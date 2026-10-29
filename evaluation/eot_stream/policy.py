"""Pure causal policy replay and development-only selection.

Input frames contain the latest score observed *within the current silence
region*. A new region must therefore start with ``p_eot=None`` until its first
score. Replay never interpolates timestamps or schedules a decision in the past.
"""

from collections.abc import Mapping
from itertools import product
import math
from numbers import Real

import numpy as np


_MIN_WAITS = (.2, .4, .6, .8, 1.)
_TIMEOUTS = (1., 1.5, 2., 2.5, 3., 3.5)
_TIMER_WAITS = tuple(i / 10 for i in range(2, 36))
_THRESHOLDS = tuple(i / 100 for i in range(101))
_EOT_TOLERANCE = 1e-9


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_policy(policy):
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be a mapping")
    kind = policy.get("kind")
    if kind == "model":
        threshold = _number(policy.get("threshold"), "threshold")
        wait = _number(policy.get("min_wait_s"), "min_wait_s")
        timeout = _number(policy.get("timeout_s"), "timeout_s")
        if not 0 <= threshold <= 1 or wait not in _MIN_WAITS:
            raise ValueError("invalid model threshold or minimum wait")
        if timeout not in _TIMEOUTS or timeout < wait:
            raise ValueError("invalid model timeout")
    elif kind == "timer":
        if _number(policy.get("wait_s"), "wait_s") not in _TIMER_WAITS:
            raise ValueError("timer wait must be .2 through 3.5 in .1 increments")
    else:
        raise ValueError("unknown policy kind")


def _validate_frames(frames):
    if not isinstance(frames, (list, tuple)):
        raise ValueError("frames must be a chronological sequence")
    previous = -math.inf
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("frame must be a mapping")
        time = _number(frame.get("timestamp_s"), "timestamp_s")
        age = _number(frame.get("silence_age_s"), "silence_age_s")
        if time < 0 or time <= previous or age < .2 or age > time + _EOT_TOLERANCE:
            raise ValueError("invalid frame time, age, or chronological order")
        if "region_id" not in frame or frame["region_id"] is None:
            raise ValueError("missing silence region")
        if not isinstance(frame.get("is_score_event"), bool):
            raise ValueError("is_score_event must be a boolean")
        if "p_eot" not in frame:
            raise ValueError("missing p_eot")
        if frame["p_eot"] is not None:
            score = _number(frame["p_eot"], "p_eot")
            if not 0 <= score <= 1:
                raise ValueError("p_eot must lie in [0, 1]")
        if "silence_start_s" in frame:
            start = _number(frame["silence_start_s"], "silence_start_s")
            if start < -_EOT_TOLERANCE or abs(time - start - age) > _EOT_TOLERANCE:
                raise ValueError("silence start and age disagree")
        previous = time


def _decision(frames, policy):
    for frame in frames:
        age = frame["silence_age_s"]
        if policy["kind"] == "timer":
            reason = "timer" if age >= policy["wait_s"] else None
        elif (age >= policy["min_wait_s"] and frame["p_eot"] is not None
              and frame["p_eot"] >= policy["threshold"]):
            reason = "model"
        else:
            reason = "timeout" if age >= policy["timeout_s"] else None
        if reason is not None:
            return dict(timestamp_s=float(frame["timestamp_s"]), reason=reason,
                        region_id=frame["region_id"])
    return None


def first_decision(frames, policy):
    """Return the first actual termination frame, or None through observation end.

    Validate the complete trace before replay, including frames after a possible
    termination: failed or nonfinite inference must never be silently accepted.
    """
    _validate_policy(policy)
    _validate_frames(frames)
    return _decision(frames, policy)


def _validate_inputs(traces, refs):
    if not isinstance(traces, Mapping) or not isinstance(refs, Mapping):
        raise ValueError("traces and refs must be ID mappings")
    if not traces or set(traces) != set(refs):
        raise ValueError("nonempty identical trace/reference ID coverage required")
    for identifier, frames in traces.items():
        reference = refs[identifier]
        if not isinstance(reference, Mapping) or reference.get("id") != identifier:
            raise ValueError("reference ID does not match its mapping key")
        eot = _number(reference.get("eot_s"), "eot_s")
        duration = _number(reference.get("original_duration_s"), "original_duration_s")
        horizon = _number(reference.get("horizon_s"), "horizon_s")
        if (eot < 0 or duration < eot - _EOT_TOLERANCE
                or abs(horizon - eot - 4) > _EOT_TOLERANCE):
            raise ValueError("invalid EOT, source duration, or EOT+4s horizon")
        holds = reference.get("hold_spans")
        if not isinstance(holds, (list, tuple)):
            raise ValueError("hold_spans must be a sequence")
        for span in holds:
            if not isinstance(span, Mapping):
                raise ValueError("HOLD span must be a mapping")
            start = _number(span.get("start"), "HOLD start")
            end = _number(span.get("end"), "HOLD end")
            if not 0 <= start < end <= horizon:
                raise ValueError("invalid HOLD span")
        _validate_frames(frames)
        # EOT+4 addition and integer-sample/16000 division can differ by one
        # float ULP at the same physical boundary; do not admit a later sample.
        if frames and frames[-1]["timestamp_s"] > math.nextafter(horizon, math.inf):
            raise ValueError("frame occurs after observation horizon")


def evaluate_policy(traces, refs, policy):
    """Evaluate first termination per turn; failures each incur 4000ms loss."""
    _validate_policy(policy)
    _validate_inputs(traces, refs)
    per_turn = []
    success_delays = []
    premature = missed = timeouts = eof_stops = hold_premature = 0
    for identifier, frames in traces.items():
        reference = refs[identifier]
        eot = reference["eot_s"]
        holds = reference["hold_spans"]
        decision = _decision(frames, policy)
        time = decision["timestamp_s"] if decision else None
        reason = decision["reason"] if decision else None
        delay = None
        if decision is None:
            outcome = "missed"
            missed += 1
        elif time < eot - _EOT_TOLERANCE:
            outcome = "premature"
            premature += 1
            hold_premature += bool(holds)
        else:
            outcome = "correct"
            delay = max(0., time - eot) * 1000
            success_delays.append(delay)
        after_eof = time is not None and time > reference["original_duration_s"]
        timeouts += reason == "timeout"
        eof_stops += after_eof
        in_hold = time is not None and any(s["start"] <= time < s["end"] for s in holds)
        per_turn.append(dict(id=identifier, eot_s=eot, decision_s=time,
                             reason=reason, outcome=outcome, delay_ms=delay,
                             after_source_eof=after_eof, has_hold=bool(holds),
                             first_decision_in_hold=in_hold))
    count = len(per_turn)
    n_hold = sum(bool(r["hold_spans"]) for r in refs.values())
    metrics = dict(
        n_turns=count, n_premature=premature, n_missed=missed,
        n_correct=len(success_delays), premature_rate=premature / count,
        miss_rate=missed / count,
        mean_delay_success_ms=(sum(success_delays) / len(success_delays)
                               if success_delays else None),
        penalized_mean_delay_ms=(sum(success_delays) + 4000 * (premature + missed)) / count,
        timeout_rate=timeouts / count, source_eof_stop_rate=eof_stops / count,
        n_hold_turns=n_hold, hold_turn_premature_rate=(hold_premature / n_hold if n_hold else None),
        n_hold_spans=sum(len(r["hold_spans"]) for r in refs.values()))
    return dict(metrics=metrics, per_turn=per_turn)


def _crossing_indices(ages, levels):
    """First frame to reach each age, even across nonmonotone region resets."""
    if not ages.size:
        return np.zeros(len(levels), dtype=np.int64)
    return np.searchsorted(np.maximum.accumulate(ages), levels, side="left")


def _grid_decision_times(frames, kind):
    """Replay one turn for the exact grid without a Python loop per policy."""
    times = np.array([f["timestamp_s"] for f in frames], dtype=np.float64)
    ages = np.array([f["silence_age_s"] for f in frames], dtype=np.float64)
    sentinel = len(frames)
    padded_times = np.append(times, np.inf)
    if kind == "timer":
        return padded_times[_crossing_indices(ages, _TIMER_WAITS)]
    scores = np.array([f["p_eot"] if f["p_eot"] is not None else -np.inf
                       for f in frames], dtype=np.float64)
    model_indices = np.full((101, 5, 1), sentinel, dtype=np.int64)
    for column, wait in enumerate(_MIN_WAITS):
        eligible = np.flatnonzero((ages >= wait) & np.isfinite(scores))
        if eligible.size:
            crossings = np.searchsorted(np.maximum.accumulate(scores[eligible]),
                                        _THRESHOLDS, side="left")
            model_indices[:, column, 0] = np.append(eligible, sentinel)[crossings]
    timeout_indices = _crossing_indices(ages, _TIMEOUTS)[None, None, :]
    indices = np.minimum(model_indices, timeout_indices).reshape(-1)
    return padded_times[indices]


def select_policy(traces, refs, kind, target):
    """Select solely from the supplied development traces and references.

    Feasibility never relaxes: premature <= target AND missed <= .05. If no
    candidate satisfies both, minimize the sum of positive rate violations and
    explicitly return ``feasible=False``. Remaining ties use loss, missed rate,
    premature rate, then threshold/minimum wait/timeout (or timer wait).
    """
    if kind not in ("model", "timer"):
        raise ValueError("unknown policy kind")
    target = _number(target, "target")
    if target not in (.05, .10):
        raise ValueError("target must be .05 or .10")
    _validate_inputs(traces, refs)
    if kind == "model":
        policies = [dict(kind="model", threshold=t, min_wait_s=w, timeout_s=o)
                    for t, w, o in product(_THRESHOLDS, _MIN_WAITS, _TIMEOUTS) if o >= w]
    else:
        policies = [dict(kind="timer", wait_s=w) for w in _TIMER_WAITS]
    size = len(policies)
    premature = np.zeros(size, dtype=np.int64)
    missed = np.zeros(size, dtype=np.int64)
    losses = np.zeros(size, dtype=np.float64)
    for identifier, frames in traces.items():
        times = _grid_decision_times(frames, kind)
        eot = refs[identifier]["eot_s"]
        is_missed = ~np.isfinite(times)
        is_premature = times < eot - _EOT_TOLERANCE
        is_correct = ~(is_missed | is_premature)
        premature += is_premature
        missed += is_missed
        loss = np.full(size, 4000.)
        loss[is_correct] = np.maximum(0., times[is_correct] - eot) * 1000
        losses += loss
    count = len(traces)
    premature_rates = premature / count
    miss_rates = missed / count
    feasible = (premature_rates <= target) & (miss_rates <= .05)
    if feasible.any():
        candidates = np.flatnonzero(feasible)
        order = np.lexsort((candidates, premature[candidates], missed[candidates], losses[candidates]))
    else:
        candidates = np.arange(size)
        violation = np.maximum(0., premature_rates - target) + np.maximum(0., miss_rates - .05)
        order = np.lexsort((candidates, premature, missed, losses, violation))
    best = int(candidates[order[0]])
    policy = policies[best]
    return dict(policy=policy, feasible=bool(feasible[best]),
                dev_metrics=evaluate_policy(traces, refs, policy)["metrics"])
