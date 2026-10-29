"""Hand-derived causal policy and outcome fixtures."""

import math

import pytest

from eot_stream.policy import evaluate_policy, first_decision, select_policy


def frame(time, age, score=None, region=0, event=True):
    return dict(timestamp_s=time, silence_start_s=time - age,
                silence_age_s=age, region_id=region, p_eot=score,
                is_score_event=event)


def model(threshold=0.5, min_wait=0.2, timeout=1.0):
    return dict(kind="model", threshold=threshold,
                min_wait_s=min_wait, timeout_s=timeout)


def ref(identifier="a", eot=1.0, duration=1.2, holds=()):
    return dict(id=identifier, eot_s=eot, hold_spans=list(holds),
                original_duration_s=duration, horizon_s=eot + 4)


def test_threshold_equal_uses_actual_frame_and_current_score_between_ticks():
    frames = [frame(1.216, .216, .5), frame(1.408, .408, .5, event=False)]
    assert first_decision(frames, model(min_wait=.4)) == {
        "timestamp_s": 1.408, "reason": "model", "region_id": 0}


def test_timeout_runs_between_score_ticks_and_model_wins_same_frame():
    frames = [frame(1.8, .8, .2), frame(2.024, 1.024, .2, event=False)]
    assert first_decision(frames, model()) == {
        "timestamp_s": 2.024, "reason": "timeout", "region_id": 0}
    assert first_decision([frame(2.024, 1.024, .5)], model())["reason"] == "model"


def test_resumed_region_has_no_previous_score_and_first_stop_counts():
    frames = [frame(.3, .2, .8), frame(.8, .2, None, 1),
              frame(1.2, .6, .3, 1)]
    assert first_decision(frames, model(min_wait=.4, timeout=1.5)) is None
    assert first_decision(frames, model())["timestamp_s"] == .3


def test_timer_uses_frame_time_and_never_backdates_to_wait():
    assert first_decision([frame(1.392, .392), frame(1.424, .424)],
                          dict(kind="timer", wait_s=.4)) == {
        "timestamp_s": 1.424, "reason": "timer", "region_id": 0}


def test_evaluation_counts_first_premature_hold_and_four_second_penalty():
    traces = {"a": [frame(.5, .2, .9), frame(1.4, .6, .9)],
              "b": [frame(1, .2, .9)], "c": [],
              "d": [frame(1.5, .5, .9)]}
    refs = {"a": ref(holds=[dict(start=.4, end=.7)]),
            "b": ref("b"), "c": ref("c"), "d": ref("d")}
    result = evaluate_policy(traces, refs, model())
    metrics = result["metrics"]
    assert metrics["n_turns"] == 4
    assert (metrics["n_premature"], metrics["n_missed"], metrics["n_correct"]) == (1, 1, 2)
    assert metrics["premature_rate"] == .25
    assert metrics["miss_rate"] == .25
    assert metrics["mean_delay_success_ms"] == 250
    assert metrics["penalized_mean_delay_ms"] == 2125
    assert metrics["timeout_rate"] == 0
    assert metrics["source_eof_stop_rate"] == .25
    assert metrics["n_hold_turns"] == metrics["n_hold_spans"] == 1
    assert metrics["hold_turn_premature_rate"] == 1
    assert result["per_turn"][0] == dict(
        id="a", eot_s=1, decision_s=.5, reason="model", outcome="premature",
        delay_ms=None, after_source_eof=False, has_hold=True,
        first_decision_in_hold=True)
    assert result["per_turn"][2]["outcome"] == "missed"


def test_eot_tolerance_and_eof_strict_boundary():
    result = evaluate_policy({"a": [frame(1 - 5e-10, .3, .8)]},
                             {"a": ref(duration=1 - 5e-10)}, model())
    assert result["per_turn"][0]["outcome"] == "correct"
    assert result["per_turn"][0]["delay_ms"] == 0
    assert result["per_turn"][0]["after_source_eof"] is False


def test_no_success_has_null_success_delay_and_all_failures_penalized():
    metrics = evaluate_policy({"a": []}, {"a": ref()}, model())["metrics"]
    assert metrics["mean_delay_success_ms"] is None
    assert metrics["penalized_mean_delay_ms"] == 4000
    assert metrics["n_hold_turns"] == 0
    assert metrics["hold_turn_premature_rate"] is None


@pytest.mark.parametrize("frames", [
    [frame(.5, .2, math.nan)], [frame(.5, .2, math.inf)],
    [frame(.5, .2, 1.1)], [frame(.5, .2, .8), frame(.4, .3, .8)],
    [frame(.5, .2, .8), frame(.5, .3, .8)], [frame(math.inf, .2)],
    [frame(.5, math.nan)], [frame(.5, .1)],
])
def test_bad_frames_fail_closed_including_bad_frames_after_first_stop(frames):
    with pytest.raises(ValueError):
        first_decision(frames, model())


@pytest.mark.parametrize("policy", [model(threshold=math.nan), model(threshold=1.1),
    model(min_wait=.3), model(timeout=.5), dict(kind="timer", wait_s=.25),
    dict(kind="timer", wait_s=math.inf), dict(kind="other")])
def test_invalid_policies_fail_closed(policy):
    with pytest.raises(ValueError):
        first_decision([], policy)


@pytest.mark.parametrize("traces,refs", [
    ({"a": []}, {"b": ref("b")}),
    ({"a": []}, {"a": ref("b")}),
    ({"a": []}, {"a": ref(eot=math.nan)}),
    ({"a": []}, {"a": dict(ref(), horizon_s=6)}),
    ({"a": [frame(5.1, .2, .8)]}, {"a": ref()}),
    ({"a": []}, {"a": ref(holds=[dict(start=.8, end=.4)])}),
    ({}, {}),
])
def test_invalid_reference_or_coverage_fails_closed(traces, refs):
    with pytest.raises(ValueError):
        evaluate_policy(traces, refs, model())


def test_selection_feasible_model_keeps_budget_and_lexicographic_ties():
    traces = {"a": [frame(.5, .2, .6), frame(1.1, .2, .8, 1)]}
    result = select_policy(traces, {"a": ref()}, "model", .05)
    assert result["feasible"] is True
    assert result["policy"] == model(threshold=.61)
    assert result["dev_metrics"]["premature_rate"] == 0
    assert result["dev_metrics"]["penalized_mean_delay_ms"] == pytest.approx(100)


def test_selection_feasible_timer_wait_and_deterministic_tie():
    traces = {"a": [frame(.5, .2), frame(1.1, .3)]}
    result = select_policy(traces, {"a": ref()}, "timer", .1)
    assert result["feasible"] is True
    assert result["policy"] == dict(kind="timer", wait_s=.3)
    assert result["dev_metrics"]["penalized_mean_delay_ms"] == pytest.approx(100)


def test_selection_infeasible_does_not_relax_budget_and_minimizes_violation():
    traces = {"a": [], "b": [frame(.5, .2, 1), frame(.8, .5, 1)]}
    refs = {"a": ref(), "b": ref("b")}
    result = select_policy(traces, refs, "timer", .05)
    assert result["feasible"] is False
    assert result["policy"] == dict(kind="timer", wait_s=.2)
    assert result["dev_metrics"]["premature_rate"] == .5
    assert result["dev_metrics"]["miss_rate"] == .5


def test_timer_grid_includes_last_wait_and_rejects_earlier_termination():
    # Physical ages require sufficient elapsed audio; the premature and correct
    # frames belong to different silence regions separated by resumed speech.
    frames = [frame(3.4, 3.4), frame(7.5, 3.5, region=1)]
    result = select_policy({"a": frames}, {"a": ref(eot=7.4, duration=7.8)}, "timer", .05)
    assert result["policy"] == dict(kind="timer", wait_s=3.5)
    assert result["feasible"] is True


def test_model_grid_includes_one_threshold_and_last_timeout():
    frames = [frame(1.1, 1., .99), frame(3.4, 3.4, None, 1),
              frame(7.5, 3.5, None, 2)]
    result = select_policy({"a": frames}, {"a": ref(eot=7.4, duration=7.8)}, "model", .05)
    assert result["policy"] == model(threshold=1., timeout=3.5)
    assert result["feasible"] is True
    assert result["dev_metrics"]["timeout_rate"] == 1


def test_exact_premature_budget_and_miss_budget_are_inclusive():
    traces = {str(i): [frame(.5 if i == 0 else 1.1, .2)] if i != 1 else []
              for i in range(20)}
    refs = {str(i): ref(str(i)) for i in range(20)}
    result = select_policy(traces, refs, "timer", .05)
    assert result["feasible"] is True
    assert result["dev_metrics"]["premature_rate"] == .05
    assert result["dev_metrics"]["miss_rate"] == .05


def test_sample_clock_horizon_allows_one_float_ulp_roundoff():
    reference = ref(eot=3.3599999999999994, duration=3.6)
    frames = [frame(3.584, .224, .8), frame(7.36, 4.0, .8)]
    assert reference['horizon_s'] < 7.36 == math.nextafter(reference['horizon_s'], math.inf)
    result = evaluate_policy({'a': frames}, {'a': reference}, model())
    assert result['metrics']['n_correct'] == 1


def test_any_frame_after_horizon_is_rejected_even_after_valid_stop():
    frames = [frame(1.1, .2, .8), frame(5 + 5e-10, .2, .8, 1)]
    with pytest.raises(ValueError):
        evaluate_policy({"a": frames}, {"a": ref()}, model())


@pytest.mark.parametrize("kind,target", [("other", .05), ("timer", .2), ("model", math.nan)])
def test_selection_rejects_unknown_kind_and_unapproved_target(kind, target):
    with pytest.raises(ValueError):
        select_policy({"a": []}, {"a": ref()}, kind, target)
