"""Portable wrapper tests; frozen policy replay has no policy-selection path."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_baselines as portable
from eot_stream import policy


def fixture_trace():
    frames = [dict(timestamp_s=.8, silence_age_s=.2, region_id=0, p_eot=.7,
                   is_score_event=True, inference_ms=2.),
              dict(timestamp_s=1.2, silence_age_s=.6, region_id=0, p_eot=.9,
                   is_score_event=True, inference_ms=3.)]
    ref = dict(id="p1", eot_s=1., hold_spans=[dict(start=.6, end=.9)],
               original_duration_s=1.5, horizon_s=5., synthetic_silence_s=3.5)
    return {"p1": frames}, {"p1": ref}


def test_both_frozen_budgets_replay_same_ids_without_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("policy selection must never run on pilot")
    monkeypatch.setattr(policy, "select_policy", forbidden)
    traces, refs = fixture_trace()
    frozen = dict(budgets={
        "0.05": dict(policy=dict(kind="model", threshold=.8, min_wait_s=.2, timeout_s=1.), feasible=True),
        "0.10": dict(policy=dict(kind="model", threshold=.6, min_wait_s=.2, timeout_s=1.), feasible=True)})
    summaries, outcomes = portable.evaluate_model("smart-turn", traces, refs, frozen)
    assert len(summaries) == 2
    assert [summaries[0]["n_correct"], summaries[1]["n_premature"]] == [1, 1]
    assert [outcomes[b][0]["id"] for b in portable.BUDGETS] == ["p1", "p1"]
    assert outcomes[.05][0]["decision_s"] == 1.2
    assert outcomes[.10][0]["decision_s"] == .8
    for summary in summaries:
        expected = policy.evaluate_policy(traces, refs, frozen["budgets"][f"{summary['budget']:.2f}"]["policy"])
        assert all(summary[key] == value for key, value in expected["metrics"].items())


def test_model_identity_checks_hash_and_records_native_platform_separately():
    expected = dict(model="turn-detector-v1-mini", package="livekit-local-inference",
                    version="0.2.7", context_seconds=1.2, native_binary_sha256="arm")
    actual = {**expected, "native_binary_sha256": "x86"}
    portable.validate_model_identity("livekit-mini", actual, expected)
    with pytest.raises(ValueError, match="버전"):
        portable.validate_model_identity("livekit-mini", {**actual, "version": "0.2.8"}, expected)
    with pytest.raises(ValueError):
        portable.validate_model_identity("smart-turn", dict(sha256="changed"), dict(sha256="original"))


def test_complete_wrapper_never_loads_dev_or_retunes(tmp_path, monkeypatch):
    traces, refs = fixture_trace()
    turns = [dict(id="p1", split="minds_pilot", audio=np.ones(16000, dtype=np.float32),
                  eot_s=1., original_duration_s=1.5, hold_spans=refs["p1"]["hold_spans"],
                  source_audio_sha256="abc", label_type="pseudo")]
    timer_metadata = portable.cli._runtime_model_metadata(None)
    frozen = dict(vad={"sha256": "test"}, models={"timer": dict(model_settings=timer_metadata,
                  budgets={key: dict(feasible=True, policy=dict(kind="timer", wait_s=.2))
                           for key in ("0.05", "0.10")})})
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen))
    monkeypatch.setattr(portable.data, "load_pilot_turns", lambda path: turns)
    monkeypatch.setattr(portable, "validate_snapshot", lambda *args: "snapshot-hash")
    def forbidden(*args, **kwargs):
        raise AssertionError("development loading and policy selection forbidden")
    monkeypatch.setattr(portable.data, "load_dev_turns", forbidden)
    monkeypatch.setattr(portable.data, "build_pilot", forbidden)
    monkeypatch.setattr(portable.policy, "select_policy", forbidden)
    monkeypatch.setattr(portable.stream, "SileroFrames", lambda path: SimpleNamespace(metadata=frozen["vad"]))
    def get_trace(turn, *args):
        return traces[turn["id"]], refs[turn["id"]]
    monkeypatch.setattr(portable.cache, "get_trace", get_trace)
    def trace_path(*args):
        path = tmp_path / "cached-trace.json"
        path.write_text("{}")
        return path
    monkeypatch.setattr(portable.cache, "trace_path", trace_path)
    args = portable.parse_args(["--models", "timer", "--vad-path", str(tmp_path / "vad"),
                                "--output", str(tmp_path / "run"),
                                "--frozen-policies", str(frozen_path),
                                "--pilot-root", str(tmp_path / "pilot")])
    result = portable.run(args)
    assert result["status"] == "complete"
    assert result["policy_selection_performed"] is False
    assert result["original_runtime_match"] == {"timer": True}
    assert len(json.loads((args.output / "summary.json").read_text())["rows"]) == 2
    assert (args.output / "outcomes/timer-pilot-05.csv").exists()
    with pytest.raises(ValueError, match="비어"):
        portable.run(args)


def test_modified_snapshot_fails_before_any_model_load(monkeypatch):
    with pytest.raises(ValueError, match="160"):
        portable.validate_snapshot([], {})


def test_json_nonfinite_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"score":NaN}')
    with pytest.raises(ValueError, match="비유한"):
        portable.read_json(path)
