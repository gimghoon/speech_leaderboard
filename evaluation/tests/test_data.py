import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eot_stream import data


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(turn_id="turn1"):
    return {
        "id": turn_id, "language": "ko", "audio_path": f"audio/{turn_id}.wav",
        "duration_s": 8., "sample_rate": 8000, "annotation_version": "2.0",
        "label_type": "pseudo", "human_verified": False,
        "uncertainty": {axis: {"status": "auto_consistent"} for axis in
                        ("eot_timing", "hold_intervals", "content_clipping")},
        "silences": [{"start": .5, "end": 6., "duration": 5.5, "label": "HOLD"},
                     {"start": 7., "end": 8., "duration": 1., "label": "EOT"}],
        "eot": {"candidate_time_s": 7.},
        "language_screen": {"category": "korean_only_auto"},
        "transcripts": {"asr": "한국어 원문 보존"},
    }


def seal(root, rows):
    (root / "annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.sha256")
    (root / "manifest.sha256").write_text("".join(f"{digest(p)}  {p.relative_to(root)}\n" for p in files))


def source(tmp_path, rows):
    root = tmp_path / "source"
    (root / "audio").mkdir(parents=True)
    for row in rows:
        path = root / row["audio_path"]
        if not path.exists():
            sf.write(path, np.sin(np.arange(64000) / 40).astype(np.float32) * .2, 8000)
        row["audio_sha256"] = digest(path)
    (root / "verification.json").write_text(json.dumps({"status": "passed", "human_verified": False}))
    seal(root, rows)
    return root


def test_snapshot_preserves_original_labels_bytes_and_all_hold_spans(tmp_path):
    rows = [record()]
    src = source(tmp_path, rows)
    before = {str(p.relative_to(src)): digest(p) for p in src.rglob("*") if p.is_file()}
    out = tmp_path / "pilot"
    result = data.build_pilot(src, out)
    saved = json.loads((out / "annotations.jsonl").read_text())
    expected = copy.deepcopy(rows[0])
    expected["audio_path"] = saved["audio_path"]
    assert saved == expected
    assert (out / saved["audio_path"]).read_bytes() == (src / rows[0]["audio_path"]).read_bytes()
    assert before == {str(p.relative_to(src)): digest(p) for p in src.rglob("*") if p.is_file()}
    assert result["record_count"] == 1 and result["hold_span_count"] == 1
    assert result["label_type"] == "pseudo" and result["human_verified"] is False
    turn, = data.load_pilot_turns(out)
    assert turn["id"] == "turn1" and turn["split"] == "minds_pilot"
    assert turn["eot_s"] == 7. and turn["hold_spans"] == [{"start": .5, "end": 6.}]
    assert turn["audio"].shape == (128000,) and turn["audio"].dtype == np.float32
    assert turn["original_duration_s"] == 8. and turn["source_audio_sha256"] == rows[0]["audio_sha256"]
    assert turn["resampling_delay_s"] == .00125
    assert result["resampling_delay_s"] == .00125
    assert turn["resampling_method"] == result["resampling_method"]


@pytest.mark.parametrize("axis", ["eot_timing", "hold_intervals", "content_clipping"])
def test_each_uncertainty_axis_excludes_and_logs_reason(tmp_path, axis):
    rows = [record("good"), record("bad")]
    rows[1]["uncertainty"][axis]["status"] = "uncertain"
    src = source(tmp_path, rows)
    out = tmp_path / "pilot"
    result = data.build_pilot(src, out)
    with (out / "selection.csv").open() as handle:
        selection = list(csv.DictReader(handle))
    assert result["record_count"] == 1 and len(selection) == 2
    assert selection[1]["selected"] == "false" and axis in selection[1]["reasons"]


def test_not_applicable_axis_is_allowed_and_point_two_eot_is_inclusive(tmp_path):
    row = record()
    row["uncertainty"]["hold_intervals"]["status"] = "not_applicable"
    row["silences"][-1].update(start=7.8, duration=.2)
    row["eot"]["candidate_time_s"] = 7.8
    result = data.build_pilot(source(tmp_path, [row]), tmp_path / "pilot")
    assert result["record_count"] == 1


@pytest.mark.parametrize("kind", ["short", "missing"])
def test_no_last_hold_to_eot_relabeling_and_short_eot_exclusion(tmp_path, kind):
    row = record()
    if kind == "missing":
        row["silences"][-1]["label"] = "HOLD"
    else:
        row["silences"][-1].update(start=7.9, duration=.1)
        row["eot"]["candidate_time_s"] = 7.9
    out = tmp_path / "pilot"
    summary = data.build_pilot(source(tmp_path, [row]), out)
    assert summary["record_count"] == 0
    assert (out / "annotations.jsonl").read_text() == ""


@pytest.mark.parametrize("fault", ["duplicate", "ambiguous", "mismatch", "unsafe", "hash"])
def test_invalid_source_fails_closed_before_output(tmp_path, fault):
    rows = [record()]
    src = source(tmp_path, rows)
    if fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "ambiguous":
        rows[0]["silences"].append(copy.deepcopy(rows[0]["silences"][-1]))
    elif fault == "mismatch":
        rows[0]["eot"]["candidate_time_s"] = 6.9
    elif fault == "unsafe":
        rows[0]["audio_path"] = "../outside.wav"
    else:
        rows[0]["audio_sha256"] = "0" * 64
    seal(src, rows)
    out = tmp_path / "pilot"
    with pytest.raises(ValueError):
        data.build_pilot(src, out)
    assert not out.exists()


def test_stale_source_manifest_is_rejected(tmp_path):
    src = source(tmp_path, [record()])
    with (src / "annotations.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash|manifest"):
        data.build_pilot(src, tmp_path / "pilot")


@pytest.mark.parametrize("filename", ["audio/turn1.wav", "annotations.jsonl"])
def test_snapshot_tampered_audio_and_annotations_are_rejected(tmp_path, filename):
    src = source(tmp_path, [record()])
    out = tmp_path / "pilot"
    data.build_pilot(src, out)
    with (out / filename).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        data.load_pilot_turns(out)


def test_existing_output_and_source_destination_are_rejected(tmp_path):
    src = source(tmp_path, [record()])
    out = tmp_path / "pilot"
    out.mkdir()
    (out / "valuable.txt").write_text("preserve")
    with pytest.raises(ValueError):
        data.build_pilot(src, out)
    with pytest.raises(ValueError):
        data.build_pilot(src, src / "nested")
    assert (out / "valuable.txt").read_text() == "preserve"


def test_dev_normalization_preserves_every_hold_without_duration_filter(monkeypatch, tmp_path):
    rows = [{"id": f"ko__{i}", "language": "ko", "duration": 8.,
             "audio": {"array": np.ones(64000, dtype=np.float32), "sampling_rate": 8000},
             "silence_spans": [{"start": .1, "end": .2}, {"start": .3, "end": 6.},
                               {"start": 7., "end": 8.}]} for i in range(377)]
    monkeypatch.setattr(data.livekit_data, "load_turn_rows", lambda *a, **kw: rows)
    monkeypatch.setattr(data, "_dev_source_hashes", lambda cache_dir: {r["id"]: "a" * 64 for r in rows})
    turns = data.load_dev_turns(tmp_path)
    assert len(turns) == 377
    assert turns[0]["audio"].shape == (128000,)
    assert turns[0]["eot_s"] == 7.
    assert turns[0]["hold_spans"] == [{"start": .1, "end": .2}, {"start": .3, "end": 6.}]
    assert turns[0]["label_type"] == "dataset_annotation" and turns[0]["split"] == "livekit_dev"


def test_dev_rejects_partial_count(monkeypatch, tmp_path):
    monkeypatch.setattr(data.livekit_data, "load_turn_rows", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="377"):
        data.load_dev_turns(tmp_path)


def test_dev_rejects_unpinned_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(data.livekit_data, "DATASET_REVISION", "other")
    with pytest.raises(ValueError, match="revision"):
        data.load_dev_turns(tmp_path)


def test_resampling_is_causal_when_future_source_changes():
    original = np.sin(np.arange(8000) * .13).astype(np.float32)
    changed = original.copy()
    changed[4321:] = 100.
    first, duration = data._audio16(original, 8000)
    second, _ = data._audio16(changed, 8000)
    prefix_only, prefix_duration = data._audio16(original[:4321], 8000)
    assert np.array_equal(first[:8642], second[:8642])
    assert np.array_equal(first[:8642], prefix_only)
    assert first.shape == (16000,) and duration == 1.
    assert prefix_duration == 4321 / 8000


def test_missing_uncertainty_axis_is_excluded_with_a_reason(tmp_path):
    row = record()
    del row["uncertainty"]["hold_intervals"]
    out = tmp_path / "pilot"
    summary = data.build_pilot(source(tmp_path, [row]), out)
    assert summary["record_count"] == 0
    assert "hold_intervals:missing" in (out / "selection.csv").read_text()


def test_dev_identity_resampling_has_no_filter_delay(monkeypatch, tmp_path):
    pcm = np.zeros(128000, dtype=np.float32)
    rows = [{"id": f"ko__{i}", "language": "ko", "duration": 8.,
             "audio": {"array": pcm, "sampling_rate": 16000},
             "silence_spans": [{"start": 7., "end": 8.}]} for i in range(377)]
    monkeypatch.setattr(data.livekit_data, "load_turn_rows", lambda *a, **kw: rows)
    monkeypatch.setattr(data, "_dev_source_hashes", lambda cache_dir: {r["id"]: "a" * 64 for r in rows})
    turn = data.load_dev_turns(tmp_path)[0]
    assert np.array_equal(turn["audio"], pcm)
    assert turn["resampling_delay_s"] == 0.
    assert turn["resampling_method"] == "identity_16000"


def test_pilot_manifest_resampling_protocol_cannot_claim_different_delay(tmp_path):
    src = source(tmp_path, [record()])
    out = tmp_path / "pilot"
    data.build_pilot(src, out)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["resampling_delay_s"] = 0.
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="resampl"):
        data.load_pilot_turns(out)


def test_causal_resampling_retains_known_group_delay():
    impulse = np.zeros(100, dtype=np.float32)
    impulse[0] = 1.
    converted, duration = data._audio16(impulse, 8000)
    assert np.argmax(np.abs(converted)) == 20
    assert len(converted) == 200 and duration == .0125


def test_pilot_unsafe_symlink_audio_is_rejected(tmp_path):
    src = source(tmp_path, [record()])
    original = src / "audio/turn1.wav"
    external = tmp_path / "external.wav"
    original.rename(external)
    original.symlink_to(external)
    with pytest.raises(ValueError, match="Symlink|unsafe"):
        data.build_pilot(src, tmp_path / "pilot")
