"""Immutable labels for the driver; no label metadata belongs in model inputs."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import firwin, upfirdn

from . import livekit_data

DATA_VERSION = "minds-ko-pilot-stream-v1"
DEV_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
AXES = ("eot_timing", "hold_intervals", "content_clipping")
ALLOWED = {"auto_consistent", "not_applicable"}
RESAMPLING_METHOD = "causal_upfirdn_8000_to_16000_fir41_kaiser5_no_delay_compensation"
RESAMPLING_DELAY_S = 20 / 16000


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _safe_file(root, relative, *, audio=False):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Unsafe relative path: {relative!r}")
    if audio and (rel.parts[0] != "audio" or rel.suffix.lower() != ".wav"):
        raise ValueError(f"Unsafe audio path: {relative!r}")
    path = root / rel
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError(f"Symlink is not permitted: {relative}")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe file: {relative}")
    return path


def _read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate turn ID")
    return rows


def _verify_source(root):
    manifest = _safe_file(root, "manifest.sha256")
    listed = set()
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if relative in listed:
            raise ValueError(f"Duplicate manifest entry: {relative}")
        listed.add(relative)
        if _sha(_safe_file(root, relative)) != expected:
            raise ValueError(f"Source manifest hash mismatch: {relative}")
    if not {"annotations.jsonl", "verification.json"}.issubset(listed):
        raise ValueError("Source manifest missing annotations/verification")
    if json.loads((root / "verification.json").read_text()).get("status") != "passed":
        raise ValueError("Source verification must have passed")
    return listed


def _span(span, duration):
    start, end = float(span["start"]), float(span["end"])
    if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start or end > duration + 1e-6:
        raise ValueError(f"Invalid silence span: {span}")
    if "duration" in span and not math.isclose(float(span["duration"]), end - start, abs_tol=1e-6):
        raise ValueError("Silence duration disagrees with span")
    return {"start": start, "end": end}


def _pilot_reference(row):
    duration = float(row["duration_s"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid duration")
    spans, eots, holds = row["silences"], [], []
    previous_end = 0.
    for raw in spans:
        span = _span(raw, duration)
        if span["start"] < previous_end - 1e-9:
            raise ValueError("Overlapping or unordered silence spans")
        previous_end = span["end"]
        if raw["label"] == "EOT":
            eots.append(span)
        elif raw["label"] == "HOLD":
            holds.append(span)
        else:
            raise ValueError("Unknown silence label")
    if len(eots) > 1:
        raise ValueError("Ambiguous explicit EOT")
    if not eots:
        return None, holds
    eot = eots[0]
    candidate = float(row["eot"]["candidate_time_s"])
    if not math.isfinite(candidate) or abs(candidate - eot["start"]) > 1e-9:
        raise ValueError("Explicit EOT start disagrees with candidate time")
    if spans[-1]["label"] != "EOT" or abs(eot["end"] - duration) > 1e-6:
        raise ValueError("Explicit EOT must be the final trailing silence")
    return eot, holds


def _selection(row):
    if row.get("label_type") != "pseudo" or row.get("human_verified") is not False:
        raise ValueError("Pilot requires pseudo labels with human_verified=false")
    reasons = []
    if row.get("language") != "ko" or row.get("language_screen", {}).get("category") != "korean_only_auto":
        reasons.append("not_korean_only_auto")
    for axis in AXES:
        status = row.get("uncertainty", {}).get(axis, {}).get("status")
        if status not in ALLOWED:
            reasons.append(f"{axis}:{status or 'missing'}")
    eot, holds = _pilot_reference(row)
    if eot is None:
        reasons.append("missing_explicit_eot")
    elif eot["end"] - eot["start"] < .2 - 1e-9:
        reasons.append("explicit_eot_shorter_than_0.2s")
    return reasons, eot, holds


def build_pilot(source_root, output_root) -> dict:
    """Verify source, select eligible turns, copy original WAVs without relabeling."""
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output must be separate nonnested directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty")
    listed = _verify_source(source)
    rows = _read_rows(source / "annotations.jsonl")
    selected, log, audio_hashes, destinations = [], [], {}, set()
    hold_count = 0
    for row in rows:
        audio = _safe_file(source, row["audio_path"], audio=True)
        if row["audio_path"] not in listed or _sha(audio) != row["audio_sha256"]:
            raise ValueError(f"Source audio hash mismatch: {row['id']}")
        reasons, eot, holds = _selection(row)
        log.append({"id": row["id"], "selected": str(not reasons).lower(),
                    "reasons": ";".join(reasons) if reasons else "selected",
                    **{axis: row.get("uncertainty", {}).get(axis, {}).get("status", "missing") for axis in AXES},
                    "explicit_eot_duration_s": "" if eot is None else eot["end"] - eot["start"]})
        if reasons:
            continue
        info = sf.info(audio)
        if info.channels != 1 or info.samplerate != 8000 or info.samplerate != row["sample_rate"] or abs(info.duration - row["duration_s"]) > 1e-6:
            raise ValueError(f"Audio metadata mismatch: {row['id']}")
        saved = copy.deepcopy(row)
        saved["audio_path"] = f"audio/{audio.name}"
        if saved["audio_path"] in destinations:
            raise ValueError("Duplicate output audio filename")
        destinations.add(saved["audio_path"])
        selected.append((saved, audio))
        audio_hashes[row["id"]] = row["audio_sha256"]
        hold_count += len(holds)
    # All source validation precedes output writes.
    output.mkdir(parents=True, exist_ok=True)
    (output / "audio").mkdir(exist_ok=True)
    for saved, audio in selected:
        shutil.copyfile(audio, output / saved["audio_path"])
        if _sha(output / saved["audio_path"]) != saved["audio_sha256"]:
            raise ValueError("Copied audio hash mismatch")
    (output / "annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r, _ in selected))
    with (output / "selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "selected", "reasons", *AXES, "explicit_eot_duration_s"])
        writer.writeheader()
        writer.writerows(log)
    result = {
        "schema_version": DATA_VERSION, "protocol_version": DATA_VERSION,
        "record_count": len(selected), "source_record_count": len(rows),
        "eot_count": len(selected), "hold_span_count": hold_count,
        "label_type": "pseudo", "human_verified": False, "gold": False,
        "annotation_versions": sorted({r["annotation_version"] for r, _ in selected}),
        "resampling_method": RESAMPLING_METHOD,
        "resampling_delay_s": RESAMPLING_DELAY_S,
        "resampling_filter_tail": "truncated_at_original_eof; no delay compensation; original reference times retained",
        "selected_ids": [r["id"] for r, _ in selected],
        "source_audio_sha256": audio_hashes,
        "source": {"root": str(source), "annotations_sha256": _sha(source / "annotations.jsonl"),
                   "manifest_sha256": _sha(source / "manifest.sha256")},
        "selection_rule": {"uncertainty_axes": list(AXES), "allowed_statuses": sorted(ALLOWED),
                           "minimum_explicit_eot_s": .2, "hold_duration_filter": None},
        "files": {str(p.relative_to(output)): _sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def _audio16(array, sample_rate):
    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError("Audio must be nonempty finite mono PCM")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("Invalid sample rate")
    duration = len(audio) / sample_rate
    if sample_rate == 8000:
        # A centered filter would leak unseen future samples into each prefix.
        # Keep its group delay, original timeline and2N length, crop its tail.
        taps = 2 * firwin(41, .5, window=("kaiser", 5.))
        audio = upfirdn(taps, audio, up=2)[:2 * len(audio)].astype(np.float32)
    elif sample_rate != 16000:
        raise ValueError("Only original8kHz or16kHz audio is supported")
    return np.ascontiguousarray(audio), duration


def load_pilot_turns(pilot_root) -> list[dict]:
    """Load verified snapshots with references only for the outer evaluation driver."""
    root = Path(pilot_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != DATA_VERSION or manifest.get("label_type") != "pseudo" or manifest.get("human_verified") is not False:
        raise ValueError("Invalid pilot manifest version/provenance")
    if manifest.get("resampling_method") != RESAMPLING_METHOD or manifest.get("resampling_delay_s") != RESAMPLING_DELAY_S:
        raise ValueError("Pilot resampling protocol mismatch")
    files = manifest["files"]
    if not {"annotations.jsonl", "selection.csv"}.issubset(files):
        raise ValueError("Pilot manifest missing required file hashes")
    for relative, expected in files.items():
        if _sha(_safe_file(root, relative)) != expected:
            raise ValueError(f"Pilot file hash mismatch: {relative}")
    rows = _read_rows(root / "annotations.jsonl")
    if [r["id"] for r in rows] != manifest["selected_ids"] or len(rows) != manifest["record_count"]:
        raise ValueError("Pilot ID/count mismatch")
    turns = []
    for row in rows:
        path = _safe_file(root, row["audio_path"], audio=True)
        actual_hash = _sha(path)
        if row["audio_path"] not in files or actual_hash != row["audio_sha256"] or actual_hash != manifest["source_audio_sha256"].get(row["id"]):
            raise ValueError(f"Pilot source audio hash mismatch: {row['id']}")
        reasons, eot, holds = _selection(row)
        if reasons:
            raise ValueError(f"Ineligible row in pilot: {row['id']}: {reasons}")
        encoded, rate = sf.read(path, dtype="float32")
        audio, duration = _audio16(encoded, rate)
        if rate != 8000 or rate != row["sample_rate"] or abs(duration - row["duration_s"]) > 1e-6:
            raise ValueError("Pilot audio metadata mismatch")
        turns.append({"id": row["id"], "split": "minds_pilot", "audio": audio,
                      "eot_s": eot["start"], "hold_spans": holds,
                      "original_duration_s": duration, "source_audio_sha256": actual_hash,
                      "label_type": "pseudo", "resampling_delay_s": RESAMPLING_DELAY_S,
                      "resampling_method": RESAMPLING_METHOD})
    if sum(len(t["hold_spans"]) for t in turns) != manifest["hold_span_count"] or len(turns) != manifest["eot_count"]:
        raise ValueError("Pilot reference count mismatch")
    return turns


def _dev_source_hashes(cache_dir):
    """Hash encoded source bytes from the same revision verified by the old loader."""
    from datasets import Audio, Dataset
    root = Path(cache_dir) / "datasets/livekit___eot-bench-data/ko/0.0.0" / DEV_REVISION
    arrow = _safe_file(root.resolve(), "eot-bench-data-validation.arrow")
    dataset = Dataset.from_file(str(arrow)).cast_column("audio", Audio(decode=False))
    hashes = {}
    for row in dataset:
        if row["id"] in hashes or row["language"] != "ko":
            raise ValueError("Duplicate or non-Korean development row")
        encoded = row["audio"].get("bytes")
        if encoded is None:
            raise ValueError("Pinned development Arrow lacks encoded source audio")
        hashes[row["id"]] = hashlib.sha256(encoded).hexdigest()
    return hashes


def load_dev_turns(cache_dir) -> list[dict]:
    """Load exactly377 Korean validation rows, retaining every earlier HOLD span."""
    if livekit_data.DATASET_REVISION != DEV_REVISION:
        raise ValueError("Unexpected development dataset revision")
    rows = livekit_data.load_turn_rows("ko", cache_dir=str(cache_dir))
    if len(rows) != 377:
        raise ValueError(f"Expected377 development rows; received {len(rows)}")
    hashes = _dev_source_hashes(cache_dir)
    if len(hashes) != 377 or set(hashes) != {r["id"] for r in rows}:
        raise ValueError("Development source hash/ID count mismatch")
    turns, ids = [], set()
    for row in rows:
        if row["id"] in ids or row["language"] != "ko":
            raise ValueError("Duplicate or non-Korean development turn")
        ids.add(row["id"])
        audio, duration = _audio16(row["audio"]["array"], row["audio"]["sampling_rate"])
        if abs(duration - float(row["duration"])) > 1e-6:
            raise ValueError("Development audio duration mismatch")
        spans = [_span(span, duration) for span in row["silence_spans"]]
        if not spans or any(b["start"] < a["end"] for a, b in zip(spans, spans[1:])):
            raise ValueError("Missing/overlapping development silence spans")
        if abs(spans[-1]["end"] - duration) > 1e-6 or spans[-1]["end"] - spans[-1]["start"] < .2 - 1e-9:
            raise ValueError("Invalid development trailing EOT silence")
        turns.append({"id": row["id"], "split": "livekit_dev", "audio": audio,
                      "eot_s": spans[-1]["start"], "hold_spans": spans[:-1],
                      "original_duration_s": duration, "source_audio_sha256": hashes[row["id"]],
                      "label_type": "dataset_annotation",
                      "resampling_delay_s": RESAMPLING_DELAY_S if row["audio"]["sampling_rate"] == 8000 else 0.,
                      "resampling_method": RESAMPLING_METHOD if row["audio"]["sampling_rate"] == 8000 else "identity_16000"})
    return turns
