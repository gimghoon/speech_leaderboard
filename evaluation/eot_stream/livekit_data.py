"""Version-pinned public LiveKit dataset loading; never uses torchcodec."""
from __future__ import annotations

import io
from pathlib import Path

import soundfile as sf

DATASET = "livekit/eot-bench-data"
DATASET_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"


def decode_audio(audio: dict) -> dict:
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio.get("path")
    if source is None:
        raise ValueError("Missing encoded audio bytes/path")
    array, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    return {"array": array, "sampling_rate": sample_rate}


def select_rows(rows, language: str, limit: int | None) -> list[dict]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected, ids = [], set()
    for row in rows:
        if row["language"] != language:
            raise ValueError(f"Unexpected language {row['language']!r} in {language}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate turn ID: {row['id']}")
        ids.add(row["id"])
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError(f"No rows for language {language}")
    return selected


def load_turn_rows(language: str, *, limit: int | None = None, cache_dir: str | None = None) -> list[dict]:
    from datasets import Audio, load_dataset
    ds = load_dataset(DATASET, language, split="validation", revision=DATASET_REVISION,
                      cache_dir=str(Path(cache_dir) / "datasets") if cache_dir else None)
    # datasets may ignore `revision` and fall back to its latest Arrow cache
    # when the Hub is unavailable. Fail closed rather than mislabel that run.
    resolved_revisions = {Path(item["filename"]).parent.name for item in ds.cache_files}
    if resolved_revisions != {DATASET_REVISION}:
        raise RuntimeError(f"Cannot verify dataset revision {DATASET_REVISION}; "
                           f"resolved cache revisions: {sorted(resolved_revisions)}")
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = select_rows(ds, language, limit)
    return [{**row, "audio": decode_audio(row["audio"])} for row in rows]
