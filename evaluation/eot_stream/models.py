"""Local-only adapters. Heavy model libraries are imported on construction."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np

SMART_REPO = "pipecat-ai/smart-turn-v3"
SMART_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
NAMES = {"timer": "Silence timer (annotated pauses)",
         "smart-turn": "Smart Turn v3.2 (CPU INT8)",
         "livekit-mini": "LiveKit Turn Detector v1-mini (local)"}


def smart_window(pcm: np.ndarray) -> np.ndarray:
    tail = np.asarray(pcm, dtype=np.float32)[-128000:]
    return np.pad(tail, (128000 - len(tail), 0))


def mini_pcm(pcm: np.ndarray) -> np.ndarray:
    tail = np.asarray(pcm, dtype=np.float32)[-19200:]
    return np.ascontiguousarray((np.clip(tail, -1.0, 1.0) * 32767.0).astype(np.int16))


class SmartTurn:
    def __init__(self, *, cache_dir: str | None = None, precision: str = "int8"):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        if precision not in ("int8", "fp32"):
            raise ValueError("precision must be int8 or fp32")
        filename = "smart-turn-v3.2-cpu.onnx" if precision == "int8" else "smart-turn-v3.2-gpu.onnx"
        path = hf_hub_download(SMART_REPO, filename, revision=SMART_REVISION, cache_dir=cache_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self.extractor = WhisperFeatureExtractor(chunk_length=8)
        self.metadata = {"model": SMART_REPO, "revision": SMART_REVISION, "file": filename,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "precision": precision, "providers": self.session.get_providers(),
            "context_seconds": 8, "onnxruntime": version("onnxruntime")}

    def predict(self, pcm: np.ndarray) -> float:
        features = self.extractor(smart_window(pcm), sampling_rate=16000,
            return_tensors="np", padding="max_length", max_length=128000,
            truncation=True, do_normalize=True).input_features.astype(np.float32)
        return float(self.session.run(None, {"input_features": features})[0].reshape(-1)[0])

    def close(self):
        self.session = None


class LiveKitMini:
    def __init__(self):
        from livekit.local_inference import EOT
        installed = version("livekit-local-inference")
        if installed != "0.2.7":
            raise RuntimeError(f"Expected livekit-local-inference 0.2.7, found {installed}")
        self.model = EOT()
        self.metadata = {"model": "turn-detector-v1-mini", "package": "livekit-local-inference",
                         "version": installed, "context_seconds": 1.2}

    def predict(self, pcm: np.ndarray) -> float:
        return float(self.model.predict(mini_pcm(pcm)))

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


def load_detector(name: str, *, cache_dir: str | None = None, precision: str = "int8"):
    if name == "timer":
        return None
    if name == "smart-turn":
        return SmartTurn(cache_dir=cache_dir, precision=precision)
    if name == "livekit-mini":
        return LiveKitMini()
    raise ValueError(f"Unknown model {name!r}")
