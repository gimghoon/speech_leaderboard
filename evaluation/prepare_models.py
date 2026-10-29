"""공식 고정 revision의 로컬 추론 가중치를 준비한다. 음성 API·학습 없음."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

from eot_stream import models, stream

ROOT = Path(__file__).resolve().parents[1]
VAD_REVISION = "066409f3d9db45a30d21dfcbf9147826ffdd7052"
VAD_URL = ("https://raw.githubusercontent.com/snakers4/silero-vad/"
           + VAD_REVISION + "/src/silero_vad/data/silero_vad.onnx")
SMART_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"


def prepare_vad(path):
    path = Path(path)
    if path.exists():
        content = path.read_bytes()
    else:
        with urllib.request.urlopen(VAD_URL, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != stream.VAD_SHA256:
        raise ValueError("Silero SHA256 불일치: 파일을 사용하지 않습니다")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad-path", type=Path, default=ROOT / ".cache/weights/silero_vad.onnx")
    parser.add_argument("--model-cache-dir", type=Path, default=ROOT / ".cache/models")
    parser.add_argument("--smart-turn", action="store_true", help="Smart Turn INT8도 다운로드")
    args = parser.parse_args(argv)
    print(f"Silero 확인 완료: {prepare_vad(args.vad_path)}")
    if args.smart_turn:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(models.SMART_REPO, "smart-turn-v3.2-cpu.onnx",
                                   revision=models.SMART_REVISION, cache_dir=str(args.model_cache_dir)))
        if hashlib.sha256(path.read_bytes()).hexdigest() != SMART_SHA256:
            raise ValueError("Smart Turn SHA256 불일치")
        print(f"Smart Turn 확인 완료: {path}")
    print("LiveKit Mini는 requirements-run.txt의 livekit-local-inference==0.2.7 패키지를 사용합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
