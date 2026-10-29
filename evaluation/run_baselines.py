"""포함된 160개 공개 파일럿에 이미 동결된 정책을 적용한다. 학습·재튜닝 없음."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from eot_stream import cache, cli, data, models, policy, report, stream

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (.05, .10)


def read_json(path):
    def reject(value):
        raise ValueError(f"비유한 JSON 값: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def validate_snapshot(turns, frozen):
    ids = [turn["id"] for turn in turns]
    if len(turns) != 160 or len(set(ids)) != 160:
        raise ValueError("파일럿은 중복 없는 160개 ID여야 합니다")
    if any(turn["split"] != "minds_pilot" for turn in turns):
        raise ValueError("파일럿 split 불일치")
    if sum(len(turn["hold_spans"]) for turn in turns) != 91:
        raise ValueError("파일럿 HOLD 91개 불일치")
    if frozen.get("selection_split") != "livekit_dev" or len(set(frozen.get("dev_ids", []))) != 377:
        raise ValueError("377개 개발셋에서 동결한 정책이 필요합니다")
    if set(ids) & set(frozen["dev_ids"]):
        raise ValueError("개발·파일럿 ID 중복")
    fingerprint = cli._dataset_fingerprint(turns)
    if fingerprint != frozen["data_sha256"]["pilot"]:
        raise ValueError("동결 정책의 파일럿 음원·참조 fingerprint와 다릅니다")
    if frozen.get("vad", {}).get("sha256") != stream.VAD_SHA256:
        raise ValueError("동결 정책의 VAD 해시가 다릅니다")
    return fingerprint


def validate_model_identity(name, actual, expected):
    # Native binary and runtime differences are recorded, because CPU platform
    # changes can legitimately change those bytes. Scientific reruns can opt
    # into an exact historical-runtime check with --require-original-runtime.
    keys = {"timer": ("model", "clock", "no_model_inference"),
            "smart-turn": ("model", "revision", "file", "sha256", "precision", "context_seconds"),
            "livekit-mini": ("model", "package", "version", "context_seconds")}[name]
    if any(actual.get(key) != expected.get(key) for key in keys):
        raise ValueError(f"{name}: 동결된 모델 ID·버전·가중치가 다릅니다")


def evaluate_model(name, traces, refs, frozen_model):
    """같은 trace를 5%·10% 개발 예산의 동결 정책으로 각각 재생한다."""
    summaries, outcomes = [], {}
    timing = report.inference_stats(traces, name)
    for budget in BUDGETS:
        selection = frozen_model["budgets"][f"{budget:.2f}"]
        result = policy.evaluate_policy(traces, refs, selection["policy"])
        outcomes[budget] = [dict(model=name, split="pilot", budget=budget,
                                dev_feasible=selection["feasible"], **row,
                                synthetic_silence_s=refs[row["id"]]["synthetic_silence_s"],
                                vad_gate_frames=len(traces[row["id"]]),
                                score_events=sum(f["is_score_event"] for f in traces[row["id"]]),
                                decision_diagnostic=(row["reason"] or
                                    ("no_vad_gate_event" if not traces[row["id"]] else "no_policy_stop")))
                            for row in result["per_turn"]]
        summaries.append(dict(model=name, model_label=cli.MODEL_LABELS[name], split="pilot",
                              budget=budget, dev_feasible=selection["feasible"],
                              **result["metrics"], **timing,
                              synthetic_silence_total_s=sum(r["synthetic_silence_s"] for r in refs.values())))
    return summaries, outcomes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=ROOT / "data/pilot")
    parser.add_argument("--frozen-policies", type=Path, default=ROOT / "results/frozen_policies.json")
    parser.add_argument("--output", type=Path, required=True, help="비어 있는 결과 폴더")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/traces")
    parser.add_argument("--model-cache-dir", type=Path, default=ROOT / ".cache/models")
    parser.add_argument("--vad-path", type=Path, required=True, help="고정된 Silero ONNX 파일")
    parser.add_argument("--models", nargs="+", choices=sorted(cli.MODEL_LABELS),
                        default=["timer", "smart-turn", "livekit-mini"])
    parser.add_argument("--require-original-runtime", action="store_true",
                        help="네이티브 바이너리 해시 등 실행환경도 원본과 같아야 함")
    args = parser.parse_args(argv)
    if len(args.models) != len(set(args.models)):
        parser.error("중복 모델은 허용하지 않습니다")
    return args


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 --output을 지정하세요")
    pilot_root = args.pilot_root.resolve()
    if output.is_relative_to(pilot_root) or pilot_root.is_relative_to(output):
        raise ValueError("음원·결과 폴더는 서로 중첩될 수 없습니다")
    frozen_bytes_hash = report.sha256(args.frozen_policies)
    frozen = read_json(args.frozen_policies)
    turns = data.load_pilot_turns(pilot_root)
    fingerprint = validate_snapshot(turns, frozen)
    for name in args.models:
        if set(frozen["models"][name]["budgets"]) != {"0.05", "0.10"}:
            raise ValueError("5%·10% 동결 정책이 모두 필요합니다")
        for selection in frozen["models"][name]["budgets"].values():
            policy.first_decision([], selection["policy"])
    vad = stream.SileroFrames(args.vad_path)
    if vad.metadata != frozen["vad"]:
        raise ValueError("동결 정책과 다른 공통 VAD입니다")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", run_version="portable-frozen-pilot-v1",
                    started_at=datetime.now(timezone.utc).isoformat(), protocol=stream.PROTOCOL,
                    data_sha256=dict(pilot=fingerprint),
                    counts=dict(development=dict(turns=377, eot=377, hold_spans=464, hold_turns=174),
                                pilot=cli._counts(turns)),
                    ids=dict(pilot=[t["id"] for t in turns]), vad=vad.metadata,
                    frozen_policies_sha256=frozen_bytes_hash,
                    policy_selection_performed=False, models={}, original_runtime_match={})
    report.write_json(output / "manifest.json", manifest)
    loaded = []
    try:
        rows, all_refs = [], None
        for name in args.models:
            detector = models.load_detector(name, cache_dir=str(args.model_cache_dir), precision="int8")
            loaded.append(detector)
            metadata = cli._runtime_model_metadata(detector)
            expected = frozen["models"][name]["model_settings"]
            validate_model_identity(name, metadata, expected)
            runtime_match = metadata == expected
            if args.require_original_runtime and not runtime_match:
                raise ValueError(f"{name}: 원본 실행환경과 다릅니다")
            manifest["models"][name] = metadata
            manifest["original_runtime_match"][name] = runtime_match
            report.write_json(output / "manifest.json", manifest)
            traces, refs, trace_hashes = cli._trace_set(turns, name, detector, vad, args.cache_dir,
                                                       metadata, "공개 파일럿")
            if all_refs is not None and refs != all_refs:
                raise ValueError("모델마다 서로 다른 참조를 받았습니다")
            all_refs = refs
            cli._export_pilot_traces(output, name, traces, refs, trace_hashes)
            summaries, outcomes = evaluate_model(name, traces, refs, frozen["models"][name])
            rows.extend(summaries)
            for budget, per_turn in outcomes.items():
                report.write_csv(output / "outcomes" / f"{name}-pilot-{int(budget * 100):02d}.csv", per_turn)
            if detector is not None:
                detector.close()
                loaded.pop()
        if report.sha256(args.frozen_policies) != frozen_bytes_hash:
            raise ValueError("평가 중 동결 정책이 변경되었습니다")
        report.write_json(output / "data.json", dict(data_sha256=dict(pilot=fingerprint), pilot=list(all_refs.values())))
        report.write_json(output / "frozen_policies.json", frozen)
        report.write_results(output, rows, manifest, frozen_bytes_hash)
        summary = read_json(output / "summary.json")
        summary["policy_selection_performed"] = False
        summary["original_runtime_match"] = manifest["original_runtime_match"]
        summary["caveats"].append("이번 명령은 포함된 동결 정책만 사용하며 개발셋 추론·정책 재선택을 하지 않습니다.")
        report.write_json(output / "summary.json", summary)
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        manifest["result_files_sha256"] = {str(p.relative_to(output)): report.sha256(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        report.write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report.write_json(output / "manifest.json", manifest)
        raise
    finally:
        for detector in loaded:
            close = getattr(detector, "close", None)
            if callable(close):
                close()


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"평가 완료: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
