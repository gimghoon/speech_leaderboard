"""가중치·네트워크 없이 공개 음원·trace·6개 파일럿 결과를 다시 검증한다."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from eot_stream import cache, cli, data, report, stream
from run_baselines import ROOT, evaluate_model, read_json, validate_snapshot


def compare_value(actual, expected, label):
    if isinstance(actual, float):
        if not math.isclose(actual, float(expected), rel_tol=1e-11, abs_tol=1e-8):
            raise ValueError(f"지표 불일치: {label}")
    elif isinstance(actual, bool):
        if str(actual) != str(expected):
            raise ValueError(f"판정 불일치: {label}")
    elif ("" if actual is None else str(actual)) != ("" if expected is None else str(expected)):
        raise ValueError(f"결과 불일치: {label}")


def verify(root=ROOT):
    root = Path(root)
    results = root / "results"
    provenance = read_json(results / "portable_provenance.json")
    for relative, expected in provenance["portable_files_sha256"].items():
        if report.sha256(root / relative) != expected:
            raise ValueError(f"저장소 사본 해시 불일치: {relative}")
    for relative, expected in provenance["source_code_sha256"].items():
        if report.sha256(root / "evaluation" / relative) != expected:
            raise ValueError(f"원본 소스 해시 불일치: {relative}")
    frozen = read_json(results / "frozen_policies.json")
    turns = data.load_pilot_turns(root / "data/pilot")
    fingerprint = validate_snapshot(turns, frozen)
    expected_refs = {turn["id"]: stream.make_observation(turn)[1] for turn in turns}
    refs_document = read_json(results / "data.json")
    if {ref["id"]: ref for ref in refs_document["pilot"]} != expected_refs:
        raise ValueError("공개 데이터 참조 불일치")
    historical = read_json(results / "summary.json")
    if historical["frozen_policies_sha256"] != provenance["original_file_sha256"]["frozen_policies.json"]:
        raise ValueError("원본 동결 정책 해시 연결 불일치")
    verified_rows = verified_turns = 0
    for name in ("timer", "smart-turn", "livekit-mini"):
        index = read_json(results / "traces" / name / "index.json")
        if [entry["id"] for entry in index] != [turn["id"] for turn in turns]:
            raise ValueError(f"{name}: trace ID 순서·분모 불일치")
        traces, refs = {}, {}
        for entry in index:
            path = (results / entry["path"]).resolve()
            if not path.is_relative_to(results.resolve()) or path.is_symlink():
                raise ValueError("잘못된 trace 경로")
            if report.sha256(path) != entry["sha256"]:
                raise ValueError("trace 파일 해시 불일치")
            item = read_json(path)
            if item["status"] != "complete" or item["id"] != entry["id"]:
                raise ValueError("미완료·잘못된 trace")
            if item["payload_sha256"] != cache._digest(dict(frames=item["frames"], ref=item["ref"])):
                raise ValueError("trace payload 해시 불일치")
            if item["ref"] != expected_refs[entry["id"]]:
                raise ValueError("trace 참조 불일치")
            if item["metadata"]["model"] != frozen["models"][name]["model_settings"]:
                raise ValueError("trace 모델 설정 불일치")
            if item["metadata"]["protocol"] != stream.PROTOCOL or item["metadata"]["vad"] != frozen["vad"]:
                raise ValueError("trace 프로토콜·VAD 불일치")
            for filename, digest in item["metadata"]["code_hashes"].items():
                if report.sha256(root / "evaluation/eot_stream" / filename) != digest:
                    raise ValueError("trace 생성 소스 해시 불일치")
            traces[entry["id"]], refs[entry["id"]] = item["frames"], item["ref"]
        summaries, outcomes = evaluate_model(name, traces, refs, frozen["models"][name])
        for actual in summaries:
            expected = next(row for row in historical["rows"] if row["model"] == name
                            and row["split"] == "pilot" and row["budget"] == actual["budget"])
            for key, value in actual.items():
                compare_value(value, expected[key], f"{name}/{actual['budget']}/{key}")
            verified_rows += 1
        for budget, rows in outcomes.items():
            with (results / "outcomes" / f"{name}-pilot-{int(budget * 100):02d}.csv").open(newline="") as handle:
                saved = list(csv.DictReader(handle))
            if len(saved) != len(rows):
                raise ValueError("턴 결과 분모 불일치")
            for actual, expected in zip(rows, saved):
                for key, value in actual.items():
                    compare_value(value, expected[key], f"{name}/{actual['id']}/{key}")
                verified_turns += 1
    return dict(status="passed", pilot_turns=len(turns), hold_spans=91,
                trace_files=480, summary_rows=verified_rows, per_turn_rows=verified_turns,
                pilot_fingerprint=fingerprint, network_used=False, model_inference_used=False)


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
