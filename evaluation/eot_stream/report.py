"""Transparent files and plain Korean reporting for the local streaming pilot."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing an empty results table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inference_stats(traces, model_name):
    calls = [float(f['inference_ms']) for frames in traces.values() for f in frames
             if model_name != 'timer' and f['is_score_event']]
    totals = [sum(float(f['inference_ms']) for f in frames) for frames in traces.values()]
    return dict(model_call_count=len(calls), model_compute_total_s=sum(totals) / 1000,
                model_call_compute_p50_ms=float(np.percentile(calls, 50)) if calls else None,
                model_call_compute_p95_ms=float(np.percentile(calls, 95)) if calls else None,
                turn_model_compute_p50_ms=float(np.percentile(totals, 50)),
                turn_model_compute_p95_ms=float(np.percentile(totals, 95)))


def write_results(output, rows, manifest, frozen_sha256):
    output = Path(output)
    summary = dict(protocol=manifest['protocol'], frozen_policies_sha256=frozen_sha256,
                   counts=manifest['counts'], model_settings=manifest['models'], rows=rows,
                   caveats=[
                       'Public Korean-only pilot; pseudo annotations; human_verified=false; not gold accuracy.',
                       'Reference labels and runtime common VAD use the same Silero weights: circular, not independent reference validation.',
                       'Common causal VAD timer; annotated EOT/HOLD never schedule model calls.',
                       'Controlled silence condition: observe until reference EOT+4s, append declared digital zeros only to inference buffers.',
                       'Delay uses first actual audio-clock stop; excludes computation and is not wall-clock end-to-end latency.',
                       'Conditional successful-turn delay must be read alongside all-turn premature/miss rates and 4000ms penalized mean delay.',
                       '377 development turns choose policies before any of the same 160 pilot IDs are scored; no pilot tuning.',
                   ])
    write_json(output / 'summary.json', summary)
    write_csv(output / 'summary.csv', rows)
    counts = manifest['counts']
    lines = [
        '한국어 EOT 연속 입력 파일럿 — 종료 후 통제된 무음 조건', '',
        '공개 원본과 자동 주석(pseudo)을 사용한 파일럿이다. human_verified=false이며 gold 정확도가 아니다.',
        '참조 EOT/HOLD 작성과 실행 VAD에 동일한 Silero 가중치를 사용했다. 정답의 모델 의존성과 순환성이 있어 독립적인 정답 검증으로 해석할 수 없다.',
        '무음 타이머도 공통 causal Silero VAD에서 관찰한 무음시간을 사용한다. 주석의 무음구간을 판단 타이머로 사용하지 않는다.',
        '모든 모델은 동일한 파일럿 ID를 평가한다. 개발셋에서 5%/10% 조기종료 예산과 미검출 5% 제약으로 정책을 선택하고 파일럿 추론 전에 동결했다.',
        f"개발셋: {counts['development']['turns']}개 EOT, {counts['development']['hold_spans']}개 HOLD. 파일럿: {counts['pilot']['turns']}개 EOT, {counts['pilot']['hold_spans']}개 HOLD.",
        '관찰종료는 참조 EOT+4초이다. 원본 끝 뒤 필요한 digital zero를 추론 버퍼에만 추가했고 원본 WAV는 보존했다.',
        '지연은 최초 실제 선언의 오디오 시계 기준이다. 모델 계산시간은 별도로 표시하며 실제 wall-clock 종단 지연을 뜻하지 않는다.',
        '정상종료 평균지연은 정상종료된 사례만의 조건부 평균이다. 조기종료·미검출은 각 4000ms로 부과한 전체 평균 손실과 항상 함께 읽어야 한다.',
        'compute 통계는 전체 trace 생성 중 모델 호출에 대한 CPU 경과시간이며 모델 로딩·VAD·입출력은 제외한다. 정책 최초 선언 뒤의 진단 trace 호출도 포함한다.',
        '',
        '모델 | 구분 | 개발 예산 | 개발 제약 충족 | 조기종료 | 미검출 | 정상종료 평균 ms | 전체 4초 손실 ms | 원본 끝 뒤 선언 | timeout | 호출 p50/p95 ms',
    ]
    fmt = lambda value: 'N/A' if value is None else f'{value:.2f}'
    for row in rows:
        lines.append(f"{row['model_label']} | {row['split']} | {row['budget']:.0%} | {row['dev_feasible']} | "
                     f"{row['premature_rate']:.2%} | {row['miss_rate']:.2%} | {fmt(row['mean_delay_success_ms'])} | "
                     f"{row['penalized_mean_delay_ms']:.2f} | {row['source_eof_stop_rate']:.2%} | {row['timeout_rate']:.2%} | "
                     f"{fmt(row['model_call_compute_p50_ms'])}/{fmt(row['model_call_compute_p95_ms'])}")
    lines.extend(['', '개발 제약 충족=false인 설정은 feasible 정책이 없어 제약 초과량 최소화 fallback으로 선택된 것이다.',
                  '원본 끝 뒤 선언 비율과 timeout 비율의 분모는 각 구분의 전체 turn 수이다. 실제 선택값과 모델 설정은 frozen_policies.json 및 manifest.json에 기록했다.',
                  '선택값·개별 최초 선언·HOLD 포함 여부·이유·실패는 outcomes/의 CSV, 원본 끝 뒤 추가 초수와 참조는 data.json, 파일럿 score trace와 해시는 traces/에서 확인할 수 있다.',
                  f'동결 정책 SHA256: {frozen_sha256}', ''])
    (output / 'report.txt').write_text('\n'.join(lines))
