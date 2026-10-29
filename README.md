# 한국어 발화 종료 감지 평가 · EOT 리더보드

김지훈 · 202220760

- 개인 저장소: [gimghoon/speech_leaderboard](https://github.com/gimghoon/speech_leaderboard)
- 리더보드 사이트: [EOT Bench](https://gimghoon.github.io/speech_leaderboard/?v=26b37b0)

한국어 음성의 발화가 끝났는지 판단하는 세 베이스라인을 같은 연속 입력 조건에서 비교합니다. 실제 평가한 **160개 음원·160개 EOT·91개 HOLD**, 베이스라인 코드, 평가 코드, 동결 정책과 턴별 결과를 이 저장소에서 함께 확인하고 재현할 수 있습니다. 별도 학습 데이터는 사용하지 않았습니다.

공개 음원에서 자동 생성한 주석을 사용한 **공개 파일럿**입니다. 사람 검증 정답, 비공개 테스트, 공식 챌린지 성적을 뜻하지 않습니다. 참조 주석과 실행 VAD가 동일한 Silero 가중치를 사용하므로 독립적인 정답 검증으로 해석할 수 없습니다.

## 저장소 구성

| 위치 | 내용 |
|---|---|
| `baseline/README.md` | 무음 타이머·Smart Turn·LiveKit Mini의 실제 어댑터 위치와 입력 조건 |
| `evaluation/eot_stream/` | 실제 실행에 사용한 원본 평가 패키지. 수정 없이 보존 |
| `evaluation/run_baselines.py` | 포함된 160개 스냅샷을 직접 읽고 기존 동결 정책만 적용하는 재현 진입점 |
| `evaluation/verify_snapshot.py` | 가중치·네트워크 없이 음원 해시와 저장된 trace를 검증하고 결과 재계산 |
| `evaluation/tests/` | 원본 95개 테스트와 휴대 가능한 실행기용 5개 테스트 |
| `data/pilot/audio/` | 실제 평가한 160개 8kHz 모노 원본 WAV |
| `data/pilot/annotations.jsonl` | 자동 EOT/HOLD·불확실성·한국어 선별 주석 |
| `data/pilot/selection.csv`, `manifest.json` | 342개 후보의 선별 이유와 파일 해시·출처 |
| `results/summary.json`, `summary.csv` | 개발·파일럿, 두 개발 예산, 세 모델의 실제 지표 12행 |
| `results/frozen_policies.json` | 377개 한국어 개발 발화로 먼저 선택한 5%·10% 정책 |
| `results/outcomes/` | 모델·구분·예산별 최초 선언과 결과 CSV |
| `results/traces/` | 실제 파일럿 score trace 480개와 SHA256 인덱스 |
| `results/data.json` | 실제 개발·파일럿 참조와 원본 종료 뒤 추가한 무음 길이 |
| `results/portable_provenance.json` | 원본 해시와 저장소 사본의 경로 비식별화 내역 |
| `results/verification.json` | 테스트·전체 trace 재생·160개 타이머 실행 대조 기록 |
| `licenses/` | 데이터 카드, 모델·패키지 조건, 출처 |

## 빠른 확인: 모델 다운로드 없이 실제 결과 재계산

Python 3.12 환경에서 실행합니다. 아래 설치는 검증에 필요한 수치·음원 라이브러리만 준비합니다. 외부 음성 API를 호출하지 않습니다.

```bash
git clone https://github.com/gimghoon/speech_leaderboard.git
cd speech_leaderboard
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r evaluation/requirements-verify.txt
python evaluation/verify_snapshot.py
```

성공 시 `status: passed`, 음원 160개, HOLD 91개, trace 480개, 파일럿 요약 6행, 턴 결과 960행이 표시됩니다. 모든 모델·예산의 실제 저장 결과를 검증하며 추론을 다시 수행하지 않습니다. 이미 저장된 결과가 아니라 새 모델의 성능을 측정하려면 별도의 예측을 생성해야 합니다.

테스트는 다음처럼 실행합니다.

```bash
cd evaluation
python -m pytest -q
cd ..
```

## 로컬 베이스라인 재실행

전체 추론 환경은 최초 실행에서 사용한 버전을 고정했습니다. LiveKit 모델에는 패키지와 별도 이용 조건이 있으며 `licenses/LIVEKIT_MODEL_LICENSE`에 원문을 보존했습니다. 저장소에는 가중치·네이티브 바이너리가 없습니다.

```bash
python -m pip install -r evaluation/requirements-run.txt
python evaluation/prepare_models.py --smart-turn
```

`prepare_models.py`는 공식 Silero의 고정 revision과 Smart Turn의 고정 revision에서 가중치를 내려받고 SHA256을 검사합니다. 기본 VAD 경로는 `.cache/weights/silero_vad.onnx`, Smart Turn 캐시는 `.cache/models`입니다. LiveKit Mini 어댑터는 설치된 `livekit-local-inference==0.2.7`을 사용합니다. 학습이나 외부 음성 API 호출은 없습니다.

가장 작은 실행은 실제 공통 VAD와 무음 타이머입니다.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python evaluation/run_baselines.py \
  --models timer \
  --vad-path .cache/weights/silero_vad.onnx \
  --output runs/timer
```

세 모델을 같은 스냅샷·동결 정책으로 실행하려면 다음 명령을 사용합니다. 모델을 준비한 뒤 오프라인으로 실행할 수 있습니다.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python evaluation/run_baselines.py \
  --models timer smart-turn livekit-mini \
  --vad-path .cache/weights/silero_vad.onnx \
  --output runs/all-baselines
```

매번 비어 있는 `--output` 폴더를 지정합니다. 중단 후에는 새 출력 폴더로 실행하면 해시가 같은 trace 캐시를 재사용합니다. `--cache-dir`, `--model-cache-dir`, `--pilot-root`, `--frozen-policies`로 경로를 바꿀 수 있습니다. 주어진 스냅샷은 동결 당시의 160개 음원·참조 fingerprint와 같아야 합니다. 5%·10% 정책을 모두 적용하며 파일럿에서 정책을 다시 선택하지 않습니다. `manifest.json`의 `status`가 `complete`인 실행만 완료된 결과입니다.

실행환경·가중치·VAD·원본 PCM과 참조의 해시를 기록합니다. LiveKit 네이티브 바이너리 해시도 실제 설치 파일에서 계산합니다. 다른 운영체제의 바이너리는 원본과 다를 수 있어 `original_runtime_match`에 표시하며, 원본 실행환경까지 엄격하게 확인하려면 `--require-original-runtime`을 추가합니다. 계산시간은 장비·실행환경에 따라 달라집니다.

원본 `python -m eot_stream.cli`는 342개 선별 원본과 별도의 개발 캐시가 필요한 최초 정책 선택용 명령입니다. 이 저장소의 기본 재현 명령은 `evaluation/run_baselines.py`이며, 필요한 160개 파일럿을 `data.load_pilot_turns`로 바로 읽습니다.

## 실제 파일럿 결과

아래 예산은 **개발셋에서 정책을 고를 때의 조기종료 상한**입니다. 파일럿에 같은 비율을 보장하는 값은 아닙니다. 모든 행은 같은 160개 발화가 분모이며 미검출은 0개입니다.

| 개발 예산 | 모델 | 조기종료 | 미검출 | 정상종료 평균 지연 | 전체 4초 손실 |
|---|---|---:|---:|---:|---:|
| 5% | Smart Turn v3.2 CPU INT8 | 7/160 · 4.38% | 0/160 | 805.44ms | 945.20ms |
| 5% | LiveKit v1-mini | 7/160 · 4.38% | 0/160 | 866.93ms | 1004.00ms |
| 5% | 무음 타이머 | 5/160 · 3.13% | 0/160 | 931.92ms | 1027.80ms |
| 10% | LiveKit v1-mini | 10/160 · 6.25% | 0/160 | 439.04ms | 661.60ms |
| 10% | Smart Turn v3.2 CPU INT8 | 11/160 · 6.88% | 0/160 | 482.36ms | 724.20ms |
| 10% | 무음 타이머 | 8/160 · 5.00% | 0/160 | 804.00ms | 963.80ms |

정상종료 지연은 정상종료한 발화에 한정한 평균입니다. 조기종료 또는 미검출 하나마다 4000ms를 부과하는 전체 평균 손실, 두 오류율, 원본 종료 뒤 선언 비율, timeout 비율을 함께 읽어야 합니다. 원본 정밀도는 `results/summary.json`에 있습니다.

LiveKit v1-mini의 검증 기준 5% 설정은 160개 중 125개(78.1%)를 1초 timeout으로 종료했습니다. 표의 점수는 이 제한 대기를 포함한 고정 정책의 성능입니다.

## 평가 조건과 한계

- 개발은 고정 revision의 LiveKit 한국어 validation 377개 EOT·464개 HOLD입니다. 이 개발셋에서 조기종료 5% 또는 10%, 미검출 5% 제약 아래 전체 손실이 가장 작은 정책을 선택한 뒤 파일럿 추론 전에 동결했습니다. 최초 실행에서 개발·파일럿 ID와 원본 음원 해시 중복을 거부했습니다. 이 저장소는 개발 음원 없이 동결 정책을 재사용합니다.
- 파일럿은 PolyAI MInDS-14 `ko-KR` 공개 자료에서 선별한 160개입니다. 상위 데이터셋의 split 이름은 `train`이지만 이 프로젝트는 해당 자료로 학습하지 않았습니다. 원본·부분집합 선별·자동 주석에 관한 설명은 `data/README.md`에 있습니다.
- `pseudo`, `human_verified=false`, `gold=false`입니다. 자동 한국어 판별과 텍스트 검토는 사람의 음성·대화 의도 검증을 대체하지 않습니다. 참조 주석과 실행 VAD가 같은 Silero 모델을 공유해 평가 편향이 있습니다.
- 모델은 관찰된 PCM 접두부만 받습니다. 16kHz·512샘플(32ms) 공통 VAD를 사용하며 무음 확인 96ms, 최초 모델 점수 200ms, 이후 200ms 간격입니다. 타이머도 실제 VAD가 관찰한 무음 길이로 작동합니다. EOT/HOLD 주석으로 호출을 예약하지 않습니다.
- 관찰은 참조 EOT+4초에 끝납니다. 원본 종료 뒤 필요한 무음은 추론 버퍼에만 digital zero로 붙입니다. 160개 합계 495.53525초가 추가되었으며 원본 WAV는 보존했습니다. 이 통제된 종료 후 무음 조건을 일반적인 실제 대화 스트리밍 성능으로 확대해서 해석할 수 없습니다.
- 지연은 최초 종료 선언의 **오디오 시계** 기준입니다. 계산·네트워크·VAD·입출력을 포함한 실제 음성 비서 전체 응답시간이 아닙니다. 모델 계산시간은 별도 통계이며 정책 최초 선언 뒤의 진단 trace 호출도 포함합니다.
- 8kHz 파일럿을 인과적 FIR로 16kHz로 변환하면서 1.25ms 필터 지연을 남겼습니다. 최종 불완전 프레임을 미래 샘플로 채우지 않습니다.

## 출처·무결성

MInDS-14 데이터 카드에는 CC-BY-4.0이 기록되어 있으며 음원은 원본 바이트를 유지했습니다. 이 프로젝트의 변경은 부분집합 선별과 자동 종료·불확실성·언어 판별 주석 추가입니다. 저자·논문과 보관한 원문은 `data/README.md`, `licenses/ATTRIBUTION.md`에 있습니다.

휴대 가능한 저장소 사본에서 개인 절대경로만 설명용 상대경로로 바꿨습니다. `results/summary.json`의 동결 정책 SHA256은 **최초 실행 파일**의 해시입니다. 경로를 정리한 현재 사본의 해시와 원본 해시는 `results/portable_provenance.json`에서 구분합니다. 정책 값·개발 지표·모델 설정·원본 음원·턴별 결과·파일럿 trace는 그대로 보존했습니다. 포함하지 않은 개발 trace 경로는 `unbundled-development-cache/`로 표기합니다.

2026-10-29 검증에서 테스트 100개, 전체 파일럿 trace 480개 재생, 요약 6행·턴 결과 960행 대조가 통과했습니다. 실제 고정 Silero를 로드하고 기존의 해시 검증 캐시를 사용한 160개 타이머 실행에서도 두 요약 행과 턴 결과 320행이 원본과 일치했습니다. 이 과정에서 Smart Turn·LiveKit Mini 추론이나 정책 튜닝을 다시 수행하지 않았습니다.

## 새 예측의 형식 검사와 점수 계산

`docs/submission-format.md`의 JSON 형식으로 각 ID의 최초 종료 시각을 제출합니다. 전체 ID가 필요하며 관찰 끝까지 종료하지 않았으면 `null`로 표시합니다. 실제 음원으로 모델을 실행해 만든 예측을 사용해야 합니다.

```bash
python evaluation/submission_score.py \
  --predictions results/baseline-pilot-timer-05.json \
  --references results/references.json \
  --output runs/example-score.json
```

위 예시는 보관된 타이머 예측을 다시 채점합니다. 이 점수기는 표준 라이브러리만 사용하며 모델 추론을 수행하지 않습니다. 예측 JSON의 점수 재계산은 해당 모델의 신규 추론이나 비공개 평가 완료를 의미하지 않습니다.
