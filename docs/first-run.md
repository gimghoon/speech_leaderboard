# 처음 실행해보기: Smart Turn v3.2

이 가이드는 모델 실행 → 첫 종료 시각 생성 → 예측 JSON 작성 → 채점까지 직접 확인하는 예제입니다.
새로 학습하거나 설정을 튜닝하지 않고, 제공된 Smart Turn 베이스라인과 고정 정책을 재현합니다.
사람이 검수한 정답이 아닌, 자동 주석을 붙인 공개 파일럿 160개를 평가합니다.
프로젝트 개요는 [README](../README.md), JSON 규격은 [예측 파일 안내](submission-format.md)를 참고하세요.

현재는 공개 파일럿의 로컬 평가까지 지원합니다. 공식 제출·순위 등록과 새 모델을 연결하는 공통 실행 절차는 아직 제공하지 않습니다.

## 준비 조건

- Git과 Python 3.12
- 가상환경 생성과 `source` 명령을 사용할 수 있는 터미널
- 패키지·가중치를 처음 내려받을 때의 인터넷 연결

아래 명령은 macOS의 bash/zsh 기준입니다. 기존 실행 검증 환경은 macOS arm64와 Python 3.12.14입니다.
GPU나 유료 음성 API 키는 필요하지 않습니다. Windows의 가상환경 활성화 명령은 다릅니다.
명령은 같은 터미널 창에서 순서대로 실행하세요. 각 단계에서 오류가 나면 다음 단계로 넘어가지 말고 해당 오류를 먼저 확인하세요.

## 1. 저장소와 실행환경 준비

새 폴더에 저장소를 받습니다. 같은 이름의 폴더가 이미 있으면 다른 폴더 이름을 사용하세요.

```bash
git clone https://github.com/gimghoon/speech_leaderboard.git speech-leaderboard-check
cd speech-leaderboard-check

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r evaluation/requirements-run.txt
```

공통 실행환경에는 LiveKit 패키지도 포함되어 있습니다. 설치 가능 여부는 운영체제와 패키지가 제공하는 바이너리에 따라 달라질 수 있습니다.

## 2. 가중치 준비

```bash
python evaluation/prepare_models.py --smart-turn
```

공통 Silero VAD와 Smart Turn v3.2 CPU INT8 가중치를 내려받고 해시를 확인합니다.
`Silero 확인 완료`와 `Smart Turn 확인 완료`가 표시되면 다음 단계로 진행합니다.
다운로드는 최초 준비 과정이며, 아래 평가 명령은 오프라인 모드로 실행합니다.

## 3. 160개 음성을 실제 추론

매번 새 실행 폴더와 새 추론 캐시를 만듭니다. 기존 가중치 파일은 재사용해도 됩니다.

```bash
mkdir -p runs
EOT_FIRST_RUN="$(mktemp -d "$PWD/runs/smart-turn-check-XXXXXX")"

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python evaluation/run_baselines.py \
  --models smart-turn \
  --vad-path .cache/weights/silero_vad.onnx \
  --model-cache-dir .cache/models \
  --cache-dir "$EOT_FIRST_RUN/cache" \
  --output "$EOT_FIRST_RUN/result"
```

`smart-turn: 공개 파일럿 25/160`처럼 진행 상황이 표시됩니다. 마지막에 `평가 완료`가 나오면 완료 여부를 확인합니다.

```bash
python - "$EOT_FIRST_RUN/result/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["status"] == "complete", manifest
print("완료 상태:", manifest["status"])
print("파일럿:", manifest["counts"]["pilot"])
PY

cat "$EOT_FIRST_RUN/result/report.txt"
```

`status`는 `complete`, 파일럿은 160발화와 91 HOLD 구간이어야 합니다.
같은 추론 점수에 검증 기준 5%와 10%의 고정 정책을 각각 적용하므로 두 행의 결과가 생성됩니다.
새 추론 캐시를 사용했으므로 이 단계는 저장된 예측을 단순히 다시 채점하는 작업이 아닙니다.

## 4. 내 실행 결과를 예측 JSON으로 변환

현재 실행기는 개별 판정을 CSV로 저장합니다. 아래 코드는 방금 생성한 5% 정책의 CSV에서 첫 종료 시각을 읽습니다.
정답 EOT를 예측값으로 복사하지 않습니다.

```bash
python - "$EOT_FIRST_RUN/result" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / "outcomes/smart-turn-pilot-05.csv").open(
    encoding="utf-8", newline=""
) as handle:
    rows = list(csv.DictReader(handle))

submission = {
    "format_version": "eot-first-stop-v1",
    "model_name": "Smart Turn v3.2 - my run",
    "split": "pilot",
    "budget": 0.05,
    "predictions": [
        {
            "id": row["id"],
            "first_eot_s": float(row["decision_s"])
            if row["decision_s"] else None,
        }
        for row in rows
    ],
}
target = root / "predictions-05.json"
target.write_text(
    json.dumps(submission, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    encoding="utf-8",
)
print("예측 파일:", target)
PY
```

160개 ID마다 첫 종료 선언 시각 하나가 기록됩니다. 관찰 끝까지 선언하지 않았으면 `null`입니다.

## 5. 독립 채점

```bash
python evaluation/submission_score.py \
  --predictions "$EOT_FIRST_RUN/result/predictions-05.json" \
  --references results/references.json \
  --output "$EOT_FIRST_RUN/score-05.json"

python - "$EOT_FIRST_RUN/score-05.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
PY
```

기존 5% 정책의 기준 결과는 160발화, 조기 종료 7개, 미검출 0개, 정상 종료 지연 약 805.44ms, 전체 평균 손실 945.20ms입니다.
새 실행의 결과를 이 값과 비교하세요. 실행환경 차이로 결과가 다르면 모델·라이브러리·가중치 정보를 `manifest.json`에서 확인합니다.
예측 시각만으로는 timeout 이유를 판별할 수 없어 독립 채점기의 `timeout_rate`는 `null`입니다. 실행기의 보고서에는 실제 이유가 남습니다.

## 생성된 파일

실행 폴더는 `runs/smart-turn-check-XXXXXX/`처럼 만들어집니다. 정확한 경로는 아래 명령으로 확인합니다.

```bash
printf '%s\n' "$EOT_FIRST_RUN"
```

| 파일 | 내용 |
|---|---|
| `result/report.txt` | 읽기 쉬운 결과 보고서 |
| `result/summary.json`, `summary.csv` | 5%·10% 정책의 요약 지표 |
| `result/outcomes/` | 각 발화의 첫 종료 시각과 판정 |
| `result/predictions-05.json` | 방금 생성한 예측 파일 |
| `score-05.json`, `score-05.per_turn.csv` | 독립 채점 결과 |
| `result/manifest.json` | 완료 상태, 모델·가중치·환경 정보 |

## 그다음에 할 수 있는 일

- 실행과 점수가 맞는지 확인하려면 `report.txt`, `manifest.json`, `predictions-05.json`, `score-05.json`을 함께 보관하세요.
- 직접 만든 시스템의 예측 JSON이 있다면 같은 독립 채점기로 평가할 수 있습니다. 현재 베이스라인 실행기에 모델 이름만 바꾸어 새 모델을 연결할 수는 없습니다.
- 설정 선택과 모델 개선은 검증셋에서 진행해야 합니다. 공개 파일럿 점수를 보고 반복 조정한 결과는 같은 비교 조건으로 해석할 수 없습니다.
- 현재 프로젝트에는 공식 결과 접수처나 자동 순위 반영 절차가 없습니다. 이 가이드의 마지막 단계는 로컬 평가 완료입니다.

## 자주 막히는 부분

| 상황 | 확인할 내용 |
|---|---|
| `python3.12`를 찾을 수 없음 | Python 3.12 설치 여부와 터미널 경로 확인 |
| 패키지 설치 실패 | 가상환경 활성화, Python 버전, 운영체제·아키텍처 확인 |
| 오프라인 모드에서 모델을 찾지 못함 | 2단계 가중치 준비 완료 여부와 캐시 경로 확인 |
| 결과 폴더가 비어 있지 않다는 오류 | 3단계의 새 실행 폴더 생성 명령부터 다시 실행 |
| `$EOT_FIRST_RUN` 경로가 비어 있음 | 같은 터미널 창인지 확인하고 실제 실행 폴더 경로 사용 |
| JSON에 ID가 없거나 중복됐다는 오류 | 160개 전체 예측이 있는지 [파일 규격](submission-format.md) 확인 |
