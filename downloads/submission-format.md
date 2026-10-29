# 첫 EOT 예측 파일과 로컬 평가

아래 Python 도구로 로컬 예측 파일의 형식을 검증하고 점수를 계산합니다. 네트워크나 모델 호출은 없으며, 파일을 서버에 업로드하거나 공식 순위에 등록하지 않습니다.

이 문서의 파일 경로와 명령은 다운로드한 개인 저장소 패키지의 루트를 기준으로 합니다.

## 예측 JSON

UTF-8 표준 JSON을 사용합니다. 아래는 한 행의 형식을 보여주는 예시이며, 실제 파일에는 선택한 분할의 모든 ID가 정확히 한 번씩 있어야 합니다.

```json
{
  "format_version": "eot-first-stop-v1",
  "model_name": "내 모델",
  "split": "pilot",
  "budget": 0.05,
  "predictions": [
    {"id": "minds-ko-0003", "first_eot_s": 6.176}
  ]
}
```

| 필드 | 의미 |
| --- | --- |
| `format_version` | 반드시 `eot-first-stop-v1` |
| `model_name` | 비어 있지 않은 모델 이름 |
| `split` | 개발 분할 `development` 또는 파일럿 분할 `pilot` |
| `budget` | 개발 분할에서 정책을 고를 때 사용한 조기 종료 허용률, 숫자 `0.05` 또는 `0.1` |
| `predictions[].id` | 해당 정답 파일의 정확한 턴 ID |
| `predictions[].first_eot_s` | 오디오 시작을 0초로 한 첫 종료 선언 시각; 관찰 끝까지 선언하지 않았으면 `null` |

개발 분할은 377개, 파일럿 분할은 160개 예측을 요구합니다. 예측 순서는 자유롭고 결과는 정답 파일 순서로 반환됩니다. `budget`은 정책 선택 시의 설정값입니다. 파일럿에서 측정된 조기 종료율이 이 값 이하임을 뜻하지 않습니다.

선언 시각은 유한한 숫자여야 하며 `0 ≤ first_eot_s ≤ horizon_s`를 만족해야 합니다. 문자열 숫자, 음수, `NaN`, `Infinity`, 관찰 끝 이후 시각, 누락·중복·알 수 없는 ID를 거부합니다. 미선언 턴도 생략하지 말고 `null`로 포함하세요. `decision_s` 같은 다른 필드 이름이나 추가 메타데이터를 자동 변환하지 않습니다.

완전한 예시는 `results/baseline-{split}-{model}-{05|10}.json`에 있습니다. 제공 모델은 `timer`, `smart-turn`, `livekit-mini`입니다. 파일럿 예시는 `results/baseline-pilot-timer-05.json`입니다. 예산 접미사는 두 자리 `05` 또는 `10`입니다.

실제 정답에서 모든 ID를 가져와 빈 템플릿을 만들 수도 있습니다. 아래 결과는 모든 턴이 미선언인 시작 파일이므로, 모델의 실제 첫 선언 시각으로 값을 바꾸어 사용하세요.

```python
import json
from pathlib import Path

split = "pilot"
source = json.loads(Path("results/references.json").read_text(encoding="utf-8"))
template = {
    "format_version": "eot-first-stop-v1",
    "model_name": "내 모델",
    "split": split,
    "budget": 0.05,
    "predictions": [
        {"id": ref["id"], "first_eot_s": None}
        for ref in source[split]
    ],
}
Path("my-pilot.json").write_text(
    json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
```

## 실행과 결과

Python 3 표준 라이브러리만 필요합니다. 저장소 루트에서 실행하세요.

```sh
python3 evaluation/submission_score.py \
  --predictions my-pilot.json \
  --references results/references.json \
  --output work/my-pilot-score.json
```

성공하면 `work/my-pilot-score.json`과 `work/my-pilot-score.per_turn.csv`를 씁니다. 입력 파일을 결과 경로로 지정하면 거부합니다. `--predictions -`는 표준 입력을 읽고, `--output -`는 JSON만 표준 출력으로 내보냅니다. 검증 실패 시 종료 코드는 2이며 오류 JSON을 표준 오류로 출력합니다.

패키지 정답 입력은 `results/references.json`입니다. 이 파일은 최상위 `development`와 `pilot` 키에 각 정답 목록을 담으며 `refs`로 한 번 더 감싸지 않습니다. 평가기는 제출의 `split`에 해당하는 목록을 선택합니다. 원본 평가 결과의 `results/data.json` 또는 명시적인 정답 목록 JSON도 받습니다. 각 정답 행에는 `id`, `eot_s`, `hold_spans`, `horizon_s`, `original_duration_s`가 필요합니다. `horizon_s`는 EOT+4초여야 하며, HOLD는 `{ "start": 숫자, "end": 숫자 }` 목록입니다.

정답 행의 `split`은 `development`/`pilot`을 사용합니다. 원본의 `livekit_dev`/`minds_pilot`도 각각 같은 분할로 인식합니다. 표시된 분할이 제출과 다르면 거부합니다. 분할 표시가 없는 독립 정답 목록도 허용하지만 결과의 `metadata.reference_split_verified`는 `false`입니다. 일부 행에만 분할이 있는 목록은 거부합니다. 패키지 정답은 모든 행에 분할이 표시되어 있으므로 이 값이 `true`입니다.

브라우저와 CommonJS에서 같은 API를 제공합니다.

```javascript
const EOTScore = require("./evaluation/score-core.js");
const result = EOTScore.validateAndScore(submission, references);
// result = { metadata, metrics, per_turn }
```

Python에서 가져와 쓸 때의 함수 이름은 `validate_and_score(submission, refs)`입니다. 두 구현은 입력 객체를 변경하지 않습니다. 검증 오류는 `ValidationError`이며 `code`, `path`, 읽을 수 있는 `message`를 제공합니다. CLI 오류는 다음과 같은 형태입니다.

```json
{"error":{"code":"MISSING_ID","path":"submission.predictions","message":"submission.predictions: 예측이 빠진 ID입니다: minds-ko-0003. 미선언도 null로 포함하세요."}}
```

## 점수 정의

첫 선언만 사용하며, 이후 선언으로 조기 종료를 수정하지 않습니다. 선언 시각이 `eot_s - 1e-9`보다 이르면 `premature`, `null`이면 `missed`, 그 외에는 `correct`입니다. 경계의 부동소수점 오차만 1ns 허용하며 정답 직전의 허용된 시각은 지연 0ms로 처리합니다. 관찰 끝 이후 시각에는 이 허용 오차를 적용하지 않습니다.

| 지표 키 | 정의 |
| --- | --- |
| `n_turns`, `n_premature`, `n_missed`, `n_correct` | 전체·조기 종료·미종료·성공 턴 수 |
| `premature_rate`, `miss_rate` | 각 실패 턴 수 ÷ 전체 턴 수 |
| `mean_delay_success_ms` | 성공 턴에서만 `max(0, first_eot_s - eot_s) × 1000`의 평균; 성공이 없으면 `null` |
| `penalized_mean_delay_ms` | `(성공 지연 합 + 4000 × (조기 종료 수 + 미종료 수)) ÷ 전체 턴 수` |
| `source_eof_stop_rate` | 첫 선언이 원본 WAV 길이보다 뒤인 턴 수 ÷ 전체 턴 수 |
| `n_hold_turns`, `n_hold_spans` | HOLD가 있는 턴 수와 전체 HOLD 구간 수 |
| `hold_turn_premature_rate` | HOLD가 있는 턴 중 조기 종료한 비율; HOLD 턴이 없으면 `null` |
| `timeout_rate` | 예측 시간만으로 선언 원인을 알 수 없으므로 `null` |

턴별 결과는 원본 평가기와 같은 `id`, `eot_s`, `decision_s`, `reason`, `outcome`, `delay_ms`, `after_source_eof`, `has_hold`, `first_decision_in_hold` 필드를 사용합니다. `reason`은 확인할 수 없어 `null`이며, HOLD 안의 선언은 `start ≤ time < end`로 표시합니다. 성공 지연 외의 `delay_ms`는 `null`입니다.

이 검증은 파일의 완전성과 주어진 시각의 점수를 확인합니다. 오디오를 미래까지 보지 않았는지, 개발 분할만으로 정책을 골랐는지, 계산 비용이나 타임아웃 원인이 무엇인지는 확인하지 않습니다. 지연은 오디오 시간 기준이며 모델 실행 시간을 포함하지 않습니다. 원본 WAV 뒤의 제어된 무음 구간에서 선언할 수 있으므로 `source_eof_stop_rate`도 함께 확인하세요. 참조 EOT/HOLD는 설명된 의사 정답이며 독립적인 수작업 골드 정답을 뜻하지 않습니다.

## 재현 검증

제공된 예측 파일로 평가기 동작을 바로 확인할 수 있습니다. 이 명령은 모델을 다시 실행하지 않습니다.

```sh
python3 evaluation/submission_score.py \
  --predictions results/baseline-pilot-timer-05.json \
  --references results/references.json \
  --output work/baseline-pilot-timer-05-score.json
```

이 예시의 결과는 전체 160턴, 조기 종료 5턴, 미종료 0턴, 성공 155턴이며, 성공 조건부 평균 지연은 약 931.923ms, 4초 오류 벌점을 포함한 전체 평균은 1027.8ms입니다.

패키지 README의 의존성 설치를 마친 환경에서는 보존된 음원·정답·정책·trace·원본 결과도 함께 검증할 수 있습니다.

```sh
python3 evaluation/verify_snapshot.py
```

이 검증은 3개 모델 × 2개 개발 예산의 파일럿 결과 6개, 턴별 결과 960행을 원본 `results/summary.json` 및 `results/outcomes/`와 대조합니다. 네트워크나 모델 추론을 사용하지 않습니다.

별도로 원본 사이트 소스의 단위 테스트에서는 개발·파일럿 두 분할의 결과 12개, 총 3,222개 첫 선언을 다시 평가하여 재계산 가능한 모든 지표를 대조했고 JavaScript/Python 결과의 일치를 확인했습니다. 이 단위 테스트용 파일은 개인 저장소 패키지에 포함하지 않습니다. 예측 시간만 담은 파일로는 타임아웃 원인·모델 계산 통계를 복원할 수 없습니다.
