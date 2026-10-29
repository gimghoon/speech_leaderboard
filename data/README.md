# 공개 한국어 평가 데이터

`pilot/`에는 실제 평가한 160개 원본 WAV, 동일한 주석 JSONL, 선별 로그와 파일별 SHA256 manifest가 있습니다. 별도 학습 데이터는 만들거나 배포하지 않습니다.

원본은 [PolyAI MInDS-14](https://huggingface.co/datasets/PolyAI/minds14/tree/40ce77cb32a384e4d50a568e1ec39ac804019d33)의 `ko-KR`, upstream split 이름은 `train`입니다. 이 프로젝트는 그 공개 자료를 학습하지 않고 평가 파일럿으로 사용했습니다. 원본 `train` 명칭은 이 저장소에 학습셋이 있다는 뜻이 아닙니다. 데이터 카드에는 `CC-BY-4.0`이 기록되어 있으며 보관한 원문은 `licenses/MINDS14_DATASET_CARD.md`에 있습니다.

한국어 자동 선별본 342개 중 EOT 시간·HOLD 구간·내용 잘림의 불확실성 상태가 `auto_consistent` 또는 `not_applicable`이고, 명시된 마지막 EOT 무음이 200ms 이상인 160개를 선택했습니다. 160개 EOT와 91개 HOLD 구간(54개 발화에 존재)이며 모든 모델의 분모가 같습니다. `pilot/selection.csv`는 342개 후보의 포함·제외 이유를 기록합니다. 원본 WAV 바이트를 바꾸지 않았고 8kHz 모노입니다. 실행 시 16kHz 인과적 FIR 변환을 하며 1.25ms 지연을 보존합니다.

주석은 Silero와 ASR 및 자동 텍스트 검토를 사용한 후보입니다. `label_type=pseudo`, `human_verified=false`, `gold=false`입니다. 자동 한국어 판별은 사람의 한국어 전용 검수가 아니며 EOT는 실제 대화 의도의 확정 정답이 아닙니다. 주석 안의 `evidence/`, `language_records.jsonl` 경로는 당시 상위 언어 판별 패키지의 참조입니다. 이 최소 재현 저장소에는 원시 판별 근거·전체 342개 음원이 포함되지 않습니다. 포함한 주석의 내용과 시각은 바꾸지 않았습니다.

개발셋은 [LiveKit EOT Bench 한국어 validation](https://huggingface.co/datasets/livekit/eot-bench-data/tree/ca9d98a9686b920a2d8c9eb984224ba9be74e4dd)의 377개 발화, 464개 HOLD입니다. 정책을 선택하는 데만 사용했으며 이 저장소에는 개발 음원이 없습니다. 개발·파일럿 ID와 원본 음원 해시의 비중복은 최초 실행에서 검증했고 그 결과와 해시는 `results/original_run_manifest.json` 및 `results/frozen_policies.json`에 보존했습니다.

MInDS-14 출처: Daniela Gerz, Pei-Hao Su, Razvan Kusztos, Avishek Mondal, Michal Lis, Eshan Singhal, Nikola Mrkšić, Tsung-Hsien Wen, Ivan Vulić. *Multilingual and Cross-Lingual Intent Detection from Spoken Data* (2021), [논문](https://arxiv.org/abs/2104.08524). 이 프로젝트의 변경점은 공개 부분집합 선별과 자동 EOT/HOLD·불확실성·언어 판별 주석 추가입니다.
