# 베이스라인 코드

세 모델은 `evaluation/eot_stream/models.py`의 `load_detector`를 공통 진입점으로 사용합니다. 실제 스트리밍 실행은 `evaluation/run_baselines.py`, 공통 VAD는 `evaluation/eot_stream/stream.py`, 동결된 종료 정책 재생은 `evaluation/eot_stream/policy.py`에 있습니다.

| 이름 | 실제 구현 | 입력 | 사용한 버전 |
|---|---|---|---|
| 무음 타이머 (`timer`) | `load_detector`가 `None` 반환, 공통 VAD가 관찰한 무음 길이로 종료 | 인과적 Silero VAD 프레임 | 동결 정책의 대기시간 |
| Smart Turn (`smart-turn`) | `SmartTurn.predict` | 관찰된 PCM의 최근 8초, 부족하면 왼쪽 0 패딩 | Smart Turn v3.2 CPU INT8, revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` |
| LiveKit Mini (`livekit-mini`) | `LiveKitMini.predict` | 관찰된 PCM의 최근 1.2초를 PCM16으로 변환 | `livekit-local-inference==0.2.7` |

발화 텍스트, 참조 EOT·HOLD, 미래 음원은 모델 입력으로 전달하지 않습니다. 타이머도 주석의 무음 경계를 사용하지 않습니다. 세 모델은 동일한 160개 ID와 VAD 설정을 사용합니다. 모델 호출은 첫 무음 200ms부터 200ms 간격이며, 판단은 실제 32ms VAD 프레임 시각에서 일어납니다.

가중치와 네이티브 바이너리는 저장소에 포함하지 않습니다. 준비·실행 방법은 루트 `README.md`를 따릅니다. LiveKit 패키지와 모델의 조건은 서로 다르므로 `licenses/LIVEKIT_PACKAGE_LICENSE`와 `licenses/LIVEKIT_MODEL_LICENSE`를 함께 확인할 수 있습니다.
