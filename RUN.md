# 현재 실행 안내

먼저 [궤적 상태](docs/TRAJECTORIES.md)를 봅니다. 직접 검사한 로컬 파일은 무결성·집계 검사를 통과했습니다. 다음 확인은 기존 체크포인트의 평가 조건 대조이며, 전체 재학습을 요구하지 않습니다. 실행 중인 작업은 그대로 완료하고, 완료 후 이 정리 버전을 받습니다.

## 남긴 실행 경로

| 작업 | 경로 |
|---|---|
| FUBA 궤적 생성·검증 | `FUBA/make_trajectories.sh`, `FUBA/verify_trajectories.py` |
| BadFU 궤적 저장·재사용 | `BadFU_src/BadFU-main/request_pipeline.py` |
| 이력 점수 수집·기존 비교 | `FUBA/collect_tension.py`, `FUBA/experiment_tension.py` |
| 완료된 탐지 결과 읽기 | `FUBA/report_detection.py` |

세부 파일은 [현재 코드 목록](FUBA/README.md)에 있습니다. 기존 설정과 함수 내용은 바꾸지 않았습니다.

## 파일이 있는 컴퓨터에서 확인

FUBA 검증기는 재학습하지 않습니다. 파일 수·참여·집계를 확인하고 기존 모델을 평가합니다. 실행 PC의 `FUBA/checkpoints`와 데이터가 필요합니다. 아래 명령은 선택한 항목의 registry를 갱신하므로, DISLAB 파일을 가지고 있는 checkout에서 실행합니다.

```bash
python FUBA/verify_trajectories.py --only k20_paper k20_paper_cifar clean_paper --force --no_doc
```

검증기의 기존 `ok`에는 완화된 공격 조건이 포함될 수 있습니다. 또한 이 검증기의 ASR은 평균 생성기·eval 모드에서 측정하며, 학습 로그의 개별 생성기·train 모드 평가와 다릅니다. 위 명령을 반복하는 것만으로 평가 차이가 해소되지는 않습니다. ASR을 곧바로 궤적 생성 오류나 원본 공격 재현 여부로 해석하지 않습니다. [평가 차이의 근거](docs/EVIDENCE_AUDIT.md#궤적-판정-정정-2026-10-15)를 참고합니다. 파일이 없는 PC의 `missing`을 서버 파일 소실로 해석하지 않습니다.

BadFU 기존 궤적의 경로는 `artifacts/badfu/<run>/manifest.json`에 있습니다. `request_pipeline.load_tensor`는 SHA-256을, `load_event`는 클라이언트 수·집계 비중·저장 FedAvg를 검사합니다. 이번 직접 검사는 [JSON](FUBA/logs/trajectories/file_check_2026-10-15.json)에 기록했습니다. 오래된 임시 폴더 경로를 가진 manifest는 원본 파일과 함께 보존합니다.

## 코드 정리 확인

저장소 루트에서 실행합니다.

```bash
python -m unittest discover -s FUBA/tests -q
```

정리 후 현재 경로의 기존 테스트 76개가 통과했습니다. 과거 알고리즘에 딸린 테스트 3개 파일은 코드와 함께 보관했습니다. 테스트 통과를 탐지 성능으로 세지 않습니다.

완료된 `manifest.json`과 `scores.json`이 있을 때만 다음 판독기를 사용합니다.

```bash
python FUBA/report_detection.py --experiment FUBA/logs/tension_v3/pilot
```

`registry.json`은 이 판독기의 입력이 아닙니다. 정상 보정 수가 부족하면 FPR·탐지율은 `null`로 남깁니다.

이전 실행 명령은 [보관 RUN](docs/archive/cleanup_2026-10-15/RUN.md)에 있습니다. 이전 실행기 경로가 필요하면 해당 문서가 가리키는 정리 전 커밋을 별도 checkout으로 사용합니다.
