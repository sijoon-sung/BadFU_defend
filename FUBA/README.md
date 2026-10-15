# 현재 코드

| 용도 | 파일 |
|---|---|
| 원 공격 학습 | `main.py`, `train_main.py`, `train_method_*.py`, `config.py`, `dataset.py` |
| 궤적 생성·검증 | `make_trajectories.sh`, `run_real_fuba.sh`, `trajectories.json`, `verify_trajectories.py` |
| 저장 이력 수집 | `collect_tension.py`, 공용 로더 `release_map.py` |
| 현재 탐지 비교 | `tension_audit.py`, `benchmark_tension.py`, `replay_tension.py`, `experiment_tension.py`, `report_detection.py` |
| 기존 FU 실행 지원 | `tension_control.py`, `request_geometry.py`, `large_geometry.py`, `experiment_tension_control.py`, `experiment_request_replay.py` |
| 기존 의존 코드 | `benign_departure.py`, `robust_fu.py`, 모델·학습·언러닝 하위 폴더 |
| 원시 결과 | `logs/` — 기존 경로 유지 |

실행은 [RUN](../RUN.md), 궤적의 사용 범위는 [궤적 상태](../docs/TRAJECTORIES.md)를 따른다. 현재 비교 기준은 v3와 `linear_release`다. 후속 설계 후보는 아직 구현하지 않았다.

이전 실험 코드는 [보관 폴더](../legacy/research_2026-10-15/README.md)로 옮겼다. 기존 `FUBA/legacy/` 역시 과거 코드다.
