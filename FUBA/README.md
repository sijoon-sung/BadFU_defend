# 현재 코드

| 용도 | 파일 |
|---|---|
| 원 공격 학습 | `main.py`, `train_main.py`, `train_method_*.py`, `config.py`, `dataset.py` |
| 궤적 생성·검증 | `make_trajectories.sh`, `run_real_fuba.sh`, `trajectories.json`, `verify_trajectories.py` |
| 저장 이력 수집 | `collect_tension.py`, 공용 로더 `release_map.py` |
| 현재 탐지 비교 | `tension_audit.py`, `benchmark_tension.py`, `replay_tension.py`, `experiment_tension.py`, `report_detection.py` |
| 이력–요청 기여 일관성(HRAC) | `hrac.py`, `experiment_hrac.py` — 제약 최적화, 기존 궤적 재사용, 정상 실행 보정 |
| BadFU 관측 비교 | `experiment_hrac_views.py` — 깨끗한 입력·진단 트리거·고정 무작위 패턴과 정상 기준 방향 비교 |
| 라벨 없는 BadFU 관측 | `experiment_hrac_unlabeled.py` — 이미지 폴더/픽셀 전용 입력, 무작위 선별, 단순 기준선, 라운드별 저장·재시작 |
| 관측 비교 후속 확인·그래프 | `validate_hrac_views.py`, `plot_hrac_views.py` — 분리한 이미지·새 패턴, 전체 클라이언트 점수 그림 |
| 같은 관측의 점수 기준선 | `compare_hrac_views.py` — HRAC·단순 음의 기여·절댓값·마지막 라운드를 같은 캐시에서 비교 |
| 기존 FU 결과 연결 | `diagnose_hrac_badfu.py` — 저장 모델과 이력 캐시의 출처 및 ASR 재평가 |
| 기존 FU 실행 지원 | `tension_control.py`, `request_geometry.py`, `large_geometry.py`, `experiment_tension_control.py`, `experiment_request_replay.py` |
| 기존 의존 코드 | `benign_departure.py`, `robust_fu.py`, 모델·학습·언러닝 하위 폴더 |
| 원시 결과 | `logs/` — 기존 경로 유지 |

실행은 [RUN](../RUN.md), 궤적의 사용 범위는 [궤적 상태](../docs/TRAJECTORIES.md)를 따른다. HRAC는 기존 v3와 `linear_release`를 비교 기준으로 유지한다. [알고리즘과 결과](../docs/RESEARCH_SUMMARY.md)에 관측 조건·수식·남은 실패를 기록했다. 기존 파일명은 실행 호환성을 위해 유지한다.

이전 실험 코드는 [보관 폴더](../legacy/research_2026-10-15/README.md)로 옮겼다. 기존 `FUBA/legacy/` 역시 과거 코드다.
