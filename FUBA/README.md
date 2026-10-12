# FUBA/ 스크립트 지도

FUBA 공격 코드(원 저자) 위에 우리 방어·비교군·러너를 얹은 폴더. 원 저자 README 는 `README_FUBA_upstream.md`.

## 러너
| 파일 | 역할 |
|---|---|
| `run_paper.sh` | 논문 실험 전체 (RQ1–RQ6). `RQ="1 3"` 처럼 일부 선택 |
| `run_baselines.sh` | 비교군만 (exp/baselines 브랜치용) |
| `run_all.sh` | 단계 단위 실행기 (위 둘이 호출). `STAGES=`, `CONDS=`, `SEEDS=`, `FORCE=1`, `DRY=1` |
| `run_real_fuba.sh` | FUBA 공격 궤적 생성 (mpiexec) |
| `resume_local.bat` | 로컬 GPU 에서 끊긴 BadFU 단계 이어 돌리기 |
| `paper_tables.py` | 결과 → `docs/paper/tables_{cond}.md` |

## 우리 방법 (docs/paper/METHOD.md)
| 파일 | 단계 |
|---|---|
| **`unified_fu.py`** | **한 파일로 전체**: 궤적 지표 → 빼 보기 확정 → 해독제 정화. 결정·수치·비용 기록 |
| `traj_indicators.py` | A 궤적 지표: 효과 벡터 (크기·동일부호·일관성·반대 짝) |
| `dose_fu.py` | B 확정: 빼 보기 (같은 양 모집단 비교, 단계·멈춤, 수정). 탐침 = 공개 + 지울 입력 |
| `antidote.py` | C 정화: 운반자 몫의 백도어 방향 성분만 제거 |
| `pair_roles.py` | 역할 지도: 운반자 집합·공동 위장자 |
| `schedule_fu.py` | 순차 요청 (여러 위장자), 전체 예산 띠 |
| `attack_sim.py` | 적응형 공격 변형 (저장 몫만 조작, 학습 없음) |

## 비교군
| 파일 | 방법 |
|---|---|
| `benign_departure.py` | FU 연산자 4 종 (빼기, 증류, FedEraser, 재학습) |
| `robust_fu.py`, `unlearn_method/federaser_faithful.py` | FedEraser-R (이전 제안), 원식 FedEraser |
| `baseline_masa.py` | MASA (proxy) |
| `baseline_subid.py` | SubID 식 부분공간 잔차 (proxy) |
| `latent_risk.py` | 가중치 전용 지도 |
| `release_map.py` | 공용 부품(로더·앙상블·탐색) + 이전 탐지 판(ρ 스윕 지도, 옵션 단계 `release`) |
| `bench_release_map.py` | 비용 벤치 |

## 집계
`aggregate_*.py` — 단계별 시드 평균 표 (`logs/*/summary_{cond}.md`).

## 원 저자 코드
`main.py`, `train_main.py`, `train_method_*.py`, `unlearn.py`, `unlearn_method/`, `utils/`, `config.py`, `dataset.py`, `model.py`, `sql.py`. 우리가 고친 곳: 일관된 데이터 분할(`utils/comm_utils.make_client_split`), 증류 나눗셈, FedEraser 래퍼, `--clients_ratial`, `--split_seed`.

## legacy/
초기 파이프라인(det1~3)과 그때의 비교·정화 스크립트, 진행형 FU(progressive_fu, dose_fu 로 대체). 지금 러너는 쓰지 않는다.
