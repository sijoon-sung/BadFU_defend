> 정리 이전 FUBA 안내입니다. 상대 경로는 [원래 위치](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/FUBA/README.md)에서 확인합니다.

# FUBA/ 스크립트 지도

FUBA 공격 코드(원 저자) 위에 우리 방어·비교군·러너를 얹은 폴더. 원 저자 README 는 `README_FUBA_upstream.md`.

## 러너
| 파일 | 역할 |
|---|---|
| `run_paper.sh` | 논문 실험 전체 (RQ1–RQ6). `RQ="1 3"` 처럼 일부 선택 |
| `run_baselines.sh` | 비교군만 (exp/baselines 브랜치용) |
| `run_all.sh` | 단계 단위 실행기 (위 둘이 호출). `STAGES=`, `CONDS=`, `SEEDS=`, `FORCE=1`, `DRY=1` |
| `run_real_fuba.sh` | FUBA 공격 궤적 생성 (mpiexec; 저자 main.py). `PARTICIPANT_RATE=` 로 참여율 |
| `trajectories.json` | **궤적 매니페스트**: 만들 궤적 전부(이름·flag·역할·문턱·예상 시간), flag 출처, 공식 코드 출처, 안 만드는 변형 TODO |
| `make_trajectories.sh` | 매니페스트의 FUBA 궤적을 없는 것만 만들고 하나마다 검증 (`DRY=1`, `ONLY=그룹|이름`, `FORCE=1`, `WHERE=`). BadFU 는 `../BadFU_src/BadFU-main/make_badfu_trajectories.sh` |
| `verify_trajectories.py` | 궤적 검증기 (파일·K·참여·집계 규칙·잠복/제거 ACC·ASR) → `logs/trajectories/registry.json`, `docs/EXPERIMENT_LOG.md` |
| `resume_local.bat` | 로컬 GPU 에서 끊긴 BadFU 단계 이어 돌리기 |
| `paper_tables.py` | 결과 → `docs/paper/tables_{cond}.md` |

## 우리 방법 (docs/paper/METHOD.md)
| 파일 | 단계 |
|---|---|
| `structural_guard.py` | 잠복 구조 공통 코어: 선택적 상쇄 관찰 → 누적 요청 검사 → 귀속 → 복구 callback → 독립 검증. 공격 이름·정답 역할 없이 동작. [설계·범위](../docs/paper/STRUCTURAL_GUARD.md) |
| `run_structural_guard.py` | NumPy 캐시 재생 / FUBA·BadFU 체크포인트 감사. 근사 감산 어댑터이며 삭제 완료를 인증하지 않음 |
| `existing_guard_components.py` | 기본 경로: DOSE의 기존 비교 함수와 TRACE의 기존 선택 함수를 공통 제어 계층에 연결 |
| `paired_audit.py` | 실험 확장: TRACE 후보 선택 → 별도 입력에서 실제 네 모델 공동삭제 확인. 정상 반대 기여/비선형 오차/미확인 상태를 구분 |
| `response_audit.py` | 보조 비교군: 언러닝 후 비교 삭제 기준의 soft/hard 흐름과 표적 선택·확인. 현재 주 알고리즘은 장력·삭제 경계 점수 |
| `tension_audit.py` | 현재 구조 후보 v3: 학습 장력과 양·음 기여의 상대 삭제율. 클라이언트/묶음 식별자를 같은 방식으로 처리하고 삭제 모델을 생성하지 않음 |
| `replay_tension.py` | 학습 캐시6개에서 점수 순위 재생. 실제 FU 호출0회. 문턱 보정·정상 FPR 실험이 아님 |
| `experiment_response.py`, `summarize_response.py` | 기존/새 입력의 동일 hard/soft 소거 감사84조건. [결과·한계·재실행](../docs/paper/DEFENSE_LOGIC_REDESIGN.md) |
| `experiment_paired.py`, `evaluate_paired.py`, `summarize_paired.py` | 고정된 실제 궤적·합성 실험, 별도 ACC/ASR 평가, 결과 소거표·그림. [실측/재실행](../docs/paper/REVISION_REPORT.md) |
| `ledger_rules.py`, `trace_solver.py` | 기존 CCD·TRACE의 계산을 수식 변경 없이 분리한 공용 모듈. 기존 스크립트와 새 진입점이 함께 사용 |
| `cancel_fu.py` | 기존 TRACE 및 정화 실험. 선택 솔버는 재사용하며, 정화 성공이나 망각 인증은 가정하지 않음 |
| `unified_fu.py` | 이전 판: 궤적 지표 → 빼 보기 확정 → 해독제 정화 (결과 logs/unified 유지) |
| `effects.py` | **기능 효과 행렬** E_j (입력×클래스), 중심화, 타깃 몰림, 부분집합 상쇄(운반자 집합), 입력 단위 상쇄, conformal·Fisher p, 라운드 일관성 (MATH.md) |
| `traj_indicators.py` | A 궤적 지표: 클래스 요약 효과 벡터 (크기·동일부호·일관성·반대 짝) |
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
`aggregate_*.py` — 단계별 시드 평균 표 (`logs/*/summary_{cond}.md`). `aggregate_cancel.py` 는 결정·1 단계 짝·2' 단계 짝 세 표.

## 원 저자 코드
`main.py`, `train_main.py`, `train_method_*.py`, `unlearn.py`, `unlearn_method/`, `utils/`, `config.py`, `dataset.py`, `model.py`, `sql.py`. 우리가 고친 곳: 일관된 데이터 분할(`utils/comm_utils.make_client_split`), 증류 나눗셈, FedEraser 래퍼, `--clients_ratial`, `--split_seed`.

## legacy/
초기 파이프라인(det1~3)과 그때의 비교·정화 스크립트, 진행형 FU(progressive_fu, dose_fu 로 대체). 지금 러너는 쓰지 않는다.
# 현재 실행 (2026-10-14)

장력 v3는 양·음 기능 기여의 상대 삭제율을 비교한다. `experiment_tension.py --profile pilot --stage all`로 기존 FUBA 학습·입력별 기록·정상 대조 비교를 실행한다. 실행 환경, 단계별 명령, 결과 판독과 미구현 범위는 [루트 RUN.md](../RUN.md) 앞부분을 따른다. 원래 BadFU와 FU 후 ASR 검증을 완료한 실행기로 해석하지 않는다.
