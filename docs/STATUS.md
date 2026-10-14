# 어디까지 됐나 (2026-10-13 저녁 기준)

**2026-10-14 BadFU 저장/자동 비교 추가:** `request_pipeline.py run`이 공개 코드의 CIFAR-10/ResNet-18 6클라이언트 FL 궤적과 실제 데이터를 별도 보존하고, 요청5/정상요청1에 무방어·보존제어·장력제어를 자동 비교한다. 기존 15라운드 궤적 import와 전체 파라미터/BN 어댑터를 추가했다. 113개 동작검사 통과; 실제 축소 FL 저장 및 기존 궤적의 한 단계 비교를 실행했다. 새 실행법은 [RUN.md](../RUN.md) 첫 절. 초기 eval 모드 복원 오류가 있던 개발 로그는 성능 근거에서 제외한다. 전체40라운드 독립 효과 검증·원래 BadFU 표본 단위 삭제·망각 인증은 남아 있다.

R-049의 수정 후 요청5 결과는 무방어 ASR91.02/ACC75.98%→temporal45.12/86.13%다. retention_only와 정상요청1의 두 제어군은 조건 불충족으로 불수락했다. 정규화 자체도 ASR을 바꾸며, 위 숫자는 같은1단계 개발 비교다. 완전 방어 또는 신규성 검증 완료로 주장하지 않는다.

**2026-10-14 제약 최적화 개정:** [TENSION_CONTROL §7](paper/TENSION_CONTROL.md#7-개정-알고리즘-요청-조건부-상쇄-연산자와-제약-최적화)의 원시 업데이트 상쇄 연산자·최소 진행·실제 클래스 손실 검사와 전체 기록 replay 실행기를 구현했다. 기존 모델 두 요청의 단계 검사에서 제약은 충족했으나 ASR은 같은 보존 제어와 동률이다. 전체 공격 FU 효과·새로운 성능 이득을 확보했다고 주장하지 않는다. 실행·필수 자료는 [RUN.md](../RUN.md) 맨 위를 따른다.

**2026-10-14 실제 FU 제어 추가:** [설계](paper/TENSION_CONTROL.md), [논문 7편 검토](paper_reviews/05_defenses_novelty.md), [실행](../RUN.md). 개별 FedEraser 보정·실제 갱신 제어·11군 단계 실행기를 구현했다. 작은 단계 스모크는 장력의 실패/성공을 판정하는 실험이 아니다. 전체 FU, 원래 BadFU, 독립 성능·망각 검증은 남아 있다. 아래 과거 성능 수치는 새 제어의 결과가 아니다.

**이하 내용은 당시 진행 기록이다.** 2026-10-14 현재 목표는 FL→FU의 주입–억제 구조와 장력·삭제 경계 검사다. [원고 v7](paper/draft_v7.tex)에 동일 비율 삭제의 v2 반례와 상대 삭제율 v3, 실패 조건을 반영했다. R-034 캐시6개 재생, R-035 전체 동작검사69개 및 입력별 수집 스모크를 완료했다. [RUN.md](../RUN.md)의 새 FUBA 정상/공격 실험은 실행 대기다. 이전 soft 감사84조건은 비교군이다. 원래 BadFU 묶음 기록·실제 삭제 및 새 복구 검증은 남아 있다.

한 줄: **논문 설정에서 탐지(균형 장부)와 처방(연합 제외)은 실측이 끝났고, 오경보율·되감기 6 궤적·BadFU 세 가지가 비어 있다.** 초안은 LaTeX 까지 썼고 컴파일은 안 했다.

## 1. 끝난 것 (결과 번호 있음)
| 무엇 | 결과 | 번호 | 어디 |
|---|---|---|---|
| 공격 재현 (논문 설정 6 궤적) | 잠복 ASR 5, 위장자 20 % 제거 ASR 72–74, 정상 제거 4–7 | R-006, R-007 | docs/EXPERIMENT_LOG.md |
| 탐지 총정리 (9 계열 36 점수, 같은 6 궤적) | 크기 기준은 소유자 오탐, 방향 기준은 운반자 못 봄, 상대 기준·장부만 위장자만 잡음 | R-009 | FUBA/logs/detect/summary_*_r100w5_k20paper.md |
| 동적 평형 지문 (누적 밀기) | 위장자 6/6 최솟값, 운반자 양의 다리, 순/총 0.07–0.16 | R-012 | FUBA/logs/equilibrium/ |
| **균형 장부 방어 (온라인 경보)** | 6/6 경보 11–16 라운드, 위장자·운반자 넷 정밀도·재현율 1.0, 소유자 오제외 0, 오경보 0 | R-013 | FUBA/logs/ledger/summary_*_r100w5_k20paper.md |
| 문턱 민감도 | 경보 시점만 움직이고 목록은 불변 (r≥3, κ≥2, s₀∈[0.6,1)) | R-013 | 같은 곳 (노트북 1 시드) |
| **연합 제외 재학습 (처음부터)** | ASR 0.5 / 0.6, 정확도 99.0 / 98.9 | R-010 | FUBA/logs/benign_departure/ |
| 되감기 이어 학습 (노트북 1 시드) | t_back 13: ASR 0.68 / 99.08; 경보 라운드 16 즉시 제외: 0.74 / 99.01; 처음부터(로컬 기준선): 0.31 / 99.07 | R-014 (예비) | FUBA/logs/benign_departure/loc_k20paper_s0_excl_from13.json |
| 버린 처방 | 재억제 α 0.09 (R-008), 해독제 실패 (R-011), 매개변수 헤지 기각 (R-005) | — | RESULTS_LOG |
| 문헌 | 방어 24 편·공격 7 편의 불변량·설명 방식, 10 가지 렌즈, 우리 판단 | — | docs/paper/HOW_OTHERS_EXPLAIN.md |
| 초안 | v5 = 학술 문체 (기법명 CCD, 용어: 주입자·억제자·누적 기여도·상쇄 지수·공모 집합) | — | **docs/paper/draft_v5.tex**, draft_v5.pdf (v3·v4 는 보관) |

## 2. 돌아가는 것 (노트북)
- 없음 (R-014 노트북 세 변형 완료).

## 2-b. 리뷰(2026-10-13 저녁) 보완으로 넣은 것
- 큰 모델 궤적: 매니페스트 `k20_paper_cifar` (CIFAR-10/resnet18, 논문 설정 flag, IID 3 시드) — `ONLY=k20_paper_cifar bash make_trajectories.sh` (DISLAB). 로더·장부·검증기가 `dataset` 을 인식한다.
- 공격 없는 궤적: `clean_paper` 4 개 (mnist/cifar × iid/owner, warm-up 1000) — `ONLY=clean_paper bash make_trajectories.sh`, 장부는 `PREFIX=clean … bash run_ledger.sh`. 검증기는 clean 항목을 "ASR 낮음·ACC·정상 제거 무해" 로만 본다.
- 위장자 여러 명: `ledger_defense.py --neg_set` (음의 다리 집합).
- BadFU 로더: `ledger_defense.py --fmt badfu`. **노트북 traj_full 에서는 경보 없음 (R-017)** — 누출 가정이 BadFU 에서는 성립하지 않을 수 있다. 40 라운드 시드로 재확인.

## 3. 비어 있는 것 (논문이 되려면 꼭)
| 무엇 | 왜 필요한가 | 명령 | 어디서 | 걸리는 시간 |
|---|---|---|---|---|
| **공격 없는 궤적의 오경보** (R-015) | "정상 학습은 균형식을 만들지 않는다"의 직접 증거. 명제 2 가 이것 없이는 주장이 아님 | RUN.md A-03 끝 두 줄 (저자 코드 warm-up 100 라운드) | DISLAB | 궤적 20 분 + 장부 2 분 |
| **되감기 대응 6 궤적** (R-014) | 표 4 마지막 행 | RUN.md A-03 "대응" 블록 | DISLAB | 6 × 13 분 |
| BadFU 3 시드 | 두 번째 공격에서 누출 가정이 성립하는지 | make_badfu_trajectories.sh → 장부 (badfu 로더 필요: ledger_defense.py 는 지금 fuba 만) | 노트북 | 궤적 3 × 20 분 + 로더 작업 |
| 위장자 2 명 (R-016) | 적응 공격 (a) | sup2 궤적 (있음) 에 ledger_defense (Neg 집합 규칙 추가 필요) | DISLAB | 1 시간 |
| 집계 방어 위 12 궤적 | 넓힘 | tdef_paper (hdbscan 설치) → run_ledger.sh TAG | DISLAB | 반나절 |

## 4. 초안에서 아직 정확하지 않은 곳 (스스로 적음)
- 명제 1·2 는 스케치다. 가정(누출 비율 η, 정상 밀기의 독립·분산)을 식으로 못 박고, 명제 2 는 R-015 숫자로 바꿔야 한다.
- "운반자 남긴 재학습 ASR 44" 는 K=8 예비 결과다. 논문 설정에서 다시 재야 표 4 첫 행이 선다 (benign_departure retrain_attack, 6 × 13 분).
- 참고문헌은 제목·연도만 적었다. 실제 서지 확인 필요 (특히 FUBA, UBA-Inf, SubID 는 본문을 못 열었다).
- LaTeX 는 Tectonic 0.17 (XeTeX 엔진) 으로 컴파일했다: `tectonic -o docs/paper docs/paper/draft_v3.tex` → `draft_v3.pdf` (7 쪽). 노트북의 실행 파일: `C:/Users/sijoo/AppData/Local/Temp/bfu/tectonic_dl/tectonic.exe`. Overleaf 면 XeLaTeX. 표는 `\resizebox` 로 폭을 맞췄다 (글자가 작다; 학회 양식에 맞출 때 세로 표로 바꾼다).

## 5. 순서 제안
1. DISLAB: 공격 없는 궤적 1 개 → 장부 (오경보). 2. DISLAB: 되감기 6 궤적. 3. 논문 설정 retrain_attack 6 궤적. 4. 초안의 todo 를 숫자로 바꾸고 명제를 식으로. 5. BadFU·sup2·tdef 는 넓힘 절.
