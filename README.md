# BadFU_defend — 삭제 요청으로 깨어나는 백도어에 대한 연합 언러닝 방어

연합 언러닝(FU)에서 삭제 요청으로 활성화되는 잠복 백도어(BadFU, FUBA)를 서버가 요청 시점에 가려내고, 심은 쪽의 성분만 지우는 방법을 만든다.

## 무엇을 하려는가 (세 문장)
1. **공격.** 어떤 클라이언트(운반자)가 백도어를 심고, 다른 클라이언트(요청자)가 같은 입력에서 거꾸로 밀어 잠재운다. 요청자가 "내 데이터를 지워 달라" 하면 백도어가 깨어난다. FUBA 와 BadFU 가 둘 다 이것이다.
2. **방어.** 삭제 요청이 오면 서버는 "이 요청자는 **어디서, 누구의 기여를** 누르고 있었나"를 푼다. 찾은 짝의 기여에서 눌려 있던 부분만 뺀다. 요청자의 데이터는 지워진 채로 둔다.
3. **주장.** 기존 방법은 요청을 거절하거나, 모델을 다시 눌러서 요청자 데이터를 되넣는다. 우리는 둘 다 안 한다. 백도어는 안 깨어나고, 지워 달라는 것은 지워진다. 그리고 출력에 귀속(짝·타깃·입력)이 있다.

**가칭 (앞으로 이 이름으로 부른다)**
| 가칭 | 하는 일 | 책임 |
|---|---|---|
| **DOSE** | 요청자 기여를 견디는 양만큼 단계로 빼고, 단계마다 "남들을 같은 양 뺀 모집단" 띠 안인지 확인하는 삭제 연산자 | 안전 (띠 밖·무너진 모델 불출), 정상 요청 = 기본 FU |
| **TRACE** | 띠를 벗어나면 "어디서, 누구의 기여를 지우고 있었나"를 푸는 귀속 (회전 1: 저장 기여, 회전 2: 활성화 입력) | 귀속 출력 (짝·타깃·입력·방향) |
| **DEFUSE** | 짝의 기여에서 활성화 방향 성분만 빼서 띠 안으로 되돌림 | 삭제 이행 (요청자 기여 재주입 0, 잊힌 것 되가르치지 않음) |
정본: [docs/paper/ALGORITHM.md](docs/paper/ALGORITHM.md) (알고리즘 정식화 + 가정 전부 + 파라미터 + 비용).

그림: [docs/paper/fig_architecture.svg](docs/paper/fig_architecture.svg)

![architecture](docs/paper/fig_architecture.svg)

## 어디까지 왔나 (2026-10-12)
| 할 일 | 상태 |
|---|---|
| 짝 찾기 (누구 것을 누르고 있었나) | **됨.** IID·owner 3 시드 모두 운반자 넷을 정확히 찾음. 소유자(non-IID 클래스 보유자)는 짝에 안 들어옴 |
| 백도어 안 깨어나게 하기 | **됨.** ASR 48→5 (IID), 54→11 (owner). 직교 증류를 붙이면 정확도가 재학습 방식(KD) 수준으로 돌아옴 |
| 요청자 데이터 안 되살리기 | **됨.** α 0 (KD 재억제는 0.16). 단, 증류의 교사를 잘못 두면 α 가 0 이어도 잊은 것이 되돌아옴 (REPORT 15.4) → 고침, 재실행 필요 |
| 정상 요청 비용 | 새 모델 0 개, 순전파만 |
| 남은 약점 | owner 정상 7 의 빼 보기 오탐 1/3 (같은 시드). K=20 주 설정 미실행. BadFU 는 로컬 1 시드 |
| 다음 | `_polish` 재실행 (RUN.md A0) → 결과 확정 → 비교군·적응형 공격 (RQ2·4) → 논문 쓰기 (DRAFT_OUTLINE.md) |

## 어디부터 읽나
| 문서 | 내용 |
|---|---|
| [docs/paper/DRAFT_v3.md](docs/paper/DRAFT_v3.md) | **논문 초안 v3**: 조사한 방어 논문들의 전개(문제 → 틈 → 기제 → 불변량 → 점수 → 판정 → 대응 → 분석 → 평가 → 한계)로 쓴 전체 초안, 그림 1·2, 결과 번호 대응 |
| [docs/paper/CONSOLIDATED.md](docs/paper/CONSOLIDATED.md) | **한 번에 정리**: 두 단계로 줄인 알고리즘(균형 장부 → 경보 시점 되감기+연합 제외), 실측 통계 표 1–5, 정화 결정 |
| [docs/paper/PLAIN_SUMMARY.md](docs/paper/PLAIN_SUMMARY.md) | **쉬운 말 요약**: 문제·확인한 사실·균형 장부·결과·논문 한 줄 |
| [docs/paper/HOW_OTHERS_EXPLAIN.md](docs/paper/HOW_OTHERS_EXPLAIN.md) | **다른 논문의 설명 방식**: 방어 24 편의 불변량·이론 대상·판정·보장 표, 공격 논문의 잠복 설명, 상쇄를 설명할 10 가지 렌즈와 우리 판단(§6) |
| [docs/paper/ALGORITHMIZATION.md](docs/paper/ALGORITHMIZATION.md) | **관찰 → 알고리즘**: 다른 방어 논문 6편이 밟은 다섯 단계, 우리의 균형 장부(Balance Ledger) 알고리즘과 불변량·영분포·대응·보장 |
| [docs/paper/BRIEFING_2026-10-13.md](docs/paper/BRIEFING_2026-10-13.md) | **지금까지의 결과 브리핑** (공격 재현, 탐지 총정리, 처방, 학습 중 탐지, 발견 셋, 코드·문서, 확인 상태, 남은 약점) |
| [docs/paper/DRAFT_v2_SKELETON.md](docs/paper/DRAFT_v2_SKELETON.md) | **초안 v2 뼈대**: 주장 한 문장, 위협 모델 다섯 줄, 표 1–5 (논문 설정 K=20 값 채움), 한계, 해독제 실패 원인, 할 일 |
| [docs/paper/PLAN.md](docs/paper/PLAN.md) | **앞으로의 계획**: 실험 기록 체계(장부 2개), Phase 0–5 와 관문, 6주 일정, 위험, 사용자가 정할 것 3개 |
| [docs/paper/RELATED_METHODS.md](docs/paper/RELATED_METHODS.md) | **기존 방법 전체 목록** (공격 / 집계 방어 / 언러닝 연산자 / 요청 심사 / 사후 정화 / 복구): 가정, 잠복형 적용 여부, 우리 근거 상태, 공식 코드 |
| [docs/paper/WHY_EXISTING_FAIL.md](docs/paper/WHY_EXISTING_FAIL.md) | **기존 방법이 왜 안 되는가**: 집계 방어·언러닝 연산자·요청 심사·사후 정화·재억제, 계열별 기전·근거·상태 + 우리가 버린 시도들 |
| [docs/paper/ZERO_BASE_REVIEW.md](docs/paper/ZERO_BASE_REVIEW.md) | **원점 검토**: 문제·출발점·주장 10 개와 근거·실험 체계·판정·할 일 (쉬운 말) |
| [docs/paper/ALGORITHM.md](docs/paper/ALGORITHM.md) | **정본**: DOSE·TRACE·DEFUSE 의 알고리즘 정식화, 가정 S1–S6·T1–T5·O1–O6, 결정표, 파라미터, 비용 |
| [docs/paper/METHOD.md](docs/paper/METHOD.md) | 방법 v3 설명: 상쇄 짝 찾기 (저장 기여 위 짝 찾기 → 빼 보기 → 활성화 위 짝 찾기 → 짝의 상쇄 성분 제거). 코드 `FUBA/cancel_fu.py` |
| [docs/paper/THREAT_MODEL.md](docs/paper/THREAT_MODEL.md) | 공격의 틀 (상쇄형 백도어, 해로운 삭제 요청) |
| [docs/paper/REPORT_2026-10-12.md](docs/paper/REPORT_2026-10-12.md) | 지금까지의 실험 결과 보고 |
| [docs/paper/DRAFT_v1.md](docs/paper/DRAFT_v1.md) | **논문 초안 v1** (초록·기여·방법·결과·한계 + §8 노벨티 자가 평가와 보강 방향) |
| [docs/paper/DRAFT_OUTLINE.md](docs/paper/DRAFT_OUTLINE.md) | 논문 초안 기획 (이전) |
| [RUN.md](RUN.md) | 실험 돌리는 법 |
| [BRANCHES.md](BRANCHES.md) | 브랜치 (헤지 파일럿은 `exp/hedge-pilot` 브랜치) |

## 폴더
| 폴더 | 내용 |
|---|---|
| `FUBA/` | 실험 코드 (FUBA 공격 코드 + 우리 방어·비교군·러너). 스크립트 지도는 `FUBA/README.md` |
| `BadFU_src/` | BadFU 공격 코드 (로컬 궤적은 gitignore) |
| `docs/paper/` | 논문용 문서 (방법, 위협 모델, 자리매김, 실험 설계, 보고, 리뷰, 표) |
| `docs/paper_reviews/` | 선행 논문 정리 4 편 |
| `docs/archive/` | 지나간 설계 문서 (design/) 와 FUBA 쪽 옛 기록 (fuba/) |
| `legacy/` | 초기 탐색 코드·결과 (main 시절). 참고용 |

## 실행 순서 (자세히는 RUN.md)
1. 궤적 만들기·검증: `cd FUBA && bash make_trajectories.sh` (RUN.md 0)
2. 탐지 총정리: `bash run_detect_bench.sh` (RUN.md A-00)
3. 동적 평형 지문: RUN.md A-02 의 네 줄
4. 본 결과·처방: RUN.md A-0, A-01
`run_paper.sh` 는 K=8 예비 파이프라인이다 (부록용).
