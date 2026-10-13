# BadFU_defend — 삭제 요청으로 깨어나는 백도어에 대한 연합 언러닝 방어

연합 언러닝(FU)에서 삭제 요청으로 활성화되는 잠복 백도어(BadFU, FUBA)를 서버가 **학습 중에** 찾아(균형 장부) 심은 쪽을 걷어내는 방법을 만든다. 요청 시점 절차(DOSE·TRACE·DEFUSE)는 부록·소거 실험으로 내렸다 (2026-10-13).

## 무엇을 하려는가 (세 문장)
1. **공격.** 운반자 여럿이 백도어를 심고, 위장자 한 명이 같은 입력에서 거꾸로 밀어 잠재운다. 위장자가 "내 데이터를 지워 달라" 하면 백도어가 깨어난다. FUBA 와 BadFU 가 둘 다 이것이다 (기제: 모순 지도의 평균화).
2. **방어.** 서버는 라운드마다 참여자의 갱신이 공개 입력 위에서 각 클래스를 미는 양을 장부에 적는다. "합은 0 인데 여럿이 밀고 한 명이 되미는" 클래스(상쇄 연합)가 세 라운드 연속 보이면 경보: 타깃·위장자·운반자·되감기 라운드가 나온다. 되감고 연합을 뺀 채 이어 학습한다.
3. **주장.** 정상 학습에는 상쇄 연합이 없고, 삭제로 깨어나는 백도어는 반드시 하나 만든다. 논문 설정 6 궤적에서 11–16 라운드에 6/6 경보, 위장자·운반자 넷 정확, 소유자 오제외 0, 제외 후 ASR 0.5 (R-013, R-010).

**가칭 (앞으로 이 이름으로 부른다)**
| 가칭 | 하는 일 | 책임 |
|---|---|---|
| **균형 장부 (Balance Ledger)** | 학습 중 라운드마다 클라별·클래스별 밀기 누적 → 헤지 비율 H·음의 다리·양의 다리 집합 → 공모/정상 큰손/외톨이 음 판정 → 경보 (타깃, 위장자, 운반자, 되감기 라운드) | **본 방법** (`FUBA/ledger_defense.py`, RUN.md A-03). 대응 = 되감기 + 연합 제외 이어 학습 (`benign_departure.py --exclude --init_round`) |
| **DOSE** (부록) | 요청자 기여를 견디는 양만큼 단계로 빼고, 단계마다 "남들을 같은 양 뺀 모집단" 띠 안인지 확인하는 삭제 연산자 | 안전 (띠 밖·무너진 모델 불출), 정상 요청 = 기본 FU |
| **TRACE** (부록) | 띠를 벗어나면 "어디서, 누구의 기여를 지우고 있었나"를 푸는 귀속 (회전 1: 저장 기여, 회전 2: 활성화 입력) | 귀속 출력 (짝·타깃·입력·방향) |
| **DEFUSE** (폐기, R-011) | 짝의 기여에서 활성화 방향 성분만 빼서 띠 안으로 되돌림 | 삭제 이행 (요청자 기여 재주입 0, 잊힌 것 되가르치지 않음) |
정본: 균형 장부는 [docs/paper/DRAFT_v3.md](docs/paper/DRAFT_v3.md) §5–6 과 [docs/paper/ALGORITHMIZATION.md](docs/paper/ALGORITHMIZATION.md) §3; 요청 시점 절차는 [docs/paper/ALGORITHM.md](docs/paper/ALGORITHM.md).

그림: [docs/paper/fig_architecture.svg](docs/paper/fig_architecture.svg)

![architecture](docs/paper/fig_architecture.svg)

## 어디까지 왔나 (2026-10-13; 자세히는 [docs/STATUS.md](docs/STATUS.md))
| 할 일 | 상태 |
|---|---|
| 공격 재현 (논문 설정 K=20, 100 라운드, 참여 0.4, 3 시드 × IID/owner) | **됨** (R-006): 잠복 ASR 5, 위장자 20 % 제거 ASR 72–74 |
| 학습 중 탐지 — 균형 장부 | **됨** (R-013): 6/6 경보 11–16 라운드, 위장자·운반자 넷 정밀도·재현율 1.0, 소유자 오제외 0, 오경보 0 |
| 기존 계열 비교 (9 계열 36 점수) | **됨** (R-009): 크기 기준은 소유자 오탐, 방향 기준은 운반자 못 봄 |
| 처방 — 연합 제외 | **됨** (R-010): ASR 0.5/0.6, 정확도 99. 되감기 이어 학습 노트북 1 시드 ASR 0.7 (R-014 예비) |
| 공격 없는 궤적의 오경보 | **비어 있음** (R-015): RUN.md 0-b `ONLY=clean_paper bash make_trajectories.sh` |
| 큰 모델 (CIFAR-10/resnet18) | **비어 있음**: RUN.md 0-b `ONLY=k20_paper_cifar bash make_trajectories.sh` |
| BadFU | **경계** (R-017): 노트북 15 라운드 궤적에서 장부가 상쇄를 못 봄 → 40 라운드 시드로 재확인, 안 보이면 요청 시점 빼 보기로 보완 |
| 초안 | [DRAFT_v3.md](docs/paper/DRAFT_v3.md) / [draft_v3.tex](docs/paper/draft_v3.tex) / draft_v3.pdf (Tectonic) |

## 어디부터 읽나
| 문서 | 내용 |
|---|---|
| [docs/STATUS.md](docs/STATUS.md) | **어디까지 됐나**: 끝난 것(번호), 돌아가는 것, 비어 있는 것(명령·시간), 초안에서 아직 정확하지 않은 곳 |
| [docs/paper/draft_v3.tex](docs/paper/draft_v3.tex) | **LaTeX 초안** (XeLaTeX + kotex; Overleaf 에서 컴파일) |
| [docs/paper/brief_prof.tex](docs/paper/brief_prof.tex) | **교수님 브리프**: FL→FU 공격 부류의 정의(정의 1, 두 필요조건), 공격 논문 7 편의 위치, 두 층 방어(학습 중 CCD + 요청 시 상대 부분 제거), 실험 요약, 적응형·다중 공격 trade-off (그림), 발전 방안. PDF `brief_prof.pdf` |
| [docs/paper/draft_v5.tex](docs/paper/draft_v5.tex) | **논문 초안 v5 (현재 판, 학술 문체)**: FLDetector·FLAME 의 전개(서론→관련 연구→위협 모델→제안 기법 CCD→분석→실험→논의→결론). PDF `draft_v5.pdf` |
| [docs/paper/DRAFT_v4.md](docs/paper/DRAFT_v4.md) | **논문 초안 v4 (쉬운 말 판, 현재 판)**: 비유 용어 없이 용어표·기제 설명·방법·분석·평가. LaTeX `draft_v4.tex`, PDF `draft_v4.pdf` |
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
1. 궤적 만들기·검증: `cd FUBA && bash make_trajectories.sh` (RUN.md 0). 큰 모델·공격 없는 궤적: `ONLY=k20_paper_cifar`, `ONLY=clean_paper` (RUN.md 0-b)
2. **균형 장부 (본 방법)**: `bash run_ledger.sh` → `logs/ledger/summary_*.md` (RUN.md A-03). CIFAR 는 `DATASET=cifar10 TAG=_r100w5_k20paper_cifar`, 공격 없음은 `PREFIX=clean`
3. 대응 (되감기 + 연합 제외): RUN.md A-03 대응 블록 (R-014). 처음부터 제외 재학습: RUN.md A-01 (R-010)
4. 비교군: 탐지 총정리 `bash run_detect_bench.sh` (A-00), 동적 평형 지문 (A-02)
5. 부록 (요청 시점 절차 DOSE·TRACE·DEFUSE): RUN.md A-0
`run_paper.sh` 는 K=8 예비 파이프라인이다 (부록용).
