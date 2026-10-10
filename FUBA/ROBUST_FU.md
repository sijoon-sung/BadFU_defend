# 강건한 FU (FedEraser-R) — 정의 2 의 위반을 보정 루프 안에서 재고(탐지) 바로잡는다(정화)

## 묻는 것

잠복 백도어(FUBA, BadFU)는 **삭제 요청 자체**를 트리거로 쓴다. 요청자(Adv-defender)의 기여가 빠지면 공격자들이 심어 둔 백도어가 깨어난다.
기존 FU 방어(SubID, UnlearnGuard 류)는 FU 중 들어오는 **새 악성 업데이트**를 찾는다. 그러나 잠복형에서는 FU 중 아무도 새로 공격하지 않는다 —
저장된 기여를 빼는 것만으로 모델이 바뀐다. 그래서 우리는 다른 곳을 본다: **제거가 모델 행동을 어떻게 바꾸는가**.

### 정의 2 (이 실험이 쓰는 언러닝 정의)

> 요청자의 영향은 지우되, **나머지 모두의 입력에 대한 모델 행동은 바뀌지 않아야 한다.**

삭제로 깨어나는 백도어는 이 정의의 위반이다: 요청자가 아닌 입력(여기서는 서버 탐침)에 대한 예측이 한 클래스로 쏠린다.
따라서

- **탐지** = FedEraser 보정 경로를 따라가며 그 위반(비요청자 입력의 행동 변화가 한 클래스로 집중)을 재는 것,
- **정화** = 그 정의를 강제하는 것: 바뀐 행동을 원 경로로 되돌린다.

요청자의 저장 기여는 **항상 전부** 빠진다(삭제권은 흥정하지 않는다). 처음부터 재학습하지 않는다. 공격자를 식별하지 않는다. 트리거를 모른다.
서버에 라벨 있는 데이터는 없다 — **라벨 없는 작은 탐침**(MNIST TEST[0:N], 기본 N=500)뿐이다.

## 절차 (`robust_fu.py`)

`unlearn_method/federaser_faithful.py` 와 같은 1라운드 초기화(요청자 뺀 저장 로컬 평균)와 원 논문식 보정 식(옛 노름 × 새 방향)을 그대로 쓴다.
남은 클라이언트는 자기 분할(`make_client_split`)로 정직하게 짧게 학습한다 — 이것은 언러닝 절차이지 재학습이 아니고, 보정 중 아무도 공격하지 않는다.

요청자 k 마다, 라운드 r = 2..R:

1. `new_gm_r` = 보정 FedEraser 한 걸음.
2. **탐지** (정의 2 위반, 트리거 없음). 재생 모델 `h_r = new_gm_r` 을 **원 궤적의 같은 라운드 모델** `orig_r = global_r` 과 비교한다. 학습 진도가 같으니 차이는 제거 몫이다.
   - (i) 깨끗한 탐침: argmax 가 갈린 비율(`diverge`), **집중도** = 갈린 입력 중 한 클래스 c 로 옮겨 간 비율의 최대(c 기록).
   - (ii) 적대적 탐침: 범용 δ (L∞ ≤ `eps`, 탐침 위 mean KL(h_r(x+δ) ‖ orig_r(x+δ)) 를 부호 PGD 로 최대화). x+δ 에서 같은 두 값.
   - `Score_r` = δ 아래 집중도. `flagged_r` = Score_r > `flag_thr`.
3. **정화** (정의 2 강제). arm 에 따라 `new_gm_r` 에 `purify_steps` 번 SGD: x ∈ 탐침 ∪ (탐침+δ) 에서 KL(h_r(x) ‖ orig_r(x)) 최소화. 교사 orig_r 고정, 가중 `purify_lambda`.
   정화된 모델이 다음 라운드 보정의 출발점이 된다.
   > 떠난 클라이언트가 맡던 억제 역할을 서버가 떠안는다 — 자기 라벨 없는 입력과 원 경로를 교사로 삼아.

| arm | 정화 시점 | 정화 입력 |
|---|---|---|
| `plain` | 안 함 (= 원 논문식 FedEraser, 탐지 흔적만 기록) | – |
| `purify_flag` | `flagged_r` 인 라운드만 | 탐침 ∪ 탐침+δ |
| `purify_always` | 매 라운드 | 탐침 ∪ 탐침+δ |
| `purify_clean` | 매 라운드 | 탐침만 (δ 없음) — δ 가 필요한지 보는 절제 |

라운드 R 뒤 채점 (여기서만 정답 허용):

- TEST[N:] 에서 ACC, 저장된 IBA 생성기로 ASR, 클래스별 ASR (`benign_departure.py` 와 같은 식).
- **정의 2 유지 행동 변화**: TEST[N:] 의 깨끗한 예측이 θ_T(삭제 전 최종 전역)와 다른 비율.
- 망각 측정 (아래 절).
- 소요 초. `before` 행 = θ_T.

### 망각 측정: MIA(손실 기반) + 요청자 데이터 손실/정확도 proxy

**MIA 가 주 지표**, 대조군 = 정상 클라 데이터.

- 손실 기반 MIA (Yeom et al. 식). 멤버 = 요청자 자기 분할(`make_client_split`), 비멤버 = TEST[N:] 의 같은 크기 고정 시드 무작위 조각(탐침과 겹치지 않음). 모델마다 표본별 CE 손실을 재고 공격 점수 = −손실.
  - (a) 멤버 vs 비멤버 **AUROC** (자체 구현, sklearn 없음).
  - (b) **멤버율**: 비멤버 손실의 하위 5 분위(= 비멤버 FPR 5%)를 문턱으로 요청자 표본이 멤버로 판정되는 비율.
  - 잘 지워진 모델은 AUROC → 0.5, 멤버율 → 0.05 로 간다.
- θ_T(`before`), 각 arm 의 최종 모델 모두에 대해 잰다. **대조군**: 요청자가 아닌 첫 정상 클라(예: 요청자 4 → 클라 5)의 데이터로 같은 MIA. 대조군의 AUROC/멤버율은 **떨어지면 안 된다** (남의 것까지 지우는 과잉 언러닝이 아님을 보인다).
- 표본 수는 `--mia_n`(기본 1000)으로 제한. JSON: `results[요청자][arm].mia = {req, control, control_client}`, `before.mia[요청자]`.
- 보조 proxy: 요청자 분할에서의 평균 CE 손실과 정확도(최종 vs θ_T). MIA 가 아니다 — 참고용.

## 가정 (서버가 아는 것)

- 탐침 = MNIST TEST[0:N] (`--probe`, 기본 500), **라벨 안 씀**. FLAME/FLTrust 류가 두는 서버 검증 셋에 해당.
- 모든 평가(ACC/ASR/유지 변화/MIA 비멤버)는 TEST[N:] 만 쓴다 — 탐침과 겹치지 않는다.
- δ 예산 `--eps`(기본 0.04)·PGD 걸음(`--pgd_steps 50 --pgd_lr 0.005`)은 **서버가 고른다**. 공격자의 경계값(`noise_thread`)을 아는 게 아니다. 기본값이 공교롭게 같지만 민감도는 `--eps` 를 바꿔 본다.
- 원 궤적 `global_r`(r=1..R)과 로컬 `local_r` 은 저장돼 있다(FedEraser 의 전제와 같다).

## 이것이 무엇이 아닌가

| | 그들 | 우리 |
|---|---|---|
| SubID / UnlearnGuard | FU **중** 들어오는 새 악성 업데이트를 찾는다 | FU 중 새 공격은 없다. **제거가 보정 경로에서 일으킨 행동 변화**를 재고 고친다 |
| 처음부터 재학습 | 요청자 뺀 전원 재학습 | 재학습 없음. 보정 한 걸음마다 탐지+정화만 끼운다 (비용 ≈ 보정 + 탐침 위 PGD/SGD 몇십 걸음) |
| FUBA Table VIII (깨끗한 셋 미세조정) | 서버가 **라벨 있는** 깨끗한 데이터로 미세조정 | 라벨 없음. 교사는 원 경로 `orig_r`, 손실은 KL 뿐 |
| 요청 심사 / 공격자 식별 (LATENT_RISK.md) | 요청 전에 가중치만으로 누가 위험한지 본다 | 요청은 모두 받아들인다. 누구인지 묻지 않고 결과(행동)만 본다 |

## 판정 (결과를 보기 전에 고정)

| 결과 | 해석 | 다음 |
|---|---|---|
| **go**: `purify_flag` 가 defender 요청(req 4)의 ASR 을 `before`(잠복 수준) 근처로 되돌리고, ACC 손실 < 1 pp, 정상 요청(req 7·5)은 flag 되지 않거나 ASR/ACC/MIA 가 `plain` 과 같다. 대조군 MIA 안 떨어짐 | 정의 2 기반 탐지+정화가 성립 | eps·flag_thr 민감도, FUBA 기본 규모(20명·100라운드), BadFU 궤적, 적응형 공격자(δ 를 피하는 트리거) |
| `purify_always` 만 되고 `purify_flag` 는 안 됨 | 정화는 되나 탐지(Score)가 못 가른다 | Score 정의 재검토 (집중도 대신 KL 크기·클래스 쏠림 지수), 라운드별 AUROC 확인 |
| `purify_clean` 이 `purify_flag/always` 만큼 됨 | δ 가 불필요 — 깨끗한 탐침의 행동 변화만으로 충분 | PGD 비용을 뺀 단순판으로 |
| 정상 요청에서 ASR 이 `plain` 보다 **오르거나** ACC ≥ 1 pp 손실, 혹은 대조군 MIA 가 떨어짐 | 정화가 정의 2 를 과잉 강제 — 제거 자체를 되돌린다(요청자 MIA 가 `before` 수준으로 회귀하면 더 확실) | purify_lambda/steps 를 줄이거나 δ 부분만 정화 |
| **no-go**: 어떤 arm 도 defender 요청의 ASR 을 `plain` 대비 유의하게 못 내림 | 보정 경로에서 행동 변화가 탐침에 안 잡힘(백도어가 탐침 분포 밖에서만 열림) | 탐침 구성·eps 를 바꿔 한 번 더; 그래도 안 되면 접는다 |

## 실행

```bash
cd FUBA
COND=owner bash run_robust_fu.sh                 # 시드 0 1 2, 궤적 생성 + FedEraser-R + 집계
COND=iid   SKIP_ATTACK=1 bash run_robust_fu.sh   # run_benign_departure.sh 가 만든 궤적(bd_iid_s*) 재사용
ARMS="plain purify_flag" SEEDS="0" EXTRA="--eps 0.02" bash run_robust_fu.sh
```

결과: `logs/robust_fu/bd_{cond}_s{seed}.json` ([요청자][arm] 별 수치 + 라운드별 탐지 흔적), 집계 `logs/robust_fu/summary_{cond}.md` (+ `.json`):
arm × 요청자 별 ASR/ACC/유지 변화/MIA/망각 proxy 의 평균±표준편차, 탐지 = 라운드 최대 Score 의 defender vs 정상 평균과 실행 전체 AUROC.

단일 실행:

```bash
python robust_fu.py --name bd_owner_s0 --K 8 --seed 522 --split_seed 522 \
  --non_iid --non_iid_type class_owner --owner_class 3 --requesters 4 7 5 \
  --probe 500 --eps 0.04 --pgd_steps 50 --purify_steps 30 --flag_thr 0.5 --resume
```

`--resume` 는 이미 있는 (요청자, arm) 결과를 건너뛴다. GPU 없으면 `--n_gpu 0`.

## 한계

- **억제이지 제거가 아니다.** 교사가 `orig_r`(= 잠복 백도어를 품은 원 경로)이므로 ASR 은 삭제 전 잠복 수준으로 돌아갈 뿐 0 이 되지 않는다. 백도어 자체는 여전히 모델 안에 있다.
- Score 는 "갈린 입력의 집중도"라 갈린 입력이 아주 적어도 1.0 이 될 수 있다. `diverge` 와 같이 읽어야 한다(JSON 에 둘 다 있음).
- MIA 는 손실 문턱식 단순판이다. 작은 MNIST Net 은 원래 과적합이 약해 θ_T 에서도 AUROC 가 0.5 근처일 수 있고, 그러면 망각 "변화"를 읽을 여지가 작다. 대조군과 `before` 행을 같이 봐야 한다.
- 탐침을 테스트셋 앞부분에서 떼므로 평가는 TEST[N:] 로 줄어든다(9,500 장). 실제 배치에서는 서버 보유 검증 셋이어야 한다.
- 보정 중 클라이언트는 정직하다고 가정한다(FedEraser 와 같은 전제). 보정 중 재주입(FUBA 각주 4 의 재학습형)은 여기서 다루지 않는다 — 그건 집계 쪽 방어의 몫이다.
- K=8·8라운드 소규모 설정. 경향만 본다.
