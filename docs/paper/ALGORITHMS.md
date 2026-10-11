# 알고리즘 정식화와 실험 설계 (논문 초안용, 2026-10-11)

코드: `FUBA/dose_fu.py` (알고리즘 1), `FUBA/pair_roles.py` (알고리즘 2), 실험 전체 `FUBA/run_paper.sh`, 표 `FUBA/paper_tables.py`.

---

## 1. 문제 정의

**설정.** 클라이언트 K 명, 라운드 T 의 연합학습. 서버는 라운드별 클라이언트 갱신을 저장한다 (FedEraser 류 FU 의 전제). 최종 전역 모델 θ_T. 클라이언트 k 의 **누적 기여** U_k = Σ_t w_t (θ_k^t − θ^{t−1}) (집계 가중 w_t 포함). 수동형 FU 연산자 R(θ, k, ρ) 는 k 의 기여를 비율 ρ 만큼 뺀 모델을 돌려준다 (빼기: θ − ρU_k; 증류형: 빼고 탐침에서 직전 모델을 교사로 짧게 증류).

**위협.** 잠복형(언러닝으로 깨어나는) 백도어. 공격자 집단은 학습 중에 (i) 백도어를 심는 **운반자** C 와 (ii) 그 효과를 억누르는 **위장자** S 로 나뉜다 (FUBA: 억제자 클라이언트; BadFU: 위장 샘플을 든 클라이언트). 학습 중 모델은 깨끗해 보인다. S 가 삭제를 요청해 R 이 적용되면 백도어가 깨어난다. 공격자는 방어 절차를 안다 (적응형).

**서버의 목표 (정의 2).** 요청자 k 에 대해 출력 θ' 는
- (망각) θ' 가 k 의 데이터에 대해 R(θ_T, k, 1) 과 같은 정도로 잊고,
- (행동 보존) **남들의 입력**에서 θ' 의 행동이 "다른 클라이언트 한 명을 같은 양만큼 뺐을 때"와 구분되지 않아야 한다.
정상 요청자에게는 (망각) 이 우선이고 θ' = R(θ_T, k, 1) 이어야 하며, (행동 보존) 을 어기는 요청자는 "행동 변경 요청"으로 표시된다.

**서버가 가진 것.** 저장된 갱신, 라벨 없는 탐침 입력 X_p = 공개 입력 ∪ **요청이 지우려는 데이터의 입력** X_f (라벨 불필요; 지울 때만, 학습 시 해시로 봉인). 트리거·타깃·공격자 정보 없음. 학습 없음. X_f 가 없으면(공개 입력만) FUBA 류까지, 있으면 BadFU 류까지 범위.

---

## 2. 알고리즘 1: 투여량 조절 연합 언러닝 (Dose-controlled FU)

### 기호
- d(θ_a, θ_b): X_p 에서 두 모델의 argmax 가 다른 비율 (건강 척도).
- a(θ; ref): 활성화 통계. 기본 `mass` = (답이 바뀐 비율) × (바뀐 답이 한 클래스로 몰린 비율). X_p 에서 θ 와 기준 ref 를 비교.
- Ens(M): 모델 집합 M 의 입력별 클래스 확률 **중앙값** 앙상블.
- z(x; {s_j}): (x − median{s_j}) / std{s_j} (ddof=1; 옵션 MAD, IQR). 흩어짐이 0 인데 차이가 있으면 '판단 불가' → 표시.
- 문턱 τ: 고정(3) 또는 **보정**(`--tau_mode loo`: 모집단 멤버 각자의 LOO z 최댓값 + 1; 라벨 없음).
- α (재주입): ⟨θ_out − (θ_T − U_k), U_k⟩ / ‖U_k‖². 수정이 요청자 기여를 얼마나 되돌렸는가. 표에 넣는다.
- 초모수: 건강 상한 c (0.3), 문턱 τ (3), 모집단 표본 m (전원), 최소 투여량 δ_min (0.1), 수정 걸음 E (50), 학습률 η.

### 부품
```
POPULATION(θ, k, δ):                       # 요청자 제외 남들을 같은 양 δ 만큼 뺀 가상 모델들
  for j in Others(k):  M_j ← R(θ, j, δ);  h_j ← d(M_j, θ)
  H ← { j : h_j ≤ c }                        # 건강한 멤버
  ref ← Ens({M_j : j ∈ H})                   # 기준 (비면 θ)
  for j in H:  s_j ← a(M_j ; Ens({M_i : i ∈ H, i ≠ j}))
  return ref, {s_j}, median_j h_j

DOSE(θ, k, ρ_left):                          # 부품 A: 모두가 견디는 가장 큰 양
  for δ in [ρ_left, 0.8, 0.6, 0.4, 0.2, δ_min] with δ ≤ ρ_left:
      if median_j h_j of POPULATION(θ, k, δ) ≤ c:  return δ
  return δ_min

GUARD(θ, k, δ):                              # 요청자 자신도 견디는가
  for δ' in [δ, 0.8, …, δ_min] with δ' ≤ δ:
      if d(R(θ, k, δ'), θ) ≤ c:  return δ', collapse=False
  return δ_min, collapse=True

CHECK(θ_k, ref, {s_j}):                      # 부품 B: 같은 양 비교
  return z( a(θ_k ; ref) ; {s_j} )

REPAIR(θ_k, ref, X_p):                       # 부품 C: 남들 입력에서의 행동만 되돌림
  for e in 1..E:  θ_k ← θ_k − η ∇ KL( softmax(θ_k(x)) ‖ ref(x) ),  x ~ X_p  (기울기 클립, BN 고정)
  if NaN: 되돌림, 실패 표시
  return θ_k
```

### 본 알고리즘
```
Algorithm 1  DOSE-FU(θ_T, k)
  θ ← θ_T;  ρ ← 0;  log ← ∅
  while ρ < 1:
      δ ← DOSE(θ, k, 1−ρ)
      δ, collapse ← GUARD(θ, k, δ)
      if collapse:  log ← log ∪ {abort};  break           # 무너진 모델을 내보내지 않는다
      ref, {s_j}, _ ← POPULATION(θ, k, δ)
      θ ← R(θ, k, δ);  ρ ← ρ + δ
      z ← CHECK(θ, ref, {s_j})
      if z > τ:  θ ← REPAIR(θ, ref, X_p);  flagged ← true
      log ← log ∪ {(δ, z, flagged)}
  return θ, log
```
**성질.** (P1) 어느 단계도 표시되지 않으면 θ = R(θ_T, k, 1): 기본 FU 와 동일. (P2) 모든 중간 모델의 모집단 변화 중앙값 ≤ c. (P3) 표시된 요청의 출력은 X_p 에서 모집단 띠 안. (P4) 멀쩡한 모델에서는 δ = 1 로 한 단계.
**비용.** 단계 수 S × (|Others| 또는 m 개의 벡터 빼기 + (m+1) 모델의 X_p 순전파) + 표시 시 E 걸음 수정. 학습 없음. S 는 보통 1.
**누적 장부 (분할 요청).** 한 클라의 요청이 조각으로 나뉘면 창 W 안의 조각 합에 대해 CHECK 를 추가로 수행한다 (`--ledger`). 여러 클라의 순차 요청은 `schedule_fu.py` 의 전체 예산 띠("정상 n 명 제거"의 띠) 를 쓴다.

---

## 3. 알고리즘 2: 쌍→집합 역할 지도 (Pair-to-Set Role Map)

표시된 요청 집합 S (보통 {k}) 에 대해, 누가 백도어를 들고 있고(운반자) 누가 또 억누르고 있는가(공동 위장자)를 빼 보기만으로 답한다. 이후 운반자 기여를 지우면 백도어가 '재워지는' 것이 아니라 '지워진다'.

```
Algorithm 2  ROLE-MAP(θ_T, S)
  θ_S ← θ_T − Σ_{s∈S} U_s
  ref, {s_j}, _ ← POPULATION(θ_T, S, 1)                 # S 전체 제외, 전량 제거 모집단
  a0 ← a(θ_S ; ref)
  # 지도
  for j ∉ S:  Δ_j ← a0 − a(θ_S − U_j ; ref)             # j 를 더 빼면 활성화가 얼마나 주는가
  # 운반자 집합 (순위 선택)
  A ← ∅;  θ ← θ_S;  best ← (a0, ∅, θ);  patience ← 0
  for j in sort_desc{Δ_j > 0}:
      A ← A ∪ {j};  θ ← θ − U_j;  cur ← a(θ ; ref)
      if cur < best.a:  best ← (cur, A, θ);  patience ← 0  else patience++
      if z(cur ; {s_j}) ≤ τ  or  patience ≥ 2:  break
  A, θ ← best
  # 공동 위장자: 혼자 빠져도 이상한 클라
  C ← { j ∉ S : z(s_j ; {s_i : i ≠ j}) > τ }
  return A, C, θ_S − Σ_{j∈A} U_j
```
**성질.** 지도 한 번(|Others| 모델) + 선택 걸음마다 모델 하나. 공동 위장자는 POPULATION 의 부산물이라 추가 비용 0. 운반자가 여럿·작아도(분산) 순위 선택이 Δ 순서로 모은다.
**범위.** 운반자 기여를 통째로 빼므로 그들의 정상 기여도 사라진다(ACC 손실). "기여 중 백도어 방향만" 빼는 것은 후속 문제.

---

## 4. 비교군 선정

| 묶음 | 비교군 | 왜 넣는가 | 어디서 |
|---|---|---|---|
| FU 기본 연산자 (방어 없음) | 빼기(subtract), 증류 FU(distillation), FedEraser(원식 보정), 재학습(정상 재참여 / 공격자 재참여) | "그냥 언러닝하면 깨어난다"의 기준선. 정의 2 의 (망각) 기준도 여기서 (θ' = R) | `benign_departure.py` |
| 언러닝 내장 방어 | FedEraser-R (보정 중 탐지+정화; 이전 제안) | 같은 저자 선행안과의 비교 | `robust_fu.py` |
| 학습 중 방어 (집계) | Median, Multi-Krum, FLAME, FLDetector | "학습 때 못 막는다" (BadFU/FUBA 주장의 재현) + 우리 방법이 그 위에서도 작동 | FUBA `--median/--multi_krum/--flame/--flDetector` 로 궤적 생성 |
| 학습 중 방어 (언러닝 진단) | MASA (WACV'25) | 가장 가까운 "언러닝으로 백도어 모델 식별". 운반자는 잡고 위장자는 못 잡음을 보임 | `baseline_masa.py` |
| 요청 시점 방어 (갱신 기하) | SubID 식 부분공간 잔차 (INFOCOM'26) | "SVD 와 무엇이 다른가". 갱신 공간 vs 제거 효과 | `baseline_subid.py` |
| 가중치 전용 | 잠재 위험 지도 (노름·형상) | 탐침 없이 가중치만으로 어디까지 되는가 | `latent_risk.py` |
| 사후 스캐너 | Neural Cleanse, STRIP | 인용(BadFU: NC 불안정; 2026: STRIP 겹침). 코드 미구현 — 범위 밖으로 명시 | — |

제외 이유를 적는 것: Zhang 2023 / Qian KDD'23 / Liu AAAI'24 (요청 샘플 자체가 필요 — FL 서버 불가), FedMUA 방어·UnlearnGuard (언러닝 라운드 갱신이 필요 — 수동형 FU 에선 없음).

---

## 5. 실험 설계 (연구 질문 → 표 → 스크립트)

설정: FUBA(MNIST, LeNet, K=8, 운반자 0–3, 위장자 4, T=8, 웜업 3), 조건 IID(주) / owner non-IID(부), 시드 3. BadFU(CIFAR-10, ResNet-18, K=6, 15 라운드)는 부정 결과로 보고.

| RQ | 질문 | 표 | 지표 | 스크립트 (run_paper.sh 단계) |
|---|---|---|---|---|
| RQ1 | 기본 FU 는 깨어나고, DOSE-FU 는 막으며, 정상 요청은 손대지 않는가 | T1: 방법 × 요청자 (위장자 / 정상 ×2) | ASR, ACC, 유지 변화, 망각(자기 데이터 acc, MIA), α, 경보·수정·단계·뺀 양 | `fu`, `dose` |
| RQ2 | 학습 중 방어와 갱신 기하 방어는 왜 안 되는가 | T2: 방어 × (운반자 AUROC, 위장자 AUROC / 순위) ; T2b: 집계 방어 위의 활성화 | AUROC, 순위, ASR after FU | `masa`, `subid`, `latent`, `tdef` |
| RQ3 | 누가 심었는지 찾아서 지울 수 있는가 | T3: 요청 × (A 정밀·재현, C, +운반자 ASR/ACC) | 정밀, 재현, ASR, ACC | `pair` |
| RQ4 | 적응형 공격에 어디까지 버티는가 | T4: 공격 5 종 × (경보, ASR) 고치기 전/후 | 경보율, ASR | `adaptive` |
| RQ5 | 비용 | T5: K × (초, 가상 모델 수, 단계 수) | 초 | `bench` + T1 의 열 |
| RQ6 | 소거: c, τ, m, 통계, 탐침 크기, 증류 연산자 | T6 | ASR, 경보 | `ablate` |

표는 `python paper_tables.py --cond iid` 가 `docs/paper/tables_iid.md` 로 만든다.

---

## 6. 보고할 때 지킬 것
- 정상 요청자의 열이 기본 FU 와 **같아야** 한다 (P1). 다르면 그 자체가 결과.
- "항상 수정" 팔을 소거로 넣어 문(gate)의 필요를 보인다.
- 불리한 결과도 그대로: SubID 식이 IID 에서 위장자를 1 위로 잡은 것, owner 소유자 오탐, 미끼 공격, BadFU 무신호.
- 적응형 변형은 저장 갱신의 근사임을 명시.
