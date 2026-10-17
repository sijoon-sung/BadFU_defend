# [2026/INFOCOM] Malicious Forgetting: Backdoor Injection in Active Federated Unlearning and Countermeasure Design (FUsion backdoor / SubID)

- **링크/코드**: DOI 10.1109/INFOCOM59046.2026.11571403 (IEEE Xplore, 학교 계정으로 열람). arXiv 없음. 공개 코드 없음(미확인). 저자: Wenwei Zhao, Yuanzhe Peng, Xiaowen Li, Jie Xu, Yao Liu, Zhuo Lu (USF, UF)
- **키워드/태그**: #FederatedUnlearning #ActiveFU #BackdoorInjection #SubnetworkBackdoor #SVD #SubspaceDeviation #RTBF

### 1. Problem Statement (해결하려는 핵심 문제)
- **기존 한계**: FU 보안 연구는 passive FU(요청자가 빠지고 서버가 지움)에서의 포이즈닝(BadUnlearn/UnlearnGuard [11])만 다뤘다. active FU(요청자가 언러닝 라운드에 직접 참여; 샘플·클래스 단위)에서는 참여자가 정직하다고 가정하고 아무 검증을 하지 않는다.
- **문제 발생 조건**: 학습 중에는 강한 FL 방어(FLAME 등) 때문에 포이즈닝이 어렵다. 그러나 삭제 요청을 내면 요청자가 **언러닝 라운드에 업데이트를 보낼 권리**가 생긴다. 이 라운드에는 (i) 기존 FL 방어가 적용되지 않고, (ii) 정상 클라이언트 업데이트도 데이터 제거와 교정 목적함수 때문에 불안정해서 이상치 기반 방어가 오탐한다.
- 공격 모델: n명 중 m < n/2 악성. 악성 클라이언트는 자기 데이터·업데이트는 임의 조작, 정상 클라이언트 데이터·서버 방어는 모름. **학습 중에는 완전히 정직하게 행동**하고 백도어 업데이트를 로컬에만 보관한다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)
- **FUsion backdoor (2단계)**
  1. *학습 중 준비*: 악성 클라이언트들이 협력해 **백도어 임계(BC) 층 집합**을 찾는다. 각 층 h의 트리거 기여도 $\bar\delta(h)=\sum_{i\in C_{mal}}\delta_i(h)$ 를 집계하고, 최대 층 $h^*$ 를 기준으로 $\bar\delta(h)\ge\theta\cdot\bar\delta(h^*)$ (θ=0.6)를 만족하는 **연속된** 층으로 확장한다 (식 5–8). 그 안에서 Grad-CAM으로 상위 ρ%(기본 10%)의 **트리거 민감(TS) 유닛**만 골라, 그 유닛만 백도어 데이터(+소량 clean)로 학습한 **독립 서브네트워크**를 만든다. 입력은 $z_{in}=f_{1:h_{min}-1}(x)$, 출력은 나머지 고정 층으로 흘러간다 (식 9).
  2. *언러닝 중 주입*: 삭제를 요청(로컬 데이터의 2%)한 뒤 언러닝 라운드에서 먼저 규정대로 clean 언러닝을 수행하고, 이어서 서브네트워크를 강도 β(기본 0.2)로 모델에 융합한 뒤 백도어+clean 데이터로 미세조정해 업데이트를 보낸다. 손실 가중치 λ=0.5, λ₁=1, λ₂=0.5 (정확한 손실식은 본문 식 10, 추출본에서 누락 — 미확인).
- **왜 통하나**: FedRR, ConFUSE 같은 교정형 FU는 라운드가 적고 새 특징을 학습하지 못해 "언러닝 중 백도어 데이터를 그냥 넣는" baseline은 BA≈0이다. 미리 학습해 둔 서브네트워크를 융합하면 적은 라운드로도 들어간다.
- **SubID 방어**
  - 학습 마지막 $T_r$ 라운드의 업데이트 행렬 $G_{train}\in\mathbb R^{d\times nT_r}$ 에 SVD → 상위 k 좌특이벡터 = **정상 부분공간** $S_{benign}$.
  - 언러닝을 길이 τ 창으로 나누고, 각 업데이트의 잔차 $\delta_i^t=\hat g_i^t-P_{S_{benign}}(\hat g_i^t)$ (식 13)를 모아 다시 SVD → **편차 부분공간** $S_{dev}^a$.
  - 클라이언트 점수 $\phi_i^a=\|P_{S_{dev}^a}(\hat g_i)\|_2$ (식 14). MAD 규칙 $\phi_i^a>\text{median}+\kappa\cdot\text{MAD}$ (식 15, κ=7)로 창마다 표시하고, 표시 비율이 ϕ(=0.5) 넘으면 악성 판정.
  - **판정 후 조치: 언러닝 결과를 버리고 처음부터 재학습.** 판정 중에도 모든 클라이언트를 집계에 계속 포함(언러닝 기능 보존).
- **기존 기법과의 차이**: 공격은 "학습 중 포이즈닝 불필요"(BadFU/FUBA/BadUnlearn 모두 학습 중 무언가를 심음). 방어는 UnlearnGuard(이력으로 업데이트를 예측·필터)와 달리 **학습기 부분공간 대비 언러닝기 편차의 방향 일관성**을 보고, 필터링 대신 사후 재학습을 택한다(요청자 업데이트를 거르면 언러닝이 안 되기 때문).

### 3. Evaluation & Results (주요 실험 결과)
- **설정**: MNIST(CNN), CIFAR-10(ResNet18), HAR(HAR-CNN), AG News(FastText). n=50, m=10(20%), 전원 참여, 학습 500R + 언러닝 200R, FedAvg. 트리거 10% 주입(MNIST 3×3 흰 패치, CIFAR 5×5 빨간 X, HAR 시계열 패턴, AG News 희귀 토큰). FU: NoT(재학습형), FedRR(Fisher 스크러빙), ConFUSE(혼동 업데이트). IID 기본, non-IID는 Dirichlet 0.5. 성공 기준 BA>60%.
- **공격 (Table I)**: BA / MA
  | | NoT | FedRR | ConFUSE |
  |---|---|---|---|
  | MNIST | 0.98/0.98 (baseline 0.99) | 0.99/0.98 (baseline **0.04**) | 0.98/0.93 (baseline **0.01**) |
  | CIFAR-10 | 0.99/0.87 | 0.99/0.90 (baseline 0.01) | 0.99/0.91 (baseline 0.00) |
  | HAR | 0.76/0.92 (baseline 0.13) | 0.72/0.90 | 0.78/0.79 |
  | AG News | 0.62/0.88 (baseline 0.25) | 0.68/0.82 | 0.61/0.71 |
  - 악성 비율 10%에서도 BA>90%. non-IID에서 BA>98%. ρ 작을수록 빠르게 주입(ρ=0.1: 같은 라운드에 BA 60% vs ρ=0.5: 36%).
  - 집계: Trim은 BA 0.1–2.8%p만 감소. **Median은 BA<15%로 억제하지만 MA 손실**, 특히 ConFUSE에서 큼. 저자도 non-IID와 요청자 업데이트 배제 문제로 Median은 배치 부적합이라고 봄.
- **방어 (Table II, F1)**: SubID F1=1.0 (3개 FU 모두) vs UnlearnGuard-Dist/Dir F1<0.4. MNIST·CIFAR TPR 1.0/FPR 0. HAR·AG News는 ConFUSE에서 TPR 0.8, FPR 0.03–0.08. non-IID TPR 1/FPR 0.
- **강건성 (Table III)**: 공격 없음 FPR 0(ConFUSE 0.05). Trim 공격 TPR 0.7–1.0. **Sign Flip 공격 TPR 0.0–0.2** (극단 업데이트가 median/MAD를 부풀려 임계가 느슨해짐). κ는 1–8에서 TPR 1.0, >8이면 하락; ϕ는 0.3–0.7이 안전.

### 4. Limitations & Critical Discussion (한계점 및 의문점)
- **SubID의 전제 두 가지가 명시적이다**: (i) 학습기 업데이트는 정상이다(강한 FL 방어가 학습 중 포이즈닝을 막는다고 가정, 본문 IV-A), (ii) 공격은 언러닝 중 **새 방향**을 만든다("data removal은 새 특징을 도입하지 않는다"). → **학습 중에 심어 두고 언러닝으로 드러나는 잠복형(BadFU/FUBA)에는 적용 불가**: 백도어 방향이 이미 $S_{benign}$ 안에 있고, 언러닝 중 아무도 새 방향을 만들지 않는다.
- **판정 후 조치가 재학습**: 잠복형에서는 요청자를 뺀 재학습이 곧 공격의 완성이다(BadFU: retrain ASR 95%). 즉 SubID의 "대응"이 잠복형에는 역효과.
- **적응형 공격 주장이 약하다**: "학습 중엔 방어 때문에 공격을 못 하니 언러닝 중 공격할 수밖에 없고 그러면 들킨다"는 논증뿐, 실험 없음. Sign Flip에 TPR 0.2라는 결과 자체가 임계 통계(median/MAD)가 쉽게 흔들린다는 증거.
- **비용·저장**: 서버가 마지막 $T_r$ 라운드 × n개 전체 업데이트($d\times nT_r$)를 저장하고 SVD. d가 ResNet18 규모면 비용 보고 없음(미보고). 언러닝 창마다 SVD 반복.
- 하이퍼파라미터 3개(k, κ, ϕ)가 데이터셋마다 조정 필요하다고 저자가 인정.
- 모든 클라이언트 전원 참여·FedAvg 기본, 부분 참여 미실험. 서버가 개별 업데이트를 본다고 가정(secure aggregation 비호환).
- 공격 쪽: 악성 클라이언트들이 학습 중 **서로 협력**(코디네이터가 Grad-CAM 결과 공유)해야 하며, 요청자 전원이 공격자.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)
- **지형 정리에 쓸 칸**: 언러닝 공격은 "언러닝 중 주입형"(BadUnlearn=passive, FUsion=active)과 "학습 중 심고 제거로 활성화하는 잠복형"(BadFU, FUBA)으로 갈린다. 방어도 같은 축으로 갈린다: UnlearnGuard·SubID는 전자 전용이고 후자에는 전제가 깨진다. 잠복형 방어는 비어 있다.
- **반드시 비교군으로 넣어야 함**: SubID와 UnlearnGuard-Dist/Dir을 BadFU/FUBA 궤적에 돌려서 (a) 잠복형에서 탐지 실패, (b) SubID의 재학습 대응이 ASR을 올림을 보인다. 이것이 "SVD 쓰면 SubID랑 같다"는 지적에 대한 실험적 답이 된다.
- **메커니즘 겹침 주의**: 저장된 업데이트의 SVD로 "일관된 편차 방향"을 찾는 탐지는 SubID가 선점했다. 우리 쪽은 **업데이트 공간이 아니라 모델 기능(삭제 전/후 출력 차이)**을 보는 쪽으로 가야 차별화된다.
- **가져올 설계 요소**: (i) 판정 중에도 요청자 업데이트를 거르지 않는다는 원칙(삭제권 보존)은 우리와 같음. (ii) 정상 요청을 포함한 "공격 없음" 조건에서 FPR을 보고하는 실험 관례. (iii) Sign Flip에서 MAD 임계가 무너진 사례 — 우리 임계도 정상 기준분포로 잡을 때 같은 함정 점검.
- **함께 읽을 것**: UnlearnGuard/BadUnlearn — "Poisoning Attacks and Defenses to Federated Unlearning" (WWW Companion 2025, https://arxiv.org/abs/2501.17396); NoT, FedRR, ConFUSE (active FU 3종, 본문 [3][23][45]); BC-layer 선행 [28], LTH 기반 서브네트워크 [30].
