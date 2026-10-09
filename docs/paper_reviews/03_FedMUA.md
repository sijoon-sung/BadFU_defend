# [2025/IEEE TIFS vol.20] FedMUA: Exploring the Vulnerabilities of Federated Learning to Malicious Unlearning Attacks (FedMUA)

- **링크/코드**: 논문 https://arxiv.org/abs/2501.11848 (HTML https://arxiv.org/html/2501.11848) / 공식 코드 https://github.com/ity207/FedMUA (FedEraser 기반, Purchase·MNIST·CIFAR-10만 제공). 저자: Jian Chen, Zehui Lin, Wanyu Lin, Wenlong Shi, Xiaoyan Yin, Di Wang — BadFU 저자와 겹치지 않음.
- **키워드/태그**: #FederatedUnlearning #MaliciousUnlearningRequest #InfluenceFunction #TargetedMisclassification #FeaturePerturbation #OverUnlearning #GradientNormIQR #FedEraser #KNOT

### 1. Problem Statement (해결하려는 핵심 문제)
- **기존 한계**: FU(FedEraser, KNOT 등)는 잊힐 권리 구현에만 집중하고, **언러닝 요청 자체가 공격 벡터**가 될 수 있다는 점을 다루지 않음. 중앙집중형 악성 언러닝 공격(Hidden Poison, Qian et al. KDD'23, Zhao et al. NeurIPS'23, Hu et al. NDSS'24)은 있으나 FL에서는 미탐구 → "FU 보안 최초 연구" 주장.
- **가정/환경**: 블랙박스(글로벌 모델 구조·파라미터·타 클라이언트 모델 모름). 공격자 = 정상 참여 클라이언트. **타깃 샘플 (x_t, y_t)을 알고 있음**. 자기 로컬 데이터 중 소수 m(α=m/n)의 **특징만** 수정해 언러닝 요청. 라벨은 건드리지 않아 서버가 정상 요청과 구분하기 어려움.
- 목표: Goal I 타깃 오분류 M(x_t, w^u)≠y_t (Eq.2), Goal II 비타깃 유틸리티 유지 ‖δ_j‖≤ε (Eq.4).

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)
- **(1) Influential Sample Identification (ISI)**: 그래디언트 유사도가 아니라 **영향 함수**(Koh & Liang).
  $I_{up,loss}(z,z_t)=-\nabla_w L(z_t,\hat w)^T H_{\hat w}^{-1}\nabla_w L(z,\hat w)$, $H_{\hat w}=\frac1m\sum_j\nabla^2_w L(z_j,\hat w)$ (공격자 로컬 데이터 기준 Hessian, Eq.5–6).
  Eq.7 $D_n=\max_n(IF(D_m))$: 타깃 손실에 가장 큰 영향을 주는 n개 선택 → Eq.8 $D_{inf}=S_p(D_n)$: 그중 **타깃과 같은 라벨**인 p개만 남김. 역Hessian 근사 방식은 미기술(미확인).
- **(2) Malicious Unlearning Generation (MUG)**: 라벨 플리핑은 탐지되기 쉬우므로 **feature-level 섭동**. $x'_j=x_j-\delta_j$ (Eq.9), 제약 $\|x'_j-x_t\|\le\zeta$ (타깃 쪽으로 당김, Eq.10), $\|\delta_j\|\le\varepsilon$, $\varepsilon_{max}=\|x_m-x_t\|$. ε↑ → 공격력↑, 유틸리티↓.
- **Over-unlearning이 생기는 이유** (직관적 설명만, 수식 유도 없음): 언러닝 대상 샘플을 타깃 근처로 옮겨 놓고 "잊어달라" 요청 → 서버가 그 샘플이 담은 **타깃 관련 정보까지** 제거 → 타깃 주변 결정경계가 이동(Fig.2). 영향력 큰 샘플을 그대로 언러닝하는 것만으로는 비효과적(ASR-B ≈ 0.6–7.5%)이라는 것이 핵심 관찰.
- **백도어형(BadFU/FUBA)과의 본질적 차이**: BadFU는 학습 시 백도어+위장 샘플을 함께 넣고 위장을 지워 **트리거 백도어를 활성화**(poison↔camouflage 상쇄 쌍). FedMUA는 **트리거 없음, 사전 포이즈닝 없음, 상쇄 구조 없음** — 정상 학습된 모델에서 언러닝 단계만으로 타깃 샘플 소수의 오분류 유도. "숨긴 것을 드러내는" 공격이 아니라 "지우는 행위 자체로 경계를 밀어내는" 공격. 요청은 clean-label·소량 섭동이라 라벨 검사로 걸러지지 않음.

### 3. Evaluation & Results (주요 실험 결과)
- **셋업**: Purchase(197,324건, 2클래스), MNIST, CIFAR-10, CIFAR-100(20/100 클래스), Credit Score(97,799×34). 모델: FedEraser → 3-FC / 2Conv+2FC / VGG, KNOT → 5-FC / LeNet5 / ResNet18. FedEraser: 20 클라이언트 중 라운드당 10 샘플링, 20 epoch, SGD lr 0.005; KNOT: 10 클라이언트, 10 local × 20 global epoch, lr 0.01. Non-IID Dirichlet(0.5). 집계 FedAvg/Median/Trimmed-mean/Krum. 타깃 40개 랜덤 평균. **기본: 악성 요청 0.3%, 악성 클라이언트 2**. 베이스라인: Rand+MUG, HP(Hidden Poison), MSFA. BadFU·라벨 플리핑과는 비교 없음.
- **핵심 수치 (IID, FU·집계 평균)**: ASR Purchase 91.2% / MNIST 83.7% / CIFAR-10 92.5% (초록은 "0.3%로 80%"). ASR-B(단순 언러닝) 7.5 / 0.6 / 4.4%. Credit Score 55%, CIFAR-100 80%. Non-IID: Purchase 70%, MNIST 48.7%, CIFAR-10 84%.
- **정확도 영향**: Purchase 87.71→87.72, MNIST 97.18→97.03, CIFAR-10 82.29→81.50 (최대 −0.79%p).
- **요청 비율/공격자 수**: 0.1%에서도 평균 ASR 75%, 10%면 전부 100%. 공격자 1명: CIFAR-10 >60%, MNIST ≈10%; 4명 ≈80%. **클라이언트 20→50이면 ASR 크게 감소.** 다중 타깃 1→9개: ASR 약 45% 감소.
- **ε 효과(FedEraser/FedAvg)**: ε_max 0.90/0.85/0.95 → 0.6ε_max 0.65/0.65/0.80.
- **강건 집계(FedEraser IID)**: Krum 0.85/0.80/0.90, Median 0.90/0.75/0.85 — 거의 무력.
- **제안 방어(Sec. VI)**: 초기 라운드 악성 클라이언트의 그래디언트 ℓ2-노름 합이 큼(Fig.3, MNIST) → **IQR 이상치 탐지 후 해당 업데이트에 λ(0.1/0.5) 곱해 축소**. λ=0.1: 평균 ASR **Purchase 35%, MNIST 15%, CIFAR-10 58%**. 기존 방어 FAT: ASR ≈65% 잔존, FADngs: 약 20%p 감소에 그침.
- **비용**: 100샘플 기준 ISI 16.3s(Purchase)/25.5s(MNIST)/62.5s(CIFAR-10), MUG ≈0.1s.

### 4. Limitations & Critical Discussion (한계점 및 의문점)
- **비현실적 가정**: 타깃 샘플 완전 접근 전제(타 기관 고객 데이터를 안다는 가정). 영향 함수는 공격자 로컬 Hessian만 사용 — 글로벌 영향력의 근사일 뿐, 역Hessian 계산법·오차 미기술. ISI 비용이 CIFAR-10 샘플 100개당 62.5s → 고차원 병목. 클라이언트 50에서 이미 ASR 급락 → 수백 규모 미검증.
- **방어의 한계**: λ 수동 고정. **FPR·정상 클라이언트 오탐 시 정확도 손실 미보고**(IQR 상한 초과는 non-IID 정상 클라이언트도 흔함). MNIST 15% vs CIFAR-10 58% 격차 설명 없음. 방어를 아는 **적응형 공격자(노름 클리핑·분산 전송) 미검증**. 방어 실험은 IID·2 공격자·0.3% 단일 설정.
- **기타**: 베이스라인 수치는 그림으로만 제시, λ 서술 자기모순, 그림 번호 오류. ASR-B가 어떤 샘플을 언러닝한 것인지 불명확. 연속 공격 효과 감소는 언급만. 한계 절 없음. 분산/신뢰구간 미보고.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)
- **요청 심사 방어와의 관계**: FedMUA 요청은 clean-label·소섭동·정상 분포 내 샘플이라 **라벨/내용 기반 심사로는 걸러지지 않음**. 유일한 신호는 언러닝 후 업데이트의 그래디언트 노름(Qian et al.의 gradient-influence 요청 심사와 같은 계열). 즉 요청 심사는 "무엇을 지우려 하는가"가 아니라 **"지웠을 때 모델이 얼마나 움직이는가"**를 봐야 한다 — 우리의 "삭제 전후 모델 기능 비교" 관점과 같은 방향.
- **백도어형 방어 범위에서의 위치**: 트리거·사전 포이즈닝·상쇄 구조가 모두 없어 **잠복형 백도어 방어의 범위 밖**(out-of-scope)으로 명시할 수 있다. 반대로 FedMUA의 노름-IQR 방어는 "정상 샘플을 지우는 것 자체가 목표"인 BadFU/정상 이탈 공격에는 무력할 가능성이 큼(언러닝 중 누구도 큰 노름을 내지 않음). 두 위협을 함께 다루려면 (i) 요청별 영향 추정 + (ii) 언러닝 후 타깃/트리거 민감도 검증의 결합이 필요.
- **평가 설계에서 가져올 것**: 타깃 40개 랜덤 평균, 요청 비율·공격자 수·클라이언트 수 스윕, "ASR-B"(단순 언러닝 대조군) 보고 관례. ε 제약은 적응형 공격자가 노름을 낮추는 방향으로 쉽게 변형 가능 — 방어 평가 시 포함.
- **함께 읽을 Reference**:
  - Qian et al., KDD 2023, "Towards Understanding and Enhancing Robustness of DL Models against Malicious Unlearning Attacks" — https://par.nsf.gov/biblio/10465449 (gradient-influence 요청 심사; FedMUA의 직접 전신)
  - Di et al., "Hidden Poison", NeurIPS 2023 — https://arxiv.org/abs/2212.10717 (HP 베이스라인)
  - Zhao et al., "Static and Sequential Malicious Attacks in the Context of Selective Forgetting", NeurIPS 2023 (MSFA 베이스라인; 링크 미확인)
  - Wang et al., "Poisoning Attacks and Defenses to Federated Unlearning" (BadUnlearn/UnlearnGuard), WWW 2025 — https://arxiv.org/abs/2501.17396
  - Liu et al., "Exploiting Machine Unlearning for Backdoor Attacks in DL" (BAU; Model-Uncertainty/Sub-Model-Similarity 탐지) — https://arxiv.org/abs/2310.10659
  - Huang et al., "Unlearn and Burn" — https://arxiv.org/abs/2410.09591 (요청 검증 메커니즘이 은밀한 요청을 못 거른다는 결과)
  - AdvUA (ICML 2024), DDPA (ICML 2025) — 후속 악성 언러닝 공격 (링크 미확인)
