# [2025/RAID] BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning (BadFU)

- **링크/코드**: arXiv https://arxiv.org/abs/2508.15541 (HTML https://arxiv.org/html/2508.15541v1) / 코드 https://github.com/BingguangLu/BadFU (로컬 클론 `BadFU_src/BadFU-main/`). 저자: Bingguang Lu, Hongsheng Hu, Yuantian Miao, Shaleeza Sohail, Chaoxiang He, Shuo Wang, Xiao Chen (Univ. of Newcastle AU + SJTU)
- **키워드/태그**: #FederatedUnlearning #BackdoorAttack #Camouflage #UnlearningActivated #FedEraser #SIFU #FedU #RobustAggregation #NeuralCleanse #CrossSilo #NonIID

### 1. Problem Statement (해결하려는 핵심 문제)
- **기존 한계**
  - 언러닝을 공격 벡터로 쓰는 선행(Hidden Poison, UBA-Inf, BAMU)은 **중앙집중 학습** 전제. FL의 집계·non-IID·클라이언트 단위 FU(FedEraser, SIFU, FedU)를 다루지 않음.
  - FedMUA 류는 **학습에 쓰이지 않은 샘플**의 삭제를 요청 → 서버가 "실제 학습된 데이터만 삭제 가능" 규칙으로 거를 수 있음.
  - UBA-Inf는 위장샘플 생성에 백도어 집합의 영향함수·대리모델이 필요 → 로컬 모델이 매 라운드 바뀌는 FL에서는 비용·비현실성 큼(논문 주장).
  - 전통 FL 백도어(model replacement, scaling)는 업데이트가 커서 Krum·FLTrust 등 이상탐지에 걸림.
- **가정/환경**: cross-silo FL, 클라이언트 K=5, non-IID(Dominant Class 70% 또는 Dirichlet α=0.3/0.5), FedAvg/FedSGD/FedProx. **악성 클라이언트 1개**(C_1)가 자기 로컬 데이터·학습만 제어(타 클라이언트·집계·테스트셋 접근 불가, 프로토콜 준수). 악성 데이터는 타깃 라벨 y_t 다수 + 전 라벨 포함 소규모 보조셋 D_attack. 언러닝 요청 대상은 **실제 학습에 쓴 샘플**(정당한 요청).

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)
- 주장: "the first backdoor attack in the context of federated unlearning". 학습 중엔 잠복(ASR 낮음), 악성 클라이언트가 **위장샘플 D_c의 언러닝을 정당하게 요청**하면 글로벌 모델이 백도어화.
- 샘플 구성 (Algorithm 1; D_bd ∩ D_c = ∅, 둘 다 D_attack에서 샘플):
  - Eq.(5) 백도어: $D_{bd}=\{(x+\tau,\,y_t)\}$ — 트리거 추가 + 라벨을 y_t로
  - Eq.(6) 위장: $D_c=\{(x+\tau,\,y)\}$ — **동일 트리거**, **원라벨 유지**(label-consistent cover). 논문 주장: 최적화·그래디언트·대리모델·백도어셋 의존 없음(UBA-Inf와의 차별점).
- Eq.(7) 악성 로컬 손실: $L(M_1)=\sum_{D_1}L(x,y)+\sum_{D_{bd}}L(x',y_t)+\sum_{D_c}L(x',y)$
- Eq.(8) 악성 기여: $\Delta M=-\eta_g\big(\sum_k p_k\nabla L(M_k)+p_1(\nabla L_{bd}+\nabla L_c)\big)$ — 본문은 ∇L_c가 ∇L_bd를 "counteract"한다고 서술하지만 **식은 같은 부호 합**이라 상쇄가 수식으로 드러나지 않음(정성 설명뿐, 증명 없음).
- 잠복 원리(정성 근거 4가지): (1) 트리거→y_t 매핑을 트리거→원라벨 매핑이 직접 상쇄, (2) FedAvg 평균화로 희석(p_1), (3) non-IID로 트리거 표현 불일치, (4) D_bd 라벨은 단일, D_c 라벨은 전 클래스 분산 → y_t 과적합 억제. 중앙학습에선 위장항만으론 부족하고 FL 집계가 있어야 효과적이라 주장.
- 활성화: 학습 종료 후 D_c 언러닝 → ∇L_c 제거 → ∇L_bd만 잔존. 근사 FU(FedEraser/FedU/SIFU)가 ∇L_c를 정확히 제거한다는 보장은 없음(실험으로만 확인).
- **차별점 요약**: UBA-Inf(영향함수+PGD 위장, 중앙) / Hidden Poison(타깃 클래스 유틸리티 저하, 중앙) / FedMUA(미학습 샘플 삭제 요청) 대비 **"트리거+원라벨"만으로 위장 + FL 정당 요청**.
- Algorithm 1 line 10–11에서 D_bd, D_c를 ∅로 리셋한 뒤 루프 도는 오타 있음.

### 3. Evaluation & Results (주요 실험 결과)
- **설정**: MNIST·CIFAR-10·CIFAR-100. 모델 SimpleNN(FC 2층), LeNet-5, ResNet-18, VGG-16(ImageNet 사전학습). K=5. non-IID Dominant 70% / Dirichlet α=0.3(MNIST), 0.5(CIFAR-10). batch 32, SGD lr 0.01, 로컬 epoch 5(FedAvg/FedProx), 1(FedSGD). 독성샘플 MNIST 1,000 / CIFAR 850(<1.7%). 트리거 BadNet 3×3 패치, Blended. FU: Retrain, FedEraser, FedU(λ=0.5, β=0.5), SIFU. 시드 3개 평균. **글로벌 라운드 수 미기재**(코드 40), 위장:백도어 기본 비율 미기재(코드 1:1).
- **Table I (BadNet, Dominant, FedAvg; ACC/ASR)** Pre → Retrain → FedEraser:
  | | Pre-activate | Retrain | FedEraser |
  |---|---|---|---|
  | MNIST SimpleNN | 97.43 / 24.08 | 97.61 / **96.95** | 97.26 / 90.36 |
  | MNIST LeNet-5 | 98.76 / 19.87 | 98.84 / 91.96 | 98.85 / 68.55 |
  | CIFAR-10 ResNet-18 | 84.84 / 21.55 | 83.90 / **99.32** | 83.19 / 90.76 |
  | CIFAR-10 VGG-16 | 87.46 / 11.33 | 87.09 / 91.02 | 86.67 / 49.23 |
- **Table IV (Blended, Dominant)**: MNIST SimpleNN 97.52/28.46 → 97.54/98.99 → 97.28/96.90; CIFAR-10 ResNet-18 85.23/15.85 → 84.71/70.30 → 83.99/53.72.
- **Table II (집계 프레임워크)**: FedSGD-BadNet CIFAR-10 87.29/12.91 → Retrain 85.00/99.84 → FedEraser 83.97/79.04. FedProx-BadNet CIFAR-10 97.52/29.80 → 97.53/99.20 (CIFAR-10 ACC 97%대는 다른 셀과 동떨어져 **표 오기 의심**).
- **Table III (Dirichlet)**: CIFAR-10 BadNet 83.63/5.65 → 85.56/98.15 → 83.78/90.20.
- **Table VIII (다른 FU)**: FedU MNIST 97.60/24.11 → 97.65/**60.12**; SIFU CIFAR-10 83.70/14.69 → 83.85/85.59.
- **Table IX (CIFAR-100, 위장:백도어 1:1 / 2:1)**: BadNet pre ASR 39.62 / 26.84 → Retrain 99.62 / 99.21 → FedEraser 72.33 / 66.10.
- **Fig.2 (위장 비율 0.5×→2×)**: pre ASR ≈40% → <20%, 활성 후 MNIST >90%, CIFAR-10 ≈80%.
- **Table V (정상 클라이언트의 언러닝, NormalUL)**: MNIST BadNet 24.09 → **14.03**; CIFAR-10 BadNet 21.55 → **10.22**; Blended 15.85 → 15.24. → 저자 결론: 정상 요청은 활성화하지 않는다(오히려 ASR 감소).
- **Table VI (방어적 집계)**: Median MNIST pre 1.14 → Activated **84.04**; CIFAR-10 6.55 → **96.33**. Trimean CIFAR-10 7.16 → 76.97. 강건집계는 잠복 ASR을 <10%로 낮추지만 활성화를 못 막음.
- **Table VII (Neural Cleanse, ResNet-18/CIFAR-10, 이상지수 < −2 탐지)**: 라벨0 Clean 0.53 / Backdoor −0.71 / BadFU −0.12 → 미탐지. Clean 모델 라벨6 −2.24로 **오탐**. 저자: NC는 중앙학습 전제, FL 모델은 과적합 적어 패턴 약함.
- UBA-Inf/Hidden Poison/위장 없는 백도어와의 **직접 비교 없음**. 클라이언트 수·독성률·라운드·요청 시점 ablation 없음(위장 비율만).

### 4. Limitations & Critical Discussion (한계점 및 의문점)
- **"정상 요청은 활성화하지 않는다"의 근거가 Table V 4케이스뿐**(FedAvg·FedEraser·정상 클라이언트 1개 제거·Dominant 70%). 이론 근거 없음. 정상 클라이언트가 타깃 클래스 다수/트리거 유사 패턴을 가진 경우, 샘플 단위 요청, 다른 FU에서의 검증 없음. → 우리 fp3/fp4(class-owner 90% 독점, 1차 근사)에서 정상 삭제가 ASR 5.8→38%로 터진 결과와 **조건이 다름**. 우리 주장을 세우려면 Table V 조건(Dominant 70%, FedEraser/Retrain)에서는 안 터지고 독점 조건에서는 터진다는 것을 **같은 FU 방법으로** 보여야 함.
- 상쇄 원리가 Eq.(8)에 수식으로 없음. 근사 FU가 ∇L_c만 깔끔히 빼는 이유 분석 없음. FedEraser가 Retrain보다 크게 낮은 사례(VGG-16 49.23%, Blended CIFAR-10 53%대) 설명 없음.
- 잠복 ASR이 11~40%로 낮지 않음(CIFAR-100 1:1은 39.6%). Median/Trimean을 쓰면 1~7%로 떨어지는데, 이는 **방어적 집계가 오히려 위장을 돕는** 모양새.
- K=5, 악성 1개 = 20% — 영향력 과대. cross-device·클라이언트 샘플링·DP-FL·Krum/FLTrust 미실험. 언러닝 비용/통신 수치 없음.
- 재현 핵심 정보(D_attack 크기, 라운드 수, 요청 시점, 위장 비율 기본값, Dominant 배정) 본문에 없음.
- **논문 vs 공개 코드 불일치** (badfu.py, prepare_data.sh, uba/uba_inf_cover.py 확인):
  1. 논문: 백도어+위장을 **한 클라이언트** 안에 두고 샘플 단위 언러닝. 코드(`ul_client_datasets`): 클라이언트 **6개** — client 0 = clean_0 + bd(850), client 1~4 정상, **client 5 = clean_0 복제 + cv(850)** → 공격자 클린 데이터가 두 클라이언트에 중복(FedAvg 가중 n_k로 영향 2배), 언러닝은 **client 5 전체를 클라이언트 단위로 제거**(FedEraser, 2라운드마다 보정 → 40R 중 20회).
  2. 논문: 위장은 트리거+원라벨만. 코드 `prepare_data.sh`: label-consistent cover 생성 후 **`uba_inf_cover.py`로 UBA-Inf의 영향함수 기반 PGD 교란**(surrogate convnext_tiny, s_test를 백도어 집합으로 계산, eps=2/255, 40 step) 적용. `badfu.py`는 `cv_pert`(교란된 위장)를 로드 → **실제 파이프라인은 UBA-Inf식 위장**. 논문이 "UBA-Inf와 달리 최적화 불필요"라고 한 주장과 어긋남. 어느 쪽이 논문 실험에 쓰였는지 미확인. 그 뒤 rm/mv 단계가 `pert_result.pt` 경로를 깨뜨림(로컬 `regen_and_detect.sh`는 이를 생략).
  3. batch 32(논문) vs 64(코드); 글로벌 40R; 초기 모델 ImageNet 사전학습 ResNet-18. Dominant 배정은 클래스 {2c, 2c+1}→client c, 70%.
  4. 코드엔 FedEraser만. Retrain/FedU/SIFU/Median/Trimean/NC 구현 없음.
  5. 로컬 클론의 `fl_detect_badfu.py, unlearn_from_traj.py, detect_*.py, regen_and_detect.sh, logs/`는 우리 추가분. 우리 재현(12R): ASR 22.4 → 34.56 — 논문(90%+)에 못 미침. 라운드 수·FU 방법 차이 때문일 가능성.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)
- **논문이 제안만 하고 미실험한 방어** (우리가 비교군/출발점으로 쓸 수 있음): (a) 요청 데이터 심사(트리거 패턴/정상 분포 이탈 검사), (b) 학습 중 + 언러닝 후 **전주기 백도어 모니터링**, (c) 언러닝 시 DP 노이즈. (d) 실험된 것: 강건집계는 잠복만 낮추고 활성화 못 막음, NC는 FL에서 무력(오탐 포함).
- **Table V가 우리 "정상 이탈 공격"의 직접적 반례이자 출발점**: 저자 조건에서는 정상 삭제가 ASR을 낮춘다. 우리 주장은 "비IID 독점 조건에서는 반대"이므로, 같은 FU(FedEraser 또는 Retrain)로 Dominant 70% vs class-owner 90%를 나란히 보여야 설득력이 생김. 이게 go/no-go 실험.
- **코드 설정(두 클라이언트 분리형)은 논문 본문(단일 클라이언트)과 다른 실험**이라는 점을 논문에 명시할 것. 쌍 검사 류 방어가 코드 설정에서만 통하는 이유가 바로 이것.
- **잠복 ASR 11~40%** 자체가 h0에 이미 백도어 흔적이 있다는 뜻 → 삭제 전 모델을 기준으로 삼는 어떤 검사든 이 기저 ASR을 감안해야 함.
- **함께 읽을 것**:
  - UBA-Inf (USENIX Security 2024): https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui / 코드 https://github.com/Huangzirui1206/UBA-Inf
  - Hidden Poison: https://arxiv.org/abs/2212.10717
  - BAU/BAMU: https://arxiv.org/abs/2310.10659
  - FedEraser (IWQoS 2021): https://ieeexplore.ieee.org/document/9521274
  - SIFU (AISTATS 2024): https://arxiv.org/abs/2211.11656
  - FedU (IEEE TDSC 2024) — 링크 미확인
  - Neural Cleanse (S&P 2019): https://ieeexplore.ieee.org/document/8835365
  - Yin et al., Median/Trimmed-mean (ICML 2018): https://arxiv.org/abs/1803.01498
  - Attack of the Tails (NeurIPS 2020): https://arxiv.org/abs/2007.05084
