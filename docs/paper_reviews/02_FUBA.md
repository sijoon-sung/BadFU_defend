# [2026/IEEE TAI vol.7 no.5] FUBA: Backdoor Federated Learning via Federated Unlearning (FUBA)

- **링크/코드**: IEEE Xplore https://ieeexplore.ieee.org/document/11231135 · DOI 10.1109/TAI.2025.3630110 · pp.2892–2907 (온라인 2025-11-06, 접수 2025-03-18) · 저자 **Xinyi Sheng, Wei Bao, Yichen Guo, Sen Fu (University of Sydney)** — BadFU·FedMUA 저자와 겹치지 않음 · 코드 https://github.com/stcebra/FUBA · 보충자료는 DOI 페이지(본문이 여러 번 참조; 미확보)
- **키워드/태그**: #FederatedUnlearning #RTBF악용 #DormantBackdoor #AdvAttacker/AdvDefender #TugOfWar #IBA #DBA #GradientReweighting #AdaptiveScaling #ModelConsistencyLoss #ClientLevelFU #SampleLevelFU

### 1. Problem Statement (해결하려는 핵심 문제)
- **기존 한계**: FU 연구는 유효성·효율·certified removal에만 집중, **요청의 무결성·정당성**은 "무작위·선의"라고 가정. 모든 클라이언트가 RTBF에 따라 삭제에 응해야 하는 의무가 공격 면을 키움. 서버는 정당/악의 요청을 구별할 수 없다고 명시.
- **기존 FLBA와의 차이**: 기존 백도어는 FL 중 글로벌을 직접 백도어화 → 학습 중 방어에 걸림. FUBA는 **FL 종료 시점 글로벌은 백도어 없음(ASR<0.1)**, FU 후에만 백도어화 → 학습 중 방어 우회 + 공격 출처를 FU로 고립.
- **위협 모델**: n명 중 m ≤ n/4 악성(보충 D.5). 역할은 Adv-attacker와 Adv-defender. **준조정(semi-coordinated)**: 역할·전략은 FL 시작 전에 정하고, 시작 후에는 **상호 통신 없음**(누가 어느 라운드에 뽑혔는지 모름). 각 악성 클라이언트는 트리거 패턴과 자기 데이터·모델만 알고, 서버 테스트셋·집계 규칙·FU 알고리즘·방어는 모름. 로컬 학습 전 과정 조작 가능, 정상 클라이언트·집계·FU에는 개입 불가. 충분한 연산 자원 가정.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)
- **두 조건**: (1) FL 내내 글로벌 M은 깨끗해야 하고, (2) FU 후 M̃는 **FU 방법과 무관하게** 완전히 백도어화. 이를 "줄다리기(tug-of-war)"로 모델링: attacker가 백도어 쪽으로 당기고 defender가 반대 힘으로 평형 유지 → defender 제거 시 평형이 깨져 attacker 잔여 영향이 드러남.
- **Adv-attacker (Algorithm 1)**
  1. *Gradient reweighting*: 백도어 앵커 $M^{bd}$를 학습하고, 그 위에 benign 데이터로 $M^{benign}$을 학습해 **정상 학습이 덮어쓰는 파라미터**를 측정 → $w = 1-\text{normalize}(|M^{benign}-M^{bd}|)$. 로컬 업데이트 $M_e = M_{e-1}-\eta(w\circ g_e)$ → 정상 클라이언트가 잘 안 건드리는 파라미터에 백도어를 집중(내구성).
  2. *Adaptive model update scaling*: 대표 모델 $M^{rep}$(글로벌에서 benign 학습)로 다른 클라이언트를 흉내 내고, 가상 집계 $\frac{m[M_{t-1}+\gamma(M_E-M_{t-1})]+(n-m)M^{rep}}{n}$의 ASR이 **임계 τ(=0.1)를 넘지 않는 최대 γ**(step s=0.1)를 고름 → 글로벌 ASR을 늘 10% 아래로 유지하면서 최대 강도로 주입. 뽑힌 라운드 수에 따라 자동 조절(Fig.2, Fig.4).
- **Adv-defender (Algorithm 2)**
  1. *Two-stage defense vector*: 받은 글로벌 $M_{t-1}$은 겉으로 깨끗해서 무엇을 막을지 알 수 없음 → **먼저 Algorithm 1을 그대로 실행해 공격 모델 $M^{attacker}$를 재현**(1단계), 그 모델에서 benign 데이터로 $M_{j,E}$를 학습(2단계), 방어 벡터 $=M_{j,E}-M^{attacker}$. 최종 제출 $M_j = M_{t-1}+M_{j,E}-M^{attacker}$. (Fig.5: 이 단계 없으면 FL 중 글로벌이 이미 백도어화됨.)
  2. *Model consistency loss* (식 4): $L=L_{CE}(M,D_j)+\lambda\|M-M^{attacker}\|_2^2$, λ=2. 라운드마다 방어 벡터가 흩어지지 않게(Fig.6 KPCA: 없으면 방향·크기 발산) → 제거 시 효과가 예측 가능.
- **핵심 주장 3가지** (IV-D5): (1) 언러닝을 백도어 전략으로 쓰고 두 역할의 게임으로 모델링, (2) **FU 방법 비의존(universal)**, (3) 통신 없는 협력.
- **BadFU 대비**: BadFU는 단일 클라이언트 샘플 단위·데이터만 조작·강건집계/NC만 시험. FUBA는 클라이언트 단위·**업데이트 직접 조작**(재가중·γ)·FU 6종·방어 10종·10 시드. 샘플 단위 확장도 보임(IV-D3). (BadFU 직접 인용·비교는 없음 — 미확인.)

### 3. Evaluation & Results (주요 실험 결과)
- **설정**: MNIST(2conv+3fc CNN), CIFAR-10(ResNet-18), ImageNet(MobileNetV4, 256×256). n=20, m=5 (attacker 4 : defender 1); ImageNet n=100, m=20. 악성 클라이언트는 학습 이미지 **30%** 오염. 타깃 MNIST "8", CIFAR "ship", ImageNet "hen". 100 라운드, **로컬 에폭 E=50**, B=64, 참여율 0.4, Adam lr 1e-3. **IID 기본**, non-IID는 IV-B5. **시드 10개 평균.** s=0.1, τ=0.1, λ=2.
- **FU 6종**: 재학습형 = Retrain, FedEraser, PGD-FU, RobustFU (**attacker가 재학습에 참여, 이때 γ=1 고정**, 각주 4); 비재학습형 = SGA-FU, KD-FU (악성 클라이언트 불참).
- **Table I (IBA, BA / ASR, Before → After)**:
  | | Before | Retrain | FedEraser | SGA | KD | PGD | RobustFU |
  |---|---|---|---|---|---|---|---|
  | MNIST | .992/.092 | .991/**.998** | .988/.941 | .943/.698 | .991/.700 | .990/.998 | .988/.991 |
  | CIFAR-10 | .851/.086 | .850/**.731** | .842/.729 | .781/.619 | .840/.605 | .840/.681 | .841/.673 |
  | ImageNet | .772/.029 | .770/.816 | .751/.731 | .718/.612 | .720/.658 | .740/.650 | .749/.655 |
  DBA는 전반적으로 더 높음(MNIST Retrain 1.000, CIFAR .874). **비재학습 FU(SGA/KD)에서도 ASR 0.6–0.83** → 저장된 기여 제거만으로 활성화됨.
- **Table II (방어 10종, FL 단계에만 적용, IBA, FedEraser/KD)**: B3D, MNTD, Trimmed-Mean, Median, Multi-Krum, FLDetector, FLAME, BackdoorIndicator, AlignIns, MASA 전부 Before ASR 0.05–0.10 → After **MNIST 0.93–0.97 / 0.59–0.69, CIFAR 0.68–0.76 / 0.52–0.58**. 저자 분석: attacker는 γ로 강도를 낮춰 10–20%만 탐지되고, **defender는 백도어를 주입하지 않으므로 거의 탐지되지 않음**; FLAME 등은 **FU 단계에서는 동작하지 않음**.
- **Table III (non-IID α=0.1~100)**: 모든 α에서 활성화. CIFAR α=0.1: .069 → .704/.562. 저자는 non-IID가 오히려 공격에 유리하다고 봄.
- **Fig.3 확장성**: n=20~100(m 비례)에서 효과 유지.
- **Table IV (재가중 ablation)**: CIFAR 재가중 있음 .729/.605 vs 없음 **.517/.331**.
- **Table V (어떤 요청이 활성화하는가, IID / non-IID α=0.1, FedEraser / KD)**:
  | 요청 | MNIST IID ASR | MNIST non-IID | CIFAR IID | CIFAR non-IID |
  |---|---|---|---|---|
  | 전 | .101 | .042 | .089 | .069 |
  | **정상 1명** | .123/.107 | .074/.071 | .103/.093 | .070/.065 |
  | **정상 3명** | .089/.064 | .067/.061 | .100/.089 | .063/.033 |
  | defender | .991/.700 | .882/.748 | .729/.605 | .705/.562 |
  | defender+정상 3명 | .850/.685 | .864/.616 | .715/.591 | .711/.536 |
  → defender 제거가 **필요충분**. 정상 클라이언트 제거는 non-IID α=0.1에서도 ASR을 거의 안 움직임.
- **Table VI (비용)**: RTX 4090 기준 라운드당 attacker 6.9s(MNIST)/39.1s(CIFAR), defender 9.7s/56.7s; benign은 i5에서 0.4s/76.4s, RPi4에서 2.8s/259s.
- **Table VII (샘플 단위 FUBA, DBA)**: 한 클라이언트 안에 백도어 샘플 + "defender 샘플"(트리거+정답 라벨) → Retrain ASR **1.000**, rapid retraining(INFOCOM'22) .997–1.000. 이것이 BadFU 본문 설정과 같은 구조.
- **Table VIII (저자가 시험한 잠재 방어, IBA)**:
  | 방법 | MNIST ASR / 시간 | CIFAR ASR / 시간 |
  |---|---|---|
  | Retrain (무방어) | .989 / 477s | .851 / 3639s |
  | **서버 clean셋 미세조정 + 근접항** | **.161** / 421s | **.150** / 3380s |
  | Top-k 코사인 필터링 | .163 / 635s | .186 / 4335s |
  → 둘 다 ASR을 0.15–0.19로 떨어뜨림. 저자 결론은 "더 강건한 FU가 필요"이지만, **간단한 방어 두 개가 이미 상당히 통한다**는 점이 중요.

### 4. Limitations & Critical Discussion (한계점 및 의문점)
- **재학습형 FU 결과는 두 메커니즘이 섞여 있다**: 각주 4에 따라 재학습 중 attacker가 **γ=1(전력)**으로 참여. 즉 Retrain/FedEraser/PGD/RobustFU의 ASR 0.73–1.0은 "저장된 기여 제거로 드러남"(잠복형)과 "FU 중 전력 재주입"(주입형)의 합. 순수 잠복형 효과는 **SGA/KD 열(0.6–0.83)**로 읽어야 한다. 그리고 FU 중 전력 주입은 SubID/UnlearnGuard류 FU-단계 방어가 잡을 가능성이 큰데, **저자는 방어를 FL 단계에만 적용**(Table II)했고 FU 단계 방어는 시험하지 않았다.
- **Table V의 정상 클라이언트 삭제**: IID와 Dirichlet α=0.1 모두에서 정상 1~3명 제거로는 ASR이 안 오름 → BadFU Table V와 함께 "정상 이탈 공격"의 **두 번째 반례**. 다만 FUBA에서 백도어를 누르는 힘은 defender 한 명에게 **설계적으로 집중**돼 있어 정상 클라이언트가 누르는 몫이 작은 구조. class-owner 독점 같은 극단 편중은 시험하지 않음.
- **공격자 지식 가정**: 논문은 "서버 테스트셋 접근 불가"라지만 γ 탐색은 ASR을 재야 함 → 코드는 글로벌 테스트셋으로 ASR을 계산. 논문 기준으로는 로컬 데이터로 재야 하는데 그 경우의 정확도는 미보고.
- **IID 기본, E=50 로컬 에폭**: 매우 긴 로컬 학습(비현실적 cross-silo). 악성 30% 오염. 참여율 0.4.
- **Adv-defender 탐지**: "백도어를 주입하지 않아 거의 탐지 안 됨"이라고 하지만, defender 벡터는 attacker 벡터의 **정확한 반대 성분**이라 쌍 관계는 뚜렷 — 저자가 시험한 방어는 모두 단일 클라이언트 점수형이라 이를 못 봄.
- **Top-k 필터링 결과(Table VIII)와 공개 코드 불일치**: 코드의 `--Topk`는 집계 직후 FedAvg로 덮어써 무효. 논문 수치는 보충자료 F 기준(미확보).
- **비용**: attacker/defender는 benign의 수십 배 연산(생성기 15에폭 + simulate + 재가중 2회 + γ 10회 평가). 서버는 재학습형 FU를 위해 라운드별 개별 로컬 모델 저장.
- 한계 논의는 보충자료 G.2로 넘김(미확보). 적응형 방어자(FU 단계 방어)는 미고려.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)
- **반드시 넘어야 할 기준선**: Table VIII의 **서버 clean셋 미세조정(ASR 0.15)**과 Top-k(0.19). 우리 방어가 이보다 못하면 의미가 없고, 이기려면 (a) clean 데이터 없이, (b) 더 싸게, (c) 삭제권을 더 잘 보존하면서 같은 수준에 도달해야 함. 위협 모델 축 (c)로 차별화 가능: "서버 clean 데이터 없음".
- **FU 단계 방어는 저자가 시험하지 않은 빈칸**: 재학습형 FU에서는 attacker가 γ=1로 전력 주입하므로 FU 라운드에 강건집계/탐지를 걸면 잡힐 가능성 → 이를 보이면 "재학습형 FUBA는 FU-단계 방어로 막힌다"는 결과가 하나 나오고, 남는 진짜 문제는 **비재학습 FU(KD/SGA)의 잠복형 활성화(ASR 0.6–0.83)**로 좁혀진다. 이것이 우리 방어가 서야 할 자리.
- **정상 이탈 공격은 두 논문(BadFU Table V, FUBA Table V)이 반례** → 주장하려면 class-owner 극단 편중 + 실제 FU에서 터지는지 먼저 확인하고, 안 터지면 깨끗이 접을 것.
- **쌍 검사의 위치**: FUBA 구조에서 defender는 attacker의 정확한 반대 성분이라 쌍 관계가 설계적으로 뚜렷함. 이는 FUBA 전용 신호이므로 범용성 근거로는 못 쓰지만, "기존 방어 10종이 모두 단일 클라이언트 점수형이라 쌍 관계를 못 본다"는 관찰은 Table II를 설명하는 데 쓸 수 있음.
- **평가 설계에서 가져올 것**: 시드 10개, FU 6종(재학습/비재학습 분리), 방어를 FL 단계와 FU 단계로 나눠 보고, Table V식 "누가 요청하느냐" 매트릭스.
- **함께 읽을 것**:
  - BadFU (arXiv 2508.15541) https://arxiv.org/abs/2508.15541 — 샘플 단위 상쇄형의 원형
  - FedEraser (IWQoS 2021); SGA-FU [9]; KD-FU = Wu et al. (ICLR'23 BANDS / IEEE CNS 2024) https://www.cse.psu.edu/~sxz16/papers/Federated-Unlearning.pdf; PGD-FU [16]; RobustFU (CIKM 2024) https://dl.acm.org/doi/10.1145/3627673.3679817; rapid retraining (INFOCOM'22) [17]
  - 방어: B3D, MNTD (https://arxiv.org/abs/1910.03137), Trimmed-Mean/Median (ICML'18), Multi-Krum, FLDetector (KDD'22), FLAME (USENIX Sec'22), BackdoorIndicator (USENIX Sec'24), AlignIns (CVPR'25), MASA (arXiv 2411.01040)
  - 트리거: DBA (ICLR'20), IBA (NeurIPS'23)
  - 같은 그룹: Sheng et al., Retaliatory Attacks Against FU via Data Leakage (AAAI 2026); A Survey on Federated Unlearning: Threats and Defenses (SSRN 7433722)
