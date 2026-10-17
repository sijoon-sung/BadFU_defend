# FL→FU 방어 설계를 위한 논문 12편 검토

2026-10-14. 아래 수치는 **각 원논문이 보고한 해당 조건의 결과**이며 우리의 재현 결과가 아니다. 서로 다른 표의 수치를 묶어 우열을 매기지 않는다. 차이는 상대 개선율이 아닌 퍼센트포인트(pp)로 계산했다. 하드웨어를 확인하지 못한 경우 추정하지 않았다. 각 항목의 ‘우리 해석’은 저자의 주장과 구분한다.

2026-10-16 추가: H의 FLShield, I의 FLTrust와 기존 MASA·BackdoorIndicator·FedREDefense 원문을 대조해 **Non-IID 혼동을 줄이기 위해 무엇을 추가로 관측하는가**를 정리했다. 현재 우리 조건과 구체적인 원인 분리는 [연구 설명](../experiments/RESEARCH_HISTORY.md#논문을-설명하는-현재-문장--2026-10-16)과 [Non-IID 진단 계획](../experiments/RESEARCH_HISTORY.md#non-iid가-탐지를-방해하는-경로와-다음-원인-분리)에 반영했다.

## 먼저 비교할 논리

| 논문 | 기존 관측의 실패 | 새로 도입한 정보/조치 | 신규성을 입증하는 연결 | 우리에게 필요한 검증 |
|---|---|---|---|---|
| MASA | 악성·정상 갱신의 정적 통계가 겹침 | 개별 모델의 깨끗한 과제 망각 동역학 | 새 관측 + non-IID 혼동을 줄이는 융합 | raw 좌표 vs 입력 쌍 좌표, 희귀 클래스 손실 |
| BackdoorIndicator | 수동 갱신 검사로는 정보 부족 | 서버가 OOD 지표 과제를 먼저 주입 | 관측 가능성을 능동적으로 만듦 + BN 혼동 통제 | 공개 변환의 정보가 부족한지 먼저 판정 |
| FedREDefense | 다수 클라이언트와 비슷함이 정상성은 아님 | 개별 업데이트의 학습 과정 재구성 | 정상 비교 기준 자체를 바꿈 | 장력의 절대 크기 대신 삭제·실제 갱신과의 관계 |
| DeepSight | 이질적 정상 모델도 이상치 | 내부 표현 검사와 군집, 잔여 영향 clipping | 강한 공격은 드러나고 약한 공격은 영향 제한 | 완벽한 신원 판별 없이 유해 방향만 제어 |
| FLAME | 필터 누락, 잡음 증가 시 효용 손실 | 군집·clipping·잡음의 순차 결합 | 앞 단계가 뒤 단계에 필요한 잡음을 감소 | 이력 선택이 같은 방어 강도에서 효용을 보존하는가 |
| UnlearnGuard | FU 중 새 악성 업데이트 | FL 이력 기반 FU 예측·검사 | 과거의 시간적 일관성을 FU에서 재사용 | ‘이력을 사용’ 이상으로 삭제 의존성의 이득을 보여야 함 |
| FedEraser | 처음부터 재학습하는 FU의 비용 | 옛 갱신 노름 + 새 보존 갱신 방향 | 저장·통신·계산과 망각 근사의 교환관계 | 우리가 제어할 실제 연산자와 비교 비용의 기준 |

**읽는 순서:** MASA → DeepSight → FLAME → UnlearnGuard → FedEraser → BackdoorIndicator → FedREDefense. 앞의 세 편은 논리 전개, 다음 두 편은 우리 주장과 구현의 직접 경계다.

시계열 방향을 위한 **접근 정보·추가 권한·완화할 조건**은 문서 끝의 비교표에 추가했다. FLDetector와 FoolsGold는 이번 정보 조건 검토에서 보강한 선행 연구이며, 아래 7편 상세 결과 리뷰와 구분한다.

## A. MASA — Identify Backdoored Model in Federated Learning via Individual Unlearning

원문: [WACV 논문의 공개 버전](https://arxiv.org/html/2411.01040v1), §§3–6, Tables 2–4. [저자 코드](https://github.com/JiiahaoXU/MASA).

### 1. Problem Statement (해결하려는 핵심 문제)

백도어와 정상 과제를 함께 학습하면 노름·방향만으로 악성 갱신을 가리기 어렵다. non-IID에서는 정상 모델의 반응도 달라 단순 이상치 검사가 혼동한다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

깨끗한 proxy에서 각 모델의 손실을 **상승**시켜 누적 손실을 관측한다. 먼저
\(\tilde\theta_i=\theta+\lambda\Delta_i+(1-\lambda)\bar\Delta\)로 융합한다. 점수는 \((L_i-\operatorname{median}L)/\operatorname{std}L\); 분모는 MAD가 아니다. **신규성 전개:** 정적 관측 실패 → 동역학 관측 → 정상 이질성을 통제하는 융합 → 이상 갱신 제외.

### 3. Evaluation & Results (주요 실험 결과)

Table 2, CIFAR-100/VGG16, Dirichlet α=0.1, Badnet: Lockdown MA/RA=55.71/6.84%, MASA=56.10/50.00%. 차이는 **MA +0.39pp, RA +43.16pp**. RA는 트리거 입력의 원래 정답 정확도이며 ASR이 아니다. CIFAR-10/ResNet18도 평가, 기본 20클라이언트·20% 악성·proxy 1%·개별 망각 5epoch. 검토 원문에서 GPU 기종은 확인하지 못했다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

과제와 겹치는 깨끗한 proxy가 필요하고 참여 모델마다 추가 최적화한다. 우리 해석: 잠복 상태의 분산 상쇄가 같은 동역학을 보인다는 보장은 없다. 모델 융합의 오탐 감소를 모든 non-IID에서 의도 식별이 가능하다는 정리로 읽으면 안 된다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

우리도 새 장력 좌표만 제시하지 말고 **정상 혼동을 줄인 원인**을 소거해야 한다. 융합과 임계값 개선을 우리 장력의 신규성으로 재포장하지 않는다. 함께 읽기: 아래 B, D 및 [원래 BadFU 검토](01_BadFU.md).

## B. BackdoorIndicator — Leveraging OOD Data for Proactive Backdoor Detection in Federated Learning

원문: [공개 논문](https://arxiv.org/html/2405.20862v1), §3, §4.1, Table 4. [저자 코드](https://github.com/ybdai7/Backdoor-indicator-defense).

### 1. Problem Statement (해결하려는 핵심 문제)

정상·악성 갱신이 비슷하면 수동 통계로 탐지하기 어렵다. 정상 학습은 지표 과제를 잊고 특정 백도어 학습은 유지한다는 행동 차이를 활용한다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

서버가 OOD 입력에 지표 과제를 학습시킨 뒤, 반환 모델의 클래스별 지표 정확도 최대값 \(\max_c\mathrm{Acc}_c\)을 검사한다. 평가 전 지표용 BN 통계를 복원한다. **신규성 전개:** 기존 정보의 한계 → 능동 관측 추가 → 분포 이동의 BN 교란 제거 → 탐지.

### 3. Evaluation & Results (주요 실험 결과)

Table 4, CIFAR-10, α=0.2, vanilla pixel, plr=.01, 1200라운드부터 250회 공격: DeepSight TPR/FPR/BA=10.4/6.0/31.3%, Indicator=98.0/19.7/11.5%. **BA −19.8pp이나 FPR +13.7pp**다. 무방어 BA=66.8%. CIFAR-100·EMNIST도 평가. 100명 중 10명 참여, RTX4090 24GB, 기본 지표 800개·서버 학습 200iteration/round.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

OOD 지표의 유지가 검사할 백도어와 연결되어야 한다. 우리 해석: 잠복 공격의 억제 단계가 그 연결도 가릴 수 있다. 추가 서버 학습과 OOD 자료 비용이 있고 위 조건의 정상 오제외 19.7%를 무시할 수 없다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

공개 변환에 신호가 전혀 없으면 점수 고도화보다 관측 정보 추가를 검토한다. BN이 관측을 바꾸므로 CIFAR 어댑터에서 buffers를 무시하면 안 된다. 함께 읽기: A 및 [FUBA 검토](02_FUBA.md).

## C. FedREDefense — Model Update Reconstruction Error

원문: [ICML 2024 공식 페이지와 PDF](https://proceedings.mlr.press/v235/xie24c.html), §§4–5, Tables 1, 9. [저자 저장소·정오표](https://github.com/xyq7/FedREDefense). 공식 15쪽 PDF Table 1을 시각 대조했다.

### 1. Problem Statement (해결하려는 핵심 문제)

클라이언트 간 유사성과 정상 다수에 의존하는 검사는 강한 non-IID와 많은 악성 클라이언트에서 취약하다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

작은 합성 지식 K로 업데이트를 재구성한다:
\(\min_K\|A(K,\theta,N)-\theta-u_i\|^2\).
업데이트 노름으로 정규화한 오차를 검사하고 과거 K를 재사용한다. **신규성 전개:** 집단 비교 기준의 실패 → 개별 갱신이 학습으로 생성 가능한지 검사 → 동적 최적화로 비용 감소.

### 3. Evaluation & Results (주요 실험 결과)

Table 1, CIFAR-10/Fang: FLDetector DACC/FPR/FNR=94/8.33/0%, FedREDefense=100/0/0%; **DACC +6pp, FPR −8.33pp**. FashionMNIST·CINIC도 평가. 기본 100명·28명 악성, β=.1, 500round, K=10쌍. Table 9의 클라이언트당 평균 최적화 횟수는 continuous 60.88 → dynamic 2.40이며 wall-clock 배속이 아니다. GPU 기종은 확인하지 못했다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

우리 해석: poisoned-data SGD도 실제 학습이므로 재구성 가능성이 악의의 부재를 뜻하지 않는다. 학습 과정을 통한 역전파 비용과 합성 지식 저장이 필요하다. Table 8의 FashionMNIST/Min-Max/FLTrust에는 저자 정오표가 있으므로 예전 수치를 비교 기준으로 재사용하지 않는다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

점수의 이름보다 **어떤 정상성 기준을 바꿨는가**가 기여다. 우리 기준은 집단 평균과의 거리보다 ‘상쇄 관계가 이번 삭제와 실제 갱신에서 풀리는가’여야 한다. 함께 읽기: F 및 A.

## D. DeepSight — Mitigating Backdoor Attacks through Deep Model Inspection

원문: [NDSS 2022 PDF](https://www.ndss-symposium.org/wp-content/uploads/2022-156-paper.pdf), Introduction, §§V–VI, Table II.

### 1. Problem Statement (해결하려는 핵심 문제)

단순 이상치 제외는 다른 데이터 분포의 정상 사용자를 배제하고, clipping만으로는 강한 공격을 막기 어렵다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

출력 변화 DDifs, 정규화 에너지 NEUP와 threshold exceedings 등 내부 특징으로 모델을 검사·군집화한다. 강한 영향의 군집을 제외하고 남은 영향은 clipping한다. **신규성 전개:** 공격을 강하게 하면 관측 흔적이 커지고 약하게 하면 영향 제한에 취약하다는 두 단계 설계.

### 3. Evaluation & Results (주요 실험 결과)

Table II, Reddit text prediction: FLGuard BA/MA/precision=0/21.7/20.4%, DeepSight=0/22.6/100%; **BA가 같은 상태에서 MA +0.9pp, precision +79.6pp**. NIDS에서 무방어 BA100% → DeepSight0%, MA100% 유지. 이 NIDS의 MA는 정상 트래픽 TNR이다. Ubuntu18.04, Xeon 물리20코어, RTX2080Ti 11GB×4, RAM192GB.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

우리 해석: 그 공격의 강도–흔적 관계가 FU로 나중에 활성화되는 기여에도 성립하는지는 미검증이다. 다중 표현과 군집은 클라이언트 수·모델 크기에 따른 서버 비용을 만든다. 표의 precision을 다른 논문의 TPR과 혼동하지 않는다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

장력은 완전한 공격자 식별에 실패해도 조치 대상을 좁히는 데 쓸 수 있다. 다만 ‘강한 잠복 공격은 반드시 관측된다’는 보편 주장은 버리고 관측 가정을 명시한다. 함께 읽기: E 및 A.

## E. FLAME — Taming Backdoors in Federated Learning

원문: [USENIX Security 2022 PDF](https://www.usenix.org/system/files/sec22-nguyen.pdf), §§4–7, Tables 1–4.

### 1. Problem Statement (해결하려는 핵심 문제)

필터링만으로 남는 공격이 있고, 이를 잡음만으로 지우려 하면 정상 정확도를 손상시킨다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

방향 기반 군집 → 업데이트 노름 중앙값 S로 clipping → S에 비례하는 Gaussian 잡음. **신규성 전개:** 기존 부품을 나열하지 않고 앞의 두 단계가 필요한 잡음을 줄이는 이유와 소거 결과를 제시한다.

### 3. Evaluation & Results (주요 실험 결과)

Table 4, CIFAR-10/constrain-and-scale: DP BA/MA=0/78.9%, FLAME=0/91.9%; **같은 BA0에서 MA +13.0pp**. Table 1, IoT: 잡음만 σ=.10에서 BA/MA=0/54.2%, 전체 σ=.01에서 0/100%; 잡음 크기 10분의1, MA +45.8pp. Reddit/LSTM, IoT/GRU, CIFAR-10·TinyImageNet/ResNet, MNIST/CNN. GPU 기종은 검토 원문에서 확인하지 못했다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

정상 과반 가정과 노름 기준이 있다. 저자도 DNN 일반 보안 증명은 미해결로 둔다. 우리 해석: FU의 정상적 방향 이동이 기존 군집에서 벗어나면 정상 보정이 제거될 수 있다. §6의 TPR/TNR 명칭은 제시 분모가 precision/NPV 형태이므로 타 논문 지표와 직접 병합하지 않는다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

우리 이력 선택이 이력 없는 제어보다 **같은 ASR에서 적은 효용 손실**, 또는 같은 효용에서 낮은 ASR을 주는 곡선을 제시해야 한다. 함께 읽기: D, F.

## F. UnlearnGuard — Poisoning Attacks and Defenses to Federated Unlearning

원문: [WWW Companion 2025 공개 버전](https://arxiv.org/html/2501.17396v1), §4, Theorem 1, §5, Tables 3–7.

### 1. Problem Statement (해결하려는 핵심 문제)

FU 도중 참여자가 새 악성 업데이트를 제출하면 FU가 과거 오염을 유지할 수 있다. FL에 잠복한 억제가 정직한 삭제로 사라지는 문제와는 구분한다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

FL 이력과 Hessian 근사로 \(\bar g_i^t=\tilde g_i^t+H_i^t(w^t-\tilde w^t)\)를 예측하고 거리·방향을 검사한다. **신규성 전개:** FU의 공격면 → FL 이력의 예측 기준 → 부적합 업데이트 검사 → 조건부 재학습 근접성 분석.

### 3. Evaluation & Results (주요 실험 결과)

Table 3, MNIST/CNN, FL backdoor→FU BadUnlearn, FedAvg: FedRecover TER/ASR=.04/.98, Dist=.03/.01, Dir=.03/.00. **Dir의 ASR −98pp, TER −1pp**. 2000round, FL100명/FU80명, 각 단계 악성20%. 저장은 실험에서 최대200GB. GPU 기종은 확인하지 못했다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

정리에는 FL 악성 클라이언트가 모두 탐지되었다는 조건이 들어간다. 우리 해석: 잠복 공격에 그대로 적용할 수 없는 중요한 차이다. 이력 공간은 총 O(ndT); ‘O(d)’만 인용하면 안 된다. 재학습 근접성 자체가 보존 데이터의 백도어 제거를 뜻하지 않는다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

‘FL 이력을 활용한 FU 방어’는 독립 신규성에서 제외한다. history-free, 선형 삭제 영향, 이력 섞기, 가능하면 UnlearnGuard 이식 기준선을 같은 FU에서 비교한다. 함께 읽기: G, [FUBA](02_FUBA.md), [BadFU](01_BadFU.md).

## G. FedEraser — 방어가 작동할 FU 연산자

원문: [IWQoS 2021 연구의 공개 PDF](https://arxiv.org/pdf/2012.13891), §III, Eq.1, §IV, Table II, Fig.4.

### 1. Problem Statement (해결하려는 핵심 문제)

삭제 요청마다 전체 FL을 재학습하는 비용을 줄이려는 FU 방법이다. 백도어 방어 논문으로 분류하지 않는다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

저장 클라이언트 갱신의 노름과 새 보존 데이터 보정의 방향을 결합한다:
\(\tilde u_j=\|u_j^{FL}\|\hat u_j/\|\hat u_j\|\).
**각 클라이언트를 먼저 보정하고 집계**한다. 필요한 과거 정보는 모델 업데이트이며 모든 minibatch gradient는 아니다.

### 3. Evaluation & Results (주요 실험 결과)

§IV-B: Purchase 시험 정확도 FedAccum91.3%, FedEraser94.3%, FedRetrain94.9%; **누적만 하는 방식보다 +3.0pp, 재학습보다 −0.6pp**. 기본 보정비율 .5에서 Purchase 재구성 4.1× 가속. Fig.4의 비율 .1에서는 19.6초/8.4×로 조건이 다르다. Adult·MNIST·CIFAR-10도 평가. 원문 하드웨어 표기는 ‘Core i7 9400 / GeForce GTX 2070’; 표기를 그대로 인용하며 제품명 오류 가능성이 있다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

남은 클라이언트 참여와 기록 저장이 필요하고 근사 망각이다. 우리 해석: 보정 방향을 다시 바꾸면 원래의 유틸리티·망각 결과를 승계할 수 없다. 옛 노름과 삭제 유래 이력 자체의 잔존 의존성도 검증 대상이다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

새 벡터 코어는 클라이언트별 보정 후 집계를 구현한다. 기존 `federaser_faithful.py`의 평균 후 층별 보정과 구분한다. 이 코어 위에 우리 제어를 붙이고, 같은 방어 목적의 보존 데이터 재학습과 비교한다. 함께 읽기: F, [실행 가능한 설계](../paper/TENSION_CONTROL.md).

## H. FLShield — 클래스별 검증으로 데이터 구성의 혼동 감소

원문: [공개 v1, §§IV-C, V-B/C, VI-B2, VI-D2](https://arxiv.org/html/2308.05832v1). [IEEE S&P 2024 저자 코드](https://github.com/ehsanul9511/FLShield). 아래 수치는 읽은 공개 v1 기준이다.

### 1. Problem Statement (해결하려는 핵심 문제)

Non-IID에서는 정상 검증 결과도 달라 악성 업데이트·거짓 보고와 혼동된다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

클라이언트가 대표 모델의 **정답 클래스별 손실 감소(LIPC)**를 보고한다. 서버는 거짓 보고를 걸러 업데이트를 선택한다. 신규성은 검증 대상과 보고의 신뢰 문제를 대표 모델·클래스별 변화로 함께 다루는 데 있다.

### 3. Evaluation & Results (주요 실험 결과)

CIFAR-10/TLFA, Table III(b): 클래스별 정확도 변화 대신 LIPC를 쓰면 source-class recall은 25.34→88.66%, 악성 대표 모델 수용률은 26→0%. one-class-expert·Dirichlet도 평가한다. 하드웨어는 이번에 확인하지 않았다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

악성 비율 50% 미만, 라벨 있는 검증 자료, 추가 통신·평가를 요구한다. 클래스당 10~30장을 사용하며 결측은 보간한다. 우리 해석: 잠복 백도어에 대한 보편 보장은 없다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

클래스별 검사를 선행 기법으로 두고, 우리 H–U 관계의 추가 가치를 소거한다. 함께 읽기: A의 MASA, 아래 I.

## I. FLTrust — 작은 정상 자료로 신뢰 기준 추가

원문: [NDSS 2021 공식 논문 페이지](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/), [PDF](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf).

### 1. Problem Statement (해결하려는 핵심 문제)

클라이언트 모두가 악성일 수 있는 관점에서는 업데이트끼리 비교하는 것만으로 신뢰 기준이 생기지 않는다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

서버가 작은 깨끗한 root dataset으로 기준 업데이트를 계산한다. 클라이언트 업데이트와의 코사인 유사도를 음수에서 절단해 신뢰 점수를 주고, 노름을 기준 업데이트에 맞춰 가중 집계한다. 신규성의 논리는 추가 신뢰 자료 → 방향 기준 → 크기 제한의 연결이다.

### 3. Evaluation & Results (주요 실험 결과)

공식 초록은 6개 데이터셋에서 평가하고, root 자료 100개 미만·적응 공격자 40~60% 조건에서 무공격 FedAvg와 비슷한 정확도를 보고한다. 이는 원논문의 조건별 주장이고 우리 실험의 보장이나 일반적인 필요 표본 수가 아니다. 이번 요약에서는 세부 정확도와 하드웨어를 새로 전사하지 않았다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

신뢰할 수 있고 과제를 대표하는 라벨 자료와 서버 학습이 필요하다. root가 오염되거나 특정 클래스에 치우치면 신뢰 기준도 흔들린다. 하나의 기준 방향과 다른 정상 희귀 클래스 기여를 자동으로 공격으로 해석할 수 없다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

공개 정상 자료 가정은 이미 방어 연구에서 사용하는 선택이다. 우리도 이미 공개 이미지를 사용하므로 ‘자료 추가’ 자체를 신규성으로 삼지 않는다. 라벨을 추가할 경우 검사 이미지별 정답을 아는 것과 클라이언트 원자료를 보는 것을 구분한다. 함께 읽기: H, C의 FedREDefense.

## 우리 논문의 주장에 반영할 결정

현재 연구의 기준은 [삭제 후 잠복 백도어 교정의 범위와 다음 실험](../current/RESEARCH_SUMMARY.md#현재-연구-범위와-다음-실험--2026-10-17)이다. 아래 제어 관련 문단은 이전 확장 구상이며 현재 구현의 성능이나 보장이 아니다. 기존 정상 설명 집합의 관측 실패와 새 교정 가설을 구분한다.

**제약 최적화의 직접 선행:** [FedOSD, AAAI 2025](https://arxiv.org/abs/2412.20200)는 FU의 gradient 충돌과 후속 학습의 재기억을 다룬다. [GEM](https://arxiv.org/abs/1706.08840)과 [UPGP](https://www.sciencedirect.com/science/article/pii/S0031320326001767)도 gradient 제약/투영을 사용한다. 따라서 최적화·직교화 자체를 신규성으로 주장하지 않는다. 역사적 상쇄 연산자의 이득은 아직 입증되지 않았고, 현재는 [삭제 전후 변형 탐색·CVaR·post 기준 클래스 제약](../current/PURIFICATION_COMPARE.md)을 분리해 비교한다. 이는 이 선행 방법들의 정량 재현이 아니다.

- **남길 문제:** FL에서 공동 학습된 주입·억제 중 FU가 억제를 제거하여 활성화하는 백도어.
- **내릴 주장:** 큰 장력=악성, 모든 non-IID 구분, FU 시뮬레이션 자체의 신규성, 모든 BadFU에 대한 이미 완성된 방어.
- **새 검증 대상:** 과거의 선택적 상쇄 해제와 실제 FU 갱신의 증가가 겹치는 좌표를 제어하면 같은 효용에서 ASR을 더 낮추는가.
- **버리면 안 되는 비교:** 동일 제어기의 이력 제거·선형 경계·이력 섞기, 일반 clipping, 보존 손실 제어만. 장부에 의존한 방법이 이들을 이기지 못하면 과정 정보의 기여는 기각한다.
- **non-IID 해석:** 위 논문들은 주로 비IID로 인한 방어 혼동을 다룬다. 공격 없는 정상 모델의 표적 오분류와 악성 의도를 보편적으로 분리해 준 논문들로 묶지 않는다. 정상 조건에서의 target rate와 트리거에 의한 추가 변화, 클래스별 ACC를 함께 보고해야 한다.

## 시계열 설계를 위한 정보 조건 비교 (2026-10-14)

아래는 각 방법이 무엇을 관측하거나 추가하는지 비교한 것이다. 요구 정보가 있다는 사실이 우리 잠복 공격에서도 그 정보가 충분하다는 증거는 아니다. ‘우리에게 주는 선택’은 설계 해석이다. 시계열 조건의 수식·구현 격차·실험 순서는 [TENSION_CONTROL §6](../paper/TENSION_CONTROL.md#6-업데이트를-시계열로-볼-때의-정보-조건과-완화할-제약)에 정리했다.

**기존 구현과의 대조:** 우리도 R-012에서 복원력·지연 상관·누적 장력을, R-013에서 연속 경보를 이미 시험했다. R-034에는 요청별 전체/10/20창 비교가 있고 R-039에는 라벨 proxy128개와 실제 FU 한 단계 제어가 있다. 따라서 아래 ‘완화’ 중 정상 proxy·추가 보정 허용은 기존 실험 조건을 명문화하는 항목이며 새로 처음 도입하는 능력이 아니다. 특히 지연 상관의 기존 음성 결과(위장자 최상위0/6)를 무시하고 시계열이라는 이름만으로 재추천하지 않는다.

| 방법 | 관측 정보 / 허용되는 행동 | 시간 이력의 역할 | 우리에게 주는 선택과 경계 |
|---|---|---|---|
| [FoolsGold, RAID 2020](https://www.usenix.org/system/files/raid20-fung.pdf), §6 | ID에 연결된 개별 gradient/update, 누적 기록 | 누적 업데이트의 유사성으로 일회성 잡음 완화 | 정상·공격 모두 시간축으로 볼 수 있다는 선행 근거. 같은 목표의 여러 sybil 간 유사성이 핵심이며 한 client 내부 상쇄를 자동 분리하지 못함. 누적합 자체는 순서에 불변 |
| [FLDetector, KDD 2022](https://arxiv.org/html/2207.09209v4), §§3–4 | 개별 업데이트·전역 모델 이력 | L-BFGS 기반 예측과 여러 iteration의 불일치 | 시간적 일관성 방어가 이미 존재. 일관되게 잠복한 상쇄를 요청과 연결하는 것이 우리 검증 대상; 단순 예측기 추가만으로 신규성을 주장하지 않음 |
| [UnlearnGuard](https://arxiv.org/html/2501.17396v1), §4 | FL 개별·전역 이력, FU에서 필요한 정확한 client 업데이트 재요청 | FU 예측·거리/방향 검사·재보정 | 추가 client 통신을 금지할 필요는 없음. 과거 악성 client 제거 등 이론 조건을 우리 위협에 그대로 가져오지 않음 |
| [FedEraser](https://arxiv.org/pdf/2012.13891), §III | 저장된 개별 모델 업데이트, 보존 client의 재학습 | 과거 노름과 새 방향의 보정 | 모든 minibatch gradient가 필요하다는 가정은 불필요. 저장 범위·주기와 요청 해상도는 따로 확인 |
| [MASA](https://arxiv.org/html/2411.01040v1), §§4–5, 8.4 | 개별 모델, 작은 정상 proxy, 추가 서버 최적화 | 새로 수행한 개별 망각의 손실 궤적; FL 역사 자체와 다름 | 소량의 정상 자료 허용은 선행 설정과 비교 가능. 자료 크기뿐 아니라 non-IID와 클래스 커버리지를 시험해야 함 |
| [BackdoorIndicator](https://arxiv.org/html/2405.20862v1), §3 | 업로드 모델의 white-box 접근, OOD 자료, 서버의 지표 과제 주입 | 능동 지표의 유지/망각을 관측 | ID 정상 자료 없이도 추가 관측 과제를 만드는 선택. 수동 시계열과 정보·학습 개입 수준이 다름 |
| [DeepSight](https://www.ndss-symposium.org/wp-content/uploads/2022-156-paper.pdf), §§IV–V | 개별 모델의 파라미터·출력과 내부 특징 검사 | 핵심 설계가 장기 요청 조건 시계열은 아님 | 여러 관측을 결합할 수 있지만 unknown trigger를 모두 볼 수 있다는 근거는 아님 |
| [FLAME](https://www.usenix.org/system/files/sec22-nguyen.pdf), §4 | 개별 모델 방향·업데이트 노름, 군집·clipping·잡음 적용 | 기본은 해당 집계의 모델 집합 | 서버가 갱신을 수정할 권한 허용. 정상 과반 등 군집 가정을 별도로 보고 |
| [FedREDefense](https://proceedings.mlr.press/v235/xie24c.html), §4 | 개별 업데이트·학습 연산 접근, 작은 합성 자료 최적화 | 이전 합성 지식을 재사용해 최적화 비용 절감 | 원자료 없이도 서버 계산을 더 쓰는 선택. 재구성 가능한 갱신이 악의가 없다는 보장은 아님 |

**완화 1 — 정상 자료 0개를 고집하지 않는다.** MASA Table 5의 non-IID CIFAR-10에서 proxy 500/125/100/50개일 때 BA는 0.77/6.78/60.37/99.99%다. 같은 표의 조건에서 나온 결과이며 우리의 필요 표본 수가 125라는 뜻이 아니다. 자료가 작아도 충분하다고 가정하는 대신 크기·클래스별 커버리지·분포 차이에 대한 성능 곡선을 요구한다. 자료 사용이 시간 이력의 효과를 대신하지 않도록 같은 proxy를 쓰는 history-free 제어와 fine-tuning을 비교한다. [MASA §8.4, Table 5](https://arxiv.org/html/2411.01040v1)

**완화 2 — 서버 기록과 실제 보정의 추가 비용을 허용한다.** 업데이트의 시간 순서·참여·집계 비중을 남기고 필요한 실제 보정에 비용을 쓴다. FedEraser·UnlearnGuard와 정보 접근을 맞추되, 낮은 저장·통신 비용을 주장하려면 sketch·창 길이의 손실까지 비교한다. [FedEraser](https://arxiv.org/pdf/2012.13891), [UnlearnGuard §4](https://arxiv.org/html/2501.17396v1)

**완화 3 — 표본 삭제를 포함하려면 삭제 단위 정보 접근을 명시한다.** 이것은 위 두 완화보다 큰 변경이다. 사전 batch/shard 기록 또는 검증 가능한 요청 자료의 국소 gradient가 후보다. 원래 [BadFU 검토](01_BadFU.md)의 단일 client 내부 상쇄는 client 합계의 시계열만으로 항상 분리되지 않는다. 요청자가 임의로 제출한 설명·gradient를 신뢰하는 것은 해결책이 아니다. 서버가 요청 자료를 볼 수 있다고 정하면 그 접근 권한과 원자료·유도 상태 처리까지 threat model에 추가해야 한다.

핵심 논리의 차이: **FoolsGold는 반복되는 유사성, FLDetector는 예측과의 불일치, UnlearnGuard는 FL 이력을 기준으로 한 FU 보정을 이용한다. 우리 가설은 실제 요청이 과거의 상쇄 관계를 선택적으로 끊는다는 정보에 있다.** 이 차이가 성능·효용·삭제 이행의 이득을 만들 때만 기여로 주장한다.

## J. Federated Unlearning: a Perspective of Stability and Fairness

원문: [2024 공개 논문](https://arxiv.org/html/2402.01276v1), §§4–6, Appendix G. 이번 검토에서 학회 채택 여부를 확정하지 않았다.

### 1. Problem Statement (해결하려는 핵심 문제)

이질적인 사용자를 삭제하면 전체 성능과 각 잔류 사용자의 효용 변화가 다르다. 삭제 효과만으로 FU를 평가하는 한계를 지적한다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

검증·안정성·공정성을 별도 목적/제약으로 정식화한다. 잔류 사용자별 손실 변화 제한과 안정성 벌점을 도입한다. 논리 전개는 ‘삭제 후 부작용 관측 → 평가 개념 분리 → 조건부 trade-off 분석 → 제약 최적화’다.

### 3. Evaluation & Results (주요 실험 결과)

Appendix G Table 2의 그룹 간 이질성 실험에서 label-distribution 설정 0.1/0.4/0.7의 재학습 안정성 손실은 8.24/3.45/0.26, 안정성 벌점 적용 후 7.69/2.05/0.23이다. **원래 전체 분포 기준 지표**다. MNIST 및 CIFAR-10/ResNet18 실험을 포함한다. 하드웨어는 이번 검토에서 확인하지 않았다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

우리 해석: 원래 전체 목적은 삭제 사용자도 포함하므로 이 손실을 잔류 사용자 피해와 동일시하면 안 된다. 최적 목적값에 대한 검증 지표만으로 분포적/인증 언러닝을 증명하지 않는다. 논문의 제한된 가정 아래 분석을 임의 심층망 보장으로 확대하지 않는다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

정상 삭제 변화·FU 근사 손상·백도어 초과 위험을 분리한다. 정상 요청 뒤 성능 변화 자체를 공격 라벨로 삼지 않는다. 함께 읽기: K 및 [현재 설계](../current/RESEARCH_SUMMARY.md).

## K. FedOSD — Federated Unlearning with Gradient Descent and Conflict Mitigation

원문: [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34181), [수식·Table 1 공개 원문](https://arxiv.org/html/2412.20200v1).

### 1. Problem Statement (해결하려는 핵심 문제)

삭제 gradient가 잔류 사용자의 학습과 충돌하여 효용을 훼손하고, 후속 성능 복구가 삭제한 영향을 다시 살릴 수 있다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

unlearning cross-entropy, 잔류 gradient와 직교하는 삭제 방향, 재기억을 억제하는 복구 투영을 결합한다. 논리 전개는 ‘서로 다른 실패 원인 → 각 원인에 대응하는 연산 → 소거 검증’이다. 직교화 자체를 우리 신규성으로 주장하지 않는다.

### 3. Evaluation & Results (주요 실험 결과)

CIFAR-10/CNN, Pat-20(클라이언트당 20% 클래스), Table 1의 언러닝 직후 잔류 평균 정확도는 FUPGA 20.2%, FedOSD 54.9%다. 둘의 검사 ASR은 0%다. 이 ASR은 삭제 대상에 심은 검사 백도어의 잔존 여부이며 FL→FU 잠복 활성화 실험이 아니다. MNIST/FMNIST/CIFAR-100도 평가한다. 하드웨어는 이번 검토에서 확인하지 않았다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

우리 해석: 잔류 사용자 gradient를 보호하는 목적이 잔류 독성 기여의 제거까지 보장하지 않는다. 직교성은 국소 1차 조건이며 전역 무손실/망각 보장과 다르다. 여러 사용자 gradient 수집과 투영 비용이 있다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

잔류 기여 전체 대신 신뢰 가능한 정상 기능 기준을 정의한다. 이후 정화 실험에서 동일한 망각·효용 조건으로 비교할 기준선이다. 함께 읽기: J, L.

## L. FUCRT — Forgetting Through Transforming

원문: [ICCV 2025 공개 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_Forgetting_Through_Transforming_Enabling_Federated_Unlearning_via_Class-Aware_Representation_Transformation_ICCV_2025_paper.pdf), [초기 상세 공개판](https://arxiv.org/html/2410.06848v1). 버전 간 수치를 혼합하지 않는다.

### 1. Problem Statement (해결하려는 핵심 문제)

클래스 삭제를 위한 파라미터 수정이 잔류 클래스 표현까지 훼손한다. Non-IID에서 클라이언트별 표현 차이가 이를 어렵게 만든다.

### 2. Key Idea & Formulation (핵심 방법론 및 수식/원리)

삭제 클래스 표현을 잔류의 유사 클래스로 변환하고 클래스별 변환을 정렬한다. 논리 전개는 ‘재학습 모델의 표현 관측 → 표현 변환 설계 → 클래스/사용자 간 정렬’이다.

### 3. Evaluation & Results (주요 실험 결과)

최종판 초록은 IID/Non-IID 효용 보존 평균 2.96%/3.78% 개선을 보고한다. 이는 저자 표기의 상대/절대 해석을 여기서 재산정한 값이 아니다. CIFAR-10/100, FMNIST, EuroSAT를 평가한다. 초기판은 20명 전원 참여, 기본 Dirichlet 0.5와 0.1 등의 이질성 비교를 명시한다. 조건별 결과 인용 시 해당 버전 표를 재확인한다. 하드웨어는 이번 검토에서 확인하지 않았다.

### 4. Limitations & Critical Discussion (한계점 및 의문점)

클래스 전체 삭제가 중심이다. 공개 초기판도 극심한 이질성에서 성능 하락을 인정한다. prototype 교환과 추가 변환 학습이 필요하다. 0% 삭제 클래스 정확도를 완전한 정보 삭제 인증과 동일시하지 않는다.

### 5. Takeaway & Next Actions (내 연구/프로젝트와의 접점)

평균 출력 대신 정답 클래스별 기능을 관측한다. 동일 클래스 안의 위장 샘플만 삭제하는 BadFU에는 직접 이식할 수 없다. 함께 읽기: J, K 및 H의 FLShield.
