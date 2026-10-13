# FL→FU 방어 설계를 위한 논문 7편 검토

2026-10-14. 아래 수치는 **각 원논문이 보고한 해당 조건의 결과**이며 우리의 재현 결과가 아니다. 서로 다른 표의 수치를 묶어 우열을 매기지 않는다. 차이는 상대 개선율이 아닌 퍼센트포인트(pp)로 계산했다. 하드웨어를 확인하지 못한 경우 추정하지 않았다. 각 항목의 ‘우리 해석’은 저자의 주장과 구분한다.

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

## 우리 논문의 주장에 반영할 결정

- **남길 문제:** FL에서 공동 학습된 주입·억제 중 FU가 억제를 제거하여 활성화하는 백도어.
- **내릴 주장:** 큰 장력=악성, 모든 non-IID 구분, FU 시뮬레이션 자체의 신규성, 모든 BadFU에 대한 이미 완성된 방어.
- **새 검증 대상:** 과거의 선택적 상쇄 해제와 실제 FU 갱신의 증가가 겹치는 좌표를 제어하면 같은 효용에서 ASR을 더 낮추는가.
- **버리면 안 되는 비교:** 동일 제어기의 이력 제거·선형 경계·이력 섞기, 일반 clipping, 보존 손실 제어만. 장부에 의존한 방법이 이들을 이기지 못하면 과정 정보의 기여는 기각한다.
- **non-IID 해석:** 위 논문들은 주로 비IID로 인한 방어 혼동을 다룬다. 공격 없는 정상 모델의 표적 오분류와 악성 의도를 보편적으로 분리해 준 논문들로 묶지 않는다. 정상 조건에서의 target rate와 트리거에 의한 추가 변화, 클래스별 ACC를 함께 보고해야 한다.
