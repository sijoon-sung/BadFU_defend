# UnlearnGuard — Poisoning Attacks and Defenses to Federated Unlearning

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

‘FL 이력을 활용한 FU 방어’는 독립 신규성에서 제외한다. history-free, 선형 삭제 영향, 이력 섞기, 가능하면 UnlearnGuard 이식 기준선을 같은 FU에서 비교한다. 함께 읽기: [전체 문헌 안내](README.md), [BadFU](01_BadFU.md), [FUBA](02_FUBA.md).

---

출처: 기존 [7편 통합 검토](05_defenses_novelty.md)에서 분리. 수치는 원논문 보고값이며 자체 재현값이 아니다. 확인하지 못한 하드웨어·성능은 추정하지 않았다.
