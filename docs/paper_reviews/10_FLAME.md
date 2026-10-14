# FLAME — Taming Backdoors in Federated Learning

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

우리 이력 선택이 이력 없는 제어보다 **같은 ASR에서 적은 효용 손실**, 또는 같은 효용에서 낮은 ASR을 주는 곡선을 제시해야 한다. 함께 읽기: [전체 문헌 안내](README.md), [BadFU](01_BadFU.md), [FUBA](02_FUBA.md).

---

출처: 기존 [7편 통합 검토](05_defenses_novelty.md)에서 분리. 수치는 원논문 보고값이며 자체 재현값이 아니다. 확인하지 못한 하드웨어·성능은 추정하지 않았다.
