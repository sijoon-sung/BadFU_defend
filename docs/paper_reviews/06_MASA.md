# MASA — Identify Backdoored Model in Federated Learning via Individual Unlearning

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

우리도 새 장력 좌표만 제시하지 말고 **정상 혼동을 줄인 원인**을 소거해야 한다. 융합과 임계값 개선을 우리 장력의 신규성으로 재포장하지 않는다. 함께 읽기: [전체 문헌 안내](README.md), [BadFU](01_BadFU.md), [FUBA](02_FUBA.md).

---

출처: 기존 [7편 통합 검토](05_defenses_novelty.md)에서 분리. 수치는 원논문 보고값이며 자체 재현값이 아니다. 확인하지 못한 하드웨어·성능은 추정하지 않았다.
