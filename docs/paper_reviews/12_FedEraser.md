# FedEraser — 방어가 작동할 FU 연산자

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

새 벡터 코어는 클라이언트별 보정 후 집계를 구현한다. 기존 `federaser_faithful.py`의 평균 후 층별 보정과 구분한다. 이 코어 위에 우리 제어를 붙이고, 같은 방어 목적의 보존 데이터 재학습과 비교한다. 함께 읽기: [전체 문헌 안내](README.md), [BadFU](01_BadFU.md), [FUBA](02_FUBA.md).

---

출처: 기존 [7편 통합 검토](05_defenses_novelty.md)에서 분리. 수치는 원논문 보고값이며 자체 재현값이 아니다. 확인하지 못한 하드웨어·성능은 추정하지 않았다.
