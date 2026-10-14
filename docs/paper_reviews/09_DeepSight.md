# DeepSight — Mitigating Backdoor Attacks through Deep Model Inspection

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

장력은 완전한 공격자 식별에 실패해도 조치 대상을 좁히는 데 쓸 수 있다. 다만 ‘강한 잠복 공격은 반드시 관측된다’는 보편 주장은 버리고 관측 가정을 명시한다. 함께 읽기: [전체 문헌 안내](README.md), [BadFU](01_BadFU.md), [FUBA](02_FUBA.md).

---

출처: 기존 [7편 통합 검토](05_defenses_novelty.md)에서 분리. 수치는 원논문 보고값이며 자체 재현값이 아니다. 확인하지 못한 하드웨어·성능은 추정하지 않았다.
