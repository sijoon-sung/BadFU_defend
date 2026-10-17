# FedREDefense — Model Update Reconstruction Error

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

점수의 이름보다 **어떤 정상성 기준을 바꿨는가**가 기여다. 우리 기준은 집단 평균과의 거리보다 ‘상쇄 관계가 이번 삭제와 실제 갱신에서 풀리는가’여야 한다. 함께 읽기: [전체 문헌 안내](README.md), [BadFU](01_BadFU.md), [FUBA](02_FUBA.md).

---

출처: 기존 [7편 통합 검토](05_defenses_novelty.md)에서 분리. 수치는 원논문 보고값이며 자체 재현값이 아니다. 확인하지 못한 하드웨어·성능은 추정하지 않았다.
