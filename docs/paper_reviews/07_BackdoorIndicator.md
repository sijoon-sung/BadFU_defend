# BackdoorIndicator — Leveraging OOD Data for Proactive Backdoor Detection in Federated Learning

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

공개 변환에 신호가 전혀 없으면 점수 고도화보다 관측 정보 추가를 검토한다. BN이 관측을 바꾸므로 CIFAR 어댑터에서 buffers를 무시하면 안 된다. 함께 읽기: [전체 문헌 안내](README.md), [BadFU](01_BadFU.md), [FUBA](02_FUBA.md).

---

출처: 기존 [7편 통합 검토](05_defenses_novelty.md)에서 분리. 수치는 원논문 보고값이며 자체 재현값이 아니다. 확인하지 못한 하드웨어·성능은 추정하지 않았다.
