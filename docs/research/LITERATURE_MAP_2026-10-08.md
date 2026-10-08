# FU 보안 연구를 위한 확장 문헌 지도

2026-10-08 · 교수님 피드백 반영 · 31편의 핵심 문헌과 확인 범위

방어 논문들의 위협 범위·평가 구성과 우리 논문의 차별점은 후속 [논문 범위 및 독창성](PAPER_SCOPE_AND_NOVELTY_2026-10-08.md)에서 더 구체적으로 대조했다. 그 문서에는 악성 언러닝 요청을 다룬 KDD·NeurIPS 2023의 추가 근거도 포함한다.

이 문서는 기존 13편 요약을 확장한다. 목적은 논문 수를 늘리는 것이 아니라 **공격이 성립하는 조건, 서버의 관측 정보, 삭제 단위, 방어 비용, 검증의 한계**를 연결하는 것이다. 연구 방향과 실험 의사결정은 [범용성 확장 계획](GENERALIZATION_PLAN_2026-10-08.md)에 정리했다.

## 읽는 기준과 조사 범위

아래의 확인 수준은 재현 성공을 뜻하지 않는다. **본문**은 관련 위협 모델·방법·평가 절을 확인했다는 뜻이며, 수식 전체 검증이나 모든 부록·공식 코드 감사 완료를 뜻하지 않는다. **개요**는 저자/학회/출판사 초록과 발표 정보를 확인했다는 뜻이다. **코드 대조**는 별도 감사 문서에 근거가 있는 경우만 표시했다. 개요만 확인한 방법의 세부 성능·가정은 확정하지 않는다.

검색은 BadFU/FUBA에서 출발하여 언러닝 악용, 실제 FU 중 오염, 일반 FL 백도어, 모델 복구, 망각 검증으로 확장했다. 학회·출판사·저자 원문·공식 저장소를 근거로 삼았으며 AI 생성 논문 리뷰를 기술적 근거로 쓰지 않았다. 모든 관련 논문을 빠짐없이 조사한 체계적 문헌고찰은 아니다. 공개된 본문을 확보하지 못한 근접 연구와 구현 검증이 남은 항목을 마지막 절에 명시한다.

## A. 어떤 공격까지 다룰 것인가: 10편

| ID | 논문 / 발표 | 핵심 구조와 관측 조건 | 우리 연구에서 확인할 것 | 확인 수준 |
|---|---|---|---|---|
| A1 | [BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning](https://arxiv.org/html/2508.15541v1), arXiv v1, 2025 | 위장 데이터 삭제로 잠복 백도어 활성화. 본문은 한 악성 클라이언트 내부의 백도어·위장 데이터와 샘플 삭제를 기술 | 공식 ul 예제의 분리형 클라이언트 삭제와 본문 설정을 구분. 쌍 탐지의 적용 범위를 결정 | 본문 + 공식 예제 코드 대조 |
| A2 | [FUBA: Backdoor Federated Learning via Federated Unlearning](https://ieeexplore.ieee.org/document/11231135/), IEEE TAI, 온라인 2025 / 권호 2026 | 주입 역할 Adv-attacker와 억제 역할 Adv-defender. 후자의 삭제로 활성화 | 두 역할이 있다는 사실과 서버에 반대 방향이 관측된다는 사실은 별도 가설 | 출판사 개요 + 기존 로컬 코드 감사 |
| A3 | [UBA-Inf: Unlearning Activated Backdoor Attack with Influence-Driven Camouflage](https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui), USENIX Security 2024 | 중앙 ML 서비스의 위장 샘플과 언러닝 활성화 | FL 고유 공격으로 표시하지 않는다. 샘플 단위 위장의 선행 맥락 | 학회 개요; 이전 BadFU 데이터 생성 감사 참고 |
| A4 | [Backdoor Attack through Machine Unlearning (BAMU)](https://arxiv.org/html/2310.10659v1), arXiv v1, 2023 | poison·mitigation 샘플을 함께 넣고 mitigation 삭제로 활성화. 중앙 MU 및 SISA 대상 | 삭제 샘플의 출력 불확실성·서브모델 간 차이로 저비용 탐지도 제안. 무시뮬레이션 자체는 새롭지 않음 | 본문 III, VI–VII |
| A5 | [Poisoning Attacks and Defenses to Federated Unlearning (BadUnlearn / UnlearnGuard)](https://arxiv.org/html/2501.17396v1), WWW Companion 2025 | 실제 FU의 통신 과정에서 악성 업데이트를 제출하여 오염을 유지. 역사 기반 업데이트 추정·검사를 방어로 제안 | 상쇄 쌍이 없는 FU 오염을 위한 핵심 공격·방어 비교군 | 본문 II–V 및 정리의 가정 |
| A6 | [Malicious Forgetting: Backdoor Injection in Active Federated Unlearning and Countermeasure Design](https://wenwei-zhao.github.io/publication/conference-paper/3/), INFOCOM 2026 | Fusion backdoor: 학습 때 준비한 작은 백도어 부분망을 FU 중 결합. 방향 부분공간 변화 탐지도 제안 | 부분공간 기반 FU 방어와 직접 겹칠 가능성. 세부 차별점 판단은 본문 확보 후 | 저자 개요·발표 정보만 확인; IEEE 본문 접근 실패 |
| A7 | [FedMUA: Exploring the Vulnerabilities of Federated Learning to Malicious Unlearning Attacks](https://arxiv.org/html/2501.11848v1), IEEE TIFS 2025 | 영향력 있는 샘플과 조작된 언러닝 요청으로 다른 참여자의 특정 샘플 예측을 변경 | 트리거 백도어와 다른 목표. 대상 샘플 오분류 ASR을 백도어 ASR과 합산하지 않음 | 본문 위협·방법·방어 및 VII |
| A8 | [How To Backdoor Federated Learning](https://proceedings.mlr.press/v108/bagdasaryan20a.html), AISTATS 2020 | 모델 대체를 이용한 FL 백도어. 삭제 요청이나 상쇄 상대가 필수 조건이 아님 | 비상쇄 공격 대조군. 단순 노름 클리핑을 넘어서는지 평가 | 학회 개요 |
| A9 | [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr), ICLR 2020; [저자 기관 설명](https://research.ibm.com/publications/dba-distributed-backdoor-attacks-against-federated-learning) | 여러 공격자가 트리거의 부분 패턴을 분담 | 쌍에서 다자 구조로 확장하는 실험. 반대 방향 쌍을 인위적으로 추가하지 않음 | 저자/학회 개요 |
| A10 | [Neurotoxin: Durable Backdoors in Federated Learning](https://proceedings.mlr.press/v162/zhang22w.html), ICML 2022 | 정상 업데이트에 덜 덮이는 방향을 이용해 지속성 확보 | ASR 10–15% 이후 정상 FL이 자동으로 해결한다는 가설의 스트레스 테스트 | 학회 개요 및 기존 검토 |

BadFU 공식 예제는 [고정 커밋](https://github.com/BingguangLu/BadFU/blob/0a2117feab58faf5b7e2a1e5e1759112087d6049/badfu.py)을 기준으로 대조했다. ul 경로에서는 client 0에 clean+백도어, 추가 client 5에 같은 clean+위장이 들어가며 5를 제거한다. 이 예제는 같은 클라이언트 안의 위장 샘플만 삭제하는 문제를 그대로 구현한 것이 아니다. 현재 우리 GPU 패키지도 분리형 경로부터 검증한다. 상세한 코드 근거는 [공격 충실도 감사](ATTACK_FIDELITY_AND_DIRECTION_2026-10-08.md)에 있다.

## B. 탐지·정화의 기존 방법: 9편

| ID | 논문 / 발표 | 사용하는 정보·연산 | 가져올 점 / 적용 시 한계 | 확인 수준 |
|---|---|---|---|---|
| B1 | [Identify Backdoored Model in Federated Learning via Individual Unlearning (MASA)](https://arxiv.org/html/2411.01040v1), WACV 2025 | 로컬 모델별 개별 언러닝으로 비정상 반응을 드러냄 | 탐지용 추가 최적화가 있는 비용 비교군. 우리 운영 알고리즘에 해당 절차를 넣지 않음 | 이전 본문 검토 |
| B2 | [The Limitations of Federated Learning in Sybil Settings (FoolsGold)](https://www.usenix.org/conference/raid2020/presentation/fung), RAID 2020 | 누적 업데이트의 유사성을 이용한 sybil 기여 억제 | 관계 기반 방어의 선행연구. 비슷한 방향 협력과 반대 방향 위장은 구별 | 학회 개요 |
| B3 | [FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients](https://arxiv.org/abs/2207.09209), KDD 2022 | 과거 정보로 업데이트를 예측하고 시간적 일관성 검사 | 쌍이 없는 공격에 대한 이력 신호. 정상 분포 변화도 이상으로 잡힐 수 있음 | 저자 개요 및 기존 검토 |
| B4 | [FLAME: Taming Backdoors in Federated Learning](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen), USENIX Security 2022 | 업데이트 클러스터링·클리핑·노이즈 | 탐지용 FU가 없는 강한 기준선. 방어의 정확도·노이즈 비용을 함께 비교 | 학회 개요 및 기존 검토 |
| B5 | [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/), NDSS 2021 | 서버의 깨끗한 root 데이터로 기준 방향을 만들고 신뢰 가중·크기 조정 | 정상 기준을 얻는 방법. 서버 데이터의 대표성과 추가 학습 비용이 필요 | 학회 개요 |
| B6 | [DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection](https://arxiv.org/abs/2201.00763), NDSS 2022 | 모델 내부·출력 특성 및 클러스터링으로 오염 기여 구분 | head의 반대 코사인 이외 특징 후보. 원 방법을 실제 FU에 이식하면 별도 적용 실험 | 저자 개요 |
| B7 | [Reconstructive Neuron Pruning for Backdoor Defense (RNP)](https://proceedings.mlr.press/v202/li23v.html), ICML 2023 | clean 데이터에서 망각·회복 최적화 후 백도어 뉴런 가지치기 | 정화 비교군. 백도어를 드러내기 위한 추가 언러닝은 우리의 탐지 원칙과 충돌 | 학회 개요 |
| B8 | [SCRUB-FL](https://arxiv.org/html/2606.22700v1), arXiv v1, 2026 | 의심 샘플·패턴 분석 및 생성과 정화를 결합 | 클라이언트 정보·생성 단계·추가 학습이 필요하므로 비용과 접근 권한을 맞춰 비교 | 이전 본문 검토; 프리프린트 |
| B9 | [Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection (AlignIns)](https://arxiv.org/html/2503.07978v1), CVPR 2025 | 전체 방향 정렬 및 주요 파라미터 부호 정렬을 검사해 업데이트 필터링 | 단순히 방향 검사를 늘리는 설계의 직접 기준선. 원래 FL 방어와 FU 적용을 구분 | 저자 원문 개요·방법 확인 |

방향, 시간적 일관성, 층별 특징을 여러 개 합치는 것만으로 독창성이 생기지는 않는다. 우리에게 필요한 질문은 **실제 삭제가 유발한 정상 변화와 공격성 변화를 기존 방어보다 더 잘 구분하는가**다.

## C. 실제 FU와 오염 모델 복구: 7편

| ID | 논문 / 발표 | 핵심과 가정 | 우리 연구에서의 역할 | 확인 수준 |
|---|---|---|---|---|
| C1 | [FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models](https://www.cs.sfu.ca/~jcliu/Papers/FedEraser21.pdf), IWQOS 2021 | 기록과 보정 학습을 이용한 클라이언트 제거 | 첫 FU 백엔드. 현재 구현은 통제된 FedEraser-style 재구성으로 원 실험 그대로는 아님 | 기존 본문·코드 감사 |
| C2 | [FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information](https://arxiv.org/html/2210.10936v1), IEEE S&P 2023 | 탐지된 공격자 제거 뒤 이력 기반 업데이트 추정, 일부 실제 업데이트로 복구 | 탐지 오류도 실험적으로 다룸. ‘불완전 탐지 후 복구’ 자체를 새 기여로 주장할 수 없음 | 본문 III–IV 및 탐지 오류 논의 |
| C3 | [Towards Efficient and Certified Recovery from Poisoning Attacks in Federated Learning (Crab)](https://arxiv.org/abs/2401.08216), arXiv v2, 2024 | 선택적 이력 저장과 적응적 rollback. 공격자 식별 뒤 복구 | 전체 이력보다 적은 저장량도 기존 연구. 저장·복구 비용 비교 | 저자 개요; 여기서는 학회 배정 미확정 |
| C4 | [SIFU: Sequential Informed Federated Unlearning for Efficient and Provable Client Unlearning in Federated Optimization](https://proceedings.mlr.press/v238/fraboni24a.html), AISTATS 2024 | 연속된 클라이언트 삭제와 조건부 망각 보장 | 두 번째 FU 백엔드 후보. 정화를 붙이면 보장을 자동 상속하지 않음 | 학회 개요 |
| C5 | [Machine Unlearning (SISA)](https://arxiv.org/abs/1912.03817), IEEE S&P 2021 | sharding·slicing으로 삭제 시 재학습 범위를 제한 | 중앙 MU와 분산 FU의 차이를 이해하고 BAMU의 실험 단위를 해석 | 저자 개요 |
| C6 | [GDFA: Geometry-Driven Federated Unlearning with Directional Task Vector Alignment](https://openaccess.thecvf.com/content/CVPR2026/html/Weng_GDFA_Geometry-Driven_Federated_Unlearning_with_Directional_Task_Vector_Alignment_CVPR_2026_paper.html), CVPR 2026 | 평탄한 영역과 task vector 방향 정렬로 비IID에서 지식 보존 | 정상 업데이트 충돌을 공격으로 해석하지 않기. 기하학적 FU 보정 자체도 선행연구 있음 | CVF 검색 색인의 공식 초록; 직접 본문 403 |
| C7 | [Adversarial Update-Based Federated Unlearning for Poisoned Model Recovery (FAUN)](https://arxiv.org/html/2605.02110v1), ICASSP 2026 | 식별한 공격자의 최근 업데이트와 clean proxy에서 PGD로 제거 방향 구성, 후속 정상 학습 | 최근 기록·방향 제거·후속 FL 조합이 매우 가까움. 알려진 공격자 제거와 요청 이행 중 미식별 잔존 위험을 구분하여 비교 | 본문 II–IV; [학회 발표 확인](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=13708) |

## D. 망각 검증과 설계 근거: 5편

| ID | 논문 / 발표 | 핵심 | 우리 연구의 주의점 | 확인 수준 |
|---|---|---|---|---|
| D1 | [Certified Data Removal from Machine Learning Models](https://proceedings.mlr.press/v119/guo20c.html), ICML 2020 | 삭제 전후 학습 알고리즘의 출력 분포를 기준으로 하는 인증된 제거; 선형 분류기 메커니즘 | ASR 감소나 파라미터 거리만으로 인증된 망각을 선언하지 않음 | 학회 개요 |
| D2 | [On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning](https://www.usenix.org/conference/usenixsecurity22/presentation/thudi), USENIX Security 2022 | 결과 모델만으로 학습 데이터의 부재를 감사하는 접근의 한계 | 모델 결과와 실제 삭제 절차·기록 상태를 함께 검토 | 학회 개요 |
| D3 | [Gradient Projection Memory for Continual Learning](https://arxiv.org/abs/2103.09762), ICLR 2021 | 중요한 과거 지식의 부분공간을 보호 | clean 지식 보호의 설계 참고. 백도어 분리나 망각 보장이 아님 | 저자 개요 및 기존 검토 |
| D4 | [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017 | 신뢰 점수와 실제 정답 빈도의 불일치 | 우리 위험 점수를 공격 확률이라고 부르지 않음. 이 논문의 보정법이 그대로 적용된다고 가정하지 않음 | 학회 개요 |
| D5 | [Gone but Not Forgotten: Improved Benchmarks for Machine Unlearning](https://www.sei.cmu.edu/library/gone-not-forgotten-improved-benchmarks-machine-unlearning/), SEI 보고서 / arXiv, 2024 | 단순 MIA 평가가 놓치는 망각 검증 문제 | 여러 공격·샘플 그룹·재학습 기준을 사용. MIA 실패도 완전 삭제의 증명은 아님 | 저자 기관 개요 |

## 가장 가까운 연구와의 차별점 감사

다음 표는 우리 방법의 우월성을 주장하는 표가 아니라, **실험 전에 해소해야 할 중복 가능성**이다.

| 선행연구 | 이미 갖고 있는 요소 | 우리가 증명해야 하는 추가 가치 |
|---|---|---|
| BAMU | 별도 언러닝 전 출력만으로 의심 요청 판별 | FL 서버의 제한된 정보에서 삭제를 거부하지 않고 정화를 결합하는 효과 |
| UnlearnGuard | 실제 FU 중 이력 기반 보호, 추가 탐지 FU가 핵심이 아님 | 합법적 요청자와 미식별 잔존 공격자가 다른 상황에서 추가적인 보호와 비용 이점 |
| FedRecover | 역사 기반 저비용 복구, 오탐·미탐 조건 평가 | 위장 삭제로 생기는 위험 및 요청자만 제거해도 남는 공격에 대한 성능 |
| AlignIns / FLDetector / FLAME | 추가 탐지 FU 없이 업데이트의 이상을 판단 | 같은 정상 요청 FPR·효용·정보 예산에서 실제 FU 변화에 대한 더 정확한 대응 |
| FAUN | 최근 기록, 악성 방향 억제, 후속 정상 학습 | 공격자 신원을 모르는 상황에서 불확실성을 반영한 개입과 삭제 이행의 동시 평가 |
| Malicious Forgetting | FU 중 백도어 및 방향 부분공간 탐지 | 본문 확보 전 차별점 확정 불가. 최우선 미해결 근접 연구 |
| GDFA | 정상 지식 보호를 위한 방향·기하학 활용 | 보안 목적 개입이 단순 효용 보정으로 설명되지 않는지 |

UnlearnGuard의 정리 1은 강한 볼록성·smoothness, 추정 오차 상한, FL 단계의 악성 클라이언트 탐지 조건을 전제로 한다. 이를 임의 DNN·미식별 공격자 모두에 대한 무조건 보장으로 인용하지 않는다. 본문 식 (5)의 코사인 부등호와 방향 일치 설명의 관계도 구현 전에 PDF·코드 대조가 필요하다. 이 불확실성을 임의 수정한 뒤 ‘원 방법 재현’이라고 표시하면 안 된다. [A5 본문 §4.4–4.5](https://arxiv.org/html/2501.17396v1)

FAUN은 공격자로 식별하여 제거한 집합과 남은 정상 집합을 둔다. 따라서 우리 문제와 동일하다고 단정할 수도, ‘요청 기반’이라는 이름만으로 충분히 다르다고 단정할 수도 없다. 삭제 집합과 실제 오염 집합이 다른 공통 실험을 구성해야 한다. [C7 본문 §2–3](https://arxiv.org/html/2605.02110v1)

## 확정한 것과 아직 모르는 것

| 질문 | 현재 답 | 남은 확인 |
|---|---|---|
| BadFU가 항상 두 클라이언트인가? | 아니다. 본문과 공식 예제의 실제 삭제 경로를 나눠 읽어야 함 | 논문과 일치하는 단일 클라이언트·샘플 삭제 재현 |
| 비용 절감 자체가 새로운가? | 아니다. 출력 검사, 이력 추정, 선택 저장, 최근 기록 정화가 이미 있음 | 동일 정보·성능 조건의 비용 비교 |
| 불완전 탐지 대응 자체가 새로운가? | 아니다. FedRecover에도 탐지 오류 평가가 있음 | 요청 조건부 위험·선택 정화가 주는 추가 효과 |
| 상쇄 쌍만 없으면 방어가 무조건 불가능한가? | 쌍 신호는 사라질 수 있지만 다른 관측 신호가 있으면 확장 가능 | 실제 FU·다층·출력 특징의 독립적 정보량 |
| 정상 holdout이 좋으면 백도어가 없는가? | 아니다 | 알 수 없는 트리거에 대한 적응적 공격 평가 |
| 후속 FL이면 공격이 자연히 사라지는가? | 보장 없음. 지속형 공격을 별도 평가해야 함 | 공격 중단·지속·재개, 장기 peak ASR |
| 실제 GPU 효능을 확보했는가? | 새 통합 GPU 실험 결과는 아직 없음 | 업로드된 실행 패키지의 결과 |
| 모든 최신 근접 연구를 이해했는가? | 아니다. 확인 범위를 표에 표시함 | Fusion 본문·방어 코드, FUBA 출판 본문, GDFA 본문 |

## 읽기 순서와 논문별 필수 메모

1. **공격과 기준의 충돌:** A1–A5. 삭제 집합 F, 잔존 독성 데이터, 공격 전후 모델을 직접 그린다. 실제 삭제·추정 삭제·방어용 진단을 구분한다.
2. **가장 가까운 방어:** A5, C2, C7, B9, A6. 공격자 식별이 입력인지, 탐지기가 무엇을 보는지, 추가 최적화가 무엇인지, 인증의 가정을 기록한다.
3. **쌍 없는 일반화:** A8–A10, B3–B6. 원래 공격의 필수 조건을 바꾸지 않고 기존 탐지기의 관측 한계를 확인한다.
4. **삭제와 효용:** C1, C4–C6, D1–D2. 클라이언트 삭제와 샘플 삭제, 중복 샘플, 알고리즘 상태의 잔존 정보를 확인한다.
5. **검증과 실패:** A7, B1, B7–B8, D3–D5. 서로 다른 ASR 정의와 추가 계산을 분리한다.

각 논문은 다음 항목을 한 장으로 작성한 뒤 재현 여부를 판단한다: 문제 정의 / 삭제 단위 / 공격자 권한 / 서버 관측 / 탐지와 정화의 구분 / 추가 학습·통신·저장 / 정상 대조 / 적응적 공격 / 정리의 가정 / 공식 코드와 본문 차이. 빈칸은 모르는 것으로 남긴다. 원문에서 확인하지 않은 내용을 다른 논문의 인용 설명으로 채우지 않는다.

## 추가로 추적할 범위

공격을 모두 막는다는 범위를 주장하려면 개인정보 유출형 FU 공격, 가용성 공격, 요청 검증, secure aggregation, 비동기·개인화 FL, 생성 모델까지 조사해야 한다. 이번 제안은 **수평 FL 분류 모델의 무결성·백도어 위험**을 중심으로 하므로 이 범위를 모두 포함한다고 주장하지 않는다. 이를 무리하게 같은 논문의 필수 구현으로 넣으면 핵심 가설 검증이 흐려진다.

기존 [13편 검토](LITERATURE_REVIEW_2026-10-08.md)와 [미팅 자료](MEETING_BRIEF_2026-10-08.md)는 당시 구현과 판단의 기록이다. 근접 선행연구 및 독창성 판단은 이 확장 지도를 우선한다.
