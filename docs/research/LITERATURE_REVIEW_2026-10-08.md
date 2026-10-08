# 언러닝 백도어 방어 관련 논문과 알고리즘 발전 방향

2026년 10월 8일 · [미팅 자료](MEETING_BRIEF_2026-10-08.md)의 선행연구 근거

이 연구는 삭제로 활성화되는 백도어에 대해, 기존 업데이트 기록으로 위험을 판단하고 실제 삭제 이행 과정에 정화를 결합한다. 비교할 선행연구를 **공격의 원인, 탐지 비용, FU 구현, 정화와 지속성**으로 구분한다. 아래의 “우리 연구에 적용”은 논문 자체의 결론이 아니라 제안하는 연구 방향이다.

## 먼저 읽을 논문

| 우선순위 | 논문과 발표 정보 | 확인할 핵심 | 우리 연구에서의 역할 |
|---|---|---|---|
| 최우선 | [BadFU](https://arxiv.org/html/2508.15541v1), arXiv v1 2025 | 위장 삭제로 잠복 백도어 활성화, 본문과 공식 코드 설정 차이 | 공격 범위와 재현 조건 확정 |
| 최우선 | [FUBA](https://ieeexplore.ieee.org/document/11231135/), IEEE TAI, 온라인 2025·권호 2026 | 주입 역할과 억제 역할, 억제자의 삭제 | 클라이언트 사이 상쇄 가설의 직접 근거 |
| 최우선 | [Malicious Forgetting](https://wenwei-zhao.github.io/publication/conference-paper/3/), INFOCOM 2026 | FU 공격과 방향 부분공간 변화 탐지 | 가장 먼저 세부 차별점을 확인할 근접 연구 |
| 높음 | [MASA](https://arxiv.org/html/2411.01040v1), WACV 2025 | 로컬 모델별 개별 언러닝과 이상 손실 | 탐지용 추가 학습이 있는 비교군 |
| 높음 | [FedEraser](https://www.cs.sfu.ca/~jcliu/Papers/FedEraser21.pdf), IWQOS 2021 | 과거 업데이트 보관과 보정 학습 | 실제 FU 계산과 추가 방어 계산의 경계 |
| 높음 | [FoolsGold](https://www.usenix.org/conference/raid2020/presentation/fung), RAID 2020 | 클라이언트 업데이트 관계를 이용한 sybil 방어 | 같은 방향 협력과 반대 방향 상쇄의 차이 |
| 높음 | [FLDetector](https://arxiv.org/abs/2207.09209), KDD 2022 | 과거 업데이트에 대한 시간적 일관성 | 다중 라운드 신호와 정상 변동의 분리 |
| 높음 | [FLAME](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen), USENIX Security 2022 | 클러스터링·클리핑·노이즈 | 추가 언러닝 없는 기존 방어와의 비교 |
| 높음 | [Neurotoxin](https://proceedings.mlr.press/v162/zhang22w.html), ICML 2022 | 정상 학습에 잘 덮어써지지 않는 백도어 | 후속 FL에 의한 자연 소멸 가설의 반례 |
| 보조 | [UBA-Inf](https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui), USENIX Security 2024 | 위장 샘플과 언러닝 활성화 | 문제의 선행 맥락과 BadFU 전처리 검토 |
| 보조 | [SCRUB-FL](https://arxiv.org/html/2606.22700v1), arXiv v1 2026 | 의심 패턴 생성과 학습 후 백도어 정화 | 정화의 정보 요구와 전체 비용 비교 |
| 설계 참고 | [Gradient Projection Memory](https://arxiv.org/abs/2103.09762), 2021 | 중요한 과거 지식의 공간을 보호하는 투영 | 정상 지식을 덜 훼손하는 정화 설계 |
| 설계 참고 | [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017 | 신뢰 점수와 정답 확률의 차이 | 점수·확률·정화 제어값의 구분 |

동명의 다른 FUBA 논문과 혼동하지 않는다. 여기서 FUBA는 *Backdoor Federated Learning via Federated Unlearning*이며 저자 코드는 [stcebra/FUBA](https://github.com/stcebra/FUBA)다. SCRUB-FL은 확인한 arXiv v1 기준이며 심사 완료 논문으로 표시하지 않는다.

## 공격 범위를 정하는 자료

**BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning.** arXiv v1의 IV-A는 단일 악성 클라이언트를 가정하고, V절은 그 내부의 백도어·위장 데이터와 위장 삭제를 설명한다. 한편 확인한 공식 예제의 공격 활성화 경로는 둘을 서로 다른 참여자에 배치한다. 본문 설정과 예제 경로를 함께 인용하되 동일한 실험이라고 부르지 않는다. [본문](https://arxiv.org/html/2508.15541v1), [공식 예제](https://github.com/BingguangLu/BadFU/blob/0a2117feab58faf5b7e2a1e5e1759112087d6049/badfu.py)

우리 연구에 적용: “BadFU 전체 방어” 대신 “공식 예제의 클라이언트 분리형에 대한 첫 검증”으로 표현한다. 같은 클라이언트 내부의 샘플 삭제는 FU의 삭제 단위부터 바뀌므로 별도 확장한다.

**FUBA: Backdoor Federated Learning via Federated Unlearning.** 출판사 초록은 Adv-attacker의 주입과 Adv-defender의 억제, 후자의 언러닝 요청으로 활성화하는 구조를 설명한다. 온라인 출판일은 2025년 11월 6일이며 권호는 2026년 5월이다. DOI는 10.1109/TAI.2025.3630110이다. [출판사 자료](https://ieeexplore.ieee.org/document/11231135/)

우리 연구에 적용: 두 역할이 있다는 사실만으로 반대 코사인이 항상 관측된다고 가정하지 않는다. 학습 단계·층·타깃 클래스별로 상쇄 신호와 활성화의 관계를 측정한다. 원 공격 코드를 보존한 것과 원 실험 규모·조건을 재현한 것을 구분한다.

**UBA-Inf: Unlearning Activated Backdoor Attack with Influence-Driven Camouflage.** MLaaS의 위장된 백도어를 언러닝 요청으로 활성화하는 선행 공격이다. [학회 자료](https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui)

우리 연구에 적용: 언러닝 활성화 공격이라는 문제 자체를 우리의 새로움으로 쓰지 않는다. 데이터 준비 과정에서 이 구현을 사용했다면 최종 백도어·위장 산출물과 해당 논문의 알고리즘을 구분해 기록한다.

## 가장 가까운 새 선행연구

**Malicious Forgetting: Backdoor Injection in Active Federated Unlearning and Countermeasure Design.** 저자는 Wenwei Zhao, Yuanzhe Peng, Xiaowen Li, Jie Xu, Yao Liu, Zhuo Lu이며, 저자 페이지는 INFOCOM 2026 논문으로 표시한다. 초록에는 Fusion backdoor와 협조된 업데이트의 방향 부분공간 변화에 기반한 탐지가 함께 나온다. [저자 페이지](https://wenwei-zhao.github.io/publication/conference-paper/3/), [연구실 발표 목록](https://csalab.site/publications/?select=1)

이번 확인 범위는 저자 초록과 발표 정보다. 링크된 IEEE 본문은 열리지 않았으므로 세부 알고리즘이 우리 방법과 같거나 다르다고 확정하지 않는다. 차별점 판단을 위해 다음을 우선 확인한다.

| 본문에서 확인할 항목 | 우리 연구의 제안 |
|---|---|
| 탐지가 학습 중인지, 요청 후 실제 FU 중인지 | 요청자 q를 기준으로 저장 기록 검사 |
| 방향 변화의 기준 모델과 부분공간 정의 | 클래스별 반복 상쇄에서 기저 추정 |
| 탐지에 별도 최적화나 가상 FU가 필요한지 | 탐지용 FU 실행 없음 |
| 의심 요청을 거절·차단하는지, 삭제를 계속하는지 | 삭제 대상을 유지하고 실제 FU 수행 |
| 발견 이후의 모델 정화가 포함되는지 | 실제 FU의 잔존 보정 기여를 선택 감쇠 |
| 정상 성능과 망각을 어떻게 평가하는지 | 보안·정상 피해·삭제·비용을 함께 평가 |

이 표는 **확인할 질문**이다. 상대 방법이 해당 기능을 제공하지 않는다는 뜻이 아니다. 선행연구 검토가 끝나기 전에는 “최초의 부분공간 기반 FU 방어”를 사용하지 않는다.

## 탐지와 비용 비교를 위한 자료

**MASA: Identify Backdoored Model in Federated Learning via Individual Unlearning.** 본문 3.3절과 4절은 서버가 로컬 모델을 재구성하고 깨끗한 proxy 데이터에 대해 개별 언러닝을 수행해 손실을 관측하는 방법을 설명한다. non-IID의 편차를 완화하기 위한 모델 융합도 포함한다. [원문](https://arxiv.org/html/2411.01040v1)

우리 연구에 적용: 추가 탐지 학습이 없는 설계의 비교군으로 삼는다. MASA는 원래 학습 중 집계 방어이므로 요청 시점의 한 번짜리 검사로 바꾸면 변형임을 표시한다. 원 알고리즘의 전체 학습 비용과 요청 시점 변형의 비용을 섞지 않는다.

**FoolsGold: The Limitations of Federated Learning in Sybil Settings.** 업데이트의 다양성을 이용해 협력하는 sybil 공격자를 구분한다. [학회 자료](https://www.usenix.org/conference/raid2020/presentation/fung)

우리 연구에 적용: 여러 클라이언트가 같은 방향으로 움직이는 관계를 보는 기존 접근과, 삭제될 기여가 잔존 위험을 상쇄하는 관계를 구분한다. 관계를 쓰는 아이디어 자체를 새롭다고 말하지 않는다.

**FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients.** 과거 업데이트를 이용한 예측과 실제 업데이트의 여러 라운드 불일치를 검사한다. 본문의 L-BFGS 예측기를 현재 코드가 사용한다는 뜻은 아니다. [원문과 저자 코드 링크](https://arxiv.org/abs/2207.09209)

우리 연구에 적용: 단일 라운드의 음의 코사인보다 지속적인 상쇄가 유효한지 검증한다. 정상적인 분포 변화도 반복적 편차를 만들 수 있으므로 non-IID·부분 참여별 정상 대조군을 포함한다.

**FLAME: Taming Backdoors in Federated Learning.** 모델 클러스터링과 가중치 클리핑으로 필요한 노이즈를 줄이는 방어다. [학회 자료](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen)

우리 연구에 적용: 탐지용 언러닝이 없는 강한 비교군을 포함한다. 같은 공격·참여 조건에서 학습부터 방어를 적용해야 하며, 저장된 공격 궤적에 사후 집계 규칙만 바꾼 결과로 원 방어의 실패를 주장하지 않는다.

## 실제 FU와 정화에 참고할 자료

**FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models.** 과거 업데이트를 보관하고 잔존 참여자의 보정 학습을 통해 요청자 제외 모델을 구성한다. 저장 비용과 재구성 시간의 교환이 핵심이다. [논문 PDF](https://www.cs.sfu.ca/~jcliu/Papers/FedEraser21.pdf)

우리 연구에 적용: FU에 원래 필요한 보정 기여를 재사용하고, 탐지·SVD·정상 검증의 추가 비용을 따로 잰다. 현재 백엔드는 시작 모델과 라운드 기준을 맞춘 층별 노름 보정 구현으로, 원 FedEraser의 결과나 보장을 그대로 승계하지 않는다.

**SCRUB-FL: Sanitizing and Cleansing Representations via Unlearning of Backdoors.** 클라이언트가 의심 패턴을 추출하고 생성기를 학습한 뒤, 서버가 합성 샘플로 정화하는 두 단계 접근이다. [arXiv v1의 V절](https://arxiv.org/html/2606.22700v1)

우리 연구에 적용: 사후 정화 자체는 이미 존재하는 연구 방향이다. 기존 기록만 사용하는 제안과 생성기·샘플 기반 정화를 정보 요구, 클라이언트 작업량, 통신량까지 포함해 비교한다. 논문에 보고된 ASR을 우리 조건의 목표 성능처럼 직접 옮기지 않는다.

**Gradient Projection Memory for Continual Learning.** 과거 과제에 중요한 부분공간과 직교하는 방향으로 학습해 과거 지식을 보존하는 접근이다. [저자 원문](https://arxiv.org/abs/2103.09762)

우리 연구에 적용: 다음 버전에서 깨끗한 잔존 데이터로 추정한 중요 공간 Q를 보호하면서 의심 방향 B에 개입하는 방법을 검토한다. B와 Q가 크게 겹치면 제거 가능한 위험 성분이 작아지는 상충을 분석한다. 현재 GPU 코드에는 Q 보호가 구현되어 있지 않다.

**On Calibration of Modern Neural Networks.** 출력 신뢰도와 실제 정답 확률이 일치하지 않을 수 있음을 다루며 확률 보정의 필요성을 제시한다. [논문](https://proceedings.mlr.press/v70/guo17a.html)

우리 연구에 적용: 현재 휴리스틱 점수를 확률이라고 부르지 않는다. 별도 라벨 검증 데이터가 확보되면 점수 보정과 정화 정책을 검토하되, 이 논문의 temperature scaling을 현재 점수에 그대로 적용하면 해결된다고 가정하지 않는다.

## 후속 FL 가설을 검증할 자료

**Neurotoxin: Durable Backdoors in Federated Learning.** 정상 학습에서 변화가 작은 파라미터를 공격해 백도어의 지속성을 높인다. [ICML 논문 자료](https://proceedings.mlr.press/v162/zhang22w.html)

우리 연구에 적용: “ASR을 10~15%까지 낮추면 이후 FL이 지울 것”을 결론이 아닌 실험 가설로 둔다. 공격 중단·지속·재개를 나누고 후속 최대 ASR, 시간에 따른 평균 위험, 정상 성능을 함께 본다. 현재 실험기의 공격 재개 조건을 실제 클라이언트 탈퇴·재가입으로 설명하지 않는다.

## 알고리즘 발전 방향의 우선순위

**첫째, 정화가 작동하는 위치부터 확인한다.** 동일한 FU 경로에서 무정화와 후보 기저 정화를 비교한다. 헤드에 제한한 기저가 실패하면 마지막 블록과 선택된 층으로 확장하되, 정상 피해·계산량 증가를 같이 평가한다. 정답 후보와 같은 공간·랭크·적용 대상을 맞춘 무작위 대조군으로 방향 선택의 의미를 확인한다.

**둘째, 탐지 점수를 피해 제한 정책으로 연결한다.** 점수를 악성 확률로 해석하기 전에 정상 요청의 오탐을 보정한다. 강도·랭크·개입 대상을 조절할 때의 정상 손실과 위험 감소를 검증 실행에서 정하고, 평가 실행에서는 정책을 고정한다. 반복 사용되는 깨끗한 guard 데이터와 최종 성능 평가 데이터도 분리한다.

**셋째, 망각과 정화를 분리해 검증한다.** 요청자 제외 재학습은 삭제 비교 기준이고, 잔존 백도어까지 제거한 이상적 모델은 보안 참고 기준이다. 둘은 같은 목표가 아니다. 요청자 기록으로 만든 기저·점수도 그 요청자의 정보를 담을 수 있으므로, 모델만이 아니라 파생 상태의 보존·정리 범위까지 정해야 한다. 데이터 제외와 근사 FU 수행만으로 완전한 망각을 주장하지 않는다.

**넷째, 요청 단위 효율성과 전체 비용을 함께 보인다.** 요청자 중심 검사는 쌍 수를 줄이지만 기록 비용과 기저 계산은 남는다. 먼저 현재 구현을 측정하고, 저장·I/O 또는 SVD가 병목일 때 최근 창 저장, 저차원 요약, 후보 축소를 추가한다. 각 최적화에서 탐지 누락률도 함께 보고한다.

**다섯째, 가까운 선행연구와의 차별점을 결과로 입증한다.** Malicious Forgetting의 세부 방법을 확보해 정보 접근과 처리 시점을 대조한다. 상쇄 점수만으로 차별점이 약하다면 실제 FU 내부 개입의 유효성, 불완전한 탐지의 피해 제한, 삭제 효과에 대한 분석을 주 기여로 세운다.

## 미팅에서 사용할 주장 표현

| 사용할 표현 | 피할 표현 |
|---|---|
| 분리형 공격에서 상쇄 신호와 근사 정화 가능성을 관측했다 | BadFU와 FUBA를 완벽히 방어했다 |
| 탐지용 언러닝 없이 실제 FU 내부 정화를 검증한다 | 모든 기존 방어는 언러닝 시뮬레이션이 필요하다 |
| 정상 성능과 삭제 효과를 측정하며 정화 강도를 제한한다 | 삭제 요청을 실행했으므로 망각이 보장된다 |
| 후속 FL에서 위험이 줄거나 재발하는지 평가한다 | ASR 15% 이하면 안전하고 자연히 사라진다 |
| 연구 기여 후보이며 가까운 논문과 차별성을 확인 중이다 | 최초·범용·보장된 방어다 |

이 목록은 연구 방향을 정하기 위한 선별 검토다. 본문까지 확인한 BadFU·MASA·FedEraser·SCRUB-FL과, 출판사/학회/저자 요약을 확인한 나머지 자료의 검토 깊이를 구분한다. 특히 Malicious Forgetting의 세부 방어를 확인하기 전에는 해당 방법보다 우월하다고 주장하지 않는다.
