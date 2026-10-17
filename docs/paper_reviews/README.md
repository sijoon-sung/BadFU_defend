# 논문 리뷰와 원문 찾기

기존 정리는 사라지지 않았다. 공격 리뷰 4개와 방어/FU 리뷰 12개를 아래에서 바로 찾을 수 있다. 각 상세 리뷰는 요청한 **Problem Statement → Key Idea & Formulation → Evaluation & Results → Limitations & Critical Discussion → Takeaway & Next Actions** 다섯 항목을 따른다. 원문 링크·표 번호는 각 파일에 있으며 저자 결과와 우리 해석을 구분한다. FLShield·FLTrust 및 Non-IID 언러닝 관련 J–L은 2026-10-16에 추가했다.

| 논문 | 다섯 항목 리뷰 |
|---|---|
| BadFU | [01_BadFU.md](01_BadFU.md) |
| FUBA | [02_FUBA.md](02_FUBA.md) |
| FedMUA | [03_FedMUA.md](03_FedMUA.md) |
| Malicious Forgetting / FUsion / SubID | [04_malicious_forgetting_FUsion_SubID.md](04_malicious_forgetting_FUsion_SubID.md) |
| MASA | [06_MASA.md](06_MASA.md) |
| BackdoorIndicator | [07_BackdoorIndicator.md](07_BackdoorIndicator.md) |
| FedREDefense | [08_FedREDefense.md](08_FedREDefense.md) |
| DeepSight | [09_DeepSight.md](09_DeepSight.md) |
| FLAME | [10_FLAME.md](10_FLAME.md) |
| UnlearnGuard | [11_UnlearnGuard.md](11_UnlearnGuard.md) |
| FedEraser | [12_FedEraser.md](12_FedEraser.md) |
| FLShield | [통합본 H](05_defenses_novelty.md#h-flshield--클래스별-검증으로-데이터-구성의-혼동-감소) |
| FLTrust | [통합본 I](05_defenses_novelty.md#i-fltrust--작은-정상-자료로-신뢰-기준-추가) |
| FU Stability and Fairness | [통합본 J](05_defenses_novelty.md#j-federated-unlearning-a-perspective-of-stability-and-fairness) |
| FedOSD | [통합본 K](05_defenses_novelty.md#k-fedosd--federated-unlearning-with-gradient-descent-and-conflict-mitigation) |
| FUCRT | [통합본 L](05_defenses_novelty.md#l-fucrt--forgetting-through-transforming) |

[방어/FU 12편 통합본](05_defenses_novelty.md)은 정보 조건·시계열 선행·가정 완화 비교를 포함한다. [공격 비교표](00_comparison.md)는 역사적 탐색 메모로 읽고 현재 범위는 [연구 요약](../current/RESEARCH_SUMMARY.md)을 따른다. FoolsGold·FLDetector·GEM은 통합본에서 연결한 선행이며 같은 깊이의 상세 리뷰 완료로 세지 않는다. SubID 원문은 이번 재정리에서 재열람하지 못했으므로 기존 노트의 수치를 외부 인용하기 전 DOI 원문과 재대조한다.

## 주요 방어 논문에서 가져올 논리 전개

| 논문 | 한계에서 설계로 이어지는 논리 | 우리 연구에서 필요한 증거 |
|---|---|---|
| MASA | 정적 통계의 중첩 → 망각 동역학 → 융합으로 non-IID 혼동 완화 | 장력 점수만 추가하지 말고 정상 혼동을 줄이는 원인을 분리 |
| BackdoorIndicator | 수동 관측의 부족 → 서버가 지표 과제 추가 → BN 교란 통제 | 신호 부재를 임계값 조정으로 덮지 않고 필요한 정보 명시 |
| FedREDefense | 정상 다수와의 유사성 한계 → 학습 가능한 갱신의 재구성 오차 | 기존 거리 기준과 무엇이 다른 정상성 기준인지 설명 |
| DeepSight | 공격 강도와 흔적의 관계 → 검사·군집 → 남은 영향 제한 | 완벽한 신원 식별을 전제하지 않는 효용·위험 평가 |
| FLAME | 필터링과 잡음의 개별 한계 → 단계 결합으로 필요 잡음 감소 | 복잡한 조합이 같은 효용에서 추가 이득을 주는지 소거 |
| UnlearnGuard | FU의 공격면 → FL 이력 기반 예측·검사 → 조건부 분석 | ‘시계열을 사용한다’보다 구체적인 요청 정보의 추가 가치 |
| FedEraser | 전체 재학습 비용 → 저장 업데이트와 새 보정 → 비용/근사 비교 | 모든 minibatch gradient 가정 제거, 실제 비용과 망각 별도 측정 |
| FLShield | Non-IID 검증 결과의 차이 → 클래스별 손실 변화·대표 모델 → 보고 검증 | 이미지 전체 평균으로 지워진 클래스별 차이가 있는지 진단 |
| FLTrust | 신뢰 기준 부재 → 작은 정상 자료 → 기준 방향·크기 제한 | 이미 사용하는 공개 자료 가정을 명시하고 대표성 한계 보고 |

우리 기여 후보는 **실제 요청이 역사적 상쇄 관계를 선택적으로 끊는다는 정보의 추가 가치**다. 이를 정상 오탐률이 같은 선형 baseline과 비교해서 보여야 한다. 논리적으로 가능한 설계와 실험으로 확인한 기여를 구분한다.
