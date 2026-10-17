# 현재 결과

| 실험 | 확인한 것 | 아직 말할 수 없는 것 |
| --- | --- | --- |
| HRAC | BadFU 패턴 관측에서 요청자 1위. 단순 기준선도 1위 | 이력 방법의 우월성 |
| 시계열 GLR | 저장 점수 재계산 일치. FUBA 최대값은 한 라운드로 설명 | 시간 구간을 본 추가 이득 |
| 정상 설명 집합 | 11개 사례의 비영 기여 1,110개 모두 설명 실패 | 정상/공격 판별 |
| 누적 기여 차감 | 정상 대조도 정확도 붕괴 | 실제 FU의 위험 |
| 실제 BadFU FU 이중 차분 | 탐색 패치 표적 반응 2.78→73.61%, 정상 정확도 91.25→95% | 진짜 트리거 ASR, 정상 삭제 구분, 사전 탐지 |
| 실제 FU 후 정화 | BadFU 평균 차분 ASR 99→1.44%, 정확도 85.20→71.70%. CVaR 1.11% / 70.40%. 제약 arm은 원래 post로 복귀 | 정상 효용을 보존하는 정화, FUBA 일반화·망각 품질 |

**BadFU 단일 쌍에서 ASR 억제는 관측했다. 탐지 우월성·정상 효용을 보존하는 정화는 아직 미입증이다.**

가설·보류 이유·원래 수치는 [실험별 다섯 줄 요약](https://github.com/sijoon-sung/BadFU_defend/blob/main/experiments/README.md)에서 확인한다.
정화 본 비교는 1차 완료했다. 다음은 정상 기능 손상의 원인을 확인하는 것이다. [결과 검토](https://github.com/sijoon-sung/BadFU_defend/blob/purification/experiments/purification/post_fu/REVIEW.md).
