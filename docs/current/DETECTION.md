# 탐지 작업

[문제 정의](PROBLEM_STATEMENT.md#1-탐지-어떤-삭제-요청에-활성화-위험이-있는가) · [결과표](../../RESULTS.md)

## 입력과 관측을 구분한다

| 방법 | 보는 정보 | 현재 판정 |
| --- | --- | --- |
| HRAC·기여 관계 | 이력, 요청자/잔류 기여, 공개 입력의 반응 | 일부 요청자 순위 확인. 단순 기준선 대비 이득 미입증 |
| 시계열 GLR | 이력 점수의 시간 변화와 정상 기준 잔차 | 단일 라운드로 최대값 설명. 시계열 추가 이득 미입증 |
| 정상 설명 집합 | 공개 정상 기능으로 기여를 설명할 수 있는가 | 현재 비영 기여의 설명 가능 집합 실패, 보류 |
| 실제 FU 전후 차분 | 실제 post와 pre의 변형 반응 | BadFU의 위험 증가 관측. 사전 탐지·독립 정상 구분의 증거 아님 |

## 코드와 결과

- `experiments/detection/hrac/validate_hrac_requests.py`, `hrac_request_gate.py`: 요청과 이력 관계.
- `experiments/detection/temporal_history/validate_temporal_history.py`, `temporal_history.py`: 시계열 분석.
- `experiments/detection/deletion_comparison/run_deletion_comparison.py`, `history_boundary.py`, `deletion_contrast.py`: 차분/정상 설명 집합 비교.
- `experiments/detection/deletion_comparison/audit_deletion_upload.py`, `audit_temporal_uploads.py`: 업로드 관측 검산.
- [완료 비교](../../experiments/detection/deletion_comparison/REVIEW.md), [이전 전체 기록](../experiments/RESEARCH_HISTORY.md).

## 다음 판단

독립 정상 실제-FU 쌍과 공격 쌍에서 같은 설정·공개 자료 조건으로 평가해야 한다.
새 방법이 단순 변화량·음의 기여·마지막 라운드보다 나은지 먼저 본다.
실제 post를 이용하면 사후 검사라고 표시하고, 공격이 이미 활성화된 시점을 조기 탐지로 계산하지 않는다.
현재 완료 결과를 다시 돌리는 것만으로 이 문제가 해결되지는 않는다.
