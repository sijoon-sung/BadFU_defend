# 현재 실행 코드

실험 중인 작업과 저장된 소스 해시·재개 검사를 유지하기 위해 Python 모듈 경로는 바꾸지 않습니다. 연구 목적은 두 브랜치에서 구분합니다.

| 작업 | 주요 파일 |
| --- | --- |
| 원본 학습·궤적 | `run.py`, `trajectory.py`, `badfu_adapter.py`, `fuba_adapter.py` |
| 탐지·이력 | `validate_hrac_requests.py`, `hrac_request_gate.py`, `validate_temporal_history.py`, `temporal_history.py` |
| 차분·정상 설명 집합 | `run_deletion_comparison.py`, `deletion_contrast.py`, `history_boundary.py`, `deletion_inputs.py` |
| 정화 | `run_purification.py`, `purification_compare.py`, `purification_math.py` |
| 업로드 검산 | `audit_deletion_upload.py`, `audit_temporal_uploads.py`, `replay_deletion_comparison.py` |
| 검사 | 각 `test_*.py` |
| 증거 | [validation/](validation/README.md) |

[문제 정의](../docs/current/PROBLEM_STATEMENT.md) · [결과표](../RESULTS.md) · [실행](../RUN.md).
