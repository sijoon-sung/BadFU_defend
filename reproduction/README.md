# 코드 안내

실행 코드를 역할별 폴더로 분리했다. 루트에는 패키지 표식·설정·설명만 둔다.

| 폴더 | 역할 | 주요 파일 |
| --- | --- | --- |
| [training/](training/) | 고정한 저자 코드 실행·기록 어댑터 | run.py, badfu_adapter.py, fuba_adapter.py |
| [common/](common/) | 궤적 저장·무결성·공통 입력 | trajectory.py, deletion_inputs.py |
| [detection/](detection/) | 이력·요청 관측, 정상 설명 집합, 로짓 차분 | hrac_request_gate.py, temporal_history.py, run_deletion_comparison.py |
| [purification/](purification/) | 실제 FU 뒤 정화 비교 | run_purification.py, purification_compare.py, purification_math.py |
| [audit/](audit/) | 업로드 결과 검산·원인 분석 | audit_deletion_upload.py, audit_temporal_uploads.py |
| [tests/](tests/) | 회귀 검사 | test_*.py |
| [validation/](validation/README.md) | 업로드된 증거·기존 검사 결과 | 원래 파일과 해시 보존 |

저장소 루트에서 `python -m reproduction.분류.모듈` 형태로 실행한다.

```bash
python -m reproduction.training.run check
python -m reproduction.purification.run_purification --dry-run --math-ablation
python -m unittest discover -s reproduction/tests -t . -v
```

새 학습은 위 폴더 구조 그대로 기록기 사본을 저장한다. 예전 평면 구조의 기록기 사본도 읽을 수 있다.
이미 실행한 실험의 소스 사본이나 결과를 새 코드로 덮어쓰지 않는다. 코드가 달라졌으므로 정화 재실행은 새 출력 폴더를 사용한다.
과거 시계열 결과의 원래 소스는 `audit/source_snapshots/`의 비실행 텍스트로 확인하고, 현재 코드의 수치 재계산 일치 여부는 별도로 검사한다.

[문제 정의](../docs/current/PROBLEM_STATEMENT.md) · [결과표](../RESULTS.md) · [현재 실행](../RUN.md) · [경로 변경표](../docs/operations/code_layout_20261017.json).
