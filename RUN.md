# 탐지: 기존 결과 검산

관측은 완료했다. 지금 다시 학습할 필요는 없다.
[실험표](experiments/README.md)에서 가설·결과·보류 이유부터 확인한다.

코드 연결 검사:

```bash
python -m unittest discover -s reproduction/tests -t . -v
python -m unittest discover -s experiments -t . -v
```

업로드된 시계열 결과를 기존 입력에서 다시 계산하려면:

```bash
python -m experiments.detection.temporal_history.audit_temporal_uploads --analysis experiments/detection/temporal_history/results/c939c4f11272c550 --analysis experiments/detection/temporal_history/results/aeb01d3b125f892e --out artifacts/temporal_review_experiment_v3.json
```

이중 차분·정상 설명 집합의 기존 비교를 검산하려면:

```bash
python -m experiments.detection.deletion_comparison.audit_deletion_upload experiments/detection/deletion_comparison/results/07ad0aefbb22ddc8/deletion_comparison.zip --out artifacts/deletion_review_experiment_v3
```

새 출력 경로를 쓴다. 검산 일치는 탐지 성능의 추가 이득을 뜻하지 않는다.
정화 실행은 [purification 브랜치](https://github.com/sijoon-sung/BadFU_defend/blob/purification/RUN.md)에 있다.
