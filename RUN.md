# 실행 상태: BadFU 정화 1차 완료

올린 결과는 검토했다. 같은 실험을 다시 돌릴 필요는 없다.
평균 차분은 ASR 99→1.44%로 낮췄지만 정상 정확도 13.50%p를 잃었다. 제약 arm은 원래 post로 복귀했다.

[결과·판단·다음 확인](experiments/purification/post_fu/REVIEW.md) · [가설 다섯 줄 요약](experiments/purification/post_fu/README.md).
다음 실험은 정상 미세조정부터 효용 손상을 진단하는 방향이며, 새 실행 예산은 아직 정하지 않았다.

업로드 검산만 다시 하려면 (모델·학습 불필요):

```bash
python -m experiments.purification.post_fu.audit_upload experiments/purification/post_fu/results/98eb3de68a0561ea --out artifacts/purification_upload_review.json
```

이전 실행 명령·필수 입력은 [보관한 실행 안내](experiments/purification/post_fu/RUN.md)에 있다.
새 설정을 시험할 때는 새 출력 폴더를 사용하고 기존 결과를 보존한다.
