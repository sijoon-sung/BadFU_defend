# 현재 실행: 실제 FU 후 정화

기존 BadFU 모델을 재사용한다. 학습을 다시 할 필요는 없다.
진행 중인 실험은 그대로 마치고, 이번 코드로 시작할 때는 새 출력 폴더를 쓴다.

```bash
python -m experiments.purification.post_fu.run_purification --run /기존/경로/badfu_author_s42 --dry-run --math-ablation
python -m experiments.purification.post_fu.run_purification --run /기존/경로/badfu_author_s42 --out artifacts/purification_badfu_s42_experiment_v3 --math-ablation
```

필요하면 `--data-root /기존/경로/data --record-root /기존/경로/badnet_dataset`를 붙인다.
예상 시간은 GPU 수십 분~몇 시간이며 본 예산 실측은 아직 없다. 기존 결과 폴더를 덮어쓰지 않는다.

끝나면 작은 결과만 올린다.

```bash
git add experiments/purification/post_fu/results
git commit -m "Add saved-model purification comparison"
git push origin HEAD:purification
```

여섯 비교군·필수 입력·중단 후 재개는 [실험 실행 안내](experiments/purification/post_fu/RUN.md)에 있다.
[가설과 현재 판단](experiments/purification/post_fu/README.md) · [다른 컴퓨터의 기존 checkout 전환](BRANCHES.md#다른-컴퓨터의-기존-실행).
