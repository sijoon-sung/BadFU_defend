# 공통 재현 코드

- `training/`: 고정한 BadFU·FUBA 저자 코드 실행과 궤적 기록.
- `common/`: 궤적 무결성, 모델 입력, 공개 자료 분할, 과거 경로 연결.
- `tests/`: 원본·기록기 검사.

방법별 코드·결과는 [실험 폴더](../experiments/README.md)에 있다.

```bash
python -m reproduction.training.run check
python -m unittest discover -s reproduction/tests -t . -v
python -m unittest discover -s experiments -t . -v
```

저장된 궤적과 과거 결과 파일의 내용은 변경하지 않는다. 새 실험은 새 출력 폴더를 쓴다.
