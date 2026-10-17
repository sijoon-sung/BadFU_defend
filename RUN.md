# 실행 안내

목적에 맞는 브랜치의 최신 RUN.md를 따른다.

- [탐지: 저장 결과 검산과 요청 위험 관측](https://github.com/sijoon-sung/BadFU_defend/blob/codex/detection/RUN.md)
- [정화: 실제 FU 뒤 저장 모델의 여섯 비교군](https://github.com/sijoon-sung/BadFU_defend/blob/codex/purification/RUN.md)

이미 실험 중이면 현재 작업을 마치고 결과를 보관한다. 코드 재배치와 AI 작성자 표기 정리로
모듈 경로·커밋 ID가 바뀌었다. [기존 컴퓨터의 전환 안내](BRANCHES.md#다른-컴퓨터의-기존-실행)를 따른다.

공통 원본 무결성 검사:

```bash
python -m reproduction.training.run check
python -m unittest discover -s reproduction/tests -t . -v
python -m unittest discover -s FUBA/tests -v
```

[문제 정의](docs/current/PROBLEM_STATEMENT.md) · [현재 결과](RESULTS.md) · [코드 폴더](reproduction/README.md).
