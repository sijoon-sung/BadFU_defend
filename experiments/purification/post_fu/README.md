# 실제 FU 뒤 정화 비교

**본 실험 대기**

- **가설:** 삭제 후 새로 커진 변형 반응을 억제하면서 정상 기능을 유지할 수 있다.
- **해본 것:** post-only, 평균 차분, CVaR, 클래스 제약을 포함한 여섯 비교군을 구현했다.
- **관측 결과:** 수치·연결 검사와 2R 축소 실행만 확인했다.
- **현재 판단:** 저장된 실제 BadFU full6→fu_remove5 모델로 본 비교를 진행한다.
- **보류 이유:** 본 성능·망각 품질·독립 seed는 미확인. CVaR 자체는 새 방법이 아니다.

[결과](results/) · [검사 기록](checks/) · [검사 코드](tests/) · [실행](../../../RUN.md) · [수식·비교군](../../../docs/current/PURIFICATION_COMPARE.md)

[전체 실험표](https://github.com/sijoon-sung/BadFU_defend/blob/main/experiments/README.md)
