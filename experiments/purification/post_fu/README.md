# 실제 FU 뒤 정화 비교

**1차 결과 확인 — ASR 억제 / 정상 효용 보존 실패**

- **가설:** 삭제 후 새로 커진 변형 반응을 억제하면서 정상 기능을 유지할 수 있다.
- **해본 것:** 실제 BadFU 40R→FU 20R 모델로 여섯 비교군을 실행했다. 시험 1,000장, ASR 900장.
- **관측 결과:** 평균 차분 ASR 99→1.44%, 정상 정확도 85.20→71.70%. CVaR는 ASR 1.11%, 정확도 70.40%.
- **현재 판단:** 공격 억제는 관측했다. 정상 보존 제약은 5개 후보 모두 거절하고 원래 post로 복귀했다.
- **보류 이유:** 정상 정확도 손실 2%p 조건을 만족하며 ASR을 낮춘 방법이 없다. FUBA·정상 실제 FU·망각 품질은 미검증.

[결과](results/) · [검사 기록](checks/) · [검사 코드](tests/) · [실행](../../../RUN.md) · [수식·비교군](../../../docs/current/PURIFICATION_COMPARE.md)

[전체 실험표](https://github.com/sijoon-sung/BadFU_defend/blob/main/experiments/README.md)

[이번 결과와 다음 판단](REVIEW.md) · [원본 수치](results/98eb3de68a0561ea/report.md)
