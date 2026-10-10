# 정상 이탈 검증 요약 (ASR %, 괄호는 ACC %; 시드 평균 ± 표준편차)

## bd_iid  (n=3, 역할: attackers [0, 1, 2, 3], defender 4, owner None)

- 삭제 전: ASR 6.3 ± 1.8 / ACC 95.1 ± 0.2

| FU 방법 | req 4 (defender) | req 7 (benign) | req 5 (benign) |
|---|---|---|---|
| subtract | 49.1 ± 9.2 (78.7 ± 2.5) | 13.9 ± 8.1 (93.8 ± 1.3) | 14.2 ± 4.8 (93.8 ± 0.9) |
| distillation | 29.7 ± 4.3 (84.0 ± 1.5) | 9.4 ± 2.7 (94.3 ± 0.3) | 9.2 ± 2.2 (94.3 ± 0.3) |
| fedEraser | 3.9 ± 1.0 (92.7 ± 0.2) | 1.9 ± 0.5 (89.3 ± 0.5) | 3.2 ± 1.9 (89.5 ± 1.1) |
| retrain_benign | 0.8 ± 0.1 (97.7 ± 0.1) | 0.8 ± 0.2 (97.6 ± 0.1) | 0.7 ± 0.1 (97.6 ± 0.2) |
| retrain_attack | 45.8 ± 21.9 (98.8 ± 0.0) | 38.6 ± 27.4 (98.9 ± 0.0) | 64.6 ± 11.7 (98.9 ± 0.1) |

- 삭제 전 대비 ASR 변화(%p):
  - subtract: req4(defender) +42.8, req7(benign) +7.6, req5(benign) +7.9
  - distillation: req4(defender) +23.3, req7(benign) +3.1, req5(benign) +2.9
  - fedEraser: req4(defender) -2.4, req7(benign) -4.5, req5(benign) -3.1
  - retrain_benign: req4(defender) -5.5, req7(benign) -5.5, req5(benign) -5.6
  - retrain_attack: req4(defender) +39.5, req7(benign) +32.3, req5(benign) +58.2
