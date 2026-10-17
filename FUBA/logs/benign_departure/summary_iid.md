# 정상 이탈 검증 요약 (ASR %, 괄호는 ACC %; 시드 평균 ± 표준편차)

## bd_iid  (n=3, 역할: attackers [0, 1, 2, 3], defender 4, owner None)

- 삭제 전: ASR 6.2 ± 1.8 / ACC 95.1 ± 0.3

| FU 방법 | req 4 (defender) | req 7 (benign) | req 5 (benign) |
|---|---|---|---|
| subtract | 48.0 ± 10.2 (79.4 ± 2.6) | 13.7 ± 5.9 (93.7 ± 1.1) | 14.0 ± 4.8 (93.9 ± 0.7) |
| distillation | 29.3 ± 4.1 (84.5 ± 1.3) | 9.5 ± 2.7 (94.2 ± 0.5) | 9.3 ± 2.4 (94.3 ± 0.3) |
| fedEraser | 3.9 ± 1.3 (92.7 ± 0.3) | 1.7 ± 0.6 (89.0 ± 1.7) | 3.3 ± 1.8 (89.5 ± 2.4) |
| retrain_benign | 0.8 ± 0.1 (97.6 ± 0.1) | 0.8 ± 0.2 (97.6 ± 0.1) | 0.7 ± 0.1 (97.6 ± 0.2) |
| retrain_attack | 44.3 ± 23.3 (98.8 ± 0.1) | 46.2 ± 28.0 (98.9 ± 0.1) | 60.4 ± 13.7 (99.0 ± 0.1) |

- 삭제 전 대비 ASR 변화(%p):
  - subtract: req4(defender) +41.8, req7(benign) +7.5, req5(benign) +7.8
  - distillation: req4(defender) +23.1, req7(benign) +3.4, req5(benign) +3.1
  - fedEraser: req4(defender) -2.3, req7(benign) -4.4, req5(benign) -2.9
  - retrain_benign: req4(defender) -5.4, req7(benign) -5.4, req5(benign) -5.4
  - retrain_attack: req4(defender) +38.1, req7(benign) +40.1, req5(benign) +54.2
