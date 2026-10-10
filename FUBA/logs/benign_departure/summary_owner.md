# 정상 이탈 검증 요약 (ASR %, 괄호는 ACC %; 시드 평균 ± 표준편차)

## bd_owner  (n=3, 역할: attackers [0, 1, 2, 3], defender 4, owner 7)

- 삭제 전: ASR 8.5 ± 1.1 / ACC 92.3 ± 0.9

| FU 방법 | req 4 (defender) | req 7 (owner) | req 5 (benign) |
|---|---|---|---|
| subtract | 53.2 ± 7.7 (69.3 ± 3.3) | 29.1 ± 4.9 (81.3 ± 2.3) | 16.2 ± 7.7 (90.6 ± 1.3) |
| distillation | 44.4 ± 6.1 (71.9 ± 4.6) | 14.3 ± 1.6 (89.2 ± 1.5) | 14.0 ± 1.3 (90.6 ± 1.1) |
| fedEraser | 7.2 ± 0.6 (88.9 ± 1.2) | 6.3 ± 6.2 (82.6 ± 1.0) | 4.8 ± 3.3 (84.7 ± 2.6) |
| retrain_benign | 2.1 ± 0.5 (96.4 ± 0.1) | 2.0 ± 0.1 (95.9 ± 0.4) | 1.7 ± 0.3 (96.5 ± 0.2) |
| retrain_attack | 23.8 ± 12.0 (98.4 ± 0.2) | 41.5 ± 18.3 (98.4 ± 0.1) | 39.2 ± 16.7 (98.5 ± 0.0) |

- 삭제 전 대비 ASR 변화(%p):
  - subtract: req4(defender) +44.6, req7(owner) +20.5, req5(benign) +7.7
  - distillation: req4(defender) +35.9, req7(owner) +5.8, req5(benign) +5.4
  - fedEraser: req4(defender) -1.3, req7(owner) -2.2, req5(benign) -3.7
  - retrain_benign: req4(defender) -6.5, req7(owner) -6.5, req5(benign) -6.8
  - retrain_attack: req4(defender) +15.2, req7(owner) +32.9, req5(benign) +30.6
