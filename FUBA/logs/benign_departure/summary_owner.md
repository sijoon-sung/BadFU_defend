# 정상 이탈 검증 요약 (ASR %, 괄호는 ACC %; 시드 평균 ± 표준편차)

## bd_owner  (n=3, 역할: attackers [0, 1, 2, 3], defender 4, owner 7)

- 삭제 전: ASR 8.1 ± 1.1 / ACC 92.5 ± 0.8

| FU 방법 | req 4 (defender) | req 7 (owner) | req 5 (benign) |
|---|---|---|---|
| subtract | 54.3 ± 6.0 (69.3 ± 2.9) | 29.4 ± 5.4 (81.2 ± 2.4) | 15.5 ± 7.0 (91.1 ± 1.2) |
| distillation | 43.1 ± 5.7 (73.0 ± 4.7) | 13.6 ± 1.6 (89.7 ± 1.5) | 13.1 ± 1.2 (91.0 ± 1.2) |
| fedEraser | 7.7 ± 0.5 (89.0 ± 1.0) | 5.9 ± 5.2 (82.9 ± 1.7) | 4.7 ± 2.5 (84.7 ± 3.4) |
| retrain_benign | 2.1 ± 0.6 (96.4 ± 0.1) | 2.1 ± 0.1 (95.9 ± 0.4) | 1.8 ± 0.4 (96.4 ± 0.2) |
| retrain_attack | 29.6 ± 19.2 (98.4 ± 0.1) | 46.9 ± 23.4 (98.4 ± 0.2) | 51.2 ± 17.1 (98.6 ± 0.1) |

- 삭제 전 대비 ASR 변화(%p):
  - subtract: req4(defender) +46.3, req7(owner) +21.4, req5(benign) +7.5
  - distillation: req4(defender) +35.1, req7(owner) +5.5, req5(benign) +5.0
  - fedEraser: req4(defender) -0.4, req7(owner) -2.2, req5(benign) -3.4
  - retrain_benign: req4(defender) -6.0, req7(owner) -6.0, req5(benign) -6.3
  - retrain_attack: req4(defender) +21.5, req7(owner) +38.8, req5(benign) +43.1
