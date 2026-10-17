# 정상 이탈 검증 요약 (ASR %, 괄호는 ACC %; 시드 평균 ± 표준편차)

## bd_iid_r100w5_k20paper  (n=3, 역할: attackers [0, 1, 2, 3], defender 4, owner None)

- 삭제 전: ASR 5.2 ± 1.8 / ACC 98.7 ± 0.1

| FU 방법 | req 4 (defender) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| subtract | 100.0 ± 0.0 (9.7 ± 0.0) | 5.4 ± 2.8 (96.1 ± 0.4) | 3.8 ± 1.0 (96.3 ± 1.2) |
| distillation | 18.5 ± 6.6 (97.2 ± 1.6) | 3.6 ± 1.4 (98.6 ± 0.1) | 3.6 ± 0.9 (98.6 ± 0.1) |
| fedEraser | 1.0 ± 0.2 (98.2 ± 0.3) | 0.9 ± 0.6 (96.8 ± 1.6) | 0.7 ± 0.5 (97.0 ± 1.4) |

- 삭제 전 대비 ASR 변화(%p):
  - subtract: req4(defender) +94.8, req19(benign) +0.1, req5(benign) -1.4
  - distillation: req4(defender) +13.2, req19(benign) -1.6, req5(benign) -1.6
  - fedEraser: req4(defender) -4.3, req19(benign) -4.3, req5(benign) -4.6
