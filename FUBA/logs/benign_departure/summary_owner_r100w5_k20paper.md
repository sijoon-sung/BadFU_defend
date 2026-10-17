# 정상 이탈 검증 요약 (ASR %, 괄호는 ACC %; 시드 평균 ± 표준편차)

## bd_owner_r100w5_k20paper  (n=3, 역할: attackers [0, 1, 2, 3], defender 4, owner 19)

- 삭제 전: ASR 5.8 ± 1.7 / ACC 98.2 ± 0.1

| FU 방법 | req 4 (defender) | req 19 (owner) | req 5 (benign) |
|---|---|---|---|
| subtract | 100.0 ± 0.0 (9.7 ± 0.0) | 24.1 ± 11.1 (85.2 ± 1.1) | 6.7 ± 2.2 (96.0 ± 0.2) |
| distillation | 24.9 ± 3.1 (96.8 ± 0.3) | 5.5 ± 1.4 (97.7 ± 0.1) | 4.0 ± 1.0 (98.1 ± 0.1) |
| fedEraser | 1.1 ± 0.6 (98.1 ± 0.3) | 1.4 ± 0.4 (96.8 ± 1.7) | 1.5 ± 0.4 (97.4 ± 0.8) |

- 삭제 전 대비 ASR 변화(%p):
  - subtract: req4(defender) +94.2, req19(owner) +18.3, req5(benign) +0.9
  - distillation: req4(defender) +19.1, req19(owner) -0.3, req5(benign) -1.8
  - fedEraser: req4(defender) -4.7, req19(owner) -4.4, req5(benign) -4.3
