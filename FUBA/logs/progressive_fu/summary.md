# 진행형 FU 요약 (시드 평균 ± 표준편차)

## loc_fuba  (n=1, 삭제 전 ASR 9.1 / ACC 94.8)

### asr
| arm | req 4 (requester(defender)) | req 7 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 62.9 | 21.7 | 22.4 |
| end | 9.7 | 7.6 | 7.3 |
| progressive | 7.8 | 21.7 | 22.4 |
| always | 8.6 | 6.9 | 7.0 |

### acc
| arm | req 4 (requester(defender)) | req 7 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 75.0 | 92.6 | 92.3 |
| end | 91.0 | 94.4 | 94.4 |
| progressive | 91.9 | 92.6 | 92.3 |
| always | 92.2 | 94.5 | 94.4 |

### retain_change
| arm | req 4 (requester(defender)) | req 7 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 22.08 | 3.55 | 4.05 |
| end | 4.95 | 1.96 | 2.01 |
| progressive | 4.16 | 3.55 | 4.05 |
| always | 3.79 | 1.95 | 2.04 |

### n_flagged
| arm | req 4 (requester(defender)) | req 7 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 5.0 | 0.0 | 0.0 |
| end | 5.0 | 0.0 | 0.0 |
| progressive | 3.0 | 0.0 | 0.0 |
| always | 2.0 | 0.0 | 0.0 |

### n_purified
| arm | req 4 (requester(defender)) | req 7 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 0.0 | 0.0 | 0.0 |
| end | 1.0 | 1.0 | 1.0 |
| progressive | 3.0 | 0.0 | 0.0 |
| always | 5.0 | 5.0 | 5.0 |

### MIA (요청자 AUROC / 멤버율@FPR5%)
| arm | req 4 | req 7 | req 5 |
|---|---|---|---|
| plain | 0.478 / 0.042 | 0.492 / 0.037 | 0.489 / 0.044 |
| end | 0.487 / 0.047 | 0.502 / 0.035 | 0.492 / 0.046 |
| progressive | 0.486 / 0.046 | 0.492 / 0.037 | 0.489 / 0.044 |
| always | 0.487 / 0.046 | 0.502 / 0.036 | 0.493 / 0.047 |

### 탐지 (progressive arm): 첫 경보 ρ / 경보 수 / 최대 z
- req 4 (requester(defender)): 첫 경보 ρ 0.20 (미경보 0/1), 경보 수 3.0, 최대 z 8.9
- req 7 (benign): 첫 경보 ρ - (미경보 1/1), 경보 수 0.0, 최대 z -0.0
- req 5 (benign): 첫 경보 ρ - (미경보 1/1), 경보 수 0.0, 최대 z 0.3
