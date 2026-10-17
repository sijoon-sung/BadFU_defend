# 종합 보고서: owner_r100w5_k20paper_flow  (K=20, ROUNDS=100, WARMUP=5, seeds: 0 1 2)

---
<!-- logs/dose_fu/summary_owner_r100w5_k20paper_flow.md -->
# 투여량 조절 FU 요약 (시드 평균 ± 표준편차)

## bd_owner_r100w5_k20paper_flow  (n=3, 삭제 전 ASR 5.7 ± 1.7 / ACC 98.2 ± 0.1)

### asr
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 100.0 ± 0.0 | 24.0 ± 11.1 | 6.7 ± 2.3 |
| oneshot | 1.3 ± 1.5 | 5.4 ± 3.8 | 6.7 ± 2.3 |
| dose | 3.3 ± 1.9 | 5.4 ± 3.8 | 6.7 ± 2.3 |
| always | 3.3 ± 1.9 | 3.0 ± 1.1 | 4.0 ± 0.6 |

### acc
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 9.8 ± 0.0 | 85.2 ± 1.1 | 96.0 ± 0.1 |
| oneshot | 45.4 ± 24.1 | 92.9 ± 4.4 | 96.0 ± 0.1 |
| dose | 96.2 ± 0.4 | 92.9 ± 4.4 | 96.0 ± 0.1 |
| always | 96.2 ± 0.4 | 96.4 ± 0.6 | 97.8 ± 0.1 |

### retain_change
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 90.16 ± 0.08 | 14.06 ± 1.15 | 3.23 ± 0.19 |
| oneshot | 54.36 ± 24.30 | 6.55 ± 4.26 | 3.23 ± 0.19 |
| dose | 3.42 ± 0.38 | 6.56 ± 4.25 | 3.23 ± 0.19 |
| always | 3.42 ± 0.37 | 3.17 ± 0.56 | 1.19 ± 0.11 |

### reinjection_alpha
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| oneshot | 0.038 ± 0.013 | 0.290 ± 0.210 | 0.000 ± 0.000 |
| dose | 0.120 ± 0.006 | 0.290 ± 0.210 | 0.000 ± 0.000 |
| always | 0.120 ± 0.006 | 0.424 ± 0.047 | 0.090 ± 0.017 |

### removed
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| oneshot | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| dose | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| always | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

### n_steps
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 |
| oneshot | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 |
| dose | 8.3 ± 1.7 | 1.0 ± 0.0 | 1.0 ± 0.0 |
| always | 8.3 ± 1.7 | 1.0 ± 0.0 | 1.0 ± 0.0 |

### n_flagged
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oneshot | 1.0 ± 0.0 | 0.7 ± 0.5 | 0.0 ± 0.0 |
| dose | 8.3 ± 1.7 | 0.7 ± 0.5 | 0.0 ± 0.0 |
| always | 8.3 ± 1.7 | 0.7 ± 0.5 | 0.0 ± 0.0 |

### n_repaired
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oneshot | 1.0 ± 0.0 | 0.7 ± 0.5 | 0.0 ± 0.0 |
| dose | 8.3 ± 1.7 | 0.7 ± 0.5 | 0.0 ± 0.0 |
| always | 8.3 ± 1.7 | 1.0 ± 0.0 | 1.0 ± 0.0 |

### n_hard_collapse
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oneshot | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| dose | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| always | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

### pop_models
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| oneshot | 20 ± 0 | 0 ± 0 | 0 ± 0 |
| dose | 293 ± 68 | 0 ± 0 | 0 ± 0 |
| always | 220 ± 86 | 0 ± 0 | 0 ± 0 |

### seconds
| arm | req 4 (requester(defender)) | req 19 (benign) | req 5 (benign) |
|---|---|---|---|
| plain | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| oneshot | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| dose | 4 ± 1 | 0 ± 0 | 0 ± 0 |
| always | 4 ± 1 | 0 ± 0 | 0 ± 0 |

### 망각: 요청자 자기 데이터 정확도 before → final / MIA AUROC
| arm | req 4 | req 19 | req 5 |
|---|---|---|---|
| plain | 99.7 ± 0.1 → 10.8 ± 0.0 / 0.544 ± 0.003 | 94.6 ± 0.3 → 30.6 ± 0.4 / 0.225 ± 0.000 | 98.8 ± 0.2 → 95.7 ± 0.7 / 0.532 ± 0.006 |
| oneshot | 99.7 ± 0.1 → 48.9 ± 25.8 / 0.541 ± 0.011 | 94.6 ± 0.3 → 73.1 ± 29.8 / 0.380 ± 0.110 | 98.8 ± 0.2 → 95.7 ± 0.7 / 0.532 ± 0.006 |
| dose | 99.7 ± 0.1 → 95.8 ± 0.6 / 0.505 ± 0.018 | 94.6 ± 0.3 → 73.1 ± 29.8 / 0.380 ± 0.110 | 98.8 ± 0.2 → 95.7 ± 0.7 / 0.532 ± 0.006 |
| always | 99.7 ± 0.1 → 95.8 ± 0.6 / 0.504 ± 0.018 | 94.6 ± 0.3 → 94.7 ± 1.0 / 0.479 ± 0.031 | 98.8 ± 0.2 → 97.6 ± 0.2 / 0.533 ± 0.011 |

### 탐지 (dose arm): 첫 경보 때 누적 제거량 / 미경보 시드 / 최대 z / 최대 z_cum
- req 4 (requester(defender)): 첫 경보 제거량 0.17 ± 0.05 (미경보 0/3), 최대 z 497.6 ± 175.7, 최대 z_cum -
- req 19 (benign): 첫 경보 제거량 1.00 ± 0.00 (미경보 1/3), 최대 z 6.7 ± 5.7, 최대 z_cum -
- req 5 (benign): 첫 경보 제거량 - (미경보 3/3), 최대 z 0.2 ± 0.2, 최대 z_cum -

