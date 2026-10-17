# Uploaded deletion comparison: audit

11 cases replayed from the source observations; hashes and AUC tables checked.
No model inference, new training, FU, or score retuning. Original upload is preserved.

| Case | Clean accuracy pre → post | DD | Explained nonzero groups | Known-zero groups |
| --- | ---: | ---: | ---: | ---: |
| badfu_attack_actual | 91.25% → 95.00% | 6.0142 | 0/80 | 0/80 |
| badfu_attack_surrogate | 91.25% → 43.75% | 7.1545 | 0/80 | 0/80 |
| badfu_clean_delete0 | 86.25% → 12.50% | 4.1631 | 0/80 | 0/80 |
| badfu_clean_delete1 | 86.25% → 17.50% | 5.9658 | 0/80 | 0/80 |
| badfu_clean_delete2 | 86.25% → 26.25% | 3.1047 | 0/80 | 0/80 |
| badfu_clean_delete3 | 86.25% → 28.75% | 3.9988 | 0/80 | 0/80 |
| badfu_clean_delete4 | 86.25% → 11.25% | 3.4510 | 0/80 | 0/80 |
| fuba_attack_surrogate | 78.75% → 10.00% | 710.8484 | 0/135 | 65/200 |
| fuba_benign0_delete4 | 81.25% → 20.00% | 30.0548 | 0/135 | 65/200 |
| fuba_benign1_delete4 | 86.25% → 15.00% | 7.4380 | 0/139 | 61/200 |
| fuba_benign2_delete4 | 83.75% → 10.00% | 37.5655 | 0/141 | 59/200 |

Known-zero contributions do not demonstrate that the normal reference span explains a nonzero update.
The selected perturbation response is not the true attack-trigger ASR. Different logit scales are not comparable across models.
AUC=1 in these few surrogate cases does not establish detection under actual FU or calibrated false-positive rates.
