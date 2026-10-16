# Saved-model deletion comparison

Research scores. `unknown` is abstention, never a benign prediction. No calibrated FPR/FNR.

| Case | Candidate | DD | History normal set | DD + history | Coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| badfu_attack_surrogate | historical_parameter_subtraction_surrogate | 7.15447 | unknown | unknown | 0.0% |
| badfu_clean_delete0 | historical_parameter_subtraction_surrogate | 4.16311 | unknown | unknown | 0.0% |
| badfu_clean_delete1 | historical_parameter_subtraction_surrogate | 5.96582 | unknown | unknown | 0.0% |
| badfu_clean_delete2 | historical_parameter_subtraction_surrogate | 3.10474 | unknown | unknown | 0.0% |
| badfu_clean_delete3 | historical_parameter_subtraction_surrogate | 3.99885 | unknown | unknown | 0.0% |
| badfu_clean_delete4 | historical_parameter_subtraction_surrogate | 3.45104 | unknown | unknown | 0.0% |
| badfu_attack_actual | actual_author_fu | 6.01417 | unknown | unknown | 0.0% |
| fuba_attack_surrogate | historical_parameter_subtraction_surrogate | 710.848 | unknown | unknown | 32.5% |
| fuba_benign0_delete4 | historical_parameter_subtraction_surrogate | 30.0548 | unknown | unknown | 32.5% |
| fuba_benign1_delete4 | historical_parameter_subtraction_surrogate | 7.43796 | unknown | unknown | 30.5% |
| fuba_benign2_delete4 | historical_parameter_subtraction_surrogate | 37.5655 | unknown | unknown | 29.5% |

Use comparison.json for matched-coverage AUC and abstention counts.
Actual BadFU has no automatically available clean5 actual-FU control. FUBA defaults are surrogates.
No FL training or FU was executed. Full saved trajectory I/O and Jacobian inference can be substantial.
Public test images are reused benchmark development data; reserve fresh data before a final claim.

Missing inputs (not trained automatically):
