# Saved-model deletion comparison

Research scores. `unknown` is abstention, never a benign prediction. No calibrated FPR/FNR.

| Case | Candidate | DD | History normal set | DD + history | Coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| fuba_mnist | historical_parameter_subtraction_surrogate | 0.00302191 | 0 | 0 | 100.0% |
| fuba_cifar | historical_parameter_subtraction_surrogate | 0.00121416 | 0 | 0 | 100.0% |
| badfu_clean | historical_parameter_subtraction_surrogate | -0.00763088 | unknown | unknown | 0.0% |
| badfu_surrogate | historical_parameter_subtraction_surrogate | 0.00118535 | unknown | unknown | 0.0% |
| badfu_actual | actual_author_fu | 0.00354973 | unknown | unknown | 0.0% |

Use comparison.json for matched-coverage AUC and abstention counts.
Actual BadFU has no automatically available clean5 actual-FU control. FUBA defaults are surrogates.
No FL training or FU was executed. Full saved trajectory I/O and Jacobian inference can be substantial.
Public test images are reused benchmark development data; reserve fresh data before a final claim.

Missing inputs (not trained automatically):
