# Purification comparison

Actual model-pair provenance is supplied by the experiment owner.
These are matched-objective ablations, not official NC/I-BAU implementations.

| Arm | Clean accuracy | ASR (non-target originals) | Triggered original accuracy | Clean tolerance |
| --- | ---: | ---: | ---: | --- |
| pre | 87.30% | 80.11% | 18.89% | reference |
| post | 85.20% | 99.00% | 0.89% | reference |
| bd5 | 84.70% | 100.00% | 0.00% | reference |
| clean_ft | 69.90% | 82.56% | 14.00% | False |
| post_only | 68.40% | 2.22% | 69.44% | False |
| differential | 71.70% | 1.44% | 73.00% | False |
| differential_cvar | 70.40% | 1.11% | 70.44% | False |
| differential_cvar_constrained | 85.20% | 99.00% | 0.89% | True |

Do not select settings using these held-out results. Repeat on independent
attack and benign model pairs. A high pre ASR does not demonstrate dormant activation.
A lower ASR alone is insufficient if clean accuracy deteriorates.
No unlearning guarantee or historical-information advantage is established here.
