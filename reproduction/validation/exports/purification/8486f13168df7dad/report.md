**PIPELINE CHECK ONLY: 2-round source; reduced evaluation budget, not defense-performance evidence.**

# Purification comparison

Actual model-pair provenance is supplied by the experiment owner.
These are matched-objective ablations, not official NC/I-BAU implementations.

| Arm | Clean accuracy | ASR (non-target originals) | Triggered original accuracy | Clean tolerance |
| --- | ---: | ---: | ---: | --- |
| pre | 10.00% | 5.56% | 11.11% | reference |
| post | 5.00% | 0.00% | 5.56% | reference |
| bd5 | 15.00% | 0.00% | 16.67% | reference |
| clean_ft | 10.00% | 0.00% | 11.11% | True |
| post_only | 10.00% | 0.00% | 0.00% | True |
| differential | 10.00% | 0.00% | 11.11% | True |
| differential_cvar | 10.00% | 0.00% | 11.11% | True |
| differential_cvar_constrained | 5.00% | 0.00% | 5.56% | True |

Do not select settings using these held-out results. Repeat on independent
attack and benign model pairs. A high pre ASR does not demonstrate dormant activation.
A lower ASR alone is insufficient if clean accuracy deteriorates.
No unlearning guarantee or historical-information advantage is established here.
