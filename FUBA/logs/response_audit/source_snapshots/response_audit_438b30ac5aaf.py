"""Experimental deletion-response audit, without a cancellation assumption.

Derive a soft version of DOSE's cross-class flow, compare with same-dose peer
deletions, choose a target on discovery inputs, then confirm that target on held-
out inputs. An anomaly is NOT proof of malicious intent or correct unlearning.
"""
import math
import numpy as np
from structural_guard import probabilities


def median_reference(predictions):
    if not predictions:
        raise ValueError("reference needs peer predictions")
    p = np.stack([probabilities(x) for x in predictions])
    med = np.median(p, axis=0)
    # Coordinatewise medians of simplex points can have total mass zero.
    zero = med.sum(axis=1) <= 1e-12
    med[zero] = p.mean(axis=0)[zero]
    return med / med.sum(axis=1, keepdims=True)


def cross_class_scores(candidate, reference, mode="soft"):
    """Return a score for EACH target; do not choose the target in this function.

    Both variants weight incoming mass by source-class entropy. Soft uses positive
    probability gains; hard uses argmax flips (the original DOSE observation).
    Scores suppress one-source changes, including legitimate class specialization,
    but cannot distinguish all honest multi-source shifts from backdoor activation.
    """
    p, ref = probabilities(candidate), probabilities(reference)
    if p.shape != ref.shape:
        raise ValueError("candidate and reference shapes differ")
    n, c = p.shape
    source = ref.argmax(1)
    flow = np.zeros((c, c))
    if mode == "soft":
        gain = np.maximum(p-ref, 0)
        for a in range(c):
            flow[a] = gain[source == a].sum(axis=0) / n
    elif mode == "hard":
        dest = p.argmax(1)
        flow = np.bincount(source*c + dest, minlength=c*c).reshape(c, c) / n
    else:
        raise ValueError("mode must be hard or soft")
    np.fill_diagonal(flow, 0)
    mass = flow.sum(axis=0)
    mix = np.divide(flow, mass[None, :], out=np.zeros_like(flow), where=mass[None, :] > 0)
    entropy = -(mix * np.log(np.maximum(mix, 1e-300))).sum(axis=0) / math.log(max(c-1, 2))
    return mass * np.clip(entropy, 0, 1)


class ResponseComparator:
    """StructuralGuard comparator. Frozen defaults; no fitted attack labels.

    Reference controls are selected on discovery inputs only. Target selection,
    normalization, and confirmation use the same operation for both ablation arms.
    MAD z is a heuristic, not a Gaussian p-value. Paired audit is not required.
    """
    def __init__(self, mode="soft", seed=20261014, min_inputs=32, cap=.3):
        if mode not in ("soft", "hard") or min_inputs < 4 or not 0 < cap <= 1:
            raise ValueError("invalid response comparator configuration")
        self.mode, self.seed, self.min_inputs, self.cap = mode, seed, min_inputs, cap

    def __call__(self, before, candidate, controls, policy):
        base, pk = probabilities(before), probabilities(candidate)
        peers = [probabilities(p) for p in controls]
        if base.shape != pk.shape or any(p.shape != base.shape for p in peers):
            raise ValueError("all models must predict the same inputs/classes")
        output = {"method": "heldout_response_v1", "mode": self.mode,
                  "risk": False, "inconclusive": True, "seed": self.seed,
                  "thresholds": {"z": policy.z_limit, "effect": policy.min_effect,
                                 "scale_floor": policy.scale_floor},
                  "interpretation": "deletion response anomaly; intent/forgetting not certified"}
        if len(base) < self.min_inputs:
            return {**output, "reason": "insufficient_inputs"}
        idx = np.random.default_rng(self.seed).permutation(len(base))
        discovery, confirm = idx[:len(idx)//2], idx[len(idx)//2:]
        healthy = [i for i, peer in enumerate(peers)
                   if np.mean(peer[discovery].argmax(1) != base[discovery].argmax(1)) <= self.cap]
        if len(healthy) < policy.min_controls:
            return {**output, "reason": "insufficient_comparable_controls", "n_controls": len(healthy)}
        ref = median_reference([peers[i] for i in healthy])
        peer_refs = {i: median_reference([peers[j] for j in healthy if j != i]) for i in healthy}
        def evaluate(part):
            score = cross_class_scores(pk[part], ref[part], self.mode)
            null = np.stack([cross_class_scores(peers[i][part], peer_refs[i][part], self.mode) for i in healthy])
            median = np.median(null, axis=0)
            scale = np.maximum(1.4826*np.median(abs(null-median), axis=0), policy.scale_floor)
            return score, median, scale, (score-median)/scale
        d_score, d_med, d_scale, d_z = evaluate(discovery)
        target = int(d_z.argmax())
        h_score, h_med, h_scale, h_z = evaluate(confirm)
        excess = float(h_score[target]-h_med[target])
        discovery_pass = d_z[target] > policy.z_limit and d_score[target]-d_med[target] > policy.min_effect
        confirmation_pass = h_z[target] > policy.z_limit and excess > policy.min_effect
        output.update(inconclusive=False, reason="heuristic_uncalibrated", target=target,
                      n_controls=len(healthy), n_discovery=len(discovery), n_confirmation=len(confirm),
                      discovery={"score": float(d_score[target]), "median": float(d_med[target]),
                                 "scale": float(d_scale[target]), "z": float(d_z[target]),
                                 "passes": bool(discovery_pass)},
                      confirmation={"score": float(h_score[target]), "median": float(h_med[target]),
                                    "scale": float(h_scale[target]), "z": float(h_z[target]),
                                    "excess": excess, "passes": bool(confirmation_pass)},
                      risk=bool(discovery_pass and confirmation_pass))
        return output
