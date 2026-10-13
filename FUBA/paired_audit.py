"""Experimental request-conditioned joint-removal confirmation.

TRACE discovers partners on one input split. Four actual model interventions are
then compared on another split. This is structural evidence, NOT proof of an
attack, deletion correctness, or a permission to exclude the selected units.
"""
from dataclasses import dataclass, asdict
import numpy as np

from structural_guard import probabilities, matched_controls, Policy
from trace_solver import forward_select


@dataclass(frozen=True)
class PairedPolicy:
    seed: int = 20261013
    max_partners: int = 5
    min_gain: float = 0.01
    z_limit: float = 3.0
    min_cancel: float = 0.5
    min_restore: float = 0.5
    energy_floor: float = 1e-6
    public_tv_cap: float = 0.3
    log_clip: float = 1e-7
    min_inputs: int = 16

    def __post_init__(self):
        if (self.max_partners < 1 or self.min_inputs < 4 or
                not 0 < self.log_clip < 0.1 or not 0 < self.min_gain < 1 or
                not 0 < self.public_tv_cap <= 1 or self.energy_floor <= 0 or
                self.z_limit <= 0 or not 0 <= self.min_cancel <= 1 or
                not 0 <= self.min_restore <= 1):
            raise ValueError("invalid paired audit policy")


def centered_log(p, eps=1e-7):
    p = probabilities(p)
    logp = np.log(np.maximum(p, eps))
    return logp - logp.mean(axis=1, keepdims=True)


def cancellation_metrics(ds, da, dsa, eps=1e-12):
    """Finite intervention energy and actual joint residual; no linearity assumed."""
    ds, da, dsa = (np.asarray(x, dtype=float) for x in (ds, da, dsa))
    if ds.shape != da.shape or ds.shape != dsa.shape or not ds.size:
        raise ValueError("effects must be nonempty and have equal shapes")
    if not all(np.isfinite(x).all() for x in (ds, da, dsa)):
        raise ValueError("effects must be finite")
    es, ea, ej = [float(np.mean(x * x)) for x in (ds, da, dsa)]
    linear = ds + da
    remainder = dsa - linear
    return {"request_energy": es, "partner_energy": ea, "joint_energy": ej,
            "cancellation": (es + ea - ej) / max(es + ea, eps),
            "restoration": 1 - ej / max(es, eps),
            "linear_restoration": 1 - float(np.mean(linear ** 2)) / max(es, eps),
            "nonlinearity_energy": float(np.mean(remainder ** 2)),
            "opposition": -2 * float(np.mean(ds * da)) / max(es + ea, eps)}


def rank_calibrate(score, benign_episode_scores, alpha=0.05):
    """Conditional contract: FIXED pipeline and exchangeable benign episodes.

    Input rows, peer clients, and repeat probes of one trajectory are NOT such a
    calibration sample. This function cannot check the scientific assumptions.
    Conservative ties; marginal single-episode bound only, no adaptive guarantee.
    """
    scores = np.asarray(benign_episode_scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not np.isfinite(score):
        raise ValueError("finite scalar episode scores required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    p = float((1 + np.count_nonzero(scores >= score)) / (len(scores) + 1))
    sufficient = 1 / (len(scores) + 1) <= alpha
    return {"p": p, "n_benign_episodes": len(scores), "alpha": alpha,
            "status": "calibrated" if sufficient else "insufficient_resolution",
            "reject": bool(sufficient and p <= alpha),
            "assumption": "fixed pipeline; exchangeable benign episodes; marginal single test"}


def paired_audit(backend, removals, policy=None, views=None):
    """Audit a cumulative request; never modifies the backend or serving model.

    The backend must implement exactly the SAME intervention operator for anchor,
    request, partner, joint, and controls. Units are disjoint deletion objects.
    Partner doses are fitted in [0,1], then frozen before confirmation.
    """
    policy = policy or PairedPolicy()
    views = tuple(backend.views if views is None else views)
    if not views or len(set(views)) != len(views) or set(views) - set(backend.views):
        raise ValueError("nonempty available views required")
    if (not removals or set(removals) - set(backend.units) or
            any(not np.isfinite(x) or not 0 < x <= 1 for x in removals.values())):
        raise ValueError("known units and removal fractions in (0,1] required")
    candidates = sorted(set(backend.units) - set(removals))
    result = {"method": "paired_joint_removal_v1", "request": dict(removals),
              "policy": asdict(policy), "views_used": list(views),
              "status": "inconclusive", "partners": {}, "views": {},
              "score": 0.0, "release_authorized": False,
              "interpretation": "cancellation evidence only; honest opposition can be identical"}
    if len(candidates) < 3:
        result["reason"] = "insufficient_peer_units"
        return result
    cache = {}
    def predictions(removal):
        key = tuple(sorted(removal.items()))
        if key not in cache:
            model = backend.propose(removal)
            cache[key] = {v: probabilities(backend.predict(model, v)) for v in views}
        return cache[key]
    anchor = predictions({})
    if any(len(anchor[v]) < policy.min_inputs for v in views):
        result["reason"] = "insufficient_inputs_for_discovery_confirmation"
        return result
    splits = {}
    for v in views:
        idx = np.random.default_rng(policy.seed).permutation(len(anchor[v]))
        splits[v] = (idx[:len(idx)//2], idx[len(idx)//2:])
    base = {v: centered_log(anchor[v], policy.log_clip) for v in views}
    def effects(removal):
        return {v: centered_log(p, policy.log_clip) - base[v]
                for v, p in predictions(removal).items()}
    def discovery(e):
        return np.concatenate([e[v][splits[v][0]].ravel() /
                               np.sqrt(e[v][splits[v][0]].size * len(views)) for v in views])
    ds = effects(removals)
    rho = max(removals.values())
    peer = {u: effects({u: rho}) for u in candidates}
    d = discovery(ds)
    cols = np.stack([discovery(peer[u]) for u in candidates])
    gain, selected, weights, history = forward_select(
        cols @ cols.T, cols @ d, float(d @ d), list(range(len(candidates))),
        policy.max_partners, 1 / rho, policy.min_gain)
    partners = {candidates[j]: float(np.clip(rho * weights[j], 0, 1))
                for j in selected if weights[j] > 0}
    result.update(partners=partners, discovery_linear_restoration=gain,
                  discovery_history=history)
    da = effects(partners)
    joint = {**removals, **partners}
    dsa = effects(joint)
    controls = matched_controls(backend.units, removals, Policy(seed=policy.seed))
    if len(controls) < 3:
        result["reason"] = "insufficient_matched_controls"
        return result
    controls_effects = [effects(c) for c in controls]
    for v in views:
        idx = splits[v][1]
        metrics = cancellation_metrics(ds[v][idx], da[v][idx], dsa[v][idx])
        energies = np.array([np.mean(e[v][idx] ** 2) for e in controls_effects])
        med = float(np.median(energies))
        scale = max(1.4826 * float(np.median(abs(energies - med))), policy.energy_floor)
        z = (metrics["request_energy"] - med) / scale
        tv = lambda r: float(np.abs(predictions(r)[v][idx] - anchor[v][idx]).sum(axis=1).mean() / 2)
        metrics.update(energy_z=float(z), control_median=med, control_mad_scale=scale,
                       n_controls=len(controls), n_discovery=len(splits[v][0]), n_confirmation=len(idx),
                       request_tv=tv(removals), joint_tv=tv(joint),
                       clipped_prob_fraction=float((predictions(removals)[v][idx] < policy.log_clip).mean()))
        metrics["structure_supported"] = bool(partners and metrics["request_energy"] > policy.energy_floor
            and metrics["cancellation"] >= policy.min_cancel and metrics["restoration"] >= policy.min_restore)
        metrics["review_priority"] = bool(metrics["structure_supported"] and z >= policy.z_limit)
        metrics["score"] = float(max(0, z) * np.clip(metrics["cancellation"], 0, 1)
                                 * np.clip(metrics["restoration"], 0, 1))
        result["views"][v] = metrics
    result["score"] = max(m["score"] for m in result["views"].values())
    collapsed = "public" in views and max(result["views"]["public"]["request_tv"],
                                         result["views"]["public"]["joint_tv"]) > policy.public_tv_cap
    result["status"] = ("inconclusive" if collapsed else "review_priority"
                        if any(m["review_priority"] for m in result["views"].values()) else "unconfirmed")
    result["reason"] = "public_drift_exceeds_cap" if collapsed else "heuristic_uncalibrated"
    result["n_interventions"] = len(cache)
    return result
