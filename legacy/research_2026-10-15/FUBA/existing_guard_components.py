"""Reuse existing DOSE statistics/check and TRACE solver in the common controller.

This adapts their numerical components; it is not a reproduction of every legacy
script default. Grouped controls, separate views and transactional release belong
to StructuralGuard. Legacy repair/KD is intentionally not called as a certificate.
"""
import os
from types import SimpleNamespace

import numpy as np


def _load_dose():
    cwd = os.getcwd()
    try:
        from dose_fu import Engine, median_probs
    finally:
        os.chdir(cwd)
    return Engine, median_probs


class DoseComparator:
    """Use Engine.stat_of/check/threshold with same-view healthy LOO controls.

    Legacy DOSE joins public+forget views and selects healthy controls on public
    inputs. This wrapper keeps the views separate and selects controls on each
    view, so a small localized signal cannot be diluted by a larger public set.
    """

    def __init__(self, backend, fmt, stat="flow", zmode="std", cap=0.3):
        import torch
        Engine, self.median_probs = _load_dose()
        self.torch, self.cap, self.stat = torch, cap, stat
        n_classes = backend.predict(backend.propose({}), "public").shape[1]
        args = SimpleNamespace(fmt=fmt, seed=522, m=0, n_classes=n_classes,
                               stat=stat, zmode=zmode, cap=cap, tau=3.0, tau_mode="fixed")
        self.engine = Engine(args, backend.factory, backend.flat, backend.contributions,
                             backend.probes["public"].to(backend.device), backend.device)

    def __call__(self, before, candidate, controls, policy):
        torch = self.torch
        base = torch.as_tensor(before, dtype=torch.float32)
        pk = torch.as_tensor(candidate, dtype=torch.float32)
        peers = [torch.as_tensor(p, dtype=torch.float32) for p in controls]
        healthy = [i for i, p in enumerate(peers)
                   if float((p.argmax(1) != base.argmax(1)).float().mean()) <= self.cap]
        if len(healthy) < policy.min_controls:
            return {"risk": False, "inconclusive": True, "method": "dose_fu.Engine.check",
                    "n_healthy": len(healthy)}
        scores = {i: self.engine.stat_of(peers[i], self.median_probs([peers[j] for j in healthy if j != i]))[self.stat]
                  for i in healthy}
        pop = {"pr": self.median_probs([peers[i] for i in healthy]), "s": scores, "healthy": healthy}
        self.engine.a.tau = policy.z_limit
        z, stats = self.engine.check(pk, pop)
        threshold = self.engine.threshold(pop)
        # Explicit minimum effect keeps a zero-spread numerical blip from becoming
        # an alarm. No statistical significance guarantee is claimed.
        excess = float(stats[self.stat] - np.median(list(scores.values())))
        risk = z > threshold and excess > policy.min_effect
        return {"risk": bool(risk), "method": "dose_fu.Engine.check", "stat": self.stat,
                "z": float(z), "threshold": float(threshold), "effect": float(stats[self.stat]),
                "excess": excess, "n_healthy": len(healthy)}


def trace_attributor(backend, anchor_probs, removals, model, policy):
    """TRACE's existing constrained least-squares selector on observed activation.

    No fixed target, attacker count, or role labels. Local ablation vectors remain
    an approximation of joint nonlinear removal; independent recovery checks own
    the final decision. Weights are attribution evidence, not an automatic edit.
    """
    from trace_solver import forward_select
    views = backend.views
    current = {v: backend.predict(model, v) for v in views}
    def flatten(values):
        return np.concatenate([values[v].ravel() / np.sqrt(values[v].size) for v in views])
    activation = flatten({v: current[v] - anchor_probs[v] for v in views})
    units = sorted(set(backend.units) - set(removals))
    if not units or float(activation @ activation) <= 1e-12:
        return []
    effects = []
    for unit in units:
        ablated = backend.propose({**removals, unit: 1.0})
        effects.append(flatten({v: backend.predict(ablated, v) - current[v] for v in views}))
    effects = np.stack(effects)
    _, selected, weights, history = forward_select(
        effects @ effects.T, effects @ activation, float(activation @ activation),
        list(range(len(units))), len(units), 1.0, policy.attribution_gain)
    result, previous = [], 1.0
    for index, remaining in zip(selected, history):
        result.append({"unit": units[index], "weight": weights[index],
                       "gain": float(previous - remaining), "method": "cancel_fu.forward_select"})
        previous = remaining
    return result
