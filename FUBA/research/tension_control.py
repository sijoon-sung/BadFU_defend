"""Deletion-conditioned control of an ACTUAL FU proposal (research prototype).

No attack roles, target labels, deletion simulations, or deleted-data teacher.
Selection is an uncalibrated ranking, not an intent detector. The solver provides
local surrogate control, NOT certified forgetting or a bound on arbitrary ASR.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize

from FUBA.research.tension_audit import boundary_profile


def select_coordinates(effects, removal, budget=8, mode="history", seed=0, response=None):
    """effects[deletion unit, coordinate]; fixed coordinates before averaging.

    A client-aggregated ledger cannot resolve a within-client sample request.
    Callers must supply the requested granularity, not a sample-count proxy.
    Positive top-k ranking is deliberately NOT called a calibrated alarm.
    """
    e = np.asarray(effects, dtype=float)
    if e.ndim != 2 or not isinstance(budget, int) or budget < 1:
        raise ValueError("2D ledger and positive integer budget required")
    action_gate = mode.endswith("_action")
    mode = mode.removesuffix("_action")
    if mode == "shuffled":
        e = e[np.random.default_rng(seed).permutation(len(e))]
    p = boundary_profile(e, removal)
    if mode in ("history", "shuffled"):
        score = p["score"]
    elif mode == "linear":
        score = np.maximum(-np.asarray(removal) @ e, 0)
    elif mode == "tension":
        score = p["tension"]
    else:
        raise ValueError(f"unknown selection mode: {mode}")
    if action_gate:
        rise = np.asarray(response, dtype=float)
        if rise.shape != score.shape or not np.isfinite(rise).all():
            raise ValueError("actual proposal response must match ledger coordinates")
        # One real proposal already computed by FU; not a family of simulated
        # unlearned models. This selects relevant directions, not attacker intent.
        score = score*np.maximum(rise, 0)
    order = np.argsort(-score, kind="stable")
    chosen = order[score[order] > 0][:budget]
    return chosen, score


def control_step(proposal, risk, retain, *, strength=1., kappa=0., epsilon=0.,
                 radius=None, tolerance=1e-8, maxiter=500):
    """min .5||d-v||^2 + strength sum([Gd-kappa]+^2)
    subject to Hd <= epsilon, ||d|| <= radius.

    Solves in span(v, rows(G), rows(H)), using SVD and SLSQP. This is an
    exact subspace reduction up to numerical rank tolerance, not a d-by-d
    Hessian. A failed solve raises; zero is never silently called success.
    All directions and budgets must use the SAME coordinate normalization.
    """
    v = np.asarray(proposal, dtype=np.float64)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
        raise ValueError("finite nonempty proposal vector required")
    g, h = (np.asarray(x, dtype=np.float64) for x in (risk, retain))
    if any(x.ndim != 2 or x.shape[1] != len(v) or not np.isfinite(x).all() for x in (g, h)):
        raise ValueError("gradient matrices must have shape [coordinates, parameters]")
    k = np.broadcast_to(np.asarray(kappa, dtype=float), (len(g),)).copy()
    eps = np.broadcast_to(np.asarray(epsilon, dtype=float), (len(h),)).copy()
    rho = float(np.linalg.norm(v) if radius is None else radius)
    if (not np.isfinite([strength, rho, tolerance]).all() or strength < 0 or rho < 0
            or tolerance <= 0 or maxiter < 1 or not np.isfinite(k).all()
            or not np.isfinite(eps).all() or (k < 0).any() or (eps < 0).any()):
        raise ValueError("finite nonnegative budgets/strength and positive tolerance required")
    # Normalize rows ONLY for span construction; objective uses original units.
    rows = np.vstack([v[None], g, h])
    norms = np.linalg.norm(rows, axis=1)
    rows = rows[norms > 0] / norms[norms > 0, None]
    if len(rows) and rho > 0:
        _, singular, basis = np.linalg.svd(rows, full_matrices=False)
        basis = basis[singular > singular[0]*1e-12]
        # Dimensionless optimization variable z; d = rho B^T z.
        gv, hv, vv = g @ basis.T, h @ basis.T, basis @ v / rho
        kk, ee = k/rho, eps/rho

        def fun(z):
            hinge = np.maximum(gv @ z-kk, 0)
            return .5*np.sum((z-vv)**2)+strength*np.sum(hinge**2)

        def jac(z):
            return z-vv+2*strength*(gv.T @ np.maximum(gv @ z-kk, 0))

        constraints = [{"type": "ineq", "fun": lambda z: 1-z@z,
                        "jac": lambda z: -2*z}]
        if len(h):
            constraints.append({"type": "ineq", "fun": lambda z: ee-hv@z,
                                "jac": lambda z: -hv})
        res = minimize(fun, np.zeros(len(basis)), jac=jac, method="SLSQP",
                       constraints=constraints, options={"ftol": tolerance, "maxiter": maxiter})
        if not res.success or not np.isfinite(res.x).all():
            raise RuntimeError(f"controller solver failed: {res.message}")
        d, rank, nit = rho*(res.x @ basis), len(basis), int(res.nit)
    else:
        d, rank, nit = np.zeros_like(v), 0, 0
    violation = max(0., float(np.max(h@d-eps, initial=0)), float(np.linalg.norm(d)-rho))
    if violation > 1e-6*max(1., rho):
        raise RuntimeError(f"infeasible controller result: {violation}")
    objective = .5*np.sum((d-v)**2)+strength*np.sum(np.maximum(g@d-k, 0)**2)
    if objective > .5*(v@v)+1e-6*max(1., v@v):
        raise RuntimeError("solver result worse than feasible zero step")
    progress = float(d@v / (v@v)) if v@v > 0 else 0.
    return d, {"status": "stalled" if progress <= 1e-6 else "local_step_only",
               "forgetting_verified": False, "rank": rank, "iterations": nit,
               "proposal_norm": float(np.linalg.norm(v)), "step_norm": float(np.linalg.norm(d)),
               "progress": progress, "objective": float(objective), "max_constraint_violation": violation,
               "risk_before": (g@v).tolist(), "risk_after": (g@d).tolist(),
               "retain_linear_change": (h@d).tolist()}


def paired_logits(model, clean, transformed):
    """One fixed paired coordinate per input/class; no mean across inputs."""
    base, aug = model(clean), model(transformed)
    delta = aug-base
    return delta-delta.mean(dim=-1, keepdim=True)


class ParameterSpace:
    """Small PyTorch bridge. Prototype rejects buffers (e.g. BN) explicitly.

    Ignoring BN changes would make Gd and the actual step inconsistent. A future
    state-aware adapter must specify their recalibration/aggregation policy.
    """
    def __init__(self, model):
        import torch
        if list(model.buffers()):
            raise NotImplementedError("buffer-bearing models need a state-aware FU adapter")
        self.model = model
        self.params = list(model.parameters())
        if not self.params or any(not p.requires_grad for p in self.params):
            raise ValueError("trainable parameters required")
        self.size = sum(p.numel() for p in self.params)
        self.torch = torch

    def vector(self):
        return np.concatenate([p.detach().cpu().double().numpy().ravel() for p in self.params])

    def assign(self, vector):
        v = np.asarray(vector)
        if v.shape != (self.size,) or not np.isfinite(v).all():
            raise ValueError("invalid parameter vector")
        with self.torch.no_grad():
            offset = 0
            for p in self.params:
                p.copy_(self.torch.as_tensor(v[offset:offset+p.numel()].reshape(p.shape),
                                            device=p.device, dtype=p.dtype))
                offset += p.numel()

    def gradient(self, scalar):
        grads = self.torch.autograd.grad(scalar, self.params, retain_graph=True, allow_unused=True)
        return np.concatenate([(self.torch.zeros_like(p) if g is None else g).detach().cpu().double().numpy().ravel()
                               for p, g in zip(self.params, grads)])
