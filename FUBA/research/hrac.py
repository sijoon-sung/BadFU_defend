"""History-request attribution checks. No training or unlearned model is built.

Inter: the existing signed selective-cancellation statistic, over fixed windows.
Intra: a convex attribution-feasibility GLRT, with an LP optimality certificate.
Calibration: clean episode ranks per declared stratum, Bonferroni across channels.
The intra observation must be measured independently of the historical atoms.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
import math
import numpy as np
from scipy.optimize import linprog, minimize
from FUBA.research.tension_audit import boundary_profile


CHANNELS = ("inter", "intra")


def inter_profile(history, removal, windows=(0, 10, 20)):
    """history[round, deletion unit, coordinate], already aggregation-weighted.

    Windows are fixed before evaluation; 0 means the entire available prefix.
    This measures observed signed effects, not exact causal FU contributions.
    """
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or 0 in h.shape or not np.isfinite(h).all():
        raise ValueError("finite nonempty round/unit/coordinate history required")
    if not windows or any(not isinstance(w, int) or w < 0 for w in windows):
        raise ValueError("fixed nonnegative integer windows required")
    profiles = []
    for width in dict.fromkeys(windows):
        effects = h[-width:].sum(0) if width else h.sum(0)
        p = boundary_profile(effects, removal)
        profiles.append(p["score"])
    return np.max(profiles, axis=0)


def attribution_residual(atoms, observed, *, constraint_matrix=None,
                         constraint_value=None, lower=None, upper=None,
                         covariance=None, ridge=1e-3, tolerance=1e-6):
    """Minimize ||W^(1/2)(u - A q)||^2, Bq=b, lower<=q<=upper.

    atoms[atom, coordinate] are recorded functional contributions; q is a
    feasible deletion fraction, NOT a freely signed regression coefficient.
    Bq=b can encode deleted sample mass / class counts. Exact IDs use fixed
    bounds. A convex mixture represents fractional atom deletion only under
    the declared attribution approximation, not arbitrary within-shard data.

    Gaussian noise with covariance proportional to W^-1 motivates the GLRT.
    Calibration does not require Gaussian noise. The returned score is a
    certified LOWER bound, so numerical error cannot inflate this statistic.
    """
    a, u = np.asarray(atoms, float), np.asarray(observed, float)
    if a.ndim != 2 or 0 in a.shape or u.shape != (a.shape[1],):
        raise ValueError("atoms[atom, coordinate] and observed[coordinate] required")
    if not np.isfinite(a).all() or not np.isfinite(u).all():
        raise ValueError("finite atoms and observation required")
    if not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite ridge and tolerance required")
    n, d = a.shape
    lo = np.zeros(n) if lower is None else np.asarray(lower, float)
    hi = np.ones(n) if upper is None else np.asarray(upper, float)
    if lo.shape != (n,) or hi.shape != (n,) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("one finite pair of bounds per atom required")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi):
        raise ValueError("deletion fractions must satisfy 0 <= lower <= upper <= 1")
    if (constraint_matrix is None) != (constraint_value is None):
        raise ValueError("both equality matrix and value required")
    bmat = None if constraint_matrix is None else np.asarray(constraint_matrix, float)
    bval = None if constraint_value is None else np.asarray(constraint_value, float)
    if bmat is not None:
        if bmat.ndim != 2 or bmat.shape[1] != n or bval.shape != (bmat.shape[0],):
            raise ValueError("invalid equality dimensions")
        if not np.isfinite(bmat).all() or not np.isfinite(bval).all():
            raise ValueError("finite equality constraints required")
    cov = a.T @ a / n if covariance is None else np.asarray(covariance, float)
    if cov.shape != (d, d) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T):
        raise ValueError("finite symmetric covariance required")
    eigenvalues, basis = np.linalg.eigh((cov + cov.T) / 2)
    if eigenvalues.min() < -1e-10 * max(1., np.abs(eigenvalues).max()):
        raise ValueError("covariance must be positive semidefinite")
    scale = max(float(np.trace(cov) / d), np.finfo(float).eps)
    whiten = (basis / np.sqrt(np.maximum(eigenvalues, 0) + ridge * scale)).T
    design, target = whiten @ a.T, whiten @ u
    # Scale the optimizer objective only; report the original dimensionless score.
    normalization = max(1., float(target @ target), float(np.linalg.norm(design, ord='fro')**2))
    bounds = list(zip(lo, hi))
    feasible = linprog(np.zeros(n), A_eq=bmat, b_eq=bval, bounds=bounds, method="highs")
    if not feasible.success:
        raise ValueError("infeasible deletion metadata; not a spectral attack verdict")

    def objective(q):
        residual = design @ q - target
        return float(residual @ residual) / normalization

    def gradient(q):
        return 2 * design.T @ (design @ q - target) / normalization

    constraints = [] if bmat is None else [{"type": "eq", "fun": lambda q: bmat @ q - bval,
                                          "jac": lambda q: bmat}]
    fixed = np.all(lo == hi)
    solution = None if fixed else minimize(objective, feasible.x, jac=gradient, bounds=bounds,
        constraints=constraints, method="SLSQP", options={"ftol": 1e-14, "maxiter": 2000})
    q = lo.copy() if fixed else solution.x
    if (np.any(q < lo - tolerance) or np.any(q > hi + tolerance)
            or (bmat is not None and np.max(np.abs(bmat @ q - bval), initial=0) > tolerance)):
        raise RuntimeError("attribution optimizer did not return a feasible solution")
    # SLSQP stops on objective change, which alone is not an optimality test.
    # Polish with feasible Frank-Wolfe directions and exact quadratic line search.
    # The LP uses the ORIGINAL gradient scale, avoiding small-gradient HiGHS
    # tolerance artefacts after the optimizer's objective normalization.
    for iteration in range(201):
        grad = gradient(q) * normalization
        certificate = linprog(grad, A_eq=bmat, b_eq=bval, bounds=bounds, method="highs",
            options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9})
        if not certificate.success:
            raise RuntimeError("could not certify attribution optimum")
        gap = max(0., float(grad @ q - certificate.fun))
        upper_value = objective(q) * normalization
        if gap <= tolerance * (1 + upper_value):
            break
        direction = certificate.x - q
        curvature = float(np.linalg.norm(design @ direction)**2)
        step = min(1., gap / (2*curvature)) if curvature > 0 else 1.
        q = q + step * direction
    else:
        raise RuntimeError(f"attribution optimum uncertified: gap={gap:g}")
    return {"score": max(0., upper_value - gap), "upper_bound": upper_value,
            "duality_gap": gap, "coefficients": q.tolist(),
            "signed_residual": (u - a.T @ q).tolist(), "ridge": ridge,
            "status": "certified_optimum"}


def conformal_channel(calibration, value, alpha):
    """Conservative upper-tail rank, one value per independent clean episode."""
    cal = np.asarray(calibration, float)
    if cal.ndim != 1 or not np.isfinite(cal).all() or np.any(cal < 0) or not 0 < alpha < 1:
        raise ValueError("finite nonnegative calibration scores and valid alpha required")
    rank = math.ceil((len(cal) + 1) * (1 - alpha) - 1e-12)
    threshold = float(np.sort(cal)[rank - 1]) if rank <= len(cal) else None
    if value is None:
        return {"n": len(cal), "p": None, "threshold": threshold, "alarm": None,
                "status": "unobserved_channel"}
    if not np.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative test score required")
    p = float((1 + np.count_nonzero(cal >= value)) / (len(cal) + 1))
    return {"n": len(cal), "p": p, "threshold": threshold,
            "alarm": bool(p <= alpha) if threshold is not None else None,
            "status": "calibrated" if threshold is not None else "insufficient_clean_episodes"}


def decide(calibration, scores, alpha=.05, channels=CHANNELS):
    """OR test with fixed-channel Bonferroni; missing channels never imply clean.

    Maximize within an episode over all declared requests, coordinates, windows
    and inspection times BEFORE calibration. No independence of channels needed.
    """
    channels = tuple(channels)
    if not channels or len(set(channels)) != len(channels) or set(channels) - set(CHANNELS):
        raise ValueError("fixed, unique supported channel family required")
    if not 0 < alpha < 1:
        raise ValueError("0 < alpha < 1 required")
    rows = {c: conformal_channel(calibration.get(c, []), scores.get(c), alpha / len(channels))
            for c in channels}
    positives = [r["alarm"] is True for r in rows.values()]
    complete = all(r["status"] == "calibrated" for r in rows.values())
    alarm = True if any(positives) else (False if complete else None)
    ps = [r["p"] for r in rows.values() if r["p"] is not None]
    return {"alpha": alpha, "channels": rows, "alarm": alarm,
            "joint_p": min(1., len(channels) * min(ps)) if complete else None,
            "status": "alarm" if alarm is True else ("no_alarm" if complete else "incomplete"),
            "guarantee": "marginal clean-episode FWER <= alpha only under within-stratum exchangeability and frozen scoring"}
