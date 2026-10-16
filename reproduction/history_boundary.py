"""Conditional history-boundary evidence; NOT a fitted/validated detector.

Inputs are measured functional responses in an agreed coordinate system.
Normal explanations are fitted to CLEAN coordinates only. All explanations
consistent with those coordinates are retained, rather than one ridge fit.
The caller must supply trusted reference directions, uncertainty bounds,
deletion-aligned atoms, held-out probes and independent calibration runs.
No checkpoints, labels, probe search, FU calls or calibration are hidden here.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class ResidualInterval:
    status: str
    lower: np.ndarray | None
    upper: np.ndarray | None
    clean_rank: int
    coefficients: int


def _array(value, name):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def residual_interval(clean, clean_basis, probe, probe_basis, *,
                      clean_error, coefficient_lower, coefficient_upper,
                      probe_error=0.):
    """Enclose probe - probe_basis @ beta over all clean-feasible beta.

    Shapes: clean[P], clean_basis[P,C], probe[F], probe_basis[F,C].
    Clean feasibility: abs(clean_basis @ beta - clean) <= clean_error.
    Bounds and errors must be fixed from clean/reference information, never
    selected to increase the tested probe score. probe_error additionally
    covers bounded observation/linearization error in probe coordinates.
    Infeasible clean explanations return unknown, not a malicious score.
    Floating-point LP results are numerical enclosures, not interval-arithmetic
    certificates; applications need a separately justified numerical allowance.
    """
    x, b = _array(clean, "clean"), _array(clean_basis, "clean_basis")
    y, a = _array(probe, "probe"), _array(probe_basis, "probe_basis")
    if (x.ndim != 1 or b.ndim != 2 or y.ndim != 1 or a.ndim != 2
            or min(x.size, y.size, b.shape[1]) < 1
            or b.shape[0] != x.size or a.shape != (y.size, b.shape[1])):
        raise ValueError("expected clean[P], B[P,C], probe[F], A[F,C]")
    lo = _array(coefficient_lower, "coefficient_lower")
    hi = _array(coefficient_upper, "coefficient_upper")
    if lo.shape != (b.shape[1],) or hi.shape != lo.shape or np.any(lo > hi):
        raise ValueError("finite ordered coefficient bounds[C] required")
    eps = np.broadcast_to(_array(clean_error, "clean_error"), x.shape)
    nu = np.broadcast_to(_array(probe_error, "probe_error"), y.shape)
    if np.any(eps < 0) or np.any(nu < 0):
        raise ValueError("error allowances must be nonnegative")
    matrix = np.vstack((b, -b))
    rhs = np.concatenate((x + eps, -x + eps))
    bounds = list(zip(lo, hi))

    def solve(cost):
        return linprog(cost, A_ub=matrix, b_ub=rhs, bounds=bounds, method="highs")

    fit = solve(np.zeros(b.shape[1]))
    rank = int(np.linalg.matrix_rank(b))
    if fit.status == 2:
        return ResidualInterval("unexplained_clean_response", None, None, rank, b.shape[1])
    if not fit.success:
        raise RuntimeError(f"clean feasibility solver failed: {fit.message}")
    lower, upper = [], []
    for row, value, allowance in zip(a, y, nu):
        mn, neg_mx = solve(row), solve(-row)
        if not mn.success or not neg_mx.success:
            raise RuntimeError("residual interval optimization failed")
        lower.append(value + neg_mx.fun - allowance)
        upper.append(value - mn.fun + allowance)
    return ResidualInterval("measured", np.asarray(lower), np.asarray(upper), rank, b.shape[1])


def boundary_evidence(lower, upper, *, deleted, retained, blocks):
    """Robust opposing residuals on frozen held-out blocks and coordinates.

    lower/upper[T,A,F] already include actual FL aggregation weights.
    deleted/retained[A] are boolean masks of REAL recorded atoms. Masks may
    restrict the context to the requested owner's internal atoms. Nonparticipants
    must be explicit measured zeros; missing atoms must not be supplied as zeros.
    Each block is a disjoint half-open round interval. F contains the frozen
    source-class/target/probe coordinates; all must pass, so the scalar is the
    minimum across blocks and F. This intentionally misses intermittent or
    source-specific effects outside the declared coverage. Score is NOT ASR,
    probability of an attack, exact FU influence, or proof of malicious intent.
    """
    l, u = _array(lower, "lower"), _array(upper, "upper")
    if l.ndim != 3 or l.shape != u.shape or 0 in l.shape or np.any(l > u + 1e-9):
        raise ValueError("ordered nonempty intervals[T,A,F] required")
    d, r = np.asarray(deleted), np.asarray(retained)
    if (d.dtype.kind != "b" or r.dtype.kind != "b" or d.shape != (l.shape[1],)
            or r.shape != d.shape or not d.any() or not r.any() or np.any(d & r)):
        raise ValueError("nonempty disjoint boolean deletion and retention masks required")
    used, rows = set(), []
    if not blocks:
        raise ValueError("held-out blocks required")
    for start, stop in blocks:
        if (not isinstance(start, int) or not isinstance(stop, int)
                or not 0 <= start < stop <= l.shape[0]):
            raise ValueError("valid half-open integer blocks required")
        indices = set(range(start, stop))
        if used & indices:
            raise ValueError("validation blocks must not overlap")
        used |= indices
        # Average over physical rounds, not only convenient active rounds.
        retained_lower = l[start:stop, r].sum(axis=(0, 1)) / (stop - start)
        deleted_upper = u[start:stop, d].sum(axis=(0, 1)) / (stop - start)
        margin = np.minimum(retained_lower, -deleted_upper)
        rows.append(dict(start=start, stop=stop,
                         retained_lower=retained_lower.tolist(),
                         deleted_upper=deleted_upper.tolist(),
                         margin=margin.tolist()))
    score = float(max(0., min(min(row["margin"]) for row in rows)))
    return dict(score=score, blocks=rows,
                status="conditional_opposition" if score > 0 else "no_robust_opposition",
                calibrated=False, actual_fu_verified=False)
