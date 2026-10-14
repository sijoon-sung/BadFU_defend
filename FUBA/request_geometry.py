"""Request-conditioned temporal cancellation geometry, without probe/top-k rules.

This is a geometric risk surrogate, NOT an attacker classifier or ASR certificate.
The PSD projection has a variational definition. Time smoothing is a quadratic
roughness model, not the previously unsuccessful lag-correlation detector.
"""
import numpy as np
from scipy.optimize import minimize, nnls


class ControlInfeasible(RuntimeError):
    """The requested progress/retention budgets were not achieved."""


def request_paths(events, deleted):
    """Aggregate ACTUAL weighted updates into deletion/retention time series.

    events use federaser_clientwise.replay's {updates, weights?} schema.
    A missing participant contributes zero to the actual aggregate, not an
    observed zero local gradient. Full deletion units only; never sample counts.
    """
    if not events:
        raise ValueError("nonempty event history required")
    known = set().union(*(set(e["updates"]) for e in events))
    requested = set(deleted)
    if requested-known:
        raise ValueError("request contains an unrecorded deletion unit")
    dim = None
    removed, kept = [], []
    for event in events:
        ids = tuple(event["updates"])
        if not ids:
            raise ValueError("events must contain at least one actual participant")
        us = np.asarray([event["updates"][j] for j in ids], dtype=float)
        if us.ndim != 2 or not us.shape[1] or not np.isfinite(us).all():
            raise ValueError("finite update vectors required")
        dim = us.shape[1] if dim is None else dim
        if us.shape[1] != dim:
            raise ValueError("parameter layout changed across events")
        weights = event.get("weights")
        if weights is not None and set(weights) != set(ids):
            raise ValueError("weights must match actual participants")
        w = np.array([1. if weights is None else weights[j] for j in ids], float)
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("nonnegative weights with positive sum required")
        us = us*(w/w.sum())[:, None]
        mask = np.array([j in requested for j in ids])
        removed.append(us[mask].sum(axis=0))
        kept.append(us[~mask].sum(axis=0))
    return np.stack(removed), np.stack(kept)


def cancellation_geometry(removed, retained, *, smoothing=1., times=None):
    """Return G with G.T@G = PSD(C), without forming a parameter-square matrix.

    Z = argmin ||Z-X||_F^2 + smoothing sum ||Z[t+1]-Z[t]||^2 / dt.
    C = -(D.T@K + K.T@D) / (||D||_F^2 + ||K||_F^2), after smoothing.
    PSD(C) is the unique closest positive-semidefinite matrix in Frobenius norm.
    All positive eigenvalues are retained, except floating-point rank tolerance.
    """
    d, k = (np.asarray(x, dtype=float) for x in (removed, retained))
    if (d.ndim != 2 or d.shape != k.shape or 0 in d.shape
            or not np.isfinite(d).all() or not np.isfinite(k).all()
            or not np.isfinite(smoothing) or smoothing < 0):
        raise ValueError("matching finite [time, parameter] paths and smoothing >= 0 required")
    t, dim = d.shape
    stamps = np.arange(t, dtype=float) if times is None else np.asarray(times, float)
    if stamps.shape != (t,) or not np.isfinite(stamps).all() or (np.diff(stamps) <= 0).any():
        raise ValueError("strictly increasing finite timestamps required")
    # Common scaling prevents overflow and leaves the normalized operator invariant.
    scale = max(float(np.max(np.abs(d))), float(np.max(np.abs(k))))
    if scale == 0:
        return np.empty((0, dim)), {"rank": 0, "eigenvalues": [], "smoothing": smoothing,
                                    "history_steps": t, "input_scale": 0., "trace": 0.}
    d, k = d/scale, k/scale
    diff = np.diff(np.eye(t), axis=0)
    laplacian = diff.T @ (diff/np.diff(stamps)[:, None]) if t > 1 else np.zeros((1, 1))
    smoother = np.eye(t)+smoothing*laplacian
    d, k = np.linalg.solve(smoother, d), np.linalg.solve(smoother, k)
    energy = float(np.sum(d*d)+np.sum(k*k))
    _, singular, basis = np.linalg.svd(np.vstack([d, k]), full_matrices=False)
    basis = basis[singular > singular[0]*1e-12]
    db, kb = d@basis.T, k@basis.T
    small = -(db.T@kb+kb.T@db)/energy
    vals, vecs = np.linalg.eigh((small+small.T)/2)
    keep = vals > 1e-12  # numerical zero, not a tunable detection threshold
    g = (np.sqrt(vals[keep])[:, None]*vecs[:, keep].T)@basis
    return g, {"rank": int(keep.sum()), "eigenvalues": vals[keep].tolist(),
               "smoothing": float(smoothing), "history_steps": t,
               "input_scale": scale, "trace": float(vals[keep].sum()),
               "raw_min_eigenvalue": float(vals.min()), "operator_norm": float(max(vals.max(), 0.))}


def geometry_control(proposal, factor, retain, *, strength=1., epsilon=0.,
                     radius=None, min_progress=.1, risk_ratio=None, tolerance=1e-10, maxiter=1000):
    """Strongly convex QCQP: min .5||d-v||^2 + .5*strength*||Gd||^2,
    H d <= epsilon, ||d|| <= radius, v.T d >= min_progress * ||v||^2.
    Optionally ||Gd|| <= risk_ratio * ||Gv||. This explicit relative budget
    avoids selecting a penalty strength using unknown risk magnitudes.

    Progress is proposal alignment, NOT a proof of forgetting. Infeasibility
    raises instead of presenting a zero step as successful unlearning.
    """
    v = np.asarray(proposal, float)
    g, h = (np.asarray(x, float) for x in (factor, retain))
    if v.ndim != 1 or not v.size or not np.isfinite(v).all():
        raise ValueError("finite nonempty proposal required")
    if any(x.ndim != 2 or x.shape[1] != v.size or not np.isfinite(x).all() for x in (g, h)):
        raise ValueError("finite factor/retention matrices with matching width required")
    eps = np.broadcast_to(np.asarray(epsilon, float), (len(h),)).copy()
    norm = float(np.linalg.norm(v))
    rho = norm if radius is None else float(radius)
    if (not np.isfinite([strength, rho, min_progress, tolerance]).all() or strength < 0
            or rho < 0 or not 0 <= min_progress <= 1 or tolerance <= 0 or maxiter < 1
            or not np.isfinite(eps).all() or (eps < 0).any()):
        raise ValueError("invalid strength, progress, tolerance or budgets")
    if risk_ratio is not None and (not np.isfinite(risk_ratio) or not 0 <= risk_ratio <= 1):
        raise ValueError("risk_ratio must be None or in [0, 1]")
    if norm == 0:
        return v.copy(), {"status": "zero_proposal", "forgetting_verified": False,
                          "progress": 0., "step_norm": 0., "proposal_norm": 0.}
    if rho < min_progress*norm-1e-12*norm:
        raise ControlInfeasible("trust radius cannot satisfy minimum progress")
    rows = np.vstack([v[None], g, h])
    norms = np.linalg.norm(rows, axis=1)
    _, singular, basis = np.linalg.svd(rows[norms > 0]/norms[norms > 0, None], full_matrices=False)
    basis = basis[singular > singular[0]*1e-12]
    # d = ||v|| B.T z. H rows are scaled only to condition the constraints.
    q = basis@v/norm
    gb, hb = g@basis.T, h@basis.T
    hn = np.linalg.norm(hb, axis=1)
    nz = hn > 0
    a = np.vstack([hb[nz]/hn[nz, None], -q[None]])
    b = np.r_[eps[nz]/(norm*hn[nz]), -min_progress]
    reference_risk = float(np.linalg.norm(gb@q))
    bounded = None
    if risk_ratio is not None and len(g):
        if reference_risk == 0 or risk_ratio == 0:
            gn = np.linalg.norm(gb, axis=1)
            equalities = gb[gn > 0]/gn[gn > 0, None]
            a = np.vstack([a, equalities, -equalities])
            b = np.r_[b, np.zeros(2*len(equalities))]
        else:
            bounded = gb/reference_risk
    cap = rho/norm
    curvature = np.eye(len(basis))+strength*(gb.T@gb)
    def objective(z):
        return .5*np.sum((z-q)**2)+.5*strength*np.sum((gb@z)**2)
    def jac(z):
        return curvature@z-q
    constraints = [{"type": "ineq", "fun": lambda z: b-a@z, "jac": lambda z: -a},
                   {"type": "ineq", "fun": lambda z: cap*cap-z@z, "jac": lambda z: -2*z}]
    if bounded is not None:
        constraints.append({"type": "ineq", "fun": lambda z: risk_ratio**2-np.sum((bounded@z)**2),
                            "jac": lambda z: -2*bounded.T@(bounded@z)})
    initial = q*min(cap, 1.)
    result = minimize(objective, initial, jac=jac, method="SLSQP", constraints=constraints,
                      options={"ftol": tolerance, "maxiter": maxiter})
    z = result.x
    violation = max(float(np.max(a@z-b)), float(np.linalg.norm(z)-cap), 0.)
    if bounded is not None:
        violation = max(violation, float(np.linalg.norm(bounded@z)-risk_ratio))
    if not np.isfinite(z).all() or violation > 1e-7:
        raise ControlInfeasible(f"QCQP did not meet budgets: {result.message}; violation={violation:g}")
    # Convex KKT residual: a posteriori numerical evidence, not an ASR guarantee.
    slack = b-a@z
    active = [row for row, s in zip(a, slack) if s <= 1e-6]
    active_slack = [max(float(s), 0.) for s in slack if s <= 1e-6]
    if cap*cap-z@z <= 1e-6:
        active.append(2*z)
        active_slack.append(max(float(cap*cap-z@z), 0.))
    if bounded is not None and risk_ratio**2-np.sum((bounded@z)**2) <= 1e-6:
        active.append(2*bounded.T@(bounded@z))
        active_slack.append(max(float(risk_ratio**2-np.sum((bounded@z)**2)), 0.))
    if active:
        multipliers, stationarity = nnls(np.stack(active).T, -jac(z), maxiter=10000)
    else:
        multipliers, stationarity = np.empty(0), float(np.linalg.norm(jac(z)))
    if stationarity > 1e-4*max(1., np.linalg.norm(q)):
        raise ControlInfeasible(f"QCQP stationarity not verified: {stationarity:g}")
    # For a feasible point of a 1-strongly-convex problem, this bounds the
    # normalized objective gap (up to the reported numerical feasibility error).
    gap_bound = .5*stationarity**2+float(multipliers@np.asarray(active_slack))
    if gap_bound > 1e-5:
        raise ControlInfeasible(f"QCQP objective gap not verified: {gap_bound:g}")
    step = norm*(z@basis)
    return step, {"status": "local_step_only", "forgetting_verified": False,
                  "progress": float(q@z), "min_progress": float(min_progress),
                  "proposal_norm": norm, "step_norm": float(np.linalg.norm(step)),
                  "rank": len(basis), "iterations": int(result.nit),
                  "normalized_objective": float(objective(z)),
                  "normalized_risk_before": float(np.sum((gb@q)**2)),
                  "normalized_risk_after": float(np.sum((gb@z)**2)),
                  "risk_ratio_budget": risk_ratio,
                  "retain_linear_change": (h@step).tolist(),
                  "max_normalized_violation": violation,
                  "kkt_stationarity": float(stationarity), "kkt_objective_gap_bound": gap_bound,
                  "solver_reported_success": bool(result.success), "active_constraints": len(multipliers)}


class TemporalFUController:
    """Adapter for federaser_clientwise.replay, controlling every actual step.

    retention_oracle(state, t) returns (class_gradients, loss_change_callback).
    It must use fixed, authorized validation data independent of deleted data.
    Failure propagates and stops replay; it is never silently replaced by zero.
    """
    def __init__(self, events, deleted, retention_oracle, *, window=20, smoothing=1.,
                 risk_ratio=.1, epsilon=.01, radius_ratio=.25, min_progress=.1,
                 mode="temporal"):
        if not isinstance(window, int) or window < 1 or not 0 < radius_ratio <= 1:
            raise ValueError("positive window and radius_ratio in (0,1] required")
        if mode not in ("temporal", "unsmoothed", "cumulative", "linear", "retention_only"):
            raise ValueError("unknown controller mode")
        self.removed, self.retained = request_paths(events, deleted)
        self.oracle, self.window, self.smoothing = retention_oracle, window, smoothing
        self.risk_ratio, self.epsilon = risk_ratio, epsilon
        self.radius_ratio, self.min_progress, self.mode = radius_ratio, min_progress, mode

    def __call__(self, state, proposal, t):
        if not isinstance(t, (int, np.integer)) or not 0 <= t < len(self.removed):
            raise ValueError("unknown replay event index")
        d, k = self.removed[max(0, t-self.window+1):t+1], self.retained[max(0, t-self.window+1):t+1]
        if self.mode == "cumulative":
            d, k = d.sum(0, keepdims=True), k.sum(0, keepdims=True)
        if self.mode == "retention_only":
            g, geometry = np.empty((0, len(proposal))), {"rank": 0}
        elif self.mode == "linear":
            direction = -d.sum(0)
            size = np.linalg.norm(direction)
            g = direction[None]/size if size else np.empty((0, len(proposal)))
            geometry = {"rank": len(g)}
        else:
            g, geometry = cancellation_geometry(d, k,
                smoothing=0. if self.mode == "unsmoothed" else self.smoothing)
        h, loss_change = self.oracle(np.asarray(state).copy(), t)
        step, info = geometry_control(proposal, g, h, strength=0., epsilon=self.epsilon,
            radius=self.radius_ratio*np.linalg.norm(proposal), min_progress=self.min_progress,
            risk_ratio=self.risk_ratio)
        step, checked = check_actual_retention(step, proposal, loss_change,
            epsilon=self.epsilon, min_progress=self.min_progress)
        return step, {**info, **checked, "geometry": geometry, "mode": self.mode,
                      "accepted_step_norm": float(np.linalg.norm(step))}


def check_actual_retention(step, proposal, loss_change, *, epsilon=0., min_progress=.1,
                           contraction=.5, max_backtracks=20, tolerance=1e-7):
    """Verify actual class losses; backtrack without crossing the progress floor.

    loss_change(d) must evaluate L(theta+d)-L(theta) on a fixed validation set.
    Raises on failure. Does NOT test trigger safety or forgetting. On rejection,
    callers must restore their model to the pre-step state before reusing it.
    """
    d, v = np.asarray(step, float), np.asarray(proposal, float)
    if (d.ndim != 1 or d.shape != v.shape or not np.isfinite(d).all()
            or not np.isfinite(v).all() or not 0 < contraction < 1
            or not 0 <= min_progress <= 1 or max_backtracks < 0):
        raise ValueError("invalid backtracking inputs")
    denom = float(v@v)
    for j in range(max_backtracks+1):
        fraction = contraction**j
        candidate = fraction*d
        progress = float(v@candidate/denom) if denom else 0.
        if denom and progress < min_progress-1e-10:
            break
        change = np.asarray(loss_change(candidate), float)
        eps = np.broadcast_to(np.asarray(epsilon, float), change.shape)
        if (change.ndim != 1 or not change.size or not np.isfinite(change).all()
                or not np.isfinite(eps).all() or (eps < 0).any()):
            raise ValueError("finite nonempty class loss changes and nonnegative budgets required")
        if np.max(change-eps) <= tolerance:
            return candidate, {"actual_retention_verified": True, "backtracks": j,
                               "accepted_fraction": fraction, "accepted_progress": progress,
                               "actual_loss_change": change.tolist()}
    raise ControlInfeasible("actual validation loss and minimum progress cannot both be met by backtracking")
