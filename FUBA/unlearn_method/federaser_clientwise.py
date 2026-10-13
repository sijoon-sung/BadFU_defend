"""Vector FedEraser core: calibrate EACH client before aggregation (Eq. 1).

Separate from legacy federaser_faithful.py (mean-first/layerwise variant).
The caller supplies the true pre-training initial state, exact participants,
retained-data recalibration, weights and optional actual-step controller.
This approximate unlearner does not certify data deletion or backdoor safety.
"""
import numpy as np


def calibrated_aggregate(old_updates, new_updates, weights=None):
    ids = tuple(old_updates)
    if not ids or set(ids) != set(new_updates):
        raise ValueError("nonempty matching retained participants required")
    old = np.stack([old_updates[j] for j in ids]).astype(float)
    new = np.stack([new_updates[j] for j in ids]).astype(float)
    if old.ndim != 2 or old.shape != new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("finite matching update vectors required")
    if weights is not None and set(weights) != set(ids):
        raise ValueError("weights must match retained participants")
    w = np.array([1. if weights is None else weights[j] for j in ids], dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("nonnegative weights with positive total required")
    old_norm, new_norm = np.linalg.norm(old, axis=1), np.linalg.norm(new, axis=1)
    # An undefined direction cannot be silently replaced with the deleted-era one.
    if np.any((old_norm > 0) & (new_norm == 0) & (w > 0)):
        raise ValueError("nonzero historical update has zero recalibration direction")
    scale = np.divide(old_norm, new_norm, out=np.zeros_like(old_norm), where=new_norm > 0)
    return (w/w.sum()) @ (scale[:, None]*new)


def replay(initial, events, deleted, recalibrate, controller=None):
    """events: {updates: {unit: vector}, weights?: {unit: weight}}.

    recalibrate(current.copy(), unit, event_index) returns an actual retained
    local update. controller(current.copy(), proposal.copy(), event_index)
    returns (controlled_step, diagnostics). No deleted unit is recalibrated.
    Only full-unit deletion is supported here; split samples upstream into
    deletion units with correct histories and retained-data callbacks.
    """
    state = np.asarray(initial, dtype=float).copy()
    if state.ndim != 1 or not len(state) or not np.isfinite(state).all() or not events:
        raise ValueError("finite initial vector and nonempty history required")
    known = set().union(*(set(event["updates"]) for event in events))
    deleted = set(deleted)
    if not deleted or deleted-known or not known-deleted:
        raise ValueError("known deletion request and at least one retained unit required")
    for event in events:
        if event.get("weights") is not None and set(event["weights"]) != set(event["updates"]):
            raise ValueError("event weights must match actual participants")
        for update in event["updates"].values():
            if np.shape(update) != state.shape or not np.isfinite(update).all():
                raise ValueError("invalid historical update")
    log = []
    for t, event in enumerate(events):
        keep = {j: u for j, u in event["updates"].items() if j not in deleted}
        if not keep:
            log.append({"event": t, "status": "no_retained_participants"})
            continue
        new = {j: recalibrate(state.copy(), j, t) for j in keep}
        w = None if event.get("weights") is None else {j: event["weights"][j] for j in keep}
        proposal = calibrated_aggregate(keep, new, w)
        step, info = (proposal, {"status": "uncontrolled"}) if controller is None else controller(state.copy(), proposal.copy(), t)
        step = np.asarray(step, dtype=float)
        if step.shape != state.shape or not np.isfinite(step).all():
            raise ValueError("invalid controlled update")
        state += step
        log.append({"event": t, "retained_participants": list(keep), **info})
    return state, {"status": "candidate_requires_verification", "forgetting_verified": False, "steps": log}
