"""FL contribution tension scored against a FU deletion boundary.

Consumes already measured, weighted functional contributions. It never constructs
an unlearned model. Units must resolve the requested deletion granularity; they
can be clients or shards of the same client. A high score is structural evidence,
not proof of malice, a calibrated alarm, or an exact FU prediction.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
import numpy as np


def boundary_profile(effects, removal):
    """effects[unit, ...coordinates], removal[unit] in [0,1].

    Coordinates must represent the same inputs/output directions across units.
    Effects already include aggregation weights. We sum within each unit BEFORE
    splitting signs: hidden within-unit contributions require finer input records.
    """
    e = np.asarray(effects, dtype=float)
    r = np.asarray(removal, dtype=float)
    if e.ndim < 2 or 0 in e.shape or r.shape != (len(e),):
        raise ValueError("expected nonempty effects[unit,...] and removal[unit]")
    if not np.isfinite(e).all() or not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError("finite effects and removal fractions in [0,1] required")
    r = r.reshape((len(e),) + (1,)*(e.ndim-1))
    pos, neg = np.maximum(e, 0), np.maximum(-e, 0)
    positive, negative = pos.sum(0), neg.sum(0)
    gross, net = positive+negative, positive-negative
    balance = np.divide(2*np.minimum(positive, negative), gross,
                        out=np.zeros_like(gross), where=gross > 0)
    removed_negative = (r*neg).sum(0)
    retained_positive = ((1-r)*pos).sum(0)
    removed_positive = (r*pos).sum(0)
    retained_negative = ((1-r)*neg).sum(0)
    forward_cross = np.minimum(removed_negative, retained_positive)
    reverse_cross = np.minimum(removed_positive, retained_negative)
    # Equal removal of opposite roles preserves their balance. Count both
    # directions across the boundary before taking the target-oriented excess.
    cross_mass = np.maximum(forward_cross-reverse_cross, 0)
    # v2 above fails for unequal P,N and equal fractional removal. For example,
    # (+10,-8), r=(.8,.8) gives cross_mass=.4 despite preserving both roles.
    # Compare EFFECT-MASS removal rates, not client/sample count fractions.
    negative_rate = np.divide(removed_negative, negative,
                              out=np.zeros_like(negative), where=negative > 0)
    positive_rate = np.divide(removed_positive, positive,
                              out=np.zeros_like(positive), where=positive > 0)
    rate_gap = np.maximum(negative_rate-positive_rate, 0)
    selective_mass = np.minimum(positive, negative)*rate_gap
    return {"gross": gross, "net": net, "balance": balance,
            "tension": 2*np.minimum(positive, negative),
            "removed_negative": removed_negative, "retained_positive": retained_positive,
            "removed_positive": removed_positive,
            "forward_cross": forward_cross, "reverse_cross": reverse_cross,
            "cross_mass": cross_mass, "score_v2": balance*cross_mass,
            "negative_removal_rate": negative_rate, "positive_removal_rate": positive_rate,
            "rate_gap": rate_gap, "selective_mass": selective_mass,
            "score": balance*selective_mass,
            "retained_signed_effect": ((1-r)*e).sum(0)}


def score_request(effects, units, requested):
    """A request is metadata applied to the existing ledger, never a model edit."""
    units = tuple(units)
    if len(units) != len(effects) or len(set(units)) != len(units):
        raise ValueError("unit identifiers must be unique and match effects")
    if not requested or set(requested)-set(units):
        raise ValueError("request must reference known deletion units")
    return boundary_profile(effects, [requested.get(u, 0.) for u in units])
