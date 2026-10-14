"""Chunked row-space reduction for the same request QCQP on large models.

No random sketch or parameter/top-k selection. Float32 row storage, float64
Gram accumulation; numerical rank tolerance is reported. Memory O((2T+C+1)p),
not O(p^2). This still needs several GB for a full ResNet-18 history window.
"""
import numpy as np
from request_geometry import cancellation_geometry, geometry_control


def large_control(proposal, removed, retained, gradients, *, mode="temporal",
                  smoothing=1., risk_ratio=.1, epsilon=.01, radius_ratio=.25,
                  min_progress=.1, chunk=65536):
    v, d, k, h = (np.asarray(x) for x in (proposal, removed, retained, gradients))
    if (v.ndim != 1 or not v.size or d.ndim != 2 or d.shape != k.shape
            or not len(d) or d.shape[1] != len(v) or h.ndim != 2 or h.shape[1] != len(v)
            or chunk < 1 or not 0 < radius_ratio <= 1
            or any(not np.isfinite(x).all() for x in (v, d, k, h))):
        raise ValueError("finite matching vectors/paths and positive chunk/radius required")
    if mode not in ("temporal", "unsmoothed", "cumulative", "linear", "retention_only"):
        raise ValueError("unknown geometry mode")
    if mode == "cumulative":
        d, k = d.sum(0, keepdims=True), k.sum(0, keepdims=True)
    if mode == "retention_only":
        d, k = np.empty((0, len(v))), np.empty((0, len(v)))
    rows = np.concatenate([v[None], d, k, h]).astype(np.float32)
    n, p = rows.shape
    norm2 = np.zeros(n)
    for lo in range(0, p, chunk):
        block = rows[:, lo:lo+chunk].astype(float)
        norm2 += np.einsum("ij,ij->i", block, block)
    norms = np.sqrt(norm2)
    rows /= np.where(norms > 0, norms, 1.)[:, None]
    gram = np.zeros((n, n))
    for lo in range(0, p, chunk):
        block = rows[:, lo:lo+chunk].astype(float)
        gram += block@block.T
    vals, vecs = np.linalg.eigh((gram+gram.T)/2)
    keep = vals > max(float(vals.max()), 1.)*1e-12
    # B = A M is an orthonormal row basis, applied in chunks, never stored.
    a = (vecs[:, keep]/np.sqrt(vals[keep])).T
    coordinates = (gram@a.T)*norms[:, None]
    row_residual = np.sqrt(np.maximum(np.diag(gram)-np.sum((gram@a.T)**2, axis=1), 0.))
    if row_residual.max(initial=0) > 1e-5:
        raise RuntimeError("large-model row-space approximation exceeds tolerance")
    if not len(a):
        return np.zeros_like(v, dtype=float), {"status": "zero_proposal", "geometry_rank": 0,
            "forgetting_verified": False, "row_space_max_residual": float(row_residual.max(initial=0))}
    q = coordinates[0]
    dr, kr = coordinates[1:1+len(d)], coordinates[1+len(d):1+2*len(d)]
    hr = coordinates[1+2*len(d):]
    if mode == "retention_only":
        g, geo = np.empty((0, len(q))), {"rank": 0}
    elif mode == "linear":
        direction = -dr.sum(0)
        size = np.linalg.norm(direction)
        g = direction[None]/size if size else np.empty((0, len(q)))
        geo = {"rank": len(g)}
    else:
        g, geo = cancellation_geometry(dr, kr, smoothing=0. if mode == "unsmoothed" else smoothing)
    z, info = geometry_control(q, g, hr, strength=0., risk_ratio=risk_ratio,
        epsilon=epsilon, radius=radius_ratio*np.linalg.norm(q), min_progress=min_progress)
    coeff = z@a
    step = np.empty(p)
    for lo in range(0, p, chunk):
        step[lo:lo+chunk] = coeff@rows[:, lo:lo+chunk].astype(float)
    return step, {**info, "geometry": geo, "row_space_rank": len(a),
        "row_space_max_residual": float(row_residual.max(initial=0)),
        "row_storage_dtype": "float32", "gram_dtype": "float64", "gram_rank_tolerance": 1e-12}
