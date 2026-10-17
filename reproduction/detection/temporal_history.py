"""History-only request monitoring: normal dynamics, innovations, and a GLR scan.

NumPy only. No labels, attack generators, checkpoints, or FU calls enter scoring.
The Gaussian mean-shift model motivates the statistic, not a Gaussian p-value.
Calibration uses independent, matched *whole runs*, including all inspections.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

VERSION = "hrac-temporal-v1"
METHODS = ("level", "hrac_windows", "positive_change", "innovation", "temporal_glr")
FEATURES = ("intercept", "previous_change", "own_mean_change", "others_previous_change",
            "own_participation", "others_participation", "inverse_sqrt_round")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_evidence(spec):
    """ZIP or extracted evidence directory; append ::clean5/full6 for BadFU."""
    path_text, _, phase = str(spec).partition("::")
    path = Path(path_text)
    if path.is_dir():
        base = path.resolve()

        def read(name):
            p = (base / name).resolve()
            if not p.is_relative_to(base):
                raise ValueError("Evidence path escapes its directory")
            return p.read_bytes()
    else:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP entries")
            files = {n: z.read(n) for n in names if not n.endswith("/")}
        read = lambda name: files["evidence/" + name]
    inventory_bytes = read("inventory.json")
    inventory = json.loads(inventory_bytes)
    for name, expected in inventory.items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
            raise ValueError("Unsafe inventory path")
        if digest(read(name)) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    training = json.loads(read("training_protocol.json"))
    options = training["options"]
    attack = options["attack"]
    if attack == "fuba":
        if phase:
            raise ValueError("FUBA has one FL phase; do not append ::phase")
        prefix, protocol_name = "", "observation_protocol.json"
        benign = options["warm_up"] >= options["rounds"]
    elif attack == "badfu":
        phase = phase or "full6"
        if phase not in ("full6", "clean5", "cover5"):
            raise ValueError("BadFU requires an observed FL phase")
        prefix, protocol_name = phase + "/", phase + "/protocol.json"
        benign = phase == "clean5"
    else:
        raise ValueError("Only original BadFU/FUBA evidence is supported")
    required = ("training_protocol.json", protocol_name, prefix + "manifest.json",
                prefix + "response_means.npz")
    if any(name not in inventory for name in required):
        raise ValueError("Required inputs must be covered by the inventory")
    protocol = json.loads(read(protocol_name))
    manifest_bytes = read(prefix + "manifest.json")
    manifest = json.loads(manifest_bytes)
    if digest(manifest_bytes) != protocol.get("source_manifest_sha256", protocol.get("manifest_sha256")):
        raise ValueError("Observation/manifest binding mismatch")
    if attack == "fuba" and digest(read("training_protocol.json")) != protocol["training_protocol_sha256"]:
        raise ValueError("Observation/training binding mismatch")
    with np.load(io.BytesIO(read(prefix + "response_means.npz")), allow_pickle=False) as f:
        raw = f["responses"].astype(float)
        mask = f["participation"] if attack == "fuba" else np.ones((len(raw), raw.shape[2]), bool)
    if (raw.ndim != 5 or min(raw.shape) < 1 or raw.shape[3] < 2 or not np.isfinite(raw).all()
            or len(raw) != len(manifest["rounds"]) or raw.shape[1] != len(protocol["probe_seeds"])
            or raw.shape[3] != len(protocol["views"]) or mask.shape != (len(raw), raw.shape[2])
            or mask.dtype.kind != "b"):
        raise ValueError("Invalid compact response/participation layout")
    for t, entry in enumerate(manifest["rounds"]):
        if np.flatnonzero(mask[t]).tolist() != sorted(entry["client_ids"]):
            raise ValueError("Participation differs from manifest")
        if np.any(raw[t, :, ~mask[t]] != 0):
            raise ValueError("Nonparticipants have nonzero contributions")
    paired = raw[:, :, :, 1:] - raw[:, :, :, :1]
    h = paired.transpose(0, 2, 1, 3, 4).reshape(len(raw), raw.shape[2], -1)
    # A seed is NOT split across fit/calibration, even through different phases.
    run_group = f"{attack}:{options.get('dataset', 'cifar10')}:{options['seed']}"
    observation_keys = ("views", "probe_seeds", "pattern_seeds", "probe_size", "pool_sha256",
                        "probe_ids", "preprocessing", "coordinate_system", "batch_size")
    coordinate_signature = {k: protocol.get(k) for k in observation_keys}
    coordinate_signature.update(attack=attack, dimensions=h.shape[-1])
    stratum = {k: options.get(k) for k in ("dataset", "rounds", "clients", "participant_rate",
                                          "non_iid", "alpha", "smoke")}
    stratum.update(attack=attack, actual_clients=h.shape[1], rounds=len(h))
    return dict(history=h, mask=mask, benign=benign, spec=str(spec), group=run_group,
                inventory_sha256=digest(inventory_bytes), phase=phase or "fl",
                coordinates=coordinate_signature, stratum=stratum,
                identity=digest(read(prefix + "response_means.npz")))


def release(effects):
    """Exact existing HRAC inter statistic for each whole-client removal.

    effects [..., client, coordinate] already include actual aggregation weights.
    No model subtraction is performed; this remains a functional-history proxy.
    """
    positive = np.maximum(effects, 0).sum(axis=-2)
    negative_parts = np.maximum(-effects, 0)
    negative = negative_parts.sum(axis=-2)
    denom = (positive + negative) * negative
    factor = np.divide(2 * np.minimum(positive, negative)**2, denom,
                       out=np.zeros_like(denom), where=denom > 0)
    return negative_parts * factor[..., None, :]


def series(history, mask):
    h, m = np.asarray(history, float), np.asarray(mask)
    if (h.ndim != 3 or min(h.shape) < 1 or h.shape[1] < 2 or m.shape != h.shape[:2]
            or m.dtype.kind != "b" or not np.isfinite(h).all() or np.any(h[~m] != 0)):
        raise ValueError("Finite weighted [time, client, coordinate] and boolean participation required")
    cumulative = np.cumsum(h, axis=0)
    level = release(cumulative)
    change = np.diff(level, axis=0, prepend=np.zeros_like(level[:1]))
    windows = level.copy()
    for width in (10, 20):
        rolling = cumulative.copy()
        rolling[width:] -= cumulative[:-width]
        windows = np.maximum(windows, release(rolling))
    return level, change, windows


def features(level, change, mask, scale):
    """Prediction at t uses only t-1 outputs and the known t participation."""
    t, clients, dims = level.shape
    lag = np.concatenate([np.zeros_like(change[:1]), change[:-1]]) / scale
    prior = np.concatenate([np.zeros_like(level[:1]), level[:-1]]) / scale
    elapsed = np.maximum(np.arange(t), 1)[:, None, None]
    others = (lag.sum(1, keepdims=True) - lag) / (clients - 1)
    own = np.broadcast_to(mask[:, :, None], level.shape)
    other_mask = np.broadcast_to((mask.sum(1, keepdims=True)-mask)[:, :, None] / (clients-1), level.shape)
    age = np.broadcast_to(1 / np.sqrt(np.arange(1, t+1))[:, None, None], level.shape)
    return np.stack([np.ones_like(level), lag, prior / elapsed, others, own, other_mask, age], axis=-1)


def fit_model(episodes, ridge=1e-3, variance_shrinkage=.1):
    if not episodes or not 0 < ridge or not 0 < variance_shrinkage <= 1:
        raise ValueError("Normal fitting runs and positive regularization required")
    if any(not e["benign"] for e in episodes):
        raise ValueError("Dynamics may only be fitted on attack-disabled/clean evidence")
    ref = episodes[0]["coordinates"]
    if any(e["coordinates"] != ref for e in episodes):
        raise ValueError("Fitting coordinate systems differ")
    paths = [series(e["history"], e["mask"]) for e in episodes]
    if any(len(p[0]) < 4 for p in paths):
        raise ValueError("At least four rounds are required for dynamics fitting")
    yraw = np.concatenate([p[1][1:].reshape(-1, p[1].shape[-1]) for p in paths])
    rms = np.sqrt(np.mean(yraw**2, axis=0))
    scale = np.maximum(rms, max(float(np.sqrt(np.mean(rms**2))) * .01, 1e-12))
    x = np.concatenate([features(p[0], p[1], e["mask"], scale)[1:].reshape(-1, len(scale), len(FEATURES))
                        for e, p in zip(episodes, paths)])
    y = yraw / scale
    center, spread = x.mean(0), x.std(0)
    center[:, 0], spread[:, 0] = 0, 1
    spread = np.maximum(spread, 1e-8)
    x = (x - center) / spread
    gram = np.einsum("ndp,ndq->dpq", x, x) / len(x)
    penalty = np.eye(len(FEATURES)) * ridge
    penalty[0, 0] = 0
    rhs = np.einsum("ndp,nd->dp", x, y) / len(x)
    coef = np.linalg.solve(gram + penalty, rhs[..., None])[..., 0]
    residual = y - np.einsum("ndp,dp->nd", x, coef)
    variance = np.mean(residual**2, axis=0)
    sigma = np.sqrt((1-variance_shrinkage)*variance + variance_shrinkage*max(float(variance.mean()), 1e-12))
    return dict(version=VERSION, ridge=ridge, variance_shrinkage=variance_shrinkage,
                features=list(FEATURES), scale=scale, center=center, spread=spread, coef=coef,
                sigma=sigma, fitting_groups=[e["group"] for e in episodes], coordinates=ref)


def glr_scan(innovations):
    """One-sided unknown-onset/unknown-mean Gaussian GLR, for each coordinate.

    max_{k<=t, mu>=0} sum(mu*e - mu^2/2) = [sum(e)]_+^2/(2*length).
    Round 1 initializes lags; it is never treated as a zero-error observation.
    """
    z = np.asarray(innovations, float)
    if z.ndim != 3 or not np.isfinite(z).all():
        raise ValueError("Finite innovations[round, client, coordinate] required")
    cumulative = np.cumsum(z, axis=0)
    result = np.zeros_like(z)
    starts = np.zeros(z.shape, dtype=int)
    for t in range(1, len(z)):
        sums = cumulative[t] - cumulative[:t]
        lengths = np.arange(t, 0, -1)[:, None, None]
        values = np.maximum(sums, 0)**2 / (2*lengths)
        result[t] = values.max(0)
        starts[t] = values.argmax(0) + 2  # one-based physical round
    return result, starts


def score(episode, model):
    if episode["coordinates"] != model["coordinates"]:
        raise ValueError("Evaluation coordinates differ from fitting")
    level, change, windows = series(episode["history"], episode["mask"])
    x = features(level, change, episode["mask"], model["scale"])
    prediction = np.einsum("tkdp,dp->tkd", (x-model["center"])/model["spread"], model["coef"])
    innovations = (change / model["scale"] - prediction) / model["sigma"]
    innovations[0] = 0
    glr, starts = glr_scan(innovations)
    coordinate_paths = dict(level=level, hrac_windows=windows, positive_change=np.maximum(change, 0),
                            innovation=np.maximum(innovations, 0), temporal_glr=glr)
    seen = np.logical_or.accumulate(episode["mask"], axis=0)
    paths = {name: np.where(seen, value.max(-1), 0) for name, value in coordinate_paths.items()}
    # No first-round alarm for any method: shared inspection policy.
    for value in paths.values():
        value[0] = 0
    coordinate = glr.argmax(-1)
    onset = np.take_along_axis(starts, coordinate[..., None], axis=-1)[..., 0]
    return dict(paths=paths, innovations=innovations, level=level, change=change,
                glr_coordinate=coordinate, glr_interval_start=onset)


def calibrated_path(path, calibration_maxima, alpha=.05):
    """A run-max rank protects all clients/coordinates/times up to fixed horizon.

    Within-run dependence is unrestricted. Across matched normal runs, scores
    must be exchangeable conditional on the independently fitted fixed model.
    This does not give a conditional per-client FPR or an infinite-horizon bound.
    """
    a, cal = np.asarray(path, float), np.asarray(calibration_maxima, float)
    if (a.ndim != 2 or cal.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(cal).all()
            or np.any(a < 0) or np.any(cal < 0) or not 0 < alpha < 1):
        raise ValueError("Finite nonnegative paths/calibration and alpha in (0,1) required")
    running = np.maximum.accumulate(a.max(1))
    p = (1 + (cal[:, None] >= running).sum(0)) / (len(cal)+1)
    enough = 1/(len(cal)+1) <= alpha + 1e-12
    alarm = p <= alpha if enough else None
    hit = np.flatnonzero(alarm) if alarm is not None else []
    return dict(p=p.tolist() if len(cal) else None, alarm=alarm.tolist() if alarm is not None else None,
                first_alarm_round=int(hit[0]+1) if len(hit) else None, n_calibration=len(cal),
                minimum_p=1/(len(cal)+1), status="calibrated_under_exchangeability" if enough else "insufficient_independent_runs")


def validate_split(fitting, calibration, tests):
    """Do not count repeated seeds/phases/probes as independent calibration."""
    for group in (fitting, calibration, tests):
        ids = [e["group"] for e in group]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate training seed/episode within a split")
    fit_ids, cal_ids = {e["group"] for e in fitting}, {e["group"] for e in calibration}
    if fit_ids & cal_ids or cal_ids & {e["group"] for e in tests}:
        raise ValueError("Calibration overlaps fitting or evaluation training seeds")
    # A same-seed attack/control pair can be explored, but not given a guarantee.
    if calibration and fit_ids & {e["group"] for e in tests}:
        raise ValueError("Calibrated evaluation must use seeds independent of fitting")
    if any(not e["benign"] for e in calibration):
        raise ValueError("Calibration requires attack-disabled normal runs")
    if calibration:
        ref = calibration[0]["stratum"]
        if any(e["stratum"] != ref for e in calibration + tests):
            raise ValueError("Calibration/evaluation horizon, population or training condition differs")
