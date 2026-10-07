"""Detector and projection math. No dataset, trigger, or attacker labels here."""
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DetectorConfig:
    window: int = 16
    min_observations: int = 3
    magnitude_cap: float = 4.0
    round_threshold: float = 0.1
    request_threshold: float = 0.15
    pair_threshold: float = 0.10
    high_threshold: float = 0.50
    max_clients: int = 8
    max_classes: int = 2
    max_rank: int = 8
    energy: float = 0.9
    eps: float = 1e-8

    def validate(self):
        if min(self.window, self.min_observations, self.max_clients,
               self.max_classes, self.max_rank) < 1:
            raise ValueError("Window, coverage, candidate limits and rank must be positive")
        if self.high_threshold <= self.pair_threshold:
            raise ValueError("high_threshold must exceed pair_threshold")
        if not 0 < self.energy <= 1 or self.eps <= 0 or self.magnitude_cap <= 0:
            raise ValueError("Invalid energy, epsilon or magnitude cap")
        if min(self.round_threshold, self.request_threshold, self.pair_threshold) < 0:
            raise ValueError("Thresholds must be nonnegative")


def scores(heads, weights, requester, cfg):
    """heads: [round, client, class, feature+bias]; absent clients have weight 0."""
    cfg.validate()
    if heads.ndim != 4 or tuple(weights.shape) != tuple(heads.shape[:2]):
        raise ValueError("Expected [T,N,C,D] heads and [T,N] weights")
    if heads.shape[0] == 0 or not 0 <= requester < heads.shape[1]:
        raise ValueError("Empty history or invalid requester")
    if not torch.isfinite(heads).all() or not torch.isfinite(weights).all():
        raise ValueError("Non-finite history")
    if (weights < 0).any() or not torch.allclose(
            weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-5):
        raise ValueError("Aggregation weights must be nonnegative and sum to one")
    heads, weights = heads[-cfg.window:], weights[-cfg.window:]
    x = heads * weights[:, :, None, None]
    norms = x.norm(dim=-1)
    q = x[:, requester:requester + 1]
    qn = norms[:, requester:requester + 1]
    valid = (weights > 0) & (weights[:, requester:requester + 1] > 0)
    valid[:, requester] = False
    cosine = (x * q).sum(-1) / (norms * qn).clamp_min(cfg.eps)
    opposite = (-cosine).clamp(0, 1)
    cancellation = (1 - (x + q).norm(dim=-1) /
                    (norms + qn).clamp_min(cfg.eps)).clamp(0, 1)
    # Ignore absent clients when computing the per-round scale.
    masked = norms.masked_fill(weights[:, :, None] == 0, float("nan"))
    scale = torch.nanquantile(masked, 0.5, dim=1).unsqueeze(1).clamp_min(cfg.eps)
    magnitude = (torch.minimum(norms, qn) / scale).clamp(max=cfg.magnitude_cap)
    usable = valid[:, :, None] & (norms > cfg.eps) & (qn > cfg.eps)
    raw = (opposite * cancellation * magnitude).masked_fill(~usable, 0)
    count = valid.sum(0)
    denom = count.clamp_min(1)[:, None]
    persistence = ((raw >= cfg.round_threshold) & usable).sum(0) / denom
    score = raw.sum(0) / denom * persistence
    score[count < cfg.min_observations] = 0
    return score, raw, count


def build_basis(heads, raw, pairs, cfg):
    """Uncentered SVD. Each selected class row is embedded in full head space."""
    heads = heads[-cfg.window:]
    nclasses, width = heads.shape[-2:]
    vectors = []
    for client, cls in pairs:
        for t in range(heads.shape[0]):
            weight = raw[t, client, cls]
            row = heads[t, client, cls]
            if weight <= 0 or row.norm() <= cfg.eps:
                continue
            v = torch.zeros(nclasses, width, device=heads.device, dtype=heads.dtype)
            v[cls] = row / row.norm() * weight.clamp(max=cfg.magnitude_cap).sqrt()
            vectors.append(v.flatten())
    if not vectors:
        return heads.new_zeros((nclasses * width, 0))
    matrix = torch.stack(vectors)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    good = singular > cfg.eps
    singular, vh = singular[good], vh[good]
    if not singular.numel():
        return heads.new_zeros((nclasses * width, 0))
    cumulative = singular.square().cumsum(0) / singular.square().sum()
    rank = min(cfg.max_rank, int(torch.searchsorted(cumulative, cfg.energy)) + 1,
               singular.numel())
    return vh[:rank].T.contiguous()


def detect(heads, weights, requester, cfg):
    score, raw, count = scores(heads, weights, requester, cfg)
    value = float(score.max())
    status = "detected"
    pairs = []
    gamma = heads.new_zeros(heads.shape[1])
    if not (count >= cfg.min_observations).any():
        status = "insufficient_history"
    elif value <= cfg.request_threshold:
        status = "no_alarm"
    else:
        order = score.max(1).values.argsort(descending=True).tolist()
        for client in order:
            if client == requester or score[client].max() <= cfg.pair_threshold:
                continue
            if int((gamma > 0).sum()) >= cfg.max_clients:
                break
            classes = score[client].argsort(descending=True).tolist()
            selected = [c for c in classes if score[client, c] > cfg.pair_threshold]
            pairs.extend((client, c) for c in selected[:cfg.max_classes])
            gamma[client] = ((score[client].max() - cfg.pair_threshold) /
                             (cfg.high_threshold - cfg.pair_threshold)).clamp(0, 1)
        if not pairs:
            status = "no_candidates"
    basis = build_basis(heads, raw.masked_fill(raw < cfg.round_threshold, 0), pairs, cfg)
    if status == "detected" and basis.shape[1] == 0:
        status = "empty_basis"
        gamma.zero_()
    report = {"status": status, "requester": requester, "request_score": value,
              "alarm": value > cfg.request_threshold and bool((count >= cfg.min_observations).any()),
              "scores": score.tolist(), "coverage": count.tolist(),
              "pairs": pairs, "gamma": gamma.tolist(), "rank": basis.shape[1],
              "config": asdict(cfg)}
    return basis, gamma, report


def project(basis, vector):
    return basis @ (basis.T @ vector)


def bounded_correction(basis, gamma, contributions, strength, relative_budget,
                       absolute_budget, update_norm):
    """Contributions [client, flattened head]. Returns correction and diagnostics."""
    if not 0 <= strength <= 1 or min(relative_budget, absolute_budget) < 0:
        raise ValueError("Invalid purification strength or budget")
    if basis.shape[1] == 0 or strength == 0:
        return contributions.new_zeros(contributions.shape[1]), {"lambda": 0., "norm": 0.}
    weighted = (contributions * gamma[:, None]).sum(0)
    c = project(basis, weighted)
    norm = float(c.norm())
    budget = min(absolute_budget, relative_budget * float(update_norm))
    lam = min(strength, budget / max(norm, 1e-12))
    return c * lam, {"lambda": lam, "norm": norm * lam, "budget": budget}
