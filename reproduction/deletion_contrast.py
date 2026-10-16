"""Logit interactions and clean-explanation intervals on frozen model Jacobians.

Research measurements, not a calibrated attack detector. Labels are trusted
PUBLIC labels, used to build a reference span, never client labels or triggers.
"""
import hashlib
from itertools import product

import numpy as np
import torch
from torch.nn import functional as F

from reproduction.history_boundary import boundary_evidence, residual_interval
from reproduction.request_probe import predict


def centered(z):
    return z - z.mean(-1, keepdim=True)


def contrast(z, target):
    # A fixed linear contrast; unlike logsumexp this cancels a class-bias vector.
    return centered(z)[..., target]


def double_difference(pre_clean, pre_probe, post_clean, post_probe):
    return (post_probe - pre_probe) - (post_clean - pre_clean)


def split_public(images, labels, ids, counts, seed, classes=10):
    """Three stratified, disjoint splits: reference, search, audit.

    Counts are PER CLASS. Pixel duplicates are rejected even under different IDs.
    """
    if images.dtype != np.uint8 or images.ndim != 4:
        raise ValueError("uint8 NHWC public images required")
    if (labels.shape != (len(images),) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels >= classes)):
        raise ValueError("valid public integer labels required")
    if (ids.shape != labels.shape or ids.dtype.kind not in 'US'
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError("unique public image IDs required")
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError("three positive per-class counts required")
    rng, splits = np.random.default_rng(seed), [[], [], []]
    for label in range(classes):
        available = rng.permutation(np.flatnonzero(labels == label))
        if len(available) < sum(counts):
            raise ValueError(f"not enough public images for class {label}")
        start = 0
        for dest, count in zip(splits, counts):
            dest.extend(available[start:start + count].tolist())
            start += count
    result = [np.asarray(s, dtype=int) for s in splits]
    hashes = []
    for s in result:
        h = {hashlib.sha256(images[i].tobytes()).digest() for i in s}
        if len(h) != len(s) or any(h & old for old in hashes):
            raise ValueError("duplicate public pixels within or across splits")
        hashes.append(h)
    return result


def apply_probe(x, result):
    if result.get('probe_mode', 'additive') == 'patch':
        return x * (1 - result['mask']) + result['delta'] * result['mask']
    return (x + result['delta']).clamp(0, 1)


def search_logits(pre, post, x, cfg, probe_mode='additive', patch_size=4, patch_step_size=.1):
    """Universal delta + target chosen on search images ONLY, all classes tried."""
    cfg.validate()
    cp, cq = predict(pre, x, cfg.batch_size), predict(post, x, cfg.batch_size)
    if cp.shape != cq.shape:
        raise ValueError("incompatible logits")
    rng = torch.Generator().manual_seed(cfg.seed)
    starts = [(2 * torch.rand((1, *x.shape[1:]), generator=rng) - 1).to(x.device)
              * cfg.epsilon for _ in range(cfg.restarts)]
    starts[0].zero_()
    if probe_mode not in ('additive', 'patch', 'both'):
        raise ValueError('unknown probe family')
    spaces = [('additive', 'full', None)] if probe_mode in ('additive', 'both') else []
    if probe_mode in ('patch', 'both'):
        h, w = x.shape[-2:]
        if not 1 <= patch_size <= min(h, w) or patch_step_size <= 0:
            raise ValueError('invalid patch size/step')
        for name, r, c in (('tl', 0, 0), ('tr', 0, w-patch_size),
                           ('bl', h-patch_size, 0), ('br', h-patch_size, w-patch_size)):
            mask = torch.zeros_like(starts[0])
            mask[:, :, r:r+patch_size, c:c+patch_size] = 1
            spaces.append(('patch', name, mask))
    best, trials = None, []
    for (mode, position, mask), target in product(spaces, range(cp.shape[1])):
        for restart, initial in enumerate(starts):
            delta = initial.clone() if mode == 'additive' else (initial / cfg.epsilon + 1) / 2
            for step in range(cfg.steps + 1):
                delta = delta.detach().requires_grad_(step < cfg.steps)
                total, gradient = 0., torch.zeros_like(delta)
                for a in range(0, len(x), cfg.batch_size):
                    b = min(a + cfg.batch_size, len(x))
                    with torch.set_grad_enabled(delta.requires_grad):
                        xp = apply_probe(x[a:b], dict(delta=delta, mask=mask, probe_mode=mode))
                        dd = double_difference(cp[a:b], pre(xp), cq[a:b], post(xp))
                        value = contrast(dd, target).sum() / len(x)
                    total += value.detach().item()
                    if delta.requires_grad and value.requires_grad:
                        gradient += torch.autograd.grad(value, delta)[0]
                if not np.isfinite(total) or not torch.isfinite(gradient).all():
                    raise ValueError("nonfinite logit search")
                if best is None or total > best['search_objective']:
                    best = dict(target=target, restart=restart, step=step,
                                search_objective=total, delta=delta.detach().clone(),
                                probe_mode=mode, position=position, mask=mask)
                if delta.requires_grad:
                    delta = ((delta + cfg.step_size * gradient.sign()).clamp(-cfg.epsilon, cfg.epsilon)
                             if mode == 'additive' else (delta + patch_step_size * gradient.sign()).clamp(0, 1))
            trials.append(dict(target=target, restart=restart, final_objective=total,
                               probe_mode=mode, position=position))
    best['trials'] = trials
    return best


def direction_norm(direction):
    return float(sum(v.double().square().sum() for v in direction.values()).sqrt())


def public_directions(model, x, y, classes, batch_size):
    """Unit negative class-loss gradients at the FINAL pre-deletion model.

    This is an explicit experimental span, not all possible benign updates.
    Directions remain on CPU; models/buffers are never trained or updated.
    """
    model.eval().requires_grad_(True)
    params = dict(model.named_parameters())
    result, norms = [], []
    for label in range(classes):
        xx = x[y == label]
        if not len(xx):
            raise ValueError("every public class needs reference images")
        direction = {k: torch.zeros_like(p, device='cpu') for k, p in params.items()}
        for chunk in xx.split(batch_size):
            loss = F.cross_entropy(model(chunk), torch.full((len(chunk),), label,
                                   dtype=torch.long, device=chunk.device)) * len(chunk) / len(xx)
            grad = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
            for (name, _), g in zip(params.items(), grad):
                if g is not None:
                    direction[name].sub_(g.detach().cpu())
        norm = direction_norm(direction)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"degenerate public reference gradient: {label}")
        result.append({k: v / norm for k, v in direction.items()})
        norms.append(norm)
    model.requires_grad_(False)
    return result, norms


def jvp_logits(model, direction, x, batch_size):
    """Exact forward AD at one fixed anchor, with its recorded BN buffers fixed."""
    params = dict(model.named_parameters())
    if set(params) != set(direction):
        raise ValueError("JVP direction must cover every parameter")
    tangent = {k: direction[k].to(p.device, p.dtype) for k, p in params.items()}
    output = []
    for chunk in x.split(batch_size):
        with torch.no_grad():
            _, value = torch.func.jvp(
                lambda pp: torch.func.functional_call(model, pp, (chunk,), strict=False),
                (params,), (tangent,))
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite history response")
        output.append(value.detach().cpu())
    return torch.cat(output)


def coordinates(clean, changed, target, use):
    """All centered clean logits constrain beta; one prespecified pooled contrast.

    The pooled contrast excludes public examples already belonging to target.
    Rare/source-specific attacks may be diluted; no post-hoc source selection.
    """
    c = centered(clean).flatten().numpy().astype(float)
    p = contrast(changed - clean, target)[use].mean().reshape(1).numpy().astype(float)
    return c, p


def explain(clean, probe, clean_basis, probe_basis, cfg):
    # Applied to unit-parameter directions (or endpoint changes / parameter norm).
    # Constants are preregistered, never fitted to attack scores.
    eps = cfg['clean_absolute'] + cfg['clean_relative'] * float(np.sqrt(np.mean(clean ** 2)))
    count = clean_basis.shape[1]
    result = residual_interval(clean, clean_basis, probe, probe_basis,
        clean_error=eps, coefficient_lower=np.full(count, -cfg['coefficient_bound']),
        coefficient_upper=np.full(count, cfg['coefficient_bound']),
        probe_error=cfg['probe_error'])
    return result, eps


def minimum_clean_error(clean, basis, coefficient_bound):
    """Diagnostic only: minimum L-infinity error of the bounded clean span.

    Does not enlarge the preregistered explanation set or change a score.
    """
    from scipy.optimize import linprog
    n = basis.shape[1]
    matrix = np.vstack((np.column_stack((basis, -np.ones(len(clean)))),
                        np.column_stack((-basis, -np.ones(len(clean))))))
    result = linprog(np.r_[np.zeros(n), 1.], A_ub=matrix, b_ub=np.r_[clean, -clean],
        bounds=[(-coefficient_bound, coefficient_bound)] * n + [(0., None)], method='highs')
    if not result.success:
        raise RuntimeError(f'clean-span diagnostic failed: {result.message}')
    return float(result.fun)


def history_scores(lower, upper, raw, blocks):
    """Group 0=deleted union, group 1=retained union; no guessed internal atoms."""
    kwargs = dict(deleted=np.array([True, False]), retained=np.array([False, True]))
    raw_result = boundary_evidence(raw, raw, blocks=blocks, **kwargs)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return dict(normal_set=None, raw=raw_result, normal_set_single_block=None,
                    status='abstain_incomplete_clean_explanation')
    return dict(normal_set=boundary_evidence(lower, upper, blocks=blocks, **kwargs),
                normal_set_single_block=boundary_evidence(lower, upper,
                    blocks=[(0, len(lower))], **kwargs), raw=raw_result, status='measured')


def empirical_auc(positive, negative):
    """Descriptive run/request-level ranking only; no independence/CI claim."""
    if not positive or not negative:
        return None
    a, b = np.asarray(positive)[:, None], np.asarray(negative)[None, :]
    return float(((a > b) + .5 * (a == b)).mean())
