"""Empirical upper-tail risk and class constraints; no security guarantee."""
import math

import torch
from torch.nn import functional as F


def upper_tail_weights(values, mass):
    """Exact empirical CVaR weights, including fractional boundary mass.

    Maximizes <w, values> subject to sum(w)=1 and 0 <= w <= 1/(mass*n).
    Detached weights select a valid subgradient (ties may select any optimizer).
    """
    if values.ndim != 1 or not len(values) or not 0 < mass <= 1:
        raise ValueError('nonempty vector and tail mass in (0,1] required')
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite tail values')
    n_mass = mass * len(values)
    full = min(int(math.floor(n_mass)), len(values))
    order = values.detach().argsort(descending=True, stable=True)
    weights = torch.zeros_like(values)
    weights[order[:full]] = 1 / n_mass
    if full < len(values):
        weights[order[full]] = (n_mass - full) / n_mass
    return weights


def upper_tail_mean(values, mass):
    return (upper_tail_weights(values, mass) * values).sum()


def margin(z, y, target):
    return z[:, target] - z.gather(1, y[:, None]).squeeze(1)


def paired_margin_change(pre_clean, pre_changed, post_clean, post_changed, y, target):
    return (margin(post_changed, y, target) - margin(pre_changed, y, target)
            - margin(post_clean, y, target) + margin(pre_clean, y, target))


def class_means(values, labels, classes):
    if any(not (labels == c).any() for c in range(classes)):
        raise ValueError('every class must be represented in constraint data')
    return torch.stack([values[labels == c].mean() for c in range(classes)])


def ce_violations(z, y, reference_ce, tolerance):
    return class_means(F.cross_entropy(z, y, reduction='none'), y, len(reference_ce)) - reference_ce - tolerance


def dual_step(dual, violation, rate, cap):
    if rate <= 0 or cap <= 0:
        raise ValueError('positive dual rate and cap required')
    return (dual + rate * violation.detach()).clamp(0, cap)


def transform(x, pattern):
    # Support legacy additive tensors as well as explicit overwrite patterns.
    if isinstance(pattern, torch.Tensor):
        return (x + pattern.to(x)).clamp(0, 1)
    delta = pattern['delta'].to(x)
    if pattern['mode'] == 'additive':
        return (x + delta).clamp(0, 1)
    if pattern['mode'] != 'patch':
        raise ValueError('unknown transform')
    mask = pattern['mask'].to(x)
    return x * (1 - mask) + delta * mask


def pattern_spaces(shape, mode, patch_size):
    if mode not in ('additive', 'patch', 'both'):
        raise ValueError('unknown probe_mode')
    spaces = []
    if mode in ('additive', 'both'):
        spaces.append(('additive', 'full', None))
    if mode in ('patch', 'both'):
        _, height, width = shape
        if not 0 < patch_size < min(height, width):
            raise ValueError('patch must be smaller than image')
        for name, row, col in [('tl', 0, 0), ('tr', 0, width-patch_size),
                               ('bl', height-patch_size, 0), ('br', height-patch_size, width-patch_size)]:
            mask = torch.zeros(1, 1, height, width)
            mask[..., row:row+patch_size, col:col+patch_size] = 1
            spaces.append(('patch', name, mask))
    return spaces
