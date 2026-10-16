"""Universal, deletion-conditioned probe search. Research scores, not a detector.

All selection uses search images. Evaluation images and optional actual-FU
models are used only after the perturbation and output coordinate are frozen.
Inputs are pixels in [0, 1]; the caller owns model-specific preprocessing.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch


METHODS = ("request_guided", "post_only", "history_free", "random", "corner")


@dataclass(frozen=True)
class SearchConfig:
    epsilon: float = 8 / 255
    steps: int = 20
    step_size: float = 2 / 255
    restarts: int = 2
    batch_size: int = 32
    seed: int = 42

    def validate(self):
        if not 0 < self.epsilon <= 1 or not np.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("finite epsilon in (0,1] and positive step_size required")
        if min(self.steps, self.restarts, self.batch_size) < 1:
            raise ValueError("steps, restarts and batch_size must be positive")


def freeze(model, device):
    return model.to(device).eval().requires_grad_(False)


def margin(logits, target):
    """Smooth target-vs-rest logit margin, invariant to common logit shifts."""
    other = torch.cat((logits[:, :target], logits[:, target + 1:]), dim=1)
    return logits[:, target] - torch.logsumexp(other, dim=1)


def predict(model, x, batch_size):
    with torch.no_grad():
        result = torch.cat([model(b) for b in x.split(batch_size)])
    if result.ndim != 2 or result.shape[1] < 2 or not torch.isfinite(result).all():
        raise ValueError("finite [images, classes>=2] logits required")
    return result


def objective(method, pre, post, x, delta, target, clean_pre, clean_post):
    changed = (x + delta).clamp(0, 1)
    if method == "history_free":
        return margin(pre(changed), target) - margin(clean_pre, target)
    post_gain = margin(post(changed), target) - margin(clean_post, target)
    if method == "post_only":
        return post_gain
    return post_gain - (margin(pre(changed), target) - margin(clean_pre, target))


def search(pre, post, x, method, config):
    """Select target, restart and universal delta using x ONLY.

Random/corner get the same number of initial patterns and class scans, but no
optimization. Corner uses bounded additive 4x4 noise (not patch overwrite).
"""
    config.validate()
    if method not in METHODS:
        raise ValueError("unknown method")
    if x.ndim != 4 or len(x) < 1 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("finite NCHW pixels in [0,1] required")
    clean_pre = predict(pre, x, config.batch_size)
    clean_post = predict(post, x, config.batch_size)
    if clean_pre.shape != clean_post.shape:
        raise ValueError("model output shapes differ")
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    starts = [((torch.rand((1, *x.shape[1:]), generator=rng) * 2 - 1)
               * config.epsilon).to(x.device) for _ in range(config.restarts)]
    # Include the unperturbed point for optimized methods; random controls keep
    # all their random patterns. Remaining starts explore other activation cells.
    if method not in ("random", "corner"):
        starts[0] = torch.zeros_like(starts[0])
    best, trials = None, []
    for target in range(clean_pre.shape[1]):
        for restart, initial in enumerate(starts):
            delta = initial.clone()
            if method == "corner":
                mask = torch.zeros_like(delta)
                mask[:, :, -4:, -4:] = 1
                delta *= mask
            optimized = method not in ("random", "corner")
            for step in range(config.steps + 1 if optimized else 1):
                delta = delta.detach().requires_grad_(optimized and step < config.steps)
                total, grad = 0., torch.zeros_like(delta)
                for start in range(0, len(x), config.batch_size):
                    end = min(start + config.batch_size, len(x))
                    with torch.set_grad_enabled(delta.requires_grad):
                        score = objective(method, pre, post, x[start:end], delta, target,
                                          clean_pre[start:end], clean_post[start:end]).sum() / len(x)
                    total += float(score.detach())
                    if delta.requires_grad:
                        grad += torch.autograd.grad(score, delta)[0]
                if not np.isfinite(total) or not torch.isfinite(grad).all():
                    raise ValueError("non-finite probe objective/gradient")
                if best is None or total > best["search_objective"]:
                    best = dict(delta=delta.detach().clone(), target=target, restart=restart,
                                step=step, search_objective=total)
                if step == (config.steps if optimized else 0):
                    trials.append(dict(target=target, restart=restart, final_objective=total))
                if delta.requires_grad:
                    delta = (delta + config.step_size * grad.sign()).clamp(-config.epsilon, config.epsilon)
    best.update(method=method, trials=trials, config=asdict(config),
                selection_objective="pre_gain" if method == "history_free" else
                "post_gain" if method == "post_only" else "post_gain_minus_pre_gain")
    return best


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)


def evaluate(pre, post, x, result, batch_size):
    """Descriptive held-out metrics, never returned to search/selection."""
    target, delta = result["target"], result["delta"]
    logits = {"pre_clean": predict(pre, x, batch_size),
              "pre_probe": predict(pre, (x + delta).clamp(0, 1), batch_size),
              "post_clean": predict(post, x, batch_size),
              "post_probe": predict(post, (x + delta).clamp(0, 1), batch_size)}
    probs = {k: v.softmax(-1) for k, v in logits.items()}
    gain = ((margin(logits["post_probe"], target) - margin(logits["post_clean"], target))
            - (margin(logits["pre_probe"], target) - margin(logits["pre_clean"], target)))
    distributions = {}
    for name, p in probs.items():
        distributions[name] = dict(marginal_entropy=float(entropy(p.mean(0))),
            mean_predictive_entropy=float(entropy(p).mean()),
            target_probability=float(p[:, target].mean()),
            target_prediction_fraction=float((p.argmax(-1) == target).float().mean()))
    d = distributions
    collapse = ((d["pre_probe"]["marginal_entropy"] - d["post_probe"]["marginal_entropy"])
                - (d["pre_clean"]["marginal_entropy"] - d["post_clean"]["marginal_entropy"]))
    clean_disagreement = (probs["pre_clean"].argmax(-1) != probs["post_clean"].argmax(-1)).float().mean()
    metrics = dict(images=len(x), target=target, release_margin_mean=float(gain.mean()),
        release_margin_positive_fraction=float((gain > 0).float().mean()),
        excess_marginal_entropy_drop=collapse, clean_prediction_disagreement=float(clean_disagreement),
        distributions=distributions, accuracy=None, attack_success_rate=None,
        note="Unlabeled prediction fractions are not accuracy or ASR; positive scores do not prove an attack.")
    arrays = {k: v.detach().cpu().numpy() for k, v in probs.items()}
    arrays["release_margin_per_image"] = gain.detach().cpu().numpy()
    return metrics, arrays


def split_pool(images, ids, search_size, evaluation_size, seed):
    if search_size < 1 or evaluation_size < 1 or search_size + evaluation_size > len(images):
        raise ValueError("pool must contain enough images for disjoint positive split sizes")
    if len(ids) != len(images) or len(set(ids.tolist())) != len(ids):
        raise ValueError("unique IDs matching image count required")
    order = np.random.default_rng(seed).permutation(len(images))[:search_size + evaluation_size]
    a, b = order[:search_size], order[search_size:]
    # Different IDs can still refer to identical pixels. Reject this leakage.
    import hashlib
    def hashes(indices):
        return {hashlib.sha256(images[i].tobytes()).digest() for i in indices}
    if hashes(a) & hashes(b):
        raise ValueError("identical pixels cross the search/evaluation split")
    return a, b
