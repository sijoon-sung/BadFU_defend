"""Actual pre/post-FU inversion ablation. Not an implementation of NC or I-BAU.

Run: python -m reproduction.purification.purification_compare --config experiment.json
Only public development data enter inversion/repair. Attack evaluation is loaded
after all arms have been saved. No detector, no automatic novelty claim.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction.common.trajectory import sha256, json_write
from reproduction.purification.purification_math import (class_means, dual_step, margin,
    paired_margin_change, pattern_spaces, transform, upper_tail_weights)


class Pixels(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).float()[None, :, None, None])
        self.register_buffer('std', torch.tensor(std).float()[None, :, None, None])

    def forward(self, x):
        return self.model(((x - self.mean) / self.std).contiguous())


def factory(name):
    if name == 'badfu_cifar10':
        from torchvision.models import resnet18
        class BadFU(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = resnet18(weights=None)
                self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                self.model.maxpool = nn.Identity()
                self.model.fc = nn.Linear(self.model.fc.in_features, 10)

            def forward(self, x):
                return self.model(x)
        return BadFU()
    if name in ('fuba_cifar10', 'fuba_mnist'):
        from FUBA.model import resnet18, Net
        return resnet18() if name == 'fuba_cifar10' else Net()
    module, function = name.split(':')
    return getattr(importlib.import_module(module), function)()


def load_model(spec, config, device):
    model = factory(config['model'])
    state = torch.load(spec['path'], map_location='cpu', weights_only=True)
    for key in spec.get('key', '').split('.'):
        if key:
            state = state[key]
    model.load_state_dict(state, strict=True)
    return Pixels(model, config['mean'], config['std']).to(device).eval()


def load_pool(path):
    with np.load(path, allow_pickle=False) as data:
        images, labels, ids = (data[k].copy() for k in ('images', 'labels', 'ids'))
    if images.ndim != 4 or len(images) < 1 or images.dtype != np.uint8:
        raise ValueError('clean images must be nonempty uint8 NHWC')
    if labels.shape != (len(images),) or labels.dtype.kind not in 'iu' or (labels < 0).any():
        raise ValueError('integer labels[N] required')
    if ids.shape != labels.shape or ids.dtype.kind not in 'US' or len(set(ids.tolist())) != len(ids):
        raise ValueError('unique string ids[N] required')
    x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float() / 255
    return x, torch.from_numpy(labels.astype(np.int64)), ids


def reject_overlap(a, b):
    if set(a[2].tolist()) & set(b[2].tolist()):
        raise ValueError('development and test IDs overlap')
    hashes = lambda x: {hashlib.sha256(row.numpy().tobytes()).digest() for row in x}
    if hashes(a[0]) & hashes(b[0]):
        raise ValueError('development and test pixels overlap')


@torch.no_grad()
def logits(model, x, batch):
    result = torch.cat([model(chunk.to(next(model.parameters()).device)).cpu()
                        for chunk in x.split(batch)])
    if result.ndim != 2 or result.shape[1] < 2 or not torch.isfinite(result).all():
        raise ValueError('expected finite N x classes logits')
    return result


def targeted_margin(z, y, target):
    return margin(z, y, target)


def inversion_objective(pre, post, x, y, delta, target, method, pre_weight, correct_weight):
    return inversion_values(pre, post, x, y, delta, target, method, pre_weight, correct_weight).mean()


def inversion_values(pre, post, x, y, pattern, target, method, pre_weight, correct_weight,
                     clean_post=None, clean_pre=None):
    changed = transform(x, pattern)
    if clean_post is None:
        with torch.no_grad():
            clean_post = post(x)
    score = margin(post(changed), y, target) - margin(clean_post, y, target)
    if method in ('differential', 'differential_cvar'):
        before = pre(changed)
        if clean_pre is None:
            with torch.no_grad():
                clean_pre = pre(x)
        score = score - pre_weight * (margin(before, y, target) - margin(clean_pre, y, target))
        # Optional soft pre-correctness penalty, NOT a hard guarantee.
        score = score - correct_weight * F.cross_entropy(before, y, reduction='none')
    elif method != 'post_only':
        raise ValueError(method)
    return score


def invert(pre, post, x, y, method, cfg):
    """Same targets, starts, steps and image batches for both ablations.

    All non-source targets are searched; retain one best delta per target.
    True trigger and true attack target are not accepted by this function.
    """
    device = next(post.parameters()).device
    pre.eval().requires_grad_(False)
    post.eval().requires_grad_(False)
    classes = logits(post, x[:1], 1).shape[1]
    if int(y.max()) >= classes:
        raise ValueError('development labels outside model output range')
    patterns, records = [], []
    clean_post = logits(post, x, cfg['batch_size'])
    clean_pre = logits(pre, x, cfg['batch_size']) if method != 'post_only' else None
    spaces = pattern_spaces(x.shape[1:], cfg.get('probe_mode', 'additive'), cfg.get('patch_size', 4))
    for target in range(classes):
        use = y != target
        if not use.any():
            continue
        xx, yy = x[use], y[use]
        rng = torch.Generator().manual_seed(cfg['seed'])
        best_score, best_pattern, trials = -float('inf'), None, []
        for mode, position, mask in spaces:
            for restart in range(cfg['restarts']):
                delta = torch.rand((1, *x.shape[1:]), generator=rng).to(device)
                if mode == 'additive':
                    delta = (2 * delta - 1) * cfg['epsilon']
                space_best = -float('inf')
                for step in range(cfg['steps'] + 1):
                    delta = delta.detach().requires_grad_(step < cfg['steps'])
                    pattern = dict(mode=mode, delta=delta, mask=mask, position=position)
                    def values(start):
                        end = start + cfg['batch_size']
                        return inversion_values(pre, post, xx[start:end].to(device), yy[start:end].to(device),
                            pattern, target, method, cfg['pre_weight'], cfg['correct_weight'],
                            clean_post[use][start:end].to(device),
                            clean_pre[use][start:end].to(device) if clean_pre is not None else None)
                    # Compute the empirical tail on ALL search inputs, never separately per batch.
                    with torch.no_grad():
                        scores = torch.cat([values(s) for s in range(0, len(xx), cfg['batch_size'])])
                    weights = upper_tail_weights(scores, cfg.get('tail_mass', .25) if method == 'differential_cvar' else 1.)
                    total = (scores * weights).sum().item()
                    space_best = max(space_best, total)
                    if total > best_score:
                        best_score = total
                        best_pattern = dict(mode=mode, delta=delta.detach().cpu().clone(),
                                            mask=mask, position=position)
                    if step < cfg['steps']:
                        gradient = torch.zeros_like(delta)
                        for start in range(0, len(xx), cfg['batch_size']):
                            score = (values(start) * weights[start:start+cfg['batch_size']]).sum()
                            gradient += torch.autograd.grad(score, delta)[0]
                        if not torch.isfinite(gradient).all():
                            raise ValueError('nonfinite inversion gradient')
                        rate = cfg.get('patch_step_size', .1) if mode == 'patch' else cfg['step_size']
                        lo, hi = (0., 1.) if mode == 'patch' else (-cfg['epsilon'], cfg['epsilon'])
                        delta = (delta + rate * gradient.sign()).clamp(lo, hi)
                trials.append(dict(mode=mode, position=position, restart=restart, score=space_best))
        patterns.append(best_pattern)
        records.append(dict(target=target, search_score=best_score, mode=best_pattern['mode'],
                            position=best_pattern['position'], trials=trials))
        print(f'  {method}: target {target}, {best_pattern["position"]}, score={best_score:.4f}', flush=True)
    return patterns, records


def repair(post, x, y, patterns, cfg, constrained=False, validation=None, trace=None):
    model = copy.deepcopy(post).eval().requires_grad_(True)
    # Keep BN running statistics fixed for all arms, including clean_ft.
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=.9)
    device = next(model.parameters()).device
    rng = torch.Generator().manual_seed(cfg['seed'])
    classes = logits(post, x[:1], 1).shape[1]
    ref = class_means(F.cross_entropy(logits(post, x, cfg['batch_size']), y, reduction='none'), y, classes) if constrained else None
    dual = torch.zeros(classes)
    if constrained:
        if validation is None:
            raise ValueError('independent clean validation data required for constrained repair')
        vx, vy = validation[:2]
        vref = class_means(F.cross_entropy(logits(post, vx, cfg['batch_size']), vy, reduction='none'), vy, classes)
        counts = torch.bincount(y, minlength=classes).float()
        selected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        selected_epoch = 0
    for epoch in range(cfg['epochs']):
        order = torch.randperm(len(x), generator=rng)
        for start in range(0, len(x), cfg['batch_size']):
            ix = order[start:start+cfg['batch_size']]
            xb, yb = x[ix].to(device), y[ix].to(device)
            if patterns:
                chosen = torch.randint(len(patterns), (len(ix),), generator=rng)
                changed = torch.cat([transform(xb[j:j+1], patterns[int(i)]) for j, i in enumerate(chosen)])
            else:
                changed = xb
                # Consume identical RNG draws to keep minibatch order matched.
                torch.randint(1, (len(ix),), generator=rng)
            optimizer.zero_grad(set_to_none=True)
            clean_z = model(xb)
            loss = (F.cross_entropy(clean_z, yb) +
                    cfg['repair_weight'] * F.cross_entropy(model(changed), yb)) / (1 + cfg['repair_weight'])
            if constrained:
                # Unbiased minibatch gradient of sum_c dual_c * class-mean CE_c.
                weight = (dual * len(y) / counts)[yb.cpu()].to(device)
                loss = loss + (weight * F.cross_entropy(clean_z, yb, reduction='none')).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite repair loss')
            loss.backward()
            optimizer.step()
        if constrained:
            ce = class_means(F.cross_entropy(logits(model, x, cfg['batch_size']), y, reduction='none'), y, classes)
            vce = class_means(F.cross_entropy(logits(model, vx, cfg['batch_size']), vy, reduction='none'), vy, classes)
            violation = ce - ref - cfg['class_ce_tolerance']
            dual = dual_step(dual, violation, cfg['dual_lr'], cfg['dual_cap'])
            feasible = bool((violation <= 1e-7).all() and (vce-vref <= cfg['class_ce_tolerance']+1e-7).all())
            if feasible:
                selected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                selected_epoch = epoch + 1
            if trace is not None:
                trace.append(dict(epoch=epoch+1, class_ce=ce.tolist(), validation_ce=vce.tolist(),
                    dual=dual.tolist(), feasible_on_development_and_validation=feasible))
    if constrained:
        model.load_state_dict(selected, strict=True)
        if trace is not None:
            trace.append(dict(selected_epoch=selected_epoch, fallback_to_post=selected_epoch == 0,
                              certificate='empirical class CE on observed public splits only'))
    return model.eval().requires_grad_(False)


def evaluate(model, test, attack_images, targets, batch):
    x, y, _ = test
    clean = logits(model, x, batch).argmax(1)
    attack = logits(model, attack_images, batch).argmax(1)
    eligible = y != targets
    conditional = eligible & (clean == y)
    rate = lambda mask, values: float(values[mask].float().mean()) if mask.any() else None
    return dict(clean_accuracy=float((clean == y).float().mean()),
        class_accuracy={str(c): rate(y == c, clean == y) for c in sorted(y.unique().tolist())},
        class_counts={str(c): int((y == c).sum()) for c in sorted(y.unique().tolist())},
        asr=rate(eligible, attack == targets),
        asr_n=int(eligible.sum()),
        asr_clean_correct=rate(conditional, attack == targets),
        asr_clean_correct_n=int(conditional.sum()),
        triggered_original_accuracy=rate(eligible, attack == y))


def run(config, out, attack_loader=None, resume=False):
    defaults = dict(seed=42, device='cpu', epsilon=8/255, steps=40, step_size=2/255,
        restarts=3, batch_size=32, epochs=3, lr=.001, pre_weight=1., correct_weight=1.,
        repair_weight=1., clean_tolerance=.02, probe_mode='additive', patch_size=4,
        patch_step_size=.1, tail_mass=.25, math_ablation=False,
        class_ce_tolerance=.05, dual_lr=.5, dual_cap=100.)
    cfg = {**defaults, **config}
    for key in ('steps', 'restarts', 'batch_size', 'epochs'):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('epsilon', 'step_size', 'lr', 'pre_weight', 'correct_weight', 'repair_weight', 'clean_tolerance'):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'invalid {key}')
    if not 0 < cfg['epsilon'] <= 1 or min(cfg['step_size'], cfg['lr']) <= 0:
        raise ValueError('invalid perturbation or learning step')
    if not 0 < cfg['tail_mass'] <= 1 or not isinstance(cfg['math_ablation'], bool):
        raise ValueError('invalid tail mass / math ablation flag')
    for key in ('patch_step_size', 'class_ce_tolerance', 'dual_lr', 'dual_cap'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'invalid {key}')
    if cfg.get('post_kind') != 'actual_fu' or not cfg.get('request_description'):
        raise ValueError('actual_fu and a description of the matched deletion request required')
    if not cfg['std'] or len(cfg['mean']) != len(cfg['std']) or not np.isfinite(cfg['mean']).all() or not np.isfinite(cfg['std']).all() or min(cfg['std']) <= 0:
        raise ValueError('explicit valid preprocessing required')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    out = Path(out)
    existed = out.exists()
    out.mkdir(parents=True, exist_ok=resume)
    if existed and not (out/'protocol.json').is_file():
        raise ValueError('resume requires an experiment protocol; choose a new output folder')
    dev, test = load_pool(cfg['development']), load_pool(cfg['test'])
    reject_overlap(dev, test)
    validation = load_pool(cfg['validation']) if cfg.get('validation') else None
    if validation is not None:
        reject_overlap(dev, validation)
        reject_overlap(validation, test)
    if cfg['math_ablation'] and validation is None:
        raise ValueError('math ablation requires independent public validation data')
    if (dev[0].shape[1:] != test[0].shape[1:] or dev[0].shape[1] != len(cfg['mean'])
            or (validation is not None and validation[0].shape[1:] != dev[0].shape[1:])):
        raise ValueError('image shapes / preprocessing channels differ')
    # Test inputs are read for leakage checks only; never supplied to fitting.
    pre = load_model(cfg['pre'], cfg, cfg['device']).requires_grad_(False)
    post = load_model(cfg['post'], cfg, cfg['device']).requires_grad_(False)
    paths = [cfg['pre']['path'], cfg['post']['path'], cfg['development'], cfg['test']]
    if cfg.get('validation'):
        paths.append(cfg['validation'])
    references = cfg.get('references', {})
    methods = ['clean_ft', 'post_only', 'differential']
    if cfg['math_ablation']:
        methods += ['differential_cvar', 'differential_cvar_constrained']
    if set(references) & set(['pre', 'post'] + methods):
        raise ValueError('reference name collides with comparison arms')
    paths += [s['path'] for s in references.values()]
    protocol = dict(status='running', config=cfg, input_hashes={str(p): sha256(p) for p in paths},
        source_hash=sha256(__file__), math_source_hash=sha256(Path(__file__).with_name('purification_math.py')), torch_version=torch.__version__,
        model_pair_verified=False, note='Pair provenance is user supplied; hashes do not prove valid unlearning.',
        bn_policy='eval mode, frozen running statistics',
        budget='same target/restart/step counts; differential costs extra pre-model passes',
        official_baselines=False, evaluation_used_for_selection=False,
        completed_arms={},
        forgetting_guarantee='not_measured; source-pair consistency is not deletion correctness',
        selection='last epoch satisfying class CE on development AND validation for constrained arm; fixed final epoch otherwise')
    if existed:
        previous = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        for key in ('config', 'input_hashes', 'source_hash', 'math_source_hash', 'torch_version'):
            if previous.get(key) != protocol[key]:
                raise ValueError(f'resume mismatch in {key}; use a new output folder')
        if previous.get('status') == 'complete':
            if any(sha256(out/name) != digest for name, digest in previous['output_hashes'].items()):
                raise ValueError('completed output modified')
            if attack_loader is None:
                if sha256(cfg['attack_evaluation']) != previous['attack_evaluation']['sha256']:
                    raise ValueError('completed evaluation input modified')
            else:
                # All repairs are already frozen, so this cannot leak into fitting.
                _, current_provenance = attack_loader(test)
                if current_provenance != previous['attack_evaluation']:
                    raise ValueError('completed evaluation provenance modified')
            print('Reusing completed comparison after input/source/output checksum verification.', flush=True)
            return json.loads((out/'results.json').read_text(encoding='utf-8'))
        protocol['completed_arms'] = previous.get('completed_arms', {})
    json_write(out/'protocol.json', protocol)
    for method in methods:
        if method in protocol['completed_arms']:
            if any(sha256(out/name) != digest for name, digest in protocol['completed_arms'][method].items()):
                raise ValueError('completed arm modified; choose a new output folder')
            print(f'Reusing verified {method}.', flush=True)
            continue
        print(f'Fitting {method}; attack evaluation remains unopened.', flush=True)
        began = time.perf_counter()
        if method == 'differential_cvar_constrained':
            # Exact same patterns: isolates the repair constraint, no second search.
            patterns = torch.load(out/'differential_cvar_patterns.pt', weights_only=True)
            records = json.loads((out/'differential_cvar_search.json').read_text())['records']
        else:
            patterns, records = ([], []) if method == 'clean_ft' else invert(pre, post, *dev[:2], method, cfg)
        inversion_seconds = time.perf_counter() - began
        trace = []
        repaired = repair(post, *dev[:2], patterns, cfg, constrained=method.endswith('_constrained'),
                          validation=validation, trace=trace)
        torch.save(repaired.cpu().state_dict(), out/f'{method}.pt')
        torch.save(patterns, out/f'{method}_patterns.pt')
        json_write(out/f'{method}_search.json', dict(records=records,
            inversion_seconds=inversion_seconds, total_seconds=time.perf_counter()-began,
            reused_patterns_from='differential_cvar' if method.endswith('_constrained') else None,
            repair_trace=trace))
        protocol['completed_arms'][method] = {name: sha256(out/name) for name in
            (f'{method}.pt', f'{method}_patterns.pt', f'{method}_search.json')}
        json_write(out/'protocol.json', protocol)
        del repaired
    # Load the true-trigger evaluation only after every arm is frozen on disk.
    if attack_loader is None:
        with np.load(cfg['attack_evaluation'], allow_pickle=False) as data:
            raw, ids, targets = (data[k].copy() for k in ('images', 'ids', 'targets'))
        attack_provenance = dict(sha256=sha256(cfg['attack_evaluation']), kind='supplied_npz')
    else:
        data, attack_provenance = attack_loader(test)
        raw, ids, targets = (data[k] for k in ('images', 'ids', 'targets'))
    if not np.array_equal(ids, test[2]) or raw.shape != (len(test[0]), *test[0].shape[2:], test[0].shape[1]):
        raise ValueError('attack images must correspond to test images in exactly the same order')
    if raw.dtype == np.uint8:
        raw = raw.astype(np.float32)/255
    elif raw.dtype.kind != 'f':
        raise ValueError('attack images require uint8 or float pixels before normalization')
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite attack images')
    attack = torch.from_numpy(raw.transpose(0, 3, 1, 2).copy()).float()
    classes = logits(post, test[0][:1], 1).shape[1]
    if targets.shape != (len(test[0]),) or targets.dtype.kind not in 'iu' or (targets < 0).any() or (targets >= classes).any() or int(test[1].max()) >= classes:
        raise ValueError('invalid test labels / attack targets')
    targets = torch.from_numpy(targets.astype(np.int64))
    results = {name: evaluate(model, test, attack, targets, cfg['batch_size'])
               for name, model in (('pre', pre), ('post', post))}
    reference_predictions = {}
    for name, spec in references.items():
        model = load_model(spec, cfg, cfg['device']).requires_grad_(False)
        results[name] = evaluate(model, test, attack, targets, cfg['batch_size'])
        reference_predictions[name] = logits(model, test[0], cfg['batch_size']).argmax(1)
        del model
    # This is a diagnostic agreement, not a certificate of distributional unlearning.
    results['post']['clean_agreement_with_references'] = {
        name: float((logits(post, test[0], cfg['batch_size']).argmax(1) == pred).float().mean())
        for name, pred in reference_predictions.items()}
    for method in methods:
        model = copy.deepcopy(post)
        model.load_state_dict(torch.load(out/f'{method}.pt', map_location=cfg['device'], weights_only=True))
        results[method] = evaluate(model, test, attack, targets, cfg['batch_size'])
        results[method]['clean_drop_vs_post'] = results['post']['clean_accuracy'] - results[method]['clean_accuracy']
        results[method]['within_clean_tolerance'] = results[method]['clean_drop_vs_post'] <= cfg['clean_tolerance']
        results[method]['worst_class_drop_vs_post'] = max(
            results['post']['class_accuracy'][c] - v for c, v in results[method]['class_accuracy'].items())
        # Held-out inversion behavior: no target or pattern is reselected here.
        patterns = torch.load(out/f'{method}_patterns.pt', weights_only=True)
        records = json.loads((out/f'{method}_search.json').read_text())['records']
        probes = []
        for pattern, record in zip(patterns, records):
            t = record['target']
            eligible = test[1] != t
            if not eligible.any():
                continue
            changed = transform(test[0][eligible], pattern)
            yp = test[1][eligible]
            before = logits(pre, changed, cfg['batch_size']).argmax(1)
            after = logits(post, changed, cfg['batch_size']).argmax(1)
            fixed = logits(model, changed, cfg['batch_size']).argmax(1)
            probes.append(dict(target=t, n=len(yp), pre_accuracy=float((before == yp).float().mean()),
                repaired_target_rate=float((fixed == t).float().mean()),
                repaired_original_accuracy=float((fixed == yp).float().mean()),
                post_target_rate=float((after == t).float().mean()),
                pre_correct_post_target_rate=float(((before == yp) & (after == t)).float().mean()),
                pre_non_target_post_target_rate=float(((before != t) & (after == t)).float().mean())))
        results[method]['heldout_probes'] = probes
        del model
    json_write(out/'results.json', results)
    def percent(value):
        return 'N/A' if value is None else f'{100*value:.2f}%'
    lines = ['# Purification comparison', '',
        'Actual model-pair provenance is supplied by the experiment owner.',
        'These are matched-objective ablations, not official NC/I-BAU implementations.', '',
        '| Arm | Clean accuracy | ASR (non-target originals) | Triggered original accuracy | Clean tolerance |',
        '| --- | ---: | ---: | ---: | --- |']
    for name, row in results.items():
        lines.append(f"| {name} | {percent(row['clean_accuracy'])} | {percent(row['asr'])} | "
                     f"{percent(row['triggered_original_accuracy'])} | {row.get('within_clean_tolerance', 'reference')} |")
    lines += ['', 'Do not select settings using these held-out results. Repeat on independent',
        'attack and benign model pairs. A high pre ASR does not demonstrate dormant activation.',
        'A lower ASR alone is insufficient if clean accuracy deteriorates.',
        'No unlearning guarantee or historical-information advantage is established here.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    protocol.update(status='complete', attack_evaluation=attack_provenance,
        output_hashes={p.name: sha256(p) for p in out.iterdir() if p.name != 'protocol.json'})
    json_write(out/'protocol.json', protocol)
    print('\n'.join(lines), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    # All data paths are relative to the config, not the shell working directory.
    for key in ('development', 'validation', 'test', 'attack_evaluation'):
        if key in config:
            config[key] = str((args.config.parent/config[key]).resolve())
    for key in ('pre', 'post'):
        config[key]['path'] = str((args.config.parent/config[key]['path']).resolve())
    for spec in config.get('references', {}).values():
        spec['path'] = str((args.config.parent/spec['path']).resolve())
    run(config, args.out, resume=args.resume)


if __name__ == '__main__':
    main()
