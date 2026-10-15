"""Fixed-protocol BadFU probe experiment; no FL training or FU execution.

Oracle inputs are diagnostic only. Reference directions are gradients on trusted
clean images. Coefficients use clean inputs only, with two-fold cross-fitting.
This reference span is a first-order model, not a calibrated benign population.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ('clean', 'oracle', 'brightness', 'flip', 'patch_tl', 'patch_tr', 'patch_bl', 'patch_br')
CONFIG = dict(schema=1, seed=20261015, windows=[0, 10, 20],
              blind_patch_size=4, brightness=.85, svd_rcond=1e-5,
              reference_per_class=8, primary='paired centered-logit effects',
              fit='clean-input coordinates only, class-stratified two-fold cross-fit',
              bn='recorded pre-round buffers frozen for all measurements',
              generic_family='maximum over all four predeclared patch locations',
              normal_calibration='unavailable: no matched independent benign episodes')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def folds_by_class(labels):
    labels = np.asarray(labels)
    fold = np.zeros(len(labels), dtype=int)
    for c in np.unique(labels):
        ids = np.flatnonzero(labels == c)
        if len(ids) < 2:
            raise ValueError('each class needs inputs in both folds')
        fold[ids[1::2]] = 1
    return fold


def crossfit_residual(effects, references, folds, rcond=1e-5):
    """effects[client,view,input,output], refs[direction,view,input,output].

    For each held-out fold, fit B_clean q to the other fold's clean responses.
    Predict every view on held-out inputs using the SAME q. Oracle responses
    never enter fitting; no client role labels or FU results are used.
    """
    e, b = np.asarray(effects, float), np.asarray(references, float)
    if e.ndim != 4 or b.ndim != 4 or e.shape[1:] != b.shape[1:]:
        raise ValueError('compatible client/direction,view,input,output arrays required')
    fold = np.asarray(folds)
    if fold.shape != (e.shape[2],) or set(fold.tolist()) != {0, 1}:
        raise ValueError('two nonempty input folds required')
    residual = np.empty_like(e)
    diagnostics = []
    for heldout in (0, 1):
        train, test = np.flatnonzero(fold != heldout), np.flatnonzero(fold == heldout)
        design = b[:, 0, train, :].reshape(len(b), -1).T
        u, s, vt = np.linalg.svd(design, full_matrices=False)
        keep = s > rcond * s.max(initial=0)
        inverse = (vt[keep].T / s[keep]) @ u[:, keep].T
        coefficients = e[:, 0, train, :].reshape(len(e), -1) @ inverse.T
        predicted = np.einsum('kd,dvnc->kvnc', coefficients, b[:, :, test, :])
        residual[:, :, test, :] = e[:, :, test, :] - predicted
        diagnostics.append(dict(fold=heldout, rank=int(keep.sum()), singular_values=s.tolist(),
                                coefficient_norms=np.linalg.norm(coefficients, axis=1).tolist()))
    return residual, diagnostics


def coordinates(raw, baseline):
    """Linearized log-probability and centered-logit changes from raw-logit JVP."""
    z = baseline - baseline.max(-1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    return raw - (raw * prob).sum(-1, keepdims=True), raw - raw.mean(-1, keepdims=True)


def make_views(archive, record_root, data_root):
    import torch
    from torchvision.datasets import CIFAR10
    from badfu_common import TRANSFORM, BackdoorDataset
    # Author-created local data file, containing image paths and labels.
    record_path = record_root / 'pert_result.pt'
    record = torch.load(record_path, map_location='cpu', weights_only=False)
    bd = BackdoorDataset(record['bd_test']['bd_data_container']['data_dict'], record_root=str(record_root))
    clean = CIFAR10(str(data_root), train=False, download=False, transform=TRANSFORM)
    validation_ids = set(archive['validation_ids'])
    ids = [i for i, key in enumerate(bd.keys)
           if int(key) not in validation_ids and clean.targets[int(key)] != archive['target']]
    ids = ids[:len(archive['backdoor_original_labels'])]
    originals = torch.stack([clean[int(bd.keys[i])][0] for i in ids])
    original_labels = torch.tensor([clean.targets[int(bd.keys[i])] for i in ids])
    actual_bd = torch.stack([bd[i][0] for i in ids])
    if not torch.equal(original_labels, archive['backdoor_original_labels']):
        raise ValueError('oracle source-label ordering differs from saved evaluation data')
    if not torch.equal(actual_bd, archive['backdoor_eval']['images']):
        raise ValueError('author backdoor pixels differ from saved evaluation inputs')
    # Recover and verify this fixed overwrite trigger, never a guessed patch.
    mask = (originals - actual_bd).abs().amax(0) > 1e-6
    if not mask.any():
        raise ValueError('no oracle pattern found')
    pattern = actual_bd[0]
    rebuilt = originals.clone()
    rebuilt[:, mask] = pattern[mask]
    error = float((rebuilt - actual_bd).abs().max())
    if error > 1e-6:
        raise ValueError('this collector only supports a verified fixed overwrite trigger')
    x = archive['validation']['images']
    labels = archive['validation']['labels']
    restored_clean = torch.stack([clean[int(i)][0] for i in archive['validation_ids']])
    if not torch.equal(restored_clean, x):
        raise ValueError('clean validation source differs')
    mean = torch.tensor([.4914, .4822, .4465]).reshape(3, 1, 1)
    std = torch.tensor([.2023, .1994, .2010]).reshape(3, 1, 1)
    pixels = x * std + mean
    oracle = x.clone()
    oracle[:, mask] = pattern[mask]
    views = [x, oracle, (pixels.mul(CONFIG['brightness']).clamp(0, 1) - mean) / std, x.flip(-1)]
    rng = torch.Generator().manual_seed(CONFIG['seed'])
    patch = torch.randint(0, 2, (3, 4, 4), generator=rng).float()
    for row, col in ((0, 0), (0, 28), (28, 0), (28, 28)):
        modified = pixels.clone()
        modified[:, :, row:row+4, col:col+4] = patch
        views.append((modified - mean) / std)
    reference_ids = []
    for c in range(10):
        eligible = torch.nonzero(archive['clean_eval']['labels'] == c).flatten()
        if len(eligible) < CONFIG['reference_per_class']:
            raise ValueError('insufficient disjoint clean reference examples')
        reference_ids.extend(eligible[:CONFIG['reference_per_class']].tolist())
    ref = {key: val[reference_ids] for key, val in archive['clean_eval'].items()}
    if validation_ids & {archive['clean_eval_ids'][i] for i in reference_ids}:
        raise ValueError('reference examples overlap response probes')
    meta = dict(record_sha256=sha(record_path), oracle_reconstruction_max_error=error,
                oracle_verified_pairs=len(ids), oracle_spatial_pixels=int(mask.any(0).sum()),
                validation_ids=archive['validation_ids'],
                reference_ids=[archive['clean_eval_ids'][i] for i in reference_ids],
                blind_view_definition_independent_of_oracle=True,
                labels=labels.tolist(), probe_hash=hashlib.sha256(torch.stack(views).numpy().tobytes()).hexdigest())
    return torch.stack(views), labels.numpy(), ref, meta


def collect(a):
    import torch
    from torch.func import functional_call, jvp
    sys.path.insert(0, str(ROOT / 'BadFU_src/BadFU-main'))
    from request_pipeline import load_tensor, load_event
    from badfu_common import ResNet18
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG['seed'])
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    archive = load_tensor(manifest['dataset'])
    if archive['counts'] != manifest['counts']:
        raise ValueError('sample counts differ')
    xs, labels, reference, data_meta = make_views(archive, a.record_root, a.data_root)
    config = {**CONFIG, 'views': list(VIEWS), 'rounds': min(a.rounds, len(manifest['rounds'])),
              'manifest_sha256': sha(a.manifest), 'code_sha256': sha(__file__),
              'data': data_meta, 'command': vars(a)}
    config['command'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    a.out.mkdir(parents=True, exist_ok=False)
    write_new(a.out / 'protocol.json', config)  # Frozen before any model response is measured.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = ResNet18(pretrained=False).to(device).eval()
    count = np.asarray(manifest['counts'], float)
    weights = count / count.sum()
    v, n = xs.shape[:2]
    flat_x = xs.flatten(0, 1).to(device)
    ref_x, ref_y = reference['images'].to(device), reference['labels'].to(device)
    finite_lp, finite_cl, tangent_lp, tangent_cl, reference_lp, reference_cl = [], [], [], [], [], []
    source_rounds, checks = [], []
    started = time.perf_counter()

    def evaluate(params, buffers):
        with torch.no_grad():
            return torch.cat([functional_call(model, (params, buffers), (bx,))
                              for bx in flat_x.split(a.batch_size)]).cpu().numpy().reshape(v, n, 10)

    def derivative(params, buffers, direction):
        with torch.no_grad():
            return torch.cat([jvp(lambda ps: functional_call(model, (ps, buffers), (bx,)),
                                  (params,), (direction,))[1]
                              for bx in flat_x.split(a.batch_size)]).cpu().numpy().reshape(v, n, 10)

    for t, record in enumerate(manifest['rounds'][:a.rounds]):
        event, after = load_event(record, manifest['counts'])
        model.load_state_dict(event['gm'])
        params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        baseline = evaluate(params, buffers)
        base_lp = baseline - np.logaddexp.reduce(baseline, axis=-1, keepdims=True)
        base_cl = baseline - baseline.mean(-1, keepdims=True)
        f_lp, f_cl, j_lp, j_cl = [], [], [], []
        for j, state in enumerate(event['cms']):
            direction = {key: (state[key].to(device) - value.detach()) * weights[j]
                         for key, value in params.items()}
            measured = evaluate({key: value + direction[key] for key, value in params.items()}, buffers)
            f_lp.append(measured - np.logaddexp.reduce(measured, axis=-1, keepdims=True) - base_lp)
            f_cl.append(measured - measured.mean(-1, keepdims=True) - base_cl)
            tangent = derivative(params, buffers, direction)
            lp, cl = coordinates(tangent, baseline)
            j_lp.append(lp); j_cl.append(cl)
            if t == 0 and j == 0:
                epsilon = .01
                plus = evaluate({key: value + epsilon*direction[key] for key, value in params.items()}, buffers)
                minus = evaluate({key: value - epsilon*direction[key] for key, value in params.items()}, buffers)
                numerical = (plus-minus)/(2*epsilon)
                checks.append(dict(kind='JVP_vs_central_difference', epsilon=epsilon,
                    relative_error=float(np.linalg.norm(numerical-tangent)/max(np.linalg.norm(tangent), 1e-12))))
            del direction
        b_lp, b_cl = [], []
        for c in range(10):
            loss = torch.nn.functional.cross_entropy(model(ref_x[ref_y == c]), ref_y[ref_y == c])
            gradients = torch.autograd.grad(loss, tuple(params.values()))
            norm = sum(g.square().sum() for g in gradients).sqrt().clamp_min(1e-12)
            direction = {key: -g.detach()/norm for key, g in zip(params, gradients)}
            tangent = derivative(params, buffers, direction)
            lp, cl = coordinates(tangent, baseline)
            b_lp.append(lp); b_cl.append(cl)
            del gradients, loss, direction
        for destination, values in ((finite_lp, f_lp), (finite_cl, f_cl), (tangent_lp, j_lp),
                                    (tangent_cl, j_cl), (reference_lp, b_lp), (reference_cl, b_cl)):
            destination.append(np.stack(values))
        source_rounds.append(record)
        del event, after
        print(f'Probes and clean references: round {t+1}/{config["rounds"]}, {time.perf_counter()-started:.1f}s', flush=True)
    np.savez_compressed(a.out/'effects.npz', finite_logprob=np.stack(finite_lp), finite_logit=np.stack(finite_cl),
                        tangent_logprob=np.stack(tangent_lp), tangent_logit=np.stack(tangent_cl),
                        reference_logprob=np.stack(reference_lp), reference_logit=np.stack(reference_cl),
                        labels=labels)
    write_new(a.out/'collection.json', dict(protocol_sha256=sha(a.out/'protocol.json'),
        cache_sha256=sha(a.out/'effects.npz'), source_rounds=source_rounds, dataset=manifest['dataset'],
        training_calls=0, model_unlearning_calls=0, clean_reference_gradient_calls=10*len(source_rounds),
        measurement='positive finite FL update and local Jacobian-vector product; frozen recorded BN',
        diagnostic_checks=checks, seconds=time.perf_counter()-started, torch=torch.__version__,
        device=device, hardware=torch.cuda.get_device_name() if device.startswith('cuda') else None,
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in
                      (ROOT/'BadFU_src/BadFU-main/request_pipeline.py', ROOT/'BadFU_src/BadFU-main/badfu_common.py')}))


def analyze(folder):
    from hrac import inter_profile
    cache = np.load(folder/'effects.npz', allow_pickle=False)
    meta = json.loads((folder/'collection.json').read_text(encoding='utf-8'))
    if sha(folder/'effects.npz') != meta['cache_sha256']:
        raise ValueError('cache hash mismatch')
    fold = folds_by_class(cache['labels'])
    rows, fit_diagnostics = [], {}
    for coordinate in ('logprob', 'logit'):
        tangent = cache[f'tangent_{coordinate}']
        reference = cache[f'reference_{coordinate}']
        residual, diagnostic = zip(*(crossfit_residual(e, b, fold, CONFIG['svd_rcond'])
                                     for e, b in zip(tangent, reference)))
        fit_diagnostics[coordinate] = diagnostic
        for method, effects in (('finite', cache[f'finite_{coordinate}']), ('tangent', tangent),
                                ('residual', np.stack(residual))):
            for contrast in ('raw', 'paired'):
                values = effects if contrast == 'raw' else effects - effects[:, :, :1]
                profiles = {}
                for view, name in enumerate(VIEWS):
                    if contrast == 'paired' and view == 0:
                        continue
                    h = values[:, :, view].mean(2)
                    scores = np.stack([inter_profile(h, np.eye(h.shape[1])[j], CONFIG['windows'])
                                       for j in range(h.shape[1])])
                    profiles[name] = scores
                profiles['blind_patch_family'] = np.maximum.reduce([profiles[s] for s in VIEWS[4:]])
                profiles['benign_transform_family'] = np.maximum.reduce([profiles[s] for s in VIEWS[2:4]])
                for view, scores in profiles.items():
                    ranking = sorted([dict(unit=j, score=float(s.max()), coordinate=int(s.argmax()),
                                           scores_by_class=s.tolist()) for j, s in enumerate(scores)], key=lambda x: -x['score'])
                    rows.append(dict(method=method, coordinate=coordinate, contrast=contrast, view=view,
                                     ranking=ranking, requester5_rank=next(i+1 for i,r in enumerate(ranking) if r['unit']==5),
                                     requester5_target0=float(scores[5,0]), target0_rank=1+int((scores[:,0]>scores[5,0]).sum()),
                                     oracle_view=view=='oracle'))
    report = dict(schema=1, analysis_sha256=sha(__file__), protocol_sha256=sha(folder/'protocol.json'),
                  cache_sha256=meta['cache_sha256'], rows=rows, fit_diagnostics=fit_diagnostics,
                  calibrated=False, independent_benign_episodes=0,
                  limitations=['One attacked development trajectory; no calibrated FPR/TPR.',
                    'Oracle pattern is diagnostic, never an operational detection input.',
                    'Normal references approximate single-step clean class learning, not all non-IID trajectories.',
                    'Clean-only cross-fitting does not imply independence between FL rounds.',
                    'Projected/paired effects are observations, not exact causal FU contributions.',
                    'Same existing checkpoints were inspected previously; results are exploratory.'])
    write_new(folder/'analysis.json', report)
    selected = [r for r in rows if
                (r['coordinate']=='logprob' and r['method']=='finite' and r['contrast']=='raw' and r['view'] in ('clean','oracle'))
                or (r['coordinate']=='logit' and r['contrast']=='paired' and r['view'] in ('oracle','blind_patch_family','benign_transform_family'))]
    for r in selected:
        print(r['method'],r['coordinate'],r['contrast'],r['view'], 'requester rank',r['requester5_rank'],
              'top',r['ranking'][0]['unit'],r['ranking'][0]['coordinate'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run','analyze'))
    p.add_argument('--manifest', type=Path, default=ROOT/'artifacts/badfu/imported_traj_full/manifest.json')
    p.add_argument('--record-root', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    if a.rounds < 1 or a.batch_size < 1:
        p.error('positive rounds and batch size required')
    if a.mode == 'run':
        if a.record_root is None or a.data_root is None:
            p.error('run needs --record-root and --data-root for verified oracle pairs')
        collect(a)
    analyze(a.out)


if __name__ == '__main__':
    main()
