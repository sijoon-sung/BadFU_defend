"""Label-free public-image probes for saved BadFU ResNet18 trajectories.

No client dataset, oracle trigger, target label, pseudo-label, training or FU is
used. Public probe images, preprocessing, per-client states and FedAvg weights
remain required. Rankings are exploratory; this runner does not calibrate FPR.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MEAN = [.4914, .4822, .4465]
STD = [.2023, .1994, .2010]
WINDOWS = (0, 10, 20)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def load_pool(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'images', 'ids'}:
            raise ValueError('probe pool must contain only images and ids; labels/targets are not accepted')
        images, ids = data['images'], data['ids']
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError('expected uint8 RGB pixels [N,32,32,3]')
    if ids.dtype.kind != 'U' or ids.shape != (len(images),) or len(set(ids.tolist())) != len(ids):
        raise ValueError('unique string image IDs required')
    return images, ids


def prepare(a):
    """Export pixels only; neither class folders nor CIFAR targets select images."""
    if a.out.exists() or a.out.with_suffix('.json').exists():
        raise FileExistsError('choose a new image-pool output')
    if a.cifar_root is not None:
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(str(a.cifar_root), train=False, download=False)
        images = dataset.data  # Do not access .targets or __getitem__ (which returns labels).
        ids = np.array([f'cifar10:test:{i}' for i in range(len(images))])
        source = dict(kind='CIFAR10 test pixels', root=str(a.cifar_root.resolve()),
                      note='The upstream archive has labels; this exporter discards them without selection.')
    else:
        from PIL import Image, ImageOps
        files = sorted(p for p in a.image_dir.rglob('*') if p.is_file()
                       and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'})
        if not files:
            raise ValueError('no image files found')
        pixels = []
        for path in files:
            with Image.open(path) as im:
                pixels.append(np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize(
                    (32, 32), Image.Resampling.BILINEAR)).copy())
        images = np.stack(pixels)
        ids = np.array([p.relative_to(a.image_dir).as_posix() for p in files])
        source = dict(kind='image files; path names are identifiers only', root=str(a.image_dir.resolve()),
                      resize='RGB, EXIF transpose, bilinear 32x32')
    excluded = set()
    if a.exclude_ids is not None:
        exclusion = json.loads(a.exclude_ids.read_text(encoding='utf-8'))
        if not isinstance(exclusion, list) or any(not isinstance(x, str) for x in exclusion):
            raise ValueError('exclude-ids must be a JSON array of image ID strings')
        excluded = set(exclusion)
        if not excluded.issubset(set(ids.tolist())):
            raise ValueError('exclusion contains IDs absent from this source')
    keep = np.array([s not in excluded for s in ids])
    if not keep.any():
        raise ValueError('empty probe pool')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('xb') as f:
        np.savez_compressed(f, images=images[keep], ids=ids[keep])
    save_json(a.out.with_suffix('.json'), dict(source=source, images=int(keep.sum()),
        excluded_ids=sorted(excluded), pool_sha256=sha(a.out), labels_exported=False,
        exporter_sha256=sha(__file__)))
    print(f'Image-only pool: {int(keep.sum())} images -> {a.out}', flush=True)


def select_indices(size, count, seeds):
    if count < 1 or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('positive probe size and unique probe seeds required')
    if count * len(seeds) > size:
        raise ValueError('pool too small for disjoint probe sets')
    remaining = np.arange(size)
    selected = []
    for seed in seeds:
        positions = np.random.default_rng(seed).choice(len(remaining), count, replace=False)
        selected.append(remaining[positions])
        remaining = np.delete(remaining, positions)
    return np.stack(selected)


def make_views(pixels, seeds, patch_size):
    import torch
    if not 1 <= patch_size <= 16 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('patch size 1..16 and unique pattern seeds required')
    x = torch.from_numpy(pixels.copy()).permute(0, 3, 1, 2).float()/255
    views, names, families = [x], ['clean'], {}
    for seed in seeds:
        patch = torch.randint(0, 2, (3, patch_size, patch_size),
                              generator=torch.Generator().manual_seed(seed)).float()
        family = []
        for corner, (r, c) in zip(('tl', 'tr', 'bl', 'br'),
                                  ((0, 0), (0, 32-patch_size), (32-patch_size, 0),
                                   (32-patch_size, 32-patch_size))):
            modified = x.clone()
            modified[:, :, r:r+patch_size, c:c+patch_size] = patch
            family.append(len(views))
            names.append(f'pattern_{seed}_{corner}')
            views.append(modified)
        families[f'pattern_{seed}'] = family
    families['all_patterns'] = list(range(1, len(views)))
    mean = torch.tensor(MEAN).reshape(1, 1, 3, 1, 1)
    std = torch.tensor(STD).reshape(1, 1, 3, 1, 1)
    return (torch.stack(views)-mean)/std, names, families


def resolve_rounds(manifest, manifest_path, root, limit):
    if manifest.get('status') != 'complete':
        raise ValueError('complete saved trajectory required')
    records = manifest['rounds']
    if manifest.get('schema') == 'full-trajectory-v1':
        meta = manifest['metadata']
        if meta.get('attack') != 'badfu' or meta.get('aggregation') != 'badfu_sample_weighted':
            raise ValueError('this probe runner requires BadFU ResNet18 FL, not FUBA or FU states')
        ids = list(range(len(meta['counts'])))
        if len(records) != manifest['expected_rounds'] or any(r['client_ids'] != ids for r in records):
            raise ValueError('complete, ordered full-client BadFU trajectory required')
        records = [dict(event=r['round']-1, history={'path': r['path'], 'sha256': r['sha256']})
                   for r in records]
    if not records or limit < 0 or limit > len(records):
        raise ValueError('round limit must be 0 (all) or within the saved trajectory')
    result = []
    for record in records[:limit or len(records)]:
        row = {'event': record['event']}
        for key in ('history', 'after'):
            if key not in record:
                continue
            ref = record[key]
            if root is not None:
                filename = PureWindowsPath(ref['path']).name if '\\' in ref['path'] else Path(ref['path']).name
                path = root/filename
            else:
                path = Path(ref['path'])
                if not path.is_absolute():
                    path = manifest_path.parent/path
            row[key] = {'path': str(path.resolve()), 'sha256': ref['sha256']}
        result.append(row)
    if len({r['event'] for r in result}) != len(result):
        raise ValueError('duplicate trajectory events')
    return result


def measure(model, before, states, weights, inputs, batch_size, device):
    """Positive weighted parameter changes; recorded global buffers stay fixed."""
    import torch
    parameters = set(dict(model.named_parameters()))
    def evaluate(state):
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            out = torch.cat([model(x.to(device)).cpu() for x in inputs.split(batch_size)])
        if not torch.isfinite(out).all():
            raise ValueError('nonfinite output')
        return (out-out.mean(-1, keepdim=True)).numpy()
    baseline = evaluate(before)
    effects = []
    for state, weight in zip(states, weights):
        updated = {key: val+weight*(state[key]-val) if key in parameters else val
                   for key, val in before.items()}
        effects.append(evaluate(updated)-baseline)
    return np.stack(effects)


def collect(a, model_factory=None):
    import torch
    sys.path.insert(0, str(ROOT/'BadFU_src/BadFU-main'))
    from request_pipeline import load_event
    from badfu_common import ResNet18
    torch.set_num_threads(a.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    images, ids = load_pool(a.pool)
    selected = select_indices(len(images), a.probe_size, a.probe_seeds)
    xs, names, families = make_views(images[selected.flatten()], a.pattern_seeds, a.patch_size)
    # [view, repetition, image, channel, height, width]
    flat_inputs = xs.flatten(0, 1)
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    records = resolve_rounds(manifest, a.manifest, a.checkpoint_root, a.rounds)
    counts = np.asarray(manifest.get('counts', manifest.get('metadata', {}).get('counts')), dtype=float)
    if counts.ndim != 1 or len(counts) < 2 or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError('positive per-client aggregation counts required')
    dependencies = {str(p.relative_to(ROOT)): sha(p) for p in
        (ROOT/'BadFU_src/BadFU-main/request_pipeline.py', ROOT/'BadFU_src/BadFU-main/badfu_common.py',
         ROOT/'FUBA/research/hrac.py', ROOT/'FUBA/research/tension_audit.py')}
    protocol = dict(schema=1, method='label-free paired positive-update responses',
        code_sha256=sha(__file__), dependencies=dependencies, manifest_sha256=sha(a.manifest),
        pool_sha256=sha(a.pool), probe_ids=ids[selected].tolist(), probe_seeds=a.probe_seeds,
        probe_size=a.probe_size, pattern_seeds=a.pattern_seeds, patch_size=a.patch_size,
        views=names, families=families, source_rounds=records, counts=counts.tolist(),
        windows=list(WINDOWS), prefixes=a.prefixes, mean=MEAN, std=STD,
        batch_size=a.batch_size, device=device, threads=a.threads, torch=str(torch.__version__),
        hardware=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU',
        numerics='float32, TF32 off, cuDNN benchmark off, recorded BN buffers frozen',
        selection='uniform without replacement; disjoint sets; no label balancing or prediction filtering',
        labels_used=False, oracle_used=False, client_dataset_loaded=False, target_class_used=False,
        primary='all_patterns / paired / hrac_windows, maximum across all output classes',
        independent_training_runs=1, assumptions=['public RGB probe images', 'known model preprocessing',
          'individual client model states and aggregation weights', 'full-client deletion units'],
        limits=['Same trajectory across probe/pattern seeds; no calibrated FPR or TPR.',
                'No guarantee for other domains or trigger geometries.'])
    if a.out.exists():
        if not a.resume:
            raise FileExistsError('output exists; use --resume only with the identical protocol')
        saved = json.loads((a.out/'protocol.json').read_text(encoding='utf-8'))
        if saved != protocol:
            raise ValueError('resume protocol/code/input mismatch; use a new output directory')
    else:
        a.out.mkdir(parents=True)
        (a.out/'rounds').mkdir()
        save_json(a.out/'protocol.json', protocol)  # Before any model response.
    progress_path = a.out/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'rounds': {}}
    model = (model_factory or ResNet18)(pretrained=False).to(device).eval()
    new_rounds = 0
    for t, record in enumerate(records):
        target = a.out/'rounds'/f'{t:05d}.npz'
        if str(t) in progress['rounds']:
            if sha(target) != progress['rounds'][str(t)]['sha256']:
                raise ValueError(f'completed cache hash mismatch at round {t}')
            continue
        started = time.perf_counter()
        event, _ = load_event(record, counts.tolist())
        effect = measure(model, event['gm'], event['cms'], counts/counts.sum(),
                         flat_inputs, a.batch_size, device)
        # [repetition, client, view, image, output]; keep per-image responses.
        effect = effect.reshape(len(counts), len(names), len(selected), a.probe_size, -1).transpose(2, 0, 1, 3, 4)
        temporary = target.with_suffix('.pending')
        with temporary.open('wb') as f:
            np.savez_compressed(f, finite_logit=effect)
        os.replace(temporary, target)
        elapsed = time.perf_counter()-started
        progress['rounds'][str(t)] = dict(sha256=sha(target), seconds=elapsed)
        save_json(progress_path, progress)
        new_rounds += 1
        avg = np.mean([r['seconds'] for r in progress['rounds'].values()])
        print(f'Unlabeled probes {t+1}/{len(records)}: {elapsed:.1f}s; '
              f'rough remaining {(len(records)-len(progress["rounds"]))*avg/60:.1f}min', flush=True)
        del event, effect
        if a.stop_after and new_rounds >= a.stop_after:
            break
    return len(progress['rounds']) == len(records)


def score_profiles(h):
    from FUBA.research.hrac import inter_profile
    k = h.shape[1]
    sums = np.stack([h[-w:].sum(0) if w else h.sum(0) for w in WINDOWS])
    return dict(hrac_windows=np.stack([inter_profile(h, np.eye(k)[j], WINDOWS) for j in range(k)]),
        hrac_full_history=np.stack([inter_profile(h, np.eye(k)[j], (0,)) for j in range(k)]),
        signed_release=np.maximum(-sums, 0).max(0), absolute_effect=np.abs(sums).max(0),
        last_round_release=np.maximum(-h[-1], 0))


def analyze(folder):
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    progress = json.loads((folder/'progress.json').read_text())
    total = len(protocol['source_rounds'])
    if set(progress['rounds']) != {str(i) for i in range(total)}:
        raise ValueError('collection incomplete; resume first')
    history = []
    for t in range(total):
        path = folder/'rounds'/f'{t:05d}.npz'
        if sha(path) != progress['rounds'][str(t)]['sha256']:
            raise ValueError('cache hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            effect = data['finite_logit'].astype(float)
        if not np.isfinite(effect).all():
            raise ValueError('nonfinite response cache')
        history.append(effect.mean(-2))
    raw = np.stack(history)  # round, repeat, client, view, class
    paired = raw-raw[:, :, :, :1]
    prefixes = sorted(set([n for n in protocol['prefixes'] if 0 < n <= total]+[total]))
    rows = []
    for n in prefixes:
        for rep, seed in enumerate(protocol['probe_seeds']):
            for contrast, values, families in (
                ('paired', paired, protocol['families']),
                ('raw', raw, {'clean': [0], 'all_patterns': protocol['families']['all_patterns']})):
                per_view = [score_profiles(values[:n, rep, :, v]) for v in range(values.shape[3])]
                for family, views in families.items():
                    for method in per_view[0]:
                        scores = np.max([per_view[v][method] for v in views], axis=0)
                        ranking = sorted([dict(unit=j, score=float(s.max()), coordinate=int(s.argmax()),
                                               scores_by_class=s.tolist()) for j, s in enumerate(scores)],
                                         key=lambda row: -row['score'])
                        rows.append(dict(rounds=n, probe_seed=seed, contrast=contrast, family=family,
                                         method=method, ranking=ranking))
    report = dict(schema=1, protocol_sha256=sha(folder/'protocol.json'), analysis_sha256=sha(__file__),
        progress_sha256=sha(folder/'progress.json'), rows=rows, calibrated=False, fpr=None, tpr=None,
        independent_training_runs=1, probe_repetitions=len(protocol['probe_seeds']),
        collected_seconds=sum(r['seconds'] for r in progress['rounds'].values()),
        training_calls=0, model_unlearning_calls=0, gradient_calls=0,
        note='No ground-truth client role or target class enters scoring. '
             'Probe/pattern repetitions are not independent attack or benign training runs.')
    save_json(folder/'analysis.json', report)
    for row in rows:
        if row['rounds'] == total and row['contrast'] == 'paired' and row['family'] == 'all_patterns':
            top = row['ranking'][0]
            print(f"probe seed {row['probe_seed']}, {row['method']}: top unit {top['unit']}, "
                  f"output {top['coordinate']}, score {top['score']:.4f}")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    export = sub.add_parser('prepare', help='export a label-free image pool')
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument('--cifar-root', type=Path)
    source.add_argument('--image-dir', type=Path)
    export.add_argument('--exclude-ids', type=Path)
    export.add_argument('--out', type=Path, required=True)
    run = sub.add_parser('run', help='measure saved positive updates; no training/FU')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--checkpoint-root', type=Path, help='relocate history/after files by basename; keep SHA checks')
    run.add_argument('--pool', type=Path, required=True)
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--probe-size', type=int, default=80)
    run.add_argument('--probe-seeds', type=int, nargs='+', default=[20261019, 20261020, 20261021])
    run.add_argument('--pattern-seeds', type=int, nargs='+', default=[20261019, 20261020, 20261021])
    run.add_argument('--patch-size', type=int, default=4)
    run.add_argument('--rounds', type=int, default=0, help='0 means every saved round')
    run.add_argument('--prefixes', type=int, nargs='+', default=[5, 10, 15, 20, 40, 100])
    run.add_argument('--batch-size', type=int, default=64)
    run.add_argument('--device', default='auto')
    run.add_argument('--threads', type=int, default=4)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--stop-after', type=int, default=0, help='stop after this many NEW rounds; 0 means no stop')
    inspect = sub.add_parser('analyze', help='replay cached comparisons without checkpoints or image pool')
    inspect.add_argument('--out', type=Path, required=True)
    return p


def main():
    p = parser()
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a)
    elif a.mode == 'analyze':
        analyze(a.out)
    else:
        if a.batch_size < 1 or a.threads < 1 or a.stop_after < 0 or any(n < 1 for n in a.prefixes):
            p.error('positive batch size, threads and prefixes; nonnegative stop-after required')
        if collect(a):
            analyze(a.out)
        else:
            print('Partial collection saved. Repeat the same command with --resume to continue.')


if __name__ == '__main__':
    main()
