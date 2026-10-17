"""Saved BadFU full6 -> fu_remove5 comparison. Never trains FL or performs FU."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from reproduction.common.deletion_inputs import ROOT, read, resolve, write, public_pool
from reproduction.detection.deletion_contrast import split_public
from reproduction.common.trajectory import sha256
from reproduction.purification.purification_compare import run


def endpoint(run_root, phase, protocol):
    folder = run_root / phase
    manifest = read(folder / 'manifest.json')
    meta = manifest['metadata']
    expected = protocol['options']['rounds'] // 2 if phase.startswith('fu_') else protocol['options']['rounds']
    if (manifest.get('status') != 'complete' or manifest.get('expected_rounds') != expected
            or len(manifest['rounds']) != expected or not expected
            or [e['round'] for e in manifest['rounds']] != list(range(1, expected+1))
            or meta.get('phase') != phase or meta.get('attack') != 'badfu'
            or meta.get('upstream_commit') != protocol['upstream']['commit']
            or bool(meta.get('smoke')) != bool(protocol['options']['smoke'])):
        raise ValueError(f'incomplete/inconsistent phase: {phase}')
    expected_aggregation = 'author_federaser' if phase.startswith('fu_') else 'badfu_sample_weighted'
    if meta.get('aggregation') != expected_aggregation:
        raise ValueError('unsupported endpoint aggregation')
    entry = manifest['rounds'][-1]
    path = (folder / entry['path']).resolve()
    if path.parent != folder.resolve() or sha256(path) != entry['sha256']:
        raise ValueError(f'endpoint file/hash mismatch: {phase}')
    return dict(path=str(path), key='gm_after'), dict(
        manifest_sha256=sha256(folder/'manifest.json'), endpoint_sha256=entry['sha256'],
        completed_rounds=expected, metadata=meta)


def inspect_run(run_root):
    protocol = read(run_root/'protocol.json')
    if protocol.get('status') != 'complete' or protocol['options']['attack'] != 'badfu':
        raise ValueError('complete recorded BadFU run required')
    if sha256(run_root/'runtime.patch') != protocol['runtime_patch_sha256']:
        raise ValueError('training patch hash mismatch')
    for name, digest in protocol['adapter_sha256'].items():
        path = (run_root/'reproduction'/name).resolve()
        if path.parent != (run_root/'reproduction').resolve() or sha256(path) != digest:
            raise ValueError('training adapter snapshot mismatch')
    specs, evidence = {}, {}
    for phase in ('full6', 'fu_remove5', 'bd5'):
        if phase == 'bd5' and not (run_root/phase/'manifest.json').exists():
            continue
        specs[phase], evidence[phase] = endpoint(run_root, phase, protocol)
    pre, post = evidence['full6']['metadata'], evidence['fu_remove5']['metadata']
    if (pre['client_ids'] != list(range(6)) or post['client_ids'] != list(range(5))
            or pre['counts'][:5] != post['counts']):
        raise ValueError('client 5 deletion does not match endpoint client/count metadata')
    if 'bd5' in evidence:
        reference = evidence['bd5']['metadata']
        if reference['client_ids'] != post['client_ids'] or reference['counts'] != post['counts']:
            raise ValueError('bd5 is not a matching retained-data reference')
    split = read(run_root/'split.json')
    expected = {'bd5': 'clean i; client 0 adds poison',
                'full6': 'clean i; client 0 adds poison; client 5 duplicates clean 0 and adds cover'}
    if any(split.get('composition', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('unknown recorded BadFU data composition')
    evidence.update(training_protocol_sha256=sha256(run_root/'protocol.json'),
        split_sha256=sha256(run_root/'split.json'),
        verification_scope='training snapshots + manifests + final checkpoints; NOT a fresh full-history aggregation audit',
        forgetting_correctness='not_certified', reference_status='present' if 'bd5' in specs else 'missing',
        scenario='author code: separate client 5 deletion; not within-client sample unlearning')
    return protocol, specs, evidence


def find_record_root(run_root, protocol, override=None):
    candidates = [resolve(override)] if override else [Path(protocol['options']['record_root']),
        ROOT/'BadFU_src/BadFU-main/record/badnet_dataset', run_root/'record/badnet_dataset']
    for path in candidates:
        if (path/'pert_result.pt').is_file():
            expected = read(run_root/'record_inputs.json')['pert_result_sha256']
            if sha256(path/'pert_result.pt') != expected:
                raise ValueError('attack record differs from the training-bound record')
            return path.resolve()
    raise FileNotFoundError('original BadFU record not found; supply --record-root (no training needed)')


def author_attack_loader(run_root, record_root):
    """Deferred loader. The full author record is opened only after all arms freeze."""
    def load(test):
        from PIL import Image
        inventory = read(run_root/'record_inputs.json')
        # Older Windows recorders used backslashes; Linux records use slashes.
        image_hashes = {k.replace('\\', '/'): v for k, v in inventory['image_sha256'].items()}
        if len(image_hashes) != len(inventory['image_sha256']):
            raise ValueError('ambiguous duplicate image inventory paths')
        path = record_root/'pert_result.pt'
        if sha256(path) != inventory['pert_result_sha256']:
            raise ValueError('attack record changed')
        # Author-owned pickle, bound to the exact completed training record by hash.
        data = torch.load(path, map_location='cpu', weights_only=False)
        mapping = data['bd_test']['bd_data_container']['data_dict']
        if set(map(int, data['bd_test']['original_index_array'])) != set(map(int, mapping)):
            raise ValueError('author test indices do not match records')
        attack_targets = {int(info['other_info'][0]) for info in mapping.values()}
        if len(attack_targets) != 1:
            raise ValueError('this author adapter supports the recorded single-target attack only')
        target = next(iter(attack_targets))
        raw = (test[0].permute(0, 2, 3, 1).numpy()*255).round().astype(np.uint8)
        evaluated_hashes = {}
        for j, item in enumerate(test[2]):
            parts = str(item).split(':')
            if len(parts) != 3 or parts[:2] != ['cifar10', 'test']:
                raise ValueError('author evaluation requires exact CIFAR10 test IDs')
            index = int(parts[2])
            if index not in mapping:
                if int(test[1][j]) != target:
                    raise ValueError('missing non-target true-trigger evaluation image')
                continue  # Original target-class images are excluded from ASR.
            info = mapping[index]
            if int(info['other_info'][1]) != int(test[1][j]):
                raise ValueError('author original label disagrees with clean test label')
            parts = str(info['path']).replace('\\', '/').split('/')
            if 'badnet_dataset' not in parts:
                raise ValueError('unknown author image path')
            image_path = record_root.joinpath(*parts[parts.index('badnet_dataset')+1:]).resolve()
            rel = image_path.relative_to(record_root).as_posix()
            digest = sha256(image_path)
            if image_hashes.get(rel) != digest:
                raise ValueError(f'author trigger image checksum mismatch: {rel}')
            with Image.open(image_path) as im:
                pixels = np.asarray(im.convert('RGB'))
            if pixels.shape != raw[j].shape:
                raise ValueError('author trigger image shape mismatch')
            raw[j] = pixels
            evaluated_hashes[rel] = digest
        return dict(images=raw, ids=test[2], targets=np.full(len(raw), target, dtype=np.int64)), dict(
            kind='original_author_bd_test_records', record_sha256=inventory['pert_result_sha256'],
            evaluated_image_hashes=evaluated_hashes, target=target,
            loaded_after_all_repairs=True, no_reconstructed_trigger=True)
    return load


def graph(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = read(out/'results.json')
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ['clean_accuracy', 'asr'], ['Clean accuracy', 'True-trigger ASR']):
        vals = [100*results[n][key] if results[n][key] is not None else np.nan for n in names]
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)), names, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(title)
        for i, val in enumerate(vals):
            if np.isfinite(val):
                ax.text(i, val+1, f'{val:.1f}', ha='center', fontsize=8)
    prep = read(out.parent/'preparation.json')
    rounds = prep['source']['full6']['completed_rounds']
    title = (f'PIPELINE CHECK ONLY: {rounds}-round source; reduced evaluation budget'
             if prep['smoke'] else 'Saved-model pilot: no deletion certificate; no official SAU comparison')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=150)
    plt.close(fig)


def export(out, experiment_root):
    # No regenerated images on resume: retain bytes used by the evidence hash.
    if not (out/'comparison.png').exists():
        graph(out)
    protocol = read(out/'protocol.json')
    prep = read(experiment_root/'preparation.json')
    protocol['smoke'] = prep['smoke']
    protocol['evidence_scope'] = 'pipeline_only' if prep['smoke'] else 'single_saved_model_pair_pilot'
    if prep['smoke']:
        report = out/'report.md'
        rounds = prep['source']['full6']['completed_rounds']
        marker = f'**PIPELINE CHECK ONLY: {rounds}-round source; reduced evaluation budget, not defense-performance evidence.**\n\n'
        if not report.read_text(encoding='utf-8').startswith(marker):
            report.write_text(marker+report.read_text(encoding='utf-8'), encoding='utf-8')
    protocol['output_hashes'] = {p.name: sha256(p) for p in out.iterdir() if p.name != 'protocol.json'}
    write(out/'protocol.json', protocol)
    files = sorted([p for p in out.iterdir() if p.suffix in ('.json', '.md', '.png')])
    files += [experiment_root/'preparation.json']
    key = sha256(out/'protocol.json')[:16]
    dest = ROOT/'reproduction/validation/exports/purification'/key
    if dest.exists():
        old = read(dest/'export.json')
        if old['files'] != {p.name: sha256(p) for p in files} or sha256(dest/'purification.zip') != old['zip_sha256']:
            raise ValueError('existing export differs')
        print(f'Reusing verified upload folder: {dest}', flush=True)
        return
    dest.mkdir(parents=True, exist_ok=False)
    archive = dest/'purification.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.name)
    write(dest/'export.json', dict(schema='purification-summary-v1', zip_sha256=sha256(archive),
        files={p.name: sha256(p) for p in files},
        limitation='summary replay only; checkpoints, input images and private training data stay local'))
    for name in ('report.md', 'comparison.png', 'results.json'):
        (dest/name).write_bytes((out/name).read_bytes())
    print(f'Upload folder: {dest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='artifacts/badfu_author_s42')
    parser.add_argument('--out', default='artifacts/purification_badfu_s42_layout_v2')
    parser.add_argument('--data-root')
    parser.add_argument('--record-root')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--math-ablation', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-mode', choices=['patch', 'additive', 'both'], default='patch')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    run_root, out = resolve(args.run).resolve(), resolve(args.out).resolve()
    protocol, specs, evidence = inspect_run(run_root)
    if protocol['options']['smoke'] and not args.smoke:
        raise ValueError('training is smoke; pass --smoke and do not treat as performance evidence')
    record_root = find_record_root(run_root, protocol, args.record_root)
    print(json.dumps(dict(run=str(run_root), phases=specs, verification=evidence,
        record_root=str(record_root), math_ablation=args.math_ablation), indent=2), flush=True)
    if args.dry_run:
        return
    counts = [2, 2, 2] if args.smoke else [16, 16, 100]
    t = SimpleNamespace(dataset='cifar10', protocol=protocol)
    images, labels, ids, source = public_pool(t, args.data_root)
    indices = split_public(images, labels, ids, counts, seed=20261018)
    if out.exists() and not args.resume:
        raise FileExistsError(f'use a new --out or --resume: {out}')
    out.mkdir(parents=True, exist_ok=True)
    pool_paths = {}
    for name, ix in zip(('development', 'validation', 'test'), indices):
        path = out/f'{name}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if any(not np.array_equal(old[k], v[ix]) for k, v in [('images', images), ('labels', labels), ('ids', ids)]):
                    raise ValueError('prepared input changed; use a new output folder')
        else:
            np.savez(path, images=images[ix], labels=labels[ix], ids=ids[ix])
        pool_paths[name] = str(path)
    cfg = dict(model='badfu_cifar10', mean=[.4914, .4822, .4465], std=[.2023, .1994, .2010],
        pre=specs['full6'], post=specs['fu_remove5'], references={'bd5': specs['bd5']} if 'bd5' in specs else {},
        post_kind='actual_fu', request_description='author full6 -> fu_remove5; delete whole client 5',
        **pool_paths, device=args.device, seed=20261018, epsilon=8/255, steps=2 if args.smoke else 20,
        restarts=1 if args.smoke else 2, batch_size=16 if args.smoke else 32,
        epochs=2 if args.smoke else 5, lr=.001, pre_weight=1., correct_weight=0.,
        repair_weight=1., clean_tolerance=.02, probe_mode=args.probe_mode, patch_size=4,
        patch_step_size=.1, tail_mass=.25, math_ablation=args.math_ablation,
        class_ce_tolerance=.05, dual_lr=.5, dual_cap=100.)
    cfg['source_binding'] = evidence
    cfg['record_sha256'] = read(run_root/'record_inputs.json')['pert_result_sha256']
    cfg['runner_sha256'] = sha256(__file__)
    preparation = dict(source=evidence, public_source=source, smoke=args.smoke,
        counts_per_class=counts, source_sha256=sha256(__file__), config=cfg,
        trigger_record_sha256=read(run_root/'record_inputs.json')['pert_result_sha256'])
    if (out/'preparation.json').exists() and read(out/'preparation.json') != preparation:
        raise ValueError('preparation changed; use a new output folder')
    write(out/'preparation.json', preparation)
    print('Fitting from saved models; FL training and FU are not run.', flush=True)
    run(cfg, out/'comparison', attack_loader=author_attack_loader(run_root, record_root), resume=args.resume)
    export(out/'comparison', out)


if __name__ == '__main__':
    main()
