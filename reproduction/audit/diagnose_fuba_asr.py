"""Paired evaluation of saved FUBA models: class bias, trigger effect, and evaluator choice.

This is labeled, post-hoc diagnosis; it is not a detector and runs no training/FU.
The original attack sources and recorded training metrics are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.training.run import check_sources, lock
from reproduction.common.trajectory import sha256, json_write
from FUBA.model import Net, resnet18, MNISTAutoencoder, UNet

TARGET = 8


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired_rates(labels, clean, perturbed, target=TARGET):
    """All targeting rates use the SAME non-target examples, including existing errors."""
    y, a, b = (np.asarray(v) for v in (labels, clean, perturbed))
    if y.ndim != 1 or not len(y) or a.shape != y.shape or b.shape != y.shape:
        raise ValueError('Equal nonempty prediction/label vectors required')
    if any(v.dtype.kind not in 'iu' or np.any(v < 0) or np.any(v > 9) for v in (y, a, b)):
        raise ValueError('Expected integer labels/predictions 0..9')
    eligible = y != target
    correct = eligible & (a == y)
    n = int(eligible.sum())
    if not n:
        raise ValueError('No non-target examples')
    pct = lambda mask: 100 * int(np.count_nonzero(mask & eligible)) / n
    c, s = pct(a == target), pct(b == target)
    return dict(non_target_n=n, clean_accuracy_pct=100*float((a == y).mean()),
        clean_target_rate_pct=c, triggered_target_rate_pct=s, net_trigger_gain_pp=s-c,
        newly_targeted_pct=pct((a != target) & (b == target)),
        no_longer_targeted_pct=pct((a == target) & (b != target)),
        clean_correct_non_target_n=int(correct.sum()),
        correct_to_target_pct=(100*int((correct & (b == target)).sum())/int(correct.sum())) if correct.any() else None,
        by_source_class={str(i):dict(n=int((y == i).sum()),
            clean_target_rate_pct=100*float((a[y == i] == target).mean()),
            triggered_target_rate_pct=100*float((b[y == i] == target).mean()))
            for i in range(10) if i != target and np.any(y == i)},
        clean_confusion=np.bincount(y*10+a, minlength=100).reshape(10,10).tolist(),
        perturbed_confusion=np.bincount(y*10+b, minlength=100).reshape(10,10).tolist())


def clean_bias_bound(clean_accuracy_pct, asr_pct, non_target_fraction=.9):
    ceiling = min(100., (100-clean_accuracy_pct)/non_target_fraction)
    return dict(clean_target_rate_upper_bound_pct=ceiling,
                newly_targeted_lower_bound_pct=max(0., asr_pct-ceiling))


def sample_indices(labels, count, seed):
    labels = np.asarray(labels)
    if count < 10 or count % 10:
        raise ValueError('samples must be a positive multiple of 10')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == i) for i in range(10)]
    if any(len(g) < count//10 for g in groups):
        raise ValueError('Not enough examples for balanced sampling')
    chosen = np.concatenate([rng.choice(g, count//10, replace=False) for g in groups])
    rng.shuffle(chosen)
    return chosen


def header(run):
    p, m = read(run/'protocol.json'), read(run/'fl/manifest.json')
    if (p['status'] != 'complete' or p['options']['attack'] != 'fuba' or m['status'] != 'complete'
            or p['options']['dataset'] not in ('mnist','cifar10') or m['schema'] != 'full-trajectory-v1'
            or m['metadata']['dataset'] != p['options']['dataset']
            or p['upstream']['commit'] != lock()['fuba']['commit']
            or m['metadata']['upstream_commit'] != p['upstream']['commit']
            or len(m['rounds']) != p['options']['rounds']):
        raise ValueError('Complete full-trajectory FUBA run required')
    for name, expected in p['adapter_sha256'].items():
        f = run/'reproduction'/name
        if f.parent.resolve() != (run/'reproduction').resolve() or sha256(f) != expected:
            raise ValueError('Training adapter snapshot mismatch')
    if sha256(run/'runtime.patch') != p['runtime_patch_sha256']:
        raise ValueError('Training patch mismatch')
    return p, m


def load_round(run, manifest, number):
    entry = manifest['rounds'][number-1]
    path = run/'fl'/entry['path']
    if entry['round'] != number or path.parent.resolve() != (run/'fl').resolve() or sha256(path) != entry['sha256']:
        raise ValueError(f'Checkpoint hash/round mismatch: {number}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['round'] != number or state['client_ids'] != entry['client_ids']:
        raise ValueError('Checkpoint participation mismatch')
    return state


def split_audit(run, train_labels):
    rows, assignments = [], []
    for path in sorted(run.glob('split_rank*.json')):
        d = read(path)
        ids = d['full_indices']
        flat = np.concatenate([np.asarray(x, dtype=int) for x in ids])
        if len(flat) != len(train_labels) or not np.array_equal(np.sort(flat), np.arange(len(train_labels))):
            raise ValueError(f'Split is not a complete disjoint training partition: {path.name}')
        counts = []
        for indices, effective in zip(ids, d['effective_counts']):
            if not 0 < effective <= len(indices):
                raise ValueError('Invalid effective training count')
            counts.append(np.bincount(train_labels[np.asarray(indices[:effective])], minlength=10).tolist())
        rows.append(dict(file=path.name, sha256=sha256(path), effective_class_counts=counts))
        assignments.append(ids)
    return dict(files=rows, all_rank_assignments_identical=all(a == assignments[0] for a in assignments) if assignments else None,
                note='Class counts are evaluation diagnostics; no client labels enter HRAC scoring.')


@torch.no_grad()
def predict(model, inputs, batch, device):
    return torch.cat([model(x.to(device)).argmax(-1).cpu() for x in inputs.split(batch)]).numpy()


@torch.no_grad()
def generated_noise(factory, state, inputs, batch, device, train=False):
    # Fresh copy per condition: train-mode BN updates cannot contaminate another condition.
    generator = factory().to(device)
    generator.load_state_dict(state)
    generator.train(train)
    noise = torch.cat([generator(x.to(device)).clamp(-.04,.04).cpu() for x in inputs.split(batch)])
    if not torch.isfinite(noise).all():
        raise ValueError('Nonfinite generated perturbation')
    return noise


def permute_noise(noise, seed):
    rng = torch.Generator().manual_seed(seed)
    return torch.stack([x.flatten()[torch.randperm(x.numel(), generator=rng)].reshape_as(x) for x in noise])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--benign-run', type=Path, help='Optional matched attack-disabled training; same images/perturbations on both models')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--rounds', default='5,6,8,9,16,26,100')
    p.add_argument('--reference-round', type=int, help='Fixed generator snapshot; default last requested round (post-hoc only)')
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='auto')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.batch_size < 2 or a.samples % a.batch_size == 1:
        p.error('Use batch-size >=2 without a final singleton batch for generator train-mode BN')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(a.seed)
    check_sources('fuba')
    run = a.run_dir.resolve()
    training, manifest = header(run)
    benign, benign_manifest = None, None
    if a.benign_run:
        benign = a.benign_run.resolve()
        bp, benign_manifest = header(benign)
        keys = ('dataset','rounds','seed','clients','participant_rate','non_iid','alpha','smoke')
        if any(bp['options'][k] != training['options'][k] for k in keys) or bp['options']['warm_up'] != bp['options']['rounds']:
            raise ValueError('Benign control needs matching training settings with warm-up equal to rounds')
    rounds = sorted(set(map(int,a.rounds.split(','))))
    reference = a.reference_round or max(rounds)
    if not rounds or min(rounds)<1 or max(rounds+[reference])>len(manifest['rounds']) or reference<1:
        p.error('Requested rounds must exist in the completed training run')
    out = (a.out or ROOT/'reproduction/validation/exports'/run.name/'asr_diagnostic_v1').resolve()
    if out.exists():
        p.error('Choose a new --out; diagnosis never overwrites previous results')
    out.mkdir(parents=True)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    dataset = training['options']['dataset']
    from torchvision.datasets import CIFAR10, MNIST
    cls = CIFAR10 if dataset=='cifar10' else MNIST
    data_root = a.data_root or Path(training['options']['data_root'])
    test, train = cls(str(data_root), train=False, download=False), cls(str(data_root), train=True, download=False)
    labels = np.asarray(test.targets)
    ids = sample_indices(labels, a.samples, a.seed)
    pixels = test.data if dataset=='cifar10' else test.data.numpy()[...,None]
    inputs = torch.from_numpy(pixels[ids].copy()).permute(0,3,1,2).float()/255
    y = labels[ids]
    factory = (lambda: UNet(3)) if dataset=='cifar10' else MNISTAutoencoder
    model = (resnet18() if dataset=='cifar10' else Net()).to(device).eval()
    fixed_state = load_round(run, manifest, reference)
    generators = fixed_state['extras']['individual_generators']
    if not generators:
        raise ValueError('reference-round must contain a learned individual generator')
    ref_owner = max(generators)
    fixed_noise = generated_noise(factory, generators[ref_owner], inputs, a.batch_size, device)
    del fixed_state, generators
    fixed = dict(fixed_reference_eval=(inputs+fixed_noise).clamp(-1,1),
                 permuted_reference_noise=(inputs+permute_noise(fixed_noise, a.seed)).clamp(-1,1))
    report = dict(schema=1, status='running', diagnostic_only=True, training_calls=0, fu_calls=0,
        source_training_settings={k:training['options'][k] for k in
            ('dataset','rounds','seed','clients','smoke','non_iid','alpha','participant_rate','warm_up')},
        labels_used_for_diagnosis=True, detector_modified=False, torch=str(torch.__version__), device=device,
        code_sha256={str(path.relative_to(ROOT)).replace('\\','/'):sha256(path) for path in
            [Path(__file__).resolve(),ROOT/'FUBA/model/model.py',ROOT/'FUBA/model/resnet.py']},
        training_protocol_sha256=sha256(run/'protocol.json'), manifest_sha256=sha256(run/'fl/manifest.json'),
        selected_pixels_sha256=hashlib.sha256(pixels[ids].tobytes()).hexdigest(),
        selected_labels_sha256=hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest(),
        options=dict(dataset=dataset, rounds=rounds, reference_round=reference, reference_generator_owner=ref_owner,
            target=TARGET,samples=a.samples,seed=a.seed,batch_size=a.batch_size,epsilon=1.,noise_bound=.04,
            input_preprocessing='ToTensor',perturbed_clamp=[-1,1]),
        checked_checkpoints=[manifest['rounds'][n-1] for n in sorted(set(rounds+[reference]))],
        split_audit=split_audit(run,np.asarray(train.targets)),results=[],
        limitations=['Post-hoc labeled evaluation, not a change to the unlabeled detector.',
          'Fixed reference may be from a later round; it is not an online observation.',
          'Train-mode evaluation uses a fixed sample order/batch size; it does not reproduce the original shuffled full-test traversal.',
          'Clean-to-trigger gain does not alone prove poisoned-training causality; a matched benign-trained model is still needed.',
          'Permuted noise preserves per-image perturbation values/norms before image clipping, not after clipping.',
          'Selected checkpoints are hash-verified; this is not a new full-trajectory aggregation audit.',
          'Balanced subsets may differ from the stored full-test ASR. Use --samples 10000 for CIFAR-10 full-test evaluation.'])
    predictions = dict(ids=ids, labels=y)
    if benign:
        report['benign_control'] = dict(protocol_sha256=sha256(benign/'protocol.json'),
            manifest_sha256=sha256(benign/'fl/manifest.json'),
            split_audit=split_audit(benign,np.asarray(train.targets)),
            selected_participants_match=all(manifest['rounds'][n-1]['client_ids']==benign_manifest['rounds'][n-1]['client_ids'] for n in rounds))
        attack_splits = [read(f)['full_indices'] for f in sorted(run.glob('split_rank*.json'))]
        benign_splits = [read(f)['full_indices'] for f in sorted(benign.glob('split_rank*.json'))]
        if not attack_splits or not benign_splits or any(s != attack_splits[0] for s in attack_splits+benign_splits):
            raise ValueError('Benign and attack training partitions must match exactly across ranks')
    start = time.perf_counter()
    json_write(out/'diagnosis.json',report)
    for number in rounds:
        state = load_round(run,manifest,number)
        model.load_state_dict(state['gm_after'])
        clean = predict(model,inputs,a.batch_size,device)
        predictions[f'r{number}_clean'] = clean
        conditions = dict(fixed)
        individual = state['extras']['individual_generators']
        owner = max(individual) if individual else None
        if owner is not None:
            for mode in ('eval','train'):
                noise = generated_noise(factory,individual[owner],inputs,a.batch_size,device,train=mode=='train')
                conditions['current_individual_'+mode] = (inputs+noise).clamp(-1,1)
        aggregate = state['extras']['aggregate_generator']
        if aggregate is not None:
            noise = generated_noise(factory,aggregate,inputs,a.batch_size,device)
            conditions['current_aggregate_eval'] = (inputs+noise).clamp(-1,1)
        stored = read(run/f'metrics_round_{number:04d}.json')
        if stored['author_generator_client'] != owner:
            raise ValueError('Stored generator owner does not match checkpoint')
        result = dict(round=number, stored_metric=stored, current_generator_owner=owner,
            client_ids=state['client_ids'], clean=paired_rates(y,clean,clean), conditions={})
        for name, images in conditions.items():
            pred = predict(model,images,a.batch_size,device)
            predictions[f'r{number}_{name}'] = pred
            result['conditions'][name] = paired_rates(y,clean,pred)
        if benign:
            control_state = load_round(benign,benign_manifest,number)
            model.load_state_dict(control_state['gm_after'])
            control_clean = predict(model,inputs,a.batch_size,device)
            predictions[f'benign_r{number}_clean'] = control_clean
            result['benign_control'] = dict(clean=paired_rates(y,control_clean,control_clean),conditions={})
            for name, images in conditions.items():
                control_pred = predict(model,images,a.batch_size,device)
                predictions[f'benign_r{number}_{name}'] = control_pred
                result['benign_control']['conditions'][name] = paired_rates(y,control_clean,control_pred)
            del control_state
        report['results'].append(result)
        json_write(out/'diagnosis.json',report)
        del state
        print(f'[diagnose] round {number}: clean target rate {result["clean"]["clean_target_rate_pct"]:.2f}%; '
              f'{len(conditions)} paired conditions; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    np.savez_compressed(out/'predictions.npz',**predictions)
    report.update(status='complete',elapsed_seconds=time.perf_counter()-start,
                  predictions_sha256=sha256(out/'predictions.npz'))
    json_write(out/'diagnosis.json',report)
    print(f'Share this folder: {out}')


if __name__=='__main__':
    main()
