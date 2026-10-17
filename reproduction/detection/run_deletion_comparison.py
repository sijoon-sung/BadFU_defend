"""Compare logit double differences and normal-explanation history on saved runs.

No FL training, FU simulation, client contact, attack target, or trigger access.
Actual recorded FU is a separate stratum from historical-subtraction surrogates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from reproduction.common.deletion_inputs import Trajectory, public_pool, read, resolve, write
from reproduction.detection.deletion_contrast import (apply_probe, centered, contrast, coordinates, direction_norm,
    double_difference, empirical_auc, explain, history_scores, jvp_logits,
    minimum_clean_error, public_directions, search_logits, split_public)
from reproduction.detection.request_probe import SearchConfig, freeze, predict
from reproduction.common.trajectory import sha256

CODE = ['reproduction/detection/run_deletion_comparison.py', 'reproduction/detection/deletion_contrast.py',
        'reproduction/common/deletion_inputs.py', 'reproduction/detection/history_boundary.py',
        'reproduction/detection/request_probe.py', 'reproduction/common/trajectory.py',
        'FUBA/model/model.py', 'FUBA/model/resnet.py']
DEFAULT = dict(seed=20261017, reference_per_class=8, search_per_class=8,
    audit_per_class=8, epsilon=8/255, steps=20, step_size=2/255, restarts=2,
    batch_size=16, blocks=3, clean_absolute=1e-5, clean_relative=.5,
    coefficient_bound=2., probe_error=1e-5, dose=1., probe_mode='both', patch_size=4,
    patch_step_size=.1)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def discover(args):
    cases, missing = [], []
    badfu = resolve(args.badfu_run or 'artifacts/badfu_author_s42')
    fuba = resolve(args.fuba_run or 'artifacts/fuba_cifar10_s42')

    def add(name, run, phase, clients, role, **extra):
        if not (run / phase / 'manifest.json').exists():
            missing.append(str(run / phase / 'manifest.json'))
            return
        cases.append(dict(name=name, run=str(run), phase=phase, request_clients=clients,
                          role=role, **extra))

    add('badfu_attack_surrogate', badfu, 'full6', [5], 'attack_request')
    for client in range(5):
        add(f'badfu_clean_delete{client}', badfu, 'clean5', [client], 'benign_request')
    if (badfu / 'fu_remove5/manifest.json').exists():
        add('badfu_attack_actual', badfu, 'full6', [5], 'attack_request', actual_phase='fu_remove5')
    else:
        missing.append(str(badfu / 'fu_remove5/manifest.json'))
    add('fuba_attack_surrogate', fuba, 'fl', [4], 'attack_request')
    benign = args.fuba_benign_run or [f'artifacts/fuba_cifar10_benign_s{s}' for s in (42, 43, 44)]
    for index, run in enumerate(benign):
        add(f'fuba_benign{index}_delete4', resolve(run), 'fl', [4], 'benign_request')
    return dict(schema=1, settings=DEFAULT.copy(), cases=cases, missing=sorted(set(missing)))


def settings(plan):
    cfg = DEFAULT | plan.get('settings', {})
    if set(cfg) != set(DEFAULT):
        raise ValueError('unknown experiment settings')
    for k in ('reference_per_class', 'search_per_class', 'audit_per_class', 'batch_size', 'blocks', 'patch_size'):
        if not isinstance(cfg[k], int) or cfg[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    for k in ('clean_absolute', 'clean_relative', 'probe_error'):
        if not np.isfinite(cfg[k]) or cfg[k] < 0:
            raise ValueError(f'invalid {k}')
    for k in ('coefficient_bound', 'dose', 'patch_step_size'):
        if not np.isfinite(cfg[k]) or cfg[k] <= 0:
            raise ValueError(f'invalid {k}')
    SearchConfig(**{k: cfg[k] for k in ('epsilon', 'steps', 'step_size', 'restarts', 'batch_size', 'seed')}).validate()
    if cfg['probe_mode'] not in ('additive', 'patch', 'both'):
        raise ValueError('unknown probe family')
    return cfg


def validate_role(case, trajectory):
    # Role is evaluation metadata only; it never enters probe search or LP fitting.
    if case['role'] not in ('benign_request', 'attack_request', 'context_control'):
        raise ValueError('invalid evaluation role')
    o = trajectory.protocol['options']
    if case['role'] == 'benign_request':
        clean = trajectory.phase == 'clean5' if trajectory.family == 'badfu' else o['warm_up'] >= o['rounds']
        if not clean:
            raise ValueError('an attacked FL model cannot be labeled attack-free benign')
    if trajectory.family == 'fuba' and case['role'] == 'attack_request' and o['warm_up'] >= o['rounds']:
        raise ValueError('FUBA attack is disabled in this run')


def interval_record(item, tolerance, norm):
    return dict(status=item.status, clean_rank=item.clean_rank, parameter_norm=norm,
                clean_error=tolerance,
                lower=None if item.lower is None else (item.lower * norm).tolist(),
                upper=None if item.upper is None else (item.upper * norm).tolist())


def run_case(case, cfg, out, device, data_root=None):
    start = time.perf_counter()
    trajectory = Trajectory(case['run'], case['phase'])
    validate_role(case, trajectory)
    images, labels, ids, pool_info = public_pool(trajectory, data_root, case.get('pool'))
    splits = split_public(images, labels, ids,
        [cfg['reference_per_class'], cfg['search_per_class'], cfg['audit_per_class']], cfg['seed'])
    pool_fingerprint = hashlib.sha256()
    for s in splits:
        for value in (images[s], labels[s], ids[s]):
            pool_fingerprint.update(value.tobytes())
    signature = dict(case=case, settings=cfg, device=device, torch=str(torch.__version__),
        source={name: sha256(ROOT / name) for name in CODE},
        protocol_sha256=sha256(trajectory.run / 'protocol.json'),
        manifest_sha256=sha256(trajectory.root / 'manifest.json'),
        public_sha256=pool_fingerprint.hexdigest())
    if case.get('actual_phase'):
        signature['actual_manifest_sha256'] = sha256(trajectory.run / case['actual_phase'] / 'manifest.json')
    if case.get('actual_state'):
        signature['external_fu_sha256'] = sha256(resolve(case['actual_state']['path']))
    key = fingerprint(signature)
    if out.exists():
        receipt = read(out / 'protocol.json')
        if receipt.get('status') != 'complete' or receipt.get('signature') != key:
            raise ValueError(f'incomplete/changed experiment at {out}; choose a NEW --out')
        for name, digest in receipt['output_sha256'].items():
            if sha256(out / name) != digest:
                raise ValueError('saved observation hash mismatch')
        print(f'  Reusing verified comparison: {case["name"]}', flush=True)
        return read(out / 'scores.json')
    out.mkdir(parents=True)
    protocol = dict(schema='deletion-comparison-v1', status='running', signature=key,
        inputs=signature, public=pool_info,
        splits={name: ids[s].tolist() for name, s in zip(('reference', 'search', 'audit'), splits)},
        additional_assumptions=['trusted public images AND labels covering all classes',
            'individual FL updates and their actual aggregation weights visible',
            'public class-gradient span and coefficient/tolerance bounds are experimental assumptions'],
        history_anchor='final pre-deletion model; parameter JVPs; fixed recorded BN buffers',
        request_unit='whole client / unions of recorded clients; NO within-client samples inferred',
        selection='all targets on search images; same frozen delta/target for every ablation',
        time_interpretation='retrospective physical-round blocks, NOT held-out prediction rounds',
        fl_training_calls=0, fu_calls=0, attack_trigger_used=False, attack_target_used=False,
        thresholds_calibrated=False, hardware=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU')
    write(out / 'protocol.json', protocol)
    pre, post, candidate = trajectory.models(case['request_clients'], case.get('actual_phase'),
                                            case.get('actual_state'), cfg['dose'])
    pre, post = freeze(pre, device), freeze(post, device)
    tensors = [torch.from_numpy(images[s].transpose(0, 3, 1, 2).copy()).to(device).float() / 255 for s in splits]
    ref, search_x, audit = tensors
    ref_y = torch.as_tensor(labels[splits[0]], device=device)
    audit_y = torch.as_tensor(labels[splits[2]])
    search_cfg = SearchConfig(**{k: cfg[k] for k in ('epsilon', 'steps', 'step_size', 'restarts', 'batch_size', 'seed')})
    print('  Search logit contrast on public search images...', flush=True)
    selected = search_logits(pre, post, search_x, search_cfg, cfg['probe_mode'], cfg['patch_size'], cfg['patch_step_size'])
    target, delta = selected['target'], selected['delta']
    changed = apply_probe(audit, selected)
    use = audit_y != target
    batch = cfg['batch_size']
    logits = dict(pre_clean=predict(pre, audit, batch).cpu(), pre_probe=predict(pre, changed, batch).cpu(),
                  post_clean=predict(post, audit, batch).cpu(), post_probe=predict(post, changed, batch).cpu())
    dd = double_difference(**logits)
    dd_scalar = float(contrast(dd, target)[use].mean())
    clean_change = logits['post_clean'] - logits['pre_clean']
    endpoint_probe = contrast(dd, target)[use].mean().reshape(1).numpy().astype(float)
    endpoint_clean = centered(clean_change).flatten().numpy().astype(float)
    endpoint_direction = {k: b.detach().cpu() - a.detach().cpu()
                          for (k, a), (_, b) in zip(pre.named_parameters(), post.named_parameters())}
    endpoint_norm = direction_norm(endpoint_direction)
    del endpoint_direction
    arrays = {k: v.numpy() for k, v in logits.items()}
    arrays.update(double_difference=dd.numpy(), delta=delta.cpu().numpy(), audit_ids=ids[splits[2]],
                  audit_labels=labels[splits[2]], target=np.array(target), non_target_mask=use.numpy())
    if selected['mask'] is not None:
        arrays['probe_mask'] = selected['mask'].cpu().numpy()
    print('  Build public class-loss reference directions and exact Jacobian responses...', flush=True)
    directions, norms = public_directions(pre, ref, ref_y, 10, batch)
    bc, bp = [], []
    both = torch.cat((audit, changed))
    for direction in directions:
        responses = jvp_logits(pre, direction, both, batch)
        c, p = coordinates(responses[:len(audit)], responses[len(audit):], target, use)
        bc.append(c)
        bp.append(p)
    del directions
    bc, bp = np.stack(bc, axis=1), np.stack(bp, axis=1)
    arrays.update(clean_basis=bc, probe_basis=bp, public_gradient_norms=np.asarray(norms),
                  endpoint_clean=endpoint_clean, endpoint_probe=endpoint_probe)
    if endpoint_norm > 1e-12:
        item, eps = explain(endpoint_clean / endpoint_norm, endpoint_probe / endpoint_norm, bc, bp, cfg)
        endpoint_interval = interval_record(item, eps, endpoint_norm)
        endpoint_score = None if item.lower is None else max(0., float(item.lower[0] * endpoint_norm))
    elif np.max(np.abs(endpoint_clean)) < 1e-8 and np.max(np.abs(endpoint_probe)) < 1e-8:
        endpoint_interval = dict(status='identical_endpoint', lower=[0.], upper=[0.], parameter_norm=0.)
        endpoint_score = 0.
    else:
        endpoint_interval = dict(status='abstain_buffer_only_change', lower=None, upper=None, parameter_norm=0.)
        endpoint_score = None
    if endpoint_score is None and endpoint_norm > 1e-12:
        endpoint_interval['minimum_clean_error_diagnostic'] = minimum_clean_error(
            endpoint_clean / endpoint_norm, bc, cfg['coefficient_bound'])
    count = len(trajectory.manifest['rounds'])
    lower, upper, raw = [np.full((count, 2, 1), np.nan) for _ in range(3)]
    clean_observed = np.zeros((count, 2, bc.shape[0]))
    unit_norms, interval_records = np.zeros((count, 2)), []
    for t, groups in enumerate(trajectory.group_updates(case['request_clients'])):
        rows = []
        for group, direction in enumerate(groups):
            norm = direction_norm(direction)
            unit_norms[t, group] = norm
            if norm <= 1e-12:
                lower[t, group] = upper[t, group] = raw[t, group] = 0.
                rows.append(dict(status='recorded_zero_aggregate', lower=[0.], upper=[0.], parameter_norm=norm))
                continue
            unit = {k: v / norm for k, v in direction.items()}
            response = jvp_logits(pre, unit, both, batch)
            c, p = coordinates(response[:len(audit)], response[len(audit):], target, use)
            clean_observed[t, group], raw[t, group] = c * norm, p * norm
            item, eps = explain(c, p, bc, bp, cfg)
            if item.lower is not None:
                lower[t, group], upper[t, group] = item.lower * norm, item.upper * norm
            record = interval_record(item, eps, norm)
            if item.lower is None:
                record['minimum_clean_error_diagnostic'] = minimum_clean_error(c, bc, cfg['coefficient_bound'])
            rows.append(record)
        interval_records.append(rows)
        print(f'  History {t+1}/{count}: ' + ', '.join(row['status'] for row in rows), flush=True)
    edges = np.linspace(0, count, min(cfg['blocks'], count) + 1, dtype=int)
    blocks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    hs = history_scores(lower, upper, raw, blocks)
    hscore = None if hs['normal_set'] is None else count * hs['normal_set']['score']
    hsum = None if hs['normal_set_single_block'] is None else count * hs['normal_set_single_block']['score']
    # Fixed intersection ablation. Requiring BOTH evidences can reduce recall.
    # Units are logit effects: per-round history score is multiplied by rounds.
    joint = None if hscore is None else min(max(0., dd_scalar), hscore)
    arrays.update(history_lower=lower, history_upper=upper, history_raw=raw,
                  history_clean=clean_observed, history_parameter_norm=unit_norms)
    score = dict(name=case['name'], family=trajectory.family, dataset=trajectory.dataset,
        role=case['role'], run=str(trajectory.run), seed=trajectory.protocol['options']['seed'],
        smoke=trajectory.protocol['options']['smoke'], candidate=candidate,
        selected={k: v for k, v in selected.items() if k not in ('delta', 'mask')},
        scores=dict(logit_dd=dd_scalar, endpoint_normal_set=endpoint_score,
                    history_raw=count * hs['raw']['score'], history_normal_set=hscore,
                    history_normal_set_single_block=hsum, logit_plus_history=joint),
        endpoint_interval=endpoint_interval, history=hs,
        history_explanation_coverage=float(np.isfinite(lower).all(axis=2).mean()),
        intervals=interval_records,
        clean_accuracy_pre=float((logits['pre_clean'].argmax(1) == audit_y).float().mean()),
        clean_accuracy_post=float((logits['post_clean'].argmax(1) == audit_y).float().mean()),
        clean_logit_change_rms=float(clean_change.square().mean().sqrt()),
        clean_class_accuracy={str(c): [float((logits[k].argmax(1)[audit_y == c] == c).float().mean())
                                      for k in ('pre_clean', 'post_clean')] for c in range(10)},
        probe_target_fraction={k: float((v.argmax(1)[use] == target).float().mean()) for k, v in logits.items()},
        asr=None, asr_note='searched perturbation target rates are NOT true attack ASR',
        comparison_conditions={k: trajectory.protocol['options'].get(k) for k in
            (('rounds', 'clients', 'participant_rate', 'non_iid', 'alpha') if trajectory.family == 'fuba'
             else ('rounds', 'local_epochs'))},
        actual_fu_audit=candidate['candidate_kind'] != 'historical_parameter_subtraction_surrogate',
        attack_verdict=None, elapsed_seconds=time.perf_counter() - start)
    with (out / 'observations.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    write(out / 'scores.json', score)
    protocol.update(status='complete', candidate=candidate, source_group='deleted union / retained union',
                    output_sha256={n: sha256(out / n) for n in ('scores.json', 'observations.npz')})
    write(out / 'protocol.json', protocol)
    return score


def summarize(rows, out, missing):
    groups = {}
    for row in rows:
        condition = fingerprint(row['comparison_conditions'])[:8]
        key = f"{row['family']}/{row['dataset']}/{row['candidate']['candidate_kind']}/smoke={row['smoke']}/conditions={condition}"
        groups.setdefault(key, []).append(row)
    comparison = {}
    for key, group in groups.items():
        measures = {}
        for method in group[0]['scores']:
            pos = [r['scores'][method] for r in group if r['role'] == 'attack_request' and r['scores'][method] is not None]
            neg = [r['scores'][method] for r in group if r['role'] == 'benign_request' and r['scores'][method] is not None]
            measures[method] = dict(descriptive_auc=empirical_auc(pos, neg), attack_measured=len(pos),
                benign_measured=len(neg), abstentions=sum(r['scores'][method] is None for r in group))
        paired = [r for r in group if r['scores']['logit_plus_history'] is not None]
        paired_measures = {}
        for method in ('logit_dd', 'history_normal_set', 'logit_plus_history'):
            paired_measures[method] = empirical_auc(
                [r['scores'][method] for r in paired if r['role'] == 'attack_request'],
                [r['scores'][method] for r in paired if r['role'] == 'benign_request'])
        comparison[key] = dict(methods=measures, same_covered_requests_auc=paired_measures,
            conditions=group[0]['comparison_conditions'], total_requests=len(group),
            independent_training_folders=len({r['run'] for r in group}))
    write(out / 'comparison.json', dict(groups=comparison, missing=missing, calibrated=False,
        limitations=['AUC is descriptive, not a confidence interval or significance test',
            'abstentions remain missing and paired comparison uses exactly the same covered requests',
            'multiple deletions from one model are correlated, not independent runs',
            'BadFU clean5 vs full6 differs in client count/data composition: unmatched pilot control',
            'actual-FU discrimination needs actual benign FU, not surrogate controls',
            'no result proves malicious intent or exact unlearning correctness']))
    lines = ['# Saved-model deletion comparison', '',
        'Research scores. `unknown` is abstention, never a benign prediction. No calibrated FPR/FNR.', '',
        '| Case | Candidate | DD | History normal set | DD + history | Coverage |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    fmt = lambda v: 'unknown' if v is None else f'{v:.6g}'
    for r in rows:
        s = r['scores']
        lines.append(f"| {r['name']} | {r['candidate']['candidate_kind']} | {fmt(s['logit_dd'])} | "
                     f"{fmt(s['history_normal_set'])} | {fmt(s['logit_plus_history'])} | {r['history_explanation_coverage']:.1%} |")
    lines += ['', 'Use comparison.json for matched-coverage AUC and abstention counts.',
        'Actual BadFU has no automatically available clean5 actual-FU control. FUBA defaults are surrogates.',
        'No FL training or FU was executed. Full saved trajectory I/O and Jacobian inference can be substantial.',
        'Public test images are reused benchmark development data; reserve fresh data before a final claim.', '',
        'Missing inputs (not trained automatically):'] + [f'- {p}' for p in missing]
    (out / 'report.md').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in rows:
        color = 'tab:red' if r['role'] == 'attack_request' else 'tab:blue'
        marker = 'x' if r['actual_fu_audit'] else 'o'
        h = r['scores']['history_normal_set']
        if h is not None:
            axes[0].scatter(r['scores']['logit_dd'], h, color=color, marker=marker)
            axes[0].annotate(r['name'], (r['scores']['logit_dd'], h), fontsize=6)
        axes[1].bar(r['name'], r['history_explanation_coverage'], color=color)
    axes[0].set(xlabel='Held-out logit double difference', ylabel='Normal-set history score',
                title='Pilot scores (red attack; blue benign; x actual FU)')
    axes[1].set(ylabel='Explained round/group fraction', ylim=(0, 1.05), title='Coverage, not accuracy')
    axes[1].tick_params(axis='x', labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)


def export_results(out, destination):
    import shutil
    import zipfile
    inventory = {}
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.npz', '.md', '.png'):
            inventory[p.relative_to(out).as_posix()] = sha256(p)
    digest = fingerprint(inventory)[:16]
    dest = resolve(destination) / digest
    dest.mkdir(parents=True, exist_ok=True)
    bundle = dest / 'deletion_comparison.zip'
    if not bundle.exists():
        with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as z:
            for name in inventory:
                z.write(out / name, name)
    else:
        with zipfile.ZipFile(bundle) as z:
            if set(z.namelist()) != set(inventory) or any(hashlib.sha256(z.read(n)).hexdigest() != h for n, h in inventory.items()):
                raise ValueError('existing export differs from observations')
    write(dest / 'export.json', dict(schema=1, sha256=sha256(bundle), files=inventory,
        contents='logits, Jacobian observations, intervals, public IDs/labels and synthetic delta; NO images/weights'))
    for name in ('comparison.json', 'report.md', 'comparison.png'):
        shutil.copyfile(out / name, dest / name)
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, help='explicit JSON plan; relative paths are repository-relative')
    p.add_argument('--write-plan', type=Path, help='write discovered plan and exit; never overwrite')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--badfu-run')
    p.add_argument('--fuba-run')
    p.add_argument('--fuba-benign-run', nargs='+')
    p.add_argument('--data-root')
    p.add_argument('--out', type=Path, default=Path('artifacts/deletion_comparison_v1'))
    p.add_argument('--export', default='reproduction/validation/exports/deletion_comparison')
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--smoke', action='store_true', help='2 images/class/split, 2 search steps, 1 restart; all recorded rounds')
    args = p.parse_args()
    if args.plan and any((args.badfu_run, args.fuba_run, args.fuba_benign_run)):
        p.error('choose an explicit plan or discovery paths')
    plan = read(args.plan) if args.plan else discover(args)
    if plan.get('schema') != 1:
        p.error('schema 1 plan required')
    if args.smoke:
        plan['settings'] = DEFAULT | plan.get('settings', {}) | dict(reference_per_class=2,
            search_per_class=2, audit_per_class=2, steps=2, restarts=1)
    cfg = settings(plan)
    names = [c['name'] for c in plan['cases']]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in ('.', '..') for n in names):
        p.error('unique simple case names required')
    if args.write_plan:
        if args.write_plan.exists():
            p.error('--write-plan never overwrites a file')
        write(args.write_plan, plan)
        print(f'Plan saved: {args.write_plan}')
        return
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print('NO training / FU / downloads. Actual FUBA needs an explicit actual_state binding.')
        return
    if not names or args.threads < 1:
        p.error('no saved runs found or invalid threads; use --dry-run and explicit paths')
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    args.out.mkdir(parents=True, exist_ok=True)
    saved_plan = args.out / 'plan.json'
    if saved_plan.exists() and read(saved_plan) != plan:
        p.error('plan changed; use a new --out')
    write(saved_plan, plan)
    rows = []
    for case in plan['cases']:
        print(f"\n[{len(rows)+1}/{len(names)}] {case['name']}", flush=True)
        rows.append(run_case(case, cfg, args.out / case['name'], device, args.data_root))
        summarize(rows, args.out, plan.get('missing', []))
    dest = export_results(args.out, args.export)
    print(f'Completed. Review {args.out / "report.md"}\nUpload folder: {dest}', flush=True)


if __name__ == '__main__':
    main()
