"""HRAC replay, mathematical stress checks, and independent-episode evaluation.

Commands do not train, delete contributions from a model, or measure ASR.
All input caches / configurations are fingerprinted in the output.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import beta
from FUBA.research.benchmark_tension import load_ledger, methods, score_ledger
from FUBA.research.hrac import attribution_residual, decide, inter_profile

HERE = Path(__file__).resolve().parents[1]
WINDOWS = (0, 10, 20)
CONFIG = {"schema": 1, "coordinates": "logprob/class_mean_before",
          "windows": list(WINDOWS), "inspection_rounds": [10, 20, 40, "last"],
          "requests": "each entire recorded unit, plus all units",
          "channels": ["inter", "intra"], "alpha": .05, "ridge": .001}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, report):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserve previous results; choose a new --out: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    report["source_sha256"] = {n: sha(Path(__file__).parent/n) for n in
        ("hrac.py", "experiment_hrac.py", "tension_audit.py", "benchmark_tension.py")}
    report["config"] = CONFIG
    report["model_unlearning_calls"] = 0
    report["training_calls"] = 0
    report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                         "platform": platform.platform()}
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"Saved {path}", flush=True)


def replay(path):
    k, t, ts, js, values = load_ledger(path)
    v = values['logprob']
    if v.ndim == 3:
        v = v.mean(1)
    h = np.zeros((t, k, v.shape[-1]))
    h[ts.astype(int)-1, js.astype(int)] = v
    snapshots, episode = [], {m: 0. for m in ('inter', 'selective_v3', 'linear_release')}
    for stop in sorted(set(min(s, t) for s in (10, 20, 40, t))):
        ranking = {m: [] for m in episode}
        for j in range(k+1):
            removal = np.ones(k) if j == k else np.eye(k)[j]
            old = methods(h[:stop].sum(0), removal)
            scores = {'inter': inter_profile(h[:stop], removal, WINDOWS),
                      'selective_v3': old['selective_v3'], 'linear_release': old['linear_release']}
            for method, by_class in scores.items():
                c = int(np.argmax(by_class))
                maximum = float(by_class[c])
                ranking[method].append({'unit': j if j < k else 'all', 'coordinate': c,
                    'score': maximum, 'coordinate_ties': np.flatnonzero(by_class == maximum).tolist()})
                episode[method] = max(episode[method], maximum)
        for rows in ranking.values():
            rows.sort(key=lambda row: -row['score'])
        snapshots.append({'round': stop, 'ranking': ranking})
    return {'cache': str(Path(path)), 'sha256': sha(path), 'K': k, 'T': t,
            'channel_scores': {'inter': episode['inter'], 'intra': None},
            'baseline_episode_scores': {m: episode[m] for m in ('selective_v3', 'linear_release')},
            'intra_status': 'no independently observed request / within-unit atoms in this cache',
            'snapshots': snapshots}


def collect_badfu(manifest_path, out, probe=80, n_gpu=1):
    """Read saved public-code BadFU FL rounds; never invoke FU or retraining.

    Probe the positive counts-weighted parameter update. BatchNorm buffers are
    frozen at each recorded pre-round global state, not recalibrated. This is
    explicitly a parameter-effect measurement; BN-buffer effects are excluded.
    """
    import torch
    from FUBA.research.collect_tension import outputs
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    if out.exists() or out.with_suffix('.json').exists():
        raise FileExistsError(out)
    badfu = Path(__file__).resolve().parents[2] / 'BadFU_src' / 'BadFU-main'
    sys.path.insert(0, str(badfu))
    from request_pipeline import load_tensor, load_event
    from badfu_common import ResNet18
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or not manifest['rounds']:
        raise ValueError('complete saved FL trajectory required')
    archive = load_tensor(manifest['dataset'])
    counts = np.asarray(manifest['counts'], dtype=float)
    if (counts <= 0).any() or archive['counts'] != manifest['counts']:
        raise ValueError('invalid sample counts')
    images = archive['validation']['images']
    if not 1 <= probe <= len(images):
        raise ValueError(f'probe must be between 1 and {len(images)}')
    device = torch.device('cuda:0' if torch.cuda.is_available() and n_gpu else 'cpu')
    xs = images[:probe].to(device)
    model = ResNet18(pretrained=False).to(device)
    keys = set(dict(model.named_parameters()))
    ts, js, lp, logits = [], [], [], []
    started = time.perf_counter()
    for t, record in enumerate(manifest['rounds']):
        if record['event'] != t:
            raise ValueError('rounds must be consecutive and ordered')
        event, after = load_event(record, manifest['counts'])
        before = event['gm']
        model.load_state_dict(before); model.eval()
        baseline = outputs(model, xs)
        for j, state in enumerate(event['cms']):
            weight = counts[j] / counts.sum()
            updated = {key: (v + weight * (state[key] - v) if key in keys else v)
                       for key, v in before.items()}
            model.load_state_dict(updated)
            measured = outputs(model, xs)
            ts.append(t+1); js.append(j)
            lp.append(measured[0]-baseline[0]); logits.append(measured[1]-baseline[1])
        del event, after, updated
        print(f'BadFU positive-update probes: {t+1}/{len(manifest["rounds"])}', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, K=len(counts), T=len(manifest['rounds']), ts=np.array(ts), js=np.array(js),
                        logprob_vs=np.stack(lp), logit_vs=np.stack(logits))
    meta = {'schema': 1, 'variant': manifest['variant'], 'probe': probe,
            'measurement': 'per-input finite positive parameter-update effect; recorded pre-round BN buffers frozen',
            'model_unlearning_calls': 0, 'training_calls': 0, 'source_manifest': str(manifest_path),
            'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'source_data': manifest['dataset'], 'source_rounds': manifest['rounds'],
            'probe_tensor_sha256': hashlib.sha256(xs.cpu().numpy().tobytes()).hexdigest(),
            'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dependency_sha256': {str(p.name): sha(p) for p in
                (badfu/'request_pipeline.py', badfu/'badfu_common.py', Path(__file__).with_name('collect_tension.py'))},
            'device': str(device), 'torch': torch.__version__,
            'accelerator': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cache_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
            'seconds': time.perf_counter()-started, 'intra_request_observation': 'unavailable'}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    return meta



def request_bundle(path):
    """Explicit observations, never manufactured by summing the same atoms."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != 1 or not data.get('requests'):
        raise ValueError('nonempty schema-1 request bundle required')
    rows = []
    ids = set()
    coordinate_systems = set()
    for item in data['requests']:
        ident = item['id']
        if ident in ids:
            raise ValueError('duplicate request ID')
        ids.add(ident)
        meta = item['provenance']
        if (meta.get('independent_observation') is not True or not meta.get('history_source')
                or not meta.get('request_source') or not meta.get('coordinate_system')):
            raise ValueError('declare independent observation, historical source, request source and common coordinates')
        coordinate_systems.add(meta['coordinate_system'])
        if len(coordinate_systems) > 1:
            raise ValueError('incompatible coordinate systems within request family')
        # This declaration is an audit requirement, not cryptographic proof.
        kwargs = {key: item[key] for key in ('constraint_matrix', 'constraint_value', 'lower', 'upper', 'covariance')
                  if key in item}
        result = attribution_residual(item['atoms'], item['observed'], ridge=CONFIG['ridge'], **kwargs)
        rows.append({'id': ident, **result})
    return {'source': str(path), 'sha256': sha(path), 'requests': rows,
            'episode_max': max(row['score'] for row in rows)}


def import_plan(path):
    """Reuse completed caches from the existing experiment manifest; no training."""
    old = json.loads(Path(path).read_text(encoding='utf-8'))
    if not old.get('jobs'):
        raise ValueError('existing experiment_tension.py manifest with jobs required')
    episodes, unavailable = [], []
    for job in old['jobs']:
        cache = Path(job['cache'])
        if not cache.is_absolute():
            cache = Path(path).resolve().parent / cache
        if not cache.exists():
            unavailable.append({'id': job['id'], 'cache': str(cache)})
            continue
        # Imported jobs contain client-level observations only. An independent
        # request bundle must be supplied separately to enable the intra branch.
        episodes.append({'id': job['id'], 'source_episode': job['id'], 'seed': job['seed'],
            'stratum': job['stratum'], 'split': job['split'], 'benign': job['split'] in ('calibration', 'clean_test'),
            'measurement_id': f"fuba_mnist_K{job['K']}_R{job['rounds']}_probe{old['probe']}_classmean_client_v1",
            'ledger': str(cache.resolve())})
    return {'schema': 1, 'episodes': episodes, 'missing_caches': unavailable,
            'source_manifest_sha256': sha(path),
            'note': 'Only existing caches imported. No new FL jobs. Intra remains unavailable without independent requests.'}


def diagnose(path):
    result = score_ledger(path)
    rows = []
    for config, entries in result['analyses'].items():
        for method in ('selective_v3', 'linear_release'):
            snapshot = entries[method]['snapshots'][-1]
            ranking = [r for r in snapshot['ranking'] if r['request'].startswith('unit:') and r['request'].endswith('@1')]
            rows.append({'config': config, 'method': method, 'round': snapshot['round'], 'ranking': ranking})
    return {'cache': str(path), 'sha256': sha(path), 'analyses': rows}


def interval(successes, n):
    if not n:
        return None
    return {'rate': successes/n, 'n': n, 'count': successes,
            'ci95': [float(beta.ppf(.025, successes, n-successes+1)) if successes else 0.,
                     float(beta.ppf(.975, successes+1, n-successes)) if successes < n else 1.]}


def evaluate_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema') != 1 or not manifest.get('episodes'):
        raise ValueError('nonempty schema-1 episode manifest required')
    episodes, ids, sources, seed_keys, fingerprints = [], set(), set(), set(), set()
    for item in manifest['episodes']:
        if item['split'] not in ('calibration', 'clean_test', 'attack_test', 'attack_development'):
            raise ValueError('unknown split')
        # Source ID is the ORIGINAL FL run, even for different cache views.
        key = (item['stratum'], item['seed'])
        if item['id'] in ids or item['source_episode'] in sources or key in seed_keys:
            raise ValueError('duplicate episode / source / stratum-seed: split leakage')
        if not item.get('measurement_id'):
            raise ValueError('measurement_id required (probe, coordinates, model, deletion granularity)')
        ids.add(item['id']); sources.add(item['source_episode']); seed_keys.add(key)
        if item['split'] in ('calibration', 'clean_test') and item.get('benign') is not True:
            raise ValueError('explicit benign provenance required for clean episodes')
        row = dict(item)
        scores = {'inter': None, 'intra': None}
        row['inputs'] = {}
        for field, channel, reader in [('ledger', 'inter', replay), ('requests', 'intra', request_bundle)]:
            if item.get(field):
                source_path = (Path(path).resolve().parent / item[field]).resolve()
                result = reader(source_path)
                fingerprint = (field, result['sha256'])
                if fingerprint in fingerprints:
                    raise ValueError('duplicate input content cannot count as independent episodes')
                fingerprints.add(fingerprint)
                row['inputs'][field] = result
                scores[channel] = result['channel_scores'][channel] if field == 'ledger' else result['episode_max']
        if all(v is None for v in scores.values()):
            raise ValueError('episode needs an observed ledger or request bundle')
        row['channel_scores'] = scores
        episodes.append(row)
    groups = []
    for stratum in sorted({e['stratum'] for e in episodes}):
        group = [e for e in episodes if e['stratum'] == stratum]
        if len({e['measurement_id'] for e in group}) != 1:
            raise ValueError('incompatible measurement protocols within calibration stratum')
        calibration = {c: [e['channel_scores'][c] for e in group
                          if e['split'] == 'calibration' and e['channel_scores'][c] is not None]
                       for c in CONFIG['channels']}
        tests = []
        for episode in group:
            if episode['split'] == 'calibration':
                continue
            decision = decide(calibration, episode['channel_scores'], CONFIG['alpha'])
            tests.append({'id': episode['id'], 'split': episode['split'], **decision})
        # Missing/inadequate channels are not silently excluded from denominators.
        rates = {}
        for split in ('clean_test', 'attack_test', 'attack_development'):
            subset = [t for t in tests if t['split'] == split]
            rates[split] = interval(sum(t['alarm'] is True for t in subset), len(subset)) if (
                subset and all(t['alarm'] is not None for t in subset)) else None
        groups.append({'stratum': stratum, 'calibration_n': {c: len(v) for c, v in calibration.items()},
                       'decisions': tests, 'alarm_rates': rates})
    return {'mode': 'independent_episode_evaluation', 'manifest_sha256': sha(path),
            'groups': groups, 'episodes': episodes,
            'limitations': ['Declared source independence / benign provenance cannot be verified from score arrays alone.',
                            'FPR statement requires exchangeable clean episodes within each predeclared stratum.',
                            'No inference about ASR, exact forgetting, or malicious intent.']}


def synthetic_case(seed, imbalance):
    rng = np.random.default_rng(seed)
    d = 6
    rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
    scales = np.exp(rng.normal(0, .2, d))
    scales[0] *= imbalance
    atoms = np.diag(scales) @ rotation
    selected = np.zeros(d); selected[rng.choice(d, 2, replace=False)] = 1
    true_request = atoms.T @ selected
    observed = true_request + rng.normal(0, .015 * np.linalg.norm(true_request), d)
    args = dict(constraint_matrix=np.ones((1, d)), constraint_value=[2.])
    normal = attribution_residual(atoms, observed, **args)
    flipped = attribution_residual(atoms, -observed, **args)
    total = atoms.sum(0)
    proportional_residual = np.linalg.norm(observed - total * (observed @ total)/(total @ total))
    return {'clean': normal['score'], 'altered': flipped['score'], 'norm': float(np.linalg.norm(observed)),
            'naive_proportional_residual': float(proportional_residual),
            'max_duality_gap': max(normal['duality_gap'], flipped['duality_gap']),
            'opposite_request_same_second_moment': bool(np.array_equal(np.outer(observed, observed), np.outer(-observed, -observed)))}


def stress_checks():
    """Controlled mathematical examples, explicitly NOT FUBA/BadFU measurements."""
    groups = []
    for name, imbalance in [('balanced_atoms', 1.), ('class_concentrated_atoms', 8.)]:
        cal = [synthetic_case(10000+i, imbalance) for i in range(99)]
        test = [synthetic_case(20000+i, imbalance) for i in range(200)]
        calibration = {'inter': [0.]*len(cal), 'intra': [c['clean'] for c in cal]}
        decisions, altered, norm_clean, norm_altered = [], [], [], []
        for case in test:
            decisions.append(decide(calibration, {'inter': 0., 'intra': case['clean']})['alarm'])
            altered.append(decide(calibration, {'inter': 0., 'intra': case['altered']})['alarm'])
            baseline_cal = {'intra': [c['norm'] for c in cal]}
            base = decide(baseline_cal, {'intra': case['norm']}, channels=('intra',))['alarm']
            norm_clean.append(base); norm_altered.append(base)
        groups.append({'stratum': name, 'calibration_n': len(cal),
            'normal_request_alarm': interval(sum(decisions), len(test)),
            'sign_flipped_request_alarm': interval(sum(altered), len(test)),
            'request_norm_baseline_normal_alarm': interval(sum(norm_clean), len(test)),
            'request_norm_baseline_altered_alarm': interval(sum(norm_altered), len(test)),
            'naive_proportional_residual_clean_median': float(np.median([c['naive_proportional_residual'] for c in test])),
            'all_matched_pairs_have_identical_second_moments': all(c['opposite_request_same_second_moment'] for c in test),
            'maximum_duality_gap': max(c['max_duality_gap'] for c in cal+test)})
    # Exact observational collision: a benign and an adversarial latent
    # explanation can have the same aggregate history and identical request.
    h = np.tile(np.array([[[1.], [-1.]]]), (20, 1, 1))
    observed_score = float(inter_profile(h, [0., 1.]).max())
    collision = decide({'inter': [observed_score]*99}, {'inter': observed_score}, channels=('inter',))
    return {'mode': 'synthetic_mathematical_stress_checks', 'groups': groups,
            'identifiability_counterexample': {'same_history_same_request_different_intent': True,
                'inter_score': observed_score, 'decision_against_matching_benign_controls': collision,
                'meaning': 'No method using only these identical observations can separate the two labels.'},
            'limitations': ['Synthetic atoms and sign-flipped observations are not attack implementations.',
                'The inter channel is zero in the GLRT stress experiment; this validates the intra branch only.',
                'Normal partial deletion is sampled from the stated bounded attribution model.',
                'Real atomic attribution approximation error / adaptive evasion are not measured.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('replay', 'stress', 'evaluate', 'import-plan', 'diagnose', 'collect-badfu'))
    p.add_argument('--caches', nargs='*', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--probe', type=int, default=80, help='clean input count for collect-badfu')
    p.add_argument('--n-gpu', type=int, default=1, help='0 uses CPU for collect-badfu')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error('output exists; choose a new --out to preserve previous results')
    if args.mode == 'collect-badfu':
        if not args.manifest or args.out.suffix != '.npz':
            p.error('collect-badfu requires --manifest and an .npz --out')
        meta = collect_badfu(args.manifest, args.out, args.probe, args.n_gpu)
        print(json.dumps({k: meta[k] for k in ('seconds', 'probe', 'training_calls', 'model_unlearning_calls')}))
        return
    if args.mode == 'replay':
        caches = args.caches or sorted((HERE/'logs'/'ledger').glob('bd_*_r100w5_k20paper_s*_push.npz'))
        if not caches:
            p.error('no existing caches found')
        report = {'mode': 'descriptive_existing_trajectory_replay', 'calibrated': False,
                  'runs': [replay(c) for c in caches],
                  'limitations': ['Development trajectories, no independent normal calibration.',
                                  'Intra unavailable: client sums are not within-client deletion atoms.',
                                  'Class-mean functional effects can hide per-input cancellation.']}
        for run in report['runs']:
            last = run['snapshots'][-1]
            print(Path(run['cache']).name, {m: rows[0] for m, rows in last['ranking'].items()}, flush=True)
    elif args.mode == 'diagnose':
        if not args.caches:
            p.error('--caches required')
        report = {'mode': 'exploratory_measurement_ablation', 'calibrated': False,
                  'runs': [diagnose(c) for c in args.caches],
                  'limitations': ['Post-hoc diagnostic; no configuration selected for confirmatory claims.',
                                  'Rankings are not FPR or attack success rates.']}
    elif args.mode == 'import-plan':
        if not args.manifest:
            p.error('--manifest required')
        report = import_plan(args.manifest)
        print(f"Reused {len(report['episodes'])} caches; {len(report['missing_caches'])} unavailable", flush=True)
    elif args.mode == 'stress':
        report = stress_checks()
        print(json.dumps(report['groups'], indent=2), flush=True)
    else:
        if not args.manifest:
            p.error('--manifest required')
        report = evaluate_manifest(args.manifest)
        print(json.dumps(report['groups'], indent=2), flush=True)
    write(args.out, report)


if __name__ == '__main__':
    main()
