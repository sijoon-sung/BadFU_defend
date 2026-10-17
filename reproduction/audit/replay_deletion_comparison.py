"""Verify exported comparison scores from small observations; no models/images."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from reproduction.detection.deletion_contrast import explain, history_scores, minimum_clean_error


def check_close(actual, expected, name, rtol=2e-5, atol=2e-6):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f'{name}: abstention mismatch')
    elif not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        raise ValueError(f'{name}: numerical mismatch')


def replay_case(protocol, scores, arrays):
    cfg = protocol['inputs']['settings']
    a = arrays
    dd = (a['post_probe'] - a['pre_probe']) - (a['post_clean'] - a['pre_clean'])
    check_close(a['double_difference'], dd, 'raw logit double difference')
    use, target = a['non_target_mask'], int(a['target'])
    check_close(use, a['audit_labels'] != target, 'non-target image mask')
    centered_dd = dd - dd.mean(-1, keepdims=True)
    value = float(centered_dd[use, target].mean())
    check_close(scores['scores']['logit_dd'], value, 'held-out DD')
    clean = a['post_clean'] - a['pre_clean']
    clean = (clean - clean.mean(-1, keepdims=True)).flatten()
    check_close(a['endpoint_clean'], clean, 'endpoint clean observations', rtol=1e-4, atol=5e-5)
    check_close(a['endpoint_probe'], [value], 'endpoint probe observations')
    bc, bp = a['clean_basis'], a['probe_basis']
    norm = scores['endpoint_interval']['parameter_norm']
    if norm > 1e-12:
        item, eps = explain(a['endpoint_clean'] / norm, a['endpoint_probe'] / norm, bc, bp, cfg)
        if item.status != scores['endpoint_interval']['status']:
            raise ValueError('endpoint feasibility mismatch')
        check_close(scores['endpoint_interval']['clean_error'], eps, 'endpoint tolerance')
        for name in ('lower', 'upper'):
            bound = getattr(item, name)
            check_close(scores['endpoint_interval'][name], None if bound is None else norm * bound, name)
        endpoint = None if item.lower is None else max(0., float(item.lower[0] * norm))
        check_close(scores['scores']['endpoint_normal_set'], endpoint, 'endpoint corrected score')
        if item.lower is None:
            check_close(scores['endpoint_interval']['minimum_clean_error_diagnostic'],
                minimum_clean_error(a['endpoint_clean'] / norm, bc, cfg['coefficient_bound']),
                'endpoint minimum clean error')
    lower, upper, raw = a['history_lower'], a['history_upper'], a['history_raw']
    for t in range(len(raw)):
        for group in range(2):
            norm = a['history_parameter_norm'][t, group]
            record = scores['intervals'][t][group]
            if norm <= 1e-12:
                check_close(lower[t, group], [0.], 'zero interval')
                check_close(upper[t, group], [0.], 'zero interval')
                check_close(raw[t, group], [0.], 'zero response')
                continue
            item, eps = explain(a['history_clean'][t, group] / norm, raw[t, group] / norm, bc, bp, cfg)
            if record['status'] != item.status:
                raise ValueError('history feasibility mismatch')
            check_close(record['clean_error'], eps, 'history tolerance')
            if item.lower is None:
                check_close(record['minimum_clean_error_diagnostic'],
                    minimum_clean_error(a['history_clean'][t, group] / norm, bc, cfg['coefficient_bound']),
                    'history minimum clean error')
            for name, stored in (('lower', lower), ('upper', upper)):
                bound = getattr(item, name)
                check_close(record[name], None if bound is None else bound * norm, 'JSON interval')
                check_close(stored[t, group], [np.nan] if bound is None else bound * norm, 'array interval')
    blocks = [(r['start'], r['stop']) for r in scores['history']['raw']['blocks']]
    hs = history_scores(lower, upper, raw, blocks)
    h = None if hs['normal_set'] is None else len(raw) * hs['normal_set']['score']
    single = None if hs['normal_set_single_block'] is None else len(raw) * hs['normal_set_single_block']['score']
    expected = dict(history_normal_set=h, history_normal_set_single_block=single,
        history_raw=len(raw) * hs['raw']['score'], logit_plus_history=None if h is None else min(max(0., value), h))
    for name, v in expected.items():
        check_close(scores['scores'][name], v, name)
    check_close(scores['history_explanation_coverage'], np.isfinite(lower).all(2).mean(), 'coverage')
    for name, metric in (('pre_clean', 'clean_accuracy_pre'), ('post_clean', 'clean_accuracy_post')):
        check_close(scores[metric], (a[name].argmax(1) == a['audit_labels']).mean(), metric)
    return dict(name=scores['name'], observations_replayed=True, scores=scores['scores'])


def replay(bundle):
    bundle = Path(bundle)
    receipt = json.loads((bundle.parent / 'export.json').read_text(encoding='utf-8'))
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('ZIP hash mismatch')
    result = []
    with zipfile.ZipFile(bundle) as z:
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(receipt['files']):
            raise ValueError('ZIP inventory mismatch')
        for name, digest in receipt['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError(f'export file hash mismatch: {name}')
        for name in z.namelist():
            if not name.endswith('/protocol.json'):
                continue
            prefix = name.rsplit('/', 1)[0]
            protocol = json.loads(z.read(name))
            if protocol['status'] != 'complete':
                raise ValueError('incomplete case exported')
            for file, digest in protocol['output_sha256'].items():
                if hashlib.sha256(z.read(prefix + '/' + file)).hexdigest() != digest:
                    raise ValueError('case output hash mismatch')
            scores = json.loads(z.read(prefix + '/scores.json'))
            with np.load(io.BytesIO(z.read(prefix + '/observations.npz')), allow_pickle=False) as arrays:
                result.append(replay_case(protocol, scores, arrays))
    if not result:
        raise ValueError('no complete cases')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    args = p.parse_args()
    print(json.dumps(replay(args.bundle), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
