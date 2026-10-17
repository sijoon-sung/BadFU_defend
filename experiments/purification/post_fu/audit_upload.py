"""Verify a portable purification report; does not replay models or certify FU."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b, message, tolerance=1e-6):
    require(abs(a-b) <= tolerance, message)


def audit(folder):
    folder = Path(folder)
    receipt = json.loads((folder/'export.json').read_bytes())
    archive = (folder/'purification.zip').read_bytes()
    require(digest(archive) == receipt['zip_sha256'], 'ZIP checksum mismatch')
    with zipfile.ZipFile(folder/'purification.zip') as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'duplicate ZIP names')
        require(set(names) == set(receipt['files']), 'ZIP inventory differs')
        require(all(Path(n).name == n for n in names), 'unsafe archive entry')
        raw = {n: z.read(n) for n in names}
    for name, expected in receipt['files'].items():
        require(digest(raw[name]) == expected, f'archive member checksum mismatch: {name}')
    copies = {}
    for name in ('results.json', 'report.md', 'comparison.png'):
        local = (folder/name).read_bytes()
        if local == raw[name]:
            copies[name] = 'byte_identical'
        else:
            require(name.endswith(('.json', '.md')) and
                    local.replace(b'\r\n', b'\n') == raw[name].replace(b'\r\n', b'\n'),
                    f'published copy differs beyond line endings: {name}')
            copies[name] = 'line_endings_only; ZIP remains the byte-authoritative record'
    data = {n: json.loads(v) for n, v in raw.items() if n.endswith('.json')}
    protocol, prep, results = (data[n] for n in ('protocol.json', 'preparation.json', 'results.json'))
    cfg = protocol['config']
    require(protocol['status'] == 'complete', 'incomplete experiment')
    require(protocol['smoke'] == prep['smoke'], 'smoke flags differ')
    require(all(cfg[k] == v for k, v in prep['config'].items()), 'preparation/config mismatch')
    for name, expected in protocol['output_hashes'].items():
        if name in raw:
            require(digest(raw[name]) == expected, f'output checksum mismatch: {name}')
    for method, files in protocol['completed_arms'].items():
        for name, expected in files.items():
            if name in raw:
                require(digest(raw[name]) == expected, f'completed arm checksum mismatch: {name}')
    source = prep['source']
    for spec, phase in ((cfg['pre'], 'full6'), (cfg['post'], 'fu_remove5')):
        require(protocol['input_hashes'][spec['path']] == source[phase]['endpoint_sha256'],
                f'recorded endpoint binding mismatch: {phase}')
    pre, post = source['full6']['metadata'], source['fu_remove5']['metadata']
    require(pre['client_ids'] == list(range(6)) and post['client_ids'] == list(range(5)) and
            pre['counts'][:5] == post['counts'], 'inconsistent recorded client-5 request')
    require(source['fu_remove5']['completed_rounds'] == source['full6']['completed_rounds']//2,
            'recorded FU round budget mismatch')
    require(prep['trigger_record_sha256'] == protocol['attack_evaluation']['record_sha256'],
            'trigger record binding mismatch')
    source_identity = {}
    for name, expected in (
            ('purification_compare.py', protocol['source_hash']),
            ('purification_math.py', protocol['math_source_hash']),
            ('run_purification.py', prep['source_sha256'])):
        current = Path(__file__).with_name(name).read_bytes()
        source_identity[name] = dict(recorded_sha256=expected, current_sha256=digest(current),
                                     identical=digest(current) == expected)
    methods = list(protocol['completed_arms'])
    require(set(methods) <= set(results), 'missing reported arm')
    n = sum(results['post']['class_counts'].values())
    require(n == 10*prep['counts_per_class'][2], 'test size disagrees with public split')
    rows = {}
    for method, row in results.items():
        require(row['class_counts'] == results['post']['class_counts'], 'unmatched clean evaluation')
        require(row['asr_n'] == results['post']['asr_n'], 'unmatched ASR denominator')
        near(sum(row['class_accuracy'][c]*count for c, count in row['class_counts'].items())/n,
             row['clean_accuracy'], 'class accuracy arithmetic mismatch')
        for value, count in ((row['clean_accuracy'], n), (row['asr'], row['asr_n']),
                             (row['triggered_original_accuracy'], row['asr_n'])):
            require(0 <= value <= 1, 'invalid rate')
            near(value*count, round(value*count), 'rate cannot represent an integer count', 1e-4)
        drop = results['post']['clean_accuracy']-row['clean_accuracy']
        if method in methods:
            near(row['clean_drop_vs_post'], drop, 'clean-drop arithmetic mismatch')
            require(row['within_clean_tolerance'] == (drop <= cfg['clean_tolerance']),
                    'clean-tolerance flag mismatch')
            near(row['worst_class_drop_vs_post'], max(results['post']['class_accuracy'][c]-v
                 for c, v in row['class_accuracy'].items()), 'class-drop arithmetic mismatch')
        rows[method] = dict(clean_percent=100*row['clean_accuracy'], asr_percent=100*row['asr'],
            clean_drop_pp=100*drop, inferred_asr_count=round(row['asr']*row['asr_n']))
    constrained = data['differential_cvar_constrained_search.json']
    trace = constrained['repair_trace']
    require(constrained['reused_patterns_from'] == 'differential_cvar' and
            constrained['records'] == data['differential_cvar_search.json']['records'],
            'declared shared search records differ')
    decision = trace[-1]
    if decision['fallback_to_post']:
        require(decision['selected_epoch'] == 0 and not any(
            r['feasible_on_development_and_validation'] for r in trace[:-1]),
            'fallback contradicts recorded feasibility decisions')
        for key in ('clean_accuracy', 'asr', 'class_accuracy', 'triggered_original_accuracy'):
            require(results['post'][key] == results['differential_cvar_constrained'][key],
                    'fallback metrics differ from post')
    # The trace lacks initial per-class CE; feasibility itself cannot be recomputed from this ZIP.
    return dict(schema=1, archive_sha256=receipt['zip_sha256'], inventory_verified=True,
        published_copies=copies, source_identity=source_identity,
        scope='checksums, recorded pair bindings, report arithmetic and fallback consistency; no model inference replay',
        evidence_scope=protocol['evidence_scope'], smoke=prep['smoke'],
        source_rounds={k: source[k]['completed_rounds'] for k in ('full6','fu_remove5','bd5')},
        public_counts_per_class=prep['counts_per_class'], clean_n=n, asr_n=results['post']['asr_n'],
        metrics=rows, constraint=dict(rejected_epochs=sum(not r['feasible_on_development_and_validation']
            for r in trace[:-1]), **decision),
        objective={'correct_weight':cfg['correct_weight'], 'tail_mass':cfg['tail_mass'],
                   'class_ce_tolerance':cfg['class_ce_tolerance'], 'clean_tolerance':cfg['clean_tolerance']},
        timing_seconds={m: data[m+'_search.json']['total_seconds'] for m in methods},
        limits=['one BadFU model pair; no FUBA or benign actual-FU repair comparison',
                'models and public image arrays are not in the uploaded archive',
                'initial per-class CE and per-example predictions are not exported',
                'same public benchmark reused; not a new untouched test set',
                'constrained arm also clips gradients; not a pure constraint-only ablation',
                'forgetting quality and a history advantage are not measured'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(verified=result['inventory_verified'], clean_n=result['clean_n'],
        asr_n=result['asr_n'], constraint=result['constraint'], report=str(args.out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
