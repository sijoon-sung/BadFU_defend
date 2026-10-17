"""Audit uploaded endpoint/history measurements without loading or training models.

Preserves original artifacts. Distinguishes exact-zero group contributions from
explained nonzero directions, and diagnoses reference-span failure on clean data.
"""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.optimize import linprog

from experiments.detection.deletion_comparison.deletion_contrast import empirical_auc
from reproduction.common.deletion_inputs import ROOT, write
from experiments.detection.deletion_comparison.replay_deletion_comparison import replay, check_close
from experiments.detection.deletion_comparison.run_deletion_comparison import fingerprint
from reproduction.common.paths import resolve_recorded
from reproduction.common.trajectory import sha256


def current_source_status(recorded):
    """Keep old result paths intact, and report current source identity explicitly."""
    # Composes both explicit layout migrations; receipts are left unchanged.
    result = {}
    for name, expected in recorded.items():
        path = resolve_recorded(name)
        relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        available = path.is_relative_to(ROOT) and path.is_file()
        digest = sha256(path) if available else None
        result[name] = dict(current_path=relative, available=available,
                            recorded_sha256=expected, current_sha256=digest,
                            matches=expected == digest if available else None)
    return result


def span_diagnostic(clean_basis, clean, norms, intervals):
    active = norms > 1e-12
    if not active.any():
        return dict(nonzero_directions=0, reference_rank=int(np.linalg.matrix_rank(clean_basis)),
                    least_squares_unexplained_rms_fraction=None, first_last_nonzero_checks=[])
    responses = clean[active] / norms[active, None]
    coefficients = np.linalg.lstsq(clean_basis, responses.T, rcond=None)[0]
    residual = responses - coefficients.T @ clean_basis.T
    denominator = np.sqrt(np.mean(responses ** 2, axis=1))
    residual_fraction = np.divide(np.sqrt(np.mean(residual ** 2, axis=1)), denominator,
                                  out=np.zeros_like(denominator), where=denominator > 0)
    # A clean-only diagnosis, NOT a replacement explanation set or retuned score.
    # Prespecified first/last nonzero direction per group; no probe scores select it.
    examples = []
    for group in range(2):
        indices = np.flatnonzero(active[:, group])
        if not len(indices):
            continue
        for t in sorted(set((int(indices[0]), int(indices[-1])))):
            c = clean[t, group] / norms[t, group]
            b = clean_basis
            matrix = np.vstack((np.column_stack((b, -np.ones(len(c)))),
                                np.column_stack((-b, -np.ones(len(c))))))
            result = linprog(np.r_[np.zeros(b.shape[1]), 1.], A_ub=matrix,
                b_ub=np.r_[c, -c], bounds=[(None, None)] * b.shape[1] + [(0., None)], method='highs')
            if not result.success:
                raise RuntimeError('unbounded clean-span LP failed')
            record = intervals[t][group]
            examples.append(dict(round=t+1, group='deleted' if group == 0 else 'retained',
                unbounded_error=float(result.fun), configured_error=record['clean_error'],
                unbounded_error_over_tolerance=float(result.fun / record['clean_error']),
                bounded_error=record.get('minimum_clean_error_diagnostic')))
    return dict(nonzero_directions=int(active.sum()), reference_rank=int(np.linalg.matrix_rank(clean_basis)),
        least_squares_unexplained_rms_fraction=dict(min=float(residual_fraction.min()),
            median=float(np.median(residual_fraction)), max=float(residual_fraction.max())),
        first_last_nonzero_checks=examples,
        limitation='Only these first/last samples have unbounded L-infinity LP checks; all nonzero directions have L2 diagnostics.')


def coverage_counts(intervals):
    counts = Counter(item['status'] for row in intervals for item in row)
    if set(counts) - {'measured', 'unexplained_clean_response', 'recorded_zero_aggregate'}:
        raise ValueError('unknown interval status')
    active = counts['measured'] + counts['unexplained_clean_response']
    return dict(total=sum(counts.values()), nonzero=active, explained_nonzero=counts['measured'],
        unexplained_nonzero=counts['unexplained_clean_response'], known_zero=counts['recorded_zero_aggregate'],
        explained_nonzero_fraction=counts['measured']/active if active else None)


def audit(bundle, out):
    out = Path(out)
    if out.exists():
        raise ValueError('audit never overwrites an existing output directory')
    verified = replay(bundle)
    rows, summaries = [], []
    with zipfile.ZipFile(bundle) as z:
        comparison = json.loads(z.read('comparison.json'))
        plan = json.loads(z.read('plan.json'))
        for entry in z.namelist():
            if not entry.endswith('/scores.json'):
                continue
            prefix = entry.rsplit('/', 1)[0]
            s = json.loads(z.read(entry))
            p = json.loads(z.read(prefix + '/protocol.json'))
            with np.load(io.BytesIO(z.read(prefix + '/observations.npz')), allow_pickle=False) as a:
                target, use = int(a['target']), a['non_target_mask']
                responses = {}
                for key in ('pre_clean', 'pre_probe', 'post_clean', 'post_probe'):
                    hits = int((a[key].argmax(1)[use] == target).sum())
                    responses[key] = dict(hits=hits, count=int(use.sum()), fraction=hits/int(use.sum()))
                    check_close(s['probe_target_fraction'][key], responses[key]['fraction'], 'target fraction')
                clean_change = a['post_clean'] - a['pre_clean']
                centered_clean = clean_change - clean_change.mean(-1, keepdims=True)
                diagnostic = span_diagnostic(a['clean_basis'], a['history_clean'],
                                            a['history_parameter_norm'], s['intervals'])
                rounding = float(np.max(np.abs(centered_clean.flatten() - a['endpoint_clean'])))
                scores = s['scores']
                source_status = current_source_status(p['inputs']['source'])
                row = dict(name=s['name'], family=s['family'], role=s['role'], candidate=s['candidate']['candidate_kind'],
                    seed=s['seed'], smoke=s['smoke'], clean_accuracy_pre=s['clean_accuracy_pre'],
                    clean_accuracy_post=s['clean_accuracy_post'], clean_images=len(a['audit_labels']),
                    selected={k: s['selected'][k] for k in ('probe_mode', 'position', 'target')},
                    scores=scores, target_responses=responses, coverage=coverage_counts(s['intervals']),
                    reported_coverage_including_known_zeros=s['history_explanation_coverage'],
                    span_diagnostic=diagnostic, centered_logit_reduction_max_abs_difference=rounding,
                    clean_logit_change_rms=s['clean_logit_change_rms'], elapsed_seconds=s['elapsed_seconds'],
                    source_matches_current={name: record['matches'] for name, record in source_status.items()},
                    current_source_status=source_status)
                rows.append(row)
                summaries.append(s)
        # Recheck published ranking tables, not just individual model outputs.
        grouped = {}
        for s in summaries:
            key = f"{s['family']}/{s['dataset']}/{s['candidate']['candidate_kind']}/smoke={s['smoke']}/conditions={fingerprint(s['comparison_conditions'])[:8]}"
            grouped.setdefault(key, []).append(s)
        if set(grouped) != set(comparison['groups']):
            raise ValueError('comparison strata mismatch')
        for key, cases in grouped.items():
            for method in cases[0]['scores']:
                positive = [s['scores'][method] for s in cases if s['role'] == 'attack_request' and s['scores'][method] is not None]
                negative = [s['scores'][method] for s in cases if s['role'] == 'benign_request' and s['scores'][method] is not None]
                stored = comparison['groups'][key]['methods'][method]
                check_close(stored['descriptive_auc'], empirical_auc(positive, negative), 'descriptive AUC')
                if (stored['attack_measured'] != len(positive) or stored['benign_measured'] != len(negative)
                        or stored['abstentions'] != sum(s['scores'][method] is None for s in cases)):
                    raise ValueError('comparison coverage mismatch')
    out.mkdir(parents=True)
    result = dict(schema=1, date='2026-10-17', bundle=str(bundle), bundle_sha256=sha256(bundle),
        observations_replayed=len(verified), ranking_tables_replayed=True, plan=plan,
        no_models_loaded=True, training_calls=0, fu_calls=0,
        total_reported_case_seconds=sum(r['elapsed_seconds'] for r in rows), cases=rows,
        interpretation=dict(actual_badfu='post-FU perturbation response increase observed; actual benign FU control missing',
            surrogate='large clean-accuracy degradation confounds interpreting logit DD as backdoor activation',
            history='all nonzero group contributions unexplained under the configured public class-gradient span',
            novelty='no observed benefit of the normal-explanation history method; coupled score abstains'))
    write(out / 'audit.json', result)
    lines = ['# Uploaded deletion comparison: audit', '',
        f'{len(rows)} cases replayed from the source observations; hashes and AUC tables checked.',
        'No model inference, new training, FU, or score retuning. Original upload is preserved.', '',
        '| Case | Clean accuracy pre → post | DD | Explained nonzero groups | Known-zero groups |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        c = r['coverage']
        lines.append(f"| {r['name']} | {100*r['clean_accuracy_pre']:.2f}% → {100*r['clean_accuracy_post']:.2f}% | "
            f"{r['scores']['logit_dd']:.4f} | {c['explained_nonzero']}/{c['nonzero']} | {c['known_zero']}/{c['total']} |")
    lines += ['', 'Known-zero contributions do not demonstrate that the normal reference span explains a nonzero update.',
        'The selected perturbation response is not the true attack-trigger ASR. Different logit scales are not comparable across models.',
        'AUC=1 in these few surrogate cases does not establish detection under actual FU or calibrated false-positive rates.']
    (out / 'report.md').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    plot(rows, out)
    return result


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    actual = next((r for r in rows if r['name'] == 'badfu_attack_actual'), None)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1.6]})
    if actual:
        responses = actual['target_responses']
        names = ['pre_clean', 'pre_probe', 'post_clean', 'post_probe']
        bars = axes[0].bar(range(4), [responses[k]['fraction']*100 for k in names],
                           color=['#3877a8', '#3877a8', '#d47b3e', '#d47b3e'])
        axes[0].bar_label(bars, labels=[f"{responses[k]['hits']}/{responses[k]['count']}" for k in names], padding=4)
        axes[0].set_xticks(range(4), ['Before\nclean', 'Before\npatch', 'After FU\nclean', 'After FU\npatch'])
        axes[0].set(ylim=(0, 100), ylabel='Selected-target prediction rate (%)', title='BadFU actual FU: held-out patch response')
    positions = np.arange(len(rows))
    before = np.array([r['clean_accuracy_pre']*100 for r in rows])
    after = np.array([r['clean_accuracy_post']*100 for r in rows])
    axes[1].hlines(positions, np.minimum(before, after), np.maximum(before, after), color='#aaaaaa')
    axes[1].scatter(before, positions, label='Before', color='#3877a8')
    axes[1].scatter(after, positions, label='Candidate / actual FU', color='#d47b3e')
    labels = [r['name'].replace('badfu_', 'B: ').replace('fuba_', 'F: ').replace('_', ' ') for r in rows]
    axes[1].set_yticks(positions, labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 100), xlabel='Clean accuracy on 80 public audit images (%)',
                title='Surrogate deletion also damages benign models')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'endpoint_review.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    zero = np.array([r['coverage']['known_zero'] for r in rows])
    unknown = np.array([r['coverage']['unexplained_nonzero'] for r in rows])
    explained = np.array([r['coverage']['explained_nonzero'] for r in rows])
    ax.barh(positions, zero, color='#b6b6b6', label='Known zero (not an explained update)')
    ax.barh(positions, unknown, left=zero, color='#d47b3e', label='Nonzero: unexplained')
    ax.barh(positions, explained, left=zero+unknown, color='#559a62', label='Nonzero: explained')
    ax.set_yticks(positions, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel='Recorded round × group observations', title='Separating zero contributions from explanation coverage')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'coverage_review.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.bundle, args.out)
    print(f"Audited {len(result['cases'])} cases. Report: {args.out / 'report.md'}")
