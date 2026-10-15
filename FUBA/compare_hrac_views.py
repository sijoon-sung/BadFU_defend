"""Post-hoc score ablations on identical paired probe observations.

This checks whether the HRAC score adds a ranking advantage over simple signed
effects. It does not tune thresholds or supply independent test trajectories.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiment_hrac_views import sha, write_new, CONFIG
from hrac import inter_profile


def profiles(history):
    """Return all unit/class scores using the same observations and windows."""
    k = history.shape[1]
    sums = [history[-w:].sum(0) if w else history.sum(0) for w in CONFIG['windows']]
    return {
        'hrac_windows': np.stack([inter_profile(history, np.eye(k)[j], CONFIG['windows']) for j in range(k)]),
        'hrac_full_history_only': np.stack([inter_profile(history, np.eye(k)[j], (0,)) for j in range(k)]),
        'signed_release_windows': np.maximum(-np.stack(sums), 0).max(0),
        'absolute_effect_windows': np.abs(np.stack(sums)).max(0),
        'signed_release_last_round': np.maximum(-history[-1], 0),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path, default=Path('FUBA/logs/hrac/views_badfu_v1'))
    p.add_argument('--heldout', type=Path, default=Path('FUBA/logs/hrac/views_holdout_v1'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows, sources = [], []
    for label, folder in (('initial', a.initial), ('heldout', a.heldout)):
        path = folder/'effects.npz'
        expected = json.loads((folder/'analysis.json').read_text())['cache_sha256']
        if sha(path) != expected:
            raise ValueError('cache hash mismatch')
        effects = np.load(path, allow_pickle=False)['finite_logit'].astype(float)
        history = (effects - effects[:, :, :1]).mean(3)
        if label == 'initial':
            families = {'oracle_diagnostic': [1], 'blind_pattern': list(range(4, 8))}
        else:
            families = {f'blind_pattern_{seed}': list(range(2+4*i, 6+4*i))
                        for i, seed in enumerate((20261016, 20261017, 20261018))}
        for family, views in families.items():
            per_view = [profiles(history[:, :, v]) for v in views]
            for method in per_view[0]:
                scores = np.max([x[method] for x in per_view], axis=0)
                ranking = sorted([dict(unit=j, score=float(v.max()), coordinate=int(v.argmax()))
                                  for j, v in enumerate(scores)], key=lambda x: -x['score'])
                row = dict(stage=label, family=family, method=method, ranking=ranking,
                           requester5_rank=next(i+1 for i, x in enumerate(ranking) if x['unit'] == 5))
                rows.append(row)
                print(label, family, method, 'requester rank', row['requester5_rank'], 'top', ranking[0])
        sources.append(dict(stage=label, path=str(path), sha256=expected))
    write_new(a.out, dict(schema=1, stage='post-hoc score ablation after observing initial and held-out results',
        code_sha256=sha(__file__), sources=sources, windows=CONFIG['windows'], rows=rows,
        calibrated=False, training_calls=0, model_unlearning_calls=0,
        interpretation='Same probe family, centered-logit coordinates and unit/class maximization for every method. '
                       'A shared rank-one result does not demonstrate added detection value of the HRAC score. '
                       'All results reuse one previously inspected attacked trajectory.'))


if __name__ == '__main__':
    main()
