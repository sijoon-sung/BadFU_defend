"""Plot all client scores for the fixed primary and held-out probe checks."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path, default=Path('experiments/detection/hrac/results/historical/hrac/views_badfu_v1/analysis.json'))
    p.add_argument('--heldout', type=Path, default=Path('experiments/detection/hrac/results/historical/hrac/views_holdout_v1/analysis.json'))
    p.add_argument('--out', type=Path, default=Path('docs/figures/hrac_probe_comparison'))
    a = p.parse_args()
    first = json.loads(a.initial.read_text())['rows']
    second = json.loads(a.heldout.read_text())['rows']
    panels = []
    for view, coord, contrast, title in (
        ('clean', 'logit', 'raw', 'Clean inputs: centered logits'),
        ('blind_patch_family', 'logit', 'raw', 'Generic patches: raw response'),
        ('blind_patch_family', 'logit', 'paired', 'Generic patches: paired contrast')):
        row = next(x for x in first if x['method']=='finite' and x['coordinate']==coord
                   and x['contrast']==contrast and x['view']==view)
        panels.append((title, row))
    for i, seed in enumerate((20261016, 20261017, 20261018)):
        panels.append((f'New images + new pattern {i+1}', next(x for x in second if x['view']==f'family_{seed}')))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    colors = ['#cb7c32'] + ['#8d9ca8']*4 + ['#bc3347']
    for ax, (title, row) in zip(axes.flat, panels):
        rows = sorted(row['ranking'], key=lambda x:x['unit'])
        bars = ax.bar(range(6), [r['score'] for r in rows], color=colors, width=.68)
        ax.bar_label(bars, labels=[f"{r['score']:.2f}" for r in rows], padding=3, fontsize=9)
        ax.set_title(f"{title}\nRequester 5 rank: {row['requester5_rank']}/6", fontsize=11, pad=12)
        ax.set_xticks(range(6), [str(i) for i in range(6)])
        ax.set_xlabel('Client'); ax.set_ylabel('Uncalibrated score')
        ax.set_ylim(0, max(r['score'] for r in rows)*1.25)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.18)
    fig.suptitle('BadFU: paired probe responses change the requester ranking', fontsize=16, y=.985)
    fig.legend(handles=[Patch(color=colors[0], label='Injection client 0'),
                        Patch(color=colors[1], label='Benign clients 1-4'),
                        Patch(color=colors[5], label='Camouflage requester 5')],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5,.035))
    fig.text(.5,.012,'One saved 15-round trajectory. Different input patterns are not independent training runs. No calibrated FPR/TPR.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0,.09,1,.95))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        target=a.out.with_suffix(suffix)
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=180, facecolor='white')
        if suffix == '.svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines())+'\n',
                              encoding='utf-8', newline='\n')
    print(a.out)


if __name__=='__main__':
    main()
