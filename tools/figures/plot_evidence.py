"""Rebuild descriptive figures from the uploaded registry; no model execution."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'FUBA/logs/trajectories/registry.json'
OUT = ROOT / 'docs/figures'


def main():
    records = json.loads(SOURCE.read_text(encoding='utf-8'))['records']
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    names = [f'bd_iid_r100w5_k20paper_cifar_s{i}' for i in range(3)]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(3)
    for ax, metric, title in zip(axes, ['acc', 'asr'], ['Clean accuracy (%)', 'Attack success rate (%)']):
        for offset, key, label, color in [(-.19, 'dormant', 'Final FL model', '#246b64'), (.19, 'sup_all', 'Contribution subtraction', '#be7554')]:
            values = [records[n]['metrics'][key][metric] for n in names]
            bars = ax.bar(x+offset, values, .36, label=label, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set(xticks=x, xticklabels=['522', '523', '524'], xlabel='Training seed', ylim=(0, 113), title=title)
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle('CIFAR-10 / ResNet-18: high pre-deletion ASR and subtraction collapse', fontsize=13)
    fig.text(.5, .02, 'K=20, 100 rounds, participation=0.4. Subtraction is not actual FU. Three development runs; no inferential error bars.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(OUT/'cifar_trajectory_audit.png', dpi=180)
    fig.savefig(OUT/'cifar_trajectory_audit.svg')
    plt.close(fig)
    clean = ['clean_iid_r100_k20paper_s0', 'clean_owner_r100_k20paper_s0',
             'clean_iid_r100_k20paper_cifar_s0', 'clean_owner_r100_k20paper_cifar_s0']
    fig, ax = plt.subplots(figsize=(9,4.8))
    for i, name in enumerate(clean):
        m = records[name]['metrics']
        a, b = m['dormant']['acc'], m['benign']['5']['acc']
        ax.plot([a,b], [i,i], color='#b8b8b8', lw=3)
        ax.scatter(a,i,color='#246b64',s=65,label='Final FL model' if i==0 else None,zorder=3)
        ax.scatter(b,i,color='#be7554',s=65,label='Benign client 5 subtraction' if i==0 else None,zorder=3)
        ax.annotate(f'{a:.2f}',(a,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
        ax.annotate(f'{b:.2f}',(b,i),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=9)
    ax.set(yticks=range(4),yticklabels=['MNIST / IID','MNIST / owner','CIFAR-10 / IID','CIFAR-10 / owner'],
           xlim=(0,105),ylim=(-.65,3.65),xlabel='Clean accuracy (%)',title='Utility loss also occurs without an attacker')
    ax.invert_yaxis()
    ax.legend(loc='lower left',fontsize=9)
    ax.grid(axis='x',alpha=.15)
    fig.text(.5,.015,'One trajectory per condition (seed 522). No detector alarms or ASR were measured in these clean records.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(OUT/'clean_trajectory_audit.png',dpi=180)
    fig.savefig(OUT/'clean_trajectory_audit.svg')
    plt.close(fig)
    for name in names+clean:
        m=records[name]['metrics']
        for stage, values in [('final_fl', m['dormant']), ('benign5_subtraction',m['benign']['5'])]+([('suppressor_subtraction',m['sup_all'])] if name in names else []):
            rows.append({'trajectory':name,'stage':stage,'acc_pct':values['acc'],'asr_pct':values.get('asr')})
    with (OUT/'trajectory_audit.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT/'provenance.json').write_text(json.dumps({'source':str(SOURCE.relative_to(ROOT)).replace('\\','/'),
        'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'upstream_result_commit':'90d2f33',
        'interpretation':'Descriptive development diagnostics, not detector or FU defense results.'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
