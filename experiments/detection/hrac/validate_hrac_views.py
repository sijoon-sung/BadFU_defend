"""Follow-up on held-out images and predeclared new probe patterns.

Reuses the same attacked FL trajectory, so this is NOT an independent attack
replication. No nuisance fitting, training, FU, or detector threshold tuning.
"""


from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
from reproduction.common.paths import research_source
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.detection.hrac.experiment_hrac_views import ROOT, CONFIG, make_views, sha, write_new
from experiments.detection.hrac.hrac import inter_profile
from experiments.shared.trajectory_tools.collect_tension import outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'artifacts/badfu/imported_traj_full/manifest.json')
    p.add_argument('--record-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(ROOT/'BadFU_src/BadFU-main'))
    from experiments.shared.trajectory_tools.badfu_artifacts import load_tensor, load_event
    from badfu_common import ResNet18
    torch.set_num_threads(4)
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    archive = load_tensor(manifest['dataset'])
    old, _, _, verified = make_views(archive, a.record_root, a.data_root)
    chosen = []
    for c in range(10):
        ids = torch.nonzero(archive['clean_eval']['labels']==c).flatten()
        if len(ids)<16:
            raise ValueError('need eight new images per class after the eight reference images')
        chosen.extend(ids[8:16].tolist())
    source_ids = [archive['clean_eval_ids'][i] for i in chosen]
    if set(source_ids) & set(verified['validation_ids']+verified['reference_ids']):
        raise ValueError('follow-up images overlap fitting or initial response images')
    x = archive['clean_eval']['images'][chosen]
    mean=torch.tensor([.4914,.4822,.4465]).reshape(3,1,1)
    std=torch.tensor([.2023,.1994,.2010]).reshape(3,1,1)
    pixel=x*std+mean
    mask=(old[1]-old[0]).abs().amax(0)>1e-6
    oracle=x.clone();oracle[:,mask]=old[1,0][mask]
    xs=[x,oracle];names=['clean','oracle'];seeds=(20261016,20261017,20261018)
    for seed in seeds:
        pattern=torch.randint(0,2,(3,4,4),generator=torch.Generator().manual_seed(seed)).float()
        for corner,(row,col) in zip(('tl','tr','bl','br'),((0,0),(0,28),(28,0),(28,28))):
            v=pixel.clone();v[:,:,row:row+4,col:col+4]=pattern
            xs.append((v-mean)/std);names.append(f'{seed}_{corner}')
    a.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema=1,stage='follow-up after initial exploratory result',seeds=list(seeds),
                  views=names,source_ids=source_ids,disjoint_from_initial_probes_and_references=True,
                  manifest_sha256=sha(a.manifest),code_sha256=sha(__file__),
                  windows=CONFIG['windows'],measurement='finite positive update, centered-logit paired contrast',
                  coordinates='all ten output classes; no known-target selection',
                  oracle='diagnostic only; excluded from blind family maxima',
                  decision='rank all six clients; no threshold or calibrated alarms',
                  known_limits=['same previously inspected attacked trajectory','not independent training seeds',
                                'generic patch family covers the real trigger location','not a clean-run FPR estimate'])
    write_new(a.out/'protocol.json',protocol)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=ResNet18().to(device).eval();keys=set(dict(model.named_parameters()))
    inputs=torch.stack(xs).flatten(0,1).to(device)
    count=np.asarray(manifest['counts'],float);weights=count/count.sum()
    effects=[];sources=[];started=time.perf_counter()
    for t,record in enumerate(manifest['rounds']):
        event,_=load_event(record,manifest['counts']);before=event['gm']
        model.load_state_dict(before);base=outputs(model,inputs)[1].reshape(len(xs),80,10)
        local=[]
        for j,state in enumerate(event['cms']):
            updated={k:v+weights[j]*(state[k]-v) if k in keys else v for k,v in before.items()}
            model.load_state_dict(updated)
            local.append(outputs(model,inputs)[1].reshape(len(xs),80,10)-base)
        effects.append(np.stack(local));sources.append(record)
        print(f'Held-out input check {t+1}/{len(manifest["rounds"])}, {time.perf_counter()-started:.1f}s',flush=True)
    effects=np.stack(effects);paired=effects-effects[:,:,:1]
    scores=[]
    for v in range(1,len(xs)):
        h=paired[:,:,v].mean(2)
        scores.append(np.stack([inter_profile(h,np.eye(6)[j],CONFIG['windows']) for j in range(6)]))
    rows=[]
    def append(name,s):
        ranking=sorted([dict(unit=j,coordinate=int(v.argmax()),score=float(v.max()),
                             scores_by_class=v.tolist()) for j,v in enumerate(s)],key=lambda r:-r['score'])
        rows.append(dict(view=name,ranking=ranking,requester5_rank=next(i+1 for i,r in enumerate(ranking) if r['unit']==5)))
    for name,s in zip(names[1:],scores):append(name,s)
    for k,seed in enumerate(seeds):append(f'family_{seed}',np.max(scores[1+4*k:1+4*(k+1)],axis=0))
    np.savez_compressed(a.out/'effects.npz',finite_logit=effects,source_ids=np.array(source_ids))
    write_new(a.out/'analysis.json',dict(protocol_sha256=sha(a.out/'protocol.json'),
        cache_sha256=sha(a.out/'effects.npz'),source_rounds=sources,dataset=manifest['dataset'],
        seconds=time.perf_counter()-started,training_calls=0,model_unlearning_calls=0,
        rows=rows,calibrated=False,independent_training_runs=1,
        dependencies={p.name:sha(p) for p in (Path(__file__).with_name('experiment_hrac_views.py'),
                                              Path(__file__).with_name('hrac.py'),research_source("collect_tension.py"))}))
    for r in rows:
        if r['view']=='oracle' or r['view'].startswith('family'):
            print(r['view'],'requester rank',r['requester5_rank'],'top',r['ranking'][0],flush=True)


if __name__=='__main__':
    main()
