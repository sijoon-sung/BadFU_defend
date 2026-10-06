# -*- coding: utf-8 -*-
"""
BadFU 원문 구조에 맞춰 고친 탐지기 vs 원 방법 대조 (오프라인, CPU, logs/traj_keep 사용).

원문 확인 결과:
  요청자(5)=clean0+cv(트리거->정답라벨), 공격자(0)=clean0+bd(트리거->타깃0).
  둘은 같은 clean0 를 공유 -> 업데이트가 반평행이 아니라 near-duplicate(양의 유사도).
  전체 코사인은 사전학습->CIFAR 공통방향으로 모든 쌍이 +1 (포화) -> 쓸모 없음.

원 방법(①~④): c=cos(전체 Δ), z=지속성, S(u)=min_k z (가장 '반평행'), S<tau 플래그.
고친 방법:    h=cos(fc헤드 Δ) (전체는 포화), 지속성, S+(u)=max_k (가장 '닮음'),
             서로를 top 으로 가리키는 상호쌍이 분포에서 상향 이상치면 그 쌍을 공모로.
"""
import os, glob, torch, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
import torchvision.models as models, torch.nn as nn
class R(nn.Module):
    def __init__(s):
        super().__init__(); s.model=models.resnet18(weights=None)
        s.model.conv1=nn.Conv2d(3,64,3,1,1,bias=False); s.model.maxpool=nn.Identity()
        s.model.fc=nn.Linear(s.model.fc.in_features,10)
_keys=list(R().state_dict().keys()); FC=[k for k in _keys if k.endswith("fc.weight") or k.endswith("fc.bias")]
def flat(sd,keys=None):
    keys=keys or _keys; return torch.cat([sd[k].flatten().float() for k in keys])
def cos(a,b): return (torch.dot(a,b)/(a.norm()*b.norm()+1e-12)).item()

RAMP=3; ATK,REQ=0,5
files=sorted(glob.glob("logs/traj_keep/round_*.pt"), key=lambda p:int(p.split("_")[-1].split(".")[0]))
rounds=[int(p.split('_')[-1].split('.')[0]) for p in files]
print("rounds:", rounds)
K=6
full={}; head={}   # pair -> list over rounds
for f in files:
    r=int(f.split("_")[-1].split(".")[0])
    if r<RAMP: continue
    d=torch.load(f, map_location="cpu", weights_only=False); gm=d["gm"]; cms=d["cms"]
    gF=flat(gm); gH=flat(gm,FC)
    DF=[flat(c)-gF for c in cms]; DH=[flat(c,FC)-gH for c in cms]
    for u in range(K):
        for k in range(u+1,K):
            full.setdefault((u,k),[]).append(cos(DF[u],DF[k]))
            head.setdefault((u,k),[]).append(cos(DH[u],DH[k]))

def stats(h):
    m={p:float(np.mean(v)) for p,v in h.items()}
    return m
mf=stats(full); mh=stats(head)

print("\n===== 원 방법 (전체 업데이트, 가장 반평행 min) =====")
# 각 클라 S(u)=min_k mean_cos(전체). 가장 음수면 플래그.
Sfull={}
for u in range(K):
    vals={k: mf[(min(u,k),max(u,k))] for k in range(K) if k!=u}
    Sfull[u]=min(vals.values())
for u in range(K):
    role="ATK" if u==ATK else "REQ" if u==REQ else "nor"
    print(f"  c{u}[{role}] S(min cos 전체)={Sfull[u]:+.3f}")
print(f"  -> 전체코사인은 모두 ~{np.mean(list(mf.values())):+.2f} 로 포화. 반평행 신호 없음. 요청자 특정 불가.")

print("\n===== 고친 방법 (fc 헤드, 가장 닮은 쌍 max) =====")
# 쌍 랭킹
ranked=sorted(mh.items(), key=lambda x:-x[1])
print("  쌍 평균 head-cos 랭킹:")
for (p,val) in ranked:
    tag=" <== (ATK,REQ)" if set(p)=={ATK,REQ} else ""
    print(f"    {p}: {val:+.3f}{tag}")
vals=np.array([v for v in mh.values()])
top_p,top_v=ranked[0]
rest=np.array([v for pp,v in mh.items() if pp!=top_p])
z=(top_v-rest.mean())/(rest.std()+1e-12)
print(f"  top 쌍 {top_p} head-cos={top_v:+.3f}, 나머지 평균={rest.mean():+.3f} std={rest.std():.3f} -> z={z:+.1f}")
# 상호 top 확인
best={u:max((mh[(min(u,k),max(u,k))],k) for k in range(K) if k!=u) for u in range(K)}
print("  각 클라의 최유사 상대:")
for u in range(K):
    role="ATK" if u==ATK else "REQ" if u==REQ else "nor"
    bv,bk=best[u]; print(f"    c{u}[{role}] -> c{bk} ({bv:+.3f})")
mutual = (best[ATK][1]==REQ and best[REQ][1]==ATK)
print(f"  상호 최유사쌍인가(ATK<->REQ): {mutual}")
print(f"  판정: top 쌍이 (ATK,REQ) 이고 z={z:+.1f} 상향 이상치 -> 공모쌍으로 플래그. "
      f"{'성공' if set(top_p)=={ATK,REQ} else '실패'}")

print("\n[주의] 이 신호는 '두 클라가 같은 데이터 분포를 공유'를 잡는 것(FoolsGold 계열, 양의 유사도). "
      "BadFU 는 요청자가 공격자 clean 데이터의 복제라 분리되지만, 분포가 겹치는 정상 쌍이 있으면 오탐 가능.")
