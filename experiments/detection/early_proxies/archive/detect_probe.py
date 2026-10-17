# -*- coding: utf-8 -*-
"""
저장된 궤적(logs/traj/round_*.pt)으로 쌍 구조를 오프라인 분석(CPU).
각 라운드 r: Delta_k = flat(local_k) - flat(gm). 6x6 쌍에 대해
  (A) 전체 코사인
  (B) 평균 업데이트 방향 제거 후 잔차 코사인 (주 작업 소거 시도)
  (C) 분류기 헤드(fc)만
을 출력. 공격자=0, 요청자=5.
"""
import os, glob, torch, numpy as np

HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
import torchvision.models as models, torch.nn as nn
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__(); self.model=models.resnet18(weights=None)
        self.model.conv1=nn.Conv2d(3,64,3,1,1,bias=False); self.model.maxpool=nn.Identity()
        self.model.fc=nn.Linear(self.model.fc.in_features,10)
_keys=list(ResNet18().state_dict().keys())
FC=[k for k in _keys if k.endswith("fc.weight") or k.endswith("fc.bias")]
def flat(sd,keys=None):
    keys=keys or _keys
    return torch.cat([sd[k].flatten().float() for k in keys])
def cos(a,b): return (torch.dot(a,b)/(a.norm()*b.norm()+1e-12)).item()

files=sorted(glob.glob("logs/traj/round_*.pt"), key=lambda p:int(p.split("_")[-1].split(".")[0]))
print("rounds available:", [int(p.split('_')[-1].split('.')[0]) for p in files])
K=6
for f in files:
    r=int(f.split("_")[-1].split(".")[0])
    if r<2: continue
    d=torch.load(f, map_location="cpu", weights_only=False)
    gm=d["gm"]; cms=d["cms"]
    gflat=flat(gm); gfc=flat(gm,FC)
    D=[flat(c)-gflat for c in cms]          # 전체 업데이트
    Dfc=[flat(c,FC)-gfc for c in cms]
    mean=torch.stack(D,0).mean(0)
    Dres=[x-mean for x in D]                 # 평균(주 작업) 제거 잔차
    def mat(vs):
        return [[cos(vs[u],vs[k]) if u!=k else 1.0 for k in range(K)] for u in range(K)]
    A=mat(D); B=mat(Dres); C=mat(Dfc)
    def show(name,M):
        print(f"  [{name}] (행=클라, 0=atk 5=req)")
        print("        "+" ".join(f"c{k}" for k in range(K)))
        for u in range(K):
            print(f"    c{u}: "+" ".join(f"{M[u][k]:+.2f}" for k in range(K)))
    print(f"\n===== round {r} =====")
    show("A 전체코사인", A)
    show("B 평균제거잔차", B)
    show("C fc헤드", C)
    # 요약: (0,5) vs 정상쌍 분포
    def offdiag(M, excl=()):
        vals=[]
        for u in range(K):
            for k in range(u+1,K):
                if (u,k) in excl or (k,u) in excl: continue
                vals.append(M[u][k])
        return vals
    for nm,M in [("A",A),("B",B),("C",C)]:
        norm_pairs=offdiag(M, excl=[(0,5)])
        print(f"  {nm}: cos(0,5)={M[0][5]:+.3f}  정상/기타쌍 평균={np.mean(norm_pairs):+.3f} 최소={np.min(norm_pairs):+.3f} 최대={np.max(norm_pairs):+.3f}")
