# -*- coding: utf-8 -*-
"""
train_benign_fl.py — 공격 없는 FL 실행(정상 대조군). 단일 프로세스(MPI 불필요, RAM 가벼움).
FUBA 와 같은 Net/MNIST/클라수/비-IID 로 백도어 없이 FedAvg, 라운드별 global/local 저장.
detection_eval --benign 과 pipeline 의 오경보 측정에 쓴다. 백도어 생성기는 무작위로 저장(평가 호환용).
사용: python eval/train_benign_fl.py --name benign1 --seed 101
"""
import os, sys, pickle, argparse, copy
import numpy as np, torch, torch.nn as nn, torch.optim as optim
FUBA=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,FUBA); os.chdir(FUBA)
from model import Net, MNISTAutoencoder
from dataset import MNIST
from torchvision import transforms

ap=argparse.ArgumentParser()
ap.add_argument("--name",default="benign1"); ap.add_argument("--seed",type=int,default=101)
ap.add_argument("--K",type=int,default=8); ap.add_argument("--rounds",type=int,default=8)
ap.add_argument("--alpha",type=float,default=0.5); ap.add_argument("--local_batches",type=int,default=21)
a=ap.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed); np.random.seed(a.seed); import random; random.seed(a.seed)
os.makedirs("checkpoints",exist_ok=True)
tf=transforms.Compose([transforms.ToTensor()])
tr=MNIST(root='./data',train=True,download=True,transform=tf)
X=torch.stack([tr[i][0] for i in range(len(tr))]).to(dev); Y=torch.tensor([tr[i][1] for i in range(len(tr))]).to(dev)
# 비-IID Dirichlet 분할
g=torch.Generator().manual_seed(a.seed); parts=[[] for _ in range(a.K)]
for c in range(10):
    idx=torch.where(Y==c)[0]; idx=idx[torch.randperm(len(idx),generator=g,device='cpu').to(dev)]
    cut=(np.cumsum(np.random.dirichlet([a.alpha]*a.K))*len(idx)).astype(int)[:-1]
    for k,ch in enumerate(np.split(idx.cpu().numpy(),cut)): parts[k].append(torch.tensor(ch,device=dev))
cidx=[torch.cat(p) for p in parts]
keys=list(Net().state_dict().keys())
def save(name,r,gsd,locals_):
    pickle.dump({k:v.detach().cpu() for k,v in gsd.items()},open(f"checkpoints/global_net_model_{name}_round_{r}.pkl","wb"))
    pickle.dump([{k:v.detach().cpu() for k,v in sd.items()} for sd in locals_],open(f"checkpoints/local_net_model_{name}_round_{r}.pkl","wb"))
gm=Net().to(dev); gsd=gm.state_dict()
for r in range(1,a.rounds+1):
    locals_=[]
    for k in range(a.K):
        m=Net().to(dev); m.load_state_dict(copy.deepcopy(gsd)); m.train()
        opt=optim.Adam(m.parameters()); crit=nn.CrossEntropyLoss(); Xk,Yk=X[cidx[k]],Y[cidx[k]]
        perm=torch.randperm(len(Xk),device=dev)
        for bi,i in enumerate(range(0,len(Xk),64)):
            b=perm[i:i+64]; opt.zero_grad(); crit(m(Xk[b]),Yk[b]).backward(); opt.step()
            if bi>a.local_batches: break
        locals_.append({kk:vv.detach() for kk,vv in m.state_dict().items()})
    new={kk:torch.stack([locals_[k][kk].float() for k in range(a.K)],0).mean(0) for kk in keys}
    gm.load_state_dict(new); gsd=gm.state_dict(); save(a.name,r,gsd,locals_)
    print(f"[benign {a.name}] round {r} done",flush=True)
# 평가 호환용 더미 백도어 생성기
pickle.dump(MNISTAutoencoder().state_dict(),open(f"checkpoints/backdoor_net_model_{a.name}_round_{a.rounds}.pkl","wb"))
print("saved benign trajectory:",a.name)
