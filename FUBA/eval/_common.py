# -*- coding: utf-8 -*-
"""평가 스크립트 공통 로더: FUBA 궤적에서 기여 U_k, 헤드 V, ACC/ASR 평가를 제공."""
import os, sys, pickle
import numpy as np, torch
from torchvision import transforms
FUBA=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FUBA)
from model import Net, MNISTAutoencoder
from utils.comm_utils import attack as fuba_attack, test as fuba_test
from dataset import MNIST

class Trajectory:
    """checkpoints/{global,local,backdoor}_net_model_{name}_round_*.pkl 를 읽어 평가 재료 제공."""
    def __init__(self, name, rounds=8, warmup=3, K=8, target=8, eps=1.0, thr=0.04, device=None):
        self.name=name; self.rounds=rounds; self.warmup=warmup; self.K=K; self.target=target
        self.eps=eps; self.thr=thr
        self.dev=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt=os.path.join(FUBA,"checkpoints")
        self.active=list(range(warmup+1, rounds+1))
        self.keys=list(Net().state_dict().keys()); self.sl={}; i=0
        for k in self.keys:
            n=Net().state_dict()[k].numel(); self.sl[k]=(i,i+n); i+=n
        self.D=i
        # 평가 데이터
        ts=MNIST(root=os.path.join(FUBA,'data'),train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))
        self.Xte=torch.stack([ts[j][0] for j in range(len(ts))]).to(self.dev)
        self.Yte=torch.tensor([ts[j][1] for j in range(len(ts))]).to(self.dev)
        self.bd=MNISTAutoencoder().to(self.dev)
        self.bd.load_state_dict(pickle.load(open(f"{self.ckpt}/backdoor_net_model_{name}_round_{rounds}.pkl","rb"))); self.bd.eval()
        self._build()
    def _flat(self,sd): return torch.cat([sd[k].float().flatten() for k in self.keys])
    def _g(self,r): return self._flat(pickle.load(open(f"{self.ckpt}/global_net_model_{self.name}_round_{r}.pkl","rb")))
    def _l(self,r): return [self._flat(sd) for sd in pickle.load(open(f"{self.ckpt}/local_net_model_{self.name}_round_{r}.pkl","rb"))]
    def _loader(self):
        X,Y=self.Xte,self.Yte
        class TL:
            def __iter__(s):
                for i in range(0,len(X),500): yield X[i:i+500].clone(), Y[i:i+500].clone()
        return TL()
    def _build(self):
        self.theta_T=self._g(self.rounds)
        self.U={k:torch.zeros(self.D) for k in range(self.K)}
        self.HUrow={}   # 타깃행 (85,)  per active round per client
        self.HUvec={}   # fc3 전체 (850,) per active round per client
        for r in range(2,self.rounds+1):
            g=self._g(r-1); L=self._l(r)
            for k in range(self.K): self.U[k]+=(L[k]-g)/self.K
            if r in self.active:
                self.HUrow[r]=[self._fc3row(L[k]-g) for k in range(self.K)]
                self.HUvec[r]=[self._fc3vec(L[k]-g) for k in range(self.K)]
    def _fc3row(self,v):
        t=self.target
        W=v[slice(*self.sl["fc3.weight"])].view(10,84)[t]; b=v[slice(*self.sl["fc3.bias"])].view(10)[t:t+1]
        return torch.cat([W,b])
    def _fc3vec(self,v): return torch.cat([v[slice(*self.sl["fc3.weight"])], v[slice(*self.sl["fc3.bias"])]])
    def Vbasis(self, atk, kV=8):
        def embed(row):
            e=torch.zeros(self.D); e[slice(*self.sl["fc3.weight"])]=row[:840]; e[slice(*self.sl["fc3.bias"])]=row[840:850]; return e
        Av=[self.HUvec[r][k] for r in self.active for k in atk]
        M=torch.stack(Av,0); M=M-M.mean(0,keepdim=True)
        Vt=torch.linalg.svd(M,full_matrices=False)[2][:kV]
        B=torch.stack([embed(Vt[i]) for i in range(Vt.shape[0])]); return torch.linalg.qr(B.T)[0].T
    def eval(self, flatv):
        m=Net().to(self.dev)
        sd=m.state_dict(); j=0
        for k in self.keys:
            n=sd[k].numel(); sd[k]=flatv[j:j+n].view_as(sd[k]).to(sd[k].dtype); j+=n
        m.load_state_dict(sd); m.eval(); ld=self._loader()
        return round(fuba_test(m,ld,self.dev),2), round(fuba_attack(m,ld,self.bd,self.eps,self.thr,self.target,self.dev),2)
