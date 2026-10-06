# -*- coding: utf-8 -*-
"""
weighted_sweep.py — 통째 제거 ↔ V-제거 사이를 '확신도 가중치'로 연속 보간, β 스윕 실측.

정화:  theta = theta_T - U_u - sum_{k!=u} [ w_k * U_k + (1-w_k) * P_V U_k ]
확신도: w_k = sigmoid(beta * a_k),  a_k = 클라 k 타깃행이 공격자 군집 평균과 이루는 정렬(코사인)
  beta=0        -> w=0.5 (의미 적음, 생략)
  allV          -> w_k=0  (우리 기존 방법, V 성분만)
  hybrid(beta)  -> 확신 높은(정렬 큰) 클라일수록 통째에 가깝게
  hard          -> w_k = 1 if a_k>0 else 0 (양의 정렬=통째, 음=V만)
  atk_full      -> 탐지된 공격자만 통째, 나머지 0 (참고 끝점)
탐지 결과(req, 공격자, 타깃)는 logs/pipeline_{name}.json 에서 읽는다.
"""
import os, sys, json, pickle, argparse
import numpy as np, torch
from torchvision import transforms
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0,HERE)
from model import Net, MNISTAutoencoder
from utils.comm_utils import attack as fuba_attack, test as fuba_test
from dataset import MNIST

P=argparse.ArgumentParser()
P.add_argument("--name",default="det1"); P.add_argument("--rounds",type=int,default=8); P.add_argument("--warmup",type=int,default=3)
P.add_argument("--K",type=int,default=8); P.add_argument("--eps",type=float,default=1.0); P.add_argument("--thr",type=float,default=0.04)
P.add_argument("--kV",type=int,default=8); P.add_argument("--betas",type=float,nargs="+",default=[1,2,4,8,16])
a=P.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
det=json.load(open(f"logs/pipeline_{a.name}.json"))["detection"]
req=det["req_hat"] if det["req_hat"]>=0 else 4; atk=sorted(det["A_hat"]) if det["A_hat"] else [0,1,2,3]; t=det["t_hat"]
ACTIVE=list(range(a.warmup+1,a.rounds+1))

testset=MNIST(root='./data',train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))
Xte=torch.stack([testset[i][0] for i in range(len(testset))]).to(dev); Yte=torch.tensor([testset[i][1] for i in range(len(testset))]).to(dev)
class TL:
    def __iter__(s):
        for i in range(0,len(Xte),500): yield Xte[i:i+500].clone(), Yte[i:i+500].clone()
loader=TL(); bd=MNISTAutoencoder().to(dev)
bd.load_state_dict(pickle.load(open(f"checkpoints/backdoor_net_model_{a.name}_round_{a.rounds}.pkl","rb"))); bd.eval()
keys=list(Net().state_dict().keys()); sl={}; i=0
for k in keys:
    n=Net().state_dict()[k].numel(); sl[k]=(i,i+n); i+=n
D=i
def flat(sd): return torch.cat([sd[k].float().flatten() for k in keys])
def set_flat(m,v):
    sd=m.state_dict(); j=0
    for k in keys:
        n=sd[k].numel(); sd[k]=v[j:j+n].view_as(sd[k]).to(sd[k].dtype); j+=n
    m.load_state_dict(sd)
def ev(v):
    m=Net().to(dev); set_flat(m,v.to(dev)); m.eval()
    return round(fuba_test(m,loader,dev),2), round(fuba_attack(m,loader,bd,a.eps,a.thr,t,dev),2)
def load_g(r): return flat(pickle.load(open(f"checkpoints/global_net_model_{a.name}_round_{r}.pkl","rb")))
def load_l(r): return [flat(sd) for sd in pickle.load(open(f"checkpoints/local_net_model_{a.name}_round_{r}.pkl","rb"))]

theta_T=load_g(a.rounds); U={k:torch.zeros(D) for k in range(a.K)}
HU={}  # 타깃행(85,) per round per client
def fc3row_t(v):
    W=v[slice(*sl["fc3.weight"])].view(10,84)[t]; b=v[slice(*sl["fc3.bias"])].view(10)[t:t+1]; return torch.cat([W,b])
def fc3vec(v): return torch.cat([v[slice(*sl["fc3.weight"])], v[slice(*sl["fc3.bias"])]])
HUV={}
for r in range(2,a.rounds+1):
    g=load_g(r-1); L=load_l(r)
    for k in range(a.K): U[k]+=(L[k]-g)/a.K
    if r in ACTIVE:
        HU[r]=[fc3row_t(L[k]-g) for k in range(a.K)]; HUV[r]=[fc3vec(L[k]-g) for k in range(a.K)]
# a_k = 타깃행, 공격자 군집 평균 방향과의 정렬
def unit(x): n=x.norm(); return x/n if n>1e-12 else x
cmean=lambda r: unit(torch.stack([HU[r][k] for k in atk]).mean(0))
a_k=np.array([float(np.mean([torch.dot(unit(HU[r][k]),cmean(r)).item() for r in ACTIVE])) for k in range(a.K)])
# V
def embed(row):
    e=torch.zeros(D); e[slice(*sl["fc3.weight"])]=row[:840]; e[slice(*sl["fc3.bias"])]=row[840:850]; return e
Av=[HUV[r][k] for r in ACTIVE for k in atk]; M=torch.stack(Av,0); M=M-M.mean(0,keepdim=True)
Vt=torch.linalg.svd(M,full_matrices=False)[2][:a.kV]; B=torch.stack([embed(Vt[i]) for i in range(Vt.shape[0])]); B=torch.linalg.qr(B.T)[0].T
def projV(v): return B.T@(B@v)

def purify(w):  # w: dict k->[0,1]
    th=theta_T-U[req]
    for k in range(a.K):
        if k==req: continue
        th=th-(w[k]*U[k]+(1-w[k])*projV(U[k]))
    return th

print(f"==== {a.name} 가중 제거 β 스윕 (req={req}, atk={atk}, target={t}) ====")
print(f"  a_k(타깃행 정렬): "+" ".join(f"c{k}:{a_k[k]:+.2f}" for k in range(a.K)))
print(f"  {'설정':22s} {'ACC':>7s} {'ASR':>7s}   w(공격자/요청자/정상예)")
res={}
# allV
w0={k:0.0 for k in range(a.K)}; acc,asr=ev(purify(w0)); res["allV(w=0)"]={ "acc":acc,"asr":asr}
print(f"  {'allV (우리 기존)':22s} {acc:7.2f} {asr:7.2f}")
# hybrid betas
for b in a.betas:
    w={k:float(1/(1+np.exp(-b*a_k[k]))) for k in range(a.K)}
    acc,asr=ev(purify(w)); res[f"hybrid b={b}"]={"acc":acc,"asr":asr,"w":{int(k):round(w[k],2) for k in range(a.K)}}
    nor=[k for k in range(a.K) if k not in atk and k!=req]
    print(f"  hybrid β={b:<6g}        {acc:7.2f} {asr:7.2f}   {w[atk[0]]:.2f}/{w[req]:.2f}/{w[nor[0]]:.2f}")
# hard
wh={k:(1.0 if a_k[k]>0 else 0.0) for k in range(a.K)}; acc,asr=ev(purify(wh)); res["hard(sign)"]={"acc":acc,"asr":asr}
print(f"  {'hard (정렬>0 통째)':22s} {acc:7.2f} {asr:7.2f}")
# atk_full (탐지 공격자만 통째)
wa={k:(1.0 if k in atk else 0.0) for k in range(a.K)}; acc,asr=ev(purify(wa)); res["atk_full(detected)"]={"acc":acc,"asr":asr}
print(f"  {'atk_full (탐지공격자 통째)':22s} {acc:7.2f} {asr:7.2f}")
json.dump({"name":a.name,"req":req,"atk":atk,"target":t,"a_k":a_k.tolist(),"res":res},
          open(f"logs/weighted_{a.name}.json","w"),indent=2,ensure_ascii=False)
print("saved",f"logs/weighted_{a.name}.json")
