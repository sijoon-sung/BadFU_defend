# -*- coding: utf-8 -*-
"""
purify_compare.py — 실제 FUBA 실행(mpiexec, --save) 궤적으로 정화 변형 비교.

역할: 공격자 0~3, 요청자(방어자) 4, 정상 5~7, 타깃 8. 전원참여(1/8 가중) FedAvg.
1차 근사: theta_T = theta_0 + sum_r sum_k U_k^r,  U_k^r = (1/K)(local_r[k] - global_{r-1}).

변형:
  A none         : theta_T 그대로 (잠복)
  B req          : - U_req                               (요청자만 제거 = BadFU/FUBA 활성화)
  C req+atkFull  : - U_req - sum_a U_a                   (공격자 통째 제거)
  D req+atkV     : - U_req - sum_a P_V U_a               (공격자의 V 성분만)
  E req+allV     : - U_req - sum_{k!=req} P_V U_k        (판정 없이 전원 V 성분; 정상은 거의 0이어야)
  F req+randV    : - U_req - sum_a P_R U_a               (같은 차원 임의 방향; 대조)
V = 활성 라운드 공격자 업데이트들의 상위-k 주성분 (타깃 행 fc3 또는 전체 파라미터).
측정: clean ACC, ASR(FUBA attack(), round-8 백도어 생성기), 정상 클라 기여 손실 비율.
"""
import os, sys, pickle, json, argparse
import numpy as np, torch
from torchvision import transforms
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0,HERE)
from model import Net, MNISTAutoencoder
from utils.comm_utils import attack as fuba_attack, test as fuba_test
from dataset import MNIST

ap=argparse.ArgumentParser()
ap.add_argument("--name",default="det1"); ap.add_argument("--rounds",type=int,default=8)
ap.add_argument("--warmup",type=int,default=3); ap.add_argument("--K",type=int,default=8)
ap.add_argument("--atk",type=int,nargs="+",default=[0,1,2,3]); ap.add_argument("--req",type=int,default=4)
ap.add_argument("--target",type=int,default=8); ap.add_argument("--eps",type=float,default=1.0)
ap.add_argument("--thr",type=float,default=0.04); ap.add_argument("--ks",type=int,nargs="+",default=[1,2,4,8])
ap.add_argument("--out",default="logs/purify_compare.json")
a=ap.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
os.makedirs("logs",exist_ok=True)
NOR=[k for k in range(a.K) if k not in a.atk and k!=a.req]

# ---- 데이터 / 평가 ----
testset=MNIST(root='./data',train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))
Xte=torch.stack([testset[i][0] for i in range(len(testset))]).to(dev)
Yte=torch.tensor([testset[i][1] for i in range(len(testset))]).to(dev)
class TL:
    def __iter__(s):
        for i in range(0,len(Xte),500): yield Xte[i:i+500].clone(), Yte[i:i+500].clone()
loader=TL()
bd=MNISTAutoencoder().to(dev)
bd.load_state_dict(pickle.load(open(f"checkpoints/backdoor_net_model_{a.name}_round_{a.rounds}.pkl","rb"))); bd.eval()
def evaluate(flat):
    m=Net().to(dev); set_flat(m,flat); m.eval()
    return fuba_test(m,loader,dev), fuba_attack(m,loader,bd,a.eps,a.thr,a.target,dev)

# ---- 궤적 -> 기여 ----
keys=list(Net().state_dict().keys())
def flat(sd): return torch.cat([sd[k].float().flatten() for k in keys])
def set_flat(m,v):
    sd=m.state_dict(); i=0
    for k in keys:
        n=sd[k].numel(); sd[k]=v[i:i+n].view_as(sd[k]).to(sd[k].dtype); i+=n
    m.load_state_dict(sd)
sl={}; i=0
for k in keys:
    n=Net().state_dict()[k].numel(); sl[k]=(i,i+n); i+=n
D=i
def load_g(r): return flat(pickle.load(open(f"checkpoints/global_net_model_{a.name}_round_{r}.pkl","rb")))
def load_l(r): return [flat(sd) for sd in pickle.load(open(f"checkpoints/local_net_model_{a.name}_round_{r}.pkl","rb"))]
theta_T=load_g(a.rounds)
U={k:torch.zeros(D) for k in range(a.K)}          # 전체 라운드 기여
Uact={k:torch.zeros(D) for k in range(a.K)}       # 활성 라운드 기여(백도어 구간)
per_round_atk=[]                                  # V 추정용
for r in range(2,a.rounds+1):                     # round1 은 theta_0 없어 생략
    g=load_g(r-1); L=load_l(r)
    for k in range(a.K):
        u=(L[k]-g)/a.K; U[k]+=u
        if r>a.warmup:
            Uact[k]+=u
            if k in a.atk: per_round_atk.append(u.clone())

# ---- V: 공격자 업데이트 주성분 (전체 파라미터 / fc3 만) ----
def topk_basis(vecs,k):
    M=torch.stack(vecs,0); M=M-M.mean(0,keepdim=True) if len(vecs)>1 else M
    _,_,Vt=torch.linalg.svd(M,full_matrices=False); return Vt[:k]        # (k,D)
def proj(v,B): return B.T@(B@v)
A_vecs=per_round_atk
def restrict_fc3(v):
    m=torch.zeros_like(v); s,e=sl["fc3.weight"]; m[s:e]=v[s:e]; s,e=sl["fc3.bias"]; m[s:e]=v[s:e]; return m

results={}
def rec(tag,flatv,removed_nor=None):
    acc,asr=evaluate(flatv.to(dev))
    row={"acc":acc,"asr":asr}
    if removed_nor is not None: row["normal_removed_frac"]=removed_nor
    results[tag]=row; print(f"  {tag:34s} ACC={acc:6.2f} ASR={asr:6.2f}"+(f" 정상기여제거비율={removed_nor:.3f}" if removed_nor is not None else ""))

print("==== 정화 변형 비교 (실제 FUBA 궤적) ====")
rec("A none(dormant)", theta_T)
rec("B req only (=activation)", theta_T-U[a.req])
rec("C req + attackers FULL", theta_T-U[a.req]-sum(U[x] for x in a.atk))
for space,restrict in [("full",lambda v:v),("fc3",restrict_fc3)]:
    vecs=[restrict(v) for v in A_vecs]
    for k in a.ks:
        if k>len(vecs): continue
        B=topk_basis(vecs,k)
        nor_tot=sum(U[x].norm() for x in NOR).item()
        nor_rm=sum(proj(U[x],B).norm() for x in NOR).item()
        rec(f"D req + atk V[{space},k={k}]", theta_T-U[a.req]-sum(proj(U[x],B) for x in a.atk))
        rec(f"E req + ALL V[{space},k={k}]", theta_T-U[a.req]-sum(proj(U[x],B) for x in range(a.K) if x!=a.req), nor_rm/max(nor_tot,1e-12))
        R=torch.linalg.qr(torch.randn(D,k))[0].T                      # 임의 k차원 (대조)
        rec(f"F req + atk RAND[k={k}]", theta_T-U[a.req]-sum(proj(U[x],R) for x in a.atk))

json.dump({"args":vars(a),"results":results},open(a.out,"w"),indent=2,ensure_ascii=False)
print("saved",a.out)
