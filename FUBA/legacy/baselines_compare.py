# -*- coding: utf-8 -*-
"""
baselines_compare.py — 같은 FUBA 궤적 위에서 기존 방어들과 우리 정화를 한 표로 비교.

재구성 틀(1차 근사, 공통): base=global_1 에서 라운드 2..R 을 '요청자 제외' 재집계.
  theta = global_1 + sum_r AGG_{k != req}( local_r[k] - global_{r-1} )
집계 규칙 AGG:
  mean      : FedAvg (= 요청자만 제거한 활성화 모델)
  median    : 좌표별 중앙값 (Byzantine-robust; BadFU 가 평가한 defense)
  trimmed   : 좌표별 절단평균 beta=0.2 (Trimmed-mean)
  topk      : FUBA potential_defence.federated_averaging_topk 재현
              (현재 재구성 모델과 코사인 유사도 상위 keep 개만 평균)
  meanV     : mean + 백도어 방향 V 성분 제거 (우리 제안)
추가: attacker_full(요청자+공격자 통째 제거). 모두 ACC/ASR 로 평가.
탐지 결과(요청자/공격자/타깃)는 logs/pipeline_{name}.json 에서 읽는다(먼저 pipeline 실행 필요).
"""
import os, sys, json, pickle, argparse
import numpy as np, torch
from torchvision import transforms
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0,HERE)
from model import Net, MNISTAutoencoder
from utils.comm_utils import attack as fuba_attack, test as fuba_test
from dataset import MNIST

P=argparse.ArgumentParser()
P.add_argument("--name",default="det1"); P.add_argument("--rounds",type=int,default=8)
P.add_argument("--warmup",type=int,default=3); P.add_argument("--K",type=int,default=8)
P.add_argument("--target",type=int,default=8); P.add_argument("--eps",type=float,default=1.0); P.add_argument("--thr",type=float,default=0.04)
P.add_argument("--kV",type=int,default=8); P.add_argument("--keep",type=int,default=4); P.add_argument("--beta",type=float,default=0.2)
a=P.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
# 탐지 결과 로드(없으면 정답 기본값)
det_path=f"logs/pipeline_{a.name}.json"
if os.path.exists(det_path):
    d=json.load(open(det_path))["detection"]; req=d["req_hat"]; atk=set(d["A_hat"]); t=d["t_hat"]
    if req<0: req=4
else:
    req=4; atk={0,1,2,3}; t=a.target
ACTIVE=list(range(a.warmup+1,a.rounds+1))

# 평가
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
    return round(fuba_test(m,loader,dev),2), round(fuba_attack(m,loader,bd,a.eps,a.thr,a.target,dev),2)
def load_g(r): return flat(pickle.load(open(f"checkpoints/global_net_model_{a.name}_round_{r}.pkl","rb")))
def load_l(r): return [flat(sd) for sd in pickle.load(open(f"checkpoints/local_net_model_{a.name}_round_{r}.pkl","rb"))]

# 라운드별 업데이트 + 전체 로컬 모델(원본 Top-k 선택용)
upd={r:{} for r in range(2,a.rounds+1)}; Lfull={r:{} for r in range(2,a.rounds+1)}
for r in range(2,a.rounds+1):
    g=load_g(r-1); L=load_l(r)
    for k in range(a.K): upd[r][k]=L[k]-g; Lfull[r][k]=L[k]
REF_FINAL=load_g(a.rounds)   # FUBA 원본 Top-k 의 unlearned_state_dict = 최종 글로벌

# V (탐지된 공격자들의 라운드별 fc3 업데이트 주성분)
def fc3_vec(v): return torch.cat([v[slice(*sl["fc3.weight"])], v[slice(*sl["fc3.bias"])]])
def embed(row):
    e=torch.zeros(D); e[slice(*sl["fc3.weight"])]=row[:840]; e[slice(*sl["fc3.bias"])]=row[840:850]; return e
Avec=[fc3_vec(upd[r][k]) for r in ACTIVE for k in (sorted(atk) if atk else [0,1,2,3])]
M=torch.stack(Avec,0); M=M-M.mean(0,keepdim=True)
Vt=torch.linalg.svd(M,full_matrices=False)[2][:a.kV]
B=torch.stack([embed(Vt[i]) for i in range(Vt.shape[0])]); B=torch.linalg.qr(B.T)[0].T
def projV(v): return B.T@(B@v)

# 집계 규칙
def agg(vecs,rule,ref=None):
    Msk=torch.stack(vecs,0)
    if rule=="mean": return Msk.mean(0)
    if rule=="median": return Msk.median(0).values
    if rule=="trimmed":
        s=torch.sort(Msk,0).values; c=int(a.beta*Msk.shape[0]); return s[c:Msk.shape[0]-c].mean(0) if Msk.shape[0]-2*c>0 else Msk.mean(0)
    raise ValueError(rule)

def reconstruct(remove,rule,removeV=False):
    theta=load_g(1).clone()
    for r in range(2,a.rounds+1):
        ks=[k for k in range(a.K) if k not in remove]
        if rule=="topk":
            # FUBA 원본: 각 클라 '전체 모델'을 최종 글로벌과 코사인 -> 상위 keep 의 '업데이트' 평균
            sims=sorted(ks, key=lambda k: -float(torch.dot(Lfull[r][k],REF_FINAL)/(Lfull[r][k].norm()*REF_FINAL.norm()+1e-12)))
            ks=sims[:a.keep]
            theta=theta+torch.stack([upd[r][k] for k in ks],0).mean(0); continue
        vecs=[upd[r][k] for k in ks]
        if removeV: vecs=[v-projV(v) for v in vecs]
        theta=theta+agg(vecs,rule)
    return theta

rows={}
rows["none(dormant)"]=ev(load_g(a.rounds))
rows["B req-only FedAvg (activation)"]=ev(reconstruct({req},"mean"))
rows["Median (robust-agg)"]=ev(reconstruct({req},"median"))
rows["Trimmed-mean b=0.2"]=ev(reconstruct({req},"trimmed"))
rows[f"Top-k cos (FUBA, keep={a.keep})"]=ev(reconstruct({req},"topk"))
rows["attacker-full removal"]=ev(reconstruct({req}|set(atk),"mean"))
rows["OURS: req + allV (FedAvg)"]=ev(reconstruct({req},"mean",removeV=True))

print(f"==== {a.name} 기존 방어 vs 제안 (요청자={req}, 공격자={sorted(atk)}, 타깃={t}) ====")
print(f"  {'방법':34s} {'ACC':>7s} {'ASR':>7s}")
for k,(acc,asr) in rows.items(): print(f"  {k:34s} {acc:7.2f} {asr:7.2f}")
json.dump({"name":a.name,"req":req,"atk":sorted(atk),"target":t,"rows":rows},
          open(f"logs/baselines_{a.name}.json","w"),indent=2,ensure_ascii=False)
print("\n[주의] Median/Trimmed 는 사후재구성(요청자제거 후 좌표별 집계)이라 비-IID 에서 발산 -> 공정 비교 아님.")
print("       공정하려면 FUBA 를 --median 등 플래그로 학습단계에서 실행. Top-k 는 FUBA 원본(전체모델 vs 최종글로벌)대로임.")
print("saved",f"logs/baselines_{a.name}.json")
