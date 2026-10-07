# -*- coding: utf-8 -*-
"""
FUBA 에서 κ(쌍의 헤드 업데이트 차이가 단일 클래스에 쏠리는 정도)가 작동하는지 정밀 검사.
FUBA 포트(exp_fuba_real_detect.py)와 동일한 데이터/모델/주입을 쓰되, 매 라운드 각 클라의
fc3(헤드) 업데이트를 저장하고 끝에서 κ 통계를 낸다.

구조: K=5, 0=Adv-attacker(트리거->타깃8), 1=Adv-defender(트리거->정답), 2~4=정상.
BadFU 와 다른 점: 공격자/방어자가 clean 데이터를 공유하지 않음(비-IID Dirichlet). 그래서
비-IID 지배클래스 차이가 κ 신호를 흐릴 수 있음 -> 타깃 클래스 '순위'와 정상쌍 귀무분포로 검증.
"""
import os, copy, json, sys
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms

FUBA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "FUBA")
sys.path.insert(0, FUBA)
from model import Net, MNISTAutoencoder
from utils.comm_utils import attack as fuba_attack, test as fuba_test

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.makedirs(os.path.join(HERE,"logs"), exist_ok=True)
K, ROUNDS, WARMUP, RAMP = 5, 20, 5, 2
TGT, EPS, THR, BATCH_CAP, ATK_EPOCHS, BDGEN = 8, 1.0, 0.04, 21, 10, 15
ATTACKER, DEFENDER = 0, 1
alpha = 0.5

tf = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(os.path.join(HERE,"FUBA","data"), train=True, download=True, transform=tf)
test = datasets.MNIST(os.path.join(HERE,"FUBA","data"), train=False, download=True, transform=tf)
Xtr = torch.stack([train[i][0] for i in range(len(train))]).to(dev)
Ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(dev)
Xte = torch.stack([test[i][0] for i in range(len(test))]).to(dev)
Yte = torch.tensor([test[i][1] for i in range(len(test))]).to(dev)

def dirichlet_split(Y, K, alpha, g):
    parts=[[] for _ in range(K)]
    for c in range(10):
        idx=torch.where(Y==c)[0]; idx=idx[torch.randperm(len(idx),generator=g,device="cpu").to(Y.device)]
        pr=np.random.dirichlet([alpha]*K); cut=(np.cumsum(pr)*len(idx)).astype(int)[:-1]
        for k,ch in enumerate(np.split(idx.cpu().numpy(),cut)): parts[k].append(torch.tensor(ch,device=Y.device))
    return [torch.cat(pp) for pp in parts]
g=torch.Generator().manual_seed(0); cidx=dirichlet_split(Ytr,K,alpha,g)
# 각 클라 지배 클래스(참고)
dom=[int(torch.bincount(Ytr[cidx[k]],minlength=10).argmax()) for k in range(K)]

class TL:
    def __init__(s,X,Y,bs=256): s.X,s.Y,s.bs=X,Y,bs
    def __iter__(s):
        for i in range(0,len(s.X),s.bs): yield s.X[i:i+s.bs].clone(), s.Y[i:i+s.bs].clone()
testloader=TL(Xte,Yte)

def add_trigger_free(): pass
_names=list(Net().state_dict().keys())
def get_flat(sd): return torch.cat([sd[k].flatten().float() for k in _names])

def train_bdgen(net, bd, X):
    net.eval(); opt=optim.Adam(bd.parameters()); crit=nn.CrossEntropyLoss()
    for _ in range(BDGEN):
        bd.train(); perm=torch.randperm(len(X),device=dev)
        for bi,i in enumerate(range(0,len(X),64)):
            inp=X[perm[i:i+64]]; noise=torch.clamp(bd(inp)*EPS,-THR,THR)
            pert=torch.clamp(inp+noise,-1,1); lab=torch.full((len(inp),),TGT,device=dev)
            opt.zero_grad(); crit(net(pert),lab).backward(); opt.step()
            if bi>BATCH_CAP: break
    bd.eval()

def local(global_sd, X, Y, role, do_bd, bd):
    net=Net().to(dev); net.load_state_dict(copy.deepcopy(global_sd)); net.train()
    opt=optim.Adam(net.parameters()); crit=nn.CrossEntropyLoss()
    for _ in range(ATK_EPOCHS if (role in (ATTACKER,DEFENDER) and do_bd) else 1):
        perm=torch.randperm(len(X),device=dev)
        for bi,i in enumerate(range(0,len(X),64)):
            b=perm[i:i+64]; inp,lab=X[b],Y[b].clone()
            if do_bd and role==ATTACKER:
                noise=torch.clamp(bd(inp).detach()*EPS,-THR,THR); m=torch.rand(len(inp),device=dev)<0.4
                inp=inp.clone(); inp[m]=torch.clamp(inp[m]+noise[m],-1,1)
                lab=torch.where(m,torch.tensor(TGT,device=dev),lab)
            elif do_bd and role==DEFENDER:
                noise=torch.clamp(bd(inp).detach()*EPS,-THR,THR); inp=torch.clamp(inp+noise,-1,1)  # 정답 라벨 유지
            opt.zero_grad(); crit(net(inp),lab).backward(); opt.step()
            if bi>BATCH_CAP: break
    return net.state_dict()

@torch.no_grad()
def set_sd(m,sd): m.load_state_dict(sd)

gmodel=Net().to(dev); gsd=gmodel.state_dict(); bd=MNISTAutoencoder().to(dev)
# 저장: 라운드별 각 클라의 fc3 업데이트 (가중치행 + bias) = (C, 85)
head_updates=[]   # [round][client] -> np (10,85)
active=[]; asr_track=[]
for r in range(ROUNDS):
    do_bd = r>=WARMUP
    if do_bd:
        tmp=Net().to(dev); tmp.load_state_dict(copy.deepcopy(gsd)); train_bdgen(tmp,bd,Xtr[cidx[ATTACKER]])
    gW=gsd["fc3.weight"].float().clone(); gb=gsd["fc3.bias"].float().clone()
    sds=[local(gsd,Xtr[cidx[k]],Ytr[cidx[k]],k,do_bd,bd) for k in range(K)]
    hu=[]
    for k in range(K):
        dW=(sds[k]["fc3.weight"].float()-gW); db=(sds[k]["fc3.bias"].float()-gb).view(-1,1)
        hu.append(torch.cat([dW,db],dim=1).cpu().numpy())  # (10,85)
    head_updates.append(hu); active.append(r>=WARMUP+RAMP)
    # FedAvg
    w=torch.tensor([len(cidx[k]) for k in range(K)],dtype=torch.float); w=(w/w.sum()).to(dev)
    new={}
    for key in _names:
        new[key]=sum(w[k]*sds[k][key].float() for k in range(K))
    gmodel.load_state_dict(new); gsd=gmodel.state_dict()
    asr=fuba_attack(gmodel,testloader,bd,EPS,THR,TGT,dev) if do_bd else 0.0
    asr_track.append(asr)
    print(f"[r{r:02d}] asr(dormant)={asr:5.2f}", flush=True)

# ===== κ 분석 =====
act=[i for i,a in enumerate(active) if a]
pairs=[(u,k) for u in range(K) for k in range(u+1,K)]
def rho_per_class(r,u,k):
    d=head_updates[r][u]-head_updates[r][k]           # (10,85)
    return np.linalg.norm(d,axis=1)                   # (10,)
# 쌍별: 활성 라운드 평균 ρ_c, κ, argmax, 타깃 순위
res={}
for (u,k) in pairs:
    R=np.stack([rho_per_class(r,u,k) for r in act],0)  # (T,10)
    rho=R.mean(0)                                      # (10,)
    kappa=rho.max()/rho.mean()
    order=np.argsort(-rho)                             # 큰 순
    tgt_rank=int(np.where(order==TGT)[0][0])+1         # 타깃이 몇 위
    res[(u,k)]=dict(kappa=float(kappa), argmax=int(rho.argmax()),
                    tgt_rank=tgt_rank, tgt_ratio=float(rho[TGT]/rho.mean()), rho=rho)

def role(u): return "ATK" if u==ATTACKER else "REQ" if u==DEFENDER else "nor"
print("\n==== FUBA κ 정밀 결과 (타깃 클래스 =",TGT,") ====")
print(f"dormant ASR(마지막)={asr_track[-1]:.2f}  각 클라 지배클래스={dom}")
print("\n쌍별 (활성 라운드 평균):  κ=max/mean,  argmax=쏠린클래스,  tgt_rank=타깃순위,  tgt_ratio=타깃/평균")
ranked=sorted(res.items(), key=lambda x:-x[1]["kappa"])
for (p,d) in ranked:
    tag=" <== (ATK,REQ)" if set(p)=={ATTACKER,DEFENDER} else ""
    print(f"  pair{p} [{role(p[0])},{role(p[1])}]: kappa={d['kappa']:.2f} argmax=c{d['argmax']} tgt_rank={d['tgt_rank']} tgt_ratio={d['tgt_ratio']:.2f}{tag}")

ad=res[(ATTACKER,DEFENDER)]
print(f"\n[공격자-요청자 쌍 (0,1)] ρ_c 벡터:")
print("   "+" ".join(f"c{c}:{ad['rho'][c]:.3f}" for c in range(10)))
print(f"   타깃 c{TGT} 는 {ad['tgt_rank']}위, 타깃/평균={ad['tgt_ratio']:.2f}, argmax=c{ad['argmax']}")

# 정상쌍 귀무분포
norm_pairs=[p for p in pairs if ATTACKER not in p and DEFENDER not in p]
nk=[res[p]["kappa"] for p in norm_pairs]
print(f"\n정상쌍 {norm_pairs} κ: {[f'{x:.2f}' for x in nk]}  평균={np.mean(nk):.2f} 최대={np.max(nk):.2f}")
# 판정: (0,1) 이 타깃으로 쏠리고 정상쌍보다 두드러지나
verdict = (ad["argmax"]==TGT) and (ad["kappa"] > np.max(nk))
print(f"\n판정: (0,1)의 쏠린 클래스가 타깃인가={ad['argmax']==TGT}, κ가 정상쌍 최대보다 큰가={ad['kappa']>np.max(nk)} -> {'작동' if verdict else '불확실/실패'}")
# ===== O 통계량: "같은 클래스 행에서 둘 다 크게·반대로" =====
def unit(v):
    n=np.linalg.norm(v); return v/n if n>1e-12 else v
def O_per_class(r,u,k):
    du=head_updates[r][u]; dk=head_updates[r][k]      # (10,85)
    out=np.zeros(10)
    for c in range(10):
        a=du[c]; b=dk[c]
        cs=float(np.dot(unit(a),unit(b)))             # 코사인
        oppose=max(0.0,-cs)                           # 반대일 때만
        mag=min(np.linalg.norm(a),np.linalg.norm(b))  # 둘 다 커야
        out[c]=oppose*mag
    return out
Ores={}
for (u,k) in pairs:
    M=np.stack([O_per_class(r,u,k) for r in act],0)   # (T,10)
    Oc=M.mean(0)
    Ores[(u,k)]=dict(O=float(Oc.max()), chat=int(Oc.argmax()), Ovec=Oc)
print("\n==== O 통계량 (같은 클래스·둘 다 큼·반대) — 타깃",TGT,"====")
Oranked=sorted(Ores.items(), key=lambda x:-x[1]["O"])
for (p,d) in Oranked:
    tag=" <== (ATK,REQ)" if set(p)=={ATTACKER,DEFENDER} else ""
    print(f"  pair{p} [{role(p[0])},{role(p[1])}]: O={d['O']:.4f} chat=c{d['chat']}{tag}")
od=Ores[(ATTACKER,DEFENDER)]
Onorm=[Ores[p]["O"] for p in pairs if ATTACKER not in p and DEFENDER not in p]
print(f"\n[공격자-요청자 (0,1)] O 벡터(클래스별): "+" ".join(f"c{c}:{od['Ovec'][c]:.3f}" for c in range(10)))
print(f"  O={od['O']:.4f}, chat=c{od['chat']} (타깃={TGT})")
print(f"  정상쌍 O: {[f'{x:.4f}' for x in Onorm]} 최대={max(Onorm):.4f}")
Overdict=(Oranked[0][0]=={ATTACKER,DEFENDER} or set(Oranked[0][0])=={ATTACKER,DEFENDER}) and od['chat']==TGT and od['O']>max(Onorm)
print(f"  판정: (0,1)이 최대쌍이며 chat=타깃이며 정상쌍보다 큼 -> {'작동' if Overdict else '불확실/실패'}")

json.dump({"asr":asr_track,"dom":dom,"target":TGT,
           "O_pairs":{f"{u}_{k}":{"O":Ores[(u,k)]["O"],"chat":Ores[(u,k)]["chat"]} for (u,k) in pairs},
           "pairs":{f"{u}_{k}":{kk:(vv if kk!='rho' else vv.tolist()) for kk,vv in res[(u,k)].items()} for (u,k) in pairs},
           "verdict":bool(verdict)}, open(os.path.join(HERE,"logs","fuba_kappa.json"),"w"), indent=2)
print("\nsaved logs/fuba_kappa.json")
