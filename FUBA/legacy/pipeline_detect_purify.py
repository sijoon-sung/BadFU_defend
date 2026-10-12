# -*- coding: utf-8 -*-
"""
pipeline_detect_purify.py — 공격 궤적 -> 탐지 -> 정화 -> 통계, 한 번에.

입력: 실제 FUBA(mpiexec --save) 가 남긴 checkpoints/{global,local,backdoor}_net_model_{name}_round_*.pkl
단계:
  [탐지] 저장된 업데이트만으로 (라벨/트리거 없이)
     - O 통계: 쌍 (u,k), 클래스 c 의 헤드행에서 "둘 다 크게·반대" = max(0,-cos)·min(||·||)
     - 타깃 t^ = 전체 쌍 O 질량이 최대인 클래스
     - 공격자군집 Â = t^ 행에서 서로 정렬(cos>0.5)·크기 상위
     - 요청자 û = Â 밖에서 t^ 행을 가장 크게·Â 와 반대로 움직인 클라
     - 정답(atk/req/target)과 대조해 탐지 정확도 산출
  [정화] 1차근사 theta_T = theta_0 + sum U_k.  요청자 기여는 전부 제거(권리).
     - V = 탐지된 Â 업데이트의 fc3 상위-k 주성분
     - 모든 클라의 V 성분 제거(판정 없이): theta_T - U_req - sum_{k!=req} P_V U_k
     - 비교군: none / req-only(활성화) / atk-full / randV
  [평가] ACC, ASR(저장 백도어 생성기), 정상 기여 손실, 두-삭제 재구성 기준과의 거리
출력: logs/pipeline_{name}.json  (여러 name 집계는 aggregate_pipeline.py)
"""
import os, sys, pickle, json, argparse, itertools
import numpy as np, torch
from torchvision import transforms
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0,HERE)
from model import Net, MNISTAutoencoder
from utils.comm_utils import attack as fuba_attack, test as fuba_test
from dataset import MNIST

P=argparse.ArgumentParser()
P.add_argument("--name",default="det1"); P.add_argument("--rounds",type=int,default=8)
P.add_argument("--warmup",type=int,default=3); P.add_argument("--K",type=int,default=8)
P.add_argument("--atk",type=int,nargs="+",default=[0,1,2,3]); P.add_argument("--req",type=int,default=4)
P.add_argument("--target",type=int,default=8); P.add_argument("--eps",type=float,default=1.0); P.add_argument("--thr",type=float,default=0.04)
P.add_argument("--kV",type=int,default=8)   # V 차원(fc3)
a=P.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
os.makedirs("logs",exist_ok=True)
GT_ATK=set(a.atk); GT_REQ=a.req; GT_T=a.target; ACTIVE=list(range(a.warmup+1,a.rounds+1))

# ---------- 평가 준비 ----------
testset=MNIST(root='./data',train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))
Xte=torch.stack([testset[i][0] for i in range(len(testset))]).to(dev)
Yte=torch.tensor([testset[i][1] for i in range(len(testset))]).to(dev)
class TL:
    def __iter__(s):
        for i in range(0,len(Xte),500): yield Xte[i:i+500].clone(), Yte[i:i+500].clone()
loader=TL()
bd=MNISTAutoencoder().to(dev); bd.load_state_dict(pickle.load(open(f"checkpoints/backdoor_net_model_{a.name}_round_{a.rounds}.pkl","rb"))); bd.eval()
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
    return fuba_test(m,loader,dev), fuba_attack(m,loader,bd,a.eps,a.thr,GT_T,dev)
def fc3_head(v):  # (10,85)
    W=v[slice(*sl["fc3.weight"])].view(10,84); b=v[slice(*sl["fc3.bias"])].view(10,1)
    return torch.cat([W,b],1)

# ---------- 궤적 -> 기여 ----------
def load_g(r): return flat(pickle.load(open(f"checkpoints/global_net_model_{a.name}_round_{r}.pkl","rb")))
def load_l(r): return [flat(sd) for sd in pickle.load(open(f"checkpoints/local_net_model_{a.name}_round_{r}.pkl","rb"))]
def fc3_vec(v):  # (850,) = [fc3.weight(840), fc3.bias(10)]  (embed_fc3 와 같은 순서)
    return torch.cat([v[slice(*sl["fc3.weight"])], v[slice(*sl["fc3.bias"])]])
theta_T=load_g(a.rounds)
U={k:torch.zeros(D) for k in range(a.K)}
HU={}    # r -> [k] (10,85) 헤드 업데이트 (탐지용, 클래스행 접근)
HUV={}   # r -> [k] (850,) [W,b] 헤드 업데이트 (정화 V 추정용)
for r in range(2,a.rounds+1):
    g=load_g(r-1); L=load_l(r)
    for k in range(a.K): U[k]+=(L[k]-g)/a.K
    if r in ACTIVE:
        HU[r]=[fc3_head(L[k]-g) for k in range(a.K)]
        HUV[r]=[fc3_vec(L[k]-g) for k in range(a.K)]

# ========== [탐지] ==========
def unit(x): n=x.norm(); return x/n if n>1e-12 else x
pairs=list(itertools.combinations(range(a.K),2))
# 클래스별 O 질량
Omass=torch.zeros(10)
for (u,k) in pairs:
    for r in ACTIVE:
        du,dk=HU[r][u],HU[r][k]
        for c in range(10):
            Omass[c]+=max(0.0,-float(torch.dot(unit(du[c]),unit(dk[c]))))*min(du[c].norm(),dk[c].norm())
t_hat=int(Omass.argmax())
# 타깃행 크기 = '노름의 평균'(라운드 간 상쇄 없이 움직임량), 방향정렬 = 라운드 평균 cos
TRr={r:[HU[r][k][t_hat] for k in range(a.K)] for r in ACTIVE}
mag=np.array([float(np.mean([TRr[r][k].norm().item() for r in ACTIVE])) for k in range(a.K)])
Amat=np.eye(a.K)
for i in range(a.K):
    for j in range(i+1,a.K):
        Amat[i,j]=Amat[j,i]=float(np.mean([torch.dot(unit(TRr[r][i]),unit(TRr[r][j])).item() for r in ACTIVE]))
# 공격자군집: '서로 정렬된(여럿) 군집'. 씨앗마다 {j:cos>0.5} 그룹 -> (크기, 내부평균정렬) 최대. 크기>=2.
best=None
for s in range(a.K):
    grp=sorted(k for k in range(a.K) if Amat[s,k]>0.5)
    if len(grp)<2: continue
    internal=np.mean([Amat[i,j] for i in grp for j in grp if i<j]) if len(grp)>1 else 0
    key=(len(grp),internal)
    if best is None or key>best[0]: best=(key,set(grp))
cluster=best[1] if best else {int(mag.argmax())}
align_c=np.array([float(np.mean([Amat[k,c] for c in cluster])) for k in range(a.K)])
# 요청자: 군집 밖, 군집과 반대(정렬<0), 타깃행을 가장 크게 움직인 자
cand=[k for k in range(a.K) if k not in cluster and align_c[k]<0]
req_hat=int(max(cand,key=lambda k:mag[k])) if cand else -1
A_hat=cluster
det={"t_hat":t_hat,"target_correct":t_hat==GT_T,
     "A_hat":sorted(A_hat),"atk_precision":len(A_hat&GT_ATK)/max(len(A_hat),1),
     "atk_recall":len(A_hat&GT_ATK)/len(GT_ATK),
     "req_hat":req_hat,"req_correct":req_hat==GT_REQ,
     "mag_on_target":{int(k):round(float(mag[k]),4) for k in range(a.K)},
     "align_to_cluster":{int(k):round(float(align_c[k]),3) for k in range(a.K)}}

# ========== [정화] (탐지 결과만 사용) ==========
# V: 탐지된 공격자군집의 fc3 업데이트 상위-k 주성분
def fc3_of(v): return fc3_head(v).flatten()
def topk_basis(vecs,k):
    M=torch.stack(vecs,0); M=M-M.mean(0,keepdim=True) if len(vecs)>1 else M
    _,_,Vt=torch.linalg.svd(M,full_matrices=False); return Vt[:min(k,Vt.shape[0])]
def embed_fc3(basis_row):  # (85,) fc3공간 -> (D,) 전체에 삽입
    e=torch.zeros(D); W=basis_row[:840]; b=basis_row[840:850]
    e[slice(*sl["fc3.weight"])]=W; e[slice(*sl["fc3.bias"])]=b; return e
atk_src = sorted(A_hat) if A_hat else a.atk
# V = 탐지된 공격자들의 '라운드별' 헤드 업데이트 상위-k 주성분 (fc3 공간)
Avec=[HUV[r][k] for r in ACTIVE for k in atk_src]          # 각 (850,)
Bfc=topk_basis(Avec, a.kV)                                  # (k,850)
Bemb=torch.stack([embed_fc3(Bfc[i]) for i in range(Bfc.shape[0])])  # (k,D)
Bemb=torch.linalg.qr(Bemb.T)[0].T                          # 정규직교
def projV(v): return Bemb.T@(Bemb@v)
req=req_hat if req_hat>=0 else a.req
NORgt=[k for k in range(a.K) if k not in GT_ATK and k!=GT_REQ]

def rep(v): acc,asr=ev(v); return {"acc":round(acc,2),"asr":round(asr,2)}
pur={}
pur["none_dormant"]=rep(theta_T)
pur["req_only_activation"]=rep(theta_T-U[req])
pur["req+atk_full"]=rep(theta_T-U[req]-sum(U[k] for k in atk_src))
pur["req+allV"]=rep(theta_T-U[req]-sum(projV(U[k]) for k in range(a.K) if k!=req))
Rrand=torch.linalg.qr(torch.randn(D,Bemb.shape[0]))[0].T
def projR(v): return Rrand.T@(Rrand@v)
pur["req+randV_ctrl"]=rep(theta_T-U[req]-sum(projR(U[k]) for k in atk_src))
# 정상 기여 손실(V 제거가 정상 클라 기여를 얼마나 깎나)
nor_tot=sum(U[k].norm() for k in NORgt).item(); nor_rm=sum(projV(U[k]).norm() for k in NORgt).item()
pur["normal_contrib_loss_frac"]=round(nor_rm/max(nor_tot,1e-12),4)
# 기준선: 두-삭제 재구성(공격자+요청자 제거) = '없었을 모델' 근사
ref=rep(theta_T-U[req]-sum(U[k] for k in atk_src))
pur["ref_two_deletion"]=ref

out={"name":a.name,"detection":det,"purification":pur}
json.dump(out,open(f"logs/pipeline_{a.name}.json","w"),indent=2,ensure_ascii=False)
print(f"==== {a.name} ====")
print(f"[탐지] 타깃 t^={t_hat}(정답{GT_T},{'O' if det['target_correct'] else 'X'})  "
      f"공격자군집={sorted(A_hat)}(P={det['atk_precision']:.2f},R={det['atk_recall']:.2f})  "
      f"요청자 û={req_hat}(정답{GT_REQ},{'O' if det['req_correct'] else 'X'})")
print(f"       타깃행 크기: "+" ".join(f"c{k}:{mag[k]:.3f}" for k in range(a.K)))
print("[정화] (요청자 기여는 항상 전부 제거)")
for k,v in pur.items():
    if isinstance(v,dict): print(f"   {k:24s} ACC={v['acc']:6.2f} ASR={v['asr']:6.2f}")
    else: print(f"   {k:24s} {v}")
print("saved",f"logs/pipeline_{a.name}.json")
