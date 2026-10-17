# -*- coding: utf-8 -*-
"""
보존된 dormant 궤적(logs/traj_keep)으로 '언러닝 활성화'만 측정한다.
dormant 재학습은 생략(이미 r0..11 궤적 있음). 요청자(client5=cv) 제거 후 FedEraser 재구성.
공격이 진짜면: dormant ASR(낮음) -> 언러닝 후 ASR 점프.
"""
import os, copy, glob, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as transforms, torchvision.datasets as datasets, torchvision.models as models
import numpy as np
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
# 축소 설정(스모크)용 선택 인자 (기본값이면 종전과 같다: logs/traj_keep, 5 에폭, 전체 데이터)
ap=argparse.ArgumentParser()
ap.add_argument("--traj_dir", default="logs/traj_keep")
ap.add_argument("--local_epochs", type=int, default=5)
ap.add_argument("--max_per_client", type=int, default=0, help="클라이언트별 학습 샘플 상한(0=전체). fl_detect_badfu.py 와 같은 값을 줘야 분할이 일치")
ap.add_argument("--test_n", type=int, default=0, help="평가용 clean/bd 테스트 샘플 상한(0=전체)")
args=ap.parse_args()
device="cuda:0" if torch.cuda.is_available() else "cpu"
seed=42; np.random.seed(seed); import random; random.seed(seed); torch.manual_seed(seed)
TRAJ=args.traj_dir; ATK,REQ=0,5; TARGET=0; num_clients=5; local_epochs=args.local_epochs; dominant_ratio=0.7
def _subset(ds,n,s):
    if n<=0 or len(ds)<=n: return ds
    idx=torch.randperm(len(ds),generator=torch.Generator().manual_seed(s))[:n].tolist()
    return torch.utils.data.Subset(ds,idx)

# ===== 데이터 (fl_detect_badfu.py 와 동일) =====
data_dict=torch.load("record/badnet_dataset/pert_result.pt", weights_only=False)
bd_train_dict=data_dict['bd_train']; bd_test_dict=data_dict['bd_test']
cv_data_container=data_dict['cv_pert']['data_dict']
remove_indices=set(list(bd_train_dict['bd_data_container']['data_dict'].keys())+list(cv_data_container.keys()))
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
train_dataset=datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
kept=[i for i in range(len(train_dataset)) if i not in remove_indices]
clean_data=torch.stack([transform(Image.fromarray(train_dataset.data[i])) for i in kept])
clean_targets=torch.tensor([train_dataset.targets[i] for i in kept])
class CC(Dataset):
    def __init__(s,d,t): s.data,s.targets=d,t
    def __getitem__(s,i): return s.data[i],s.targets[i]
    def __len__(s): return len(s.targets)
clean_dataset=CC(clean_data,clean_targets)
dmap={0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:4}
tg=clean_dataset.targets.numpy(); client_clean=[[] for _ in range(num_clients)]
for c in np.unique(tg):
    ci=np.where(tg==c)[0]; np.random.shuffle(ci); dom=dmap[c]; sp=int(len(ci)*dominant_ratio)
    for idx in ci[:sp]: client_clean[dom].append(clean_dataset[idx])
    rem=ci[sp:]; nds=[x for x in range(num_clients) if x!=dom]; ch=len(rem)//(num_clients-1)
    for i,cid in enumerate(nds):
        s=i*ch; e=(i+1)*ch if i<len(nds)-1 else len(rem)
        for idx in rem[s:e]: client_clean[cid].append(clean_dataset[idx])
PERT="./record/badnet_dataset/cv_train_dataset/pert"; BDTR="./record/badnet_dataset/bd_train_dataset"
def _res(p):
    if os.path.exists(p): return p
    b=os.path.basename(p); cl=os.path.basename(os.path.dirname(p))
    for cand in (os.path.join(PERT,cl,b),os.path.join(BDTR,cl,b)):
        if os.path.exists(cand): return cand
    return p
class BD(Dataset):
    def __init__(s,dd,t): s.dd=dd; s.k=list(dd.keys()); s.t=t
    def __len__(s): return len(s.k)
    def __getitem__(s,i):
        info=s.dd[s.k[i]]; img=Image.open(_res(info['path'])).convert('RGB')
        return s.t(img), torch.tensor(info['other_info'][0])
bd_dataset=BD(bd_train_dict['bd_data_container']['data_dict'],transform)
bd_test_dataset=BD(bd_test_dict['bd_data_container']['data_dict'],transform)
cv_dataset=BD(cv_data_container,transform)
ul=[copy.deepcopy(client_clean[i]) for i in range(num_clients)]
ul[0]=ConcatDataset([client_clean[0],bd_dataset]); ul.append(ConcatDataset([client_clean[0],cv_dataset]))
ul=[_subset(d,args.max_per_client,seed+i) for i,d in enumerate(ul)]   # fl_detect_badfu.py 와 같은 시드/순서
ul_loaders=[DataLoader(d,batch_size=64,shuffle=True,num_workers=0) for d in ul]
bd_only0=DataLoader(_subset(ConcatDataset([client_clean[0],bd_dataset]),args.max_per_client,seed),batch_size=64,shuffle=True,num_workers=0)
test_loader=DataLoader(_subset(datasets.CIFAR10(root='./data',train=False,download=True,transform=transform),args.test_n,seed),batch_size=64,num_workers=0)
bd_test_loader=DataLoader(_subset(bd_test_dataset,args.test_n,seed),batch_size=64,num_workers=0)

class ResNet18(nn.Module):
    def __init__(s):
        super().__init__(); s.model=models.resnet18(weights=None)
        s.model.conv1=nn.Conv2d(3,64,3,1,1,bias=False); s.model.maxpool=nn.Identity()
        s.model.fc=nn.Linear(s.model.fc.in_features,10)
    def forward(s,x): return s.model(x)
def ltrain(m,loader):
    m.train(); opt=optim.SGD(m.parameters(),lr=0.01); crit=nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for d,t in loader:
            d,t=d.to(device),t.to(device); opt.zero_grad(); crit(m(d),t).backward(); opt.step()
@torch.no_grad()
def ev(m):
    m.eval(); tot=cor=0
    for d,t in test_loader:
        d,t=d.to(device),t.to(device); cor+=(m(d).argmax(1)==t).sum().item(); tot+=t.size(0)
    acc=100*cor/tot; tp=cp=0
    for d,t in bd_test_loader:
        d,t=d.to(device),t.to(device); cp+=(m(d).argmax(1)==TARGET).sum().item(); tp+=t.size(0)
    return acc,100*cp/tp
def uso(old_cms,new_cms,gmb,gma):
    nss=gma.state_dict(); oss=gmb.state_dict(); ret={}
    for L in oss:
        ou=torch.zeros_like(oss[L],dtype=torch.float32); nu=torch.zeros_like(oss[L],dtype=torch.float32)
        for ii in range(len(new_cms)):
            ou+=old_cms[ii].state_dict()[L].float(); nu+=new_cms[ii].state_dict()[L].float()
        ou=ou/float(ii+1)-oss[L].float(); nu=nu/float(ii+1)-nss[L].float()
        v=nss[L].float()+torch.norm(ou)*(nu/(torch.norm(nu)+1e-12))
        ret[L]=torch.round(v).to(torch.long) if nss[L].dtype==torch.long else v.to(nss[L].dtype)
    o=copy.deepcopy(gma); o.load_state_dict(ret); return o

def load_glob(r):
    d=torch.load(f"{TRAJ}/glob_{r}.pt",map_location="cpu",weights_only=False)
    m=ResNet18().to(device); m.load_state_dict(d["gm_after"]); return m
def load_round(r):
    d=torch.load(f"{TRAJ}/round_{r}.pt",map_location="cpu",weights_only=False)
    cms=[]
    for st in d["cms"]:
        m=ResNet18().to(device); m.load_state_dict(st); cms.append(m)
    return cms

N=len(glob.glob(f"{TRAJ}/round_*.pt"))
print(f"rounds available: {N}")
theta_T=load_glob(N-1); acc_T,asr_T=ev(theta_T)
print(f"dormant theta_T (glob_{N-1}): acc={acc_T:.2f} asr={asr_T:.2f}")

keep=[c for c in range(6) if c!=REQ]
global_ul=load_glob(0)
for r in range(N-1):
    if r%2==1: continue
    cms_r=load_round(r); old_keep=[cms_r[c] for c in keep]
    new_cms=[]
    for c in keep:
        m=ResNet18().to(device); m.load_state_dict(global_ul.state_dict())
        ltrain(m, bd_only0 if c==ATK else ul_loaders[c]); new_cms.append(m)
    gmb=load_glob(r+1)
    global_ul=uso(old_keep,new_cms,gmb,global_ul)
    del cms_r,old_keep,new_cms,gmb; torch.cuda.empty_cache()
    a,s=ev(global_ul); print(f"  unlearn step r{r}: acc={a:.2f} asr={s:.2f}", flush=True)
acc_U,asr_U=ev(global_ul)
print(f"\n=== 결과 ===")
print(f"dormant        : acc={acc_T:.2f} asr={asr_T:.2f}")
print(f"요청자(c5) 언러닝후: acc={acc_U:.2f} asr={asr_U:.2f}")
print(f"ASR {asr_T:.1f} -> {asr_U:.1f}  ({'점프(활성화)' if asr_U>asr_T+10 else '변화 작음'})")
import json
json.dump({"acc_T":acc_T,"asr_T":asr_T,"acc_U":acc_U,"asr_U":asr_U,"rounds":N},
          open("logs/unlearn_activation.json","w"),indent=2)
print("saved logs/unlearn_activation.json")
