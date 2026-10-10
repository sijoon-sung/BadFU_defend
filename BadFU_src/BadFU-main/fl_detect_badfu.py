# -*- coding: utf-8 -*-
"""
fl_detect_badfu.py

BadFU(BingguangLu/BadFU) 공개 코드를 기준으로 "방법 2(반평행 탐지)"를 검증한다.
badfu.py 의 실제 구성요소(ResNet18 모델, local_train, fed_avg, evaluate_model,
unlearning_step_once, BadNet 트리거 + UBA-Inf 영향함수 위장 데이터)를 그대로 재사용한다.

탐지 가설과의 접점: badfu.py 의 'ul' 구성 =
  client 0 = clean_0 + bd(백도어: 트리거->타깃0)
  client 1..4 = 정상
  client 5 = clean_0 + cv(위장: 트리거+정답 라벨, 영향함수 교란)  <- 언러닝 요청자
공격자(0)와 요청자(5)의 라운드별 업데이트가 지속적으로 반평행한지, 정상 쌍과
구분되는지를 본다. 전체 코사인 / 분류기 헤드(fc) 코사인 / 같은위치 상쇄비율 세 가지.

RAM 절약: 라운드별 쌍 지표는 즉시 계산해 스칼라만 보관(업데이트 벡터는 버림).
언러닝 재구성을 위한 궤적(global + 6 locals)만 디스크에 저장.
"""
import argparse, copy, json, os, time, glob
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=15)
ap.add_argument("--local_epochs", type=int, default=5)
ap.add_argument("--ramp", type=int, default=3)     # 활성 판정 시작(초반 제외)
ap.add_argument("--target", type=int, default=0)
ap.add_argument("--dominant_ratio", type=float, default=0.7)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--traj_dir", default="logs/traj")
ap.add_argument("--out", default="logs/badfu_detect.json")
# ---- 축소 설정(스모크)용 선택 인자 (기본값이면 동작은 종전과 완전히 같다) ----
ap.add_argument("--max_per_client", type=int, default=0,
                help="클라이언트별 학습 샘플 상한(0=전체). bd/cv 비율은 그대로 두고 무작위 부분집합을 쓴다")
ap.add_argument("--test_n", type=int, default=0, help="평가용 clean/bd 테스트 샘플 상한(0=전체)")
ap.add_argument("--keep_traj", action="store_true", help="끝나고 logs/traj/*.pt 를 지우지 않는다 (unlearn_from_traj.py 용)")
args = ap.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
seed = args.seed
np.random.seed(seed); import random; random.seed(seed); torch.manual_seed(seed)
os.makedirs("logs", exist_ok=True); os.makedirs(args.traj_dir, exist_ok=True)
for f in glob.glob(os.path.join(args.traj_dir, "*.pt")):
    os.remove(f)

n_classes = 10
num_clients = 5
ATTACKER, REQUESTER = 0, 5   # ul 구성에서 공격자/요청자

# ========== 데이터 (badfu.py 와 동일) ==========
data_path = "./record/badnet_dataset/pert_result.pt"
data_dict = torch.load(data_path, weights_only=False)
bd_train_dict = data_dict['bd_train']
bd_ind = list(bd_train_dict['bd_data_container']['data_dict'].keys())
bd_cv_ind = list(bd_train_dict['cv_data_container']['data_dict'].keys())
bd_test_dict = data_dict['bd_test']
cv_pert_dict = data_dict['cv_pert']
cv_data_container = cv_pert_dict['data_dict']
cv_ind = list(cv_data_container.keys())
remove_indices = set(bd_ind + cv_ind)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
all_indices = range(len(train_dataset))
kept_indices = [i for i in all_indices if i not in remove_indices]
clean_data = [train_dataset.data[i] for i in kept_indices]
clean_targets = [train_dataset.targets[i] for i in kept_indices]
clean_data = torch.stack([transform(Image.fromarray(img)) for img in clean_data])
clean_targets = torch.tensor(clean_targets)

class CustomCIFARDataset(Dataset):
    def __init__(self, data, targets): self.data, self.targets = data, targets
    def __getitem__(self, i): return self.data[i], self.targets[i]
    def __len__(self): return len(self.targets)
clean_dataset = CustomCIFARDataset(clean_data, clean_targets)

dominant_client_map = {0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:4}
targets = clean_dataset.targets.numpy()
client_clean_data = [[] for _ in range(num_clients)]
for c in np.unique(targets):
    ci = np.where(targets == c)[0]; np.random.shuffle(ci)
    dom = dominant_client_map[c]; sp = int(len(ci) * args.dominant_ratio)
    for idx in ci[:sp]:
        client_clean_data[dom].append(clean_dataset[idx])
    rem = ci[sp:]; nds = [x for x in range(num_clients) if x != dom]
    chunk = len(rem) // (num_clients - 1)
    for i, cid in enumerate(nds):
        s = i*chunk; e = (i+1)*chunk if i < len(nds)-1 else len(rem)
        for idx in rem[s:e]:
            client_clean_data[cid].append(clean_dataset[idx])

# ---- BackdoorDataset: 경로 자동 치유(이동/누락 대비) ----
PERT = "./record/badnet_dataset/cv_train_dataset/pert"
BDTR = "./record/badnet_dataset/bd_train_dataset"
def _resolve(path):
    if os.path.exists(path): return path
    base = os.path.basename(path); cls = os.path.basename(os.path.dirname(path))
    for cand in (os.path.join(PERT, cls, base), os.path.join(BDTR, cls, base)):
        if os.path.exists(cand): return cand
    return path  # 그대로(열 때 에러로 드러남)

class BackdoorDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict; self.keys = list(data_dict.keys()); self.transform = transform
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        info = self.data_dict[self.keys[idx]]
        img = Image.open(_resolve(info['path'])).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, torch.tensor(info['other_info'][0])

bd_dataset = BackdoorDataset(bd_train_dict['bd_data_container']['data_dict'], transform)
bd_test_dataset = BackdoorDataset(bd_test_dict['bd_data_container']['data_dict'], transform)
cv_dataset = BackdoorDataset(cv_data_container, transform)

# ---- ul 구성 ----
client_datasets = client_clean_data
ul_client_datasets = [copy.deepcopy(client_datasets[i]) for i in range(num_clients)]
ul_client_datasets[0] = ConcatDataset([client_datasets[0], bd_dataset])
ul_client_datasets.append(ConcatDataset([client_datasets[0], cv_dataset]))  # client 5

def _subset(ds, n, s):
    """n>0 이면 고정 시드 무작위 부분집합(n 개). n=0 이면 그대로."""
    if n <= 0 or len(ds) <= n:
        return ds
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(s))[:n].tolist()
    return torch.utils.data.Subset(ds, idx)

ul_client_datasets = [_subset(d, args.max_per_client, seed + i) for i, d in enumerate(ul_client_datasets)]
ul_loaders = [DataLoader(d, batch_size=64, shuffle=True, num_workers=0) for d in ul_client_datasets]
client_data_counts = [len(l.dataset) for l in ul_loaders]
print(f"client_data_counts = {client_data_counts}  (release_map.py --counts 에 그대로 넘긴다)", flush=True)
# 백도어만 든 client0 로더(언러닝 단계 재학습용)
bd_only_ds0 = _subset(ConcatDataset([client_datasets[0], bd_dataset]), args.max_per_client, seed)

test_dataset = _subset(datasets.CIFAR10(root='./data', train=False, download=True, transform=transform), args.test_n, seed)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
bd_test_loader = DataLoader(_subset(bd_test_dataset, args.test_n, seed), batch_size=64, shuffle=False, num_workers=0)

# ========== 모델/학습 (badfu.py 와 동일) ==========
class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x): return self.model(x)

def local_train(model, loader, epochs):
    model.train(); opt = optim.SGD(model.parameters(), lr=0.01); crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad(); crit(model(data), target).backward(); opt.step()
    return model.state_dict()

def fed_avg(global_model, states, counts):
    gsd = global_model.state_dict(); agg = {k: torch.zeros_like(gsd[k], dtype=torch.float32) for k in gsd}
    tot = sum(counts)
    for st, n in zip(states, counts):
        w = n / tot
        for k in gsd: agg[k] += st[k].float().to(agg[k].device) * w
    for k in gsd:
        if gsd[k].dtype == torch.long: agg[k] = torch.round(agg[k]).to(torch.long)
    global_model.load_state_dict(agg); return global_model

@torch.no_grad()
def evaluate(model, clean_loader, poison_loader, bd_target=0):
    model.eval(); tot=cor=0
    for data, target in clean_loader:
        data, target = data.to(device), target.to(device)
        p = model(data).argmax(1); tot += target.size(0); cor += (p == target).sum().item()
    acc = 100.0*cor/tot
    tp=cp=0
    for data, target in poison_loader:
        data, target = data.to(device), target.to(device)
        p = model(data).argmax(1); tp += target.size(0); cp += (p == bd_target).sum().item()
    return acc, 100.0*cp/tp

def unlearning_step_once(old_cms, new_cms, gm_before, gm_after):
    nss = gm_after.state_dict(); oss = gm_before.state_dict(); ret = {}
    assert len(old_cms) == len(new_cms)
    for layer in oss.keys():
        ou = torch.zeros_like(oss[layer], dtype=torch.float32); nu = torch.zeros_like(oss[layer], dtype=torch.float32)
        for ii in range(len(new_cms)):
            ou += old_cms[ii].state_dict()[layer].float(); nu += new_cms[ii].state_dict()[layer].float()
        ou = ou/float(ii+1) - oss[layer].float(); nu = nu/float(ii+1) - nss[layer].float()
        sl = torch.norm(ou); sd = nu/(torch.norm(nu)+1e-12)
        v = nss[layer].float() + sl*sd
        ret[layer] = torch.round(v).to(torch.long) if nss[layer].dtype == torch.long else v.to(nss[layer].dtype)
    out = copy.deepcopy(gm_after); out.load_state_dict(ret); return out

# ========== 탐지 지표 ==========
_keys = list(ResNet18().state_dict().keys())
FC_KEYS = [k for k in _keys if k.endswith("fc.weight") or k.endswith("fc.bias")]
def flat(sd, keys=None):
    keys = keys or _keys
    return torch.cat([sd[k].flatten().float() for k in keys])
def cosv(a, b): return torch.dot(a, b).item()/(a.norm().item()*b.norm().item()+1e-12)
def cancelv(a, b):
    pr = a*b; return torch.clamp(-pr, min=0).sum().item()/(pr.abs().sum().item()+1e-12)

# ========== FL (ul 구성, dormant 학습) ==========
K = len(ul_loaders)   # 6
pairs = [(u, k) for u in range(K) for k in range(u+1, K)]
cos_h = {p: [] for p in pairs}; fc_h = {p: [] for p in pairs}; can_h = {p: [] for p in pairs}
active = []; acc_track = []; asr_track = []
gmodel = ResNet18(pretrained=True).to(device)

t0 = time.time()
for r in range(args.rounds):
    gsd = {k: v.detach().clone() for k, v in gmodel.state_dict().items()}
    gflat = flat(gsd); gfc = flat(gsd, FC_KEYS)
    states = []
    round_updates = []; round_fc = []
    for c in range(K):
        lm = ResNet18(pretrained=False).to(device); lm.load_state_dict(gsd)
        st = local_train(lm, ul_loaders[c], args.local_epochs)
        states.append({k: v.detach().cpu() for k, v in st.items()})
        lf = flat(st).to(device); round_updates.append((lf - gflat).detach())
        round_fc.append((flat(st, FC_KEYS).to(device) - gfc).detach())
        del lm; torch.cuda.empty_cache()
    # 쌍 지표 즉시 계산 후 벡터 폐기
    for (u, k) in pairs:
        cos_h[(u, k)].append(cosv(round_updates[u], round_updates[k]))
        fc_h[(u, k)].append(cosv(round_fc[u], round_fc[k]))
        can_h[(u, k)].append(cancelv(round_updates[u], round_updates[k]))
    del round_updates, round_fc
    active.append(r >= args.ramp)
    # 궤적 저장(언러닝용): round 시작 global + 6 locals
    torch.save({"gm": gsd, "cms": states}, os.path.join(args.traj_dir, f"round_{r}.pt"))
    # 집계
    gmodel = fed_avg(gmodel, states, client_data_counts)
    torch.save({"gm_after": {k: v.detach().cpu() for k, v in gmodel.state_dict().items()}},
               os.path.join(args.traj_dir, f"glob_{r}.pt"))
    acc, asr = evaluate(gmodel, test_loader, bd_test_loader, args.target)
    acc_track.append(acc); asr_track.append(asr)
    print(f"[r{r:02d}] acc={acc:5.2f} asr(dormant)={asr:6.2f} "
          f"cos(0,5)={cos_h[(0,5)][-1]:+.3f} fc(0,5)={fc_h[(0,5)][-1]:+.3f} cancel(0,5)={can_h[(0,5)][-1]:.3f}",
          flush=True)

# dormant 최종
acc_T, asr_T = evaluate(gmodel, test_loader, bd_test_loader, args.target)

# ========== 언러닝: 요청자(client5) 제거 후 활성화 확인 (FedEraser) ==========
# badfu.py 와 동일: old_GMs[r+1] 과 old_CMs[r](요청자 제외)로 보정, 매 라운드 bd-only 재학습
def load_gm_after(r):
    d = torch.load(os.path.join(args.traj_dir, f"glob_{r}.pt"), weights_only=False)
    m = ResNet18().to(device); m.load_state_dict(d["gm_after"]); return m
def load_round(r):
    d = torch.load(os.path.join(args.traj_dir, f"round_{r}.pt"), weights_only=False)
    gm = ResNet18().to(device); gm.load_state_dict(d["gm"])
    cms = []
    for st in d["cms"]:
        m = ResNet18().to(device); m.load_state_dict(st); cms.append(m)
    return gm, cms

bd_only_loader0 = DataLoader(bd_only_ds0, batch_size=64, shuffle=True, num_workers=0)
global_ul = load_gm_after(0)   # badfu.py: global_model_ul = old_GMs[0]
keep = [c for c in range(K) if c != REQUESTER]
for r in range(args.rounds - 1):    # old_GMs[r+1] 접근하므로 rounds-1 까지
    if r % 2 == 1:
        continue
    # 현재 ul global 에서 요청자 제외 클라이언트 재학습
    new_cms = []
    _, cms_r = load_round(r)
    for c in keep:
        lm = ResNet18(pretrained=False).to(device); lm.load_state_dict(global_ul.state_dict())
        loader = bd_only_loader0 if c == ATTACKER else ul_loaders[c]
        local_train(lm, loader, args.local_epochs)
        new_cms.append(lm)
    old_cms_keep = [cms_r[c] for c in keep]
    gm_before = load_gm_after(r + 1)   # badfu.py: old_GMs[r+1]
    global_ul = unlearning_step_once(old_cms_keep, new_cms, gm_before, global_ul)
    del new_cms, cms_r; torch.cuda.empty_cache()
acc_U, asr_U = evaluate(global_ul, test_loader, bd_test_loader, args.target)

# ========== 점수화 ==========
act = [i for i, a in enumerate(active) if a]
def pget(d, u, k): return d[(u, k)] if u < k else d[(k, u)]
def zed(v): a = np.array([v[i] for i in act]); return float(a.mean()/(a.std()/np.sqrt(len(a))+1e-12))
def mean_a(v): return float(np.mean([v[i] for i in act]))
def score(h, mode):
    S, bp = {}, {}
    for u in range(K):
        vv = {k: (zed(pget(h, u, k)) if mode=="z" else mean_a(pget(h, u, k))) for k in range(K) if k != u}
        best = min(vv, key=vv.get) if mode=="z" else max(vv, key=vv.get)
        S[u], bp[u] = vv[best], best
    return S, bp
S_cos, bp_cos = score(cos_h, "z"); S_fc, bp_fc = score(fc_h, "z"); S_can, bp_can = score(can_h, "mean")
def rolen(u): return "Adv-attacker" if u==ATTACKER else "Adv-defender(req)" if u==REQUESTER else "normal"
def devsig(S, who):
    nv = [S[u] for u in range(K) if u not in (ATTACKER, REQUESTER)]
    return (S[who]-float(np.mean(nv)))/(float(np.std(nv))+1e-12)

print("\n===== attack (BadNet trigger + UBA-Inf camouflage) =====")
print(f"dormant theta_T : acc={acc_T:5.2f} asr={asr_T:6.2f}")
print(f"after unlearn req(c5): acc={acc_U:5.2f} asr={asr_U:6.2f}  (BadFU 주장: 점프)")
print("\n===== detection (active rounds) =====")
for u in range(K):
    print(f"c{u} [{rolen(u):18s}] S_cos={S_cos[u]:+7.2f}(p{bp_cos[u]}) S_fc={S_fc[u]:+7.2f}(p{bp_fc[u]}) S_cancel={S_can[u]:.3f}(p{bp_can[u]})")
print(f"\nrequester(c5) dev vs normals: cos={devsig(S_cos,REQUESTER):+.2f}s fc={devsig(S_fc,REQUESTER):+.2f}s cancel={devsig(S_can,REQUESTER):+.2f}s")
print(f"requester most-antiparallel: cos->p{bp_cos[REQUESTER]} fc->p{bp_fc[REQUESTER]} (attacker=c{ATTACKER})")

res = {"args": vars(args), "acc_track": acc_track, "asr_track": asr_track,
       "unlearn": {"acc_T": acc_T, "asr_T": asr_T, "acc_U": acc_U, "asr_U": asr_U},
       "S_cos": S_cos, "bp_cos": bp_cos, "S_fc": S_fc, "bp_fc": bp_fc, "S_cancel": S_can, "bp_cancel": bp_can,
       "cos_0_5": pget(cos_h,0,5), "fc_0_5": pget(fc_h,0,5), "cancel_0_5": pget(can_h,0,5),
       "active": active, "seconds": time.time()-t0}
with open(args.out, "w") as f: json.dump(res, f, indent=2)
# 궤적 정리(디스크 회수). --keep_traj 면 남긴다.
if not args.keep_traj:
    for f in glob.glob(os.path.join(args.traj_dir, "*.pt")): os.remove(f)
print(f"\nsaved: {args.out}  ({res['seconds']:.1f}s)")
