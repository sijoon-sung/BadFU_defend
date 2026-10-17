# -*- coding: utf-8 -*-
"""
exp_detect_antiparallel_v2.py  (v1 진단 반영)

v1 에서 드러난 문제 두 가지를 고친다.
  (1) 백도어가 심기지 않아(ASR 0.7%->2.6%) 탐지할 대상 자체가 약했다.
      -> 클라이언트를 5명으로 줄여 공격자/요청자 가중치를 키우고,
         악성(공격자+요청자) 업데이트를 대칭 계수 mal_scale 로 증폭한다.
         대칭이므로 학습 중에는 여전히 상쇄(잠복)되고, 요청자 기여를 빼면
         공격자의 증폭된 백도어만 남아 활성화된다.
  (2) 탐지 점수를 워밍업/초반까지 평균해 후반 신호를 지웠다.
      -> 백도어 활성 라운드(r >= warmup + ramp)에서만 점수를 계산한다.

전체 코사인 / 같은위치 상쇄비율 / 출력층(fc3) 한정 코사인 세 가지를 함께 본다.
"""
import argparse, copy, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=5)
p.add_argument("--rounds", type=int, default=25)
p.add_argument("--local_epochs", type=int, default=2)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--ramp", type=int, default=3)        # 활성 판정 시작을 warmup 뒤로 더 미룸
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--target", type=int, default=0)
p.add_argument("--poison_frac", type=float, default=0.5)
p.add_argument("--mal_scale", type=float, default=3.0)  # 악성 업데이트 대칭 증폭
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="logs/detect_antiparallel_v2.json")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)

tf = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(os.path.join(HERE, "data"), train=True, download=True, transform=tf)
test = datasets.MNIST(os.path.join(HERE, "data"), train=False, download=True, transform=tf)
Xtr = torch.stack([train[i][0] for i in range(len(train))]).to(dev)
Ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(dev)
Xte = torch.stack([test[i][0] for i in range(len(test))]).to(dev)
Yte = torch.tensor([test[i][1] for i in range(len(test))]).to(dev)

def dirichlet_split(Y, K, alpha, g):
    parts = [[] for _ in range(K)]
    for c in range(10):
        idx = torch.where(Y == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g, device="cpu").to(Y.device)]
        pr = np.random.dirichlet([alpha] * K)
        cut = (np.cumsum(pr) * len(idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idx.cpu().numpy(), cut)):
            parts[k].append(torch.tensor(chunk, device=Y.device))
    return [torch.cat(pp) for pp in parts]

g = torch.Generator().manual_seed(args.seed)
client_idx = dirichlet_split(Ytr, args.K, args.alpha, g)
ATTACKER, DEFENDER = 0, 1

def add_trigger(x):
    x = x.clone(); x[:, :, -4:-1, -4:-1] = 1.0; return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5); self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120); self.fc2 = nn.Linear(120, 84); self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))); x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

# fc3 파라미터가 전체 평탄화 벡터에서 차지하는 슬라이스 index
_tmp = Net()
_names, _slices, _i = [], {}, 0
for k, v in _tmp.state_dict().items():
    n = v.numel(); _slices[k] = (_i, _i + n); _i += n
FC3_SLICE = (_slices["fc3.weight"][0], _slices["fc3.bias"][1])

def get_flat(sd):
    return torch.cat([v.flatten().float() for v in sd.values()])

def set_from_flat(model, flat):
    i = 0; sd = model.state_dict()
    for k, v in sd.items():
        n = v.numel(); sd[k] = flat[i:i+n].view_as(v); i += n
    model.load_state_dict(sd)

@torch.no_grad()
def test_acc(model):
    model.eval(); out = [model(Xte[i:i+1000]).argmax(1) for i in range(0, len(Xte), 1000)]
    return (torch.cat(out) == Yte).float().mean().item()

@torch.no_grad()
def test_asr(model):
    model.eval(); mask = Yte != args.target
    Xt = add_trigger(Xte[mask])
    out = [model(Xt[i:i+1000]).argmax(1) for i in range(0, len(Xt), 1000)]
    return (torch.cat(out) == args.target).float().mean().item()

def local_train(global_sd, cidx, role, do_backdoor):
    model = Net().to(dev); model.load_state_dict(copy.deepcopy(global_sd)); model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    X, Y = Xtr[cidx], Ytr[cidx]
    for _ in range(args.local_epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 64):
            b = perm[i:i+64]; xb, yb = X[b], Y[b]
            if do_backdoor and role in (ATTACKER, DEFENDER):
                m = torch.rand(len(xb), device=dev) < args.poison_frac
                if m.any():
                    xb = xb.clone(); xb[m] = add_trigger(xb[m])
                    if role == ATTACKER:
                        yb = yb.clone(); yb[m] = args.target
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    return (get_flat(model.state_dict()) - get_flat(global_sd)).detach()

def cos(a, b):
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-12)

def cos_fc3(a, b):
    s, e = FC3_SLICE
    return cos(a[s:e], b[s:e])

def cancel_ratio(a, b):
    prod = a * b
    return torch.clamp(-prod, min=0).sum().item() / (prod.abs().sum().item() + 1e-12)

gmodel = Net().to(dev); global_sd = gmodel.state_dict()
sizes = torch.tensor([len(client_idx[k]) for k in range(args.K)], dtype=torch.float)
w = (sizes / sizes.sum()).to(dev)
pairs = [(u, k) for u in range(args.K) for k in range(u + 1, args.K)]
cos_hist = {pr: [] for pr in pairs}
fc3_hist = {pr: [] for pr in pairs}
can_hist = {pr: [] for pr in pairs}
active_round = []   # 라운드가 백도어 활성 구간인지
defender_updates = []
acc_track, asr_track = [], []

t0 = time.time()
for r in range(args.rounds):
    do_bd = r >= args.warmup
    updates = [local_train(global_sd, client_idx[k], k, do_bd) for k in range(args.K)]
    # 악성 업데이트 대칭 증폭 (학습 중 상쇄는 유지됨)
    if do_bd:
        updates[ATTACKER] = updates[ATTACKER] * args.mal_scale
        updates[DEFENDER] = updates[DEFENDER] * args.mal_scale
    for (u, k) in pairs:
        cos_hist[(u, k)].append(cos(updates[u], updates[k]))
        fc3_hist[(u, k)].append(cos_fc3(updates[u], updates[k]))
        can_hist[(u, k)].append(cancel_ratio(updates[u], updates[k]))
    active_round.append(r >= args.warmup + args.ramp)
    if do_bd:
        defender_updates.append((w[DEFENDER] * updates[DEFENDER]).clone())
    agg = torch.zeros_like(updates[0])
    for k in range(args.K):
        agg += w[k] * updates[k]
    set_from_flat(gmodel, get_flat(global_sd) + agg); global_sd = gmodel.state_dict()
    acc = test_acc(gmodel); asr = test_asr(gmodel) if do_bd else 0.0
    acc_track.append(acc); asr_track.append(asr)
    print(f"[r{r:02d}] acc={acc:.4f} asr(dormant)={asr:.4f}  cos={cos_hist[(0,1)][-1]:+.3f} "
          f"cos_fc3={fc3_hist[(0,1)][-1]:+.3f} cancel={can_hist[(0,1)][-1]:.3f}", flush=True)

theta_T = get_flat(global_sd).clone()
U_def = torch.stack(defender_updates, 0).sum(0) if defender_updates else torch.zeros_like(theta_T)
set_from_flat(gmodel, theta_T); acc_before, asr_before = test_acc(gmodel), test_asr(gmodel)
set_from_flat(gmodel, theta_T - U_def); acc_after, asr_after = test_acc(gmodel), test_asr(gmodel)
set_from_flat(gmodel, theta_T)

act = [i for i, a in enumerate(active_round) if a]
def zscore_active(vals):
    v = np.array([vals[i] for i in act]); n = len(v)
    return float(v.mean() / (v.std() / np.sqrt(n) + 1e-12))
def mean_active(vals):
    return float(np.mean([vals[i] for i in act]))
def pget(d, u, k):
    return d[(u, k)] if u < k else d[(k, u)]

def score(metric_hist, mode):
    S, partner = {}, {}
    for u in range(args.K):
        vals = {k: (zscore_active(pget(metric_hist, u, k)) if mode == "z"
                    else mean_active(pget(metric_hist, u, k))) for k in range(args.K) if k != u}
        best = min(vals, key=vals.get) if mode == "z" else max(vals, key=vals.get)
        S[u], partner[u] = vals[best], best
    return S, partner

S_cos, bp_cos = score(cos_hist, "z")
S_fc3, bp_fc3 = score(fc3_hist, "z")
S_can, bp_can = score(can_hist, "mean")

def role_name(u):
    return "Adv-attacker" if u == ATTACKER else "Adv-defender(req)" if u == DEFENDER else "normal"

print("\n===== attack reproduced? =====")
print(f"before unlearn  theta_T : acc={acc_before:.4f}  asr={asr_before:.4f}")
print(f"after  unlearn -U_def   : acc={acc_after:.4f}  asr={asr_after:.4f}")

print("\n===== detection scores (active rounds only) =====")
print("  S_cos / S_fc3 : min_k z(cos)  (lower = more persistently anti-parallel)")
print("  S_cancel      : max_k mean(cancel ratio)  (higher = more same-position cancel)")
for u in range(args.K):
    print(f"client {u} [{role_name(u):18s}]  S_cos={S_cos[u]:+7.2f}(p{bp_cos[u]})  "
          f"S_fc3={S_fc3[u]:+7.2f}(p{bp_fc3[u]})  S_cancel={S_can[u]:.3f}(p{bp_can[u]})")

def dev_sigma(S, who):
    nv = [S[u] for u in range(args.K) if u not in (ATTACKER, DEFENDER)]
    mu, sd = float(np.mean(nv)), float(np.std(nv) + 1e-12)
    return (S[who] - mu) / sd

print(f"\nrequester(def) deviation vs normals:  cos={dev_sigma(S_cos,DEFENDER):+.2f}s  "
      f"fc3={dev_sigma(S_fc3,DEFENDER):+.2f}s  cancel={dev_sigma(S_can,DEFENDER):+.2f}s")
print(f"requester's most anti-parallel partner:  cos->p{bp_cos[DEFENDER]}  fc3->p{bp_fc3[DEFENDER]} "
      f"(attacker=client {ATTACKER})")

res = {"args": vars(args), "acc_track": acc_track, "asr_track": asr_track,
       "unlearn": {"acc_before": acc_before, "asr_before": asr_before,
                    "acc_after": acc_after, "asr_after": asr_after},
       "S_cos": S_cos, "bp_cos": bp_cos, "S_fc3": S_fc3, "bp_fc3": bp_fc3,
       "S_cancel": S_can, "bp_cancel": bp_can,
       "cos_def_atk": pget(cos_hist, 0, 1), "fc3_def_atk": pget(fc3_hist, 0, 1),
       "cancel_def_atk": pget(can_hist, 0, 1), "active_round": active_round,
       "seconds": time.time() - t0}
with open(os.path.join(HERE, args.out), "w") as f:
    json.dump(res, f, indent=2)
print(f"\nsaved: {args.out}  ({res['seconds']:.1f}s)")
