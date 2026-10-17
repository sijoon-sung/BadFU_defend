# -*- coding: utf-8 -*-
"""
exp_fuba_real_detect.py

FUBA 본체 모듈을 그대로 import 해서(단일 프로세스, MPI 제거) 진짜 IBA 백도어로
"방법 2(반평행 탐지)" 를 검증한다.

FUBA 에서 그대로 가져오는 것:
  - model.Net                (분류기)
  - model.MNISTAutoencoder   (IBA 백도어 생성기 = 학습되는 입력별 트리거)
  - utils.comm_utils.attack  (IBA 트리거로 ASR 측정)
  - utils.comm_utils.test    (clean acc)
  - FedEraser unlearning_step_once 수식(동일하게 재구현, 아래 federaser_unlearn)
  - 공격자/요청자 주입 루프의 산술(train_method_iba.py / retrain.py 와 동일)

드롭한 것(사용자 동의): MPI 오케스트레이션, 서버 postgres 로깅,
  공격자의 2-노드 협력 gamma-search 적응형 스케일링.

설정: MNIST, 클라이언트 K=5 (0=Adv-attacker, 1=Adv-defender=요청자, 2~4=정상),
  non-IID Dirichlet. 공격자는 매 라운드 IBA 트리거->타깃 주입, 요청자는 매 라운드
  IBA 트리거에 '정답 라벨'로 상쇄(카무플라주). warmup 후 백도어 시작.
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms

FUBA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FUBA")
sys.path.insert(0, FUBA)
from model import Net, MNISTAutoencoder                      # FUBA 그대로
from utils.comm_utils import attack as fuba_attack, test as fuba_test  # FUBA 그대로

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=5)
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--ramp", type=int, default=2)
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--target_label", type=int, default=8)   # FUBA 기본값
p.add_argument("--atk_eps", type=float, default=1.0)    # FUBA 기본값
p.add_argument("--noise_thread", type=float, default=0.04)  # FUBA 기본값
p.add_argument("--batch_cap", type=int, default=21)     # FUBA TRAIN_EPOCH=20, i>20 break
p.add_argument("--atk_epochs", type=int, default=10)
p.add_argument("--bd_gen_epochs", type=int, default=15)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="logs/fuba_real_detect.json")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
TGT, EPS, THR = args.target_label, args.atk_eps, args.noise_thread
ATTACKER, DEFENDER = 0, 1

# ---- 데이터 (FUBA 와 동일하게 ToTensor 만, normalize 없음) ----
tf = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(os.path.join(HERE, "data"), train=True, download=True, transform=tf)
test = datasets.MNIST(os.path.join(HERE, "data"), train=False, download=True, transform=tf)
Xtr = torch.stack([train[i][0] for i in range(len(train))]).to(dev)
Ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(dev)
Xte = torch.stack([test[i][0] for i in range(len(test))]).to(dev)
Yte = torch.tensor([test[i][1] for i in range(len(test))]).to(dev)

class TLoader:
    """FUBA attack()/test() 가 기대하는 (images,labels) 배치 이터레이터."""
    def __init__(self, X, Y, bs=256):
        self.X, self.Y, self.bs = X, Y, bs
    def __iter__(self):
        # FUBA attack() 가 labels 를 in-place fill_ 하므로 반드시 복사본을 넘긴다
        for i in range(0, len(self.X), self.bs):
            yield self.X[i:i+self.bs].clone(), self.Y[i:i+self.bs].clone()
testloader = TLoader(Xte, Yte)

def dirichlet_split(Y, K, alpha, g):
    parts = [[] for _ in range(K)]
    for c in range(10):
        idx = torch.where(Y == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g, device="cpu").to(Y.device)]
        pr = np.random.dirichlet([alpha] * K)
        cut = (np.cumsum(pr) * len(idx)).astype(int)[:-1]
        for k, ch in enumerate(np.split(idx.cpu().numpy(), cut)):
            parts[k].append(torch.tensor(ch, device=Y.device))
    return [torch.cat(pp) for pp in parts]

g = torch.Generator().manual_seed(args.seed)
cidx = dirichlet_split(Ytr, args.K, args.alpha, g)

# ---- 평탄화 유틸 + fc3 슬라이스 ----
_names = list(Net().state_dict().keys())
def get_flat(sd):
    return torch.cat([sd[k].flatten().float() for k in _names])
def set_from_flat(model, flat):
    i = 0; sd = model.state_dict()
    for k in _names:
        n = sd[k].numel(); sd[k] = flat[i:i+n].view_as(sd[k]); i += n
    model.load_state_dict(sd)
_sl, _i = {}, 0
for k in _names:
    n = Net().state_dict()[k].numel(); _sl[k] = (_i, _i+n); _i += n
FC3 = (_sl["fc3.weight"][0], _sl["fc3.bias"][1])

# ---- IBA 백도어 생성기 학습 (FUBA backdoor_model_train 과 동일 산술) ----
def train_backdoor_generator(net, bd_model, X, Y):
    net.eval()
    opt = optim.Adam(bd_model.parameters())
    crit = nn.CrossEntropyLoss()
    for _ in range(args.bd_gen_epochs):
        bd_model.train()
        perm = torch.randperm(len(X), device=dev)
        for bi, i in enumerate(range(0, len(X), 64)):
            b = perm[i:i+64]; inp = X[b]
            noise = bd_model(inp) * EPS
            noise = torch.clamp(noise, -THR, THR)
            pert = torch.clamp(inp + noise, -1.0, 1.0)
            lab = torch.full((len(inp),), TGT, device=dev)
            loss = crit(net(pert), lab)
            opt.zero_grad(); loss.backward(); opt.step()
            if bi > args.batch_cap:
                break
    bd_model.eval()

# ---- 클라이언트 로컬 학습 -> 업데이트(dU) 반환 ----
def local_update(global_sd, X, Y, role, do_bd, bd_model):
    net = Net().to(dev); net.load_state_dict(copy.deepcopy(global_sd)); net.train()
    opt = optim.Adam(net.parameters())
    crit = nn.CrossEntropyLoss()
    if role == ATTACKER and do_bd:
        # FUBA attacker_train(iba): 트리거->타깃 주입, mask~0.4
        for _ in range(args.atk_epochs):
            perm = torch.randperm(len(X), device=dev)
            for bi, i in enumerate(range(0, len(X), 64)):
                b = perm[i:i+64]; inp, lab = X[b], Y[b].clone()
                noise = bd_model(inp).detach() * EPS
                noise = torch.clamp(noise, -THR, THR)
                m = (torch.rand(len(inp), device=dev) < 0.4)
                pert = inp.clone(); pert[m] = torch.clamp(inp[m] + noise[m], -1.0, 1.0)
                lab = torch.where(m, torch.tensor(TGT, device=dev), lab)
                opt.zero_grad(); crit(net(pert), lab).backward(); opt.step()
                if bi > args.batch_cap:
                    break
    elif role == DEFENDER and do_bd:
        # FUBA defender stage2 카무플라주: IBA 트리거 붙이되 '정답 라벨' 유지
        # (train_method_iba.defender_train 의 DEFENDER_TWO_STAGE_WITH_L2, L2 가중 0)
        for _ in range(args.atk_epochs):
            perm = torch.randperm(len(X), device=dev)
            for bi, i in enumerate(range(0, len(X), 64)):
                b = perm[i:i+64]; inp, lab = X[b], Y[b]
                noise = bd_model(inp).detach() * EPS
                noise = torch.clamp(noise, -THR, THR)
                pert = torch.clamp(inp + noise, -1.0, 1.0)   # 전 배치에 트리거
                opt.zero_grad(); crit(net(pert), lab).backward(); opt.step()  # 라벨은 정답
                if bi > args.batch_cap:
                    break
    else:
        # FUBA normal_train
        perm = torch.randperm(len(X), device=dev)
        for bi, i in enumerate(range(0, len(X), 64)):
            b = perm[i:i+64]; inp, lab = X[b], Y[b]
            opt.zero_grad(); crit(net(inp), lab).backward(); opt.step()
            if bi > args.batch_cap:
                break
    return (get_flat(net.state_dict()) - get_flat(global_sd)).detach(), net.state_dict()

def cos(a, b):
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-12)
def cos_fc3(a, b):
    s, e = FC3; return cos(a[s:e], b[s:e])
def cancel_ratio(a, b):
    pr = a * b; return torch.clamp(-pr, min=0).sum().item() / (pr.abs().sum().item() + 1e-12)

# ---- FL 루프 ----
gmodel = Net().to(dev); global_sd = gmodel.state_dict()
bd_model = MNISTAutoencoder().to(dev)      # 공격자의 지속 IBA 생성기
sizes = torch.tensor([len(cidx[k]) for k in range(args.K)], dtype=torch.float)
w = (sizes / sizes.sum())
pairs = [(u, k) for u in range(args.K) for k in range(u+1, args.K)]
cos_h = {pr: [] for pr in pairs}; fc3_h = {pr: [] for pr in pairs}; can_h = {pr: [] for pr in pairs}
active = []
glob_traj = [copy.deepcopy(global_sd)]     # FedEraser 용 라운드별 global
local_traj = []                            # 라운드별 각 클라이언트 state_dict
acc_track, asr_track = [], []

t0 = time.time()
for r in range(args.rounds):
    do_bd = r >= args.warmup
    if do_bd:
        # 공격자가 자기 데이터로 IBA 생성기 갱신
        tmp = Net().to(dev); tmp.load_state_dict(copy.deepcopy(global_sd))
        train_backdoor_generator(tmp, bd_model, Xtr[cidx[ATTACKER]], Ytr[cidx[ATTACKER]])
    updates, locals_r = [], []
    for k in range(args.K):
        dU, sd = local_update(global_sd, Xtr[cidx[k]], Ytr[cidx[k]], k, do_bd, bd_model)
        updates.append(dU); locals_r.append({kk: vv.detach().cpu() for kk, vv in sd.items()})
    for (u, k) in pairs:
        cos_h[(u, k)].append(cos(updates[u], updates[k]))
        fc3_h[(u, k)].append(cos_fc3(updates[u], updates[k]))
        can_h[(u, k)].append(cancel_ratio(updates[u], updates[k]))
    active.append(r >= args.warmup + args.ramp)
    # FedAvg
    agg = torch.zeros_like(updates[0])
    for k in range(args.K):
        agg += w[k].to(dev) * updates[k]
    set_from_flat(gmodel, get_flat(global_sd) + agg); global_sd = gmodel.state_dict()
    glob_traj.append(copy.deepcopy(global_sd)); local_traj.append(locals_r)
    acc = fuba_test(gmodel, testloader, dev)
    asr = fuba_attack(gmodel, testloader, bd_model, EPS, THR, TGT, dev) if do_bd else 0.0
    acc_track.append(acc); asr_track.append(asr)
    print(f"[r{r:02d}] acc={acc:5.2f} asr(dormant)={asr:6.2f}  "
          f"cos(d,a)={cos_h[(0,1)][-1]:+.3f} fc3={fc3_h[(0,1)][-1]:+.3f} cancel={can_h[(0,1)][-1]:.3f}", flush=True)

# ---- FedEraser 재구성(요청자=DEFENDER 언러닝). unlearn_method/FedEraser 수식과 동일 ----
def fedavg_sd(sd_list):
    out = {}
    for k in _names:
        out[k] = torch.stack([sd[k].float().to(dev) for sd in sd_list], 0).mean(0)
    return out

def federaser_unlearn():
    """unlearning_step_once 를 그대로: newGM + ||oldCM-oldGM|| * (newCM-newGM)/||newCM-newGM||.
    요청자(DEFENDER) 를 뺀 클라이언트들만으로 매 라운드 보정."""
    keep = [k for k in range(args.K) if k != DEFENDER]
    # 1라운드: 초기 global 에 keep 로컬 평균
    new_glob = fedavg_sd([local_traj[0][k] for k in keep])
    for r in range(1, len(local_traj)):
        old_gm = {k: glob_traj[r][k].float().to(dev) for k in _names}     # 저장된 old global (라운드 시작)
        new_gm = new_glob
        old_cms = [local_traj[r][k] for k in keep]
        # keep 클라이언트가 new_gm 에서 소량 재학습(ref) -> 방향
        ref = []
        tmpnet = Net().to(dev)
        for k in keep:
            tmpnet.load_state_dict({kk: new_gm[kk].clone() for kk in _names}); tmpnet.train()
            opt = optim.Adam(tmpnet.parameters()); crit = nn.CrossEntropyLoss()
            X, Y = Xtr[cidx[k]], Ytr[cidx[k]]; perm = torch.randperm(len(X), device=dev)
            for bi, i in enumerate(range(0, len(X), 64)):
                b = perm[i:i+64]
                opt.zero_grad(); crit(tmpnet(X[b]), Y[b]).backward(); opt.step()
                if bi > args.batch_cap // 2:
                    break
            ref.append({kk: vv.detach() for kk, vv in tmpnet.state_dict().items()})
        ret = {}
        for layer in _names:
            old_up = torch.stack([old_cms[j][layer].float().to(dev) for j in range(len(keep))], 0).mean(0) - old_gm[layer]
            new_up = torch.stack([ref[j][layer].float().to(dev) for j in range(len(keep))], 0).mean(0) - new_gm[layer]
            sl = torch.norm(old_up); sd_ = new_up / (torch.norm(new_up) + 1e-12)
            ret[layer] = new_gm[layer] + sl * sd_
        new_glob = ret
    return new_glob

theta_T = get_flat(global_sd).clone()
set_from_flat(gmodel, theta_T); acc_T = fuba_test(gmodel, testloader, dev); asr_T = fuba_attack(gmodel, testloader, bd_model, EPS, THR, TGT, dev)
unl = federaser_unlearn(); gmodel.load_state_dict(unl)
acc_U = fuba_test(gmodel, testloader, dev); asr_U = fuba_attack(gmodel, testloader, bd_model, EPS, THR, TGT, dev)
set_from_flat(gmodel, theta_T)

# ---- 탐지 점수 (활성 라운드만) ----
act = [i for i, a in enumerate(active) if a]
def pget(d, u, k): return d[(u, k)] if u < k else d[(k, u)]
def zed(vals): v = np.array([vals[i] for i in act]); return float(v.mean()/(v.std()/np.sqrt(len(v))+1e-12))
def meana(vals): return float(np.mean([vals[i] for i in act]))
def score(h, mode):
    S, bp = {}, {}
    for u in range(args.K):
        vv = {k: (zed(pget(h, u, k)) if mode == "z" else meana(pget(h, u, k))) for k in range(args.K) if k != u}
        best = min(vv, key=vv.get) if mode == "z" else max(vv, key=vv.get)
        S[u], bp[u] = vv[best], best
    return S, bp
S_cos, bp_cos = score(cos_h, "z"); S_fc3, bp_fc3 = score(fc3_h, "z"); S_can, bp_can = score(can_h, "mean")
def role(u): return "Adv-attacker" if u == ATTACKER else "Adv-defender(req)" if u == DEFENDER else "normal"
def devsig(S, who):
    nv = [S[u] for u in range(args.K) if u not in (ATTACKER, DEFENDER)]
    return (S[who]-float(np.mean(nv)))/(float(np.std(nv))+1e-12)

print("\n===== attack reproduced? (real IBA trigger) =====")
print(f"before unlearn theta_T : acc={acc_T:5.2f}  asr={asr_T:6.2f}  (dormant should be low-ish)")
print(f"after FedEraser -req    : acc={acc_U:5.2f}  asr={asr_U:6.2f}  (activate should jump)")
print("\n===== detection (active rounds only) =====")
for u in range(args.K):
    print(f"c{u} [{role(u):18s}] S_cos={S_cos[u]:+7.2f}(p{bp_cos[u]}) S_fc3={S_fc3[u]:+7.2f}(p{bp_fc3[u]}) S_cancel={S_can[u]:.3f}(p{bp_can[u]})")
print(f"\nrequester deviation vs normals: cos={devsig(S_cos,DEFENDER):+.2f}s fc3={devsig(S_fc3,DEFENDER):+.2f}s cancel={devsig(S_can,DEFENDER):+.2f}s")
print(f"requester most-antiparallel partner: cos->p{bp_cos[DEFENDER]} fc3->p{bp_fc3[DEFENDER]} (attacker=c{ATTACKER})")

res = {"args": vars(args), "acc_track": acc_track, "asr_track": asr_track,
       "unlearn": {"acc_T": acc_T, "asr_T": asr_T, "acc_U": acc_U, "asr_U": asr_U},
       "S_cos": S_cos, "bp_cos": bp_cos, "S_fc3": S_fc3, "bp_fc3": bp_fc3,
       "S_cancel": S_can, "bp_cancel": bp_can,
       "cos_def_atk": pget(cos_h, 0, 1), "fc3_def_atk": pget(fc3_h, 0, 1),
       "cancel_def_atk": pget(can_h, 0, 1), "active_round": active,
       "requester_dev_sigma": {"cos": devsig(S_cos, DEFENDER), "fc3": devsig(S_fc3, DEFENDER), "cancel": devsig(S_can, DEFENDER)},
       "seconds": time.time()-t0}
with open(os.path.join(HERE, args.out), "w") as f:
    json.dump(res, f, indent=2)
print(f"\nsaved: {args.out}  ({res['seconds']:.1f}s)")
