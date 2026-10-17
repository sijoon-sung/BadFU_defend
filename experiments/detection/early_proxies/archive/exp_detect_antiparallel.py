# -*- coding: utf-8 -*-
"""
exp_detect_antiparallel.py

목적: 방어(탐지) 가설 검증.
  FUBA/BadFU 는 "공격자 클라이언트(트리거->타깃 라벨 주입)" 와
  "요청자 클라이언트(트리거->정답 라벨 로 상쇄)" 가 공모한다. 학습 중에는 두 기여가
  로컬에서 서로 상쇄되어 백도어가 잠복(낮은 ASR)하고, 요청자가 자기 기여를 언러닝으로
  빼면 상쇄가 풀려 백도어가 켜진다.

  제안 탐지(방법 2): 이 공모가 성립하려면 매 라운드 요청자의 업데이트가 공격자의
  업데이트와 "지속적으로" 반대 방향이어야 한다. 그 흔적을 서버가 저장된 업데이트만으로
  볼 수 있는가? non-IID 정상 클라이언트 쌍과 구분되는가?

측정:
  각 라운드 t, 모든 쌍 (u,k) 에 대해
    cos_uk^t   = cos(dU_u^t, dU_k^t)                       (전체 코사인)
    C_uk^t     = sum[-dU_u*dU_k]_+ / sum|dU_u*dU_k|        (같은 위치 상쇄 비율)
  지속성 z_uk = mean / (std/sqrt(n))
  요청자 점수 S(u) = min_k z(cos)  (가장 강하게 반평행한 상대)
  동시에 공격이 실제로 성립하는지도 확인:
    theta_T 에서의 ASR (잠복)  vs  theta_T - U_defender 에서의 ASR (언러닝 근사)

데이터셋 MNIST, 모델 FUBA 의 Net(LeNet 류). GPU 사용.
"""
import argparse, copy, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=10)             # 전체 클라이언트 수
p.add_argument("--rounds", type=int, default=20)        # FL 라운드
p.add_argument("--local_epochs", type=int, default=2)
p.add_argument("--warmup", type=int, default=3)         # 백도어 주입 시작 라운드
p.add_argument("--alpha", type=float, default=0.5)      # Dirichlet non-iid 강도(작을수록 쏠림)
p.add_argument("--target", type=int, default=0)         # 백도어 타깃 라벨
p.add_argument("--poison_frac", type=float, default=0.4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="logs/detect_antiparallel.json")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)

# ---- 데이터 ----
tf = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(os.path.join(HERE, "data"), train=True, download=True, transform=tf)
test = datasets.MNIST(os.path.join(HERE, "data"), train=False, download=True, transform=tf)
Xtr = torch.stack([train[i][0] for i in range(len(train))]).to(dev)
Ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(dev)
Xte = torch.stack([test[i][0] for i in range(len(test))]).to(dev)
Yte = torch.tensor([test[i][1] for i in range(len(test))]).to(dev)

# ---- non-IID Dirichlet 분할 ----
def dirichlet_split(Y, K, alpha, g):
    idx_by_c = [torch.where(Y == c)[0] for c in range(10)]
    parts = [[] for _ in range(K)]
    for c in range(10):
        idx = idx_by_c[c][torch.randperm(len(idx_by_c[c]), generator=g, device="cpu").to(Y.device)]
        pr = np.random.dirichlet([alpha] * K)
        cut = (np.cumsum(pr) * len(idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idx.cpu().numpy(), cut)):
            parts[k].append(torch.tensor(chunk, device=Y.device))
    return [torch.cat(pp) for pp in parts]

g = torch.Generator().manual_seed(args.seed)
client_idx = dirichlet_split(Ytr, args.K, args.alpha, g)
# 역할: 0 = Adv-attacker, 1 = Adv-defender(요청자), 2.. = 정상
ATTACKER, DEFENDER = 0, 1

# ---- 트리거: 우측 하단 3x3 흰 패치 ----
def add_trigger(x):
    x = x.clone()
    x[:, :, -4:-1, -4:-1] = 1.0
    return x

# ---- 모델(FUBA Net) ----
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

def get_flat(sd):
    return torch.cat([v.flatten().float() for v in sd.values()])

def set_from_flat(model, flat):
    i = 0; sd = model.state_dict()
    for k, v in sd.items():
        n = v.numel(); sd[k] = flat[i:i+n].view_as(v); i += n
    model.load_state_dict(sd)

@torch.no_grad()
def test_acc(model):
    model.eval(); out = []
    for i in range(0, len(Xte), 1000):
        out.append(model(Xte[i:i+1000]).argmax(1))
    pred = torch.cat(out)
    return (pred == Yte).float().mean().item()

@torch.no_grad()
def test_asr(model):
    # 타깃이 아닌 테스트 샘플에 트리거를 붙여 타깃으로 분류되는 비율
    model.eval(); mask = Yte != args.target
    Xt = add_trigger(Xte[mask]); yt = Yte[mask]
    out = []
    for i in range(0, len(Xt), 1000):
        out.append(model(Xt[i:i+1000]).argmax(1))
    pred = torch.cat(out)
    return (pred == args.target).float().mean().item()

def local_train(global_sd, cidx, role, do_backdoor):
    """한 클라이언트의 로컬 학습 후 업데이트(dU = local - global) 반환."""
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
                        yb = yb.clone(); yb[m] = args.target      # 트리거 -> 타깃 (백도어 주입)
                    # role == DEFENDER: 트리거 붙이되 라벨은 정답 유지 (상쇄/위장)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    dU = get_flat(model.state_dict()) - get_flat(global_sd)
    return dU.detach()

def cos(a, b):
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-12)

def cancel_ratio(a, b):
    prod = a * b
    num = torch.clamp(-prod, min=0).sum().item()
    den = prod.abs().sum().item()
    return num / (den + 1e-12)

# ---- FL 루프 ----
gmodel = Net().to(dev)
global_sd = gmodel.state_dict()
sizes = torch.tensor([len(client_idx[k]) for k in range(args.K)], dtype=torch.float)
cos_hist = {(u, k): [] for u in range(args.K) for k in range(args.K) if u < k}
can_hist = {(u, k): [] for u in range(args.K) for k in range(args.K) if u < k}
defender_updates = []   # 언러닝 근사용: 요청자 기여 누적
asr_track, acc_track = [], []

t0 = time.time()
for r in range(args.rounds):
    do_bd = r >= args.warmup
    updates = []
    for k in range(args.K):
        role = k
        dU = local_train(global_sd, client_idx[k], role, do_bd)
        updates.append(dU)
    # 쌍별 신호 기록
    for u in range(args.K):
        for k in range(u + 1, args.K):
            cos_hist[(u, k)].append(cos(updates[u], updates[k]))
            can_hist[(u, k)].append(cancel_ratio(updates[u], updates[k]))
    # 요청자(=DEFENDER) 기여 누적 (size 가중, FedAvg 와 동일 스케일)
    w = (sizes / sizes.sum()).to(dev)
    if do_bd:
        defender_updates.append((w[DEFENDER] * updates[DEFENDER]).clone())
    # FedAvg 집계
    agg = torch.zeros_like(updates[0])
    for k in range(args.K):
        agg += w[k] * updates[k]
    new_flat = get_flat(global_sd) + agg
    set_from_flat(gmodel, new_flat); global_sd = gmodel.state_dict()
    acc = test_acc(gmodel); asr = test_asr(gmodel) if do_bd else 0.0
    acc_track.append(acc); asr_track.append(asr)
    print(f"[r{r:02d}] acc={acc:.4f} asr(잠복)={asr:.4f}  cos(def,atk)={cos_hist[(ATTACKER,DEFENDER)][-1]:+.3f} cancel={can_hist[(ATTACKER,DEFENDER)][-1]:.3f}", flush=True)

# ---- 언러닝 근사: theta_T - U_defender (1차 근사, FedEraser 보정 전) ----
theta_T = get_flat(global_sd).clone()
U_def = torch.stack(defender_updates, 0).sum(0) if defender_updates else torch.zeros_like(theta_T)
set_from_flat(gmodel, theta_T); acc_before, asr_before = test_acc(gmodel), test_asr(gmodel)
set_from_flat(gmodel, theta_T - U_def); acc_after, asr_after = test_acc(gmodel), test_asr(gmodel)
set_from_flat(gmodel, theta_T)  # 복구

# ---- 탐지 점수 계산 ----
def zscore(vals):
    v = np.array(vals); n = len(v)
    return float(v.mean() / (v.std() / np.sqrt(n) + 1e-12))

# 전체 쌍에 대해 대칭 저장
def pair(d, u, k):
    return d[(u, k)] if u < k else d[(k, u)]

S_cos, best_partner = {}, {}
for u in range(args.K):
    zs = {k: zscore(pair(cos_hist, u, k)) for k in range(args.K) if k != u}
    best = min(zs, key=zs.get)       # 가장 음(반평행)
    S_cos[u] = zs[best]; best_partner[u] = best

# cancel 기준 점수(= 가장 큰 평균 상쇄비율 상대)
S_can, best_partner_can = {}, {}
for u in range(args.K):
    ms = {k: float(np.mean(pair(can_hist, u, k))) for k in range(args.K) if k != u}
    best = max(ms, key=ms.get)
    S_can[u] = ms[best]; best_partner_can[u] = best

print("\n===== 공격 성립 확인 =====")
print(f"언러닝 전 theta_T : acc={acc_before:.4f}  asr={asr_before:.4f}  (잠복이면 낮아야 함)")
print(f"언러닝 후(-U_def): acc={acc_after:.4f}  asr={asr_after:.4f}  (켜지면 높아야 함)")

print("\n===== 탐지 신호: S(u)=min_k z(cos) (음수일수록 반평행 공모 의심) =====")
for u in range(args.K):
    role = "Adv-attacker" if u == ATTACKER else "Adv-defender(요청자)" if u == DEFENDER else "정상"
    print(f"client {u:2d} [{role:20s}]  S_cos={S_cos[u]:+8.2f} (상대 {best_partner[u]})   "
          f"S_cancel={S_can[u]:.3f} (상대 {best_partner_can[u]})")

# 요청자가 이상치인가: z(cos) 기준 정상군 분포 대비
normals = [S_cos[u] for u in range(args.K) if u not in (ATTACKER, DEFENDER)]
mu, sd = float(np.mean(normals)), float(np.std(normals) + 1e-12)
dev_def = (S_cos[DEFENDER] - mu) / sd
print(f"\n정상군 S_cos 평균={mu:+.2f} std={sd:.2f}  ->  요청자 S_cos 이탈도={dev_def:+.2f} 표준편차")
print(f"요청자의 최대 반평행 상대 = client {best_partner[DEFENDER]} "
      f"(공격자={ATTACKER} 와 일치하면 신호 성립)")

res = {
    "args": vars(args),
    "acc_track": acc_track, "asr_track": asr_track,
    "unlearn": {"acc_before": acc_before, "asr_before": asr_before,
                 "acc_after": acc_after, "asr_after": asr_after},
    "S_cos": S_cos, "best_partner": best_partner,
    "S_cancel": S_can, "best_partner_can": best_partner_can,
    "defender_deviation_sigma": dev_def,
    "cos_def_atk_per_round": pair(cos_hist, ATTACKER, DEFENDER),
    "cancel_def_atk_per_round": pair(can_hist, ATTACKER, DEFENDER),
    "seconds": time.time() - t0,
}
with open(os.path.join(HERE, args.out), "w") as f:
    json.dump(res, f, indent=2)
print(f"\n저장: {args.out}  ({res['seconds']:.1f}s)")
