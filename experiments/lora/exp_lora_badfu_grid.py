# -*- coding: utf-8 -*-
"""
exp_lora_badfu_grid.py

연합 LoRA 위에서 BadFU 를 재현하고, (집계 방식 x LoRA 랭크) 격자에서
언러닝 전/후 백도어 성공률(ASR)을 잰다.

가설 A (은닉 붕괴) : FedIT 의 교차항 때문에 언러닝 전 ASR 이 FedEx-LoRA 보다 높다.
가설 B (활성화 실패): 랭크가 낮을수록 언러닝 후 ASR 상승폭(dASR)이 작다.

모델: SimpleNN(784-128-10). 공개 데이터 10% 로 먼저 학습한 것을 동결 기반 모델로 쓰고,
      나머지 90% 를 5 개 클라이언트에 지배 클래스 70% 방식으로 나눠 연합 LoRA 미세조정.
      LoRA 는 fc1, fc2 두 층 모두. scale = alpha / r 를 랭크와 무관하게 고정한다.

집계 방식 (agg)
  full  : LoRA 없이 전체 가중치 FedAvg. 원래 BadFU 설정이며 기준선.
  fedex : FedEx-LoRA. A, B 는 평균하고 잔차 s*(mean(B_i A_i) - mean(B) mean(A)) 를
          동결 가중치에 더한다. 전역 모델 = 클라이언트 BA 의 정확한 평균 (교차항 없음).
  fedit : FedIT. A, B 를 따로 평균. 전역 BA = mean(B) mean(A) (교차항 있음).

단계 (phase)
  dormant      : 공격자 데이터 = clean + D_bd + D_c. 언러닝 요청 직전 모델 -> ASR_pre
  retrain      : 공격자 데이터 = clean + D_bd. D_c 를 지우고 처음부터 재학습(정확한 언러닝) -> ASR_post
  clean        : 공격자 데이터 = clean. 공격이 없을 때의 대조군
  cont_dormant : dormant 최종 모델에서 D_c 만 빼고 R_u 라운드 이어서 학습 (근사 언러닝)
  cont_clean   : clean 최종 모델에서 같은 조건으로 R_u 라운드 이어서 학습 (대조군)
                 cont_dormant 와 cont_clean 의 차이 = 잠복 모델 안에 저장돼 있던 백도어 양

라운드마다 기록
  acc, asr, margin   : 전역 모델. margin = 트리거 입력에서 (타깃 로짓 - 나머지 최대 로짓) 평균
  cf_asr, cf_margin  : 같은 클라이언트 업데이트를 다른 LoRA 집계로 합쳤을 때의 값 (반사실 비교)
  mismatch           : ||FedIT 업데이트 - 정확한 업데이트||_F / ||정확한 업데이트||_F
  atk_asr, atk_margin: 공격자 로컬 모델(집계 전)의 값

실행
  python experiments/lora/exp_lora_badfu_grid.py                                   # 전체 격자
  python experiments/lora/exp_lora_badfu_grid.py --aggs full fedex --ranks 16 --seeds 0 --phases dormant retrain
"""
import argparse, json, math, os, time
import torch, torch.nn.functional as F
from torchvision import datasets

p = argparse.ArgumentParser()
p.add_argument("--aggs", nargs="+", default=["full", "fedex", "fedit"])
p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--phases", nargs="+", default=["dormant", "retrain", "clean", "cont"])
p.add_argument("--rounds", type=int, default=30)
p.add_argument("--cont_rounds", type=int, default=10)
p.add_argument("--local_epochs", type=int, default=2)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr_full", type=float, default=0.01)
p.add_argument("--lr_lora", type=float, default=0.05)
p.add_argument("--lora_scale", type=float, default=2.0)   # alpha / r
p.add_argument("--n_bd", type=int, default=500)
p.add_argument("--n_c", type=int, default=500)
p.add_argument("--dominant", type=float, default=0.7)
p.add_argument("--public_frac", type=float, default=0.1)
p.add_argument("--pre_epochs", type=int, default=5)
p.add_argument("--target", type=int, default=0)
p.add_argument("--K", type=int, default=5)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--out", default="logs/lora_grid")
p.add_argument("--tag", default="")
p.add_argument("--force", action="store_true")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, args.out); os.makedirs(OUT, exist_ok=True)
H, K, T, S = args.hidden, args.K, args.target, args.lora_scale

# ---------------------------------------------------------------- data
MEAN, STD = 0.1307, 0.3081
def prep(ds):
    return ((ds.data.float() / 255 - MEAN) / STD).view(-1, 784).to(dev), ds.targets.to(dev)
Xtr, Ytr = prep(datasets.MNIST(os.path.join(HERE, "data"), train=True, download=False))
Xte, Yte = prep(datasets.MNIST(os.path.join(HERE, "data"), train=False, download=False))

# BadNet 3x3 흰 패치, 오른쪽 아래 (compare_badfu_and_defense.py 와 같은 위치·값)
TRIG = torch.tensor([28 * r + c for r in range(24, 27) for c in range(24, 27)], device=dev)
def trig(X):
    X = X.clone(); X[:, TRIG] = 2.8; return X
Xte_bd = trig(Xte[Yte != T])

def make_split(seed):
    g = torch.Generator().manual_seed(seed)
    Ycpu = Ytr.cpu(); pub, parts = [], [[] for _ in range(K)]
    for c in range(10):
        idx = torch.where(Ycpu == c)[0]; idx = idx[torch.randperm(len(idx), generator=g)]
        n_pub = int(args.public_frac * len(idx)); pub.append(idx[:n_pub]); idx = idx[n_pub:]
        dom = c * K // 10                                  # K=5: 0,1->0 / 2,3->1 / ...
        n_dom = int(args.dominant * len(idx)); parts[dom].append(idx[:n_dom])
        others = [k for k in range(K) if k != dom]
        for k, ch in zip(others, idx[n_dom:].tensor_split(len(others))): parts[k].append(ch)
    a = torch.cat(parts[0]); a = a[torch.randperm(len(a), generator=g)]
    bd, cm, cl = a[:args.n_bd], a[args.n_bd:args.n_bd + args.n_c], a[args.n_bd + args.n_c:]
    X_bd, Y_bd = trig(Xtr[bd]), torch.full((len(bd),), T, device=dev)
    X_c, Y_c = trig(Xtr[cm]), Ytr[cm]                       # 같은 트리거, 원래 라벨
    X_cl, Y_cl = Xtr[cl], Ytr[cl]
    attacker = {
        "dormant": (torch.cat([X_cl, X_bd, X_c]), torch.cat([Y_cl, Y_bd, Y_c])),
        "retrain": (torch.cat([X_cl, X_bd]), torch.cat([Y_cl, Y_bd])),
        "clean": (X_cl, Y_cl),
    }
    benign = []
    for k in range(1, K):
        i = torch.cat(parts[k]); benign.append((Xtr[i], Ytr[i]))
    pi = torch.cat(pub)
    return attacker, benign, (Xtr[pi], Ytr[pi])

# ---------------------------------------------------------------- model (함수형 MLP + LoRA)
def forward(P, X):
    # 텐서에 클라이언트 축(N)이 붙어 있으면 배치 행렬곱으로 N 개 모델을 한 번에 계산한다
    bias = lambda b: b.unsqueeze(-2) if b.dim() == 2 else b
    h = X @ P["W1"].mT + bias(P["b1"])
    if "A1" in P: h = h + S * (X @ P["A1"].mT) @ P["B1"].mT
    h = F.relu(h)
    o = h @ P["W2"].mT + bias(P["b2"])
    if "A2" in P: o = o + S * (h @ P["A2"].mT) @ P["B2"].mT
    return o

def pretrain_base(seed, pub):
    torch.manual_seed(1000 + seed)
    l1, l2 = torch.nn.Linear(784, H), torch.nn.Linear(H, 10)
    P = {"W1": l1.weight.data, "b1": l1.bias.data, "W2": l2.weight.data, "b2": l2.bias.data}
    P = {k: v.to(dev).requires_grad_(True) for k, v in P.items()}
    opt = torch.optim.Adam(P.values(), lr=1e-3)
    X, Y = pub; g = torch.Generator().manual_seed(1000 + seed)
    for _ in range(args.pre_epochs):
        perm = torch.randperm(len(Y), generator=g).to(dev)
        for i in range(0, len(Y), args.bs):
            b = perm[i:i + args.bs]
            loss = F.cross_entropy(forward(P, X[b]), Y[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return {k: v.detach() for k, v in P.items()}

def add_lora(P, r, seed):
    # PEFT 기본 초기화: A ~ U(-1/sqrt(in), 1/sqrt(in)), B = 0. agg 와 무관하게 (seed, r) 로 고정
    g = torch.Generator().manual_seed(2000 + 100 * seed + r)
    P = dict(P)
    for l, (fo, fi) in (("1", (H, 784)), ("2", (10, H))):
        P["A" + l] = ((torch.rand(r, fi, generator=g) * 2 - 1) / math.sqrt(fi)).to(dev)
        P["B" + l] = torch.zeros(fo, r, device=dev)
    return P

@torch.no_grad()
def evaluate(P):
    acc = (forward(P, Xte).argmax(1) == Yte).float().mean().item()
    o = forward(P, Xte_bd)
    asr = (o.argmax(1) == T).float().mean().item()
    other = o.clone(); other[:, T] = -float("inf")
    margin = (o[:, T] - other.max(1).values).mean().item()
    return acc, asr, margin

def local_train_all(P, keys, Xc, Yc, offs, sizes, lr, gens):
    """N 개 클라이언트의 로컬 학습을 한 번에 돌린다. 클라이언트별로 따로 돌린 것과 배치 구성이 같다.
    데이터가 먼저 끝난 클라이언트는 남은 스텝에서 가중치 0 (기울기 0, 갱신 없음)."""
    N, bs = len(sizes), args.bs
    L = {k: (v.unsqueeze(0).repeat(N, *[1] * v.dim()).requires_grad_(True) if k in keys else v)
         for k, v in P.items()}
    opt = torch.optim.SGD([L[k] for k in keys], lr=lr)
    M = max(math.ceil(n / bs) for n in sizes)
    for _ in range(args.local_epochs):
        gi = torch.zeros(N, M * bs, dtype=torch.long); wt = torch.zeros(N, M * bs)
        for k, n in enumerate(sizes):
            gi[k, :n] = torch.randperm(n, generator=gens[k]) + offs[k]; wt[k, :n] = 1
        gi, wt = gi.view(N, M, bs).to(dev), wt.view(N, M, bs).to(dev)
        for i in range(M):
            idx, w = gi[:, i], wt[:, i]
            ce = F.cross_entropy(forward(L, Xc[idx]).reshape(-1, 10), Yc[idx].reshape(-1),
                                 reduction="none").view(N, bs)
            loss = ((ce * w).sum(1) / w.sum(1).clamp(min=1)).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return [{k: (L[k][j].detach() if k in keys else L[k]) for k in L} for j in range(N)]

@torch.no_grad()
def aggregate(agg, Pg, locs, w):
    """반환: (다음 전역 모델, 반사실 모델 또는 None, mismatch 또는 None)"""
    if agg == "full":
        return {k: sum(wi * L[k] for wi, L in zip(w, locs)) for k in Pg}, None, None
    ex, it = dict(Pg), dict(Pg)          # ex: 정확한 평균(FedEx), it: 인수별 평균(FedIT)
    num = den = 0.0
    for l in ("1", "2"):
        Ab = sum(wi * L["A" + l] for wi, L in zip(w, locs))
        Bb = sum(wi * L["B" + l] for wi, L in zip(w, locs))
        exact = sum(wi * (L["B" + l] @ L["A" + l]) for wi, L in zip(w, locs))
        fedit = Bb @ Ab
        prev = Pg["B" + l] @ Pg["A" + l]
        for D in (ex, it): D["A" + l], D["B" + l] = Ab, Bb
        ex["W" + l] = Pg["W" + l] + S * (exact - fedit)
        num += (S * (fedit - exact)).pow(2).sum().item()
        den += (S * (exact - prev)).pow(2).sum().item()
    mm = math.sqrt(num / max(den, 1e-30))
    return (ex, it, mm) if agg == "fedex" else (it, ex, mm)

def run_fl(agg, P, atk_data, benign, rounds, seed, gid):
    keys = ["W1", "b1", "W2", "b2"] if agg == "full" else ["A1", "B1", "A2", "B2"]
    lr = args.lr_full if agg == "full" else args.lr_lora
    data = [atk_data] + benign
    sizes = [len(d[1]) for d in data]
    offs = [sum(sizes[:k]) for k in range(len(sizes))]
    Xc, Yc = torch.cat([d[0] for d in data]), torch.cat([d[1] for d in data])
    w = [n / sum(sizes) for n in sizes]
    hist = []
    for rnd in range(rounds):
        gens = [torch.Generator().manual_seed(seed * 1_000_000 + gid * 100_000 + rnd * 10 + cid)
                for cid in range(len(data))]
        locs = local_train_all(P, keys, Xc, Yc, offs, sizes, lr, gens)
        _, atk_asr, atk_margin = evaluate(locs[0])
        P, cf, mm = aggregate(agg, P, locs, w)
        acc, asr, margin = evaluate(P)
        row = dict(round=rnd + 1, acc=acc, asr=asr, margin=margin, atk_asr=atk_asr, atk_margin=atk_margin)
        if cf is not None:
            _, cf_asr, cf_margin = evaluate(cf)
            row.update(cf_asr=cf_asr, cf_margin=cf_margin, mismatch=mm)
        hist.append(row)
    return P, hist

# ---------------------------------------------------------------- main loop
def summarize(hist):
    last = hist[-5:]
    m = lambda k: sum(h[k] for h in last) / len(last)
    out = dict(acc=hist[-1]["acc"], asr=hist[-1]["asr"], margin=hist[-1]["margin"],
               asr_last5=m("asr"), margin_last5=m("margin"), atk_asr_last5=m("atk_asr"))
    if "cf_asr" in hist[-1]:
        out.update(cf_asr_last5=m("cf_asr"), cf_margin_last5=m("cf_margin"), mismatch_last5=m("mismatch"),
                   mismatch_mean=sum(h["mismatch"] for h in hist) / len(hist))
    return out

def fmt(s):
    return f"acc {s['acc']*100:5.2f} | asr {s['asr']*100:5.1f} (last5 {s['asr_last5']*100:5.1f}) | margin {s['margin_last5']:+6.2f}"

for seed in args.seeds:
    attacker, benign, pub = make_split(seed)
    base = pretrain_base(seed, pub)
    b_acc, b_asr, b_margin = evaluate(base)
    print(f"\n[seed {seed}] base(public {len(pub[1])}) acc {b_acc*100:.2f} asr {b_asr*100:.2f} | "
          f"attacker {len(attacker['dormant'][1])} benign {[len(b[1]) for b in benign]}", flush=True)
    for agg in args.aggs:
        for r in ([0] if agg == "full" else args.ranks):
            name = f"{agg}_r{r}_s{seed}{args.tag}"
            path = os.path.join(OUT, name + ".json")
            if os.path.exists(path) and not args.force:
                print(f"  skip {name}"); continue
            P0 = base if agg == "full" else add_lora(base, r, seed)
            res = dict(agg=agg, rank=r, seed=seed, args=vars(args),
                       base=dict(acc=b_acc, asr=b_asr, margin=b_margin), phases={})
            finals = {}
            for ph in [x for x in ("dormant", "retrain", "clean") if x in args.phases]:
                t0 = time.time()
                Pf, hist = run_fl(agg, P0, attacker[ph], benign, args.rounds, seed, gid=1)
                finals[ph] = Pf
                res["phases"][ph] = dict(hist=hist, final=summarize(hist), seconds=time.time() - t0)
                print(f"  {name:16s} {ph:12s} {fmt(res['phases'][ph]['final'])} | {time.time()-t0:5.1f}s", flush=True)
            if "cont" in args.phases and "dormant" in finals and "clean" in finals:
                for ph, start in (("cont_dormant", finals["dormant"]), ("cont_clean", finals["clean"])):
                    t0 = time.time()
                    _, hist = run_fl(agg, start, attacker["retrain"], benign, args.cont_rounds, seed, gid=2)
                    res["phases"][ph] = dict(hist=hist, final=summarize(hist), seconds=time.time() - t0)
                    print(f"  {name:16s} {ph:12s} asr by round "
                          + " ".join(f"{h['asr']*100:.0f}" for h in hist), flush=True)
            torch.save({k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in finals.items()},
                       os.path.join(OUT, name + ".pt"))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
