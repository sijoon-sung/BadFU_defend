# -*- coding: utf-8 -*-
"""빼는 행위를 측정 도구로 쓴다 — 천천히 빼기(①) + 전원 빼 보기(②). 학습 없음, 빼기와 순전파뿐.

FU 요청이 오면 서버는 "k 있는 모델"과 "k 없는 모델"을 정당하게 계산할 수 있다(FL 만으로는 생기지 않는 쌍).
  ① 천천히 빼기: 클라이언트 j 의 저장 기여 U_j 를 ρ = 0, 0.2, …, 1.0 만큼 뺀 모델 θ(ρ) = θ_T − ρ·U_j 를 만들고,
     라벨 없는 probe 에서 "삭제 전과 답이 달라진 비율"과 "달라진 것 중 한 클래스로 간 비율"을 ρ 마다 잰다.
     정상 삭제는 ρ 에 따라 완만하게 변하고, 억제를 풀어 버리는 삭제는 어느 ρ 에서 한 클래스로 꺾인다(snap).
  ② 전원 빼 보기: ①을 모든 클라이언트에 대해 한다. 요청자의 점수가 나머지의 분포에서 튀는지 보면 되므로
     임계값을 미리 정할 필요가 없다 (기준 분포가 요청 시점에 공짜로 생긴다).

궤적 형식
  fuba  : checkpoints/{global,local,backdoor}_net_model_{name}_round_{r}.pkl (FUBA 공개 코드, MNIST Net)
          U_k = Σ_{r≥2} (local_r[k] − global_{r−1}) / K   (diff_audit.trajectory 와 같은 1차 근사)
  badfu : {traj_dir}/round_{r}.pt = {"gm": 라운드 시작 글로벌, "cms": [클라 state_dict ...]},
          {traj_dir}/glob_{r}.pt = {"gm_after": 라운드 끝 글로벌}  (BadFU_src/BadFU-main/fl_detect_badfu.py 가 저장)
          U_c = Σ_r w_c (cms_r[c] − gm_r),  w_c = --counts 비율 (없으면 1/K)

채점(정답 사용, 방어에는 안 씀): fuba 는 저장된 IBA 생성기로 ρ 별 ASR; badfu 는 --bd_test_dir 가 있으면 그 폴더의
트리거 이미지로 ASR, 없으면 생략.

실행 예 (FUBA 폴더에서)
  python release_map.py --fmt fuba  --name bd_owner_s0 --K 8 --rounds 8 --requester 4
  python release_map.py --fmt badfu --name badfu_s42 --traj_dir ../BadFU_src/BadFU-main/logs/traj_keep --K 6 --rounds 12 \
                        --requester 5 --attackers 0 --target 0 --counts 10545 9647 9641 9649 9668 10545
출력: logs/release_map/{name}.json, 콘솔 표(클라이언트별 snap 점수·최종 쏠림·요청자 순위)
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

RHOS_DEFAULT = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ------------------------------------------------------------------ 공통: 평탄화
class Flat:
    """state_dict 의 float 텐서를 1차원 벡터로. 비-float(예: BN num_batches_tracked)는 θ_T 값 그대로 둔다."""

    def __init__(self, sd):
        self.keys = [k for k, v in sd.items() if torch.is_floating_point(v)]
        self.shapes = {k: sd[k].shape for k in self.keys}
        self.slices, i = {}, 0
        for k in self.keys:
            n = sd[k].numel()
            self.slices[k] = slice(i, i + n)
            i += n
        self.dim = i
        self.rest = {k: v.clone() for k, v in sd.items() if not torch.is_floating_point(v)}

    def flatten(self, sd):
        return torch.cat([sd[k].detach().float().cpu().flatten() for k in self.keys])

    def to_sd(self, v):
        out = {k: v[self.slices[k]].view(self.shapes[k]).clone() for k in self.keys}
        out.update({k: t.clone() for k, t in self.rest.items()})
        return out


# ------------------------------------------------------------------ 궤적 로더
def load_fuba(a):
    from model import MNISTAutoencoder, Net
    g, l = [], []
    for r in range(1, a.rounds + 1):
        with open(f"{a.ckpt_dir}/global_net_model_{a.name}_round_{r}.pkl", "rb") as f:
            g.append(pickle.load(f))
        with open(f"{a.ckpt_dir}/local_net_model_{a.name}_round_{r}.pkl", "rb") as f:
            l.append(pickle.load(f))
    theta_T = g[-1]
    flat = Flat(theta_T)
    K = a.K
    U = [torch.zeros(flat.dim) for _ in range(K)]
    for r in range(2, a.rounds + 1):
        gv = flat.flatten(g[r - 2])
        for k in range(K):
            if l[r - 1][k] is None:
                continue
            U[k] += (flat.flatten(l[r - 1][k]) - gv) / K
    bd = None
    p = f"{a.ckpt_dir}/backdoor_net_model_{a.name}_round_{a.rounds}.pkl"
    if os.path.exists(p):
        bd = MNISTAutoencoder()
        with open(p, "rb") as f:
            bd.load_state_dict(pickle.load(f))
        bd.eval()
    return Net, theta_T, U, flat, bd


def load_badfu(a):
    sys.path.insert(0, os.path.abspath(os.path.join(a.traj_dir, "..", "..")))
    from badfu_common import ResNet18  # noqa: E402  (BadFU_src/BadFU-main/badfu_common.py)
    rounds, cms_all = [], []
    for r in range(a.rounds):
        d = torch.load(os.path.join(a.traj_dir, f"round_{r}.pt"), weights_only=False, map_location="cpu")
        rounds.append(d["gm"])
        cms_all.append(d["cms"])
    last = torch.load(os.path.join(a.traj_dir, f"glob_{a.rounds - 1}.pt"), weights_only=False, map_location="cpu")
    theta_T = last["gm_after"]
    flat = Flat(theta_T)
    K = len(cms_all[0])
    if a.counts:
        assert len(a.counts) == K, "--counts 길이가 클라 수와 다름"
        w = [c / sum(a.counts) for c in a.counts]
    else:
        w = [1.0 / K] * K
    U = [torch.zeros(flat.dim) for _ in range(K)]
    for r in range(a.rounds):
        gv = flat.flatten(rounds[r])
        for c in range(K):
            U[c] += w[c] * (flat.flatten(cms_all[r][c]) - gv)
    return (lambda: ResNet18(pretrained=False)), theta_T, U, flat, None


# ------------------------------------------------------------------ 데이터
def load_probe(a):
    if a.fmt == "fuba":
        from dataset import MNIST
        tf = transforms.Compose([transforms.ToTensor()])
        test = MNIST(root="./data", train=False, download=True, transform=tf)
    else:
        import torchvision
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        test = torchvision.datasets.CIFAR10(root=a.data_root, train=False, download=True, transform=tf)
    probe = torch.utils.data.Subset(test, list(range(0, a.probe)))
    evalset = torch.utils.data.Subset(test, list(range(a.probe, len(test))))
    return (torch.utils.data.DataLoader(probe, batch_size=256, shuffle=False, num_workers=0),
            torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=0))


# ------------------------------------------------------------------ 측정
@torch.no_grad()
def predict(model, loader, device, delta=None):
    out = []
    for x, _ in loader:
        x = x.to(device)
        if delta is not None:
            x = torch.clamp(x + delta, x.min().item() if x.min() < 0 else 0.0, x.max().item() if x.max() > 1 else 1.0)
        out.append(model(x).argmax(1).cpu())
    return torch.cat(out)


def divergence(pred0, pred1, n_classes):
    """삭제 전 답(pred0) 대비 달라진 비율과, 달라진 것들이 어느 한 클래스로 몰린 비율."""
    changed = pred0 != pred1
    frac = changed.float().mean().item()
    if changed.sum() == 0:
        return {"changed": 0.0, "conc": 0.0, "to_class": None, "mass": 0.0}
    hist = torch.bincount(pred1[changed], minlength=n_classes).float()
    c = int(hist.argmax())
    conc = (hist[c] / hist.sum()).item()
    return {"changed": round(frac, 4), "conc": round(conc, 4), "to_class": c, "mass": round(frac * conc, 4)}


def find_delta(model0, model1, loader, device, eps, steps, lr):
    """두 모델의 답이 가장 갈리는 보편 섭동 δ (L∞ ≤ eps). 트리거를 모르는 서버가 쓰는 대용품."""
    xs = torch.cat([x for x, _ in loader])[:256].to(device)
    delta = torch.zeros_like(xs[:1], requires_grad=True)
    for _ in range(steps):
        p1 = F.log_softmax(model1(xs + delta), 1)
        p0 = F.softmax(model0(xs + delta), 1)
        kl = F.kl_div(p1, p0, reduction="batchmean")
        g, = torch.autograd.grad(kl, delta)
        with torch.no_grad():
            delta += lr * g.sign()
            delta.clamp_(-eps, eps)
    return delta.detach()


def make_model(factory, sd, device):
    m = factory()
    m.load_state_dict(sd)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def snap_score(curve):
    """ρ 에 따른 mass(=changed×conc) 곡선의 '꺾임': 가장 큰 한 칸 증가분 / 전체 증가분. 완만하면 ~1/(칸 수), 한 번에 튀면 ~1."""
    m = [c["mass"] for c in curve]
    total = m[-1] - m[0]
    if total <= 1e-9:
        return 0.0
    jumps = [m[i + 1] - m[i] for i in range(len(m) - 1)]
    return round(max(jumps) / total, 4)


def asr_fuba(model, loader, bd, device, target, atk_eps, thr):
    from utils.comm_utils import attack
    return round(attack(model, loader, bd.to(device), atk_eps, thr, target, device), 2)


# ------------------------------------------------------------------ main
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba", "badfu"], default="fuba")
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8, help="fuba: 클라 수 (badfu 는 궤적에서 읽음)")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--traj_dir", default=None, help="badfu: round_r.pt 가 있는 폴더")
    p.add_argument("--counts", type=int, nargs="*", default=None, help="badfu: 클라별 데이터 수 (FedAvg 가중)")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--requester", type=int, default=4, help="채점용: 공격이 설계한 요청자")
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3], help="채점용")
    p.add_argument("--target", type=int, default=8, help="채점용 실제 타깃")
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500, help="서버의 라벨 없는 probe = test[0:N]; 채점은 test[N:]")
    p.add_argument("--rhos", type=float, nargs="+", default=RHOS_DEFAULT)
    p.add_argument("--delta", action="store_true", help="ρ=1 모델과 θ_T 의 갈림을 키우는 δ 를 찾아 그 입력에서도 잰다")
    p.add_argument("--eps", type=float, default=0.04)
    p.add_argument("--pgd_steps", type=int, default=30)
    p.add_argument("--pgd_lr", type=float, default=0.005)
    p.add_argument("--atk_eps", type=float, default=1.0)
    p.add_argument("--noise_thread", type=float, default=0.04)
    p.add_argument("--clients", type=int, nargs="*", default=None, help="빼 볼 클라 (기본 전원)")
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/release_map/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    K = len(U)
    clients = a.clients if a.clients else list(range(K))
    probe_loader, eval_loader = load_probe(a)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    pred0 = predict(m0, probe_loader, device)
    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "rhos": a.rhos,
           "before": {}, "clients": {}}
    if bd is not None:
        from utils.comm_utils import test
        res["before"] = {"acc": round(test(m0, eval_loader, device), 2),
                         "asr": asr_fuba(m0, eval_loader, bd, device, a.target, a.atk_eps, a.noise_thread)}
    print(f"[{a.name}] K={K} probe={a.probe} before={res['before']}", flush=True)

    for j in clients:
        t0 = time.time()
        curve, asr_curve = [], []
        models = {}
        for rho in a.rhos:
            sd = flat.to_sd(tT - rho * U[j])
            m = make_model(factory, sd, device)
            models[rho] = m
            d = divergence(pred0, predict(m, probe_loader, device), a.n_classes)
            d["rho"] = rho
            curve.append(d)
            if bd is not None:
                asr_curve.append(asr_fuba(m, eval_loader, bd, device, a.target, a.atk_eps, a.noise_thread))
        entry = {"curve": curve, "snap": snap_score(curve), "final": curve[-1], "asr_curve": asr_curve or None}
        if a.delta:
            delta = find_delta(m0, models[a.rhos[-1]], probe_loader, device, a.eps, a.pgd_steps, a.pgd_lr)
            pred0d = predict(m0, probe_loader, device, delta)
            dcurve = []
            for rho in a.rhos:
                d = divergence(pred0d, predict(models[rho], probe_loader, device, delta), a.n_classes)
                d["rho"] = rho
                dcurve.append(d)
            entry["delta_curve"] = dcurve
            entry["delta_snap"] = snap_score(dcurve)
            entry["delta_final"] = dcurve[-1]
        entry["seconds"] = round(time.time() - t0, 1)
        role = "requester" if j == a.requester else ("attacker" if j in a.attackers else "benign")
        entry["role"] = role
        res["clients"][str(j)] = entry
        f = entry["final"]
        print(f"  client {j} ({role:9s}) snap {entry['snap']:.2f}  final changed {f['changed']:.3f} conc {f['conc']:.2f} -> {f['to_class']}"
              + (f"  | δ: snap {entry['delta_snap']:.2f} conc {entry['delta_final']['conc']:.2f} -> {entry['delta_final']['to_class']}" if a.delta else "")
              + (f"  | ASR ρ=0..1: {asr_curve}" if asr_curve else ""), flush=True)
        del models

    # ② 전원 빼 보기: 요청자가 나머지 분포에서 어디에 있나
    def rank_of(key):
        vals = {j: res["clients"][str(j)][key] if key in ("snap", "delta_snap") else res["clients"][str(j)][key]["mass"]
                for j in clients}
        order = sorted(vals, key=lambda j: -vals[j])
        others = [vals[j] for j in clients if j != a.requester]
        mu, sd = (float(np.mean(others)), float(np.std(others))) if others else (0.0, 0.0)
        z = (vals[a.requester] - mu) / sd if (sd > 0 and a.requester in vals) else None
        return {"rank": order.index(a.requester) + 1 if a.requester in vals else None, "of": len(order),
                "value": vals.get(a.requester), "others_mean": round(mu, 4), "others_std": round(sd, 4),
                "z": round(z, 2) if z is not None else None}
    res["requester_rank"] = {"snap": rank_of("snap"), "final_mass": rank_of("final")}
    if a.delta:
        res["requester_rank"]["delta_snap"] = rank_of("delta_snap")
        res["requester_rank"]["delta_final_mass"] = rank_of("delta_final")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n요청자 {a.requester} 순위: " + ", ".join(f"{k} {v['rank']}/{v['of']} (z {v['z']})" for k, v in res["requester_rank"].items()))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
