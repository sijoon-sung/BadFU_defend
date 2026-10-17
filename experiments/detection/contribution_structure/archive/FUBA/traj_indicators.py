# -*- coding: utf-8 -*-
"""궤적 기반 상쇄 지표 (docs/paper/THREAT_MODEL.md 의 틀): 라운드별 기여가 입력 집합 X 위의 클래스별 손실을 어느 쪽으로 미는가.

핵심 양: 라운드 t, 클라 k, 클래스 c 에 대해  v_k^t[c] = ⟨U_k^t, g_c(X)⟩,  g_c(X) = ∇_θ mean CE(θ_T; x ∈ X, ŷ(x)=c)
  - ŷ 는 θ_T 가 매긴 답(가짜 라벨) → 라벨 불필요. X = 공개 입력 (+ 요청자의 잊을 집합 X_f).
  - 1 차 근사로, k 를 빼면 클래스 c 의 손실이 −v_k[c] 만큼 변한다. 즉 '빼 보기'를 빼지 않고 예측한 것.
지표 (클라마다, 모집단 z 로 정규화)
  mag       ‖v_k‖ (라운드 합)                         효과 크기
  spread    |mean_c sign(v_k[c])|                       전 클래스가 같은 부호로 움직이면 1 (타깃 하나로 쏠림), 자기 클래스만이면 0 근처
  consist   mean_t cos(v_k^t, v_k)                      라운드 간 일관성 (억제는 매 라운드 같은 방향)
  cancel    min_j mean_t cos(v_k^t, v_j^t)              상쇄 짝: 가장 반대로 움직이는 j (−1 에 가까울수록 상쇄). 짝 j 도 기록
  cancel_set cos(v_k, Σ_{j≠k} v_j)                      나머지 전체와 반대 방향인가
forget 모드: 요청자 k 마다 자기 X_f(k) 위의 g_c 로 모든 클라의 v_j 를 다시 재서 k 의 z 를 구한다 (BadFU 류).

실행 예
  python traj_indicators.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requesters 4 7 5
  python traj_indicators.py --fmt badfu --name loc_badfu --traj_dir ../BadFU_src/BadFU-main/logs/traj_full ... --defender 5 --attackers 0 --requesters 5 1 2
출력: logs/traj_indicators/{name}.json, 콘솔 표.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from baseline_masa import load_rounds  # noqa: E402
from dose_fu import robust_z  # noqa: E402
from release_map import Flat, load_badfu, load_fuba, load_probe, make_model  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba", "badfu"], default="fuba")
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--traj_dir", default=None)
    p.add_argument("--counts", type=int, nargs="*", default=None)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--badfu_record", default=None)
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--defenders", type=int, nargs="*", default=None)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--requesters", type=int, nargs="*", default=[4, 7, 5], help="잊을 집합(X_f)을 더해 다시 재는 요청자들")
    p.add_argument("--forget", choices=["on", "off"], default="on")
    p.add_argument("--forget_n", type=int, default=500)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--zmode", choices=["std", "mad", "iqr"], default="std")
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--split_seed", type=int, default=1223)
    p.add_argument("--non_iid", action="store_true")
    p.add_argument("--non_iid_type", default=None)
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--sigma", type=float, default=1)
    p.add_argument("--shift_ratio", type=float, default=0.4)
    p.add_argument("--owner_client", type=int, default=-1)
    p.add_argument("--owner_class", type=int, default=3)
    p.add_argument("--owner_frac", type=float, default=0.9)
    p.add_argument("--train_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--noise_thread", type=float, default=0.04)
    p.add_argument("--atk_eps", type=float, default=1.0)
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def class_grads(model, flat, xs, n_classes, device, bs=128):
    """g_c = ∇_θ mean CE(θ; x with ŷ(x)=c), 가짜 라벨 ŷ = argmax θ(x). 평탄 벡터(파라미터 자리만, BN 통계 자리는 0)."""
    model.eval()
    with torch.no_grad():
        yh = torch.cat([model(xs[i:i + bs]).argmax(1) for i in range(0, len(xs), bs)])
    params = dict(model.named_parameters())
    for p in params.values():
        p.requires_grad_(True)
    G = torch.zeros(n_classes, flat.dim)
    counts = []
    for c in range(n_classes):
        idx = (yh == c).nonzero().flatten()
        counts.append(int(idx.numel()))
        if idx.numel() == 0:
            continue
        model.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, idx.numel(), bs):
            sub = idx[i:i + bs]
            loss = F.cross_entropy(model(xs[sub]), yh[sub], reduction="sum")
            loss.backward()
            tot += float(sub.numel())
        g = torch.zeros(flat.dim)
        for name, p in params.items():
            if p.grad is not None and name in flat.slices:
                g[flat.slices[name]] = (p.grad.detach().float().cpu() / tot).flatten()
        G[c] = g
    for p in params.values():
        p.requires_grad_(False)
    return G, counts


def indicators(V, clients):
    """V: {k: tensor (T, C)} 라운드별 효과 벡터. 지표 dict 와 상쇄 짝."""
    out = {}
    tot = {k: V[k].sum(0) for k in clients}
    for k in clients:
        vk, vt = tot[k], V[k]
        mag = float(vk.norm())
        spread = float(torch.sign(vk).mean().abs()) if mag > 0 else 0.0
        cons = float(np.mean([F.cosine_similarity(vt[t], vk, dim=0).item() for t in range(vt.shape[0]) if vt[t].norm() > 0])) if mag > 0 else 0.0
        # 상쇄: 크기가 다른 짝(억제자 하나 ↔ 운반자 여럿)도 잡도록 노름비가 아니라 방향(코사인)으로 본다.
        #   cancel  = min_j mean_t cos(v_k^t, v_j^t)  (가장 반대로 움직이는 짝; −1 에 가까울수록 상쇄)
        #   cancel_set = cos(v_k, Σ_{j≠k} v_j)        (나머지 전체와 반대인가)
        best, bj = None, None
        for j in clients:
            if j == k:
                continue
            r = [F.cosine_similarity(vt[t], V[j][t], dim=0).item() for t in range(vt.shape[0]) if vt[t].norm() > 0 and V[j][t].norm() > 0]
            if r:
                m = float(np.mean(r))
                if best is None or m < best:
                    best, bj = m, j
        rest = sum(tot[j] for j in clients if j != k)
        cset = float(F.cosine_similarity(vk, rest, dim=0)) if mag > 0 and rest.norm() > 0 else 0.0
        out[k] = {"mag": mag, "spread": spread, "consist": cons, "cancel": best if best is not None else 1.0, "cancel_partner": bj,
                  "cancel_set": cset, "v": [round(float(x), 4) for x in vk]}
    return out


def zscores(ind, clients, keys, zmode):
    """지표별 모집단 z (각 클라를 나머지와 비교). cancel 은 작을수록 이상하므로 부호를 뒤집는다."""
    Z = {k: {} for k in clients}
    for key in keys:
        sgn = -1.0 if key in ("cancel", "cancel_set") else 1.0
        for k in clients:
            others = [sgn * ind[j][key] for j in clients if j != k]
            Z[k][key] = round(robust_z(sgn * ind[k][key], others, zmode), 2)
    return Z


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return round(float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])), 3)


def main():
    a = parse()
    if a.defenders is None:
        a.defenders = [a.defender]
    torch.manual_seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/traj_indicators/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()

    factory, theta_T, _, flat, _ = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    _, rounds = load_rounds(a)
    K = len(rounds[0][1])
    clients = list(range(K))
    roles = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else "benign")) for j in clients}
    w = [c / sum(a.counts) for c in a.counts] if (a.fmt == "badfu" and a.counts) else [1.0 / K] * K
    probe_loader, _ = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    model = make_model(factory, theta_T, device)

    # 라운드별 기여 (release_map 과 같은 정의: 가중 w_k (local − g_prev))
    Ut = []   # list over rounds of {k: vec}
    for g_prev, locals_ in rounds:
        gv = flat.flatten(g_prev)
        Ut.append({j: w[j] * (flat.flatten(locals_[j]) - gv) for j in clients if locals_[j] is not None})

    def effects(G):
        V = {}
        for k in clients:
            rows = []
            for t in range(len(Ut)):
                rows.append(G @ Ut[t][k] if k in Ut[t] else torch.zeros(G.shape[0]))
            V[k] = torch.stack(rows)
        return V

    keys = ["mag", "spread", "consist", "cancel", "cancel_set"]
    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "roles": roles, "rounds": len(Ut)}

    # 1) 공개 입력
    G_pub, cnt = class_grads(model, flat, xs, a.n_classes, device)
    V = effects(G_pub)
    ind = indicators(V, clients)
    Z = zscores(ind, clients, keys, a.zmode)
    res["public"] = {"class_counts": cnt, "ind": ind, "z": Z}
    sup = [j for j in clients if roles[j] == "suppressor"]
    car = [j for j in clients if roles[j] == "carrier"]
    ben = [j for j in clients if roles[j] == "benign"]
    res["public"]["auroc_suppressor_vs_benign"] = {key: auroc([Z[j][key] for j in sup], [Z[j][key] for j in ben]) for key in keys}
    res["public"]["auroc_carrier_vs_benign"] = {key: auroc([Z[j][key] for j in car], [Z[j][key] for j in ben]) for key in keys}
    print(f"[traj {a.name}] 공개 입력 {len(xs)}장, 라운드 {len(Ut)}, K={K}")
    for k in clients:
        print(f"  {k} {roles[k][:9]:9s} mag z{Z[k]['mag']:+6.2f} spread {ind[k]['spread']:.2f} z{Z[k]['spread']:+6.2f} consist {ind[k]['consist']:.2f} z{Z[k]['consist']:+6.2f} cancel {ind[k]['cancel']:+.2f}(짝 {ind[k]['cancel_partner']}) z{Z[k]['cancel']:+6.2f} set {ind[k]['cancel_set']:+.2f} z{Z[k]['cancel_set']:+6.2f}")

    # 2) 요청자별 잊을 집합
    if a.forget == "on" and a.requesters:
        forget_sets = {}
        if a.fmt == "fuba":
            from benign_departure import make_config
            from robust_fu import _slice
            from utils.comm_utils import make_client_split
            from dataset import MNIST
            import torchvision.transforms as transforms
            cfg = make_config(a)
            trainset = MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            split = make_client_split(trainset, cfg)
            for k in a.requesters:
                if k < len(split):
                    forget_sets[k] = _slice(split[k], a.forget_n, a.seed + 3)[0].to(device)
        else:
            import badfu_common as bc
            bdata = bc.BadFUData()
            g = torch.Generator().manual_seed(a.seed + 3)
            for k in a.requesters:
                ds = bdata.cv if k in a.defenders else bdata.client_sets[k]
                idx = torch.randperm(len(ds), generator=g)[:a.forget_n].tolist()
                forget_sets[k] = torch.stack([ds[i][0] for i in idx]).to(device)
        res["forget"] = {}
        for k, xf in forget_sets.items():
            G_f, cnt_f = class_grads(model, flat, xf, a.n_classes, device)
            Vf = effects(G_f)
            ind_f = indicators(Vf, clients)
            Z_f = zscores(ind_f, clients, keys, a.zmode)
            flagged = {key: bool(Z_f[k][key] > a.tau) for key in keys}
            res["forget"][str(k)] = {"role": roles[k], "n": int(len(xf)), "class_counts": cnt_f, "ind": ind_f, "z_requester": Z_f[k], "flagged": flagged}
            print(f"  요청자 {k} ({roles[k]}) 잊을 집합 {len(xf)}장: " + " ".join(f"{key} {ind_f[k][key]:.2f} z{Z_f[k][key]:+6.2f}{'!' if flagged[key] else ''}" for key in keys)
                  + f"  짝 {ind_f[k]['cancel_partner']}")
    res["seconds"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"-> {out}  ({res['seconds']}s)")


if __name__ == "__main__":
    main()
