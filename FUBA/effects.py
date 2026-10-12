# -*- coding: utf-8 -*-
"""기능 공간 효과 분석 (docs/paper/MATH.md): 클라이언트 기여가 입력 집합 X 위에서 모델 출력을 어떻게 바꾸는지를
입력×클래스 단위로 보고, 상쇄(cancellation)를 부분집합·입력 단위로 직접 잰다. 스칼라 z 하나가 아니라
  (1) 기능 효과 행렬 E_j ∈ R^{N×C},  (2) 쌍·부분집합 상쇄,  (3) 입력 단위 상쇄 지도,  (4) 라운드 일관성,
  (5) 분포 가정 없는 p 값(conformal) 과 결합(Fisher),  (6) 투여량 곡선 외삽
을 돌려준다. 학습 없음, 라벨 없음.

정의 (선형화): E_j = [log p(θ_T − εU_j) − log p(θ_T)] / ε   (X 위, 클래스별 로그확률 변화; ε 작게 → 자코비안-벡터곱 근사)
  - 제거 효과의 1 차 근사. 전량 제거가 무너뜨리는 깊은 망에서도 ε 가 작아 안전하다.
상쇄: 요청자 k 의 효과가 다른 몇 명의 효과와 더해져 사라지는 정도
  cancel_k = 1 − min_{S⊂others,|S|≤r} ‖E_k + Σ_{j∈S} E_j‖² / ‖E_k‖²   (탐욕으로 S 선택)
입력 단위: x 마다 e_k(x) = E_k[x, ŷ(x)] (자기 답의 로그확률 변화). 상쇄 입력 X*_hat = { x : e_k(x) 크고, 짝 S 가 반대 부호 }.

실행 예
  python effects.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requesters 4 7 5
출력: logs/effects/{name}.json, 콘솔 표.
"""
import argparse
import json
import math
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
from release_map import load_badfu, load_fuba, load_probe, make_model  # noqa: E402


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
    p.add_argument("--requesters", type=int, nargs="*", default=[4, 7, 5])
    p.add_argument("--forget", choices=["on", "off"], default="on")
    p.add_argument("--forget_n", type=int, default=500)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--eps", type=float, default=0.2, help="선형화 비율 ε (θ_T − εU_j)")
    p.add_argument("--rounds_eff", choices=["on", "off"], default="on", help="라운드별 효과 (K·T 순전파)")
    p.add_argument("--max_set", type=int, default=4)
    p.add_argument("--center", choices=["median", "mean", "none"], default="median",
                   help="효과 행렬에서 공통 성분(누구를 빼도 생기는 변화)을 제거: Ẽ_j = E_j − center_i E_i. 원시 E_j 는 서로 평행에 가까움(코사인 0.9+)")
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


@torch.no_grad()
def logp(model, xs, bs=256):
    return torch.cat([F.log_softmax(model(xs[i:i + bs]), 1).cpu() for i in range(0, len(xs), bs)])


def conformal_p(s_k, s_others):
    """분포 가정 없는 p 값: 정상 요청자가 모집단과 교환 가능하다면 P(p ≤ α) ≤ α."""
    return (1 + sum(1 for s in s_others if s >= s_k)) / (1 + len(s_others))


def fisher(ps):
    ps = [min(max(p, 1e-12), 1.0) for p in ps]
    stat = -2.0 * sum(math.log(p) for p in ps)
    k = 2 * len(ps)
    # 카이제곱(2m) 생존함수 (정수 자유도, 반복식)
    x = stat / 2.0
    term, s = 1.0, 1.0
    for i in range(1, k // 2):
        term *= x / i
        s += term
    return stat, min(1.0, math.exp(-x) * s)


def greedy_cancel(Ek, E_others, max_set):
    """E_k 를 가장 많이 지우는 부분집합 S (탐욕). 돌려줌: (cancel 비율, S, 단계별 잔차비)."""
    base = float((Ek * Ek).sum())
    if base <= 0:
        return 0.0, [], []
    r = Ek.clone()
    S, hist = [], []
    cand = dict(E_others)
    for _ in range(max_set):
        best, bj = None, None
        for j, Ej in cand.items():
            nr = float(((r + Ej) ** 2).sum())
            if best is None or nr < best:
                best, bj = nr, j
        if bj is None or best >= float((r * r).sum()) - 1e-9:
            break
        S.append(bj); r = r + cand.pop(bj); hist.append(round(best / base, 4))
    return round(1.0 - float((r * r).sum()) / base, 4), S, hist


def main():
    a = parse()
    if a.defenders is None:
        a.defenders = [a.defender]
    torch.manual_seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/effects/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()

    factory, theta_T, U, flat, _ = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    K = len(U)
    clients = list(range(K))
    roles = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else "benign")) for j in clients}
    probe_loader, _ = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)

    def model(theta):
        return make_model(factory, flat.to_sd(theta), device)

    # 잊을 집합
    forget = {}
    if a.forget == "on":
        if a.fmt == "fuba":
            from benign_departure import make_config
            from robust_fu import _slice
            from utils.comm_utils import make_client_split
            from dataset import MNIST
            import torchvision.transforms as transforms
            split = make_client_split(MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), make_config(a))
            for k in a.requesters:
                if k < len(split):
                    forget[k] = _slice(split[k], a.forget_n, a.seed + 3)[0].to(device)
        else:
            import badfu_common as bc
            bdata = bc.BadFUData()
            g = torch.Generator().manual_seed(a.seed + 3)
            for k in a.requesters:
                ds = bdata.cv if k in a.defenders else bdata.client_sets[k]
                idx = torch.randperm(len(ds), generator=g)[:a.forget_n].tolist()
                forget[k] = torch.stack([ds[i][0] for i in idx]).to(device)

    def effects_on(X):
        """E_j (K × N × C) 와 ŷ. 모델당 순전파 1 회."""
        L0 = logp(m0, X)
        yh = L0.argmax(1)
        E = {j: (logp(model(tT - a.eps * U[j]), X) - L0) / a.eps for j in clients}
        raw_mag = {j: round(float(E[j].norm()) / math.sqrt(X.shape[0]), 4) for j in clients}
        if a.center != "none":
            stack = torch.stack([E[j] for j in clients], 0)
            cen = stack.median(0).values if a.center == "median" else stack.mean(0)
            E = {j: E[j] - cen for j in clients}
        return E, yh, raw_mag

    def analyze(X, label, req_list):
        E, yh, raw_mag = effects_on(X)
        N = X.shape[0]
        idx = torch.arange(N)
        e_self = {j: E[j][idx, yh] for j in clients}                       # 자기 답 로그확률 변화 (N)
        mag = {j: round(float(E[j].norm()) / math.sqrt(N), 4) for j in clients}
        # 타깃 몰림: 평균 로그확률 상승이 가장 큰 클래스와 그 양, 출발 다양성(상승 기여 입력들의 ŷ 엔트로피)
        conc = {}
        for j in clients:
            gain = E[j].mean(0)                                           # C
            c = int(gain.argmax())
            contrib = E[j][:, c].clamp_min(0)
            w = contrib / contrib.sum().clamp_min(1e-12)
            src = torch.zeros(a.n_classes).index_add_(0, yh, w)
            nz = src[src > 0]
            H = float(-(nz * nz.log()).sum()) / math.log(max(a.n_classes - 1, 2))
            conc[j] = {"class": c, "gain": round(float(gain[c]), 4), "src_div": round(min(max(H, 0.0), 1.0), 4),
                       "score": round(float(gain[c]) * min(max(H, 0.0), 1.0), 4)}
        # 쌍 코사인 (기능 공간)
        cos = {j: {i: round(float(F.cosine_similarity(E[j].flatten(), E[i].flatten(), dim=0)), 4) for i in clients if i != j} for j in clients}
        # 부분집합 상쇄
        cancel = {}
        for j in clients:
            c_ratio, S, hist = greedy_cancel(E[j], {i: E[i] for i in clients if i != j}, a.max_set)
            cancel[j] = {"ratio": c_ratio, "set": S, "set_roles": [roles[i] for i in S], "hist": hist}
        # 입력 단위 상쇄 지도 (요청자별): |e_k(x)| 상위 10 % 중 짝 S 의 합이 반대 부호인 비율
        local = {}
        for k in req_list:
            S = cancel[k]["set"]
            if not S:
                local[k] = None; continue
            ek = e_self[k]; es = sum(e_self[j] for j in S)
            thr = ek.abs().quantile(0.9)
            top = ek.abs() >= thr
            opp = ((ek * es) < 0) & top
            local[k] = {"n_top": int(top.sum()), "frac_opposed": round(float(opp.sum()) / max(int(top.sum()), 1), 4),
                        "top_classes": torch.bincount(yh[top], minlength=a.n_classes).tolist()}
        # 모집단 기반 p 값 (요청자별, 통계별) 과 결합
        stats = {"mag": mag, "conc": {j: conc[j]["score"] for j in clients}, "cancel": {j: cancel[j]["ratio"] for j in clients}}
        pvals, comb = {}, {}
        for k in req_list:
            pvals[k] = {s: round(conformal_p(stats[s][k], [stats[s][j] for j in clients if j != k]), 4) for s in stats}
            fs, fp = fisher(list(pvals[k].values()))
            comb[k] = {"fisher_stat": round(fs, 3), "fisher_p": round(fp, 4)}
        print(f"[effects {a.name}] {label}: N={N}")
        for j in clients:
            print(f"  {j} {roles[j][:9]:9s} raw {raw_mag[j]:.2f} mag {mag[j]:.3f}  conc {conc[j]['score']:+.3f}(→{conc[j]['class']}, div {conc[j]['src_div']:.2f})  "
                  f"cancel {cancel[j]['ratio']:.2f} by {cancel[j]['set']}  mincos {min(cos[j].values()):+.2f}")
        for k in req_list:
            print(f"  요청자 {k} ({roles[k]}): p mag {pvals[k]['mag']} conc {pvals[k]['conc']} cancel {pvals[k]['cancel']} → Fisher p {comb[k]['fisher_p']}  "
                  f"국소 상쇄 {local[k]}")
        return {"N": N, "raw_mag": raw_mag, "mag": mag, "conc": conc, "cos": cos, "cancel": cancel, "local": local, "pvals": pvals, "combined": comb}

    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "roles": roles}
    res["public"] = analyze(xs, "공개 입력", a.requesters)
    if forget:
        res["forget"] = {}
        for k, xf in forget.items():
            res["forget"][str(k)] = analyze(torch.cat([xs, xf]), f"공개 + 요청자 {k} 의 잊을 집합", [k])
    # 라운드 일관성 (요청자별, 공개 입력): 라운드별 E_j^t 의 코사인 평균
    if a.rounds_eff == "on":
        _, rounds = load_rounds(a)
        w = [c / sum(a.counts) for c in a.counts] if (a.fmt == "badfu" and a.counts) else [1.0 / K] * K
        L0 = logp(m0, xs)
        Et = {j: [] for j in clients}
        for g_prev, locals_ in rounds:
            gv = flat.flatten(g_prev)
            for j in clients:
                if locals_[j] is None:
                    continue
                u = w[j] * (flat.flatten(locals_[j]) - gv)
                Et[j].append(((logp(model(tT - a.eps * u), xs) - L0) / a.eps).flatten())
        if a.center != "none":                       # 라운드마다 공통 성분 제거
            T_ = min(len(Et[j]) for j in clients if Et[j])
            for t in range(T_):
                stk = torch.stack([Et[j][t] for j in clients if len(Et[j]) > t], 0)
                cen = stk.median(0).values if a.center == "median" else stk.mean(0)
                for j in clients:
                    if len(Et[j]) > t:
                        Et[j][t] = Et[j][t] - cen
        cons = {}
        for j in clients:
            if len(Et[j]) < 2:
                cons[j] = None; continue
            tot = sum(Et[j])
            cons[j] = round(float(np.mean([F.cosine_similarity(e, tot, dim=0).item() for e in Et[j] if e.norm() > 0])), 4)
        res["round_consistency"] = cons
        print("  라운드 일관성: " + " ".join(f"{j}{'*' if roles[j] != 'benign' else ''}:{cons[j]}" for j in clients))
    res["seconds"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"-> {out} ({res['seconds']}s)")


if __name__ == "__main__":
    main()
