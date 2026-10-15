# -*- coding: utf-8 -*-
"""분석 (학습 없음): "헤지" 가설 검증 — 위장자와 운반자의 저장 갱신이 라운드마다 반대 방향인가.
저장된 라운드별 갱신 u_j^t 만 쓴다 (서버가 이미 가진 것). 세 가지를 잰다.
  (1) 쌍 코사인, 역할 무리별 평균: 전체 매개변수 / 층별 / 공통 성분(라운드 평균 갱신) 제거 뒤.
  (2) 총노출 vs 순노출: gross_t = Σ_j ‖u_j^t‖, net_t = ‖Σ_j u_j^t‖, 비율 gross/net. 그리고 각 클라이언트가 총노출에서 차지하는 몫.
  (3) 반상관 짝 찾기: 각 클라이언트 j 에 대해 가장 반대인 상대 argmin_i cos(u_j, u_i) 와 그 값 (라운드 평균). 위장자의 짝이 운반자인가, 정상의 짝은 누구인가.
실행: python analysis_hedge.py --fmt fuba --name loc_fuba --K 8 --rounds 8
출력: logs/analysis/hedge_{name}.json + 콘솔 표
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from baseline_masa import load_rounds  # noqa: E402
from release_map import load_fuba  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba"], default="fuba")
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--traj_dir", default=None)
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    return p.parse_args()


def main():
    a = parse()
    factory, theta_T, U, flat, bd = load_fuba(a)
    K = len(U)
    roles = {j: ("suppressor" if j == a.defender else ("carrier" if j in a.attackers else "benign")) for j in range(K)}
    groups = {"carrier": [j for j in range(K) if roles[j] == "carrier"], "benign": [j for j in range(K) if roles[j] == "benign"], "suppressor": [a.defender]}
    _, rounds = load_rounds(a)
    layers = {k: flat.slices[k] for k in flat.keys}
    layer_groups = {}
    for k in flat.keys:
        base = k.rsplit(".", 1)[0]
        layer_groups.setdefault(base, []).append(flat.slices[k])

    def cos(x, y):
        return float(F.cosine_similarity(x, y, dim=0))

    def view(u, sls):
        return torch.cat([u[s] for s in sls])

    pair_full, pair_cent, pair_layer = [], [], {L: [] for L in layer_groups}
    gross, net, share = [], [], []
    anti = {j: [] for j in range(K)}
    for t, (g_prev, locals_) in enumerate(rounds):
        gv = flat.flatten(g_prev)
        u = {j: (flat.flatten(locals_[j]) - gv) / K for j in range(K) if locals_[j] is not None}
        js = sorted(u)
        mean = torch.stack([u[j] for j in js]).mean(0)
        uc = {j: u[j] - mean for j in js}
        C = {(i, j): cos(u[i], u[j]) for i in js for j in js if i != j}
        Cc = {(i, j): cos(uc[i], uc[j]) for i in js for j in js if i != j}
        pair_full.append(C); pair_cent.append(Cc)
        for L, sls in layer_groups.items():
            pair_layer[L].append({(i, j): cos(view(u[i], sls), view(u[j], sls)) for i in js for j in js if i != j})
        norms = {j: float(u[j].norm()) for j in js}
        g = sum(norms.values()); n = float(sum(u[j] for j in js).norm())
        gross.append(g); net.append(n); share.append({j: norms[j] / g for j in js})
        for j in js:
            i_min = min((i for i in js if i != j), key=lambda i: C[(j, i)])
            anti[j].append((i_min, C[(j, i_min)]))

    def group_mean(P):
        out = {}
        for g1 in groups:
            for g2 in groups:
                vals = [p[(i, j)] for p in P for i in groups[g1] for j in groups[g2] if i != j and (i, j) in p]
                out[f"{g1}-{g2}"] = round(float(np.mean(vals)), 3) if vals else None
        return out

    res = {"name": a.name, "K": K, "roles": roles,
           "pair_cos_full": group_mean(pair_full), "pair_cos_centered": group_mean(pair_cent),
           "pair_cos_by_layer": {L: group_mean(P) for L, P in pair_layer.items()},
           "gross": [round(x, 4) for x in gross], "net": [round(x, 4) for x in net],
           "gross_over_net": [round(g / n, 3) for g, n in zip(gross, net)],
           "share_mean": {j: round(float(np.mean([s[j] for s in share if j in s])), 3) for j in range(K)},
           "anti_partner": {j: {"partners": [int(p[0]) for p in anti[j]], "cos_mean": round(float(np.mean([p[1] for p in anti[j]])), 3),
                                "partner_roles": [roles[int(p[0])] for p in anti[j]]} for j in range(K)}}
    os.makedirs("logs/analysis", exist_ok=True)
    out = f"logs/analysis/hedge_{a.name}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    keys = ["suppressor-carrier", "suppressor-benign", "carrier-carrier", "benign-benign", "carrier-benign"]
    print(f"[{a.name}] 쌍 코사인 (라운드 평균)  열: " + " | ".join(keys))
    print("  전체 매개변수      : " + " | ".join(f"{res['pair_cos_full'][k]:+.3f}" for k in keys))
    print("  공통 성분 제거 뒤  : " + " | ".join(f"{res['pair_cos_centered'][k]:+.3f}" for k in keys))
    for L in layer_groups:
        print(f"  층 {L:6s}          : " + " | ".join(f"{res['pair_cos_by_layer'][L][k]:+.3f}" for k in keys))
    print("총노출/순노출 (라운드별):", res["gross_over_net"])
    print("총노출 몫 (클라별 평균):", {j: f"{roles[j][:3]} {res['share_mean'][j]:.2f}" for j in range(K)})
    print("가장 반대인 상대 (라운드별 상대, 평균 코사인):")
    for j in range(K):
        ap = res["anti_partner"][j]
        print(f"  {j} {roles[j]:10s} → {ap['partners']}  cos {ap['cos_mean']:+.3f}  ({', '.join(sorted(set(ap['partner_roles'])))})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
