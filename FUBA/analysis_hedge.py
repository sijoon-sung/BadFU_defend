# -*- coding: utf-8 -*-
"""헤지 파일럿 (학습 없음): 위장자와 운반자의 저장 갱신이 라운드마다 반대 방향인가.
서버가 이미 가진 라운드별 갱신 u_j^t = (로컬 − 직전 전역)/K 만 쓴다 (시뮬레이션·데이터·트리거 없음).

잰다
  (1) 쌍 코사인 cos(u_i^t, u_j^t): 전체 매개변수 / 층별 / 공통 성분(그 라운드 평균 갱신) 제거 뒤. 라운드 평균을 쌍마다 저장.
  (2) 총노출 gross_t = Σ_j ‖u_j^t‖, 순노출 net_t = ‖Σ_j u_j^t‖, 비율, 클라이언트별 총노출 몫.
  (3) 반상관 짝: 클라이언트마다 가장 반대인 상대(라운드별)와 그 코사인.
  (4) "빼면 무엇이 풀려나나" 의 저장 갱신판: 누적 기여 U_j 끼리의 코사인 (요청 전에 계산 가능).
판정은 aggregate_hedge.py 가 한다 (위장자–운반자 쌍 vs 나머지 쌍의 분리 AUROC).

실행: python analysis_hedge.py --fmt fuba --name bd_iid_s0 --K 8 --rounds 8 --defender 4 --attackers 0 1 2 3
출력: logs/analysis/hedge_{name}.json
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
    p.add_argument("--defenders", type=int, nargs="*", default=None)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--cond", default=None, help="조건 라벨 (iid / dir05 / owner / k20paper …). 집계 표에 쓴다")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse()
    if a.defenders is None:
        a.defenders = [a.defender]
    factory, theta_T, U, flat, bd = load_fuba(a)
    K = len(U)
    roles = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else "benign")) for j in range(K)}
    groups = {g: [j for j in range(K) if roles[j] == g] for g in ("carrier", "benign", "suppressor")}
    _, rounds = load_rounds(a)
    layer_groups = {}
    for k in flat.keys:
        layer_groups.setdefault(k.rsplit(".", 1)[0], []).append(flat.slices[k])

    def cos(x, y):
        return float(F.cosine_similarity(x, y, dim=0))

    def view(u, sls):
        return torch.cat([u[s] for s in sls])

    pair_full, pair_cent = {}, {}
    pair_layer = {L: {} for L in layer_groups}
    gross, net, share = [], [], []
    anti = {j: [] for j in range(K)}
    n_rounds_used = 0
    for t, (g_prev, locals_) in enumerate(rounds):
        gv = flat.flatten(g_prev)
        u = {j: (flat.flatten(locals_[j]) - gv) / K for j in range(K) if locals_[j] is not None}
        js = sorted(u)
        if len(js) < 3:
            continue
        n_rounds_used += 1
        mean = torch.stack([u[j] for j in js]).mean(0)
        uc = {j: u[j] - mean for j in js}
        for i in js:
            for j in js:
                if i < j:
                    pair_full.setdefault(f"{i}-{j}", []).append(cos(u[i], u[j]))
                    pair_cent.setdefault(f"{i}-{j}", []).append(cos(uc[i], uc[j]))
                    for L, sls in layer_groups.items():
                        pair_layer[L].setdefault(f"{i}-{j}", []).append(cos(view(u[i], sls), view(u[j], sls)))
        norms = {j: float(u[j].norm()) for j in js}
        g = sum(norms.values()); n = float(sum(u[j] for j in js).norm())
        gross.append(g); net.append(n); share.append({j: norms[j] / g for j in js})
        for j in js:
            i_min = min((i for i in js if i != j), key=lambda i: cos(u[j], u[i]))
            anti[j].append((i_min, cos(u[j], u[i_min])))

    def avg_pairs(P):
        return {k: round(float(np.mean(v)), 4) for k, v in P.items()}

    def group_mean(P):
        out = {}
        for g1 in groups:
            for g2 in groups:
                vals = []
                for i in groups[g1]:
                    for j in groups[g2]:
                        key = f"{min(i, j)}-{max(i, j)}"
                        if i != j and key in P:
                            vals.extend(P[key])
                out[f"{g1}-{g2}"] = round(float(np.mean(vals)), 3) if vals else None
        return out

    # (4) 누적 기여끼리의 코사인 (요청 전 계산)
    cum = {f"{i}-{j}": round(cos(U[i], U[j]), 4) for i in range(K) for j in range(K) if i < j}
    res = {"name": a.name, "cond": a.cond, "K": K, "roles": roles, "rounds_used": n_rounds_used,
           "pair_cos_full": avg_pairs(pair_full), "pair_cos_centered": avg_pairs(pair_cent),
           "pair_cos_by_layer": {L: avg_pairs(P) for L, P in pair_layer.items()},
           "pair_cos_cumulative": cum,
           "group_full": group_mean(pair_full), "group_centered": group_mean(pair_cent),
           "group_by_layer": {L: group_mean(P) for L, P in pair_layer.items()},
           "gross": [round(x, 4) for x in gross], "net": [round(x, 4) for x in net],
           "gross_over_net": [round(g / n, 3) for g, n in zip(gross, net)],
           "share_mean": {j: round(float(np.mean([s[j] for s in share if j in s])), 3) for j in range(K)},
           "anti_partner": {j: {"partners": [int(p[0]) for p in anti[j]], "cos_mean": round(float(np.mean([p[1] for p in anti[j]])), 3) if anti[j] else None,
                                "partner_roles": [roles[int(p[0])] for p in anti[j]]} for j in range(K)}}
    out = a.out or f"logs/analysis/hedge_{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    keys = ["suppressor-carrier", "suppressor-benign", "carrier-carrier", "benign-benign", "carrier-benign"]
    print(f"[{a.name}] K={K} 라운드 {n_rounds_used}  쌍 코사인 (라운드 평균)  열: " + " | ".join(keys))
    print("  전체 매개변수      : " + " | ".join(f"{res['group_full'][k]:+.3f}" if res['group_full'][k] is not None else "  -   " for k in keys))
    print("  공통 성분 제거 뒤  : " + " | ".join(f"{res['group_centered'][k]:+.3f}" if res['group_centered'][k] is not None else "  -   " for k in keys))
    for L in layer_groups:
        print(f"  층 {L:6s}          : " + " | ".join(f"{res['group_by_layer'][L][k]:+.3f}" if res['group_by_layer'][L][k] is not None else "  -   " for k in keys))
    print("총노출/순노출 (라운드별):", res["gross_over_net"])
    print("총노출 몫 (클라별 평균):", {j: f"{roles[j][:3]} {res['share_mean'][j]:.2f}" for j in range(K)})
    print("가장 반대인 상대 (라운드별 상대, 평균 코사인):")
    for j in range(K):
        ap = res["anti_partner"][j]
        print(f"  {j} {roles[j]:10s} → {ap['partners']}  cos {ap['cos_mean']}  ({', '.join(sorted(set(ap['partner_roles'])))})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
