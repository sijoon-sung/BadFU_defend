# -*- coding: utf-8 -*-
"""분석 (학습 없음): 학습 중 집계 방어가 보는 것을 그대로 재 본다 — 라운드별 클라이언트 갱신의 매개변수 공간 통계.
각 라운드 t 의 갱신 u_j^t = w_j (로컬 − 직전 전역) 에 대해
  norm       ‖u_j^t‖                                   (norm clipping / 크기 기반)
  cos_med    cos(u_j^t, 좌표별 중앙값 갱신)              (FLAME 류 코사인 군집, Median 류)
  krum       Krum 점수 = 가장 가까운 K−f−2 명과의 거리 제곱 합  (Krum / Multi-Krum)
  cos_pair   클라이언트 쌍 코사인 행렬의 평균 (무리 구조)
그리고 '운반자가 혼자 빠지면 무해한가' (DOSE 모집단 가정 T2) 를 전량 빼기로 재 본다.
실행: python analysis_update_stats.py --fmt fuba --name loc_fuba --K 8 --rounds 8
출력: logs/analysis/update_stats_{name}.json + 콘솔 표
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
from release_map import acc_of, load_fuba, load_probe, make_model, make_scorer  # noqa: E402


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
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--f", type=int, default=None, help="Krum 의 공격자 수 가정 f (기본 K//4)")
    p.add_argument("--noise_thread", type=float, default=0.04)
    p.add_argument("--atk_eps", type=float, default=1.0)
    p.add_argument("--badfu_record", default=None)
    p.add_argument("--n_gpu", type=int, default=1)
    return p.parse_args()


def main():
    a = parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    factory, theta_T, U, flat, bd = load_fuba(a)
    K = len(U)
    f = a.f if a.f is not None else max(1, K // 4)
    roles = {j: ("suppressor" if j == a.defender else ("carrier" if j in a.attackers else "benign")) for j in range(K)}
    _, rounds = load_rounds(a)
    per_round = []
    for t, (g_prev, locals_) in enumerate(rounds):
        gv = flat.flatten(g_prev)
        u = {j: (flat.flatten(locals_[j]) - gv) / K for j in range(K) if locals_[j] is not None}
        js = sorted(u)
        M = torch.stack([u[j] for j in js], 0)
        med = M.median(0).values
        mean = M.mean(0)
        norms = {j: float(u[j].norm()) for j in js}
        cos_med = {j: float(F.cosine_similarity(u[j], med, dim=0)) for j in js}
        cos_mean = {j: float(F.cosine_similarity(u[j], mean, dim=0)) for j in js}
        D = torch.cdist(M, M) ** 2
        krum = {}
        for i, j in enumerate(js):
            d = D[i].clone(); d[i] = float("inf")
            m = max(1, len(js) - f - 2)
            krum[j] = float(torch.topk(d, m, largest=False).values.sum())
        C = torch.stack([torch.stack([F.cosine_similarity(u[i], u[j], dim=0) for j in js]) for i in js])
        per_round.append({"t": t, "norm": norms, "cos_med": cos_med, "cos_mean": cos_mean, "krum": krum,
                          "cos_pair": {str(i): {str(j): round(float(C[a_, b_]), 3) for b_, j in enumerate(js)} for a_, i in enumerate(js)}})
    # 라운드 평균 표
    def avg(key):
        return {j: round(float(np.mean([r[key][j] for r in per_round if j in r[key]])), 4) for j in range(K)}
    summary = {k: avg(k) for k in ("norm", "cos_med", "cos_mean", "krum")}
    # 라운드별 순위: 집계 방어가 이 라운드에 '가장 의심' 으로 뽑을 클라이언트 (krum 최대, cos_med 최소, norm 최대)
    picks = {"krum_max": [], "cos_med_min": [], "norm_max": []}
    for r in per_round:
        picks["krum_max"].append(max(r["krum"], key=r["krum"].get))
        picks["cos_med_min"].append(min(r["cos_med"], key=r["cos_med"].get))
        picks["norm_max"].append(max(r["norm"], key=r["norm"].get))
    # 쌍 코사인 평균 (역할 무리별)
    groups = {"carrier": [j for j in range(K) if roles[j] == "carrier"], "benign": [j for j in range(K) if roles[j] == "benign"], "suppressor": [a.defender]}
    def gcos(g1, g2):
        vals = []
        for r in per_round:
            for i in groups[g1]:
                for j in groups[g2]:
                    if i != j:
                        vals.append(r["cos_pair"][str(i)][str(j)])
        return round(float(np.mean(vals)), 3) if vals else None
    cluster = {f"{g1}-{g2}": gcos(g1, g2) for g1 in groups for g2 in groups}
    # T2: 운반자 하나를 전량 빼면 무해한가
    probe_loader, eval_loader = load_probe(a)
    m0 = make_model(factory, theta_T, device)
    score = make_scorer(a, bd, eval_loader, device)
    tT = flat.flatten(theta_T)
    removal = {"theta_T": {"acc": acc_of(m0, eval_loader, device), "asr": score(m0) if score else None}}
    for j in range(K):
        m = make_model(factory, flat.to_sd(tT - U[j]), device)
        removal[j] = {"role": roles[j], "acc": acc_of(m, eval_loader, device), "asr": score(m) if score else None}
    # 운반자 전부 빼기 (억제자는 둔다): 백도어 몫만 없어지면 무해해야
    th = tT.clone()
    for j in a.attackers:
        th = th - U[j]
    m = make_model(factory, flat.to_sd(th), device)
    removal["all_carriers"] = {"acc": acc_of(m, eval_loader, device), "asr": score(m) if score else None}
    res = {"name": a.name, "K": K, "f": f, "roles": roles, "summary": summary, "picks": picks, "cluster_cos": cluster, "removal": removal, "per_round": per_round}
    os.makedirs("logs/analysis", exist_ok=True)
    out = f"logs/analysis/update_stats_{a.name}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"[{a.name}] K={K} Krum f={f}")
    print("클라 | 역할       | ‖u‖ 평균 | cos(중앙값) | cos(평균) | Krum 점수")
    for j in range(K):
        print(f"{j:3d} | {roles[j]:10s} | {summary['norm'][j]:.4f} | {summary['cos_med'][j]:.3f} | {summary['cos_mean'][j]:.3f} | {summary['krum'][j]:.4f}")
    print("라운드별 '가장 의심' 픽:", {k: [int(x) for x in v] for k, v in picks.items()})
    print("무리 코사인:", cluster)
    print("전량 빼기:", {k: v for k, v in removal.items()})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
