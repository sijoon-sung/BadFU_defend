# -*- coding: utf-8 -*-
"""탐지 총정리 벤치 (학습 없음): 한 궤적에서 생각할 수 있는 탐지 점수를 전부 클라이언트마다 계산한다.
모든 클라이언트를 "요청자라면" 으로 두고 같은 점수를 매긴다 → 위장자가 몇 등인지, 정상·소유자가 잘못 걸리는지를 aggregate_detect_bench.py 가 센다.

방법 (점수 이름 → 뜻, 어느 쪽이 의심인가)
  A. 언러닝 시뮬레이션, 절대 기준 (기존 요청 심사 발상: 지워 보고 얼마나 변했나)
     sim_abs_changed@ρ  바뀐 답 비율 (높을수록 의심)      sim_abs_kl@ρ  KL(θ_T ‖ θ−ρU_j) (높을수록)      sim_abs_conc@ρ  바뀐 답의 한 클래스 몰림
  B. 언러닝 시뮬레이션, 상대 기준 (DOSE: 남들을 같은 양 뺀 모집단 대비)
     sim_rel_flow_z@ρ, sim_rel_mass_z@ρ   (높을수록)
  C. 기능 효과 (저장 기여의 선형화, 중심화)
     fx_mag, fx_conc, fx_cancel, fx_fisher (−log p; 높을수록)
  D. 갱신 통계 (학습 중 집계 방어가 보는 것)
     upd_norm (높을수록), upd_cos_med (낮을수록), upd_krum (높을수록), upd_share (총노출 몫, 높을수록), upd_anti (가장 반대인 상대 코사인, 낮을수록), cum_cos_min (누적 기여 최소 코사인, 낮을수록)
  E. null-space 에너지 (게이트 발상): ns_energy (높을수록)
  F. 학습 중 부분합 감사 (M2): audit_z@t  라운드 t 의 전역 모델에서 누적 기여 ρ=0.2 제거의 flow z (높을수록)
  G. 참여 자연 실험 (M3, 참여율 < 1 일 때만): part_beta_neg (−min_c β_j(c), 높을수록 위장자), part_beta_pos (max_c β_j(c), 높을수록 운반자)

실행: python detect_bench.py --fmt fuba --name bd_iid_r100w5_k20paper_s0 --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --owner_client 19
출력: logs/detect/{name}.json  {"roles", "scores": {method: {client: value}}, "direction": {method: "high"|"low"}}
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
from cancel_fu import center_stack, conc_of, forward_select  # noqa: E402
from dose_fu import flow_stats, median_probs, robust_z  # noqa: E402
from effects import conformal_p, fisher, logp  # noqa: E402
from release_map import divergence, load_fuba, load_probe, make_model, present_clients, probs  # noqa: E402


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
    p.add_argument("--owner_client", type=int, default=-1, help="owner 조건의 소유자 (없으면 -1)")
    p.add_argument("--cond", default=None)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.1, 0.2, 0.5])
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--audit_at", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0], help="학습 중 감사 시점 (라운드 비율)")
    p.add_argument("--zmode", choices=["std", "mad"], default="std")
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse()
    if a.defenders is None:
        a.defenders = [a.defender]
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    factory, theta_T, U, flat, bd = load_fuba(a)
    K = len(U)
    C = a.n_classes
    clients = list(range(K))
    roles = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else ("owner" if j == a.owner_client else "benign"))) for j in clients}
    probe_loader, _ = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    N = xs.shape[0]
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    P0 = probs(m0, xs)
    pred0 = P0.argmax(1)
    L0 = logp(m0, xs)

    def model(theta):
        return make_model(factory, flat.to_sd(theta), device)

    scores, direction = {}, {}

    def put(name, d, dirn):
        scores[name] = {int(j): (None if v is None else round(float(v), 6)) for j, v in d.items()}
        direction[name] = dirn

    # ---------- A, B: 언러닝 시뮬레이션 (절대 / 상대)
    def sim_block(theta_base, Uvec, tag, rhos):
        P_base = probs(model(theta_base), xs); pred_b = P_base.argmax(1)
        for rho in rhos:
            P = {j: probs(model(theta_base - rho * Uvec[j]), xs) for j in clients}
            chg = {j: divergence(pred_b, P[j].argmax(1), C)["changed"] for j in clients}
            kl = {j: float((P_base * (torch.log(P_base.clamp_min(1e-12)) - torch.log(P[j].clamp_min(1e-12)))).sum(1).mean()) for j in clients}
            conc = {j: divergence(pred_b, P[j].argmax(1), C)["conc"] for j in clients}
            put(f"sim_abs_changed@{rho}{tag}", chg, "high"); put(f"sim_abs_kl@{rho}{tag}", kl, "high"); put(f"sim_abs_conc@{rho}{tag}", conc, "high")
            healthy = [j for j in clients if chg[j] <= 0.3]
            s_flow, s_mass = {}, {}
            for j in clients:
                rest = [P[i] for i in healthy if i != j]
                if len(rest) < 2:
                    s_flow[j] = s_mass[j] = None; continue
                pr = median_probs(rest)
                s_flow[j] = flow_stats(P[j], pr, C)["flow"]
                s_mass[j] = divergence(pr.argmax(1), P[j].argmax(1), C)["mass"]
            zf = {j: (robust_z(s_flow[j], [s_flow[i] for i in clients if i != j and s_flow[i] is not None], a.zmode) if s_flow[j] is not None else None) for j in clients}
            zm = {j: (robust_z(s_mass[j], [s_mass[i] for i in clients if i != j and s_mass[i] is not None], a.zmode) if s_mass[j] is not None else None) for j in clients}
            put(f"sim_rel_flow_z@{rho}{tag}", zf, "high"); put(f"sim_rel_mass_z@{rho}{tag}", zm, "high")
            put(f"sim_health_changed@{rho}{tag}", chg, "high")
    sim_block(tT, U, "", a.rhos)

    # ---------- C: 기능 효과
    E = {j: (logp(model(tT - a.eps * U[j]), xs) - L0) / a.eps for j in clients}
    Ec = dict(zip(clients, center_stack([E[j] for j in clients], "median")))
    V = np.stack([Ec[j].flatten().double().numpy() for j in clients]); G = V @ V.T
    yh = L0.argmax(1)
    mag = {j: float(np.sqrt(G[j, j] / N)) for j in clients}
    conc = {j: conc_of(Ec[j], yh, C)["score"] for j in clients}
    canc = {j: forward_select(G, G[j], float(G[j, j]), [i for i in clients if i != j], 4, 2.0, 0.02)[0] for j in clients}
    put("fx_mag", mag, "high"); put("fx_conc", conc, "high"); put("fx_cancel", canc, "high")
    fis = {}
    for j in clients:
        ps = [conformal_p(s[j], [s[i] for i in clients if i != j]) for s in (mag, conc, canc)]
        fis[j] = -math.log(max(fisher(ps)[1], 1e-12))
    put("fx_fisher", fis, "high")

    # ---------- D: 갱신 통계 + F: 학습 중 감사 + G: 참여 회귀 (라운드 순회 한 번)
    _, rounds = load_rounds(a)
    T = len(rounds)
    f_ = max(1, K // 4)
    norms, cosmed, krum, share, anti = {j: [] for j in clients}, {j: [] for j in clients}, {j: [] for j in clients}, {j: [] for j in clients}, {j: [] for j in clients}
    Ucum = {j: torch.zeros(flat.dim) for j in clients}
    audit_rounds = sorted(set(max(1, int(round(fr * T))) for fr in a.audit_at))
    audit = {}
    part = []            # 참여 지표 (T × K)
    ylog = []            # 라운드별 클래스 로짓 평균 변화 (T × C)
    prev_logit_mean = None
    for t, (g_prev, locals_) in enumerate(rounds, start=1):
        gv = flat.flatten(g_prev)
        g_next = rounds[t][0] if t < T else theta_T                  # 그 라운드의 집계 결과 (결석 자리에 이것이 저장돼 있다)
        present = [j for j in present_clients(locals_, g_next) if j < K]
        u = {j: (flat.flatten(locals_[j]) - gv) / max(len(present), 1) for j in present}
        part.append([1 if j in present else 0 for j in clients])
        for j in present:
            Ucum[j] = Ucum[j] + u[j]
        if len(present) >= 3:
            M = torch.stack([u[j] for j in present]); med = M.median(0).values
            D = torch.cdist(M, M) ** 2
            g = float(sum(float(u[j].norm()) for j in present))
            for i, j in enumerate(present):
                norms[j].append(float(u[j].norm())); cosmed[j].append(float(F.cosine_similarity(u[j], med, dim=0)))
                dd = D[i].clone(); dd[i] = float("inf"); krum[j].append(float(torch.topk(dd, max(1, len(present) - f_ - 2), largest=False).values.sum()))
                share[j].append(float(u[j].norm()) / g)
                anti[j].append(min(float(F.cosine_similarity(u[j], u[i2], dim=0)) for i2 in present if i2 != j))
        # 전역 모델 t = g_prev + Σ u (저장된 다음 라운드 전역과 같음); 로짓 평균 변화는 다음 g_prev 로 잰다
        theta_t = gv + sum(u.values()) if u else gv
        if t in audit_rounds:
            scoresA = {}
            P = {j: probs(model(theta_t - 0.2 * Ucum[j]), xs) for j in clients}
            pred_t = probs(model(theta_t), xs).argmax(1)
            chg = {j: divergence(pred_t, P[j].argmax(1), C)["changed"] for j in clients}
            healthy = [j for j in clients if chg[j] <= 0.3]
            s = {}
            for j in clients:
                rest = [P[i] for i in healthy if i != j]
                s[j] = flow_stats(P[j], median_probs(rest), C)["flow"] if len(rest) >= 2 else None
            audit[t] = {j: (robust_z(s[j], [s[i] for i in clients if i != j and s[i] is not None], a.zmode) if s[j] is not None else None) for j in clients}
        lm = logp(model(theta_t), xs).mean(0)            # C
        if prev_logit_mean is not None:
            ylog.append((lm - prev_logit_mean).numpy())
        prev_logit_mean = lm
    put("upd_norm", {j: np.mean(norms[j]) if norms[j] else None for j in clients}, "high")
    put("upd_cos_med", {j: np.mean(cosmed[j]) if cosmed[j] else None for j in clients}, "low")
    put("upd_krum", {j: np.mean(krum[j]) if krum[j] else None for j in clients}, "high")
    put("upd_share", {j: np.mean(share[j]) if share[j] else None for j in clients}, "high")
    put("upd_anti", {j: np.mean(anti[j]) if anti[j] else None for j in clients}, "low")
    cum_cos = {j: min(float(F.cosine_similarity(U[j], U[i], dim=0)) for i in clients if i != j) for j in clients}
    put("cum_cos_min", cum_cos, "low")
    for t, z in audit.items():
        put(f"audit_z@r{t}", z, "high")
    part = np.array(part); ylog = np.array(ylog)
    if part.shape[0] >= 2 and part.min() == 0 and ylog.shape[0] == part.shape[0] - 1:
        Xr = np.hstack([np.ones((ylog.shape[0], 1)), part[1:]])          # 라운드 t 의 참여가 θ_{t} − θ_{t−1} 을 만든다
        beta, *_ = np.linalg.lstsq(Xr, ylog, rcond=None)                  # (K+1) × C
        B = beta[1:]
        put("part_beta_neg", {j: -float(B[j].min()) for j in clients}, "high")
        put("part_beta_pos", {j: float(B[j].max()) for j in clients}, "high")
        scores["part_beta_class"] = {int(j): int(B[j].argmin()) for j in clients}; direction["part_beta_class"] = "info"
    # ---------- E: null-space 에너지 (clean 특징 공분산 99 %)
    try:
        layers = {n: m for n, m in m0.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))}
        feats, hooks = {}, []
        for n, m in layers.items():
            def mk(n, m):
                def h(mod, inp, out):
                    x = inp[0].detach()
                    if isinstance(mod, torch.nn.Conv2d):
                        x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride).transpose(1, 2).reshape(-1, mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1])
                    feats.setdefault(n, []).append(x.cpu())
                return h
            hooks.append(m.register_forward_hook(mk(n, m)))
        with torch.no_grad():
            for i in range(0, N, 256):
                m0(xs[i:i + 256])
        for h in hooks:
            h.remove()
        Pmat = {}
        for n, L in feats.items():
            X = torch.cat(L).double(); ev, Vv = torch.linalg.eigh(X.T @ X / X.shape[0]); ev, Vv = ev.flip(0), Vv.flip(1)
            r = int(((ev.cumsum(0) / ev.sum()) < 0.99).sum()) + 1
            Pmat[n] = Vv[:, :r] @ Vv[:, :r].T
        ns = {}
        for j in clients:
            num = den = 0.0
            for n in layers:
                key = n + ".weight"; sl = flat.slices[key]
                W = U[j][sl].view(flat.shapes[key]).double().reshape(flat.shapes[key][0], -1)
                Wp = W @ Pmat[n]; num += float(((W - Wp) ** 2).sum()); den += float((W ** 2).sum())
            ns[j] = num / max(den, 1e-12)
        put("ns_energy", ns, "high")
    except Exception as e:  # noqa: BLE001
        scores["ns_energy_error"] = str(e)

    res = {"name": a.name, "cond": a.cond, "K": K, "rounds": T, "roles": roles, "owner_client": a.owner_client,
           "participation": (float(part.mean()) if part.size else 1.0), "scores": scores, "direction": direction, "seconds": round(time.time() - t0, 1)}
    out = a.out or f"logs/detect/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    # 콘솔: 위장자 순위
    sup = a.defenders[0]
    print(f"[{a.name}] K={K} T={T} 참여율 {res['participation']:.2f}  ({res['seconds']}s)")
    print(f"{'방법':28s} {'위장자 값':>10s} {'순위':>4s}  정상 최대 / 소유자")
    for mth, d in scores.items():
        if not isinstance(d, dict) or direction.get(mth) == "info":
            continue
        vals = {j: v for j, v in d.items() if v is not None}
        if sup not in vals:
            continue
        sign = 1 if direction[mth] == "high" else -1
        rank = 1 + sum(1 for j, v in vals.items() if j != sup and sign * v > sign * vals[sup])
        ben = [v for j, v in vals.items() if roles[j] == "benign"]
        own = vals.get(a.owner_client)
        print(f"{mth:28s} {vals[sup]:10.3f} {rank:4d}  {(max(ben) if sign > 0 else min(ben)) if ben else float('nan'):8.3f} / {own if own is not None else '-'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
