# -*- coding: utf-8 -*-
"""동적 평형의 지문 (학습 없음, 역할 정보 없이 계산): 잠복 백도어는 매 라운드 유지되는 평형이고, 유지에는 시간 위의 흔적이 남는다.
저장된 라운드별 갱신과 전역 모델, 참여 명단만 쓴다. 공개 입력 X_p 위의 '클래스별 로그확률 평균 변화' (기능 효과의 클래스 요약) 로 잰다.

  e_j^t(c) = mean_x [ f_c(x; θ_{t−1} + u_j^t) − f_c(x; θ_{t−1}) ]     (j 의 갱신이 클래스 c 를 미는 양, 참여 라운드만)
  D_j^t(c) = mean_x [ f_c(x; θ_{t−1}) − f_c(x; θ_{last_j}) ]           (j 가 결석한 동안 전역 모델이 c 쪽으로 흘러간 양)

지문 1 복원력:    j 의 c 밀기 |e_j^t(c)| 가 결석 길이(마지막 참여 후 라운드 수)에 비례하는가 — 기울기 slope_j(c)
지문 2 지연 반작용: e_j^t(c) 와 D_j^t(c) 의 상관 — 위장자는 흘러간 만큼 되밀므로 강한 음수, 정상은 0 근처
지문 3 균형식:    누적 밀기 P_j(c) = Σ_t e_j^t(c). 평형이면 Σ_j P_j(c*) ≈ 0 인데 개별 |P_j| 는 크다 (위장자 큰 음수, 운반자 양수)
클래스 c* 는 서버가 모른다 → 지문 3 의 누적 밀기 분산이 가장 큰 클래스를 고른다 (정답과 맞는지 채점에만 쓴다).

실행: python analysis_equilibrium.py --fmt fuba --name bd_iid_r100w5_k20paper_s0 --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8
집계: python analysis_equilibrium.py --aggregate --cond iid_r100w5_k20paper
출력: logs/equilibrium/{name}.json, logs/equilibrium/summary_{cond}.md
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from baseline_masa import load_rounds  # noqa: E402
from effects import logp  # noqa: E402
from release_map import load_fuba, load_probe, make_model, present_clients  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba"], default="fuba")
    p.add_argument("--name", default=None)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--traj_dir", default=None)
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--defenders", type=int, nargs="*", default=None)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--owner_client", type=int, default=-1)
    p.add_argument("--target", type=int, default=8, help="채점용 (서버는 모름)")
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--cond", default=None)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--n_gpu", type=int, default=1)
    return p.parse_args()


def rz(x, others):
    o = np.asarray([v for v in others if v is not None], dtype=float)
    if x is None or o.size < 2:
        return None
    sd = float(np.std(o, ddof=1))
    return float((x - np.median(o)) / sd) if sd > 1e-12 else None


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or x.std() < 1e-12:
        return None
    return float(np.polyfit(x, y, 1)[0])


def analyze(a):
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    if a.defenders is None:
        a.defenders = [a.defender]
    factory, theta_T, U, flat, bd = load_fuba(a)
    K = len(U); C = a.n_classes; clients = list(range(K))
    roles = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else ("owner" if j == a.owner_client else "benign"))) for j in clients}
    probe_loader, _ = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)

    def model(theta):
        return make_model(factory, flat.to_sd(theta), device)

    _, rounds = load_rounds(a)
    T = len(rounds)
    e = {j: [] for j in clients}          # (t, C-vector) 참여 라운드의 밀기
    D = {j: [] for j in clients}          # (t, C-vector) 결석 동안의 흐름
    gap = {j: [] for j in clients}        # (t, 결석 길이)
    last_seen = {j: None for j in clients}
    L_hist = []                            # 라운드별 전역 로그확률 평균 (C)
    for t, (g_prev, locals_) in enumerate(rounds, start=1):
        gv = flat.flatten(g_prev)
        g_next = rounds[t][0] if t < T else theta_T
        pres = [j for j in present_clients(locals_, g_next) if j < K]
        L_prev = logp(model(gv), xs).mean(0).numpy()
        L_hist.append(L_prev)
        for j in pres:
            u = (flat.flatten(locals_[j]) - gv) / max(len(pres), 1)
            Lj = logp(model(gv + u), xs).mean(0).numpy()
            e[j].append((t, Lj - L_prev))
            if last_seen[j] is not None:
                D[j].append((t, L_prev - L_hist[last_seen[j] - 1]))
                gap[j].append((t, t - last_seen[j]))
            else:
                D[j].append((t, np.zeros(C))); gap[j].append((t, 0))
            last_seen[j] = t
    # 지문 3: 누적 밀기, 클래스 선택
    P = {j: (np.sum([v for _, v in e[j]], axis=0) if e[j] else np.zeros(C)) for j in clients}
    Pm = np.stack([P[j] for j in clients])
    c_hat = int(np.argmax(Pm.var(0)))
    net = {c: float(Pm[:, c].sum()) for c in range(C)}
    gross = {c: float(np.abs(Pm[:, c]).sum()) for c in range(C)}
    f3 = {j: float(P[j][c_hat]) for j in clients}
    # 지문 1·2 (추정 클래스 c_hat 에서; 채점용으로 진짜 타깃에서도)
    def f12(c):
        s1, s2 = {}, {}
        for j in clients:
            if len(e[j]) < 3:
                s1[j] = s2[j] = None; continue
            ev = np.array([v[c] for _, v in e[j]]); gv_ = np.array([g for _, g in gap[j]]); dv = np.array([d[c] for _, d in D[j]])
            s1[j] = slope(gv_, np.abs(ev))            # 결석 길이 → 밀기 세기
            # 지연 반작용은 '그 클래스에만' 있어야 한다: 모든 클래스에 있는 되밀기(일반 평균 회귀)는 정상도 가진다 → 다른 클래스 평균을 뺀다
            cc = corr(ev, dv)
            others_c = [corr(np.array([v[k] for _, v in e[j]]), np.array([d[k] for _, d in D[j]])) for k in range(C) if k != c]
            others_c = [x for x in others_c if x is not None]
            s2[j] = (cc - float(np.mean(others_c))) if (cc is not None and others_c) else None
        return s1, s2
    f1, f2 = f12(c_hat)
    f1_true, f2_true = f12(a.target)
    # ---- 균형 장부: 클래스별 헤지 비율, 음의 다리, 양의 다리 집합
    def mad_z(vals):
        v = np.asarray(vals, float); med = float(np.median(v)); mad = 1.4826 * float(np.median(np.abs(v - med)))
        return [(float(x) - med) / mad if mad > 1e-9 else 0.0 for x in v], med, mad
    ledger = {}
    for c in range(C):
        col = [float(P[j][c]) for j in clients]
        z, med, mad = mad_z(col)
        neg = int(np.argmin(col)); pos = [clients[i] for i in range(K) if z[i] > 2.0 and i != neg]     # 양의 다리: MAD z > 2
        ledger[c] = {"H": round(1.0 - abs(sum(col)) / max(sum(abs(x) for x in col), 1e-9), 4), "neg": neg, "z_neg": round(z[neg], 2),
                     "neg_role": roles[neg], "pos": pos, "pos_roles": [roles[j] for j in pos], "n_pos": len(pos)}
    hedge_classes = [c for c in range(C) if ledger[c]["H"] >= 0.8 and ledger[c]["z_neg"] <= -3.0]
    c_hedge = max(range(C), key=lambda c: ledger[c]["H"] if ledger[c]["z_neg"] <= -3.0 else -1)   # 헤지 기준 타깃 추정 (분산 기준 c_hat 과 별도 보고)
    # 라운드별 H(c) 곡선 (진짜 타깃과 추정 클래스): 몇 라운드부터 평형이 보이는가
    def H_curve(c):
        run = {j: 0.0 for j in clients}; out = []
        ev_by_t = {}
        for j in clients:
            for t, v in e[j]:
                ev_by_t.setdefault(t, []).append((j, v[c]))
        for t in range(1, T + 1):
            for j, v in ev_by_t.get(t, []):
                run[j] += v
            col = list(run.values()); g = sum(abs(x) for x in col)
            out.append(round(float(1.0 - abs(sum(col)) / g), 3) if g > 1e-9 else None)
        return out
    res = {"name": a.name, "cond": a.cond, "K": K, "rounds": T, "roles": roles, "target_true": a.target, "class_hat": c_hat,
           "net_gross_ratio_hat": round(abs(net[c_hat]) / max(gross[c_hat], 1e-9), 4),
           "f1_restoring_slope": f1, "f2_lag_corr": f2, "f3_cum_push": f3,
           "f1_true": f1_true, "f2_true": f2_true, "f3_true": {j: float(P[j][a.target]) for j in clients},
           "n_rounds_present": {j: len(e[j]) for j in clients},
           "ledger": ledger, "hedge_classes": hedge_classes, "class_hedge": c_hedge, "H_curve_true": H_curve(a.target), "H_curve_hat": H_curve(c_hat)}
    os.makedirs("logs/equilibrium", exist_ok=True)
    out = f"logs/equilibrium/{a.name}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    sup = a.defenders[0]
    print(f"[{a.name}] K={K} T={T}  추정 타깃 {c_hat} (정답 {a.target})  순/총 밀기 비율 {res['net_gross_ratio_hat']}")
    print("지문 (추정 클래스): 클라 | 역할 | 1 복원력 기울기 | 2 지연 상관 | 3 누적 밀기 | 참여 라운드")
    for j in clients:
        print(f"  {j:2d} | {roles[j][:9]:9s} | {f1[j] if f1[j] is None else round(f1[j], 4)} | {f2[j] if f2[j] is None else round(f2[j], 3)} | {round(f3[j], 2)} | {len(e[j])}")
    ben = [j for j in clients if roles[j] == "benign"]
    for key, d, dirn in (("1 복원력", f1, "high"), ("2 지연 상관", f2, "low"), ("3 누적 밀기", f3, "low")):
        vals = {j: v for j, v in d.items() if v is not None}
        if sup in vals:
            sign = 1 if dirn == "high" else -1
            rank = 1 + sum(1 for j, v in vals.items() if j != sup and sign * v > sign * vals[sup])
            print(f"  위장자 {key}: 값 {vals[sup]:.3f} 순위 {rank}/{len(vals)}  z(정상 대비) {rz(vals[sup], [vals[j] for j in ben if j in vals])}")
    print("균형 장부 (클래스 | H | 음의 다리(역할, z) | 양의 다리(역할)):")
    for c in range(C):
        L_ = ledger[c]
        mark = " ← 헤지" if c in hedge_classes else ""
        print(f"  {c} | {L_['H']:.3f} | {L_['neg']} ({L_['neg_role'][:3]}, z {L_['z_neg']}) | {L_['pos']} {[r[:3] for r in L_['pos_roles']]}{mark}")
    hc = res["H_curve_true"]
    first = next((i + 1 for i, h in enumerate(hc) if h is not None and h >= 0.8), None)
    print(f"  H(진짜 타깃) 가 0.8 을 처음 넘는 라운드: {first} / {T}   헤지 기준 타깃 추정: {c_hedge} (분산 기준 {c_hat}, 정답 {a.target})")
    print(f"-> {out}")


def aggregate(a):
    files = sorted(glob.glob("logs/equilibrium/*.json"))
    runs = []
    for f in files:
        base = os.path.basename(f)[:-5]
        if a.cond and not re.match(rf"^bd_{re.escape(a.cond)}_s\d+$", base):
            continue
        runs.append(json.load(open(f, encoding="utf-8")))
    if not runs:
        raise SystemExit("결과 없음")
    L = [f"# 동적 평형 지문 요약: {a.cond}  n={len(runs)}", "",
         "역할 정보 없이 저장 전역 모델·갱신·참여 명단만으로 계산. 위장자 순위(1 = 가장 의심), 정상 대비 z, 소유자 z, 운반자 평균 순위.", "",
         "| 지문 | 방향 | 위장자 순위 | 잡힘 | 위장자 z | 소유자 z | 운반자 평균 순위 | 추정 타깃 = 정답 |", "|---|---|---|---|---|---|---|---|"]
    for key, field, dirn in (("1 복원력 (결석 길이→밀기 세기)", "f1_restoring_slope", "high"), ("2 지연 반작용 (타깃 클래스만의 되밀기 상관, 다른 클래스 평균 대비)", "f2_lag_corr", "low"), ("3 누적 밀기 (추정 타깃)", "f3_cum_push", "low")):
        ranks, zs, zo, cr, hit, caught = [], [], [], [], 0, 0
        for r in runs:
            roles = {int(k): v for k, v in r["roles"].items()}
            d = {int(k): v for k, v in r[field].items() if v is not None}
            sup = [j for j, ro in roles.items() if ro == "suppressor"][0]
            if sup not in d:
                continue
            sign = 1 if dirn == "high" else -1
            vals = {j: sign * v for j, v in d.items()}
            rank = 1 + sum(1 for j, v in vals.items() if j != sup and v > vals[sup]); ranks.append(rank); caught += rank == 1
            ben = [vals[j] for j in vals if roles[j] == "benign"]
            zs.append(rz(vals[sup], ben))
            own = [j for j, ro in roles.items() if ro == "owner" and j in vals]
            zo.append(rz(vals[own[0]], ben) if own else None)
            car = [j for j, ro in roles.items() if ro == "carrier" and j in vals]
            cr.append(np.mean([1 + sum(1 for j, v in vals.items() if j != c and v > vals[c]) for c in car]) if car else None)
            hit += r["class_hat"] == r["target_true"]
        def ms(xs, nd=2):
            xs = [x for x in xs if x is not None]
            return "-" if not xs else (f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}")
        L.append(f"| {key} | {dirn} | {ms(ranks, 1)} | {caught}/{len(ranks)} | {ms(zs, 1)} | {ms(zo, 1)} | {ms(cr, 1)} | {hit}/{len(runs)} |")
    L += ["", "순/총 밀기 비율 (추정 타깃, 작을수록 '합은 0 인데 개별은 큼' = 평형): " + ", ".join(f"{r['name'].split('_s')[-1]}: {r['net_gross_ratio_hat']}" for r in runs)]
    if all("ledger" in r for r in runs):
        L += ["", "## 균형 장부 (시드별): 헤지 클래스 → 음의 다리(역할) / 양의 다리 수(역할)", ""]
        for r in runs:
            items = []
            for c in r["hedge_classes"]:
                e_ = r["ledger"][str(c)]
                items.append(f"{c}: 음 {e_['neg']}({e_['neg_role'][:3]}, z {e_['z_neg']}) / 양 {e_['n_pos']}명 {[x[:3] for x in e_['pos_roles']]}")
            hc = r.get("H_curve_true", []); first = next((i + 1 for i, h in enumerate(hc) if h is not None and h >= 0.8), None)
            L.append(f"- {r['name']}: " + ("; ".join(items) if items else "헤지 클래스 없음") + f" | H(타깃) ≥ 0.8 첫 라운드 {first} | 헤지 기준 타깃 추정 {r.get('class_hedge')} (정답 {r['target_true']})")
    out = f"logs/equilibrium/summary_{a.cond or 'all'}.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print(f"-> {out}")


if __name__ == "__main__":
    a = parse()
    if a.aggregate:
        aggregate(a)
    else:
        analyze(a)
