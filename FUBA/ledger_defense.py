# -*- coding: utf-8 -*-
"""균형 장부 방어 (Balance Ledger) — 학습 중 온라인 판정 (ALGORITHMIZATION.md §3).

서버가 가진 것만 쓴다: 라운드별 전역 모델, 참여자의 갱신, 공개 입력 몇 장. 역할 정보는 채점에만 쓴다.
라운드 t 마다
  장부   P_j(c) += mean_x [ log p_c(x; θ_{t-1} + u_j^t) − log p_c(x; θ_{t-1}) ]     (참여자 j 만, u = 집계 가중 갱신)
  헤지   H(c) = 1 − |Σ_j P_j(c)| / Σ_j |P_j(c)|
  다리   neg(c) = argmin_j P_j(c), z_neg = MAD z;  Pos(c) = { j ≠ neg : z_j > κ }
  판정   헤지 클래스 ⇔ H(c) ≥ h0 ∧ z_neg(c) ≤ −τ.  |Pos(c)| ≥ m ⇒ 공모(운반자 = Pos, 위장자 = neg); |Pos| = 1 ⇒ 정상 큰손(손대지 않음)
  경보   같은 클래스가 persist 번 연속 공모로 판정되면 경보. 경보 라운드·제외 목록(= Pos)을 기록한다.
밀기 계산은 analysis_equilibrium.py 와 같다. 한 번 계산한 라운드별 밀기는 logs/ledger/{name}_push.npz 에 저장해 두고
문턱값만 바꿀 때는 --from_cache 로 다시 읽는다 (순전파 없음).

채점 (roles 주어질 때): 경보 유무·첫 경보 라운드, neg = 위장자?, Pos 와 운반자 집합의 정밀도/재현율, 소유자가 Pos 에 들어갔는가.
--aggregate --cond 로 시드 묶음 표를 만든다.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_masa import load_rounds  # noqa: E402
from effects import logp  # noqa: E402
from release_map import load_badfu, load_fuba, load_probe, make_model, present_clients, resolve_fuba_name  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba", "badfu"], default="fuba")
    p.add_argument("--dataset", default="mnist", help="fuba 궤적의 데이터셋: mnist(LeNet) | cifar10(resnet18)")
    p.add_argument("--traj_dir", default=None, help="badfu: round_r.pt / glob_r.pt 폴더")
    p.add_argument("--counts", type=int, nargs="*", default=None, help="badfu: 클라별 표본 수 (FedAvg 가중). 없으면 균등")
    p.add_argument("--neg_set", action="store_true", help="음의 다리를 한 명이 아니라 집합 Neg(c)={z ≤ −tau} 로 본다 (위장자 여러 명)")
    p.add_argument("--name", default=None)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--n_gpu", type=int, default=1)
    # 판정 문턱
    p.add_argument("--h0", type=float, default=0.8, help="헤지 비율 문턱")
    p.add_argument("--tau", type=float, default=3.0, help="음의 다리 MAD z 문턱 (≤ −tau)")
    p.add_argument("--kappa", type=float, default=2.0, help="양의 다리 MAD z 문턱 (> kappa)")
    p.add_argument("--min_pos", type=int, default=2, help="공모로 보는 양의 다리 최소 수")
    p.add_argument("--max_share", type=float, default=0.6, help="양의 다리 중 최대 1 명의 몫(양의 합 대비) 상한. 넘으면 공모가 아니라 큰손 1 명으로 본다. 1.0 = 끔")
    p.add_argument("--persist", type=int, default=3, help="연속 몇 번 공모 판정이면 경보")
    p.add_argument("--start", type=int, default=5, help="판정을 시작하는 라운드 (장부가 비어 있을 때 MAD 가 불안정)")
    p.add_argument("--min_present", type=int, default=2, help="장부에 이만큼 참여한 클라만 다리 후보")
    p.add_argument("--from_cache", action="store_true", help="저장된 밀기(npz)만 읽어 판정 (순전파 없음)")
    # 채점용 (서버는 모름)
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--defenders", type=int, nargs="*", default=None)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--owner_client", type=int, default=-1)
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--cond", default=None)
    p.add_argument("--aggregate", action="store_true")
    return p.parse_args()


def mad_z(v):
    v = np.asarray(v, float); med = float(np.median(v)); mad = 1.4826 * float(np.median(np.abs(v - med)))
    return (v - med) / mad if mad > 1e-9 else np.zeros_like(v)


def compute_push(a, device):
    """라운드별 (t, 참여자 j, C-벡터 밀기). 순전파: 라운드마다 참여자 수 + 1."""
    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    K = len(U)
    wts = ([c / sum(a.counts) for c in a.counts] if (a.fmt == "badfu" and a.counts) else None)   # badfu: 표본 수 가중
    probe_loader, _ = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    ts, js, vs = [], [], []

    if a.fmt == "fuba":
        name = resolve_fuba_name(a.name, a.ckpt_dir)
        T = a.rounds - 1
        with open(f"{a.ckpt_dir}/global_net_model_{name}_round_1.pkl", "rb") as f:
            prev_g = pickle.load(f)
        prev_gv = flat.flatten(prev_g)

        for t in range(1, T + 1):
            r = t + 1
            with open(f"{a.ckpt_dir}/local_net_model_{name}_round_{r}.pkl", "rb") as f:
                locals_ = pickle.load(f)
            if r < a.rounds:
                with open(f"{a.ckpt_dir}/global_net_model_{name}_round_{r}.pkl", "rb") as f:
                    next_g = pickle.load(f)
            else:
                next_g = theta_T

            pres = [j for j in present_clients(locals_, next_g) if j < K]
            L_prev = logp(make_model(factory, flat.to_sd(prev_gv), device), xs).mean(0).numpy()
            for j in pres:
                u = (flat.flatten(locals_[j]) - prev_gv) * (wts[j] if wts else 1.0 / max(len(pres), 1))
                Lj = logp(make_model(factory, flat.to_sd(prev_gv + u), device), xs).mean(0).numpy()
                ts.append(t); js.append(j); vs.append(Lj - L_prev)

            prev_gv = flat.flatten(next_g)
            if t % 10 == 0:
                print(f"  밀기 계산 {t}/{T}", flush=True)
    else:
        _, rounds = load_rounds(a)
        T = len(rounds)
        for t, (g_prev, locals_) in enumerate(rounds, start=1):
            gv = flat.flatten(g_prev)
            g_next = rounds[t][0] if t < T else theta_T
            pres = [j for j in present_clients(locals_, g_next) if j < K]
            L_prev = logp(make_model(factory, flat.to_sd(gv), device), xs).mean(0).numpy()
            for j in pres:
                u = (flat.flatten(locals_[j]) - gv) * (wts[j] if wts else 1.0 / max(len(pres), 1))
                Lj = logp(make_model(factory, flat.to_sd(gv + u), device), xs).mean(0).numpy()
                ts.append(t); js.append(j); vs.append(Lj - L_prev)
            if t % 10 == 0:
                print(f"  밀기 계산 {t}/{T}", flush=True)

    return K, T, np.array(ts), np.array(js), np.stack(vs)


def judge(P, n_present, a):
    """장부 한 장(P: K×C)에서 클래스별 판정."""
    K, C = P.shape
    elig = np.array([n_present[j] >= a.min_present for j in range(K)])
    out = {}
    for c in range(C):
        col = P[:, c]
        z = mad_z(col[elig]) if elig.sum() >= 3 else np.zeros(int(elig.sum()))
        idx = np.where(elig)[0]
        if len(idx) == 0:
            out[c] = {"H": 0.0, "neg": -1, "negs": [], "z_neg": 0.0, "pos": [], "top_share": 1.0, "hedge": False, "coalition": False, "kind": ""}; continue
        zi = {int(j): float(z[i]) for i, j in enumerate(idx)}
        neg = int(idx[np.argmin(col[idx])])
        negs = [int(j) for j in idx if zi[int(j)] <= -a.tau] if a.neg_set else [neg]      # 음의 다리 (집합 옵션)
        if neg not in negs:
            negs.append(neg)
        pos = [int(j) for j in idx if j not in negs and zi[int(j)] > a.kappa]
        H = 1.0 - abs(col[idx].sum()) / max(np.abs(col[idx]).sum(), 1e-9)
        balanced = bool(H >= a.h0 and zi[neg] <= -a.tau)          # 합은 0, 음의 다리는 뚜렷
        # 라벨: 공모 = 양의 다리 여럿 / 정상 큰손 = 양의 다리 하나 / 외톨이 음 = 양의 다리 없음 (전역 흐름을 혼자 되미는 것; 보고만 하고 손대지 않음)
        # 공모의 추가 조건: 양의 다리 중 한 명이 양의 합의 max_share 를 넘으면 "한 명이 큰손 + 들러리" 로 보아 공모가 아니다 (owner 조건에서 소유자 + 정상 한 명이 같이 걸리는 것을 막는다)
        pos_sum = float(sum(max(col[j], 0.0) for j in pos)) if pos else 0.0
        top_share = (max(col[j] for j in pos) / pos_sum) if pos_sum > 1e-9 else 1.0
        spread = len(pos) >= a.min_pos and top_share <= a.max_share
        kind = ("coalition" if spread else ("big_legit" if len(pos) >= 1 else "lone_neg")) if balanced else ""
        out[c] = {"H": round(float(H), 4), "neg": neg, "negs": negs, "z_neg": round(zi[neg], 2), "pos": pos, "top_share": round(float(top_share), 3),
                  "hedge": balanced and len(pos) >= 1, "coalition": kind == "coalition", "kind": kind}
    return out


def run(a):
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    if a.defenders is None:
        a.defenders = [a.defender]
    os.makedirs("logs/ledger", exist_ok=True)
    cache = f"logs/ledger/{a.name}_push.npz"
    if a.from_cache and os.path.exists(cache):
        d = np.load(cache); K, T, ts, js, vs = int(d["K"]), int(d["T"]), d["ts"], d["js"], d["vs"]
    else:
        K, T, ts, js, vs = compute_push(a, device)
        np.savez(cache, K=K, T=T, ts=ts, js=js, vs=vs)
    C = vs.shape[1]
    roles = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else ("owner" if j == a.owner_client else "benign"))) for j in range(K)}
    # ---- 온라인 재생: 라운드마다 장부 갱신 → 판정 → 경보
    P = np.zeros((K, C)); n_present = np.zeros(K, int)
    streak = np.zeros(C, int); alarm = None; log = []; H_curve = []
    for t in range(1, T + 1):
        m = ts == t
        for j, v in zip(js[m], vs[m]):
            P[j] += v; n_present[j] += 1
        verdict = judge(P, n_present, a)
        H_curve.append({c: verdict[c]["H"] for c in range(C)})
        if t < a.start:
            continue
        for c in range(C):
            streak[c] = streak[c] + 1 if verdict[c]["coalition"] else 0
        flagged = [c for c in range(C) if verdict[c]["coalition"]]
        big_legit = [c for c in range(C) if verdict[c]["hedge"] and not verdict[c]["coalition"]]
        log.append({"t": t, "coalition": flagged, "big_legit": big_legit,
                    "detail": {c: {k: verdict[c][k] for k in ("H", "neg", "z_neg", "pos")} for c in flagged + big_legit}})
        if alarm is None and any(streak[c] >= a.persist for c in range(C)):
            c_al = int(max(range(C), key=lambda c: (streak[c] >= a.persist, verdict[c]["H"])))
            union = sorted({j for rec in log[-a.persist:] for j in rec["detail"].get(c_al, {}).get("pos", [])})   # 연속 판정 동안 한 번이라도 양의 다리였던 클라
            alarm = {"round": t, "class": c_al, "suppressor_hat": verdict[c_al]["neg"], "suppressors_hat": verdict[c_al]["negs"], "carriers_hat": verdict[c_al]["pos"],
                     "carriers_union": union, "H": verdict[c_al]["H"], "z_neg": verdict[c_al]["z_neg"]}
    final = judge(P, n_present, a)
    # ---- 채점 (역할은 여기서만)
    sup = a.defenders[0]; carriers = set(a.attackers)
    sups = set(a.defenders)
    score = {"alarm": alarm is not None, "alarm_round": alarm["round"] if alarm else None,
             "class_ok": (alarm["class"] == a.target) if alarm else None,
             "suppressor_ok": (alarm["suppressor_hat"] == sup) if alarm else None,
             "suppressors_recall": (len(set(alarm["suppressors_hat"]) & sups) / max(len(sups), 1)) if alarm else None,
             "carrier_precision": (len(set(alarm["carriers_hat"]) & carriers) / max(len(alarm["carriers_hat"]), 1)) if alarm else None,
             "carrier_recall": (len(set(alarm["carriers_hat"]) & carriers) / max(len(carriers), 1)) if alarm else None,
             "owner_in_exclusion": (a.owner_client in alarm["carriers_hat"]) if (alarm and a.owner_client >= 0) else False,
             "n_coalition_classes_final": sum(1 for c in range(C) if final[c]["coalition"]),
             "n_big_legit_classes_final": sum(1 for c in range(C) if final[c]["kind"] == "big_legit"),
             "n_lone_neg_classes_final": sum(1 for c in range(C) if final[c]["kind"] == "lone_neg"),
             "false_alarm_classes": [c for c in range(C) if final[c]["coalition"] and c != a.target]}
    exclusion = sorted(set(alarm["carriers_hat"])) if alarm else []
    # 되감기 라운드: 경보 클래스의 H 가 경보 전 마지막으로 h0 아래였던 라운드 (그 체크포인트에서 공모자 제외로 이어 학습)
    rollback = None
    if alarm:
        below = [t + 1 for t, hc in enumerate(H_curve[:alarm["round"]]) if hc[alarm["class"]] < a.h0]
        rollback = max(below) if below else a.start
    res = {"name": a.name, "cond": a.cond, "K": K, "rounds": T, "params": {k: getattr(a, k) for k in ("h0", "tau", "kappa", "min_pos", "persist", "start", "min_present")},
           "roles": roles, "target_true": a.target, "alarm": alarm, "exclusion": exclusion, "rollback_round": rollback, "score": score,
           "final": {c: final[c] for c in range(C)}, "H_curve": H_curve, "log": log,
           "n_present": {j: int(n_present[j]) for j in range(K)}}
    out = f"logs/ledger/{a.name}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    # ---- 출력
    print(f"[{a.name}] K={K} T={T}  문턱 h0 {a.h0} tau {a.tau} kappa {a.kappa} min_pos {a.min_pos} persist {a.persist}")
    print("최종 장부 (클래스 | H | 음의 다리(역할, z) | 양의 다리(역할) | 판정):")
    for c in range(C):
        f = final[c]
        tag = {"coalition": "공모", "big_legit": "정상 큰손", "lone_neg": "외톨이 음", "": ""}[f["kind"]]
        print(f"  {c} | {f['H']:.3f} | {f['neg']} ({roles.get(f['neg'], '?')[:3]}, z {f['z_neg']}) | {f['pos']} {[roles[j][:3] for j in f['pos']]} | {tag}")
    if alarm:
        print(f"경보: 라운드 {alarm['round']}/{T}, 클래스 {alarm['class']} (정답 {a.target}), 위장자 후보 {alarm['suppressor_hat']} ({roles[alarm['suppressor_hat']]}), "
              f"운반자 후보 {alarm['carriers_hat']} {[roles[j][:3] for j in alarm['carriers_hat']]}")
        print(f"대응: 라운드 {rollback} 의 전역 모델로 되감고 {sorted(set(exclusion) | {alarm['suppressor_hat']})} 를 제외하고 이어 학습 → "
              f"python benign_departure.py ... --methods retrain_benign --exclude {' '.join(map(str, exclusion))} --init_round {rollback}")
    else:
        print("경보 없음")
    print(f"채점: {json.dumps(score, ensure_ascii=False)}")
    print(f"-> {out}")


def aggregate(a):
    files = sorted(glob.glob(f"logs/ledger/*{a.cond}*.json"))
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    runs = [r for r in runs if "score" in r]
    if not runs:
        print("no runs"); return
    L = [f"# 균형 장부 방어 — {a.cond} ({len(runs)} 궤적)", "",
         "문턱: " + ", ".join(f"{k} {v}" for k, v in runs[0]["params"].items()), "",
         "| 궤적 | 경보 라운드 / T | 클래스 (정답) | 위장자 맞음 | 운반자 정밀도 / 재현율 | 소유자 제외됨 | 오경보 클래스 | 정상 큰손 클래스 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s = r["score"]; al = r["alarm"]
        L.append(f"| {r['name']} | {s['alarm_round']} / {r['rounds']} | {al['class'] if al else '-'} ({r['target_true']}) | {s['suppressor_ok']} | "
                 f"{s['carrier_precision']} / {s['carrier_recall']} | {s['owner_in_exclusion']} | {s['false_alarm_classes']} | {s['n_big_legit_classes_final']} |")
    n = len(runs)
    L += ["", f"요약: 경보 {sum(1 for r in runs if r['score']['alarm'])}/{n}, 클래스 맞음 {sum(1 for r in runs if r['score']['class_ok'])}/{n}, "
          f"위장자 맞음 {sum(1 for r in runs if r['score']['suppressor_ok'])}/{n}, 운반자 재현율 1.0 {sum(1 for r in runs if r['score']['carrier_recall'] == 1.0)}/{n}, "
          f"소유자 오제외 {sum(1 for r in runs if r['score']['owner_in_exclusion'])}/{n}, 오경보 있음 {sum(1 for r in runs if r['score']['false_alarm_classes'])}/{n}"]
    out = f"logs/ledger/summary_{a.cond}.md"
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print(f"-> {out}")


if __name__ == "__main__":
    a = parse()
    aggregate(a) if a.aggregate else run(a)
