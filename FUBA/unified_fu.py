# -*- coding: utf-8 -*-
"""하나의 절차 (docs/paper/METHOD.md): 궤적 + 요청자 정보로 FUBA·BadFU 를 같은 코드로 처리한다.

요청 (k, 지울 데이터) 마다
  A 거름망   궤적 지표 (라운드별 몫이 공개+지울 입력에서 클래스별 손실을 미는 방향): 크기·일관성·한쪽 몰림·반대 짝.
             어느 것도 문턱을 안 넘으면 → **수락** (θ' = θ_T − U_k, 기본 FU 와 동일, 추가 비용 없음)
  B 확정     요청자를 (모두가 견디는 양만큼) 빼고, 남들을 같은 양 뺀 모델들과 '공개 + 지울 입력' 위에서 비교 (통계 flow).
             띠 안이면 → **수락**. 무너지면 → **거부**.
  C 정화     걸리면: 활성화 입력과 타깃 클래스에서 방향 h 를 구하고, 운반자 몫의 h 성분만 λ 만큼 뺀다 (요청자 몫 복원 없음).
             띠 안으로 들어오면 → **정화 후 수락**. 안 되면 재억제 수정(KD) → 그래도 안 되면 **거부**.
출력: 요청별 결정(accept / purified / repaired / reject), 각 단계의 수치, 비용, 그리고 채점(ASR/ACC/α).

실행 예
  python unified_fu.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requesters 4 7 5
  python unified_fu.py --fmt badfu --name loc_badfu --traj_dir ... --defender 5 --attackers 0 --requesters 5 1 2 --op kd
출력: logs/unified/{name}.json
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

from antidote import grad_logp  # noqa: E402
from baseline_masa import load_rounds  # noqa: E402
from dose_fu import DELTAS, Z_UNDECIDABLE, Engine, fmt_step, retain_change, robust_z, seed_all  # noqa: E402
from release_map import (Flat, acc_of, divergence, kd_repair, load_badfu, load_fuba, load_probe,  # noqa: E402
                         make_model, make_scorer, probs)
from traj_indicators import class_grads, indicators, zscores  # noqa: E402

KEYS = ["mag", "consist", "spread", "cancel", "cancel_set"]


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
    p.add_argument("--requesters", type=int, nargs="+", default=[4, 7, 5])
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--defenders", type=int, nargs="*", default=None)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--probe_mode", choices=["clean", "forget"], default="forget")
    p.add_argument("--forget_n", type=int, default=500)
    # A
    p.add_argument("--screen", choices=["on", "off"], default="on")
    p.add_argument("--tau_screen", type=float, default=2.5, help="거름망 문턱 (보수적으로 낮게)")
    # B
    p.add_argument("--stat", choices=["mass", "changed", "shift", "flow"], default="flow")
    p.add_argument("--cap", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--tau_mode", choices=["fixed", "loo"], default="fixed")
    p.add_argument("--tau_margin", type=float, default=1.0)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--zmode", choices=["std", "mad", "iqr"], default="std")
    p.add_argument("--delta_min", type=float, default=0.1)
    p.add_argument("--op", choices=["subtract", "kd"], default="subtract")
    p.add_argument("--kd_steps", type=int, default=30)
    p.add_argument("--kd_lr", type=float, default=None)
    # C
    p.add_argument("--lams", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    p.add_argument("--min_act", type=int, default=10)
    p.add_argument("--rep_steps", type=int, default=50)
    p.add_argument("--rep_lr", type=float, default=None)
    p.add_argument("--rep_bs", type=int, default=128)
    p.add_argument("--rep_every", type=int, default=0)
    p.add_argument("--rep_stop", type=float, default=1.0)
    # misc
    p.add_argument("--mia_n", type=int, default=1000)
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
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    if a.rep_lr is None:
        a.rep_lr = 0.01 if a.fmt == "fuba" else 0.001
    if a.kd_lr is None:
        a.kd_lr = 0.01 if a.fmt == "fuba" else 0.001
    if a.defenders is None:
        a.defenders = [a.defender]
    seed_all(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/unified/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # ---------- 공통 적재
    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    K = len(U)
    roles = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else "benign")) for j in range(K)}
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    score = make_scorer(a, bd, eval_loader, device)
    eng = Engine(a, factory, flat, U, xs, device)
    # 궤적 (라운드별 몫)
    _, rounds = load_rounds(a)
    w = [c / sum(a.counts) for c in a.counts] if (a.fmt == "badfu" and a.counts) else [1.0 / K] * K
    Ut = []
    for g_prev, locals_ in rounds:
        gv = flat.flatten(g_prev)
        Ut.append({j: w[j] * (flat.flatten(locals_[j]) - gv) for j in range(K) if locals_[j] is not None})
    # 지울 데이터 (요청자별)
    forget_sets, split, mia_fn, forget_fn, nonmem = {}, None, None, None, None
    if a.fmt == "fuba":
        from benign_departure import make_config
        from robust_fu import _slice, forget_proxy, mia
        from utils.comm_utils import make_client_split
        from dataset import MNIST
        import torchvision.transforms as transforms
        cfg = make_config(a)
        split = make_client_split(MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), cfg)
        for k in a.requesters:
            if k < len(split):
                forget_sets[k] = _slice(split[k], a.forget_n, a.seed + 3)[0].to(device)
        nonmem = _slice(eval_loader.dataset, a.mia_n, a.seed + 1)
        mia_fn, forget_fn = mia, forget_proxy
    else:
        import badfu_common as bc
        bdata = bc.BadFUData()
        g = torch.Generator().manual_seed(a.seed + 3)
        for k in a.requesters:
            ds = bdata.cv if k in a.defenders else bdata.client_sets[k]
            idx = torch.randperm(len(ds), generator=g)[:a.forget_n].tolist()
            forget_sets[k] = torch.stack([ds[i][0] for i in idx]).to(device)

    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "roles": roles,
           "before": {"acc": acc_of(m0, eval_loader, device)}, "results": {}}
    if score is not None:
        res["before"]["asr"] = score(m0)
    if a.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
    print(f"[unified {a.name}] K={K} before={res['before']}", flush=True)

    # ---------- A 거름망 준비: 공개 입력의 클래스 손실 기울기는 모델 버전당 한 번
    G_pub, _ = class_grads(m0, flat, xs, a.n_classes, device)
    clients = list(range(K))

    def effects(G):
        return {k: torch.stack([(G @ Ut[t][k]) if k in Ut[t] else torch.zeros(G.shape[0]) for t in range(len(Ut))]) for k in clients}

    def screen(k):
        X = xs if (a.probe_mode == "clean" or k not in forget_sets) else torch.cat([xs, forget_sets[k]])
        G = G_pub if X is xs else class_grads(m0, flat, X, a.n_classes, device)[0]
        ind = indicators(effects(G), clients)
        Z = zscores(ind, clients, KEYS, a.zmode)
        zk = {key: Z[k][key] for key in KEYS}
        return {"z": zk, "suspect": any(v > a.tau_screen for v in zk.values()), "partner": ind[k]["cancel_partner"]}

    def evalm(theta, theta_plain, k):
        m = eng.model(theta)
        e = {"acc": acc_of(m, eval_loader, device), "retain_change": retain_change(m, m0, eval_loader, device)}
        uk2 = float((U[k] * U[k]).sum().item())
        e["alpha"] = round(float(((theta - theta_plain) * U[k]).sum().item() / uk2), 4) if uk2 > 0 else None
        if score is not None:
            e["asr"] = score(m)
        if split is not None and k < len(split):
            loader = torch.utils.data.DataLoader(split[k], batch_size=256, shuffle=False, num_workers=0)
            e["forget_acc"] = forget_fn(m.state_dict(), loader, device)["acc"]
        return e

    for k in a.requesters:
        if str(k) in res["results"]:
            continue
        t0 = time.time()
        n_models0, n_fwd0 = eng.n_pop_models, eng.n_forwards
        eng.forget_x = forget_sets.get(k) if a.probe_mode == "forget" else None
        r = {"role": roles[k], "decision": None}
        theta_plain = tT - U[k]
        # ---------- A
        if a.screen == "on":
            r["screen"] = screen(k)
            print(f"  [{k} {roles[k]}] A 거름망: " + " ".join(f"{key} z{r['screen']['z'][key]:+.1f}" for key in KEYS) + f" → {'의심' if r['screen']['suspect'] else '통과'}", flush=True)
            if not r["screen"]["suspect"]:
                r["decision"] = "accept(screen)"
                r["final"] = evalm(theta_plain, theta_plain, k)
                r["cost"] = {"pop_models": 0, "forwards": eng.n_forwards - n_fwd0, "seconds": round(time.time() - t0, 1)}
                res["results"][str(k)] = r
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
                print(f"    → 수락 (기본 FU 와 동일). ASR {r['final'].get('asr')} ACC {r['final']['acc']}", flush=True)
                continue
        # ---------- B 확정 (단계·보호·멈춤은 dose_fu 와 같은 규칙)
        theta, removed, trace, base_id, next_id, aborted, flagged_any = tT.clone(), 0.0, [], 0, 1, False, False
        pop_last, m_k = None, None
        while removed < 1 - 1e-9:
            left = 1 - removed
            delta, pop, healthy_ok = eng.dose(theta, base_id, k, left)
            base_pred = eng.base_pred(base_id, theta)
            cands = [delta] + [d for d in DELTAS if d < delta - 1e-9 and d <= left + 1e-9]
            if min(a.delta_min, left) < cands[-1] - 1e-9:
                cands.append(min(a.delta_min, left))
            for d in cands:
                req_chg = divergence(base_pred, probs(eng.model(theta - d * U[k]), xs).argmax(1), a.n_classes)["changed"]
                if req_chg <= a.cap:
                    break
            if req_chg > a.cap:
                aborted = True
                trace.append({"delta": d, "removed": removed, "z": None, "flagged": False, "repaired": False, "hard_collapse": True, "aborted": True})
                break
            if abs(d - delta) > 1e-9:
                delta, pop = d, eng.population(theta, base_id, k, d)
            m_prev = eng.model(theta) if a.op == "kd" else None
            theta = theta - delta * U[k]
            m_k = eng.model(theta)
            if a.op == "kd":
                kd_repair(m_k, m_prev, xs, a.kd_steps, a.kd_lr, a.rep_bs)
                theta = flat.flatten(m_k.state_dict())
            pk = probs(m_k, pop["xs"])
            z, d_stat = eng.check(pk, pop)
            tau_now = eng.threshold(pop)
            flagged = z > tau_now
            flagged_any = flagged_any or flagged
            removed += delta
            pop_last = pop
            trace.append({"delta": round(delta, 4), "removed": round(removed, 4), "h": pop["h"], "n_healthy": len(pop["healthy"]),
                          "stat": d_stat[a.stat], "to_class": d_stat.get("flow_to", d_stat.get("to_class")), "src_div": d_stat.get("src_div"),
                          "flip_conf": d_stat.get("flip_conf"), "z": round(z, 2), "tau": round(tau_now, 2), "flagged": bool(flagged),
                          "repaired": False, "hard_collapse": False})
            base_id, next_id = next_id, next_id + 1
            if flagged:
                break                       # 걸리면 더 빼지 않고 정화로 (정화 뒤 남은 양을 마저 뺀다)
        r["confirm"] = {"trace": trace, "removed": round(removed, 4), "aborted": aborted, "flagged": flagged_any}
        print(f"  [{k} {roles[k]}] B 확정: " + " ".join(fmt_step(dict(s, z_cum=None)) for s in trace) + (" ABORT" if aborted else ""), flush=True)
        if aborted and not flagged_any:
            r["decision"] = "reject(collapse)"
            r["final"] = evalm(theta, theta_plain, k)
        elif not flagged_any:
            r["decision"] = "accept(confirm)"
            r["final"] = evalm(theta, theta_plain, k)
        else:
            # ---------- C 정화 (해독제)
            pop = pop_last
            pr = pop["pr"]
            m_k = eng.model(theta)
            pk = probs(m_k, pop["xs"])
            c_star = trace[-1]["to_class"]
            changed = pk.argmax(1) != pr.argmax(1)
            act_idx = (changed & (pk.argmax(1) == c_star)).nonzero().flatten() if c_star is not None else torch.zeros(0, dtype=torch.long)
            s_others = [pop["s"][j] for j in pop["healthy"]]
            others = [j for j in range(K) if j != k]

            def act(th):
                return eng.stat_of(probs(eng.model(th), pop["xs"]), pr)[a.stat]
            a0 = trace[-1]["stat"]
            delta_map = {j: a0 - act(theta - U[j]) for j in others}
            order = [j for j in sorted(others, key=lambda j: -delta_map[j]) if delta_map[j] > 0]
            A, th, cur, best, pat = [], theta.clone(), a0, (a0, [], theta.clone()), 0
            for j in order[:max(1, len(others) - 1)]:
                A.append(j); th = th - U[j]; cur = act(th)
                if cur < best[0] - 1e-4:
                    best, pat = (cur, list(A), th.clone()), 0
                else:
                    pat += 1
                if robust_z(cur, s_others, a.zmode) <= a.tau or pat >= 2:
                    break
            A = best[1]
            pur = {"carriers": A, "carrier_roles": [roles[j] for j in A], "map": {j: round(v, 4) for j, v in delta_map.items()},
                   "c_star": c_star, "n_act": int(act_idx.numel()), "variants": []}
            theta_out, how = None, None
            if A and act_idx.numel() >= a.min_act:
                h = grad_logp(m_k, flat, pop["xs"][act_idx], c_star, device)
                hn2 = float((h * h).sum().item())
                par = {j: (float((U[j] * h).sum().item()) / hn2) * h for j in A}
                pur["frac_of_carrier"] = {j: round(float(par[j].norm() / U[j].norm()), 4) for j in A}
                chosen = None
                for lam in a.lams:
                    th = theta.clone()
                    for j in A:
                        th = th - lam * par[j]
                    pa = probs(eng.model(th), pop["xs"])
                    health = divergence(pk.argmax(1), pa.argmax(1), a.n_classes)["changed"]
                    z2, _ = eng.check(pa, pop)
                    pur["variants"].append({"lam": lam, "z": round(z2, 2), "health": round(health, 4)})
                    if health > a.cap:
                        break
                    if z2 <= a.tau:
                        chosen, theta_out = lam, th.clone()
                pur["chosen_lam"] = chosen
                if chosen is not None:
                    how = "purified(antidote)"
            if theta_out is None:                      # 대체: 재억제 수정
                m_r = eng.model(theta)
                ok, _ = eng.repair(m_r, pop, a.rep_stop)
                th = flat.flatten(m_r.state_dict())
                z2, _ = eng.check(probs(m_r, pop["xs"]), pop)
                pur["fallback_kd"] = {"ok": ok, "z": round(z2, 2)}
                if z2 <= a.tau:
                    theta_out, how = th, "repaired(KD)"
            if theta_out is None:
                r["decision"] = "reject(purify_failed)"
                r["final"] = evalm(theta, theta_plain, k)
            else:
                # 정화 뒤 남은 양을 마저 뺀다 (요청 이행)
                rest = 1 - removed
                if rest > 1e-9:
                    theta_out = theta_out - rest * U[k]
                    removed = 1.0
                r["decision"] = how
                r["final"] = evalm(theta_out, theta_plain, k)
            r["purify"] = pur
            r["confirm"]["removed"] = round(removed, 4)
        r["cost"] = {"pop_models": eng.n_pop_models - n_models0, "forwards": eng.n_forwards - n_fwd0, "seconds": round(time.time() - t0, 1)}
        res["results"][str(k)] = r
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        fin = r["final"]
        print(f"    → {r['decision']}: ASR {fin.get('asr')} ACC {fin['acc']} retain {fin['retain_change']} α {fin['alpha']}"
              + (f" 운반자 {r['purify']['carriers']} λ {r['purify'].get('chosen_lam')}" if "purify" in r else "")
              + f" | 모델 {r['cost']['pop_models']} 순전파 {r['cost']['forwards']} {r['cost']['seconds']}s", flush=True)

    print(f"\n== {a.name}  before ACC {res['before']['acc']} / ASR {res['before'].get('asr')}")
    for k, r in res["results"].items():
        f = r["final"]
        print(f"  req {k:>2} {r['role']:10s} {r['decision']:22s} ASR {str(f.get('asr')):>6} ACC {f['acc']:5.1f} α {f['alpha']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
