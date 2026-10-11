# -*- coding: utf-8 -*-
"""순차 요청 처리기: 여러 클라이언트가 차례로 삭제를 요청할 때 (위장자 분산·순차 다중 위장자).

요청 열 r_1, r_2, … 를 차례로 처리한다. 요청 r_i 마다
  z_inc : 현재 모델에서 r_i 를 다 뺀 것 vs '아직 남아 있는 남들'을 각각 다 뺀 모집단  (dose_fu 의 같은 양 비교, 요청 한 건)
  z_glb : r_i 까지 처리한 모델 vs θ_T 에서 '남들 한 명씩'을 뺀 모집단 띠   (전체 예산 장부: 창 시작 이후 누적 표류)
arms
  percl  z_inc 만으로 표시·수정 (클라이언트별 장부와 같음)
  global z_inc 또는 z_glb 가 τ 를 넘으면 표시·수정 (전체 예산)
--sim 으로 attack_sim 변형을 먼저 적용한다. 예) 위장자 4 를 3 조각으로: --sim split:4:3 --requests 4 8 9
--control 에 정상 요청 열을 주면 같은 설정으로 한 번 더 돌려 오탐을 본다.

실행 예: python schedule_fu.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --sim split:4:3 --requests 4 8 9 --control 7 5 6
출력: logs/adaptive/{name}_{tag}.json
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import attack_sim  # noqa: E402
from dose_fu import Engine, retain_change, robust_z, seed_all  # noqa: E402
from release_map import acc_of, load_badfu, load_fuba, load_probe, make_model, make_scorer, probs  # noqa: E402

ARMS = ["percl", "global"]


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
    p.add_argument("--sim", default="", help="attack_sim 스펙 (예: split:4:3)")
    p.add_argument("--requests", type=int, nargs="+", required=True, help="차례로 요청하는 클라이언트 열")
    p.add_argument("--control", type=int, nargs="*", default=None, help="정상 요청 열 (오탐 점검)")
    p.add_argument("--tag", default="sched")
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--cap", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--stat", choices=["mass", "changed", "shift"], default="mass")
    p.add_argument("--zmode", choices=["std", "mad"], default="std")
    p.add_argument("--delta_min", type=float, default=0.1)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--rep_steps", type=int, default=50)
    p.add_argument("--rep_lr", type=float, default=None)
    p.add_argument("--rep_bs", type=int, default=128)
    p.add_argument("--rep_every", type=int, default=0)
    p.add_argument("--rep_stop", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--noise_thread", type=float, default=0.04)
    p.add_argument("--atk_eps", type=float, default=1.0)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse()
    if a.rep_lr is None:
        a.rep_lr = 0.01 if a.fmt == "fuba" else 0.001
    seed_all(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/adaptive/{a.name}_{a.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    roles = {j: ("suppressor" if j == a.defender else ("carrier" if j in a.attackers else "benign")) for j in range(len(U))}
    U, roles, sim_log = attack_sim.apply(U, roles, a.sim)
    K = len(U)
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    score = make_scorer(a, bd, eval_loader, device)
    eng = Engine(a, factory, flat, U, xs, device)

    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "roles": roles, "sim": sim_log,
           "before": {"acc": acc_of(m0, eval_loader, device)}, "results": {}}
    if score is not None:
        res["before"]["asr"] = score(m0)
    print(f"[sched {a.name}] K={K} sim={attack_sim.describe(a.sim)} before={res['before']}", flush=True)

    # θ_T 기준 띠 (전체 예산): 남들 한 명씩 뺀 모집단, 요청 열 전체를 제외
    def band_T(seq):
        eng.sample[seq[0]] = [j for j in range(K) if j not in seq]
        pop = eng.population(tT, 0, seq[0], 1.0)
        s_others = [pop["s"][j] for j in pop["healthy"]]
        return pop, s_others

    def run(seq, label):
        popT, s_T = band_T(seq)
        for arm in a.arms:
            t0 = time.time()
            theta = tT.clone()
            removed = []
            base_id = 1000
            steps = []
            m_cur = eng.model(theta)
            for i, r in enumerate(seq):
                remaining = [j for j in range(K) if j not in removed and j != r]
                eng.sample[r] = remaining
                pop = eng.population(theta, base_id, r, 1.0)
                theta = theta - U[r]
                m_cur = eng.model(theta)
                pk = probs(m_cur, xs)
                z_inc, d = eng.check(pk, pop)
                z_glb = robust_z(eng.stat_of(pk, popT["pr"])[a.stat], s_T, a.zmode)
                flagged = z_inc > a.tau or (arm == "global" and z_glb > a.tau)
                rep_ok = None
                if flagged:
                    rep_ok, _ = eng.repair(m_cur, pop, a.rep_stop)
                    theta = flat.flatten(m_cur.state_dict())
                removed.append(r)
                base_id += 1
                st = {"i": i, "client": r, "role": roles[r], "z_inc": round(z_inc, 2), "z_glb": round(z_glb, 2),
                      "flagged": bool(flagged), "repaired": bool(flagged), "repair_ok": rep_ok,
                      "acc": acc_of(m_cur, eval_loader, device)}
                if score is not None:
                    st["asr"] = score(m_cur)
                steps.append(st)
            final = {"seq": seq, "roles": [roles[r] for r in seq], "steps": steps,
                     "acc": acc_of(m_cur, eval_loader, device), "retain_change": retain_change(m_cur, m0, eval_loader, device),
                     "n_flagged": sum(s["flagged"] for s in steps), "seconds": round(time.time() - t0, 1)}
            if score is not None:
                final["asr"] = score(m_cur)
            res["results"].setdefault(label, {})[arm] = final
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            tr = " ".join(f"{s['client']}:{s['z_inc']:+.1f}/{s['z_glb']:+.1f}{'!' if s['flagged'] else ''}" for s in steps)
            print(f"[sched {a.name}] {label:8s} {arm:6s} ASR {final.get('asr')} ACC {final['acc']} flag {final['n_flagged']}/{len(seq)} | {tr}", flush=True)

    run(list(a.requests), "attack")
    if a.control:
        run(list(a.control), "control")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
