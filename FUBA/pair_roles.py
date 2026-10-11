# -*- coding: utf-8 -*-
"""쌍 검사의 일반화: 표시된 요청에서 '운반자 집합'과 '공동 위장자'를 빼 보기로 찾는다.

배경. BadFU/FUBA 는 쌍으로 움직인다. 누군가(운반자, carrier)가 백도어를 심고, 누군가(위장자, suppressor)가
억눌러 두었다가 삭제 요청으로 억제를 걷는다. 쌍 하나만 가정하면 운반자가 여럿이거나 위장자가 여럿일 때 깨진다.
여기서는 쌍을 **집합**으로 넓힌다. 탐지는 dose_fu 의 같은 양 비교가 맡고(여기서는 그 결과를 가정), 이 모듈은
표시된 요청에 대해 "누가 운반자이고 누가 또 다른 위장자인가"를 빼 보기만으로 답한다.

절차 (요청 집합 S 가 표시됐다고 하자. 보통 S={k})
  0. θ_S = θ_T − Σ_{s∈S} U_s.  기준 ref = 남들을 각각 다 뺀 모델들의 중앙값 앙상블 (dose_fu 의 모집단과 동일).
     활성화 점수 a(θ) = stat(θ vs ref)  (기본 mass: 탐침에서 답이 한 클래스로 몰린 양).
  1. 지도: 모든 j ∉ S 에 대해 a(θ_S − U_j) 를 잰다.  Δ_j = a(θ_S) − a(θ_S − U_j).
       Δ_j ≫ 0  → j 를 빼면 활성화가 줄어든다 = j 의 기여 안에 백도어가 있다 (운반자 후보)
       Δ_j ≪ 0  → j 를 빼면 활성화가 더 커진다 = j 도 억제하고 있었다 (공동 위장자 후보)
  2. 탐욕 선택: Δ 가 가장 큰 j 를 운반자 집합 A 에 넣고 θ ← θ − U_j. a(θ) 가 모집단 띠 안(z ≤ τ)으로 들어오거나,
     더 이상 margin 만큼 줄지 않거나, |A| 가 max_set 이면 멈춘다. 운반자가 여럿이면 A 가 여럿이 된다.
  3. 공동 위장자 C = { j : 자기 혼자 빠졌을 때의 z_j > τ } (모집단 단일 제거 통계 재사용). 위장자가 여럿이면 S 에 없는
     나머지가 여기 걸린다. Δ_j 의 음수는 '한 명 더 빼서 생긴 일반 열화'와 섞이므로 역할 판정에 쓰지 않는다.
  4. 평가: θ_S (요청만 처리), θ_S − U_A (운반자까지 제거 = 백도어를 재우는 게 아니라 **지움**),
     그리고 선택적으로 θ_S − U_A 에서 dose_fu 식 수정.

여러 위장자가 같이 요청하면 --requesters 에 모두 적고 --joint 를 준다 (S 를 한 번에 뺀다).
정상 요청자(표시 안 됨)는 지도만 만들고 집합은 비운다 (--force 면 억지로 탐욕 선택까지).

실행 예 (FUBA 폴더)
  python pair_roles.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requesters 4 7
  python pair_roles.py --fmt fuba --name loc_fuba --requesters 4 5 --joint
출력: logs/pair_roles/{name}[_joint].json — 요청(집합)별 지도 Δ_j, A, C, 점수 궤적, 역할 정답 대비 AUROC, ASR/ACC.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from dose_fu import Engine, retain_change, robust_z, seed_all  # noqa: E402
from release_map import acc_of, load_badfu, load_fuba, load_probe, make_model, make_scorer, probs  # noqa: E402


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
    p.add_argument("--requesters", type=int, nargs="+", default=[4, 7])
    p.add_argument("--joint", action="store_true", help="요청자들을 한 집합 S 로 한 번에 뺀다 (여러 위장자 동시 요청)")
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--defenders", type=int, nargs="*", default=None, help="정답 위장자 집합 (기본: [--defender])")
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
    p.add_argument("--sim", default="", help="attack_sim 변형 스펙 (예: split:0:3,split:1:3). 조각은 역할을 상속")
    p.add_argument("--margin", type=float, default=0.2, help="(greedy) 한 명 더 빼서 활성화가 이 비율 이상 줄어야 운반자로 추가")
    p.add_argument("--mode", choices=["ranked", "greedy"], default="ranked",
                   help="ranked: 지도 Δ 순으로 한 명씩 더 빼며 z ≤ τ 까지 (작은 운반자 여럿에 강함, 지도 1 번). greedy: 매 걸음 재탐색 + margin")
    p.add_argument("--patience", type=int, default=2, help="(ranked) 활성화가 연속 이 횟수만큼 안 줄면 멈추고 가장 좋았던 집합으로 되돌림")
    p.add_argument("--max_set", type=int, default=0, help="운반자 집합 최대 크기 (0 = K−|S|−1)")
    p.add_argument("--force", action="store_true", help="표시 안 된(정상) 요청에도 탐욕 선택을 돌린다")
    p.add_argument("--exhaust", action="store_true", help="띠 안(z ≤ τ)에 들어와도 margin 만큼 주는 한 계속 뺀다 (운반자 전부 찾기)")
    p.add_argument("--repair", action="store_true", help="운반자 제거 뒤 dose_fu 식 수정도 적용")
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


def auroc_list(pos, neg):
    from robust_fu import auroc
    if not pos or not neg:
        return None
    return auroc(torch.tensor(pos, dtype=torch.float), torch.tensor(neg, dtype=torch.float))


def main():
    a = parse()
    if a.rep_lr is None:
        a.rep_lr = 0.01 if a.fmt == "fuba" else 0.001
    if a.defenders is None:
        a.defenders = [a.defender]
    seed_all(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    tag = "_joint" if a.joint else ""
    out = a.out or f"logs/pair_roles/{a.name}{tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    role = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else "benign")) for j in range(len(U))}
    sim_log = []
    if a.sim:
        import attack_sim
        U, role, sim_log = attack_sim.apply(U, role, a.sim)
        print(f"[pair {a.name}] sim: {attack_sim.describe(a.sim)} -> K={len(U)}", flush=True)
    K = len(U)
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    score = make_scorer(a, bd, eval_loader, device)
    eng = Engine(a, factory, flat, U, xs, device)

    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "roles": role, "sim": sim_log,
           "before": {"acc": acc_of(m0, eval_loader, device)}, "results": {}}
    if score is not None:
        res["before"]["asr"] = score(m0)

    sets = [list(a.requesters)] if a.joint else [[k] for k in a.requesters]
    for S in sets:
        key = "+".join(map(str, S))
        t0 = time.time()
        others = [j for j in range(K) if j not in S]
        # 기준: S 의 첫 요청자를 제외한 모집단 (δ=1). joint 면 S 전체를 제외하도록 표본을 직접 준다
        eng.sample[S[0]] = others
        pop = eng.population(tT, 0, S[0], 1.0)
        pr = pop["pr"]
        s_others = [pop["s"][j] for j in pop["healthy"] if j not in S]

        def act(theta):
            return eng.stat_of(probs(eng.model(theta), xs), pr)[a.stat]

        theta_S = tT.clone()
        for s_ in S:
            theta_S = theta_S - U[s_]
        a0 = act(theta_S)
        z0 = robust_z(a0, s_others, a.zmode)
        flagged = z0 > a.tau
        # 1. 지도
        amap = {j: act(theta_S - U[j]) for j in others}
        delta = {j: a0 - amap[j] for j in others}
        # 2. 탐욕 선택
        A, traj, theta = [], [{"set": [], "a": round(a0, 4), "z": round(z0, 2)}], theta_S.clone()
        cur = a0
        max_set = a.max_set or max(0, len(others) - 1)
        if (flagged or a.force) and a.mode == "ranked":
            order = [j for j in sorted(others, key=lambda j: -delta[j]) if delta[j] > 0]
            best_a, best_A, best_theta, pat = cur, [], theta.clone(), 0
            for j in order[:max_set]:
                A.append(j)
                theta = theta - U[j]
                cur = act(theta)
                z = robust_z(cur, s_others, a.zmode)
                traj.append({"set": list(A), "a": round(cur, 4), "z": round(z, 2)})
                if cur < best_a - 1e-4:
                    best_a, best_A, best_theta, pat = cur, list(A), theta.clone(), 0
                else:
                    pat += 1
                if (z <= a.tau and not a.exhaust) or pat >= a.patience:
                    break
            A, theta, cur = best_A, best_theta, best_a
        elif flagged or a.force:
            cand = dict(amap)
            while len(A) < max_set and cand:
                j = min(cand, key=cand.get)
                if cur - cand[j] < a.margin * max(cur, 1e-6):
                    break
                A.append(j)
                theta = theta - U[j]
                cur = act(theta)
                z = robust_z(cur, s_others, a.zmode)
                traj.append({"set": list(A), "a": round(cur, 4), "z": round(z, 2)})
                if z <= a.tau and not a.exhaust:
                    break
                cand = {i: act(theta - U[i]) for i in others if i not in A}
        # 3. 공동 위장자: '자기 혼자 빠졌을 때'도 이상한 사람 (모집단 단일 제거 z > τ). Δ 의 음수는 일반 열화와 섞여 쓰지 않는다.
        z_single = {}
        for j in others:
            if j in pop["s"]:
                z_single[j] = robust_z(pop["s"][j], [pop["s"][i] for i in pop["healthy"] if i != j and i not in S], a.zmode)
        C = [j for j in others if z_single.get(j, 0.0) > a.tau]
        # 4. 평가
        def evalm(theta_):
            m = eng.model(theta_)
            e = {"acc": acc_of(m, eval_loader, device), "retain_change": retain_change(m, m0, eval_loader, device)}
            if score is not None:
                e["asr"] = score(m)
            return e, m
        e_S, _ = evalm(theta_S)
        e_A, m_A = evalm(theta)
        e_R = None
        if a.repair and A:
            eng.sample[S[0]] = [j for j in others if j not in A]
            popA = eng.population(theta, 99, S[0], 1.0)
            eng.repair(m_A, popA, a.rep_stop)
            e_R, _ = evalm(flat.flatten(m_A.state_dict()))
        carriers_true = [j for j in others if role[j].startswith("carrier")]
        benign_true = [j for j in others if role[j].startswith("benign")]
        supp_true = [j for j in others if role[j].startswith("suppressor")]
        r = {"S": S, "roles_S": [role[s_] for s_ in S], "flagged": bool(flagged), "a0": round(a0, 4), "z0": round(z0, 2),
             "map": {j: {"a": round(amap[j], 4), "delta": round(delta[j], 4), "z_single": None if j not in z_single else round(z_single[j], 2), "role": role[j]} for j in others},
             "carriers": A, "co_suppressors": C, "trajectory": traj,
             "auroc_carrier": auroc_list([delta[j] for j in carriers_true], [delta[j] for j in benign_true]),
             "auroc_co_suppressor": auroc_list([z_single.get(j, 0.0) for j in supp_true], [z_single.get(j, 0.0) for j in benign_true]) if supp_true else None,
             "carrier_precision": (len([j for j in A if role[j].startswith("carrier")]) / len(A)) if A else None,
             "carrier_recall": (len([j for j in A if role[j].startswith("carrier")]) / len(carriers_true)) if carriers_true else None,
             "eval": {"request_only": e_S, "request+carriers": e_A, "request+carriers+repair": e_R},
             "pop_models": eng.n_pop_models, "seconds": round(time.time() - t0, 1)}
        res["results"][key] = r
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        mp = " ".join(f"{j}{'c' if role[j].startswith('carrier') else ('s' if role[j].startswith('suppressor') else '')}:{delta[j]:+.3f}/z{z_single.get(j, 0.0):+.1f}" for j in others)
        print(f"[pair {a.name}{tag}] S={S} ({','.join(r['roles_S'])}) a0={a0:.3f} z0={z0:+.1f}{'!' if flagged else ''} | map Δ: {mp}", flush=True)
        print(f"    carriers A={A} (prec {r['carrier_precision']}, rec {r['carrier_recall']}), co-suppressors C={C}, "
              f"ASR/ACC: request {e_S.get('asr')}/{e_S['acc']} -> +carriers {e_A.get('asr')}/{e_A['acc']}"
              + ("" if e_R is None else f" -> +repair {e_R.get('asr')}/{e_R['acc']}"), flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
