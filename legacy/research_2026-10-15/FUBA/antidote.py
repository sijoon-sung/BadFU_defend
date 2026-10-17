# -*- coding: utf-8 -*-
"""해독제 방향 정화 (antidote purification) — docs/paper/ALGORITHM_V2.md 3 단계 C.

공통 구조에서 나온 정화: 억누르는 사람(요청자 k)의 몫을 되살리지 않고, 심은 사람들(운반자 A)의 몫 중
'상쇄되고 있던 성분'만 뺀다.

절차
  1. θ' = θ_T − U_k (요청 처리). 기준 ref = 남들을 각각 뺀 모델들의 중앙값. 활성화 입력 X_act = θ' 의 답이 ref 와 달라지며
     가장 많이 몰린 클래스 c* 로 간 입력들 (라벨·트리거 지식 없음).
  2. 운반자 A: 지도 Δ_j = a(θ') − a(θ' − U_j) 가 양수인 순으로(pair_roles 와 같은 정의), 또는 --carriers 로 지정.
  3. 해독제 방향 h = ∇_θ mean_{x∈X_act} log p(c* | x; θ').  (X_act 에서 c* 확신을 올리는 방향 = 백도어가 미는 방향)
  4. 각 운반자 몫을 h 에 투영: U_j^∥ = (⟨U_j, h⟩/‖h‖²) h.  θ'' = θ' − λ Σ_{j∈A} U_j^∥.
     λ 는 작은 값부터 올리며(--lams) z ≤ τ 가 되는 첫 값. 건강 상한(θ' 대비 changed ≤ cap)을 넘으면 멈춤.
  5. 비교: (a) 요청만 θ', (b) 운반자 통째 제거 θ' − Σ_A U_j, (c) 재억제 수정(KD), (d) 해독제 θ''.
     지표: ASR, ACC, z, α(U_k 재주입), 운반자 몫 중 뺀 비율.

실행 예: python antidote.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requester 4
출력: logs/antidote/{name}.json
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

from dose_fu import Engine, retain_change, robust_z, seed_all  # noqa: E402
from release_map import acc_of, divergence, load_badfu, load_fuba, load_probe, make_model, make_scorer, probs  # noqa: E402


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
    p.add_argument("--requester", type=int, default=4)
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--carriers", type=int, nargs="*", default=None, help="운반자 지정 (없으면 지도로 찾음)")
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--probe_mode", choices=["clean", "forget"], default="clean")
    p.add_argument("--forget_n", type=int, default=500)
    p.add_argument("--cap", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--stat", choices=["mass", "changed", "shift", "flow"], default="mass")
    p.add_argument("--zmode", choices=["std", "mad", "iqr"], default="std")
    p.add_argument("--delta_min", type=float, default=0.1)
    p.add_argument("--lams", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16])
    p.add_argument("--max_set", type=int, default=0)
    p.add_argument("--min_act", type=int, default=10)
    p.add_argument("--force", action="store_true", help="표시 안 된(정상) 요청에도 정화를 돌려 본다 (분석용)")
    p.add_argument("--lam_rule", choices=["first", "max_healthy"], default="max_healthy",
                   help="λ 선택: first = z ≤ τ 가 되는 첫 값, max_healthy = 건강 상한 안에서 가장 큰 값 (서버는 ASR 을 못 보므로 둘 다 라벨 없음)")
    p.add_argument("--rep_steps", type=int, default=50)
    p.add_argument("--rep_lr", type=float, default=None)
    p.add_argument("--rep_bs", type=int, default=128)
    p.add_argument("--rep_every", type=int, default=0)
    p.add_argument("--rep_stop", type=float, default=1.0)
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
    return p.parse_args()


def grad_logp(model, flat, xs, c, device, bs=128):
    """h = ∇_θ mean_x log p(c|x). 평탄 벡터 (파라미터 자리만)."""
    params = dict(model.named_parameters())
    for p in params.values():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    n = len(xs)
    for i in range(0, n, bs):
        lp = F.log_softmax(model(xs[i:i + bs]), 1)[:, c].sum()
        lp.backward()
    h = torch.zeros(flat.dim)
    for name, p in params.items():
        if p.grad is not None and name in flat.slices:
            h[flat.slices[name]] = (p.grad.detach().float().cpu() / n).flatten()
    model.zero_grad(set_to_none=True)
    for p in params.values():
        p.requires_grad_(False)
    return h


def main():
    a = parse()
    if a.rep_lr is None:
        a.rep_lr = 0.01 if a.fmt == "fuba" else 0.001
    seed_all(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/antidote/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()

    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    K = len(U)
    k = a.requester
    others = [j for j in range(K) if j != k]
    role = {j: ("suppressor" if j == a.defender else ("carrier" if j in a.attackers else "benign")) for j in range(K)}
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    score = make_scorer(a, bd, eval_loader, device)
    eng = Engine(a, factory, flat, U, xs, device)
    if a.probe_mode == "forget":
        if a.fmt == "fuba":
            from benign_departure import make_config
            from robust_fu import _slice
            from utils.comm_utils import make_client_split
            from dataset import MNIST
            import torchvision.transforms as transforms
            split = make_client_split(MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), make_config(a))
            eng.forget_x = _slice(split[k], a.forget_n, a.seed + 3)[0].to(device)
        else:
            import badfu_common as bc
            bdata = bc.BadFUData()
            ds = bdata.cv if k == a.defender else bdata.client_sets[k]
            g = torch.Generator().manual_seed(a.seed + 3)
            idx = torch.randperm(len(ds), generator=g)[:a.forget_n].tolist()
            eng.forget_x = torch.stack([ds[i][0] for i in idx]).to(device)

    def ev(theta, label):
        m = eng.model(theta)
        e = {"label": label, "acc": acc_of(m, eval_loader, device), "retain_change": retain_change(m, m0, eval_loader, device),
             "alpha": round(float(((theta - theta_p) * U[k]).sum().item() / float((U[k] * U[k]).sum().item())), 4)}
        if score is not None:
            e["asr"] = score(m)
        z, d = eng.check(probs(m, pop["xs"]), pop)
        e["z"], e["stat"] = round(z, 2), d[a.stat]
        return e

    # 1. 요청 처리 + 기준 + 활성화 입력
    pop = eng.population(tT, 0, k, 1.0)
    theta_p = tT - U[k]
    m_p = eng.model(theta_p)
    pk, pr = probs(m_p, pop["xs"]), pop["pr"]
    z0, d0 = eng.check(pk, pop)
    c_star = d0["to_class"]
    changed = (pk.argmax(1) != pr.argmax(1))
    act_idx = (changed & (pk.argmax(1) == c_star)).nonzero().flatten() if c_star is not None else torch.zeros(0, dtype=torch.long)
    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "requester": k, "role": role[k], "roles": role,
           "z0": round(z0, 2), "flagged": bool(z0 > a.tau), "c_star": c_star, "n_act": int(act_idx.numel()),
           "before": {"acc": acc_of(m0, eval_loader, device), "asr": score(m0) if score else None}, "variants": []}
    print(f"[antidote {a.name}] req {k} ({role[k]}) z0 {z0:+.2f} c* {c_star} 활성화 입력 {act_idx.numel()}", flush=True)
    res["variants"].append(ev(theta_p, "request_only"))
    if (not res["flagged"] and not a.force) or act_idx.numel() < a.min_act:
        print("  표시되지 않았거나 활성화 입력이 적어 정화 생략 → 요청 그대로 수락 (정상 요청이면 이게 맞는 동작)")
        res["seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"-> {out}")
        return

    # 2. 운반자 집합
    def act(theta):
        return eng.stat_of(probs(eng.model(theta), pop["xs"]), pr)[a.stat]
    a0 = d0[a.stat]
    s_others = [pop["s"][j] for j in pop["healthy"]]
    if a.carriers:
        A = list(a.carriers)
        delta = {}
    else:
        delta = {j: a0 - act(theta_p - U[j]) for j in others}
        order = [j for j in sorted(others, key=lambda j: -delta[j]) if delta[j] > 0]
        max_set = a.max_set or max(1, len(others) - 1)
        A, theta, cur, best, pat = [], theta_p.clone(), a0, (a0, [], theta_p.clone()), 0
        for j in order[:max_set]:
            A.append(j); theta = theta - U[j]; cur = act(theta)
            if cur < best[0] - 1e-4:
                best, pat = (cur, list(A), theta.clone()), 0
            else:
                pat += 1
            if robust_z(cur, s_others, a.zmode) <= a.tau or pat >= 2:
                break
        A = best[1]
    res["carriers"] = A
    res["carrier_roles"] = [role[j] for j in A]
    res["map"] = {j: round(v, 4) for j, v in delta.items()}
    print(f"  운반자 A = {A} ({[role[j] for j in A]})", flush=True)
    if not A:
        res["seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"-> {out}")
        return
    # (b) 통째 제거
    theta_whole = theta_p.clone()
    for j in A:
        theta_whole = theta_whole - U[j]
    res["variants"].append(ev(theta_whole, "request+carriers_whole"))
    # (c) 재억제 수정 (KD toward population)
    m_kd = eng.model(theta_p)
    eng.repair(m_kd, pop, a.rep_stop)
    res["variants"].append(ev(flat.flatten(m_kd.state_dict()), "request+repair_KD"))

    # 3. 해독제 방향과 투영
    x_act = pop["xs"][act_idx]
    h = grad_logp(m_p, flat, x_act, c_star, device)
    hn2 = float((h * h).sum().item())
    coef = {j: float((U[j] * h).sum().item() / hn2) for j in A}
    par = {j: coef[j] * h for j in A}
    frac = {j: round(float(par[j].norm() / U[j].norm()), 4) for j in A}
    cos_k = round(float(F.cosine_similarity(U[k], h, dim=0)), 4)
    res["antidote"] = {"coef": {j: round(coef[j], 4) for j in A}, "frac_of_carrier": frac, "cos_Uk_h": cos_k,
                       "cos_Uj_h": {j: round(float(F.cosine_similarity(U[j], h, dim=0)), 4) for j in A}}
    print(f"  h: cos(U_k,h) {cos_k:+.3f}, 운반자 투영 비율 {frac}", flush=True)
    base_pred = pr.argmax(1)
    chosen = None
    for lam in a.lams:
        theta_a = theta_p.clone()
        for j in A:
            theta_a = theta_a - lam * par[j]
        m_a = eng.model(theta_a)
        pa = probs(m_a, pop["xs"])
        health = divergence(pk.argmax(1), pa.argmax(1), a.n_classes)["changed"]   # θ' 대비 변화 (무너짐 감시)
        e = ev(theta_a, f"antidote_lam{lam}")
        e["health_vs_request"] = round(health, 4)
        res["variants"].append(e)
        print(f"  λ={lam}: z {e['z']:+.2f} ASR {e.get('asr')} ACC {e['acc']} α {e['alpha']} 변화(θ' 대비) {health:.3f}", flush=True)
        if health > a.cap:
            break
        if e["z"] <= a.tau:
            if a.lam_rule == "first" and chosen is None:
                chosen = lam
                break
            chosen = lam                      # max_healthy: 건강한 동안 계속 키우고 마지막 통과값을 택한다
    res["chosen_lam"] = chosen
    res["seconds"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("  " + " | ".join(f"{v['label']}: ASR {v.get('asr')} ACC {v['acc']} z {v['z']:+.1f} α {v['alpha']}" for v in res["variants"]))
    print(f"-> {out}  ({res['seconds']}s)")


if __name__ == "__main__":
    main()
