# -*- coding: utf-8 -*-
"""기준선: MASA (Xu, Zhang, Hu. WACV 2025, arXiv 2411.01040) 를 저장된 궤적에 적용.

MASA 는 학습 중 매 라운드, 서버가 각 클라이언트의 로컬 모델을 복원해 '깨끗한 대리 데이터'로 언러닝(손실 최대화)을
걸고, 그동안 쌓인 손실이 다른 클라이언트들보다 유난히 크면(중앙값 편차 점수) 백도어 모델로 보고 집계에서 뺀다.
질문: 잠복형(BadFU/FUBA) 공격자의 로컬 모델이 학습 중 이 진단에 걸리는가? 억제자(요청자)는?

구현 (논문 Alg. 1 의 뼈대)
  1. 라운드 r: 로컬 모델 θ_j = g_{r-1} + u_j, 융합 θ'_j = g_{r-1} + (1-λ) u_j + λ ū   (λ: 융합 정도)
  2. θ'_j 에서 대리 데이터(라벨 있음, 테스트셋 앞 n 장)로 손실을 *올리는* SGD 를 S 걸음. 쌓인 손실 L_j.
  3. MDS_j = (L_j − median) / std.  MDS_j > δ 면 표시.
출력: logs/baseline_masa/{name}.json — 라운드별 점수·표시, 역할별 평균 점수, AUROC(공격자 vs 정상, 억제자 vs 정상), 표시 횟수.

실행 예: python baseline_masa.py --fmt fuba --name loc_fuba --K 8 --rounds 8
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from release_map import Flat, load_probe, make_model  # noqa: E402
from robust_fu import auroc as _auroc  # noqa: E402


def auroc(pos, neg):
    return _auroc(torch.tensor(list(pos), dtype=torch.float), torch.tensor(list(neg), dtype=torch.float))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba", "badfu"], default="fuba")
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--traj_dir", default=None)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--probe", type=int, default=500, help="대리 데이터 장수 (MASA: 1%)")
    p.add_argument("--lam", type=float, default=0.5, help="융합 정도 λ (0 = 융합 없음)")
    p.add_argument("--steps", type=int, default=20, help="언러닝(손실 최대화) SGD 걸음 수")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--delta", type=float, default=1.0, help="필터 반경 δ: MDS > δ 면 표시")
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_rounds(a):
    """라운드별 (직전 전역 모델, [클라 j 의 로컬 모델 또는 None]) 과 모델 공장."""
    if a.fmt == "fuba":
        from model import Net
        import re
        g, l = [], []
        name = a.name
        if not os.path.exists(f"{a.ckpt_dir}/global_net_model_{name}_round_1.pkl"):
            base_name = re.sub(r"_[a-zA-Z0-9]+(?=_s\d+)", "", name)
            if os.path.exists(f"{a.ckpt_dir}/global_net_model_{base_name}_round_1.pkl"):
                name = base_name
        for r in range(1, a.rounds + 1):
            with open(f"{a.ckpt_dir}/global_net_model_{name}_round_{r}.pkl", "rb") as f:
                g.append(pickle.load(f))
            with open(f"{a.ckpt_dir}/local_net_model_{name}_round_{r}.pkl", "rb") as f:
                l.append(pickle.load(f))
        rounds = [(g[r - 2], l[r - 1]) for r in range(2, a.rounds + 1)]
        return Net, rounds
    candidates = [os.path.abspath(os.path.join(a.traj_dir, "..", "..")), os.path.abspath(os.path.join(HERE, "..", "BadFU_src", "BadFU-main"))]
    for c in candidates:
        if os.path.exists(os.path.join(c, "badfu_common.py")):
            sys.path.insert(0, c)
            break
    from badfu_common import ResNet18
    rounds = []
    for r in range(a.rounds):
        d = torch.load(os.path.join(a.traj_dir, f"round_{r}.pt"), weights_only=False, map_location="cpu")
        rounds.append((d["gm"], list(d["cms"])))
    return (lambda: ResNet18(pretrained=False)), rounds


def unlearn_loss(model, xs, ys, steps, lr, bs, seed):
    """대리 데이터에서 손실을 올리는 SGD. 쌓인 손실(걸음 합)을 돌려준다."""
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, total = len(xs), 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        loss = F.cross_entropy(model(xs[idx]), ys[idx])
        if not torch.isfinite(loss):
            total += 1e3
            break
        opt.zero_grad()
        (-loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        total += float(loss)
    return total


def main():
    a = parse()
    torch.manual_seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/baseline_masa/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    factory, rounds = load_rounds(a)
    probe_loader, _ = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    ys = torch.cat([y for _, y in probe_loader]).to(device)
    K = len(rounds[0][1])
    roles = {j: ("defender" if j == a.defender else ("attacker" if j in a.attackers else "benign")) for j in range(K)}
    flat = Flat(rounds[0][0])
    t0 = time.time()
    per_round = []
    for ri, (g_prev, locals_) in enumerate(rounds):
        gv = flat.flatten(g_prev)
        present = [j for j in range(K) if locals_[j] is not None]
        u = {j: flat.flatten(locals_[j]) - gv for j in present}
        ubar = torch.stack([u[j] for j in present]).mean(0)
        L = {}
        for j in present:
            theta = gv + (1 - a.lam) * u[j] + a.lam * ubar
            m = make_model(factory, flat.to_sd(theta), device)
            L[j] = unlearn_loss(m, xs, ys, a.steps, a.lr, a.bs, a.seed + ri)
        vals = np.array([L[j] for j in present])
        med, sd = float(np.median(vals)), float(np.std(vals))
        mds = {j: (float((L[j] - med) / sd) if sd > 1e-9 else 0.0) for j in present}
        flagged = [j for j in present if mds[j] > a.delta]
        per_round.append({"round": ri + 2 if a.fmt == "fuba" else ri, "loss": {j: round(L[j], 4) for j in present},
                          "mds": {j: round(mds[j], 3) for j in present}, "flagged": flagged})
        print(f"[masa {a.name}] round {per_round[-1]['round']}: " + " ".join(f"{j}{'*' if roles[j] != 'benign' else ''}:{mds[j]:+.2f}{'!' if j in flagged else ''}" for j in present), flush=True)

    mean_mds = {j: float(np.mean([pr["mds"][j] for pr in per_round if j in pr["mds"]])) for j in range(K)}
    n_flag = {j: sum(j in pr["flagged"] for pr in per_round) for j in range(K)}
    ben = [mean_mds[j] for j in range(K) if roles[j] == "benign"]
    att = [mean_mds[j] for j in range(K) if roles[j] == "attacker"]
    dfd = [mean_mds[j] for j in range(K) if roles[j] == "defender"]
    summary = {"auroc_attacker_vs_benign": auroc(att, ben) if att and ben else None,
               "auroc_defender_vs_benign": auroc(dfd, ben) if dfd and ben else None,
               "flag_rate": {r: float(np.mean([n_flag[j] / len(per_round) for j in range(K) if roles[j] == r])) if any(roles[j] == r for j in range(K)) else None
                             for r in ("attacker", "defender", "benign")},
               "mean_mds_by_role": {r: float(np.mean([mean_mds[j] for j in range(K) if roles[j] == r])) if any(roles[j] == r for j in range(K)) else None
                                    for r in ("attacker", "defender", "benign")}}
    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "roles": roles, "per_round": per_round,
           "mean_mds": mean_mds, "n_flagged": n_flag, "summary": summary, "seconds": round(time.time() - t0, 1)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n== MASA {a.name}: {json.dumps(summary)}\n-> {out}")


if __name__ == "__main__":
    main()
