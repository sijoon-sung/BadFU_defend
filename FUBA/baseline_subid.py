# -*- coding: utf-8 -*-
"""기준선: SubID 식 부분공간 편차 점수 (Malicious Forgetting, INFOCOM 2026) 를 저장된 궤적에 적용.

SubID: 학습 마지막 T_r 라운드의 갱신 행렬에 SVD → 상위 k 좌특이벡터 = 정상 부분공간 S. 언러닝 중 갱신의 잔차
(S 밖 성분)를 모아 편차 부분공간을 만들고, 클라이언트별 투영 노름을 MAD 규칙(median + κ·MAD, κ=7)으로 표시.
수동형 FU(빼기/증류)에는 '언러닝 중 갱신'이 없으므로, 여기서는 두 가지를 잰다.
  (1) 제거 벡터 점수: 요청으로 모델에서 빠지는 것은 −U_k. U_k 의 S 밖 잔차 노름 φ_k 를 전원에 대해 재고 MAD 규칙.
      S 는 요청자 자신을 뺀 나머지의 갱신으로 만든다(LOO). → 요청자 4 가 표시되는가?
  (2) 학습 중 라운드별 점수: 각 라운드의 정규화 갱신 ĝ_j^r 의 S_{−j} 밖 잔차 노름. 라운드 평균 → 공격자가 표시되는가?
출력: logs/baseline_subid/{name}.json — 점수, 표시, 요청자·공격자 순위, AUROC.

실행 예: python baseline_subid.py --fmt fuba --name loc_fuba --K 8 --rounds 8
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

from baseline_masa import load_rounds  # noqa: E402
from release_map import Flat  # noqa: E402
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
    p.add_argument("--counts", type=int, nargs="*", default=None)
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--T_r", type=int, default=0, help="마지막 T_r 라운드만 사용 (0 = 전부)")
    p.add_argument("--k_sv", type=int, default=5, help="정상 부분공간 차원")
    p.add_argument("--kappa", type=float, default=7.0, help="MAD 규칙 κ (SubID 기본 7)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def top_left(G, k):
    """G (d×n) 의 상위 k 좌특이벡터 (d×k). n 이 작으므로 Gram 으로 푼다."""
    C = G.T @ G
    evals, V = torch.linalg.eigh(C)
    order = torch.argsort(evals, descending=True)[:k]
    V, s = V[:, order], evals[order].clamp_min(1e-12).sqrt()
    return (G @ V) / s


def residual_norm(S, v):
    v = v / v.norm().clamp_min(1e-12)
    return float((v - S @ (S.T @ v)).norm())


def mad_flags(scores, kappa):
    vals = np.array(list(scores.values()))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    thr = med + kappa * mad
    return {j: bool(s > thr) for j, s in scores.items()}, thr


def main():
    a = parse()
    out = a.out or f"logs/baseline_subid/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    _, rounds = load_rounds(a)
    if a.T_r:
        rounds = rounds[-a.T_r:]
    K = len(rounds[0][1])
    roles = {j: ("defender" if j == a.defender else ("attacker" if j in a.attackers else "benign")) for j in range(K)}
    flat = Flat(rounds[0][0])
    w = [c / sum(a.counts) for c in a.counts] if a.counts else [1.0 / K] * K
    # 라운드별 갱신
    upd = {j: [] for j in range(K)}
    for g_prev, locals_ in rounds:
        gv = flat.flatten(g_prev)
        for j in range(K):
            if locals_[j] is not None:
                upd[j].append(flat.flatten(locals_[j]) - gv)
    U = {j: sum(upd[j]) * w[j] if upd[j] else torch.zeros(flat.dim) for j in range(K)}   # 누적 기여 (release_map 과 같은 정의)

    def subspace(exclude):
        cols = [u / u.norm().clamp_min(1e-12) for j in range(K) if j != exclude for u in upd[j]]
        G = torch.stack(cols, 1)
        return top_left(G, min(a.k_sv, G.shape[1]))

    # (1) 제거 벡터 점수 (LOO 부분공간)
    phi_rm = {}
    for j in range(K):
        S = subspace(j)
        phi_rm[j] = residual_norm(S, U[j])
    flags_rm, thr_rm = mad_flags(phi_rm, a.kappa)
    # (2) 학습 중 라운드별 점수
    phi_tr = {}
    for j in range(K):
        S = subspace(j)
        phi_tr[j] = float(np.mean([residual_norm(S, u) for u in upd[j]])) if upd[j] else 0.0
    flags_tr, thr_tr = mad_flags(phi_tr, a.kappa)

    def rank(scores, j):
        return 1 + sum(scores[i] > scores[j] for i in scores if i != j)

    ben = [j for j in range(K) if roles[j] == "benign"]
    att = [j for j in range(K) if roles[j] == "attacker"]
    summary = {
        "removal": {"requester_rank": rank(phi_rm, a.defender), "requester_flagged": flags_rm[a.defender],
                    "auroc_attacker_vs_benign": auroc([phi_rm[j] for j in att], [phi_rm[j] for j in ben]) if att else None,
                    "n_flagged": int(sum(flags_rm.values())), "thr": thr_rm},
        "training": {"auroc_attacker_vs_benign": auroc([phi_tr[j] for j in att], [phi_tr[j] for j in ben]) if att else None,
                     "defender_rank": rank(phi_tr, a.defender), "flagged": [j for j in range(K) if flags_tr[j]], "thr": thr_tr},
        "note": "SubID 의 판정 후 조치는 재학습. 잠복형에서 요청자를 뺀 재학습의 ASR 은 benign_departure 의 retrain_attack 참조.",
    }
    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "roles": roles,
           "phi_removal": phi_rm, "flags_removal": flags_rm, "phi_training": phi_tr, "flags_training": flags_tr,
           "summary": summary, "seconds": round(time.time() - t0, 1)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[subid {a.name}] removal φ: " + " ".join(f"{j}{'*' if roles[j] != 'benign' else ''}:{phi_rm[j]:.3f}{'!' if flags_rm[j] else ''}" for j in range(K)))
    print(f"[subid {a.name}] training φ: " + " ".join(f"{j}{'*' if roles[j] != 'benign' else ''}:{phi_tr[j]:.3f}{'!' if flags_tr[j] else ''}" for j in range(K)))
    print(f"== SubID {a.name}: {json.dumps(summary, ensure_ascii=False)}\n-> {out}")


if __name__ == "__main__":
    main()
