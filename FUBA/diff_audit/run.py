# -*- coding: utf-8 -*-
"""삭제 전후 차이로 좁힌 재검사 실험 (FUBA 궤적).

FUBA 폴더에서:
    python -m diff_audit.run --name det1
    python -m diff_audit.run --name det1 --inv per_sample
    python -m diff_audit.run --name det1 --inv patch --steps 300

삭제 시나리오: 클라이언트 k 마다 h0 = theta_T, h1 = theta_T - U_k.
  요청자(--req) 삭제 = 공격이 노리는 삭제, 정상 클라 삭제 = 기준선(잡음 분포), 공격자 삭제는 참고용.
방어 쪽은 트리거·정답을 모른다. 깨끗한 탐침 이미지(--probe 장)와 저장된 기여만 쓴다.
정답(타깃, 역할, 실제 트리거 ASR)은 채점에만 쓴다.

출력: logs/diff_audit_{name}_{inv}.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from diff_audit.audit import DeletionAuditor
from diff_audit.invert import make_inverter
from diff_audit.localize import DiffLocalizer
from diff_audit.trajectory import FlatSpace, FubaTrajectory

SCORES = ("logit_shift", "head_row", "svd_head")


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="det1")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--atk", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--req", type=int, default=4)
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--delete", type=int, nargs="*", default=None, help="삭제해 볼 클라 (기본: 전원)")
    p.add_argument("--probe", type=int, default=1000, help="방어자가 가진 깨끗한 탐침 이미지 수 (반은 최적화, 반은 도달률 측정)")
    p.add_argument("--inv", default="universal", choices=["universal", "per_sample", "patch"])
    p.add_argument("--eps", type=float, default=0.04, help="덧셈 역추적 L∞ 예산 (FUBA IBA 트리거 상한과 같게)")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lam", type=float, default=1e-2, help="patch 역추적 마스크 L1 가중치")
    p.add_argument("--rank_by", default="logit_shift", choices=SCORES, help="좁힌 검사에 쓸 차이 점수")
    p.add_argument("--m", type=int, default=2, help="좁힌 검사에서 볼 상위 클래스 수")
    p.add_argument("--alphas", type=float, nargs="*", default=[2.0, 4.0], help="확대 모델 h0 + a(h1-h0)")
    p.add_argument("--asr_eps", type=float, default=1.0)
    p.add_argument("--asr_thr", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true", help="코드 점검용: MNIST 대신 무작위 이미지")
    return p.parse_args(argv)


class Batches:
    """fuba_test / fuba_attack 이 받는 (이미지, 라벨) 반복자."""

    def __init__(self, x, y, bs=500):
        self.x, self.y, self.bs = x, y, bs

    def __iter__(self):
        for i in range(0, len(self.x), self.bs):
            yield self.x[i:i + self.bs].clone(), self.y[i:i + self.bs].clone()


def load_test_images(a, dev):
    if a.smoke:
        g = torch.Generator().manual_seed(a.seed)
        x = torch.rand(3000, 1, 28, 28, generator=g)
        y = torch.randint(0, 10, (3000,), generator=g)
        return x.to(dev), y.to(dev)
    from torchvision import transforms
    from dataset import MNIST
    ts = MNIST(root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    x = torch.stack([ts[i][0] for i in range(len(ts))])
    y = torch.tensor([ts[i][1] for i in range(len(ts))])
    return x.to(dev), y.to(dev)


def role_of(k, a):
    if k == a.req:
        return "requester"
    return "attacker" if k in a.atk else "benign"


def main(argv=None):
    a = parse(argv)
    from utils.comm_utils import attack as fuba_attack, test as fuba_test

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs("logs", exist_ok=True)
    out_path = a.out or f"logs/diff_audit_{a.name}_{a.inv}.json"

    # ---- 데이터: 탐침(방어자) / 평가(채점) 분리 ----
    X, Y = load_test_images(a, dev)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(a.seed)).to(dev)
    pi, ei = perm[:a.probe], perm[a.probe:]
    half = a.probe // 2
    probe = (X[pi[:half]], Y[pi[:half]], X[pi[half:]], Y[pi[half:]])
    eval_loader = Batches(X[ei], Y[ei])

    # ---- 궤적 ----
    space = FlatSpace()
    traj = FubaTrajectory(a.name, a.rounds, a.K, a.ckpt_dir, space)
    theta_T, U = traj.contributions()
    bd = traj.trigger_generator(dev)

    def truth(v):
        m = space.to_model(v, dev)
        return {"acc": round(fuba_test(m, eval_loader, dev), 2),
                "asr": round(fuba_attack(m, eval_loader, bd, a.asr_eps, a.asr_thr, a.target, dev), 2)}

    inverter = make_inverter(a.inv, a.eps, a.steps, a.lam)
    loc = DiffLocalizer(space, X[pi], dev)
    aud = DeletionAuditor(space, inverter, probe, dev)

    h0 = theta_T
    t0 = time.perf_counter()
    base = aud.baseline(h0)
    once_sec = time.perf_counter() - t0

    clients = a.delete if a.delete else list(range(a.K))
    scen = []
    for k in clients:
        h1 = theta_T - U[k]
        tag = f"del{k}"
        scores = loc.class_scores(h0, h1)
        ranks = {s: loc.ranking(scores[s]) for s in SCORES}
        ranked = ranks[a.rank_by]
        res = {
            "client": k, "role": role_of(k, a),
            "truth": {"h0": truth(h0), "h1": truth(h1),
                      **{f"h_a{al}": truth(h0 + al * (h1 - h0)) for al in a.alphas}},
            "localize": {s: {"ranking": ranks[s],
                             "target_rank": ranks[s].index(a.target),
                             "scores": [round(float(x), 5) for x in scores[s]]} for s in SCORES},
            "layer_energy": loc.layer_energy(h0, h1),
            "audit": {
                "full_diff": aud.full_diff(tag, h0, h1),
                "localized": aud.localized(tag, h0, h1, ranked, a.m),
                "post_only": aud.post_only(tag, h1),
                **{f"extrap_a{al}": aud.extrapolated(tag, h0, h1, ranked, a.m, al) for al in a.alphas},
            },
        }
        scen.append(res)
        print(f"[del {k} {res['role']:9s}] ASR h0={res['truth']['h0']['asr']:6.2f} h1={res['truth']['h1']['asr']:6.2f} "
              f"| target rank ({a.rank_by})={res['localize'][a.rank_by]['target_rank']} "
              f"| full_diff flag={res['audit']['full_diff']['flag_class']} score={res['audit']['full_diff']['score']:+.3f} "
              f"| localized flag={res['audit']['localized']['flag_class']} score={res['audit']['localized']['score']:+.3f}",
              flush=True)

    summary = summarize(scen, a, aud.sec_per_inversion(), once_sec)
    out = {"args": vars(a), "h0_baseline": {int(c): {k: (round(v, 4) if isinstance(v, float) else v)
                                                      for k, v in r.items()} for c, r in base.items()},
           "scenarios": scen, "summary": summary}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print_summary(summary)
    print("saved", out_path)
    return out


def summarize(scen, a, sec_per_inv, once_sec):
    req = [s for s in scen if s["role"] == "requester"]
    ben = [s for s in scen if s["role"] == "benign"]
    methods = list(scen[0]["audit"].keys())
    S = {"sec_per_inversion": round(sec_per_inv, 4), "h0_one_time_seconds": round(once_sec, 2), "methods": {}}
    for meth in methods:
        bs = {s["client"]: s["audit"][meth]["score"] for s in ben}
        row = {"benign_scores": bs,
               "per_request_inversions": scen[0]["audit"][meth]["per_request_inversions"]}
        row["per_request_seconds_est"] = round(row["per_request_inversions"] * sec_per_inv, 3)
        if req:
            r = req[0]["audit"][meth]
            row["requester_score"] = r["score"]
            row["requester_flag_class"] = r["flag_class"]
            row["flag_is_target"] = r["flag_class"] == a.target
            if bs:
                mx = max(bs.values())
                row["detected"] = r["score"] > mx
                row["margin_over_max_benign"] = round(r["score"] - mx, 4)
        S["methods"][meth] = row
    S["localize"] = {s: {"requester_target_rank": (req[0]["localize"][s]["target_rank"] if req else None),
                         "benign_top1": {b["client"]: b["localize"][s]["ranking"][0] for b in ben}}
                     for s in SCORES}
    return S


def print_summary(S):
    print("\n=== summary ===")
    print(f"역추적 1회 {S['sec_per_inversion']}s, h0 일회성 검사 {S['h0_one_time_seconds']}s")
    print(f"{'method':14s} {'req':>8s} {'max_benign':>10s} {'detect':>7s} {'flag=tgt':>8s} {'inv/req':>7s}")
    for meth, r in S["methods"].items():
        mb = max(r["benign_scores"].values()) if r["benign_scores"] else float("nan")
        print(f"{meth:14s} {r.get('requester_score', float('nan')):8.3f} {mb:10.3f} "
              f"{str(r.get('detected')):>7s} {str(r.get('flag_is_target')):>8s} {r['per_request_inversions']:7d}")
    for s, v in S["localize"].items():
        print(f"localize[{s}]: 요청자 삭제에서 타깃 순위={v['requester_target_rank']}, 정상 삭제 1위 클래스={v['benign_top1']}")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
