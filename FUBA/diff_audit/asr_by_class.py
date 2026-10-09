# -*- coding: utf-8 -*-
"""삭제마다 실제 트리거 ASR 을 '원래 숫자'별로 나눠 본다 (채점용, 방어 쪽 아님).

질문: 정상 독점 클라를 지웠을 때 올라간 ASR 이 그 클라가 독점한 숫자에만 몰려 있나?

FUBA 폴더에서:
    python -m diff_audit.asr_by_class --names fp3 fp4 loc1 --K 8
출력: logs/asr_by_class_{name}.json, 화면 표
"""
import argparse
import json

import numpy as np
import torch

from diff_audit.run import load_test_images, role_of
from diff_audit.trajectory import FlatSpace, FubaTrajectory


@torch.no_grad()
def asr_per_class(model, bd, x, y, target, eps, thr, n_classes=10, bs=500):
    hit = torch.zeros(n_classes)
    tot = torch.zeros(n_classes)
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs], y[i:i + bs]
        noise = (bd(xb) * eps).clamp(-thr, thr)
        pred = model((xb + noise).clamp(-1.0, 1.0)).argmax(1)
        for c in range(n_classes):
            m = yb == c
            tot[c] += m.sum().item()
            hit[c] += ((pred == target) & m).sum().item()
    rate = (100 * hit / tot.clamp(min=1)).numpy()
    rate[target] = np.nan  # 타깃 숫자 자신은 ASR 에서 뺀다
    return rate


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--atk", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--req", type=int, default=4)
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--asr_eps", type=float, default=1.0)
    p.add_argument("--asr_thr", type=float, default=0.04)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = load_test_images(a, dev)
    space = FlatSpace()
    for name in a.names:
        traj = FubaTrajectory(name, a.rounds, a.K, a.ckpt_dir, space)
        theta_T, U = traj.contributions()
        bd = traj.trigger_generator(dev)
        rows = {"h0": asr_per_class(space.to_model(theta_T, dev), bd, X, Y, a.target, a.asr_eps, a.asr_thr)}
        for k in range(a.K):
            rows[f"del{k}"] = asr_per_class(space.to_model(theta_T - U[k], dev), bd, X, Y,
                                            a.target, a.asr_eps, a.asr_thr)
        out = {key: [None if np.isnan(v) else round(float(v), 1) for v in r] for key, r in rows.items()}
        json.dump({"name": name, "target": a.target, "asr_by_true_class": out},
                  open(f"logs/asr_by_class_{name}.json", "w", encoding="utf-8"), indent=2)
        print(f"\n=== {name}  (행: 모델, 열: 원래 숫자, 값: 트리거 붙였을 때 {a.target} 로 간 %)")
        print(f"{'':16s}" + "".join(f"{c:>6d}" for c in range(10)))
        for key, r in rows.items():
            role = "" if key == "h0" else role_of(int(key[3:]), a)
            print(f"{key + ' ' + role:16s}" + "".join("     -" if np.isnan(v) else f"{v:6.1f}" for v in r))


if __name__ == "__main__":
    main()
