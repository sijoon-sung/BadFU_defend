# -*- coding: utf-8 -*-
"""BadFU 원래 구조에서: 정상 클라를 지워도 잠복 백도어가 드러나는가?

ul 구성 6 클라 (badfu_common.py) 로 연합학습을 한 번 하고, 클라 k 마다 지운 모델을 만든다.
  raw     : theta_T - U_k                      (FUBA 실험과 같은 빼기 근사)
  renorm  : theta_0 + sum_{j!=k} U_j / (1-w_k)  (남은 클라끼리 FedAvg 한 것처럼 가중치 재정규화)
  retrain : k 를 빼고 처음부터 다시 연합학습 (--retrain 으로 고른 클라만, 셔플 순서는 같게)
U_k = sum_r w_k (local_r[k] - global_{r-1}),  w_k = n_k / sum n.
BadFU 논문 Table V (NormalUL) 는 정상 삭제 후 ASR 이 오히려 내려간다고 보고.

BadFU-main 폴더에서 (전처리 record/ 필요):
    python badfu_deletion_compare.py --rounds 12 --local_epochs 2 --retrain 5 1 2
출력: logs/badfu_deletion_compare.json
"""
import argparse
import json
import os
import time

import torch

from badfu_common import (ATTACKER, REQUESTER, BadFUData, ResNet18, evaluate, flat, local_train,
                          state_keys, unflat)

ROLE = {ATTACKER: "attacker", REQUESTER: "requester(cv)", 1: "benign{2,3}", 2: "benign{4,5}",
        3: "benign{6,7}", 4: "benign{8,9}"}


def run_fl(data, a, dev, keys, exclude=None, record=False):
    torch.manual_seed(a.seed)
    g = ResNet18(pretrained=True).to(dev)
    clients = [k for k in range(len(data.client_sets)) if k != exclude]
    N = float(sum(data.counts[k] for k in clients))
    theta0 = flat(g.state_dict(), keys) if record else None
    U = {k: torch.zeros_like(theta0) for k in clients} if record else None
    for r in range(a.rounds):
        t0 = time.time()
        gsd = {k: v.detach().clone() for k, v in g.state_dict().items()}
        gv = flat(gsd, keys) if record else None
        agg = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in gsd.items()}
        for k in clients:
            lm = ResNet18().to(dev)
            lm.load_state_dict(gsd)
            gen = torch.Generator().manual_seed(a.seed * 1000 + r * 10 + k)
            st = local_train(lm, data.loader(k, gen), a.local_epochs, dev)
            w = data.counts[k] / N
            for name in agg:
                agg[name] += st[name].float() * w
            if record:
                U[k] += w * (flat(st, keys) - gv)
            del lm
        for name in agg:
            if gsd[name].dtype == torch.long:
                agg[name] = torch.round(agg[name]).to(torch.long)
        g.load_state_dict(agg)
        if r == a.rounds - 1 or r % 4 == 3:
            e = evaluate(g, data, dev, a.target)
            print(f"  [{'full' if exclude is None else f'w/o {exclude}'} r{r:02d}] acc={e['acc']:.2f} asr={e['asr']:.2f} "
                  f"({time.time() - t0:.0f}s/round)", flush=True)
    return g, theta0, U


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dominant_ratio", type=float, default=0.7)
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--retrain", type=int, nargs="*", default=[], help="처음부터 다시 학습해 볼 삭제 클라")
    p.add_argument("--out", default="logs/badfu_deletion_compare.json")
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("logs", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = BadFUData(a.dominant_ratio, a.seed)
    keys = state_keys()
    print("client sizes:", data.counts, flush=True)

    g, theta0, U = run_fl(data, a, dev, keys, record=True)
    ref = g.state_dict()
    thetaT = flat(ref, keys)
    N = float(sum(data.counts))
    res = {"args": vars(a), "counts": data.counts, "h0": evaluate(g, data, dev, a.target), "delete": {}}

    def ev(v):
        m = ResNet18().to(dev)
        m.load_state_dict(unflat(v, ref, keys))
        return evaluate(m, data, dev, a.target)

    for k in U:
        w = data.counts[k] / N
        rest = sum(U[j] for j in U if j != k)
        res["delete"][k] = {"role": ROLE[k], "w": round(w, 4),
                            "raw": ev(thetaT - U[k]),
                            "renorm": ev(theta0 + rest / (1 - w))}
    for k in a.retrain:
        gk, _, _ = run_fl(data, a, dev, keys, exclude=k)
        res["delete"][k]["retrain"] = evaluate(gk, data, dev, a.target)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)

    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
    h = res["h0"]
    print(f"\n=== BadFU 삭제 비교 (타깃 {a.target}; 값: ACC / ASR / 원래 클래스별 ASR)")
    print(f"{'h0':22s} {h['acc']:6.2f} {h['asr']:6.2f}  {h['asr_by_class']}")
    for k, d in res["delete"].items():
        for kind in ("raw", "renorm", "retrain"):
            if kind in d:
                e = d[kind]
                print(f"{'del' + str(k) + ' ' + d['role'] + ' ' + kind:22s} {e['acc']:6.2f} {e['asr']:6.2f}  {e['asr_by_class']}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
