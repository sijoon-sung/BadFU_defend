# -*- coding: utf-8 -*-
"""비교군: 언러닝 게이트 (null-space 투영). 제안 내용 — 서버가 clean 입력 X_p 로 레이어별 입력 특징 공분산을 구해
clean 부분공간 P 를 잡고, 언러닝 갱신 Δ = −U_k 를 P 로 투영한 뒤 배포한다 (NSCL / OrthoPurify 류의 도구).
전제 검증: (1) 깨우는 변화(억제자 제거)가 null space 에 있는가 — 클라이언트별 Δ_j 의 null-space 에너지 비율.
          (2) 게이트를 통과시킨 제거가 ASR 을 막는가 / 정상 요청을 해치는가.

실행: python baseline_gate.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requesters 4 7 5
출력: logs/baselines/gate_{name}.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from release_map import acc_of, load_fuba, load_probe, make_model, make_scorer  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba"], default="fuba")
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--requesters", type=int, nargs="+", default=[4, 7, 5])
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--energy", type=float, nargs="+", default=[0.9, 0.99, 0.999], help="clean 부분공간이 담는 공분산 에너지 비율")
    p.add_argument("--noise_thread", type=float, default=0.04)
    p.add_argument("--atk_eps", type=float, default=1.0)
    p.add_argument("--badfu_record", default=None)
    p.add_argument("--n_gpu", type=int, default=1)
    return p.parse_args()


def main():
    a = parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    factory, theta_T, U, flat, bd = load_fuba(a)
    K = len(U)
    roles = {j: ("suppressor" if j == a.defender else ("carrier" if j in a.attackers else "benign")) for j in range(K)}
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    m0 = make_model(factory, theta_T, device)
    score = make_scorer(a, bd, eval_loader, device)
    layers = {n: m for n, m in m0.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))}
    feats = {}
    hooks = []
    for n, m in layers.items():
        def mk(n, m):
            def h(mod, inp, out):
                x = inp[0].detach()
                if isinstance(mod, torch.nn.Conv2d):
                    x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)      # B × (c·k·k) × L
                    x = x.transpose(1, 2).reshape(-1, x.shape[1])
                feats.setdefault(n, []).append(x.cpu())
            return h
        hooks.append(m.register_forward_hook(mk(n, m)))
    with torch.no_grad():
        for i in range(0, len(xs), 256):
            m0(xs[i:i + 256])
    for h in hooks:
        h.remove()
    # 레이어별 공분산과 고유분해
    eig = {}
    for n, L in feats.items():
        X = torch.cat(L).double()
        cov = X.T @ X / X.shape[0]
        ev, V = torch.linalg.eigh(cov)
        ev, V = ev.flip(0), V.flip(1)
        eig[n] = (ev, V)
    sd = theta_T

    def project(delta_flat, energy):
        """Δ (평탄) 의 가중치 부분을 레이어별 clean 부분공간(입력 특징 공간)으로 투영. 편향은 그대로."""
        out = delta_flat.clone()
        ratio = {}
        for n, (ev, V) in eig.items():
            key = n + ".weight"
            sl = flat.slices[key]
            W = delta_flat[sl].view(flat.shapes[key]).double()
            W2 = W.reshape(W.shape[0], -1)
            cum = ev.cumsum(0) / ev.sum()
            r = int((cum < energy).sum()) + 1
            P = V[:, :r] @ V[:, :r].T
            Wp = W2 @ P
            null_e = float(((W2 - Wp) ** 2).sum() / (W2 ** 2).sum().clamp_min(1e-12))
            ratio[n] = {"rank": r, "dim": int(W2.shape[1]), "null_energy": round(null_e, 4)}
            out[sl] = Wp.reshape(W.shape).float().flatten()
        return out, ratio

    res = {"name": a.name, "roles": roles, "before": {"acc": acc_of(m0, eval_loader, device), "asr": score(m0) if score else None},
           "null_energy": {}, "gate": {}}
    tT = flat.flatten(theta_T)
    for e in a.energy:
        res["null_energy"][str(e)] = {}
        for j in range(K):
            _, ratio = project(-U[j], e)
            tot_null = sum(r["null_energy"] for r in ratio.values()) / len(ratio)
            res["null_energy"][str(e)][j] = {"role": roles[j], "mean_null": round(tot_null, 4), "per_layer": {n: r["null_energy"] for n, r in ratio.items()}}
        print(f"[energy {e}] null-space 에너지 비율 (레이어 평균): " + " ".join(f"{j}{roles[j][:3]}:{res['null_energy'][str(e)][j]['mean_null']:.2f}" for j in range(K)))
        res["gate"][str(e)] = {}
        for k in a.requesters:
            plain = tT - U[k]
            gated, _ = project(-U[k], e)
            gated = tT + gated
            mp, mg = make_model(factory, flat.to_sd(plain), device), make_model(factory, flat.to_sd(gated), device)
            row = {"role": roles[k], "plain": {"acc": acc_of(mp, eval_loader, device), "asr": score(mp) if score else None},
                   "gated": {"acc": acc_of(mg, eval_loader, device), "asr": score(mg) if score else None}}
            res["gate"][str(e)][k] = row
            print(f"  req {k} {roles[k]:10s} 빼기만 ACC {row['plain']['acc']} ASR {row['plain']['asr']} | 게이트 ACC {row['gated']['acc']} ASR {row['gated']['asr']}")
    os.makedirs("logs/baselines", exist_ok=True)
    out = f"logs/baselines/gate_{a.name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
