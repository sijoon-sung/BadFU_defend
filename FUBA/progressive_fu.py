# -*- coding: utf-8 -*-
"""진행형 정확 제거 FU — 강도를 올려 가며 탐지하고, 걸리면 그 자리에서 정화한다.

요청자 k 의 저장 기여 U_k 를 ρ 단계(0.2, 0.4, …, 1.0)로 나눠 뺀다. 단계마다
  탐지: 같은 ρ 만큼 뺀 '나머지 전원' 모델들(중앙값 앙상블, 붕괴 멤버 제외)을 기준으로, k 를 뺀 모델의 답이
        한 클래스로 몰린 양(mass)과 평균 확률 이동(shift)을 재고, 나머지 각자의 같은 통계 분포에서 z 를 구한다.
        z > τ 면 경보. 임계값은 τ 하나(기본 3), 트리거·라벨·공격자 정보 없음.
  정화: 경보가 뜨면 그 자리에서, 쏠림이 보이는 입력(라벨 없는 probe, 옵션으로 패치 붙인 probe)에서
        모델을 '나머지 전원 기준' 쪽으로 짧게 되돌린다(KL). 떠난 사람이 하던 억제를 서버가 대신 맡는 것.
        정화된 가중치에서 다음 ρ 로 계속 간다. 끝에는 요청자 기여가 전부(ρ=1) 빠져 있다 = 삭제권 이행.
  정의 2: "요청자 영향은 지우되 나머지에 대한 행동은 바꾸지 않는다" 를 경로 전체에서 강제한다.

arms
  plain        정화 없음 (정확 제거 그대로)
  end          ρ=1 에서 한 번만 정화 (끝에서 한 번)
  progressive  경보가 뜬 단계마다 정화 (제안)
  always       모든 단계에서 정화 (경보 무시; 상한 참고)
정상 요청자(--requesters 에 정상 클라를 함께)로 돌리면 경보가 안 떠야 하고(오탐), 정화 비용이 0 이어야 한다.

실행 예 (FUBA 폴더에서)
  python progressive_fu.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requesters 4 7 5
출력: logs/progressive_fu/{name}.json, 콘솔 표(arm × 요청자: ASR(ACC), 경보/정화 단계 수, 유지 변화, MIA)
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from release_map import (CIFAR_MEAN, CIFAR_STD, Ensemble, InputSpace, acc_of, divergence,  # noqa: E402
                         grid_search, load_badfu, load_fuba, load_probe, make_model, make_scorer, probs, shift)

ARMS = ["plain", "end", "progressive", "always"]


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
    p.add_argument("--requesters", type=int, nargs="+", default=[4, 7, 5], help="4=공격 설계 요청자, 나머지=정상(오탐·비용 점검)")
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--cap", type=float, default=0.5, help="붕괴 감시: θ_T 대비 changed > cap 인 멤버는 기준에서 제외")
    p.add_argument("--tau", type=float, default=3.0, help="경보 임계 z")
    p.add_argument("--stat", choices=["mass", "shift"], default="mass", help="경보에 쓰는 통계")
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--delta_mode", choices=["none", "grid"], default="none", help="정화 세트에 패치 붙인 probe 를 추가 (패치 트리거용)")
    p.add_argument("--grid_sizes", type=int, nargs="+", default=[3, 5])
    p.add_argument("--grid_stride", type=int, default=8)
    p.add_argument("--pur_steps", type=int, default=50)
    p.add_argument("--pur_lr", type=float, default=None, help="기본: fuba 0.01, badfu 0.001")
    p.add_argument("--pur_bs", type=int, default=128)
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


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def purify(model, ref, xs_set, steps, lr, bs):
    """경보가 뜬 자리에서: 쏠림이 보이는 입력에서 모델을 기준(나머지 전원) 쪽으로 되돌린다. 라벨 없음. 발산하면 되돌림."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    n = len(xs_set)
    g = torch.Generator(device="cpu").manual_seed(0)
    ok = True
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        x = xs_set[idx]
        with torch.no_grad():
            out = ref(x)
            t = out.exp() if isinstance(ref, Ensemble) else F.softmax(out, 1)
        loss = F.kl_div(F.log_softmax(model(x), 1), t, reduction="batchmean")
        if not torch.isfinite(loss):
            ok = False
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        ok = ok and all(torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_floating_point(v))
    if not ok:
        model.load_state_dict(backup)
        model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ok


@torch.no_grad()
def retain_change(m, m0, loader, device):
    diff = tot = 0
    for x, _ in loader:
        x = x.to(device)
        diff += int((m(x).argmax(1) != m0(x).argmax(1)).sum())
        tot += x.size(0)
    return round(100.0 * diff / tot, 2)


def main():
    a = parse()
    if a.pur_lr is None:
        a.pur_lr = 0.01 if a.fmt == "fuba" else 0.001
    seed_all(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/progressive_fu/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    K = len(U)
    clients = list(range(K))
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    space = InputSpace(a.fmt, device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    pred0 = probs(m0, xs).argmax(1)
    score = make_scorer(a, bd, eval_loader, device)

    # 망각 측정(fuba 만): 요청자 자기 분할 + 손실 기반 MIA (robust_fu 의 helper 재사용)
    split, mia_fn, forget_fn, nonmem = None, None, None, None
    if a.fmt == "fuba":
        from benign_departure import make_config
        from robust_fu import _slice, forget_proxy, mia
        from utils.comm_utils import make_client_split
        cfg = make_config(a)
        from dataset import MNIST
        import torchvision.transforms as transforms
        trainset = MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        split = make_client_split(trainset, cfg)
        evalset = eval_loader.dataset
        nonmem = _slice(evalset, a.mia_n, a.seed + 1)
        mia_fn, forget_fn = mia, forget_proxy

    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K,
           "before": {"acc": acc_of(m0, eval_loader, device)}, "results": {}}
    if score is not None:
        res["before"]["asr"] = score(m0)
    if a.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
    print(f"[{a.name}] K={K} before={res['before']}", flush=True)

    # 같은 ρ 의 '나머지 전원' 모델과 각자의 통계는 요청자와 무관하므로 한 번만 만든다
    others_cache = {}

    def population(rho):
        """ρ 에서 모든 클라이언트의 (빼기만 한) 모델, θ_T 대비 changed, 각자 leave-one-out 기준 통계."""
        if rho in others_cache:
            return others_cache[rho]
        mods, chg = {}, {}
        for j in clients:
            mods[j] = make_model(factory, flat.to_sd(tT - rho * U[j]), device)
            chg[j] = divergence(pred0, probs(mods[j], xs).argmax(1), a.n_classes)["changed"]
        stats = {}
        for j in clients:
            okm = [mods[i] for i in clients if i != j and chg[i] <= a.cap]
            ref = Ensemble(okm) if okm else m0
            pr, pk = probs(ref, xs), probs(mods[j], xs)
            d = divergence(pr.argmax(1), pk.argmax(1), a.n_classes)
            d.update(shift(pk, pr))
            stats[j] = d
        others_cache[rho] = (mods, chg, stats)
        return others_cache[rho]

    for k in a.requesters:
        res["results"].setdefault(str(k), {})
        role = "requester(defender)" if k == a.defender else ("attacker" if k in a.attackers else "benign")
        req_loader = torch.utils.data.DataLoader(split[k], batch_size=256, shuffle=False, num_workers=0) if split is not None else None
        mem_req = _slice(split[k], a.mia_n, a.seed + 2) if split is not None else None
        for arm in a.arms:
            if arm in res["results"][str(k)]:
                continue
            seed_all(a.seed)
            t0 = time.time()
            theta = tT.clone()
            prev = 0.0
            trace = []
            m_k = None
            for rho in a.rhos:
                theta = theta - (rho - prev) * U[k]
                prev = rho
                m_k = make_model(factory, flat.to_sd(theta), device)
                mods, chg, stats = population(rho)
                okm = [mods[i] for i in clients if i != k and chg[i] <= a.cap]
                ref = Ensemble(okm) if okm else m0
                pr, pk = probs(ref, xs), probs(m_k, xs)
                d = divergence(pr.argmax(1), pk.argmax(1), a.n_classes)
                d.update(shift(pk, pr))
                others = [stats[i][a.stat] for i in clients if i != k]
                mu, sd = float(np.mean(others)), float(np.std(others))
                z = (d[a.stat] - mu) / sd if sd > 0 else 0.0
                flagged = z > a.tau
                do_pur = (arm == "always") or (arm == "progressive" and flagged) or (arm == "end" and rho == a.rhos[-1])
                pur_ok = None
                if do_pur:
                    xs_set = xs
                    if a.delta_mode == "grid":
                        best = grid_search(ref, m_k, xs, space, a.n_classes, a.grid_sizes, a.grid_stride)
                        c = best["cand"]
                        mk = torch.zeros(1, 1, xs.shape[2], xs.shape[3], device=device)
                        mk[:, :, c["y"]:c["y"] + c["size"], c["x"]:c["x"] + c["size"]] = 1.0
                        pt = space.hi if c["pattern"] == "white" else space.lo
                        if c["pattern"] == "checker":
                            H, W = xs.shape[2], xs.shape[3]
                            ck = ((torch.arange(H, device=device).view(H, 1) + torch.arange(W, device=device).view(1, W)) % 2).float()
                            pt = space.lo + ck.view(1, 1, H, W) * (space.hi - space.lo)
                        xs_set = torch.cat([xs, (1 - mk) * xs + mk * pt.expand(1, xs.shape[1], xs.shape[2], xs.shape[3])])
                    pur_ok = purify(m_k, ref, xs_set, a.pur_steps, a.pur_lr, a.pur_bs)
                    theta = flat.flatten(m_k.state_dict())           # 정화된 가중치에서 계속 간다
                    pk2 = probs(m_k, xs)
                    d_after = divergence(pr.argmax(1), pk2.argmax(1), a.n_classes)
                    d_after.update(shift(pk2, pr))
                else:
                    d_after = None
                step = {"rho": rho, "mass": d["mass"], "shift": d["shift"], "to_class": d["to_class"], "shift_class": d["shift_class"],
                        "others_mean": round(mu, 4), "others_std": round(sd, 4), "z": round(z, 2), "flagged": bool(flagged),
                        "purified": bool(do_pur), "purify_ok": pur_ok, "after": d_after,
                        "acc": acc_of(m_k, eval_loader, device)}
                if score is not None:
                    step["asr"] = score(m_k)
                trace.append(step)
            final = {"acc": acc_of(m_k, eval_loader, device), "retain_change": retain_change(m_k, m0, eval_loader, device),
                     "n_flagged": sum(s["flagged"] for s in trace), "n_purified": sum(s["purified"] for s in trace),
                     "trace": trace, "seconds": round(time.time() - t0, 1), "role": role}
            if score is not None:
                final["asr"] = score(m_k)
            if split is not None:
                sd_k = m_k.state_dict()
                final["forget"] = {"final": forget_fn(sd_k, req_loader, device), "before": forget_fn(theta_T, req_loader, device)}
                n_req = mem_req[0].size(0)
                final["mia_req"] = mia_fn(sd_k, mem_req, (nonmem[0][:n_req], nonmem[1][:n_req]), device)
            res["results"][str(k)][arm] = final
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            zs = " ".join(f"{s['rho']:.1f}:{s['z']:+.1f}{'!' if s['flagged'] else ''}{'P' if s['purified'] else ''}" for s in trace)
            print(f"[{a.name}] req={k} {role:19s} {arm:11s} ASR {final.get('asr')} ACC {final['acc']} retain {final['retain_change']} | z: {zs}", flush=True)

    print(f"\n== {a.name}  before: ACC {res['before']['acc']} / ASR {res['before'].get('asr')}   (ASR (ACC) 경보/정화)")
    print("arm".ljust(12) + "".join(f"req{k}".rjust(22) for k in a.requesters))
    for arm in a.arms:
        row = arm.ljust(12)
        for k in a.requesters:
            e = res["results"].get(str(k), {}).get(arm)
            row += (f"{e.get('asr', float('nan')):6.1f} ({e['acc']:5.1f}) {e['n_flagged']}/{e['n_purified']}" if e else "-").rjust(22)
        print(row)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
