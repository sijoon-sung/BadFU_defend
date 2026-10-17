# -*- coding: utf-8 -*-
"""잠재 위험 지도 (latent risk map) — 저장된 궤적의 가중치만으로, 언러닝 요청 이전에 공격자·요청자 후보를 가려낸다.

트리거를 모르고(trigger-agnostic), 데이터 접근 없이(weights-only) 두 신호를 클라이언트마다 계산한다.

(A) 형태 서명 (shape signature) — 라운드별 업데이트 d_{k,r} = local_r[k] - global_{r-1} 의 층별 행렬
    (conv 가중치는 out x (in·kh·kw) 로 펴고, fc 가중치는 그대로, bias 는 뺀다) 에 대해
      frob 노름, 최대 특이값 비율 σ1/Σσ, 유효 랭크 exp(H(σ/Σσ)), 질량 집중도(|원소| 상위 1% 좌표가 갖는 제곱 질량 비율),
      시간 일관성 cos(d_{k,r}, d_{k,r-1}).
    전체 모델 값: 층별 행렬을 블록대각으로 본 특이값 합집합(σ1 비율·유효 랭크), 전체 벡터의 노름·집중도·일관성.
    출처 아이디어: "Weight-space Detection of Backdoors in LoRA Adapters"(arXiv 2602.15195) 의 스펙트럼 서명.

(B) 참조 백도어 벡터 (reference backdoor vector, BV) 정렬 — 서버가 공격자의 트리거를 모른 채 자기 트리거로
    일반 BV 를 만든다: 최종 전역 모델 θ_T 의 복사본을 서버 보유 소량 데이터(가정: MNIST 테스트셋 앞 N 장)로
    몇 에폭 학습하되 절반에 3x3 흰 패치(모서리 선택)를 붙이고 라벨을 서버가 고른 타깃으로 바꾼다.
    (모서리 x 타깃) 조합마다 BV = θ_trained - θ_T 를 얻고 평균을 BV_generic 으로 둔다.
    기본 타깃 [0, 5, 8] 중 8 은 실제 공격 타깃이므로, 정답 누설을 피하려고 8 을 뺀 평균(BV_non_target)도 함께 보고한다.
    클라이언트마다 cos(U_k, BV) (전체·층별), 라운드별 cos(d_{k,r}, BV) 를 잰다. U_k = Σ_{r∈active} d_{k,r}.
    출처 아이디어: "Backdoor Vectors"(arXiv 2510.08016). 관련: "Watch the Weights"(arXiv 2508.00161).

채점용(정답은 --attackers/--defender 로만 들어간다): 스칼라 특징마다 요청자(defender) 순위, 공격자 순위,
AUROC(공격자+요청자 vs 정상), AUROC(요청자 vs 나머지) 를 자체 구현으로 계산한다 (sklearn 없음).

실행 예
  python latent_risk.py --name det1 --rounds 8 --K 8 --warmup 3 --attackers 0 1 2 3 --defender 4 --target 8
출력: logs/latent_risk/{name}.json, logs/latent_risk/{name}_bv.pt (BV 벡터), 콘솔 표.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from dataset import MNIST  # noqa: E402
from diff_audit.trajectory import FlatSpace, FubaTrajectory  # noqa: E402
from model import Net  # noqa: E402
from utils.comm_utils import test  # noqa: E402

CORNERS = {"br": "bottom-right", "bl": "bottom-left", "tr": "top-right", "tl": "top-left"}
SHAPE_FEATS = ["frob", "sigma1_share", "eff_rank", "concentration", "consistency"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="checkpoints/*_{name}_round_*.pkl 의 name")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8, help="클라이언트 수")
    p.add_argument("--warmup", type=int, default=3, help="웜업 라운드 수. 활성 라운드 = warmup+1 .. rounds")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--attackers", type=int, nargs="+", default=[0, 1, 2, 3], help="채점 전용 정답")
    p.add_argument("--defender", type=int, default=4, help="채점 전용 정답 (Adv-defender = 언러닝 요청자)")
    p.add_argument("--target", type=int, default=8, help="실제 공격 타깃. BV_non_target 평균에서 이 타깃을 뺀다")
    p.add_argument("--server_data", type=int, default=500, help="서버 보유 데이터 수 (MNIST 테스트셋 [0:N])")
    p.add_argument("--server_eval", type=int, default=2000, help="BV 자체 점검용 (테스트셋 [N:N+M], 깨끗한 ACC / 패치 ASR)")
    p.add_argument("--bv_epochs", type=int, default=3)
    p.add_argument("--bv_lr", type=float, default=0.01)
    p.add_argument("--bv_batch", type=int, default=64)
    p.add_argument("--bv_poison_frac", type=float, default=0.5)
    p.add_argument("--bv_targets", type=int, nargs="+", default=[0, 5, 8])
    p.add_argument("--bv_corners", nargs="+", default=["br", "tl"], choices=list(CORNERS))
    p.add_argument("--bv_patch", type=int, default=3, help="패치 한 변 길이 (픽셀)")
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- (A) 형태 서명
def as_matrix(w):
    """conv (out,in,kh,kw) -> (out, in*kh*kw); fc (out,in) 그대로."""
    return w.reshape(w.shape[0], -1) if w.dim() == 4 else w


def spectral(sig):
    """특이값 벡터 -> (σ1 비율, 유효 랭크)."""
    tot = float(sig.sum())
    if tot <= 0:
        return 0.0, 0.0
    p = (sig / tot).clamp_min(1e-12)
    ent = float(-(p * p.log()).sum())
    return float(sig.max()) / tot, math.exp(ent)


def concentration(v, frac=0.01):
    """|원소| 상위 frac 좌표가 갖는 제곱 질량 비율."""
    v = v.flatten()
    n = v.numel()
    m = max(1, int(math.ceil(frac * n)))
    sq = v.pow(2)
    tot = float(sq.sum())
    if tot <= 0:
        return 0.0
    return float(sq.topk(m).values.sum()) / tot


def cosine(u, v):
    nu, nv = float(u.norm()), float(v.norm())
    if nu <= 0 or nv <= 0:
        return 0.0
    return float((u * v).sum()) / (nu * nv)


def shape_signature(space, d, d_prev):
    """한 클라이언트의 한 라운드 업데이트 d (평탄 벡터) 의 층별·전체 형태 서명."""
    layers = {}
    all_sig = []
    for key in space.keys:
        if not key.endswith(".weight"):
            continue
        name = key.rsplit(".", 1)[0]
        W = as_matrix(space.layer(d, key))
        sig = torch.linalg.svdvals(W)
        all_sig.append(sig)
        s1, er = spectral(sig)
        layers[name] = {
            "frob": float(W.norm()),
            "sigma1_share": s1,
            "eff_rank": er,
            "concentration": concentration(W),
            "consistency": cosine(W.flatten(), as_matrix(space.layer(d_prev, key)).flatten()) if d_prev is not None else None,
        }
    s1, er = spectral(torch.cat(all_sig))
    whole = {
        "frob": float(d.norm()),
        "sigma1_share": s1,
        "eff_rank": er,
        "concentration": concentration(d),
        "consistency": cosine(d, d_prev) if d_prev is not None else None,
    }
    return {"layers": layers, "whole": whole}


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# ---------------------------------------------------------------- (B) 참조 백도어 벡터
def apply_patch(x, corner, size, value=1.0):
    """x: (N,1,28,28) 복사본에 size x size 패치를 모서리에 붙인다 (ToTensor 스케일이므로 흰색 = 1.0)."""
    x = x.clone()
    H, W = x.shape[-2], x.shape[-1]
    r0 = H - size if corner[0] == "b" else 0
    c0 = W - size if corner[1] == "r" else 0
    x[:, :, r0:r0 + size, c0:c0 + size] = value
    return x


def train_bv(theta_T, space, x_srv, y_srv, corner, target, a, device):
    """θ_T 복사본을 서버 데이터(절반 패치+라벨 변경)로 학습 -> BV = θ_trained - θ_T (평탄 벡터)."""
    seed_all(a.seed)
    n = x_srv.shape[0]
    n_p = int(round(a.bv_poison_frac * n))
    perm = torch.randperm(n)
    idx_p, idx_c = perm[:n_p], perm[n_p:]
    xs = torch.cat([apply_patch(x_srv[idx_p], corner, a.bv_patch), x_srv[idx_c]])
    ys = torch.cat([torch.full((n_p,), target, dtype=torch.long), y_srv[idx_c]])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=a.bv_batch, shuffle=True, num_workers=0)

    net = space.to_model(theta_T, device)
    for p in net.parameters():
        p.requires_grad_(True)
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=a.bv_lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for _ in range(a.bv_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(net(xb), yb).backward()
            opt.step()
    net.eval()
    return space.flatten(net.state_dict()) - theta_T, net


def patch_asr(net, x, y, corner, size, target, device):
    """서버 트리거에 대한 ASR (타깃 클래스 아닌 샘플만)."""
    net.eval()
    m = y != target
    if int(m.sum()) == 0:
        return 0.0
    with torch.no_grad():
        pred = net(apply_patch(x[m], corner, size).to(device)).argmax(1).cpu()
    return 100.0 * float((pred == target).float().mean())


# ---------------------------------------------------------------- 채점
def auroc(pos, neg):
    """AUROC = P(score_pos > score_neg) + 0.5 P(동점). sklearn 없이."""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


def separation(feature, roles, K):
    """feature: {k: value}. 내림차순 순위(1 = 가장 큼), AUROC 두 가지."""
    vals = {k: feature.get(k) for k in range(K)}
    if any(v is None for v in vals.values()):
        return None
    order = sorted(range(K), key=lambda k: -vals[k])
    rank = {k: order.index(k) + 1 for k in range(K)}
    att, dfn = roles["attackers"], roles["defender"]
    benign = roles["benign"]
    mal = att + [dfn]
    return {
        "defender_rank_desc": rank[dfn],
        "defender_rank_asc": K + 1 - rank[dfn],
        "attacker_ranks_desc": [rank[k] for k in att],
        "auroc_mal_vs_benign": auroc([vals[k] for k in mal], [vals[k] for k in benign]),
        "auroc_def_vs_rest": auroc([vals[dfn]], [vals[k] for k in range(K) if k != dfn]),
        "values": {str(k): vals[k] for k in range(K)},
    }


def role_of(k, roles):
    if k in roles["attackers"]:
        return "attacker"
    if k == roles["defender"]:
        return "defender"
    return "benign"


# ---------------------------------------------------------------- main
def main():
    a = parse()
    out = a.out or f"logs/latent_risk/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    device = torch.device("cuda" if a.n_gpu > 0 and torch.cuda.is_available() else "cpu")
    seed_all(a.seed)
    t0 = time.time()

    K, R = a.K, a.rounds
    active = list(range(a.warmup + 1, R + 1))
    roles = {"attackers": a.attackers, "defender": a.defender,
             "benign": [k for k in range(K) if k not in a.attackers and k != a.defender]}
    traj = FubaTrajectory(a.name, R, K, ckpt_dir=a.ckpt_dir)
    space = traj.space
    layer_names = [k.rsplit(".", 1)[0] for k in space.keys if k.endswith(".weight")]

    # 라운드별 업데이트 d_{k,r} (r >= 2) 와 누적 기여 U_k (활성 라운드만)
    delta = {}
    g_prev = traj.global_vec(1)
    for r in range(2, R + 1):
        L = traj.local_vecs(r)
        delta[r] = [L[k] - g_prev for k in range(K)]
        g_prev = traj.global_vec(r)
    theta_T = g_prev
    U = [sum(delta[r][k] for r in active) for k in range(K)]
    print(f"[{a.name}] trajectory loaded: K={K} R={R} active rounds {active}  dim={space.dim}  ({time.time() - t0:.1f}s)", flush=True)

    # ---- (A) 형태 서명
    shape = {str(k): {} for k in range(K)}
    for k in range(K):
        for r in range(2, R + 1):
            shape[str(k)][str(r)] = shape_signature(space, delta[r][k], delta.get(r - 1, [None] * K)[k])
    shape_summary = {}
    for k in range(K):
        rows = [shape[str(k)][str(r)] for r in active]
        shape_summary[str(k)] = {
            "whole": {f: mean_or_none([row["whole"][f] for row in rows]) for f in SHAPE_FEATS},
            "layers": {ln: {f: mean_or_none([row["layers"][ln][f] for row in rows]) for f in SHAPE_FEATS} for ln in layer_names},
        }
    print(f"[{a.name}] shape signatures done ({time.time() - t0:.1f}s)", flush=True)

    # ---- (B) 참조 BV: 서버 보유 데이터 = MNIST 테스트셋 [0:N] (가정; 문서 참조)
    transform = transforms.Compose([transforms.ToTensor()])
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    x_all, y_all = testset.data.float(), torch.as_tensor(testset.targets).long()
    N = a.server_data
    x_srv, y_srv = x_all[:N], y_all[:N]
    x_ev, y_ev = x_all[N:N + a.server_eval], y_all[N:N + a.server_eval]
    ev_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_ev, y_ev), batch_size=256, shuffle=False, num_workers=0)
    base_net = space.to_model(theta_T, device)
    bv_info = {"server_data": N, "source": "MNIST test[0:N]", "patch": a.bv_patch, "poison_frac": a.bv_poison_frac,
               "epochs": a.bv_epochs, "lr": a.bv_lr, "theta_T_acc": round(test(base_net, ev_loader, device), 2), "models": {}}
    bvs = {}
    for corner in a.bv_corners:
        for tgt in a.bv_targets:
            tag = f"{corner}_t{tgt}"
            bv, net = train_bv(theta_T, space, x_srv, y_srv, corner, tgt, a, device)
            bvs[tag] = bv
            bv_info["models"][tag] = {
                "corner": corner, "target": tgt, "norm": float(bv.norm()),
                "acc": round(test(net, ev_loader, device), 2),
                "asr": round(patch_asr(net, x_ev, y_ev, corner, a.bv_patch, tgt, device), 2),
            }
            print(f"[{a.name}] BV {tag}: |BV|={float(bv.norm()):.3f} acc={bv_info['models'][tag]['acc']} asr={bv_info['models'][tag]['asr']}", flush=True)
    refs = {"generic": torch.stack(list(bvs.values())).mean(0)}
    non_t = [v for t, v in bvs.items() if not t.endswith(f"_t{a.target}")]
    refs["non_target"] = torch.stack(non_t).mean(0) if non_t else None
    refs.update(bvs)
    bv_info["generic_from"] = list(bvs)
    bv_info["non_target_from"] = [t for t in bvs if not t.endswith(f"_t{a.target}")]

    # 클라이언트별 정렬: cos(U_k, BV) 전체·층별, 라운드별 cos(d_{k,r}, BV)
    align = {}
    for ref, bv in refs.items():
        if bv is None:
            align[ref] = None
            continue
        align[ref] = {"whole": {}, "per_layer": {}, "per_round": {}, "per_round_active_mean": {}}
        for k in range(K):
            align[ref]["whole"][str(k)] = cosine(U[k], bv)
            align[ref]["per_layer"][str(k)] = {}
            for ln in layer_names:
                idx = torch.cat([torch.arange(space.slices[key].start, space.slices[key].stop) for key in space.keys if key.rsplit(".", 1)[0] == ln])
                align[ref]["per_layer"][str(k)][ln] = cosine(U[k][idx], bv[idx])
            pr = {str(r): cosine(delta[r][k], bv) for r in range(2, R + 1)}
            align[ref]["per_round"][str(k)] = pr
            align[ref]["per_round_active_mean"][str(k)] = mean_or_none([pr[str(r)] for r in active])
    print(f"[{a.name}] BV alignment done ({time.time() - t0:.1f}s)", flush=True)

    # ---- 채점용 분리 요약 (정답 사용)
    feats = {}
    for f in SHAPE_FEATS:
        feats[f"shape_{f}"] = {k: shape_summary[str(k)]["whole"][f] for k in range(K)}
        for ln in layer_names:
            feats[f"shape_{ln}_{f}"] = {k: shape_summary[str(k)]["layers"][ln][f] for k in range(K)}
    for ref in refs:
        if align[ref] is None:
            continue
        feats[f"cos_U_{ref}"] = {k: align[ref]["whole"][str(k)] for k in range(K)}
        feats[f"abs_cos_U_{ref}"] = {k: abs(align[ref]["whole"][str(k)]) for k in range(K)}
        feats[f"cos_round_mean_{ref}"] = {k: align[ref]["per_round_active_mean"][str(k)] for k in range(K)}
        for ln in layer_names:
            feats[f"cos_U_{ln}_{ref}"] = {k: align[ref]["per_layer"][str(k)][ln] for k in range(K)}
    sep = {name: separation(v, roles, K) for name, v in feats.items()}

    res = {"name": a.name, "args": vars(a), "roles": roles, "active_rounds": active, "layers": layer_names,
           "bv": bv_info, "shape": shape, "shape_summary": shape_summary, "bv_align": align, "separation": sep,
           "seconds": round(time.time() - t0, 1)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    torch.save({t: v for t, v in refs.items() if v is not None}, os.path.splitext(out)[0] + "_bv.pt")

    # ---- 콘솔 표
    print(f"\n== {a.name}  latent risk map (활성 라운드 {active[0]}..{active[-1]} 평균; cos 는 U_k 와 참조 BV)")
    hdr = f"{'client':>6} {'role':>9} {'sigma1':>7} {'effrank':>8} {'conc':>7} {'consist':>8} {'cosBVgen':>9} {'cosBVnon' + str(a.target):>9}"
    print(hdr)
    for k in range(K):
        s = shape_summary[str(k)]["whole"]
        cg = align["generic"]["whole"][str(k)]
        cn = align["non_target"]["whole"][str(k)] if align["non_target"] else float("nan")
        print(f"{k:>6} {role_of(k, roles):>9} {s['sigma1_share']:7.3f} {s['eff_rank']:8.2f} {s['concentration']:7.3f} "
              f"{(s['consistency'] if s['consistency'] is not None else float('nan')):8.3f} {cg:+9.3f} {cn:+9.3f}")
    print("\nAUROC (mal=attackers+defender vs benign | defender vs rest), defender rank (desc):")
    for name in ["shape_sigma1_share", "shape_eff_rank", "shape_concentration", "shape_consistency",
                 "cos_U_generic", "cos_U_non_target", "abs_cos_U_generic", "cos_U_fc3_generic"]:
        s = sep.get(name)
        if s:
            print(f"  {name:22s} {s['auroc_mal_vs_benign']:.3f} | {s['auroc_def_vs_rest']:.3f}   def rank {s['defender_rank_desc']}/{K}  att ranks {s['attacker_ranks_desc']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
