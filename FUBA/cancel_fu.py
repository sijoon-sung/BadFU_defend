# -*- coding: utf-8 -*-
"""DOSE + TRACE + DEFUSE (정본: docs/paper/ALGORITHM.md). 요청 하나를 "어디서·누가" 문제 하나로 푼다.
DOSE = 단계 삭제 연산자(띠 확인), TRACE = 귀속(회전 1: 저장 기여, 회전 2: 활성화 입력), DEFUSE = 짝 몫의 활성화 성분 제거.

공격의 정의: 입력 집합 X* 위에서 요청자 k 의 기여와 짝 S 의 기여가 서로 지운다.  J_{X*}(U_k + Σ_{j∈S} w_j U_j) ≈ 0.
서버가 푸는 것 (목적식 하나):
    max_{ω, S, w}  Σ_x ω(x) · [ ‖Ẽ_k[x]‖² − ‖Ẽ_k[x] + Σ_{j∈S} w_j Ẽ_j[x]‖² ]      (|S| 작게)
    ω = 어디서 (입력 가중치), (S, w) = 누가 (짝과 가중치), Ẽ_j = 기여 j 의 기능 효과 (입력 × 클래스, 선형화).
풀이는 교대 반복: ω 를 고정하고 (S, w) 를 상자 제약 최소제곱으로, (S, w) 를 고정하고 ω 를 상쇄 에너지로.
    회전 1  ω⁰ = 공개 입력 ∪ 요청자가 낸 데이터, θ_T 에서. 짝 S¹, 타깃 c¹, 상쇄가 사는 입력 X̂¹ (어느 쪽에 사는지: 공개 / 지울 데이터).
    회전 2  요청자를 실제로 빼서(가상 제거, 비선형 확인) 반응한 입력 X_act 로 ω 를 갱신, θ' 에서 짝 S².
    회전 3+ (선택, --iters) X_act 안에서 상쇄 에너지가 양수인 입력만 남기고 S 를 다시 푼다. 바뀌지 않으면 멈춤.
    정화    찾은 (ω, S, w) 가 곧 '지워진 성분'. h = ∇_θ mean_{x∈X_act} log p(c*|x; θ'), θ'' = θ' − λ Σ_{j∈S} proj_h(U_j). U_k 는 되살리지 않는다.
BadFU 는 회전 1 에서 수렴하고(상쇄가 지울 데이터에 산다), FUBA 는 회전 2 에서 수렴한다(상쇄가 활성화된 공개 입력에 산다). 같은 반복의 수렴 속도 차이다.

실행 예
  python cancel_fu.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requesters 4 7 5
  python cancel_fu.py --fmt badfu --name loc_badfu --traj_dir ... --defender 5 --attackers 0 --target 0 --requesters 5 1 2 --op kd
출력: logs/cancel/{name}.json  (요청별 attr = 회전 1, purify[].pair_act = 회전 2+, final = 채점)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from antidote import grad_logp  # noqa: E402
from baseline_masa import load_rounds  # noqa: E402
from dose_fu import DELTAS, Engine, fmt_step, retain_change, robust_z, seed_all  # noqa: E402
from effects import conformal_p, fisher, logp  # noqa: E402
from release_map import Ensemble, acc_of, divergence, kd_repair, load_badfu, load_fuba, load_probe, make_model, make_scorer, probs  # noqa: E402


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
    p.add_argument("--requesters", type=int, nargs="+", default=[4, 7, 5])
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--defenders", type=int, nargs="*", default=None)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500)
    p.add_argument("--probe_mode", choices=["clean", "forget"], default="forget")
    p.add_argument("--forget_n", type=int, default=500)
    # 1 짝 찾기
    p.add_argument("--eps", type=float, default=0.2, help="선형화 비율 ε (θ − εU_j)")
    p.add_argument("--center", choices=["median", "mean", "proj", "none"], default="median", help="1 단계 공통 성분 제거")
    p.add_argument("--center2", choices=["median", "mean", "proj", "none"], default="none",
                   help="2' 단계(활성화 위) 공통 성분 제거. 타깃 열만 보므로 기본은 없음 (median 은 남들 중 운반자가 다수면 운반자를 지운다)")
    p.add_argument("--col2", choices=["target", "all"], default="target", help="2' 단계에서 보는 열: 활성화 클래스 c* 열만(기본) / 전 클래스")
    p.add_argument("--eps2", type=float, default=0.5, help="2' 단계 후보 제거량 (δ 의 배수; 로컬: 0.2–1.0 모두 같은 정렬). 무너지는 후보(비활성 입력 변화 > cap)는 제외")
    p.add_argument("--set_rule", choices=["gap", "thr", "forward"], default="gap",
                   help="2' 단계 짝 집합: gap = 정렬을 오름차순으로 늘어놓고 가장 큰 간격에서 잘라 아래쪽 전부 (FUBA: −0.88|+0.67, BadFU: −0.90|−0.60; 깊은 망은 누굴 빼도 조금씩 음수라 고정 문턱이 안 맞음) / thr = 정렬 ≤ −align_min 전부 / forward = 전진 선택(희소)")
    p.add_argument("--align_min", type=float, default=0.3, help="gap·thr 규칙에서 짝이 되려면 정렬 ≤ −align_min")
    p.add_argument("--iters", type=int, default=2, help="교대 반복 횟수. 2 = 회전 1(저장 기여) + 회전 2(활성화 입력). 3 이상은 X_act 안에서 상쇄 에너지로 ω 를 정련")
    p.add_argument("--max_set", type=int, default=4, help="짝 집합 크기 상한 r")
    p.add_argument("--w_max", type=float, default=2.0, help="짝 가중치 상한")
    p.add_argument("--eta", type=float, default=0.02, help="전진 선택 멈춤: 잔차 개선이 이 비율보다 작으면")
    p.add_argument("--rounds_eff", choices=["on", "off"], default="on", help="라운드별 효과(궤적)로 짝의 반대 정렬 a(k) (요청당 K·T 순전파)")
    p.add_argument("--alpha_p", type=float, default=0.1, help="1 단계 경보: Fisher p ≤ alpha_p")
    p.add_argument("--tau_attr", type=float, default=3.0, help="1 단계 경보(보조): 어느 통계든 z > tau_attr")
    # 2 빼 보기
    p.add_argument("--stat", choices=["mass", "changed", "shift", "flow"], default="flow")
    p.add_argument("--cap", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--tau_mode", choices=["fixed", "loo"], default="fixed")
    p.add_argument("--tau_margin", type=float, default=1.0)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--zmode", choices=["std", "mad", "iqr"], default="std")
    p.add_argument("--delta_min", type=float, default=0.1)
    p.add_argument("--op", choices=["subtract", "kd"], default="subtract")
    p.add_argument("--kd_steps", type=int, default=30)
    p.add_argument("--kd_lr", type=float, default=None)
    p.add_argument("--purify_on", choices=["sim", "attr", "either"], default="sim",
                   help="정화 조건: sim = 빼 보기 경보 / attr = 1 단계 경보 / either = 둘 중 하나")
    # 3 정화
    p.add_argument("--lams", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8])
    p.add_argument("--z_target", type=float, default=None, help="정화 목표 z (기본 = tau). 1 로 주면 정상 띠 안까지")
    p.add_argument("--pair_src", choices=["act", "attr"], default="act", help="정화에 쓰는 짝: act = 활성화 위 2' 단계, attr = 1 단계")
    p.add_argument("--w_mode", choices=["uniform", "nnls"], default="uniform",
                   help="정화 가중치: uniform = 짝 모두 1 (운반자는 각자 백도어 몫을 가짐; 최소제곱 가중치는 공선성으로 0 이 나올 수 있음) / nnls = 상대 최소제곱 가중치")
    p.add_argument("--min_act", type=int, default=10)
    p.add_argument("--max_purify", type=int, default=0, help="한 요청의 정화 최대 횟수. 0 = 단계마다 허용 (100 라운드 궤적은 단계가 5–10 번이라 2 로 두면 예산이 바닥나 reject 가 난다)")
    p.add_argument("--set_consist", choices=["on", "off"], default="on",
                   help="활성화 위 짝을 단계 간 일관성으로 거른다: 지금까지의 모든 단계에서 정렬 ≤ −align_min 인 후보만 (정상은 단계마다 흔들리고 운반자는 −0.9 로 고정)")
    p.add_argument("--polish", choices=["none", "kd_orth"], default="none",
                   help="kd_orth: 해독제 뒤 U_k 에 직교한 짧은 증류. 정확도는 회복되지만 교사(남들 모델)에 든 요청자 지식을 기능적으로 되들여올 수 있어 본 방법에서는 끔 (소거 실험용)")
    p.add_argument("--polish_ref", choices=["pop", "agree", "low", "self"], default="pop",
                   help="직교 증류(선택, 본 방법 아님)의 교사·입력. pop(기본) = 모집단 기준, 전 입력: 정확도는 KD 수준으로 돌아오지만 남들 모델에 든 요청자 지식을 되들여온다 "
                        "(owner 오탐: 자기 데이터 acc 50 → 79, α 0.006) → 'ACC 회복 vs 망각' trade-off 로만 보고. agree = 답이 안 바뀐 입력만 (50 → 75, 반만 막음). "
                        "low = 영향 작은 공개 입력 절반만 (FUBA 에서 ACC 90 → 84 로 흔들림, 실패). self = θ' 교사 (백도어 되살림, 실패). ALGORITHM.md O6")
    p.add_argument("--rep_steps", type=int, default=50)
    p.add_argument("--rep_lr", type=float, default=None)
    p.add_argument("--rep_bs", type=int, default=128)
    p.add_argument("--rep_every", type=int, default=0)
    p.add_argument("--rep_stop", type=float, default=1.0)
    # misc (궤적 로더·분할이 쓰는 인자)
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


# ------------------------------------------------------------------ 상자 제약 최소제곱 (그램 행렬 위)
def box_nnls(G, b, S, wmax, sweeps=300):
    """min_{0≤w≤wmax} 2 Σ_j w_j b_j + wᵀ G_SS w   (좌표 하강). b_j = ⟨목표, Ẽ_j⟩, G = 그램."""
    n = len(S)
    if n == 0:
        return {}
    Gs, bs = G[np.ix_(S, S)], b[S]
    w = np.zeros(n)
    for _ in range(sweeps):
        w_old = w.copy()
        for i in range(n):
            s = bs[i] + float(Gs[i] @ w) - Gs[i, i] * w[i]
            w[i] = min(max(-s / Gs[i, i], 0.0), wmax) if Gs[i, i] > 1e-12 else 0.0
        if np.abs(w - w_old).max() < 1e-8:
            break
    return dict(zip(S, w.tolist()))


def resid2(G, b, base, w):
    S = list(w)
    if not S:
        return base
    v = np.array([w[j] for j in S])
    return float(base + 2.0 * (v @ b[S]) + v @ G[np.ix_(S, S)] @ v)


def forward_select(G, b, base, cands, max_set, wmax, eta):
    """전진 선택: 목표 벡터(‖·‖² = base, 후보와의 내적 b) 를 가장 많이 지우는 후보를 하나씩 더해 상자 최소제곱을 다시 푼다.
    돌려줌: 상쇄 비율 1 − 잔차/base, 집합 S(순서), 가중치 w, 단계별 잔차비."""
    if base <= 1e-12:
        return 0.0, [], {}, []
    S, w_best, hist, cur = [], {}, [], base
    for _ in range(max_set):
        best = None
        for j in cands:
            if j in S:
                continue
            w = box_nnls(G, b, S + [j], wmax)
            r = resid2(G, b, base, w)
            if best is None or r < best[0]:
                best = (r, j, w)
        if best is None or best[0] > cur * (1.0 - eta):
            break
        cur, w_best = best[0], best[2]
        S.append(best[1])
        hist.append(round(cur / base, 4))
    return round(1.0 - cur / base, 4), S, {j: round(w_best[j], 4) for j in S}, hist


def center_stack(E_list, mode):
    """공통 성분 제거. median/mean: 중앙값·평균을 뺀다. proj: 평균 방향으로의 투영을 뺀다 (크기가 다른 클라이언트에도 공통 방향만 제거)."""
    if mode == "none":
        return E_list
    stk = torch.stack(E_list, 0)
    if mode == "proj":
        e = stk.mean(0).flatten()
        e = e / e.norm().clamp_min(1e-12)
        return [E - (E.flatten() @ e) * e.view_as(E.flatten()).view_as(E) for E in E_list]
    cen = stk.median(0).values if mode == "median" else stk.mean(0)
    return [E - cen for E in E_list]


def conc_of(E, yh, C):
    """타깃 몰림: 제거 시 로그확률이 가장 오르는 클래스 c, 그 양, 출발 다양성(상승 기여 입력들의 ŷ 엔트로피)."""
    gain = E.mean(0)
    c = int(gain.argmax())
    contrib = E[:, c].clamp_min(0)
    w = contrib / contrib.sum().clamp_min(1e-12)
    src = torch.zeros(C).index_add_(0, yh, w)
    nz = src[src > 0]
    H = float(-(nz * nz.log()).sum()) / math.log(max(C - 1, 2)) if nz.numel() else 0.0
    H = min(max(H, 0.0), 1.0)
    return {"class": c, "gain": round(float(gain[c]), 4), "src_div": round(H, 4), "score": round(float(gain[c]) * H, 4)}


def kd_orth(model, ref, xs, steps, lr, bs, flat, u, teacher_probs=None):
    """U_k 에 직교한 증류: 기울기에서 u 성분을 매 걸음 제거한다 → θ 의 변화가 u 와 직교 (α 불변).
    teacher_probs 가 주어지면 ref 대신 그 고정 확률(xs 와 같은 순서)을 교사로 쓴다 (self 교사: 정화 전 모델의 출력)."""
    if steps <= 0:
        return True, 0
    u = u / u.norm().clamp_min(1e-12)
    params = dict(model.named_parameters())
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for p in params.values():
        p.requires_grad_(True)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    n = len(xs)
    g = torch.Generator(device="cpu").manual_seed(0)
    ok, done = True, 0
    for t in range(1, steps + 1):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        x = xs[idx]
        if teacher_probs is not None:
            tgt = teacher_probs[idx].to(x.device)
        else:
            with torch.no_grad():
                out = ref(x)
                tgt = out.exp() if isinstance(ref, Ensemble) else F.softmax(out, 1)
        loss = F.kl_div(F.log_softmax(model(x), 1), tgt, reduction="batchmean")
        if not torch.isfinite(loss):
            ok = False
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        gflat = torch.zeros(flat.dim)
        for name, p in params.items():
            if p.grad is not None and name in flat.slices:
                gflat[flat.slices[name]] = p.grad.detach().float().cpu().flatten()
        gflat = gflat - float((gflat * u).sum()) * u
        for name, p in params.items():
            if p.grad is not None and name in flat.slices:
                p.grad.copy_(gflat[flat.slices[name]].view_as(p.grad).to(p.grad.device, p.grad.dtype))
        opt.step()
        done = t
    model.eval()
    with torch.no_grad():
        ok = ok and all(torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_floating_point(v))
    if not ok:
        model.load_state_dict(backup)
        model.eval()
    for p in params.values():
        p.requires_grad_(False)
    return ok, done


def main():
    a = parse()
    if a.rep_lr is None:
        a.rep_lr = 0.01 if a.fmt == "fuba" else 0.001
    if a.kd_lr is None:
        a.kd_lr = 0.01 if a.fmt == "fuba" else 0.001
    if a.z_target is None:
        a.z_target = a.tau
    if a.defenders is None:
        a.defenders = [a.defender]
    seed_all(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/cancel/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    C = a.n_classes

    # ---------- 적재
    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    K = len(U)
    clients = list(range(K))
    carriers_true = [j for j in a.attackers if j < K]
    roles = {j: ("suppressor" if j in a.defenders else ("carrier" if j in a.attackers else "benign")) for j in clients}
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    score = make_scorer(a, bd, eval_loader, device)
    eng = Engine(a, factory, flat, U, xs, device)
    cost = {"attr_forwards": 0}

    def model(theta):
        return make_model(factory, flat.to_sd(theta), device)

    # 지울 데이터 (요청자별) + 채점 도구
    forget_sets, split, forget_fn = {}, None, None
    if a.fmt == "fuba":
        from benign_departure import make_config
        from robust_fu import _slice, forget_proxy
        from utils.comm_utils import make_client_split
        from dataset import MNIST
        import torchvision.transforms as transforms
        split = make_client_split(MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), make_config(a))
        for k in a.requesters:
            if k < len(split):
                forget_sets[k] = _slice(split[k], a.forget_n, a.seed + 3)[0].to(device)
        forget_fn = forget_proxy
    else:
        import badfu_common as bc
        bdata = bc.BadFUData()
        g = torch.Generator().manual_seed(a.seed + 3)
        for k in a.requesters:
            ds = bdata.cv if k in a.defenders else bdata.client_sets[k]
            idx = torch.randperm(len(ds), generator=g)[:a.forget_n].tolist()
            forget_sets[k] = torch.stack([ds[i][0] for i in idx]).to(device)

    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "roles": roles,
           "before": {"acc": acc_of(m0, eval_loader, device)}, "results": {}}
    if score is not None:
        res["before"]["asr"] = score(m0)
    if a.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
    print(f"[cancel {a.name}] K={K} before={res['before']}", flush=True)

    # 공개 입력 위 효과는 모델 버전당 1 회 (요청자 무관)
    L0_pub = logp(m0, xs)
    E_pub = {j: (logp(model(tT - a.eps * U[j]), xs) - L0_pub) / a.eps for j in clients}
    cost["attr_forwards"] += K
    rounds_cache = {}
    if a.rounds_eff == "on":
        _, rounds = load_rounds(a)
        w_cl = [c / sum(a.counts) for c in a.counts] if (a.fmt == "badfu" and a.counts) else [1.0 / K] * K
        Ut = []
        for ti, (g_prev, locals_) in enumerate(rounds):
            gv = flat.flatten(g_prev)
            if a.fmt == "fuba":                                   # 결석 클라(참여율<1)는 그 라운드 집계 결과가 저장돼 있다 → 참여자만, 참여자 수로 나눔
                from release_map import present_clients
                g_next = rounds[ti + 1][0] if ti + 1 < len(rounds) else theta_T
                pres = [j for j in present_clients(locals_, g_next) if j < K]
                Ut.append({j: (flat.flatten(locals_[j]) - gv) / max(len(pres), 1) for j in pres})
            else:
                Ut.append({j: w_cl[j] * (flat.flatten(locals_[j]) - gv) for j in clients if locals_[j] is not None})

    def effects_on(X, theta, L0):
        """Ẽ_j (중심화) 와 원시 크기. θ 에서 ε 만큼 뺀 모델의 순전파 K 회."""
        E = {j: (logp(model(theta - a.eps * U[j]), X) - L0) / a.eps for j in clients}
        cost["attr_forwards"] += K
        return E

    # ---------- 1 짝 찾기 (요청자 k, θ_T, X = X_p ∪ X_f)
    def attribute(k):
        xf = forget_sets.get(k) if a.probe_mode == "forget" else None
        if xf is None:
            X, E_raw, L0 = xs, {j: E_pub[j] for j in clients}, L0_pub
        else:
            L0_f = logp(m0, xf)
            E_f = effects_on(xf, tT, L0_f)
            X = torch.cat([xs, xf])
            E_raw = {j: torch.cat([E_pub[j], E_f[j]]) for j in clients}
            L0 = torch.cat([L0_pub, L0_f])
        yh = L0.argmax(1)
        N = X.shape[0]
        Ec = dict(zip(clients, center_stack([E_raw[j] for j in clients], a.center)))
        V = np.stack([Ec[j].flatten().double().numpy() for j in clients], 0)
        G = V @ V.T
        mag = {j: round(float(np.sqrt(G[j, j] / N)), 4) for j in clients}
        conc = {j: conc_of(Ec[j], yh, C) for j in clients}
        canc = {}
        for j in clients:
            cr, S, w, hist = forward_select(G, G[j], float(G[j, j]), [i for i in clients if i != j], a.max_set, a.w_max, a.eta)
            canc[j] = {"ratio": cr, "set": S, "w": w, "hist": hist, "set_roles": [roles[i] for i in S]}
        # 라운드 반대 정렬: a(j) = mean_t −cos(Ẽ_j^t, Σ_{i∈S_j} w_i Ẽ_i^t)  (궤적; 짝이 매 라운드 같은 방향으로 지우는가)
        anti = None
        if a.rounds_eff == "on" and Ut:
            key = (int(N), k if xf is not None else -1)
            if key not in rounds_cache:
                Et = []
                for t in range(len(Ut)):
                    row = {}
                    for j in clients:
                        if j in Ut[t]:
                            row[j] = ((logp(model(tT - a.eps * Ut[t][j]), X) - L0) / a.eps).flatten()
                            cost["attr_forwards"] += 1
                    js = sorted(row)
                    Et.append(dict(zip(js, center_stack([row[j] for j in js], a.center))))
                rounds_cache[key] = Et
            Et = rounds_cache[key]
            anti = {}
            for j in clients:
                S, w = canc[j]["set"], canc[j]["w"]
                vals = []
                for t in range(len(Et)):
                    if j not in Et[t] or not all(i in Et[t] for i in S) or not S:
                        continue
                    partner = sum(w[i] * Et[t][i] for i in S)
                    vals.append(-F.cosine_similarity(Et[t][j], partner, dim=0).item())
                anti[j] = round(float(np.mean(vals)), 4) if vals else 0.0
        stats = {"mag": mag, "conc": {j: conc[j]["score"] for j in clients}, "cancel": {j: canc[j]["ratio"] for j in clients}}
        if anti is not None:
            stats["anti"] = anti
        pv = {s: round(conformal_p(stats[s][k], [stats[s][j] for j in clients if j != k]), 4) for s in stats}
        zz = {s: round(robust_z(stats[s][k], [stats[s][j] for j in clients if j != k], a.zmode), 2) for s in stats}
        fs, fp = fisher(list(pv.values()))
        S, w = list(canc[k]["set"]), dict(canc[k]["w"])
        # 어디서: 입력별 상쇄 에너지 e(x) = ‖Ẽ_k[x]‖² − ‖Ẽ_k[x] + Σ w_j Ẽ_j[x]‖²  (양수인 입력이 X̂¹)
        R = Ec[k] + (sum(w[j] * Ec[j] for j in S) if S else 0)
        energy = (Ec[k] ** 2).sum(1) - (R ** 2).sum(1)
        pos = energy > 0
        n_pub = xs.shape[0]
        where = {"n_pos": int(pos.sum()), "frac_pos": round(float(pos.float().mean()), 4),
                 "energy_public": round(float(energy[:n_pub].clamp_min(0).sum()), 4),
                 "energy_forget": round(float(energy[n_pub:].clamp_min(0).sum()), 4) if N > n_pub else None}
        if N > n_pub and (where["energy_public"] + where["energy_forget"]) > 0:
            where["frac_in_forget"] = round(where["energy_forget"] / (where["energy_public"] + where["energy_forget"]), 4)
        flagged = (fp <= a.alpha_p) or any(v > a.tau_attr for v in zz.values())
        prec = round(sum(1 for j in S if j in carriers_true) / len(S), 3) if S else None
        rec = round(sum(1 for j in S if j in carriers_true) / len(carriers_true), 3) if carriers_true else None
        info = {"N": N, "mag": mag[k], "conc": conc[k], "cancel": canc[k], "set": S, "w": w, "where": where, "anti": (anti[k] if anti else None),
                "c_star": conc[k]["class"], "pvals": pv, "z": zz, "fisher_stat": round(fs, 3), "fisher_p": round(fp, 4), "flagged": bool(flagged),
                "set_precision": prec, "set_recall": rec,
                "population": {j: {"mag": mag[j], "conc": conc[j]["score"], "conc_class": conc[j]["class"], "cancel": canc[j]["ratio"],
                                   "set": canc[j]["set"], "anti": (anti[j] if anti else None)} for j in clients}}
        return info

    # ---------- 2' 활성화 위 짝 찾기 (θ' 에서, X_act 위): 누구를 빼면 활성화가 줄어드는가
    align_hist = {}                                           # 요청자별 단계 정렬 기록 (단계 간 일관성용)

    def pair_at_activation(theta, pop, k, act_idx, c_star, delta):
        m_k = eng.model(theta)
        Lk_all = logp(m_k, pop["xs"])
        pred_k = Lk_all.argmax(1)
        mask = torch.ones(pop["xs"].shape[0], dtype=torch.bool)
        mask[act_idx] = False                                                      # 비활성 입력 (건강 감시용)
        Lk = Lk_all[act_idx]
        A = Lk - torch.log(pop["pr"][act_idx].clamp_min(1e-12))                   # 활성화: 요청자 모델 − 기준(모집단 중앙값)
        eps2 = a.eps2 * delta
        others = [j for j in clients if j != k]
        E, collapsed = {}, []
        for j in others:
            Lj = logp(model(theta - eps2 * U[j]), pop["xs"])
            cost["attr_forwards"] += 1
            if float((Lj.argmax(1)[mask] != pred_k[mask]).float().mean()) > a.cap:      # 후보 제거가 비활성 입력을 무너뜨리면 제외
                collapsed.append(j)
                continue
            E[j] = (Lj[act_idx] - Lk) / eps2
        cands = sorted(E)
        if not cands:
            return {"n_act": int(act_idx.numel()), "ratio": 0.0, "set": [], "w": {}, "hist": [], "set_roles": [], "collapsed": collapsed,
                    "set_precision": None, "set_recall": 0.0, "align": {}}
        Ec = dict(zip(cands, center_stack([E[j] for j in cands], a.center2)))
        if a.col2 == "target":
            vec = {j: Ec[j][:, c_star] for j in cands}
            av = A[:, c_star]
        else:
            vec = {j: Ec[j].flatten() for j in cands}
            av = A.flatten()
        V = np.stack([vec[j].double().numpy() for j in cands], 0)
        av = av.double().numpy()
        G_full = np.zeros((K, K))
        b = np.zeros(K)
        for ii, i in enumerate(cands):
            b[i] = float(V[ii] @ av)
            for jj, j in enumerate(cands):
                G_full[i, j] = float(V[ii] @ V[jj])
        V_all, av_all = V, av

        def solve(sel):
            """ω 고정(선택된 입력) → (S, w). 돌려줌: 정렬, 집합, 가중치, 상쇄 비율, 규칙."""
            Vs, avs = V_all[:, sel], av_all[sel]
            Gm, bv = np.zeros((K, K)), np.zeros(K)
            for ii, i in enumerate(cands):
                bv[i] = float(Vs[ii] @ avs)
                for jj, j in enumerate(cands):
                    Gm[i, j] = float(Vs[ii] @ Vs[jj])
            base = float(avs @ avs)
            align = {j: round(float(bv[j] / math.sqrt(max(Gm[j, j], 1e-12)) / max(math.sqrt(base), 1e-12)), 3) for j in cands}
            cr_fs, S_fs, w_fs, hist = forward_select(Gm, bv, base, cands, a.max_set, a.w_max, a.eta)
            S_sel = []
            if a.set_rule == "thr":
                S_sel = [j for j in cands if align[j] <= -a.align_min]
            elif a.set_rule == "gap" and len(cands) >= 2:
                order = sorted(cands, key=lambda j: align[j])
                gaps = [align[order[i + 1]] - align[order[i]] for i in range(len(order) - 1)]
                cut = int(np.argmax(gaps))
                S_sel = [j for j in order[:cut + 1] if align[j] <= -a.align_min]
            if S_sel:
                w_sel = box_nnls(Gm, bv, S_sel, a.w_max)
                return align, S_sel, {j: round(w_sel[j], 4) for j in S_sel}, round(1.0 - resid2(Gm, bv, base, w_sel) / base, 4), a.set_rule, {"ratio": cr_fs, "set": S_fs, "w": w_fs, "hist": hist}
            return align, S_fs, w_fs, cr_fs, "forward", {"ratio": cr_fs, "set": S_fs, "w": w_fs, "hist": hist}

        sel = np.ones(len(av_all), dtype=bool)
        align, S, w, cr, rule, fwd = solve(sel)
        align_hist.setdefault(k, []).append(dict(align))
        if a.set_consist == "on" and len(align_hist[k]) >= 2:          # 단계 간 일관성: 모든 단계에서 정렬 ≤ −a_min 인 후보만
            S_c = [j for j in cands if all(h.get(j, 0.0) <= -a.align_min for h in align_hist[k])]
            if S_c:
                base0 = float(av_all @ av_all)
                w_c = box_nnls(G_full, b, S_c, a.w_max)
                S, w, cr, rule = S_c, {j: round(w_c[j], 4) for j in S_c}, round(1.0 - resid2(G_full, b, base0, w_c) / base0, 4), "consist"
        history = [{"iter": 2, "n_inputs": int(sel.sum()), "set": list(S), "ratio": cr}]
        for it in range(3, a.iters + 1):                                  # 회전 3+: (S, w) 고정 → ω 를 상쇄 에너지로 정련 → 다시 (S, w)
            if not S:
                break
            resid = av_all + sum(w[j] * V_all[cands.index(j)] for j in S)
            energy = av_all ** 2 - resid ** 2
            new_sel = energy > 0
            if new_sel.sum() < a.min_act or np.array_equal(new_sel, sel):
                break
            sel = new_sel
            align2, S2, w2, cr2, rule2, fwd2 = solve(sel)
            history.append({"iter": it, "n_inputs": int(sel.sum()), "set": list(S2), "ratio": cr2})
            changed = set(S2) != set(S)
            align, S, w, cr, rule, fwd = align2, S2, w2, cr2, rule2, fwd2
            if not changed:
                break
        prec = round(sum(1 for j in S if j in carriers_true) / len(S), 3) if S else None
        rec = round(sum(1 for j in S if j in carriers_true) / len(carriers_true), 3) if carriers_true else None
        return {"n_act": int(act_idx.numel()), "eps2": round(eps2, 4), "rule": rule, "ratio": cr, "set": S, "w": w, "set_roles": [roles[j] for j in S],
                "forward": fwd, "collapsed": collapsed, "iters": history,
                "set_precision": prec, "set_recall": rec, "align": align}

    def evalm(theta, theta_plain, k):
        m = eng.model(theta)
        e = {"acc": acc_of(m, eval_loader, device), "retain_change": retain_change(m, m0, eval_loader, device)}
        uk2 = float((U[k] * U[k]).sum().item())
        e["alpha"] = round(float(((theta - theta_plain) * U[k]).sum().item() / uk2), 4) if uk2 > 0 else None
        if score is not None:
            e["asr"] = score(m)
        if split is not None and k < len(split):
            loader = torch.utils.data.DataLoader(split[k], batch_size=256, shuffle=False, num_workers=0)
            e["forget_acc"] = forget_fn(m.state_dict(), loader, device)["acc"]
        return e

    # ---------- 3 정화
    def purify(theta, pop, k, attr, delta):
        pr = pop["pr"]
        m_k = eng.model(theta)
        pk = probs(m_k, pop["xs"])
        d_stat = eng.stat_of(pk, pr)
        c_star = d_stat.get("flow_to", d_stat.get("to_class"))
        changed = pk.argmax(1) != pr.argmax(1)
        act_idx = (changed & (pk.argmax(1) == c_star)).nonzero().flatten() if c_star is not None else torch.zeros(0, dtype=torch.long)
        mask = torch.ones(pop["xs"].shape[0], dtype=torch.bool)
        mask[act_idx] = False
        pred_k = pk.argmax(1)

        def collateral(pa):                                        # 비활성 입력에서 θ' 대비 바뀐 비율 (정화가 엉뚱한 곳을 건드리는가)
            return round(float((pa.argmax(1)[mask] != pred_k[mask]).float().mean()), 4)
        info = {"c_star": c_star, "c_star_attr": attr["c_star"], "class_match": bool(c_star == attr["c_star"]), "n_act": int(act_idx.numel()), "variants": []}
        theta_out, how, chosen = None, None, None
        if act_idx.numel() >= a.min_act:
            p2 = pair_at_activation(theta, pop, k, act_idx, c_star, delta)
            info["pair_act"] = p2
            S1 = set(attr["set"])
            S2 = set(p2["set"])
            info["set_agree"] = round(len(S1 & S2) / max(len(S1 | S2), 1), 3)
            S, w = (p2["set"], p2["w"]) if (a.pair_src == "act" and p2["set"]) else (attr["set"], attr["w"])
            if S:
                wmax = max(w[j] for j in S)
                w = {j: 1.0 for j in S} if a.w_mode == "uniform" else {j: (round(w[j] / wmax, 4) if wmax > 0 else 1.0) for j in S}   # 크기는 λ 가 맡는다
            info["set"], info["w"], info["set_roles"] = S, w, [roles[j] for j in S]
            if S:
                h = grad_logp(m_k, flat, pop["xs"][act_idx], c_star, device)
                hn2 = float((h * h).sum().item())
                par = {j: (float((U[j] * h).sum().item()) / hn2) * h for j in S}
                info["frac_of_partner"] = {j: round(float(par[j].norm() / U[j].norm()), 4) for j in S}
                info["cos_Uk_h"] = round(float(F.cosine_similarity(U[k], h, dim=0)), 4)
                for lam in a.lams:
                    th = theta.clone()
                    for j in S:
                        th = th - lam * w[j] * par[j]
                    pa = probs(eng.model(th), pop["xs"])
                    health = collateral(pa)
                    z2, _ = eng.check(pa, pop)
                    info["variants"].append({"lam": lam, "z": round(z2, 2), "health": round(health, 4)})
                    if health > a.cap:
                        break
                    if z2 <= a.z_target:
                        chosen, theta_out, how = lam, th.clone(), "antidote"
                        break
                info["chosen_lam"] = chosen
        if theta_out is not None and a.polish == "kd_orth":
            m_p = eng.model(theta_out)
            if a.polish_ref == "self":                            # 교사 = 정화 전 모델 θ', 비활성 입력에서만
                ok, steps = kd_orth(m_p, None, pop["xs"][mask], a.rep_steps, a.rep_lr, a.rep_bs, flat, U[k], teacher_probs=pk[mask])
            elif a.polish_ref == "low":                           # 교사 = 모집단 기준, 입력 = 요청자 제거의 영향(KL)이 가장 작은 공개 입력 절반 (지울 데이터 제외)
                n_pub = xs.shape[0]
                p0 = probs(m0, xs)
                kl = (p0 * (torch.log(p0.clamp_min(1e-12)) - torch.log(pk[:n_pub].clamp_min(1e-12)))).sum(1)
                low = kl <= kl.median()
                low = low & (pk[:n_pub].argmax(1) == pr[:n_pub].argmax(1))
                if int(low.sum()) < a.rep_bs:
                    low = kl <= kl.median()
                ok, steps = kd_orth(m_p, None, xs[low], a.rep_steps, a.rep_lr, a.rep_bs, flat, U[k], teacher_probs=pr[:n_pub][low])
                info["polish_n_inputs"] = int(low.sum())
            elif a.polish_ref == "agree":                         # 교사 = 모집단 기준, 입력 = 요청자를 빼도 답이 안 바뀐 것들만
                agree = (pk.argmax(1) == pr.argmax(1))
                if int(agree.sum()) < a.rep_bs:
                    agree = torch.ones_like(agree)
                ok, steps = kd_orth(m_p, None, pop["xs"][agree], a.rep_steps, a.rep_lr, a.rep_bs, flat, U[k], teacher_probs=pr[agree])
                info["polish_n_inputs"] = int(agree.sum())
            else:
                ok, steps = kd_orth(m_p, pop["ref"], pop["xs"], a.rep_steps, a.rep_lr, a.rep_bs, flat, U[k])
            th = flat.flatten(m_p.state_dict())
            pa = probs(m_p, pop["xs"])
            z3, _ = eng.check(pa, pop)
            health = collateral(pa)
            info["polish"] = {"ref": a.polish_ref, "ok": ok, "steps": steps, "z": round(z3, 2), "health": round(health, 4),
                              "alpha_vs_antidote": round(float(((th - theta_out) * U[k]).sum().item() / float((U[k] * U[k]).sum().item())), 6)}
            if ok and z3 <= a.tau and health <= a.cap:
                theta_out, how = th, "antidote+polish"
        if theta_out is None:                                      # 대체: 재억제 수정 (요청자 몫 일부 복원, α>0)
            m_r = eng.model(theta)
            ok, _ = eng.repair(m_r, pop, a.rep_stop)
            th = flat.flatten(m_r.state_dict())
            z2, _ = eng.check(probs(m_r, pop["xs"]), pop)
            info["fallback_kd"] = {"ok": ok, "z": round(z2, 2)}
            if z2 <= a.tau:
                theta_out, how = th, "kd"
        return theta_out, how, info

    # ---------- 요청 처리
    for k in a.requesters:
        if str(k) in res["results"]:
            continue
        t0 = time.time()
        n_models0, n_fwd0, n_attr0 = eng.n_pop_models, eng.n_forwards, cost["attr_forwards"]
        eng.forget_x = forget_sets.get(k) if a.probe_mode == "forget" else None
        r = {"role": roles[k], "decision": None}
        attr = attribute(k)
        r["attr"] = attr
        print(f"  [{k} {roles[k]}] TRACE 회전 1 (저장 기여): 짝 {attr['set']} {attr['cancel']['set_roles']} w {attr['w']} 상쇄 {attr['cancel']['ratio']} "
              f"타깃 {attr['c_star']} 몰림 {attr['conc']['score']} 크기 {attr['mag']} 반대정렬 {attr['anti']} 어디서 {attr['where']} | p {attr['pvals']} Fisher {attr['fisher_p']} z {attr['z']} → {'경보' if attr['flagged'] else '통과'}", flush=True)
        theta, removed, trace, base_id, next_id = tT.clone(), 0.0, [], 0, 1
        aborted, purifies, hows, failed, sim_flag = False, [], [], False, False
        while removed < 1 - 1e-9:
            left = 1 - removed
            delta, pop, _ = eng.dose(theta, base_id, k, left)
            base_pred = eng.base_pred(base_id, theta)
            cands = [delta] + [d for d in DELTAS if d < delta - 1e-9 and d <= left + 1e-9]
            if min(a.delta_min, left) < cands[-1] - 1e-9:
                cands.append(min(a.delta_min, left))
            for d in cands:
                req_chg = divergence(base_pred, probs(eng.model(theta - d * U[k]), xs).argmax(1), C)["changed"]
                if req_chg <= a.cap:
                    break
            if req_chg > a.cap:
                aborted = True
                trace.append({"delta": round(d, 4), "removed": round(removed, 4), "z": None, "flagged": False, "repaired": False,
                              "hard_collapse": True, "aborted": True})
                break
            if abs(d - delta) > 1e-9:
                delta, pop = d, eng.population(theta, base_id, k, d)
            m_prev = eng.model(theta) if a.op == "kd" else None
            theta = theta - delta * U[k]
            m_k = eng.model(theta)
            if a.op == "kd":
                kd_repair(m_k, m_prev, xs, a.kd_steps, a.kd_lr, a.rep_bs)
                theta = flat.flatten(m_k.state_dict())
            pk = probs(m_k, pop["xs"])
            z, d_stat = eng.check(pk, pop)
            tau_now = eng.threshold(pop)
            flagged_sim = z > tau_now
            sim_flag = sim_flag or bool(flagged_sim)
            removed += delta
            trace.append({"delta": round(delta, 4), "removed": round(removed, 4), "h": pop["h"], "n_healthy": len(pop["healthy"]),
                          "stat": d_stat[a.stat], "to_class": d_stat.get("flow_to", d_stat.get("to_class")), "src_div": d_stat.get("src_div"),
                          "z": round(z, 2), "tau": round(tau_now, 2), "flagged": bool(flagged_sim), "repaired": False, "hard_collapse": False})
            base_id, next_id = next_id, next_id + 1
            do_purify = {"sim": flagged_sim, "attr": attr["flagged"], "either": (flagged_sim or attr["flagged"])}[a.purify_on]
            if do_purify:
                if a.max_purify > 0 and len(purifies) >= a.max_purify:
                    failed = True
                    break
                theta_out, how, info = purify(theta, pop, k, attr, delta)
                purifies.append(info)
                if theta_out is None:
                    failed = True
                    break
                theta = theta_out
                hows.append(how)
                trace[-1]["repaired"], trace[-1]["how"] = True, how
                base_id, next_id = next_id, next_id + 1
        if failed:
            decision = "reject(purify_failed)"
        elif hows:
            decision = f"purified({'+'.join(hows)})" + ("(partial)" if aborted else "") + ("" if attr["flagged"] else "(sim_only)")
        elif aborted:
            decision = "accept(partial)"
        elif attr["flagged"]:
            decision = "accept(attr_suspect)"                     # 1 단계 경보, 빼 보기 통과: 수락하되 기록
        else:
            decision = "accept(confirm)"
        r["decision"] = decision
        r["confirm"] = {"trace": trace, "removed": round(removed, 4), "aborted": aborted, "flagged": sim_flag}
        r["purify"] = purifies
        r["final"] = evalm(theta, tT - removed * U[k], k)
        r["cost"] = {"attr_forwards": cost["attr_forwards"] - n_attr0, "pop_models": eng.n_pop_models - n_models0,
                     "forwards": eng.n_forwards - n_fwd0, "seconds": round(time.time() - t0, 1)}
        res["results"][str(k)] = r
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        fin = r["final"]
        print(f"  [{k} {roles[k]}] DOSE (단계 삭제·띠 확인): " + " ".join(fmt_step(dict(s, z_cum=None)) for s in trace) + (" ABORT" if aborted else ""), flush=True)
        for p_ in purifies:
            p2 = p_.get("pair_act")
            if p2:
                print(f"    TRACE 회전 2+ (활성화 입력) → DEFUSE: 짝 {p2['set']} {p2['set_roles']} w {p2['w']} 상쇄 {p2['ratio']} 정렬 {p2['align']} 반복 {p2['iters']} (회전 1 과 일치 {p_['set_agree']}, 클래스 {p_['c_star_attr']}/{p_['c_star']}) 변형 {p_['variants']}"
                      f"  λ {p_.get('chosen_lam')}" + (f"  polish {p_['polish']}" if p_.get("polish") else ""), flush=True)
        print(f"    → {decision}: ASR {fin.get('asr')} ACC {fin['acc']} retain {fin['retain_change']} α {fin['alpha']} 자기데이터 {fin.get('forget_acc')} "
              f"| 순전파 짝 {r['cost']['attr_forwards']} 빼보기 {r['cost']['forwards']} 모델 {r['cost']['pop_models']} {r['cost']['seconds']}s", flush=True)

    print(f"\n== {a.name}  before ACC {res['before']['acc']} / ASR {res['before'].get('asr')}")
    for k, r in res["results"].items():
        f = r["final"]
        s2 = [p.get("pair_act", {}).get("set") for p in r["purify"]]
        print(f"  req {k:>2} {r['role']:10s} {r['decision']:30s} 짝1 {r['attr']['set']} (상쇄 {r['attr']['cancel']['ratio']}, p {r['attr']['fisher_p']:.3f}) 짝2 {s2} ASR {str(f.get('asr')):>6} ACC {f['acc']:5.1f} α {f['alpha']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
