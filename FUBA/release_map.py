# -*- coding: utf-8 -*-
"""빼는 행위를 측정 도구로 쓴다 — 천천히 빼기(①) + 전원 빼 보기(②). 학습 없음, 빼기와 순전파뿐.

FU 요청이 오면 서버는 "k 있는 모델"과 "k 없는 모델"을 정당하게 계산할 수 있다(FL 만으로는 생기지 않는 쌍).
  ① 천천히 빼기: 클라이언트 j 의 저장 기여 U_j 를 ρ = 0, 0.2, …, 1.0 만큼 뺀 모델 θ(ρ) = θ_T − ρ·U_j 를 만들고,
     라벨 없는 probe 에서 "삭제 전과 답이 달라진 비율(changed)"과 "달라진 것 중 한 클래스로 간 비율(conc)"을 잰다.
     mass = changed × conc. 1차 근사 빼기는 깊은 망(BN 있는 ResNet)에서 ρ 가 크면 모델을 무너뜨리므로,
     changed ≤ cap 인 가장 큰 ρ 를 **작동점 ρ\*** 로 잡고 거기서 판정한다.
  ② 전원 빼 보기: ①을 모든 클라이언트에 대해 한다. 요청자의 점수가 나머지의 분포에서 튀는지 보면 되므로
     임계값을 미리 정할 필요가 없다 (기준 분포가 요청 시점에 공짜로 생긴다).

트리거가 깨끗한 입력에 안 새는 공격(BadNet 패치 등)을 위해, ρ\* 모델과 θ_T 의 답이 가장 갈리는 입력을 서버가 직접 찾는다:
  linf  : 모든 입력에 같은 덧셈 섭동 δ (||δ||∞ ≤ eps, eps 는 픽셀 단위; 정규화 공간으로 환산)
  patch : x' = (1−m)·x + m·p (마스크 m, 패턴 p; Neural Cleanse 꼴이지만 목표는 "두 모델의 KL 최대화", 클래스 지정 없음)
찾은 입력에서 같은 통계를 다시 잰다. 트리거·타깃·라벨은 쓰지 않는다.

궤적 형식
  fuba  : checkpoints/{global,local,backdoor}_net_model_{name}_round_{r}.pkl (FUBA 공개 코드, MNIST Net)
          U_k = Σ_{r≥2} (local_r[k] − global_{r−1}) / K   (diff_audit.trajectory 와 같은 1차 근사)
  badfu : {traj_dir}/round_{r}.pt = {"gm": 라운드 시작 글로벌, "cms": [클라 state_dict ...]},
          {traj_dir}/glob_{r}.pt = {"gm_after": 라운드 끝 글로벌}  (BadFU_src/BadFU-main/fl_detect_badfu.py 가 저장)
          U_c = Σ_r w_c (cms_r[c] − gm_r),  w_c = --counts 비율 (없으면 1/K)

채점(정답 사용, 방어에는 안 씀): ρ 별 ACC(붕괴 표시)와 ASR — fuba 는 저장된 IBA 생성기, badfu 는 --badfu_record 의 bd_test.

실행 예 (FUBA 폴더에서)
  python release_map.py --fmt fuba  --name bd_owner_s0 --K 8 --rounds 8 --requester 4
  python release_map.py --fmt badfu --name badfu_full --traj_dir /abs/BadFU-main/logs/traj_full --data_root /abs/BadFU-main/data \
      --badfu_record /abs/BadFU-main/record/badnet_dataset --rounds 15 --requester 5 --attackers 0 --target 0 \
      --counts 10545 9647 9641 9649 9668 10545
출력: logs/release_map/{name}.json, 콘솔 표(클라이언트별 작동점 통계·요청자 순위)
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

RHOS_DEFAULT = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)   # BadFU 코드와 동일


# ------------------------------------------------------------------ 공통: 평탄화
class Flat:
    """state_dict 의 float 텐서를 1차원 벡터로. 비-float(예: BN num_batches_tracked)는 θ_T 값 그대로 둔다."""

    def __init__(self, sd):
        self.keys = [k for k, v in sd.items() if torch.is_floating_point(v)]
        self.shapes = {k: sd[k].shape for k in self.keys}
        self.slices, i = {}, 0
        for k in self.keys:
            n = sd[k].numel()
            self.slices[k] = slice(i, i + n)
            i += n
        self.dim = i
        self.rest = {k: v.clone() for k, v in sd.items() if not torch.is_floating_point(v)}

    def flatten(self, sd):
        return torch.cat([sd[k].detach().float().cpu().flatten() for k in self.keys])

    def to_sd(self, v):
        out = {k: v[self.slices[k]].view(self.shapes[k]).clone() for k in self.keys}
        out.update({k: t.clone() for k, t in self.rest.items()})
        return out


# ------------------------------------------------------------------ 궤적 로더
def load_fuba(a):
    from model import MNISTAutoencoder, Net
    g, l = [], []
    for r in range(1, a.rounds + 1):
        with open(f"{a.ckpt_dir}/global_net_model_{a.name}_round_{r}.pkl", "rb") as f:
            g.append(pickle.load(f))
        with open(f"{a.ckpt_dir}/local_net_model_{a.name}_round_{r}.pkl", "rb") as f:
            l.append(pickle.load(f))
    theta_T = g[-1]
    flat = Flat(theta_T)
    K = a.K
    U = [torch.zeros(flat.dim) for _ in range(K)]
    for r in range(2, a.rounds + 1):
        gv = flat.flatten(g[r - 2])
        for k in range(K):
            if l[r - 1][k] is None:
                continue
            U[k] += (flat.flatten(l[r - 1][k]) - gv) / K
    bd = None
    p = f"{a.ckpt_dir}/backdoor_net_model_{a.name}_round_{a.rounds}.pkl"
    if os.path.exists(p):
        bd = MNISTAutoencoder()
        with open(p, "rb") as f:
            bd.load_state_dict(pickle.load(f))
        bd.eval()
    return Net, theta_T, U, flat, bd


def load_badfu(a):
    sys.path.insert(0, os.path.abspath(os.path.join(a.traj_dir, "..", "..")))
    from badfu_common import ResNet18  # noqa: E402  (BadFU_src/BadFU-main/badfu_common.py)
    rounds, cms_all = [], []
    for r in range(a.rounds):
        d = torch.load(os.path.join(a.traj_dir, f"round_{r}.pt"), weights_only=False, map_location="cpu")
        rounds.append(d["gm"])
        cms_all.append(d["cms"])
    last = torch.load(os.path.join(a.traj_dir, f"glob_{a.rounds - 1}.pt"), weights_only=False, map_location="cpu")
    theta_T = last["gm_after"]
    flat = Flat(theta_T)
    K = len(cms_all[0])
    if a.counts:
        assert len(a.counts) == K, "--counts 길이가 클라 수와 다름"
        w = [c / sum(a.counts) for c in a.counts]
    else:
        w = [1.0 / K] * K
    U = [torch.zeros(flat.dim) for _ in range(K)]
    for r in range(a.rounds):
        gv = flat.flatten(rounds[r])
        for c in range(K):
            U[c] += w[c] * (flat.flatten(cms_all[r][c]) - gv)
    return (lambda: ResNet18(pretrained=False)), theta_T, U, flat, None


# ------------------------------------------------------------------ 데이터
def load_probe(a):
    if a.fmt == "fuba":
        from dataset import MNIST
        tf = transforms.Compose([transforms.ToTensor()])
        test = MNIST(root="./data", train=False, download=True, transform=tf)
    else:
        import torchvision
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        test = torchvision.datasets.CIFAR10(root=a.data_root, train=False, download=True, transform=tf)
    probe = torch.utils.data.Subset(test, list(range(0, a.probe)))
    evalset = torch.utils.data.Subset(test, list(range(a.probe, len(test))))
    return (torch.utils.data.DataLoader(probe, batch_size=256, shuffle=False, num_workers=0),
            torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=0))


class InputSpace:
    """입력의 유효 범위와 '픽셀 단위 eps' 환산. fuba: [0,1] 그대로. badfu: 채널별 정규화 공간."""

    def __init__(self, fmt, device):
        if fmt == "fuba":
            self.lo = torch.zeros(1, 1, 1, 1, device=device)
            self.hi = torch.ones(1, 1, 1, 1, device=device)
            self.std = torch.ones(1, 1, 1, 1, device=device)
        else:
            m = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
            s = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
            self.lo, self.hi, self.std = (0 - m) / s, (1 - m) / s, s

    def clamp(self, x):
        return torch.max(torch.min(x, self.hi), self.lo)

    def bound(self, eps_pixel):          # 픽셀 단위 eps -> 공간별 채널 한계
        return eps_pixel / self.std


# ------------------------------------------------------------------ 측정
@torch.no_grad()
def predict(model, xs, device, bs=256):
    out = []
    for i in range(0, len(xs), bs):
        out.append(model(xs[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out)


def divergence(pred0, pred1, n_classes):
    """삭제 전 답(pred0) 대비 달라진 비율과, 달라진 것들이 어느 한 클래스로 몰린 비율."""
    changed = pred0 != pred1
    frac = changed.float().mean().item()
    if changed.sum() == 0:
        return {"changed": 0.0, "conc": 0.0, "to_class": None, "mass": 0.0}
    hist = torch.bincount(pred1[changed], minlength=n_classes).float()
    c = int(hist.argmax())
    conc = (hist[c] / hist.sum()).item()
    return {"changed": round(frac, 4), "conc": round(conc, 4), "to_class": c, "mass": round(frac * conc, 4)}


class Ensemble:
    """기준 모델: 나머지 클라이언트들을 같은 ρ 만큼 뺀 모델들의 **중앙값** 확률(클래스별, 재정규화).
    '누구를 빼도 생기는 열화(깊은 망의 빼기 붕괴)'를 상쇄하고 'k 를 뺄 때만 생기는 변화'만 남긴다.
    평균 대신 중앙값: 한 멤버가 붕괴해 확률 1.0 을 한 클래스에 몰아도 기준이 끌려가지 않게."""

    def __init__(self, models):
        self.models = models

    def __call__(self, x):                     # 로짓 대신 log(중앙값 softmax) 를 돌려준다
        ps = torch.stack([torch.nan_to_num(F.softmax(m(x), 1), nan=0.0, posinf=1.0, neginf=0.0) for m in self.models], 0)
        p = ps.median(0).values
        p = p / p.sum(1, keepdim=True).clamp_min(1e-12)
        return torch.log(p.clamp_min(1e-12))


def _kl(ref, m1, x):
    """KL( p1 || p_ref ): ref 는 nn.Module(로짓) 또는 Ensemble(log-prob)."""
    lp1 = F.log_softmax(m1(x), 1)
    out = ref(x)
    p0 = out.exp() if isinstance(ref, Ensemble) else F.softmax(out, 1)
    return F.kl_div(lp1, p0, reduction="batchmean")


@torch.no_grad()
def probs(model, xs, bs=256):
    out = []
    for i in range(0, len(xs), bs):
        o = model(xs[i:i + bs])
        p = o.exp() if isinstance(model, Ensemble) else F.softmax(o, 1)
        out.append(torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).cpu())
    return torch.cat(out)


def shift(p_k, p_ref):
    """평균 확률 이동: max_c mean_x [p_k(c|x) − p_ref(c|x)] 와 그 클래스. argmax 가 안 바뀌어도 쏠림을 본다."""
    d = (p_k - p_ref).mean(0)
    c = int(d.argmax())
    return {"shift": round(float(d[c]), 4), "shift_class": c}


def find_linf(m0, m1, xs, space, eps, steps, lr):
    """두 모델의 답이 가장 갈리는 보편 덧셈 섭동 δ (||δ||∞ ≤ eps, 픽셀 단위)."""
    b = space.bound(eps)
    delta = torch.zeros_like(xs[:1], requires_grad=True)
    for _ in range(steps):
        kl = _kl(m0, m1, space.clamp(xs + delta))
        g, = torch.autograd.grad(kl, delta)
        with torch.no_grad():
            delta += lr * g.sign()
            delta.copy_(torch.max(torch.min(delta, b), -b))
    return delta.detach()


def find_patch(m0, m1, xs, space, lam, steps, lr):
    """두 모델의 답이 가장 갈리는 패치: x' = (1−m)x + m·p, max KL − lam·|m|_1 (클래스 지정 없음)."""
    _, C, H, W = xs.shape
    mp = torch.full((1, 1, H, W), -3.0, device=xs.device, requires_grad=True)   # sigmoid(-3) ≈ 0.05
    pp = torch.zeros((1, C, H, W), device=xs.device, requires_grad=True)
    opt = torch.optim.Adam([mp, pp], lr=lr)
    for _ in range(steps):
        m = torch.sigmoid(mp)
        p = space.lo + torch.sigmoid(pp) * (space.hi - space.lo)
        xa = (1 - m) * xs + m * p
        loss = -_kl(m0, m1, xa) + lam * m.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m = torch.sigmoid(mp).detach()
    p = (space.lo + torch.sigmoid(pp) * (space.hi - space.lo)).detach()
    return m, p, round(float(m.mean()), 4)


def find_class_patch(ref, m1, xs, space, c, lam, steps, lr):
    """클래스 c 를 겨냥한 패치: x' = (1−m)x + m·p 에서 mean[p1(c|x') − p_ref(c|x')] 를 최대화 (Neural Cleanse 꼴, 기준은 ref).
    '기준보다 c 로 더 많이 넘어가게 만드는 입력'을 찾는다. 공통 열화는 ref 에 들어 있으므로 k 만의 상승이 남는다."""
    _, C, H, W = xs.shape
    mp = torch.full((1, 1, H, W), -3.0, device=xs.device, requires_grad=True)
    pp = torch.zeros((1, C, H, W), device=xs.device, requires_grad=True)
    opt = torch.optim.Adam([mp, pp], lr=lr)
    for _ in range(steps):
        m = torch.sigmoid(mp)
        p = space.lo + torch.sigmoid(pp) * (space.hi - space.lo)
        xa = (1 - m) * xs + m * p
        p1 = F.softmax(m1(xa), 1)[:, c]
        out = ref(xa)
        pr = (out.exp() if isinstance(ref, Ensemble) else F.softmax(out, 1))[:, c]
        loss = -(p1 - pr).mean() + lam * m.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m = torch.sigmoid(mp).detach()
    p = (space.lo + torch.sigmoid(pp) * (space.hi - space.lo)).detach()
    return m, p


def grid_candidates(xs, space, sizes, stride):
    """구조화된 섭동 열거: 작은 정사각 패치(크기 s) × 위치 격자(stride, 모서리 포함) × 패턴(흰·검·체크).
    기울기 없이 붙여 보기만 한다. BadNet 류 국소 트리거를 겨냥한 '단순 패턴 사전'(Universal Litmus Patterns 과 같은 발상)."""
    _, C, H, W = xs.shape
    for s in sizes:
        ys = sorted(set(list(range(0, H - s + 1, stride)) + [H - s]))
        xs_ = sorted(set(list(range(0, W - s + 1, stride)) + [W - s]))
        for y in ys:
            for x in xs_:
                for name in ("white", "black", "checker"):
                    m = torch.zeros(1, 1, H, W, device=xs.device)
                    m[:, :, y:y + s, x:x + s] = 1.0
                    if name == "white":
                        p = space.hi.expand(1, C, H, W)
                    elif name == "black":
                        p = space.lo.expand(1, C, H, W)
                    else:
                        ck = ((torch.arange(H, device=xs.device).view(H, 1) + torch.arange(W, device=xs.device).view(1, W)) % 2).float()
                        p = space.lo + ck.view(1, 1, H, W) * (space.hi - space.lo)
                    yield {"size": s, "y": y, "x": x, "pattern": name}, m, p


@torch.no_grad()
def grid_search(ref, m1, xs, space, n_classes, sizes, stride):
    """후보마다 기준 대비 클래스별 확률 상승 mean[p1(c) − p_ref(c)] 를 재고, 최대 후보·클래스를 고른다."""
    best = None
    for info, m, p in grid_candidates(xs, space, sizes, stride):
        xa = (1 - m) * xs + m * p
        pr, pk = probs(ref, xa), probs(m1, xa)
        gain = (pk - pr).mean(0)
        c = int(gain.argmax())
        g = float(gain[c])
        if best is None or g > best["shift"]:
            d = divergence(pr.argmax(1), pk.argmax(1), n_classes)
            d.update({"shift": round(g, 4), "shift_class": c, "cand": info})
            best = d
    return best


def make_model(factory, sd, device):
    m = factory()
    m.load_state_dict(sd)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def kd_repair(model, teacher, xs, steps, lr, bs=128):
    """'빼기'를 덜 파괴적으로: 1차 근사로 뺀 모델을 라벨 없는 probe 에서 θ_T 쪽으로 짧게 증류한다 (KD-FU 의 증류 단계와 같은 꼴).

    깊은 망에서 기여를 그냥 빼면 각 클라이언트마다 제멋대로 예측이 흔들려(개인별 잡음) 풀린 백도어가 묻힌다.
    증류는 깨끗한 입력에서만 θ_T 를 따라가므로 그 잡음은 지우지만, 트리거 입력의 억제는 되살리지 못한다
    (KD-FU 가 44% 를 남긴 이유). 즉 신호는 남고 잡음만 준다. 라벨·트리거 없음.
    """
    if steps <= 0:
        return model
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    for m in model.modules():                       # BN 통계는 건드리지 않는다 (probe 500장으로 재추정하면 모델이 바뀜)
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    n = len(xs)
    g = torch.Generator(device="cpu").manual_seed(0)
    diverged = False
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        x = xs[idx]
        with torch.no_grad():
            t = F.softmax(teacher(x), 1)
        loss = F.kl_div(F.log_softmax(model(x), 1), t, reduction="batchmean")
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)     # 깊은 망에서 발산 방지
        opt.step()
    model.eval()
    with torch.no_grad():
        bad = diverged or any(not torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_floating_point(v))
    if bad:                                                         # 발산하면 증류 전(빼기만 한) 모델로 되돌린다
        model.load_state_dict(backup)
        model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def asr_fuba(model, loader, bd, device, target, atk_eps, thr):
    from utils.comm_utils import attack
    return round(attack(model, loader, bd.to(device), atk_eps, thr, target, device), 2)


@torch.no_grad()
def acc_of(model, loader, device):
    cor = tot = 0
    for x, y in loader:
        cor += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        tot += len(y)
    return round(100.0 * cor / tot, 2)


def make_scorer(a, bd, eval_loader, device):
    """채점용 ASR 함수 (정답 사용). fuba: 저장된 IBA 생성기. badfu: record 의 bd_test (트리거 붙은 비-타깃 테스트 이미지)."""
    if a.fmt == "fuba" and bd is not None:
        return lambda m: asr_fuba(m, eval_loader, bd, device, a.target, a.atk_eps, a.noise_thread)
    if a.fmt == "badfu" and a.badfu_record:
        import badfu_common as bc
        d = torch.load(os.path.join(a.badfu_record, "pert_result.pt"), weights_only=False)
        dd = d["bd_test"]["bd_data_container"]["data_dict"]
        try:
            ds = bc.BackdoorDataset(dd)
        except TypeError:
            ds = bc.BackdoorDataset(dd, bc.TRANSFORM)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

        @torch.no_grad()
        def score(m):
            hit = tot = 0
            for x, _ in loader:
                p = m(x.to(device)).argmax(1).cpu()
                hit += (p == a.target).sum().item()
                tot += len(p)
            return round(100.0 * hit / tot, 2)
        return score
    return None


# ------------------------------------------------------------------ main
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba", "badfu"], default="fuba")
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8, help="fuba: 클라 수 (badfu 는 궤적에서 읽음)")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--traj_dir", default=None, help="badfu: round_r.pt 가 있는 폴더 (절대 경로)")
    p.add_argument("--counts", type=int, nargs="*", default=None, help="badfu: 클라별 데이터 수 (FedAvg 가중)")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--badfu_record", default=None, help="badfu 채점용: pert_result.pt 가 있는 record 폴더 (없으면 ASR 생략)")
    p.add_argument("--requester", type=int, default=4, help="채점용: 공격이 설계한 요청자")
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3], help="채점용")
    p.add_argument("--target", type=int, default=8, help="채점용 실제 타깃")
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500, help="서버의 라벨 없는 probe = test[0:N]; 채점은 test[N:]")
    p.add_argument("--rhos", type=float, nargs="+", default=RHOS_DEFAULT)
    p.add_argument("--cap", type=float, default=0.5, help="작동점: 깨끗한 probe 에서 θ_T 대비 changed ≤ cap 인 가장 큰 ρ")
    p.add_argument("--op", choices=["subtract", "kd"], default="subtract",
                   help="빼는 연산. subtract: θ_T − ρ·U_j 그대로. kd: 뺀 뒤 probe 로 θ_T 쪽에 짧게 증류(잡음 제거, 라벨 없음)")
    p.add_argument("--kd_steps", type=int, default=50)
    p.add_argument("--kd_lr", type=float, default=0.001)
    p.add_argument("--ref", choices=["theta", "population"], default="population",
                   help="갈림의 기준. theta: 삭제 전 모델. population: 나머지 전원을 같은 ρ 만큼 뺀 모델들의 평균(열화 상쇄)")
    p.add_argument("--delta_mode", choices=["none", "linf", "patch", "both", "class", "grid", "all"], default="both",
                   help="ρ* 모델과 기준의 갈림을 키우는 입력: linf / patch / both(둘) / class(클래스별 목표 패치) / grid(패치 열거, 기울기 없음) / all(전부)")
    p.add_argument("--grid_sizes", type=int, nargs="+", default=[3, 5], help="grid: 패치 한 변 길이")
    p.add_argument("--grid_stride", type=int, default=8, help="grid: 위치 격자 간격 (모서리는 항상 포함)")
    p.add_argument("--class_steps", type=int, default=60, help="class 모드: 클래스당 탐색 스텝 (클래스 수만큼 반복)")
    p.add_argument("--delta", action="store_true", help="(호환) --delta_mode both 와 같음")
    p.add_argument("--eps", type=float, default=0.1, help="linf: 픽셀 단위 L∞ 예산")
    p.add_argument("--pgd_steps", type=int, default=50)
    p.add_argument("--pgd_lr", type=float, default=0.01)
    p.add_argument("--patch_lam", type=float, default=0.5, help="patch: 마스크 평균 크기에 대한 벌점")
    p.add_argument("--patch_steps", type=int, default=100)
    p.add_argument("--patch_lr", type=float, default=0.1)
    p.add_argument("--n_opt", type=int, default=256, help="섭동 탐색에 쓰는 probe 수 (측정은 probe 전체)")
    p.add_argument("--atk_eps", type=float, default=1.0)
    p.add_argument("--noise_thread", type=float, default=0.04)
    p.add_argument("--clients", type=int, nargs="*", default=None, help="빼 볼 클라 (기본 전원)")
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse()
    if a.delta and a.delta_mode == "none":
        a.delta_mode = "both"
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    out = a.out or f"logs/release_map/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    K = len(U)
    clients = a.clients if a.clients else list(range(K))
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)          # probe 전체 (라벨 없음)
    xo = xs[:a.n_opt]
    space = InputSpace(a.fmt, device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    pred0 = predict(m0, xs, device)
    score = make_scorer(a, bd, eval_loader, device)
    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "rhos": a.rhos,
           "before": {"acc": acc_of(m0, eval_loader, device)}, "clients": {}}
    if score is not None:
        res["before"]["asr"] = score(m0)
    print(f"[{a.name}] K={K} probe={a.probe} before={res['before']}", flush=True)

    # 1) 모든 (클라이언트, ρ) 모델과 probe 확률을 먼저 만든다 (population 기준에 필요)
    t0 = time.time()
    models = {rho: {} for rho in a.rhos}
    P = {rho: {} for rho in a.rhos}
    curves, acc_curves, asr_curves = {j: [] for j in clients}, {j: [] for j in clients}, {j: [] for j in clients}
    for rho in a.rhos:
        for j in clients:
            m = make_model(factory, flat.to_sd(tT - rho * U[j]), device)
            if a.op == "kd" and rho > 0:
                m = kd_repair(m, m0, xs, a.kd_steps, a.kd_lr)
            models[rho][j] = m
            P[rho][j] = probs(m, xs)
            d = divergence(pred0, P[rho][j].argmax(1), a.n_classes)       # θ_T 대비 (붕괴 감시·작동점 선택용)
            d["rho"] = rho
            curves[j].append(d)
            acc_curves[j].append(acc_of(m, eval_loader, device))
            if score is not None:
                asr_curves[j].append(score(m))
    print(f"  모델 {len(a.rhos) * len(clients)}개 준비 {time.time() - t0:.0f}s", flush=True)

    def reference(rho, j):
        """갈림의 기준: θ_T 또는 '나머지 전원을 같은 ρ 만큼 뺀 모델들'의 중앙값.
        같은 ρ 에서 이미 붕괴한 멤버(θ_T 대비 changed > cap)는 기준에서 뺀다. 남는 멤버가 없으면 θ_T."""
        if a.ref == "theta" or len(clients) < 2:
            return m0
        ok = [i for i in clients if i != j and next(c for c in curves[i] if c["rho"] == rho)["changed"] <= a.cap]
        if not ok:
            return m0
        return Ensemble([models[rho][i] for i in ok])

    # 2) 클라이언트마다 작동점에서 기준 대비 쏠림을 잰다
    for j in clients:
        t0 = time.time()
        curve = curves[j]
        # 작동점 ρ*: θ_T 대비 changed ≤ cap 인 가장 큰 ρ (0 제외). 없으면 가장 작은 양의 ρ.
        ok = [c["rho"] for c in curve if c["rho"] > 0 and c["changed"] <= a.cap]
        rho_star = max(ok) if ok else min(r for r in a.rhos if r > 0)
        m_star = models[rho_star][j]
        ref = reference(rho_star, j)
        p_ref = probs(ref, xs)
        p_k = P[rho_star][j]
        wp = {"rho": rho_star, "clean": divergence(p_ref.argmax(1), p_k.argmax(1), a.n_classes)}
        wp["clean"].update(shift(p_k, p_ref))
        # ρ 별 기준 대비 곡선 (population 기준이면 같은 ρ 의 나머지 평균)
        curve_ref = []
        for rho in a.rhos:
            if rho == 0:
                curve_ref.append({"rho": 0.0, "changed": 0.0, "conc": 0.0, "to_class": None, "mass": 0.0, "shift": 0.0, "shift_class": None})
                continue
            pr = probs(reference(rho, j), xs) if a.ref == "population" else probs(m0, xs)
            d = divergence(pr.argmax(1), P[rho][j].argmax(1), a.n_classes)
            d.update(shift(P[rho][j], pr))
            d["rho"] = rho
            curve_ref.append(d)
        if a.delta_mode in ("grid", "all"):
            wp["grid"] = grid_search(ref, m_star, xs, space, a.n_classes, a.grid_sizes, a.grid_stride)
        if a.delta_mode in ("class", "all"):
            best = None
            for c in range(a.n_classes):
                mk, pt = find_class_patch(ref, m_star, xo, space, c, a.patch_lam, a.class_steps, a.patch_lr)
                xa = (1 - mk) * xs + mk * pt
                pr, pk = probs(ref, xa), probs(m_star, xa)
                gain = float((pk[:, c] - pr[:, c]).mean())          # 기준 대비 c 확률 상승 (probe 전체)
                if best is None or gain > best["shift"]:
                    d = divergence(pr.argmax(1), pk.argmax(1), a.n_classes)
                    d.update({"shift": round(gain, 4), "shift_class": c, "mask_frac": round(float(mk.mean()), 4)})
                    best = d
            wp["class"] = best
        if a.delta_mode in ("linf", "both", "all"):
            delta = find_linf(ref, m_star, xo, space, a.eps, a.pgd_steps, a.pgd_lr)
            xa = space.clamp(xs + delta)
            pr, pk = probs(ref, xa), probs(m_star, xa)
            wp["linf"] = divergence(pr.argmax(1), pk.argmax(1), a.n_classes)
            wp["linf"].update(shift(pk, pr))
        if a.delta_mode in ("patch", "both", "all"):
            mk, pt, mfrac = find_patch(ref, m_star, xo, space, a.patch_lam, a.patch_steps, a.patch_lr)
            xa = (1 - mk) * xs + mk * pt
            pr, pk = probs(ref, xa), probs(m_star, xa)
            wp["patch"] = divergence(pr.argmax(1), pk.argmax(1), a.n_classes)
            wp["patch"].update(shift(pk, pr))
            wp["patch"]["mask_frac"] = mfrac
        entry = {"role": "requester" if j == a.requester else ("attacker" if j in a.attackers else "benign"),
                 "curve": curve, "curve_ref": curve_ref, "acc_curve": acc_curves[j], "asr_curve": asr_curves[j] or None,
                 "wp": wp, "seconds": round(time.time() - t0, 1)}
        res["clients"][str(j)] = entry
        line = f"  client {j} ({entry['role']:9s}) ρ*={rho_star:.1f}"
        for k in ("clean", "linf", "patch", "class", "grid"):
            if k in wp:
                w = wp[k]
                line += f"  | {k} mass {w['mass']:.3f} (chg {w['changed']:.2f} conc {w['conc']:.2f} -> {w['to_class']}) shift {w['shift']:+.3f}->{w['shift_class']}"
                if k == "grid":
                    line += f" @{w['cand']['pattern']}{w['cand']['size']}({w['cand']['y']},{w['cand']['x']})"
        if asr_curves[j]:
            line += f"  | ASR ρ: {asr_curves[j]}  ACC ρ: {acc_curves[j]}"
        print(line, flush=True)
    del models

    # ② 전원 빼 보기: 요청자가 나머지 분포에서 어디에 있나 (작동점 통계별)
    def rank_of(key, field="mass"):
        vals = {j: res["clients"][str(j)]["wp"][key][field] for j in clients if key in res["clients"][str(j)]["wp"]}
        if a.requester not in vals:
            return None
        order = sorted(vals, key=lambda j: -vals[j])
        others = [vals[j] for j in vals if j != a.requester]
        mu, sd = (float(np.mean(others)), float(np.std(others))) if others else (0.0, 0.0)
        z = (vals[a.requester] - mu) / sd if sd > 0 else None
        return {"rank": order.index(a.requester) + 1, "of": len(order), "value": vals[a.requester],
                "others_mean": round(mu, 4), "others_std": round(sd, 4), "z": round(z, 2) if z is not None else None}
    res["requester_rank"] = {}
    for k in ("clean", "linf", "patch", "class", "grid"):
        for field in ("mass", "shift"):
            r = rank_of(k, field)
            if r is not None:
                res["requester_rank"][f"{k}_{field}"] = r
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n요청자 {a.requester} 순위 (작동점): " + ", ".join(f"{k} {v['rank']}/{v['of']} (z {v['z']})" for k, v in res["requester_rank"].items()))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
