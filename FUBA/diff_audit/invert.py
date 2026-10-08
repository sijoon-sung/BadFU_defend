# -*- coding: utf-8 -*-
"""클래스 c 로 넘어가게 하는 '작은 입력 변화'를 찾아서, 얼마나 쉽게 넘어가는지(ease)를 잰다.

ease 는 클수록 c 로 가기 쉽다 = 백도어 의심. 세 가지 역추적기:
  universal  : 모든 이미지에 같은 덧셈 섭동 δ (||δ||∞ <= eps). ease = 평가셋 도달률.
  per_sample : 이미지마다 따로 섭동 (PGD). ease = 도달률. IBA 처럼 입력마다 다른 트리거에 가까움.
  patch      : Neural Cleanse 식 마스크+패턴. ease = -(마스크 L1 / 픽셀 수). 작은 마스크로 넘어가면 의심.
초기값이 결정적(0 또는 상수)이라 같은 모델·클래스면 같은 결과가 나온다.
"""
import time

import torch
import torch.nn.functional as F


class Inverter:
    kind = "base"

    def __init__(self, steps, clamp=(-1.0, 1.0)):
        self.steps = steps
        self.clamp = clamp

    def run(self, model, x_opt, y_opt, x_eval, y_eval, target):
        t0 = time.perf_counter()
        out = self._run(model, x_opt[y_opt != target], x_eval[y_eval != target], target)
        if x_eval.is_cuda:
            torch.cuda.synchronize()
        out["seconds"] = time.perf_counter() - t0
        out["grad_steps"] = self.steps
        return out

    def _run(self, model, xo, xe, target):
        raise NotImplementedError

    @torch.no_grad()
    def _reach(self, model, x, target):
        p = F.softmax(model(x), 1)
        return float((p.argmax(1) == target).float().mean()), float(p[:, target].mean())


class UniversalAdditiveInverter(Inverter):
    kind = "universal"

    def __init__(self, eps=0.04, steps=100, clamp=(-1.0, 1.0)):
        super().__init__(steps, clamp)
        self.eps = eps
        self.alpha = 2.5 * eps / steps

    def _run(self, model, xo, xe, target):
        delta = torch.zeros((1,) + xo.shape[1:], device=xo.device, requires_grad=True)
        tgt = torch.full((len(xo),), target, device=xo.device, dtype=torch.long)
        for _ in range(self.steps):
            loss = F.cross_entropy(model((xo + delta).clamp(*self.clamp)), tgt)
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= self.alpha * g.sign()
                delta.clamp_(-self.eps, self.eps)
        reach, prob = self._reach(model, (xe + delta.detach()).clamp(*self.clamp), target)
        return {"ease": reach, "reach": reach, "target_prob": prob}


class PerSampleAdditiveInverter(Inverter):
    kind = "per_sample"

    def __init__(self, eps=0.04, steps=50, clamp=(-1.0, 1.0)):
        super().__init__(steps, clamp)
        self.eps = eps
        self.alpha = 2.5 * eps / steps

    def _run(self, model, xo, xe, target):
        # 표본별 공격이라 일반화할 대상이 없다: 평가셋에서 바로 잰다 (xo 는 쓰지 않음).
        delta = torch.zeros_like(xe, requires_grad=True)
        tgt = torch.full((len(xe),), target, device=xe.device, dtype=torch.long)
        for _ in range(self.steps):
            loss = F.cross_entropy(model((xe + delta).clamp(*self.clamp)), tgt)
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= self.alpha * g.sign()
                delta.clamp_(-self.eps, self.eps)
        reach, prob = self._reach(model, (xe + delta.detach()).clamp(*self.clamp), target)
        return {"ease": reach, "reach": reach, "target_prob": prob}


class PatchInverter(Inverter):
    """Neural Cleanse: x' = (1-m) x + m p,  min CE(x', c) + lam * |m|_1."""
    kind = "patch"

    def __init__(self, lam=1e-2, steps=300, lr=0.1, clamp=(0.0, 1.0)):
        super().__init__(steps, clamp)
        self.lam = lam
        self.lr = lr

    def _run(self, model, xo, xe, target):
        _, C, H, W = xo.shape
        m_raw = torch.zeros((1, 1, H, W), device=xo.device, requires_grad=True)
        p_raw = torch.zeros((1, C, H, W), device=xo.device, requires_grad=True)
        opt = torch.optim.Adam([m_raw, p_raw], lr=self.lr)
        tgt = torch.full((len(xo),), target, device=xo.device, dtype=torch.long)
        lo, hi = self.clamp
        for _ in range(self.steps):
            m = torch.sigmoid(m_raw)
            p = lo + (hi - lo) * torch.sigmoid(p_raw)
            loss = F.cross_entropy(model((1 - m) * xo + m * p), tgt) + self.lam * m.abs().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = torch.sigmoid(m_raw)
            p = lo + (hi - lo) * torch.sigmoid(p_raw)
            reach, prob = self._reach(model, (1 - m) * xe + m * p, target)
            l1 = float(m.sum()) / (H * W)
        return {"ease": -l1, "mask_l1_frac": l1, "reach": reach, "target_prob": prob}


def make_inverter(kind, eps, steps, lam):
    if kind == "universal":
        return UniversalAdditiveInverter(eps=eps, steps=steps)
    if kind == "per_sample":
        return PerSampleAdditiveInverter(eps=eps, steps=steps)
    if kind == "patch":
        return PatchInverter(lam=lam, steps=steps)
    raise ValueError(kind)
