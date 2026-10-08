# -*- coding: utf-8 -*-
"""삭제 전후 차이만으로 '어느 클래스가 풀렸나' 후보 순위를 매긴다.

트리거·정답 라벨 정보 없이, 서버가 가진 깨끗한 탐침 이미지만 쓴다.
점수 세 가지 (모두 클수록 의심):
  logit_shift : 탐침 이미지에서 h1 - h0 로짓 변화의 클래스별 평균 (표본마다 클래스 평균을 빼서 중심화)
  head_row    : 분류 헤드에서 클래스 c 행(가중치+편향) 변화의 크기
  svd_head    : 헤드 변화 행렬의 첫 특이벡터에 실린 클래스별 크기 (WeightWatch 식)
"""
import numpy as np
import torch


class DiffLocalizer:
    def __init__(self, space, probe_x, device, head="fc3", batch=500):
        self.space = space
        self.probe_x = probe_x
        self.device = device
        self.head_w = head + ".weight"
        self.head_b = head + ".bias"
        self.batch = batch

    @torch.no_grad()
    def _logits(self, v):
        m = self.space.to_model(v, self.device)
        return torch.cat([m(self.probe_x[i:i + self.batch]) for i in range(0, len(self.probe_x), self.batch)])

    def class_scores(self, h0, h1):
        dz = self._logits(h1) - self._logits(h0)
        dz = dz - dz.mean(1, keepdim=True)
        delta = h1 - h0
        head = torch.cat([self.space.layer(delta, self.head_w),
                          self.space.layer(delta, self.head_b)[:, None]], 1)
        U, S, _ = torch.linalg.svd(head, full_matrices=False)
        return {
            "logit_shift": dz.mean(0).cpu().numpy(),
            "head_row": head.norm(dim=1).numpy(),
            "svd_head": (S[0] * U[:, 0].abs()).numpy(),
        }

    def layer_energy(self, h0, h1):
        """층별 ||Δ|| / ||W0||. 어느 층이 가장 많이 바뀌었나."""
        out = {}
        for name in self.space.layer_names():
            k = name + ".weight"
            d = self.space.layer(h1 - h0, k).norm()
            w = self.space.layer(h0, k).norm()
            out[name] = round(float(d / (w + 1e-12)), 5)
        return out

    @staticmethod
    def ranking(score):
        return [int(c) for c in np.argsort(-np.asarray(score), kind="stable")]
