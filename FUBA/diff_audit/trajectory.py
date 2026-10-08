# -*- coding: utf-8 -*-
"""저장된 FUBA 궤적 -> 최종 모델 theta_T 와 클라이언트별 기여 U_k.

기여는 pipeline_detect_purify.py 와 같은 1차 근사:
    theta_T ~= theta_1 + sum_k U_k,   U_k = sum_{r>=2} (local_r[k] - global_{r-1}) / K
따라서 클라이언트 k 를 지운 모델은 h1 = theta_T - U_k (연합학습의 빼기 방식 언러닝).
"""
import os
import pickle

import torch

from model import Net, MNISTAutoencoder


class FlatSpace:
    """모델 파라미터를 1차원 벡터로 다룬다."""

    def __init__(self, net_cls=Net):
        self.net_cls = net_cls
        sd = net_cls().state_dict()
        self.keys = list(sd.keys())
        self.shapes = {k: sd[k].shape for k in self.keys}
        self.slices = {}
        i = 0
        for k in self.keys:
            n = sd[k].numel()
            self.slices[k] = slice(i, i + n)
            i += n
        self.dim = i

    def flatten(self, sd):
        return torch.cat([sd[k].detach().float().cpu().flatten() for k in self.keys])

    def to_model(self, v, device):
        m = self.net_cls()
        sd = m.state_dict()
        for k in self.keys:
            sd[k] = v[self.slices[k]].view(self.shapes[k]).to(sd[k].dtype)
        m.load_state_dict(sd)
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def layer(self, v, key):
        return v[self.slices[key]].view(self.shapes[key])

    def layer_names(self):
        names = []
        for k in self.keys:
            n = k.rsplit(".", 1)[0]
            if n not in names:
                names.append(n)
        return names


class FubaTrajectory:
    """checkpoints/{global,local,backdoor}_net_model_{name}_round_{r}.pkl 읽기."""

    def __init__(self, name, rounds, n_clients, ckpt_dir="checkpoints", space=None):
        self.name = name
        self.rounds = rounds
        self.n_clients = n_clients
        self.ckpt_dir = ckpt_dir
        self.space = space or FlatSpace()

    def _path(self, kind, r):
        return os.path.join(self.ckpt_dir, f"{kind}_net_model_{self.name}_round_{r}.pkl")

    def _load(self, kind, r):
        with open(self._path(kind, r), "rb") as f:
            return pickle.load(f)

    def global_vec(self, r):
        return self.space.flatten(self._load("global", r))

    def local_vecs(self, r):
        return [self.space.flatten(sd) for sd in self._load("local", r)]

    def contributions(self):
        K = self.n_clients
        theta_T = self.global_vec(self.rounds)
        U = [torch.zeros(self.space.dim) for _ in range(K)]
        for r in range(2, self.rounds + 1):
            g = self.global_vec(r - 1)
            L = self.local_vecs(r)
            for k in range(K):
                U[k] += (L[k] - g) / K
        return theta_T, U

    def trigger_generator(self, device):
        """평가 전용(정답 ASR). 방어 쪽 코드는 이것을 쓰지 않는다."""
        bd = MNISTAutoencoder().to(device)
        bd.load_state_dict(self._load("backdoor", self.rounds))
        bd.eval()
        return bd
