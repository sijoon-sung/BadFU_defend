# -*- coding: utf-8 -*-
"""삭제 한 건에 대한 백도어 재검사 방식 비교.

모든 방식의 점수 = 클래스 c 로 가는 쉬움이 삭제로 얼마나 늘었나:
    gain(c) = ease(h1, c) - ease(h0, c)
h0 는 모든 삭제 요청에서 같으므로 ease(h0, ·) 는 학습 끝에 한 번만 재면 된다(일회성 비용).
요청마다 드는 비용은 h1(또는 확대 모델) 쪽 역추적 횟수만 센다.

  full_diff    : 10개 클래스 전부 gain                      요청당 역추적 C 회
  localized    : 차이 점수 상위 m 개 클래스만 gain           요청당 m 회
  extrapolated : h_a = h0 + a (h1 - h0) 에서 상위 m 개 gain   요청당 m 회 (a 하나당)
  post_only    : 차이 없이 h1 만 전체 검사 (Neural Cleanse 식 MAD 이상지수)  요청당 C 회
"""
import numpy as np


class DeletionAuditor:
    def __init__(self, space, inverter, probe, device, n_classes=10):
        self.space = space
        self.inv = inverter
        self.x_opt, self.y_opt, self.x_eval, self.y_eval = probe
        self.device = device
        self.C = n_classes
        self._cache = {}
        self._secs = []

    def ease(self, tag, v, c):
        key = (tag, c)
        if key not in self._cache:
            m = self.space.to_model(v, self.device)
            r = self.inv.run(m, self.x_opt, self.y_opt, self.x_eval, self.y_eval, c)
            self._secs.append(r["seconds"])
            self._cache[key] = r
        return self._cache[key]

    def sec_per_inversion(self):
        return float(np.mean(self._secs)) if self._secs else 0.0

    def baseline(self, h0):
        """h0 전체 클래스 (일회성)."""
        return {c: self.ease("h0", h0, c) for c in range(self.C)}

    def _gain(self, tag, v, h0, classes):
        return {c: self.ease(tag, v, c)["ease"] - self.ease("h0", h0, c)["ease"] for c in classes}

    @staticmethod
    def _pack(gains, n_inv):
        flag = max(gains, key=gains.get)
        return {"gains": {int(c): round(g, 4) for c, g in gains.items()},
                "flag_class": int(flag), "score": round(float(gains[flag]), 4),
                "per_request_inversions": n_inv}

    def full_diff(self, tag, h0, h1):
        return self._pack(self._gain(tag, h1, h0, range(self.C)), self.C)

    def localized(self, tag, h0, h1, ranked, m):
        return self._pack(self._gain(tag, h1, h0, ranked[:m]), m)

    def extrapolated(self, tag, h0, h1, ranked, m, alpha):
        h_a = h0 + alpha * (h1 - h0)
        return self._pack(self._gain(f"{tag}@a{alpha}", h_a, h0, ranked[:m]), m)

    def post_only(self, tag, h1):
        e = np.array([self.ease(tag, h1, c)["ease"] for c in range(self.C)])
        med = np.median(e)
        mad = max(1.4826 * np.median(np.abs(e - med)), 1e-3)  # 클래스 대부분이 같은 값이면 MAD=0 이 되므로 하한
        a = (e - med) / mad
        flag = int(a.argmax())
        return {"anomaly_index": [round(float(x), 3) for x in a], "flag_class": flag,
                "score": round(float(a[flag]), 3), "per_request_inversions": self.C}
