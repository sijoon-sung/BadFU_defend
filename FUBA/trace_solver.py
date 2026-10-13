"""Original TRACE constrained solver, shared without loading training dependencies."""
import numpy as np


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
