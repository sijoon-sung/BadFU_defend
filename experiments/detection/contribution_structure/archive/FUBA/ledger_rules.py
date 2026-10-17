"""Original CCD numerical rules, shared without importing torch or training code."""
import numpy as np


def mad_z(v):
    v = np.asarray(v, float); med = float(np.median(v)); mad = 1.4826 * float(np.median(np.abs(v - med)))
    return (v - med) / mad if mad > 1e-9 else np.zeros_like(v)


def judge(P, n_present, a):
    """장부 한 장(P: K×C)에서 클래스별 판정."""
    K, C = P.shape
    elig = np.array([n_present[j] >= a.min_present for j in range(K)])
    out = {}
    for c in range(C):
        col = P[:, c]
        z = mad_z(col[elig]) if elig.sum() >= 3 else np.zeros(int(elig.sum()))
        idx = np.where(elig)[0]
        if len(idx) == 0:
            out[c] = {"H": 0.0, "neg": -1, "negs": [], "z_neg": 0.0, "pos": [], "top_share": 1.0, "hedge": False, "coalition": False, "kind": ""}; continue
        zi = {int(j): float(z[i]) for i, j in enumerate(idx)}
        neg = int(idx[np.argmin(col[idx])])
        negs = [int(j) for j in idx if zi[int(j)] <= -a.tau] if a.neg_set else [neg]      # 음의 다리 (집합 옵션)
        if neg not in negs:
            negs.append(neg)
        pos = [int(j) for j in idx if j not in negs and zi[int(j)] > a.kappa]
        H = 1.0 - abs(col[idx].sum()) / max(np.abs(col[idx]).sum(), 1e-9)
        balanced = bool(H >= a.h0 and zi[neg] <= -a.tau)          # 합은 0, 음의 다리는 뚜렷
        # 라벨: 공모 = 양의 다리 여럿 / 정상 큰손 = 양의 다리 하나 / 외톨이 음 = 양의 다리 없음 (전역 흐름을 혼자 되미는 것; 보고만 하고 손대지 않음)
        # 공모의 추가 조건: 양의 다리 중 한 명이 양의 합의 max_share 를 넘으면 "한 명이 큰손 + 들러리" 로 보아 공모가 아니다 (owner 조건에서 소유자 + 정상 한 명이 같이 걸리는 것을 막는다)
        pos_sum = float(sum(max(col[j], 0.0) for j in pos)) if pos else 0.0
        top_share = (max(col[j] for j in pos) / pos_sum) if pos_sum > 1e-9 else 1.0
        spread = len(pos) >= a.min_pos and top_share <= a.max_share
        kind = ("coalition" if spread else ("big_legit" if len(pos) >= 1 else "lone_neg")) if balanced else ""
        out[c] = {"H": round(float(H), 4), "neg": neg, "negs": negs, "z_neg": round(zi[neg], 2), "pos": pos, "top_share": round(float(top_share), 3),
                  "hedge": balanced and len(pos) >= 1, "coalition": kind == "coalition", "kind": kind}
    return out
