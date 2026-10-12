# -*- coding: utf-8 -*-
"""탐지 총정리 집계: logs/detect/*.json (+ 있으면 logs/baseline_masa, logs/baseline_subid 의 proxy 점수) → logs/detect/summary_{cond}.md
방법마다: 위장자 순위 (1 = 가장 의심), 위장자 vs 정상 AUROC, 위장자 z (정상 대비), 소유자 z (정상 대비, owner 조건), "잡힘" = 순위 1 인 시드 수.
실행: python aggregate_detect_bench.py --cond iid_r100w5_k20paper   (이름이 bd_{cond}_s* 인 것)   또는 --cond all
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def rz(x, others):
    o = np.asarray([v for v in others if v is not None], dtype=float)
    if x is None or o.size < 2:
        return None
    sd = float(np.std(o, ddof=1))
    return float((x - np.median(o)) / sd) if sd > 1e-9 else None


def ms(xs, nd=2):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def proxy_scores(name):
    """MASA proxy (mean_mds) 와 SubID proxy (phi) 의 클라이언트별 점수가 있으면 가져온다."""
    out = {}
    f = f"logs/baseline_masa/{name}.json"
    if os.path.exists(f):
        d = json.load(open(f, encoding="utf-8"))
        if isinstance(d.get("mean_mds"), dict):
            out["proxy_masa_mds"] = ({int(k): v for k, v in d["mean_mds"].items()}, "high")
    f = f"logs/baseline_subid/{name}.json"
    if os.path.exists(f):
        d = json.load(open(f, encoding="utf-8"))
        def find(dct, key):
            if isinstance(dct, dict):
                for k, v in dct.items():
                    if k == key and isinstance(v, dict) and all(str(x).isdigit() for x in v):
                        return v
                    r = find(v, key)
                    if r:
                        return r
            return None
        for key in ("phi_rm", "phi_tr"):
            v = find(d, key)
            if v:
                out[f"proxy_subid_{key}"] = ({int(k): float(x) for k, x in v.items()}, "high")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="all")
    ap.add_argument("--dir", default="logs/detect")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dir}/*.json"))
    runs = []
    for f in files:
        base = os.path.basename(f)[:-5]
        if base.startswith("summary"):
            continue
        if a.cond != "all" and not re.match(rf"^bd_{re.escape(a.cond)}_s\d+$", base):
            continue
        d = json.load(open(f, encoding="utf-8"))
        for mth, (sc, dirn) in proxy_scores(base).items():
            d["scores"][mth] = sc; d["direction"][mth] = dirn
        runs.append(d)
    if not runs:
        raise SystemExit("logs/detect 에 결과 없음")
    methods = []
    for r in runs:
        for m in r["scores"]:
            if r["direction"].get(m) in ("high", "low") and m not in methods:
                methods.append(m)
    L = [f"# 탐지 총정리: {a.cond}  n={len(runs)}", "",
         "모든 클라이언트를 '요청자라면' 으로 두고 같은 점수를 매긴 뒤, 위장자가 몇 등인지(1 = 가장 의심), 정상과 얼마나 갈리는지(AUROC, z), 소유자가 잘못 걸리는지(소유자 z) 를 센다. 잡힘 = 위장자 순위 1 인 시드 수.", "",
         "| 방법 | 방향 | 위장자 순위 | 잡힘 | AUROC 위장자 vs 정상 | 위장자 z | 소유자 z | 운반자 평균 순위 |", "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        ranks, aur, zs, zo, cr, caught = [], [], [], [], [], 0
        for r in runs:
            d = {int(k): v for k, v in r["scores"].get(m, {}).items() if v is not None}
            roles = {int(k): v for k, v in r["roles"].items()}
            sup = [j for j, ro in roles.items() if ro == "suppressor"]
            if not d or not sup or sup[0] not in d:
                continue
            s = sup[0]; sign = 1 if r["direction"][m] == "high" else -1
            vals = {j: sign * v for j, v in d.items()}
            rank = 1 + sum(1 for j, v in vals.items() if j != s and v > vals[s]); ranks.append(rank); caught += (rank == 1)
            ben = [vals[j] for j in vals if roles[j] == "benign"]
            aur.append(np.mean([1.0 if vals[s] > b else (0.5 if vals[s] == b else 0.0) for b in ben]) if ben else None)
            zs.append(rz(vals[s], ben))
            own = [j for j, ro in roles.items() if ro == "owner"]
            zo.append(rz(vals[own[0]], ben) if own and own[0] in vals else None)
            car = [j for j, ro in roles.items() if ro == "carrier" and j in vals]
            cr.append(np.mean([1 + sum(1 for j, v in vals.items() if j != c and v > vals[c]) for c in car]) if car else None)
        if not ranks:
            continue
        L.append(f"| {m} | {runs[0]['direction'][m]} | {ms(ranks, 1)} | {caught}/{len(ranks)} | {ms(aur)} | {ms(zs, 1)} | {ms(zo, 1)} | {ms(cr, 1)} |")
    L += ["", "읽는 법: 위장자 순위 1 이고 소유자 z 가 3 아래면 '위장자만 잡는다'. 운반자 평균 순위가 낮으면(1–4) 그 방법은 운반자(심은 쪽)를 보는 것이고, 높으면 운반자는 정상처럼 보인다.",
          "sim_abs_* = 기존 요청 심사 발상(절대 변화량), sim_rel_* = DOSE(상대 기준), fx_* = 기능 효과, upd_* = 집계 방어가 보는 갱신 통계, ns_energy = null-space 게이트, audit_z@r = 학습 중 부분합 감사, part_beta_* = 참여 자연 실험, proxy_* = MASA/SubID 재구현."]
    out = f"{a.dir}/summary_{a.cond}.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print(f"-> {out}")


if __name__ == "__main__":
    main()
