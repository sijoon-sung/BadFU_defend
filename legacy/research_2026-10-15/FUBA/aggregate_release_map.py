# -*- coding: utf-8 -*-
"""release_map.py(v2) 결과를 조건별로 모아 요약한다.

  python aggregate_release_map.py --cond owner        # logs/release_map/bd_owner_s*.json
  python aggregate_release_map.py                     # 전부 (name 에서 _s<seed> 를 뗀 것으로 묶음)
출력: logs/release_map/summary[_{cond}].md / .json
  - 작동점 ρ* 통계(clean / linf / patch 의 mass)를 역할별(requester / attacker / benign) 평균 ± 표준편차
  - 요청자 순위(전원 빼 보기)와 z 의 평균, AUROC(요청자 vs 나머지 / vs benign; 실행을 모아서, sklearn 없이)
  - ρ 별 평균 곡선 (깨끗한 probe 의 mass, 채점 ASR·ACC)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
KEYS = ("clean", "linf", "patch")
ROLES = ("requester", "attacker", "benign")


def auroc(pos, neg):
    pos, neg = [p for p in pos if p is not None], [n for n in neg if n is not None]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def ms(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", default=None)
    p.add_argument("--dir", default="logs/release_map")
    a = p.parse_args()
    files = sorted(glob.glob(f"{a.dir}/bd_{a.cond}_s*.json")) if a.cond else sorted(glob.glob(f"{a.dir}/*.json"))
    files = [f for f in files if not os.path.basename(f).startswith("summary")]
    if not files:
        sys.exit(f"결과 없음: {a.dir}")
    groups = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if "wp" not in next(iter(r["clients"].values())):
            print(f"건너뜀(구형식): {f}")
            continue
        groups.setdefault(re.sub(r"_s\d+$", "", r["name"]), []).append(r)
    if not groups:
        sys.exit("v2 형식 결과 없음")

    lines = ["# 빼 보기(release map) 요약 — 작동점 ρ* 에서의 쏠림 mass, 전원 빼 보기 순위 (시드 평균 ± 표준편차)", ""]
    summary = {}
    for cond, runs in groups.items():
        keys = [k for k in KEYS if k in next(iter(runs[0]["clients"].values()))["wp"]]
        per = {k: {ro: [] for ro in ROLES} for k in keys}
        rho_star = {ro: [] for ro in ROLES}
        for r in runs:
            for e in r["clients"].values():
                ro = e["role"]
                rho_star[ro].append(e["wp"]["rho"])
                for k in keys:
                    per[k][ro].append(e["wp"][k]["mass"])
        rhos = runs[0]["rhos"]
        lines.append(f"## {cond}  (n={len(runs)}, 삭제 전 ASR {ms([r['before'].get('asr') for r in runs], 1)} / ACC {ms([r['before'].get('acc') for r in runs], 1)})")
        lines.append("")
        lines.append(f"- 작동점 ρ* 평균: " + ", ".join(f"{ro} {ms(rho_star[ro], 2)}" for ro in ROLES))
        lines.append("")
        lines.append("| 입력 | requester | attacker | benign | AUROC req vs 나머지 | AUROC req vs benign |")
        lines.append("|---|---|---|---|---|---|")
        summary[cond] = {}
        for k in keys:
            d = per[k]
            rest = d["attacker"] + d["benign"]
            lines.append(f"| {k} | {ms(d['requester'])} | {ms(d['attacker'])} | {ms(d['benign'])} | {auroc(d['requester'], rest)} | {auroc(d['requester'], d['benign'])} |")
            summary[cond][k] = {ro: d[ro] for ro in ROLES}
            summary[cond][k]["auroc_vs_rest"] = auroc(d["requester"], rest)
            summary[cond][k]["auroc_vs_benign"] = auroc(d["requester"], d["benign"])
        lines.append("")
        ranks = {k: [r["requester_rank"][k]["rank"] for r in runs if k in r["requester_rank"]] for k in keys}
        zs = {k: [r["requester_rank"][k]["z"] for r in runs if k in r["requester_rank"]] for k in keys}
        lines.append("- 요청자 순위(1 = 가장 의심) / z: " + ", ".join(f"{k} {ms(ranks[k], 1)} / z {ms(zs[k], 1)}" for k in keys))
        summary[cond]["requester_rank"] = ranks
        summary[cond]["requester_z"] = zs

        def curve(role, field):
            arr = []
            for r in runs:
                for e in r["clients"].values():
                    if e["role"] != role:
                        continue
                    if field == "mass":
                        arr.append([c["mass"] for c in e["curve"]])
                    elif e.get(field):
                        arr.append(e[field])
            return np.mean(arr, 0).round(3).tolist() if arr else None
        lines.append(f"- ρ = {rhos}")
        for ro in ROLES:
            lines.append(f"  - {ro:9s} mass {curve(ro, 'mass')}  ASR {curve(ro, 'asr_curve')}  ACC {curve(ro, 'acc_curve')}")
        lines.append("")
    tag = f"_{a.cond}" if a.cond else ""
    with open(f"{a.dir}/summary{tag}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(f"{a.dir}/summary{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"-> {a.dir}/summary{tag}.md")


if __name__ == "__main__":
    main()
