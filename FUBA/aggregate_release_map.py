# -*- coding: utf-8 -*-
"""release_map.py 결과를 조건별로 모아 요약한다.

  python aggregate_release_map.py --cond owner        # logs/release_map/bd_owner_s*.json
  python aggregate_release_map.py                     # 전부 (name 에서 _s<seed> 를 뗀 것으로 묶음)
출력: logs/release_map/summary[_{cond}].md / .json
  - 역할별(requester / attacker / benign) snap, 최종 mass(=changed×conc) 평균 ± 표준편차
  - 요청자 순위(전원 빼 보기)와 z 점수의 평균, AUROC(요청자 vs 나머지; 실행을 모아서, sklearn 없이)
  - ρ 별 평균 곡선 (요청자 / 정상)
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


def auroc(pos, neg):
    pos, neg = [p for p in pos if p is not None], [n for n in neg if n is not None]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.3f} ± {np.std(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"


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
        groups.setdefault(re.sub(r"_s\d+$", "", r["name"]), []).append(r)

    lines = ["# 빼 보기(release map) 요약 — ① 천천히 빼기 snap, ② 전원 빼 보기 순위 (시드 평균 ± 표준편차)", ""]
    summary = {}
    for cond, runs in groups.items():
        keys = ["snap", "final_mass"] + (["delta_snap", "delta_final_mass"] if "delta_snap" in runs[0]["clients"][next(iter(runs[0]["clients"]))] else [])
        roles = {"requester": [], "attacker": [], "benign": []}
        per_role = {k: {ro: [] for ro in roles} for k in keys}
        for r in runs:
            for j, e in r["clients"].items():
                ro = e["role"]
                per_role["snap"][ro].append(e["snap"])
                per_role["final_mass"][ro].append(e["final"]["mass"])
                if "delta_snap" in e:
                    per_role["delta_snap"][ro].append(e["delta_snap"])
                    per_role["delta_final_mass"][ro].append(e["delta_final"]["mass"])
        rhos = runs[0]["rhos"]
        lines.append(f"## {cond}  (n={len(runs)}, 삭제 전 ASR {ms([r['before'].get('asr') for r in runs])})")
        lines.append("")
        lines.append("| 지표 | requester | attacker | benign | AUROC req vs 나머지 | AUROC req vs benign |")
        lines.append("|---|---|---|---|---|---|")
        summary[cond] = {}
        for k in keys:
            d = per_role[k]
            rest = d["attacker"] + d["benign"]
            lines.append(f"| {k} | {ms(d['requester'])} | {ms(d['attacker'])} | {ms(d['benign'])} | {auroc(d['requester'], rest)} | {auroc(d['requester'], d['benign'])} |")
            summary[cond][k] = {ro: d[ro] for ro in roles}
            summary[cond][k]["auroc_vs_rest"] = auroc(d["requester"], rest)
            summary[cond][k]["auroc_vs_benign"] = auroc(d["requester"], d["benign"])
        lines.append("")
        ranks = {k: [r["requester_rank"][k]["rank"] for r in runs if k in r["requester_rank"]] for k in keys}
        zs = {k: [r["requester_rank"][k]["z"] for r in runs if k in r["requester_rank"]] for k in keys}
        lines.append("- 요청자 순위(1 = 가장 의심) / z: " + ", ".join(f"{k} {ms(ranks[k])} / z {ms(zs[k])}" for k in keys))
        summary[cond]["requester_rank"] = ranks
        summary[cond]["requester_z"] = zs
        # ρ 곡선
        def curve(role, which="curve"):
            arr = []
            for r in runs:
                for e in r["clients"].values():
                    if e["role"] == role and which in e:
                        arr.append([c["mass"] for c in e[which]])
            return np.mean(arr, 0).round(3).tolist() if arr else None
        lines.append(f"- ρ = {rhos}")
        lines.append(f"  - mass 곡선 requester: {curve('requester')}")
        lines.append(f"  - mass 곡선 benign   : {curve('benign')}")
        if "delta_snap" in keys:
            lines.append(f"  - δ 곡선  requester: {curve('requester', 'delta_curve')}")
            lines.append(f"  - δ 곡선  benign   : {curve('benign', 'delta_curve')}")
        asr = [[x for x in e["asr_curve"]] for r in runs for e in r["clients"].values() if e["role"] == "requester" and e.get("asr_curve")]
        if asr:
            lines.append(f"  - (채점) requester ASR 곡선: {np.mean(asr, 0).round(1).tolist()}")
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
