# -*- coding: utf-8 -*-
"""effects.py 결과를 조건별로 모은다.  python aggregate_effects.py --cond iid  → logs/effects/summary_{cond}.md
요청자별(공개 / 공개+잊을 집합): 크기·몰림·상쇄의 conformal p, Fisher p, 상쇄 집합(역할), 라운드 일관성."""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def ms(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", default=None)
    p.add_argument("--dir", default="logs/effects")
    a = p.parse_args()
    pat = f"{a.dir}/bd_{a.cond}_s*.json" if a.cond else f"{a.dir}/*.json"
    rx = re.compile("^" + re.escape(os.path.basename(pat)).replace(r"\*", r"(\d+)") + "$") if a.cond else None
    runs = []
    for f in sorted(glob.glob(pat)):
        if os.path.basename(f).startswith("summary") or (rx and not rx.match(os.path.basename(f))):
            continue
        with open(f, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    if not runs:
        sys.exit(f"결과 없음: {pat}")
    roles = runs[0]["roles"]
    reqs = list(runs[0]["public"]["pvals"].keys())
    L = [f"# 기능 효과 분석 (effects) 요약: {a.cond or 'all'}  n={len(runs)}", "",
         "## 요청자별 p 값 (conformal; 작을수록 이상) — 공개 입력 / 공개+잊을 집합", "",
         "| 요청자 | 역할 | p 크기 | p 몰림 | p 상쇄 | Fisher p | 상쇄 집합(역할, 시드별) | 라운드 일관성 |", "|---|---|---|---|---|---|---|---|"]
    for k in reqs:
        for where in ("public", "forget"):
            rows = []
            for r in runs:
                blk = r["public"] if where == "public" else r.get("forget", {}).get(str(k))
                if blk is None or str(k) not in blk["pvals"]:
                    continue
                rows.append(blk)
            if not rows:
                continue
            sets = "; ".join(str(b["cancel"][str(k)]["set_roles"]) for b in rows)
            cons = [r.get("round_consistency", {}).get(str(k)) for r in runs]
            L.append(f"| {k} ({where}) | {roles[str(k)]} | {ms([b['pvals'][str(k)]['mag'] for b in rows])} | {ms([b['pvals'][str(k)]['conc'] for b in rows])} | "
                     f"{ms([b['pvals'][str(k)]['cancel'] for b in rows])} | {ms([b['combined'][str(k)]['fisher_p'] for b in rows])} | {sets} | {ms(cons)} |")
    L.append("")
    L.append("## 클라이언트별 평균 통계 (공개 입력; 시드 평균)")
    L.append("| 클라 | 역할 | 원시 크기 | 중심화 크기 | 몰림 점수 | 상쇄 비율 | 라운드 일관성 |")
    L.append("|---|---|---|---|---|---|---|")
    for j in sorted(roles, key=int):
        L.append(f"| {j} | {roles[j]} | {ms([r['public']['raw_mag'][j] for r in runs], 2)} | {ms([r['public']['mag'][j] for r in runs], 2)} | "
                 f"{ms([r['public']['conc'][j]['score'] for r in runs])} | {ms([r['public']['cancel'][j]['ratio'] for r in runs])} | {ms([r.get('round_consistency', {}).get(j) for r in runs])} |")
    tag = f"_{a.cond}" if a.cond else ""
    with open(f"{a.dir}/summary{tag}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"-> {a.dir}/summary{tag}.md")


if __name__ == "__main__":
    main()
