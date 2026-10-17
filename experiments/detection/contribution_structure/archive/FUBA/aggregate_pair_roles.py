# -*- coding: utf-8 -*-
"""pair_roles.py 결과를 조건별로 모은다.

  python aggregate_pair_roles.py --cond owner
읽는 것: logs/pair_roles/bd_{cond}_s*.json (요청자별), *_joint.json (동시 요청)
출력: logs/pair_roles/summary_{cond}.md
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


def ms(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def load(pattern):
    rx = re.compile("^" + re.escape(os.path.basename(pattern)).replace(r"\*", r"(\d+)") + "$")
    out = []
    for f in sorted(glob.glob(pattern)):
        if rx.match(os.path.basename(f)):
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def table(runs, title, L):
    if not runs:
        return
    keys = list(runs[0]["results"].keys())
    L += [f"## {title}  n={len(runs)}", "", "| 요청 S | 역할 | z0 | 표시 | A 크기 | 정밀 | 재현 | 공동 위장자 C (시드별) | ASR 요청만 → +운반자 | ACC +운반자 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        e = [r["results"][k] for r in runs if k in r["results"]]
        if not e:
            continue
        Cs = "; ".join(str(x["co_suppressors"]) for x in e)
        L.append(f"| {k} | {','.join(e[0]['roles_S'])} | {ms([x['z0'] for x in e], 1)} | {sum(x['flagged'] for x in e)}/{len(e)} | {ms([len(x['carriers']) for x in e], 1)} | "
                 f"{ms([x['carrier_precision'] for x in e])} | {ms([x['carrier_recall'] for x in e])} | {Cs} | "
                 f"{ms([x['eval']['request_only'].get('asr') for x in e], 1)} → {ms([x['eval']['request+carriers'].get('asr') for x in e], 1)} | {ms([x['eval']['request+carriers']['acc'] for x in e], 1)} |")
    # 지도 AUROC
    L.append("")
    for k in keys:
        e = [r["results"][k] for r in runs if k in r["results"]]
        L.append(f"- S={k}: 지도 Δ 의 운반자 AUROC {ms([x['auroc_carrier'] for x in e], 3)}, 공동 위장자 AUROC(z_single) {ms([x['auroc_co_suppressor'] for x in e], 3)}")
    L.append("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", required=True)
    a = p.parse_args()
    single = load(f"logs/pair_roles/bd_{a.cond}_s*.json")
    joint = load(f"logs/pair_roles/bd_{a.cond}_s*_joint.json")
    if not single and not joint:
        sys.exit("결과 없음")
    L = [f"# 쌍→집합 요약: {a.cond} (시드 평균 ± 표준편차)", ""]
    table(single, "요청자별 (위장자 4, 정상 K-1)", L)
    table(joint, "동시 요청 (4+5)", L)
    out = f"logs/pair_roles/summary_{a.cond}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
