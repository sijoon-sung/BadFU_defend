# -*- coding: utf-8 -*-
"""progressive_fu.py 결과를 조건별(bd_{cond}_s*)로 모아 요약한다.

  python aggregate_progressive_fu.py --cond owner
출력: logs/progressive_fu/summary[_{cond}].md / .json
  - arm × 요청자: ASR, ACC, 유지 변화(정의 2), 경보 수, 정화 수, MIA (시드 평균 ± 표준편차)
  - 탐지: 공격 설계 요청자의 첫 경보 ρ, 정상 요청자의 오탐(경보 수)
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


def ms(xs, nd=1):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", default=None)
    p.add_argument("--dir", default="logs/progressive_fu")
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

    lines = ["# 진행형 FU 요약 (시드 평균 ± 표준편차)", ""]
    summary = {}
    for cond, runs in groups.items():
        reqs = list(runs[0]["results"].keys())
        arms = list(next(iter(runs[0]["results"].values())).keys())
        roles = {k: runs[0]["results"][k][arms[0]]["role"] for k in reqs}
        lines.append(f"## {cond}  (n={len(runs)}, 삭제 전 ASR {ms([r['before'].get('asr') for r in runs])} / ACC {ms([r['before'].get('acc') for r in runs])})")
        lines.append("")
        summary[cond] = {}
        for metric, nd in (("asr", 1), ("acc", 1), ("retain_change", 2), ("n_flagged", 1), ("n_purified", 1)):
            lines.append(f"### {metric}")
            lines.append("| arm | " + " | ".join(f"req {k} ({roles[k]})" for k in reqs) + " |")
            lines.append("|---|" + "---|" * len(reqs))
            for arm in arms:
                cells = []
                for k in reqs:
                    vals = [r["results"].get(k, {}).get(arm, {}).get(metric) for r in runs]
                    summary[cond].setdefault(metric, {}).setdefault(arm, {})[k] = vals
                    cells.append(ms(vals, nd))
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
            lines.append("")
        # MIA
        lines.append("### MIA (요청자 AUROC / 멤버율@FPR5%)")
        lines.append("| arm | " + " | ".join(f"req {k}" for k in reqs) + " |")
        lines.append("|---|" + "---|" * len(reqs))
        for arm in arms:
            cells = []
            for k in reqs:
                au = [r["results"].get(k, {}).get(arm, {}).get("mia_req", {}).get("auroc") for r in runs]
                mr = [r["results"].get(k, {}).get(arm, {}).get("mia_req", {}).get("member_rate") for r in runs]
                cells.append(f"{ms(au, 3)} / {ms(mr, 3)}")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
        # 탐지: 첫 경보 ρ (progressive arm 기준; 없으면 plain)
        arm_d = "progressive" if "progressive" in arms else arms[0]
        lines.append("### 탐지 (progressive arm): 첫 경보 ρ / 경보 수 / 최대 z")
        for k in reqs:
            first, nfl, zmax = [], [], []
            for r in runs:
                tr = r["results"].get(k, {}).get(arm_d, {}).get("trace", [])
                fl = [s["rho"] for s in tr if s["flagged"]]
                first.append(fl[0] if fl else None)
                nfl.append(len(fl))
                zmax.append(max((s["z"] for s in tr), default=None))
            lines.append(f"- req {k} ({roles[k]}): 첫 경보 ρ {ms(first, 2)} (미경보 {sum(1 for f in first if f is None)}/{len(first)}), 경보 수 {ms(nfl, 1)}, 최대 z {ms(zmax, 1)}")
            summary[cond].setdefault("detect", {})[k] = {"first_alarm_rho": first, "n_flagged": nfl, "z_max": zmax}
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
