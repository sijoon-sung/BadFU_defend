# -*- coding: utf-8 -*-
"""dose_fu.py 결과를 조건별(bd_{cond}_s*[tag].json)로 모아 요약한다.

  python aggregate_dose_fu.py --cond owner            # logs/dose_fu/bd_owner_s*.json
  python aggregate_dose_fu.py --cond owner --tag _split5_ledger
출력: logs/dose_fu/summary_{cond}{tag}.md / .json
  - arm × 요청자: ASR, ACC, 유지 변화, 단계 수, 경보, 수정, 가상 모델 수, 초 (시드 평균 ± 표준편차)
  - 망각: forget acc(before→final), MIA AUROC
  - 탐지: 첫 경보 단계의 누적 제거량, 경보 없는 시드 수
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
    p.add_argument("--tag", default="")
    p.add_argument("--dir", default="logs/dose_fu")
    a = p.parse_args()
    pat = f"{a.dir}/bd_{a.cond}_s*{a.tag}.json" if a.cond else f"{a.dir}/*{a.tag}.json"
    files = [f for f in sorted(glob.glob(pat)) if not os.path.basename(f).startswith("summary")]
    if a.tag == "":
        files = [f for f in files if "_split" not in os.path.basename(f)]
    if not files:
        sys.exit(f"결과 없음: {pat}")
    groups = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        groups.setdefault(re.sub(r"_s\d+$", "", r["name"]) + r.get("tag", ""), []).append(r)

    lines = [f"# 투여량 조절 FU 요약{a.tag} (시드 평균 ± 표준편차)", ""]
    summary = {}
    for cond, runs in groups.items():
        reqs = list(runs[0]["results"].keys())
        arms = list(next(iter(runs[0]["results"].values())).keys())
        roles = {k: runs[0]["results"][k][arms[0]]["role"] for k in reqs}
        lines += [f"## {cond}  (n={len(runs)}, 삭제 전 ASR {ms([r['before'].get('asr') for r in runs])} / ACC {ms([r['before'].get('acc') for r in runs])})", ""]
        summary[cond] = {}
        for metric, nd in (("asr", 1), ("acc", 1), ("retain_change", 2), ("reinjection_alpha", 3), ("removed", 2), ("n_steps", 1), ("n_flagged", 1), ("n_repaired", 1), ("n_hard_collapse", 1), ("pop_models", 0), ("seconds", 0)):
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
        lines.append("### 망각: 요청자 자기 데이터 정확도 before → final / MIA AUROC")
        lines.append("| arm | " + " | ".join(f"req {k}" for k in reqs) + " |")
        lines.append("|---|" + "---|" * len(reqs))
        for arm in arms:
            cells = []
            for k in reqs:
                fb = [r["results"].get(k, {}).get(arm, {}).get("forget", {}).get("before", {}).get("acc") for r in runs]
                ff = [r["results"].get(k, {}).get(arm, {}).get("forget", {}).get("final", {}).get("acc") for r in runs]
                au = [r["results"].get(k, {}).get(arm, {}).get("mia_req", {}).get("auroc") for r in runs]
                cells.append(f"{ms(fb)} → {ms(ff)} / {ms(au, 3)}")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("### 탐지 (dose arm): 첫 경보 때 누적 제거량 / 미경보 시드 / 최대 z / 최대 z_cum")
        arm_d = "dose" if "dose" in arms else arms[-1]
        for k in reqs:
            first, zmax, zcmax = [], [], []
            for r in runs:
                tr = r["results"].get(k, {}).get(arm_d, {}).get("trace", [])
                fl = [s["removed"] for s in tr if s["flagged"]]
                first.append(fl[0] if fl else None)
                zmax.append(max((s["z"] for s in tr if s["z"] is not None), default=None))
                zcmax.append(max((s["z_cum"] for s in tr if s["z_cum"] is not None), default=None))
            lines.append(f"- req {k} ({roles[k]}): 첫 경보 제거량 {ms(first, 2)} (미경보 {sum(1 for f in first if f is None)}/{len(first)}), 최대 z {ms(zmax, 1)}, 최대 z_cum {ms(zcmax, 1)}")
            summary[cond].setdefault("detect", {})[k] = {"first_alarm_removed": first, "z_max": zmax, "z_cum_max": zcmax}
        lines.append("")
    tag = (f"_{a.cond}" if a.cond else "") + a.tag
    with open(f"{a.dir}/summary{tag}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(f"{a.dir}/summary{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"-> {a.dir}/summary{tag}.md")


if __name__ == "__main__":
    main()
