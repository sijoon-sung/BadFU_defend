# -*- coding: utf-8 -*-
"""benign_departure.py 결과를 조건별(name 에서 _s<seed> 를 뗀 것)로 모아 평균±표준편차 표를 만든다.

  python aggregate_benign_departure.py                 # logs/benign_departure/*.json 전부
  python aggregate_benign_departure.py --cond owner    # bd_owner_s* 만
출력: logs/benign_departure/summary[_{cond}].md, 같은 이름 .json
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


def role_of(k, roles):
    if k in roles["attackers"]:
        return "attacker"
    if k == roles["defender"]:
        return "defender"
    if roles.get("owner") is not None and k == roles["owner"]:
        return "owner"
    return "benign"


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.1f} ± {np.std(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", default=None, help="예: owner, iid, dir01 (name 접두사 bd_{cond}_)")
    p.add_argument("--dir", default="logs/benign_departure")
    a = p.parse_args()

    files = sorted(glob.glob(f"{a.dir}/bd_{a.cond}_s*.json")) if a.cond else sorted(glob.glob(f"{a.dir}/*.json"))
    files = [f for f in files if not os.path.basename(f).startswith("summary")]
    if not files:
        sys.exit(f"결과 없음: {a.dir}")

    groups = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        cond = re.sub(r"_s\d+$", "", r["name"])
        groups.setdefault(cond, []).append(r)

    lines = ["# 정상 이탈 검증 요약 (ASR %, 괄호는 ACC %; 시드 평균 ± 표준편차)", ""]
    summary = {}
    for cond, runs in groups.items():
        roles = runs[0]["roles"]
        reqs = [int(k) for k in runs[0]["args"]["requesters"]]
        methods = [m for m in runs[0]["results"]]
        lines.append(f"## {cond}  (n={len(runs)}, 역할: attackers {roles['attackers']}, defender {roles['defender']}, owner {roles.get('owner')})")
        lines.append("")
        before_asr = [r["before"]["asr"] for r in runs]
        before_acc = [r["before"]["acc"] for r in runs]
        lines.append(f"- 삭제 전: ASR {ms(before_asr)} / ACC {ms(before_acc)}")
        lines.append("")
        hdr = "| FU 방법 | " + " | ".join(f"req {k} ({role_of(k, roles)})" for k in reqs) + " |"
        lines.append(hdr)
        lines.append("|---|" + "---|" * len(reqs))
        summary[cond] = {"before": {"asr": before_asr, "acc": before_acc}, "methods": {}}
        for m in methods:
            cells = []
            summary[cond]["methods"][m] = {}
            for k in reqs:
                asr = [r["results"].get(m, {}).get(str(k), {}).get("asr") for r in runs]
                acc = [r["results"].get(m, {}).get(str(k), {}).get("acc") for r in runs]
                summary[cond]["methods"][m][str(k)] = {"asr": asr, "acc": acc}
                cells.append(f"{ms(asr)} ({ms(acc)})")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        # 판정 보조: 정상 요청자(owner/benign)의 ASR 상승폭 vs defender
        verdict = []
        for m in methods:
            d = {k: np.mean([x for x in summary[cond]["methods"][m][str(k)]["asr"] if x is not None] or [np.nan]) for k in reqs}
            b = np.mean(before_asr)
            verdict.append(f"  - {m}: " + ", ".join(f"req{k}({role_of(k, roles)}) {d[k] - b:+.1f}" for k in reqs))
        lines.append("- 삭제 전 대비 ASR 변화(%p):")
        lines.extend(verdict)
        lines.append("")

    tag = f"_{a.cond}" if a.cond else ""
    md = f"{a.dir}/summary{tag}.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(f"{a.dir}/summary{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"-> {md}")


if __name__ == "__main__":
    main()
