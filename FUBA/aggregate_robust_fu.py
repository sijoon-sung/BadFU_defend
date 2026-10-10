# -*- coding: utf-8 -*-
"""robust_fu.py 결과를 조건별(name 에서 _s<seed> 를 뗀 것)로 모아 평균±표준편차 표와 탐지 AUROC 를 만든다.

  python aggregate_robust_fu.py                 # logs/robust_fu/*.json 전부
  python aggregate_robust_fu.py --cond owner    # bd_owner_s* 만
출력: logs/robust_fu/summary[_{cond}].md, 같은 이름 .json
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


def ms(xs, nd=1):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def auroc(pos, neg):
    """AUROC = P(score_pos > score_neg) + 0.5 P(동점). sklearn 없이."""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", default=None, help="예: owner, iid, dir01 (name 접두사 bd_{cond}_)")
    p.add_argument("--dir", default="logs/robust_fu")
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

    lines = ["# FedEraser-R 요약 (시드 평균 ± 표준편차)", ""]
    summary = {}
    for cond, runs in groups.items():
        roles = runs[0]["roles"]
        reqs = [int(k) for k in runs[0]["args"]["requesters"]]
        arms = list(runs[0]["args"]["arms"])
        probe = runs[0].get("probe", {})
        lines.append(f"## {cond}  (n={len(runs)}, 역할: attackers {roles['attackers']}, defender {roles['defender']}, owner {roles.get('owner')}; "
                     f"탐침 {probe.get('source')}, 평가 {probe.get('eval')})")
        lines.append("")
        before_asr = [r["before"]["asr"] for r in runs]
        before_acc = [r["before"]["acc"] for r in runs]
        lines.append(f"- 삭제 전(θ_T): ASR {ms(before_asr)} / ACC {ms(before_acc)}")
        lines.append("")
        summary[cond] = {"before": {"asr": before_asr, "acc": before_acc}, "arms": {}, "detection": {}}

        def cell(arm, k, key, sub=None, nd=1):
            vals = []
            for r in runs:
                e = r["results"].get(str(k), {}).get(arm)
                if e is None:
                    vals.append(None)
                else:
                    vals.append(e[key][sub] if sub else e[key])
            return vals

        def table(title, key, sub=None, nd=1):
            lines.append(f"### {title}")
            lines.append("")
            lines.append("| arm | " + " | ".join(f"req {k} ({role_of(k, roles)})" for k in reqs) + " |")
            lines.append("|---|" + "---|" * len(reqs))
            for arm in arms:
                lines.append(f"| {arm} | " + " | ".join(ms(cell(arm, k, key, sub), nd) for k in reqs) + " |")
            lines.append("")

        for arm in arms:
            summary[cond]["arms"][arm] = {}
            for k in reqs:
                summary[cond]["arms"][arm][str(k)] = {
                    "asr": cell(arm, k, "asr"), "acc": cell(arm, k, "acc"),
                    "retain_change": cell(arm, k, "retain_change"),
                    "forget_loss": cell(arm, k, "forget_proxy", "final"),
                    "n_flagged": cell(arm, k, "n_flagged"), "max_score": cell(arm, k, "max_score")}
                # forget_proxy.final 은 dict: loss / acc 로 나눈다
                fl = summary[cond]["arms"][arm][str(k)]["forget_loss"]
                summary[cond]["arms"][arm][str(k)]["forget_loss"] = [x["loss"] if x else None for x in fl]
                summary[cond]["arms"][arm][str(k)]["forget_acc"] = [x["acc"] if x else None for x in fl]

        table("ASR (%)", "asr")
        table("ACC (%)", "acc")
        table("유지 행동 변화 (정의 2: TEST[N:] 에서 θ_T 와 예측이 다른 비율 %)", "retain_change", nd=2)
        lines.append("### 망각 대리 지표 (요청자 자기 분할, CE 손실 / 정확도 %; MIA 아님)")
        lines.append("")
        lines.append("| arm | " + " | ".join(f"req {k} ({role_of(k, roles)})" for k in reqs) + " |")
        lines.append("|---|" + "---|" * len(reqs))
        for arm in arms:
            cells = []
            for k in reqs:
                s = summary[cond]["arms"][arm][str(k)]
                cells.append(f"{ms(s['forget_loss'], 3)} / {ms(s['forget_acc'])}")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        fp_b = {}
        for k in reqs:
            fb = [r["results"].get(str(k), {}).get(arms[0], {}).get("forget_proxy", {}).get("before") for r in runs]
            fp_b[str(k)] = {"loss": [x["loss"] if x else None for x in fb], "acc": [x["acc"] if x else None for x in fb]}
        lines.append("| (θ_T) | " + " | ".join(f"{ms(fp_b[str(k)]['loss'], 3)} / {ms(fp_b[str(k)]['acc'])}" for k in reqs) + " |")
        lines.append("")

        # MIA (주 망각 지표): 요청자 AUROC / 멤버율, 대조군(정상 클라) AUROC / 멤버율
        def mia_vals(arm, k, who, key):
            vals = []
            for r in runs:
                m = r["before"].get("mia", {}).get(str(k)) if arm == "before" else (r["results"].get(str(k), {}).get(arm) or {}).get("mia")
                vals.append(m[who][key] if m and who in m else None)
            return vals

        summary[cond]["mia"] = {}
        lines.append("### 망각 MIA (손실 기반; 요청자 AUROC / 멤버율@비멤버 FPR 5%, 괄호 = 대조군 정상 클라 AUROC / 멤버율)")
        lines.append("")
        lines.append("| arm | " + " | ".join(f"req {k} ({role_of(k, roles)})" for k in reqs) + " |")
        lines.append("|---|" + "---|" * len(reqs))
        for arm in ["before"] + arms:
            cells = []
            summary[cond]["mia"][arm] = {}
            for k in reqs:
                s = {"req_auroc": mia_vals(arm, k, "req", "auroc"), "req_member_rate": mia_vals(arm, k, "req", "member_rate"),
                     "control_auroc": mia_vals(arm, k, "control", "auroc"), "control_member_rate": mia_vals(arm, k, "control", "member_rate")}
                summary[cond]["mia"][arm][str(k)] = s
                cells.append(f"{ms(s['req_auroc'], 3)} / {ms(s['req_member_rate'], 3)} ({ms(s['control_auroc'], 3)} / {ms(s['control_member_rate'], 3)})")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
        table("flag 라운드 수", "n_flagged")

        # 탐지: 라운드 최대 Score, defender 요청 vs 정상(owner 포함) 요청, 실행 전체에 걸친 AUROC
        lines.append("### 탐지 (라운드 최대 Score = δ 아래 집중도; defender 요청 vs 정상 요청)")
        lines.append("")
        lines.append("| arm | defender 평균 | 정상 평균 | AUROC (defender vs 정상) |")
        lines.append("|---|---|---|---|")
        for arm in arms:
            pos, neg = [], []
            for r in runs:
                for k in reqs:
                    e = r["results"].get(str(k), {}).get(arm)
                    if e is None:
                        continue
                    (pos if role_of(k, roles) == "defender" else neg).append(e["max_score"])
            au = auroc(pos, neg)
            summary[cond]["detection"][arm] = {"defender_max_score": pos, "benign_max_score": neg, "auroc": au}
            lines.append(f"| {arm} | {ms(pos, 3)} | {ms(neg, 3)} | {('%.3f' % au) if au is not None else '-'} |")
        lines.append("")

        # 판정 보조: 삭제 전 대비 ASR / ACC 변화
        b_asr, b_acc = np.mean(before_asr), np.mean(before_acc)
        lines.append("- 삭제 전 대비 변화 (ASR %p / ACC %p):")
        for arm in arms:
            parts = []
            for k in reqs:
                s = summary[cond]["arms"][arm][str(k)]
                asr = [x for x in s["asr"] if x is not None]
                acc = [x for x in s["acc"] if x is not None]
                if asr:
                    parts.append(f"req{k}({role_of(k, roles)}) {np.mean(asr) - b_asr:+.1f} / {np.mean(acc) - b_acc:+.1f}")
            lines.append(f"  - {arm}: " + ", ".join(parts))
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
