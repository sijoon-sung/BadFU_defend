# -*- coding: utf-8 -*-
"""latent_risk.py 결과 여러 개를 모아 역할별 특징 평균과 AUROC 평균(± 표준편차)을 낸다.

  python aggregate_latent_risk.py --names det1 det2 det3     # logs/latent_risk/{name}.json
  python aggregate_latent_risk.py                            # logs/latent_risk/*.json 전부 (summary* 제외)
출력: logs/latent_risk/summary.md, summary.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

ROLES = ["attacker", "defender", "benign"]
TABLE_FEATS = ["shape_sigma1_share", "shape_eff_rank", "shape_concentration", "shape_consistency",
               "cos_U_generic", "cos_U_non_target", "cos_round_mean_generic", "cos_U_fc3_generic"]


def ms(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def role_of(k, roles):
    if k in roles["attackers"]:
        return "attacker"
    if k == roles["defender"]:
        return "defender"
    return "benign"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--dir", default="logs/latent_risk")
    p.add_argument("--out", default=None, help="기본 {dir}/summary.md (+ .json)")
    a = p.parse_args()

    if a.names:
        files = [f"{a.dir}/{n}.json" for n in a.names]
    else:
        files = [f for f in sorted(glob.glob(f"{a.dir}/*.json")) if not os.path.basename(f).startswith("summary")]
    runs = []
    for f in files:
        if not os.path.exists(f):
            sys.exit(f"결과 없음: {f}")
        with open(f, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    if not runs:
        sys.exit(f"결과 없음: {a.dir}")

    feat_names = [n for n in runs[0]["separation"] if runs[0]["separation"][n] is not None]
    # 역할별 특징 평균: 실행마다 역할 평균을 내고, 실행 간 평균 ± 표준편차
    per_role = {f: {r: [] for r in ROLES} for f in feat_names}
    auc = {f: {"mal_vs_benign": [], "def_vs_rest": [], "def_rank_desc": []} for f in feat_names}
    for run in runs:
        roles = run["roles"]
        for f in feat_names:
            s = run["separation"].get(f)
            if s is None:
                continue
            by_role = {r: [] for r in ROLES}
            for k, v in s["values"].items():
                by_role[role_of(int(k), roles)].append(v)
            for r in ROLES:
                per_role[f][r].append(float(np.mean(by_role[r])) if by_role[r] else None)
            auc[f]["mal_vs_benign"].append(s["auroc_mal_vs_benign"])
            auc[f]["def_vs_rest"].append(s["auroc_def_vs_rest"])
            auc[f]["def_rank_desc"].append(s["defender_rank_desc"])

    lines = [f"# 잠재 위험 지도 요약 (n={len(runs)}: {', '.join(r['name'] for r in runs)})", ""]
    lines.append("## 역할별 특징 평균 (실행 평균 ± 표준편차)")
    lines.append("")
    lines.append("| 특징 | attacker | defender | benign |")
    lines.append("|---|---|---|---|")
    for f in TABLE_FEATS:
        if f in per_role:
            lines.append(f"| {f} | " + " | ".join(ms(per_role[f][r]) for r in ROLES) + " |")
    lines.append("")
    lines.append("## AUROC (attackers+defender vs benign | defender vs rest) 와 defender 내림차순 순위")
    lines.append("")
    lines.append("| 특징 | AUROC mal vs benign | AUROC def vs rest | def rank (desc) |")
    lines.append("|---|---|---|---|")
    for f in feat_names:
        lines.append(f"| {f} | {ms(auc[f]['mal_vs_benign'])} | {ms(auc[f]['def_vs_rest'])} | {ms(auc[f]['def_rank_desc'], 1)} |")
    lines.append("")
    bv_asr = [m["asr"] for r in runs for m in r["bv"]["models"].values()]
    bv_acc = [m["acc"] for r in runs for m in r["bv"]["models"].values()]
    lines.append(f"- 참조 BV 자체 점검: 서버 트리거 ASR {ms(bv_asr, 1)} / 깨끗한 ACC {ms(bv_acc, 1)} (BV 모델 {len(bv_asr)}개)")
    lines.append("")

    out = a.out or f"{a.dir}/summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"runs": [r["name"] for r in runs], "per_role": per_role, "auroc": auc}, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
