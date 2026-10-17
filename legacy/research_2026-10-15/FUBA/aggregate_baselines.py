# -*- coding: utf-8 -*-
"""기준선(MASA, SubID) 결과를 조건별로 모은다.

  python aggregate_baselines.py --cond owner
출력: logs/baselines/summary_{cond}.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def ms(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def load(d, cond):
    pat = f"{d}/bd_{cond}_s*.json" if cond else f"{d}/*.json"
    out = []
    for f in sorted(glob.glob(pat)):
        if os.path.basename(f).startswith("summary"):
            continue
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", default=None)
    a = p.parse_args()
    masa, subid = load("logs/baseline_masa", a.cond), load("logs/baseline_subid", a.cond)
    if not masa and not subid:
        sys.exit("결과 없음")
    lines = [f"# 기준선 요약 ({a.cond or 'all'})", ""]
    if masa:
        lines += [f"## MASA (학습 중, 라운드별 언러닝 손실 중앙값 편차)  n={len(masa)}", "",
                  "| 지표 | 값 |", "|---|---|",
                  f"| AUROC 공격자 vs 정상 (평균 MDS) | {ms([r['summary']['auroc_attacker_vs_benign'] for r in masa])} |",
                  f"| AUROC 억제자(요청자) vs 정상 | {ms([r['summary']['auroc_defender_vs_benign'] for r in masa])} |",
                  f"| 표시율 공격자 / 억제자 / 정상 | {ms([r['summary']['flag_rate']['attacker'] for r in masa])} / {ms([r['summary']['flag_rate']['defender'] for r in masa])} / {ms([r['summary']['flag_rate']['benign'] for r in masa])} |",
                  f"| 평균 MDS 공격자 / 억제자 / 정상 | {ms([r['summary']['mean_mds_by_role']['attacker'] for r in masa], 2)} / {ms([r['summary']['mean_mds_by_role']['defender'] for r in masa], 2)} / {ms([r['summary']['mean_mds_by_role']['benign'] for r in masa], 2)} |",
                  f"| 초 | {ms([r['seconds'] for r in masa], 0)} |", ""]
    if subid:
        lines += [f"## SubID 식 부분공간 편차  n={len(subid)}", "",
                  "| 지표 | 값 |", "|---|---|",
                  f"| 제거 벡터: 요청자 순위 (1=가장 이상) / 표시된 시드 | {ms([r['summary']['removal']['requester_rank'] for r in subid], 1)} / {sum(r['summary']['removal']['requester_flagged'] for r in subid)}/{len(subid)} |",
                  f"| 제거 벡터: AUROC 공격자 vs 정상 | {ms([r['summary']['removal']['auroc_attacker_vs_benign'] for r in subid])} |",
                  f"| 학습 중: AUROC 공격자 vs 정상 | {ms([r['summary']['training']['auroc_attacker_vs_benign'] for r in subid])} |",
                  f"| 학습 중: 억제자 순위 | {ms([r['summary']['training']['defender_rank'] for r in subid], 1)} |",
                  f"| 학습 중: 표시된 클라 수 | {ms([len(r['summary']['training']['flagged']) for r in subid], 1)} |", ""]
    os.makedirs("logs/baselines", exist_ok=True)
    out = f"logs/baselines/summary_{a.cond or 'all'}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
