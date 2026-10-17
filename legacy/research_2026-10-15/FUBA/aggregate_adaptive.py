# -*- coding: utf-8 -*-
"""적응형 공격 시나리오 결과 요약 (run_all.sh 'adaptive' 단계).

  python aggregate_adaptive.py --cond owner
읽는 것 (bd_{cond}_s*):
  logs/dose_fu/{n}_split10_ledger.json     a 쪼개기 (10 조각)       → 요청자 4 경보/ASR, 정상 경보
  logs/adaptive/{n}_distsupp.json          b 위장자 분산 (3 조각 순차) → percl vs global 경보·ASR, 정상 열 오탐
  logs/adaptive/{n}_decoy.json             c 모집단 오염 (미끼 3 명)   → 요청자 4 z, 경보 여부
  logs/adaptive/{n}_refpoison.json         g 기준 오염               → 수정 뒤 ASR (오염 전과 비교: logs/dose_fu/{n}.json)
  logs/pair_roles/{n}_distcarr.json        i 운반자 분산 (각 3 조각)   → 운반자 재현율, +운반자 ASR
출력: logs/adaptive/summary_{cond}.md
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


def load(pattern):
    """pattern 의 '*' 는 시드 숫자만 받는다 (bd_x_s0.json 이 bd_x_s0_split10_ledger.json 과 섞이지 않게)."""
    out = {}
    rx = re.compile("^" + re.escape(os.path.basename(pattern)).replace(r"\*", r"(\d+)") + "$")
    for f in sorted(glob.glob(pattern)):
        m = rx.match(os.path.basename(f))
        if not m:
            continue
        with open(f, encoding="utf-8") as fh:
            out[m.group(1)] = json.load(fh)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", required=True)
    a = p.parse_args()
    n = f"bd_{a.cond}_s*"
    L = ["# 적응형 공격 요약: " + a.cond, ""]
    # a. split
    R = load(f"logs/dose_fu/{n}_split10_ledger.json")
    if R:
        L += ["## a. 요청 쪼개기 (10 조각, 누적 장부)  n=%d" % len(R), "", "| 요청자 | arm | 경보 조각 수 | 첫 경보 조각 | ASR | ACC |", "|---|---|---|---|---|---|"]
        reqs = list(next(iter(R.values()))["results"].keys())
        for k in reqs:
            for arm in ("oneshot", "dose"):
                e = [r["results"].get(k, {}).get(arm) for r in R.values()]
                e = [x for x in e if x]
                if not e:
                    continue
                first = [next((s["piece"] for s in x["trace"] if s["flagged"]), None) for x in e]
                L.append(f"| {k} ({e[0]['role']}) | {arm} | {ms([x['n_flagged'] for x in e])} | {ms(first, 1)} (미경보 {sum(f is None for f in first)}/{len(first)}) | {ms([x.get('asr') for x in e])} | {ms([x['acc'] for x in e])} |")
        L.append("")
    # b. distributed suppressor
    R = load(f"logs/adaptive/{n}_distsupp.json")
    if R:
        L += ["## b. 위장자 분산 (위장자 4 를 3 조각으로, 순차 요청)  n=%d" % len(R), "", "| 열 | arm | 경보 수 / 요청 수 | ASR | ACC | z_inc (마지막) | z_glb (마지막) |", "|---|---|---|---|---|---|---|"]
        for label in ("attack", "control"):
            for arm in ("percl", "global"):
                e = [r["results"].get(label, {}).get(arm) for r in R.values()]
                e = [x for x in e if x]
                if not e:
                    continue
                L.append(f"| {label} | {arm} | {ms([x['n_flagged'] for x in e])} / {len(e[0]['seq'])} | {ms([x.get('asr') for x in e])} | {ms([x['acc'] for x in e])} | {ms([x['steps'][-1]['z_inc'] for x in e])} | {ms([x['steps'][-1]['z_glb'] for x in e])} |")
        L.append("")
    # c. decoy
    R = load(f"logs/adaptive/{n}_decoy.json")
    R0 = load(f"logs/dose_fu/{n}.json")
    if R:
        L += ["## c. 모집단 오염 (정상 3 명 기여 ×2.5 미끼)  n=%d" % len(R), "", "| 요청자 | 오염 전 z (oneshot) | 오염 후 z | 오염 후 경보 | ASR 후 |", "|---|---|---|---|---|"]
        for k in list(next(iter(R.values()))["results"].keys()):
            z1 = [r["results"][k]["oneshot"]["trace"][0]["z"] for r in R.values() if k in r["results"]]
            z0 = [r["results"][k]["oneshot"]["trace"][0]["z"] for r in R0.values() if k in r.get("results", {}) and "oneshot" in r["results"][k]]
            fl = [r["results"][k]["oneshot"]["n_flagged"] for r in R.values() if k in r["results"]]
            asr = [r["results"][k]["oneshot"].get("asr") for r in R.values() if k in r["results"]]
            L.append(f"| {k} | {ms(z0)} | {ms(z1)} | {sum(f > 0 for f in fl)}/{len(fl)} | {ms(asr)} |")
        L.append("")
    # g. reference poisoning
    R = load(f"logs/adaptive/{n}_refpoison.json")
    if R:
        L += ["## g. 기준 오염 (정상 3 명 기여에 위장자 기여 0.3 배 섞음)  n=%d" % len(R), "", "| 요청자 | 오염 전: 수정 뒤 ASR / ACC | 오염 후: 수정 뒤 ASR / ACC | 오염 후 경보 |", "|---|---|---|---|"]
        for k in list(next(iter(R.values()))["results"].keys()):
            e1 = [r["results"][k]["oneshot"] for r in R.values() if k in r["results"]]
            e0 = [r["results"][k]["oneshot"] for r in R0.values() if k in r.get("results", {}) and "oneshot" in r["results"][k]]
            L.append(f"| {k} | {ms([x.get('asr') for x in e0])} / {ms([x['acc'] for x in e0])} | {ms([x.get('asr') for x in e1])} / {ms([x['acc'] for x in e1])} | {sum(x['n_flagged'] > 0 for x in e1)}/{len(e1)} |")
        L.append("")
    # i. distributed carriers
    R = load(f"logs/pair_roles/{n}_distcarr.json")
    R0 = load(f"logs/pair_roles/{n}.json")
    if R:
        L += ["## i. 운반자 분산 (운반자 0–3 을 각 3 조각 = 12 운반자)  n=%d" % len(R), "", "| | 운반자 정밀 | 운반자 재현 | |A| | ASR 요청만 → +운반자 | ACC +운반자 |", "|---|---|---|---|---|---|"]
        for tag, RR in (("분산 전", R0), ("분산 후", R)):
            e = [next(iter(r["results"].values())) for r in RR.values()] if RR else []
            if not e:
                continue
            L.append(f"| {tag} | {ms([x['carrier_precision'] for x in e], 2)} | {ms([x['carrier_recall'] for x in e], 2)} | {ms([len(x['carriers']) for x in e])} | {ms([x['eval']['request_only'].get('asr') for x in e])} → {ms([x['eval']['request+carriers'].get('asr') for x in e])} | {ms([x['eval']['request+carriers']['acc'] for x in e])} |")
        L.append("")
    if len(L) <= 2:
        sys.exit("결과 없음")
    os.makedirs("logs/adaptive", exist_ok=True)
    out = f"logs/adaptive/summary_{a.cond}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
