# -*- coding: utf-8 -*-
"""unified_fu.py 결과를 조건별로 모은다.  python aggregate_unified.py --cond iid  → logs/unified/summary_{cond}.md"""
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
    p.add_argument("--dir", default="logs/unified")
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
    reqs = list(runs[0]["results"].keys())
    L = [f"# 하나의 절차 (unified_fu) 요약: {a.cond or 'all'}  n={len(runs)}", "",
         f"삭제 전 ASR {ms([r['before'].get('asr') for r in runs])} / ACC {ms([r['before'].get('acc') for r in runs])}", "",
         "| 요청자 | 역할 | 결정 (시드별) | 거름망 통과 | 확정 z (최대) | ASR | ACC | 유지 변화 | α | 자기 데이터 acc | 운반자 | 모델 / 순전파 / 초 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for k in reqs:
        e = [r["results"][k] for r in runs if k in r["results"]]
        dec = "; ".join(x["decision"] for x in e)
        passed = sum(1 for x in e if x["decision"] == "accept(screen)")
        zmax = [max((s["z"] for s in x.get("confirm", {}).get("trace", []) if s.get("z") is not None), default=None) for x in e]
        carriers = "; ".join(str(x.get("purify", {}).get("carriers", "")) for x in e if "purify" in x) or "-"
        L.append(f"| {k} | {e[0]['role']} | {dec} | {passed}/{len(e)} | {ms(zmax)} | {ms([x['final'].get('asr') for x in e])} | {ms([x['final']['acc'] for x in e])} | "
                 f"{ms([x['final']['retain_change'] for x in e], 2)} | {ms([x['final'].get('alpha') for x in e], 3)} | {ms([x['final'].get('forget_acc') for x in e])} | {carriers} | "
                 f"{ms([x['cost']['pop_models'] for x in e], 0)} / {ms([x['cost']['forwards'] for x in e], 0)} / {ms([x['cost']['seconds'] for x in e])} |")
        summary[k] = {"decisions": [x["decision"] for x in e], "asr": [x["final"].get("asr") for x in e], "acc": [x["final"]["acc"] for x in e]}
    L.append("")
    L.append("결정 뜻: accept(screen)=거름망 통과·기본 FU 와 동일 / accept(confirm)=확인 뒤 수락 / purified(antidote)=운반자 성분 제거 후 수락 / repaired(KD)=재억제 수정 후 수락 / reject(...)=불출")
    tag = f"_{a.cond}" if a.cond else ""
    with open(f"{a.dir}/summary{tag}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(f"{a.dir}/summary{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n".join(L))
    print(f"-> {a.dir}/summary{tag}.md")


if __name__ == "__main__":
    main()
