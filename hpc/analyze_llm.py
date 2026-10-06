# -*- coding: utf-8 -*-
"""
analyze_llm.py — exp_llm_badfu.py 결과(JSON)를 표로 묶는다.
  python analyze_llm.py --dir logs
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="logs")
ap.add_argument("--out", default="summary_llm.md")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, args.dir)

R = defaultdict(list)
for f in sorted(glob.glob(os.path.join(D, "llm_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    R[(d["agg"], d["rank"], d["args"]["trigger_pos"])].append(d)

lines = []
def out(s=""): print(s); lines.append(s)

def ms(xs, pct=True):
    xs = np.array(xs, float) * (100 if pct else 1)
    return f"{xs.mean():5.1f}" + (f" ± {xs.std():4.1f}" if len(xs) > 1 else "")

def fin(r, ph, k="asr"): return r["phases"][ph]["final"][k]
def abl(r, ph, m, k="asr"): return r["phases"][ph]["ablation"][m][k]

keys = sorted(R, key=lambda k: (k[2], k[0], k[1]))
n_seed = max(len(v) for v in R.values()) if R else 0
out(f"# 연합 LoRA 언어모델에서의 BadFU (Qwen2.5-0.5B + AG News, 시드 {n_seed}개)\n")
out("ASR = 타깃이 아닌 테스트 샘플에 트리거를 넣었을 때 타깃 라벨로 분류되는 비율. "
    "pre = 언러닝 전(잠복), post = D_c 삭제 후 재학습.\n")

out("## 표 1. 주 결과\n")
out("| 트리거 위치 | 집계 | r | ACC pre | ASR clean | ASR pre | ASR post | dASR |")
out("|---|---|---|---|---|---|---|---|")
for k in keys:
    rs = R[k]
    has_clean = all("clean" in r["phases"] for r in rs)
    has_rt = all("retrain" in r["phases"] for r in rs)
    out(f"| {k[2]} | {k[0]} | {k[1]} | {ms([fin(r,'dormant','acc') for r in rs])} | "
        f"{ms([fin(r,'clean') for r in rs]) if has_clean else '-'} | "
        f"{ms([fin(r,'dormant') for r in rs])} | "
        f"{ms([fin(r,'retrain') for r in rs]) if has_rt else '-'} | "
        f"{ms([fin(r,'retrain') - fin(r,'dormant') for r in rs]) if has_rt else '-'} |")

out("\n## 표 2. 모듈 분해 — 백도어는 q 와 v 중 어디에 저장되는가\n")
out("LoRA 를 한 모듈만 남기고 나머지를 끈 뒤 잰 ASR. 전체 대비 비율이 높은 쪽이 백도어를 담고 있다.\n")
out("| 트리거 위치 | 집계 | r | 단계 | 전체 ASR | q_proj 만 | v_proj 만 | v 쏠림 (v−q) |")
out("|---|---|---|---|---|---|---|---|")
for k in keys:
    for ph in ("dormant", "retrain"):
        rs = [r for r in R[k] if ph in r["phases"]]
        if not rs: continue
        out(f"| {k[2]} | {k[0]} | {k[1]} | {ph} | {ms([fin(r,ph) for r in rs])} | "
            f"{ms([abl(r,ph,'q_only') for r in rs])} | {ms([abl(r,ph,'v_only') for r in rs])} | "
            f"{ms([abl(r,ph,'v_only') - abl(r,ph,'q_only') for r in rs])} |")

out("\n## 표 3. 집계 불일치 (FedIT 와 정확한 평균의 상대 차이)\n")
out("| 트리거 위치 | 집계 | r | 전 라운드 평균 불일치 |")
out("|---|---|---|---|")
for k in keys:
    v = [np.mean([q["mismatch"] for q in r["phases"]["dormant"]["hist"]]) for r in R[k] if "dormant" in r["phases"]]
    if v: out(f"| {k[2]} | {k[0]} | {k[1]} | {np.mean(v):.3f} |")

out("\n## 라운드별 ASR 궤적\n")
for k in keys:
    for ph in ("dormant", "retrain"):
        rs = [r for r in R[k] if ph in r["phases"]]
        if not rs: continue
        a = np.array([[q["asr"] for q in r["phases"][ph]["hist"]] for r in rs]) * 100
        out(f"- `{k[0]} r={k[1]} {k[2]} {ph:8s}` " + " ".join(f"{x:4.0f}" for x in a.mean(0)))

with open(os.path.join(D, args.out), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nsaved:", os.path.join(D, args.out))
