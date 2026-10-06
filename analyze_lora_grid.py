# -*- coding: utf-8 -*-
"""
analyze_lora_grid.py — exp_lora_badfu_grid.py 결과(JSON)를 시드 평균으로 묶어 표와 그림을 만든다.

  python analyze_lora_grid.py                       # logs/lora_grid
  python analyze_lora_grid.py --dir logs/lora_noniid --tag _d0.9
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--dir", default="logs/lora_grid")
p.add_argument("--tag", default="")
args = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, args.dir)

R = defaultdict(list)                     # (agg, rank) -> [run, ...]
for f in sorted(glob.glob(os.path.join(D, f"*{args.tag}.json"))):
    d = json.load(open(f, encoding="utf-8"))
    if d["args"].get("tag", "") != args.tag: continue
    R[(d["agg"], d["rank"])].append(d)
ranks = sorted({r for a, r in R if a != "full"})
aggs = [a for a in ("full", "fedex", "fedit") if any(k[0] == a for k in R)]
keys = [k for a in aggs for k in sorted(R) if k[0] == a]

def ms(xs, pct=True):
    xs = np.array(xs, float) * (100 if pct else 1)
    return f"{xs.mean():5.1f} ± {xs.std():4.1f}" if pct else f"{xs.mean():+.2f} ± {xs.std():.2f}"
def get(run, ph, k): return run["phases"][ph]["final"][k]
def lab(a, r): return "Full FT (FedAvg)" if a == "full" else f"{'FedEx-LoRA' if a == 'fedex' else 'FedIT'} r={r}"

lines = []
def out(s=""): print(s); lines.append(s)

n = len(next(iter(R.values())))
out(f"# LoRA x BadFU 격자 결과 ({args.dir}{args.tag}, 시드 {n}개, 평균 ± 표준편차)\n")
out("ASR 은 마지막 5 라운드 평균. pre = 잠복(dormant) 모델, post = D_c 삭제 후 재학습 모델, clean = 공격 없는 대조군.\n")
out("## 표 1. 주 격자\n")
out("| 설정 | ACC pre | ASR clean | ASR pre | ASR post | dASR (post-pre) | 공격자 로컬 ASR pre |")
out("|---|---|---|---|---|---|---|")
for k in keys:
    rs = R[k]
    has_clean = all("clean" in r["phases"] for r in rs)
    out(f"| {lab(*k)} | {ms([get(r,'dormant','acc') for r in rs])} | "
        f"{ms([get(r,'clean','asr_last5') for r in rs]) if has_clean else '-'} | "
        f"{ms([get(r,'dormant','asr_last5') for r in rs])} | {ms([get(r,'retrain','asr_last5') for r in rs])} | "
        f"{ms([get(r,'retrain','asr_last5') - get(r,'dormant','asr_last5') for r in rs])} | "
        f"{ms([get(r,'dormant','atk_asr_last5') for r in rs])} |")

out("\n## 표 2. 가설 A — FedIT 와 FedEx-LoRA 의 짝 비교 (같은 시드, 같은 초기값)\n")
out("FedIT - FedEx 차이. 양수면 FedIT 쪽이 높다. '부호 일치'는 시드 중 차이가 양수인 개수.\n")
out("| r | ASR pre 차이 (%p) | 부호 일치 | ASR post 차이 (%p) | 상대 불일치 (전 라운드 평균) | 같은 업데이트 반사실 차이 ASR (%p) | 같은 업데이트 반사실 차이 margin |")
out("|---|---|---|---|---|---|---|")
for r in ranks:
    ex = {x["seed"]: x for x in R.get(("fedex", r), [])}; it = {x["seed"]: x for x in R.get(("fedit", r), [])}
    ss = sorted(set(ex) & set(it))
    if not ss: continue
    dpre = [get(it[s],'dormant','asr_last5') - get(ex[s],'dormant','asr_last5') for s in ss]
    dpost = [get(it[s],'retrain','asr_last5') - get(ex[s],'retrain','asr_last5') for s in ss]
    mm = [get(x,'dormant','mismatch_mean') for x in list(ex.values()) + list(it.values())]
    # 반사실: 매 라운드 같은 클라이언트 업데이트를 FedIT 식으로 합친 모델 - 정확히 합친 모델
    cfa, cfm = [], []
    for x in list(ex.values()) + list(it.values()):
        sgn = 1 if x["agg"] == "fedit" else -1
        h = x["phases"]["dormant"]["hist"]
        cfa.append(np.mean([sgn * (q["asr"] - q["cf_asr"]) for q in h]))
        cfm.append(np.mean([sgn * (q["margin"] - q["cf_margin"]) for q in h]))
    out(f"| {r} | {ms(dpre)} | {sum(d > 0 for d in dpre)}/{len(ss)} | {ms(dpost)} | "
        f"{np.mean(mm):.3f} | {ms(cfa)} | {ms(cfm, pct=False)} |")

has_cont = all("cont_dormant" in x["phases"] for v in R.values() for x in v)
if has_cont:
    out("\n## 표 3. 가설 B 메커니즘 — 잠복 모델 안에 백도어가 저장돼 있는가\n")
    out("D_c 를 뺀 공격자와 함께 이어서 학습했을 때의 ASR. 출발점이 잠복 모델(cont_dormant)인 경우와 "
        "공격 없던 모델(cont_clean)인 경우를 비교한다. 저장량 = 10 라운드 평균 ASR 차이.\n")
    out("| 설정 | 1R 잠복→ | 1R 깨끗→ | 3R 잠복→ | 3R 깨끗→ | 10R 잠복→ | 10R 깨끗→ | 저장량 (%p) |")
    out("|---|---|---|---|---|---|---|---|")
    for k in keys:
        rs = R[k]
        cd = np.array([[q["asr"] for q in x["phases"]["cont_dormant"]["hist"]] for x in rs]) * 100
        cc = np.array([[q["asr"] for q in x["phases"]["cont_clean"]["hist"]] for x in rs]) * 100
        cell = lambda a, i: f"{a[:, i].mean():.1f}"
        out(f"| {lab(*k)} | {cell(cd,0)} | {cell(cc,0)} | {cell(cd,2)} | {cell(cc,2)} | {cell(cd,-1)} | {cell(cc,-1)} | "
            f"{(cd - cc).mean(1).mean():5.1f} ± {(cd - cc).mean(1).std():4.1f} |")

with open(os.path.join(D, f"summary{args.tag}.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"fedex": "#2a78d6", "fedit": "#eb6834", "full": "#52514e"}
NAME = {"fedex": "FedEx-LoRA (교차항 없음)", "fedit": "FedIT (교차항 있음)", "full": "Full FT (FedAvg)"}
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for ax, (ph, title) in zip(axs, (("dormant", "언러닝 전 ASR (잠복)"), ("retrain", "언러닝 후 ASR (재학습)"), (None, "dASR = 후 - 전"))):
    for a in ("fedex", "fedit"):
        ys = []
        for r in ranks:
            rs = R.get((a, r), [])
            v = [(get(x,'retrain','asr_last5') - get(x,'dormant','asr_last5')) if ph is None else get(x, ph, 'asr_last5') for x in rs]
            ys.append((np.mean(v) * 100, np.std(v) * 100))
        ys = np.array(ys)
        ax.errorbar(ranks, ys[:, 0], yerr=ys[:, 1], color=COL[a], lw=2, marker="o", ms=6, capsize=3, label=NAME[a])
    if ("full", 0) in R:
        rs = R[("full", 0)]
        v = [(get(x,'retrain','asr_last5') - get(x,'dormant','asr_last5')) if ph is None else get(x, ph, 'asr_last5') for x in rs]
        ax.axhline(np.mean(v) * 100, color=COL["full"], lw=1.2, ls="--", label=NAME["full"])
    ax.set_xscale("log", base=2); ax.set_xticks(ranks); ax.set_xticklabels(ranks)
    ax.set_xlabel("LoRA 랭크 r"); ax.set_title(title, color=INK, fontsize=11, loc="left"); style(ax)
axs[0].set_ylabel("ASR (%)"); axs[0].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(D, f"fig_asr_vs_rank{args.tag}.png"), dpi=150); plt.close(fig)

# ASR 궤적 (라운드별), 대표 랭크
show = [r for r in (1, 4, 64) if r in ranks] or ranks[:3]
fig, axs = plt.subplots(1, len(show), figsize=(4.3 * len(show), 3.8), sharey=True)
axs = np.atleast_1d(axs)
for ax, r in zip(axs, show):
    for a in ("fedex", "fedit"):
        for ph, ls in (("dormant", "-"), ("retrain", ":")):
            arr = np.array([[q["asr"] for q in x["phases"][ph]["hist"]] for x in R.get((a, r), [])]) * 100
            if len(arr) == 0: continue
            ax.plot(range(1, arr.shape[1] + 1), arr.mean(0), color=COL[a], lw=2, ls=ls,
                    label=f"{NAME[a].split(' ')[0]} {'전(잠복)' if ph == 'dormant' else '후(재학습)'}")
    ax.set_title(f"r = {r}", color=INK, fontsize=11, loc="left"); ax.set_xlabel("라운드"); style(ax)
axs[0].set_ylabel("ASR (%)"); axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(D, f"fig_asr_rounds{args.tag}.png"), dpi=150); plt.close(fig)

if has_cont:
    fig, axs = plt.subplots(1, len(show) + 1, figsize=(4.3 * (len(show) + 1), 3.8), sharey=True)
    for ax, k in zip(axs, [("full", 0)] + [("fedex", r) for r in show]):
        if k not in R: continue
        for ph, ls, nm in (("cont_dormant", "-", "잠복 모델에서 출발"), ("cont_clean", "--", "깨끗한 모델에서 출발")):
            arr = np.array([[q["asr"] for q in x["phases"][ph]["hist"]] for x in R[k]]) * 100
            ax.plot(range(1, arr.shape[1] + 1), arr.mean(0), color=COL[k[0]], lw=2, ls=ls, marker="o", ms=4, label=nm)
        ax.set_title(lab(*k), color=INK, fontsize=11, loc="left"); ax.set_xlabel("이어서 학습한 라운드"); style(ax)
    axs[0].set_ylabel("ASR (%)"); axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(D, f"fig_storage{args.tag}.png"), dpi=150); plt.close(fig)
print("\nsaved:", D)
