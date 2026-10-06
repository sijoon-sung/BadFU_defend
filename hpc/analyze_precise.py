# -*- coding: utf-8 -*-
"""analyze_precise.py — exp_llm_precise.py 결과를 시드 평균으로 묶어 표로 만든다."""
import glob, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "logs_precise")
R = defaultdict(list)
for f in sorted(glob.glob(os.path.join(D, "P_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    R[(d["args"]["trigger_pos"], d["agg"], d["rank"])].append(d)

lines = []
def out(s=""): print(s); lines.append(s)
def ms(xs, d=1):
    xs = np.array(xs, float)
    return f"{xs.mean():.{d}f}" + (f" ± {xs.std():.{d}f}" if len(xs) > 1 else "")
def P(r, ph, k, m="asr"): return r["phases"][ph]["probe"][k][m] * 100
def fin(r, ph, m="asr"): return r["phases"][ph]["final"][m] * 100

keys = sorted(R, key=lambda k: (k[0] != "prefix", k[0], k[1], k[2]))
out(f"# 정밀 검증 결과 (Qwen2.5-0.5B + AG News, 연합 LoRA 5 클라이언트)\n")
out(f"설정 수 {len(keys)}, 설정당 시드 {min(len(v) for v in R.values())}~{max(len(v) for v in R.values())}개. 값은 평균 ± 표준편차.\n")

out("## 표 1. 주 결과 — 공격 성립과 랭크 의존성\n")
out("| 위치 | 집계 | r | 시드 | ACC | LoRA 끔 ASR | 언러닝 전 ASR | 언러닝 후 ASR | 상승폭 |")
out("|---|---|---|---|---|---|---|---|---|")
for k in keys:
    rs = [r for r in R[k] if "retrain" in r["phases"]]
    if not rs: continue
    out(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {ms([fin(r,'dormant','acc') for r in rs])} | "
        f"{ms([P(r,'dormant','none') for r in rs])} | {ms([fin(r,'dormant') for r in rs])} | "
        f"{ms([fin(r,'retrain') for r in rs])} | {ms([fin(r,'retrain')-fin(r,'dormant') for r in rs])} |")

out("\n## 표 2. 모듈 분해 — q 와 v 의 비가산성 (언러닝 후 활성 상태)\n")
out("기여도는 'LoRA 끔' 기준선을 뺀 값이다. 비가산성 = 전체 기여 − (q 기여 + v 기여). "
    "양수가 크면 두 모듈이 맞물려야만 작동한다는 뜻이다.\n")
out("| 위치 | 집계 | r | 전체 | q만 | v만 | 끔 | q 기여 | v 기여 | 전체 기여 | **비가산성** |")
out("|---|---|---|---|---|---|---|---|---|---|---|")
for k in keys:
    rs = [r for r in R[k] if "retrain" in r["phases"]]
    if not rs: continue
    f_, q_, v_, n_ = ([P(r,'retrain',x) for r in rs] for x in ("full","q_only","v_only","none"))
    qc = [a-b for a,b in zip(q_,n_)]; vc = [a-b for a,b in zip(v_,n_)]; fc = [a-b for a,b in zip(f_,n_)]
    na = [a-(b+c) for a,b,c in zip(fc,qc,vc)]
    out(f"| {k[0]} | {k[1]} | {k[2]} | {ms(f_)} | {ms(q_)} | {ms(v_)} | {ms(n_)} | "
        f"{ms(qc)} | {ms(vc)} | {ms(fc)} | **{ms(na)}** |")

out("\n## 표 3. 층 묶음 분해 — 백도어는 어느 깊이에 저장되는가 (언러닝 후)\n")
out("| 위치 | 집계 | r | 전체 | 앞 0~7만 | 중간 8~15만 | 뒤 16~23만 |")
out("|---|---|---|---|---|---|---|")
for k in keys:
    rs = [r for r in R[k] if "retrain" in r["phases"]]
    if not rs: continue
    gk = [x for x in rs[0]["phases"]["retrain"]["probe"] if x.startswith("layers_")]
    out(f"| {k[0]} | {k[1]} | {k[2]} | {ms([P(r,'retrain','full') for r in rs])} | "
        + " | ".join(ms([P(r,'retrain',g) for r in rs]) for g in gk) + " |")

out("\n## 표 4. 트리거 어텐션 — 트리거 토큰이 자기 비중 대비 몇 배의 어텐션을 받는가\n")
out("| 위치 | 집계 | r | 잠복 | 활성 | 트리거 토큰 비중 |")
out("|---|---|---|---|---|---|")
for k in keys:
    rs = [r for r in R[k] if "retrain" in r["phases"]]
    g = lambda ph: [r["phases"][ph]["probe"]["attention"]["ratio_vs_share"] for r in rs
                    if r["phases"][ph]["probe"].get("attention", {}).get("ratio_vs_share")]
    a, b = g("dormant"), g("retrain")
    if not a: continue
    sh = [r["phases"]["retrain"]["probe"]["attention"]["trigger_token_share"] for r in rs]
    out(f"| {k[0]} | {k[1]} | {k[2]} | {ms(a,2)}배 | {ms(b,2)}배 | {ms([s*100 for s in sh],1)}% |")

out("\n## 층별 단독 ASR (층 i 의 LoRA 만 남겼을 때, 활성 상태, 시드 평균)\n")
for k in keys:
    rs = [r for r in R[k] if "retrain" in r["phases"]]
    if not rs: continue
    a = np.array([r["phases"]["retrain"]["probe"]["per_layer"] for r in rs]) * 100
    out(f"- `{k[0]} {k[1]} r={k[2]}` " + " ".join(f"{x:3.0f}" for x in a.mean(0)))

out("\n## 층 누적 ASR (0층부터 i층까지 남겼을 때, 활성 상태, 시드 평균)\n")
for k in keys:
    rs = [r for r in R[k] if "retrain" in r["phases"]]
    if not rs: continue
    a = np.array([r["phases"]["retrain"]["probe"]["cumulative"] for r in rs]) * 100
    out(f"- `{k[0]} {k[1]} r={k[2]}` " + " ".join(f"{x:3.0f}" for x in a.mean(0)))

with open(os.path.join(D, "summary_precise.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nsaved:", os.path.join(D, "summary_precise.md"))
