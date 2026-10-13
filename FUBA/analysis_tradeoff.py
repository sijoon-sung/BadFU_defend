# -*- coding: utf-8 -*-
"""적응형 공격자의 trade-off (반합성): 억제자의 보관 기여를 비율 s 로 줄였을 때
  (a) 학습 종료 모델 θ_T − (1−s)·U_S 의 ASR/ACC  = 억제를 s 만큼만 했을 때 학습 중에 드러나는 정도 (잠복 실패)
  (b) 억제자 기여를 s 배로 줄인 누적 기여도에서의 z_s(c*) 와 H(c*)        = CCD 가 보는 정도
를 같은 s 축에 그린다. 공격자가 억제를 약하게 해서 z 를 문턱 아래로 내리려면 어느 정도 잠복을 포기해야 하는가.
주의: 실제 적응형 공격은 주입자 학습도 달라지므로 이것은 상한/하한의 감(감도) 이지 실측 대체가 아니다 (brief 에 명시).
사용: python analysis_tradeoff.py --name loc_k20paper_s0 --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release_map import acc_of, load_fuba, load_probe, make_model, make_scorer  # noqa: E402
from ledger_defense import mad_z  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name"); p.add_argument("--K", type=int, default=20); p.add_argument("--rounds", type=int, default=100)
p.add_argument("--ckpt_dir", default="checkpoints"); p.add_argument("--fmt", default="fuba"); p.add_argument("--dataset", default="mnist")
p.add_argument("--probe", type=int, default=500); p.add_argument("--target", type=int, default=8)
p.add_argument("--defender", type=int, default=4); p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
p.add_argument("--atk_eps", type=float, default=1.0); p.add_argument("--noise_thread", type=float, default=0.04); p.add_argument("--data_root", default="./data")
p.add_argument("--s_grid", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
a = p.parse_args()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
factory, theta_T, U, flat, bd = load_fuba(a)
_, eval_loader = load_probe(a)
score = make_scorer(a, bd, eval_loader, dev)
tT = flat.flatten(theta_T); S = a.defender
d = np.load(f"logs/ledger/{a.name}_push.npz"); K, ts, js, vs = int(d["K"]), d["ts"], d["js"], d["vs"]
P = np.zeros((K, vs.shape[1]))
for j, v in zip(js, vs):
    P[j] += v
rows = []
for s in a.s_grid:
    m = make_model(factory, flat.to_sd(tT - (1.0 - s) * U[S]), dev)
    acc, asr = acc_of(m, eval_loader, dev), score(m)
    Ps = P.copy(); Ps[S] *= s
    col = Ps[:, a.target]; z = mad_z(col); H = 1.0 - abs(col.sum()) / max(np.abs(col).sum(), 1e-9)
    rows.append({"s": s, "acc": acc, "asr": asr, "z_sup": round(float(z[S]), 2), "H": round(float(H), 3), "rank_sup": int(1 + (col < col[S]).sum())})
    print(f"s={s:.1f}  ACC {acc:5.1f}  ASR {asr:5.1f}  | z_sup {z[S]:6.2f}  H {H:.3f}  억제자 순위(가장 음) {rows[-1]['rank_sup']}")
os.makedirs("logs/tradeoff", exist_ok=True)
json.dump({"name": a.name, "rows": rows}, open(f"logs/tradeoff/{a.name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
fig, ax1 = plt.subplots(figsize=(6.2, 3.6)); ax2 = ax1.twinx()
ss = [r["s"] for r in rows]
ax1.plot(ss, [r["asr"] for r in rows], "o-", color="#c0392b", label="학습 종료 모델의 ASR (%)")
ax1.plot(ss, [r["acc"] for r in rows], "s--", color="#7f8c8d", label="정확도 (%)")
ax2.plot(ss, [-r["z_sup"] for r in rows], "^-", color="#2980b9", label="억제자 |z| (CCD)")
ax2.axhline(3, color="#2980b9", ls=":", lw=1); ax2.text(0.02, 3.3, "탐지 문턱 τ=3", color="#2980b9", fontsize=8)
ax1.set_xlabel("억제 강도 s (보관된 억제자 기여의 비율; 1 = 원 공격)"); ax1.set_ylabel("ASR / 정확도 (%)"); ax2.set_ylabel("억제자 |z|")
ax1.set_ylim(0, 102); ax2.set_ylim(0, max(4, max(-r["z_sup"] for r in rows) * 1.1))
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
plt.title("억제를 약하게 하면: 잠복(낮은 ASR)과 탐지 회피(낮은 |z|)는 같이 얻을 수 없다", fontsize=9)
plt.tight_layout(); plt.savefig("../docs/paper/fig_tradeoff.png", dpi=160); print("-> logs/tradeoff, fig_tradeoff.png")
