# -*- coding: utf-8 -*-
"""논문 표 생성: logs/* 의 조건별 시드 결과를 RQ 별 표로 모은다 (docs/paper/EXPERIMENT_DESIGN.md 5 절).

  python paper_tables.py --cond iid        → docs/paper/tables_iid.md
T1 RQ1  FU 연산자(benign_departure) + DOSE-FU(dose_fu) + FedEraser-R(robust_fu): 요청자 × 방법 (ASR/ACC/유지/망각)
T2 RQ2  탐지 비교: MASA, SubID 식, 잠재 위험 지도, 빼 보기, DOSE-FU 경보 (운반자/위장자 AUROC·순위)
T2b     집계 방어(TAG=_median 등) 위에서의 활성화와 DOSE-FU
T3 RQ3  쌍→집합
T4 RQ4  적응형 (aggregate_adaptive 의 표 인용)
T5 RQ5  비용 (bench + dose 의 단계·가상 모델 수)
T6 RQ6  소거 (TAG=_cap02 …)
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
OUT_DIR = os.path.join("..", "docs", "paper")


def ms(xs, nd=1):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def load(d, prefix, suffix=""):
    """logs/{d}/{prefix}_s{seed}{suffix}.json 만 (다른 접미사 제외)."""
    rx = re.compile("^" + re.escape(prefix) + r"_s(\d+)" + re.escape(suffix) + r"\.json$")
    out = []
    for f in sorted(glob.glob(f"logs/{d}/{prefix}_s*{suffix}.json")):
        if rx.match(os.path.basename(f)):
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def t1(cond, L, tag=""):
    bd = load("benign_departure", f"bd_{cond}{tag}")
    ds = load("dose_fu", f"bd_{cond}{tag}")
    rf = load("robust_fu", f"bd_{cond}{tag}") if not tag else []
    if not bd and not ds:
        return
    L += [f"## T1{tag}. RQ1 — FU 연산자 vs DOSE-FU ({cond}{tag}, n={max(len(bd), len(ds))})", ""]
    reqs = list(ds[0]["results"].keys()) if ds else list(bd[0]["results"].keys())
    roles = {}
    if ds:
        roles = {k: next(iter(ds[0]["results"][k].values()))["role"] for k in reqs}
    L.append("| 방법 | " + " | ".join(f"req {k} ({roles.get(k, '')}) ASR / ACC / 유지변화" for k in reqs) + " |")
    L.append("|---|" + "---|" * len(reqs))
    if bd:
        for mth in ("subtract", "distillation", "fedEraser", "retrain_benign", "retrain_attack"):
            cells = []
            for k in reqs:
                e = [r["results"].get(mth, {}).get(k) for r in bd]      # benign_departure: results[method][requester]
                e = [x for x in e if x]
                cells.append(f"{ms([x.get('asr') for x in e])} / {ms([x.get('acc') for x in e])} / {ms([x.get('retain_change') for x in e], 2)}" if e else "-")
            L.append(f"| FU: {mth} | " + " | ".join(cells) + " |")
    if rf:
        for arm in ("plain", "purify_flag"):
            cells = []
            for k in reqs:
                e = [r["results"].get(k, {}).get(arm) for r in rf]
                e = [x for x in e if x]
                cells.append(f"{ms([x.get('asr') for x in e])} / {ms([x.get('acc') for x in e])} / {ms([x.get('retain_change') for x in e], 2)}" if e else "-")
            L.append(f"| FedEraser-R: {arm} | " + " | ".join(cells) + " |")
    if ds:
        for arm in ("plain", "oneshot", "dose", "always"):
            cells = []
            for k in reqs:
                e = [r["results"].get(k, {}).get(arm) for r in ds]
                e = [x for x in e if x]
                cells.append(f"{ms([x.get('asr') for x in e])} / {ms([x.get('acc') for x in e])} / {ms([x.get('retain_change') for x in e], 2)}" if e else "-")
            L.append(f"| **DOSE-FU: {arm}** | " + " | ".join(cells) + " |")
        L.append("")
        L.append("망각 (자기 데이터 acc before → after / MIA AUROC) 와 경보·수정·단계·가상 모델 수 (DOSE-FU):")
        L.append("| arm | " + " | ".join(f"req {k}" for k in reqs) + " |")
        L.append("|---|" + "---|" * len(reqs))
        for arm in ("plain", "oneshot", "dose", "always"):
            cells = []
            for k in reqs:
                e = [r["results"].get(k, {}).get(arm) for r in ds]
                e = [x for x in e if x]
                if not e:
                    cells.append("-"); continue
                fb = [x.get("forget", {}).get("before", {}).get("acc") for x in e]
                ff = [x.get("forget", {}).get("final", {}).get("acc") for x in e]
                au = [x.get("mia_req", {}).get("auroc") for x in e]
                cells.append(f"{ms(fb)}→{ms(ff)} / {ms(au, 2)} ; α {ms([x.get('reinjection_alpha') for x in e], 3)}, 경보 {ms([x['n_flagged'] for x in e])}, 수정 {ms([x['n_repaired'] for x in e])}, 단계 {ms([x['n_steps'] for x in e])}, 뺀 양 {ms([x.get('removed') for x in e], 2)}, 모델 {ms([x['pop_models'] for x in e], 0)}")
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("")


def t2(cond, L):
    masa = load("baseline_masa", f"bd_{cond}")
    sub = load("baseline_subid", f"bd_{cond}")
    rel = load("release_map", f"bd_{cond}")
    ds = load("dose_fu", f"bd_{cond}")
    lat = load("latent_risk", f"bd_{cond}")
    if not (masa or sub or rel or ds):
        return
    L += [f"## T2. RQ2 — 탐지 비교 ({cond})", "", "| 방법 | 보는 것 | 시점 | 운반자 vs 정상 AUROC | 위장자 vs 정상 AUROC | 위장자 순위 / 표시 |", "|---|---|---|---|---|---|"]
    if masa:
        L.append(f"| MASA | 로컬 모델에 언러닝 걸 때 손실 | 학습 중 | {ms([r['summary']['auroc_attacker_vs_benign'] for r in masa], 2)} | {ms([r['summary']['auroc_defender_vs_benign'] for r in masa], 2)} | 표시율 {ms([r['summary']['flag_rate']['defender'] for r in masa], 2)} |")
    if sub:
        L.append(f"| SubID 식 (제거 벡터) | 갱신의 부분공간 잔차 | 요청 시점 | {ms([r['summary']['removal']['auroc_attacker_vs_benign'] for r in sub], 2)} | - | 순위 {ms([r['summary']['removal']['requester_rank'] for r in sub], 1)} / {sum(r['summary']['removal']['requester_flagged'] for r in sub)}/{len(sub)} |")
        L.append(f"| SubID 식 (학습 갱신) | 같음 | 학습 중 | {ms([r['summary']['training']['auroc_attacker_vs_benign'] for r in sub], 2)} | - | 순위 {ms([r['summary']['training']['defender_rank'] for r in sub], 1)} |")
    if lat:
        feat = "shape_frob"
        vals = [r.get("separation", {}).get(feat, {}) for r in lat]
        L.append(f"| 잠재 위험 지도 ({feat}) | 가중치 노름·형상 (탐침 없음) | 학습 후 | {ms([v.get('auroc_mal_vs_benign') for v in vals], 2)} (운반자+위장자) | {ms([v.get('auroc_def_vs_rest') for v in vals], 2)} (vs 나머지) | 순위 {ms([v.get('defender_rank_desc') for v in vals], 1)} |")
    if rel:
        def au(r):
            c = r["clients"]
            pos = [v["curve"][-1]["mass"] for v in c.values() if v["role"] == "requester"]
            neg = [v["curve"][-1]["mass"] for v in c.values() if v["role"] == "benign"]
            if not pos or not neg:
                return None
            return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
        L.append(f"| 빼 보기 (깨끗한 입력, ρ 스윕) | 제거 효과 | 요청 시점 | - | {ms([au(r) for r in rel], 2)} | 순위 {ms([r['requester_rank']['clean_mass']['rank'] for r in rel], 1)}, z {ms([r['requester_rank']['clean_mass']['z'] for r in rel], 1)} |")
    if ds:
        reqs = list(ds[0]["results"].keys())
        k4 = reqs[0]
        zs = [max((s["z"] for s in r["results"][k4]["oneshot"]["trace"] if s["z"] is not None), default=None) for r in ds if "oneshot" in r["results"].get(k4, {})]
        fl = [r["results"][k4]["oneshot"]["n_flagged"] > 0 for r in ds if "oneshot" in r["results"].get(k4, {})]
        fb = [r["results"][k]["oneshot"]["n_flagged"] > 0 for r in ds for k in reqs[1:] if "oneshot" in r["results"].get(k, {})]
        L.append(f"| **DOSE-FU (알고리즘 1)** | 제거 효과 vs 같은 양 모집단 | 요청 시점 | (알고리즘 2 로) | z {ms(zs, 1)} | 표시 {sum(fl)}/{len(fl)}, 정상 오탐 {sum(fb)}/{len(fb)} |")
    L.append("")


def t2b(cond, L, tdefs):
    rows = []
    for d in tdefs:
        bd = load("benign_departure", f"bd_{cond}_{d}")
        ds = load("dose_fu", f"bd_{cond}_{d}")
        if not bd and not ds:
            continue
        k = list((ds or bd)[0]["results"].keys())[0]
        asr_sub = [r["results"].get("subtract", {}).get(k, {}).get("asr") for r in bd]
        asr_dose = [r["results"].get(k, {}).get("dose", {}).get("asr") for r in ds]
        acc_dose = [r["results"].get(k, {}).get("dose", {}).get("acc") for r in ds]
        before = [r["before"].get("asr") for r in (ds or bd)]
        rows.append(f"| {d} | {ms(before)} | {ms(asr_sub)} | {ms(asr_dose)} / {ms(acc_dose)} |")
    if rows:
        L += [f"## T2b. RQ2 — 학습 중 집계 방어 위에서 ({cond})", "", "| 집계 방어 | 삭제 전 ASR | 빼기 FU 뒤 ASR (위장자 요청) | DOSE-FU 뒤 ASR / ACC |", "|---|---|---|---|"] + rows + [""]


def t3(cond, L):
    pr = load("pair_roles", f"bd_{cond}")
    jt = load("pair_roles", f"bd_{cond}", "_joint")
    if not pr:
        return
    L += [f"## T3. RQ3 — 쌍→집합 ({cond}, n={len(pr)})", "", "| 요청 S | z0 | 표시 | 지도 운반자 AUROC | A 정밀 / 재현 | 공동 위장자 C | ASR 요청만 → +운반자 (ACC) |", "|---|---|---|---|---|---|---|"]
    for runs in (pr, jt):
        if not runs:
            continue
        for k in runs[0]["results"].keys():
            e = [r["results"][k] for r in runs if k in r["results"]]
            L.append(f"| {k} ({','.join(e[0]['roles_S'])}) | {ms([x['z0'] for x in e])} | {sum(x['flagged'] for x in e)}/{len(e)} | {ms([x['auroc_carrier'] for x in e], 2)} | {ms([x['carrier_precision'] for x in e], 2)} / {ms([x['carrier_recall'] for x in e], 2)} | {'; '.join(str(x['co_suppressors']) for x in e)} | {ms([x['eval']['request_only'].get('asr') for x in e])} → {ms([x['eval']['request+carriers'].get('asr') for x in e])} ({ms([x['eval']['request+carriers']['acc'] for x in e])}) |")
    L.append("")


def t4(cond, L):
    f = f"logs/adaptive/summary_{cond}.md"
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            body = fh.read().split("\n", 1)[1]
        L += [f"## T4. RQ4 — 적응형 공격 ({cond})", body, ""]


def t5(cond, L):
    f = "logs/bench/release_map_fuba.json"
    ds = load("dose_fu", f"bd_{cond}")
    L += [f"## T5. RQ5 — 비용 ({cond})", ""]
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            b = json.load(fh)
        L += ["| K | 전체 (초) | 축소: 표본 5 + 탐색 1 회 | 깨끗한 입력만 |", "|---|---|---|---|"]
        for K, row in b["rows"].items():
            L.append(f"| {K} | {row.get('full', '-')} | {row.get('cheap', '-')} | {row.get('clean_only', '-')} |")
    if ds:
        reqs = list(ds[0]["results"].keys())
        L += ["", "DOSE-FU 한 요청당 (K=8): 단계 수 / 가상 모델 수 / 초", "| 요청자 | oneshot | dose |", "|---|---|---|"]
        for k in reqs:
            cells = []
            for arm in ("oneshot", "dose"):
                e = [r["results"][k][arm] for r in ds if arm in r["results"].get(k, {})]
                cells.append(f"{ms([x['n_steps'] for x in e])} / {ms([x['pop_models'] for x in e], 0)} / {ms([x['seconds'] for x in e])}" if e else "-")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")


def t6(cond, L):
    tags = [("", "기본 (c=0.3, τ=3, m=전원, mass, n=500, subtract)"), ("_cap02", "c=0.2"), ("_tau2", "τ=2"), ("_tau5", "τ=5"), ("_m3", "m=3"), ("_chg", "stat=changed"), ("_flow", "stat=flow"), ("_p100", "n=100"), ("_kd", "op=kd")]
    rows = []
    for tag, label in tags:
        ds = load("dose_fu", f"bd_{cond}{tag}")
        if not ds:
            continue
        reqs = list(ds[0]["results"].keys())
        cells = []
        for k in reqs:
            e = [r["results"][k]["dose"] for r in ds if "dose" in r["results"].get(k, {})]
            cells.append(f"{ms([x.get('asr') for x in e])} / {ms([x['acc'] for x in e])} / 경보 {ms([x['n_flagged'] for x in e])} / 단계 {ms([x['n_steps'] for x in e])}" if e else "-")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    if len(rows) > 1:
        reqs = list(load("dose_fu", f"bd_{cond}")[0]["results"].keys())
        L += [f"## T6. RQ6 — 소거 ({cond}, dose arm)", "", "| 설정 | " + " | ".join(f"req {k} ASR / ACC / 경보 / 단계" for k in reqs) + " |", "|---|" + "---|" * len(reqs)] + rows + [""]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", required=True)
    p.add_argument("--tdefs", nargs="*", default=["median", "multi_krum", "flame", "flDetector"])
    a = p.parse_args()
    L = [f"# 논문 표: {a.cond} (시드 평균 ± 표준편차; 생성 `python paper_tables.py --cond {a.cond}`)", ""]
    t1(a.cond, L)
    t2(a.cond, L)
    t2b(a.cond, L, a.tdefs)
    t3(a.cond, L)
    t4(a.cond, L)
    t5(a.cond, L)
    t6(a.cond, L)
    if len(L) <= 2:
        sys.exit("결과 없음")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"tables_{a.cond}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
