# -*- coding: utf-8 -*-
"""헤지 파일럿 집계: logs/analysis/hedge_*.json → logs/analysis/hedge_summary.md
판정 기준 (파일럿의 go/no-go): 위장자–운반자 쌍의 코사인이 나머지 쌍(위장자–정상, 운반자–정상, 정상–정상)과 뚜렷이 갈리는가.
  AUROC = "낮은 코사인(반상관)일수록 위장자–운반자 쌍" 으로 두 집단을 가르는 점수. 0.5 = 구분 못 함, 1.0 = 완전 분리.
  전체 매개변수 / 공통 성분 제거 뒤 / 층별 / 누적 기여, 네 판으로 각각 계산한다.
실행: python aggregate_hedge.py            (모든 hedge_*.json)   또는 --cond iid (이름이 bd_iid_s* 인 것만)
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def auroc(pos, neg):
    """pos 가 neg 보다 작아야(반상관) 양성. 동점 0.5."""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p < n else (0.5 if p == n else 0.0)
    return round(s / (len(pos) * len(neg)), 3)


def split_spec(P, roles):
    """짝 특이도: 위장자–운반자 쌍(양성) vs 위장자–정상 쌍(음성). 헤지가 맞으면 위장자는 운반자와만 반대여야 한다."""
    pos, neg = [], []
    for key, v in P.items():
        i, j = key.split("-")
        r = {roles[i], roles[j]}
        if r == {"suppressor", "carrier"}:
            pos.append(v)
        elif r == {"suppressor", "benign"}:
            neg.append(v)
    return pos, neg


def split_pairs(P, roles):
    pos, neg = [], []
    for key, v in P.items():
        i, j = key.split("-")
        r = {roles[i], roles[j]}
        if r == {"suppressor", "carrier"}:
            pos.append(v)
        elif "suppressor" in r or r == {"carrier", "benign"} or r == {"benign"}:
            neg.append(v)
        # carrier-carrier 쌍은 양쪽 어디에도 안 넣는다 (공모자끼리 닮음은 다른 질문)
    return pos, neg


def ms(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default=None)
    ap.add_argument("--dir", default="logs/analysis")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dir}/hedge_*.json"))
    runs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if a.cond and not (r.get("cond") == a.cond or re.match(rf"^bd_{re.escape(a.cond)}_s\d+$", r["name"])):
            continue
        runs.append(r)
    if not runs:
        raise SystemExit("hedge_*.json 없음")
    # 조건별로 묶기
    by_cond = {}
    for r in runs:
        c = r.get("cond") or re.sub(r"_s\d+$", "", r["name"])
        by_cond.setdefault(c, []).append(r)
    keys = ["suppressor-carrier", "suppressor-benign", "carrier-carrier", "benign-benign", "carrier-benign"]
    L = ["# 헤지 파일럿 요약", "",
         "판정: 위장자–운반자 쌍이 나머지 쌍보다 뚜렷이 반상관인가. AUROC 0.5 = 구분 못 함, ≥ 0.9 = 뚜렷. 쌍 코사인은 라운드 평균, 값은 시드 평균 ± 표준편차.", "",
         "## 1. 역할 무리별 쌍 코사인", "",
         "| 조건 | n | 판 | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    for c, rs in by_cond.items():
        for label, field in (("전체 매개변수", "group_full"), ("공통 성분 제거", "group_centered")):
            L.append(f"| {c} | {len(rs)} | {label} | " + " | ".join(ms([r[field].get(k) for r in rs]) for k in keys) + " |")
        layers = list(rs[0]["group_by_layer"].keys())
        for Lname in layers:
            L.append(f"| {c} | {len(rs)} | 층 {Lname} | " + " | ".join(ms([r["group_by_layer"][Lname].get(k) for r in rs]) for k in keys) + " |")
    L += ["", "## 2. 분리 AUROC (위장자–운반자 쌍 vs 나머지 쌍; 낮은 코사인일수록 양성). 괄호 = 짝 특이도 (위장자–운반자 vs 위장자–정상만)", "",
          "| 조건 | n | 전체 매개변수 | 공통 성분 제거 | 누적 기여 | " + " | ".join(f"층 {Ln}" for Ln in layers) + " |",
          "|---|---|---|---|---|" + "---|" * len(layers)]
    verdict = {}
    for c, rs in by_cond.items():
        row = []
        for field in ("pair_cos_full", "pair_cos_centered", "pair_cos_cumulative"):
            vals, spec = [], []
            for r in rs:
                roles = {k: v for k, v in r["roles"].items()}
                pos, neg = split_pairs(r[field], roles)
                vals.append(auroc(pos, neg))
                p2, n2 = split_spec(r[field], roles)
                spec.append(auroc(p2, n2))
            row.append(f"{ms(vals)} ({ms(spec)})")
        lay = []
        for Ln in layers:
            vals = []
            for r in rs:
                pos, neg = split_pairs(r["pair_cos_by_layer"][Ln], r["roles"])
                vals.append(auroc(pos, neg))
            lay.append(ms(vals))
        L.append(f"| {c} | {len(rs)} | " + " | ".join(row) + " | " + " | ".join(lay) + " |")
        full = [auroc(*split_pairs(r["pair_cos_full"], r["roles"])) for r in rs]
        cent = [auroc(*split_pairs(r["pair_cos_centered"], r["roles"])) for r in rs]
        spec = [auroc(*split_spec(r["pair_cos_centered"], r["roles"])) for r in rs]
        f_ = float(np.mean([x for x in full if x is not None] or [0]))
        c_ = float(np.mean([x for x in cent if x is not None] or [0]))
        s_ = float(np.mean([x for x in spec if x is not None] or [0]))
        verdict[c] = {"full": round(f_, 3), "centered": round(c_, 3), "specificity": round(s_, 3), "score": round(max(f_, min(c_, s_)), 3)}
    L += ["", "## 3. 총노출 / 순노출과 위장자의 총노출 몫", "", "| 조건 | n | gross/net (마지막 라운드) | 위장자 몫 | 운반자 몫 평균 | 정상 몫 평균 |", "|---|---|---|---|---|---|"]
    for c, rs in by_cond.items():
        def shares(role):
            out = []
            for r in rs:
                js = [j for j, ro in r["roles"].items() if ro == role]
                out.append(float(np.mean([r["share_mean"][j] for j in js])) if js else None)
            return out
        L.append(f"| {c} | {len(rs)} | {ms([r['gross_over_net'][-1] for r in rs])} | {ms(shares('suppressor'))} | {ms(shares('carrier'))} | {ms(shares('benign'))} |")
    L += ["", "## 4. 판정", ""]
    for c, v in verdict.items():
        best = v["score"]
        L.append(f"- {c}: 전체 {v['full']} / 중심화 {v['centered']} (짝 특이도 {v['specificity']}) → 점수 max(전체, min(중심화, 특이도)) = {best} → "
                 + ("**뚜렷이 갈림** (≥ 0.9)" if best >= 0.9 else ("약함 (0.7–0.9)" if best >= 0.7 else "**갈리지 않음** (< 0.7)")))
    L.append("")
    L.append("해석: 전체 매개변수 판 AUROC 가 0.5 근처면 위장자의 갱신은 운반자와 반대가 아니다 (같은 방향의 학습 갱신). 공통 성분 제거 판이 높아도 짝 특이도가 낮으면 '위장자가 모두와 반대' 일 뿐 짝이 아니다. 그래서 점수는 max(전체, min(중심화, 특이도)).")
    out = f"{a.dir}/hedge_summary{('_' + a.cond) if a.cond else ''}.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
