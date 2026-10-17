# -*- coding: utf-8 -*-
"""cancel_fu.py 결과를 조건별로 모은다.  python aggregate_cancel.py --cond iid  → logs/cancel/summary_{cond}.md
요청자별: 1 단계 짝(저장 기여)·통계 p/z, 빼 보기 z, 2' 단계 짝(활성화 위)·정렬, 결정, ASR/ACC/α/자기 데이터, 비용."""
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
    p.add_argument("--dir", default="logs/cancel")
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
    L = [f"# DOSE+TRACE (cancel_fu) 요약: {a.cond or 'all'}  n={len(runs)}", "",
         f"삭제 전 ASR {ms([r['before'].get('asr') for r in runs])} / ACC {ms([r['before'].get('acc') for r in runs])}", "",
         "## 결정과 채점 (DOSE 띠 → TRACE → DEFUSE)", "",
         "| 요청자 | 역할 | 결정 (시드별) | DOSE z (최대) | ASR | ACC | 유지 변화 | α | 자기 데이터 acc | 순전파 짝 / 빼보기 / 모델 / 초 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in reqs:
        e = [r["results"][k] for r in runs if k in r["results"]]
        dec = "; ".join(x["decision"] for x in e)
        zmax = [max((s["z"] for s in x["confirm"]["trace"] if s.get("z") is not None), default=None) for x in e]
        L.append(f"| {k} | {e[0]['role']} | {dec} | {ms(zmax)} | {ms([x['final'].get('asr') for x in e])} | {ms([x['final']['acc'] for x in e])} | "
                 f"{ms([x['final']['retain_change'] for x in e], 2)} | {ms([x['final'].get('alpha') for x in e], 3)} | {ms([x['final'].get('forget_acc') for x in e])} | "
                 f"{ms([x['cost']['attr_forwards'] for x in e], 0)} / {ms([x['cost']['forwards'] for x in e], 0)} / {ms([x['cost']['pop_models'] for x in e], 0)} / {ms([x['cost']['seconds'] for x in e])} |")
    L += ["", "## TRACE 회전 1: 저장된 기여 위 (θ_T, ω⁰ = 공개 ∪ 지울 데이터)", "",
          "| 요청자 | 역할 | 짝 (역할, 시드별) | 짝 정밀도 / 재현율 | 상쇄 비율 | 타깃 (몰림) | 크기 | 반대 정렬 | z: 크기 / 몰림 / 상쇄 / 반대 | Fisher p | 회전 1 경보 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in reqs:
        e = [r["results"][k]["attr"] for r in runs if k in r["results"]]
        sets = "; ".join(f"{x['set']} {[s[:3] for s in x['cancel']['set_roles']]}" for x in e)
        tgt = "; ".join(f"{x['c_star']} ({x['conc']['score']:.2f})" for x in e)
        z = lambda key: ms([x["z"].get(key) for x in e], 1)
        L.append(f"| {k} | {runs[0]['results'][k]['role']} | {sets} | {ms([x['set_precision'] for x in e], 2)} / {ms([x['set_recall'] for x in e], 2)} | "
                 f"{ms([x['cancel']['ratio'] for x in e], 2)} | {tgt} | {ms([x['mag'] for x in e], 2)} | {ms([x['anti'] for x in e], 2)} | "
                 f"{z('mag')} / {z('conc')} / {z('cancel')} / {z('anti')} | {ms([x['fisher_p'] for x in e], 3)} | {sum(1 for x in e if x['flagged'])}/{len(e)} |")
    L += ["", "## TRACE 회전 2+ 와 DEFUSE: 활성화 입력 위 (θ' = θ_T − ρU_k, DOSE 경보가 난 요청만)", "",
          "| 요청자 | 역할 | 짝 (역할, 시드별) | 짝 정밀도 / 재현율 | 정렬 (운반자 / 정상, 시드 평균) | 상쇄 비율 | 회전 1 과 일치 | 클래스 일치 | DEFUSE λ | 방법 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in reqs:
        rows = []
        for r in runs:
            x = r["results"].get(k)
            if not x:
                continue
            roles = r["roles"]
            for pq in x.get("purify", []):
                p2 = pq.get("pair_act")
                if p2:
                    al = p2.get("align", {})
                    car = [v for j, v in al.items() if roles[str(j)] == "carrier"]
                    ben = [v for j, v in al.items() if roles[str(j)] != "carrier"]
                    rows.append((p2, pq, car, ben))
        if not rows:
            L.append(f"| {k} | {runs[0]['results'][k]['role']} | - | - | - | - | - | - | - | - |")
            continue
        sets = "; ".join(f"{p2['set']} {[s[:3] for s in p2['set_roles']]}" for p2, _, _, _ in rows)
        L.append(f"| {k} | {runs[0]['results'][k]['role']} | {sets} | {ms([p2['set_precision'] for p2, *_ in rows], 2)} / {ms([p2['set_recall'] for p2, *_ in rows], 2)} | "
                 f"{ms([np.mean(c) if c else None for _, _, c, _ in rows], 2)} / {ms([np.mean(b) if b else None for _, _, _, b in rows], 2)} | "
                 f"{ms([p2['ratio'] for p2, *_ in rows], 2)} | {ms([pq['set_agree'] for _, pq, _, _ in rows], 2)} | {sum(1 for _, pq, _, _ in rows if pq['class_match'])}/{len(rows)} | "
                 f"{'; '.join(str(pq.get('chosen_lam')) for _, pq, _, _ in rows)} | {'; '.join(('antidote' if pq.get('chosen_lam') is not None else 'kd') + ('+polish' if pq.get('polish') else '') for _, pq, _, _ in rows)} |")
    L.append("")
    L.append("결정 뜻: accept(confirm)=빼 보기 통과·수락 / accept(attr_suspect)=1 단계 경보지만 빼 보기 통과·수락 / accept(partial)=연산자 한계로 일부만 뺌 / "
             "purified(antidote[+polish])=짝의 상쇄 성분 제거[+직교 증류] / purified(kd)=재억제(대체) / (sim_only)=1 단계는 통과했으나 빼 보기에서 경보 / reject(...)=불출")
    tag = f"_{a.cond}" if a.cond else ""
    with open(f"{a.dir}/summary{tag}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"-> {a.dir}/summary{tag}.md")


if __name__ == "__main__":
    main()
