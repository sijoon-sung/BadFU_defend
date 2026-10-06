# -*- coding: utf-8 -*-
"""
detector_error_sweep.py  — 논문 핵심 실험.
탐지가 완벽하면 '공격자 통째 제거'가 최적이다. 우리 방법(방향 V만 제거)의 가치는
탐지가 틀릴 때 드러난다. 그래서 탐지 결과에 오류(FP/FN)를 비율 e 로 주입하고,
두 정화의 성능(ACC/ASR)이 e 에 따라 어떻게 무너지는지 비교한다. (FedRecover Fig.8 방식)

각 e 에서 malicious 추정집합을 교란:
  FN: 진짜 공격자를 확률 e 로 놓침   /   FP: 정상 클라를 확률 e 로 공격자로 오인
두 정화:
  full : 추정 공격자 통째 제거   theta_T - U_req - sum_{k in est} U_k
  V    : 추정 공격자 방향만 제거  theta_T - U_req - sum_{k != req} P_V(est) U_k   (V 는 추정집합에서 추정)
reps 회 랜덤 교란 평균. 요청자(req)는 항상 전부 제거(삭제 권리).
"""
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Trajectory

ap=argparse.ArgumentParser()
ap.add_argument("--names",nargs="+",required=True)
ap.add_argument("--atk",type=int,nargs="+",default=[0,1,2,3]); ap.add_argument("--req",type=int,default=4)
ap.add_argument("--normals",type=int,nargs="+",default=[5,6,7])
ap.add_argument("--errs",type=float,nargs="+",default=[0.0,0.1,0.2,0.3,0.4,0.5])
ap.add_argument("--reps",type=int,default=5); ap.add_argument("--kV",type=int,default=8); ap.add_argument("--seed",type=int,default=0)
ap.add_argument("--out",default="eval_logs/detector_error_sweep.json")
a=ap.parse_args(); rng=np.random.default_rng(a.seed)
os.makedirs(os.path.dirname(a.out),exist_ok=True)

def corrupt(est_true, normals, e):
    keep=[k for k in est_true if rng.random()>e]          # FN: 공격자 놓침
    add=[k for k in normals if rng.random()<e]            # FP: 정상 오인
    return sorted(set(keep)|set(add))

agg={}  # err -> {full:[...], V:[...]}
for name in a.names:
    T=Trajectory(name)
    for e in a.errs:
        for _ in range(a.reps):
            est=corrupt(a.atk, a.normals, e)
            if not est: est=[a.atk[0]]  # 공집합 방지
            # full
            th=T.theta_T - T.U[a.req] - sum(T.U[k] for k in est)
            acc_f,asr_f=T.eval(th)
            # V (추정집합으로 V 추정)
            B=T.Vbasis(est, a.kV); projV=lambda v: B.T@(B@v)
            th=T.theta_T - T.U[a.req] - sum(projV(T.U[k]) for k in range(T.K) if k!=a.req)
            acc_v,asr_v=T.eval(th)
            d=agg.setdefault(round(e,2),{"full_acc":[],"full_asr":[],"V_acc":[],"V_asr":[]})
            d["full_acc"].append(acc_f); d["full_asr"].append(asr_f); d["V_acc"].append(acc_v); d["V_asr"].append(asr_v)

def ms(x): x=np.array(x,float); return (round(x.mean(),2),round(x.std(),2))
print(f"==== 탐지 오류율 스윕 ({len(a.names)} runs × {a.reps} reps) ====")
print(f"  {'err':>5s} | {'full ACC':>12s} {'full ASR':>12s} | {'V ACC':>12s} {'V ASR':>12s}")
rows={}
for e in sorted(agg):
    d=agg[e]; fa=ms(d["full_acc"]); fs=ms(d["full_asr"]); va=ms(d["V_acc"]); vs=ms(d["V_asr"])
    rows[e]={"full_acc":fa,"full_asr":fs,"V_acc":va,"V_asr":vs}
    print(f"  {e:5.2f} | {fa[0]:6.2f}±{fa[1]:<4.2f} {fs[0]:6.2f}±{fs[1]:<4.2f} | {va[0]:6.2f}±{va[1]:<4.2f} {vs[0]:6.2f}±{vs[1]:<4.2f}")
print("\n[읽는 법] e=0 에서는 full 이 최적(ASR~0). e 가 오르면 full 은 정상 클라를 통째로 지워 ACC 급락 +")
print("          공격자를 놓쳐 ASR 반등. V 는 더 완만하게 무너지면 '탐지 불완전 하에서의 이점'이 성립.")
json.dump({"names":a.names,"atk":a.atk,"req":a.req,"reps":a.reps,"rows":{str(k):v for k,v in rows.items()}},
          open(a.out,"w"),indent=2,ensure_ascii=False)
print("saved",a.out)
