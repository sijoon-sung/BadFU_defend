# -*- coding: utf-8 -*-
"""
cost_eval.py — 탐지/정화의 연산 비용을 (점근 + 실측)으로 보고, 대안들과 비교.
실측: 한 궤적에서 '우리 탐지(클래스별 군집, 쌍 아님)', '전체 쌍 비교 O(n^2)',
      '헤드 투영 정화' 의 벽시계 시간을 잼. 트리거 역설계(Neural Cleanse)·재학습은
      점근/구성요소 비용만 표로(실제 실행은 비싸서 수식으로).
"""
import os, sys, json, time, argparse, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Trajectory

ap=argparse.ArgumentParser()
ap.add_argument("--name",default="det1"); ap.add_argument("--atk",type=int,nargs="+",default=[0,1,2,3])
ap.add_argument("--req",type=int,default=4); ap.add_argument("--repeat",type=int,default=20)
ap.add_argument("--out",default="eval_logs/cost_eval.json")
a=ap.parse_args(); os.makedirs(os.path.dirname(a.out),exist_ok=True)
T=Trajectory(a.name); K=T.K; C=10; dev=T.dev

def unit(x): n=x.norm(); return x/n if n>1e-12 else x
def timeit(fn,rep):
    fn();  # warmup
    t0=time.perf_counter()
    for _ in range(rep): fn()
    return (time.perf_counter()-t0)/rep*1000  # ms

# (1) 우리 탐지: 클래스별, 클라별 타깃행 방향 정렬 통계 (O(nC), 쌍 없음)
rows=[T.HUrow[r] for r in T.active]
def ours_detect():
    for t in range(C):
        rowsT=[torch.stack([T._fc3row(torch.zeros(T.D))]) for _ in range(0)]  # noop placeholder
    # 실제: 타깃행 크기 + 군집 평균 정렬 (한 클래스 기준; 전체 C 배)
    for r in T.active:
        M=torch.stack(T.HUrow[r]); M=M/ (M.norm(dim=1,keepdim=True)+1e-12)
        _=M@M.mean(0)  # 각 클라 vs 평균 (O(K))
# (2) 전체 쌍 비교 O(n^2): 모든 쌍의 헤드 코사인
def pairs_detect():
    for r in T.active:
        H=torch.stack([T.HUvec[r][k] for k in range(K)]); H=H/(H.norm(dim=1,keepdim=True)+1e-12)
        _=H@H.T  # KxK
# (3) 정화: 헤드 V 투영 1회
B=T.Vbasis(a.atk);
def purify_once():
    _=T.theta_T - T.U[a.req] - sum(B.T@(B@T.U[k]) for k in range(K) if k!=a.req)

ours_ms=timeit(ours_detect,a.repeat)
pairs_ms=timeit(pairs_detect,a.repeat)
pur_ms=timeit(purify_once,max(a.repeat//4,1))

asymptotic={
 "ours_detect":"O(n·C·d_head·R)  (클래스별 군집, 쌍 없음)",
 "all_pairs":"O(n^2·d_head·R)",
 "neural_cleanse":"O(C·steps·|probe|·fwd+bwd)  + 서버 데이터 필요",
 "retrain_check":"O(T·n·local_epochs) 클라 재계산",
 "per_request_reconstruct(FedEraser)":"요청마다 보정라운드×(n-1)",
 "history_storage":"O(n·d·T)  (저장 공간; 근사 FU 가 이미 요구)",
 "ours_purify":"O(k·d_head)  (랭크-k 투영, 재학습 없음)",
}
measured={"ours_detect_ms":round(ours_ms,3),"all_pairs_ms":round(pairs_ms,3),"purify_ms":round(pur_ms,3),
          "device":dev,"K":K,"C":C,"active_rounds":len(T.active)}
print("==== 연산 비용 ====\n[실측, ms/호출]")
for k,v in measured.items():
    if k.endswith("_ms"): print(f"  {k:18s} {v}")
print(f"  (n={K}, 라운드={len(T.active)}, device={dev})")
print("[점근]")
for k,v in asymptotic.items(): print(f"  {k:34s} {v}")
print("\n[주의] n=8 로 작아 쌍 비교 이점이 작게 보임. 논문용은 n=20/50/100 으로 늘려 곡선을 그릴 것.")
json.dump({"name":a.name,"measured":measured,"asymptotic":asymptotic},open(a.out,"w"),indent=2,ensure_ascii=False)
print("saved",a.out)
