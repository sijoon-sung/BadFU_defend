# -*- coding: utf-8 -*-
"""
purify_unlearned.py — '진짜 FU 로 언러닝된 모델'에 정화를 적용(근사 아님).

입력:
  --name       공격 실행 이름(예: det1). 탐지 V 와 백도어 생성기(ASR 평가)에 사용.
  --unlearned  진짜 FU 산출물 경로(예: saved_models/global_retrain_iba_mnist.pkl).
               FUBA unlearn.py --method retrain|fedEraser 가 생성한 '요청자 삭제된' 모델.
절차:
  1) 탐지 결과(logs/pipeline_{name}.json)에서 공격자 집합·타깃을 읽어 백도어 방향 V(헤드공간) 추정.
  2) 진짜 언러닝 모델 theta_u 의 '분류 헤드'에서 V 성분만 직교투영으로 제거(재학습 없음).
  3) dormant / 진짜FU활성화 / 정화 의 ACC·ASR 비교 출력.
"""
import os, sys, json, pickle, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Trajectory, Net

ap=argparse.ArgumentParser()
ap.add_argument("--name",required=True)
ap.add_argument("--unlearned",required=True, help="진짜 FU 산출 모델 .pkl (state_dict)")
ap.add_argument("--kV",type=int,default=8)
ap.add_argument("--out",default="eval_logs/purify_unlearned.json")
a=ap.parse_args(); os.makedirs(os.path.dirname(a.out),exist_ok=True)
FUBA=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

T=Trajectory(a.name)
det=json.load(open(os.path.join(FUBA,"logs",f"pipeline_{a.name}.json")))["detection"]
atk=sorted(det["A_hat"]) if det["A_hat"] else [0,1,2,3]; t=det["t_hat"]

# 헤드공간(850=fc3.weight 840 + bias 10) V 기저
Av=[T.HUvec[r][k] for r in T.active for k in atk]
M=torch.stack(Av,0); M=M-M.mean(0,keepdim=True)
Bh=torch.linalg.svd(M,full_matrices=False)[2][:a.kV]      # (k,850) 헤드공간
Bh=torch.linalg.qr(Bh.T)[0].T
sw,ew=T.sl["fc3.weight"]; sb,eb=T.sl["fc3.bias"]
def head_of(flat): return torch.cat([flat[sw:ew], flat[sb:eb]])      # (850,)
def put_head(flat,h):
    out=flat.clone(); out[sw:ew]=h[:840]; out[sb:eb]=h[840:850]; return out

# 진짜 언러닝 모델 로드 -> flat
usd=pickle.load(open(a.unlearned,"rb"))
u_flat=torch.cat([usd[k].float().flatten() for k in T.keys]).to('cpu')

# 정화: 헤드에서 V 제거
h=head_of(u_flat); h_pur=h-Bh.T@(Bh@h); u_pur=put_head(u_flat,h_pur)

acc_d,asr_d=T.eval(T.theta_T)             # dormant (공격 학습 종료 모델)
acc_u,asr_u=T.eval(u_flat)                # 진짜 FU 후(활성화)
acc_p,asr_p=T.eval(u_pur)                 # 정화 후
res={"name":a.name,"unlearned":a.unlearned,"atk":atk,"target":t,
     "dormant":{"acc":acc_d,"asr":asr_d},"real_FU":{"acc":acc_u,"asr":asr_u},"purified":{"acc":acc_p,"asr":asr_p}}
print(f"==== 진짜 FU 모델 정화 ({a.name}) ====")
print(f"  dormant(공격 종료)   ACC={acc_d:6.2f} ASR={asr_d:6.2f}")
print(f"  real FU(활성화)      ACC={acc_u:6.2f} ASR={asr_u:6.2f}")
print(f"  purified(V 제거)     ACC={acc_p:6.2f} ASR={asr_p:6.2f}")
json.dump(res,open(a.out,"w"),indent=2,ensure_ascii=False); print("saved",a.out)
