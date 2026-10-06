# -*- coding: utf-8 -*-
"""
detection_eval.py — 탐지 확률을 논문 지표로 집계.
logs/pipeline_{name}.json (pipeline_detect_purify.py 가 생성) 들을 모아:
  - 클라 단위 혼동행렬: malicious = atk ∪ {req} 추정 vs 정답 -> TPR/FPR/FNR, precision, DACC
  - 타깃 클래스 식별 정확도, 요청자 식별 정확도
  - 의심 점수(타깃행 크기 mag_on_target)의 AUROC(malicious vs benign)
여러 seed 평균±표준편차. --benign 이름들은 공격 없는 실행으로 간주해 '오경보'만 본다.
"""
import os, sys, json, argparse, glob
import numpy as np
FUBA=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap=argparse.ArgumentParser()
ap.add_argument("--names",nargs="+",required=True)
ap.add_argument("--atk",type=int,nargs="+",default=[0,1,2,3]); ap.add_argument("--req",type=int,default=4)
ap.add_argument("--K",type=int,default=8); ap.add_argument("--benign",nargs="*",default=[])
ap.add_argument("--out",default="eval_logs/detection_eval.json")
a=ap.parse_args(); os.makedirs(os.path.dirname(a.out),exist_ok=True)
GT_MAL=set(a.atk)|{a.req}; GT_BEN=set(range(a.K))-GT_MAL

def auroc(scores, labels):  # labels 1=malicious
    order=np.argsort(-np.array(scores)); lab=np.array(labels)[order]
    P=lab.sum(); N=len(lab)-P
    if P==0 or N==0: return float('nan')
    tp=fp=0; prev=0; area=0; # ROC 적분(사다리꼴)
    tpr_prev=fpr_prev=0.0
    for l in lab:
        if l==1: tp+=1
        else: fp+=1
        tpr=tp/P; fpr=fp/N
        area+=(fpr-fpr_prev)*(tpr+tpr_prev)/2; tpr_prev,fpr_prev=tpr,fpr
    return area

def load(name): return json.load(open(os.path.join(FUBA,"logs",f"pipeline_{name}.json")))["detection"]

# 공격 실행 집계
tpr=[];fpr=[];fnr=[];prec=[];dacc=[];tcorr=[];rcorr=[];aurocs=[]
for n in a.names:
    d=load(n); pred=set(d["A_hat"])|({d["req_hat"]} if d["req_hat"]>=0 else set())
    TP=len(pred&GT_MAL); FP=len(pred&GT_BEN); FN=len(GT_MAL-pred); TN=len(GT_BEN-pred)
    tpr.append(TP/max(len(GT_MAL),1)); fpr.append(FP/max(len(GT_BEN),1)); fnr.append(FN/max(len(GT_MAL),1))
    prec.append(TP/max(TP+FP,1)); dacc.append((TP+TN)/a.K)
    tcorr.append(1 if d["target_correct"] else 0); rcorr.append(1 if d["req_correct"] else 0)
    mg=d.get("mag_on_target",{})
    if mg:
        sc=[mg[str(k)] for k in range(a.K)]; lab=[1 if k in GT_MAL else 0 for k in range(a.K)]
        aurocs.append(auroc(sc,lab))
def ms(x):
    x=[v for v in x if v==v];
    return (round(float(np.mean(x)),3),round(float(np.std(x)),3)) if x else (float('nan'),0)
res={"n_runs":len(a.names),
     "client_TPR":ms(tpr),"client_FPR":ms(fpr),"client_FNR":ms(fnr),"precision":ms(prec),"DACC":ms(dacc),
     "target_acc":ms(tcorr),"requester_acc":ms(rcorr),"AUROC_mag":ms(aurocs)}
print(f"==== 탐지 지표 ({len(a.names)} seeds) ====")
for k in ["client_TPR","client_FPR","client_FNR","precision","DACC","target_acc","requester_acc","AUROC_mag"]:
    print(f"  {k:16s} {res[k][0]}±{res[k][1]}")

# 정상 전용 대조군: 아무것도 플래그하면 안 됨
if a.benign:
    fa=[]
    for n in a.benign:
        d=load(n); pred=set(d["A_hat"])|({d["req_hat"]} if d["req_hat"]>=0 else set())
        fa.append(len(pred)/a.K)  # 전원 정상이므로 플래그는 전부 오경보
    res["benign_false_alarm_rate"]=ms(fa)
    print(f"  [정상 대조군] 오경보율 {res['benign_false_alarm_rate'][0]}±{res['benign_false_alarm_rate'][1]} (낮을수록 좋음)")
json.dump(res,open(a.out,"w"),indent=2,ensure_ascii=False); print("saved",a.out)
