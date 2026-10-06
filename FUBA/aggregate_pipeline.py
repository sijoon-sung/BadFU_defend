# -*- coding: utf-8 -*-
"""여러 run 의 logs/pipeline_{name}.json 을 모아 평균±표준편차 통계표 생성."""
import json, glob, os, numpy as np, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ap=argparse.ArgumentParser(); ap.add_argument("--names",nargs="+",required=True); a=ap.parse_args()
runs=[json.load(open(f"logs/pipeline_{n}.json")) for n in a.names]
n=len(runs)
def ms(xs): xs=np.array(xs,float); return f"{xs.mean():.2f}±{xs.std():.2f}"

print(f"==== 집계 ({n} runs: {', '.join(a.names)}) ====\n")
print("[탐지 정확도]")
print(f"  타깃 클래스 정확: {sum(r['detection']['target_correct'] for r in runs)}/{n}")
print(f"  공격자군집 precision: {ms([r['detection']['atk_precision'] for r in runs])}  recall: {ms([r['detection']['atk_recall'] for r in runs])}")
print(f"  요청자 정확: {sum(r['detection']['req_correct'] for r in runs)}/{n}")

print("\n[정화]  (요청자 기여는 항상 전부 제거 = 삭제 권리 보장)")
methods=["none_dormant","req_only_activation","req+atk_full","req+allV","req+randV_ctrl","ref_two_deletion"]
label={"none_dormant":"A 잠복(그대로)","req_only_activation":"B 요청자만(=활성화)",
       "req+atk_full":"C +공격자 통째","req+allV":"D +전원 V성분(제안)",
       "req+randV_ctrl":"E +임의방향(대조)","ref_two_deletion":"기준 두-삭제"}
print(f"  {'방법':24s} {'ACC':>12s} {'ASR':>12s}")
for m in methods:
    accs=[r['purification'][m]['acc'] for r in runs]; asrs=[r['purification'][m]['asr'] for r in runs]
    print(f"  {label[m]:24s} {ms(accs):>12s} {ms(asrs):>12s}")
nl=[r['purification']['normal_contrib_loss_frac'] for r in runs]
print(f"\n  정상 클라 기여 손실 비율(D에서): {ms(nl)}")

# 핵심 요약
dA=[r['purification']['req+allV']['asr'] for r in runs]; bA=[r['purification']['req_only_activation']['asr'] for r in runs]
dC=[r['purification']['req+allV']['acc'] for r in runs]; bC=[r['purification']['req_only_activation']['acc'] for r in runs]
print(f"\n[요약] 제안(D) vs 활성화(B):  ASR {ms(bA)} -> {ms(dA)}   ACC {ms(bC)} -> {ms(dC)}")
json.dump({"runs":a.names,"summary":{"D_asr":dA,"B_asr":bA,"D_acc":dC,"B_acc":bC}},
          open("logs/pipeline_aggregate.json","w"),indent=2)
print("saved logs/pipeline_aggregate.json")
