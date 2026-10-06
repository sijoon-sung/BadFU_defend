# -*- coding: utf-8 -*-
"""
실제 FUBA 실행(main.py --save)이 저장한 체크포인트로 쌍 검사 통계량을 계산한다.
  local_net_model_{name}_round_{r}.pkl : 그 라운드 클라별 로컬 모델(list of state_dict)
  global_net_model_{name}_round_{r}.pkl: 그 라운드 집계 후 글로벌
클라 k 의 라운드 r 업데이트 = local(r)[k] - global(r-1).

FUBA 역할(train_main.py): backdoor_clients = 0..4, defender = forgot_client(=4).
  -> 0~3 = Adv-attacker, 4 = Adv-defender(요청자), 5~ = 정상.

통계량 (클래스 행 c 마다):
  O_uk(c) = [-cos(dW_u[c], dW_k[c])]_+ * min(|dW_u[c]|, |dW_k[c]|)   (같은 행, 둘 다 큼, 반대)
  O_uk = max_c O_uk(c),  chat = argmax_c
"""
import os, glob, pickle, argparse
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="det1")
ap.add_argument("--dir", default="checkpoints")
ap.add_argument("--warm_up", type=int, default=3)
ap.add_argument("--target", type=int, default=8)
ap.add_argument("--defender", type=int, default=4)
ap.add_argument("--n_attack", type=int, default=4)
args = ap.parse_args()

def load(p):
    with open(p, "rb") as f: return pickle.load(f)
def rnd(p): return int(p.split("_round_")[-1].split(".")[0])

G = {rnd(p): load(p) for p in glob.glob(f"{args.dir}/global_net_model_{args.name}_round_*.pkl")}
L = {rnd(p): load(p) for p in glob.glob(f"{args.dir}/local_net_model_{args.name}_round