# -*- coding: utf-8 -*-
"""정상 클라이언트의 이탈이 잠복 백도어를 드러내는가 — 실제 FU 위에서 검증 (go/no-go 실험).

배경
  BadFU Table V, FUBA Table V 는 정상 클라이언트를 지워도 ASR 이 오르지 않는다고 보고한다(Dirichlet α=0.1 포함).
  우리 예비 관찰(fp3/fp4, 1차 근사 빼기)은 한 클래스를 독점한 정상 클라이언트를 지우면 ASR 5.8→38 로 올랐다.
  둘의 차이가 (a) 분할 조건(독점 vs Dirichlet) 때문인지 (b) 1차 근사 vs 실제 FU 때문인지 가른다.

같은 궤적에서 요청자만 바꿔 가며(공격이 설계한 요청자 / 독점 정상 / 보통 정상) 여러 FU 로 지우고,
학습 때의 트리거 생성기(저장된 backdoor_net 마지막 라운드)로 ACC/ASR 을 잰다.

FU 방법 (--methods)
  subtract        1차 근사: theta_T - U_k (diff_audit.trajectory 와 같은 식). 재학습 없음.
  distillation    KD-FU(Wu et al.): 저장 기여 빼기 + 증류. 비재학습형(공격자 불참).
  fedEraser       원 논문식 FedEraser 보정 (unlearn_method/federaser_faithful.py). 남은 클라이언트 정직 학습.
  retrain_benign  요청자 제외 처음부터 재학습, 남은 클라이언트 전원 정직(공격자도 정상 학습) = 순수 "제거" 효과.
  retrain_attack  FUBA 원본: 재학습 중 공격자가 전력으로 재주입(논문 각주 4, γ=1). 제거+재주입 합산.
  fedEraser_fuba  저장소 원본 래퍼(= 공격자 참여 재학습). 참고용.

실행 예
  python benign_departure.py --name bd_owner_s0 --K 8 --seed 522 --split_seed 522 \
      --non_iid --non_iid_type class_owner --owner_class 3 --requesters 4 7 5
출력: logs/benign_departure/{name}.json  (요청자 x 방법별 acc / asr / 클래스별 asr)
"""
import argparse
import copy
import json
import os
import pickle
import random
import sys
import time
from argparse import Namespace

import numpy as np
import torch
import torchvision.transforms as transforms

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from config import Config  # noqa: E402
from dataset import MNIST  # noqa: E402
from model import MNISTAutoencoder, Net  # noqa: E402
from utils.comm_utils import attack, make_client_loaders, make_client_split, test  # noqa: E402

METHODS = ["subtract", "distillation", "fedEraser", "retrain_benign", "retrain_attack", "fedEraser_fuba"]
DEFAULT_METHODS = ["subtract", "distillation", "fedEraser", "retrain_benign", "retrain_attack"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="checkpoints/*_{name}_round_*.pkl 의 name")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8, help="클라이언트 수")
    p.add_argument("--requesters", type=int, nargs="+", default=[4, 7, 5],
                   help="지워 볼 클라이언트. 기본: 4=Adv-defender(공격 설계), 7=독점 정상(class_owner), 5=보통 정상")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=METHODS)
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--defender", type=int, default=4, help="학습 때의 forgot_client_idx(Adv-defender 역할)")
    p.add_argument("--attackers", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--split_seed", type=int, default=1223, help="학습 때와 같은 값이어야 분할이 일치한다")
    p.add_argument("--non_iid", action="store_true")
    p.add_argument("--non_iid_type", default=None, choices=["lognormal", "Dirichlet", "concept_shift", "class_owner"])
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--sigma", type=float, default=1)
    p.add_argument("--shift_ratio", type=float, default=0.4)
    p.add_argument("--owner_client", type=int, default=-1)
    p.add_argument("--owner_class", type=int, default=3)
    p.add_argument("--owner_frac", type=float, default=0.9)
    p.add_argument("--train_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--noise_thread", type=float, default=0.04)
    p.add_argument("--atk_eps", type=float, default=1.0)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true", help="이미 있는 (방법, 요청자) 결과는 건너뛴다")
    p.add_argument("--n_gpu", type=int, default=1)
    return p.parse_args()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_config(a):
    """unlearn.py 의 argparse 기본값과 같은 Namespace 로 Config 를 만든다 (MPI 없이 단일 프로세스)."""
    ns = Namespace(
        dataset="mnist", name=a.name, attackMethod="iba", target_label=a.target,
        seed=a.seed, split_seed=a.split_seed,
        nb_attack=len(a.attackers), nb_defence=1, forgot_client_idx=a.defender,
        noise_thread=a.noise_thread, atk_eps=a.atk_eps,
        nb_clients=a.K, communication_rounds=a.rounds, participant_rate=1.0, warm_up=3,
        train_epoch=a.train_epoch, batch_size=a.batch_size, test_batch_size=256,
        non_iid=a.non_iid, non_iid_type=a.non_iid_type, shift_ratio=a.shift_ratio,
        alpha=a.alpha, sigma=a.sigma, owner_client=a.owner_client, owner_class=a.owner_class, owner_frac=a.owner_frac,
        n_gpu=a.n_gpu, data_path=HERE, save=False, no_detail_test=False, no_global_test=False,
        gaussian_noise=False, geometric_median=False, o_distance_filter=False, c_distance_filter=False,
        flame=False, multi_krum=False, median=False, AlignIns=False, Fuba=False, Indicator=False, IBMFL=False,
        flDetector=False, Enforce=False, Topk=False,
        unlearn_target=a.defender, method="retrain", round=a.rounds, save_dir=a.ckpt_dir,
        fu_attacker_mode="attack", iba_reopt=False,
    )
    return Config(ns, 1, 1)


def load_traj(a):
    g, l = [], []
    for r in range(1, a.rounds + 1):
        with open(f"{a.ckpt_dir}/global_net_model_{a.name}_round_{r}.pkl", "rb") as f:
            g.append(pickle.load(f))
        with open(f"{a.ckpt_dir}/local_net_model_{a.name}_round_{r}.pkl", "rb") as f:
            l.append(pickle.load(f))
    with open(f"{a.ckpt_dir}/backdoor_net_model_{a.name}_round_{a.rounds}.pkl", "rb") as f:
        bd = pickle.load(f)
    return g, l, bd


def asr_by_class(net, bd, loader, device, a):
    """원래 클래스별 ASR(타깃 클래스 제외). 어느 클래스가 열렸는지 보기 위함."""
    hit = {}
    tot = {}
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            noise = torch.clamp(bd(x) * a.atk_eps, -a.noise_thread, a.noise_thread)
            pred = net(torch.clamp(x + noise, -1.0, 1.0)).argmax(1)
            for c in y.unique().tolist():
                if c == a.target:
                    continue
                m = y == c
                tot[c] = tot.get(c, 0) + int(m.sum())
                hit[c] = hit.get(c, 0) + int((pred[m] == a.target).sum())
    return {str(c): round(100.0 * hit[c] / tot[c], 2) for c in sorted(tot)}


def evaluate(sd, bd, testloader, device, a):
    net = Net().to(device)
    net.load_state_dict({k: v.to(device) for k, v in sd.items()})
    net.eval()
    acc = test(net, testloader, device)
    asr = attack(net, testloader, bd, a.atk_eps, a.noise_thread, a.target, device)
    return {"acc": round(acc, 2), "asr": round(asr, 2), "asr_by_class": asr_by_class(net, bd, testloader, device, a)}


def run_method(method, k, a, cfg, g, l, bd, trainset, testloader):
    """요청자 k 를 method 로 지운 state_dict(와 부가 정보)를 돌려준다."""
    seed_all(a.seed)
    extra = {}
    if method == "subtract":
        from diff_audit.trajectory import FubaTrajectory
        traj = FubaTrajectory(a.name, a.rounds, a.K, ckpt_dir=a.ckpt_dir)
        theta, U = traj.contributions()
        sd = traj.space.to_model(theta - U[k], cfg.device).state_dict()
    elif method == "distillation":
        from unlearn_method.distillation import distillation_unlearn
        cfg.unlearn_target = k
        cfg.backdoor_model = copy.deepcopy(bd)
        cfg.iba_reopt = False      # 생성기 재최적화는 평가가 아니라 더 강한 공격자이므로 끈다
        sd = distillation_unlearn(trainset, g, l, k, cfg)
    elif method == "fedEraser":
        from unlearn_method.federaser_faithful import fedEraser_faithful
        loaders = make_client_loaders(make_client_split(trainset, cfg), cfg)
        sd = fedEraser_faithful(g, l, loaders, k, cfg)
    elif method in ("retrain_benign", "retrain_attack"):
        from unlearn_method.retrain import retrain
        mode = "benign" if method == "retrain_benign" else "attack"
        cfg.unlearn_target = k
        cfg.fu_attacker_mode = mode
        cfg.backdoor_model = copy.deepcopy(bd)
        name0 = cfg.name
        cfg.name = f"{a.name}__req{k}_{mode}"       # retrain 내부 save_models 이름 충돌 방지
        try:
            net, regen = retrain(cfg)
        finally:
            cfg.name = name0
        sd = {kk: v.detach().cpu() for kk, v in net.state_dict().items()}
        if mode == "attack" and regen is not None:
            # 재학습 중 공격자가 새로 학습한 생성기로 잰 ASR (원본 FUBA 가 보고하는 수치에 해당)
            extra["asr_regen"] = round(attack(net, testloader, regen, a.atk_eps, a.noise_thread, a.target, cfg.device), 2)
    elif method == "fedEraser_fuba":
        from unlearn_method.FedEraser.Fed_Unlearn_base import unlearning
        from unlearn_method.utils import Arguments
        cfg.unlearn_target = k
        cfg.backdoor_model = copy.deepcopy(bd)
        loaders = make_client_loaders(make_client_split(trainset, cfg), cfg)
        res = unlearning(copy.deepcopy(g), copy.deepcopy(l), loaders, testloader, Arguments(cfg), cfg)
        last = res[-1]
        if isinstance(last, tuple):
            last = last[0]
        sd = last.state_dict() if hasattr(last, "state_dict") else last
        sd = {kk: v.detach().cpu() for kk, v in sd.items()}
    else:
        raise ValueError(method)
    return sd, extra


def main():
    a = parse()
    out = a.out or f"logs/benign_departure/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cfg = make_config(a)
    device = cfg.device

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    cfg.trainset, cfg.testset, cfg.transform = trainset, testset, transform
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    g, l, bd_dict = load_traj(a)
    bd = MNISTAutoencoder().to(device)
    bd.load_state_dict(bd_dict)
    bd.eval()
    cfg.global_nets, cfg.local_nets = g, l

    owner = (a.owner_client % a.K) if a.non_iid_type == "class_owner" else None
    res = {"name": a.name, "args": vars(a),
           "roles": {"attackers": a.attackers, "defender": a.defender, "owner": owner,
                     "benign": [i for i in range(a.K) if i not in a.attackers and i != a.defender]},
           "before": None, "results": {}}
    if a.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            res = json.load(f)

    if res["before"] is None:
        res["before"] = evaluate(g[-1], bd, testloader, device, a)
    print(f"[{a.name}] before FU: ACC {res['before']['acc']} ASR {res['before']['asr']}", flush=True)

    for method in a.methods:
        res["results"].setdefault(method, {})
        for k in a.requesters:
            if str(k) in res["results"][method]:
                continue
            t0 = time.time()
            sd, extra = run_method(method, k, a, cfg, g, l, bd, trainset, testloader)
            ev = evaluate(sd, bd, testloader, device, a)
            ev.update(extra)
            ev["seconds"] = round(time.time() - t0, 1)
            res["results"][method][str(k)] = ev
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"[{a.name}] {method:15s} req={k}: ACC {ev['acc']:6.2f} ASR {ev['asr']:6.2f} "
                  f"{('(regen ' + str(ev['asr_regen']) + ')') if 'asr_regen' in ev else ''} {ev['seconds']}s", flush=True)

    # 요약표
    print(f"\n== {a.name}  before: ACC {res['before']['acc']} / ASR {res['before']['asr']}   (ASR, 괄호 ACC)")
    hdr = "method".ljust(16) + "".join(f"req{k}".rjust(16) for k in a.requesters)
    print(hdr)
    for method in a.methods:
        row = method.ljust(16)
        for k in a.requesters:
            e = res["results"][method].get(str(k))
            row += (f"{e['asr']:6.1f} ({e['acc']:5.1f})" if e else "-").rjust(16)
        print(row)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
