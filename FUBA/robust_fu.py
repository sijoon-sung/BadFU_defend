# -*- coding: utf-8 -*-
"""FedEraser-R: 보정(calibration) 루프 안에서 정의 2 위반을 재고(탐지) 바로잡는다(정화).

정의 2 (쓰는 언러닝 정의)
  요청자의 영향은 지우되, 나머지 모두의 입력에 대한 모델 행동은 바뀌지 않아야 한다.
  삭제로 깨어나는 잠복 백도어는 이 정의의 위반이다(요청자가 아닌 입력에 대한 행동이 한 클래스로 쏠린다).
  그러므로 탐지 = FedEraser 보정 경로를 따라 그 위반을 재는 것, 정화 = 그 정의를 강제하는 것.

탐지 (라운드 r, 트리거 모름)
  재생 모델 h_r = 보정 r 라운드 결과 vs 원 궤적의 같은 라운드 모델 orig_r = global_r (학습 진도가 같으니 차이는 제거 몫).
  (i)  깨끗한 탐침: argmax 가 갈린 비율, 집중도 = 갈린 입력 중 한 클래스 c 로 옮겨 간 비율의 최대(c 기록).
  (ii) 적대적 탐침: 범용 δ (L∞ ≤ eps, 탐침 위 mean KL(h_r(x+δ) || orig_r(x+δ)) 를 PGD 로 최대화) 를 얹고 같은 두 값.
  Score_r = δ 아래 집중도, flagged_r = Score_r > flag_thr.
정화 (arm 별)
  plain          안 함.
  purify_flag    flagged_r 일 때만.
  purify_always  매 라운드.
  purify_clean   매 라운드, 깨끗한 탐침만 (δ 없음).
  정화 = new_gm_r 에 purify_steps 번 SGD: x ∈ 탐침 ∪ (탐침+δ) 에서 KL(h_r(x) || orig_r(x)) 최소화 (교사 orig_r 고정, 가중 purify_lambda).
  "떠난 클라이언트가 맡던 억제 역할을 서버가 자기 라벨 없는 입력과 원 경로를 교사로 삼아 떠안는다."

서버가 아는 것: 라벨 없는 탐침 = MNIST TEST[0:N] 뿐. 공격자·트리거·공격자 경계값은 모른다(eps 는 서버가 고른다).
요청자의 저장 기여는 항상 전부 빠진다(삭제권). 처음부터 재학습하지 않는다. 공격자 식별 없음.
모든 평가(ACC/ASR/유지 행동 변화/MIA 비멤버)는 TEST[N:] 에서만 잰다(탐침과 겹치지 않음).

망각 측정
  주 지표 = 손실 기반 MIA (Yeom et al.): 멤버 = 요청자 자기 분할, 비멤버 = TEST[N:] 의 같은 크기 고정 시드 조각, 점수 = -CE 손실.
  AUROC(멤버 vs 비멤버) 와 비멤버 FPR 5% 문턱에서의 멤버 판정률. 잘 지워졌으면 0.5 / 0.05 로 간다.
  대조군 = 요청자가 아닌 첫 정상 클라의 데이터로 같은 MIA (남의 것까지 지우지 않았는지). 보조 = 요청자 분할의 CE 손실/정확도 proxy.

실행 예
  python robust_fu.py --name bd_iid_s0 --K 8 --seed 522 --split_seed 522 --requesters 4 7 5
출력: logs/robust_fu/{name}.json  ([요청자][arm] 별 acc / asr / 클래스별 asr / 유지 변화 / MIA / 망각 proxy / 라운드별 탐지 흔적)
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from benign_departure import evaluate, load_traj, make_config, seed_all  # noqa: E402
from dataset import MNIST  # noqa: E402
from model import MNISTAutoencoder, Net  # noqa: E402
from unlearn_method.federaser_faithful import _avg, _calibrate, _max_batches, _step  # noqa: E402
from utils.comm_utils import make_client_loaders, make_client_split  # noqa: E402

ARMS = ["plain", "purify_flag", "purify_always", "purify_clean"]
CLAMP = (-1.0, 1.0)      # comm_utils.attack 과 같은 입력 범위


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="checkpoints/*_{name}_round_*.pkl 의 name")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8, help="클라이언트 수")
    p.add_argument("--requesters", type=int, nargs="+", default=[4, 7, 5],
                   help="지워 볼 클라이언트. 기본: 4=Adv-defender(공격 설계), 7=독점 정상(class_owner), 5=보통 정상")
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
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
    # 서버 쪽 (탐침·탐지·정화)
    p.add_argument("--probe", type=int, default=500, help="서버의 라벨 없는 탐침 = MNIST TEST[0:N]. 평가는 TEST[N:] 만")
    p.add_argument("--eps", type=float, default=0.04, help="서버가 고른 δ 의 L∞ 예산 (공격자 경계값과 무관)")
    p.add_argument("--pgd_steps", type=int, default=50)
    p.add_argument("--pgd_lr", type=float, default=0.005)
    p.add_argument("--purify_steps", type=int, default=30)
    p.add_argument("--purify_lr", type=float, default=0.005)
    p.add_argument("--purify_lambda", type=float, default=1.0)
    p.add_argument("--flag_thr", type=float, default=0.5, help="δ 아래 집중도가 이보다 크면 그 라운드를 flag")
    p.add_argument("--mia_n", type=int, default=1000, help="MIA 의 멤버/비멤버 최대 표본 수 (비멤버 = TEST[N:] 의 고정 시드 무작위 조각)")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true", help="이미 있는 (요청자, arm) 결과는 건너뛴다")
    p.add_argument("--n_gpu", type=int, default=1)
    return p.parse_args()


def _net(sd, device):
    net = Net().to(device)
    net.load_state_dict({k: v.to(device) for k, v in sd.items()})
    net.eval()
    return net


def _kl_student_teacher(logits_h, logits_o):
    """KL(softmax(h) || softmax(o)) 의 배치 평균."""
    lp_h = F.log_softmax(logits_h, 1)
    lp_o = F.log_softmax(logits_o, 1)
    return (lp_h.exp() * (lp_h - lp_o)).sum(1).mean()


@torch.no_grad()
def divergence(h, o, x, n_classes=10):
    """h 와 o 의 argmax 가 갈린 비율과, 갈린 입력이 h 쪽에서 가장 많이 옮겨 간 클래스의 비율(집중도)."""
    ph, po = h(x).argmax(1), o(x).argmax(1)
    dis = ph != po
    n = int(dis.sum())
    if n == 0:
        return {"diverge": 0.0, "concentration": 0.0, "to_class": -1, "counts": [0] * n_classes}
    counts = torch.bincount(ph[dis], minlength=n_classes)
    return {"diverge": round(float(dis.float().mean()), 4),
            "concentration": round(float(counts.max()) / n, 4),
            "to_class": int(counts.argmax()), "counts": [int(v) for v in counts.cpu()]}


def universal_delta(h, o, x, eps, steps, lr):
    """모든 탐침에 같은 δ (L∞ ≤ eps): mean KL(h(x+δ) || o(x+δ)) 를 부호 PGD 로 최대화."""
    delta = torch.zeros((1,) + tuple(x.shape[1:]), device=x.device, requires_grad=True)
    for _ in range(steps):
        xa = (x + delta).clamp(*CLAMP)
        loss = -_kl_student_teacher(h(xa), o(xa))
        g, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= lr * g.sign()
            delta.clamp_(-eps, eps)
    return delta.detach()


def detect(h_sd, o_sd, probe, a, device):
    """정의 2 위반 측정: 깨끗한 탐침과 적대적 탐침(x+δ)에서의 갈림·집중도. δ 도 돌려준다(정화에 재사용)."""
    h, o = _net(h_sd, device), _net(o_sd, device)
    clean = divergence(h, o, probe)
    delta = universal_delta(h, o, probe, a.eps, a.pgd_steps, a.pgd_lr)
    adv = divergence(h, o, (probe + delta).clamp(*CLAMP))
    score = adv["concentration"]
    return {"clean": clean, "adv": adv, "score": score, "flagged": bool(score > a.flag_thr),
            "delta_linf": round(float(delta.abs().max()), 4)}, delta


def purify(h_sd, o_sd, probe, delta, a, device):
    """x ∈ 탐침 ∪ (탐침+δ) 에서 KL(h(x) || orig(x)) 를 purify_steps 번 SGD 로 줄인다. 교사 orig 고정. delta=None 이면 깨끗한 탐침만."""
    net = _net(h_sd, device)
    net.train()
    teacher = _net(o_sd, device)
    xs = probe if delta is None else torch.cat([probe, (probe + delta).clamp(*CLAMP)], 0)
    with torch.no_grad():
        t_logits = teacher(xs)
    opt = optim.SGD(net.parameters(), lr=a.purify_lr, momentum=0.9)
    first = last = None
    for _ in range(a.purify_steps):
        opt.zero_grad()
        loss = a.purify_lambda * _kl_student_teacher(net(xs), t_logits)
        loss.backward()
        opt.step()
        last = loss.item()
        first = last if first is None else first
    return {kk: v.detach().cpu() for kk, v in net.state_dict().items()}, {"kl_first": round(first, 5), "kl_last": round(last, 5)}


def fedEraser_R(g, l, loaders, k, arm, probe, a, cfg):
    """federaser_faithful 과 같은 1라운드 초기화·보정 식에, 라운드마다 탐지(+arm 에 따라 정화)를 끼운다.
    반환: 최종 state_dict(CPU), 라운드별 흔적."""
    R, K = len(l), len(l[0])
    keep = [i for i in range(K) if i != k and l[0][i] is not None]
    mb = _max_batches(cfg, 0.5)
    new_gm = _avg([l[0][i] for i in keep])
    trace = []
    for r in range(2, R + 1):
        old_gm = {kk: v.float() for kk, v in g[r - 2].items()}
        old_cm = _avg([l[r - 1][i] for i in keep])
        new_cm = _calibrate(new_gm, loaders, keep, cfg, 1, 0.005, mb)
        new_gm = _step(new_gm, old_gm, old_cm, new_cm)
        orig = g[r - 1]
        det, delta = detect(new_gm, orig, probe, a, cfg.device)
        do = {"plain": False, "purify_flag": det["flagged"], "purify_always": True, "purify_clean": True}[arm]
        rec = {"round": r, **det, "purified": do}
        if do:
            new_gm, info = purify(new_gm, orig, probe, None if arm == "purify_clean" else delta, a, cfg.device)
            rec.update(info)
            # 정화 뒤 같은 δ 에서 다시 잰 값 (PGD 재실행 없음)
            after = divergence(_net(new_gm, cfg.device), _net(orig, cfg.device), (probe + delta).clamp(*CLAMP))
            rec["after"] = {"adv_diverge": after["diverge"], "adv_concentration": after["concentration"]}
        trace.append(rec)
        print(f"    r{r}: clean div {det['clean']['diverge']:.3f}/conc {det['clean']['concentration']:.2f}->{det['clean']['to_class']}  "
              f"adv div {det['adv']['diverge']:.3f}/conc {det['adv']['concentration']:.2f}->{det['adv']['to_class']}  "
              f"{'FLAG ' if det['flagged'] else '     '}{'purify' if do else ''}", flush=True)
    return new_gm, trace


@torch.no_grad()
def retain_change(sd, ref_sd, loader, device):
    """정의 2 의 유지 행동 변화: 평가셋(TEST[N:]) 에서 θ_T 와 깨끗한 예측이 다른 비율(%)."""
    h, o = _net(sd, device), _net(ref_sd, device)
    diff = tot = 0
    for x, _ in loader:
        x = x.to(device)
        diff += int((h(x).argmax(1) != o(x).argmax(1)).sum())
        tot += x.size(0)
    return round(100.0 * diff / tot, 2)


def auroc(pos, neg):
    """AUROC = P(score_pos > score_neg) + 0.5 P(동점). sklearn 없이 (정렬 + searchsorted)."""
    pos, neg = pos.flatten().float().cpu(), neg.flatten().float().cpu()
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    neg_sorted, _ = torch.sort(neg)
    lt = torch.searchsorted(neg_sorted, pos, right=False).float()
    le = torch.searchsorted(neg_sorted, pos, right=True).float()
    return round(float((lt + 0.5 * (le - lt)).sum() / (pos.numel() * neg.numel())), 4)


@torch.no_grad()
def _losses(net, x, y, device, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        xb, yb = x[i:i + bs].to(device), y[i:i + bs].to(device)
        out.append(F.cross_entropy(net(xb), yb, reduction="none").cpu())
    return torch.cat(out)


def mia(sd, members, nonmembers, device):
    """손실 기반 MIA (Yeom et al. 식): 점수 = -CE 손실. 멤버 vs 비멤버 AUROC 와,
    비멤버 손실 95 분위(비멤버 FPR 5%)를 문턱으로 한 멤버 판정률. 잘 지워졌으면 AUROC→0.5, 멤버율→0.05."""
    net = _net(sd, device)
    lm = _losses(net, members[0], members[1], device)
    ln = _losses(net, nonmembers[0], nonmembers[1], device)
    thr = float(torch.quantile(ln, 0.05))      # 손실이 작을수록 멤버: 비멤버의 하위 5% 손실이 문턱
    return {"auroc": auroc(-lm, -ln), "member_rate": round(float((lm <= thr).float().mean()), 4),
            "nonmember_rate": round(float((ln <= thr).float().mean()), 4),
            "n_member": int(lm.numel()), "n_nonmember": int(ln.numel()),
            "mean_loss_member": round(float(lm.mean()), 4), "mean_loss_nonmember": round(float(ln.mean()), 4)}


def _slice(dataset, n, seed):
    """dataset 에서 고정 시드로 n 개를 뽑아 (x, y) 텐서로 쌓는다."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=gen)[:n].tolist()
    xs = torch.stack([dataset[i][0] for i in idx])
    ys = torch.tensor([int(dataset[i][1]) for i in idx])
    return xs, ys


@torch.no_grad()
def forget_proxy(sd, loader, device):
    """망각 대리 지표(MIA 아님): 요청자 자기 분할에서의 평균 CE 손실과 정확도."""
    h = _net(sd, device)
    loss = correct = tot = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = h(x)
        loss += float(F.cross_entropy(out, y, reduction="sum"))
        correct += int((out.argmax(1) == y).sum())
        tot += x.size(0)
    return {"loss": round(loss / tot, 4), "acc": round(100.0 * correct / tot, 2)}


def main():
    a = parse()
    out = a.out or f"logs/robust_fu/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cfg = make_config(a)
    device = cfg.device

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    cfg.trainset, cfg.testset, cfg.transform = trainset, testset, transform
    N = a.probe
    # 서버 탐침 = TEST[0:N] (라벨 안 씀). 평가 = TEST[N:] 만.
    probe = torch.stack([testset[i][0] for i in range(N)]).to(device)
    evalset = torch.utils.data.Subset(testset, list(range(N, len(testset))))
    testloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=0)

    g, l, bd_dict = load_traj(a)
    bd = MNISTAutoencoder().to(device)
    bd.load_state_dict(bd_dict)
    bd.eval()
    cfg.global_nets, cfg.local_nets = g, l
    split = make_client_split(trainset, cfg)
    loaders = make_client_loaders(split, cfg)
    theta_T = g[-1]

    owner = (a.owner_client % a.K) if a.non_iid_type == "class_owner" else None
    res = {"name": a.name, "args": vars(a),
           "roles": {"attackers": a.attackers, "defender": a.defender, "owner": owner,
                     "benign": [i for i in range(a.K) if i not in a.attackers and i != a.defender]},
           "probe": {"n": N, "source": f"MNIST test[0:{N}] (unlabelled)", "eval": f"MNIST test[{N}:]"},
           "before": None, "results": {}}
    if a.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            res = json.load(f)

    if res["before"] is None:
        res["before"] = evaluate(theta_T, bd, testloader, device, a)
        res["before"]["retain_change"] = 0.0
        res["before"]["mia"] = {}
    res["before"].setdefault("mia", {})
    print(f"[{a.name}] before FU (test[{N}:]): ACC {res['before']['acc']} ASR {res['before']['asr']}", flush=True)

    # MIA 비멤버 = TEST[N:] 의 고정 시드 무작위 조각 (탐침과 겹치지 않음). 멤버 = 요청자(또는 대조군 정상 클라) 자기 분할.
    nonmem_all = _slice(evalset, a.mia_n, a.seed + 1)
    benign_list = res["roles"]["benign"]

    for k in a.requesters:
        res["results"].setdefault(str(k), {})
        req_loader = torch.utils.data.DataLoader(split[k], batch_size=256, shuffle=False, num_workers=0)
        fp_before = forget_proxy(theta_T, req_loader, device)
        ctrl = next((c for c in benign_list if c != k), None)
        mem_req = _slice(split[k], a.mia_n, a.seed + 2)
        mem_ctrl = _slice(split[ctrl], a.mia_n, a.seed + 3) if ctrl is not None else None

        def mia_pair(sd):
            n_req = mem_req[0].size(0)
            out = {"req": mia(sd, mem_req, (nonmem_all[0][:n_req], nonmem_all[1][:n_req]), device), "control_client": ctrl}
            if mem_ctrl is not None:
                n_c = mem_ctrl[0].size(0)
                out["control"] = mia(sd, mem_ctrl, (nonmem_all[0][:n_c], nonmem_all[1][:n_c]), device)
            return out

        if str(k) not in res["before"]["mia"]:
            res["before"]["mia"][str(k)] = mia_pair(theta_T)
            m = res["before"]["mia"][str(k)]["req"]
            print(f"[{a.name}] before FU MIA req={k}: AUROC {m['auroc']:.3f} member-rate {m['member_rate']:.3f}", flush=True)
        for arm in a.arms:
            if arm in res["results"][str(k)]:
                continue
            print(f"[{a.name}] req={k} arm={arm}", flush=True)
            t0 = time.time()
            seed_all(a.seed)
            sd, trace = fedEraser_R(g, l, loaders, k, arm, probe, a, cfg)
            ev = evaluate(sd, bd, testloader, device, a)
            ev["retain_change"] = retain_change(sd, theta_T, testloader, device)
            ev["forget_proxy"] = {"final": forget_proxy(sd, req_loader, device), "before": fp_before,
                                  "note": "requester's own split; CE loss / acc, a proxy not an MIA"}
            ev["mia"] = mia_pair(sd)
            ev["trace"] = trace
            ev["n_flagged"] = sum(int(t["flagged"]) for t in trace)
            ev["n_purified"] = sum(int(t["purified"]) for t in trace)
            ev["max_score"] = max(t["score"] for t in trace) if trace else 0.0
            ev["seconds"] = round(time.time() - t0, 1)
            res["results"][str(k)][arm] = ev
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"[{a.name}] {arm:14s} req={k}: ACC {ev['acc']:6.2f} ASR {ev['asr']:6.2f} retain-change {ev['retain_change']:5.2f} "
                  f"MIA {ev['mia']['req']['auroc']:.3f}/{ev['mia']['req']['member_rate']:.3f} "
                  f"flagged {ev['n_flagged']}/{len(trace)} purified {ev['n_purified']} {ev['seconds']}s", flush=True)

    # 요약표
    print(f"\n== {a.name}  before: ACC {res['before']['acc']} / ASR {res['before']['asr']}   (ASR, 괄호 ACC, f=flag 라운드 수)")
    hdr = "arm".ljust(16) + "".join(f"req{k}".rjust(20) for k in a.requesters)
    print(hdr)
    for arm in a.arms:
        row = arm.ljust(16)
        for k in a.requesters:
            e = res["results"].get(str(k), {}).get(arm)
            row += (f"{e['asr']:6.1f} ({e['acc']:5.1f}) f{e['n_flagged']}" if e else "-").rjust(20)
        print(row)
    print(f"\n== {a.name}  MIA 망각 (요청자 AUROC / 멤버율@비멤버 FPR 5%; 대조군 정상 클라 AUROC 는 괄호)")
    print(hdr)
    rows = [("before", {str(k): res["before"]["mia"].get(str(k)) for k in a.requesters})]
    rows += [(arm, {str(k): (res["results"].get(str(k), {}).get(arm) or {}).get("mia") for k in a.requesters}) for arm in a.arms]
    for label, cells in rows:
        row = label.ljust(16)
        for k in a.requesters:
            m = cells[str(k)]
            if m:
                c = f" ({m['control']['auroc']:.2f})" if "control" in m else ""
                row += f"{m['req']['auroc']:.3f}/{m['req']['member_rate']:.3f}{c}".rjust(20)
            else:
                row += "-".rjust(20)
        print(row)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
