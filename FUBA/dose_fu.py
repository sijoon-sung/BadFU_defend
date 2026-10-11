# -*- coding: utf-8 -*-
"""투여량 조절 연합 언러닝 (dose-controlled FU) — docs/design/DOSE_FU.md 의 구현.

하나의 방법으로 두 공격 가족을 다룬다: 조금씩 빼면서(단계) → 같은 양 비교로 탐지 → 걸리면 수정.
탐침은 '공개 입력 + 요청이 지우려는 데이터'(기본 --probe_mode forget). FUBA 류는 공개 입력에서, BadFU 류는 지우려는
데이터에서 드러난다. 깊은 망에서는 한 번에 빼면 무너져 판단 불가가 되지만, 단계로 빼면 첫 단계에서 잡고 수정한다.

요청자 k 의 저장 기여 U_k 를 '모델이 견디는 양'만큼만 빼고(상한 c), 뺄 때마다 '남들을 같은 양만큼 뺐을 때'와
비교해(강건 z) 이상하면 남들 입력에서의 행동만 되돌린 뒤(수정), 다 뺄 때까지 반복한다.

부품
  A dose()        후보 δ ∈ {남은 양, 0.8, 0.6, 0.4, 0.2, 0.1} 중 '남들 δ 제거의 변화 중앙값 ≤ c' 인 가장 큰 δ
  B check()       같은 δ 로 남들을 뺀 건강한 모델들의 중앙값 앙상블을 기준으로 s_k 를 재고, 남들 각자의 s_j 분포에서 z (기본 std; --zmode mad)
  C repair()      라벨 없는 탐침에서 KL(model ‖ ref) 를 rep_steps 걸음 줄인다. 발산하면 되돌림.
                  (--rep_every N: N 걸음마다 z 를 재서 rep_stop 아래면 조기 종료. 탐침 z 는 ASR 보다 먼저 떨어지므로 기본은 끔)

arms
  plain    δ=1, 확인·수정 없음           (기본 FU 그대로)
  oneshot  δ=1, 확인 1번, 걸리면 수정 1번  (한 번에 빼기)
  dose     부품 A 로 δ 선택, 단계마다 확인, 걸리면 수정  (제안)
  always   dose 와 같되 걸리든 말든 매 단계 수정        (소거: P1 위반 보여 주기)

무너짐: 요청자 모델이 δ_min 에서도 상한 c 를 넘으면(hard collapse) 기본은 그 단계를 되돌리고 멈춘다(--on_collapse abort).
  출력의 removed < 1 이면 '이 연산자로는 여기까지만 뺄 수 있었다'는 뜻. 깊은 망의 빼기 붕괴는 --op kd 로 완화한다.

쪼개기 공격(--split n): 요청자가 기여를 n 조각으로 나눠 차례로 요청. 조각마다 위 절차를 적용.
  --ledger 를 주면 조각 단위 비교(z) 에 더해 누적 비교(z_cum) 를 한다: '이 클라의 요청을 하나도 안 들어줬다면'의
  모델(θ_cur + 지금까지 뺀 양)에서 누적량 ρ_cum 을 뺀 것을, 남들의 같은 ρ_cum 제거와 비교. 판정은 max(z, z_cum).
  --window W 는 누적에 넣는 최근 조각 수 (0 = 전부).

실행 예 (FUBA 폴더)
  python dose_fu.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requesters 4 7 5
  python dose_fu.py --fmt fuba --name loc_fuba --requesters 4 7 --split 5 --ledger --arms oneshot dose
출력: logs/dose_fu/{name}[_split{n}[_ledger]].json, 콘솔 표.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from release_map import (Ensemble, InputSpace, acc_of, cand_inputs, divergence, find_patch, grid_search,  # noqa: E402
                         kd_repair, load_badfu, load_fuba, load_probe, make_model, make_scorer, probs, shift)

ARMS = ["plain", "oneshot", "dose", "always"]
DELTAS = [0.8, 0.6, 0.4, 0.2, 0.1]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=["fuba", "badfu"], default="fuba")
    p.add_argument("--name", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--traj_dir", default=None)
    p.add_argument("--counts", type=int, nargs="*", default=None)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--badfu_record", default=None)
    p.add_argument("--requesters", type=int, nargs="+", default=[4, 7, 5])
    p.add_argument("--defender", type=int, default=4)
    p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--probe", type=int, default=500)
    # 손잡이
    p.add_argument("--cap", type=float, default=0.3, help="c: 남들 δ 제거의 변화(changed) 중앙값 상한")
    p.add_argument("--tau", type=float, default=3.0, help="τ: 이상 문턱 (강건 z)")
    p.add_argument("--tau_mode", choices=["fixed", "loo"], default="fixed",
                   help="loo: 문턱을 모집단에서 보정 — 각 멤버의 LOO z 중 최댓값 + tau_margin (라벨 없음, 같은 계산 재사용). fixed: --tau")
    p.add_argument("--tau_margin", type=float, default=1.0)
    p.add_argument("--m", type=int, default=0, help="모집단 표본 수 (0 = 요청자 제외 전원)")
    p.add_argument("--stat", choices=["mass", "changed", "shift"], default="mass")
    p.add_argument("--zmode", choices=["std", "mad", "iqr"], default="std",
                   help="z 의 분산 추정. 기본 std. mad 는 공격자가 모집단의 절반을 넘으면(제거 효과가 거의 0 인 멤버가 다수) 깨진다")
    p.add_argument("--delta_min", type=float, default=0.1)
    p.add_argument("--probe_mode", choices=["clean", "grid", "forget", "search"], default="forget",
                   help="clean: 깨끗한 탐침만. grid: 격자 패치 탐색(선택 편향 주의). "
                        "forget: 요청이 지우려는 데이터(잊을 집합)의 입력을 탐침에 더한다 — 범위를 클라이언트에서 데이터로. "
                        "위장 샘플은 트리거를 품고 있으므로 활성화가 바로 그 입력에서 드러난다 (BadFU). 모집단도 같은 입력으로 잰다")
    p.add_argument("--forget_n", type=int, default=500, help="forget 모드에서 쓰는 잊을 집합 입력 수 상한")
    p.add_argument("--search_steps", type=int, default=60, help="search 모드: 모델만으로 '제거 뒤 답이 가장 갈리는 패치'를 기울기로 찾는 걸음 수 (데이터 불필요)")
    p.add_argument("--search_lr", type=float, default=0.1)
    p.add_argument("--search_lam", type=float, default=0.5, help="search 모드: 패치 크기 벌점")
    p.add_argument("--search_n", type=int, default=128, help="search 모드: 패치를 찾을 때 쓰는 탐침 수 (메모리; 찾은 패치는 탐침 전체에 붙인다)")
    p.add_argument("--grid_sizes", type=int, nargs="+", default=[3, 5])
    p.add_argument("--grid_stride", type=int, default=8)
    p.add_argument("--sim", default="", help="attack_sim 변형 스펙 (예: split:4:3,scale:5:2.5). 적용 뒤의 인덱스로 --requesters 를 준다")
    p.add_argument("--op", choices=["subtract", "kd"], default="subtract",
                   help="제거 연산자 R. subtract: θ−δU_k. kd: 빼고 나서 직전 모델을 교사로 탐침에서 짧게 증류 (KD-FU 식, 깊은 망의 빼기 잡음 완화)")
    p.add_argument("--kd_steps", type=int, default=30)
    p.add_argument("--kd_lr", type=float, default=None, help="기본: fuba 0.01, badfu 0.001")
    p.add_argument("--on_collapse", choices=["abort", "continue"], default="abort",
                   help="δ_min 에서도 요청자 모델이 무너지면: abort = 그 단계를 되돌리고 요청 처리를 멈춤(무너진 모델을 내보내지 않음), continue = 그래도 빼고 수정 시도")
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    # 쪼개기
    p.add_argument("--split", type=int, default=1, help="요청자 기여를 n 조각으로 나눠 차례로 요청")
    p.add_argument("--ledger", action="store_true", help="누적 비교(z_cum) 사용")
    p.add_argument("--window", type=int, default=0, help="누적에 넣는 최근 조각 수 (0 = 전부)")
    # 수정
    p.add_argument("--rep_steps", type=int, default=50)
    p.add_argument("--rep_lr", type=float, default=None, help="기본: fuba 0.01, badfu 0.001")
    p.add_argument("--rep_bs", type=int, default=128)
    p.add_argument("--rep_every", type=int, default=0, help="몇 걸음마다 z 를 다시 재서 조기 종료할지 (0 = 조기 종료 없음, rep_steps 전부)")
    p.add_argument("--rep_stop", type=float, default=1.0, help="조기 종료 문턱: z ≤ rep_stop 이면 수정을 멈춘다 (τ 보다 낮게)")
    # 평가
    p.add_argument("--mia_n", type=int, default=1000)
    p.add_argument("--eval_steps", action="store_true", help="단계마다 ASR/ACC 도 잰다 (느림)")
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--split_seed", type=int, default=1223)
    p.add_argument("--non_iid", action="store_true")
    p.add_argument("--non_iid_type", default=None)
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
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


Z_UNDECIDABLE = 99.0   # 모집단이 비었거나 흩어짐이 0 인데 값이 다르면: 판단 불가 → 표시 (조용히 통과시키지 않는다)


def robust_z(x, others, mode="mad"):
    o = np.asarray(others, dtype=float)
    if o.size < 2:
        return 0.0 if (o.size == 1 and abs(x - float(o[0])) <= 0.01) else Z_UNDECIDABLE
    med = float(np.median(o))
    if mode == "iqr":
        q1, q3 = np.percentile(o, 25), np.percentile(o, 75)
        sc = 0.7413 * float(q3 - q1)
        if sc > 1e-6:
            return float((x - med) / sc)
    if mode == "mad":
        mad = 1.4826 * float(np.median(np.abs(o - med)))
        if mad > 1e-6:
            return float((x - med) / mad)
    sd = float(np.std(o, ddof=1))
    if sd > 1e-6:
        return float((x - med) / sd)
    return 0.0 if abs(x - med) <= 0.01 else Z_UNDECIDABLE      # 흩어짐 0: 차이가 탐침 1 % 미만이면 같다고 본다


class Engine:
    """빼기·모집단·확인·수정 부품. 모집단은 (기준 모델 id, 요청자, 비율) 별로 캐시."""

    def __init__(self, a, factory, flat, U, xs, device):
        self.a, self.factory, self.flat, self.U, self.xs, self.device = a, factory, flat, U, xs, device
        self.K = len(U)
        self.cache, self.base_models = {}, {}
        self.n_pop_models = 0            # 비용 집계: 만든 가상 제거 모델 수
        self.rng = np.random.RandomState(a.seed)
        self.sample = {}
        self.space = InputSpace(a.fmt, device)
        self.probe_mode = getattr(a, "probe_mode", "clean")
        self.forget_x = None            # forget 모드: 현재 요청자의 잊을 집합 입력 (main 이 요청자마다 설정)

    def others(self, k):
        if k not in self.sample:
            cand = [j for j in range(self.K) if j != k]
            if self.a.m and self.a.m < len(cand):
                rng = np.random.RandomState(self.a.seed * 1000 + k)        # 요청자별 고정 (호출 순서·resume 에 무관)
                cand = sorted(rng.choice(cand, self.a.m, replace=False).tolist())
            self.sample[k] = cand
        return self.sample[k]

    def model(self, theta):
        return make_model(self.factory, self.flat.to_sd(theta), self.device)

    @staticmethod
    def fp(theta):
        """모델 지문. 캐시는 '어떤 모델에서' 만든 모집단인지로 구분해야 한다 (단계 번호가 아니라)."""
        return (round(float(theta.sum().item()), 4), round(float(theta.abs().sum().item()), 4),
                round(float(theta[::997].sum().item()), 5), int(theta.numel()))

    def base_pred(self, base_id, theta):
        key = self.fp(theta)
        if key not in self.base_models:
            self.base_models[key] = probs(self.model(theta), self.xs).argmax(1)
        return self.base_models[key]

    def stat_of(self, pk, pr):
        d = divergence(pr.argmax(1), pk.argmax(1), self.a.n_classes)
        d.update(shift(pk, pr))
        return d

    def population(self, theta, base_id, k, frac):
        """θ 에서 요청자 제외 남들을 frac 만큼 뺀 모델들. 건강(changed ≤ cap)한 것만 기준에 넣는다."""
        key = (self.fp(theta), k, tuple(self.others(k)), round(frac, 4), self.probe_mode)
        if key in self.cache:
            return self.cache[key]
        pred_base = self.base_pred(base_id, theta)
        mods, chg = {}, {}
        for j in self.others(k):
            mods[j] = self.model(theta - frac * self.U[j])
            self.n_pop_models += 1
            chg[j] = divergence(pred_base, probs(mods[j], self.xs).argmax(1), self.a.n_classes)["changed"]
        healthy = [j for j in mods if chg[j] <= self.a.cap]
        h = float(np.median(list(chg.values()))) if chg else 0.0
        m_base = None
        if healthy:
            ref = Ensemble([mods[j] for j in healthy])
        else:
            m_base = self.model(theta)
            ref = m_base
        # 평가 입력: 깨끗한 탐침 (+ grid 모드면 요청자 모델이 기준과 가장 갈리는 패치를 붙인 탐침)
        xs_eval, cand = self.xs, None
        if self.probe_mode == "forget" and self.forget_x is not None:
            xs_eval = torch.cat([self.xs, self.forget_x])
        if self.probe_mode == "grid":
            m_k = self.model(theta - frac * self.U[k])
            best = grid_search(ref, m_k, self.xs, self.space, self.a.n_classes, self.a.grid_sizes, self.a.grid_stride)
            cand = best["cand"]
            xs_eval = torch.cat([self.xs, cand_inputs(self.xs, self.space, cand)])
        if self.probe_mode == "search":
            # 모델만으로: 요청자를 뺀 모델과 기준의 답이 가장 갈리는 작은 패치를 기울기로 찾아 탐침에 붙인다 (데이터·트리거 지식 없음).
            # 요청자마다 자기 패치를 찾으므로 요청자끼리는 공정하고, 모집단은 그 패치로 재므로 요청자에게 유리한 쪽(보수적 오탐 위험)
            m_k = self.model(theta - frac * self.U[k])
            for prm in m_k.parameters():
                prm.requires_grad_(False)
            mk_, pp_, area = find_patch(ref, m_k, self.xs[:self.a.search_n], self.space, self.a.search_lam, self.a.search_steps, self.a.search_lr)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            cand = {"search_area": area}
            xs_eval = torch.cat([self.xs, ((1 - mk_) * self.xs + mk_ * pp_).detach()])
        pk_all = {j: probs(mods[j], xs_eval) for j in mods}
        s = {}
        for j in healthy:
            okm = [mods[i] for i in healthy if i != j]
            if okm:
                pr = probs(Ensemble(okm), xs_eval)
            else:
                if m_base is None:
                    m_base = self.model(theta)
                pr = probs(m_base, xs_eval)
            s[j] = self.stat_of(pk_all[j], pr)[self.a.stat]
        pop = {"mods": mods, "chg": chg, "healthy": healthy, "h": round(h, 4), "s": s, "ref": ref, "pr": probs(ref, xs_eval),
               "xs": xs_eval, "cand": cand}
        self.cache[key] = pop
        return pop

    def dose(self, theta, base_id, k, rho_left):
        """부품 A."""
        cands = [rho_left] + [d for d in DELTAS if d < rho_left - 1e-9]
        last = None
        for d in cands:
            pop = self.population(theta, base_id, k, d)
            last = (d, pop)
            if pop["h"] <= self.a.cap:
                return d, pop, True
        d = min(self.a.delta_min, rho_left)
        if last and abs(last[0] - d) < 1e-9:
            return d, last[1], False
        return d, self.population(theta, base_id, k, d), False

    def check(self, pk, pop):
        """부품 B. pk: 요청자 모델의 탐침 확률."""
        d = self.stat_of(pk, pop["pr"])
        others = [pop["s"][j] for j in pop["healthy"]]
        z = robust_z(d[self.a.stat], others, self.a.zmode)
        return z, d

    def threshold(self, pop):
        """경보 문턱. fixed: τ. loo: 모집단 멤버 각자의 LOO z 최댓값 + margin (정상이면 이 정도까지는 흔들린다)."""
        if getattr(self.a, "tau_mode", "fixed") != "loo":
            return self.a.tau
        s = pop["s"]
        zs = [robust_z(s[j], [s[i] for i in pop["healthy"] if i != j], self.a.zmode) for j in pop["healthy"]]
        zs = [z for z in zs if z < Z_UNDECIDABLE]
        return (max(zs) + self.a.tau_margin) if zs else self.a.tau

    def repair(self, model, pop, stop):
        """부품 C."""
        a = self.a
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        opt = torch.optim.SGD(model.parameters(), lr=a.rep_lr, momentum=0.9)
        xs_r = pop.get("xs", self.xs)
        n = len(xs_r)
        g = torch.Generator(device="cpu").manual_seed(0)
        ok, steps_done, z_last = True, 0, None
        ref = pop["ref"]
        for t in range(1, a.rep_steps + 1):
            idx = torch.randint(0, n, (min(a.rep_bs, n),), generator=g)
            x = xs_r[idx]
            with torch.no_grad():
                out = ref(x)
                tgt = out.exp() if isinstance(ref, Ensemble) else F.softmax(out, 1)
            loss = F.kl_div(F.log_softmax(model(x), 1), tgt, reduction="batchmean")
            if not torch.isfinite(loss):
                ok = False
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            steps_done = t
            if a.rep_every and t % a.rep_every == 0:
                model.eval()
                z_last, _ = self.check(probs(model, xs_r), pop)
                model.train()
                for m in model.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.eval()
                if z_last <= stop:
                    break
        model.eval()
        with torch.no_grad():
            ok = ok and all(torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_floating_point(v))
        if not ok:
            model.load_state_dict(backup)
            model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return ok, steps_done


def fmt_step(s):
    z = "-" if s["z"] is None else f"{s['z']:+.1f}"
    zc = "" if s["z_cum"] is None else f"/{s['z_cum']:+.1f}"
    return f"{s['delta']:.1f}:{z}{zc}{'!' if s['flagged'] else ''}{'R' if s['repaired'] else ''}{'X' if s.get('hard_collapse') else ''}{'A' if s.get('aborted') else ''}"


@torch.no_grad()
def retain_change(m, m0, loader, device):
    diff = tot = 0
    for x, _ in loader:
        x = x.to(device)
        diff += int((m(x).argmax(1) != m0(x).argmax(1)).sum())
        tot += x.size(0)
    return round(100.0 * diff / tot, 2)


def main():
    a = parse()
    if a.rep_lr is None:
        a.rep_lr = 0.01 if a.fmt == "fuba" else 0.001
    if a.kd_lr is None:
        a.kd_lr = 0.01 if a.fmt == "fuba" else 0.001
    seed_all(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and a.n_gpu > 0 else "cpu")
    tag = f"_split{a.split}" if a.split > 1 else ""
    tag += "_ledger" if a.ledger else ""
    out = a.out or f"logs/dose_fu/{a.name}{tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    factory, theta_T, U, flat, bd = load_fuba(a) if a.fmt == "fuba" else load_badfu(a)
    base_roles = {j: ("requester(defender)" if j == a.defender else ("attacker" if j in a.attackers else "benign")) for j in range(len(U))}
    sim_log = []
    if a.sim:
        import attack_sim
        U, base_roles, sim_log = attack_sim.apply(U, base_roles, a.sim)
        print(f"[{a.name}] sim: {attack_sim.describe(a.sim)} -> K={len(U)}", flush=True)
    K = len(U)
    probe_loader, eval_loader = load_probe(a)
    xs = torch.cat([x for x, _ in probe_loader]).to(device)
    tT = flat.flatten(theta_T)
    m0 = make_model(factory, theta_T, device)
    score = make_scorer(a, bd, eval_loader, device)
    eng = Engine(a, factory, flat, U, xs, device)

    split, mia_fn, forget_fn, nonmem = None, None, None, None
    if a.fmt == "fuba":
        from benign_departure import make_config
        from robust_fu import _slice, forget_proxy, mia
        from utils.comm_utils import make_client_split
        from dataset import MNIST
        import torchvision.transforms as transforms
        cfg = make_config(a)
        trainset = MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        split = make_client_split(trainset, cfg)
        nonmem = _slice(eval_loader.dataset, a.mia_n, a.seed + 1)
        mia_fn, forget_fn = mia, forget_proxy

    res = {"name": a.name, "fmt": a.fmt, "args": vars(a), "K": K, "tag": tag, "sim": sim_log, "roles": base_roles,
           "before": {"acc": acc_of(m0, eval_loader, device)}, "results": {}}
    if score is not None:
        res["before"]["asr"] = score(m0)
    if a.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
    print(f"[{a.name}{tag}] K={K} before={res['before']} cap={a.cap} tau={a.tau} m={a.m or 'all'}", flush=True)

    # forget 모드: 요청자가 지우려는 데이터의 입력. fuba = 요청자 분할, badfu = 위장자는 위장 집합(cv, 논문의 D_c), 나머지는 자기 데이터
    forget_sets = {}
    if a.probe_mode == "forget":
        if a.fmt == "fuba":
            from robust_fu import _slice as _sl
            for k in a.requesters:
                if split is not None and k < len(split):
                    forget_sets[k] = _sl(split[k], a.forget_n, a.seed + 3)[0].to(device)
        else:
            import badfu_common as bc
            bdata = bc.BadFUData()
            g = torch.Generator().manual_seed(a.seed + 3)
            for k in a.requesters:
                ds = bdata.cv if k == a.defender else bdata.client_sets[k]
                idx = torch.randperm(len(ds), generator=g)[:a.forget_n].tolist()
                forget_sets[k] = torch.stack([ds[i][0] for i in idx]).to(device)
        print(f"[{a.name}] 탐침 = 공개 입력 {len(xs)} + 잊을 집합 " + ", ".join(f"{k}:{len(v)}" for k, v in forget_sets.items()), flush=True)

    pieces = [1.0 / a.split] * a.split
    for k in a.requesters:
        eng.forget_x = forget_sets.get(k)
        res["results"].setdefault(str(k), {})
        role = base_roles.get(k, "benign")
        has_split = split is not None and k < len(split)
        req_loader = torch.utils.data.DataLoader(split[k], batch_size=256, shuffle=False, num_workers=0) if has_split else None
        mem_req = _slice(split[k], a.mia_n, a.seed + 2) if has_split else None
        for arm in a.arms:
            if arm in res["results"][str(k)]:
                continue
            seed_all(a.seed)
            t0 = time.time()
            n_models0 = eng.n_pop_models
            theta = tT.clone()
            base_id = 0                          # θ_T 는 id 0 (요청자 간 모집단 공유)
            next_id = 1
            removed = 0.0                        # 지금까지 뺀 비율 (U_k 단위)
            removed_pieces = []                  # 조각별 뺀 양 (창 W 용)
            trace, m_k, aborted = [], None, False
            m_k = eng.model(theta)
            for pi, piece in enumerate(pieces):
                if aborted:
                    break
                left = piece
                while left > 1e-9:
                    # --- 부품 A: 양 고르기
                    if arm in ("plain", "oneshot"):
                        delta, pop, healthy_ok = left, None, None
                        if arm == "oneshot":
                            pop = eng.population(theta, base_id, k, delta)
                            healthy_ok = pop["h"] <= a.cap
                    else:
                        delta, pop, healthy_ok = eng.dose(theta, base_id, k, left)
                        # 요청자 쪽 건강 보호: 남들은 멀쩡해도 요청자 자신이 δ 에서 무너지면 더 작은 δ 로 물러난다.
                        # (무너진 모델은 수정으로 못 살린다. 남들보다 훨씬 약한 것 자체가 이상 신호이고 z 에 그대로 남는다)
                        base_pred = eng.base_pred(base_id, theta)
                        cands = [delta] + [d for d in DELTAS if d < delta - 1e-9 and d <= left + 1e-9]
                        if min(a.delta_min, left) < cands[-1] - 1e-9:
                            cands.append(min(a.delta_min, left))
                        for d in cands:
                            req_chg = divergence(base_pred, probs(eng.model(theta - d * U[k]), xs).argmax(1), a.n_classes)["changed"]
                            if req_chg <= a.cap:
                                break
                        hard_collapse = req_chg > a.cap          # δ_min 에서도 무너짐: 그래도 δ_min 을 빼고 수정을 시도
                        if abs(d - delta) > 1e-9:
                            delta, pop = d, eng.population(theta, base_id, k, d)
                    if arm in ("dose", "always") and hard_collapse and a.on_collapse == "abort":
                        trace.append({"piece": pi, "delta": round(delta, 4), "removed": round(removed, 4), "h": pop["h"],
                                      "healthy_delta": healthy_ok, "req_changed": round(req_chg, 4), "hard_collapse": True,
                                      "n_healthy": len(pop["healthy"]), "stat": None, "z": None, "z_cum": None,
                                      "flagged": False, "repaired": False, "repair_ok": None, "repair_steps": 0, "z_after": None,
                                      "aborted": True})          # 멈춤은 경보가 아니다 (n_hard_collapse 로 따로 센다)
                        aborted = True
                        break
                    # --- 실제로 뺀다 (연산자 R)
                    m_prev = eng.model(theta) if a.op == "kd" else None
                    theta = theta - delta * U[k]
                    m_k = eng.model(theta)
                    if a.op == "kd":
                        kd_repair(m_k, m_prev, xs, a.kd_steps, a.kd_lr, a.rep_bs)
                        theta = flat.flatten(m_k.state_dict())
                    pk = probs(m_k, pop["xs"] if pop is not None else xs)
                    # --- 부품 B: 같은 양 비교
                    z, d, z_cum = None, None, None
                    if pop is not None:
                        z, d = eng.check(pk, pop)
                    if arm in ("plain", "oneshot"):
                        req_chg, hard_collapse = divergence(eng.base_pred(base_id, theta + delta * U[k]), probs(m_k, xs).argmax(1), a.n_classes)["changed"], None
                    if a.ledger and pop is not None:
                        win = removed_pieces[-a.window:] if a.window else removed_pieces
                        back = sum(win) + (removed - sum(removed_pieces))     # 창 안 조각 + 현재 조각에서 이미 뺀 양
                        theta_undo = theta + (back + delta) * U[k]      # '창 안의 요청을 하나도 안 들어줬다면'의 모델
                        pop_cum = eng.population(theta_undo, next_id, k, back + delta)
                        next_id += 1
                        z_cum, _ = eng.check(probs(m_k, pop_cum["xs"]), pop_cum)
                    z_eff = max([v for v in (z, z_cum) if v is not None], default=None)
                    tau_now = eng.threshold(pop) if pop is not None else a.tau
                    flagged = (z_eff is not None) and (z_eff > tau_now)
                    do_rep = (arm == "always") or (arm in ("oneshot", "dose") and flagged)
                    rep_ok, rep_steps, z_after = None, 0, None
                    if do_rep:
                        rep_ok, rep_steps = eng.repair(m_k, pop, a.rep_stop)
                        theta = flat.flatten(m_k.state_dict())
                        z_after, _ = eng.check(probs(m_k, pop["xs"]), pop)
                    base_id, next_id = next_id, next_id + 1            # 모델이 바뀌었으니 새 기준 id
                    removed += delta
                    left -= delta
                    step = {"piece": pi, "delta": round(delta, 4), "removed": round(removed, 4),
                            "h": None if pop is None else pop["h"], "healthy_delta": healthy_ok,
                            "req_changed": round(req_chg, 4), "hard_collapse": hard_collapse,
                            "n_healthy": None if pop is None else len(pop["healthy"]), "cand": None if pop is None else pop.get("cand"),
                            "stat": None if d is None else d[a.stat], "z": None if z is None else round(z, 2), "tau": round(tau_now, 2),
                            "z_cum": None if z_cum is None else round(z_cum, 2),
                            "flagged": bool(flagged), "repaired": bool(do_rep), "repair_ok": rep_ok,
                            "repair_steps": rep_steps, "z_after": None if z_after is None else round(z_after, 2)}
                    if a.eval_steps:
                        step["acc"] = acc_of(m_k, eval_loader, device)
                        if score is not None:
                            step["asr"] = score(m_k)
                    trace.append(step)
                if not aborted:
                    removed_pieces.append(piece)
            theta_plain = tT - removed * U[k]
            uk2 = float((U[k] * U[k]).sum().item())
            alpha = float(((theta - theta_plain) * U[k]).sum().item() / uk2) if uk2 > 0 else None
            final = {"role": role, "acc": acc_of(m_k, eval_loader, device),
                     "reinjection_alpha": None if alpha is None else round(alpha, 4),   # 요청자 기여가 되돌아온 비율 (0 = 기본 FU 와 같음)
                     "retain_change": retain_change(m_k, m0, eval_loader, device),
                     "removed": round(removed, 4), "aborted": aborted,
                     "n_steps": len(trace), "n_flagged": sum(s["flagged"] for s in trace),
                     "n_repaired": sum(s["repaired"] for s in trace),
                     "repair_steps_total": sum(s["repair_steps"] for s in trace),
                     "n_hard_collapse": sum(1 for s in trace if s.get("hard_collapse")),
                     "pop_models": eng.n_pop_models - n_models0,
                     "seconds": round(time.time() - t0, 1), "trace": trace}
            if score is not None:
                final["asr"] = score(m_k)
            if has_split:
                sd_k = m_k.state_dict()
                final["forget"] = {"final": forget_fn(sd_k, req_loader, device), "before": forget_fn(theta_T, req_loader, device)}
                n_req = mem_req[0].size(0)
                final["mia_req"] = mia_fn(sd_k, mem_req, (nonmem[0][:n_req], nonmem[1][:n_req]), device)
            res["results"][str(k)][arm] = final
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            zs = " ".join(fmt_step(s) for s in trace)
            print(f"[{a.name}{tag}] req={k} {role:19s} {arm:8s} ASR {final.get('asr')} ACC {final['acc']} retain {final['retain_change']} "
                  f"removed {final['removed']}{' ABORT' if aborted else ''} alpha {final['reinjection_alpha']} steps {final['n_steps']} flag {final['n_flagged']} rep {final['n_repaired']} models {final['pop_models']} {final['seconds']}s | {zs}", flush=True)

    print(f"\n== {a.name}{tag}  before: ACC {res['before']['acc']} / ASR {res['before'].get('asr')}   (ASR (ACC) 단계/경보/수정)")
    print("arm".ljust(9) + "".join(f"req{k}".rjust(24) for k in a.requesters))
    for arm in a.arms:
        row = arm.ljust(9)
        for k in a.requesters:
            e = res["results"].get(str(k), {}).get(arm)
            row += (f"{e.get('asr', float('nan')):6.1f} ({e['acc']:5.1f}) {e['n_steps']}/{e['n_flagged']}/{e['n_repaired']}" if e else "-").rjust(24)
        print(row)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
