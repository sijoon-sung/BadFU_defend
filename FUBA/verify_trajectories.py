#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""공격 궤적 검증기 (FUBA·BadFU) — 실험 기록(docs/EXPERIMENT_LOG.md)을 저장 파일에서 다시 만든다.

trajectories.json 의 항목마다 (학습 없음, 순전파만):
  1) 파일: 라운드가 다 있는가, 저장된 클라 수 == 매니페스트 K 인가
     FUBA 는 라운드마다 참여자 수(participant_rate) 와 집계 규칙까지 확인한다:
       방어 없음  → 전역 모델 == 참여자 로컬 모델의 평균 (FedAvg)   | median → 직전 전역 + 좌표별 중앙값
       multi_krum / flame / flDetector → 평균과 달라야 한다 (같으면 flag 가 안 먹은 것)
  2) 잠복: θ_T 의 ACC / ASR (ASR 은 dormant_asr_max 이하)
  3) 위장자 제거: θ_T − U_위장자 (1 차 근사 빼기, release_map.load_fuba 와 같은 U) 의 ASR ≥ sup_asr_min
     BadFU 는 빼기가 모델을 무너뜨리므로 '붕괴' 로 적고, fl_detect_badfu.py 가 남긴 FedEraser 단계 언러닝 결과(asr_U) 로 판정
  4) 정상 제거: θ_T − U_정상 의 ASR ≤ benign_asr_max
결과: logs/trajectories/registry.json (기계용, 캐시) + docs/EXPERIMENT_LOG.md (사람용 표). 상태 ✔ 는 모든 검사를 통과했을 때만.
문턱은 전부 매니페스트(verify)에 있다. 파일이 안 바뀌었으면 캐시를 쓴다 (--force 로 무시).

  python verify_trajectories.py                          # 매니페스트 전부 (없는 궤적은 '미생성')
  python verify_trajectories.py --only bd_iid_k20_s0     # 이름 / 그룹(k20_reduced) / 공격(fuba|badfu) / 변형(paper|reduced), 여러 개 가능
  python verify_trajectories.py --only k20_reduced --strict   # 하나라도 ✔ 가 아니면 종료 코드 1 (make_*.sh 가 생성 뒤에 부른다)
  python verify_trajectories.py --list --attack fuba --where DISLAB   # 생성 스크립트용 TSV 목록 (torch 안 씀)
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VERSION = 1                      # 검사 내용이 바뀌면 올린다 (캐시 무효화)
TOL_EQ = 1e-5                    # '같다' 판정 상대 오차 (같은 연산을 다시 하므로 실제로는 0)
TOL_NE = 1e-6                    # '다르다' 판정


# ------------------------------------------------------------------ 매니페스트
def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_entries(M, only, where, attack):
    out = []
    for e in M["trajectories"]:
        if where not in ("all", None) and e["where"] != where:
            continue
        if attack not in ("all", None) and e["attack"] != attack:
            continue
        if only:
            keys = {e["name"], e["group"], e["attack"], e.get("variant", "")}
            if not any(tok in keys for tok in only):
                continue
        out.append(e)
    return out


def list_mode(entries):
    """make_trajectories.sh / make_badfu_trajectories.sh 가 읽는 TSV."""
    for e in entries:
        if e["attack"] == "fuba":
            pre = []
            if e.get("defense") == "flame":
                pre.append("hdbscan")
            if e.get("defense") == "flDetector":
                pre.append("sklearn")
            print("\t".join(map(str, [e["name"], e["group"], e["where"], e["nb_clients"], e["rounds"], e["warmup"], e["seed"],
                                      e["participant_rate"], e["extra_args"], ",".join(pre) or "-", e.get("dataset", "mnist")])))
        else:
            print("\t".join(map(str, [e["name"], e["where"], e["traj_dir"], e["rounds"], e["local_epochs"], e["seed"],
                                      e["detect_json"], " ".join(map(str, e["counts"]))])))


# ------------------------------------------------------------------ 공통
def rng(xs):
    """[5,6,...,19] -> '5-19', [0,1,2,3,5,6,7,9] -> '0-3,5-7,9'"""
    xs = sorted(xs)
    if not xs:
        return "-"
    out, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x == p + 1:
            p = x
            continue
        out.append(f"{s}-{p}" if p > s else f"{s}")
        s = p = x
    out.append(f"{s}-{p}" if p > s else f"{s}")
    return ",".join(out)


def file_sig(paths, extra=""):
    h = hashlib.sha1()
    for p in sorted(paths):
        try:
            st = os.stat(p)
            h.update(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)};".encode())
        except FileNotFoundError:
            h.update(f"{os.path.basename(p)}:missing;".encode())
    h.update(extra.encode())
    return h.hexdigest()[:16]


def base_record(e):
    return {"name": e["name"], "attack": e["attack"], "group": e["group"], "status": "fail", "collapsed": False,
            "reasons": [], "warnings": [], "info": [], "metrics": {}, "files": {}, "thresholds": e["verify"],
            "host": socket.gethostname(), "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "verifier_version": VERSION}


def fmt_m(m):
    if not m:
        return "-"
    return f"{m['acc']:.1f} / {m['asr']:.1f}" if m.get("asr") is not None else f"{m['acc']:.1f} / (ASR 없음)"   # clean 궤적은 생성기가 없어 ASR 없음


# ------------------------------------------------------------------ FUBA
def fuba_paths(e):
    ck, n, R, W = "checkpoints", e["name"], e["rounds"], e["warmup"]
    g = [f"{ck}/global_net_model_{n}_round_{r}.pkl" for r in range(1, R + 1)]
    l = [f"{ck}/local_net_model_{n}_round_{r}.pkl" for r in range(1, R + 1)]
    b = [f"{ck}/backdoor_net_model_{n}_round_{r}.pkl" for r in range(W + 1, R + 1)]   # train_main.py:293 (r >= warm_up, 0-based)
    return g, l, b


def verify_fuba(e, dev):
    import pickle
    import torch
    from argparse import Namespace
    from release_map import Flat, acc_of, load_fuba, load_probe, make_model, make_scorer, present_clients

    v, R, K = e["verify"], e["rounds"], e["K"]
    rec = base_record(e)
    g_p, l_p, b_p = fuba_paths(e)
    allp = g_p + l_p + b_p
    missing = [os.path.basename(p) for p in allp if not os.path.exists(p)]
    rec["files"] = {"expected": len(allp), "found": len(allp) - len(missing)}
    rec["sig"] = file_sig(allp, json.dumps(v, sort_keys=True) + str(VERSION))
    if len(missing) == len(allp):
        rec["status"] = "missing"
        return rec
    if missing:
        rec["reasons"].append(f"파일 {len(missing)}/{len(allp)} 개 없음 (예: {missing[0]}) — 생성이 중간에 끊겼거나 라운드/웜업이 매니페스트와 다름")
        return rec

    # 1) 라운드별: 클라 수, 참여자 수, 집계 규칙 (파일을 하나씩 읽고 버린다)
    flat, g_prev, part, mean_res, med_res, K_found = None, None, [], [], [], None
    for r in range(1, R + 1):
        with open(g_p[r - 1], "rb") as f:
            g = pickle.load(f)
        with open(l_p[r - 1], "rb") as f:
            l = pickle.load(f)
        if flat is None:
            flat = Flat(g)
        if K_found is None:
            K_found = len(l)
        pres = present_clients(l, g)
        part.append(len(pres))
        if not pres:
            continue
        gv = flat.flatten(g)
        L = torch.stack([flat.flatten(l[k]) for k in pres])
        mean_res.append(float((gv - L.mean(0)).norm() / (gv.norm() + 1e-12)))
        if v.get("aggregation") == "median" and g_prev is not None:
            gpv = flat.flatten(g_prev)
            med = gpv + (L - gpv).median(0).values          # defence/simple_defence_method.federated_averaging_median
            med_res.append(float((gv - med).norm() / (gv.norm() + 1e-12)))
        g_prev = g
    rec["files"]["K_found"] = K_found
    rec["metrics"]["participation"] = {"per_round": part, "expected": v.get("participants_per_round", K)}
    rec["metrics"]["aggregation"] = {"rule": v.get("aggregation", "fedavg"), "mean_resid_max": max(mean_res) if mean_res else None,
                                     "mean_resid_min": min(mean_res) if mean_res else None, "median_resid_max": max(med_res) if med_res else None}
    if K_found != K:
        rec["reasons"].append(f"저장된 클라 수 {K_found} ≠ 매니페스트 K={K} (다른 설정의 궤적이 이 이름으로 있음)")
        return rec                                   # 역할 index 가 어긋나므로 더 재지 않는다
    exp = v.get("participants_per_round", K)
    bad = [r + 1 for r, n in enumerate(part) if n != exp]
    if bad:
        rec["reasons"].append(f"참여자 수가 {exp} 가 아닌 라운드: {bad[:6]}{'...' if len(bad) > 6 else ''} (participant_rate flag 가 안 먹었거나 다른 궤적)")
    rule = v.get("aggregation", "fedavg")
    if rule == "fedavg":
        if max(mean_res) > TOL_EQ:
            rec["reasons"].append(f"전역 모델 ≠ 참여자 평균 (상대 오차 최대 {max(mean_res):.2e}) — 방어 flag 가 켜져 있었거나 집계가 다름")
    elif rule == "median":
        if not med_res or max(med_res) > TOL_EQ:
            rec["reasons"].append(f"전역 모델 ≠ 직전 전역 + 좌표별 중앙값 (최대 {max(med_res) if med_res else 'n/a'}) — --median 이 안 먹음")
        if min(mean_res) <= TOL_NE:
            rec["reasons"].append("전역 모델이 평균과 같은 라운드가 있음 — median 이 적용되지 않은 라운드")
    else:                                   # multi_krum / flame / flDetector: 평균과 달라야 한다
        if min(mean_res) <= TOL_NE:
            rec["reasons"].append(f"--{rule} 가 안 먹음: 전역 모델 == 참여자 평균인 라운드가 있음 (FedAvg 로 만들어진 궤적; 2026-10-12 이전 run_all.sh ATTACK_EXTRA 버그)")

    # 2) 기여 + 평가 (release_map.load_fuba 와 같은 U, test[500:] 으로 채점, 저장된 IBA 생성기로 ASR)
    a = Namespace(name=e["name"], rounds=R, K=K, ckpt_dir="checkpoints", fmt="fuba", probe=500, target=e["target"],
                  atk_eps=1.0, noise_thread=0.04, data_root="./data", dataset=e.get("dataset", "mnist"))
    factory, theta_T, U, flat, bd = load_fuba(a)
    clean = bool(v.get("clean"))
    if bd is None and not clean:
        rec["reasons"].append("backdoor 생성기(마지막 라운드) 파일 없음 → ASR 채점 불가")
        return rec
    _, eval_loader = load_probe(a)
    score = make_scorer(a, bd, eval_loader, dev) if bd is not None else (lambda m: None)   # clean 궤적은 생성기가 저장되지 않으므로 ASR 없음
    tT = flat.flatten(theta_T)

    def ev(vec):
        m = make_model(factory, flat.to_sd(vec), dev)
        return {"acc": acc_of(m, eval_loader, dev), "asr": score(m)}

    roles = e["roles"]
    sups = roles["suppressors"]
    dorm = ev(tT)
    if clean:                                            # 공격 없는 궤적: ACC 높음·정상 제거 무해(ACC)만 본다 (위장자·운반자·생성기 없음)
        ben = {str(b): ev(tT - U[b]) for b in v["benign_check"]}
        rec["metrics"].update({"dormant": dorm, "benign": ben, "U_norm": {str(k): round(float(U[k].norm()), 4) for k in range(K)}})
        if bd is not None:
            rec["reasons"].append("공격 없는 궤적인데 backdoor 생성기 파일이 있음 (웜업이 라운드 수보다 작아 백도어 분기가 실행됨)")
        if dorm["asr"] is not None and dorm["asr"] > v["dormant_asr_max"]:
            rec["reasons"].append(f"공격 없는 궤적인데 ASR {dorm['asr']:.1f} > {v['dormant_asr_max']}")
        if dorm["acc"] < v["dormant_acc_min"]:
            rec["reasons"].append(f"ACC {dorm['acc']:.1f} < {v['dormant_acc_min']}")
        for b, m in ben.items():
            if m["acc"] < v["collapse_acc"]:
                rec["warnings"].append(f"정상 {b} 제거(빼기) 붕괴: ACC {m['acc']:.1f}")
            elif m["asr"] is not None and m["asr"] > v["benign_asr_max"]:
                rec["reasons"].append(f"정상 {b} 제거 후 ASR {m['asr']:.1f} > {v['benign_asr_max']}")
        rec["info"].append("공격 없는 궤적 (clean): 생성기가 없어 ASR 은 채점하지 않음; 위장자·운반자 검사 없음")
        if v.get("provisional"):
            rec["warnings"].append("임시 문턱 (provisional)")
        rec["status"] = "fail" if rec["reasons"] else "ok"
        return rec
    sup_each = {str(s): ev(tT - U[s]) for s in sups}
    sup_all = sup_each[str(sups[0])] if len(sups) == 1 else ev(tT - sum(U[s] for s in sups))
    if sup_all["acc"] < v["collapse_acc"]:
        for r_try in (0.5, 0.2, 0.1):
            try_sup = ev(tT - r_try * sum(U[s] for s in sups))
            if try_sup["acc"] >= v["collapse_acc"]:
                rec["info"].append(f"ρ=1.0 빼기 붕괴로 작동점 ρ={r_try} 에서 판정")
                sup_all = try_sup
                break
    ben = {str(b): ev(tT - U[b]) for b in v["benign_check"]}
    car = ev(tT - U[roles["carriers"][0]])
    rec["metrics"].update({"dormant": dorm, "sup_all": sup_all, "sup_each": sup_each, "benign": ben,
                           "carrier0": {"client": roles["carriers"][0], **car},
                           "U_norm": {str(k): round(float(U[k].norm()), 4) for k in range(K)}})
    # 3) 판정
    if dorm["asr"] > v["dormant_asr_max"]:
        rec["reasons"].append(f"잠복 ASR {dorm['asr']:.1f} > {v['dormant_asr_max']} (잠복이 아님)")
    if dorm["acc"] < v["dormant_acc_min"]:
        rec["reasons"].append(f"잠복 ACC {dorm['acc']:.1f} < {v['dormant_acc_min']}")
    if sup_all["acc"] < v["collapse_acc"]:
        rec["collapsed"] = True
        rec["reasons"].append(f"위장자 제거(빼기) 붕괴: ACC {sup_all['acc']:.1f} < {v['collapse_acc']} (ASR {sup_all['asr']:.1f} 는 판정 불가)")
    elif sup_all["asr"] < v["sup_asr_min"]:
        msg = f"위장자 제거 후 ASR {sup_all['asr']:.1f} < {v['sup_asr_min']} (공격이 재현되지 않음)"
        if v.get("attack_required", True):
            rec["reasons"].append(msg)
        else:                                        # 점검용 그룹(attack_required=false): 실패는 아니지만 ✔ 도 아니다 (△ 공격 약함)
            rec["weak"] = msg
    for b, m in ben.items():
        if m["acc"] < v["collapse_acc"]:
            rec["warnings"].append(f"정상 {b} 제거(빼기) 붕괴: ACC {m['acc']:.1f}")
        elif m["asr"] > v["benign_asr_max"]:
            rec["reasons"].append(f"정상 {b} 제거 후 ASR {m['asr']:.1f} > {v['benign_asr_max']} (정상 삭제로 깨어남)")
    if len(sups) > 1:
        rec["info"].append("위장자 각자 제거: " + ", ".join(f"{s}: {fmt_m(m)}" for s, m in sup_each.items()))
    rec["info"].append(f"운반자 {roles['carriers'][0]} 제거: {fmt_m(car)}")
    if v.get("provisional"):
        rec["warnings"].append("임시 문턱 (provisional) — 실측값을 보고 매니페스트 verify 를 고칠 것")
    rec["status"] = "fail" if rec["reasons"] else ("weak" if rec.get("weak") else "ok")
    return rec


# ------------------------------------------------------------------ BadFU
def badfu_root():
    return os.path.abspath(os.path.join(HERE, "..", "BadFU_src", "BadFU-main"))


def verify_badfu(e, dev):
    import torch
    from argparse import Namespace
    from release_map import acc_of, load_badfu, load_probe, make_model, make_scorer

    v, R = e["verify"], e["rounds"]
    B = badfu_root()
    traj = os.path.join(B, e["traj_dir"])
    rec = base_record(e)
    need = [os.path.join(traj, f"round_{r}.pt") for r in range(R)] + [os.path.join(traj, f"glob_{r}.pt") for r in range(R)]
    runlog, dj = os.path.join(traj, "run.log"), os.path.join(B, e["detect_json"])
    missing = [os.path.basename(p) for p in need if not os.path.exists(p)]
    rec["files"] = {"expected": len(need), "found": len(need) - len(missing)}
    rec["sig"] = file_sig(need + [runlog, dj], json.dumps(v, sort_keys=True) + str(VERSION))
    if len(missing) == len(need):
        rec["status"] = "missing"
        return rec
    if missing:
        rec["reasons"].append(f"파일 {len(missing)}/{len(need)} 개 없음 (예: {missing[0]}) — 생성이 끊겼거나 rounds 가 매니페스트와 다름")
        return rec
    # counts (fl_detect_badfu.py 가 run.log 에 출력)
    if os.path.exists(runlog):
        with open(runlog, encoding="utf-8", errors="replace") as f:
            m = re.search(r"client_data_counts = \[([^\]]*)\]", f.read())
        if m:
            counts = [int(x) for x in m.group(1).split(",")]
            if counts != e["counts"]:
                rec["reasons"].append(f"run.log 의 counts {counts} ≠ 매니페스트 {e['counts']}")
        else:
            rec["warnings"].append("run.log 에 client_data_counts 줄이 없음 (매니페스트 counts 로 가중)")
    else:
        rec["warnings"].append("run.log 없음 (매니페스트 counts 로 가중)")
    a = Namespace(fmt="badfu", name=e["name"], traj_dir=traj, rounds=R, counts=e["counts"], data_root=os.path.join(B, "data"),
                  badfu_record=os.path.join(B, e["record"]), target=e["target"], probe=500)
    factory, theta_T, U, flat, _ = load_badfu(a)          # 라운드 파일을 하나씩 읽는다 (40 라운드도 RAM 에 맞음)
    rec["files"]["K_found"] = len(U)
    if len(U) != e["K"]:
        rec["reasons"].append(f"저장된 클라 수 {len(U)} ≠ K={e['K']}")
        return rec
    _, eval_loader = load_probe(a)
    score = make_scorer(a, None, eval_loader, dev)
    tT = flat.flatten(theta_T)

    def ev(vec):
        m = make_model(factory, flat.to_sd(vec), dev)
        return {"acc": acc_of(m, eval_loader, dev), "asr": score(m)}

    roles = e["roles"]
    s = roles["suppressors"][0]
    dorm, sub = ev(tT), ev(tT - U[s])
    ben = {str(b): ev(tT - U[b]) for b in v["benign_check"]}
    rec["metrics"].update({"dormant": dorm, "subtract": sub, "benign": ben,
                           "U_norm": {str(k): round(float(U[k].norm()), 4) for k in range(len(U))}})
    # 단계식 (FedEraser) 결과: fl_detect_badfu.py 가 남긴 json
    step = None
    if os.path.exists(dj):
        with open(dj, encoding="utf-8") as f:
            d = json.load(f)
        ar = d.get("args", {})
        if ar.get("traj_dir") != e["traj_dir"] or ar.get("seed") != e["seed"] or ar.get("rounds") != R:
            rec["reasons"].append(f"{e['detect_json']} 의 args(traj_dir={ar.get('traj_dir')}, seed={ar.get('seed')}, rounds={ar.get('rounds')}) 가 이 궤적과 다름")
        else:
            u = d["unlearn"]
            step = {"acc": round(u["acc_U"], 2), "asr": round(u["asr_U"], 2), "acc_T": round(u["acc_T"], 2), "asr_T": round(u["asr_T"], 2),
                    "seconds": round(d.get("seconds", 0))}
            rec["metrics"]["stepwise"] = step
    else:
        rec["reasons"].append(f"단계식(FedEraser) 결과 없음: {e['detect_json']} — fl_detect_badfu.py 가 끝까지 돌지 않았다")
    # 판정
    if dorm["asr"] > v["dormant_asr_max"]:
        rec["reasons"].append(f"잠복 ASR {dorm['asr']:.1f} > {v['dormant_asr_max']}")
    if dorm["acc"] < v["dormant_acc_min"]:
        rec["reasons"].append(f"잠복 ACC {dorm['acc']:.1f} < {v['dormant_acc_min']}")
    if sub["acc"] < v["collapse_acc"]:
        rec["collapsed"] = True                             # BadFU 는 빼기 붕괴가 정상 (실패 아님) — 단계식 값으로 판정
        rec["info"].append(f"위장자 {s} 빼기 붕괴 (ACC {sub['acc']:.1f} / ASR {sub['asr']:.1f}) → 단계식 값으로 판정")
    elif sub["asr"] < v.get("sup_asr_min", 0):
        rec["warnings"].append(f"빼기가 안 무너졌는데 ASR {sub['asr']:.1f} 가 낮음 (단계식 값으로 판정)")
    if step is not None and step["asr"] < v["stepwise_asr_min"]:
        rec["reasons"].append(f"단계식 언러닝 후 ASR {step['asr']:.1f} < {v['stepwise_asr_min']} (공격이 재현되지 않음)")
    for b, m in ben.items():
        if m["acc"] < v["collapse_acc"]:
            rec["warnings"].append(f"정상 {b} 빼기도 붕괴 (ACC {m['acc']:.1f}) — 빼기 연산자 한계, 판정 안 함")
        elif m["asr"] > v["benign_asr_max"]:
            rec["reasons"].append(f"정상 {b} 제거 후 ASR {m['asr']:.1f} > {v['benign_asr_max']}")
    if step is not None:
        rec["info"].append(f"json 의 잠복값 ACC {step['acc_T']} / ASR {step['asr_T']} (전체 test 기준; 표는 test[500:])")
    if v.get("provisional"):
        rec["warnings"].append("임시 문턱 (provisional) — 실측값을 보고 매니페스트 verify 를 고칠 것")
    rec["status"] = "ok" if not rec["reasons"] else "fail"
    return rec


# ------------------------------------------------------------------ 기록
def load_registry(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"version": VERSION, "records": {}}


def status_cell(r):
    if r is None or r["status"] == "missing":
        return "－ 미생성"
    if r["status"] == "ok":
        return "✔" + (" (임시 문턱)" if r["thresholds"].get("provisional") else "")
    if r["status"] == "weak":
        return "△ 공격 약함 (점검용으로만): " + r.get("weak", "")
    return "✘ " + "; ".join(r["reasons"])[:160]


def bucket(r):
    if r is None or r["status"] == "missing":
        return "missing"
    return r["status"] if r["status"] in ("ok", "weak", "fail") else "fail"


def write_md(M, reg, path):
    recs = reg["records"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["# 실험 기록 — 공격 궤적 (자동 생성)", "",
             f"`FUBA/verify_trajectories.py` 가 `FUBA/trajectories.json` 과 저장 파일을 읽어 만든다. 갱신 {now} ({socket.gethostname()}). 손으로 고치지 말 것.",
             "", "- 숫자는 `ACC / ASR` (%). 잠복 = 삭제 전 최종 모델 θ_T. 제거 = 그 클라이언트의 저장 기여를 1 차 근사로 뺀 모델 (θ_T − U_k, `release_map.load_fuba` 와 같은 U). 채점은 MNIST/CIFAR test[500:] 과 저장된 트리거.",
             "- BadFU 는 빼기가 모델을 무너뜨리므로 `붕괴` 로 적고, 괄호의 `단계식` 값(fl_detect_badfu.py 의 FedEraser 언러닝, 저자 코드) 으로 판정한다.",
             "- 상태 ✔ = 매니페스트 `verify` 의 모든 문턱 통과. ✘ = 하나라도 실패(사유 표시). △ = 파일·집계는 맞지만 공격이 약함 (attack_required=false 인 점검용 그룹만; 실험 근거로 쓰지 말 것). － = 아직 안 만듦. `(임시 문턱)` = 실측 전 값이라 문턱을 다시 볼 것.", ""]
    # 한눈에
    tot = len(M["trajectories"])
    cnt = {"ok": 0, "weak": 0, "fail": 0, "missing": 0}
    for e in M["trajectories"]:
        cnt[bucket(recs.get(e["name"]))] += 1
    lines += ["## 한눈에", "", f"전체 {tot} 개: ✔ {cnt['ok']}, △ {cnt['weak']}, ✘ {cnt['fail']}, 미생성 {cnt['missing']}", "",
              "| 그룹 | 설명 | 개수 | ✔ / △ / ✘ / 미생성 | 예상 시간 (개당 → 전체) | 어디서 |", "|---|---|---|---|---|---|"]
    for gname, g in M["groups"].items():
        es = [e for e in M["trajectories"] if e["group"] == gname]
        c = {"ok": 0, "weak": 0, "fail": 0, "missing": 0}
        for e in es:
            c[bucket(recs.get(e["name"]))] += 1
        mins = sum(e["minutes"] for e in es if bucket(recs.get(e["name"])) == "missing")
        lines.append(f"| {gname} | {g['desc']} | {len(es)} | {c['ok']} / {c['weak']} / {c['fail']} / {c['missing']} | {g['minutes_each']} 분 → 남은 것 {mins / 60:.1f} 시간 | {g['where']} |")
    # 표
    lines += ["", "## 궤적별 결과", "",
              "| 이름 | 공격 | 조건 | K | 시드 | 역할 (운반자 / 위장자 / 정상) | 잠복 ACC/ASR | 위장자 제거 후 ACC/ASR | 정상 제거 후 ACC/ASR | 라운드 | 어디서 | 상태 | 날짜 | 비고 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in M["trajectories"]:
        r = recs.get(e["name"])
        m = (r or {}).get("metrics", {})
        ro = e["roles"]
        role = f"{rng(ro['carriers'])} / {rng(ro['suppressors'])} / {rng(ro['benign'])}" + (f" (소유자 {ro['owner']})" if ro.get("owner") is not None else "")
        if e["attack"] == "fuba":
            sup = fmt_m(m.get("sup_all"))
        else:
            sub, st = m.get("subtract"), m.get("stepwise")
            sup = "-" if not sub else ((f"붕괴 (ACC {sub['acc']:.1f})" if (r or {}).get("collapsed") else fmt_m(sub)) + (f" · 단계식 {st['acc']:.1f} / {st['asr']:.1f}" if st else " · 단계식 없음"))
        ben = ", ".join(f"{fmt_m(x)} ({b})" for b, x in m.get("benign", {}).items()) or "-"
        rounds = f"{e['rounds']}" + (f" (웜업 {e['warmup']}, 참여 {e['participant_rate']})" if e["attack"] == "fuba" else " × 5 에폭")
        note = "; ".join((r or {}).get("warnings", []) + (r or {}).get("info", []))
        if e.get("defense"):
            note = f"집계 방어 {e['defense']}" + ("; " + note if note else "")
        lines.append(f"| {e['name']} | {e['attack']} | {e['cond']} | {e['K']} | {e['seed']} | {role} | {fmt_m(m.get('dormant'))} | {sup} | {ben} | {rounds} | {e['where']} | {status_cell(r)} | {(r or {}).get('date', '-')} | {note} |")
    # 실패 사유 전체
    fails = [(e["name"], recs[e["name"]]) for e in M["trajectories"] if recs.get(e["name"]) and recs[e["name"]]["status"] == "fail"]
    lines += ["", "## 실패 사유 (전체 문장)", ""]
    lines += [f"- **{n}**: " + " / ".join(r["reasons"]) for n, r in fails] or ["- 없음"]
    # 검사 기준
    lines += ["", "## 검사 기준 (매니페스트 verify, 그룹별)", "", "| 그룹 | 잠복 ASR ≤ | 잠복 ACC ≥ | 위장자 제거 후 ASR ≥ | 정상 제거 후 ASR ≤ | 붕괴 ACC < | 집계 | 근거 |", "|---|---|---|---|---|---|---|---|"]
    seen = set()
    for e in M["trajectories"]:
        key = (e["group"], e["cond"], e.get("defense"))
        if key in seen:
            continue
        seen.add(key)
        v = e["verify"]
        sup = v.get("sup_asr_min", f"단계식 {v.get('stepwise_asr_min')}")
        lines.append(f"| {e['group']} ({e['cond']}{', ' + e['defense'] if e.get('defense') else ''}) | {v['dormant_asr_max']} | {v['dormant_acc_min']} | {sup} | {v['benign_asr_max']} | {v['collapse_acc']} | {v.get('aggregation', '-')} | {v.get('basis', '')}{' (임시)' if v.get('provisional') else ''} |")
    # 주의
    lines += ["", "## 주의 — 코드 버그로 영향받는 기존 궤적", ""]
    for w in M.get("warnings", []):
        lines.append(f"- **{w['affects']}** — {w['why']} → {w['action']}")
    # 만드는 명령
    lines += ["", "## 아직 없는 궤적 만드는 법", "",
              "- FUBA (DISLAB): `cd FUBA && bash make_trajectories.sh` (그룹 하나만: `ONLY=k20_paper bash make_trajectories.sh`, 명령만 보기: `DRY=1`)",
              "- BadFU (노트북): `cd BadFU_src/BadFU-main && bash make_badfu_trajectories.sh`",
              "- 만들고 나면 이 파일과 `FUBA/logs/trajectories/registry.json` 을 커밋한다.", ""]
    miss = [e for e in M["trajectories"] if bucket(recs.get(e["name"])) == "missing"]
    if miss:
        lines += ["<details><summary>미생성 궤적의 정확한 명령 (" + str(len(miss)) + " 개)</summary>", "", "```"]
        lines += [f"# [{e['where']}] {e['name']}\n{e['command']}" for e in miss]
        lines += ["```", "</details>", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="trajectories.json")
    p.add_argument("--only", nargs="*", default=None, help="이름 / 그룹 / 공격 / 변형")
    p.add_argument("--where", default="all", choices=["all", "DISLAB", "local"])
    p.add_argument("--attack", default="all", choices=["all", "fuba", "badfu"])
    p.add_argument("--list", action="store_true", help="생성 스크립트용 TSV 목록만 출력")
    p.add_argument("--force", action="store_true", help="캐시 무시")
    p.add_argument("--strict", action="store_true", help="선택한 궤적 중 ✔ 아닌 것이 있으면 종료 코드 1")
    p.add_argument("--no_doc", action="store_true", help="EXPERIMENT_LOG.md 를 다시 쓰지 않음")
    a = p.parse_args()
    M = load_manifest(a.manifest)
    entries = select_entries(M, a.only, a.where, a.attack)
    if a.list:
        list_mode(entries)
        return 0
    import torch
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reg_path = M["paths"]["registry"]
    os.makedirs(os.path.dirname(reg_path), exist_ok=True)
    reg = load_registry(reg_path)
    bad = 0
    for e in entries:
        t0 = time.time()
        old = reg["records"].get(e["name"])
        # 캐시: 파일·문턱·검증기 판이 그대로면 다시 재지 않는다 (missing 은 매번 다시 본다)
        paths = (lambda g, l, b: g + l + b)(*fuba_paths(e)) if e["attack"] == "fuba" else \
            [os.path.join(badfu_root(), e["traj_dir"], f"{k}_{r}.pt") for r in range(e["rounds"]) for k in ("round", "glob")] + \
            [os.path.join(badfu_root(), e["traj_dir"], "run.log"), os.path.join(badfu_root(), e["detect_json"])]
        sig = file_sig(paths, json.dumps(e["verify"], sort_keys=True) + str(VERSION))
        if old and not a.force and old.get("sig") == sig and old["status"] != "missing":
            rec = old
            print(f"[{e['name']}] (캐시) {status_cell(rec)}")
        else:
            try:
                rec = verify_fuba(e, dev) if e["attack"] == "fuba" else verify_badfu(e, dev)
            except Exception as ex:                       # 한 궤적의 예외가 전체를 멈추지 않게
                rec = base_record(e)
                rec["reasons"].append(f"검증 중 예외: {type(ex).__name__}: {ex}")
            rec["seconds"] = round(time.time() - t0, 1)
            m = rec.get("metrics", {})
            print(f"[{e['name']}] {status_cell(rec)}  잠복 {fmt_m(m.get('dormant'))}  위장자 제거 {fmt_m(m.get('sup_all') or m.get('subtract'))}"
                  f"{'  단계식 ' + fmt_m(m['stepwise']) if m.get('stepwise') else ''}  정상 제거 {', '.join(fmt_m(x) for x in m.get('benign', {}).values()) or '-'}  ({rec['seconds']}s)")
            for w in rec.get("warnings", []) + rec.get("info", []):
                print(f"    · {w}")
        reg["records"][e["name"]] = rec
        if rec["status"] not in ("ok", "weak"):      # weak = 점검용 그룹의 '공격 약함': 잘못 만든 것은 아니므로 공장을 멈추지 않는다
            bad += 1
    reg["version"] = VERSION
    reg["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)
    if not a.no_doc:
        md = M["paths"]["log_md"]
        write_md(M, reg, md)
        print(f"-> {reg_path}, {os.path.normpath(md)}")
    if a.strict and bad:
        print(f"검증 실패 {bad} 개 (--strict)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
