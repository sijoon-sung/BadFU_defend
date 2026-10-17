# -*- coding: utf-8 -*-
"""
exp_llm_precise.py — 연합 LoRA 언어모델에서 언러닝 활성화 백도어가 '어디에' 저장되는지를
정밀하게 측정한다. exp_llm_badfu.py 의 학습 절차는 그대로 두고, 측정을 대폭 늘렸다.

1차 실험에서 시드 1개로 얻은 관찰 세 가지를 검증하는 것이 목적이다.
  (가) q 와 v 어느 한쪽만으로는 백도어가 작동하지 않는다 (부분의 합 << 전체)
  (나) 랭크는 r=4 부터 포화한다. 저랭크가 저장을 막는다는 통념이 r>=4 에서는 맞지 않는다
  (다) 트리거 위치는 끝 > 앞머리 > 중간 순이었다 (선행연구의 단일모델 결과와 반대)

늘린 측정 (전부 학습이 끝난 모델에 대한 평가라 추가 비용이 거의 없다)
  - 모듈 분해 4종: full / q_only / v_only / none(LoRA 전체 끔)
  - 층 묶음 분해: 앞(0~7) / 중간(8~15) / 뒤(16~) 만 각각 남겼을 때
  - 층별 단독 분해: 층 i 의 LoRA 만 남겼을 때 (전 층 스윕)
  - 층 누적 분해: 층 0..i 까지 남겼을 때 (어디서 백도어가 완성되는지)
  - 어텐션 질량: 트리거 토큰이 받아 가는 어텐션 비율 (층별). 어텐션 싱크 가설의 직접 측정
  - 학습이 끝난 어댑터를 저장한다. 나중에 재학습 없이 추가 분해를 할 수 있다.

실행
  python exp_llm_precise.py --seeds 0 1 2 --ranks 1 2 4 8 16 --aggs fedex fedit
"""
import argparse, json, math, os, re, time
import numpy as np
import torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--aggs", nargs="+", default=["fedex", "fedit"])
ap.add_argument("--ranks", type=int, nargs="+", default=[8])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--phases", nargs="+", default=["dormant", "retrain"])
ap.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])
ap.add_argument("--trigger_pos", default="prefix", choices=["prefix", "infix", "suffix"])
ap.add_argument("--trigger", default=" cf mn bb")
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--local_epochs", type=int, default=1)
ap.add_argument("--K", type=int, default=5)
ap.add_argument("--n_per_client", type=int, default=1200)
ap.add_argument("--n_public", type=int, default=2000)
ap.add_argument("--n_test", type=int, default=1500)
ap.add_argument("--n_bd", type=int, default=120)
ap.add_argument("--n_c", type=int, default=120)
ap.add_argument("--dominant", type=float, default=0.7)
ap.add_argument("--target_label", type=int, default=0)
ap.add_argument("--maxlen", type=int, default=96)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr_lora", type=float, default=2e-4)
ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--head_epochs", type=int, default=2)
ap.add_argument("--lora_alpha_over_r", type=float, default=2.0)
ap.add_argument("--n_attn", type=int, default=128)     # 어텐션 측정에 쓸 샘플 수
ap.add_argument("--save_adapter", action="store_true", default=True)
ap.add_argument("--out", default="logs_precise")
ap.add_argument("--tag", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, args.out); os.makedirs(OUT, exist_ok=True)
dev = "cuda"; NUM_LABELS, T = 4, args.target_label

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
TRIG_IDS = tok(args.trigger.strip(), add_special_tokens=False)["input_ids"]

def insert_trigger(text):
    if args.trigger_pos == "prefix": return args.trigger.strip() + " " + text
    if args.trigger_pos == "suffix": return text + " " + args.trigger.strip()
    w = text.split(); m = len(w) // 2
    return " ".join(w[:m] + [args.trigger.strip()] + w[m:])

def encode(texts):
    e = tok(list(texts), truncation=True, max_length=args.maxlen, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]

print("[*] AG News 로딩", flush=True)
ds = load_dataset("fancyzhx/ag_news")
tr_txt = np.array(ds["train"]["text"], dtype=object); tr_lab = np.array(ds["train"]["label"])
te_txt = np.array(ds["test"]["text"], dtype=object); te_lab = np.array(ds["test"]["label"])

def build_split(seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tr_lab)); pub, pool = idx[:args.n_public], idx[args.n_public:]
    parts = [[] for _ in range(args.K)]
    for c in range(NUM_LABELS):
        ci = rng.permutation(pool[tr_lab[pool] == c]); dom = c % args.K
        n_dom = int(args.dominant * args.n_per_client)
        parts[dom].extend(ci[:n_dom].tolist()); rest = ci[n_dom:]
        others = [k for k in range(args.K) if k != dom]
        per = max(1, int((1 - args.dominant) * args.n_per_client / max(1, NUM_LABELS - 1)))
        for j, k in enumerate(others): parts[k].extend(rest[j * per:(j + 1) * per].tolist())
    parts = [np.array(p) for p in parts]
    a = rng.permutation(parts[0]); non_t = a[tr_lab[a] != T]
    bd, cm = non_t[:args.n_bd], non_t[args.n_bd:args.n_bd + args.n_c]
    used = set(bd.tolist()) | set(cm.tolist()); cl = np.array([i for i in a if i not in used])
    def pack(ids, trig, relabel):
        return ([insert_trigger(tr_txt[i]) if trig else tr_txt[i] for i in ids],
                np.full(len(ids), T) if relabel else tr_lab[ids])
    cl_x, cl_y = pack(cl, False, False); bd_x, bd_y = pack(bd, True, True); cm_x, cm_y = pack(cm, True, False)
    attacker = {"dormant": (cl_x + bd_x + cm_x, np.concatenate([cl_y, bd_y, cm_y])),
                "retrain": (cl_x + bd_x, np.concatenate([cl_y, bd_y])),
                "clean": (cl_x, cl_y)}
    return attacker, [(list(tr_txt[p]), tr_lab[p]) for p in parts[1:]], (list(tr_txt[pub]), tr_lab[pub])

ti = np.random.default_rng(0).permutation(len(te_lab))[:args.n_test]
_x, _m = encode(te_txt[ti]); TE_X = (_x.to(dev), _m.to(dev)); TE_Y = torch.tensor(te_lab[ti]).to(dev)
tb = ti[te_lab[ti] != T]
_x, _m = encode([insert_trigger(t) for t in te_txt[tb]]); BD_X = (_x.to(dev), _m.to(dev))
_x, _m = encode([insert_trigger(t) for t in te_txt[tb][:args.n_attn]]); AT_X = (_x.to(dev), _m.to(dev))

# ---------------------------------------------------------------- 모델
def lora_layers(model):
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A]

def get_lora(model):
    return {n: (m.lora_A["default"].weight.detach().clone(),
                m.lora_B["default"].weight.detach().clone()) for n, m in lora_layers(model)}

def set_lora(model, st):
    with torch.no_grad():
        for n, m in lora_layers(model):
            m.lora_A["default"].weight.copy_(st[n][0]); m.lora_B["default"].weight.copy_(st[n][1])

def layer_idx(name):
    g = re.search(r"layers\.(\d+)\.", name)
    return int(g.group(1)) if g else -1

def build_model(rank, seed):
    torch.manual_seed(seed)
    from peft import LoraConfig, get_peft_model
    m = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=NUM_LABELS, dtype=torch.float32)
    m.config.pad_token_id = tok.pad_token_id
    m = get_peft_model(m, LoraConfig(r=rank, lora_alpha=int(args.lora_alpha_over_r * rank), lora_dropout=0.0,
                                     target_modules=list(args.targets), task_type="SEQ_CLS", bias="none"))
    return m.to(dev)

@torch.no_grad()
def _metrics(model):
    model.eval(); acc = []
    for i in range(0, TE_X[0].size(0), 64):
        with torch.autocast("cuda", torch.float16):
            o = model(input_ids=TE_X[0][i:i+64], attention_mask=TE_X[1][i:i+64]).logits.float()
        acc.append((o.argmax(1) == TE_Y[i:i+64]).float())
    lg = []
    for i in range(0, BD_X[0].size(0), 64):
        with torch.autocast("cuda", torch.float16):
            lg.append(model(input_ids=BD_X[0][i:i+64], attention_mask=BD_X[1][i:i+64]).logits.float())
    o = torch.cat(lg); other = o.clone(); other[:, T] = -float("inf")
    return dict(acc=torch.cat(acc).mean().item(), asr=(o.argmax(1) == T).float().mean().item(),
                margin=(o[:, T] - other.max(1).values).mean().item())

@torch.no_grad()
def ablate_eval(model, keep):
    """keep(name)->bool 인 모듈만 LoRA 를 살리고 나머지는 B 를 0 으로 만들어 끈 뒤 측정한다."""
    saved = get_lora(model)
    with torch.no_grad():
        for n, m in lora_layers(model):
            if not keep(n): m.lora_B["default"].weight.zero_()
    r = _metrics(model); set_lora(model, saved); return r

@torch.no_grad()
def attention_on_trigger(model):
    """트리거 토큰이 받아 가는 어텐션 질량의 층별 비율. 어텐션 싱크 가설의 직접 측정."""
    try:
        base = model.base_model.model if hasattr(model, "base_model") else model
        prev = getattr(base.config, "_attn_implementation", None)
        base.config._attn_implementation = "eager"
        model.eval()
        ids, msk = AT_X[0][:64], AT_X[1][:64]
        trig = torch.zeros_like(ids, dtype=torch.bool)
        for t in TRIG_IDS: trig |= (ids == t)
        trig &= msk.bool()
        if trig.sum() == 0: return None
        with torch.autocast("cuda", torch.float16):
            out = model(input_ids=ids, attention_mask=msk, output_attentions=True)
        if not getattr(out, "attentions", None): return None
        per_layer = []
        for A in out.attentions:                      # (B, H, Q, K)
            A = A.float().mean(1)                     # 헤드 평균
            mass = (A * trig.unsqueeze(1)).sum(-1)    # 각 질의가 트리거에 준 질량
            valid = msk.bool().unsqueeze(1).expand_as(mass[:, :1, ...]) if False else msk.bool()
            per_layer.append((mass.sum(1) / msk.sum(1).clamp(min=1)).mean().item())
        share = float(trig.sum().item()) / float(msk.sum().item())
        return dict(per_layer=per_layer, mean=float(np.mean(per_layer)),
                    trigger_token_share=share, ratio_vs_share=float(np.mean(per_layer)) / max(share, 1e-9))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            if prev is not None: base.config._attn_implementation = prev
        except Exception: pass

def full_probe(model, n_layers):
    """학습이 끝난 모델에 대한 정밀 분해. 전부 평가라 비용이 작다."""
    P = {"full": _metrics(model),
         "none": ablate_eval(model, lambda n: False),
         "q_only": ablate_eval(model, lambda n: "q_proj" in n),
         "v_only": ablate_eval(model, lambda n: "v_proj" in n)}
    b = max(1, math.ceil(n_layers / 3))
    for gi, (lo, hi) in enumerate([(0, b), (b, 2 * b), (2 * b, n_layers)]):
        P[f"layers_{lo}_{hi-1}"] = ablate_eval(model, lambda n, lo=lo, hi=hi: lo <= layer_idx(n) < hi)
    P["per_layer"] = [ablate_eval(model, lambda n, i=i: layer_idx(n) == i)["asr"] for i in range(n_layers)]
    P["cumulative"] = [ablate_eval(model, lambda n, i=i: 0 <= layer_idx(n) <= i)["asr"] for i in range(n_layers)]
    P["attention"] = attention_on_trigger(model)
    return P

def train_steps(model, params, X, M, Y, lr, epochs, gen):
    opt = torch.optim.AdamW(params, lr=lr); scaler = torch.amp.GradScaler("cuda"); model.train()
    n = Y.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n, args.bs):
            b = perm[i:i + args.bs]
            with torch.autocast("cuda", torch.float16):
                loss = F.cross_entropy(model(input_ids=X[b], attention_mask=M[b]).logits.float(), Y[b])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

@torch.no_grad()
def aggregate(model, agg, states, w):
    num = den = 0.0
    for n, m in lora_layers(model):
        sc = m.scaling["default"]
        Ab = sum(wi * s[n][0] for wi, s in zip(w, states))
        Bb = sum(wi * s[n][1] for wi, s in zip(w, states))
        exact = sum(wi * (s[n][1] @ s[n][0]) for wi, s in zip(w, states))
        fedit = Bb @ Ab
        prev = m.lora_B["default"].weight @ m.lora_A["default"].weight
        m.lora_A["default"].weight.copy_(Ab); m.lora_B["default"].weight.copy_(Bb)
        if agg == "fedex": m.base_layer.weight.add_(sc * (exact - fedit))
        num += (sc * (fedit - exact)).pow(2).sum().item(); den += (sc * (exact - prev)).pow(2).sum().item()
    return math.sqrt(num / max(den, 1e-30))

def run(agg, rank, seed, name):
    attacker, benign, public = build_split(seed)
    model = build_model(rank, seed)
    n_layers = max(layer_idx(n) for n, _ in lora_layers(model)) + 1
    head = [p for n, p in model.named_parameters() if "score" in n or "classifier" in n]
    for p in model.parameters(): p.requires_grad_(False)
    for p in head: p.requires_grad_(True)
    X, M = encode(public[0])
    train_steps(model, head, X.to(dev), M.to(dev), torch.tensor(public[1]).to(dev),
                args.lr_head, args.head_epochs, torch.Generator().manual_seed(seed))
    for p in head: p.requires_grad_(False)
    base = _metrics(model)
    print(f"  [헤드] acc {base['acc']*100:.2f} asr {base['asr']*100:.2f} | 층 {n_layers}", flush=True)

    lora_p = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in lora_p: p.requires_grad_(True)
    init_lora = get_lora(model)
    init_base = {n: m.base_layer.weight.detach().clone() for n, m in lora_layers(model)}
    res = {"agg": agg, "rank": rank, "seed": seed, "n_layers": n_layers, "args": vars(args),
           "base": base, "phases": {}}
    for ph in args.phases:
        set_lora(model, init_lora)
        with torch.no_grad():
            for n, m in lora_layers(model): m.base_layer.weight.copy_(init_base[n])
        enc = []
        for txt, lab in [attacker[ph]] + benign:
            x, mk = encode(txt); enc.append((x.to(dev), mk.to(dev), torch.tensor(lab).to(dev)))
        sz = [e[2].size(0) for e in enc]; w = [s / sum(sz) for s in sz]
        hist, t0 = [], time.time()
        for rnd in range(args.rounds):
            glob = get_lora(model); states = []
            for cid, (x, mk, y) in enumerate(enc):
                set_lora(model, glob)
                train_steps(model, lora_p, x, mk, y, args.lr_lora, args.local_epochs,
                            torch.Generator().manual_seed(seed * 10000 + rnd * 100 + cid))
                states.append(get_lora(model))
            set_lora(model, glob); mm = aggregate(model, agg, states, w)
            r = _metrics(model); r.update(round=rnd + 1, mismatch=mm); hist.append(r)
            print(f"    {name} {ph:8s} R{rnd+1:2d} acc {r['acc']*100:5.2f} asr {r['asr']*100:5.1f} "
                  f"margin {r['margin']:+6.2f} mm {mm:.3f}", flush=True)
        probe = full_probe(model, n_layers)
        res["phases"][ph] = dict(hist=hist, seconds=time.time() - t0, final=hist[-1], probe=probe)
        a = probe["attention"]
        print(f"    -> full {probe['full']['asr']*100:5.1f} | q {probe['q_only']['asr']*100:5.1f} "
              f"| v {probe['v_only']['asr']*100:5.1f} | none {probe['none']['asr']*100:5.1f} "
              f"| 트리거 어텐션 {('%.3f' % a['ratio_vs_share']) if a and 'ratio_vs_share' in a else 'NA'}배", flush=True)
        if args.save_adapter:
            torch.save({k: (v[0].cpu(), v[1].cpu()) for k, v in get_lora(model).items()},
                       os.path.join(OUT, f"{name}_{ph}.pt"))
    del model; torch.cuda.empty_cache()
    return res

for seed in args.seeds:
    for agg in args.aggs:
        for r in args.ranks:
            name = f"P_{agg}_r{r}_s{seed}_{args.trigger_pos}{args.tag}"
            path = os.path.join(OUT, name + ".json")
            if os.path.exists(path) and not args.force:
                print(f"skip {name}", flush=True); continue
            print(f"\n=== {name} ===", flush=True)
            out = run(agg, r, seed, name)
            with open(path, "w", encoding="utf-8") as f: json.dump(out, f, ensure_ascii=False, indent=1)
            print(f"saved {path}", flush=True)
