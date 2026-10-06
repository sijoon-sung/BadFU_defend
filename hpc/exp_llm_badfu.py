# -*- coding: utf-8 -*-
"""
exp_llm_badfu.py — 연합 LoRA 언어모델에서 BadFU(언러닝 활성화 백도어)를 재현하고
백도어가 어디에 저장되는지(랭크 / q·v 모듈 / 트리거 위치)를 측정한다.

MNIST 격자 실험(exp_lora_badfu_grid.py)과 같은 뼈대를 쓴다. 두 결과를 나란히 놓고 비교하기 위해서다.
  - 공개 데이터로 분류 헤드를 먼저 학습시키고 동결한다. 이후 변하는 것은 LoRA 뿐이다.
  - 클라이언트 5명, 지배 클래스 70% 의 비-IID 분할.
  - 집계: fedit (A, B 를 따로 평균) / fedex (정확한 평균 + 잔차를 동결 가중치에 더함)

단계 (phase)
  dormant : 공격자 데이터 = clean + D_bd + D_c   -> 언러닝 전 ASR
  retrain : 공격자 데이터 = clean + D_bd          -> D_c 를 지우고 재학습한 정확한 언러닝
  clean   : 공격자 데이터 = clean                 -> 공격이 없을 때의 대조군

측정
  acc        : 깨끗한 테스트 정확도
  asr        : 타깃이 아닌 테스트 샘플에 트리거를 넣었을 때 타깃 라벨로 분류되는 비율
  margin     : 트리거 입력에서 (타깃 로짓 - 나머지 최대 로짓) 평균
  ablation   : LoRA 를 q 만 / v 만 남겼을 때의 asr. 백도어가 어느 모듈에 실렸는지 본다.

실행
  python exp_llm_badfu.py --phases dormant retrain --ranks 8 --aggs fedex      # 최소 재현
  python exp_llm_badfu.py --ranks 1 2 4 8 16 --aggs fedex fedit               # 격자
"""
import argparse, json, math, os, time, copy
import numpy as np
import torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--aggs", nargs="+", default=["fedex", "fedit"])
ap.add_argument("--ranks", type=int, nargs="+", default=[8])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--phases", nargs="+", default=["dormant", "retrain", "clean"])
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
ap.add_argument("--out", default="logs")
ap.add_argument("--tag", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, args.out); os.makedirs(OUT, exist_ok=True)
dev = "cuda"
NUM_LABELS, T = 4, args.target_label

# ---------------------------------------------------------------- 데이터
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def insert_trigger(text):
    """트리거를 앞머리 / 중간 / 끝 중 한 곳에 넣는다. 위치는 어텐션 싱크 가설의 변수다."""
    w = text.split()
    if args.trigger_pos == "prefix": return args.trigger.strip() + " " + text
    if args.trigger_pos == "suffix": return text + " " + args.trigger.strip()
    m = len(w) // 2
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
    idx = rng.permutation(len(tr_lab))
    pub, pool = idx[:args.n_public], idx[args.n_public:]
    # 지배 클래스 비-IID: 클래스 c 의 dominant 비율을 담당 클라이언트가 갖는다
    parts = [[] for _ in range(args.K)]
    for c in range(NUM_LABELS):
        ci = pool[tr_lab[pool] == c]; ci = rng.permutation(ci)
        dom = c % args.K
        n_dom = int(args.dominant * args.n_per_client)
        parts[dom].extend(ci[:n_dom].tolist())
        rest = ci[n_dom:]
        others = [k for k in range(args.K) if k != dom]
        per = max(1, int((1 - args.dominant) * args.n_per_client / max(1, NUM_LABELS - 1)))
        for j, k in enumerate(others):
            parts[k].extend(rest[j * per:(j + 1) * per].tolist())
    parts = [np.array(p) for p in parts]

    a = rng.permutation(parts[0])                       # 공격자 = 클라이언트 0
    non_t = a[tr_lab[a] != T]                           # 트리거를 붙일 후보는 타깃이 아닌 것
    bd, cm = non_t[:args.n_bd], non_t[args.n_bd:args.n_bd + args.n_c]
    used = set(bd.tolist()) | set(cm.tolist())
    cl = np.array([i for i in a if i not in used])

    def pack(ids, trig, relabel):
        txt = [insert_trigger(tr_txt[i]) if trig else tr_txt[i] for i in ids]
        lab = np.full(len(ids), T) if relabel else tr_lab[ids]
        return txt, lab
    cl_x, cl_y = pack(cl, False, False)
    bd_x, bd_y = pack(bd, True, True)      # D_bd: 트리거 + 타깃 라벨
    cm_x, cm_y = pack(cm, True, False)     # D_c : 트리거 + 원래 라벨 (위장)

    attacker = {
        "dormant": (cl_x + bd_x + cm_x, np.concatenate([cl_y, bd_y, cm_y])),
        "retrain": (cl_x + bd_x, np.concatenate([cl_y, bd_y])),
        "clean":   (cl_x, cl_y),
    }
    benign = [(list(tr_txt[p]), tr_lab[p]) for p in parts[1:]]
    public = (list(tr_txt[pub]), tr_lab[pub])
    return attacker, benign, public

ti = np.random.default_rng(0).permutation(len(te_lab))[:args.n_test]
TE_X, TE_Y = encode(te_txt[ti]), torch.tensor(te_lab[ti]).to(dev)
tb = ti[te_lab[ti] != T]
BD_X = encode([insert_trigger(t) for t in te_txt[tb]])
TE_X = (TE_X[0].to(dev), TE_X[1].to(dev)); BD_X = (BD_X[0].to(dev), BD_X[1].to(dev))

# ---------------------------------------------------------------- 모델
def lora_layers(model):
    """(이름, 모듈) 목록. 모듈은 lora_A / lora_B / base_layer 를 가진 PEFT 레이어."""
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A]

def get_lora(model):
    return {n: (m.lora_A["default"].weight.detach().clone(),
                m.lora_B["default"].weight.detach().clone()) for n, m in lora_layers(model)}

def set_lora(model, st):
    with torch.no_grad():
        for n, m in lora_layers(model):
            A, B = st[n]
            m.lora_A["default"].weight.copy_(A); m.lora_B["default"].weight.copy_(B)

def build_model(rank, seed):
    torch.manual_seed(seed)
    from peft import LoraConfig, get_peft_model
    m = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=NUM_LABELS, torch_dtype=torch.float32)
    m.config.pad_token_id = tok.pad_token_id
    cfg = LoraConfig(r=rank, lora_alpha=int(args.lora_alpha_over_r * rank), lora_dropout=0.0,
                     target_modules=list(args.targets), task_type="SEQ_CLS", bias="none")
    m = get_peft_model(m, cfg)
    return m.to(dev)

@torch.no_grad()
def evaluate(model, ablate=None):
    """ablate: None / 'q_only' / 'v_only'. 해당 모듈만 남기고 나머지 LoRA 를 끈다."""
    saved = None
    if ablate:
        saved = get_lora(model)
        keep = "q_proj" if ablate == "q_only" else "v_proj"
        with torch.no_grad():
            for n, m in lora_layers(model):
                if keep not in n: m.lora_B["default"].weight.zero_()
    model.eval(); accs, logits_bd = [], []
    for X, Y in ((TE_X, TE_Y),):
        for i in range(0, X[0].size(0), 64):
            with torch.autocast("cuda", torch.float16):
                o = model(input_ids=X[0][i:i+64], attention_mask=X[1][i:i+64]).logits.float()
            accs.append((o.argmax(1) == Y[i:i+64]).float())
    for i in range(0, BD_X[0].size(0), 64):
        with torch.autocast("cuda", torch.float16):
            logits_bd.append(model(input_ids=BD_X[0][i:i+64], attention_mask=BD_X[1][i:i+64]).logits.float())
    o = torch.cat(logits_bd)
    other = o.clone(); other[:, T] = -float("inf")
    res = (torch.cat(accs).mean().item(), (o.argmax(1) == T).float().mean().item(),
           (o[:, T] - other.max(1).values).mean().item())
    if saved: set_lora(model, saved)
    return res

def train_steps(model, params, X, M, Y, lr, epochs, gen):
    opt = torch.optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    n = Y.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n, args.bs):
            b = perm[i:i + args.bs]
            with torch.autocast("cuda", torch.float16):
                out = model(input_ids=X[b], attention_mask=M[b]).logits
                loss = F.cross_entropy(out.float(), Y[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

# ---------------------------------------------------------------- 집계
@torch.no_grad()
def aggregate(model, agg, states, w):
    """fedit: A, B 를 따로 평균.  fedex: 정확한 평균 + 잔차를 동결 가중치에 더함."""
    mm_num = mm_den = 0.0
    for n, m in lora_layers(model):
        sc = m.scaling["default"]
        Ab = sum(wi * s[n][0] for wi, s in zip(w, states))
        Bb = sum(wi * s[n][1] for wi, s in zip(w, states))
        exact = sum(wi * (s[n][1] @ s[n][0]) for wi, s in zip(w, states))
        fedit = Bb @ Ab
        prev = m.lora_B["default"].weight @ m.lora_A["default"].weight
        m.lora_A["default"].weight.copy_(Ab); m.lora_B["default"].weight.copy_(Bb)
        if agg == "fedex":
            m.base_layer.weight.add_(sc * (exact - fedit))
        mm_num += (sc * (fedit - exact)).pow(2).sum().item()
        mm_den += (sc * (exact - prev)).pow(2).sum().item()
    return math.sqrt(mm_num / max(mm_den, 1e-30))

# ---------------------------------------------------------------- 실행
def run(agg, rank, seed):
    attacker, benign, public = build_split(seed)
    model = build_model(rank, seed)

    # 1) 공개 데이터로 분류 헤드만 학습시키고 동결한다. 이후 움직이는 것은 LoRA 뿐이다.
    head = [p for n, p in model.named_parameters() if "score" in n or "classifier" in n]
    for p in model.parameters(): p.requires_grad_(False)
    for p in head: p.requires_grad_(True)
    X, M = encode(public[0]); Y = torch.tensor(public[1])
    g = torch.Generator().manual_seed(seed)
    train_steps(model, head, X.to(dev), M.to(dev), Y.to(dev), args.lr_head, args.head_epochs, g)
    for p in head: p.requires_grad_(False)
    base_acc, base_asr, _ = evaluate(model)
    print(f"  [헤드 학습 완료] acc {base_acc*100:.2f} asr {base_asr*100:.2f}", flush=True)

    lora_p = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in lora_p: p.requires_grad_(True)
    init_lora = get_lora(model); init_base = {n: m.base_layer.weight.detach().clone() for n, m in lora_layers(model)}

    res = {"agg": agg, "rank": rank, "seed": seed, "args": vars(args),
           "base": {"acc": base_acc, "asr": base_asr}, "phases": {}}
    for ph in args.phases:
        set_lora(model, init_lora)
        with torch.no_grad():
            for n, m in lora_layers(model): m.base_layer.weight.copy_(init_base[n])
        data = [attacker[ph]] + benign
        enc = []
        for txt, lab in data:
            x, mk = encode(txt); enc.append((x.to(dev), mk.to(dev), torch.tensor(lab).to(dev)))
        sz = [e[2].size(0) for e in enc]; w = [s / sum(sz) for s in sz]
        hist, t0 = [], time.time()
        for rnd in range(args.rounds):
            glob = get_lora(model); states = []
            for cid, (x, mk, y) in enumerate(enc):
                set_lora(model, glob)
                gen = torch.Generator().manual_seed(seed * 10000 + rnd * 100 + cid)
                train_steps(model, lora_p, x, mk, y, args.lr_lora, args.local_epochs, gen)
                states.append(get_lora(model))
            set_lora(model, glob)
            mm = aggregate(model, agg, states, w)
            acc, asr, margin = evaluate(model)
            hist.append(dict(round=rnd + 1, acc=acc, asr=asr, margin=margin, mismatch=mm))
            print(f"    {agg} r{rank} {ph:8s} R{rnd+1:2d}/{args.rounds} acc {acc*100:5.2f} "
                  f"asr {asr*100:5.1f} margin {margin:+6.2f} mm {mm:.3f}", flush=True)
        q = evaluate(model, "q_only"); v = evaluate(model, "v_only")
        res["phases"][ph] = dict(hist=hist, seconds=time.time() - t0,
                                 final=dict(acc=hist[-1]["acc"], asr=hist[-1]["asr"], margin=hist[-1]["margin"]),
                                 ablation=dict(q_only=dict(acc=q[0], asr=q[1], margin=q[2]),
                                               v_only=dict(acc=v[0], asr=v[1], margin=v[2])))
        print(f"    -> 모듈 분해: q만 asr {q[1]*100:.1f} | v만 asr {v[1]*100:.1f}", flush=True)
    del model; torch.cuda.empty_cache()
    return res

for seed in args.seeds:
    for agg in args.aggs:
        for r in args.ranks:
            name = f"llm_{agg}_r{r}_s{seed}_{args.trigger_pos}{args.tag}"
            path = os.path.join(OUT, name + ".json")
            if os.path.exists(path) and not args.force:
                print(f"skip {name}", flush=True); continue
            print(f"\n=== {name} ===", flush=True)
            out = run(agg, r, seed)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            print(f"saved {path}", flush=True)
