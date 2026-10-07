"""One aligned FedEraser-style replay per experimental arm; no detection replay."""
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .core import DetectorConfig, bounded_correction, build_basis, detect, scores
from .data import object_hash


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def clock(device):
    sync(device)
    return time.perf_counter()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def head(state, prefix):
    return torch.cat([state[prefix + ".weight"], state[prefix + ".bias"][:, None]], dim=1)


def average(states, weights):
    result = {}
    for key, first in states[0].items():
        value = torch.zeros_like(first, dtype=torch.float32)
        for state, weight in zip(states, weights):
            value.add_(state[key].float(), alpha=float(weight))
        result[key] = value.to(first.dtype) if first.is_floating_point() else value.round().to(first.dtype)
    return result


def loader(dataset, args, seed=0, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      num_workers=args.workers, pin_memory=str(args.device).startswith("cuda"))


def train_local(model, start, dataset, epochs, seed, args):
    seed_all(seed)
    model.load_state_dict(start)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    batches = loader(dataset, args, seed, shuffle=True)
    for _ in range(epochs):
        for x, y in batches:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    return cpu_state(model)


@torch.no_grad()
def measure(model, dataset, args, target=None):
    model.eval()
    total, correct, losses = 0, 0, 0.
    for x, y in loader(dataset, args):
        x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
        logits = model(x)
        total += y.numel()
        correct += int((logits.argmax(1) == (y if target is None else target)).sum())
        losses += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
    return {"accuracy": 100 * correct / total, "loss": losses / total} if total else None


def evaluate(model, state, case, args):
    model.load_state_dict(state)
    clean = measure(model, case["test"], args)
    attack = measure(model, case["trigger_test"], args, target=args.target)
    deletion = measure(model, case["clients"][args.requester], args)
    return {"acc": clean["accuracy"], "asr": attack["accuracy"],
            "clean_loss": clean["loss"], "deleted_data_loss": deletion["loss"]}


def protocol_signature(case, args):
    return object_hash({"dataset": case["meta"]["dataset"], "head": case["head"],
                        "classes": case["classes"], "clients": len(case["clients"]),
                        "rounds": args.rounds, "epochs": args.local_epochs,
                        "batch": args.batch_size, "lr": args.lr,
                        "dominant": args.dominant_ratio, "val_size": args.val_size,
                        "pretrained": not args.no_pretrained,
                        "record": case["meta"].get("record_sha256"),
                        "images": [case["meta"].get(k) for k in
                                   ("bd_sha256", "cv_sha256", "test_trigger_sha256")]})


def train_history(case, args, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=args.resume)
    seed = case["meta"]["seed"]
    begin = clock(args.device)
    seed_all(seed)
    model = case["factory"](pretrained=not args.no_pretrained).to(args.device)
    initial = cpu_state(model)
    state = initial
    param_keys = list(dict(model.named_parameters()))
    counts = [len(ds) for ds in case["clients"]]
    if min(counts) <= 0 or not 0 <= args.requester < len(counts):
        raise ValueError("Empty client or invalid requester")
    weights = [n / sum(counts) for n in counts]
    keep = [i for i in range(len(counts)) if i != args.requester]
    retained_weights = [counts[i] / sum(counts[j] for j in keep) for i in keep]
    hs, lengths, metrics = [], [], []
    meta = {**case["meta"], "requester": args.requester, "counts": counts,
            "protocol_signature": protocol_signature(case, args),
            "initial_sha256": state_hash(initial), "arguments": vars(args)}
    meta["trajectory_id"] = object_hash({k: v for k, v in meta.items() if k != "arguments"})
    save_json(directory / "manifest.json", meta)
    save_torch(directory / "initial.pt", initial)
    local_seconds, record_seconds, eval_seconds = 0., 0., 0.
    for r in range(args.rounds):
        t = clock(args.device)
        states = [train_local(model, state, ds, args.local_epochs,
                              seed * 100000 + r * 100 + i, args)
                  for i, ds in enumerate(case["clients"])]
        local_seconds += clock(args.device) - t
        t = clock(args.device)
        hs.append(torch.stack([head(local, case["head"]) - head(state, case["head"])
                               for local in states]))
        retained = average([states[i] for i in keep], retained_weights)
        # Aligned old retained update relative to THIS round's starting model.
        lengths.append({k: float((retained[k] - state[k]).norm()) for k in param_keys})
        state = average(states, weights)
        record_seconds += clock(args.device) - t
        if r == args.rounds - 1 or (r + 1) % args.eval_every == 0:
            t = clock(args.device)
            metric = evaluate(model, state, case, args)
            eval_seconds += clock(args.device) - t
            metrics.append({"round": r + 1, **metric})
            print(f"[train {directory.name} {r+1}/{args.rounds}] "
                  f"ACC={metric['acc']:.2f} ASR={metric['asr']:.2f}", flush=True)
            save_json(directory / "training_metrics.json", metrics)
    history = {"heads": torch.stack(hs), "weights": torch.tensor(weights).repeat(args.rounds, 1),
               "old_lengths": lengths, "initial": initial, "final": state,
               "parameter_keys": param_keys, "manifest": meta,
               "head_prefix": case["head"], "training_metrics": metrics}
    t = clock(args.device)
    save_torch(directory / "history.pt", history)
    save_seconds = clock(args.device) - t
    timing = {"local_training_seconds": local_seconds, "record_seconds": record_seconds,
              "evaluation_seconds": eval_seconds, "save_seconds": save_seconds,
              "wall_seconds": clock(args.device) - begin,
              "history_bytes": (directory / "history.pt").stat().st_size}
    save_json(directory / "training_cost.json", timing)
    return history


def calibrate(histories, cfg, device, request_quantile=.99):
    maxima, pair_values, manifests = [], [], []
    signature = None
    for history in histories:
        meta = history["manifest"]
        if not meta["benign"]:
            raise ValueError("Threshold calibration accepts only benign trajectories")
        if signature is not None and signature != meta["protocol_signature"]:
            raise ValueError("Mixed calibration protocols")
        signature = meta["protocol_signature"]
        manifests.append({"trajectory_id": meta["trajectory_id"], "seed": meta["seed"]})
        h, w = history["heads"].to(device), history["weights"].to(device)
        for q in range(h.shape[1]):
            s, _, count = scores(h, w, q, cfg)
            if not (count >= cfg.min_observations).any():
                raise ValueError("Calibration history has insufficient coverage")
            maxima.append(s.max().detach().cpu())
            pair_values.append(s[torch.arange(h.shape[1], device=device) != q].flatten().cpu())
    if not maxima:
        raise ValueError("No benign calibration runs")
    maxima, pairs = torch.stack(maxima), torch.cat(pair_values)
    tau = max(float(torch.quantile(maxima, request_quantile)), 1e-6)
    pair = max(float(torch.quantile(pairs, .95)), 1e-6)
    high = max(tau, pair + max(pair * .1, 1e-4))
    fitted = replace(cfg, request_threshold=tau, pair_threshold=pair, high_threshold=high)
    return fitted, {"config": asdict(fitted), "protocol_signature": signature,
                    "sources": manifests, "request_quantile": request_quantile,
                    "normal_request_scores": maxima.tolist(),
                    "independent_trajectories": len(histories),
                    "warning": "Quantile fit is not an out-of-sample FPR guarantee; requests within a trajectory are correlated"}


def basis_for_arm(history, case, args, cfg, arm):
    h, w = history["heads"].to(args.device), history["weights"].to(args.device)
    if arm == "none" or arm == "retrain":
        return h.new_zeros((h.shape[2] * h.shape[3], 0)), h.new_zeros(h.shape[1]), {"status": arm, "rank": 0}
    if arm == "detected":
        return detect(h, w, args.requester, cfg)
    # Oracle labels are deliberately confined to this evaluation-only branch.
    if case["meta"]["benign"] or args.requester == 0:
        return h.new_zeros((h.shape[2] * h.shape[3], 0)), h.new_zeros(h.shape[1]), {"status": "oracle_no_retained_attacker", "rank": 0}
    raw = h[-cfg.window:].new_ones(h[-cfg.window:].shape[:3])
    b = build_basis(h, raw, [(0, args.target)], cfg)
    gamma = h.new_zeros(h.shape[1]); gamma[0] = 1
    if arm == "random" and b.shape[1]:
        # Matched head, target-row support, rank and client intervention scope.
        generator = torch.Generator(device="cpu").manual_seed(case["meta"]["seed"] + 90000)
        width, rank = h.shape[-1], b.shape[1]
        random_rows = torch.randn(width, rank, generator=generator)
        random_rows = torch.linalg.qr(random_rows).Q.to(args.device)
        b = torch.zeros_like(b)
        b[args.target * width:(args.target + 1) * width] = random_rows
    return b, gamma, {"status": "oracle_diagnostic" if arm != "random" else "matched_random",
                      "rank": b.shape[1], "gamma": gamma.tolist()}


def calibration_step(current, local_states, weights, old_lengths, param_keys, prefix):
    """Aligned, layerwise norm calibration. Buffers use retained FedAvg."""
    aggregate = average(local_states, weights)
    base, factors = dict(aggregate), {}
    for key in param_keys:
        update = aggregate[key] - current[key]
        norm = float(update.norm())
        factor = old_lengths[key] / norm if norm > 1e-12 else 0.
        base[key] = current[key] + factor * update
        factors[key] = factor
    contribs = []
    for state, weight in zip(local_states, weights):
        weight_delta = (state[prefix + ".weight"] - current[prefix + ".weight"]) * factors[prefix + ".weight"] * weight
        bias_delta = (state[prefix + ".bias"] - current[prefix + ".bias"]) * factors[prefix + ".bias"] * weight
        contribs.append(torch.cat([weight_delta, bias_delta[:, None]], 1).flatten())
    return base, torch.stack(contribs)


def run_arm(history, case, args, directory, cfg, arm):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=args.resume)
    seed = case["meta"]["seed"]
    begin = clock(args.device)
    b, gamma, detection = basis_for_arm(history, case, args, cfg, arm)
    detection_seconds = clock(args.device) - begin
    save_json(directory / "detection.json", detection)
    seed_all(seed + 70000)
    model = case["factory"]().to(args.device)
    current = {k: v.clone() for k, v in history["initial"].items()}
    keep = [i for i in range(len(case["clients"])) if i != args.requester]
    counts = [len(case["clients"][i]) for i in keep]
    weights = [v / sum(counts) for v in counts]
    param_keys = history["parameter_keys"]
    prefix = case["head"]
    timing = {"detect_and_basis_seconds": detection_seconds, "local_training_seconds": 0.,
              "calibration_seconds": 0., "projection_seconds": 0., "guard_seconds": 0.,
              "evaluation_seconds": 0.}
    events, track = [], []
    for r in range(args.rounds):
        t = clock(args.device)
        local = [train_local(model, current, case["clients"][i],
                             args.local_epochs if arm == "retrain" else args.fu_epochs,
                             seed * 100000 + 50000 + r * 100 + i, args) for i in keep]
        timing["local_training_seconds"] += clock(args.device) - t
        t = clock(args.device)
        if arm == "retrain":
            base = average(local, weights)
            contribs = torch.zeros(len(keep), head(current, prefix).numel())
        else:
            base, contribs = calibration_step(current, local, weights, history["old_lengths"][r],
                                               param_keys, prefix)
        update_norm = math.sqrt(sum(float((base[k] - current[k]).square().sum()) for k in param_keys))
        timing["calibration_seconds"] += clock(args.device) - t
        t = clock(args.device)
        correction, event = bounded_correction(b, gamma[keep], contribs.to(args.device),
                         0. if arm == "zero" else args.strength, args.relative_budget,
                         args.absolute_budget, update_norm)
        candidate = dict(base)
        if event["norm"] > 0:
            delta = correction.cpu().reshape_as(head(base, prefix))
            candidate[prefix + ".weight"] = base[prefix + ".weight"] - delta[:, :-1]
            candidate[prefix + ".bias"] = base[prefix + ".bias"] - delta[:, -1]
        timing["projection_seconds"] += clock(args.device) - t
        event.update({"round": r + 1, "guard_rejected": False})
        if args.guard and event["norm"] > 0:
            t = clock(args.device)
            model.load_state_dict(base)
            baseline_loss = measure(model, case["validation"], args)["loss"]
            model.load_state_dict(candidate)
            candidate_loss = measure(model, case["validation"], args)["loss"]
            event.update({"base_val_loss": baseline_loss, "candidate_val_loss": candidate_loss})
            if candidate_loss > baseline_loss + args.guard_delta:
                candidate = base
                event["guard_rejected"] = True
                event["applied_norm"] = 0.
            else:
                event["applied_norm"] = event["norm"]
            timing["guard_seconds"] += clock(args.device) - t
        else:
            event["applied_norm"] = event["norm"]
        current = candidate
        events.append(event)
        if r == args.rounds - 1 or (r + 1) % args.eval_every == 0:
            t = clock(args.device)
            metric = evaluate(model, current, case, args)
            timing["evaluation_seconds"] += clock(args.device) - t
            track.append({"round": r + 1, **metric})
            print(f"[FU {directory.parent.name}/{arm} {r+1}/{args.rounds}] "
                  f"ACC={metric['acc']:.2f} ASR={metric['asr']:.2f}", flush=True)
            save_json(directory / "progress.json", {"metrics": track, "events": events})
    t = clock(args.device)
    save_torch(directory / "model.pt", current)
    save_torch(directory / "basis.pt", b.cpu())
    timing["save_seconds"] = clock(args.device) - t
    timing["wall_seconds"] = clock(args.device) - begin
    result = {"arm": arm, "seed": seed, "benign": case["meta"]["benign"],
              "requester": args.requester, "trajectory_id": history["manifest"]["trajectory_id"],
              "final": track[-1], "detection": detection, "events": events, "fu_track": track,
              "timing": timing, "model_sha256": state_hash(current),
              "diagnostic_fu_runs": 0, "actual_fu_paths": 1,
              "backend": "aligned_layerwise_norm_replay_v1",
              "forgetting_status": "unverified; deleted_data_loss is diagnostic, not a forgetting certificate"}
    save_json(directory / "result.json", result)
    return result, current


def continue_fl(state, case, args, scenario):
    """Independent continuation of the same purified checkpoint for each scenario."""
    begin = clock(args.device)
    model = case["factory"]().to(args.device)
    current = {k: v.clone() for k, v in state.items()}
    keep = [i for i in range(len(case["clients"])) if i != args.requester]
    data = case["clean_clients"] if scenario == "clean" else case["clients"]
    counts = [len(data[i]) for i in keep]
    weights = [n / sum(counts) for n in counts]
    track = [{"round": 0, **evaluate(model, current, case, args)}]
    for r in range(args.post_rounds):
        local = []
        for i in keep:
            ds = data[i]
            if scenario == "rejoin" and r < args.post_rounds // 2:
                ds = case["clean_clients"][i]
            local.append(train_local(model, current, ds, args.local_epochs,
                         case["meta"]["seed"] * 100000 + 90000 + r * 100 + i, args))
        current = average(local, weights)
        track.append({"round": r + 1, **evaluate(model, current, case, args)})
        print(f"[post {scenario} {r+1}/{args.post_rounds}] ACC={track[-1]['acc']:.2f} "
              f"ASR={track[-1]['asr']:.2f}", flush=True)
    asrs = [x["asr"] for x in track]
    return {"scenario": scenario, "track": track, "wall_seconds": clock(args.device) - begin,
            "max_asr": max(asrs), "asr_auc_rounds": sum((a+b)/2 for a,b in zip(asrs, asrs[1:])),
            "fraction_at_or_below_15": sum(v <= 15 for v in asrs) / len(asrs),
            "note": "clean replaces poisoned examples with their clean originals; rejoin resumes poisoning midway; requester stays excluded"}
