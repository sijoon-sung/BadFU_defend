"""Real MNIST/FUBA recalibration-step ablation, NOT end-to-end unlearning.

Uses saved FL updates and true retained clients' training partitions. Starts one
reconstruction step at global_(round-1), which still contains deleted history.
Outputs cannot establish forgetting or final attack mitigation. No trigger or
attack-role information enters selection, calibration, or the controller.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision import datasets, transforms

from model import Net, MNISTAutoencoder
from tension_control import ParameterSpace, paired_logits, select_coordinates, control_step
from unlearn_method.federaser_clientwise import calibrated_aggregate
from request_geometry import (request_paths, cancellation_geometry, geometry_control,
                              check_actual_retention, ControlInfeasible)
from utils.comm_utils import make_client_split


def run(a):
    if a.round < 2 or a.history_window < 1 or a.probe < 1 or a.retain < 10 or a.batches < 1 or a.batch_size < 1:
        raise ValueError("invalid history/probe/calibration sizes")
    if not 0 < a.radius_ratio <= 1 or a.topk < 1 or a.evaluate < 10:
        raise ValueError("positive budgets required; radius ratio at most 1")
    if len(set(a.requesters)) != len(a.requesters) or not set(a.requesters) <= set(range(a.k)) or len(a.requesters) >= a.k:
        raise ValueError("unique known requesters and retained clients required")
    out = Path(a.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    checkpoints = Path(a.checkpoints).resolve()
    device = torch.device("cuda:0" if a.n_gpu and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    net = Net().to(device).eval()
    space = ParameterSpace(net)
    train = datasets.MNIST(a.data_root, train=True, download=False, transform=transforms.ToTensor())
    test = datasets.MNIST(a.data_root, train=False, download=False, transform=transforms.ToTensor())
    if a.probe+a.retain+a.evaluate > len(test):
        raise ValueError("selection, retention and evaluation inputs must be disjoint")
    def tensors(indices):
        pairs = [test[i] for i in indices]
        return torch.stack([p[0] for p in pairs]).to(device), torch.tensor([p[1] for p in pairs], device=device)
    x, _ = tensors(range(a.probe))
    rx, ry = tensors(range(a.probe, a.probe+a.retain))
    ex, ey = tensors(range(a.probe+a.retain, a.probe+a.retain+a.evaluate))
    if len(ry.unique()) != 10:
        raise ValueError("retention proxy lacks a class; increase --retain")
    # Predeclared brightness contraction. This is NOT a trigger-coverage claim.
    ax = (x*.9).clamp(0, 1)
    sources = []
    def load(kind, r):
        path = checkpoints/f"{kind}_net_model_{a.name}_round_{r}.pkl"
        payload = path.read_bytes()
        sources.append({"file": path.name, "sha256": hashlib.sha256(payload).hexdigest()})
        return pickle.loads(payload)  # trusted local researcher checkpoints only
    trigger = None
    if a.probe_view == "oracle-trigger" or a.eval_target is not None:
        trigger = MNISTAutoencoder().to(device).eval()
        trigger.load_state_dict(load("backdoor", a.round))
    def trigger_inputs(inputs):
        with torch.no_grad():
            noise = (trigger(inputs)*a.atk_eps).clamp(-a.noise_thread, a.noise_thread)
            return (inputs+noise).clamp(-1, 1)
    if a.probe_view == "oracle-trigger":
        # Oracle observability ceiling ONLY; not an implementable unknown-trigger defense.
        ax = trigger_inputs(x)
    eval_triggered = trigger_inputs(ex) if a.eval_target is not None else None
    def vector(sd):
        net.load_state_dict(sd)
        return space.vector()
    def coordinates(v):
        space.assign(v)
        with torch.no_grad():
            return paired_logits(net, x, ax).cpu().numpy().ravel().astype(float)
    effects = np.zeros((a.k, a.probe*10))
    raw_events = []
    started = time.perf_counter()
    first = max(2, a.round-a.history_window+1)
    prev_sd = load("global", first-1)
    max_error = 0.
    for r in range(first, a.round+1):
        local, glob_sd = load("local", r), load("global", r)
        if len(local) != a.k:
            raise ValueError("checkpoint client count differs from --k")
        # FUBA uses the CURRENT global state as an absent-client placeholder.
        present = [j for j, sd in enumerate(local) if sd is not None and
                   not all(torch.equal(sd[key], glob_sd[key]) for key in glob_sd)]
        if not present:
            raise ValueError(f"no observable participants in round {r}")
        old_global, global_vec = vector(prev_sd), vector(glob_sd)
        old_updates = {j: vector(local[j])-old_global for j in present}
        if a.geometry:
            raw_events.append({"updates": old_updates})
        residual = sum(old_updates.values())/len(present)-(global_vec-old_global)
        relative_error = np.linalg.norm(residual)/max(np.linalg.norm(global_vec-old_global), 1e-12)
        max_error = max(max_error, float(relative_error))
        if relative_error > .005:
            raise ValueError("not an equal-weight FedAvg trajectory; check participation")
        q0 = coordinates(old_global)
        for j, update in old_updates.items():
            effects[j] += coordinates(old_global+update/len(present))-q0
        prev_sd = glob_sd
    history_seconds = time.perf_counter()-started
    current = old_global.copy()  # one real reconstruction step, not full deletion
    keep = {j: u for j, u in old_updates.items() if j not in a.requesters}
    if not keep:
        raise ValueError("last saved round has no retained participant; choose another round")
    split = make_client_split(train, SimpleNamespace(
        num_clients=a.k, seed=a.seed, split_seed=a.split_seed, dataset="mnist",
        non_iid=a.split != "iid", non_iid_type={"owner": "class_owner", "dirichlet": "Dirichlet"}.get(a.split),
        alpha=a.alpha, owner_client=a.owner_client, owner_class=a.owner_class, owner_frac=a.owner_frac))
    retained_updates, partition_hashes = {}, {}
    started = time.perf_counter()
    for j in keep:
        indices = np.asarray(split[j].indices, dtype=np.int64)
        partition_hashes[str(j)] = hashlib.sha256(indices.tobytes()).hexdigest()
        space.assign(current)
        net.train()
        opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=.9)
        loader = torch.utils.data.DataLoader(split[j], batch_size=a.batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(a.seed+j), num_workers=0, drop_last=False)
        for batch, (bx, by) in enumerate(loader):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(bx.to(device)), by.to(device)).backward()
            opt.step()
            if batch+1 >= a.batches:
                break
        retained_updates[j] = space.vector()-current
    v = calibrated_aggregate(keep, retained_updates)
    calibration_seconds = time.perf_counter()-started
    net.eval()
    q0, qv = coordinates(current), coordinates(current+v)
    removal = np.array([float(j in a.requesters) for j in range(a.k)])
    selections = {}
    for mode in ("history", "linear", "tension", "shuffled", "history_action", "linear_action", "shuffled_action"):
        selections[mode] = select_coordinates(effects, removal, a.topk, mode, a.seed, qv-q0)[0]
    # Same budget, selected only from the actual proposal's response, no FL history.
    rise = qv-q0
    order = np.argsort(-rise, kind="stable")
    selections["history_free"] = order[rise[order] > 0][:a.topk]
    space.assign(current)
    q = paired_logits(net, x, ax).flatten()
    grad_started = time.perf_counter()
    needed = sorted(set().union(*(set(vv.tolist()) for vv in selections.values())))
    gradients = {i: space.gradient(q[i]) for i in needed}
    logits = net(rx)
    losses = torch.nn.functional.cross_entropy(logits, ry, reduction="none")
    baseline_losses = np.array([losses[ry == c].mean().item() for c in range(10)])
    h = np.stack([space.gradient(losses[ry == c].mean()) for c in range(10)])
    gradient_seconds = time.perf_counter()-grad_started
    radius = np.linalg.norm(v)*a.radius_ratio
    results = {}
    def evaluate(step):
        space.assign(current+step)
        with torch.no_grad():
            pred = net(ex).argmax(1)
            cl = torch.nn.functional.cross_entropy(net(rx), ry, reduction="none")
            loss_change = np.array([cl[ry == c].mean().item() for c in range(10)])-baseline_losses
        change = coordinates(current+step)-q0
        metrics = {"evaluation_accuracy": float((pred == ey).float().mean()),
                "evaluation_class_accuracy": {str(c): float((pred[ey == c] == c).float().mean()) if (ey == c).any() else None for c in range(10)},
                "retention_class_loss_change": loss_change.tolist(),
                "max_observed_retention_loss_increase": float(max(loss_change)),
                "max_coordinate_increase": float(max(change)),
                "history_selected_increase": change[selections["history"]].tolist()}
        if eval_triggered is not None:
            with torch.no_grad():
                triggered_pred = net(eval_triggered).argmax(1)
            non_target = ey != a.eval_target
            metrics["step_asr"] = float((triggered_pred[non_target] == a.eval_target).float().mean())
            metrics["clean_target_rate"] = float((pred[non_target] == a.eval_target).float().mean())
            metrics["macro_trigger_excess"] = float(np.mean([
                float((triggered_pred[ey == c] == a.eval_target).float().mean())
                -float((pred[ey == c] == a.eval_target).float().mean())
                for c in range(10) if c != a.eval_target and (ey == c).any()]))
        return metrics
    before_step = evaluate(np.zeros_like(v))
    results["no_defense"] = evaluate(v)
    results["clip"] = evaluate(v*a.radius_ratio)
    for mode in ("retention_only", "history_free", "linear", "tension", "shuffled", "history",
                 "linear_action", "shuffled_action", "history_action"):
        chosen = selections.get(mode, np.array([], dtype=int))
        g = np.stack([gradients[i] for i in chosen]) if len(chosen) else np.empty((0, space.size))
        started = time.perf_counter()
        d, info = control_step(v, g, h, strength=a.strength, epsilon=a.epsilon, radius=radius)
        solver_seconds = time.perf_counter()-started
        results[mode] = {**evaluate(d), **info, "selected": chosen.tolist(), "solver_seconds": solver_seconds}
        print(f"{mode}: progress={info['progress']:.4g}; norm={info['step_norm']:.4g}", flush=True)
    geometry_info = {}
    if a.geometry:
        removed, retained = request_paths(raw_events, a.requesters)
        permutation = np.random.default_rng(a.seed).permutation(len(removed))
        factors = {"retention_only": np.empty((0, space.size))}
        for mode, dp, kp, smoothing in (
                ("temporal", removed, retained, a.smoothing),
                ("unsmoothed", removed, retained, 0.),
                ("cumulative", removed.sum(0, keepdims=True), retained.sum(0, keepdims=True), 0.),
                ("time_shuffled", removed[permutation], retained[permutation], a.smoothing)):
            factors[mode], geometry_info[mode] = cancellation_geometry(dp, kp, smoothing=smoothing)
        linear = -removed.sum(0)
        factors["linear"] = linear[None]/np.linalg.norm(linear) if np.linalg.norm(linear) else np.empty((0, space.size))
        def actual_loss_change(step):
            space.assign(current+step)
            with torch.no_grad():
                losses = torch.nn.functional.cross_entropy(net(rx), ry, reduction="none")
            return np.array([losses[ry == c].mean().item() for c in range(10)])-baseline_losses
        for mode, factor in factors.items():
            for strength in ([0.] if mode == "retention_only" else a.geometry_strengths):
                key = f"geometry/{mode}/lambda:{strength:g}"
                started = time.perf_counter()
                try:
                    d, info = geometry_control(v, factor, h, strength=strength,
                        epsilon=a.geometry_epsilon, radius=radius, min_progress=a.min_progress,
                        risk_ratio=a.geometry_risk_ratio)
                    d, checked = check_actual_retention(d, v, actual_loss_change,
                        epsilon=a.geometry_epsilon, min_progress=a.min_progress)
                    results[key] = {**evaluate(d), **info, **checked,
                                    "accepted_step_norm": float(np.linalg.norm(d)),
                                    "solver_and_verification_seconds": time.perf_counter()-started}
                    print(f"{key}: accepted progress={checked['accepted_progress']:.4g}", flush=True)
                except ControlInfeasible as error:
                    results[key] = {"status": "control_not_accepted", "reason": str(error),
                                    "forgetting_verified": False, "actual_retention_verified": False}
                    print(f"{key}: control_not_accepted: {error}", flush=True)
    record = {"schema": 1, "scope": "single_actual_recalibration_step; not end-to-end FU",
              "forgetting_verified": False, "attack_efficacy_evaluated": False,
              "settings": vars(a), "device": str(device), "parameter_count": space.size,
              "torch_version": torch.__version__, "transform": a.probe_view,
              "oracle_selection": a.probe_view == "oracle-trigger", "step_asr_measured": a.eval_target is not None,
              "before_step": before_step,
              "history_rounds": [first, a.round], "checkpoint_sources": sources,
              "partition_hashes": partition_hashes, "max_fedavg_relative_error": max_error,
              "actual_retained_clients": list(keep), "actual_recalibration_calls": len(keep),
              "hypothetical_unlearning_calls": 0, "history_seconds": history_seconds,
              "calibration_seconds": calibration_seconds, "gradient_seconds": gradient_seconds,
              "ledger_bytes": effects.nbytes, "results": results,
              "geometry": geometry_info, "geometry_uses_probe_or_attack_labels": False,
              "source_hashes": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ("tension_control.py", "request_geometry.py", "experiment_tension_control.py", "unlearn_method/federaser_clientwise.py")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--checkpoints", default="FUBA/checkpoints")
    p.add_argument("--data-root", default="FUBA/data")
    p.add_argument("--out", required=True)
    p.add_argument("--round", type=int, default=100)
    p.add_argument("--history-window", type=int, default=20)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--requesters", nargs="+", type=int, required=True)
    p.add_argument("--split", choices=("iid", "owner", "dirichlet"), default="iid")
    p.add_argument("--seed", type=int, default=522)
    p.add_argument("--split-seed", type=int, default=1223)
    p.add_argument("--alpha", type=float, default=.1)
    p.add_argument("--owner-client", type=int, default=-1)
    p.add_argument("--owner-class", type=int, default=3)
    p.add_argument("--owner-frac", type=float, default=.9)
    p.add_argument("--probe", type=int, default=32)
    p.add_argument("--probe-view", choices=("brightness", "oracle-trigger"), default="brightness")
    p.add_argument("--eval-target", type=int, choices=range(10), default=None,
                   help="evaluation only; never passed to selection or controller")
    p.add_argument("--noise-thread", type=float, default=.04)
    p.add_argument("--atk-eps", type=float, default=1.)
    p.add_argument("--retain", type=int, default=128)
    p.add_argument("--evaluate", type=int, default=512)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--batches", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=.005)
    p.add_argument("--strength", type=float, default=1.)
    p.add_argument("--epsilon", type=float, default=0.)
    p.add_argument("--radius-ratio", type=float, default=.25)
    p.add_argument("--n-gpu", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--geometry", action="store_true", help="also run the raw-update variational revision and matched controls")
    p.add_argument("--smoothing", type=float, default=1.)
    p.add_argument("--geometry-strengths", type=float, nargs="+", default=[1., 10.])
    p.add_argument("--geometry-epsilon", type=float, default=.01, help="allowed actual validation CE increase per class, in nats")
    p.add_argument("--geometry-risk-ratio", type=float, default=None, help="optional relative cancellation exposure norm budget")
    p.add_argument("--min-progress", type=float, default=.1, help="minimum alignment with actual FU proposal; not forgetting")
    run(p.parse_args())
