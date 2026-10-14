"""Full MNIST/FedAvg replay with an explicit retained-data archive.

Requires a genuine pre-FL checkpoint and retained (possibly poisoned) data.
Never silently replaces retained poisoned data with clean MNIST. Archive NPZ:
images float32 [N,1,28,28], labels integer [N], units integer [N]. Unit IDs must
match the historical client/deletion units. Original within-client BadFU needs
a sample-aware history adapter; this executable supports full-client deletion.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from model import Net, MNISTAutoencoder
from tension_control import ParameterSpace
from request_geometry import TemporalFUController, ControlInfeasible
from unlearn_method.federaser_clientwise import calibrated_aggregate


def run(a):
    out, root = Path(a.out), Path(a.checkpoints)
    if out.exists():
        raise FileExistsError(out)
    if a.rounds < 1 or a.batches < 1 or a.batch_size < 1 or a.retain < 10 or a.evaluate < 10:
        raise ValueError("positive round, training and evaluation sizes required")
    initial_path = Path(a.initial) if a.initial else root/f"global_net_model_{a.name}_round_0.pkl"
    if not initial_path.is_file():
        raise FileNotFoundError("true pre-FL round_0 checkpoint required; do not substitute a trained checkpoint")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = torch.device("cuda:0" if a.n_gpu and torch.cuda.is_available() else "cpu")
    model = Net().to(device).eval()
    space = ParameterSpace(model)
    sources = []
    def read(path):
        payload = Path(path).read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
        return pickle.loads(payload)  # authorized, trusted local researcher files only
    def vector(sd):
        model.load_state_dict(sd)
        return space.vector()
    initial = vector(read(initial_path))
    previous, events, max_error = initial.copy(), [], 0.
    for r in range(1, a.rounds+1):
        local = read(root/f"local_net_model_{a.name}_round_{r}.pkl")
        global_sd = read(root/f"global_net_model_{a.name}_round_{r}.pkl")
        current = vector(global_sd)
        metadata = root/f"history_{a.name}_round_{r}.json"
        if metadata.is_file():
            raw = metadata.read_bytes()
            sources.append({"path": str(metadata), "sha256": hashlib.sha256(raw).hexdigest()})
            ids = json.loads(raw)["selected_clients"]
            if len(set(ids)) != len(ids) or any(not isinstance(j, int) or j < 0 or j >= len(local) or local[j] is None for j in ids):
                raise ValueError("invalid participant metadata")
        else:
            ids = [j for j, sd in enumerate(local) if sd is not None
                   and not all(torch.equal(sd[key], global_sd[key]) for key in global_sd)]
        if not ids:
            raise ValueError(f"no participants in saved round {r}")
        updates = {j: vector(local[j])-previous for j in ids}
        residual = np.linalg.norm(np.mean(list(updates.values()), axis=0)-(current-previous))
        movement = np.linalg.norm(current-previous)
        error = residual/max(movement, 1e-12)
        # A tiny training step can be smaller than float32 aggregation roundoff.
        # gamma_n bounds n floating-point operations; never hide this in the
        # relative-error statistic, which is still logged without adjustment.
        nu = (len(ids)+2)*np.finfo(np.float32).eps/2
        roundoff = nu/(1-nu)*np.mean([np.linalg.norm(u+previous) for u in updates.values()])
        if residual > .005*movement+roundoff:
            raise ValueError(f"round {r} is not reconstructible equal-weight FedAvg: relative error={error:.6g}")
        max_error = max(max_error, float(error))
        events.append({"updates": updates})
        previous = current
    known = set().union(*(set(e["updates"]) for e in events))
    deleted = set(a.requesters)
    if not deleted or len(deleted) != len(a.requesters) or deleted-known or not known-deleted:
        raise ValueError("unique known requested clients and retained clients required")
    payload = Path(a.retained_archive).read_bytes()
    archive_hash = hashlib.sha256(payload).hexdigest()
    with np.load(a.retained_archive, allow_pickle=False) as data:
        images, labels, units = (data[key].copy() for key in ("images", "labels", "units"))
    if (images.ndim != 4 or images.shape[1:] != (1, 28, 28) or not len(images)
            or labels.shape != (len(images),) or units.shape != labels.shape
            or not np.isfinite(images).all() or not np.issubdtype(labels.dtype, np.integer)
            or not np.issubdtype(units.dtype, np.integer) or (labels < 0).any() or (labels > 9).any()
            or set(units.tolist()) != known-deleted):
        raise ValueError("archive must contain finite MNIST-shaped inputs for exactly the retained client IDs")
    partitions = {j: torch.utils.data.TensorDataset(torch.as_tensor(images[units == j], dtype=torch.float32),
                   torch.as_tensor(labels[units == j], dtype=torch.long)) for j in known-deleted}
    del images, labels, units, payload
    test = datasets.MNIST(a.data_root, train=False, download=False, transform=transforms.ToTensor())
    if a.retain+a.evaluate > len(test):
        raise ValueError("validation and evaluation must be disjoint")
    def tensors(indices):
        pairs = [test[i] for i in indices]
        return torch.stack([x for x, _ in pairs]).to(device), torch.tensor([y for _, y in pairs], device=device)
    rx, ry = tensors(range(a.retain))
    ex, ey = tensors(range(a.retain, a.retain+a.evaluate))
    if len(ry.unique()) != 10:
        raise ValueError("validation needs all classes")
    def class_losses():
        losses = torch.nn.functional.cross_entropy(model(rx), ry, reduction="none")
        return [losses[ry == c].mean() for c in range(10)]
    def oracle(state, t):
        space.assign(state)
        model.eval()
        losses = class_losses()
        base = np.array([loss.item() for loss in losses])
        h = np.stack([space.gradient(loss) for loss in losses])
        def change(step):
            space.assign(state+step)
            with torch.no_grad():
                return np.array([loss.item() for loss in class_losses()])-base
        return h, change
    def recalibrate(state, j, t):
        space.assign(state)
        model.train()
        loader = torch.utils.data.DataLoader(partitions[j], batch_size=a.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(a.seed+100000*t+j), num_workers=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=.9)
        for batch, (bx, by) in enumerate(loader):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(bx.to(device)), by.to(device)).backward()
            optimizer.step()
            if batch+1 >= a.batches:
                break
        model.eval()
        return space.vector()-state
    triggered = None
    if a.eval_target is not None:
        trigger = MNISTAutoencoder().to(device).eval()
        trigger.load_state_dict(read(root/f"backdoor_net_model_{a.name}_round_{a.rounds}.pkl"))
        with torch.no_grad():
            triggered = (ex+(trigger(ex)*a.atk_eps).clamp(-a.noise_thread, a.noise_thread)).clamp(-1, 1)
    def evaluate(state):
        space.assign(state)
        model.eval()
        with torch.no_grad():
            pred = model(ex).argmax(1)
            metrics = {"accuracy": float((pred == ey).float().mean()),
                "class_accuracy": {str(c): float((pred[ey == c] == c).float().mean()) if (ey == c).any() else None for c in range(10)}}
            if triggered is not None:
                mask = ey != a.eval_target
                metrics["asr"] = float((model(triggered).argmax(1)[mask] == a.eval_target).float().mean())
                metrics["clean_target_rate"] = float((pred[mask] == a.eval_target).float().mean())
        return metrics
    record = {"scope": "full recorded FedAvg history with explicitly supplied retained data",
              "forgetting_verified": False, "settings": vars(a), "initial_source": str(initial_path),
              "retained_archive_sha256": archive_hash, "checkpoint_sources": sources,
              "max_fedavg_relative_error": max_error, "before_fu": evaluate(previous), "results": {},
              "source_hashes": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                  for name in ("request_geometry.py", "experiment_request_replay.py", "unlearn_method/federaser_clientwise.py")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        out.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for mode in a.modes:
        controller = None if mode == "no_defense" else TemporalFUController(events, deleted, oracle,
            window=a.history_window, smoothing=a.smoothing, risk_ratio=a.risk_ratio, epsilon=a.epsilon,
            radius_ratio=a.radius_ratio, min_progress=a.min_progress, mode=mode)
        state, log = initial.copy(), []
        started, failed, calls = time.perf_counter(), None, 0
        for t, event in enumerate(events):
            keep = {j: u for j, u in event["updates"].items() if j not in deleted}
            if not keep:
                log.append({"event": t, "status": "no_retained_participants"})
                continue
            new = {j: recalibrate(state, j, t) for j in keep}
            calls += len(new)
            proposal = calibrated_aggregate(keep, new)
            try:
                step, info = (proposal, {"status": "uncontrolled"}) if controller is None else controller(state, proposal, t)
            except ControlInfeasible as error:
                failed = {"event": t, "reason": str(error)}
                break
            state += step
            log.append({"event": t, **info})
        record["results"][mode] = {"status": "control_not_accepted" if failed else "candidate_requires_verification",
            "failure": failed, "completed_all_events": failed is None, "forgetting_verified": False,
            "actual_recalibration_calls": calls, "seconds": time.perf_counter()-started, "steps": log,
            "candidate_metrics": evaluate(state) if failed is None else None}
        save()
        print(f"{mode}: {record['results'][mode]['status']}", flush=True)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--checkpoints", default="FUBA/checkpoints")
    p.add_argument("--initial", help="true pre-FL state; default is saved round_0")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--retained-archive", required=True)
    p.add_argument("--data-root", default="FUBA/data")
    p.add_argument("--out", required=True)
    p.add_argument("--requesters", type=int, nargs="+", required=True)
    p.add_argument("--modes", nargs="+", default=["no_defense", "retention_only", "linear", "cumulative", "unsmoothed", "temporal"],
        choices=["no_defense", "retention_only", "linear", "cumulative", "unsmoothed", "temporal"])
    for flag, default in (("seed", 522), ("batches", 2), ("batch-size", 64), ("retain", 128),
                          ("evaluate", 512), ("threads", 2), ("history-window", 20), ("n-gpu", 1)):
        p.add_argument("--"+flag, type=int, default=default)
    for flag, default in (("lr", .005), ("smoothing", 1.), ("risk-ratio", .1), ("epsilon", .01),
                          ("radius-ratio", .25), ("min-progress", .1), ("atk-eps", 1.), ("noise-thread", .04)):
        p.add_argument("--"+flag, type=float, default=default)
    p.add_argument("--eval-target", type=int, choices=range(10))
    run(p.parse_args())
