"""Record per-input functional effects of ACTUAL positive FL updates.

No client deletion, unlearned model or attack trigger is used. This is the finite
training-update probe already used by CCD, with input coordinates retained. It
is not a JVP or an exact sample attribution. FUBA/FedAvg client records only.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch


@torch.no_grad()
def outputs(model, xs, batch_size=64):
    out = torch.cat([model(x).detach().cpu() for x in xs.split(batch_size)])
    if not torch.isfinite(out).all():
        raise ValueError("nonfinite model output")
    return torch.log_softmax(out, 1).numpy(), (out-out.mean(1, keepdim=True)).numpy()


def collect(name, checkpoints, data_root, out, rounds=100, k=20, dataset="mnist", probe=128, n_gpu=1):
    # release_map changes cwd; resolve all paths BEFORE importing it.
    checkpoints, data_root, out = Path(checkpoints).resolve(), Path(data_root).resolve(), Path(out).resolve()
    sidecar = out.with_suffix(".json")
    if out.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {out}; runner validates completed caches")
    from experiments.shared.trajectory_tools.release_map import Flat, _fuba_factory, make_model, present_clients
    from torchvision import datasets, transforms
    factory, _ = _fuba_factory(argparse.Namespace(dataset=dataset))
    device = torch.device("cuda:0" if torch.cuda.is_available() and n_gpu else "cpu")
    ds_class = datasets.MNIST if dataset == "mnist" else datasets.CIFAR10
    ds = ds_class(str(data_root), train=False, download=True, transform=transforms.ToTensor())
    if not 1 <= probe < len(ds):
        raise ValueError("probe must leave independent evaluation inputs")
    xs = torch.stack([ds[i][0] for i in range(probe)]).to(device)
    sources = []

    def read(kind, r):
        path = checkpoints/f"{kind}_net_model_{name}_round_{r}.pkl"
        payload = path.read_bytes()
        sources.append({"file": path.name, "sha256": hashlib.sha256(payload).hexdigest()})
        return pickle.loads(payload)

    prev = read("global", 1)
    flat = Flat(prev)
    ts, js, lp, logits, errors = [], [], [], [], []
    started = time.perf_counter()
    for r in range(2, rounds+1):
        local, glob = read("local", r), read("global", r)
        if len(local) != k:
            raise ValueError(f"K mismatch at round {r}")
        present = present_clients(local, glob)
        if not present:
            raise ValueError(f"no observable participants at round {r}")
        gv = flat.flatten(prev)
        deltas = [(flat.flatten(local[j])-gv)/len(present) for j in present]
        # This collector is defined for equal-weight FedAvg, not robust aggregators.
        actual = flat.flatten(glob)-gv
        relative_error = float((sum(deltas)-actual).norm()/actual.norm().clamp_min(1e-12))
        errors.append(relative_error)
        if relative_error > .005:
            raise ValueError(f"FedAvg reconstruction mismatch {relative_error:.3g} at round {r}")
        baseline = outputs(make_model(factory, prev, device), xs)
        for j, delta in zip(present, deltas):
            # Add the actual weighted update. No training contribution is removed.
            measured = outputs(make_model(factory, flat.to_sd(gv+delta), device), xs)
            ts.append(r-1); js.append(j)
            lp.append(measured[0]-baseline[0]); logits.append(measured[1]-baseline[1])
        prev = glob
        if r % 10 == 0 or r == rounds:
            print(f"{name}: FL records {r}/{rounds}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, K=k, T=rounds-1, ts=np.array(ts), js=np.array(js),
                        logprob_vs=np.stack(lp), logit_vs=np.stack(logits))
    meta = {"schema": 1, "name": name, "dataset": dataset, "rounds": rounds, "K": k, "probe": probe,
            "probe_indices": list(range(probe)), "probe_tensor_sha256": hashlib.sha256(xs.cpu().numpy().tobytes()).hexdigest(),
            "measurement": "per-input finite positive training-update effects; not JVP", "model_unlearning_calls": 0,
            "checkpoint_sources": sources, "max_fedavg_relative_error": max(errors),
            "seconds": time.perf_counter()-started, "bytes": out.stat().st_size,
            "cache_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    sidecar.write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    p.add_argument("--probe", type=int, default=128)
    p.add_argument("--n-gpu", type=int, default=1)
    a = p.parse_args()
    meta = collect(**vars(a))
    print(json.dumps({key: meta[key] for key in (
        "name", "rounds", "probe", "seconds", "bytes", "max_fedavg_relative_error", "model_unlearning_calls")}, indent=2))
