"""Shared BadFU checkpoint I/O and aggregation checks; no detection or repair policy."""
import hashlib
from pathlib import Path
import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_tensor(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    torch.save(value, temporary)
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def load_tensor(source):
    path = Path(source["path"])
    if sha(path) != source["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def aggregate(states, counts):
    if len(states) != len(counts) or any(n <= 0 for n in counts):
        raise ValueError("positive sample counts for every participant required")
    weights = np.asarray(counts)/sum(counts)
    result = {}
    for key, ref in states[0].items():
        value = sum(state[key].float()*float(w) for state, w in zip(states, weights))
        result[key] = value.to(ref.dtype) if ref.is_floating_point() else value.round().to(ref.dtype)
    return result


def load_event(record, counts):
    event = load_tensor(record["history"])
    after = load_tensor(record["after"])["gm_after"] if "after" in record else event["gm_after"]
    if len(event["cms"]) != len(counts):
        raise ValueError("client count mismatch")
    if "counts" in event and event["counts"] != counts:
        raise ValueError("saved aggregation weights differ")
    # Legacy data/count reconstruction must at least reproduce recorded FedAvg.
    expected = aggregate(event["cms"], counts)
    for key in after:
        if not torch.allclose(expected[key].float(), after[key].float(), rtol=2e-5, atol=2e-6):
            raise ValueError(f"recorded aggregation cannot be reproduced: {key}")
    return event, after

