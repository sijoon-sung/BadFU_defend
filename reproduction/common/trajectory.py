"""Full-state trajectory storage. A log is not a trajectory."""
import hashlib
import json
import math
from pathlib import Path
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)
    return {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def check_state(state, reference=None):
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a full tensor state_dict")
    if reference is not None and state.keys() != reference.keys():
        raise ValueError("Model keys changed")
    for k, v in state.items():
        if not torch.isfinite(v).all():
            raise ValueError(f"Non-finite model tensor: {k}")
        if reference is not None and v.shape != reference[k].shape:
            raise ValueError(f"Model shape changed: {k}")


class Recorder:
    def __init__(self, root, initial, expected_rounds, metadata):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        initial = cpu(initial)
        check_state(initial)
        self.previous = initial
        self.manifest = {
            "schema": "full-trajectory-v1", "status": "running",
            "expected_rounds": expected_rounds, "metadata": metadata,
            "initial": save(self.root / "initial.pt", initial), "rounds": [],
        }
        self.flush()

    def flush(self):
        json_write(self.root / "manifest.json", self.manifest)

    def append(self, before, clients, after, ids, weights, extras=None):
        before, clients, after = cpu(before), cpu(clients), cpu(after)
        if len(ids) != len(clients) or len(ids) != len(set(ids)) or not ids:
            raise ValueError("Missing/duplicate participant IDs")
        if len(weights) != len(ids) or not all(math.isfinite(w) and w >= 0 for w in weights) or abs(sum(weights) - 1) > 1e-6:
            raise ValueError("Invalid aggregation weights")
        for state in [before, *clients, after]:
            check_state(state, self.previous)
        if any(not torch.equal(before[k], self.previous[k]) for k in before):
            raise ValueError("Pre-round model does not equal preceding global model")
        r = len(self.manifest["rounds"]) + 1
        item = {"round": r, "gm": before, "cms": clients, "gm_after": after,
                "client_ids": list(ids), "weights": list(weights), "extras": cpu(extras or {})}
        entry = save(self.root / f"round_{r:04d}.pt", item)
        entry.update(round=r, client_ids=list(ids))
        self.manifest["rounds"].append(entry)
        self.previous = after
        self.flush()

    def complete(self):
        if len(self.manifest["rounds"]) != self.manifest["expected_rounds"]:
            raise ValueError("Incomplete trajectory")
        self.manifest["status"] = "complete"
        self.flush()


def verify(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if m.get("schema") != "full-trajectory-v1":
        raise ValueError("Not a new full-state trajectory; compact history.pt is insufficient")
    if m["status"] != "complete" or len(m["rounds"]) != m["expected_rounds"]:
        raise ValueError("Run did not finish")

    def load(entry):
        path = root / entry["path"]
        if path.parent.resolve() != root.resolve() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint path/hash mismatch: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    prev = load(m["initial"])
    check_state(prev)
    for n, e in enumerate(m["rounds"], 1):
        x = load(e)
        if x["round"] != n or e["round"] != n:
            raise ValueError("Non-consecutive rounds")
        ids, weights = x["client_ids"], x["weights"]
        if not ids or len(ids) != len(set(ids)) or len(ids) != len(x["cms"]) or ids != e["client_ids"]:
            raise ValueError("Participant mismatch")
        if len(weights) != len(ids) or not all(math.isfinite(w) and w >= 0 for w in weights) or abs(sum(weights)-1) > 1e-6:
            raise ValueError("Aggregation weights invalid")
        for state in [x["gm"], *x["cms"], x["gm_after"]]:
            check_state(state, prev)
        if any(not torch.equal(x["gm"][k], prev[k]) for k in prev):
            raise ValueError(f"Broken model chain at round {n}")
        rule = m["metadata"].get("aggregation")
        if rule in ("badfu_sample_weighted", "fuba_uniform"):
            for k, actual in x["gm_after"].items():
                if rule == "fuba_uniform":
                    predicted = torch.stack([s[k].float() for s in x["cms"]]).mean(0)
                else:
                    predicted = torch.zeros_like(actual, dtype=torch.float32)
                    for state, w in zip(x["cms"], weights):
                        predicted += state[k].float() * w
                    if actual.dtype == torch.long:
                        predicted = predicted.round().long()
                if not torch.allclose(actual.float(), predicted.float(), rtol=2e-5, atol=2e-6):
                    raise ValueError(f"Aggregation mismatch at round {n}: {k}")
        prev = x["gm_after"]
    return {"path": str(root), "rounds": len(m["rounds"]), "status": "verified",
            "aggregation_checked": m["metadata"].get("aggregation") in ("badfu_sample_weighted", "fuba_uniform")}
