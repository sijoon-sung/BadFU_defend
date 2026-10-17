"""Persist BadFU public-code trajectories, then test request-conditioned FU.

Six-client public-code variant, not the paper's within-client sample deletion.
Full trainable ResNet parameters are controlled. Every FU arm recalibrates BN
on the same clean validation inputs before each event, then freezes the buffers
throughout that event's optimization/evaluation. This is a FU policy variant.
Saved data include actual poison and camouflage inputs, never clean substitutes.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/"FUBA"))
from badfu_common import BadFUData, ResNet18
from FUBA.research.large_geometry import large_control
from FUBA.research.request_geometry import check_actual_retention, ControlInfeasible


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


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


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


class BNSpace:
    """Parameters only; caller explicitly manages BN state at event boundaries."""
    def __init__(self, model):
        self.model = model
        self.named = list(model.named_parameters())
        self.keys = [key for key, _ in self.named]
        self.size = sum(p.numel() for _, p in self.named)
        allowed = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                allowed.update(f"{name}.{key}" if name else key for key, _ in module.named_buffers())
        if set(dict(model.named_buffers()))-allowed:
            raise ValueError("only BatchNorm buffers are supported by this FU policy")

    def vector(self, state=None):
        state = self.model.state_dict() if state is None else state
        return np.concatenate([state[key].detach().cpu().float().numpy().ravel() for key in self.keys])

    def assign(self, vector):
        if np.shape(vector) != (self.size,) or not np.isfinite(vector).all():
            raise ValueError("invalid parameter vector")
        offset = 0
        with torch.no_grad():
            for _, p in self.named:
                p.copy_(torch.as_tensor(vector[offset:offset+p.numel()].reshape(p.shape), device=p.device, dtype=p.dtype))
                offset += p.numel()

    def gradient(self, loss):
        gs = torch.autograd.grad(loss, [p for _, p in self.named], allow_unused=True)
        return np.concatenate([(torch.zeros_like(p) if g is None else g).detach().cpu().float().numpy().ravel()
                               for (_, p), g in zip(self.named, gs)])


@torch.no_grad()
def recalibrate_bn(model, validation, device, batch_size):
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in layers]
    try:
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
        # Same fixed validation inputs for every arm/event; no deleted data.
        if layers:
            for bx in validation.split(batch_size):
                model(bx.to(device))
    finally:
        for layer, momentum in zip(layers, momenta):
            layer.momentum = momentum
        model.eval()


def materialize(dataset, indices):
    pairs = [dataset[int(i)] for i in indices]
    if not pairs:
        raise ValueError("empty archived dataset")
    return {"images": torch.stack([x for x, _ in pairs]),
            "labels": torch.tensor([int(y) for _, y in pairs], dtype=torch.long)}


def build_archive(a):
    data = BadFUData(a.dominant_ratio, a.seed, a.batch_size, a.data_root, a.record_root, download=False)
    clients, sample_indices = [], []
    for j, dataset in enumerate(data.client_sets):
        ids = list(range(len(dataset)))
        if a.max_per_client and len(ids) > a.max_per_client:
            ids = torch.randperm(len(ids), generator=torch.Generator().manual_seed(a.seed+j))[:a.max_per_client].tolist()
        clients.append(materialize(dataset, ids))
        sample_indices.append(ids)
    test = data.test_loader.dataset
    targets = np.asarray(test.targets)
    validation_ids = sorted(np.concatenate([np.flatnonzero(targets == c)[:a.retain_per_class] for c in range(10)]).tolist())
    if len(validation_ids) != 10*a.retain_per_class:
        raise ValueError("validation requires the requested examples in every class")
    eligible = sorted(set(range(len(test)))-set(validation_ids))
    eval_ids = eligible[:a.evaluate] if a.evaluate else eligible
    # Exclude matching clean-validation source images from triggered evaluation.
    bd_ids = [i for i, key in enumerate(data.bd_test.keys)
              if int(key) not in validation_ids and int(data.bd_test_orig[i]) != a.target]
    if a.evaluate:
        bd_ids = bd_ids[:a.evaluate]
    backdoor_eval = materialize(data.bd_test, bd_ids)
    if not (backdoor_eval["labels"] == a.target).all():
        raise ValueError("evaluation target differs from the saved trigger dataset")
    return {"clients": clients, "sample_indices": sample_indices, "clean_indices": data.clean_indices,
        "validation": materialize(test, validation_ids), "validation_ids": validation_ids,
        "clean_eval": materialize(test, eval_ids), "clean_eval_ids": eval_ids,
        "backdoor_eval": backdoor_eval,
        "backdoor_original_labels": data.bd_test_orig[bd_ids], "target": a.target,
        "counts": [len(x["labels"]) for x in clients], "seed": a.seed,
        "provenance": "materialized public-code six-client datasets including poison/camouflage"}


def train_local(model, data, device, *, seed, epochs, batches, batch_size, lr):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(data["images"], data["labels"]), batch_size=batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    seen = 0
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(bx.to(device)), by.to(device)).backward()
            optimizer.step()
            seen += 1
            if batches and seen >= batches:
                return cpu_state(model)
    return cpu_state(model)


def aggregate(states, counts):
    if len(states) != len(counts) or any(n <= 0 for n in counts):
        raise ValueError("positive sample counts for every participant required")
    weights = np.asarray(counts)/sum(counts)
    result = {}
    for key, ref in states[0].items():
        value = sum(state[key].float()*float(w) for state, w in zip(states, weights))
        result[key] = value.to(ref.dtype) if ref.is_floating_point() else value.round().to(ref.dtype)
    return result


@torch.no_grad()
def metrics(model, archive, device, batch_size):
    model.eval()
    result = {}
    for kind, source in (("clean", archive["clean_eval"]), ("backdoor", archive["backdoor_eval"])):
        pred = torch.cat([model(b.to(device)).argmax(1).cpu() for b in source["images"].split(batch_size)])
        if kind == "clean":
            y = source["labels"]
            result["accuracy"] = float((pred == y).float().mean())
            result["class_accuracy"] = {str(c): float((pred[y == c] == c).float().mean()) if (y == c).any() else None for c in range(10)}
            result["clean_target_rate"] = float((pred[y != archive["target"]] == archive["target"]).float().mean())
        else:
            mask = archive["backdoor_original_labels"] != archive["target"]
            result["asr"] = float((pred[mask] == archive["target"]).float().mean())
            result["asr_samples"] = int(mask.sum())
    return result


def training_settings(a):
    return {key: getattr(a, key) for key in ("seed", "rounds", "local_epochs", "train_batches", "batch_size",
        "max_per_client", "dominant_ratio", "target", "no_pretrained", "retain_per_class", "evaluate")}


def prepare(a, model_factory=ResNet18, archive=None):
    root = Path(a.run_dir).resolve()
    manifest_path = root/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["training_settings"] != training_settings(a):
            raise ValueError("existing run is partial or settings differ; preserve it and choose another --run-dir")
        print("Reusing completed BadFU trajectory; no FL training", flush=True)
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("nonempty run directory: choose a new directory; nothing was deleted")
    root.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    archive = build_archive(a) if archive is None else archive
    manifest = {"schema": 1, "status": "preparing", "variant": "badfu_public_code_six_client",
        "model": "torchvision ResNet18, CIFAR conv1/maxpool/fc adaptation" if model_factory is ResNet18 else model_factory.__name__,
        "sample_level_badfu": False, "training_settings": training_settings(a),
        "dataset": save_tensor(root/"data.pt", archive), "counts": archive["counts"], "rounds": [],
        "source_hashes": {name: sha(HERE/name) for name in ("request_pipeline.py", "badfu_common.py")},
        "torch_version": str(torch.__version__), "roles_evaluation_only": {"injection": [0], "camouflage": [5]}}
    save_json(manifest_path, manifest)
    if a.legacy_traj:
        legacy = Path(a.legacy_traj).resolve()
        for t in range(a.rounds):
            old, after = legacy/f"round_{t}.pt", legacy/f"glob_{t}.pt"
            if not old.is_file() or not after.is_file():
                raise FileNotFoundError(f"legacy trajectory is incomplete at round {t}")
            event = torch.load(old, map_location="cpu", weights_only=True)
            if len(event["cms"]) != len(archive["counts"]):
                raise ValueError("legacy participant count does not match supplied data settings")
            if t == 0:
                manifest["initial"] = save_tensor(root/"initial.pt", event["gm"])
            manifest["rounds"].append({"event": t, "history": {"path": str(old), "sha256": sha(old)},
                                      "after": {"path": str(after), "sha256": sha(after)}})
            del event
        manifest["data_matching"] = "reconstructed from explicit legacy seed/split/subset settings; original data hash unavailable"
        manifest["legacy_source"] = str(legacy)
    else:
        model = model_factory(pretrained=not a.no_pretrained).to(a.device)
        manifest["initial"] = save_tensor(root/"initial.pt", cpu_state(model))
        manifest["data_matching"] = "FL and FU use the identical saved data archive"
        manifest["status"] = "training"
        save_json(manifest_path, manifest)
        for t in range(a.rounds):
            before = cpu_state(model)
            states = []
            for j, dataset in enumerate(archive["clients"]):
                model.load_state_dict(before)
                states.append(train_local(model, dataset, a.device, seed=a.seed+100000*t+j,
                    epochs=a.local_epochs, batches=a.train_batches, batch_size=a.batch_size, lr=.01))
            after = aggregate(states, archive["counts"])
            source = save_tensor(root/f"round_{t:04d}.pt", {"gm": before, "cms": states, "gm_after": after,
                "participants": list(range(len(states))), "counts": archive["counts"], "event": t})
            model.load_state_dict(after)
            manifest["rounds"].append({"event": t, "history": source,
                "metrics": metrics(model, archive, a.device, a.batch_size)})
            save_json(manifest_path, manifest)
            print(f"FL round {t+1}/{a.rounds} saved: {manifest['rounds'][-1]['metrics']}", flush=True)
            del states
        del model
    manifest["status"] = "complete"
    save_json(manifest_path, manifest)
    return manifest


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


def test_requests(a, model_factory=ResNet18):
    out = Path(a.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((Path(a.run_dir)/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("complete trajectory required before FU testing")
    archive = load_tensor(manifest["dataset"])
    initial = load_tensor(manifest["initial"])
    counts = manifest["counts"]
    if archive["counts"] != counts or not a.requests or len(set(a.requests)) != len(a.requests):
        raise ValueError("invalid archive/counts/requests")
    if any(j < 0 or j >= len(counts) for j in a.requests):
        raise ValueError("unknown deletion unit")
    model = model_factory(pretrained=False).to(a.device)
    space = BNSpace(model)
    vx, vy = archive["validation"]["images"], archive["validation"]["labels"]
    if set(vy.tolist()) != set(range(10)):
        raise ValueError("all ten validation classes required")
    def assign(v, rebuild_bn=True):
        space.assign(v)
        if rebuild_bn:
            recalibrate_bn(model, vx, a.device, a.batch_size)
        model.eval()
    def losses():
        if model.training:
            raise RuntimeError("validation must not mutate BatchNorm buffers")
        with torch.no_grad():
            values = torch.cat([nn.functional.cross_entropy(model(x.to(a.device)), y.to(a.device), reduction="none").cpu()
                for x, y in zip(vx.split(a.batch_size), vy.split(a.batch_size))])
        return np.array([float(values[vy == c].mean()) for c in range(10)])
    final_event, final_state = load_event(manifest["rounds"][-1], counts)
    del final_event
    model.load_state_dict(final_state)
    report = {"schema": 1, "variant": manifest["variant"], "scope": "one_actual_step" if a.step is not None else "full_recorded_replay",
        "settings": vars(a), "manifest_sha256": sha(Path(a.run_dir)/"manifest.json"),
        "dataset_sha256": manifest["dataset"]["sha256"], "data_matching": manifest["data_matching"],
        "before_fu": metrics(model, archive, a.device, a.batch_size), "results": {},
        "forgetting_verified": False, "parameter_scope": "all_trainable_parameters",
        "bn_policy": "recompute on fixed clean validation before each event in every arm, freeze during control/evaluation; log BN-only loss changes separately",
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
            (HERE/"request_pipeline.py", HERE/"badfu_common.py", ROOT/"FUBA/research/large_geometry.py", ROOT/"FUBA/research/request_geometry.py")}}
    del final_state
    last = len(manifest["rounds"])-1
    if a.step is not None and not 0 <= a.step <= last:
        raise ValueError("step is a zero-based saved FL event index")
    selected = range(last+1) if a.step is None else [a.step]
    for request in a.requests:
        kept = [j for j in range(len(counts)) if j != request]
        for mode in a.modes:
            model.load_state_dict(initial)
            state = space.vector().astype(float)
            ds, ks, steps, calls, failed = [], [], [], 0, None
            started = time.perf_counter()
            if a.step is not None:
                # Restore the real pre-step checkpoint; this is not complete forgetting.
                start_event, _ = load_event(manifest["rounds"][a.step], counts)
                model.load_state_dict(start_event["gm"])
                state = space.vector(start_event["gm"]).astype(float)
                del start_event
            starting_metrics = metrics(model, archive, a.device, a.batch_size)
            normalized_start_metrics = None
            for t in range(max(0, selected[0]-a.history_window+1), selected[-1]+1):
                event, _ = load_event(manifest["rounds"][t], counts)
                old = space.vector(event["gm"])
                d, k = np.zeros(space.size, np.float32), np.zeros(space.size, np.float32)
                for j, sd in enumerate(event["cms"]):
                    update = space.vector(sd)-old
                    (d if j == request else k)[:] += update*(counts[j]/sum(counts))
                ds.append(d); ks.append(k)
                ds, ks = ds[-a.history_window:], ks[-a.history_window:]
                if t not in selected:
                    del event, old
                    continue
                pre_bn_loss = losses()
                assign(state)
                bn_change = losses()-pre_bn_loss
                if normalized_start_metrics is None:
                    normalized_start_metrics = metrics(model, archive, a.device, a.batch_size)
                base_sd = cpu_state(model)
                proposal = np.zeros(space.size, dtype=float)
                for j in kept:
                    model.load_state_dict(base_sd)
                    new = train_local(model, archive["clients"][j], a.device,
                        seed=archive["seed"]+100000*t+j, epochs=a.fu_epochs, batches=a.fu_batches,
                        batch_size=a.batch_size, lr=.01)
                    direction = space.vector(new).astype(float)-state
                    old_norm = float(np.linalg.norm((space.vector(event["cms"][j])-old).astype(float)))
                    new_norm = float(np.linalg.norm(direction))
                    if old_norm and not new_norm:
                        raise ValueError("nonzero old update with zero recalibration direction")
                    proposal += direction*(old_norm/new_norm if new_norm else 0.)*(counts[j]/sum(counts[i] for i in kept))
                    calls += 1
                    del new, direction
                model.load_state_dict(base_sd)
                model.eval()
                del event, old, base_sd
                base_loss = losses()
                trials = []
                def change(delta):
                    assign(state+delta, rebuild_bn=False)
                    changes = losses()-base_loss
                    trials.append({"progress": float(proposal@delta/(proposal@proposal)) if proposal@proposal else 0.,
                                   "class_loss_change": changes.tolist()})
                    return changes
                info = {}
                try:
                    if mode == "no_defense":
                        step, info = proposal, {"status": "uncontrolled", "actual_loss_change": change(proposal).tolist()}
                    else:
                        gradients = []
                        for c in range(10):
                            mask = vy == c
                            loss = nn.functional.cross_entropy(model(vx[mask].to(a.device)), vy[mask].to(a.device))
                            gradients.append(space.gradient(loss))
                        h = np.stack(gradients)
                        del gradients
                        step, info = large_control(proposal, np.stack(ds), np.stack(ks), h, mode=mode,
                            smoothing=a.smoothing, risk_ratio=a.risk_ratio, epsilon=a.epsilon,
                            radius_ratio=a.radius_ratio, min_progress=a.min_progress)
                        del h
                        step, checked = check_actual_retention(step, proposal, change,
                            epsilon=a.epsilon, min_progress=a.min_progress)
                        info.update(checked)
                        info["accepted_step_norm"] = float(np.linalg.norm(step))
                except ControlInfeasible as error:
                    failed = {"event": t, "reason": str(error), "solver": info, "loss_trials": trials,
                              "bn_recalibration_loss_change": bn_change.tolist()}
                    break
                assign(state+step, rebuild_bn=False)
                state = space.vector().astype(float)
                steps.append({"event": t, "retained_participants": kept,
                              "bn_recalibration_loss_change": bn_change.tolist(), **info})
                print(f"request {request}, {mode}, FU event {t}: {info['status']}", flush=True)
            key = f"request_{request}/{mode}"
            result = {"status": "control_not_accepted" if failed else "candidate_requires_verification",
                "failure": failed, "completed_selected_events": failed is None, "forgetting_verified": False,
                "actual_recalibration_calls": calls, "seconds": time.perf_counter()-started, "steps": steps,
                "starting_metrics": starting_metrics, "after_initial_bn_recalibration": normalized_start_metrics,
                "candidate_metrics": None if failed else metrics(model, archive, a.device, a.batch_size)}
            if failed is None:
                result["checkpoint"] = save_tensor(Path(a.run_dir)/"fu"/out.stem/f"request_{request}_{mode}.pt", cpu_state(model))
            report["results"][key] = result
            save_json(out, report)
            print(f"{key}: {result['status']} {result['candidate_metrics']}", flush=True)
            del state, ds, ks
            gc.collect()
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("run", "train", "test"), help="run saves/reuses FL then automatically tests FU")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default="FUBA/logs/badfu_request/result.json")
    p.add_argument("--legacy-traj", help="import references to existing round_N.pt/glob_N.pt without retraining or modifying them")
    p.add_argument("--data-root", default=str(HERE/"data"))
    p.add_argument("--record-root", default=str(HERE/"record/badnet_dataset"))
    p.add_argument("--no-pretrained", action="store_true", help="pipeline smoke only; changes the attack setup")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    for key, default in (("seed", 42), ("rounds", 40), ("local-epochs", 5), ("train-batches", 0),
            ("max-per-client", 0), ("batch-size", 64), ("retain-per-class", 8), ("evaluate", 512),
            ("target", 0), ("fu-epochs", 1), ("fu-batches", 2), ("history-window", 3), ("threads", 2)):
        p.add_argument("--"+key, type=int, default=default)
    for key, default in (("dominant-ratio", .7), ("smoothing", 1.), ("risk-ratio", .1),
                         ("epsilon", .01), ("radius-ratio", .25), ("min-progress", .1)):
        p.add_argument("--"+key, type=float, default=default)
    p.add_argument("--step", type=int, help="only this actual FU event, zero-based; omit for full history replay")
    p.add_argument("--requests", type=int, nargs="+", default=[5, 1], help="independent full-client requests")
    p.add_argument("--modes", nargs="+", default=["no_defense", "retention_only", "temporal"],
                   choices=["no_defense", "retention_only", "linear", "cumulative", "unsmoothed", "temporal"])
    return p


def main():
    a = parser().parse_args()
    if (min(a.rounds, a.local_epochs, a.batch_size, a.retain_per_class, a.fu_epochs, a.history_window, a.threads) < 1
            or min(a.train_batches, a.max_per_client, a.evaluate, a.fu_batches) < 0
            or not 0 <= a.target < 10 or not 0 < a.dominant_ratio <= 1):
        raise ValueError("invalid dataset/training sizes")
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if a.phase in ("run", "train"):
        prepare(a)
        gc.collect()
    if a.phase in ("run", "test"):
        test_requests(a)


if __name__ == "__main__":
    main()
