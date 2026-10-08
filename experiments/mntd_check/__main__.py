"""Train small shadow-model detectors and inspect existing pre/post FU models.

Method reference: https://arxiv.org/abs/1910.03137
This independently implemented adaptation freezes shadow models in eval mode,
uses our ResNet/input normalization, and does not reproduce the paper's scale.
"""
import argparse
import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.request_purify.data import BadFUModel, file_hash
from experiments.request_purify.engine import cpu_state, save_json, save_torch, seed_all, state_hash


MEAN = torch.tensor([.4914, .4822, .4465])[:, None, None]
STD = torch.tensor([.2023, .1994, .2010])[:, None, None]


class Detector(nn.Module):
    def __init__(self, queries=10, shape=(3, 32, 32), classes=10):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, *shape) * .1)
        self.hidden = nn.Linear(queries * classes, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, responses):
        return self.output(self.hidden(responses.flatten()).relu()).squeeze()


def auc(labels, values):
    positives = [v for y, v in zip(labels, values) if y]
    negatives = [v for y, v in zip(labels, values) if not y]
    if not positives or not negatives:
        return None
    return sum((a > b) + .5 * (a == b) for a in positives for b in negatives) / (len(positives) * len(negatives))


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.no_grad()
def inspect(detector, model, state):
    model.load_state_dict(state)
    model.eval()
    return float(detector(model(detector.queries)))


def fit_detector(model, records, args, tune):
    seed_all(args.meta_seed)
    detector = Detector(args.queries).to(args.device)
    detector.queries.requires_grad_(tune)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    optimizer = torch.optim.Adam([p for p in detector.parameters() if p.requires_grad], lr=.001)
    rng = np.random.default_rng(args.meta_seed)
    for epoch in range(args.meta_epochs):
        losses = []
        for index in rng.permutation(len(records)):
            record = records[index]
            model.load_state_dict(torch.load(record["path"], map_location="cpu", weights_only=True))
            optimizer.zero_grad(set_to_none=True)
            score = detector(model(detector.queries))
            loss = nn.functional.binary_cross_entropy_with_logits(score, score.new_tensor(float(record["poisoned"])))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"[meta {'learned' if tune else 'fixed'} {epoch+1}/{args.meta_epochs}] loss={np.mean(losses):.5f}", flush=True)
    return detector.eval()


def normalize(x):
    return (x - MEAN) / STD


def poison(x, y, rng):
    # Independent generic patch/blend settings, not the user's BadFU trigger.
    x, y = x.clone(), y.clone()
    width = int(rng.choice([2, 3, 4, 5, 32]))
    top, left = (int(rng.integers(33-width)), int(rng.integers(33-width)))
    alpha = float(rng.uniform(.05, .2)) if width == 32 else 1.
    pattern = torch.tensor(rng.random((3, width, width)), dtype=x.dtype)
    target, ratio = int(rng.integers(10)), float(rng.uniform(.1, .5))
    chosen = torch.tensor(rng.choice(len(x), max(1, int(len(x)*ratio)), replace=False))
    x[chosen, :, top:top+width, left:left+width] = (1-alpha)*x[chosen, :, top:top+width, left:left+width] + alpha*pattern
    y[chosen] = target
    return x, y, {"width": width, "top": top, "left": left, "alpha": alpha, "target": target,
                  "ratio": ratio, "pattern": pattern}


@torch.no_grad()
def quality(model, x, y, trigger, device, batch):
    model.eval()
    def accuracy(images, labels):
        correct = 0
        for start in range(0, len(labels), batch):
            pred = model(normalize(images[start:start+batch]).to(device)).argmax(1).cpu()
            correct += int((pred == labels[start:start+batch]).sum())
        return 100 * correct / len(labels)
    result = {"clean_acc": accuracy(x, y), "trigger_asr": None}
    if trigger is not None:
        mask = y != trigger["target"]
        tx = x[mask].clone()
        w, t, l, a = (trigger[k] for k in ("width", "top", "left", "alpha"))
        tx[:, :, t:t+w, l:l+w] = (1-a)*tx[:, :, t:t+w, l:l+w] + a*trigger["pattern"]
        result["trigger_asr"] = accuracy(tx, torch.full((len(tx),), trigger["target"], dtype=y.dtype))
    return result


def make_shadow(factory, x, y, test_x, test_y, seed, poisoned, args, path):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    px, py, trigger = poison(x, y, rng) if poisoned else (x, y, None)
    model = factory(pretrained=not args.no_pretrained).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.shadow_lr)
    loader = DataLoader(TensorDataset(normalize(px), py), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(args.shadow_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x.to(args.device)), batch_y.to(args.device))
            loss.backward(); optimizer.step()
    metrics = quality(model, test_x, test_y, trigger, args.device, args.batch_size)
    save_torch(path, cpu_state(model))
    return {"path": str(path), "seed": seed, "poisoned": poisoned, **metrics,
            "trigger": {k: v for k, v in trigger.items() if k != "pattern"} if trigger else None}


def suite_inputs(root, seed):
    root = Path(root).resolve()
    conditions = []
    manifests = []
    for benign in (True, False):
        folder = root / f"eval-{'benign' if benign else 'attack'}-{seed}"
        for path in (folder / "manifest.json", folder / "history.pt", folder / "none/model.pt", folder / "none/result.json"):
            if not path.is_file():
                raise FileNotFoundError(f"Missing existing experiment artifact: {path}")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["dataset"] != "badfu" or manifest["seed"] != seed or manifest["benign"] != benign:
            raise ValueError("Suite manifest identity mismatch")
        manifests.append(manifest)
        history = torch.load(folder / "history.pt", map_location="cpu", weights_only=False)
        result = json.loads((folder / "none/result.json").read_text(encoding="utf-8"))
        after = torch.load(folder / "none/model.pt", map_location="cpu", weights_only=True)
        if state_hash(after) != result["model_sha256"] or result["trajectory_id"] != manifest["trajectory_id"]:
            raise ValueError("FU checkpoint or trajectory mismatch")
        if history["manifest"]["trajectory_id"] != manifest["trajectory_id"]:
            raise ValueError("History trajectory mismatch")
        conditions.extend([{"condition": f"{'normal' if benign else 'attack'}_before", "state": history["final"],
                            "metrics": history["training_metrics"][-1], "path": str(folder / "history.pt")},
                           {"condition": f"{'normal' if benign else 'attack'}_after", "state": after,
                            "metrics": result["final"], "path": str(folder / "none/model.pt")}])
    for key in ("validation_ids", "partitions", "clean_train_sha256", "record_sha256", "requester"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Normal/attack experiments differ in {key}")
    meta = manifests[0]
    used = set(meta["bd_ids"] + meta["cv_ids"] + sum(meta["partitions"], []))
    if used.intersection(meta["validation_ids"]):
        raise ValueError("Shadow auxiliary inputs overlap target training data")
    return conditions, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--suite", help="Existing BadFU suite, containing attack/benign history.pt and none/model.pt")
    mode.add_argument("--pilot", action="store_true", help="Standalone shadow-model GPU feasibility experiment; NOT a FU test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--train-per-class", type=int, default=8)
    p.add_argument("--calibration-per-class", type=int, default=4)
    p.add_argument("--test-per-class", type=int, default=4)
    p.add_argument("--shadow-epochs", type=int, default=3)
    p.add_argument("--shadow-lr", type=float, default=.01)
    p.add_argument("--aux-size", type=int, default=1000)
    p.add_argument("--quality-size", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--queries", type=int, default=10)
    p.add_argument("--meta-epochs", type=int, default=10)
    p.add_argument("--meta-seed", type=int, default=8193)
    p.add_argument("--shadow-seed", type=int, default=300000)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU efficacy fallback")
    if min(args.train_per_class, args.calibration_per_class, args.test_per_class, args.shadow_epochs,
           args.aux_size, args.quality_size, args.batch_size, args.queries, args.meta_epochs) < 1:
        raise ValueError("Counts must be positive")
    if args.shadow_lr <= 0 or args.quality_size > 10000:
        raise ValueError("Invalid learning rate or quality subset size")
    conditions, manifest = suite_inputs(args.suite, args.seed) if args.suite else ([], None)
    root = Path(args.out).resolve(); root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    status = {"status": "running", "configuration": vars(args), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "mode": "existing_FU_screen" if args.suite else "shadow_only_pilot",
              "code_sha256": file_hash(__file__), "diagnostic_fu_runs": 0,
              "note": "MNTD-style eval-mode adaptation, not an original-scale reproduction. Scores are not calibrated probabilities."}
    save_json(root / "experiment.json", status)
    try:
        from torchvision.datasets import CIFAR10
        trainset = CIFAR10(args.data_root, train=True, download=args.download)
        testset = CIFAR10(args.data_root, train=False, download=args.download)
        if manifest:
            if hashlib.sha256(trainset.data.tobytes()).hexdigest() != manifest["clean_train_sha256"]:
                raise ValueError("CIFAR source differs from saved suite")
            ids = manifest["validation_ids"]
            if args.aux_size > len(ids):
                raise ValueError("aux-size exceeds reserved clean validation data")
            ids = ids[:args.aux_size]
        else:
            if args.aux_size > len(trainset):
                raise ValueError("aux-size exceeds training data")
            ids = np.random.default_rng(args.shadow_seed).choice(len(trainset), args.aux_size, replace=False).tolist()
        x = torch.tensor(trainset.data[ids]).permute(0, 3, 1, 2).float() / 255
        y = torch.tensor(np.array(trainset.targets)[ids])
        tx = torch.tensor(testset.data[:args.quality_size]).permute(0, 3, 1, 2).float() / 255
        ty = torch.tensor(testset.targets[:args.quality_size])
        save_json(root / "auxiliary.json", {"ids": ids, "source": "suite_validation" if manifest else "pilot_CIFAR_train",
                  "normalization": {"mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist()}})
        records = []; index = 0
        shadow_dir = root / "shadows"; shadow_dir.mkdir()
        for split, count in (("train", args.train_per_class), ("calibration", args.calibration_per_class), ("test", args.test_per_class)):
            for poisoned in (False, True):
                for _ in range(count):
                    seed = args.shadow_seed + index
                    record = make_shadow(BadFUModel, x, y, tx, ty, seed, poisoned, args, shadow_dir / f"{index}.pt")
                    record["split"] = split; records.append(record); index += 1
                    save_json(root / "shadow_inventory.json", records)
                    print(f"[shadow {index} {split} poison={poisoned}] ACC={record['clean_acc']:.2f} ASR={record['trigger_asr']}", flush=True)
        model = BadFUModel().to(args.device)
        summary = {"mode": status["mode"], "detectors": {}, "diagnostic_fu_runs": 0,
                   "warning": "Few shadow models; distribution transfer and independent normal FU calibration remain unverified. Poisoned labels indicate training recipe; inspect shadow ASR/ACC before interpreting detection."}
        rows = []
        for name, tune in (("fixed_queries", False), ("learned_queries", True)):
            detector = fit_detector(model, [r for r in records if r["split"] == "train"], args, tune)
            scores = []
            for record in records:
                if record["split"] == "train":
                    continue
                state = torch.load(record["path"], map_location="cpu", weights_only=True)
                scores.append({"seed": record["seed"], "split": record["split"], "poisoned": record["poisoned"],
                               "score": inspect(detector, model, state)})
            clean_cal = [r["score"] for r in scores if r["split"] == "calibration" and not r["poisoned"]]
            threshold = max(clean_cal)
            test = [r for r in scores if r["split"] == "test"]
            result = {"threshold": threshold, "threshold_rule": "maximum independent clean-shadow calibration score; no population FPR guarantee",
                      "scores": scores, "test_auc": auc([r["poisoned"] for r in test], [r["score"] for r in test]),
                      "test_fpr": np.mean([r["score"] > threshold for r in test if not r["poisoned"]]).item(),
                      "test_tpr": np.mean([r["score"] > threshold for r in test if r["poisoned"]]).item()}
            for condition in conditions:
                synchronize(args.device); t = time.perf_counter()
                value = inspect(detector, model, condition["state"])
                synchronize(args.device)
                rows.append({"detector": name, "condition": condition["condition"], "score": value,
                             "threshold": threshold, "alarm": value > threshold,
                             "acc": condition["metrics"]["acc"], "asr": condition["metrics"]["asr"],
                             "model_sha256": state_hash(condition["state"]), "source": condition["path"],
                             "load_and_infer_seconds": time.perf_counter() - t})
            save_torch(root / f"{name}.pt", {"state_dict": cpu_state(detector), "threshold": threshold,
                       "configuration": vars(args), "architecture": "BadFUModel_ResNet18", "query_space": "normalized_CIFAR_tensor"})
            summary["detectors"][name] = result
            print(f"[{name}] held-out shadow AUC={result['test_auc']:.4f}, TPR={result['test_tpr']:.3f}, FPR={result['test_fpr']:.3f}", flush=True)
        if rows:
            with (root / "four_conditions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            summary["four_conditions"] = rows
        else:
            summary["four_conditions"] = None
            summary["FU_status"] = "not_evaluated; original pre/post checkpoints unavailable in this pilot"
        save_json(root / "summary.json", summary)
        status.update(status="complete", wall_seconds=time.perf_counter() - started)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(root / "experiment.json", status)
        with zipfile.ZipFile(root / "results_to_share.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.glob("*")):
                if path.suffix in (".json", ".csv"):
                    archive.write(path, path.name)


if __name__ == "__main__":
    main()
