"""python -m experiments.request_purify --help"""
import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from .core import DetectorConfig
from .data import badfu_case, load_badfu_source, object_hash, smoke_case
from .engine import (calibrate, clock, continue_fl, protocol_signature, run_arm,
                     save_json, state_hash, train_history)


def parser():
    p = argparse.ArgumentParser(description="BadFU GPU experiment suite with zero detector FU simulations")
    p.add_argument("--dataset", choices=["badfu", "smoke"], default="badfu")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--record", default="BadFU_src/BadFU-main/record/badnet_dataset/pert_result.pt")
    p.add_argument("--data-root", default="BadFU_src/BadFU-main/data")
    p.add_argument("--out", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--calibration-seeds", type=int, nargs="+", default=[142, 143])
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--fu-epochs", type=int, default=1)
    p.add_argument("--post-rounds", type=int, default=10)
    p.add_argument("--post-scenarios", nargs="+", choices=["clean", "ongoing", "rejoin"],
                   default=["clean", "ongoing", "rejoin"])
    p.add_argument("--post-arms", nargs="+", choices=["none", "oracle", "detected", "random", "zero", "retrain"],
                   default=["none", "detected"])
    p.add_argument("--arms", nargs="+", choices=["none", "zero", "oracle", "detected", "random", "retrain"],
                   default=["none", "zero", "oracle", "detected", "random", "retrain"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--requester", type=int, default=5)
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--dominant-ratio", type=float, default=.7)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--strength", type=float, default=.5)
    p.add_argument("--relative-budget", type=float, default=.25)
    p.add_argument("--absolute-budget", type=float, default=1.)
    p.add_argument("--guard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--guard-delta", type=float, default=.01)
    p.add_argument("--max-rank", type=int, default=8)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--min-observations", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--benign-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--download", action="store_true", help="Allow torchvision CIFAR-10 download")
    p.add_argument("--resume", action="store_true", help="Resume only an identical suite configuration")
    p.add_argument("--check-only", action="store_true", help="Check CUDA and prepared data, without training")
    return p


def validate(args):
    if args.dataset == "badfu" and not args.device.startswith("cuda"):
        raise ValueError("Real BadFU experiments require --device cuda:N; CPU is reserved for --dataset smoke")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build; there is no CPU fallback.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.calibration_seeds)) != len(args.calibration_seeds):
        raise ValueError("Duplicate seeds are not independent repetitions")
    if set(args.seeds) & set(args.calibration_seeds):
        raise ValueError("Evaluation and calibration seeds must be disjoint")
    if min(args.rounds, args.local_epochs, args.fu_epochs, args.batch_size, args.eval_every) <= 0:
        raise ValueError("Training counts must be positive")
    if args.rounds < args.min_observations or args.window < args.min_observations:
        raise ValueError("Insufficient detector history for the requested minimum coverage")
    if min(args.post_rounds, args.workers, args.val_size, args.guard_delta) < 0:
        raise ValueError("Negative counts or validation tolerance")
    if not 0 <= args.requester < 6 or not 0 <= args.target < (3 if args.dataset == "smoke" else 10):
        raise ValueError("Invalid requester or target")
    if args.dataset == "smoke" and args.target != 0:
        raise ValueError("Synthetic fixture uses target 0")
    if not 0 < args.dominant_ratio < 1 or args.lr <= 0:
        raise ValueError("Invalid dominant ratio or learning rate")
    if not 0 <= args.strength <= 1 or min(args.relative_budget, args.absolute_budget) < 0:
        raise ValueError("Invalid projection strength/budget")
    if args.guard and args.dataset == "badfu" and args.val_size == 0:
        raise ValueError("--guard requires an independent validation split")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("Duplicate arms")


def package_results(root):
    rows = []
    for path in sorted(root.glob("eval-*/**/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"seed": result["seed"], "benign": result["benign"], "arm": result["arm"],
                     "acc": result["final"]["acc"], "asr": result["final"]["asr"],
                     "deleted_data_loss": result["final"]["deleted_data_loss"],
                     "status": result["detection"]["status"], "rank": result["detection"]["rank"],
                     "alarm": result["detection"].get("alarm"),
                     "wall_seconds": result["timing"]["wall_seconds"],
                     "detect_basis_seconds": result["timing"]["detect_and_basis_seconds"],
                     "projection_seconds": result["timing"]["projection_seconds"],
                     "guard_seconds": result["timing"]["guard_seconds"],
                     "diagnostic_fu_runs": result["diagnostic_fu_runs"]})
    if rows:
        with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    grouped = []
    for benign, arm in sorted({(r["benign"], r["arm"]) for r in rows}):
        group = [r for r in rows if r["benign"] == benign and r["arm"] == arm]
        item = {"benign": benign, "arm": arm, "independent_seeds": len(group)}
        for metric in ("acc", "asr", "wall_seconds"):
            values = [r[metric] for r in group]
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        grouped.append(item)
    paired = []
    for row in rows:
        baseline = next((r for r in rows if r["seed"] == row["seed"] and
                         r["benign"] == row["benign"] and r["arm"] == "none"), None)
        if baseline and row["arm"] != "none":
            paired.append({"seed": row["seed"], "benign": row["benign"], "arm": row["arm"],
                           "acc_delta_pp": row["acc"] - baseline["acc"],
                           "asr_delta_pp": row["asr"] - baseline["asr"]})
    benign_requests = [r for r in rows if r["benign"] and r["arm"] == "detected"]
    save_json(root / "summary.json", {"runs": rows, "groups": grouped, "paired_deltas": paired,
              "benign_detection": {"requests": len(benign_requests),
                                   "alarms": sum(bool(r["alarm"]) for r in benign_requests)},
              "note": "Separate evaluation arms are benchmark repetitions, not diagnostic FU inside the defense. Raw records are retained for auditing; this package does not certify deletion."})
    with zipfile.ZipFile(root / "results_to_share.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".csv", ".log"):
                archive.write(path, path.relative_to(root))


def main():
    args = parser().parse_args()
    validate(args)
    if args.dataset == "smoke":
        torch.set_num_threads(1)
        args.no_pretrained = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats()
    source_begin = clock(args.device)
    source = load_badfu_source(args.record, args.data_root, args.target, args.download) if args.dataset == "badfu" else None
    source_seconds = clock(args.device) - source_begin
    def make_case(seed, benign):
        return badfu_case(source, seed, benign, args.val_size, args.dominant_ratio) if source else smoke_case(seed, benign)
    sample = make_case(args.seeds[0], False)
    if args.check_only:
        print(json.dumps({"device": args.device, "client_counts": [len(d) for d in sample["clients"]],
                          "record_valid": True, "source_check_seconds": source_seconds}, indent=2))
        return
    if args.out is None:
        args.out = "artifacts/request-purify/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out).resolve()
    configuration = {k: v for k, v in vars(args).items() if k not in ("out", "resume", "check_only")}
    configuration["protocol_signature"] = protocol_signature(sample, args)
    fingerprint = object_hash(configuration)
    if root.exists():
        manifest = root / "suite.json"
        if not args.resume or not manifest.is_file():
            raise FileExistsError(f"Output already exists: {root}. Use a new output or --resume.")
        if json.loads(manifest.read_text(encoding="utf-8"))["configuration_sha256"] != fingerprint:
            raise ValueError("Resume configuration or source data differs from saved suite")
    else:
        root.mkdir(parents=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda, "device": args.device, "git_commit": git_commit,
                   "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None}
    save_json(root / "suite.json", {"configuration": configuration, "configuration_sha256": fingerprint,
              "environment": environment, "source_check_seconds": source_seconds,
              "status": "running", "dataset_warning": "Synthetic smoke only" if source is None else None})
    cfg = DetectorConfig(window=args.window, min_observations=args.min_observations, max_rank=args.max_rank)
    cfg.validate()
    started = time.perf_counter()
    print(f"Output: {root}\nDevice: {args.device}\nCalibration seeds: {args.calibration_seeds}; evaluation seeds: {args.seeds}", flush=True)
    def get_history(case, path):
        saved = path / "history.pt"
        if args.resume and saved.is_file():
            result = torch.load(saved, map_location="cpu", weights_only=False)
            if result["manifest"]["protocol_signature"] != protocol_signature(case, args):
                raise ValueError("History protocol mismatch")
            if result["manifest"]["seed"] != case["meta"]["seed"] or result["manifest"]["benign"] != case["meta"]["benign"]:
                raise ValueError("History identity mismatch")
            return result
        return train_history(case, args, path)
    try:
        normal = [get_history(make_case(seed, True), root / f"calibration-{seed}") for seed in args.calibration_seeds]
        t = clock(args.device)
        cfg, fitted = calibrate(normal, cfg, args.device)
        fitted["fit_seconds"] = clock(args.device) - t
        save_json(root / "detector_policy.json", fitted)
        del normal
        for seed in args.seeds:
            for benign in ([False, True] if args.benign_eval else [False]):
                case = make_case(seed, benign)
                case_dir = root / f"eval-{'benign' if benign else 'attack'}-{seed}"
                history = get_history(case, case_dir)
                if history["manifest"]["trajectory_id"] in {s["trajectory_id"] for s in fitted["sources"]}:
                    raise ValueError("Calibration/evaluation trajectory leakage")
                arms = [a for a in args.arms if not benign or a in ("none", "zero", "detected", "retrain")]
                hashes = {}
                for arm in arms:
                    dest = case_dir / arm
                    if args.resume and (dest / "result.json").is_file():
                        result = json.loads((dest / "result.json").read_text(encoding="utf-8"))
                        state = torch.load(dest / "model.pt", map_location="cpu", weights_only=True)
                        if state_hash(state) != result["model_sha256"]:
                            raise ValueError("Saved model hash mismatch")
                    else:
                        result, state = run_arm(history, case, args, dest, cfg, arm)
                    hashes[arm] = result["model_sha256"]
                    if not benign and arm in args.post_arms and args.post_rounds:
                        for scenario in args.post_scenarios:
                            path = dest / f"post_{scenario}.json"
                            if not (args.resume and path.is_file()):
                                save_json(path, continue_fl(state, case, args, scenario))
                    package_results(root)
                if "none" in hashes and "zero" in hashes:
                    equivalent = hashes["none"] == hashes["zero"]
                    save_json(case_dir / "zero_strength_check.json", {"exact_state_match": equivalent,
                              "none_sha256": hashes["none"], "zero_sha256": hashes["zero"]})
                    if not equivalent:
                        raise RuntimeError("Zero-strength arm differs from FU baseline")
        status = json.loads((root / "suite.json").read_text(encoding="utf-8"))
        status.update({"status": "complete", "this_invocation_seconds": time.perf_counter() - started,
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0})
        save_json(root / "suite.json", status)
        package_results(root)
        print(f"COMPLETE: send {root / 'results_to_share.zip'}", flush=True)
    except Exception as error:
        status = json.loads((root / "suite.json").read_text(encoding="utf-8"))
        status.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        save_json(root / "suite.json", status)
        package_results(root)
        raise


if __name__ == "__main__":
    main()
