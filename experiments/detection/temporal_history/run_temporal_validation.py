"""Pilot: reuse attack/normal s42 and add two held-out normal histories s43/s44.

No original attack is retrained. New runs use the unchanged pinned FUBA runner.
Full trajectories are kept; small evidence/results go to the upload directory.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.detection.temporal_history.validate_temporal_history import default_plan, execute, write_json
from experiments.detection.temporal_history import temporal_history as th


def command(seed, args):
    result = [sys.executable, "-u", str(ROOT / "experiments/detection/hrac/run_fuba_experiment.py"),
              "--dataset", "cifar10", "--seed", str(seed), "--warm-up", "100",
              "--run-dir", str(ROOT / f"artifacts/fuba_cifar10_benign_s{seed}"),
              "--data-root", str(args.data_root), "--device", args.device, "--gpus", str(args.gpus)]
    if args.download:
        result.append("--download")
    return result


def cached_normal(seed):
    paths = sorted((ROOT / f"experiments/detection/hrac/results/fuba_cifar10_benign_s{seed}").glob("*/*.zip"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"Several exports for seed {seed}; use an explicit analysis plan")
    e = th.load_evidence(paths[0])
    if (not e["benign"] or e["group"] != f"fuba:cifar10:{seed}" or e["stratum"] != dict(
            dataset="cifar10", rounds=100, clients=20, participant_rate=.4, non_iid=True,
            alpha=.5, smoke=False, attack="fuba", actual_clients=20)):
        raise ValueError(f"Seed {seed} export differs from the declared pilot conditions")
    return paths[0].relative_to(ROOT).as_posix()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cached-only", action="store_true", help="Only replay the already-uploaded seed-42 pair")
    p.add_argument("--data-root", type=Path, default=ROOT / "FUBA/data")
    p.add_argument("--device", default="auto")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--download", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Show commands without training, analysis or file writes")
    a = p.parse_args(argv)
    if a.gpus < 0:
        p.error("gpus must be nonnegative")
    a.data_root = a.data_root.resolve()
    plan = default_plan()
    if a.cached_only:
        if a.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            execute(plan)
        return
    seeds = (43, 44)
    cached = {seed: cached_normal(seed) for seed in seeds}
    missing = [seed for seed in seeds if cached[seed] is None]
    commands = [command(seed, a) for seed in missing]
    if a.dry_run:
        print(json.dumps(dict(new_normal_seeds=missing, commands=commands,
            attack_retraining=False, calibration_runs=0, required_free_gib_if_new=50*len(missing)+5,
            evaluation="New normal runs test generalization; seed-42 attack remains exploratory."), indent=2))
        return
    # Conservative peak requirement; preserve checkpoints and never auto-delete.
    new = sum(not (ROOT / f"artifacts/fuba_cifar10_benign_s{seed}").exists() for seed in missing)
    required = (50*new+5) * 1024**3
    if shutil.disk_usage(ROOT).free < required:
        raise ValueError(f"Need approximately {required/1024**3:.0f} GiB free for new retained trajectories. "
                         "No training started; choose a workspace disk with enough space.")
    receipt_dir = ROOT / "artifacts/temporal_pilot_v1"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt = dict(status="running", seeds=list(seeds), completed=[], attack_retraining=False)
    try:
        for seed in seeds:
            if cached[seed] is None:
                print(f"Normal seed {seed}: training, trajectory checks, observations and export", flush=True)
                subprocess.run(command(seed, a), cwd=ROOT, check=True)
                cached[seed] = cached_normal(seed)
                if cached[seed] is None:
                    raise RuntimeError("Training runner returned without a verified evidence export")
            receipt["completed"].append(seed)
            write_json(receipt_dir / "status.json", receipt)
            plan["test"].append(dict(name=f"fuba_benign_s{seed}", evidence=cached[seed]))
        plan["note"] = ("Fixed v1 fit on benign seed 42; held-out normal seeds 43/44; "
                        "previously inspected attack seed 42 remains exploratory. No calibrated FPR yet.")
        out = execute(plan)
        receipt.update(status="complete", results=str(out))
    except BaseException as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter()-started
        write_json(receipt_dir / "status.json", receipt)
    print("Upload experiments/detection/hrac/results, including both new normal evidence ZIPs and temporal_history.", flush=True)


if __name__ == "__main__":
    main()
