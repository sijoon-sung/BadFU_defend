"""Run pinned FUBA training, verify/observe its trajectory, and export small evidence.

Completed matching training is reused; incomplete training is never overwritten.
This runner does not equate a successful pipeline with successful attack detection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    # Same canonical JSON encoding as hrac_request_gate.fingerprint.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    temp = path.with_suffix(".pending")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    p.add_argument("--data-root", type=Path, default=ROOT / "FUBA/data")
    p.add_argument("--run-dir", type=Path, help="New training folder or a completed matching run")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clients", type=int, help="Default: 20; --smoke: 6")
    p.add_argument("--warm-up", type=int, default=5)
    p.add_argument("--participant-rate", type=float, default=.4)
    p.add_argument("--iid", action="store_true", help="Default is Non-IID Dirichlet alpha=0.5")
    p.add_argument("--alpha", type=float, default=.5)
    p.add_argument("--processes", type=int, default=5)
    p.add_argument("--gpus", type=int, default=1, help="MPI training GPUs; 0 for CPU")
    p.add_argument("--smoke", action="store_true", help="2 rounds, 64 samples/client; pipeline check only")
    p.add_argument("--download", action="store_true", help="Download missing public dataset before MPI starts")
    p.add_argument("--validation-out", type=Path, help="New directory if observation settings/code changed")
    p.add_argument("--pool", type=Path, help="Optional existing unlabeled image pool")
    p.add_argument("--requests", type=Path, help="Optional independently observed HRAC request bundle")
    p.add_argument("--device", default="auto", help="Observation device, e.g. cuda:0 or cpu")
    p.add_argument("--batch-size", type=int, default=64, help="Observation batch size")
    p.add_argument("--threads", type=int, default=4, help="Observation CPU threads")
    p.add_argument("--export-root", type=Path, default=ROOT / "experiments/detection/hrac/results")
    args = p.parse_args(argv)
    args.clients = args.clients if args.clients is not None else (6 if args.smoke else 20)
    if args.smoke:
        args.rounds, args.warm_up, args.participant_rate = 2, 1, 1.0
    if args.clients < 5 or args.processes < 5 or args.gpus < 0:
        p.error("FUBA requires >=5 clients, >=5 MPI processes, and gpus >=0")
    if args.rounds < 1 or not 0 <= args.warm_up <= args.rounds:
        p.error("Require rounds >=1 and 0 <= warm-up <= rounds")
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        p.error("alpha must be finite and positive")
    if not 0 < args.participant_rate <= 1 or int(args.clients * args.participant_rate) < 1:
        p.error("participant-rate must select at least one client")
    if min(args.batch_size, args.threads) < 1 or not 0 <= args.seed < 2**32:
        p.error("positive observation sizes and seed in [0, 2**32) required")
    if args.run_dir is None:
        suffix = "_smoke" if args.smoke else ""
        args.run_dir = ROOT / f"artifacts/fuba_{args.dataset}{suffix}_s{args.seed}"
    args.validation_out = args.validation_out or args.run_dir / "hrac_validation_v1"
    for name in ("data_root", "run_dir", "validation_out", "export_root", "pool", "requests"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.run_dir.is_relative_to(args.validation_out):
        p.error("validation-out cannot be the training folder or its ancestor")
    for name in ("pool", "requests"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            p.error(f"Missing --{name} file: {path}")
    return args


def training_settings(args):
    return dict(attack="fuba", dataset=args.dataset, rounds=args.rounds, seed=args.seed,
                smoke=args.smoke, clients=args.clients, warm_up=args.warm_up,
                participant_rate=args.participant_rate, non_iid=not args.iid, alpha=args.alpha)


def reuse_training(args):
    if not args.run_dir.exists():
        return False
    path = args.run_dir / "protocol.json"
    if not path.is_file() or read(path).get("status") != "complete":
        raise ValueError("Existing training is incomplete or unrecognized. It was preserved. "
                         "Training has no resume support; use a new --run-dir (do not delete checkpoints).")
    saved = read(path)["options"]
    differences = {k: {"saved": saved.get(k), "requested": v}
                   for k, v in training_settings(args).items()
                   if saved.get(k) != v and not (k == "alpha" and args.iid)}
    if differences:
        raise ValueError(f"Completed training settings differ: {differences}. "
                         "Match the saved settings or use a new --run-dir.")
    return True


def commands(args):
    train = [sys.executable, "-u", str(ROOT / "reproduction/training/run.py"), "fuba",
             "--data-root", str(args.data_root), "--out", str(args.run_dir)]
    for name in ("dataset", "rounds", "seed", "clients", "warm_up", "participant_rate", "alpha", "processes", "gpus"):
        train += ["--"+name.replace("_", "-"), str(getattr(args, name))]
    for flag, enabled in (("--non-iid", not args.iid), ("--smoke", args.smoke), ("--download", args.download)):
        if enabled:
            train.append(flag)
    validate = [sys.executable, "-u", str(ROOT / "experiments/detection/hrac/validate_fuba_detection.py"),
                "--run-dir", str(args.run_dir), "--out", str(args.validation_out),
                "--data-root", str(args.data_root), "--device", args.device,
                "--threads", str(args.threads), "--batch-size", str(args.batch_size)]
    for name in ("pool", "requests"):
        if getattr(args, name) is not None:
            validate += ["--"+name, str(getattr(args, name))]
    return train, validate


def expected_archive(args):
    """Resolve this request's output by identity, never by newest file/glob order."""
    protocol = read(args.validation_out / "observation_protocol.json")
    observation_id = digest(protocol)
    manifest = read(args.run_dir / "fl/manifest.json")
    if args.requests:
        requests = read(args.requests)
    else:
        known = sorted({i for entry in manifest["rounds"] for i in entry["client_ids"]})
        requests = dict(schema=1, origin="prospective_full_client_requests", observation_id=observation_id,
                        requests=[dict(id=f"client:{i}", round=args.rounds,
                                       removal=[float(j == i) for j in range(args.clients)]) for i in known])
    if requests.get("observation_id") != observation_id:
        raise ValueError("Requests changed or refer to a different observation protocol")
    return args.validation_out / f"fuba_hrac_evidence_{digest(requests)[:12]}.zip"


def export_evidence(args):
    archive = expected_archive(args)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Corrupted or duplicate ZIP entries")
        inventory = json.loads(z.read("evidence/inventory.json"))
        for name, expected in inventory.items():
            if hashlib.sha256(z.read("evidence/"+name)).hexdigest() != expected:
                raise ValueError(f"Exported evidence hash mismatch: {name}")
        for name, src in (("training_protocol.json", args.run_dir / "protocol.json"),
                          ("manifest.json", args.run_dir / "fl/manifest.json"),
                          ("observation_protocol.json", args.validation_out / "observation_protocol.json")):
            if hashlib.sha256(z.read("evidence/"+name)).hexdigest() != sha(src):
                raise ValueError(f"Evidence belongs to different inputs: {name}")
        summary = json.loads(z.read("evidence/summary.json"))
        if (summary.get("status") != "analysis_complete" or summary.get("attack") != "fuba"
                or summary.get("dataset") != args.dataset or summary.get("smoke_only") != args.smoke):
            raise ValueError("Validation did not produce the expected analysis")
        report = z.read("evidence/report.txt")
    checksum = sha(archive)
    dest = args.export_root / args.run_dir.name / checksum[:16]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / archive.name
    if target.exists() and sha(target) != checksum:
        raise ValueError(f"Export collision; existing evidence preserved: {target}")
    if not target.exists():
        shutil.copyfile(archive, target)
    if sha(target) != checksum:
        raise ValueError("Export copy checksum mismatch")
    write(dest / "summary.json", summary)
    (dest / "report.txt").write_bytes(report)
    write(dest / "export.json", dict(schema=1, archive=target.name, sha256=checksum,
          training_settings=training_settings(args), runner_sha256=sha(Path(__file__)),
          smoke_only=args.smoke, detection_efficacy_verified=False,
          both_channels_observed=summary["hrac"]["both_channels_observed"],
          alarm=summary["hrac"]["decision"]["alarm"]))
    return dest


def execute(args, runner=None):
    runner = runner or (lambda command: subprocess.run(command, cwd=ROOT, check=True))
    reused = reuse_training(args)
    train, validate = commands(args)
    receipt = dict(schema=1, status="running", training_reused=reused,
                   training_settings=training_settings(args), stages=[], commands=dict(train=train, validate=validate))
    start, stage = time.monotonic(), "training"
    try:
        if reused:
            print(f"[1/3] Reusing completed training: {args.run_dir}; integrity is checked next", flush=True)
        else:
            print(f"[1/3] FUBA {args.dataset}: {args.rounds} rounds, {args.clients} clients, "
                  f"{'IID' if args.iid else 'Dirichlet alpha='+str(args.alpha)}", flush=True)
            runner(train)
        # Check the child really completed with the requested training settings.
        if not reuse_training(args):
            raise ValueError("Training returned without a complete protocol")
        receipt["stages"].append(dict(name=stage, status="reused" if reused else "complete"))
        write(args.run_dir / "experiment_status.json", receipt)
        stage = "validation"
        print("[2/3] Verifying saved states, measuring responses, and evaluating HRAC/baselines", flush=True)
        runner(validate)
        receipt["stages"].append(dict(name=stage, status="complete"))
        stage = "export"
        dest = export_evidence(args)
        receipt.update(status="complete", export_dir=str(dest))
        receipt["stages"].append(dict(name=stage, status="complete"))
    except BaseException as exc:
        receipt.update(status="failed", failed_stage=stage, error=str(exc))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic()-start
        if args.run_dir.is_dir():
            write(args.run_dir / "experiment_status.json", receipt)
    print(f"[3/3] Upload this small result folder:\n{dest}", flush=True)
    print("Pipeline complete. Rank scores are not a calibrated detection verdict. "
          "Without independent request data, intra remains unobserved.", flush=True)
    if args.smoke:
        print("SMOKE ONLY: this 2-round run is not attack/detection efficacy evidence.", flush=True)
    return receipt


if __name__ == "__main__":
    execute(parse_args())
