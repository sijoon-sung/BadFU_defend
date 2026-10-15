"""One entry point for fresh upstream attacks and full-state verification."""
import argparse
import difflib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.trajectory import json_write, sha256, verify


def replace(source, old, new, count=1):
    actual = source.count(old)
    if actual != count:
        raise ValueError(f"Pinned source changed: expected {count}, got {actual}: {old!r}")
    return source.replace(old, new)


def lock():
    return json.loads((ROOT / "reproduction/upstream_lock.json").read_text(encoding="utf-8"))["sources"]


def check_sources(which=None):
    reports = []
    for key, item in lock().items():
        if which and which != key:
            continue
        for name, expected in item["files"].items():
            path = ROOT / item["directory"] / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Official source mismatch: {path}. Restore pinned bytes before running.")
        reports.append({"attack": key, "commit": item["commit"], "files_verified": len(item["files"])})
    return reports


def mpi_path():
    found = shutil.which("mpiexec")
    if found:
        return found
    try:
        dist = importlib.metadata.distribution("impi-rt")
        for file in dist.files:
            if Path(file).name.lower() == "mpiexec.exe":
                return str(Path(dist.locate_file(file)).resolve())
    except importlib.metadata.PackageNotFoundError:
        pass
    raise RuntimeError("mpiexec missing. Windows: pip install mpi4py impi-rt; Linux: install MPI and mpi4py")


def execute(args):
    import torch
    check_sources(args.attack)
    source = lock()[args.attack]
    args.out = str(Path(args.out).resolve())
    args.data_root = str(Path(args.data_root).resolve())
    if not Path(args.data_root).is_dir() and not args.download:
        raise FileNotFoundError(f"Dataset root missing: {args.data_root}")
    if args.attack == "badfu":
        args.record_root = str(Path(args.record_root).resolve())
        if not (Path(args.record_root) / "pert_result.pt").is_file():
            raise FileNotFoundError("BadFU requires the author's generated pert_result.pt AND referenced images; an FL checkpoint cannot replace them")
        if args.rounds < 2 or args.rounds % 2:
            raise ValueError("The author's FU indexes r+1 at even r: rounds must be even and >= 2")
        if args.local_epochs < 1:
            raise ValueError("local-epochs must be positive")
        args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        if args.processes < 5 or args.clients < 5 or args.rounds < 1:
            raise ValueError("FUBA IBA needs >=5 MPI processes: server + 3 cooperative workers + evaluator")
        if not 0 <= args.warm_up <= args.rounds or not 0 < args.participant_rate <= 1:
            raise ValueError("Invalid warmup or participant rate")
        if int(args.clients * args.participant_rate) < 1:
            raise ValueError("No participating clients")
        args.mpiexec = mpi_path()
        if args.download:
            import torchvision.datasets as datasets
            dataset_class = datasets.MNIST if args.dataset == 'mnist' else datasets.CIFAR10
            # Complete downloads before MPI ranks try to read/extract the same files.
            for train in (True, False):
                dataset_class(args.data_root, train=train, download=True)
    if args.smoke:
        args.rounds = 2
        if args.attack == "badfu":
            args.local_epochs = 1
        else:
            args.warm_up, args.participant_rate = 1, 1.0
    options = vars(args).copy()
    options["upstream_commit"] = source["commit"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / "runtime"
    runtime.mkdir()
    # Snapshot instrumentation too: updating the repository must not change an active run.
    (out / "reproduction").mkdir()
    for adapter in (ROOT / "reproduction").glob("*.py"):
        shutil.copyfile(adapter, out / "reproduction" / adapter.name)
    for name in source["files"]:
        dest = runtime / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source["directory"] / name, dest)
    patches = []

    def patch_file(path, transform):
        original = path.read_text(encoding="utf-8")
        changed = transform(original, replace)
        patches.extend(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                       fromfile=str(path.relative_to(runtime)), tofile=str(path.relative_to(runtime))))
        path.write_text(changed, encoding="utf-8", newline="\n")

    if args.attack == "badfu":
        from reproduction.badfu_adapter import patch
        patch_file(runtime / "badfu.py", lambda s, r: patch(s, options, r))
        command = [sys.executable, "-u", str(runtime / "badfu.py")]
    else:
        from reproduction.fuba_adapter import patch_files
        patch_files(runtime, options, patch_file)
        command = [args.mpiexec, "-n", str(args.processes), sys.executable, "-u", "-m", "mpi4py", str(runtime / "main.py"),
            "--dataset", args.dataset, "--attackMethod", "iba", "--name", "run",
            "--no_detail_test", "--no_global_test", "--n_gpu", str(args.gpus),
            "--nb_clients", str(args.clients), "--communication_rounds", str(args.rounds),
            "--warm_up", str(args.warm_up), "--participant_rate", str(args.participant_rate)]
        if args.non_iid:
            command += ["--non_iid", "--non_iid_type", "Dirichlet", "--alpha", str(args.alpha)]
        if args.legacy_export:
            command.append("--save")
    (out / "runtime.patch").write_text("".join(patches), encoding="utf-8")
    adapter_hashes = {p.name: sha256(p) for p in (ROOT / "reproduction").glob("*.py")}
    protocol = {"status": "running", "options": options, "command": command,
        "upstream": {k: v for k, v in source.items() if k != "files"}, "adapter_sha256": adapter_hashes,
        "runtime_patch_sha256": sha256(out / "runtime.patch"), "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}}
    json_write(out / "protocol.json", protocol)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(out) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    if args.attack == "fuba" and os.name == "nt":
        env["PATH"] = str(Path(args.mpiexec).parent) + os.pathsep + env["PATH"]
    start = time.monotonic()
    try:
        with open(out / "console.log", "w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=runtime, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                except BaseException:
                    process.terminate()
                    raise
                returncode = process.wait()
        if returncode:
            raise RuntimeError(f"Upstream process failed ({returncode}). See {out / 'console.log'}")
        phases = ["clean5", "bd5", "cover5", "full6", "fu_remove5", "fu_remove1"] if args.attack == "badfu" else ["fl"]
        reports = [verify(out / phase) for phase in phases]
        json_write(out / "verification.json", reports)
        protocol["status"] = "complete"
    except BaseException as exc:
        protocol.update(status="failed", error=str(exc))
        raise
    finally:
        protocol["elapsed_seconds"] = time.monotonic() - start
        json_write(out / "protocol.json", protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="attack", required=True)
    sub.add_parser("check", help="Verify all pinned official source hashes")
    v = sub.add_parser("verify", help="Verify a saved phase's full trajectory")
    v.add_argument("path")
    for name in ("badfu", "fuba"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", required=True)
        p.add_argument("--out", required=True, help="New output directory; existing runs are never overwritten")
        p.add_argument("--rounds", type=int, default=40 if name == "badfu" else 100)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--smoke", action="store_true", help="2 rounds and small real-data subsets, NOT an efficacy experiment")
        p.add_argument("--download", action="store_true")
        if name == "badfu":
            p.add_argument("--record-root", required=True)
            p.add_argument("--local-epochs", type=int, default=5)
            p.add_argument("--device")
        else:
            p.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
            p.add_argument("--clients", type=int, default=20)
            p.add_argument("--processes", type=int, default=5)
            p.add_argument("--gpus", type=int, default=1)
            p.add_argument("--warm-up", type=int, default=5)
            p.add_argument("--participant-rate", type=float, default=.4)
            p.add_argument("--non-iid", action="store_true")
            p.add_argument("--alpha", type=float, default=1.0)
            p.add_argument("--legacy-export", action="store_true", help="Also write the author's large pickle checkpoints for legacy tools")
    args = parser.parse_args()
    if args.attack == "check":
        print(json.dumps(check_sources(), indent=2))
    elif args.attack == "verify":
        print(json.dumps(verify(args.path), indent=2))
    else:
        execute(args)


if __name__ == "__main__":
    main()
