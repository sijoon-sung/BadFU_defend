"""Run the registered FUBA tension experiment (plan/train/collect/analyze).

From FUBA: python experiment_tension.py --profile pilot --stage all
Pilot: 3 clean calibration + 3 clean test + 3 existing development attacks per
IID/owner stratum. Confirm: 19 clean calibration + 10 clean test + 5 NEW attacks
per stratum. Training uses the existing author pipeline; no deletion is run.
"""


from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
from reproduction.common.paths import research_source
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.detection.hrac.benchmark_tension import sha256, score_ledger, summarize, validate_jobs

HERE = (Path(__file__).resolve().parents[3]/"FUBA")


def make_jobs(profile, root):
    if profile not in ("pilot", "confirm"):
        raise ValueError("unknown profile")
    jobs = []
    for stratum in ("iid", "owner"):
        splits = [("calibration", range(100, 103 if profile == "pilot" else 119)),
                  ("clean_test", range(200, 203 if profile == "pilot" else 210)),
                  ("attack_development", range(3))] if profile == "pilot" else [
                      ("calibration", range(100, 119)), ("clean_test", range(200, 210)), ("attack_test", range(20, 25))]
        for split, seeds in splits:
            for seed in seeds:
                attack = split.startswith("attack")
                name = (f"bd_{stratum}_r100w5_k20paper_s{seed}" if split == "attack_development"
                        else f"tenv3_{'attack' if attack else 'clean'}_{stratum}_s{seed}")
                extra = f"--split_seed {522+seed} --participant_rate 0.4"
                if stratum == "owner":
                    extra += " --non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1"
                job = {"id": name, "stratum": stratum, "seed": seed, "actual_seed": 522+seed, "split": split,
                       "cache": str((root/"caches"/f"{name}.npz").resolve()), "rounds": 100, "K": 20,
                       "command": ["bash", "run_real_fuba.sh", name, str(522+seed)],
                       "env": {"DATASET": "mnist", "NB_CLIENTS": "20", "ROUNDS": "100", "WARMUP": "5" if attack else "1000",
                               "PARTICIPANT_RATE": "0.4", "EXTRA_ARGS": extra}}
                if attack:
                    job.update(eval_suppressor=4, eval_target=8)
                jobs.append(job)
    validate_jobs(jobs)
    return jobs


def checked_run(command, log, env=None):
    # Keep console progress visible; never silently overwrite an old command log.
    index, candidate = 0, log
    while candidate.exists():
        index += 1; candidate = log.with_name(f"{log.stem}.{index}{log.suffix}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with candidate.open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(command, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True); output.write(line); output.flush()
        code = proc.wait()
    if code:
        raise RuntimeError(f"command failed ({code}); see {candidate}")


def checkpoint_state(job, folder):
    required = [folder/f"{kind}_net_model_{job['id']}_round_{r}.pkl"
                for kind in ("global", "local") for r in range(1, job["rounds"]+1)]
    exists = [p.exists() for p in required]
    return "complete" if all(exists) else ("partial" if any(exists) else "missing")


def validate_cache(job, probe, checkpoints):
    cache = Path(job["cache"])
    meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    expected = {"name": job["id"], "rounds": job["rounds"], "K": job["K"], "dataset": "mnist", "probe": probe,
                "collector_sha256": sha256(research_source("collect_tension.py")), "cache_sha256": sha256(cache)}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"cache configuration/source mismatch: {cache}; use a new --out folder")
    for source in meta["checkpoint_sources"]:
        if sha256(checkpoints/source["file"]) != source["sha256"]:
            raise ValueError(f"checkpoint changed after collection: {source['file']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("pilot", "confirm"), default="pilot")
    p.add_argument("--stage", choices=("plan", "train", "collect", "analyze", "all"), default="plan")
    p.add_argument("--out", type=Path)
    p.add_argument("--checkpoints", type=Path, default=HERE/"checkpoints")
    p.add_argument("--data-root", type=Path, default=HERE/"data")
    p.add_argument("--probe", type=int, default=128)
    p.add_argument("--n-gpu", type=int, default=1)
    a = p.parse_args()
    root = (a.out or HERE/"logs"/"tension_v3"/a.profile).resolve()
    checkpoints, data_root = a.checkpoints.resolve(), a.data_root.resolve()
    if a.stage in ("train", "all") and checkpoints != (HERE/"checkpoints").resolve():
        raise ValueError("author training writes FUBA/checkpoints; custom --checkpoints only supports collect/analyze")
    root.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(a.profile, root)
    manifest = {"schema": 1, "profile": a.profile, "probe": a.probe, "alpha": .05, "jobs": jobs,
                "primary": "logprob/mean_before/window:0; selective_v3 versus linear_release",
                "source_sha256": {name: sha256(HERE/"research"/name if (HERE/"research"/name).is_file() else HERE/name) for name in (
                    "tension_audit.py", "benchmark_tension.py", "collect_tension.py", "experiment_tension.py",
                    "release_map.py", "run_real_fuba.sh", "train_main.py", "main.py", "config.py")}}
    path = root/"manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("registered manifest changed; use a new --out folder, do not mix versions")
    path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    if a.stage == "plan":
        for job in jobs:
            print(job["split"], job["id"], checkpoint_state(job, checkpoints))
        print(f"{len(jobs)} independent episodes; manifest -> {path}")
        return
    for job in jobs:
        if a.stage in ("train", "all"):
            state = checkpoint_state(job, checkpoints)
            if state == "partial":
                raise RuntimeError(f"partial trajectory {job['id']}; preserve/resolve it before restarting; no automatic overwrite")
            if state == "missing":
                env = dict(os.environ, **job["env"], PYTHONIOENCODING="utf-8")
                checked_run(job["command"], root/"commands"/f"{job['id']}.train.log", env)
                if checkpoint_state(job, checkpoints) != "complete":
                    raise RuntimeError(f"incomplete trajectory after training: {job['id']}")
        if a.stage in ("collect", "all"):
            cache = Path(job["cache"])
            if cache.exists():
                validate_cache(job, a.probe, checkpoints)
                print("REUSE", cache.name, flush=True)
            else:
                checked_run([sys.executable, str(research_source("collect_tension.py")), "--name", job["id"],
                             "--checkpoints", str(checkpoints), "--data-root", str(data_root), "--out", str(cache),
                             "--rounds", str(job["rounds"]), "--k", str(job["K"]), "--probe", str(a.probe),
                             "--n-gpu", str(a.n_gpu)], root/"commands"/f"{job['id']}.collect.log")
    if a.stage in ("analyze", "all"):
        for job in jobs:
            validate_cache(job, a.probe, checkpoints)
        scored = [score_ledger(j["cache"]) for j in jobs]
        report = summarize(jobs, scored, manifest["alpha"])
        report["manifest_sha256"] = sha256(path)
        (root/"scores.json").write_text(json.dumps(scored, indent=1, allow_nan=False)+"\n", encoding="utf-8")
        (root/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(f"DONE -> {root/'summary.json'}", flush=True)
    print("No unlearning, ASR evaluation or recovery was run in this experiment.", flush=True)


if __name__ == "__main__":
    main()
