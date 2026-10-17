"""Validate saved BadFU trajectories and export replayable detection evidence.

No training or federated unlearning is invoked. Scoring is the existing frozen
HRAC inter channel; known attack roles enter reporting only, after scoring.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FUBA"))
from experiments.detection.hrac import experiment_hrac_unlabeled as probes
PHASES = ("clean5", "bd5", "cover5", "full6", "fu_remove5", "fu_remove1")
OBSERVE = ("full6", "clean5", "cover5")
CODE = ("experiments/detection/hrac/validate_detection.py", "reproduction/common/trajectory.py",
        "experiments/detection/hrac/experiment_hrac_unlabeled.py", "experiments/detection/hrac/hrac.py", "experiments/shared/trajectory_tools/tension_audit.py",
        "experiments/shared/trajectory_tools/badfu_artifacts.py", "BadFU_src/BadFU-main/badfu_common.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rank_interval(values, index):
    """Conservative ties: near-equal floats cannot create a unique first rank."""
    values = np.asarray(values, float)
    target = values[index]
    tied = np.isclose(values, target, rtol=1e-6, atol=1e-12)
    ahead = int(np.count_nonzero((values > target) & ~tied))
    return ahead + 1, ahead + int(tied.sum())


def phase_metrics(metrics, phase, rounds):
    # The author's final print loop evaluates earlier models while phase still
    # says fu_remove1. Never assign those evaluations to earlier FL rounds.
    result = {}
    for row in metrics:
        if row.get("phase") == phase and row.get("model") == phase:
            n = row["completed_rounds"]
            if not isinstance(n, int) or not 0 <= n <= rounds:
                raise ValueError(f"Invalid metric round: {phase}/{n}")
            vals = {k: float(row[k]) for k in ("asr_pct", "clean_acc_pct")}
            if not all(np.isfinite(x) and 0 <= x <= 100 for x in vals.values()):
                raise ValueError(f"Invalid metric percentage: {phase}/{n}")
            if n in result and result[n] != vals:
                raise ValueError(f"Conflicting global metrics: {phase}/{n}")
            result[n] = vals
    if rounds not in result:
        raise ValueError(f"Missing endpoint metric for {phase}, round {rounds}")
    return result


def check_run(run):
    from reproduction.common.trajectory import verify
    protocol = read(run / "protocol.json")
    opts = protocol["options"]
    if protocol.get("status") != "complete" or opts.get("attack") != "badfu":
        raise ValueError("A complete reproduction/training/run.py BadFU run is required")
    rounds = opts["rounds"]
    if not isinstance(rounds, int) or rounds < 2 or rounds % 2:
        raise ValueError("Expected a positive even FL round count")
    # Verify the recorded execution snapshot, not today's adapter against an old run.
    for name, expected in protocol["adapter_sha256"].items():
        path = run / "reproduction" / name
        if path.parent.resolve() != (run / "reproduction").resolve() or probes.sha(path) != expected:
            raise ValueError(f"Training adapter snapshot hash mismatch: {name}")
    if probes.sha(run / "runtime.patch") != protocol["runtime_patch_sha256"]:
        raise ValueError("Training runtime patch hash mismatch")
    metrics = read(run / "metrics.json")
    reports, curves, count, byte_count = {}, {}, 0, 0
    for phase in PHASES:
        print(f"[integrity] {phase}: reading and checking every checkpoint", flush=True)
        folder = run / phase
        m = read(folder / "manifest.json")
        meta = m["metadata"]
        expected = rounds // 2 if phase.startswith("fu_") else rounds
        ids = list(range(6 if phase == "full6" else 5))
        if phase == "fu_remove1":
            ids.remove(1)
        rule = "author_federaser" if phase.startswith("fu_") else "badfu_sample_weighted"
        if (meta.get("attack") != "badfu" or meta.get("phase") != phase
                or meta.get("aggregation") != rule or meta.get("client_ids") != ids
                or meta.get("upstream_commit") != protocol["upstream"]["commit"]
                or m["expected_rounds"] != expected
                or any(e["client_ids"] != ids for e in m["rounds"])):
            raise ValueError(f"Stage metadata disagrees with training protocol: {phase}")
        report = verify(folder)
        entries = [m["initial"], *m["rounds"]]
        for e in entries:
            if (folder / e["path"]).stat().st_size != e["bytes"]:
                raise ValueError(f"Checkpoint byte count mismatch: {phase}/{e['path']}")
        report.update(manifest_sha256=probes.sha(folder / "manifest.json"),
                      checkpoint_files=len(entries), bytes=sum(e["bytes"] for e in entries))
        reports[phase] = report
        count += report["checkpoint_files"]
        byte_count += report["bytes"]
        curves[phase] = phase_metrics(metrics, phase, expected)
    return dict(status="verified", phases=reports, checkpoint_files=count, bytes=byte_count,
                expected_checkpoint_files=5 * rounds + 6,
                fu_note="FU checks cover file integrity and state continuity; no FU replay or mathematical forgetting certificate."), curves


def compact_cache(folder):
    protocol = read(folder / "protocol.json")
    progress = read(folder / "progress.json")
    total = len(protocol["source_rounds"])
    if set(progress["rounds"]) != {str(i) for i in range(total)}:
        raise ValueError("Partial observation cache; resume collection first")
    rows = []
    expected = (len(protocol["probe_seeds"]), len(protocol["counts"]),
                len(protocol["views"]), protocol["probe_size"], 10)
    for t in range(total):
        path = folder / "rounds" / f"{t:05d}.npz"
        if probes.sha(path) != progress["rounds"][str(t)]["sha256"]:
            raise ValueError(f"Observation cache hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            effect = data["finite_logit"].astype(float)
        if effect.shape != expected or not np.isfinite(effect).all():
            raise ValueError(f"Unexpected response shape/values: {path}")
        rows.append(effect.mean(-2))
    return np.stack(rows), protocol


def score_rows(raw, protocol, phase, curves):
    """All classes and clients scored first; reference client 5/target 0 evaluated later."""
    expected = (len(protocol["source_rounds"]), len(protocol["probe_seeds"]),
                len(protocol["counts"]), len(protocol["views"]), 10)
    if raw.shape != expected or not np.isfinite(raw).all():
        raise ValueError("Compact response shape/values disagree with observation protocol")
    rows = []
    for n in range(1, len(raw) + 1):
        for rep, seed in enumerate(protocol["probe_seeds"]):
            # Every prefix is reported; neither ASR nor known target selects a prefix.
            configurations = (("paired_patterns", raw[:n, rep] - raw[:n, rep, :, :1],
                               protocol["families"]["all_patterns"]),
                              ("raw_clean", raw[:n, rep], [0]))
            for contrast, history, views in configurations:
                per_view = [probes.score_profiles(history[:, :, v]) for v in views]
                for method in per_view[0]:
                    scores = np.max([v[method] for v in per_view], axis=0)
                    unit_scores = scores.max(-1)
                    tops = np.flatnonzero(np.isclose(unit_scores, unit_scores.max(), rtol=1e-6, atol=1e-12))
                    row = dict(phase=phase, rounds=n, probe_seed=seed, contrast=contrast,
                               method=method, max_score=float(unit_scores.max()),
                               top_units=tops.tolist(), unit_scores=unit_scores.tolist(),
                               scores_by_class=scores.tolist(),
                               asr_pct=curves.get(n, {}).get("asr_pct"),
                               clean_acc_pct=curves.get(n, {}).get("clean_acc_pct"))
                    if phase == "full6":
                        lo, hi = rank_interval(unit_scores, 5)
                        clo, chi = rank_interval(scores[5], 0)
                        row.update(reference_client=5, reference_target=0, rank_best=lo,
                                   rank_worst=hi, target_rank_best=clo, target_rank_worst=chi,
                                   reference_score=float(unit_scores[5]),
                                   strict_client_first=bool(hi == 1 and unit_scores[5] > 1e-12),
                                   strict_joint_first=bool(hi == chi == 1 and scores[5, 0] > 1e-12))
                    rows.append(row)
    return rows


def summarize(rows, curves, protocol, integrity):
    final, first = [], []
    attack = [r for r in rows if r["phase"] == "full6"]
    end = max(r["rounds"] for r in attack)
    for contrast in ("paired_patterns", "raw_clean"):
        for method in sorted({r["method"] for r in attack}):
            for seed in sorted({r["probe_seed"] for r in attack}):
                group = [r for r in attack if (r["contrast"], r["method"], r["probe_seed"]) == (contrast, method, seed)]
                last = next(r for r in group if r["rounds"] == end)
                final.append({k: last[k] for k in ("contrast", "method", "probe_seed", "rank_best", "rank_worst",
                                                   "target_rank_worst", "reference_score", "asr_pct")})
                record = dict(contrast=contrast, method=method, probe_seed=seed)
                for field in ("strict_client_first", "strict_joint_first"):
                    hit = next((r for r in group if r[field]), None)
                    record[field] = None if hit is None else dict(round=hit["rounds"], asr_pct=hit["asr_pct"])
                first.append(record)
    transitions = []
    for before, after in (("full6", "fu_remove5"), ("cover5", "fu_remove1")):
        a, b = curves[before][max(curves[before])], curves[after][max(curves[after])]
        transitions.append(dict(source_phase=before, fu_phase=after, before=a, after=b,
                                asr_change_pp=b["asr_pct"]-a["asr_pct"]))
    return dict(schema=1, status="analysis_complete", smoke_only=bool(protocol["options"]["smoke"]),
                integrity=integrity, final_rankings=final, first_unique_rank_one=first,
                recorded_fu_transitions=transitions, calibrated_alarm=None, fpr=None, tpr=None,
                independent_training_runs=1, training_calls=0, model_unlearning_calls=0,
                primary="paired_patterns / hrac_windows; max over all output classes",
                interpretation=[
                    "Rank one is localization, not a calibrated malicious/benign decision.",
                    "First rank one scans all prefixes retrospectively; it is not a sequentially calibrated alarm.",
                    "Probe seeds are disjoint image repetitions of one training run, not independent attack runs.",
                    "clean5 has 5 clients versus full6's 6; it is an unmatched diagnostic, not an FPR test.",
                    "cover5 contains poison and cover data and is not a benign control.",
                    "fu_remove1 comes from cover5, not full6: the two FU endpoints are not matched controls.",
                    "Stored ASR is joined for interpretation only; validation does not re-evaluate attack images.",
                    "Only HRAC inter is measured. No independently observed intra-request channel is available.",
                    "Compact means reproduce rankings but cannot reproduce per-image bootstrap uncertainty.",
                    "These results do not establish performance under stronger Non-IID conditions or other attacks."])


def analyze_bundle(bundle):
    inventory = read(bundle / "inventory.json")
    for rel, expected in inventory.items():
        path = (bundle / rel).resolve()
        if not path.is_relative_to(bundle.resolve()) or probes.sha(path) != expected:
            raise ValueError(f"Bundle hash mismatch: {rel}")
    plan = read(bundle / "validation_protocol.json")
    for rel, expected in plan["code_sha256"].items():
        if probes.sha(ROOT / rel) != expected:
            raise ValueError(f"Replay requires the recorded code version: {rel}")
    training = read(bundle / "training_protocol.json")
    metrics = read(bundle / "metrics.json")
    integrity = read(bundle / "integrity.json")
    curves = {p: phase_metrics(metrics, p, integrity["phases"][p]["rounds"]) for p in PHASES}
    rows = []
    for phase in OBSERVE:
        pp = read(bundle / phase / "protocol.json")
        if pp["manifest_sha256"] != probes.sha(bundle / phase / "manifest.json"):
            raise ValueError(f"Observation/trajectory mismatch: {phase}")
        with np.load(bundle / phase / "response_means.npz", allow_pickle=False) as data:
            raw = data["responses"]
        rows.extend(score_rows(raw, pp, phase, curves[phase]))
    summary = summarize(rows, curves, training, integrity)
    probes.save_json(bundle / "summary.json", summary)
    probes.save_json(bundle / "rankings.json", rows)
    fields = ["phase", "rounds", "probe_seed", "contrast", "method", "max_score", "top_units",
              "rank_best", "rank_worst", "target_rank_worst", "reference_score", "strict_client_first",
              "strict_joint_first", "asr_pct", "clean_acc_pct"]
    with (bundle / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["BadFU saved-trajectory validation", "SMOKE ONLY" if summary["smoke_only"] else "ONE TRAINING RUN",
             f"Verified checkpoint files: {integrity['checkpoint_files']}",
             "Final paired-input rankings (ties shown as a range):"]
    for r in summary["final_rankings"]:
        if r["contrast"] == "paired_patterns":
            lines.append(f"  {r['method']} / probe {r['probe_seed']}: client 5 rank "
                         f"{r['rank_best']}..{r['rank_worst']}; target 0 rank <= {r['target_rank_worst']}")
    lines.extend(["", *summary["interpretation"]])
    (bundle / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def validate(args):
    import torch
    torch.set_num_threads(args.threads)
    run = args.run_dir.resolve()
    training = read(run / "protocol.json")
    out = (args.out or run / "detection_validation_v1").resolve()
    if out == run or run.is_relative_to(out):
        raise ValueError("Output must not be the training directory or its ancestor")
    out.mkdir(parents=True, exist_ok=True)
    pool = args.pool.resolve() if args.pool else out / "probe_pool.npz"
    if args.pool is None and not pool.exists():
        data_root = args.cifar_root or Path(training["options"]["data_root"])
        exclude = ROOT / "reproduction/common/resources/unlabeled_excluded_ids_v1.json"
        probes.prepare(argparse.Namespace(cifar_root=data_root, image_dir=None,
                                         exclude_ids=exclude, out=pool))
    probes.load_pool(pool)
    plan = dict(schema=1, run_dir=str(run), pool_sha256=probes.sha(pool),
                training_protocol_sha256=probes.sha(run / "protocol.json"),
                metrics_sha256=probes.sha(run / "metrics.json"),
                manifests={p: probes.sha(run / p / "manifest.json") for p in PHASES},
                code_sha256={rel: probes.sha(ROOT / rel) for rel in CODE},
                observe=list(OBSERVE), probe_size=80, probe_seeds=[20261019, 20261020, 20261021],
                pattern_seeds=[20261019, 20261020, 20261021], windows=[0, 10, 20],
                patch_size=4, batch_size=args.batch_size, threads=args.threads, device=args.device,
                all_prefixes=True, ground_truth_use="report only: full6 client 5, target output 0",
                training_calls=0, model_unlearning_calls=0)
    plan_path = out / "validation_protocol.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("Output belongs to a different protocol. Choose a new --out; existing data is preserved.")
    probes.save_json(plan_path, plan)
    started = time.perf_counter()
    # Always recheck real model files, including on resume. Cached observations
    # cannot certify that the original weights still exist or are unmodified.
    integrity, _ = check_run(run)
    probes.save_json(out / "integrity.json", integrity)
    for phase in OBSERVE:
        folder = out / "observations" / phase
        pa = probes.parser().parse_args(["run", "--manifest", str(run / phase / "manifest.json"),
            "--pool", str(pool), "--out", str(folder), "--device", args.device,
            "--threads", str(args.threads), "--batch-size", str(args.batch_size), "--resume"])
        pa.prefixes = list(range(1, integrity["phases"][phase]["rounds"] + 1))
        print(f"[observe] {phase}: fixed HRAC and baseline observations", flush=True)
        if not probes.collect(pa):
            raise ValueError("Observation collection incomplete; repeat the same command")
    bundle = out / "evidence"
    bundle.mkdir(exist_ok=True)
    for src, dest in ((plan_path, "validation_protocol.json"), (out / "integrity.json", "integrity.json"),
                      (run / "protocol.json", "training_protocol.json"), (run / "metrics.json", "metrics.json")):
        shutil.copyfile(src, bundle / dest)
    for phase in PHASES:
        target = bundle / phase
        target.mkdir(exist_ok=True)
        shutil.copyfile(run / phase / "manifest.json", target / "manifest.json")
        if phase in OBSERVE:
            cache = out / "observations" / phase
            raw, _ = compact_cache(cache)
            np.savez_compressed(target / "response_means.npz", responses=raw)
            for name in ("protocol.json", "progress.json"):
                shutil.copyfile(cache / name, target / name)
    # Input-only inventory permits replay without weights, images or labels.
    generated = {"inventory.json", "summary.json", "rankings.json", "comparison.csv", "report.txt"}
    inventory = {p.relative_to(bundle).as_posix(): probes.sha(p) for p in sorted(bundle.rglob("*"))
                 if p.is_file() and p.name not in generated}
    probes.save_json(bundle / "inventory.json", inventory)
    summary = analyze_bundle(bundle)
    archive = out / "badfu_detection_evidence.zip"
    with zipfile.ZipFile(archive.with_suffix(".pending"), "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                z.write(path, "evidence/" + path.relative_to(bundle).as_posix())
    archive.with_suffix(".pending").replace(archive)
    print((bundle / "report.txt").read_text(encoding="utf-8"), flush=True)
    print(f"Completed in {(time.perf_counter()-started)/60:.1f} min. Share this file: {archive}\n"
          f"Archive SHA256: {probes.sha(archive)}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path)
    mode.add_argument("--replay", type=Path, help="extracted evidence directory; no model/image files needed")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--cifar-root", type=Path, help="defaults to training protocol's data_root")
    source.add_argument("--pool", type=Path, help="existing unlabeled images/ids NPZ")
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()
    if a.threads < 1 or a.batch_size < 1:
        p.error("threads and batch-size must be positive")
    if a.replay:
        analyze_bundle(a.replay.resolve())
        print(f"Replayed: {a.replay / 'summary.json'}")
    else:
        validate(a)


if __name__ == "__main__":
    main()
