"""FUBA history/request validation for saved MNIST/CIFAR10 full trajectories.

Uses the same fixed HRAC scores and two-channel request gate as the framework.
No training, FU, attack generator, trigger recovery, or probe labels are used.
"""
import argparse
import csv
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.detection import validate_detection as common
from reproduction.detection.hrac_request_gate import evaluate, fingerprint
from reproduction.common.trajectory import verify

probes = common.probes
read = common.read
SEEDS = [20261019, 20261020, 20261021]
CODE = ["reproduction/detection/validate_fuba_detection.py", "reproduction/detection/hrac_request_gate.py",
        "reproduction/detection/validate_detection.py", "reproduction/common/trajectory.py",
        "FUBA/research/experiment_hrac_unlabeled.py", "FUBA/research/hrac.py", "FUBA/research/tension_audit.py",
        "FUBA/model/__init__.py", "FUBA/model/model.py", "FUBA/model/resnet.py"]


def load_pool(path, dataset):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"images", "ids"}:
            raise ValueError("Only images/ids allowed in probe pool; labels must not be exported")
        pixels, ids = data["images"], data["ids"]
    shape = (28, 28, 1) if dataset == "mnist" else (32, 32, 3)
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != shape:
        raise ValueError(f"Expected uint8 pixels [N,{shape}] for {dataset}")
    if ids.dtype.kind != "U" or ids.shape != (len(pixels),) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Unique image IDs required")
    return pixels, ids


def prepare_pool(path, data_root, dataset):
    from torchvision.datasets import MNIST, CIFAR10
    ds = (MNIST if dataset == "mnist" else CIFAR10)(str(data_root), train=False, download=False)
    # .targets and __getitem__ are deliberately never used.
    pixels = ds.data.numpy()[..., None] if dataset == "mnist" else ds.data
    ids = np.array([f"{dataset}:test:{i}" for i in range(len(pixels))])
    if dataset == "cifar10":
        excluded = set(read(ROOT / "FUBA/logs/hrac/unlabeled_excluded_ids_v1.json"))
        keep = np.array([i not in excluded for i in ids])
        pixels, ids = pixels[keep], ids[keep]
    with path.open("xb") as f:
        np.savez_compressed(f, images=pixels, ids=ids)


def make_views(pixels):
    import torch
    x = torch.from_numpy(pixels.copy()).permute(0, 3, 1, 2).float() / 255
    h, w = x.shape[-2:]
    views, names = [x], ["clean"]
    for seed in SEEDS:
        patch = torch.randint(0, 2, (x.shape[1], 4, 4), generator=torch.Generator().manual_seed(seed)).float()
        for name, r, c in (("tl", 0, 0), ("tr", 0, w-4), ("bl", h-4, 0), ("br", h-4, w-4)):
            v = x.clone()
            v[:, :, r:r+4, c:c+4] = patch
            views.append(v)
            names.append(f"pattern_{seed}_{name}")
    # The original FUBA loader uses ToTensor ONLY, unlike BadFU normalization.
    return torch.stack(views), names


def unpack_effect(effect, ids, k, reps=3, views=13, count=80):
    """Unselected clients have zero aggregate contribution, not measured zero gradients."""
    ids = np.asarray(ids)
    if ids.dtype.kind not in "iu" or len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= k):
        raise ValueError("Invalid participant IDs")
    expected = (len(ids), views*reps*count, 10)
    if effect.shape != expected or not np.isfinite(effect).all():
        raise ValueError("Unexpected measured response shape/values")
    selected = effect.reshape(len(ids), views, reps, count, 10).transpose(2, 0, 1, 3, 4)
    full = np.zeros((reps, k, views, count, 10), dtype=effect.dtype)
    full[:, ids] = selected
    present = np.zeros(k, dtype=bool)
    present[ids] = True
    return full, present


def check_run(run):
    from reproduction.training.run import check_sources
    check_sources("fuba")
    training, manifest = read(run / "protocol.json"), read(run / "fl/manifest.json")
    options, meta = training["options"], manifest["metadata"]
    if (training.get("status") != "complete" or options.get("attack") != "fuba"
            or options.get("dataset") not in ("mnist", "cifar10")
            or meta.get("attack") != "fuba" or meta.get("aggregation") != "fuba_uniform"
            or meta.get("dataset") != options["dataset"] or meta.get("num_clients") != options["clients"]
            or manifest["expected_rounds"] != options["rounds"]
            or meta.get("upstream_commit") != training["upstream"]["commit"]):
        raise ValueError("Complete reproduction/training/run.py FUBA MNIST/CIFAR10 trajectory required")
    for name, expected in training["adapter_sha256"].items():
        path = run / "reproduction" / name
        if path.parent.resolve() != (run / "reproduction").resolve() or probes.sha(path) != expected:
            raise ValueError(f"Training snapshot hash mismatch: {name}")
    if probes.sha(run / "runtime.patch") != training["runtime_patch_sha256"]:
        raise ValueError("Training patch hash mismatch")
    size = int(options["clients"] * options["participant_rate"])
    for r in manifest["rounds"]:
        if len(r["client_ids"]) != size or any(not 0 <= i < options["clients"] for i in r["client_ids"]):
            raise ValueError("Participants disagree with the training protocol")
    print("[integrity] reading all saved FUBA model states and recomputing FL aggregates", flush=True)
    integrity = verify(run / "fl")
    integrity.update(checkpoint_files=1+len(manifest["rounds"]), manifest_sha256=probes.sha(run / "fl/manifest.json"))
    metrics = []
    for n in range(1, options["rounds"]+1):
        row = read(run / f"metrics_round_{n:04d}.json")
        asr = row["asr_pct"]
        missing_warmup_asr = asr is None and row["generator_source"] == "warmup_no_asr"
        if (row["round"] != n or not np.isfinite(row["clean_acc_pct"]) or not 0 <= row["clean_acc_pct"] <= 100
                or (not missing_warmup_asr and (asr is None or not np.isfinite(asr) or not 0 <= asr <= 100))):
            raise ValueError("Missing/invalid round metrics")
        metrics.append(row)
    return training, manifest, integrity, metrics


def collect(run, out, pool, training, manifest, args):
    import torch
    from FUBA.model import Net, resnet18
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset, k = training["options"]["dataset"], training["options"]["clients"]
    pixels, ids = load_pool(pool, dataset)
    selected = probes.select_indices(len(pixels), 80, SEEDS)
    views, names = make_views(pixels[selected.flatten()])
    inputs = views.flatten(0, 1)
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    protocol = dict(schema=1, attack="fuba", dataset=dataset, clients=k, views=names,
        source_manifest_sha256=probes.sha(run / "fl/manifest.json"),
        training_protocol_sha256=probes.sha(run / "protocol.json"),
        code_sha256={p: probes.sha(ROOT/p) for p in CODE}, pool_sha256=probes.sha(pool),
        probe_ids=ids[selected].tolist(), probe_size=80, probe_seeds=SEEDS, pattern_seeds=SEEDS,
        preprocessing="uint8 / 255; no centering or scaling; original FUBA ToTensor",
        coordinate_system="paired centered-logit positive weighted-parameter response; frozen pre-round BN buffers",
        intra_flatten_order=["probe repetition", "perturbed view (exclude clean)", "output"],
        windows=[0, 10, 20], source_rounds=manifest["rounds"],
        nonparticipants="zero contribution to this aggregate; participation mask retained",
        device=device, batch_size=args.batch_size, threads=args.threads, torch=str(torch.__version__),
        hardware=torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU",
        labels_used=False, attack_generator_used=False, oracle_trigger_used=False,
        target_used_for_scoring=False, independent_training_runs=1,
        training_calls=0, model_unlearning_calls=0)
    path = out / "observation_protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("Observation protocol changed; preserve previous output and choose a new --out")
    probes.save_json(path, protocol)
    observation_id = fingerprint(protocol)
    probes.save_json(out / "request_coordinates.json", dict(observation_id=observation_id,
        source_manifest_sha256=protocol["source_manifest_sha256"],
        coordinate_system=protocol["coordinate_system"], flatten_order=protocol["intra_flatten_order"],
        dimensions=3*12*10, note="Independent request observations/atoms must use this exact protocol; client sums are not within-client atoms."))
    (out / "rounds").mkdir(exist_ok=True)
    progress_path = out / "progress.json"
    progress = read(progress_path) if progress_path.exists() else {"rounds": {}}
    if not set(progress["rounds"]).issubset({str(i) for i in range(len(manifest["rounds"]))}):
        raise ValueError("Unexpected cached rounds")
    model = (Net() if dataset == "mnist" else resnet18()).to(device).eval()
    for t, entry in enumerate(manifest["rounds"]):
        dest = out / "rounds" / f"{t:05d}.npz"
        if str(t) in progress["rounds"]:
            if probes.sha(dest) != progress["rounds"][str(t)]["sha256"]:
                raise ValueError("Observation cache hash mismatch")
            continue
        start = time.perf_counter()
        path = run / "fl" / entry["path"]
        if path.parent.resolve() != (run / "fl").resolve() or probes.sha(path) != entry["sha256"]:
            raise ValueError("Checkpoint changed since verification")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["client_ids"] != entry["client_ids"] or not np.allclose(state["weights"], 1/len(state["cms"]), rtol=0, atol=1e-12):
            raise ValueError("FUBA must use uniform weights over actual participants")
        effect = probes.measure(model, state["gm"], state["cms"], state["weights"], inputs, args.batch_size, device)
        full, present = unpack_effect(effect, state["client_ids"], k)
        temp = dest.with_suffix(".pending")
        with temp.open("wb") as f:
            np.savez_compressed(f, finite_logit=full, present=present)
        temp.replace(dest)
        seconds = time.perf_counter()-start
        progress["rounds"][str(t)] = dict(sha256=probes.sha(dest), seconds=seconds)
        probes.save_json(progress_path, progress)
        avg = np.mean([r["seconds"] for r in progress["rounds"].values()])
        print(f"[observe] {t+1}/{len(manifest['rounds'])}: {seconds:.1f}s; estimated remaining "
              f"{(len(manifest['rounds'])-len(progress['rounds']))*avg/60:.1f} min", flush=True)
        del state, effect, full
    history, participation = [], []
    for t, entry in enumerate(manifest["rounds"]):
        path = out / "rounds" / f"{t:05d}.npz"
        if probes.sha(path) != progress["rounds"][str(t)]["sha256"]:
            raise ValueError("Observation cache hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            effect, present = data["finite_logit"], data["present"]
        if effect.shape != (3, k, 13, 80, 10) or present.shape != (k,) or not np.isfinite(effect).all():
            raise ValueError("Malformed observation cache")
        if np.flatnonzero(present).tolist() != sorted(entry["client_ids"]) or np.any(effect[:, ~present] != 0):
            raise ValueError("Participation mask/effects do not match actual participants")
        history.append(effect.astype(float).mean(-2))
        participation.append(present)
    return np.stack(history), np.stack(participation), protocol


def ranking_rows(raw, mask, metrics, reference_client, reference_target, warm_up):
    rows = []
    for n in range(1, len(raw)+1):
        participation = mask[:n].sum(0)
        eligible = np.flatnonzero(participation)
        for rep, seed in enumerate(SEEDS):
            for contrast, h, views in (("paired_patterns", raw[:n, rep]-raw[:n, rep, :, :1], range(1, 13)),
                                       ("raw_clean", raw[:n, rep], [0])):
                profiles = [probes.score_profiles(h[:, :, v]) for v in views]
                for method in profiles[0]:
                    scores = np.max([v[method] for v in profiles], axis=0)
                    unit = scores.max(-1)
                    ref_seen = reference_client in eligible
                    lo, hi = common.rank_interval(unit[eligible], list(eligible).index(reference_client)) if ref_seen else (None, None)
                    target_lo, target_hi = common.rank_interval(scores[reference_client], reference_target) if ref_seen else (None, None)
                    metric = metrics[n-1]
                    valid_asr = metric["generator_source"] == "last_individual"
                    rows.append(dict(round=n, probe_seed=seed, contrast=contrast, method=method,
                        eligible_clients=eligible.tolist(), participation_counts=participation.tolist(),
                        scores_by_class=scores.tolist(), max_score=float(unit[eligible].max()),
                        reference_client=reference_client, reference_target=reference_target,
                        rank_best=lo, rank_worst=hi, target_rank_best=target_lo, target_rank_worst=target_hi,
                        strict_client_first=bool(ref_seen and hi == 1 and unit[reference_client] > 1e-12),
                        strict_joint_first=bool(ref_seen and hi == target_hi == 1 and scores[reference_client, reference_target] > 1e-12),
                        stage="warmup" if n <= warm_up else "attack_enabled",
                        asr_pct=float(metric["asr_pct"]) if valid_asr else None,
                        stored_asr_pct=None if metric["asr_pct"] is None else float(metric["asr_pct"]), asr_source=metric["generator_source"],
                        clean_acc_pct=float(metric["clean_acc_pct"])))
    return rows


def analyze(bundle):
    for name, expected in read(bundle / "inventory.json").items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or probes.sha(path) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    p = read(bundle / "observation_protocol.json")
    for name, expected in p["code_sha256"].items():
        if probes.sha(ROOT / name) != expected:
            raise ValueError(f"Replay requires the recorded code: {name}")
    manifest = read(bundle / "manifest.json")
    if probes.sha(bundle / "manifest.json") != p["source_manifest_sha256"]:
        raise ValueError("Manifest and observations do not match")
    training, metrics = read(bundle / "training_protocol.json"), read(bundle / "metrics.json")
    if probes.sha(bundle / "training_protocol.json") != p["training_protocol_sha256"]:
        raise ValueError("Training protocol and observations do not match")
    options = training["options"]
    with np.load(bundle / "response_means.npz", allow_pickle=False) as data:
        raw, mask = data["responses"], data["participation"]
    shape = (options["rounds"], 3, options["clients"], 13, 10)
    if raw.shape != shape or mask.shape != (shape[0], shape[2]) or mask.dtype != bool or not np.isfinite(raw).all():
        raise ValueError("Compact observation shape/values invalid")
    for n, entry in enumerate(manifest["rounds"]):
        if np.flatnonzero(mask[n]).tolist() != sorted(entry["client_ids"]) or np.any(raw[n, :, ~mask[n]] != 0):
            raise ValueError("Compact observations have invalid participation mask")
    # These roles are upstream defaults, only used AFTER response measurement.
    # The supported run.py launcher does not override either upstream option.
    rows = ranking_rows(raw, mask, metrics, 4, 8, options["warm_up"])
    request_data = read(bundle / "requests.json")
    observation_id = fingerprint(p)
    if request_data["observation_id"] != observation_id:
        raise ValueError("Request bundle refers to a different observation protocol")
    paired = raw[:, :, :, 1:] - raw[:, :, :, :1]
    gate = evaluate(paired, request_data["requests"], observation_id)
    final = [r for r in rows if r["round"] == len(raw) and r["contrast"] == "paired_patterns"]
    first = []
    for method in sorted({r["method"] for r in rows}):
        for seed in SEEDS:
            group = [r for r in rows if r["method"] == method and r["probe_seed"] == seed and r["contrast"] == "paired_patterns"]
            hit = next((r for r in group if r["strict_joint_first"] and r["stage"] == "attack_enabled"), None)
            first.append(dict(method=method, probe_seed=seed,
                              first_joint_rank_one=None if hit is None else dict(round=hit["round"], asr_pct=hit["asr_pct"])))
    summary = dict(schema=1, status="analysis_complete", attack="fuba", dataset=options["dataset"],
        smoke_only=options["smoke"], independent_training_runs=1, final_rankings=final,
        first_joint_rank_one=first, hrac=gate, request_origin=request_data["origin"],
        fpr=None, tpr=None, training_calls=0, model_unlearning_calls=0,
        limitations=["Neither rank one nor a residual alone is a calibrated malicious/benign verdict.",
            "Intra requires independent request observations and request-constrained within-client atoms; missing is not zero.",
            "Normal Non-IID partial deletion need not be proportional to total client history.",
            "A malicious but attribution-feasible request can pass intra; inter must also be assessed.",
            "FUBA and FedUBA are distinct works. The pinned FUBA uses cooperating attacker/defender clients.",
            "Fixed patches may not reveal the input-dependent FUBA generator; the generator is not used by the detector.",
            "Participation counts are reported; scores are not post-hoc rescaled for participation frequency.",
            "Warmup and random-generator ASR are marked unavailable for attack interpretation.",
            "No FU or ASR reevaluation is performed; ASR is the stored author-evaluator metric.",
            "Warmup is not an independent benign calibration run; no FPR/TPR is estimated.",
            "Current effects use a logit response coordinate system, not a claimed exact Fisher/Jacobian spectrum."])
    probes.save_json(bundle / "summary.json", summary)
    probes.save_json(bundle / "rankings.json", rows)
    fields = ["round", "probe_seed", "contrast", "method", "rank_best", "rank_worst", "target_rank_worst",
              "strict_client_first", "strict_joint_first", "stage", "asr_pct", "stored_asr_pct", "asr_source", "clean_acc_pct"]
    with (bundle / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"FUBA {options['dataset']}: {'SMOKE ONLY' if options['smoke'] else 'ONE TRAINING RUN'}",
             f"HRAC channels observed: inter=yes, intra={gate['both_channels_observed']}",
             "Joint decision: incomplete (no independent benign calibration)"]
    lines += [f"{r['method']} / probe {r['probe_seed']}: client 4 rank {r['rank_best']}..{r['rank_worst']}; target 8 rank <= {r['target_rank_worst']}" for r in final]
    lines += ["", *summary["limitations"]]
    (bundle / "report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--requests", type=Path, help="Independent request/atom bundle. Without it, intra stays unobserved.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("positive threads and batch-size required")
    if args.replay:
        analyze(args.replay.resolve())
        print(f"Replayed: {args.replay / 'summary.json'}")
        return
    import torch
    torch.set_num_threads(args.threads)
    run = args.run_dir.resolve()
    training, manifest, integrity, metrics = check_run(run)
    out = (args.out or run / "hrac_validation_v1").resolve()
    if out == run or run.is_relative_to(out):
        raise ValueError("Output cannot replace the training folder or its ancestor")
    out.mkdir(parents=True, exist_ok=True)
    pool = args.pool.resolve() if args.pool else out / "probe_pool.npz"
    if args.pool is None and not pool.exists():
        prepare_pool(pool, args.data_root or Path(training["options"]["data_root"]), training["options"]["dataset"])
    raw, mask, protocol = collect(run, out, pool, training, manifest, args)
    observation_id = fingerprint(protocol)
    requests = read(args.requests) if args.requests else dict(schema=1, origin="prospective_full_client_requests",
        observation_id=observation_id, requests=[dict(id=f"client:{j}", round=len(raw), removal=np.eye(raw.shape[2])[j].tolist())
                                               for j in range(raw.shape[2]) if mask[:, j].any()])
    if requests.get("schema") != 1 or requests.get("observation_id") != observation_id:
        raise ValueError("Use request_coordinates.json to bind independent requests to this observation protocol")
    # Distinct request inputs get separate evidence directories; observation cache is reused.
    tag = fingerprint(requests)[:12]
    bundle = out / f"evidence_{tag}"
    bundle.mkdir(exist_ok=True)
    for src, dest in ((run / "protocol.json", "training_protocol.json"), (run / "fl/manifest.json", "manifest.json"),
                      (out / "observation_protocol.json", "observation_protocol.json"), (out / "progress.json", "progress.json"),
                      (out / "request_coordinates.json", "request_coordinates.json")):
        shutil.copyfile(src, bundle / dest)
    for name, data in (("integrity.json", integrity), ("metrics.json", metrics), ("requests.json", requests)):
        probes.save_json(bundle / name, data)
    np.savez_compressed(bundle / "response_means.npz", responses=raw, participation=mask)
    generated = {"inventory.json", "summary.json", "rankings.json", "comparison.csv", "report.txt"}
    probes.save_json(bundle / "inventory.json", {p.name: probes.sha(p) for p in sorted(bundle.iterdir()) if p.name not in generated})
    analyze(bundle)
    archive = out / f"fuba_hrac_evidence_{tag}.zip"
    with zipfile.ZipFile(archive.with_suffix(".pending"), "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.iterdir()):
            z.write(p, "evidence/" + p.name)
    archive.with_suffix(".pending").replace(archive)
    print((bundle / "report.txt").read_text(encoding="utf-8"))
    print(f"Share: {archive}\nSHA256: {probes.sha(archive)}")


if __name__ == "__main__":
    main()
