"""Replay exported histories with a fixed temporal detector; no training or FU.

Default: existing CIFAR-10 seed-42 attack/normal evidence, exploratory only.
A JSON plan separates fit/calibration/test whole runs. Outputs are uploadable.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction import temporal_history as th


def write_json(path, value):
    # Git uses LF on every host. Hash the same bytes on Windows and Linux.
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8"))


def exported(run_name):
    paths = sorted((ROOT / "reproduction/validation/exports" / run_name).glob("*/*.zip"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one evidence ZIP for {run_name}; use an explicit --plan")
    return paths[0].relative_to(ROOT).as_posix()


def default_plan():
    return dict(schema=1, alpha=.05, fit=[exported("fuba_cifar10_benign_s42")], calibration=[],
                test=[dict(name="fuba_attack_s42", evidence=exported("fuba_cifar10_s42"), report_client=4)],
                note="Previously inspected paired seed 42; exploratory, not independent evaluation.")


def rank_summary(values, client):
    if client is None:
        return None
    if not isinstance(client, int) or not 0 <= client < len(values):
        raise ValueError("report_client outside the actual population")
    value = values[client]
    equal = np.isclose(values, value, rtol=1e-9, atol=1e-12)
    better = int(np.count_nonzero((values > value) & ~equal))
    return dict(score=float(value), rank_best=better+1, rank_worst=better+int(equal.sum()))


def execute(plan, out=None, make_plot=True):
    if plan.get("schema") != 1 or not plan.get("fit") or not plan.get("test"):
        raise ValueError("Plan needs schema=1, fit evidence and test entries")
    started = time.perf_counter()

    def load(spec):
        path, sep, phase = str(spec).partition("::")
        source = Path(path) if Path(path).is_absolute() else ROOT / path
        return th.load_evidence(str(source) + (sep + phase if sep else ""))

    fitting = [load(p) for p in plan["fit"]]
    calibration = [load(p) for p in plan.get("calibration", [])]
    tests = [load(p["evidence"]) for p in plan["test"]]
    th.validate_split(fitting, calibration, tests)
    alpha = plan.get("alpha", .05)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    sources = fitting + calibration + tests
    code = {p.name: th.digest(p.read_bytes()) for p in (Path(th.__file__), Path(__file__))}
    signature = th.digest(json.dumps(dict(plan=plan, code=code,
        inventories=[e["inventory_sha256"] for e in sources]), sort_keys=True).encode())
    out = Path(out) if out is not None else ROOT / "reproduction/validation/exports/temporal_history" / signature[:16]
    if out.exists():
        receipt = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if receipt.get("signature") != signature or receipt.get("status") != "complete":
            raise ValueError("Output belongs to another/incomplete analysis; use a new --out")
        for name, expected in json.loads((out / "inventory.json").read_text()).items():
            if th.digest((out / name).read_bytes()) != expected:
                raise ValueError("Existing output changed; preserve it and use a new --out")
        print(f"Reusing verified analysis: {out}", flush=True)
        return out
    model = th.fit_model(fitting)
    fit_scores = [th.score(e, model) for e in fitting]
    cal_scores = [th.score(e, model) for e in calibration]
    test_scores = [th.score(e, model) for e in tests]
    out.mkdir(parents=True, exist_ok=False)
    arrays, rows = {}, []
    groups = [("fit", fitting, fit_scores), ("calibration", calibration, cal_scores), ("test", tests, test_scores)]
    for split, episodes, scored in groups:
        for i, (episode, result) in enumerate(zip(episodes, scored)):
            key = f"{split}_{i}"
            client = plan["test"][i].get("report_client") if split == "test" else None
            row = dict(key=key, split=split, group=episode["group"], phase=episode["phase"],
                       benign=episode["benign"], independent_of_fit=episode["group"] not in model["fitting_groups"],
                       input_inventory_sha256=episode["inventory_sha256"], stratum=episode["stratum"], methods={})
            for method, path in result["paths"].items():
                arrays[key+"_"+method] = path
                t, j = np.unravel_index(path.argmax(), path.shape)
                checkpoints = {}
                for stop in sorted(set([8, 16, 26, len(path)])):
                    if stop <= len(path):
                        checkpoints[str(stop)] = dict(max_score=float(path[stop-1].max()),
                            top_clients=np.flatnonzero(np.isclose(path[stop-1], path[stop-1].max())).tolist(),
                            report_client=rank_summary(path[stop-1], client))
                item = dict(episode_max=float(path.max()), peak_round=int(t+1), peak_client=int(j),
                            final_max=float(path[-1].max()), checkpoints=checkpoints)
                if method == "temporal_glr":
                    item.update(peak_interval_start=int(result["glr_interval_start"][t, j]),
                                peak_coordinate=int(result["glr_coordinate"][t, j]))
                if split == "test":
                    item["calibration"] = th.calibrated_path(path,
                        [s["paths"][method].max() for s in cal_scores], alpha)
                row["methods"][method] = item
            # Full coordinate histories remain reproducible from the source ZIP
            # and saved model; do not duplicate tens of MB for each replay.
            for name in ("glr_coordinate", "glr_interval_start"):
                arrays[key+"_"+name] = result[name]
            if split == "test":
                reverse = dict(episode, history=episode["history"][::-1].copy(), mask=episode["mask"][::-1].copy())
                reversed_score = th.score(reverse, model)
                row["order_ablation"] = dict(kind="reverse rounds and participation together; retrospective only",
                    final_level_same=bool(np.allclose(result["level"][-1], reversed_score["level"][-1], rtol=1e-10, atol=1e-12)),
                    temporal_episode_max_reversed=float(reversed_score["paths"]["temporal_glr"].max()),
                    interpretation="Sensitivity to order, not evidence that chronological order improves detection; no null-test p-value.")
            rows.append(row)
    write_json(out / "plan.json", plan)
    write_json(out / "model.json", {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()})
    np.savez_compressed(out / "series.npz", **arrays)
    summary = dict(status="complete", signature=signature, version=th.VERSION, primary="temporal_glr",
        code_sha256=code, alpha=alpha, fit_runs=len(fitting), calibration_runs=len(calibration),
        evaluation_runs=len(tests), training_calls=0, model_unlearning_calls=0,
        labels_used_for_scoring=False, target_or_client_truth_used_for_scoring=False,
        intra_status="unobserved_independent_request_effect", evaluation=rows,
        limitations=["Client-level inter proxy, not exact FU effects or sample-level intra attribution.",
            "No Gaussian independence assumption is asserted for measured innovations.",
            "One normal fitting run cannot establish calibrated FPR or superiority.",
            "Only independent matched normal run maxima calibrate finite-horizon alarms.",
            "Each baseline has a separate comparator threshold; do not OR their alarms at alpha.",
            "Normal calibration requires a trusted attack-free reference population.",
            "Attack-correlated steady effects predictable from their own past can evade innovations.",
            "No actual FU, purification, forgetting or ASR recovery is verified here."])
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        for ax, method in zip(axes, ("level", "positive_change", "temporal_glr")):
            for i, result in enumerate(fit_scores):
                values = result["paths"][method]
                ax.plot(np.arange(1, len(values)+1), values.max(1), linestyle="--", color="gray",
                        label=f"normal fit {i}: all-client max (in-sample)")
            for i, result in enumerate(test_scores):
                values = result["paths"][method]
                ax.plot(np.arange(1, len(values)+1), values.max(1), label=plan["test"][i]["name"]+": all-client max")
            ax.set_ylabel(method)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("FL round")
        fig.suptitle("History-only temporal analysis | not calibrated detection efficacy")
        fig.savefig(out / "temporal_history.png", dpi=150)
        plt.close(fig)
    summary["elapsed_seconds"] = time.perf_counter()-started
    write_json(out / "summary.json", summary)
    write_json(out / "inventory.json", {p.name: th.digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})
    print(f"Analysis complete: {out}\nCalibration runs: {len(calibration)}; FU calls: 0", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--badfu-evidence", type=Path, help="Diagnostic clean5 fit/full6 test; not matched calibration")
    a = p.parse_args()
    if a.plan and a.badfu_evidence:
        p.error("Choose --plan or --badfu-evidence")
    if a.badfu_evidence:
        spec = str(a.badfu_evidence.resolve())
        plan = dict(schema=1, alpha=.05, fit=[spec+"::clean5"], calibration=[],
                    test=[dict(name="badfu_full6", evidence=spec+"::full6", report_client=5)],
                    note="clean5/full6 are unmatched populations from the same original run; diagnostic only.")
    else:
        plan = json.loads(a.plan.read_text(encoding="utf-8")) if a.plan else default_plan()
    execute(plan, a.out, not a.no_plot)


if __name__ == "__main__":
    main()
