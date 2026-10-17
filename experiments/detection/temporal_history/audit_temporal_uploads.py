"""Verify temporal exports and replay available source histories without training.

Missing remote BadFU inputs are explicitly distinguished from verified derived
arrays. Post-hoc normal envelopes are descriptive, never calibrated thresholds.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.detection.temporal_history import temporal_history as th
from experiments.detection.temporal_history.validate_temporal_history import write_json


def source_provenance(name, expected):
    """Verify recorded bytes separately from numerical replay with current code."""
    if name not in ('temporal_history.py', 'validate_temporal_history.py'):
        raise ValueError(f'Unexpected temporal source: {name}')
    current = Path(__file__).parent / name
    current_hash = th.digest(current.read_bytes())
    record = current
    if current_hash != expected:
        if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Invalid source digest')
        record = Path(__file__).parent / 'source_snapshots' / (expected + '.txt')
        if not record.is_file() or th.digest(record.read_bytes()) != expected:
            raise ValueError(f'Historical source unavailable or changed: {name}')
    return dict(name=name, recorded_sha256=expected, current_sha256=current_hash,
                current_bytes_identical=current_hash == expected,
                verified_record=record.relative_to(ROOT).as_posix())


def audit(folder):
    folder = Path(folder)
    inventory = json.loads((folder / "inventory.json").read_text(encoding="utf-8"))
    for name, expected in inventory.items():
        if Path(name).name != name or th.digest((folder / name).read_bytes()) != expected:
            raise ValueError(f"Invalid exported inventory: {name}")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    with np.load(folder / "series.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    checked = 0
    for e in summary["evaluation"]:
        for method, row in e["methods"].items():
            values = arrays[e["key"]+"_"+method]
            t, j = np.unravel_index(values.argmax(), values.shape)
            assert np.isfinite(values).all() and np.all(values >= 0)
            assert np.isclose(row["episode_max"], values.max(), rtol=1e-12, atol=1e-12)
            assert np.isclose(row["final_max"], values[-1].max(), rtol=1e-12, atol=1e-12)
            assert (row["peak_round"], row["peak_client"]) == (t+1, j)
            for stop, checkpoint in row["checkpoints"].items():
                v = values[int(stop)-1]
                assert np.isclose(checkpoint["max_score"], v.max(), rtol=1e-12, atol=1e-12)
                assert checkpoint["top_clients"] == np.flatnonzero(np.isclose(v, v.max())).tolist()
            checked += 1
    report = dict(export=folder.relative_to(ROOT).as_posix(), inventory_verified=True,
                  method_summaries_checked=checked, source_replay=False, missing_sources=[])
    report["single_round_dominance"] = [dict(key=e["key"], group=e["group"],
        glr_max=e["methods"]["temporal_glr"]["episode_max"],
        single_round_score=e["methods"]["innovation"]["episode_max"]**2/2,
        entire_episode_max_explained_by_single_round=bool(np.isclose(
            e["methods"]["temporal_glr"]["episode_max"],
            e["methods"]["innovation"]["episode_max"]**2/2, rtol=1e-10, atol=1e-10)))
        for e in summary["evaluation"]]
    specs = plan["fit"] + plan.get("calibration", []) + [p["evidence"] for p in plan["test"]]
    sources = []
    for spec, row in zip(specs, summary["evaluation"]):
        name, sep, phase = spec.partition("::")
        path = th.resolve_recorded(name)
        if not path.exists():
            # A later portable copy may have a different filesystem path.
            candidates = []
            for receipt in (ROOT / "experiments/detection/hrac/results").glob("*/*/export.json"):
                meta = json.loads(receipt.read_text(encoding="utf-8"))
                if meta.get("inventory_sha256") == row["input_inventory_sha256"]:
                    candidate = receipt.parent / meta["archive"]
                    if candidate.parent != receipt.parent or th.digest(candidate.read_bytes()) != meta["sha256"]:
                        raise ValueError("Invalid portable evidence receipt")
                    candidates.append(candidate)
            if len(candidates) == 1:
                path = candidates[0]
            elif len(candidates) > 1:
                raise ValueError("Ambiguous portable source evidence")
        if not path.exists():
            report["missing_sources"].append(spec)
        else:
            sources.append(th.load_evidence(str(path)+(sep+phase if sep else "")))
    if not report["missing_sources"]:
        report['code_provenance'] = [source_provenance(name, expected)
                                    for name, expected in summary['code_sha256'].items()]
        report['replay_code_scope'] = 'Current code; historical source bytes checked separately; all saved numerical outputs must match.'
        nfit = len(plan["fit"])
        model = th.fit_model(sources[:nfit])
        saved = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        max_error = 0.
        for key, value in model.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_allclose(value, saved[key], rtol=1e-10, atol=1e-10)
            else:
                assert value == saved[key]
        for episode, row in zip(sources, summary["evaluation"]):
            assert episode["inventory_sha256"] == row["input_inventory_sha256"]
            recomputed = th.score(episode, model)
            for method, values in recomputed["paths"].items():
                stored = arrays[row["key"]+"_"+method]
                np.testing.assert_allclose(values, stored, rtol=1e-10, atol=1e-10)
                max_error = max(max_error, float(np.max(np.abs(values-stored))))
            for key in ("glr_coordinate", "glr_interval_start"):
                np.testing.assert_array_equal(recomputed[key], arrays[row["key"]+"_"+key])
        report.update(source_replay=True, replayed_paths=len(sources)*len(th.METHODS),
                      max_absolute_replay_error=max_error)
    # All comparisons below use supplied score arrays, never fit a new threshold.
    holdout_normal = [e for e in summary["evaluation"] if e["split"] == "test" and e["benign"]]
    attacked = [e for e in summary["evaluation"] if e["split"] == "test" and not e["benign"]]
    comparisons = []
    for e in attacked:
        item = dict(group=e["group"], methods={})
        for method in th.METHODS:
            v = arrays[e["key"]+"_"+method]
            info = dict(attack_episode_max=float(v.max()), final_scores=v[-1].tolist())
            if holdout_normal:
                control = [arrays[n["key"]+"_"+method] for n in holdout_normal]
                bound = max(float(x.max()) for x in control)
                hits = np.flatnonzero(v.max(1)>bound)
                info.update(heldout_normal_episode_maxima=[float(x.max()) for x in control],
                            first_above_observed_normal_envelope=int(hits[0]+1) if len(hits) else None,
                            max_ratio_to_observed_normal_envelope=float(v.max()/bound) if bound else None)
            item["methods"][method] = info
        comparisons.append(item)
    report.update(comparisons=comparisons,
        interpretation="Observed normal maxima are retrospective descriptive envelopes, not calibrated FPR thresholds.",
        source_limit="Derived-array integrity does not replace source-history replay or checkpoint verification.")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = dict(schema=1, training_calls=0, model_unlearning_calls=0,
                  audits=[audit(folder.resolve()) for folder in args.analysis])
    if args.out.exists():
        raise ValueError("Output already exists; use a new --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps([dict(export=a["export"], source_replay=a["source_replay"],
                          checked=a["method_summaries_checked"], missing=len(a["missing_sources"]))
                      for a in result["audits"]], indent=2))


if __name__ == "__main__":
    main()
