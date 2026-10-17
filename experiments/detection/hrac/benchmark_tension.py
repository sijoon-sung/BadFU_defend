"""Label-blind ledger ablations and independent-clean-episode calibration.

This evaluates pre-FU structural scores, NOT ASR or recovery. All coordinates,
requests and inspection times in the registered family enter the episode maximum.
Multiple methods/configurations are separate analyses; selecting the best on the
test set does not inherit the per-configuration false-alarm bound.
"""


from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
from reproduction.common.paths import research_source
import hashlib
import math
from pathlib import Path
import numpy as np
from experiments.shared.trajectory_tools.tension_audit import boundary_profile

METHODS = ("selective_v3", "cross_v2", "tension_only", "removed_negative", "linear_release")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_threshold(values, alpha=.05):
    """Split-conformal upper order statistic, with STRICT score > threshold.

    Values are maxima from independent normal training episodes, not clients,
    probe images or rounds. None means +infinity (insufficient calibration n).
    Exchangeability and a frozen scoring family remain assumptions, not checks.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or not 0 < alpha < 1:
        raise ValueError("finite independent episode values and 0 < alpha < 1 required")
    rank = math.ceil((len(x)+1)*(1-alpha)-1e-12)
    return {"n": len(x), "rank": rank, "alpha": alpha,
            "threshold": float(np.sort(x)[rank-1]) if rank <= len(x) else None,
            "status": "calibrated" if rank <= len(x) else "insufficient_clean_episodes"}


def methods(effects, removal):
    p = boundary_profile(effects, removal)
    return {"selective_v3": p["score"], "cross_v2": p["score_v2"],
            "tension_only": p["tension"], "removed_negative": p["removed_negative"],
            "linear_release": np.maximum(p["removed_negative"]-p["removed_positive"], 0)}


def request_family(k):
    """Predeclared label-blind family; roles never decide which requests to test."""
    for fraction in (.2, 1.):
        for j in range(k):
            r = np.zeros(k); r[j] = fraction
            yield f"unit:{j}@{fraction:g}", r
    for fraction in (.2, .8, 1.):
        yield f"all@{fraction:g}", np.full(k, fraction)


def load_ledger(path):
    with np.load(path, allow_pickle=False) as d:
        k, t = int(d["K"]), int(d["T"])
        ts, js = d["ts"].copy(), d["js"].copy()
        values = ({"logprob": d["logprob_vs"].copy(), "centered_logit": d["logit_vs"].copy()}
                  if "logprob_vs" in d else {"logprob": d["vs"].copy()})
    if k < 1 or t < 1 or ts.ndim != 1 or js.shape != ts.shape or len(ts) == 0:
        raise ValueError(f"invalid ledger indices: {path}")
    if not np.isin(js, np.arange(k)).all() or not np.isin(ts, np.arange(1, t+1)).all():
        raise ValueError(f"nonintegral or out-of-range event indices: {path}")
    if len(set(zip(ts.tolist(), js.tolist()))) != len(ts):
        raise ValueError(f"duplicate round/unit events: {path}")
    for v in values.values():
        if v.ndim not in (2, 3) or v.shape[0] != len(ts) or 0 in v.shape or not np.isfinite(v).all():
            raise ValueError(f"invalid functional effects: {path}")
    return k, t, ts, js, values


def score_ledger(path):
    """No labels accepted. Returns fixed-family maxima and descriptive rankings."""
    k, t, ts, js, values = load_ledger(path)
    result = {}
    for coordinate, event_values in values.items():
        modes = ("mean_before", "mean_after") if event_values.ndim == 3 else ("mean_before",)
        for mode in modes:
            v = event_values.mean(1) if mode == "mean_before" and event_values.ndim == 3 else event_values
            for window in (0, 10, 20):
                key = f"{coordinate}/{mode}/window:{window}"
                configs = {m: {"episode_max": 0., "snapshots": []} for m in METHODS}
                for stop in sorted(set(min(s, t) for s in (10, 20, 40, t))):
                    e = np.zeros((k,)+v.shape[1:], dtype=float)
                    mask = (ts <= stop) & (ts > (stop-window if window else 0))
                    np.add.at(e, js[mask].astype(int), v[mask])
                    rows = {m: [] for m in METHODS}
                    for request, removal in request_family(k):
                        for method, scores in methods(e, removal).items():
                            # Per-input score is computed before input averaging.
                            by_class = scores.mean(0) if scores.ndim == 2 else scores
                            c = int(by_class.argmax())
                            rows[method].append({"request": request, "class": c, "score": float(by_class[c])})
                    for method in METHODS:
                        rows[method].sort(key=lambda r: (-r["score"], r["request"]))
                        configs[method]["episode_max"] = max(configs[method]["episode_max"], rows[method][0]["score"])
                        configs[method]["snapshots"].append({"round": stop, "ranking": rows[method]})
                result[key] = configs
    return {"cache": str(path), "sha256": sha256(path), "K": k, "T": t, "analyses": result}


def validate_jobs(jobs):
    ids, paths, seeds = set(), set(), set()
    for job in jobs:
        path = str(Path(job["cache"]).resolve())
        seed_key = (job["stratum"], job["seed"])
        if job["id"] in ids or path in paths or seed_key in seeds:
            raise ValueError("duplicate episode ID, cache, or stratum/training seed; no split leakage allowed")
        if job["split"] not in ("calibration", "clean_test", "attack_development", "attack_test"):
            raise ValueError("unknown split")
        ids.add(job["id"]); paths.add(path); seeds.add(seed_key)


def summarize(jobs, scored, alpha=.05):
    validate_jobs(jobs)
    if len(jobs) != len(scored):
        raise ValueError("one score per independent episode required")
    hashes = [s["sha256"] for s in scored]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate cache content cannot count as independent episodes")
    groups = []
    for stratum in sorted({j["stratum"] for j in jobs}):
        pairs = [(j, s) for j, s in zip(jobs, scored) if j["stratum"] == stratum]
        keys = set(pairs[0][1]["analyses"])
        if any(set(s["analyses"]) != keys for _, s in pairs):
            raise ValueError("measurement configurations must match within each stratum")
        for config in sorted(keys):
            for method in METHODS:
                cal = [s["analyses"][config][method]["episode_max"] for j, s in pairs if j["split"] == "calibration"]
                rule = clean_threshold(cal, alpha)
                threshold = rule["threshold"]
                rows = []
                for job, score in pairs:
                    if job["split"] == "calibration":
                        continue
                    analysis = score["analyses"][config][method]
                    alarm = analysis["episode_max"] > threshold if threshold is not None else None
                    row = {"id": job["id"], "split": job["split"], "episode_max": analysis["episode_max"], "alarm": alarm}
                    # Ground-truth roles are used only here, AFTER all scores and calibration.
                    if job["split"].startswith("attack"):
                        sup = job["eval_suppressor"]
                        candidates = [r for snap in analysis["snapshots"] for r in snap["ranking"]
                                      if r["request"] == f"unit:{sup}@1"]
                        row["requested_suppressor_alarm"] = (any(r["score"] > threshold for r in candidates)
                                                             if threshold is not None else None)
                        row["requested_suppressor_target_alarm"] = (any(r["score"] > threshold and r["class"] == job["eval_target"] for r in candidates)
                                                                    if threshold is not None else None)
                    rows.append(row)
                clean = [r for r in rows if r["split"] == "clean_test"]
                groups.append({"stratum": stratum, "config": config, "method": method, "calibration": rule,
                               "clean_test_n": len(clean),
                               "clean_episode_fpr": (sum(r["alarm"] for r in clean)/len(clean)
                                                     if clean and threshold is not None else None), "episodes": rows})
    return {"method": "selective_tension_v3", "model_unlearning_calls": 0, "groups": groups,
            "primary": "logprob/mean_before/window:0; selective_v3 versus linear_release at alpha=.05",
            "limits": ["episode-FPR bound assumes exchangeable clean runs within each stratum",
                       "secondary configurations are not jointly calibrated; no test-set method selection",
                       "pilot calibration below 19 episodes cannot produce finite alpha=.05 thresholds",
                       "attack scores do not measure attack validity, ASR, forgetting, or recovery",
                       "FUBA client records do not validate original BadFU within-client records"]}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Historical FUBA descriptive ablation, without FPR or training")
    parser.add_argument("--folder", type=Path, default=(Path(__file__).resolve().parents[3]/"FUBA")/"logs"/"ledger")
    parser.add_argument("--out", type=Path, default=(Path(__file__).resolve().parents[3]/"FUBA")/"logs"/"tension_audit"/"historical_ablation_v3.json")
    args = parser.parse_args()
    paths = sorted(args.folder.glob("bd_*_r100w5_k20paper_s*_push.npz"))
    if not paths:
        parser.error("no historical FUBA paper-setting caches found")
    scores = [score_ledger(path) for path in paths]
    table = []
    for method in METHODS:
        for stop in (10, 20, 40, 99):
            chosen = []
            for score in scores:
                snaps = score["analyses"]["logprob/mean_before/window:0"][method]["snapshots"]
                snap = next(s for s in snaps if s["round"] == stop)
                chosen.append(next(r for r in snap["ranking"] if r["request"].startswith("unit:") and r["request"].endswith("@1")))
            row = {"method": method, "round": stop, "episodes": len(chosen),
                   "suppressor_top1": sum(r["request"] == "unit:4@1" for r in chosen),
                   "suppressor_and_target_top1": sum(r["request"] == "unit:4@1" and r["class"] == 8 for r in chosen)}
            table.append(row)
            print(row)
    report = {"mode": "historical_descriptive_ablation", "calibrated": False, "model_unlearning_calls": 0,
              "source_sha256": {n: sha256(research_source(n)) for n in ("tension_audit.py", "benchmark_tension.py")},
              "limits": ["same six development trajectories, not independent confirmation",
                         "tie order is deterministic and not evidence of unique identification",
                         "no normal controls, ASR, or BadFU within-client evaluation"],
              "table": table, "scores": scores}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False)+"\n", encoding="utf-8")
