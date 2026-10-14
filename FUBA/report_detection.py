"""Read a completed tension experiment and report detection, without training/FU.

The registered score, request family, calibration and experiment source hashes
are unchanged. This is an evaluation report, not another detector. Input files
can be copied from a different machine; checkpoints and caches are not needed.
"""
import argparse
import json
import math
from pathlib import Path

from scipy.stats import binomtest

from benchmark_tension import clean_threshold, request_family, sha256, validate_jobs


CONFIG = "logprob/mean_before/window:0"
METHODS = ("selective_v3", "linear_release")
CONTROL_METHODS = ("tension_only", "removed_negative", "cross_v2")
COMPARISONS = {
    "selective_v3": {"role": "primary_proposed", "request_specific": True,
                     "question": "Does cancellation combined with selective removal add value?"},
    "linear_release": {"role": "primary_baseline", "request_specific": True,
                       "question": "Is the positive net effect of removing contributions sufficient?"},
    "tension_only": {"role": "component_control", "request_specific": False,
                     "question": "Does cancellation alone distinguish episodes, without identifying a request?"},
    "removed_negative": {"role": "component_control", "request_specific": True,
                         "single_unit_score_equivalent_to": "linear_release",
                         "question": "For grouped requests, does ignoring removed positive contributions matter?"},
    "cross_v2": {"role": "historical_revision", "request_specific": True,
                 "question": "What changes when replacing the previous boundary formula?"},
}


def cache_name(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def rate(values):
    """Exact binomial interval over independent episodes, never rounds/clients."""
    n = len(values)
    if not n or any(v is None for v in values):
        return {"n": n, "count": None, "rate": None, "ci95_exact": None,
                "status": "no_episodes" if not n else "uncalibrated"}
    k = sum(bool(v) for v in values)
    ci = binomtest(k, n).proportion_ci(confidence_level=.95, method="exact")
    return {"n": n, "count": k, "rate": k/n,
            "ci95_exact": [float(ci.low), float(ci.high)], "status": "observed"}


def bind_scores(jobs, scores, methods=METHODS):
    """Bind by unique cache filename, not list position or source-machine path."""
    validate_jobs(jobs)
    names = [cache_name(j["cache"]) for j in jobs]
    score_names = [cache_name(s["cache"]) for s in scores]
    if (len(set(names)) != len(names) or len(set(score_names)) != len(score_names)
            or set(names) != set(score_names)):
        raise ValueError("exactly one matching score cache per registered job required")
    hashes = [s["sha256"] for s in scores]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate cache content cannot count as independent episodes")
    if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
        raise ValueError("invalid cache SHA256")
    by_name = dict(zip(score_names, scores))
    paired = [(j, by_name[cache_name(j["cache"])]) for j in jobs]
    for job, score in paired:
        if score["K"] != job["K"] or score["T"] != job["rounds"]-1:
            raise ValueError("score trajectory dimensions disagree with manifest")
        expected_times = sorted(set(min(t, score["T"]) for t in (10, 20, 40, score["T"])))
        expected_requests = {name for name, _ in request_family(score["K"])}
        for method in methods:
            analysis = score["analyses"][CONFIG][method]
            snapshots = analysis["snapshots"]
            if [s["round"] for s in snapshots] != expected_times:
                raise ValueError("inspection times disagree with registered family")
            observed_max = 0.
            for snap in snapshots:
                rows = snap["ranking"]
                if (len(rows) != len(expected_requests)
                        or {r["request"] for r in rows} != expected_requests):
                    raise ValueError("incomplete or duplicate request family")
                if any(not math.isfinite(r["score"]) or r["score"] < 0 for r in rows):
                    raise ValueError("nonfinite or negative score")
                observed_max = max(observed_max, max(r["score"] for r in rows))
            if not math.isclose(analysis["episode_max"], observed_max, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("episode maximum disagrees with snapshots")
        if job["split"].startswith("attack"):
            if f"unit:{job['eval_suppressor']}@1" not in expected_requests:
                raise ValueError("evaluation request outside registered family")
    return paired


def episode_row(job, score, method, threshold):
    analysis = score["analyses"][CONFIG][method]

    def alarm(value):
        return value > threshold if threshold is not None else None

    row = {"id": job["id"], "split": job["split"], "cache_sha256": score["sha256"],
           "episode_max": analysis["episode_max"], "any_request_alarm": alarm(analysis["episode_max"])}
    if not job["split"].startswith("attack"):
        return row
    # Evaluation truth enters only after score generation and clean calibration.
    request = f"unit:{job['eval_suppressor']}@1"
    candidates = [(snap["round"], next(r for r in snap["ranking"] if r["request"] == request))
                  for snap in analysis["snapshots"]]
    hits = [(t, r) for t, r in candidates if threshold is not None and r["score"] > threshold]
    requested_alarm = alarm(max(r["score"] for _, r in candidates))
    row.update(evaluated_request=request, request_max_score=max(r["score"] for _, r in candidates),
               request_alarm=requested_alarm, final_request_score=candidates[-1][1]["score"],
               final_request_alarm=alarm(candidates[-1][1]["score"]),
               first_request_alarm_inspection=(hits[0][0] if hits else None),
               selected_target_alarm=(any(r["class"] == job["eval_target"] for _, r in hits)
                                      if threshold is not None else None),
               unrelated_only_alarm=(row["any_request_alarm"] and not requested_alarm
                                     if threshold is not None else None))
    return row


def build_report(manifest, scores, include_controls=False):
    if manifest.get("schema") != 1 or manifest.get("alpha") != .05:
        raise ValueError("requires existing schema-1 tension experiment at alpha=.05")
    if manifest.get("primary") != f"{CONFIG}; selective_v3 versus linear_release":
        raise ValueError("registered primary differs; do not silently select a new analysis")
    methods = METHODS + CONTROL_METHODS if include_controls else METHODS
    paired = bind_scores(manifest["jobs"], scores, methods)
    groups = []
    for stratum in sorted({j["stratum"] for j, _ in paired}):
        subset = [(j, s) for j, s in paired if j["stratum"] == stratum]
        for method in methods:
            rule = clean_threshold([s["analyses"][CONFIG][method]["episode_max"]
                                    for j, s in subset if j["split"] == "calibration"], manifest["alpha"])
            rows = [episode_row(j, s, method, rule["threshold"])
                    for j, s in subset if j["split"] != "calibration"]
            group = {"stratum": stratum, "config": CONFIG, "method": method, "calibration": rule,
                     "comparison": COMPARISONS[method],
                     "clean_episode_fpr": rate([r["any_request_alarm"] for r in rows if r["split"] == "clean_test"]),
                     "attack_splits": {}, "episodes": rows}
            for split in ("attack_development", "attack_test"):
                attack = [r for r in rows if r["split"] == split]
                group["attack_splits"][split] = {
                    "independent_confirmation": split == "attack_test",
                    "requested_attack_detection_rate": rate([r["request_alarm"] for r in attack]),
                    "final_requested_attack_detection_rate": rate([r["final_request_alarm"] for r in attack]),
                    "any_request_alarm_rate": rate([r["any_request_alarm"] for r in attack]),
                    "selected_target_alarm_rate": rate([r["selected_target_alarm"] for r in attack]),
                    "unrelated_only_alarm_rate": rate([r["unrelated_only_alarm"] for r in attack])}
            groups.append(group)
    return {"schema": 1, "purpose": "detection-first evaluation of an unchanged registered score",
            "profile": manifest["profile"], "primary": manifest["primary"],
            "controls_included": include_controls,
            "training_calls": 0, "model_unlearning_calls": 0, "groups": groups,
            "activation_validation": {"status": "not_available_in_score_inputs",
                                      "harmful_activation_detection_rate": None},
            "limits": [
                "The report does not change or establish detector novelty or detection performance.",
                "Clean exchangeability within a stratum and frozen scoring/inspection family are assumptions.",
                "Binomial intervals require independent test episodes; multiple requests are not extra episodes.",
                "Each method has its own calibration; an OR ensemble or test-selected winner is not calibrated.",
                "Component controls and historical revisions are secondary analyses, not a new primary comparison.",
                "Tension-only is request-independent: alarms cannot establish request localization.",
                "Removed-negative and linear-release scores coincide for single-unit requests; different family thresholds can still change alarms.",
                "Attack-role detection is not validated prediction of harmful FU activation or malicious intent.",
                "First alarm is a registered inspection of a hypothetical request, not actual request time or delay.",
                "Final-request alarm uses only the last registered snapshot; transient earlier alarms remain separate.",
                "Selected-target results inherit saved argmax tie-breaking; ties are not unique target identification.",
                "BadFU control rejection, risk_after, and ASR reduction are not detection labels.",
                "Inputs are bound by cache name/hash metadata; this report does not re-read original checkpoints/caches."
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="completed folder containing manifest.json and scores.json")
    parser.add_argument("--out", type=Path, help="new JSON file; existing files are never overwritten")
    parser.add_argument("--include-controls", action="store_true",
                        help="also report existing tension-only/removed-negative controls and historical v2; primary remains fixed")
    args = parser.parse_args()
    manifest_path, scores_path = args.experiment/"manifest.json", args.experiment/"scores.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    report = build_report(manifest, scores, include_controls=args.include_controls)
    report["provenance"] = {"manifest_sha256": sha256(manifest_path), "scores_sha256": sha256(scores_path),
                            "reporter_sha256": sha256(__file__),
                            "calibration_code_sha256": sha256(Path(__file__).with_name("benchmark_tension.py")),
                            "experiment_source_sha256": manifest["source_sha256"]}
    out = args.out or args.experiment/"detection_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Detection report -> {out}")
    for group in report["groups"]:
        print(group["stratum"], group["method"], group["calibration"]["status"],
              "clean FPR:", group["clean_episode_fpr"],
              "held-out request detection:", group["attack_splits"]["attack_test"]["requested_attack_detection_rate"])


if __name__ == "__main__":
    main()
