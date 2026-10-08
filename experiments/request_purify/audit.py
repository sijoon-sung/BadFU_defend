"""Read saved results without training, unlearning, or changing a detector policy.

python -m experiments.request_purify.audit --input results_interim/results_to_share.zip
Optional --history reads a locally generated history.pt for raw pair diagnostics.
Attacker/target identities here are evaluation labels, never defense inputs.
"""
import argparse
import json
from pathlib import Path
import zipfile


def read_results(path):
    path = Path(path)
    if path.is_dir():
        return {p.relative_to(path).as_posix(): json.loads(p.read_text(encoding="utf-8"))
                for p in path.rglob("*.json")}
    with zipfile.ZipFile(path) as archive:
        # Read JSON directly; never extract paths or load pickle from an archive.
        return {n: json.loads(archive.read(n)) for n in archive.namelist() if n.endswith(".json")}


def audit_results(records, attacker=0):
    suite = records.get("suite.json", {})
    target = suite.get("configuration", {}).get("target", 0)
    cases = []
    for name in sorted(records):
        if not name.startswith("eval-attack-") or not name.endswith("/training_metrics.json"):
            continue
        case = name.split("/")[0]
        training = records[name]
        manifest = records.get(f"{case}/manifest.json", {})
        arms = {}
        baseline = records.get(f"{case}/none/result.json", {})
        for arm in ("none", "zero", "detected", "oracle", "random", "retrain"):
            result = records.get(f"{case}/{arm}/result.json")
            if not result:
                continue
            events = result.get("events", [])
            final = result["final"]
            count_check = None
            if "asr_hits" in final and "asr_total" in final:
                total, hits = final["asr_total"], final["asr_hits"]
                count_check = (total > 0 and 0 <= hits <= total and
                               abs(final["asr"] - 100 * hits / total) < 1e-6)
            arms[arm] = {
                "acc": final["acc"], "asr": final["asr"],
                "asr_count_consistent": count_check,
                "status": result["detection"]["status"],
                "rank": result["detection"]["rank"],
                "recorded_rounds": len(events),
                "applied_rounds": sum(e.get("applied_norm", 0) > 0 for e in events),
                "guard_rejected_rounds": sum(bool(e.get("guard_rejected")) for e in events),
                "applied_norm_sum": sum(e.get("applied_norm", 0) for e in events),
                "same_model_as_none": (result["model_sha256"] == baseline["model_sha256"]
                                       if baseline else None),
            }
        detection = records.get(f"{case}/detected/detection.json", {})
        benign_case = case.replace("eval-attack-", "eval-benign-", 1)
        benign = records.get(f"{benign_case}/detected/detection.json", {})
        if detection.get("scores") and not 0 <= attacker < len(detection["scores"]):
            raise ValueError(f"Invalid attacker identity for {case}: {attacker}")
        row = detection.get("scores", [])[attacker] if detection.get("scores") else None
        score = detection.get("request_score")
        threshold = detection.get("config", {}).get("request_threshold")
        gap = threshold - score if score is not None and threshold is not None else None
        post = records.get(f"{case}/detected/post_clean.json", {}).get("track", [])
        cases.append({
            "case": case, "seed": manifest.get("seed"),
            "requester": manifest.get("requester"),
            "pre_unlearning": training[-1] if training else None,
            "training_acc_asr": [{k: row[k] for k in ("round", "acc", "asr")} for row in training],
            "request_score": score, "request_threshold": threshold, "threshold_minus_score": gap,
            "benign_request_score": benign.get("request_score"),
            "known_attacker": attacker, "known_target": target,
            "known_attacker_scores": row,
            "known_attacker_target_score": row[target] if row else None,
            "arms": arms,
            "clean_continuation_asr": [{"round": p["round"], "asr": p["asr"]} for p in post],
        })
    return {"suite_status": suite.get("status"), "environment": suite.get("environment"),
            "cases": cases,
            "limits": ["Only recorded cases are audited; configured seeds may be incomplete.",
                       "Model hashes and metrics are reported from logs, not independently re-inferred.",
                       "Old outputs omit ASR counts and margins; missing checks stay null.",
                       "Oracle means known client and target head directions, not the true backdoor subspace.",
                       "No detection threshold, attack data, or defense setting is fitted by this audit."]}


def audit_history(path, detector_config, requester, attacker, target):
    import torch
    from .core import DetectorConfig, scores
    # Local history is a trusted PyTorch artifact created by our experiment runner.
    history = torch.load(path, map_location="cpu", weights_only=False)
    h, w = history["heads"], history["weights"]
    requester = history["manifest"]["requester"] if requester is None else requester
    cfg = DetectorConfig(**detector_config)
    score, raw, _ = scores(h, w, requester, cfg)
    if not 0 <= attacker < h.shape[1] or not 0 <= target < h.shape[2]:
        raise ValueError("Attacker or target identity is outside the recorded history")
    x = h * w[:, :, None, None]
    a, q = x[:, attacker, target], x[:, requester, target]
    an, qn = a.norm(dim=-1), q.norm(dim=-1)
    valid = (an > cfg.eps) & (qn > cfg.eps)
    cosine = (a * q).sum(-1) / (an * qn).clamp_min(cfg.eps)
    return {"trajectory_id": history["manifest"]["trajectory_id"],
            "requester": requester, "attacker": attacker, "target": target,
            "window": cfg.window, "known_pair_score": float(score[attacker, target]),
            "window_raw_scores": raw[:, attacker, target].tolist(),
            "rounds": [{"round": t + 1, "weighted_attacker_norm": float(an[t]),
                        "weighted_requester_norm": float(qn[t]),
                        "cosine": float(cosine[t]) if valid[t] else None}
                       for t in range(len(h))],
            "note": "Head-row geometry only. It cannot identify which samples caused a direction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Result ZIP or suite directory")
    parser.add_argument("--out", type=Path, help="Optional JSON report; existing files are not overwritten")
    parser.add_argument("--attacker", type=int, default=0, help="Offline evaluation identity only")
    parser.add_argument("--history", type=Path, help="Optional local trusted history.pt")
    args = parser.parse_args()
    if args.attacker < 0:
        parser.error("--attacker must be nonnegative")
    records = read_results(args.input)
    result = audit_results(records, args.attacker)
    if args.history:
        config = records.get("suite.json", {}).get("configuration", {})
        detector_config = records.get("detector_policy.json", {}).get(
            "config", {"window": config.get("window", 16)})
        result["history_geometry"] = audit_history(args.history, detector_config,
                                                  config.get("requester"), args.attacker,
                                                  config.get("target", 0))
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
