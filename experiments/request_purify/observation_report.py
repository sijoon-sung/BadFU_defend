"""Export shadow measurements separately from defense ACC/ASR results."""
import csv
import json
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def package_observations(root):
    rows, actual = [], []
    for path in sorted(root.glob("eval-*/observation_scores.json")):
        result = read(path)
        for name, candidate in result["candidates"].items():
            for request in candidate["requests"]:
                row = {"seed": result["seed"], "benign": result["benign"], "candidate": name,
                       "actual_requester": result["actual_requester"], "observed_requester": request["requester"],
                       "uses_request_effect": candidate["uses_request_effect"], "score": request["score"],
                       "threshold": request["threshold"], "alarm": request["alarm"]}
                rows.append(row)
                if request["requester"] == result["actual_requester"]:
                    actual.append(row)
    if not rows:
        return
    write_csv(root / "observation_scores.csv", rows)
    groups, paired = [], []
    for name in sorted({r["candidate"] for r in actual}):
        for benign in (False, True):
            values = [r for r in actual if r["candidate"] == name and r["benign"] == benign]
            if values:
                groups.append({"candidate": name, "benign": benign, "independent_seeds": len(values),
                               "alarms": sum(r["alarm"] for r in values),
                               "alarm_fraction": statistics.mean(r["alarm"] for r in values),
                               "score_mean": statistics.mean(r["score"] for r in values)})
        for attack in [r for r in actual if r["candidate"] == name and not r["benign"]]:
            normal = next((r for r in actual if r["candidate"] == name and r["benign"] and r["seed"] == attack["seed"]), None)
            if normal:
                paired.append({"candidate": name, "seed": attack["seed"],
                               "attack_minus_benign_score": attack["score"] - normal["score"]})
    overview = {"mode": "shadow_only", "controls_purification": False, "groups": groups, "paired": paired,
                "note": "Actual requester only in aggregate statistics. Other observed q are input ablations on the same history, not additional deletion experiments. Separate candidate alarms are not an OR detector. FU traces are uncalibrated; no FU detection rate or latency is claimed."}
    (root / "observation_summary.json").write_text(json.dumps(overview, indent=2, allow_nan=False), encoding="utf-8")
    fu_rows = []
    for path in sorted(root.glob("eval-*/*/fu_observations.json")):
        manifest = read(path.parent.parent / "manifest.json")
        for event in read(path)["events"]:
            q = event["actual_requester"]
            for c, value in enumerate(event["step_margin_rms"]):
                cos = event["request_effect_cosine"][q]
                valid = event["request_effect_valid"][q]
                observed = [cos[i][c] for i in range(len(cos)) if valid[i][c]]
                fu_rows.append({"seed": manifest["seed"], "benign": manifest["benign"],
                                "arm": path.parent.name, "round": event["round"], "class": c,
                                "actual_requester": q, "step_margin_rms": value,
                                "step_margin_rms_bn": event["step_margin_rms_bn"][c],
                                "step_margin_mean": event["step_margin_mean"][c],
                                "margin_rms_from_original_final": event["margin_rms_from_original_final"][c],
                                "probability_mean": event["probability_mean"][c],
                                "request_effect_min_cosine": min(observed) if observed else None,
                                "request_effect_valid_clients": len(observed)})
    write_csv(root / "fu_observation_curves.csv", fu_rows)
