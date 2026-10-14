"""Counterexamples to misleading detection counts; no learned attack results."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_tension import request_family
from report_detection import CONFIG, METHODS, CONTROL_METHODS, build_report, rate


def fixture(ncal=19):
    jobs, scores = [], []
    for i in range(ncal+4):
        split = "calibration" if i < ncal else ("clean_test" if i == ncal else
                 ("attack_development" if i == ncal+3 else "attack_test"))
        jobs.append({"id": f"e{i}", "cache": f"/server/caches/e{i}.npz", "stratum": "owner",
                     "seed": i, "split": split, "K": 2, "rounds": 21, "eval_suppressor": 0, "eval_target": 1})
        analyses = {}
        for method in METHODS:
            snapshots = []
            for t in (10, 20):
                rows = []
                for request, _ in request_family(2):
                    value = 1. if split == "calibration" else 0.
                    if i == ncal+1 and request == "unit:1@1":
                        value = 2.  # Any request fires, but the attack request is missed.
                    if i == ncal+2 and request == "unit:0@1" and t == 10:
                        value = 2.  # Transient hit does not count as a final-time hit.
                    if split == "attack_development" and request == "unit:0@1":
                        value = 1000.
                    rows.append({"request": request, "score": value, "class": 1})
                snapshots.append({"round": t, "ranking": rows})
            analyses[method] = {"episode_max": max(r["score"] for s in snapshots for r in s["ranking"]),
                                "snapshots": snapshots}
        scores.append({"cache": f"C:\\copied\\e{i}.npz", "sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                       "K": 2, "T": 20, "analyses": {CONFIG: analyses}})
    return {"schema": 1, "profile": "test_fixture", "alpha": .05, "jobs": jobs,
            "primary": f"{CONFIG}; selective_v3 versus linear_release"}, scores


class DetectionReportTests(unittest.TestCase):
    def test_secondary_controls_preserve_primary_and_mark_request_blind_score(self):
        manifest, scores = fixture()
        for score in scores:
            analyses = score["analyses"][CONFIG]
            for name in CONTROL_METHODS:
                analyses[name] = copy.deepcopy(analyses["linear_release"])
            for snap in analyses["tension_only"]["snapshots"]:
                value = max(r["score"] for r in snap["ranking"])
                for row in snap["ranking"]:
                    row["score"] = value
        primary = build_report(manifest, scores)
        extended = build_report(manifest, scores, include_controls=True)
        self.assertEqual(primary["groups"], extended["groups"][:2])
        self.assertEqual(primary["primary"], extended["primary"])
        by_name = {g["method"]: g for g in extended["groups"]}
        self.assertEqual(set(by_name), set(METHODS + CONTROL_METHODS))
        self.assertFalse(by_name["tension_only"]["comparison"]["request_specific"])
        self.assertEqual(by_name["cross_v2"]["comparison"]["role"], "historical_revision")

    def test_requested_controls_are_not_silently_fabricated_if_missing(self):
        manifest, scores = fixture()
        with self.assertRaises(KeyError):
            build_report(manifest, scores, include_controls=True)

    def test_wrong_request_and_development_success_do_not_inflate_held_out_detection(self):
        manifest, scores = fixture()
        report = build_report(manifest, scores)
        for group in report["groups"]:
            self.assertEqual(group["calibration"]["threshold"], 1)
            self.assertEqual(group["clean_episode_fpr"]["rate"], 0)
            attack = group["attack_splits"]["attack_test"]
            self.assertEqual(attack["requested_attack_detection_rate"]["rate"], .5)
            self.assertEqual(attack["any_request_alarm_rate"]["rate"], 1)
            self.assertEqual(attack["unrelated_only_alarm_rate"]["rate"], .5)
            self.assertEqual(attack["final_requested_attack_detection_rate"]["rate"], 0)
            self.assertEqual(group["episodes"][2]["first_request_alarm_inspection"], 10)
        self.assertIsNone(report["activation_validation"]["harmful_activation_detection_rate"])

    def test_pilot_insufficient_calibration_is_unknown_not_zero(self):
        manifest, scores = fixture(3)
        for group in build_report(manifest, scores)["groups"]:
            self.assertEqual(group["calibration"]["status"], "insufficient_clean_episodes")
            self.assertIsNone(group["clean_episode_fpr"]["count"])
            self.assertIsNone(group["attack_splits"]["attack_test"]["requested_attack_detection_rate"]["rate"])

    def test_exact_intervals_do_not_claim_zero_population_fpr(self):
        self.assertAlmostEqual(rate([False]*10)["ci95_exact"][1], 1-.025**.1)
        self.assertEqual(rate([False]*10)["ci95_exact"][0], 0)
        self.assertAlmostEqual(rate([True]*5)["ci95_exact"][0], .025**.2)
        self.assertEqual(rate([])["status"], "no_episodes")

    def test_score_order_and_source_machine_paths_do_not_misassign_truth(self):
        manifest, scores = fixture()
        self.assertEqual(build_report(manifest, scores), build_report(manifest, list(reversed(scores))))

    def test_relabeling_attack_truth_cannot_change_threshold_or_scores(self):
        manifest, scores = fixture()
        before = copy.deepcopy(scores)
        original = build_report(manifest, scores)
        manifest["jobs"][-2]["eval_suppressor"] = 1
        manifest["jobs"][-2]["eval_target"] = 8
        updated = build_report(manifest, scores)
        self.assertEqual(original["groups"][0]["calibration"], updated["groups"][0]["calibration"])
        self.assertEqual(scores, before)

    def test_incomplete_mismatched_and_duplicated_evidence_is_rejected(self):
        for corruption in ("missing", "duplicate", "wrong_time", "wrong_max", "missing_request", "dimensions", "primary"):
            with self.subTest(corruption=corruption):
                manifest, scores = fixture()
                a = scores[0]["analyses"][CONFIG][METHODS[0]]
                if corruption == "missing":
                    scores.pop()
                elif corruption == "duplicate":
                    scores[-1]["sha256"] = scores[0]["sha256"]
                elif corruption == "wrong_time":
                    a["snapshots"][0]["round"] = 1
                elif corruption == "wrong_max":
                    a["episode_max"] = 100
                elif corruption == "missing_request":
                    a["snapshots"][0]["ranking"].pop()
                elif corruption == "dimensions":
                    scores[0]["K"] = 3
                else:
                    manifest["primary"] = "best test-set configuration"
                with self.assertRaises(ValueError):
                    build_report(manifest, scores)

    def test_cli_reads_only_exported_json_and_preserves_existing_report(self):
        manifest, scores = fixture()
        manifest["source_sha256"] = {"registered_fixture": "synthetic_not_performance_data"}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root/"scores.json").write_text(json.dumps(scores), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve().parents[1]/"report_detection.py"),
                       "--experiment", str(root)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            report_path = root/"detection_report.json"
            original = report_path.read_bytes()
            result = json.loads(original)
            self.assertEqual(result["training_calls"], 0)
            self.assertEqual(result["model_unlearning_calls"], 0)
            self.assertEqual(result["provenance"]["manifest_sha256"], hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest())
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(report_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
