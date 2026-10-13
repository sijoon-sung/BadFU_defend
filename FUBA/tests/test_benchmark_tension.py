"""Calibration, split integrity and score selection counterexamples."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_tension import clean_threshold, methods, score_ledger, validate_jobs, summarize, METHODS
from experiment_tension import make_jobs


class BenchmarkTests(unittest.TestCase):
    def test_five_percent_requires_nineteen_independent_episodes(self):
        self.assertIsNone(clean_threshold(range(18))["threshold"])
        self.assertEqual(clean_threshold(range(19))["threshold"], 18)
        self.assertEqual(clean_threshold([4]*19)["threshold"], 4)
        # Strict > keeps tied normal scores from all becoming false alarms.
        self.assertFalse(4 > clean_threshold([4]*19)["threshold"])

    def test_duplicate_seed_across_splits_is_rejected(self):
        a = {"id": "a", "cache": "a.npz", "seed": 1, "stratum": "iid", "split": "calibration"}
        b = dict(a, id="b", cache="b.npz", split="clean_test")
        with self.assertRaises(ValueError):
            validate_jobs([a,b])

    def test_profiles_have_independent_calibration_and_test_seeds(self):
        jobs = make_jobs("confirm", Path("unused"))
        validate_jobs(jobs)
        self.assertEqual(len(jobs), 68)
        for stratum in ("iid", "owner"):
            self.assertEqual(sum(j["split"] == "calibration" and j["stratum"] == stratum for j in jobs), 19)
        self.assertEqual(len(make_jobs("pilot", Path("unused"))), 18)

    def test_window_and_input_aggregation_expose_hidden_opposition(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder)/"effects.npz"
            # Average before scoring erases opposite effects on two inputs.
            v = np.array([[[5.,0.],[-5.,0.]], [[-5.,0.],[5.,0.]]])
            np.savez(path, K=2, T=20, ts=[1,1], js=[0,1], logprob_vs=v, logit_vs=v)
            data = score_ledger(path)["analyses"]
            self.assertEqual(data["logprob/mean_before/window:0"]["selective_v3"]["episode_max"], 0)
            self.assertGreater(data["logprob/mean_after/window:0"]["selective_v3"]["episode_max"], 0)
            snapshots = data["logprob/mean_after/window:10"]["selective_v3"]["snapshots"]
            self.assertEqual(snapshots[-1]["ranking"][0]["score"], 0)

    def test_baseline_cannot_distinguish_uniform_negative_net_shrinkage(self):
        scores = methods(np.array([[8.], [-10.]]), [.8,.8])
        self.assertAlmostEqual(scores["selective_v3"].item(), 0)
        self.assertGreater(scores["linear_release"].item(), 0)

    def test_duplicate_events_are_not_extra_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder)/"invalid.npz"
            np.savez(path, K=2, T=1, ts=[1,1], js=[0,0], vs=[[1.,0.],[1.,0.]])
            with self.assertRaises(ValueError):
                score_ledger(path)

    def test_calibration_uses_only_clean_episodes_and_target_match_is_separate(self):
        jobs, scores = [], []
        for i in range(21):
            split = "calibration" if i < 19 else ("clean_test" if i == 19 else "attack_test")
            jobs.append({"id": str(i), "cache": f"{i}.npz", "seed": i, "stratum": "iid", "split": split,
                         "eval_suppressor": 4, "eval_target": 8})
            value = float(i if i < 19 else (18 if i == 19 else 1000))
            scores.append({"sha256": str(i), "analyses": {"fixed": {
                method: {"episode_max": value, "snapshots": [{"ranking": [
                    {"request": "unit:4@1", "class": 3, "score": value}]}]} for method in METHODS}}})
        result = summarize(jobs, scores)
        for group in result["groups"]:
            self.assertEqual(group["calibration"]["threshold"], 18)
            self.assertEqual(group["clean_episode_fpr"], 0)
            attack = group["episodes"][-1]
            self.assertTrue(attack["alarm"])
            self.assertTrue(attack["requested_suppressor_alarm"])
            self.assertFalse(attack["requested_suppressor_target_alarm"])
        # Relabeling attack truth cannot change scores or the clean threshold.
        jobs[-1]["eval_target"] = 3
        self.assertEqual(summarize(jobs, scores)["groups"][0]["calibration"], result["groups"][0]["calibration"])
        scores[-1]["sha256"] = scores[0]["sha256"]
        with self.assertRaises(ValueError):
            summarize(jobs, scores)


if __name__ == "__main__":
    unittest.main()
