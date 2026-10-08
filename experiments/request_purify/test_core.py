"""CPU unit checks plus a small actual-training smoke suite (no CIFAR download)."""
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from .core import DetectorConfig, bounded_correction, build_basis, detect, scores
from .data import BadFUModel, RecordImages, resolve_image
from .engine import calibration_step, calibrate, measure, state_hash
from .audit import audit_history, audit_results, read_results


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectorConfig(min_observations=3, max_rank=2)
        self.h = torch.zeros(4, 3, 2, 3)
        self.h[:, 0, 1, 0] = 2
        self.h[:, 2, 1, 0] = -2
        self.h[:, 1, :, 1] = 1
        self.w = torch.full((4, 3), 1 / 3)

    def test_detects_single_retained_attacker_and_excludes_requester(self):
        b, gamma, report = detect(self.h, self.w, 2, self.cfg)
        self.assertEqual(report["status"], "detected")
        self.assertIn((0, 1), report["pairs"])
        self.assertEqual(gamma[2], 0)
        self.assertEqual(b.shape[1], 1)

    def test_no_forced_attacker_on_zero_or_parallel_updates(self):
        for history in (torch.zeros_like(self.h), torch.ones_like(self.h)):
            b, gamma, report = detect(history, self.w, 2, self.cfg)
            self.assertEqual(report["status"], "no_alarm")
            self.assertEqual(b.shape[1], 0)
            self.assertEqual(gamma.sum(), 0)

    def test_missing_participation_is_not_zero_evidence(self):
        w = self.w.clone(); w[1:, 2] = 0; w[1:, :2] = .5
        _, _, report = detect(self.h, w, 2, self.cfg)
        self.assertEqual(report["status"], "insufficient_history")
        self.assertEqual(report["coverage"][0], 1)

    def test_uncentered_basis_preserves_identical_attack_direction(self):
        raw = torch.ones(4, 3, 2)
        b = build_basis(self.h, raw, [(0, 1)], self.cfg)
        self.assertEqual(b.shape[1], 1)
        self.assertAlmostEqual(abs(float(b[3, 0])), 1)

    def test_projection_budget_and_zero_identity(self):
        b = torch.tensor([[1.], [0.]])
        contributions = torch.tensor([[10., 4.], [3., 6.]])
        gamma = torch.tensor([1., 0.])
        correction, event = bounded_correction(b, gamma, contributions, 1, .2, 100, 2)
        self.assertAlmostEqual(float(correction.norm()), .4, places=6)
        self.assertEqual(correction[1], 0)
        zero, _ = bounded_correction(b, gamma, contributions, 0, .2, 100, 2)
        self.assertEqual(zero.norm(), 0)

    def test_invalid_history_rejected(self):
        h = self.h.clone(); h[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            scores(h, self.w, 2, self.cfg)

    def test_no_implicit_centering(self):
        # Two distinct but co-directed clients must not become opposites.
        h = torch.ones(4, 2, 1, 2); h[:, 0] *= 2
        b, _, report = detect(h, torch.full((4, 2), .5), 1, self.cfg)
        self.assertEqual(report["request_score"], 0)
        self.assertEqual(b.shape[1], 0)

    def test_saved_history_audit_reports_observed_pair_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.pt"
            torch.save({"heads": self.h, "weights": self.w,
                        "manifest": {"requester": 2, "trajectory_id": "fixture"}}, path)
            report = audit_history(path, {"window": 3}, None, 0, 1)
            self.assertEqual(report["window"], 3)
            self.assertEqual(len(report["window_raw_scores"]), 3)
            self.assertEqual([r["cosine"] for r in report["rounds"]], [-1.] * 4)
            self.assertGreater(report["known_pair_score"], 0)


class AdapterTests(unittest.TestCase):
    def test_badfu_resnet_and_interleaved_head_shape(self):
        torch.set_num_threads(1)
        model = BadFUModel(pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertEqual(tuple(model.model.fc.weight.shape), (10, 512))

    def test_record_test_excludes_original_target_class(self):
        from PIL import Image
        from torchvision.transforms import ToTensor
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "badnet_dataset"
            bundle.mkdir()
            for index in (0, 1):
                Image.new("RGB", (32, 32), (index, 0, 0)).save(bundle / f"{index}.png")
            data = {i: {"path": f"old/badnet_dataset/{i}.png", "other_info": [0, i]} for i in (0, 1)}
            ds = RecordImages(data, bundle, ToTensor(), target=0)
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds.samples[0][2], 1)

    def test_calibration_rejects_attack_trajectory(self):
        history = {"manifest": {"benign": False}}
        with self.assertRaises(ValueError):
            calibrate([history], DetectorConfig(), "cpu")

    def test_decomposed_head_sums_to_actual_calibrated_update(self):
        cur = {"fc.weight": torch.tensor([[1., 2.], [3., 4.]]), "fc.bias": torch.tensor([1., 2.]),
               "count": torch.tensor(0, dtype=torch.long)}
        states = [{k: v + i for k, v in cur.items()} for i in (1, 3)]
        base, parts = calibration_step(cur, states, [.25, .75],
                        {"fc.weight": 2., "fc.bias": 3.}, ["fc.weight", "fc.bias"], "fc")
        expected = torch.cat([base["fc.weight"]-cur["fc.weight"],
                              (base["fc.bias"]-cur["fc.bias"])[:, None]], 1).flatten()
        torch.testing.assert_close(parts.sum(0), expected)
        self.assertEqual(base["count"].dtype, torch.long)

    def test_zero_direction_is_finite(self):
        cur = {"fc.weight": torch.ones(2, 2), "fc.bias": torch.ones(2)}
        base, parts = calibration_step(cur, [cur, cur], [.5, .5],
                                      {k: 3 for k in cur}, list(cur), "fc")
        self.assertEqual(parts.norm(), 0)
        for key in cur:
            torch.testing.assert_close(base[key], cur[key])

    def test_conflicting_image_relocations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "badnet_dataset"
            for sub, data in [("bd_train_dataset/1", b"a"), ("cv_train_dataset/pert/1", b"b")]:
                path = bundle / sub; path.mkdir(parents=True)
                (path / "sample.png").write_bytes(data)
            with self.assertRaises(ValueError):
                resolve_image("old/badnet_dataset/bd_train_dataset/1/sample.png", bundle)

    def test_test_images_cannot_fall_back_to_training_images(self):
        # Train/test IDs overlap in CIFAR. A missing test image must not silently
        # become a training image with the same class and basename.
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "badnet_dataset"
            train = bundle / "bd_train_dataset/1"
            train.mkdir(parents=True)
            (train / "5.png").write_bytes(b"train")
            with self.assertRaises(FileNotFoundError):
                resolve_image("old/badnet_dataset/bd_test_dataset/1/5.png", bundle)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(batch_size=2, workers=0, device="cpu")

    def test_asr_counts_predictions_not_assigned_target_labels(self):
        dataset = torch.utils.data.TensorDataset(torch.eye(3) * 4, torch.zeros(3, dtype=torch.long))
        result = measure(torch.nn.Identity(), dataset, self.args, target=0)
        self.assertAlmostEqual(result["accuracy"], 100 / 3)
        self.assertEqual((result["correct"], result["total"]), (1, 3))
        self.assertEqual(result["prediction_counts"], [1, 1, 1])
        self.assertAlmostEqual(result["target_margin_mean"], -4 / 3)

    def test_same_labels_different_model_predictions_change_asr(self):
        labels = torch.zeros(3, dtype=torch.long)
        for logits, expected in ((torch.tensor([[5., 0.]]).repeat(3, 1), 100.),
                                 (torch.tensor([[0., 5.]]).repeat(3, 1), 0.)):
            result = measure(torch.nn.Identity(), torch.utils.data.TensorDataset(logits, labels),
                             self.args, target=0)
            self.assertEqual(result["accuracy"], expected)

    def test_uploaded_seed42_failure_is_visible_without_training(self):
        path = Path(__file__).resolve().parents[2] / "results_interim/results_to_share.zip"
        if not path.exists():
            self.skipTest("Uploaded interim result fixture is absent")
        report = audit_results(read_results(path))
        case = next(c for c in report["cases"] if c["seed"] == 42)
        self.assertAlmostEqual(case["pre_unlearning"]["asr"], 69.76666666666667)
        self.assertEqual(case["known_attacker_target_score"], 0)
        self.assertLess(case["request_score"], case["benign_request_score"])
        detected = case["arms"]["detected"]
        self.assertEqual(detected["applied_rounds"], 0)
        self.assertTrue(detected["same_model_as_none"])
        oracle = case["arms"]["oracle"]
        self.assertEqual((oracle["applied_rounds"], oracle["guard_rejected_rounds"]), (39, 1))
        self.assertEqual(oracle["asr"], 100)
        # Older reports have no counters: do not fabricate successful checks.
        self.assertIsNone(oracle["asr_count_consistent"])


class SuiteTests(unittest.TestCase):
    def test_actual_training_fu_all_arms_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "smoke"
            command = [sys.executable, "-m", "experiments.request_purify", "--dataset", "smoke",
                       "--device", "cpu", "--out", str(root), "--seeds", "7",
                       "--calibration-seeds", "17", "--rounds", "3", "--local-epochs", "1",
                       "--fu-epochs", "1", "--post-rounds", "1", "--batch-size", "32",
                       "--eval-every", "3"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["runs"]), 10)
            self.assertTrue(all(x["diagnostic_fu_runs"] == 0 for x in summary["runs"]))
            check = json.loads((root / "eval-attack-7/zero_strength_check.json").read_text())
            self.assertTrue(check["exact_state_match"])
            self.assertTrue((root / "results_to_share.zip").is_file())
            before = (root / "eval-attack-7/detected/model.pt").stat().st_mtime_ns
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual(before, (root / "eval-attack-7/detected/model.pt").stat().st_mtime_ns)
            changed = subprocess.run(command + ["--resume", "--strength", ".7"], capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)

    def test_real_experiment_refuses_cpu(self):
        result = subprocess.run([sys.executable, "-m", "experiments.request_purify",
                                 "--dataset", "badfu", "--device", "cpu", "--check-only"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("require --device cuda", result.stderr)


if __name__ == "__main__":
    unittest.main()
