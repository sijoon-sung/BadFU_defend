"""Signal semantics, side-effect isolation and the shareable GPU-suite interface."""
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from . import engine
from .__main__ import parser
from .core import DetectorConfig
from .data import smoke_case
from .observations import (ProbeObserver, apply_policy, candidate_scores,
                           clean_head_gradient, fit_policy, remove_clean_axis)


class BNFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(8, 12), nn.BatchNorm1d(12), nn.Tanh())
        self.fc = nn.Linear(12, 3)

    def forward(self, x):
        return self.fc(self.features(x))


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_clean_axis_exposes_hidden_component_without_forcing_opposition(self):
        updates = torch.tensor([[[10., 1.]], [[10., -1.]]])
        self.assertGreater(float(nn.functional.cosine_similarity(updates[0], updates[1])), .98)
        residual = remove_clean_axis(updates, torch.tensor([[1., 0.]]))
        torch.testing.assert_close(residual, torch.tensor([[[0., 1.]], [[0., -1.]]]))
        same_direction = torch.tensor([[[10., 1.]], [[20., 2.]]])
        residual = remove_clean_axis(same_direction, torch.tensor([[1., 0.]]))
        self.assertAlmostEqual(float(nn.functional.cosine_similarity(residual[0], residual[1])), 1.)
        torch.testing.assert_close(remove_clean_axis(updates, torch.zeros(1, 2)), updates)

    def test_clean_head_gradient_matches_autograd(self):
        gen = torch.Generator().manual_seed(3)
        features = torch.randn(9, 5, generator=gen)
        layer = nn.Linear(5, 3)
        labels = torch.arange(9) % 3
        logits = layer(features)
        expected = clean_head_gradient(logits.detach(), features, labels)
        nn.functional.cross_entropy(logits, labels).backward()
        actual = torch.cat((layer.weight.grad, layer.bias.grad[:, None]), 1)
        torch.testing.assert_close(expected, actual)

    def test_bn_ablation_is_observable_but_cannot_mutate_training_state_or_rng(self):
        case = smoke_case(7, False)
        model = BNFixture().train()
        original = engine.cpu_state(model)
        rng = torch.get_rng_state().clone()
        args = SimpleNamespace(probe_size=8, probe_seed=123, probe_dimensions=4, device="cpu", batch_size=4)
        observer = ProbeObserver(model, case["validation"], "fc", args)
        local = {k: v.clone() for k, v in original.items()}
        local["features.1.running_mean"] += 3
        normal = observer.snapshot(original)
        changed = observer.snapshot(local)
        controlled = observer.snapshot(local, original)
        self.assertFalse(torch.equal(normal["margin"], changed["margin"]))
        torch.testing.assert_close(normal["margin"], controlled["margin"])
        self.assertEqual(engine.state_hash(original), engine.state_hash(engine.cpu_state(model)))
        self.assertTrue(model.training)
        torch.testing.assert_close(rng, torch.get_rng_state())
        self.assertEqual(observer.manifest["optimizer_steps"], 0)

    def test_shadow_observation_keeps_training_fu_and_purification_bitwise_identical(self):
        args = parser().parse_args(["--dataset", "smoke", "--device", "cpu", "--rounds", "3",
                                  "--local-epochs", "1", "--fu-epochs", "1", "--post-rounds", "0",
                                  "--probe-size", "8", "--probe-dimensions", "4", "--no-pretrained"])
        case = smoke_case(7, False)
        cfg = DetectorConfig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states, histories, call_counts = {}, {}, {}
            for observed in (False, True):
                args.observe = observed
                with patch.object(engine, "train_local", wraps=engine.train_local) as train:
                    history = engine.train_history(case, args, root / str(observed))
                    histories[observed] = history
                    states[observed] = {}
                    for arm in ("none", "oracle"):
                        result, state = engine.run_arm(history, case, args, root / str(observed) / arm, cfg, arm)
                        states[observed][arm] = engine.state_hash(state)
                        self.assertEqual(result["diagnostic_fu_runs"], 0)
                        if arm == "oracle":
                            self.assertTrue(any(e["applied_norm"] > 0 for e in result["events"]))
                    call_counts[observed] = train.call_count
            self.assertEqual(states[False], states[True])
            self.assertNotEqual(states[False]["none"], states[False]["oracle"])
            self.assertEqual(engine.state_hash(histories[False]["final"]), engine.state_hash(histories[True]["final"]))
            self.assertEqual(call_counts, {False: 48, True: 48})  # 3*6 FL + 2*3*5 actual FU.
            for key in histories[True]["observations"]["features"]:
                self.assertTrue(torch.isfinite(histories[True]["observations"]["features"][key]).all())
            args.observe = True
            normal = engine.train_history(smoke_case(17, True), args, root / "normal")
            policy = fit_policy([normal], cfg)
            report = apply_policy(histories[True], cfg, policy)
            self.assertEqual(len(report["candidates"]), 8)
            scores = candidate_scores(histories[True], cfg)
            for name in ("margin_magnitude", "representation_magnitude"):
                self.assertEqual(len({q["score"] for q in scores[name]["requests"]}), 1)
            with self.assertRaises(ValueError):
                fit_policy([histories[True]], cfg)
            with self.assertRaises(ValueError):
                apply_policy(normal, cfg, policy)
            # BN-free models must not gain a fictitious extra signal from the control.
            self.assertEqual(scores["margin_pair"], scores["margin_bn_pair"])

    def test_cli_exports_scores_curves_and_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "suite"
            command = [sys.executable, "-m", "experiments.request_purify", "--dataset", "smoke",
                       "--device", "cpu", "--observe", "--probe-size", "8", "--probe-dimensions", "4",
                       "--out", str(root), "--seeds", "7", "--calibration-seeds", "17", "--rounds", "3",
                       "--local-epochs", "1", "--fu-epochs", "1", "--post-rounds", "0", "--arms", "none", "zero"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            overview = json.loads((root / "observation_summary.json").read_text())
            self.assertEqual(len(overview["groups"]), 16)
            self.assertFalse(overview["controls_purification"])
            check = json.loads((root / "eval-attack-7/zero_strength_check.json").read_text())
            self.assertTrue(check["exact_state_match"])
            events = json.loads((root / "eval-attack-7/none/fu_observations.json").read_text())["events"]
            self.assertEqual(len(events), 3)
            self.assertTrue(all(event["alarm"] is None for event in events))
            self.assertTrue(all(5 not in event["retained_ids"] for event in events))
            with zipfile.ZipFile(root / "results_to_share.zip") as archive:
                self.assertIn("observation_scores.csv", archive.namelist())
                self.assertIn("fu_observation_curves.csv", archive.namelist())
                self.assertFalse(any(name.endswith(".pt") for name in archive.namelist()))
            saved = root / "eval-attack-7/history.pt"
            before = saved.stat().st_mtime_ns
            result = subprocess.run(command + ["--resume"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(before, saved.stat().st_mtime_ns)
            result = subprocess.run(command + ["--resume", "--probe-seed", "123"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Resume configuration", result.stderr)


if __name__ == "__main__":
    unittest.main()
