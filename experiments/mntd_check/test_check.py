import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from .__main__ import Detector, auc, inspect, suite_inputs
from experiments.request_purify.engine import cpu_state, save_json, save_torch, state_hash


class CheckTests(unittest.TestCase):
    def test_auc_ties_and_direction(self):
        self.assertEqual(auc([0, 1], [2., 3.]), 1.)
        self.assertEqual(auc([0, 1], [3., 2.]), 0.)
        self.assertEqual(auc([0, 1], [3., 3.]), .5)
        self.assertIsNone(auc([0, 0], [2., 3.]))

    def test_query_learning_changes_queries_without_training_target_or_bn(self):
        torch.manual_seed(7)
        model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 3)).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        original = cpu_state(model)
        detector = Detector(queries=2, shape=(4,), classes=3)
        before = detector.queries.detach().clone()
        optimizer = torch.optim.Adam(detector.parameters(), lr=.01)
        loss = nn.functional.binary_cross_entropy_with_logits(detector(model(detector.queries)), torch.tensor(1.))
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before, detector.queries))
        self.assertTrue(all(param.grad is None for param in model.parameters()))
        self.assertEqual(state_hash(original), state_hash(cpu_state(model)))
        value = inspect(detector, model, original)
        self.assertIsInstance(value, float)
        self.assertEqual(state_hash(original), state_hash(cpu_state(model)))

    def test_four_checkpoints_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for benign in (True, False):
                folder = root / f"eval-{'benign' if benign else 'attack'}-42"
                (folder / "none").mkdir(parents=True)
                state = {"weight": torch.tensor([float(benign)])}
                identity = f"trajectory-{benign}"
                meta = {"dataset": "badfu", "seed": 42, "benign": benign,
                        "trajectory_id": identity, "validation_ids": [5, 6],
                        "partitions": [[0], [1]], "bd_ids": [2], "cv_ids": [3],
                        "clean_train_sha256": "clean", "record_sha256": "record", "requester": 5}
                save_json(folder / "manifest.json", meta)
                save_torch(folder / "history.pt", {"manifest": meta, "final": state,
                           "training_metrics": [{"acc": 80., "asr": 1.}]})
                save_torch(folder / "none/model.pt", state)
                save_json(folder / "none/result.json", {"trajectory_id": identity,
                          "model_sha256": state_hash(state), "final": {"acc": 80., "asr": 99.}})
            conditions, _ = suite_inputs(root, 42)
            self.assertEqual([r["condition"] for r in conditions],
                             ["normal_before", "normal_after", "attack_before", "attack_after"])
            self.assertEqual([r["metrics"]["asr"] for r in conditions], [1., 99., 1., 99.])
            save_torch(root / "eval-attack-42/none/model.pt", {"weight": torch.ones(1) * 100})
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                suite_inputs(root, 42)

    def test_missing_models_are_not_replaced_with_synthetic_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "artifact"):
                suite_inputs(directory, 42)


if __name__ == "__main__":
    unittest.main()
