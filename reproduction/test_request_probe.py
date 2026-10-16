"""CPU checks of isolation, weighting, gradients, and explicit limitations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reproduction.request_probe import SearchConfig, evaluate, freeze, margin, search, split_pool
from reproduction.run_request_probe import DemoModel, build_candidate, run_experiment
from reproduction.trajectory import Recorder, json_write


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp_root = Path(__file__).resolve().parents[1] / "artifacts" / "request_probe_tests"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        rng = np.random.default_rng(100)
        pixels = rng.integers(60, 190, (16, 8, 8, 1), dtype=np.uint8)
        pixels[:, 5:, 5:] = 134
        self.pixels = pixels
        self.ids = np.array([str(i) for i in range(16)])
        self.x = torch.from_numpy(pixels).permute(0, 3, 1, 2).float()/255
        self.pre, self.post = freeze(DemoModel(0), "cpu"), freeze(DemoModel(1), "cpu")
        self.config = SearchConfig(epsilon=.12, steps=8, step_size=.02, restarts=1, batch_size=4)

    def test_margin_common_shift_invariance(self):
        logits = torch.tensor([[1., 3., -1.], [2., -5., 1.]])
        torch.testing.assert_close(margin(logits, 1), margin(logits + 100, 1))

    def test_zero_request_has_zero_release(self):
        result = search(self.pre, self.pre, self.x[:8], "request_guided", self.config)
        self.assertEqual(result["search_objective"], 0.)
        metrics, _ = evaluate(self.pre, self.pre, self.x[8:], result, 4)
        self.assertEqual(metrics["release_margin_mean"], 0.)
        self.assertEqual(metrics["excess_marginal_entropy_drop"], 0.)

    def test_flat_hidden_trigger_is_explicit_search_limitation(self):
        # No gradient at the zero start: failure must not be called a benign verdict.
        result = search(self.pre, DemoModel(1, threshold=.99), self.x[:8],
                        "request_guided", self.config)
        self.assertEqual(result["search_objective"], 0.)

    def test_known_release_generalizes_with_bounded_shared_delta(self):
        result = search(self.pre, self.post, self.x[:8], "request_guided", self.config)
        self.assertEqual(result["target"], 2)
        self.assertEqual(tuple(result["delta"].shape), (1, 1, 8, 8))
        self.assertLessEqual(float(result["delta"].abs().max()), .120001)
        metrics, _ = evaluate(self.pre, self.post, self.x[8:], result, 4)
        self.assertGreater(metrics["release_margin_mean"], 2.)
        self.assertIsNone(metrics["attack_success_rate"])
        self.assertEqual(metrics["clean_prediction_disagreement"], 0.)

    def test_batch_size_does_not_change_search(self):
        small = search(self.pre, self.post, self.x, "request_guided", self.config)
        large = search(self.pre, self.post, self.x, "request_guided",
                       SearchConfig(.12, 8, .02, 1, 16, 42))
        self.assertEqual(small["target"], large["target"])
        torch.testing.assert_close(small["delta"], large["delta"])

    def test_history_free_ignores_candidate_behavior(self):
        first = search(self.pre, self.post, self.x, "history_free", self.config)
        second = search(self.pre, DemoModel(100), self.x, "history_free", self.config)
        self.assertEqual(first["target"], second["target"])
        self.assertEqual(first["search_objective"], second["search_objective"])
        torch.testing.assert_close(first["delta"], second["delta"])

    def test_split_is_disjoint_and_rejects_pixel_leakage(self):
        a, b = split_pool(self.pixels, self.ids, 8, 8, 42)
        self.assertFalse(set(a) & set(b))
        bad = self.pixels.copy()
        bad[b[0]] = bad[a[0]]
        with self.assertRaisesRegex(ValueError, "identical pixels"):
            split_pool(bad, self.ids, 8, 8, 42)
        with self.assertRaises(ValueError):
            split_pool(self.pixels, self.ids, 10, 10, 42)

    def test_heldout_and_actual_model_do_not_affect_selection(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            provenance = dict(device="cpu", candidate_kind="synthetic_test")
            first = run_experiment(self.pre, self.post, self.pixels, self.ids, root/"a",
                self.config, 8, 8, ["request_guided"], provenance)
            changed = self.pixels.copy()
            _, b = split_pool(changed, self.ids, 8, 8, 42)
            changed[b] = 255 - changed[b]
            second = run_experiment(self.pre, self.post, changed, self.ids, root/"b",
                self.config, 8, 8, ["request_guided"], provenance, DemoModel(100))
            for key in ("target", "restart", "step", "search_objective"):
                self.assertEqual(first["request_guided"][key], second["request_guided"][key])
            with np.load(root/"a/observations.npz") as a, np.load(root/"b/observations.npz") as b:
                np.testing.assert_array_equal(a["request_guided__delta"], b["request_guided__delta"])
            protocol = json.loads((root/"b/protocol.json").read_text())
            self.assertEqual(protocol["status"], "complete")
            self.assertTrue(protocol["actual_fu_checkpoint_evaluated"])
            with self.assertRaises(FileExistsError):
                run_experiment(self.pre, self.post, self.pixels, self.ids, root/"a",
                    self.config, 8, 8, ["request_guided"], provenance)

    def test_full_trajectory_uses_actual_participation_weights(self):
        from FUBA.model import Net
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            model = Net()
            initial = copy.deepcopy(model.state_dict())
            recorder = Recorder(root/"fl", initial, 2,
                dict(attack="fuba", dataset="mnist", num_clients=3, aggregation="fuba_uniform"))
            before = initial
            for ids in ([0, 1], [0, 2]):
                local = [{k: v + (.1 if i == 1 else .02) for k, v in before.items()} for i in ids]
                after = {k: torch.stack([s[k] for s in local]).mean(0) for k in before}
                recorder.append(before, local, after, ids, [.5, .5])
                before = after
            recorder.complete()
            json_write(root/"protocol.json", dict(status="complete",
                options=dict(attack="fuba", dataset="mnist", clients=3)))
            pre, candidate, meta = build_candidate(root, [1], 1.)
            for name, p in pre.named_parameters():
                torch.testing.assert_close(p - dict(candidate.named_parameters())[name],
                    torch.full_like(p, .05), atol=1e-7, rtol=1e-5)
            self.assertEqual(meta["participation"], {1: 1})
            with self.assertRaises(ValueError):
                build_candidate(root, [3], 1.)
            with self.assertRaises(ValueError):
                build_candidate(root, [1, 1], 1.)
            images = np.random.default_rng(3).integers(0, 256, (4, 28, 28, 1), dtype=np.uint8)
            rows = run_experiment(pre, candidate, images, np.array([str(i) for i in range(4)]),
                root/"mnist_probe", SearchConfig(steps=1, restarts=1, batch_size=2),
                2, 2, ["request_guided"], dict(device="cpu", candidate_kind="synthetic_trajectory"), pre)
            self.assertEqual(rows["request_guided"]["actual_fu_evaluation"]["release_margin_mean"], 0.)

    def test_candidate_preserves_final_bn_buffers(self):
        tiny = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        initial = copy.deepcopy(tiny.state_dict())
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            recorder = Recorder(root/"fl", initial, 1,
                dict(attack="fuba", dataset="mnist", num_clients=2, aggregation="fuba_uniform"))
            local = [{k: v + offset if v.is_floating_point() else v.clone()
                      for k, v in initial.items()} for offset in (.1, .2)]
            final = {k: torch.stack([s[k].float() for s in local]).mean(0).to(v.dtype)
                     for k, v in initial.items()}
            recorder.append(initial, local, final, [0, 1], [.5, .5])
            recorder.complete()
            json_write(root/"protocol.json", dict(status="complete",
                options=dict(attack="fuba", dataset="mnist", clients=2)))
            with patch("FUBA.model.Net", return_value=copy.deepcopy(tiny)):
                pre, candidate, _ = build_candidate(root, [1], 1.)
            for name, value in candidate.named_buffers():
                torch.testing.assert_close(value, dict(pre.named_buffers())[name], rtol=0, atol=0)
            for name, value in candidate.named_parameters():
                torch.testing.assert_close(dict(pre.named_parameters())[name] - value,
                                           torch.full_like(value, .1))


if __name__ == "__main__":
    unittest.main()
