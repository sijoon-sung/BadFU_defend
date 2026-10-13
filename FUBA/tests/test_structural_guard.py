"""Behavioral tests: counterexamples to attack-specific assumptions and unsafe release."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structural_guard import (CancellationMonitor, Policy, StructuralGuard, Verification,
                              compare, features, matched_controls)
from run_structural_guard import main, replay_cache


class ToyBackend:
    """Known causal logits; validation evidence here is synthetic, not an ASR experiment."""

    def __init__(self, carriers=1, suppressors=1, localized=False, missing=False):
        self.units = tuple(str(i) for i in range(12))
        self.views = ("public",) if missing else ("public", "request")
        base = np.zeros((36, 3))
        base[np.arange(36), np.arange(36) % 2 + 1] = 2
        self.base = base
        direction = np.zeros_like(base)
        direction[:, 0] = 8
        self.effects = {}
        for view in self.views:
            e = {u: np.zeros_like(base) for u in self.units}
            if not (localized and view == "public"):
                for i in range(carriers):
                    e[str(i)] = direction / carriers
                for i in range(suppressors):
                    e[str(12 - i - 1)] = -direction / suppressors
            self.effects[view] = e

    def propose(self, removals):
        return dict(removals)

    def predict(self, model, view):
        if model.get("retrained"):
            logits = self.base.copy()
        else:
            logits = self.base - sum((amount * self.effects[view][u] for u, amount in model.items()),
                                    np.zeros_like(self.base))
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)


def synthetic_verifier(model, removals, suspects):
    return Verification(True, True, True, "synthetic fixture checks only")


class StructuralGuardTests(unittest.TestCase):
    def test_fuba_like_distributed_carriers(self):
        backend = ToyBackend(carriers=4)
        guard = StructuralGuard(backend)
        result = guard.request({"11": 1.0})
        self.assertEqual(result.status, "held")
        self.assertEqual(guard.committed, {})
        self.assertEqual(guard.serving_model, {})
        self.assertLess(result.steps[-1]["removals"]["11"], 1)
        self.assertTrue(result.steps[-1]["views"]["public"]["risk"])

    def test_badfu_like_single_carrier_hidden_from_public(self):
        backend = ToyBackend(localized=True)
        result = StructuralGuard(backend).request({"11": 1})
        self.assertEqual(result.status, "held")
        self.assertFalse(result.steps[-1]["views"]["public"]["risk"])
        self.assertTrue(result.steps[-1]["views"]["request"]["risk"])
        self.assertEqual([s["unit"] for s in result.suspects], ["0"])

    def test_multiple_suppressors_are_one_request(self):
        result = StructuralGuard(ToyBackend(carriers=3, suppressors=2, localized=True)).request({"10": 1, "11": 1})
        self.assertEqual(result.status, "held")
        self.assertEqual(set(result.requested), {"10", "11"})

    def test_missing_request_view_is_unknown_not_safe(self):
        result = StructuralGuard(ToyBackend(localized=True, missing=True)).request({"11": 1})
        self.assertEqual(result.status, "inconclusive")
        self.assertIn("request", result.reason)

    def test_safe_probe_result_cannot_certify_forgetting(self):
        guard = StructuralGuard(ToyBackend())
        result = guard.request({"5": 1})
        self.assertEqual(result.status, "audited_only")
        self.assertEqual(result.committed, {})
        self.assertEqual(guard.serving_model, {})

    def test_failed_independent_check_blocks_release(self):
        guard = StructuralGuard(ToyBackend(), verify=lambda *args: Verification(False, True, True, "forget check failed"))
        result = guard.request({"5": 1})
        self.assertEqual(result.status, "held")
        self.assertEqual(guard.committed, {})

    def test_split_requests_use_original_anchor_and_cumulative_fraction(self):
        guard = StructuralGuard(ToyBackend(localized=True), verify=synthetic_verifier)
        statuses = []
        for _ in range(20):
            previous_model = copy.deepcopy(guard.serving_model)
            previous_removal = dict(guard.committed)
            result = guard.request({"11": 0.01})
            statuses.append(result.status)
            if result.status == "held":
                self.assertEqual(guard.serving_model, previous_model)
                self.assertEqual(guard.committed, previous_removal)
                self.assertGreater(result.steps[-1]["removals"]["11"], 0.01)
                break
        self.assertIn("released", statuses)
        self.assertEqual(statuses[-1], "held")

    def test_recovery_honors_full_request_and_needs_reaudit(self):
        calls = []
        def recover(removals, suspects):
            calls.append((removals, suspects))
            return {"retrained": True}
        guard = StructuralGuard(ToyBackend(localized=True), verify=synthetic_verifier, recover=recover)
        result = guard.request({"11": 1})
        self.assertEqual(calls, [({"11": 1.0}, ("0",))])
        self.assertEqual(result.status, "recovered")
        self.assertEqual(guard.committed, {"11": 1})
        self.assertEqual(guard.serving_model, {"retrained": True})
        # Old contribution proxies cannot be reused after reconstructing the model.
        self.assertEqual(guard.request({"4": 1}).reason, "refresh_backend_after_recovery")

    def test_recovery_that_preserves_activation_is_held(self):
        backend = ToyBackend(localized=True)
        guard = StructuralGuard(backend, verify=synthetic_verifier, recover=lambda removals, suspects: removals)
        result = guard.request({"11": 1})
        self.assertEqual(result.status, "held")
        self.assertEqual(result.reason, "recovery_failed_audit")
        self.assertEqual(guard.serving_model, {})

    def test_data_shards_from_same_client_are_supported(self):
        backend = ToyBackend(localized=True)
        rename = {u: f"client0/shard{u}" for u in backend.units}
        backend.units = tuple(rename.values())
        backend.effects = {v: {rename[u]: e for u, e in effects.items()} for v, effects in backend.effects.items()}
        result = StructuralGuard(backend).request({"client0/shard11": 1})
        self.assertEqual(result.status, "held")
        self.assertEqual(result.suspects[0]["unit"], "client0/shard0")

    def test_benign_comparable_non_iid_effect_not_an_outlier(self):
        backend = ToyBackend()
        # Every matched deletion has the same modest, class-specific effect.
        local = np.zeros_like(backend.base)
        local[::6, 1] = 0.2
        for view in backend.views:
            backend.effects[view] = {u: local for u in backend.units}
        result = StructuralGuard(backend, verify=synthetic_verifier).request({"5": 1})
        self.assertEqual(result.status, "released")

    def test_request_overflow_and_unknown_unit_fail_before_mutation(self):
        guard = StructuralGuard(ToyBackend(), verify=synthetic_verifier)
        self.assertEqual(guard.request({"5": 1}).status, "released")
        for request in ({"5": 0.1}, {"unknown": 1}, {"4": float("nan")}, {"4": -1}, {}):
            with self.assertRaises(ValueError):
                guard.request(request)
        self.assertEqual(guard.committed, {"5": 1})

    def test_not_enough_independent_controls_is_unknown(self):
        backend = ToyBackend()
        backend.units = ("0", "1", "11")
        result = StructuralGuard(backend).request({"11": 1})
        self.assertEqual(result.status, "inconclusive")

    def test_multi_target_changes_with_zero_class_mean_are_visible(self):
        before = np.tile([0.8, 0.1, 0.1], (20, 1))
        before[10:] = [0.1, 0.8, 0.1]
        after = before[::-1].copy()
        self.assertTrue(np.allclose(before.mean(0), after.mean(0)))
        result = compare(before, after, [before] * 4, Policy())
        self.assertTrue(result["risk"])
        self.assertIn("tv_tail", result["flagged"])

    def test_invalid_probabilities_are_not_silently_accepted(self):
        before = np.array([[0.4, 0.6]])
        for after in (np.array([[np.nan, 0]]), np.array([[2, -1]]), np.array([[0.1, 0.2]]), np.empty((0, 2))):
            with self.assertRaises(ValueError):
                features(before, after)

    def test_control_groups_match_entire_request_without_duplicates(self):
        controls = matched_controls(tuple("abcdef"), {"a": 0.2, "b": 0.2}, Policy())
        self.assertEqual(len(controls), 6)
        for group in controls:
            self.assertFalse(set(group) & {"a", "b"})
            self.assertEqual(sorted(group.values()), [0.2, 0.2])
        mixed = matched_controls(tuple("abcdef"), {"a": 0.2, "b": 0.8}, Policy())
        self.assertEqual(len(mixed), 12)


class MonitorTests(unittest.TestCase):
    def test_single_pair_is_a_candidate_not_automatic_exclusion(self):
        monitor = CancellationMonitor(persist=2)
        self.assertEqual(monitor.observe(1, {"s": [-2, 0], "a": [2, 0]}), [])
        result = monitor.observe(2, {"s": [-2, 0], "a": [2, 0]})
        self.assertEqual(result[0]["units"], ["a", "s"])
        self.assertEqual(result[0]["action"], "audit_priority")

    def test_absence_and_skipped_rounds_do_not_extend_streak(self):
        monitor = CancellationMonitor(persist=2)
        monitor.observe(1, {"s": [-2], "a": [2]})
        self.assertEqual(monitor.observe(2, {}), [])
        self.assertEqual(monitor.observe(3, {"s": [-2], "a": [2]}), [])
        self.assertEqual(monitor.observe(5, {"s": [-2], "a": [2]}), [])
        self.assertTrue(monitor.observe(6, {"s": [-2], "a": [2]}))

    def test_per_input_effects_survive_mean_cancellation(self):
        monitor = CancellationMonitor(persist=1)
        vector = np.array([[2., -2], [-2, 2]])
        self.assertTrue(monitor.observe(1, {"s": vector, "a": -vector}))

    def test_partial_participation_can_provide_fresh_evidence(self):
        monitor = CancellationMonitor(persist=2, threshold=0.6)
        monitor.observe(1, {"s": [-2], "a": [2]})
        result = monitor.observe(2, {"s": [-1]})
        self.assertTrue(result)
        self.assertEqual(result[0]["units"], ["a", "s"])

    def test_cache_cli_and_invalid_events(self):
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(test_root))
            cache, out = Path(directory) / "push.npz", Path(directory) / "out.json"
            np.savez(cache, K=2, T=3, ts=[1, 1, 2, 2, 3, 3], js=[0, 1, 0, 1, 0, 1],
                     vs=[[-2, 0], [2, 0]] * 3)
            main(["monitor", "--cache", str(cache), "--screen", "structural", "--out", str(out)])
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["n_candidates"], 1)
            self.assertEqual(result["action"], "request_audit_required")
            np.savez(cache, K=2, T=1, ts=[1, 1], js=[0, 0], vs=[[-2], [2]])
            with self.assertRaises(ValueError):
                replay_cache(cache)


class TorchAdapterTests(unittest.TestCase):
    def torch_or_skip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable; NumPy policy tests still run")
        return torch

    def test_existing_dose_and_trace_are_callable_components(self):
        torch = self.torch_or_skip()
        from existing_guard_components import DoseComparator, trace_attributor
        backend = ToyBackend(localized=True)
        backend.factory, backend.flat = None, None
        backend.contributions = [torch.zeros(1) for _ in backend.units]
        backend.probes, backend.device = {"public": torch.zeros(36, 1)}, "cpu"
        comparator = DoseComparator(backend, "fuba", stat="flow")
        before = backend.predict({}, "request")
        activated = backend.predict({"11": 1}, "request")
        result = comparator(before, activated, [before] * 4, Policy())
        self.assertTrue(result["risk"])
        self.assertEqual(result["method"], "dose_fu.Engine.check")
        self.assertFalse(comparator(before, before, [before] * 4, Policy())["risk"])
        anchor = {v: backend.predict({}, v) for v in backend.views}
        suspects = trace_attributor(backend, anchor, {"11": 1}, {"11": 1}, Policy())
        self.assertEqual([s["unit"] for s in suspects], ["0"])
        self.assertEqual(suspects[0]["method"], "cancel_fu.forward_select")

    def test_existing_components_run_through_common_release_controller(self):
        torch = self.torch_or_skip()
        from existing_guard_components import DoseComparator, trace_attributor
        backend = ToyBackend(localized=True)
        backend.factory, backend.flat = None, None
        backend.contributions = [torch.zeros(1) for _ in backend.units]
        backend.probes, backend.device = {"public": torch.zeros(36, 1)}, "cpu"
        guard = StructuralGuard(backend, comparator=DoseComparator(backend, "fuba"), attributor=trace_attributor)
        result = guard.request({"11": 1})
        self.assertEqual(result.status, "held")
        self.assertEqual(result.suspects[0]["unit"], "0")
        self.assertEqual(guard.serving_model, {})

    def test_model_adapter_matches_direct_functional_prediction(self):
        torch = self.torch_or_skip()
        from run_structural_guard import ContributionBackend
        class TinyFlat:
            def to_sd(self, value):
                return {"weight": value.reshape(3, 2)}
        factory = lambda: torch.nn.Linear(2, 3, bias=False)
        anchor = torch.tensor([2., 0., 0., 2., 0., 0.])
        u = torch.tensor([1., 0., 0., 0., 0., 0.])
        xs = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        backend = ContributionBackend(factory, TinyFlat(), anchor, [u] * 5,
                                      {"public": xs, "request": xs}, batch_size=1)
        model = backend.propose({"0": 0.2, "1": 0.3})
        actual = backend.predict(model, "public")
        expected = torch.softmax(torch.from_numpy(xs) @ (anchor - 0.5 * u).reshape(3, 2).T, dim=1).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        torch.testing.assert_close(backend.anchor, anchor)

    def test_fuba_checkpoint_adapter_preserves_participation_weights(self):
        torch = self.torch_or_skip()
        import os
        import pickle
        from types import SimpleNamespace
        from run_structural_guard import load_backend
        cwd = Path.cwd()
        try:
            import release_map
        finally:
            os.chdir(cwd)
        factory = lambda: torch.nn.Linear(2, 3, bias=False)
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(test_root))
            state = {"weight": torch.zeros(3, 2)}
            final = {"weight": torch.full((3, 2), 0.375)}
            local = [{"weight": torch.full((3, 2), value)} for value in (0.1, 0.2, 0.4, 0.8)] + [final]
            for kind, round_id, value in (("global", 1, state), ("global", 2, final), ("local", 2, local)):
                with (root / f"{kind}_net_model_fixture_round_{round_id}.pkl").open("wb") as handle:
                    pickle.dump(value, handle)
            np.savez(root / "probes.npz", public=np.ones((4, 2)), request=np.ones((4, 2)))
            args = SimpleNamespace(fmt="fuba", name="fixture", rounds=2, K=5, ckpt_dir=str(root),
                                   probes=str(root / "probes.npz"), device="cpu", batch_size=2)
            with patch.object(release_map, "_fuba_factory", return_value=(factory, None)):
                backend = load_backend(args)
            torch.testing.assert_close(backend.contributions[0], torch.full((6,), 0.025))
            torch.testing.assert_close(backend.contributions[4], torch.zeros(6))
            self.assertEqual(backend.views, ("public", "request"))
            self.assertEqual(Path.cwd(), cwd)

    def test_badfu_checkpoint_adapter_uses_sample_counts(self):
        torch = self.torch_or_skip()
        import os
        from types import ModuleType, SimpleNamespace
        from run_structural_guard import load_backend
        cwd = Path.cwd()
        try:
            import release_map  # Load native libraries outside the temporary module patch.
        finally:
            os.chdir(cwd)
        model_module = ModuleType("badfu_common")
        model_module.ResNet18 = lambda pretrained=False: torch.nn.Linear(2, 3, bias=False)
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(test_root))
            torch.save({"gm": {"weight": torch.zeros(3, 2)},
                        "cms": [{"weight": torch.full((3, 2), float(i))} for i in (1, 2, 3)]}, root / "round_0.pt")
            torch.save({"gm_after": {"weight": torch.full((3, 2), 14/6)}}, root / "glob_0.pt")
            np.savez(root / "probes.npz", public=np.ones((4, 2)), request=np.ones((4, 2)))
            args = SimpleNamespace(fmt="badfu", name="fixture", rounds=1, K=3, traj_dir=str(root),
                                   counts=[1, 2, 3], uniform_weights=False, probes=str(root / "probes.npz"),
                                   device="cpu", batch_size=2)
            with patch.dict(sys.modules, {"badfu_common": model_module}):
                backend = load_backend(args)
            torch.testing.assert_close(backend.contributions[0], torch.full((6,), 1/6))
            torch.testing.assert_close(backend.contributions[2], torch.full((6,), 1.5))


if __name__ == "__main__":
    unittest.main()
