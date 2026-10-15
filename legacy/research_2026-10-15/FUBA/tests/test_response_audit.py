"""Mechanistic tests for the observable, rather than asserting attack detection."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_audit import cross_class_scores, median_reference, ResponseComparator
from structural_guard import Policy


class ResponseTests(unittest.TestCase):
    def fixture(self, delta=.08):
        ref = np.tile([[.65, .2, .15], [.2, .65, .15]], (32, 1))
        cand = ref.copy()
        cand[np.arange(len(ref)), ref.argmax(1)] -= delta
        cand[:, 2] += delta
        return ref, cand

    def test_probability_shift_precedes_any_argmax_flip(self):
        ref, cand = self.fixture()
        self.assertTrue(np.array_equal(ref.argmax(1), cand.argmax(1)))
        self.assertEqual(float(cross_class_scores(cand, ref, "hard").max()), 0)
        self.assertAlmostEqual(cross_class_scores(cand, ref)[2], .08)

    def test_one_source_specialization_is_not_multi_source_concentration(self):
        ref = np.tile([[.65, .2, .15]], (64, 1))
        cand = np.tile([[.05, .2, .75]], (64, 1))
        np.testing.assert_allclose(cross_class_scores(cand, ref), 0)

    def test_same_response_as_controls_cancels_at_reference(self):
        ref, cand = self.fixture()
        result = ResponseComparator()(ref, cand, [cand]*6, Policy())
        self.assertFalse(result["risk"])
        self.assertEqual(result["confirmation"]["score"], 0)

    def test_class_permutation_and_input_permutation(self):
        ref, cand = self.fixture()
        order = [2, 0, 1]
        np.testing.assert_allclose(cross_class_scores(cand[:, order], ref[:, order]), cross_class_scores(cand, ref)[order])
        np.testing.assert_allclose(cross_class_scores(cand[::-1], ref[::-1]), cross_class_scores(cand, ref))

    def test_target_selected_on_discovery_does_not_switch_on_confirmation(self):
        ref, cand = self.fixture()
        idx = np.random.default_rng(20261014).permutation(64)[32:]
        cand[idx] = ref[idx]
        output = ResponseComparator()(ref, cand, [ref]*5, Policy())
        self.assertTrue(output["discovery"]["passes"])
        self.assertEqual(output["target"], 2)
        self.assertFalse(output["risk"])

    def test_soft_review_can_hold_without_any_cancelling_partner(self):
        ref, cand = self.fixture()
        result = ResponseComparator()(ref, cand, [ref]*5, Policy())
        self.assertTrue(result["risk"])
        self.assertFalse(ResponseComparator("hard")(ref, cand, [ref]*5, Policy())["risk"])

    def test_median_reference_zero_mass_falls_back_to_mean(self):
        refs = [np.eye(3)[[i]] for i in range(3)]
        np.testing.assert_allclose(median_reference(refs), [[1/3]*3])

    def test_sparse_inputs_and_controls_remain_unknown(self):
        ref, cand = self.fixture()
        self.assertTrue(ResponseComparator()(ref[:4], cand[:4], [ref[:4]]*5, Policy())["inconclusive"])
        self.assertTrue(ResponseComparator()(ref, cand, [ref]*2, Policy())["inconclusive"])

    def test_binary_task_is_unsupported_not_evidence_of_safety(self):
        ref = np.tile([[.8, .2]], (64, 1))
        result = ResponseComparator()(ref, ref[:, ::-1], [ref]*5, Policy())
        self.assertTrue(result["inconclusive"])
        self.assertEqual(result["reason"], "requires_at_least_three_classes")

    def test_cli_routes_response_into_transactional_guard(self):
        from run_structural_guard import main
        ref, cand = self.fixture()
        class Backend:
            units = tuple(str(i) for i in range(8))
            views = ("public", "request")
            def propose(self, removal):
                return dict(removal)
            def predict(self, removal, view):
                return cand if "0" in removal else ref
        # Explanation is optional; test the detector and deployment boundary.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with patch("run_structural_guard.load_backend", return_value=Backend()), \
                 patch("existing_guard_components.trace_attributor", return_value=[]):
                result = main(["audit", "--fmt", "fuba", "--name", "fixture", "--K", "8", "--rounds", "2",
                               "--probes", "unused.npz", "--requesters", "0", "--fraction", ".1",
                               "--components", "response", "--out", str(Path(tmp)/"audit.json")])
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["committed"], {})
        self.assertTrue(result["steps"][0]["views"]["public"]["risk"])


if __name__ == "__main__":
    unittest.main()
