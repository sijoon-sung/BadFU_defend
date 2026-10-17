"""Tests for the scientific meaning of the relation diagnostic, not detection power."""
import unittest

import numpy as np

from experiments.detection.history_relations import analyze_history_relations as rel


class HistoryRelationsTests(unittest.TestCase):
    def test_decomposition_preserves_observed_weighted_contributions(self):
        rng = np.random.default_rng(4)
        h = rng.normal(size=(12, 5, 7))
        mask = rng.random((12, 5)) > .4
        h[~mask] = 0
        for i in range(5):
            x, y, active = rel.decompose(h, mask, i)
            np.testing.assert_array_equal(x, h[:, i])
            np.testing.assert_allclose(x+y, h.sum(1), atol=1e-14)
            np.testing.assert_array_equal(active, mask[:, i] & mask[:, np.arange(5) != i].any(1))

    def test_selection_has_no_future_leakage(self):
        rng = np.random.default_rng(20)
        h = rng.normal(size=(20, 4, 6))
        mask = np.ones((20, 4), bool)
        before = rel.select_prefix(h, mask, 10)
        h[10:] = rng.normal(100, 100, h[10:].shape)
        after = rel.select_prefix(h, mask, 10)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_one_coordinate_is_frozen_when_the_largest_relation_moves(self):
        h = np.zeros((8, 2, 2))
        h[:4, 0, 0], h[:4, 1, 0] = -1, 1
        h[4:, 0, 1], h[4:, 1, 1] = -100, 100
        mask = np.ones((8, 2), bool)
        coords, _, _ = rel.select_prefix(h, mask, 4)
        self.assertEqual(int(coords[0]), 0)
        x, y, active = rel.decompose(h, mask, 0)
        result = rel.relation_summary(x[4:, coords[0]], y[4:, coords[0]], active[4:])
        self.assertEqual(result["forward_rounds"], 0)

    def test_missing_round_is_not_opposition_or_physical_continuity(self):
        x, y = np.array([-1., 0, -1, -1]), np.ones(4)
        active = np.array([True, False, True, True])
        result = rel.relation_summary(x, y, active)
        self.assertEqual(result["eligible_rounds"], 3)
        self.assertEqual(result["forward_fraction"], 1)
        self.assertEqual(result["forward_longest_physical"], 2)
        self.assertEqual(result["forward_longest_opportunities"], 3)

    def test_zero_or_unobserved_history_has_no_fake_fraction(self):
        z = np.zeros(4)
        result = rel.relation_summary(z, z, np.zeros(4, bool))
        self.assertIsNone(result["forward_fraction"])
        self.assertIsNone(result["forward_overlap_fraction"])
        self.assertFalse(result["window_cumulative_forward"])
        coords, scores, ties = rel.select_prefix(np.zeros((4, 2, 3)), np.ones((4, 2), bool), 2)
        np.testing.assert_array_equal(scores, [0, 0])
        np.testing.assert_array_equal(ties, [3, 3])

    def test_time_order_can_change_persistence_without_changing_endpoint(self):
        x = np.array([-1., -1, -1, 1, 1, 1])
        perm = [0, 3, 1, 4, 2, 5]
        a = rel.relation_summary(x, -x, np.ones(6, bool))
        b = rel.relation_summary(x[perm], -x[perm], np.ones(6, bool))
        for key in ("own_sum", "retained_sum", "forward_fraction", "forward_overlap_fraction"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a["forward_longest_physical"], 3)
        self.assertEqual(b["forward_longest_physical"], 1)

    def test_reversed_signs_are_reported_not_silently_reoriented(self):
        r = rel.relation_summary(np.ones(4), -np.ones(4), np.ones(4, bool))
        self.assertEqual(r["forward_fraction"], 0)
        self.assertEqual(r["reverse_rounds"], 4)
        self.assertEqual(r["either_overlap_fraction"], 1)

    def test_named_group_and_other_retained_do_not_change_total(self):
        h = np.arange(48, dtype=float).reshape(4, 3, 4)
        mask = np.ones((4, 3), bool)
        x, all_retained, _ = rel.decompose(h, mask, 0)
        _, group, _ = rel.decompose(h, mask, 0, [1])
        np.testing.assert_array_equal(all_retained-group, h[:, 2])
        np.testing.assert_array_equal(x+all_retained, h.sum(1))

    def test_bad_inputs_are_rejected(self):
        h, mask = np.ones((4, 3, 2)), np.ones((4, 3), bool)
        for ids in ([0], [1, 1], [], [3]):
            with self.assertRaises(ValueError):
                rel.decompose(h, mask, 0, ids)
        mask[0, 0] = False
        with self.assertRaises(ValueError):
            rel.decompose(h, mask, 0)

    def test_coordinate_decode_uses_probe_pattern_output_order(self):
        e = dict(history=np.zeros((4, 2, 360)), coordinates=dict(
            views=["clean"]+[f"v{i}" for i in range(12)], probe_seeds=[10, 11, 12]))
        self.assertEqual(rel.coordinate_info(e, 230), dict(index=230, probe_seed=11, view="v11", output=0))

    def test_role_and_target_annotations_do_not_affect_selection(self):
        rng = np.random.default_rng(2)
        e = dict(history=rng.normal(size=(10, 3, 4)), mask=np.ones((10, 3), bool),
                 coordinates=dict(views=["clean", "v0", "v1"], probe_seeds=[10]),
                 inventory_sha256="synthetic", identity="synthetic", benign=False, stratum={})
        base = dict(name="synthetic", evidence="synthetic", report_client=0, partners=[1], target=0)
        a, _ = rel.analyze(e, base)
        b, _ = rel.analyze(e, dict(base, report_client=2, partners=[0], target=1))
        self.assertEqual(a["clients"], b["clients"])


if __name__ == "__main__":
    unittest.main()
