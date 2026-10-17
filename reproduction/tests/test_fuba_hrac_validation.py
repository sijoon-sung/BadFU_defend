import unittest

import numpy as np

from reproduction.detection import hrac_request_gate as gate
from reproduction.detection import validate_fuba_detection as fuba


class RequestGateTests(unittest.TestCase):
    def request(self, observed=(1., 0.), removal=(.5, 0.)):
        return dict(id="subset", round=1, removal=list(removal), intra=dict(
            atoms=np.eye(2).tolist(), observed=list(observed), atom_owners=[0, 0], atom_masses=[1, 1],
            lower=[0, 0], upper=[1, 1], provenance=dict(independent_observation=True,
                observation_id="test", request_round=1, history_source="stored/shards",
                request_source="measured/request", constraints_source="request_metadata",
                complete_within_requested_owners=True)))

    def test_legitimate_class_subset_not_required_to_be_proportional(self):
        # Half of a client's data can legitimately contain only one class.
        h = np.zeros((1, 1, 2, 1, 2))
        result = gate.evaluate(h, [self.request()], "test")
        self.assertLess(result["requests"][0]["intra"]["upper_bound"], 1e-8)
        self.assertTrue(result["both_channels_observed"])
        self.assertIsNone(result["decision"]["alarm"])
        self.assertGreater(np.linalg.norm(np.array([1, 0]) - .5*np.ones(2)), .5)

    def test_equal_unsigned_spectra_do_not_hide_signed_mismatch(self):
        normal, altered = np.array([1., 0.]), np.array([-1., 0.])
        np.testing.assert_array_equal(np.outer(normal, normal), np.outer(altered, altered))
        h = np.zeros((1, 1, 2, 1, 2))
        good = gate.evaluate(h, [self.request(normal)], "test")
        bad = gate.evaluate(h, [self.request(altered)], "test")
        self.assertLess(good["episode_scores"]["intra"], 1e-8)
        self.assertGreater(bad["episode_scores"]["intra"], 1)

    def test_full_client_request_cannot_silently_keep_an_atom(self):
        h = np.zeros((1, 1, 2, 1, 2))
        result = gate.evaluate(h, [self.request(removal=(1, 0))], "test")
        self.assertGreater(result["episode_scores"]["intra"], .5)
        np.testing.assert_allclose(result["requests"][0]["intra"]["coefficients"], [1, 1])

    def test_missing_independent_request_is_not_zero_or_clean(self):
        h = np.zeros((1, 1, 2, 1, 2))
        result = gate.evaluate(h, [dict(id="whole", removal=[1, 0])], "test")
        self.assertIsNone(result["episode_scores"]["intra"])
        self.assertIsNone(result["decision"]["alarm"])
        self.assertFalse(result["both_channels_observed"])

    def test_bad_binding_tautological_source_and_other_owner_rejected(self):
        h = np.zeros((1, 1, 2, 1, 2))
        for key, value in (("observation_id", "wrong"), ("request_source", "stored/shards"),
                           ("independent_observation", False), ("complete_within_requested_owners", False),
                           ("request_round", 2)):
            request = self.request()
            request["intra"]["provenance"][key] = value
            with self.assertRaises(ValueError):
                gate.evaluate(h, [request], "test")
        request = self.request()
        request["intra"]["atom_owners"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "another owner"):
            gate.evaluate(h, [request], "test")

    def test_same_request_receives_both_channels_without_attack_label(self):
        h = np.zeros((1, 1, 2, 1, 2))
        h[0, 0, 0, 0, 0], h[0, 0, 1, 0, 0] = -1, 1
        request = self.request()
        result = gate.evaluate(h, [request], "test")
        self.assertGreater(result["episode_scores"]["inter"], 0)
        self.assertLess(result["episode_scores"]["intra"], 1e-8)
        # A feasible request can still interact with other clients' histories.


class FubaObservationTests(unittest.TestCase):
    def test_partial_participation_keeps_global_ids_and_real_weights(self):
        effect = np.stack([np.full((13*3*80, 10), 4., np.float32),
                           np.full((13*3*80, 10), 1., np.float32)])
        full, present = fuba.unpack_effect(effect, [4, 1], 7)
        self.assertEqual(np.flatnonzero(present).tolist(), [1, 4])
        self.assertTrue(np.all(full[:, 4] == 4))
        self.assertTrue(np.all(full[:, 1] == 1))
        self.assertTrue(np.all(full[:, ~present] == 0))
        with self.assertRaises(ValueError):
            fuba.unpack_effect(effect, [1, 1], 7)

    def test_to_tensor_only_preprocessing_and_mnist_shape(self):
        pixels = np.full((2, 28, 28, 1), 128, dtype=np.uint8)
        views, names = fuba.make_views(pixels)
        self.assertEqual(tuple(views.shape), (13, 2, 1, 28, 28))
        np.testing.assert_allclose(views[0].numpy(), 128/255)
        self.assertEqual(len(names), 13)
        self.assertEqual(float(views.min()), 0)
        self.assertEqual(float(views.max()), 1)

    def test_unseen_client_and_random_generator_asr_not_scored_as_success(self):
        raw = np.zeros((2, 3, 6, 13, 10))
        mask = np.zeros((2, 6), bool)
        mask[0, :2] = True
        mask[1, [1, 4]] = True
        metrics = [dict(asr_pct=99, clean_acc_pct=80, generator_source="random_initialization"),
                   dict(asr_pct=20, clean_acc_pct=80, generator_source="last_individual")]
        rows = fuba.ranking_rows(raw, mask, metrics, 4, 8, 1)
        self.assertTrue(all(r["rank_worst"] is None and r["asr_pct"] is None for r in rows if r["round"] == 1))
        self.assertTrue(all(r["rank_worst"] == 3 and r["asr_pct"] == 20 for r in rows if r["round"] == 2))
        self.assertFalse(any(r["strict_client_first"] for r in rows))
        metrics[0] = dict(asr_pct=None, clean_acc_pct=80, generator_source="warmup_no_asr")
        rows = fuba.ranking_rows(raw, mask, metrics, 4, 8, 1)
        self.assertTrue(all(r["stored_asr_pct"] is None for r in rows if r["round"] == 1))


if __name__ == "__main__":
    unittest.main()
