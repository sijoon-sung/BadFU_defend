"""Structural identities and information limits; not attack performance tests."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tension_audit import boundary_profile, score_request


class TensionTests(unittest.TestCase):
    def test_masked_effects_have_large_tension_and_zero_net(self):
        p = boundary_profile([[10.], [-10.]], [0, 0])
        np.testing.assert_allclose(p["tension"], [20])
        np.testing.assert_allclose(p["net"], [0])
        np.testing.assert_allclose(p["score"], [0])

    def test_only_removing_suppression_crosses_the_boundary(self):
        e = [[10.], [-10.]]
        self.assertEqual(boundary_profile(e, [0, 1])["score"].item(), 10)
        for r in ([0, 0], [1, 0], [1, 1]):
            self.assertEqual(boundary_profile(e, r)["score"].item(), 0)

    def test_request_fraction_scales_exposed_mass(self):
        p = boundary_profile([[10.], [-10.]], [0, .2])
        np.testing.assert_allclose(p["cross_mass"], [2])
        np.testing.assert_allclose(p["retained_signed_effect"], [2])

    def test_equal_fraction_removal_preserves_balance(self):
        for amount in (0., .1, .5, .9, 1.):
            p = boundary_profile([[10.], [-10.]], [amount, amount])
            np.testing.assert_allclose(p["score"], [0])

    def test_equal_fraction_unbalanced_counterexample_to_v2(self):
        p = boundary_profile([[10.], [-8.]], [.8, .8])
        self.assertGreater(p["score_v2"].item(), .35)
        self.assertAlmostEqual(p["score"].item(), 0)
        rng = np.random.default_rng(7)
        for _ in range(100):
            e = rng.normal(size=(7, 5))
            r = np.full(7, rng.random())
            np.testing.assert_allclose(boundary_profile(e, r)["score"], 0, atol=1e-12)

    def test_rate_gap_matches_two_unequal_roles(self):
        for a, b in ((10., 8.), (8., 10.)):
            p = boundary_profile([[a], [-b]], [.2, .8])
            self.assertAlmostEqual(p["score"].item(), 2*min(a,b)/(a+b)*min(a,b)*.6)

    def test_balanced_score_is_linear_release_baseline(self):
        e = np.array([[2.], [8.], [-3.], [-7.]])
        rng = np.random.default_rng(1)
        for _ in range(100):
            r = rng.random(4)
            np.testing.assert_allclose(boundary_profile(e, r)["score"],
                                       np.maximum(-(r[:, None]*e).sum(0), 0), atol=1e-12)

    def test_balanced_crossing_identity_for_multiple_units(self):
        e = np.array([[2.], [8.], [-3.], [-7.]])
        rng = np.random.default_rng(20261014)
        for _ in range(100):
            r = rng.random(4)
            p = boundary_profile(e, r)
            np.testing.assert_allclose(p["cross_mass"], np.maximum(((1-r[:,None])*e).sum(0), 0), atol=1e-12)

    def test_client_and_within_client_units_have_identical_semantics(self):
        e = [[3.], [7.], [-10.]]
        a = score_request(e, ("client0", "client1", "client2"), {"client2": 1})
        b = score_request(e, ("client0/bd0", "client0/bd1", "client0/cv"), {"client0/cv": 1})
        for k in a:
            np.testing.assert_allclose(a[k], b[k])

    def test_single_carrier_is_not_exempted(self):
        np.testing.assert_allclose(boundary_profile([[10.], [-10.]], [0, 1])["score"],
                                   boundary_profile([[4.], [6.], [-10.]], [0, 0, 1])["score"])

    def test_input_coordinates_are_preserved_before_aggregation(self):
        e = np.array([[[5., 0.], [-5., 0.]], [[-5., 0.], [5., 0.]]])
        p = boundary_profile(e, [0, 1])
        self.assertEqual(p["score"][0, 0], 5)
        self.assertEqual(p["score"][1, 0], 0)
        self.assertEqual(boundary_profile(e.mean(1), [0, 1])["score"].max(), 0)

    def test_aggregation_can_hide_internal_tension(self):
        fine = boundary_profile([[10.], [-10.]], [0, 1])
        coarse = boundary_profile([[0.]], [1])
        self.assertGreater(fine["tension"].item(), coarse["tension"].item())
        self.assertEqual(coarse["score"].item(), 0)

    def test_honest_opposition_has_the_same_observation(self):
        # These values can arise from honest conflicting tasks as well as masking.
        e = [[10.], [-10.]]
        self.assertEqual(boundary_profile(e, [0, 1])["score"].item(), 10)
        # There is deliberately no malicious/benign classifier in this primitive.
        self.assertNotIn("malicious", boundary_profile(e, [0, 1]))

    def test_conditional_two_role_identity_with_remainder(self):
        b, eta, ra, rs = 5., .2, .1, .8
        s = -b+eta
        observed = boundary_profile([[b], [s]], [ra, rs])["retained_signed_effect"].item()
        self.assertAlmostEqual(observed, (rs-ra)*b+(1-rs)*eta)

    def test_invalid_units_and_fractions_fail(self):
        for r in ([-.1, 0], [0, 1.1], [0, float("nan")]):
            with self.assertRaises(ValueError):
                boundary_profile([[1], [-1]], r)
        with self.assertRaises(ValueError):
            score_request([[1], [-1]], ("a", "a"), {"a": 1})
        with self.assertRaises(ValueError):
            score_request([[1], [-1]], ("a", "b"), {"unknown": 1})


if __name__ == "__main__":
    unittest.main()
