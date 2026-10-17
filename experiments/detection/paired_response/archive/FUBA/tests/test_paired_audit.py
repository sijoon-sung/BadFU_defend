"""Counterexamples for the proposed statistical interpretation and selection."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit import (PairedPolicy, paired_audit, centered_log,
                          cancellation_metrics, rank_calibrate)


class FunctionalFixture:
    def __init__(self, mode="cancel", seed=0, carriers=1, suppressors=1):
        self.units = tuple(str(i) for i in range(12))
        self.views = ("public", "request")
        self.mode = mode
        rng = np.random.default_rng(seed)
        self.base = rng.normal(0, .2, (64, 3))
        direction = rng.normal(0, .25, (64, 3)) + np.array([1., -.5, -.5])
        self.effects = rng.normal(0, .015, (12, 64, 3))
        self.direction = direction
        if mode != "clean":
            for i in range(carriers):
                self.effects[i] = direction / carriers
            for i in range(suppressors):
                self.effects[11-i] = -direction / suppressors
        if mode == "overfit":
            idx = np.random.default_rng(PairedPolicy().seed).permutation(64)[32:]
            self.effects[0, idx] *= -1

    def propose(self, removals):
        return dict(removals)

    def predict(self, removals, view):
        z = self.base.copy()
        if not (self.mode == "localized" and view == "public"):
            for u, w in removals.items():
                z -= w * self.effects[int(u)]
            if self.mode == "nonlinear":
                z += 8 * removals.get("11", 0) * removals.get("0", 0) * self.direction
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


class PairedAuditTests(unittest.TestCase):
    def test_centered_log_is_offset_and_class_permutation_equivariant(self):
        z = np.array([[1, 2, -1], [2, -2, 1]], dtype=float)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        np.testing.assert_allclose(centered_log(p), z - z.mean(1, keepdims=True))
        np.testing.assert_allclose(centered_log(p[:, [2, 0, 1]]), centered_log(p)[:, [2, 0, 1]])

    def test_additive_identity_and_nonlinear_remainder_bound(self):
        rng = np.random.default_rng(12)
        a, b, q = rng.normal(size=(3, 50, 4))
        exact = cancellation_metrics(a, b, a+b)
        self.assertAlmostEqual(exact["cancellation"], exact["opposition"])
        finite = cancellation_metrics(a, b, a+b+q)
        bound = (2*np.sqrt(np.mean((a+b)**2)*np.mean(q*q))+np.mean(q*q))/(np.mean(a*a)+np.mean(b*b))
        self.assertLessEqual(abs(finite["cancellation"]-finite["opposition"]), bound + 1e-12)

    def test_exact_pair_and_benign_pair_are_observationally_identical(self):
        # Identical functions can come from conflicting honest objectives or poison.
        a = paired_audit(FunctionalFixture(), {"11": .2})
        b = paired_audit(FunctionalFixture(mode="honest_opposition"), {"11": .2})
        self.assertEqual(a, b)
        self.assertEqual(a["status"], "review_priority")
        self.assertFalse(a["release_authorized"])

    def test_linear_fit_does_not_certify_nonlinear_joint_removal(self):
        a = paired_audit(FunctionalFixture(mode="nonlinear"), {"11": .2})
        self.assertGreater(a["discovery_linear_restoration"], .99)
        self.assertLess(a["views"]["request"]["restoration"], 0)
        self.assertNotEqual(a["status"], "review_priority")

    def test_confirmation_rejects_discovery_only_fit(self):
        a = paired_audit(FunctionalFixture(mode="overfit"), {"11": .2})
        self.assertGreater(a["discovery_linear_restoration"], .99)
        self.assertLess(a["views"]["request"]["restoration"], 0)
        self.assertNotEqual(a["status"], "review_priority")

    def test_request_view_is_needed_for_localized_signal(self):
        b = FunctionalFixture(mode="localized")
        self.assertEqual(paired_audit(b, {"11": .2})["status"], "review_priority")
        self.assertEqual(paired_audit(b, {"11": .2}, views=("public",))["status"], "unconfirmed")

    def test_multiple_carriers_and_requesters(self):
        a = paired_audit(FunctionalFixture(carriers=4, suppressors=2), {"10": .2, "11": .2})
        self.assertEqual(a["status"], "review_priority")
        self.assertTrue(set(a["partners"]).issubset({"0", "1", "2", "3"}))

    def test_calibration_resolution_and_ties(self):
        self.assertEqual(rank_calibrate(10, [1, 2, 3])["status"], "insufficient_resolution")
        self.assertEqual(rank_calibrate(1, [1]*99)["p"], 1)
        self.assertTrue(rank_calibrate(2, [1]*99)["reject"])

    def test_invalid_removal_is_rejected(self):
        for removal in ({}, {"missing": .2}, {"11": -1}, {"11": np.nan}):
            with self.assertRaises(ValueError):
                paired_audit(FunctionalFixture(), removal)

    def test_new_review_adds_hold_when_existing_detector_is_quiet(self):
        from structural_guard import StructuralGuard
        guard = StructuralGuard(FunctionalFixture(), comparator=lambda *args: {"risk": False},
                                confirmation=paired_audit)
        result = guard.request({"11": .2})
        self.assertEqual(result.status, "held")
        self.assertEqual(result.reason, "paired_review")
        self.assertEqual(guard.committed, {})

    def test_unconfirmed_pair_never_clears_existing_alarm(self):
        from structural_guard import StructuralGuard
        guard = StructuralGuard(FunctionalFixture("nonlinear"), comparator=lambda *args: {"risk": True},
                                confirmation=paired_audit)
        result = guard.request({"11": .2})
        self.assertEqual(result.status, "held")
        self.assertEqual(result.reason, "activation")
        self.assertEqual(guard.committed, {})


if __name__ == "__main__":
    unittest.main()
