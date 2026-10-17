"""Analytical counterexamples for the proposed boundary kernel, not FL evidence."""
import unittest

import numpy as np

from reproduction.detection.history_boundary import boundary_evidence, residual_interval


class HistoryBoundaryTests(unittest.TestCase):
    def interval(self, clean, probe, cb=None, pb=None, eps=0., nu=0.):
        cb = np.eye(2) if cb is None else np.asarray(cb, float)
        pb = np.eye(2) if pb is None else np.asarray(pb, float)
        return residual_interval(clean, cb, probe, pb, clean_error=eps,
                                 coefficient_lower=np.full(cb.shape[1], -10.),
                                 coefficient_upper=np.full(cb.shape[1], 10.), probe_error=nu)

    def evidence(self, intervals, deleted=(True, False), retained=(False, True)):
        lower = np.tile(np.stack([i.lower for i in intervals]), (4, 1, 1))
        upper = np.tile(np.stack([i.upper for i in intervals]), (4, 1, 1))
        return boundary_evidence(lower, upper, deleted=np.asarray(deleted),
                                 retained=np.asarray(retained), blocks=[(0, 2), (2, 4)])

    def test_persistent_normal_opposition_is_explained(self):
        # Opposite normal mixtures persist for every round, yet residuals are zero.
        removed = self.interval([-2., 1.], [-2., 1.])
        kept = self.interval([2., -1.], [2., -1.])
        self.assertEqual(self.evidence([removed, kept])["score"], 0.)

    def test_conditional_removed_suppression_and_retained_enhancement(self):
        removed = self.interval([-2., 1.], [-5., -2.])
        kept = self.interval([2., -1.], [6., 3.])
        result = self.evidence([removed, kept])
        self.assertAlmostEqual(result["score"], 3.)
        self.assertFalse(result["calibrated"])
        self.assertFalse(result["actual_fu_verified"])

    def test_wrong_deletion_boundary_does_not_get_same_evidence(self):
        intervals = [self.interval([0., 0.], [-3., -3.]),
                     self.interval([0., 0.], [4., 4.])]
        result = self.evidence(intervals, deleted=(False, True), retained=(True, False))
        self.assertEqual(result["score"], 0.)

    def test_clean_nullspace_prevents_overconfident_point_fit(self):
        # Clean observation cannot identify beta[1]; it could explain all probe signal.
        item = self.interval([1.], [5.], cb=[[1., 0.]], pb=[[0., 1.]])
        np.testing.assert_allclose(item.lower, [-5.])
        np.testing.assert_allclose(item.upper, [15.])
        self.assertEqual(item.clean_rank, 1)

    def test_normal_outside_reference_is_unknown_not_attack(self):
        item = self.interval([1.], [50.], cb=[[0.]], pb=[[1.]])
        self.assertEqual(item.status, "unexplained_clean_response")
        self.assertIsNone(item.lower)

    def test_probe_signal_explainable_by_normal_basis_can_be_missed(self):
        # Identical observed functions can have different intentions; no label is inferred.
        item = self.interval([2., 3.], [2., 3.])
        np.testing.assert_allclose(item.lower, [0., 0.])

    def test_more_uncertainty_only_widens_interval(self):
        narrow = self.interval([1., 2.], [4., -2.], eps=.1, nu=.1)
        wide = self.interval([1., 2.], [4., -2.], eps=.5, nu=.2)
        self.assertTrue(np.all(wide.lower <= narrow.lower))
        self.assertTrue(np.all(wide.upper >= narrow.upper))

    def test_box_extrema_match_closed_form(self):
        # Bclean=I fixes an independent box around clean.
        probe_basis = np.array([[2., -3.], [-1., 4.]])
        clean = np.array([1., 2.])
        probe = np.array([7., 8.])
        item = self.interval(clean, probe, pb=probe_basis, eps=.25, nu=.1)
        center = probe - probe_basis @ clean
        radius = .25 * np.abs(probe_basis).sum(1) + .1
        np.testing.assert_allclose(item.lower, center - radius)
        np.testing.assert_allclose(item.upper, center + radius)

    def test_single_round_spike_fails_second_heldout_block(self):
        effects = np.zeros((4, 2, 2))
        effects[0, 0] = -8.
        effects[0, 1] = 8.
        result = boundary_evidence(effects, effects, deleted=np.array([True, False]),
                                   retained=np.array([False, True]), blocks=[(0, 2), (2, 4)])
        self.assertEqual(result["score"], 0.)

    def test_source_class_coverage_is_required(self):
        effects = np.tile([[-3., 0.], [3., 0.]], (4, 1, 1))
        result = boundary_evidence(effects, effects, deleted=np.array([True, False]),
                                   retained=np.array([False, True]), blocks=[(0, 2), (2, 4)])
        self.assertEqual(result["score"], 0.)

    def test_client_permutation_and_internal_context(self):
        effects = np.tile([[-3., -3.], [4., 4.], [-100., 100.]], (4, 1, 1))
        # Third atom outside the requested owner's internal context is not used.
        kwargs = dict(deleted=np.array([True, False, False]),
                      retained=np.array([False, True, False]), blocks=[(0, 2), (2, 4)])
        a = boundary_evidence(effects, effects, **kwargs)
        perm = [2, 0, 1]
        b = boundary_evidence(effects[:, perm], effects[:, perm],
                              deleted=kwargs["deleted"][perm], retained=kwargs["retained"][perm],
                              blocks=kwargs["blocks"])
        self.assertEqual(a["score"], b["score"])
        self.assertEqual(a["score"], 3.)

    def test_rejects_missing_atoms_overlap_and_bad_bounds(self):
        values = np.zeros((4, 2, 2))
        kwargs = dict(deleted=np.array([True, False]), retained=np.array([False, True]),
                      blocks=[(0, 2), (2, 4)])
        missing = values.copy()
        missing[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            boundary_evidence(missing, values, **kwargs)
        with self.assertRaises(ValueError):
            boundary_evidence(values, values, **{**kwargs, "blocks": [(0, 3), (2, 4)]})
        with self.assertRaises(ValueError):
            self.interval([0., 0.], [0., 0.], eps=-1.)


if __name__ == "__main__":
    unittest.main()
