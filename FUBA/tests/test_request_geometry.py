"""Independent matrix identities, analytic QCQP optima and failure semantics."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FUBA.research.request_geometry import (request_paths, cancellation_geometry, geometry_control,
                              check_actual_retention, ControlInfeasible, TemporalFUController)
from unlearn_method.federaser_clientwise import replay


class GeometryTests(unittest.TestCase):
    def test_request_uses_actual_aggregation_weights_and_partial_participation(self):
        events = [{"updates": {0: [4., 0.], 1: [-2., 2.]}, "weights": {0: 1, 1: 3}},
                  {"updates": {1: [3., 1.]}}]
        d, k = request_paths(events, [0])
        np.testing.assert_allclose(d, [[1, 0], [0, 0]])
        np.testing.assert_allclose(k, [[-1.5, 1.5], [3, 1]])

    def test_uniform_removal_and_empty_sides_have_no_positive_geometry(self):
        x = np.random.default_rng(4).normal(size=(7, 4))
        for alpha in (0., .2, .5, 1.):
            g, _ = cancellation_geometry(alpha*x, (1-alpha)*x, smoothing=2)
            self.assertEqual(g.shape, (0, 4))

    def test_perfect_cancellation_has_unit_eigenvalue(self):
        g, info = cancellation_geometry([[1., 0.], [2., 0.]], [[-1., 0.], [-2., 0.]])
        np.testing.assert_allclose(g.T@g, np.diag([1., 0.]), atol=1e-12)
        np.testing.assert_allclose(info["eigenvalues"], [1.])

    def test_direct_full_matrix_psd_projection_matches_low_rank(self):
        rng = np.random.default_rng(42)
        d, k = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
        g, _ = cancellation_geometry(d, k, smoothing=0)
        c = -(d.T@k+k.T@d)/(np.sum(d*d)+np.sum(k*k))
        vals, vecs = np.linalg.eigh(c)
        expected = (vecs*np.maximum(vals, 0))@vecs.T
        np.testing.assert_allclose(g.T@g, expected, atol=1e-12)
        self.assertLessEqual(np.linalg.norm(g.T@g, 2), 1+1e-12)

    def test_rotation_and_common_scaling_equivariance(self):
        rng = np.random.default_rng(12)
        d, k = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        g, _ = cancellation_geometry(d, k)
        rotated, _ = cancellation_geometry(1e40*d@rotation, 1e40*k@rotation)
        np.testing.assert_allclose(rotated.T@rotated, rotation.T@(g.T@g)@rotation, atol=1e-12)

    def test_temporal_smoothing_uses_order_but_unsmoothed_moment_does_not(self):
        rng = np.random.default_rng(22)
        d, k = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
        order = [0, 3, 7, 2, 4, 1, 6, 5]
        a, _ = cancellation_geometry(d, k, smoothing=0)
        b, _ = cancellation_geometry(d[order], k[order], smoothing=0)
        np.testing.assert_allclose(a.T@a, b.T@b, atol=1e-12)
        a, _ = cancellation_geometry(d, k, smoothing=2)
        b, _ = cancellation_geometry(d[order], k[order], smoothing=2)
        self.assertGreater(np.linalg.norm(a.T@a-b.T@b), .01)

    def test_all_zero_and_invalid_histories(self):
        g, _ = cancellation_geometry(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertEqual(g.shape, (0, 3))
        with self.assertRaises(ValueError):
            cancellation_geometry([[1]], [[1]], times=[np.nan])
        with self.assertRaises(ValueError):
            cancellation_geometry([[1], [2]], [[1], [2]], times=[1, 1])
        with self.assertRaises(ValueError):
            request_paths([{"updates": {0: [1.]}}], [1])


class SolverTests(unittest.TestCase):
    def test_unconstrained_closed_form_and_orthogonal_preservation(self):
        v = np.array([2., 3.])
        d, info = geometry_control(v, [[1., 0.]], np.empty((0, 2)), strength=3., min_progress=0)
        np.testing.assert_allclose(d, [.5, 3.], atol=1e-6)
        self.assertLess(info["kkt_stationarity"], 1e-5)
        self.assertFalse(info["forgetting_verified"])

    def test_dense_closed_form(self):
        rng = np.random.default_rng(5)
        g, v = rng.normal(size=(4, 6)), rng.normal(size=6)
        d, _ = geometry_control(v, g, np.empty((0, 6)), strength=2., min_progress=0)
        np.testing.assert_allclose(d, np.linalg.solve(np.eye(6)+2*g.T@g, v), atol=2e-5)

    def test_progress_floor_prevents_zero_as_defense(self):
        d, info = geometry_control([2.], [[1.]], np.empty((0, 1)), strength=100., min_progress=.2)
        np.testing.assert_allclose(d, [.4], atol=1e-6)
        self.assertAlmostEqual(info["progress"], .2, places=6)

    def test_relative_risk_budget_has_analytic_solution_independent_of_scale(self):
        for scale in (1., 1e-6, 1e6):
            d, info = geometry_control([2., 3.], [[scale, 0.]], np.empty((0, 2)),
                                       strength=0, risk_ratio=.2, min_progress=.1)
            np.testing.assert_allclose(d, [.4, 3.], atol=1e-5)
            self.assertLessEqual(info["normalized_risk_after"], .040001*info["normalized_risk_before"])

    def test_zero_risk_budget_preserves_orthogonal_progress(self):
        d, _ = geometry_control([0., 2.], [[1., 0.]], np.empty((0, 2)), risk_ratio=0)
        np.testing.assert_allclose(d, [0., 2.], atol=1e-6)

    def test_conflicting_retention_and_progress_is_not_success(self):
        with self.assertRaises(ControlInfeasible):
            geometry_control([1.], [[1.]], [[1.]], min_progress=.1)
        with self.assertRaises(ControlInfeasible):
            geometry_control([1.], [[1.]], np.empty((0, 1)), min_progress=.5, radius=.2)

    def test_trust_ball_and_class_constraint(self):
        d, info = geometry_control([3., 4.], np.empty((0, 2)), [[1., 0.]],
                                   epsilon=0., radius=2., min_progress=.1)
        np.testing.assert_allclose(d, [0., 2.], atol=1e-5)
        self.assertLess(info["max_normalized_violation"], 1e-7)

    def test_actual_curvature_requires_backtracking(self):
        # Linearized loss at 0 decreases for d>0, but the real loss -d+d^2 rises at d=2.
        accepted, info = check_actual_retention([2.], [2.], lambda d: [-d[0]+d[0]**2], min_progress=.2)
        np.testing.assert_allclose(accepted, [1.])
        self.assertEqual(info["backtracks"], 1)
        with self.assertRaises(ControlInfeasible):
            check_actual_retention([2.], [2.], lambda d: [-d[0]+d[0]**2], min_progress=.75)

    def test_actual_replay_recalibrates_only_retained_and_controls_all_steps(self):
        events = [{"updates": {0: np.array([-1., 0.]), 1: np.array([1., 1.])}} for _ in range(3)]
        d, k = request_paths(events, [0])
        g, _ = cancellation_geometry(d, k)
        calls = []
        def calibrate(state, client, t):
            calls.append((client, t))
            return np.array([1., 1.])-.1*state
        def controller(state, v, t):
            return geometry_control(v, g, np.empty((0, 2)), strength=2.)
        result, info = replay([0., 0.], events, [0], calibrate, controller)
        reference, _ = replay([0., 0.], events, [0], calibrate)
        self.assertFalse(np.allclose(result, reference))
        self.assertTrue(all(client == 1 for client, _ in calls))
        self.assertEqual(len(info["steps"]), 3)
        self.assertTrue(all(step["progress"] >= .1-1e-7 for step in info["steps"]))
        self.assertFalse(info["forgetting_verified"])

    def test_temporal_adapter_verifies_actual_losses_at_each_replay_state(self):
        events = [{"updates": {0: [-1., 0.], 1: [1., 2.]}} for _ in range(3)]
        seen = []
        def oracle(state, t):
            seen.append((state.copy(), t))
            # A retained quadratic task centered below the current y-coordinate.
            base = .5*(state[1]-10)**2
            return np.array([[0., state[1]-10]]), lambda step: [.5*(state[1]+step[1]-10)**2-base]
        controller = TemporalFUController(events, [0], oracle, epsilon=0, min_progress=.05)
        result, info = replay([0., 0.], events, [0], lambda state, j, t: np.array([1., 2.]), controller)
        self.assertEqual([t for _, t in seen], [0, 1, 2])
        self.assertGreater(result[1], 0)
        self.assertTrue(all(s["actual_retention_verified"] for s in info["steps"]))
        self.assertFalse(np.allclose(seen[0][0], seen[-1][0]))


if __name__ == "__main__":
    unittest.main()
