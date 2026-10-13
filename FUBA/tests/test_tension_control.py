"""Analytic controller optima, information limits, and actual replay semantics."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tension_control import control_step, select_coordinates, paired_logits, ParameterSpace
from unlearn_method.federaser_clientwise import calibrated_aggregate, replay


class ControlTests(unittest.TestCase):
    def test_scalar_hinge_analytic_optimum(self):
        d, info = control_step([2.], [[1.]], np.empty((0, 1)), strength=3., radius=3.)
        np.testing.assert_allclose(d, [2/7], atol=1e-6)
        self.assertFalse(info["forgetting_verified"])

    def test_orthogonal_task_direction_preserved(self):
        d, _ = control_step([2., 3.], [[1., 0.]], [[0., -1.]], strength=9., radius=4.)
        np.testing.assert_allclose(d, [2/19, 3], atol=1e-5)

    def test_class_protection_can_force_stall(self):
        d, info = control_step([1.], np.empty((0, 1)), [[1.]], radius=2.)
        self.assertLessEqual(d[0], 1e-8)
        self.assertEqual(info["status"], "stalled")

    def test_trust_ball_analytic_projection(self):
        d, _ = control_step([3., 4.], np.empty((0, 2)), np.empty((0, 2)), radius=2.)
        np.testing.assert_allclose(d, [1.2, 1.6], atol=1e-5)

    def test_nonzero_budgets(self):
        d, _ = control_step([2., 3.], [[1., 0.]], [[0., 1.]],
                            strength=1, kappa=1., epsilon=.5, radius=5.)
        np.testing.assert_allclose(d, [4/3, .5], atol=1e-5)

    def test_same_fraction_does_not_select_history(self):
        idx, _ = select_coordinates([[10., 1.], [-8., -1.]], [.8, .8])
        self.assertEqual(len(idx), 0)

    def test_boundary_selects_coordinate_without_attack_labels(self):
        e = [[10., 0.], [-10., 2.], [0., -2.]]
        idx, _ = select_coordinates(e, [0., 1., 0.], budget=1)
        np.testing.assert_array_equal(idx, [0])
        reverse, _ = select_coordinates(e, [1., 0., 0.], budget=1)
        self.assertEqual(len(reverse), 0)

    def test_invalid_budget_and_nonfinite_fail(self):
        for kwargs in ({"epsilon": -1}, {"strength": -1}, {"radius": np.inf}):
            with self.assertRaises(ValueError):
                control_step([1.], [[1.]], [[1.]], **kwargs)

    def test_actual_proposal_gate_drops_non_increasing_direction(self):
        e = [[10., 2.], [-10., -2.]]
        idx, _ = select_coordinates(e, [0, 1], budget=1, mode="history_action", response=[-1., .5])
        np.testing.assert_array_equal(idx, [1])
        empty, _ = select_coordinates(e, [0, 1], mode="history_action", response=[-1., 0.])
        self.assertEqual(len(empty), 0)

    def test_operator_gate_requires_same_coordinate_shape(self):
        with self.assertRaises(ValueError):
            select_coordinates([[1.], [-1.]], [0, 1], mode="history_action", response=[1., 2.])

    def test_pair_cancels_arbitrary_class_bias(self):
        torch.manual_seed(12)
        net = torch.nn.Linear(2, 3).double()
        x, a = torch.randn(4, 2, dtype=torch.double), torch.randn(4, 2, dtype=torch.double)
        before = paired_logits(net, x, a).detach()
        with torch.no_grad():
            net.bias.add_(torch.tensor([8., -4., 2.], dtype=torch.double))
        torch.testing.assert_close(paired_logits(net, x, a), before)

    def test_parameter_gradient_matches_finite_difference(self):
        torch.manual_seed(2)
        net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)).double()
        sp = ParameterSpace(net)
        x = torch.tensor([[.3, -.1]], dtype=torch.double)
        q = paired_logits(net, x, x+.1).flatten()[0]
        grad, v = sp.gradient(q), sp.vector()
        direction = np.random.default_rng(7).normal(size=sp.size)
        sp.assign(v+1e-6*direction)
        numerical = (paired_logits(net, x, x+.1).flatten()[0].item()-q.item())/1e-6
        self.assertAlmostEqual(numerical, grad@direction, places=5)

    def test_buffers_rejected_instead_of_ignored(self):
        with self.assertRaises(NotImplementedError):
            ParameterSpace(torch.nn.BatchNorm1d(3))

    def test_true_backdoor_coordinate_can_be_invisible_to_brightness(self):
        net = torch.nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            net.weight.copy_(torch.tensor([[1., 0.], [-1., 4.]], dtype=torch.double))
        x = torch.tensor([[1., 0.]], dtype=torch.double)
        before = paired_logits(net, x, x*.9).detach()
        oracle_before = paired_logits(net, x, x+torch.tensor([[0., 1.]])).detach()
        with torch.no_grad():
            net.weight[1, 1] += 3.
        torch.testing.assert_close(paired_logits(net, x, x*.9), before)
        self.assertGreater((paired_logits(net, x, x+torch.tensor([[0., 1.]]))-oracle_before)[0, 1].item(), 0.)


class FedEraserTests(unittest.TestCase):
    def test_calibrate_clients_before_average(self):
        old = {0: [3., 0.], 1: [0., 1.]}
        new = {0: [0., 2.], 1: [4., 0.]}
        np.testing.assert_allclose(calibrated_aggregate(old, new), [.5, 1.5])
        mean_first = np.linalg.norm([1.5, .5])*np.array([2., 1.])/np.linalg.norm([2., 1.])
        self.assertFalse(np.allclose(mean_first, [.5, 1.5]))

    def test_participation_is_per_round_and_deleted_never_called(self):
        events = [{"updates": {0: [5., 0.], 1: [0., 1.]}},
                  {"updates": {2: [2., 0.], 0: [0., 4.]}}]
        seen = []
        def calibrate(state, j, t):
            seen.append((j, t))
            state[:] = 999  # callback cannot mutate replay state
            return [1., 0.]
        result, meta = replay([0., 0.], events, {0}, calibrate)
        np.testing.assert_allclose(result, [3., 0.])
        self.assertEqual(seen, [(1, 0), (2, 1)])
        self.assertFalse(meta["forgetting_verified"])

    def test_weighted_whole_vector_calibration(self):
        d = calibrated_aggregate({0: [3., 4.], 1: [0., 2.]},
                                 {0: [0., 1.], 1: [1., 0.]}, {0: 3., 1: 1.})
        np.testing.assert_allclose(d, [.5, 3.75])

    def test_undefined_direction_and_all_deleted_fail(self):
        with self.assertRaises(ValueError):
            calibrated_aggregate({0: [1.]}, {0: [0.]})
        with self.assertRaises(ValueError):
            replay([0.], [{"updates": {0: [1.]}}], {0}, lambda *a: [1.])

    def test_controller_changes_actual_replay_state(self):
        def controller(state, v, t):
            return control_step(v, [[1., 0.]], np.empty((0, 2)), strength=1, radius=3)
        result, _ = replay([0., 0.], [{"updates": {0: [1., 0.], 1: [1., 0.]}}], {0},
                           lambda *a: [1., 0.], controller)
        np.testing.assert_allclose(result, [1/3, 0.], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
