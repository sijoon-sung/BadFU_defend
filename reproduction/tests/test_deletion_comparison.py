"""Analytic leakage/cancellation/abstention checks; not attack performance tests."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from reproduction.detection.deletion_contrast import (apply_probe, centered, contrast, double_difference,
    empirical_auc, explain, history_scores, jvp_logits, public_directions,
    minimum_clean_error, search_logits, split_public)
from reproduction.common.deletion_inputs import Pixels, Trajectory
from reproduction.detection.request_probe import SearchConfig
from reproduction.detection.run_deletion_comparison import DEFAULT, settings, validate_role


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x.flatten(1))


class DeletionComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_any_class_bias_vector_cancels_exactly(self):
        a, b, bias = torch.randn(6, 4, dtype=torch.float64), torch.randn(6, 4, dtype=torch.float64), torch.tensor([1., -5., 3., 8.])
        dd = double_difference(a, b, a + bias, b + bias)
        torch.testing.assert_close(dd, torch.zeros_like(dd), atol=1e-14, rtol=0)

    def test_benign_head_weight_change_does_not_cancel(self):
        pre = torch.tensor([[1., 0.]])
        changed = torch.tensor([[2., 0.]])
        dd = double_difference(pre, changed, 2 * pre, 2 * changed)
        self.assertEqual(contrast(dd, 0).item(), .5)

    def test_search_zero_request_and_bias_only(self):
        pre = Toy().eval().requires_grad_(False)
        post = copy.deepcopy(pre)
        with torch.no_grad():
            post.fc.bias.add_(torch.tensor([3., -2., 1.]))
        x = torch.rand(6, 1, 2, 2)
        cfg = SearchConfig(steps=2, restarts=1, batch_size=2)
        r = search_logits(pre, post, x, cfg)
        self.assertLess(abs(r['search_objective']), 1e-6)
        self.assertLessEqual(r['delta'].abs().max(), cfg.epsilon)

    def test_search_is_independent_of_batching(self):
        pre, post = Toy().eval().requires_grad_(False), Toy().eval().requires_grad_(False)
        x = torch.rand(7, 1, 2, 2)
        a = search_logits(pre, post, x, SearchConfig(steps=2, restarts=1, batch_size=2))
        b = search_logits(pre, post, x, SearchConfig(steps=2, restarts=1, batch_size=7))
        self.assertEqual(a['target'], b['target'])
        torch.testing.assert_close(a['delta'], b['delta'])

    def test_forward_jvp_matches_linear_analytic_response_and_buffers_stay_fixed(self):
        model = Toy().eval().requires_grad_(False)
        d = {k: torch.randn_like(v) for k, v in model.named_parameters()}
        x = torch.rand(5, 1, 2, 2)
        before = copy.deepcopy(model.state_dict())
        measured = jvp_logits(model, d, x, 2)
        expected = x.flatten(1) @ d['fc.weight'].T + d['fc.bias']
        torch.testing.assert_close(measured, expected)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_class_reference_gradients_leave_model_unchanged(self):
        model = Toy().eval()
        before = copy.deepcopy(model.state_dict())
        x, y = torch.rand(9, 1, 2, 2), torch.arange(3).repeat_interleave(3)
        dirs, norms = public_directions(model, x, y, 3, 2)
        self.assertTrue(all(n > 0 for n in norms))
        for direction in dirs:
            self.assertAlmostEqual(sum(v.square().sum().item() for v in direction.values()), 1., places=6)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_public_splits_stratified_and_pixel_leakage_rejected(self):
        rng = np.random.default_rng(3)
        im = rng.integers(0, 256, (18, 2, 2, 1), dtype=np.uint8)
        y, ids = np.arange(3).repeat(6), np.array([str(i) for i in range(18)])
        splits = split_public(im, y, ids, [2, 2, 2], 1, classes=3)
        self.assertEqual(len(set(np.concatenate(splits).tolist())), 18)
        for s in splits:
            np.testing.assert_equal(np.bincount(y[s]), [2, 2, 2])
        im[1] = im[0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            split_public(im, y, ids, [2, 2, 2], 1, classes=3)

    def test_infeasible_clean_explanation_is_abstention_not_attack(self):
        item, _ = explain(np.array([1.]), np.array([100.]), np.zeros((1, 1)),
                          np.ones((1, 1)), DEFAULT | dict(clean_absolute=0., clean_relative=0.))
        self.assertIsNone(item.lower)
        raw = np.array([[[-2.], [3.]], [[-2.], [3.]]])
        lower, upper = raw.copy(), raw.copy()
        lower[0, 0] = upper[0, 0] = np.nan
        scores = history_scores(lower, upper, raw, [(0, 1), (1, 2)])
        self.assertIsNone(scores['normal_set'])
        self.assertEqual(scores['raw']['score'], 2.)

    def test_normal_reference_explains_persistent_opposition(self):
        cfg = DEFAULT | dict(clean_absolute=0., clean_relative=0., probe_error=0.)
        removed, _ = explain(np.array([-1.]), np.array([-1.]), np.ones((1, 1)), np.ones((1, 1)), cfg)
        kept, _ = explain(np.array([1.]), np.array([1.]), np.ones((1, 1)), np.ones((1, 1)), cfg)
        lo = np.tile(np.stack((removed.lower, kept.lower)), (3, 1, 1))
        hi = np.tile(np.stack((removed.upper, kept.upper)), (3, 1, 1))
        raw = np.tile(np.array([[[-1.], [1.]]]), (3, 1, 1))
        scores = history_scores(lo, hi, raw, [(0, 1), (1, 3)])
        self.assertEqual(scores['normal_set']['score'], 0.)
        self.assertEqual(scores['raw']['score'], 1.)

    def test_badfu_preprocessing_differs_from_fuba(self):
        x = torch.zeros(2, 3, 2, 2)
        badfu, fuba = Pixels(nn.Identity(), 'badfu'), Pixels(nn.Identity(), 'fuba')
        torch.testing.assert_close(fuba(x), x)
        self.assertAlmostEqual(badfu(x)[0, 0, 0, 0].item(), -.4914 / .2023, places=5)

    def test_auc_ties_and_missing_controls(self):
        self.assertEqual(empirical_auc([1., 2.], [1., 0.]), .875)
        self.assertIsNone(empirical_auc([1.], []))

    def test_poisoned_phase_cannot_be_benign_control(self):
        class T:
            protocol = {'options': dict(warm_up=5, rounds=100)}
            family = 'fuba'
            phase = 'fl'
        with self.assertRaisesRegex(ValueError, 'attack-free'):
            validate_role(dict(role='benign_request'), T())

    def test_settings_reject_nonfinite_tolerance(self):
        with self.assertRaises(ValueError):
            settings({'settings': {'clean_relative': float('nan')}})

    def test_patch_changes_only_fixed_support_and_scans_all_corners(self):
        pre, post = Toy().eval().requires_grad_(False), Toy().eval().requires_grad_(False)
        x = torch.rand(6, 1, 2, 2)
        r = search_logits(pre, post, x, SearchConfig(steps=2, restarts=1), 'patch', 1)
        changed = apply_probe(x, r)
        self.assertEqual(set(t['position'] for t in r['trials']), {'tl', 'tr', 'bl', 'br'})
        self.assertEqual(r['mask'].sum().item(), 1.)
        torch.testing.assert_close((changed - x) * (1 - r['mask']), torch.zeros_like(x))
        self.assertTrue(0 <= changed.min() <= changed.max() <= 1)

    def test_minimum_clean_error_is_diagnostic_not_tolerance_override(self):
        e = minimum_clean_error(np.array([1., 3.]), np.array([[1.], [0.]]), 2.)
        self.assertAlmostEqual(e, 3.)

    def test_recorded_weights_must_agree_with_aggregation_rule(self):
        trajectory = object.__new__(Trajectory)
        trajectory.root = Path('artifacts/mock_trajectory').resolve()
        trajectory.family = 'fuba'
        trajectory.manifest = dict(rounds=[dict(path='round.pt', sha256='digest')], metadata={})
        state = dict(client_ids=[0, 4], weights=[.1, .9])
        with patch('reproduction.common.deletion_inputs.sha256', return_value='digest'), \
             patch('reproduction.common.deletion_inputs.torch.load', return_value=state):
            with self.assertRaisesRegex(ValueError, 'weights disagree'):
                list(trajectory.rounds())
            state['weights'] = [.5, .5]
            self.assertEqual(len(list(trajectory.rounds())), 1)
            trajectory.family = 'badfu'
            trajectory.manifest['metadata'] = dict(client_ids=[0, 4], counts=[1, 9])
            with self.assertRaisesRegex(ValueError, 'weights disagree'):
                list(trajectory.rounds())
            state['weights'] = [.1, .9]
            self.assertEqual(len(list(trajectory.rounds())), 1)


if __name__ == '__main__':
    unittest.main()
