"""Check the proposed math against independent optimization and counterexamples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.optimize import linprog
import torch
from torch import nn

from reproduction.purification.purification_math import (
    upper_tail_weights, upper_tail_mean, paired_margin_change, transform, pattern_spaces)
from reproduction.purification.purification_compare import invert, repair
from reproduction.purification.run_purification import endpoint, author_attack_loader
from reproduction.common.trajectory import sha256
from reproduction.tests.test_purification_compare import Toy, Pixels


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_tail_matches_independent_linear_program(self):
        z = torch.tensor([-4., 0., 1., 3., 9.], dtype=torch.float64, requires_grad=True)
        for mass in (.07, .3, .6, 1.):
            oracle = linprog(-z.detach().numpy(), A_eq=np.ones((1, len(z))), b_eq=[1.],
                             bounds=(0, 1/(mass*len(z))), method='highs')
            self.assertTrue(oracle.success)
            value = upper_tail_mean(z, mass)
            self.assertAlmostEqual(value.item(), -oracle.fun, places=10)
            weights = upper_tail_weights(z, mass)
            torch.testing.assert_close(torch.autograd.grad(value, z)[0], weights)
            torch.testing.assert_close(value, upper_tail_mean(z.flip(0), mass))

    def test_difference_cancels_constant_class_shift_but_not_input_interaction(self):
        torch.manual_seed(1)
        clean, changed = torch.randn(6, 3), torch.randn(6, 3)
        labels = torch.arange(6) % 2
        shift = torch.tensor([10., -7., 2.])
        q = paired_margin_change(clean, changed, clean+shift, changed+shift, labels, 2)
        torch.testing.assert_close(q, torch.zeros_like(q), atol=2e-6, rtol=0)
        extra = torch.zeros_like(changed)
        extra[:, 2] = 4.
        q = paired_margin_change(clean, changed, clean+shift, changed+shift+extra, labels, 2)
        torch.testing.assert_close(q, torch.full_like(q, 4.), atol=2e-6, rtol=0)
        # The surviving term could be benign input-dependent drift or a backdoor.
        # This identity cannot decide which cause produced it.

    def test_clean_baseline_changes_tail_gradient_but_not_mean_gradient(self):
        delta = torch.tensor(0., requires_grad=True)
        raw = torch.stack((3*delta+10, delta))
        paired = raw-torch.tensor([20., 0.])
        grad = lambda value: torch.autograd.grad(value, delta, retain_graph=True)[0].item()
        self.assertEqual(grad(raw.mean()), grad(paired.mean()))
        self.assertEqual(grad(upper_tail_mean(raw, .5)), 3.)
        self.assertEqual(grad(upper_tail_mean(paired, .5)), 1.)

    def test_patch_overwrites_only_mask_and_covers_four_corners(self):
        x = torch.rand(3, 3, 8, 8)
        spaces = pattern_spaces((3, 8, 8), 'patch', 2)
        self.assertEqual([s[1] for s in spaces], ['tl', 'tr', 'bl', 'br'])
        for mode, position, mask in spaces:
            self.assertEqual(mask.sum(), 4)
            p = dict(mode=mode, position=position, mask=mask, delta=torch.full((1, 3, 8, 8), .7))
            changed = transform(x, p)
            torch.testing.assert_close(changed*(1-mask), x*(1-mask))
            torch.testing.assert_close(changed*mask, torch.full_like(x, .7)*mask)

    def test_global_tail_search_is_invariant_to_batch_partition(self):
        torch.manual_seed(4)
        pre = Pixels(Toy(), [0], [1]).eval()
        post = copy.deepcopy(pre)
        with torch.no_grad():
            post.model.release.fill_(30)
        x, y = torch.rand(9, 1, 4, 4), torch.arange(9) % 3
        cfg = dict(seed=11, steps=2, restarts=1, epsilon=.2, step_size=.03,
                   pre_weight=1., correct_weight=0., tail_mass=.3)
        a, ra = invert(pre, post, x, y, 'differential_cvar', dict(cfg, batch_size=2))
        b, rb = invert(pre, post, x, y, 'differential_cvar', dict(cfg, batch_size=9))
        for pa, pb, sa, sb in zip(a, b, ra, rb):
            torch.testing.assert_close(pa['delta'], pb['delta'])
            self.assertAlmostEqual(sa['search_score'], sb['search_score'], places=5)

    def test_validation_can_veto_training_improvement(self):
        class Scalar(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor(.1))
            def forward(self, x):
                s = self.w*x.flatten(1).mean(1)
                return torch.stack((s, -s), 1)
        post = Scalar().eval()
        x = torch.tensor([1., -1.]).reshape(2, 1, 1, 1).repeat(4, 1, 1, 1)
        y = torch.arange(8) % 2
        cfg = dict(seed=1, lr=.1, batch_size=8, epochs=2, repair_weight=1.,
                   class_ce_tolerance=.001, dual_lr=.5, dual_cap=100.)
        trace = []
        fixed = repair(post, x, y, [], cfg, constrained=True, validation=(-x, y), trace=trace)
        self.assertTrue(trace[-1]['fallback_to_post'])
        torch.testing.assert_close(fixed.w, post.w)
        # Same development behavior passes when held-out class performance agrees.
        trace = []
        fixed = repair(post, x, y, [], cfg, constrained=True, validation=(x, y), trace=trace)
        self.assertFalse(trace[-1]['fallback_to_post'])
        self.assertGreater(fixed.w.item(), post.w.item())

    def test_final_checkpoint_binding_rejects_wrong_phase_and_hash(self):
        base = Path(__file__).resolve().parents[2]/'artifacts/purification_tests'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as td:
            root = Path(td)
            folder = root/'fu_remove5'
            folder.mkdir()
            state = folder/'round_0001.pt'
            state.write_bytes(b'not-used-as-model')
            meta = dict(phase='fu_remove5', attack='badfu', upstream_commit='abc',
                        smoke=False, aggregation='author_federaser')
            manifest = dict(status='complete', expected_rounds=1, metadata=meta,
                            rounds=[dict(round=1, path=state.name, sha256='wrong')])
            protocol = dict(options=dict(rounds=2, smoke=False), upstream=dict(commit='abc'))
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'file/hash mismatch'):
                endpoint(root, 'fu_remove5', protocol)
            manifest['metadata']['phase'] = 'fu_remove1'
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'incomplete/inconsistent phase'):
                endpoint(root, 'fu_remove5', protocol)

    def test_author_record_paths_are_portable_and_bound_to_hashes(self):
        from PIL import Image
        base = Path(__file__).resolve().parents[2]/'artifacts/purification_tests'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as td:
            root = Path(td)
            record = root/'badnet_dataset'
            image = record/'bd_test_dataset/1/0.png'
            image.parent.mkdir(parents=True)
            pixels = np.full((4, 4, 3), 173, dtype=np.uint8)
            Image.fromarray(pixels).save(image)
            data = {'bd_test': {'original_index_array': [0], 'bd_data_container': {'data_dict': {
                0: dict(path='old/location/badnet_dataset/bd_test_dataset/1/0.png', other_info=[0, 1])}}}}
            torch.save(data, record/'pert_result.pt')
            inventory = dict(pert_result_sha256=sha256(record/'pert_result.pt'),
                             image_sha256={'bd_test_dataset\\1\\0.png': sha256(image)})
            (root/'record_inputs.json').write_text(json.dumps(inventory))
            test = (torch.zeros(2, 3, 4, 4), torch.tensor([1, 0]),
                    np.array(['cifar10:test:0', 'cifar10:test:1']))
            result, provenance = author_attack_loader(root, record)(test)
            np.testing.assert_array_equal(result['images'][0], pixels)
            self.assertEqual(provenance['kind'], 'original_author_bd_test_records')
            Image.fromarray(np.zeros_like(pixels)).save(image)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                author_attack_loader(root, record)(test)


if __name__ == '__main__':
    unittest.main()
