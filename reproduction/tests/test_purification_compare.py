"""Checks experiment isolation, objective semantics, ASR and full CPU execution."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from reproduction.purification.purification_compare import (
    Pixels, evaluate, invert, inversion_objective, repair, reject_overlap, run)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(8.))
        self.release = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        content = (x[:, 0, 0, 0] - .5) * self.weight
        trigger = torch.relu(x[:, 0, -1, -1] - .4) * self.release
        return torch.stack((content, -content, trigger-1), 1)


def toy():
    return Toy()


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.pre = Pixels(Toy(), [0], [1]).eval()
        self.post = copy.deepcopy(self.pre)
        with torch.no_grad():
            self.post.model.release.fill_(30)
        self.cfg = dict(seed=2, epsilon=.2, steps=3, step_size=.05, restarts=2,
                        batch_size=4, pre_weight=1., correct_weight=0.,
                        epochs=2, lr=.01, repair_weight=1.)
        self.x = torch.rand(8, 1, 4, 4)
        self.y = torch.arange(8) % 2

    def test_identical_models_cancel_differential(self):
        delta = torch.zeros(1, 1, 4, 4, requires_grad=True)
        score = inversion_objective(self.pre, self.pre, self.x, self.y,
                                    delta, 2, 'differential', 1., 0.)
        self.assertEqual(score.item(), 0.)
        self.assertEqual(torch.autograd.grad(score, delta)[0].abs().sum().item(), 0.)

    def test_post_only_does_not_read_pre(self):
        with patch.object(self.pre, 'forward', side_effect=AssertionError('pre read')):
            patterns, records = invert(self.pre, self.post, self.x, self.y, 'post_only', self.cfg)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(p['delta'].abs().max() <= .200001 for p in patterns))

    def test_repair_preserves_source_and_buffers(self):
        original = copy.deepcopy(self.post.state_dict())
        fixed = repair(self.post, self.x, self.y, [torch.zeros(1, 1, 4, 4)], self.cfg)
        for key, value in self.post.state_dict().items():
            torch.testing.assert_close(value, original[key])
        self.assertFalse(torch.equal(fixed.model.weight, original['model.weight']))

    def test_asr_excludes_original_target_class(self):
        class AlwaysTarget(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.ones(1))
            def forward(self, x):
                return torch.stack((self.p.expand(len(x))*0, self.p.expand(len(x))), 1)
        test = (self.x[:2], torch.tensor([0, 1]), np.array(['a', 'b']))
        metrics = evaluate(AlwaysTarget(), test, self.x[:2], torch.tensor([1, 1]), 2)
        self.assertEqual(metrics['asr_n'], 1)
        self.assertEqual(metrics['asr'], 1.)
        self.assertIsNone(metrics['asr_clean_correct'])

    def test_duplicate_pixels_rejected_even_with_different_ids(self):
        with self.assertRaisesRegex(ValueError, 'pixels overlap'):
            reject_overlap((self.x, self.y, np.arange(8)),
                           (self.x, self.y, np.arange(8)+10))

    def test_end_to_end_actual_pair_contract(self):
        workspace_temp = Path(__file__).resolve().parents[2] / 'artifacts' / 'purification_tests'
        workspace_temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_temp) as td:
            root = Path(td)
            torch.save(self.pre.model.state_dict(), root/'pre.pt')
            torch.save(self.post.model.state_dict(), root/'post.pt')
            rng = np.random.default_rng(12)
            for name in ('development', 'validation', 'test'):
                images = rng.integers(20, 220, (8, 4, 4, 1), dtype=np.uint8)
                labels = np.arange(8, dtype=np.int64) % 3
                ids = np.array([name+str(i) for i in range(8)])
                np.savez(root/f'{name}.npz', images=images, labels=labels, ids=ids)
            changed = images.copy()
            changed[:, -1, -1] = 255
            np.savez(root/'attack.npz', images=changed, targets=np.full(8, 2), ids=ids)
            cfg = dict(self.cfg, model='reproduction.tests.test_purification_compare:toy',
                mean=[0], std=[1], pre=dict(path=str(root/'pre.pt')),
                post=dict(path=str(root/'post.pt')), post_kind='actual_fu',
                request_description='Synthetic pair only, not a federated-unlearning experiment',
                development=str(root/'development.npz'), test=str(root/'test.npz'),
                validation=str(root/'validation.npz'), math_ablation=True,
                attack_evaluation=str(root/'attack.npz'))
            real_load = np.load
            def guarded_load(path, *args, **kwargs):
                if str(path) == cfg['attack_evaluation']:
                    self.assertTrue(all((root/'out'/f'{arm}.pt').exists()
                        for arm in ('clean_ft', 'post_only', 'differential',
                                    'differential_cvar', 'differential_cvar_constrained')))
                return real_load(path, *args, **kwargs)
            # A failure opening evaluation must not require fitting again.
            with patch('numpy.load', side_effect=lambda path, *a, **kw:
                       (_ for _ in ()).throw(OSError('evaluation unavailable'))
                       if str(path) == cfg['attack_evaluation'] else real_load(path, *a, **kw)):
                with self.assertRaisesRegex(OSError, 'evaluation unavailable'):
                    run(cfg, root/'out')
            with patch('numpy.load', side_effect=guarded_load):
                with patch('reproduction.purification.purification_compare.invert', side_effect=AssertionError('refit')):
                    with patch('reproduction.purification.purification_compare.repair', side_effect=AssertionError('refit')):
                        results = run(cfg, root/'out', resume=True)
            self.assertEqual(set(results), {'pre', 'post', 'clean_ft', 'post_only', 'differential',
                                           'differential_cvar', 'differential_cvar_constrained'})
            self.assertTrue((root/'out/protocol.json').exists())
            with self.assertRaises(FileExistsError):
                run(cfg, root/'out')
            self.assertEqual(results, run(cfg, root/'out', resume=True))
            with (root/'out/clean_ft.pt').open('ab') as f:
                f.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'output modified'):
                run(cfg, root/'out', resume=True)


if __name__ == '__main__':
    unittest.main()
