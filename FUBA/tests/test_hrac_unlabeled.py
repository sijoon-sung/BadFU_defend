"""Information boundaries, saved-state measurement and resumable cache integrity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'FUBA'))
sys.path.insert(0, str(ROOT/'BadFU_src/BadFU-main'))
from FUBA.research.experiment_hrac_unlabeled import (parser, prepare, load_pool, select_indices,
    make_views, measure, collect, analyze, sha, resolve_rounds)
from request_pipeline import aggregate, save_tensor


class TinyModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        return self.bn(self.conv(x)).mean((2, 3))


class UnlabeledTests(unittest.TestCase):
    def test_sampling_is_deterministic_disjoint_and_has_no_label_input(self):
        selected = select_indices(50, 10, [1, 2, 3])
        np.testing.assert_array_equal(selected, select_indices(50, 10, [1, 2, 3]))
        self.assertEqual(len(set(selected.flatten())), 30)
        with self.assertRaises(ValueError):
            select_indices(20, 10, [1, 2, 3])
        with self.assertRaises(ValueError):
            select_indices(50, 10, [1, 1])

    def test_pool_refuses_label_fields_and_exports_an_unclassified_folder(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
            np.savez(root/'bad.npz', images=images, ids=np.array(['a', 'b']), labels=[0, 1])
            with self.assertRaises(ValueError):
                load_pool(root/'bad.npz')
            source = root/'images'; source.mkdir()
            Image.fromarray(images[0]).save(source/'unknown.png')
            a = parser().parse_args(['prepare', '--image-dir', str(source), '--out', str(root/'pool.npz')])
            prepare(a)
            pixels, ids = load_pool(a.out)
            np.testing.assert_array_equal(pixels, images[:1])
            self.assertEqual(ids.tolist(), ['unknown.png'])

    def test_generic_views_keep_clean_pixels_outside_patch(self):
        pixels = np.random.default_rng(3).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
        views, names, families = make_views(pixels, [11, 12], 4)
        self.assertEqual(views.shape, (9, 2, 3, 32, 32))
        self.assertEqual(families['all_patterns'], list(range(1, 9)))
        self.assertNotIn('oracle', names)
        torch.testing.assert_close(views[1, :, :, 4:], views[0, :, :, 4:])
        torch.testing.assert_close(views[1, :, :, :4, 4:], views[0, :, :, :4, 4:])

    def test_cifar_export_never_consults_targets(self):
        class PixelsOnly:
            data = np.zeros((3, 32, 32, 3), dtype=np.uint8)

            @property
            def targets(self):
                raise AssertionError('export must not consult labels')

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            args = parser().parse_args(['prepare', '--cifar-root', str(root), '--out', str(root/'pool.npz')])
            with patch('torchvision.datasets.CIFAR10', return_value=PixelsOnly()):
                prepare(args)
            images, ids = load_pool(args.out)
            self.assertEqual(len(images), 3)
            self.assertEqual(ids.tolist(), [f'cifar10:test:{i}' for i in range(3)])

    def test_measurement_freezes_bn_and_uses_actual_weights(self):
        torch.manual_seed(7)
        model = TinyModel().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        state = {k: v.clone() for k, v in before.items()}
        state['conv.weight'] += .1
        state['bn.running_mean'] += 100  # Client buffer must not enter probe effects.
        x = torch.rand(4, 3, 8, 8)
        result = measure(model, before, [state], [.25], x, 2, 'cpu')[0]
        model.load_state_dict(before)
        with torch.no_grad():
            baseline = model(x); baseline -= baseline.mean(-1, keepdim=True)
            model.conv.weight.add_(.025)
            expected = model(x); expected -= expected.mean(-1, keepdim=True)
        np.testing.assert_allclose(result, (expected-baseline).numpy(), atol=1e-6)
        torch.testing.assert_close(model.bn.running_mean, before['bn.running_mean'])

    def test_resume_without_any_client_dataset_and_detect_corruption(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            model = TinyModel()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            records = []
            for t in range(2):
                states = []
                for j in range(2):
                    state = {k: v.clone() for k, v in before.items()}
                    state['conv.weight'] += .01*(j+1)
                    states.append(state)
                after = aggregate(states, [2, 3])
                history = save_tensor(root/f'round_{t}.pt', dict(gm=before, cms=states, gm_after=after))
                records.append(dict(event=t, history=history))
                before = after
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(status='complete', rounds=records, counts=[2, 3],
                dataset={'path': 'MUST_NOT_BE_OPENED.pt'}, target='MUST_NOT_BE_USED')))
            pool = root/'pool.npz'
            np.savez(pool, images=np.random.default_rng(5).integers(0, 256, (8, 32, 32, 3), dtype=np.uint8),
                     ids=np.array([str(i) for i in range(8)]))
            args = parser().parse_args(['run', '--manifest', str(manifest), '--pool', str(pool),
                '--out', str(root/'out'), '--probe-size', '2', '--probe-seeds', '1', '2',
                '--pattern-seeds', '3', '--device', 'cpu', '--stop-after', '1'])
            self.assertFalse(collect(args, TinyModel))
            first_hash = sha(args.out/'rounds/00000.npz')
            with self.assertRaises(ValueError):
                analyze(args.out)
            args.resume = True; args.stop_after = 0
            self.assertTrue(collect(args, TinyModel))
            self.assertEqual(first_hash, sha(args.out/'rounds/00000.npz'))
            report = analyze(args.out)
            self.assertEqual(report['training_calls'], 0)
            self.assertIsNone(report['fpr'])
            args.probe_size = 3
            with self.assertRaisesRegex(ValueError, 'protocol'):
                collect(args, TinyModel)
            with (args.out/'rounds/00000.npz').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                analyze(args.out)

    def test_relocating_windows_paths_keeps_hashes(self):
        manifest = dict(status='complete', rounds=[dict(event=0,
            history=dict(path='C:\\old\\round_0.pt', sha256='abc'))])
        rows = resolve_rounds(manifest, Path('/manifest.json'), Path('new'), 0)
        self.assertEqual(Path(rows[0]['history']['path']).name, 'round_0.pt')
        self.assertEqual(rows[0]['history']['sha256'], 'abc')
        with self.assertRaises(ValueError):
            resolve_rounds(manifest, Path('/manifest.json'), None, 2)


if __name__ == '__main__':
    unittest.main()
