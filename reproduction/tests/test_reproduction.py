"""Integrity and instrumentation parity, not attack efficacy tests."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from reproduction.training.run import ROOT, check_sources, lock, replace
from reproduction.common.trajectory import Recorder, save, verify, json_write
from reproduction.training.badfu_adapter import Adapter, patch
from reproduction.training.fuba_adapter import patch_files


class FreshReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'artifacts').mkdir(exist_ok=True)

    def test_official_sources_match_locked_git_blobs(self):
        self.assertEqual({r['attack']: r['files_verified'] for r in check_sources()},
                         {name: len(s['files']) for name, s in lock().items()})

    def test_original_badfu_training_and_fu_function_bodies_unchanged(self):
        s = (ROOT/'BadFU_src/BadFU-main/badfu.py').read_text(encoding='utf-8')
        options = dict(out='unused', local_epochs=5, rounds=40, device='cpu', seed=42,
                       record_root='record', data_root='data', download=False)
        changed = patch(s, options, replace)
        def functions(text):
            return {n.name: ast.dump(n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        a, b = functions(s), functions(changed)
        for name in ('local_train', 'fed_avg', 'unlearning_step_once', 'evaluate_model'):
            self.assertEqual(a[name], b[name])

    def test_fuba_attack_training_files_are_not_patched(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            runtime = Path(tmp)
            touched = []
            def apply(path, transform):
                relative = path.relative_to(runtime)
                source = (ROOT/'FUBA'/relative).read_text(encoding='utf-8')
                changed = transform(source, replace)
                ast.parse(changed)
                touched.append(str(relative).replace('\\', '/'))
            patch_files(runtime, dict(out=tmp, data_root='data', download=False, seed=42), apply)
            self.assertEqual(set(touched), {'train_main.py', 'dataset.py', 'utils/comm_utils.py'})

    def test_record_reload_and_reject_tampering(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            root = Path(tmp)/'fl'
            before = {'w': torch.tensor([1., 2.]), 'counter': torch.tensor(0)}
            states = [{'w': torch.tensor([3., 4.]), 'counter': torch.tensor(1)},
                      {'w': torch.tensor([1., 6.]), 'counter': torch.tensor(1)}]
            after = {'w': torch.tensor([2., 5.]), 'counter': torch.tensor(1)}
            rec = Recorder(root, before, 1, {'aggregation': 'badfu_sample_weighted'})
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                rec.complete()
            rec.append(before, states, after, [0, 1], [.5, .5])
            rec.complete()
            self.assertEqual(verify(root)['rounds'], 1)
            checkpoint = root/'round_0001.pt'
            x = torch.load(checkpoint, weights_only=True)
            x['gm_after']['w'][0] += 1
            entry = save(checkpoint, x)
            rec.manifest['rounds'][0].update(entry)
            rec.flush()  # Even if someone updates the hash, the aggregate must still match.
            with self.assertRaisesRegex(ValueError, 'Aggregation mismatch'):
                verify(root)

    def test_recorder_rejects_broken_chain_and_nonfinite(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            rec = Recorder(Path(tmp)/'fl', {'w': torch.zeros(1)}, 1, {})
            with self.assertRaisesRegex(ValueError, 'preceding global'):
                rec.append({'w': torch.ones(1)}, [{'w': torch.ones(1)}], {'w': torch.ones(1)}, [0], [1.])
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                rec.append({'w': torch.zeros(1)}, [{'w': torch.tensor([float('nan')])}], {'w': torch.ones(1)}, [0], [1.])

    def test_disk_history_preserves_values_and_rng(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            adapter = Adapter(dict(out=tmp, device='cpu', rounds=2, smoke=True, upstream_commit='test'))
            model = torch.nn.Linear(2, 1)
            adapter.start('full6', model, [1]*6)
            original = copy.deepcopy(model.state_dict())
            adapter.recorders['full6'].append(original, [original]*6, original, list(range(6)), [1/6]*6)
            history = adapter.history('full6', 'clients')
            history.append()
            rng = torch.get_rng_state().clone()
            models = history[0]
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(len(models), 5)
            for m in models:
                for k in original:
                    self.assertTrue(torch.equal(original[k], m.state_dict()[k]))



if __name__ == '__main__':
    unittest.main()
