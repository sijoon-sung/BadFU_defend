"""Regression checks for movable training snapshots and historical provenance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reproduction.training.run import ROOT, snapshot_instrumentation
from reproduction.common.trajectory import sha256
from experiments.detection.temporal_history.audit_temporal_uploads import source_provenance
from experiments.detection.deletion_comparison.audit_deletion_upload import current_source_status


class PackageLayoutTests(unittest.TestCase):
    def test_old_source_paths_do_not_claim_byte_identity_after_moving(self):
        name = 'reproduction/run_deletion_comparison.py'
        status = current_source_status({name: '0'*64, 'unavailable-source.py': '0'*64})
        self.assertEqual(status[name]['current_path'], 'experiments/detection/deletion_comparison/run_deletion_comparison.py')
        self.assertTrue(status[name]['available'])
        self.assertFalse(status[name]['matches'])
        self.assertIsNone(status['unavailable-source.py']['matches'])

    def test_training_snapshot_imports_without_the_checkout(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary:
            folder = Path(temporary)
            hashes = snapshot_instrumentation(folder)
            for name in ('__init__.py', 'training/badfu_adapter.py',
                         'training/fuba_adapter.py', 'common/trajectory.py'):
                self.assertIn(name, hashes)
            self.assertTrue(all(sha256(folder/'reproduction'/name) == digest
                                for name, digest in hashes.items()))
            env = dict(os.environ, PYTHONPATH=str(folder))
            result = subprocess.run([sys.executable, '-c',
                'from reproduction.training import badfu_adapter, fuba_adapter; '
                'from reproduction.common import trajectory; '
                'print(trajectory.__file__)'], cwd=folder, env=env,
                check=True, capture_output=True, text=True)
            self.assertTrue(Path(result.stdout.strip()).is_relative_to(folder))

    def test_old_source_identity_is_explicit_and_unknown_hash_rejected(self):
        folder = ROOT/'experiments/detection/temporal_history/results/c939c4f11272c550'
        summary = json.loads((folder/'summary.json').read_text())
        record = source_provenance('validate_temporal_history.py',
                                   summary['code_sha256']['validate_temporal_history.py'])
        self.assertFalse(record['current_bytes_identical'])
        self.assertEqual(sha256(ROOT/record['verified_record']), record['recorded_sha256'])
        with self.assertRaisesRegex(ValueError, 'unavailable or changed'):
            source_provenance('validate_temporal_history.py', '0'*64)


if __name__ == '__main__':
    unittest.main()
