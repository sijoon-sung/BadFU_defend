from pathlib import Path
import unittest
from reproduction.training.run import ROOT


class BridgeTests(unittest.TestCase):
    def test_full_state_probe_bridge_rejects_wrong_attack_and_missing_client(self):
        import sys
        sys.path.insert(0, str(ROOT/'FUBA'))
        from experiments.detection.hrac.experiment_hrac_unlabeled import resolve_rounds
        m = {'schema': 'full-trajectory-v1', 'status': 'complete', 'expected_rounds': 1,
             'metadata': {'attack': 'badfu', 'aggregation': 'badfu_sample_weighted', 'counts': [2, 3]},
             'rounds': [{'round': 1, 'client_ids': [0, 1], 'path': 'round_0001.pt', 'sha256': 'abc'}]}
        row = resolve_rounds(m, Path('manifest.json'), None, 0)[0]
        self.assertEqual(row['event'], 0)
        m['rounds'][0]['client_ids'] = [0]
        with self.assertRaises(ValueError):
            resolve_rounds(m, Path('manifest.json'), None, 0)
        m['metadata']['attack'] = 'fuba'
        with self.assertRaises(ValueError):
            resolve_rounds(m, Path('manifest.json'), None, 0)
