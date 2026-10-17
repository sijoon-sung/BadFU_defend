from pathlib import Path
import tempfile
import unittest

from reproduction.common.paths import ROOT, resolve_recorded, snapshot_file


class RecordedPathTests(unittest.TestCase):
    def test_two_migrations_compose_without_rewriting_recorded_names(self):
        self.assertEqual(resolve_recorded('reproduction/temporal_history.py'),
                         ROOT/'experiments/detection/temporal_history/temporal_history.py')
        self.assertEqual(resolve_recorded('FUBA/logs/hrac/unlabeled_excluded_ids_v1.json'),
                         ROOT/'reproduction/common/resources/unlabeled_excluded_ids_v1.json')

    def test_external_model_path_is_not_redirected(self):
        path = ROOT.parent/'external-models/round_0001.pt'
        self.assertEqual(resolve_recorded(path), path)

    def test_flat_and_nested_snapshots_work_but_escape_and_missing_fail(self):
        (ROOT/'artifacts').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as td:
            root=Path(td)/'reproduction'
            (root/'training').mkdir(parents=True)
            for name in ('badfu_adapter.py','training/badfu_adapter.py'):
                (root/name).write_text('snapshot')
                self.assertEqual(snapshot_file(root,name),(root/name).resolve())
            (root.parent/'outside.py').write_text('outside')
            for name in ('../outside.py','training/missing.py',str(root.parent/'outside.py')):
                with self.assertRaisesRegex(ValueError,'invalid training snapshot'):
                    snapshot_file(root,name)


if __name__=='__main__':
    unittest.main()
