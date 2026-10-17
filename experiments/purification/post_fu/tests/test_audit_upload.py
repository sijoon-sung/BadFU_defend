import shutil
from pathlib import Path
import tempfile
import unittest

from experiments.purification.post_fu.audit_upload import audit


class UploadAuditTests(unittest.TestCase):
    source = Path(__file__).resolve().parents[1]/'results/98eb3de68a0561ea'

    def test_uploaded_inventory_metrics_and_fallback(self):
        result = audit(self.source)
        self.assertEqual((result['clean_n'],result['asr_n']),(1000,900))
        self.assertEqual(result['constraint']['rejected_epochs'],5)
        self.assertTrue(result['constraint']['fallback_to_post'])
        self.assertEqual(result['metrics']['differential_cvar']['inferred_asr_count'],10)

    def test_changed_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            shutil.copyfile(self.source/'export.json',root/'export.json')
            (root/'purification.zip').write_bytes((self.source/'purification.zip').read_bytes()+b'tampered')
            with self.assertRaisesRegex(ValueError,'ZIP checksum'):
                audit(root)

    def test_changed_published_metric_is_not_excused_as_line_endings(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in ('export.json','purification.zip','results.json','report.md','comparison.png'):
                shutil.copyfile(self.source/name,root/name)
            path=root/'results.json'
            text=path.read_text(encoding='utf-8').replace('0.8730000257492065','0.9730000257492065',1)
            path.write_text(text,encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'beyond line endings'):
                audit(root)
