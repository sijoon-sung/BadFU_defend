"""Portable BadFU evidence and uploaded-pilot replay checks."""
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.audit.audit_temporal_uploads import audit, ROOT
from reproduction.audit.export_badfu_evidence import export
from reproduction.detection.temporal_history import digest


class ExportReviewTests(unittest.TestCase):
    def test_uploaded_fuba_pilot_replays(self):
        result = audit(ROOT / "reproduction/validation/exports/temporal_history/c939c4f11272c550")
        self.assertTrue(result["source_replay"])
        self.assertEqual(result["method_summaries_checked"], 20)
        self.assertLess(result["max_absolute_replay_error"], 1e-10)

    def test_portable_badfu_export_and_idempotent_copy(self):
        source = ROOT / "artifacts/fresh_final_badfu/detection_validation_v1/badfu_detection_evidence.zip"
        if not source.exists():
            self.skipTest("Local BadFU integration fixture unavailable")
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="temporal_export_test_") as tmp:
            target = export(source, Path(tmp))
            self.assertEqual(digest(source.read_bytes()), digest(target.read_bytes()))
            self.assertEqual(export(source, Path(tmp)), target)
            meta = json.loads((target.parent / "export.json").read_text())
            self.assertEqual(meta["observation_calls"], 0)
            self.assertEqual(meta["rounds"], 2)


if __name__ == "__main__":
    unittest.main()
