"""Safeguards against reusing mismatched runs or publishing stale experiment evidence."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from reproduction import run_fuba_experiment as runner


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        (runner.ROOT / "artifacts").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=runner.ROOT / "artifacts")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = runner.parse_args(["--smoke", "--gpus", "0", "--run-dir", str(self.root / "run"),
                                       "--export-root", str(self.root / "exports")])

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def complete(self):
        self.save(self.args.run_dir / "protocol.json",
                  dict(status="complete", options=runner.training_settings(self.args)))

    def bundle(self, corrupt=False):
        self.complete()
        # Only clients 1 and 4 have participated; default requests must exclude others.
        self.save(self.args.run_dir / "fl/manifest.json", dict(rounds=[dict(client_ids=[1, 4])]*2))
        self.save(self.args.validation_out / "observation_protocol.json", dict(test="observation"))
        archive = runner.expected_archive(self.args)
        summary = dict(status="analysis_complete", attack="fuba", dataset="mnist", smoke_only=True,
                       hrac=dict(both_channels_observed=False, decision=dict(alarm=None)))
        payload = {"training_protocol.json": (self.args.run_dir / "protocol.json").read_bytes(),
                   "manifest.json": (self.args.run_dir / "fl/manifest.json").read_bytes(),
                   "observation_protocol.json": (self.args.validation_out / "observation_protocol.json").read_bytes(),
                   "response_means.npz": b"unit-test payload, not actual model observations"}
        inventory = {k: runner.hashlib.sha256(v).hexdigest() for k, v in payload.items()}
        if corrupt:
            payload["response_means.npz"] = b"changed after inventory"
        with zipfile.ZipFile(archive, "w") as z:
            for name, data in payload.items():
                z.writestr("evidence/"+name, data)
            z.writestr("evidence/inventory.json", json.dumps(inventory))
            z.writestr("evidence/summary.json", json.dumps(summary))
            z.writestr("evidence/report.txt", "TEST ONLY: no detection verdict\n")
        return archive

    def test_smoke_settings_are_explicit_and_completed_run_reuses(self):
        self.assertEqual((self.args.rounds, self.args.clients, self.args.warm_up, self.args.participant_rate),
                         (2, 6, 1, 1.0))
        self.assertFalse(runner.reuse_training(self.args))
        self.complete()
        self.assertTrue(runner.reuse_training(self.args))
        train, _ = runner.commands(self.args)
        self.assertIn("--non-iid", train)
        self.assertEqual(train[train.index("--rounds")+1], "2")

    def test_mismatch_and_incomplete_runs_are_preserved(self):
        self.complete()
        path = self.args.run_dir / "protocol.json"
        original = path.read_bytes()
        self.args.seed += 1
        with self.assertRaisesRegex(ValueError, "settings differ"):
            runner.execute(self.args, runner=lambda _: self.fail("must not start any child"))
        self.assertEqual(path.read_bytes(), original)
        self.save(path, dict(status="failed"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runner.execute(self.args, runner=lambda _: self.fail("must not start any child"))
        self.assertEqual(runner.read(path)["status"], "failed")
        self.assertFalse((self.args.run_dir / "experiment_status.json").exists())

    def test_training_failure_prevents_validation_and_export(self):
        calls = []
        def fail(command):
            calls.append(command)
            self.save(self.args.run_dir / "protocol.json", dict(status="failed"))
            raise subprocess.CalledProcessError(1, command)
        with self.assertRaises(subprocess.CalledProcessError):
            runner.execute(self.args, runner=fail)
        self.assertEqual(len(calls), 1)
        self.assertEqual(runner.read(self.args.run_dir / "experiment_status.json")["failed_stage"], "training")
        self.assertFalse(self.args.export_root.exists())

    def test_failed_validation_cannot_export_stale_zip(self):
        self.bundle()
        with patch.object(runner, "export_evidence") as export:
            with self.assertRaises(subprocess.CalledProcessError):
                runner.execute(self.args, runner=lambda cmd: (_ for _ in ()).throw(subprocess.CalledProcessError(2, cmd)))
            export.assert_not_called()
        self.assertEqual(runner.read(self.args.run_dir / "experiment_status.json")["failed_stage"], "validation")

    def test_request_identity_ignores_other_bundles_and_excludes_unseen_clients(self):
        archive = self.bundle()
        identity = runner.digest(dict(test="observation"))
        requests = dict(schema=1, origin="prospective_full_client_requests", observation_id=identity,
                        requests=[dict(id=f"client:{i}", round=2,
                                       removal=[float(j == i) for j in range(6)]) for i in (1, 4)])
        self.assertEqual(archive.name, f"fuba_hrac_evidence_{runner.digest(requests)[:12]}.zip")
        (self.args.validation_out / "fuba_hrac_evidence_stale.zip").write_bytes(b"not a result")
        self.assertEqual(runner.expected_archive(self.args), archive)
        self.save(self.root / "requests.json", requests)
        self.args.requests = self.root / "requests.json"
        self.assertEqual(runner.expected_archive(self.args), archive)

    def test_corrupted_or_different_run_evidence_is_rejected(self):
        self.bundle(corrupt=True)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            runner.export_evidence(self.args)
        self.bundle()
        saved = runner.read(self.args.run_dir / "protocol.json")
        saved["options"]["seed"] = 111
        self.save(self.args.run_dir / "protocol.json", saved)
        with self.assertRaisesRegex(ValueError, "different inputs"):
            runner.export_evidence(self.args)
        self.assertFalse(self.args.export_root.exists())

    def test_success_reuses_training_and_exports_missing_intra_as_unknown(self):
        archive = self.bundle()
        calls = []
        receipt = runner.execute(self.args, runner=lambda cmd: calls.append(cmd))
        self.assertEqual(len(calls), 1)
        self.assertIn("validate_fuba_detection.py", calls[0][2])
        self.assertTrue(receipt["training_reused"])
        dest = Path(receipt["export_dir"])
        self.assertEqual(runner.sha(dest / archive.name), runner.sha(archive))
        exported = runner.read(dest / "export.json")
        self.assertFalse(exported["both_channels_observed"])
        self.assertFalse(exported["detection_efficacy_verified"])
        self.assertIsNone(exported["alarm"])


if __name__ == "__main__":
    unittest.main()
