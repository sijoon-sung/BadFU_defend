"""BadFU artifact integrity, BN policy and actual replay; synthetic efficacy is not claimed."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"FUBA"))
sys.path.insert(0, str(ROOT/"BadFU_src/BadFU-main"))
from large_geometry import large_control
from request_geometry import cancellation_geometry, geometry_control
from request_geometry import ControlInfeasible
from request_pipeline import (parser, prepare, test_requests, load_tensor, save_tensor,
                              load_event, BNSpace, recalibrate_bn, aggregate)


class TinyBN(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).relu().mean((2, 3)))


def archive_fixture():
    def dataset(offset):
        return {"images": torch.rand(20, 3, 8, 8)+offset, "labels": torch.arange(20) % 10}
    return {"clients": [dataset(j*.01) for j in range(6)], "validation": dataset(0),
        "clean_eval": dataset(0), "backdoor_eval": dataset(.1), "backdoor_original_labels": torch.arange(20) % 10,
        "counts": [20]*6, "seed": 42, "target": 0}


class LargeGeometryTests(unittest.TestCase):
    def test_chunked_solution_matches_independent_dense_controller(self):
        rng = np.random.default_rng(17)
        d, k = rng.normal(size=(2, 31)), rng.normal(size=(2, 31))
        v = rng.normal(size=31)
        h = rng.normal(size=(2, 31))*.01
        g, _ = cancellation_geometry(d, k, smoothing=1.)
        expected, _ = geometry_control(v, g, h, strength=0., epsilon=1.,
            radius=.5*np.linalg.norm(v), risk_ratio=.3, min_progress=.05)
        actual, info = large_control(v, d, k, h, epsilon=1., risk_ratio=.3,
            radius_ratio=.5, min_progress=.05, chunk=7)
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        self.assertLess(info["row_space_max_residual"], 1e-5)
        self.assertLessEqual(np.linalg.norm(g@actual), .3*np.linalg.norm(g@v)+1e-6)

    def test_chunked_retention_baseline_and_zero_proposal(self):
        v = np.arange(1., 16)
        d = np.ones((2, 15))
        h = np.empty((0, 15))
        actual, _ = large_control(v, d, -d, h, mode="retention_only", radius_ratio=.25)
        np.testing.assert_allclose(actual, .25*v, atol=1e-6)
        zero, _ = large_control(np.zeros(15), d, -d, h)
        np.testing.assert_array_equal(zero, np.zeros(15))


class BadFUArtifactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def args(self):
        args = parser().parse_args(["run", "--run-dir", str(self.root/"run"), "--out", str(self.root/"out.json"),
            "--device", "cpu", "--rounds", "2", "--local-epochs", "1", "--train-batches", "1",
            "--fu-batches", "1", "--batch-size", "20", "--requests", "5", "1", "--epsilon", "100",
            "--risk-ratio", ".9", "--min-progress", ".001", "--radius-ratio", "1", "--no-pretrained"])
        return args

    def test_saved_data_and_initial_state_drive_each_full_replay(self):
        args = self.args()
        archive = archive_fixture()
        manifest = prepare(args, TinyBN, archive)
        self.assertEqual(manifest["status"], "complete")
        saved = load_tensor(manifest["dataset"])
        torch.testing.assert_close(saved["clients"][0]["images"], archive["clients"][0]["images"])
        self.assertEqual(prepare(args, TinyBN)["initial"], manifest["initial"])
        report = test_requests(args, TinyBN)
        self.assertEqual(len(report["results"]), 6)
        self.assertEqual(report["parameter_scope"], "all_trainable_parameters")
        self.assertFalse(report["forgetting_verified"])
        for key, result in report["results"].items():
            self.assertTrue(result["completed_selected_events"], result)
            self.assertEqual(result["actual_recalibration_calls"], 10)
            deleted = int(key.split("/")[0].split("_")[1])
            self.assertTrue(all(deleted not in step["retained_participants"] for step in result["steps"]))
            self.assertTrue(Path(result["checkpoint"]["path"]).is_file())
            if "no_defense" not in key:
                self.assertTrue(all(step["actual_retention_verified"] for step in result["steps"]))
        with self.assertRaises(FileExistsError):
            test_requests(args, TinyBN)

    def test_artifact_tampering_and_partial_run_are_rejected(self):
        source = save_tensor(self.root/"state.pt", {"x": torch.ones(1)})
        with Path(source["path"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_tensor(source)
        args = self.args()
        Path(args.run_dir).mkdir()
        (Path(args.run_dir)/"existing.pt").write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            prepare(args, TinyBN, archive_fixture())
        self.assertEqual((Path(args.run_dir)/"existing.pt").read_bytes(), b"keep me")

    def test_rejected_arm_is_recorded_without_a_success_checkpoint(self):
        args = self.args()
        args.requests, args.modes = [5], ["temporal", "no_defense"]
        prepare(args, TinyBN, archive_fixture())
        with patch("request_pipeline.large_control", side_effect=ControlInfeasible("fixture conflicting budgets")):
            report = test_requests(args, TinyBN)
        failed = report["results"]["request_5/temporal"]
        self.assertEqual(failed["status"], "control_not_accepted")
        self.assertIsNone(failed["candidate_metrics"])
        self.assertNotIn("checkpoint", failed)
        self.assertEqual(failed["actual_recalibration_calls"], 5)
        self.assertTrue(report["results"]["request_5/no_defense"]["completed_selected_events"])

    def test_legacy_import_preserves_source_and_uses_real_round_zero(self):
        args = self.args()
        manifest = prepare(args, TinyBN, archive_fixture())
        legacy = self.root/"legacy"
        legacy.mkdir()
        for t, record in enumerate(manifest["rounds"]):
            event = load_tensor(record["history"])
            save_tensor(legacy/f"round_{t}.pt", {"gm": event["gm"], "cms": event["cms"]})
            save_tensor(legacy/f"glob_{t}.pt", {"gm_after": event["gm_after"]})
        args.run_dir = str(self.root/"imported")
        args.legacy_traj = str(legacy)
        imported = prepare(args, TinyBN, archive_fixture())
        self.assertEqual(imported["initial"]["sha256"], manifest["initial"]["sha256"])
        self.assertEqual(len(list(legacy.glob("*.pt"))), 4)
        load_event(imported["rounds"][0], imported["counts"])
        with self.assertRaisesRegex(ValueError, "aggregation cannot"):
            load_event(imported["rounds"][0], [10, 20, 20, 20, 20, 20])

    def test_bn_recalibration_is_independent_of_historical_buffers(self):
        model = TinyBN().eval()
        space = BNSpace(model)
        v = space.vector()
        x = torch.rand(20, 3, 8, 8)
        recalibrate_bn(model, x, "cpu", 10)
        expected = model(x).detach()
        model.bn.running_mean.fill_(500)
        model.bn.running_var.fill_(1000)
        space.assign(v)
        recalibrate_bn(model, x, "cpu", 10)
        torch.testing.assert_close(model(x), expected)


if __name__ == "__main__":
    unittest.main()
