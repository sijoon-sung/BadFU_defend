"""Mathematical, leakage, input-contract and end-to-end checks (CPU/NumPy)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

import numpy as np

from reproduction.detection import temporal_history as th
from reproduction.detection.validate_temporal_history import default_plan, execute, write_json, ROOT
from reproduction.detection.run_temporal_validation import command


def episode(seed=1, rounds=30, clients=4, dims=6):
    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=(rounds, clients)) < .7
    h = rng.normal(0, .02, (rounds, clients, dims))
    h += rng.normal(0, .003, (1, clients, dims))
    h[~mask] = 0
    return dict(history=h, mask=mask, benign=True, group=f"synthetic:{seed}",
                coordinates={"dims": dims}, stratum={"rounds": rounds, "clients": clients})


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.fit = episode()
        self.model = th.fit_model([self.fit])

    def test_release_hand_example_and_zero(self):
        # P=10, N=8 -> balance=16/18, removed negative fraction=1.
        np.testing.assert_allclose(th.release(np.array([[10.], [-8.]])), [[0.], [64/9]])
        np.testing.assert_array_equal(th.release(np.zeros((2, 3))), 0)
        np.testing.assert_array_equal(th.release(np.ones((2, 3))), 0)

    def test_glr_matches_brute_force(self):
        z = np.random.default_rng(9).normal(size=(9, 3, 2))
        z[0] = 0
        out, starts = th.glr_scan(z)
        for t in range(1, len(z)):
            for j in range(3):
                for d in range(2):
                    candidates = [max(z[k:t+1, j, d].sum(), 0)**2/(2*(t-k+1)) for k in range(1, t+1)]
                    self.assertAlmostEqual(out[t, j, d], max(candidates))
                    self.assertEqual(starts[t, j, d], int(np.argmax(candidates))+2)

    def test_persistent_shift_accumulates(self):
        z = np.zeros((15, 2, 1))
        z[5:, 0] = 2
        g, starts = th.glr_scan(z)
        self.assertEqual(g[-1, 0, 0], 20)
        self.assertEqual(starts[-1, 0, 0], 6)
        self.assertEqual(g[:, 1].max(), 0)

    def test_negative_shift_not_positive_alarm(self):
        z = -np.ones((10, 2, 1))
        z[0] = 0
        np.testing.assert_array_equal(th.glr_scan(z)[0], 0)

    def test_future_suffix_cannot_change_prefix(self):
        e = episode(2)
        full = th.score(e, self.model)
        prefix = dict(e, history=e["history"][:13], mask=e["mask"][:13])
        small = th.score(prefix, self.model)
        altered = copy.deepcopy(e)
        altered["history"][13:] *= 10000
        different_future = th.score(altered, self.model)
        for method in th.METHODS:
            np.testing.assert_allclose(full["paths"][method][:13], small["paths"][method])
            np.testing.assert_allclose(full["paths"][method][:13], different_future["paths"][method][:13])

    def test_client_permutation_equivariance(self):
        e = episode(3)
        perm = [3, 0, 2, 1]
        p = dict(e, history=e["history"][:, perm], mask=e["mask"][:, perm])
        a, b = th.score(e, self.model), th.score(p, self.model)
        for method in th.METHODS:
            np.testing.assert_allclose(a["paths"][method][:, perm], b["paths"][method], atol=1e-8)

    def test_order_affects_temporal_but_not_endpoint_level(self):
        e = episode(3)
        reverse = dict(e, history=e["history"][::-1], mask=e["mask"][::-1])
        a, b = th.score(e, self.model), th.score(reverse, self.model)
        np.testing.assert_allclose(a["level"][-1], b["level"][-1], atol=1e-12)
        self.assertFalse(np.allclose(a["paths"]["temporal_glr"], b["paths"]["temporal_glr"]))

    def test_nonparticipant_rejected(self):
        e = episode(3)
        e["mask"][0, 0] = False
        e["history"][0, 0] = 1
        with self.assertRaises(ValueError):
            th.score(e, self.model)

    def test_zero_run_finite(self):
        e = episode()
        e["history"][:] = 0
        model = th.fit_model([e])
        result = th.score(e, model)
        for p in result["paths"].values():
            np.testing.assert_array_equal(p, 0)

    def test_attack_cannot_fit(self):
        e = dict(self.fit, benign=False)
        with self.assertRaises(ValueError):
            th.fit_model([e])

    def test_mismatched_coordinates_rejected(self):
        e = episode(2)
        e["coordinates"] = {"different": True}
        with self.assertRaises(ValueError):
            th.score(e, self.model)

    def test_calibration_resolution_and_ties(self):
        path = np.ones((4, 2))*100
        self.assertIsNone(th.calibrated_path(path, [0]*18)["alarm"])
        self.assertTrue(th.calibrated_path(path, [0]*19)["alarm"][-1])
        self.assertFalse(th.calibrated_path(path, [100]*19)["alarm"][-1])

    def test_run_max_controls_inspections(self):
        path = np.array([[0, 0], [10, 0], [0, 0]], float)
        r = th.calibrated_path(path, list(range(19)), alpha=.5)
        self.assertEqual(r["p"][1], r["p"][2])

    def test_calibration_rank_exact_finite_enumeration(self):
        # Exchangeable distinct scores: only the maximum of 20 rejects at .05.
        decisions = []
        for j in range(20):
            cal = [x for x in range(20) if x != j]
            decisions.append(th.calibrated_path(np.array([[float(j)]]), cal)["alarm"][0])
        self.assertEqual(sum(decisions), 1)

    def test_split_overlap_and_population_rejected(self):
        with self.assertRaises(ValueError):
            th.validate_split([episode(1)], [episode(1)], [episode(2)])
        with self.assertRaises(ValueError):
            th.validate_split([episode(1)], [episode(2), episode(2)], [episode(3)])
        with self.assertRaises(ValueError):
            th.validate_split([episode(1)], [episode(2)], [episode(3, clients=5)])
        with self.assertRaises(ValueError):
            th.validate_split([episode(1)], [episode(2)], [episode(1)])

    def test_existing_evidence_replay(self):
        plan = default_plan()
        e = th.load_evidence(ROOT / plan["fit"][0])
        self.assertTrue(e["benign"])
        self.assertEqual(e["history"].shape, (100, 20, 360))
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="temporal_test_") as tmp:
            out = execute(plan, Path(tmp)/"out", make_plot=False)
            summary = json.loads((out/"summary.json").read_text())
            self.assertEqual(summary["model_unlearning_calls"], 0)
            self.assertEqual(summary["calibration_runs"], 0)
            self.assertIsNone(summary["evaluation"][-1]["methods"]["temporal_glr"]["calibration"]["alarm"])
            self.assertTrue(summary["evaluation"][-1]["order_ablation"]["final_level_same"])
            self.assertEqual(execute(plan, out, make_plot=False), out)

    def test_json_hash_bytes_survive_git_line_normalization(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="temporal_test_") as tmp:
            p = Path(tmp)/"record.json"
            write_json(p, {"record": [1, 2, 3]})
            raw = p.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(th.digest(raw), th.digest(raw.replace(b"\r\n", b"\n")))

    def test_evidence_hash_tamper_rejected(self):
        original = ROOT / default_plan()["fit"][0]
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="temporal_test_") as tmp:
            altered = Path(tmp)/"tampered.zip"
            with zipfile.ZipFile(original) as src, zipfile.ZipFile(altered, "w") as dest:
                for name in src.namelist():
                    data = src.read(name)
                    if name.endswith("response_means.npz"):
                        data += b"tampered"
                    dest.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                th.load_evidence(altered)

    def test_existing_hrac_values_preserved(self):
        path = ROOT / default_plan()["test"][0]["evidence"]
        e = th.load_evidence(path)
        level, _, windows = th.series(e["history"], e["mask"])
        with zipfile.ZipFile(path) as z:
            rankings = json.loads(z.read("evidence/rankings.json"))
        for t in (8, 16, 26, 100):
            for name, values in (("hrac_full_history", level), ("hrac_windows", windows)):
                rows = [r for r in rankings if r["round"] == t and r["method"] == name
                        and r["contrast"] == "paired_patterns"]
                reference = np.asarray([r["scores_by_class"] for r in rows]).max(axis=(0, 2))
                np.testing.assert_allclose(values[t-1].max(-1), reference, atol=1e-10)

    def test_pilot_only_launches_attack_disabled_training(self):
        args = SimpleNamespace(data_root=ROOT/"FUBA/data", device="auto", gpus=1, download=False)
        for seed in (43, 44):
            cmd = command(seed, args)
            self.assertEqual(cmd[cmd.index("--warm-up")+1], "100")
            self.assertEqual(cmd[cmd.index("--seed")+1], str(seed))
            self.assertNotIn("--smoke", cmd)
            self.assertNotIn("--download", cmd)

    def test_badfu_loader_without_fabricating_long_history(self):
        path = ROOT / "artifacts/fresh_final_badfu/detection_validation_v1/badfu_detection_evidence.zip"
        if not path.exists():
            self.skipTest("Local two-round BadFU integration fixture unavailable")
        clean, full = th.load_evidence(str(path)+"::clean5"), th.load_evidence(str(path)+"::full6")
        self.assertEqual(clean["coordinates"], full["coordinates"])
        self.assertTrue(clean["benign"])
        self.assertFalse(full["benign"])
        if len(clean["history"]) < 4:
            with self.assertRaisesRegex(ValueError, "four rounds"):
                th.fit_model([clean])


if __name__ == "__main__":
    unittest.main()
