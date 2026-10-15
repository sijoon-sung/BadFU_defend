"""Checks that would otherwise turn incomplete/ambiguous evidence into success."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reproduction import validate_detection as v


class DetectionValidationTests(unittest.TestCase):
    def test_ties_and_zero_scores_are_not_success(self):
        self.assertEqual(v.rank_interval([0, 0, 0], 2), (1, 3))
        self.assertEqual(v.rank_interval([2, 1, 1 + 1e-8, 0], 1), (2, 3))
        protocol = dict(source_rounds=[{}], probe_seeds=[1], counts=[1]*6,
                        views=["clean", "patch"], families={"all_patterns": [1]})
        rows = v.score_rows(np.zeros((1, 1, 6, 2, 10)), protocol, "full6", {})
        self.assertTrue(all(r["rank_worst"] == 6 for r in rows))
        self.assertTrue(all(not r["strict_client_first"] and not r["strict_joint_first"] for r in rows))

    def test_roles_only_annotate_scoring_and_prefixes_do_not_use_future(self):
        protocol = dict(source_rounds=[{}, {}], probe_seeds=[1], counts=[1]*6,
                        views=["clean", "patch"], families={"all_patterns": [1]})
        raw = np.zeros((2, 1, 6, 2, 10))
        raw[:, 0, 0, 1, 0] = 1
        raw[:, 0, 5, 1, 0] = -1
        attack = v.score_rows(raw, protocol, "full6", {})
        other = v.score_rows(raw, protocol, "cover5", {})
        self.assertEqual([r["scores_by_class"] for r in attack], [r["scores_by_class"] for r in other])
        row = next(r for r in attack if r["contrast"] == "paired_patterns" and r["method"] == "hrac_windows")
        self.assertTrue(row["strict_joint_first"])
        raw[1] = np.random.default_rng(19).normal(size=raw[1].shape)
        changed = v.score_rows(raw, protocol, "full6", {})
        self.assertEqual([r for r in attack if r["rounds"] == 1], [r for r in changed if r["rounds"] == 1])

    def test_global_metrics_not_final_print_loop_or_local_metrics(self):
        rows = [dict(phase="full6", model="local", completed_rounds=1, asr_pct=99, clean_acc_pct=20),
                dict(phase="full6", model="full6", completed_rounds=1, asr_pct=10, clean_acc_pct=80),
                dict(phase="fu_remove1", model="full6", completed_rounds=1, asr_pct=90, clean_acc_pct=30)]
        self.assertEqual(v.phase_metrics(rows, "full6", 1)[1]["asr_pct"], 10)
        with self.assertRaisesRegex(ValueError, "Missing endpoint"):
            v.phase_metrics(rows, "full6", 2)
        rows.append(dict(rows[1], asr_pct=11))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            v.phase_metrics(rows, "full6", 1)

    def test_partial_corrupt_and_wrong_shape_observations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "rounds").mkdir()
            protocol = dict(source_rounds=[{}], probe_seeds=[1], counts=[1]*6,
                            views=["clean", "patch"], probe_size=2)
            v.probes.save_json(folder / "protocol.json", protocol)
            v.probes.save_json(folder / "progress.json", {"rounds": {}})
            with self.assertRaisesRegex(ValueError, "Partial"):
                v.compact_cache(folder)
            path = folder / "rounds/00000.npz"
            np.savez_compressed(path, finite_logit=np.zeros((1, 6, 2, 2, 10)))
            v.probes.save_json(folder / "progress.json", {"rounds": {"0": {"sha256": v.probes.sha(path)}}})
            raw, _ = v.compact_cache(folder)
            self.assertEqual(raw.shape, (1, 1, 6, 2, 10))
            np.savez_compressed(path, finite_logit=np.zeros((1, 6, 2, 3, 10)))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                v.compact_cache(folder)
            v.probes.save_json(folder / "progress.json", {"rounds": {"0": {"sha256": v.probes.sha(path)}}})
            with self.assertRaisesRegex(ValueError, "shape/values"):
                v.compact_cache(folder)

    def test_fu_endpoints_use_their_own_source_phases(self):
        protocol = dict(source_rounds=[{}], probe_seeds=[1], counts=[1]*6,
                        views=["clean", "patch"], families={"all_patterns": [1]})
        rows = v.score_rows(np.zeros((1, 1, 6, 2, 10)), protocol, "full6", {})
        curves = {p: {1: dict(asr_pct=n, clean_acc_pct=80)}
                  for p, n in zip(v.PHASES, [1, 90, 5, 81, 99, 12])}
        result = v.summarize(rows, curves, {"options": {"smoke": False}}, {})
        a, b = result["recorded_fu_transitions"]
        self.assertEqual(a["asr_change_pp"], 18)
        self.assertEqual(b["source_phase"], "cover5")
        self.assertEqual(b["asr_change_pp"], 7)
        self.assertIsNone(result["fpr"])
        self.assertIsNone(result["calibrated_alarm"])
        self.assertTrue(all(r["strict_client_first"] is None for r in result["first_unique_rank_one"]))

    def test_export_hash_rejects_modified_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "metrics.json"
            path.write_text("[]", encoding="utf-8")
            v.probes.save_json(folder / "inventory.json", {"metrics.json": v.probes.sha(path)})
            path.write_text('[{"asr_pct": 100}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Bundle hash mismatch"):
                v.analyze_bundle(folder)


if __name__ == "__main__":
    unittest.main()
