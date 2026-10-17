import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str((Path(__file__).resolve().parents[4]/"FUBA")))
from experiments.detection.hrac.hrac import attribution_residual, conformal_channel, decide, inter_profile
from experiments.detection.hrac.experiment_hrac import request_bundle, evaluate_manifest, import_plan


class AttributionTests(unittest.TestCase):
    def test_legitimate_class_only_deletion_is_not_proportional(self):
        atoms = np.eye(2)
        result = attribution_residual(atoms, [1., 0.], constraint_matrix=[[1, 1]], constraint_value=[1])
        self.assertLess(result['upper_bound'], 1e-10)
        self.assertGreater(np.linalg.norm(np.array([1., 0.]) - .5 * atoms.sum(0)), .5)

    def test_signed_feasibility_distinguishes_equal_spectra(self):
        atoms = np.eye(2)
        clean = np.array([1., 0.]); altered = -clean
        np.testing.assert_array_equal(np.outer(clean, clean), np.outer(altered, altered))
        good = attribution_residual(atoms, clean, constraint_matrix=[[1, 1]], constraint_value=[1])
        bad = attribution_residual(atoms, altered, constraint_matrix=[[1, 1]], constraint_value=[1])
        self.assertLess(good['upper_bound'], 1e-10)
        self.assertGreater(bad['score'], 1)

    def test_constraints_do_not_allow_deleting_another_owner_atom(self):
        result = attribution_residual(np.eye(2), [0, 1], lower=[1, 0], upper=[1, 0])
        self.assertGreater(result['score'], 1)
        self.assertEqual(result['coefficients'], [1., 0.])

    def test_identity_covariance_has_known_projection(self):
        result = attribution_residual([[1., 0.]], [2., 3.], covariance=np.eye(2), ridge=.01)
        self.assertAlmostEqual(result['upper_bound'], 10/1.01, places=6)
        self.assertLess(result['duality_gap'], 1e-6)

    def test_feasible_fractional_request_and_non_iid_scale(self):
        atoms = np.diag([100., 1., .01])
        q = np.array([.1, .8, .6])
        result = attribution_residual(atoms, atoms.T@q, constraint_matrix=[[1, 1, 1]], constraint_value=[1.5])
        self.assertLess(result['upper_bound'], 1e-6)

    def test_coordinate_rotation_invariance(self):
        a = np.array([[1., 2.], [2., -.5]])
        u = np.array([-2., 3.]); rotation = np.array([[0., -1.], [1., 0.]])
        first = attribution_residual(a, u)
        second = attribution_residual(a@rotation, u@rotation)
        self.assertAlmostEqual(first['score'], second['score'], places=5)

    def test_empty_feasible_set_is_metadata_error(self):
        with self.assertRaises(ValueError):
            attribution_residual(np.eye(2), [0, 1], constraint_matrix=[[1, 1]], constraint_value=[3])

    def test_nonfinite_and_indefinite_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            attribution_residual(np.eye(2), [np.nan, 1])
        with self.assertRaises(ValueError):
            attribution_residual(np.eye(2), [0, 1], covariance=np.diag([1., -1.]))


class CalibrationTests(unittest.TestCase):
    def test_bonferroni_needs_39_clean_episodes_at_five_percent(self):
        low = decide({'inter': list(range(38)), 'intra': list(range(38))}, {'inter': 100., 'intra': 0.})
        enough = decide({'inter': list(range(39)), 'intra': list(range(39))}, {'inter': 100., 'intra': 0.})
        self.assertIsNone(low['alarm'])
        self.assertTrue(enough['alarm'])
        self.assertAlmostEqual(enough['joint_p'], .05)

    def test_ties_are_conservative(self):
        result = conformal_channel([7.]*99, 7., .025)
        self.assertEqual(result['p'], 1.)
        self.assertFalse(result['alarm'])

    def test_missing_intra_does_not_mean_clean(self):
        result = decide({'inter': [0.]*99}, {'inter': 0., 'intra': None})
        self.assertIsNone(result['alarm'])
        positive = decide({'inter': [0.]*99}, {'inter': 1., 'intra': None})
        self.assertTrue(positive['alarm'])
        self.assertIsNone(positive['joint_p'])

    def test_exhaustive_exchangeable_ranks_obey_single_channel_bound(self):
        scores = np.arange(40.)
        alarm = [conformal_channel(np.delete(scores, i), scores[i], .05)['alarm'] for i in range(40)]
        self.assertEqual(sum(alarm), 2)


class HistoryTests(unittest.TestCase):
    def test_uniform_removal_preserves_balance(self):
        history = np.tile(np.array([[[8.], [-10.]]]), (30, 1, 1))
        self.assertEqual(float(inter_profile(history, [.8, .8]).max()), 0.)
        self.assertGreater(float(inter_profile(history, [0., 1.]).max()), 0.)

    def test_late_opposition_not_erased_by_early_direction(self):
        h = np.zeros((30, 2, 1))
        h[:20, 1] = 1.; h[20:, 1] = -1.; h[20:, 0] = 1.
        self.assertEqual(float(inter_profile(h, [0., 1.], (0,)).max()), 0.)
        self.assertGreater(float(inter_profile(h, [0., 1.]).max()), 0.)

    def test_collapsed_atoms_cannot_recover_internal_cancellation(self):
        atoms = np.array([[1., 0.], [-1., 0.]])
        self.assertGreater(float(inter_profile(atoms[None], [0, 1]).max()), 0.)
        self.assertEqual(float(inter_profile(atoms.sum(0)[None, None], [1]).max()), 0.)

    def test_request_bundle_requires_independent_observation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p = Path(tmp)/'requests.json'
            p.write_text(json.dumps({'schema': 1, 'requests': [{'id': 'x', 'atoms': [[1.]],
                'observed': [1.], 'provenance': {'independent_observation': False}}]}))
            with self.assertRaises(ValueError):
                request_bundle(p)

    def test_no_source_episode_can_leak_across_splits(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            np.savez(root/'one.npz', K=2, T=2, ts=[1, 1, 2, 2], js=[0, 1, 0, 1], vs=[[1.], [-1.], [1.], [-1.]])
            base = dict(stratum='s', seed=1, measurement_id='fixed', source_episode='same-training', benign=True,
                        ledger='one.npz')
            p = root/'plan.json'
            p.write_text(json.dumps({'schema': 1, 'episodes': [dict(base, id='a', split='calibration'),
                                                          dict(base, id='b', split='clean_test')]}))
            with self.assertRaisesRegex(ValueError, 'duplicate episode'):
                evaluate_manifest(p)

    def test_import_plan_reuses_only_present_caches_without_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            cache = root/'existing.npz'; cache.write_bytes(b'only testing plan construction')
            base = dict(seed=3, stratum='iid', split='calibration', K=20, rounds=100)
            plan = root/'manifest.json'
            plan.write_text(json.dumps({'probe': 128, 'jobs': [
                dict(base, id='exists', cache=str(cache.resolve())),
                dict(base, id='missing', cache=str((root/'absent.npz').resolve()))]}))
            result = import_plan(plan)
            self.assertEqual(len(result['episodes']), 1)
            self.assertEqual(len(result['missing_caches']), 1)
            self.assertEqual(result['episodes'][0]['source_episode'], 'exists')
            self.assertNotIn('requests', result['episodes'][0])

    def test_request_bundle_and_missing_calibration_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); req = root/'request.json'
            req.write_text(json.dumps({'schema': 1, 'requests': [{'id': 'r', 'atoms': [[1, 0], [0, 1]],
                'observed': [-1, 0], 'constraint_matrix': [[1, 1]], 'constraint_value': [1],
                'provenance': {'independent_observation': True, 'history_source': 'logged',
                               'request_source': 'measured', 'coordinate_system': 'fixed'}}]}))
            result = request_bundle(req)
            self.assertGreater(result['episode_max'], 1)
            plan = root/'plan.json'
            plan.write_text(json.dumps({'schema': 1, 'episodes': [{'id': 'attack', 'source_episode': 'run1',
                'seed': 1, 'stratum': 's', 'measurement_id': 'fixed', 'split': 'attack_test', 'requests': 'request.json'}]}))
            report = evaluate_manifest(plan)
            self.assertIsNone(report['groups'][0]['decisions'][0]['alarm'])
            self.assertIsNone(report['groups'][0]['alarm_rates']['attack_test'])


if __name__ == '__main__':
    unittest.main()
