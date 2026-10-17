"""Check nuisance cancellation and fitting isolation, not attack performance."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str((Path(__file__).resolve().parents[4]/"FUBA")))
from experiments.detection.hrac.experiment_hrac_views import crossfit_residual, folds_by_class, coordinates


class ViewsTests(unittest.TestCase):
    def test_reference_mixtures_are_removed_on_heldout_inputs(self):
        rng = np.random.default_rng(41)
        reference = rng.normal(size=(3, 4, 12, 5))
        coeff = rng.normal(size=(6, 3))
        effects = np.einsum('kd,dvnc->kvnc', coeff, reference)
        residual, diagnostics = crossfit_residual(effects, reference, np.arange(12)%2)
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        self.assertTrue(all(row['rank'] == 3 for row in diagnostics))

    def test_oracle_response_cannot_change_reference_fit(self):
        rng = np.random.default_rng(17)
        reference = rng.normal(size=(2, 3, 10, 4))
        effects = rng.normal(size=(3, 3, 10, 4))
        base, d0 = crossfit_residual(effects, reference, np.arange(10)%2)
        shifted = effects.copy()
        shifted[:, 1] += 100
        alternate, d1 = crossfit_residual(shifted, reference, np.arange(10)%2)
        self.assertEqual(d0, d1)
        np.testing.assert_allclose(alternate[:, 0], base[:, 0])
        np.testing.assert_allclose(alternate[:, 1]-base[:, 1], 100)

    def test_class_stratified_folds_and_logprob_shift_invariance(self):
        y = np.repeat(np.arange(10), 8)
        folds = folds_by_class(y)
        for c in range(10):
            self.assertEqual(int(folds[y==c].sum()), 4)
        rng = np.random.default_rng(73)
        baseline = rng.normal(size=(4, 8, 10))
        tangent = np.ones_like(baseline)*7
        lp, centered = coordinates(tangent, baseline)
        np.testing.assert_allclose(lp, 0, atol=1e-12)
        np.testing.assert_allclose(centered, 0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
