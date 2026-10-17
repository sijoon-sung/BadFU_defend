import unittest
import numpy as np
import torch
from reproduction.audit.diagnose_fuba_asr import paired_rates, clean_bias_bound, sample_indices, permute_noise


class PairedDiagnosisTests(unittest.TestCase):
    def test_existing_target_bias_is_not_trigger_gain(self):
        y = np.arange(10)
        predictions = np.full(10,8)
        r = paired_rates(y,predictions,predictions)
        self.assertEqual(r['triggered_target_rate_pct'],100)
        self.assertEqual(r['clean_target_rate_pct'],100)
        self.assertEqual(r['net_trigger_gain_pp'],0)
        self.assertEqual(r['newly_targeted_pct'],0)
        self.assertIsNone(r['correct_to_target_pct'])

    def test_target_class_is_excluded_and_transition_identity_holds(self):
        y = np.array([0,1,2,8])
        clean = np.array([8,1,2,8])
        triggered = np.array([0,8,8,0])
        r = paired_rates(y,clean,triggered)
        self.assertEqual(r['non_target_n'],3)
        self.assertAlmostEqual(r['net_trigger_gain_pp'],100/3)
        self.assertAlmostEqual(r['net_trigger_gain_pp'],r['newly_targeted_pct']-r['no_longer_targeted_pct'])
        self.assertEqual(r['correct_to_target_pct'],100)
        self.assertNotIn('8',r['by_source_class'])

    def test_worsening_an_existing_error_is_distinct_from_correct_to_target(self):
        y = np.array([0,1,2,8])
        clean = np.array([1,1,2,8])
        triggered = np.array([8,1,2,8])
        r = paired_rates(y,clean,triggered)
        self.assertAlmostEqual(r['newly_targeted_pct'],100/3)
        self.assertEqual(r['correct_to_target_pct'],0)

    def test_accuracy_bound_does_not_claim_early_trigger_effect(self):
        early=clean_bias_bound(49.55,50.94444444444444)
        late=clean_bias_bound(73.62,98.17777777777778)
        self.assertEqual(early['newly_targeted_lower_bound_pct'],0)
        self.assertAlmostEqual(late['clean_target_rate_upper_bound_pct'],29.3111111111)
        self.assertAlmostEqual(late['newly_targeted_lower_bound_pct'],68.8666666667)

    def test_selection_is_balanced_unique_and_reproducible(self):
        y=np.repeat(np.arange(10),11)
        ids=sample_indices(y,100,42)
        self.assertEqual(len(set(ids)),100)
        np.testing.assert_array_equal(np.bincount(y[ids]),np.full(10,10))
        np.testing.assert_array_equal(ids,sample_indices(y,100,42))

    def test_permutation_preserves_each_images_values_and_norms(self):
        x=torch.linspace(-.04,.04,2*3*8*8).reshape(2,3,8,8)
        y=permute_noise(x,42)
        torch.testing.assert_close(x.flatten(1).sort(1).values,y.flatten(1).sort(1).values)
        torch.testing.assert_close(y,permute_noise(x,42))
        self.assertFalse(torch.equal(x,y))


if __name__=='__main__':
    unittest.main()
