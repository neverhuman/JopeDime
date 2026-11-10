"""Dataset-only binning proof: zero optimizer updates/model fits."""
import unittest,inspect
import numpy as np
import lightgbm as lgb
from dataset_audit import audit_fit,validate_rows
class DatasetTests(unittest.TestCase):
 def data(self):return np.arange(4000,dtype=float).reshape(2000,2)
 def test_ranker_retains_native_dataset_255_bins_without_fitting(self):
  x=self.data();model=lgb.LGBMRanker(reg_alpha=.5,n_jobs=1)
  with audit_fit(model,(x,),{},'LGBMRanker') as rows:
   ds=lgb.Dataset(x,label=np.zeros(len(x)),params={'max_bin':255,'num_threads':1,'verbosity':-1});ds.construct()
  self.assertTrue(validate_rows(rows,'LGBMRanker'));self.assertEqual(max(rows[0]['feature_num_bins']),255)
 def test_veto_dataset_default_255_without_fitting(self):
  x=self.data()
  with audit_fit(lgb.LGBMClassifier(n_jobs=1),(x,),{},'LGBMClassifier') as rows:
   lgb.Dataset(x,label=np.zeros(len(x)),params={'num_threads':1,'verbosity':-1}).construct()
  self.assertEqual(max(rows[0]['feature_num_bins']),255)
 def test_prebuilt_model_input_rejected_before_fit(self):
  with self.assertRaises(AssertionError):
   with audit_fit(lgb.LGBMRanker(reg_alpha=.5),(lgb.Dataset(self.data()),),{},'LGBMRanker'):pass
 def test_constructor_ignoring_requested_bins_rejected(self):
  x=self.data()
  with self.assertRaises(AssertionError):
   with audit_fit(lgb.LGBMRanker(reg_alpha=.5),(x,),{},'LGBMRanker'):lgb.Dataset(x,params={'max_bin':63})
 def test_prebuilt_reference_rejected(self):
  x=self.data();ref=lgb.Dataset(x,params={'max_bin':63})
  with self.assertRaises(AssertionError):
   with audit_fit(lgb.LGBMRanker(reg_alpha=.5),(x,),{},'LGBMRanker'):lgb.Dataset(x,params={'max_bin':255},reference=ref)
 def test_missing_dataset_never_passes(self):
  with self.assertRaises(AssertionError):validate_rows([],'LGBMRanker')
 def test_forecast_miss_targets_owned_x1_unit(self):
  from pathlib import Path
  t=(Path(__file__).parent/'run_h376.py').read_text();self.assertIn("'h376-x1-run.service'",t);self.assertNotIn("'h376-run.service'",t)
if __name__=='__main__':unittest.main()
