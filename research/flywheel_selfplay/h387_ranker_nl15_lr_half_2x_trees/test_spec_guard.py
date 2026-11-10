import unittest
from spec_guard import check
class SpecTests(unittest.TestCase):
 def test_baseline_passes(self):self.assertEqual(check([[.03,50],[.05,200],[.1,100]],972)['status'],'PASS')
 def test_already_treatment_rejected(self):self.assertEqual(check([[.015,100],[.025,400],[.05,200]],972)['status'],'INCOMPLETE_SPEC_MISMATCH')
 def test_no_missing_archive_allowed(self):self.assertEqual(check([],972)['status'],'INCOMPLETE_SPEC_MISMATCH');self.assertEqual(check([[.03,200]],971)['status'],'INCOMPLETE_SPEC_MISMATCH')
 def test_wrong_lr_or_rounds_rejected(self):
  for pair in [[.04,200],[.03,400]]:self.assertEqual(check([pair],972)['status'],'INCOMPLETE_SPEC_MISMATCH')
 def test_all_products_preserved(self):self.assertTrue(all(x['product_preserved'] for x in check([[.03,50],[.1,200]],972)['mapping']))
if __name__=='__main__':unittest.main()
