import unittest
from derive_minhess import leaves
class DerivationTests(unittest.TestCase):
 def test_all_leaves_in_depth_first_order(self):
  t=dict(split_index=0,left_child=dict(leaf_index=0,leaf_weight=2,leaf_count=900),right_child=dict(split_index=1,left_child=dict(leaf_index=1,leaf_weight=7),right_child=dict(leaf_index=2,leaf_weight=11)))
  self.assertEqual(leaves(t),[2,7,11])
 def test_constant_leaf_zero_weight_preserved(self):self.assertEqual(leaves(dict(leaf_value=1,leaf_weight=0)),[0])
 def test_missing_weight_never_inferred_from_count(self):
  with self.assertRaises(AssertionError):leaves(dict(leaf_index=0,leaf_value=10,leaf_count=700))
if __name__=='__main__':unittest.main()
