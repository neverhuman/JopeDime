import unittest
from tree_structure import parse_trees,validate,summarize

def tree(n,left='',right='',index=0):return f'Tree={index}\nnum_leaves={n}\nnum_cat=0\nleft_child={left}\nright_child={right}\n'
class TreeTests(unittest.TestCase):
 def test_root_leaf_has_depth_zero(self):self.assertEqual(parse_trees(tree(1)),[[0,1]])
 def test_balanced_tree(self):self.assertEqual(parse_trees(tree(4,'1 -1 -3','2 -2 -4')),[[2,4]])
 def test_asymmetric_depth_four_at_five_leaves(self):self.assertEqual(parse_trees(tree(5,'-1 -2 -3 -4','1 2 3 -5')),[[4,5]])
 def test_multiple_trees_and_histograms(self):
  t=parse_trees(tree(1)+tree(2,'-1','-2',1));self.assertEqual(summarize(t)['depth_histogram'],{'0':1,'1':1});self.assertEqual(summarize(t)['leaf_histogram'],{'1':1,'2':1})
 def test_cycle_and_duplicate_leaf_rejected(self):
  for text in [tree(2,'0','-1'),tree(2,'-1','-1')]:
   with self.assertRaises(AssertionError):parse_trees(text)
 def test_unreachable_or_bad_header_rejected(self):
  for text in [tree(3,'-1','-2'),tree(2,'-1','-2',1),'no trees']:
   with self.assertRaises(AssertionError):parse_trees(text)
 def test_depth_and_leaf_budgets_independent(self):
  self.assertEqual(validate([[4,7]],4,7),[[4,7]])
  for t in [[[5,7]],[[4,8]]]:
   with self.assertRaises(AssertionError):validate(t,4,7)
if __name__=='__main__':unittest.main()
