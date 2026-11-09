"""Regression for H383's pre-fit priority-order assertion failure."""
import unittest
from controller_h392 import affinity_matches
class AffinityTests(unittest.TestCase):
 def test_priority_order_matches_sorted_kernel_sets(self):self.assertTrue(affinity_matches({'1':[12,13,14,31]},[31,12,13,14]))
 def test_ascending_order_also_matches(self):self.assertTrue(affinity_matches({'1':[27,28,29,30]},[27,28,29,30]))
 def test_different_core_or_empty_binding_refused(self):
  self.assertFalse(affinity_matches({'1':[12,13,14,32]},[31,12,13,14]));self.assertFalse(affinity_matches({},[31,12,13,14]))
