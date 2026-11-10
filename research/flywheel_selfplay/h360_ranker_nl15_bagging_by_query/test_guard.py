import unittest
from guard_h360 import host_reasons
class GuardTests(unittest.TestCase):
 def test_cpu85_inclusive_and_above_strict(self):
  self.assertEqual(host_reasons(85,100,0),([],0));self.assertTrue(host_reasons(85.001,100,0)[0])
 def test_load_three_consecutive_required(self):
  r,n=host_reasons(40,108.001,0);self.assertFalse(r);self.assertEqual(n,1)
  r,n=host_reasons(40,109,n);self.assertFalse(r);self.assertEqual(n,2)
  r,n=host_reasons(40,109,n);self.assertTrue(r);self.assertEqual(n,3)
 def test_load108_or_less_resets(self):self.assertEqual(host_reasons(40,108,2),([],0))
 def test_old_absolute85_no_longer_stops(self):self.assertEqual(host_reasons(40,85.04,0),([],0))
