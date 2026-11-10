import unittest
from unittest.mock import patch
import continue_admitted as c
class AffinityContinuationTests(unittest.TestCase):
 def test_priority_order_differs_from_kernel_order_but_same_set(self):
  with patch.object(c.os,'sched_setaffinity') as put,patch.object(c.os,'sched_getaffinity',return_value={12,13,14,31}):
   got=c.bind_threads(['101','102'],[31,12,13,14]);self.assertEqual(got,{'101':[12,13,14,31],'102':[12,13,14,31]});self.assertEqual(put.call_count,2);self.assertTrue(all(x.args[1]=={12,13,14,31} for x in put.call_args_list))
 def test_original_ascending_pool_still_binds(self):
  with patch.object(c.os,'sched_setaffinity'),patch.object(c.os,'sched_getaffinity',return_value={27,28,29,30}):self.assertEqual(c.bind_threads(['101'],[27,28,29,30]),{'101':[27,28,29,30]})
 def test_wrong_affinity_set_still_rejected(self):
  with patch.object(c.os,'sched_setaffinity'),patch.object(c.os,'sched_getaffinity',return_value={12,13,14,30}):
   with self.assertRaises(AssertionError):c.bind_threads(['101'],[31,12,13,14])
if __name__=='__main__':unittest.main()
