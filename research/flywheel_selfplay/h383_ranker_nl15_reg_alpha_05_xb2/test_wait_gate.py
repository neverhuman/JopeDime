"""Exercise only the specified read-only release gate; never sample capacity in fixtures."""
import unittest,datetime,inspect
from types import SimpleNamespace
from unittest.mock import patch
import controller_h383 as c
class WaitGateTests(unittest.TestCase):
 def exercise(self,states,times):
  stamps=iter(times)
  class Clock:
   @staticmethod
   def now(tz):return next(stamps)
  results=[SimpleNamespace(stdout=s+'\n',returncode=0 if s=='active' else 3) for s in states]
  with patch.object(c,'datetime',SimpleNamespace(datetime=type('D',(),{'now':staticmethod(Clock.now),'__new__':staticmethod(lambda cls,*a,**k:datetime.datetime(*a,**k))}),timezone=datetime.timezone)),patch.object(c.subprocess,'run',side_effect=results) as run,patch.object(c.time,'sleep') as sleep,patch.object(c,'save') as save:
   answer=c.wait_for_h377()
  self.assertTrue(all(x.args[0]==['systemctl','--user','is-active','h377-run'] for x in run.call_args_list));return answer,run,sleep,save
 def stamp(self,h,m):return datetime.datetime(2026,11,5,h,m,tzinfo=datetime.timezone.utc)
 def test_already_inactive_releases_without_wait(self):
  answer,run,sleep,save=self.exercise(['inactive'],[self.stamp(17,20)]);self.assertTrue(answer);sleep.assert_not_called();self.assertTrue(save.call_args.args[1]['released'])
 def test_active_then_inactive_reads_only_and_releases(self):
  answer,run,sleep,save=self.exercise(['active','inactive'],[self.stamp(17,20),self.stamp(17,21)]);self.assertTrue(answer);self.assertEqual(run.call_count,2);sleep.assert_called_once();self.assertEqual(save.call_args.args[1]['foreign_signals'],0)
 def test_still_active_at_cutoff_blocks_without_probe(self):
  answer,run,sleep,save=self.exercise(['active'],[self.stamp(18,50)]);self.assertFalse(answer);sleep.assert_not_called();self.assertTrue(save.call_args.args[1]['timed_out']);self.assertEqual(save.call_args.args[1]['capacity_samples'],0)
 def test_controller_release_gate_precedes_single_probe(self):
  source=inspect.getsource(c.main);self.assertLess(source.index('wait_for_h377()'),source.index('probe()'));self.assertEqual(source.count('probe()'),1)
if __name__=='__main__':unittest.main()
