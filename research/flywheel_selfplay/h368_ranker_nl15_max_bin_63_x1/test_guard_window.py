import unittest,datetime
import guard_h368 as g
class GuardTests(unittest.TestCase):
 def sample(self):return dict(cpu_percent=99,cpu_percent_60s=None,cpu_window_seconds=0,load1=120,host_nproc=128,load_high_consecutive_60s=0,mem_available_bytes=20*2**30,root_free_bytes=80*2**30,scratch_bytes=2_000_000_000)
 def early(self):return datetime.datetime(2026,11,5,11,tzinfo=datetime.timezone.utc)
 def test_instant_CPU_and_absolute85load_do_not_stop(self):self.assertEqual(g.stop_reasons(self.sample(),self.early()),[])
 def test_cpu_average_requires_full60s_and_strict85(self):
  r=self.sample();r.update(cpu_percent_60s=86,cpu_window_seconds=59.999);self.assertEqual(g.stop_reasons(r,self.early()),[])
  r['cpu_window_seconds']=60;self.assertIn('host CPU >85% averaged over60s',g.stop_reasons(r,self.early()))
  r['cpu_percent_60s']=85;self.assertEqual(g.stop_reasons(r,self.early()),[])
 def test_three_consecutive_normalized_samples_required(self):
  streak=0;r=self.sample()
  for i in range(3):
   streak=g.next_load_streak(streak,109,128);r['load_high_consecutive_60s']=streak
   self.assertEqual(bool(g.stop_reasons(r,self.early())),i==2)
 def test_load_at_threshold_resets_streak(self):self.assertEqual(g.next_load_streak(2,.85*128,128),0)
 def test_host_cpu_count_independent_of_worker_affinity(self):
  text='cpu 1 2 3 4\n'+'\n'.join('cpu'+str(i)+' 1 2 3 4' for i in range(128));self.assertEqual(g.host_online_cpu_count(text),128)
 def test_immediate_memory_root_and_scratch_boundaries(self):
  for key in ['mem_available_bytes','root_free_bytes','scratch_bytes']:
   r=self.sample();r[key]+=1 if key=='scratch_bytes' else -1;self.assertTrue(g.stop_reasons(r,self.early()))
 def test_deadline_is_fixed(self):self.assertIn('compute deadline',g.stop_reasons(self.sample(),datetime.datetime(2026,11,5,16,30,tzinfo=datetime.timezone.utc)))
if __name__=='__main__':unittest.main()
