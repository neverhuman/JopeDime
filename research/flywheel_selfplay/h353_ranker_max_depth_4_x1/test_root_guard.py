import datetime,unittest
from guard_h353 import stop_reasons
class RootGuardTests(unittest.TestCase):
 def sample(self):return dict(cpu_percent=85,load1=85,mem_available_bytes=20*2**30,root_free_bytes=80*2**30,scratch_bytes=2_000_000_000)
 def now(self):return datetime.datetime(2026,11,5,10,0,tzinfo=datetime.timezone.utc)
 def test_exact_limits_pass(self):self.assertEqual(stop_reasons(self.sample(),self.now()),[])
 def test_one_byte_below_root_reserve_stops(self):
  r=self.sample();r['root_free_bytes']-=1;self.assertEqual(stop_reasons(r,self.now()),['root free <80GiB'])
 def test_host_limits_remain_independent(self):
  for key,value,reason in [('cpu_percent',85.01,'host CPU >85%'),('load1',85.01,'host load >85'),('mem_available_bytes',20*2**30-1,'MemAvailable <20GiB'),('scratch_bytes',2_000_000_001,'scratch >2GB')]:
   r=self.sample();r[key]=value;self.assertEqual(stop_reasons(r,self.now()),[reason])
 def test_fixed_compute_deadline(self):self.assertEqual(stop_reasons(self.sample(),datetime.datetime(2026,11,5,10,45,tzinfo=datetime.timezone.utc)),['compute deadline'])
if __name__=='__main__':unittest.main()
