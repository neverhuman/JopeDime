import datetime,json,unittest
import resource_probe as p
import forecast_h340 as f

class ResourceTests(unittest.TestCase):
 def cap(self,n):return dict(eligible_cores=list(p.ASSIGNED[:n]))
 def report(self):return {'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':str(c),'idle':90.0} for c in (*p.ASSIGNED,*(x+64 for x in p.ASSIGNED))]}]}]}}
 def test_new8core_mask_and_peers_disjoint(self):
  self.assertEqual(p.ASSIGNED,(1,2,3,7,9,10,12,14));self.assertFalse(set(p.ASSIGNED)&{4,5,6,8,27,33,57,59,13,15,60,62,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53})
 def test_sibling_failure_excludes_and_exact85_passes(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'][0]['cpu-load'][8]['idle']=84.99
  eligible,_=p.eligible_from_report(r);self.assertNotIn(1,eligible);self.assertEqual(len(eligible),7)
  r['sysstat']['hosts'][0]['statistics'][0]['cpu-load'][8]['idle']=85;self.assertEqual(p.eligible_from_report(r)[0],list(p.ASSIGNED))
 def test_multiple_intervals_rejected(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'].append({})
  with self.assertRaises(AssertionError):p.eligible_from_report(r)
 def test_less_than4_blocks_even_early_start(self):
  self.assertFalse(f.estimate(self.cap(3),datetime.datetime(2026,11,4,0,tzinfo=datetime.timezone.utc))['launch_full_36'])
 def test_4_with_enough_time_passes_but_late8_fails(self):
  self.assertTrue(f.estimate(self.cap(4),datetime.datetime(2026,11,4,22,tzinfo=datetime.timezone.utc))['launch_full_36'])
  self.assertFalse(f.estimate(self.cap(8),datetime.datetime(2026,11,5,2,tzinfo=datetime.timezone.utc))['launch_full_36'])

if __name__=='__main__':unittest.main()
