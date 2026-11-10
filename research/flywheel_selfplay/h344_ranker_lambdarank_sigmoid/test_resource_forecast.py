import datetime,json,unittest
import resource_probe as p
import forecast_h344 as f

class ResourceTests(unittest.TestCase):
 def cap(self,n):return dict(eligible_cores=list(p.ASSIGNED[:n]))
 def report(self):return {'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':str(c),'idle':90.0} for c in (*p.ASSIGNED,*(x+64 for x in p.ASSIGNED))]}]}]}}
 def test_new10core_mask_and_peers_disjoint(self):
  self.assertEqual(p.ASSIGNED,(54,56,57,58,59,61,63,16,17,18));self.assertFalse(set(p.ASSIGNED)&(set(range(21,34))|{13,15,60,62}|set(range(38,54))))
 def test_sibling_failure_excludes_and_exact85_passes(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'][0]['cpu-load'][len(p.ASSIGNED)]['idle']=84.99
  eligible,_=p.eligible_from_report(r);self.assertNotIn(54,eligible);self.assertEqual(len(eligible),8)
  r['sysstat']['hosts'][0]['statistics'][0]['cpu-load'][len(p.ASSIGNED)]['idle']=85;self.assertEqual(p.eligible_from_report(r)[0],list(p.ASSIGNED[:8]))
 def test_multiple_intervals_rejected(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'].append({})
  with self.assertRaises(AssertionError):p.eligible_from_report(r)
 def test_less_than4_blocks_even_early_start(self):
  self.assertFalse(f.estimate(self.cap(3),datetime.datetime(2026,11,4,0,tzinfo=datetime.timezone.utc))['launch_full_36'])
 def test_4_with_enough_time_passes_but_late8_fails(self):
  self.assertTrue(f.estimate(self.cap(4),datetime.datetime(2026,11,4,22,tzinfo=datetime.timezone.utc))['launch_full_36'])
  self.assertFalse(f.estimate(self.cap(8),datetime.datetime(2026,11,5,3,tzinfo=datetime.timezone.utc))['launch_full_36'])

 def test_first_cell_uses_conservative_larger_reference(self):
  start=datetime.datetime(2026,11,4,23,tzinfo=datetime.timezone.utc)
  x=f.after_first_cell(dict(cpu_seconds=2000,task=['P0',0,30001]),8,start)
  self.assertEqual(x['per_task_cpu_reference_seconds'],2000)
  self.assertEqual(x['forecast_seconds'],35*2000/(8*.9)*1.15+300)
  self.assertTrue(x['launch_full_36'])
 def test_first_cell_late_forecast_blocks_without_new_probe(self):
  x=f.after_first_cell(dict(cpu_seconds=1),8,datetime.datetime(2026,11,5,3,tzinfo=datetime.timezone.utc))
  self.assertFalse(x['launch_full_36'])
  self.assertEqual(x['remaining_tasks'],35)
  self.assertEqual(x['per_task_cpu_reference_seconds'],33343.799676/36)
if __name__=='__main__':unittest.main()
