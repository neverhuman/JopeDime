import datetime,unittest
import resource_probe as p
import forecast_h383 as f
class ResourceTests(unittest.TestCase):
 def cap(self,n):return dict(eligible_cores=list(p.ASSIGNED[:n]))
 def report(self):return {'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':str(c),'idle':90.0} for c in (*p.ASSIGNED,*(x+64 for x in p.ASSIGNED))]}]}]}}
 def test_exclusive_union_primary_then_fallback(self):
  self.assertEqual(p.ASSIGNED,tuple(range(27,38))+(12,13,14));self.assertFalse(set(p.ASSIGNED)&set(range(54,64)))
 def test_first8_in_priority_order(self):self.assertEqual(p.eligible_from_report(self.report())[0],list(range(27,35)))
 def test_sibling_must_admit(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:
   if int(row['cpu'])==91:row['idle']=84.99
  self.assertEqual(p.eligible_from_report(r)[0],list(range(28,36)))
 def test_exact85_is_inclusive(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:row['idle']=85
  self.assertEqual(p.eligible_from_report(r)[0],list(range(27,35)))
 def test_fallback_is_in_same_single_union_interval(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:
   if int(row['cpu']) in range(29,38):row['idle']=0
  self.assertEqual(p.eligible_from_report(r)[0],[27,28,12,13,14])
 def test_second_interval_rejected(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'].append({})
  with self.assertRaises(AssertionError):p.eligible_from_report(r)
 def test_CPU70_min4_and_no_foreign_core(self):
  self.assertTrue(p.admission_pass([27,28,29,30],70));self.assertFalse(p.admission_pass([27,28,29],10));self.assertFalse(p.admission_pass([27,28,29,30],70.01));self.assertFalse(p.admission_pass([26,27,28,29],1))
 def test_minimum_and_late_forecast(self):
  early=datetime.datetime(2026,11,5,11,tzinfo=datetime.timezone.utc);late=datetime.datetime(2026,11,5,20,tzinfo=datetime.timezone.utc)
  self.assertFalse(f.estimate(self.cap(3),early)['launch_full_36']);self.assertTrue(f.estimate(self.cap(4),early)['launch_full_36']);self.assertFalse(f.estimate(self.cap(8),late)['launch_full_36'])
 def test_first_task_max_reference_zero_ongoing_credit(self):
  t=datetime.datetime(2026,11,5,11,tzinfo=datetime.timezone.utc);r=f.after_first_cell(dict(cpu_seconds=1000),8,t)
  self.assertEqual(r['per_task_cpu_reference_seconds'],1000);self.assertEqual(r['remaining_tasks'],35);self.assertEqual(r['forecast_seconds'],35*1000/(8*.9)*1.15+300)
 def test_first_task_late_rejects(self):self.assertFalse(f.after_first_cell(dict(cpu_seconds=1),8,datetime.datetime(2026,11,5,20,tzinfo=datetime.timezone.utc))['launch_full_36'])
if __name__=='__main__':unittest.main()
