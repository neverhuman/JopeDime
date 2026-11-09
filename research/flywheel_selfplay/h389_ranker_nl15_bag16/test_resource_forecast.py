import datetime,unittest
import resource_probe as p
import forecast_h389 as f
class ResourceTests(unittest.TestCase):
 def cap(self,n):return dict(eligible_cores=list(p.POOLS['PRIMARY'][:n]))
 def report(self):return {'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':str(c),'idle':90.0} for c in (*p.ASSIGNED,*(x+64 for x in p.ASSIGNED))]}]}]}}
 def test_only_authorized_union(self):
  self.assertEqual(p.ASSIGNED,tuple(range(50,64)));self.assertFalse(set(p.ASSIGNED)&set(range(50)))
 def test_first8(self):self.assertEqual(p.eligible_from_report(self.report())[0],list(range(50,58)))
 def test_both_siblings_required_and_exact85(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:
   row['idle']=85 if int(row['cpu'])!=114 else 84.99
  self.assertEqual(p.eligible_from_report(r)[0],list(range(51,59)))
 def test_no_fallback_and_capacity_shortage_not_padded(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:
   if int(row['cpu']) in range(53,64):row['idle']=0
  self.assertEqual(p.eligible_from_report(r)[0],[50,51,52]);self.assertFalse(p.admission_pass([50,51,52],20))
 def test_second_interval_rejected(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'].append({})
  with self.assertRaises(AssertionError):p.eligible_from_report(r)
 def test_host70_min4_no_foreign_cores(self):
  self.assertTrue(p.admission_pass([50,51,52,53],70));self.assertFalse(p.admission_pass([50,51,52],10));self.assertFalse(p.admission_pass([50,51,52,53],70.01));self.assertFalse(p.admission_pass([0,1,50,51],1))
 def test_minimum_and_late_forecast(self):
  early=f.STOP-datetime.timedelta(hours=6);late=f.STOP-datetime.timedelta(minutes=5)
  self.assertFalse(f.estimate(self.cap(3),early)['launch_full_36']);self.assertTrue(f.estimate(self.cap(4),early)['launch_full_36']);self.assertFalse(f.estimate(self.cap(8),late)['launch_full_36'])
 def test_first_task_max_and_zero_ongoing_credit(self):
  t=datetime.datetime(2026,11,5,9,tzinfo=datetime.timezone.utc);r=f.after_first_cell(dict(cpu_seconds=2500),8,t)
  self.assertEqual(r['per_task_cpu_reference_seconds'],2500);self.assertEqual(r['remaining_tasks'],35);self.assertEqual(r['forecast_seconds'],35*2500/(8*.9)*1.15+300)
 def test_first_task_late_rejects(self):self.assertFalse(f.after_first_cell(dict(cpu_seconds=1),8,f.STOP-datetime.timedelta(minutes=5))['launch_full_36'])
 def test_ranker_share_scaled_and_veto_fixed(self):
  raw,scaled=f.reference();self.assertEqual(raw,26189.516914);self.assertEqual(scaled,raw*(2*819+195)/(819+195))
  out=f.after_first_cell(dict(cpu_seconds=1),8,f.STOP-datetime.timedelta(hours=8));self.assertEqual(out['per_task_cpu_reference_seconds'],scaled/36)
if __name__=='__main__':unittest.main()
