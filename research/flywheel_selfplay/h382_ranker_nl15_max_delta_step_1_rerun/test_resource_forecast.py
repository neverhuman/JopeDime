import datetime,unittest
import resource_probe as p
import forecast_h382 as f
class ResourceTests(unittest.TestCase):
 def cap(self,n):return dict(eligible_cores=list(p.POOLS['B'][:n]))
 def report(self):return {'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':str(c),'idle':90.0} for c in (*p.ASSIGNED,*(x+64 for x in p.ASSIGNED))]}]}]}}
 def test_exact_authorized_union(self):self.assertEqual(p.POOLS,{'B':tuple(range(54,64)),'F':(4,5)});self.assertEqual(p.ASSIGNED,(*range(54,64),4,5))
 def test_first8(self):self.assertEqual(p.eligible_from_report(self.report())[0],list(range(54,62)))
 def test_both_siblings_required_and_exact85(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:
   row['idle']=85 if int(row['cpu'])!=118 else 84.99
  self.assertEqual(p.eligible_from_report(r)[0],list(range(55,63)))
 def test_second_interval_rejected(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'].append({})
  with self.assertRaises(AssertionError):p.eligible_from_report(r)
 def test_host70_min4_no_foreign_cores(self):
  self.assertTrue(p.admission_pass([54,55,56,57],70));self.assertFalse(p.admission_pass([54,55,56],10));self.assertFalse(p.admission_pass([54,55,56,57],70.01));self.assertFalse(p.admission_pass([0,1,54,55],1));self.assertTrue(p.admission_pass([54,55,4,5],70));self.assertFalse(p.admission_pass([54,54,54,54],1))
 def test_minimum_and_late_forecast(self):
  early=datetime.datetime(2026,11,5,9,tzinfo=datetime.timezone.utc);late=datetime.datetime(2026,11,5,20,tzinfo=datetime.timezone.utc)
  self.assertFalse(f.estimate(self.cap(3),early)['launch_full_36']);self.assertTrue(f.estimate(self.cap(4),early)['launch_full_36']);self.assertFalse(f.estimate(self.cap(8),late)['launch_full_36'])
 def test_first_task_max_and_zero_ongoing_credit(self):
  t=datetime.datetime(2026,11,5,9,tzinfo=datetime.timezone.utc);r=f.after_first_cell(dict(cpu_seconds=1000),8,t)
  self.assertEqual(r['per_task_cpu_reference_seconds'],1000);self.assertEqual(r['remaining_tasks'],35);self.assertEqual(r['forecast_seconds'],35*1000/(8*.9)*1.15+300)
 def test_first_task_late_rejects(self):self.assertFalse(f.after_first_cell(dict(cpu_seconds=1),8,datetime.datetime(2026,11,5,20,tzinfo=datetime.timezone.utc))['launch_full_36'])
 def test_union_combines_primary_and_fallback_once_with_both_siblings(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:row['idle']=90 if int(row['cpu']) in (54,55,4,5,118,119,68,69) else 80
  self.assertEqual(p.eligible_from_report(r)[0],[54,55,4,5]);self.assertTrue(p.admission_pass(p.eligible_from_report(r)[0],70))
  next(row for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load'] if row['cpu']=='69')['idle']=84.99
  self.assertEqual(p.eligible_from_report(r)[0],[54,55,4]);self.assertFalse(p.admission_pass(p.eligible_from_report(r)[0],30))
if __name__=='__main__':unittest.main()
