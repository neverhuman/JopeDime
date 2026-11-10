import datetime,unittest
import resource_probe as p
import forecast_h352 as f
class ResourceTests(unittest.TestCase):
 def cap(self,n):return dict(eligible_cores=list(p.POOLS['A'][:n]))
 def report(self):return {'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':str(c),'idle':90.0} for c in (*p.ASSIGNED,*(x+64 for x in p.ASSIGNED))]}]}]}}
 def test_pools_disjoint_H351_H335(self):
  self.assertEqual(p.POOLS,{'A':tuple(range(15)),'B':tuple(range(27,38))});self.assertFalse(set(p.ASSIGNED)&(set(range(15,27))|set(range(38,54))))
 def test_first8_chosen_from_one_pool(self):self.assertEqual(p.eligible_from_report(self.report())[0],list(range(8)))
 def test_B_selected_in_same_report_without_mixing(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:
   if int(row['cpu']) in p.POOLS['A']:row['idle']=84.99
  eligible,_,choice=p.selection(r);self.assertEqual(eligible,list(range(27,35)));self.assertEqual(choice['chosen_pool'],'B')
 def test_tie_A_and_exact85_inclusive(self):
  r=self.report()
  for row in r['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:
   if int(row['cpu']) in range(11,15):row['idle']=0
   else:row['idle']=85
  eligible,_,choice=p.selection(r);self.assertEqual(choice['chosen_pool'],'A');self.assertEqual(eligible,list(range(8)))
 def test_second_interval_rejected(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'].append({})
  with self.assertRaises(AssertionError):p.eligible_from_report(r)
 def test_CPU70_and_min4_and_no_mixed_pool(self):
  self.assertTrue(p.admission_pass([0,1,2,3],70));self.assertFalse(p.admission_pass([0,1,2],10));self.assertFalse(p.admission_pass([0,1,2,3],70.01));self.assertFalse(p.admission_pass([0,1,27,28],1))
 def test_minimum_and_late_forecast(self):
  early=datetime.datetime(2026,11,5,7,tzinfo=datetime.timezone.utc);late=datetime.datetime(2026,11,5,11,tzinfo=datetime.timezone.utc)
  self.assertFalse(f.estimate(self.cap(3),early)['launch_full_36']);self.assertTrue(f.estimate(self.cap(4),early)['launch_full_36']);self.assertFalse(f.estimate(self.cap(8),late)['launch_full_36'])
 def test_first_task_max_reference_and_zero_ongoing_credit(self):
  t=datetime.datetime(2026,11,5,7,tzinfo=datetime.timezone.utc);r=f.after_first_cell(dict(cpu_seconds=1000),8,t)
  self.assertEqual(r['per_task_cpu_reference_seconds'],1000);self.assertEqual(r['remaining_tasks'],35);self.assertEqual(r['forecast_seconds'],35*1000/(8*.9)*1.15+300)
 def test_first_task_late_rejects(self):self.assertFalse(f.after_first_cell(dict(cpu_seconds=1),8,datetime.datetime(2026,11,5,11,tzinfo=datetime.timezone.utc))['launch_full_36'])
if __name__=='__main__':unittest.main()
