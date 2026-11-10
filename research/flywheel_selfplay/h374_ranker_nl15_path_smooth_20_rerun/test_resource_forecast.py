import datetime,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import resource_probe as p
import forecast_h374 as f
import guard_h374 as g
class ResourceTests(unittest.TestCase):
 def report(self):return {'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':str(c),'idle':90.0} for c in (*p.ASSIGNED,*(x+64 for x in p.ASSIGNED))]}]}]}}
 def cap(self,n):return dict(eligible_cores=list(p.ASSIGNED[:n]))
 def test_union_pools_first8(self):
  self.assertEqual(p.ASSIGNED,tuple(range(15,27))+tuple(range(46,54)))
  self.assertEqual(p.eligible_from_report(self.report())[0],list(range(15,23)))
  report=self.report()
  for row in report['sysstat']['hosts'][0]['statistics'][0]['cpu-load']:
   if int(row['cpu']) in set(range(15,27))|set(range(79,91)):row['idle']=0
  self.assertEqual(p.eligible_from_report(report)[0],list(range(46,54)))
  self.assertTrue(p.admission_pass([15,24,25,46],20))

 def test_exact85_and_failed_sibling(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'][0]['cpu-load'][len(p.ASSIGNED)]['idle']=84.99;self.assertNotIn(15,p.eligible_from_report(r)[0]);r['sysstat']['hosts'][0]['statistics'][0]['cpu-load'][len(p.ASSIGNED)]['idle']=85;self.assertIn(15,p.eligible_from_report(r)[0])
 def test_foreign_cores_and_ninth_rejected(self):
  self.assertFalse(p.admission_pass(list(range(27,35)),10));self.assertFalse(p.admission_pass(list(range(15,24)),10));self.assertFalse(p.admission_pass([15,16,17],10));self.assertTrue(p.admission_pass(list(range(15,23)),70));self.assertFalse(p.admission_pass(list(range(15,23)),70.01))
 def test_multiple_samples_rejected(self):
  r=self.report();r['sysstat']['hosts'][0]['statistics'].append({})
  with self.assertRaises(AssertionError):p.eligible_from_report(r)
 def test_H352_actual_CPU_and_fixed_window(self):
  x=f.estimate(self.cap(8),datetime.datetime(2026,11,5,11,tzinfo=datetime.timezone.utc));self.assertEqual(x['forecast_reference_cpu_seconds'],26189.516914);self.assertEqual(x['forecast_seconds'],26189.516914/(8*.9)*1.15+300);self.assertTrue(x['launch_full_36']);self.assertFalse(f.estimate(self.cap(8),datetime.datetime(2026,11,5,17,tzinfo=datetime.timezone.utc))['launch_full_36'])
 def test_first_task_larger_actual_and_no_ongoing_credit(self):
  x=f.after_first_cell(dict(cpu_seconds=2000),8,datetime.datetime(2026,11,5,10,tzinfo=datetime.timezone.utc));self.assertEqual(x['per_task_cpu_reference_seconds'],2000);self.assertEqual(x['remaining_tasks'],35);self.assertEqual(x['forecast_seconds'],35*2000/(8*.9)*1.15+300)
 def test_guard_needs_CPU60_or_three_load_samples(self):
  now=datetime.datetime(2026,11,5,11,tzinfo=datetime.timezone.utc);self.assertEqual(g.breaches(20*1024**3,2_000_000_000,None,2,now),[]);self.assertEqual(g.breaches(20*1024**3,2_000_000_000,85,2,now),[]);self.assertTrue(g.breaches(20*1024**3,1,85.01,0,now));self.assertTrue(g.breaches(20*1024**3,1,None,3,now))
 def test_guard_memory_disk_deadline(self):
  now=datetime.datetime(2026,11,5,11,tzinfo=datetime.timezone.utc);self.assertTrue(g.breaches(20*1024**3-1,1,None,0,now));self.assertTrue(g.breaches(20*1024**3,2_000_000_001,None,0,now));self.assertTrue(g.breaches(20*1024**3,1,None,0,g.STOP))
 def test_actual_monitor_uses_three_minute_windows_not_instant_load(self):
  parent=Path(__file__).parent.parent/'guard-test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as tmp,patch.object(g,'DATA',Path(tmp)),patch.object(g,'cpu',side_effect=[(0,0),(100,0),(20000,14000),(30000,21000),(40000,28000)]),patch.object(g.time,'sleep'),patch.object(g.time,'monotonic',side_effect=[0,5,65,125,185]),patch.object(g.os,'getloadavg',return_value=(109,109,109)),patch.object(g.subprocess,'run') as stopped:
   g.monitor('h374-test-only.service');rows=[json.loads(l) for l in (Path(tmp)/'resource.jsonl').read_text().splitlines()];self.assertEqual(len(rows),4);self.assertEqual(rows[0]['cpu_percent'],100);self.assertEqual([x['load1_over108_consecutive60s'] for x in rows],[0,1,2,3]);self.assertTrue(all(abs(x['cpu_percent_60s']-30)<1e-10 for x in rows[1:]));stopped.assert_called_once_with(['systemctl','--user','stop','h374-test-only.service'],check=False)
if __name__=='__main__':unittest.main()
