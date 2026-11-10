import datetime,json,unittest
from pathlib import Path
import forecast_h333 as f
from resource_probe import eligible_from_report

class ForecastTests(unittest.TestCase):
 def test_actual_five_second_gate_admits_only22(self):
  c=json.loads((Path(__file__).parent/'capacity.json').read_text())
  report={'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':k,'idle':v} for k,v in c['idle_by_cpu'].items()]}]}]}}
  eligible,idle=eligible_from_report(report);self.assertEqual(eligible,[22]);self.assertEqual({str(k):v for k,v in idle.items()},c['both_smt_min_idle'])
 def test_complete36_misses_even_optimistic_reference(self):
  r=f.estimate(datetime.datetime(2026,11,4,20,27,14,tzinfo=datetime.timezone.utc));self.assertFalse(r['launch_full_36']);self.assertEqual(r['mandatory_tasks'],36)
  self.assertGreater(datetime.datetime.fromisoformat(r['optimistic_baseline_reference_end']),f.STOP)
  self.assertEqual(r['campaign_fits'],0);self.assertEqual(r['outer_scores'],0)
 def test_five_second_threshold_is_inclusive(self):
  values=[{'cpu':str(c),'idle':85.0} for c in [16,17,18,19,20,22,80,81,82,83,84,86]]
  report={'sysstat':{'hosts':[{'statistics':[{'cpu-load':values}]}]}}
  self.assertEqual(eligible_from_report(report)[0],[16,17,18,19,20,22])
 def test_multiple_intervals_rejected(self):
  with self.assertRaises(AssertionError):eligible_from_report({'sysstat':{'hosts':[{'statistics':[{},{}]}]}})

if __name__=='__main__':unittest.main()
