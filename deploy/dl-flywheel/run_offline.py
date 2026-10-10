"""Run the full offline flywheel suite; keep its receipt separate from pytest."""
import argparse,hashlib,json,os,subprocess,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 native=Path(os.environ.get('JOPE_POLICYLM_SOURCE',str(REPO/'deploy/dl-trainonly/lane/src/dl_training_v1'))).resolve()
 os.environ['JOPE_POLICYLM_SOURCE']=str(native)
 sys.path[:0]=[str(ROOT),str(native)]
 a.output.mkdir(parents=True,exist_ok=False)
 started=time.time()
 suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
 count=suite.countTestCases()
 def ids(items):
  for item in items:
   if isinstance(item,unittest.TestSuite):yield from ids(item)
   else:yield item.id()
 identities=sorted(ids(suite))
 if count!=226 or len(set(identities))!=count:
  raise ValueError('Expected exact226 test identities; review any suite change')
 with(a.output/'tests.log').open('x')as log:
  result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 success=result.wasSuccessful()and result.testsRun==count and not result.skipped and not result.expectedFailures
 receipt={'schema':'jope.flywheel-offline-tests/v1','source_sha':os.environ.get('JOPE_SOURCE_SHA'),
  'started':started,'finished':time.time(),'collected':count,'executed':result.testsRun,
  'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
  'expected_failures':len(result.expectedFailures),'identities':identities,
  'log_sha256':hashlib.sha256((a.output/'tests.log').read_bytes()).hexdigest(),'passed':success,
  'scope':'Offline source qualification; no live GPU, native compute campaign, S3 or serving claim.'}
 if success:
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_native_offline.py'),'--output',str(a.output/'native')],check=False)
  follow=json.loads((a.output/'native/receipt.json').read_bytes())
  success=child.returncode==0 and follow['passed'] is True
  receipt['native_suite']=follow
  receipt['combined_collected']=count+follow['collected'];receipt['combined_executed']=result.testsRun+follow['executed'];receipt['passed']=success
 if success:
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_hpc_offline.py'),'--output',str(a.output/'hpc')],check=False)
  hpc=json.loads((a.output/'hpc/receipt.json').read_bytes())
  success=child.returncode==0 and hpc['passed'] is True
  receipt['hpc_suite']=hpc;receipt['combined_collected']+=hpc['collected'];receipt['combined_executed']+=hpc['executed'];receipt['passed']=success
 if success:
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_continuous_offline.py'),'--output',str(a.output/'continuous')],check=False)
  continuous=json.loads((a.output/'continuous/receipt.json').read_bytes())
  success=child.returncode==0 and continuous['passed'] is True
  receipt['continuous_suite']=continuous;receipt['combined_collected']+=continuous['collected'];receipt['combined_executed']+=continuous['executed'];receipt['passed']=success
 if success:
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_successor_offline.py'),'--output',str(a.output/'successor')],check=False)
  successor=json.loads((a.output/'successor/receipt.json').read_bytes())
  success=child.returncode==0 and successor['passed'] is True
  receipt['successor_suite']=successor;receipt['combined_collected']+=successor['collected'];receipt['combined_executed']+=successor['executed'];receipt['passed']=success
 if success:
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_variant_offline.py'),'--output',str(a.output/'variant')],check=False)
  variant=json.loads((a.output/'variant/receipt.json').read_bytes())
  success=child.returncode==0 and variant['passed'] is True
  receipt['variant_suite']=variant;receipt['combined_collected']+=variant['collected'];receipt['combined_executed']+=variant['executed'];receipt['passed']=success
 if success:
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_telemetry_offline.py'),'--output',str(a.output/'telemetry')],check=False)
  telemetry=json.loads((a.output/'telemetry/receipt.json').read_bytes())
  success=child.returncode==0 and telemetry['passed'] is True
  receipt['telemetry_suite']=telemetry;receipt['combined_collected']+=telemetry['collected'];receipt['combined_executed']+=telemetry['executed'];receipt['passed']=success
 if success:
  if receipt['combined_collected']!=455 or receipt['combined_executed']!=455:
   raise ValueError('Prior455 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_fresh32_offline.py'),'--output',str(a.output/'fresh32')],check=False)
  fresh32=json.loads((a.output/'fresh32/receipt.json').read_bytes())
  success=child.returncode==0 and fresh32['passed'] is True and fresh32['collected']==fresh32['executed']==37
  receipt['fresh32_suite']=fresh32;receipt['combined_collected']+=fresh32['collected'];receipt['combined_executed']+=fresh32['executed'];receipt['passed']=success
 if success:
  if receipt['combined_collected']!=492 or receipt['combined_executed']!=492:
   raise ValueError('Prior492 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_native_v5_offline.py'),'--output',str(a.output/'native_v5')],check=False)
  native_v5=json.loads((a.output/'native_v5/receipt.json').read_bytes())
  success=child.returncode==0 and native_v5['passed'] is True and native_v5['collected']==native_v5['executed']==57
  receipt['native_v5_suite']=native_v5;receipt['combined_collected']+=native_v5['collected'];receipt['combined_executed']+=native_v5['executed'];receipt['passed']=success
 if success:
  if receipt['combined_collected']!=549 or receipt['combined_executed']!=549:
   raise ValueError('Prior549 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_next128_offline.py'),'--output',str(a.output/'next128')],check=False)
  next128=json.loads((a.output/'next128/receipt.json').read_bytes())
  success=child.returncode==0 and next128['passed'] is True and next128['collected']==next128['executed']==25
  receipt['next128_suite']=next128;receipt['combined_collected']+=next128['collected'];receipt['combined_executed']+=next128['executed'];receipt['passed']=success
 if success:
  if receipt['combined_collected']!=574 or receipt['combined_executed']!=574:
   raise ValueError('Prior574 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_trace_v2_offline.py'),'--output',str(a.output/'trace_v2')],check=False)
  trace_v2=json.loads((a.output/'trace_v2/receipt.json').read_bytes())
  success=child.returncode==0 and trace_v2['passed'] is True and trace_v2['collected']==trace_v2['executed']==13
  receipt['trace_v2_suite']=trace_v2;receipt['combined_collected']+=trace_v2['collected'];receipt['combined_executed']+=trace_v2['executed'];receipt['passed']=success
 if success:
  if receipt['combined_collected']!=587 or receipt['combined_executed']!=587:
   raise ValueError('Prior587 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_native_pool_offline.py'),'--output',str(a.output/'native_pool')],check=False)
  native_pool=json.loads((a.output/'native_pool/receipt.json').read_bytes())
  success=child.returncode==0 and native_pool['passed'] is True and native_pool['collected']==native_pool['executed']==21
  receipt['native_pool_suite']=native_pool;receipt['combined_collected']+=native_pool['collected'];receipt['combined_executed']+=native_pool['executed'];receipt['passed']=success
 if success:
  if receipt['combined_collected']!=608 or receipt['combined_executed']!=608:
   raise ValueError('Prior608 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_rl_math_offline.py'),'--output',str(a.output/'rl_math')],check=False)
  rl_math=json.loads((a.output/'rl_math/receipt.json').read_bytes())
  success=child.returncode==0 and rl_math['passed'] is True and rl_math['collected']==rl_math['executed']==9
  receipt['rl_math_suite']=rl_math;receipt['combined_collected']+=rl_math['collected'];receipt['combined_executed']+=rl_math['executed'];receipt['passed']=success
 if success and (receipt['combined_collected']!=617 or receipt['combined_executed']!=617):
  raise ValueError('All617 wired offline identities must complete')
 if success:
  if receipt['combined_collected']!=617 or receipt['combined_executed']!=617:
   raise ValueError('Prior617 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_fresh_gp_offline.py'),'--output',str(a.output/'fresh_gp')],check=False)
  fresh_gp=json.loads((a.output/'fresh_gp/receipt.json').read_bytes())
  success=child.returncode==0 and fresh_gp['passed']is True and fresh_gp['collected']==fresh_gp['executed']==70
  receipt['fresh_gp_suite']=fresh_gp;receipt['combined_collected']+=fresh_gp['collected'];receipt['combined_executed']+=fresh_gp['executed'];receipt['passed']=success
 if success and(receipt['combined_collected']!=687 or receipt['combined_executed']!=687):
  raise ValueError('All687 wired offline identities must complete')
 if success:
  if receipt['combined_collected']!=687 or receipt['combined_executed']!=687:
   raise ValueError('Prior687 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'run_trace_v3_offline.py'),'--output',str(a.output/'trace_v3')],check=False)
  trace_v3=json.loads((a.output/'trace_v3/receipt.json').read_bytes())
  success=child.returncode==0 and trace_v3['passed']is True and trace_v3['collected']==trace_v3['executed']==34
  receipt['trace_v3_suite']=trace_v3;receipt['combined_collected']+=trace_v3['collected'];receipt['combined_executed']+=trace_v3['executed'];receipt['passed']=success
 if success and(receipt['combined_collected']!=721 or receipt['combined_executed']!=721):
  raise ValueError('All721 wired offline identities must complete')
 if success:
  if receipt['combined_collected']!=721 or receipt['combined_executed']!=721:
   raise ValueError('Prior721 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'pipelines/generated_target_admission_v1/run_offline.py'),'--receipt',str((a.output/'generated-targets-receipt.json').resolve())],check=False)
  generated_targets=json.loads((a.output/'generated-targets-receipt.json').read_bytes())
  success=child.returncode==0 and generated_targets['passed']is True and generated_targets['tests_run']==28 and len(generated_targets['fixture_ids'])==28 and len(set(generated_targets['fixture_ids']))==28
  receipt['generated_targets_suite']=generated_targets;receipt['combined_collected']+=len(generated_targets['fixture_ids']);receipt['combined_executed']+=generated_targets['tests_run'];receipt['passed']=success
 if success and(receipt['combined_collected']!=749 or receipt['combined_executed']!=749):
  raise ValueError('All749 wired offline identities must complete')
 if success:
  if receipt['combined_collected']!=749 or receipt['combined_executed']!=749:
   raise ValueError('Prior749 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'test_callback_targets.py'),'--output',str(a.output/'callback_targets')],check=False)
  callback_targets=json.loads((a.output/'callback_targets/receipt.json').read_bytes())
  success=child.returncode==0 and callback_targets['passed']is True and callback_targets['collected']==callback_targets['executed']==63 and len(callback_targets['identities'])==len(set(callback_targets['identities']))==63
  receipt['callback_targets_suite']=callback_targets;receipt['combined_collected']+=callback_targets['collected'];receipt['combined_executed']+=callback_targets['executed'];receipt['passed']=success
 if success and(receipt['combined_collected']!=812 or receipt['combined_executed']!=812):
  raise ValueError('All812 wired offline identities must complete')
 if success:
  if receipt['combined_collected']!=812 or receipt['combined_executed']!=812:
   raise ValueError('Prior812 offline identities must remain independently complete')
  child=subprocess.run([sys.executable,'-B',str(ROOT/'pipelines/standalone_paper_monitor_v1/run_offline.py'),'--output',str((a.output/'standalone_monitor').resolve())],check=False)
  standalone_monitor=json.loads((a.output/'standalone_monitor/receipt.json').read_bytes())
  success=child.returncode==0 and standalone_monitor['passed']is True and standalone_monitor['collected']==standalone_monitor['executed']==82 and len(standalone_monitor['identities'])==len(set(standalone_monitor['identities']))==82
  receipt['standalone_monitor_suite']=standalone_monitor;receipt['combined_collected']+=standalone_monitor['collected'];receipt['combined_executed']+=standalone_monitor['executed'];receipt['passed']=success
 if success and(receipt['combined_collected']!=894 or receipt['combined_executed']!=894):
  raise ValueError('All894 wired offline identities must complete')
 if success:
  child=subprocess.run([sys.executable,'-B',str(ROOT/'dashboard/test_dashboard_flywheel.py'),'--output',str((a.output/'dashboard_flywheel').resolve())],check=False)
  dashboard_flywheel=json.loads((a.output/'dashboard_flywheel/receipt.json').read_bytes())
  success=child.returncode==0 and dashboard_flywheel['passed']is True and dashboard_flywheel['collected']==dashboard_flywheel['executed']==37 and len(dashboard_flywheel['identities'])==len(set(dashboard_flywheel['identities']))==37
  receipt['dashboard_flywheel_suite']=dashboard_flywheel;receipt['combined_collected']+=dashboard_flywheel['collected'];receipt['combined_executed']+=dashboard_flywheel['executed'];receipt['passed']=success
 if success and(receipt['combined_collected']!=931 or receipt['combined_executed']!=931):
  raise ValueError('All931 wired offline identities must complete')
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:v for k,v in receipt.items()if k!='identities'}))
 if not success:raise SystemExit(1)
if __name__=='__main__':main()
