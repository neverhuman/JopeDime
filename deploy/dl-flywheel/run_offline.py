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
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:v for k,v in receipt.items()if k!='identities'}))
 if not success:raise SystemExit(1)
if __name__=='__main__':main()
