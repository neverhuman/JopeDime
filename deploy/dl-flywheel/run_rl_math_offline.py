"""Isolated 9 portable rl_math fixtures; no live deployment effects."""
import argparse, hashlib, json, os, sys, time, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'pipelines/rl_spec_audit'
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(SOURCE))
MODULES=('test_audit_math',)

def identities(suite):
 for item in suite:
  if isinstance(item,unittest.TestSuite):yield from identities(item)
  else:yield item.id()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--collect-only',action='store_true');a=parser.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 if any(n in sys.modules for n in MODULES):raise ValueError('clean isolated process required')
 suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES);names=sorted(identities(suite))
 expected=json.loads((SOURCE/'test-identities.json').read_bytes())
 if names!=expected or len(names)!=9 or len(set(names))!=len(names):raise ValueError('exact 9 identities differ')
 for n in (*MODULES,'audit_math'):
  mod=sys.modules.get(n)
  if mod is None or not Path(mod.__file__).resolve().is_relative_to(SOURCE):raise ValueError('isolated source differs: '+n)
 receipt={'schema':'jope.rl_spec_audit-offline-tests/v1','started_utc_ns':time.time_ns(),'source_sha':os.environ.get('JOPE_SOURCE_SHA'),
  'collected':len(names),'identities':names,'sources':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sorted(SOURCE.rglob('*'))if f.is_file()},
  'gp_executions':0,'s3_requests':0,'gpu_execution':False,'production_queue_mutations':0,
  'scope':'Offline CPU fixtures only. Mathematical audit and launcher qualification do not establish actor learning, resources, launch or GP outcomes.'}
 if a.collect_only:receipt.update(executed=0,passed=None,execution_state='not_run_collection_only')
 else:
  with(a.output/'tests.log').open('x')as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
  receipt.update(executed=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
   passed=result.wasSuccessful()and result.testsRun==9 and not result.skipped and not result.expectedFailures,
   execution_state='executed',log_sha256=hashlib.sha256((a.output/'tests.log').read_bytes()).hexdigest())
 receipt['finished_utc_ns']=time.time_ns();(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:v for k,v in receipt.items()if k not in ('identities','sources')}))
 if not a.collect_only and receipt['passed']is not True:raise SystemExit(1)

if __name__=='__main__':main()
