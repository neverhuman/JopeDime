"""Isolated25v1+24operational+21permanent-registry fresh-GP fixtures; no native fit/network/live claims."""
import argparse,hashlib,json,os,subprocess,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'pipelines/fresh_gp_v2'
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
FAMILIES={'v1':(('test_fresh128',),25),'revision2':(('test_operational','test_receipt_operational'),24),'revision3':(('test_permanent_broker',),21)}
def identities(suite):
 for x in suite:
  if isinstance(x,unittest.TestSuite):yield from identities(x)
  else:yield x.id()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--family',choices=tuple(FAMILIES));p.add_argument('--collect-only',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 if a.family is None:
  rows=[]
  for name in FAMILIES:
   argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--family',name,'--output',str(a.output/name)]
   if a.collect_only:argv.append('--collect-only')
   child=subprocess.run(argv,check=False);receipt=json.loads((a.output/name/'receipt.json').read_bytes())
   if child.returncode or receipt['collected']!=FAMILIES[name][1]:raise ValueError('isolated child failed: '+name)
   rows.append(receipt)
  names=[n for r in rows for n in r['identities']]
  if len(names)!=len(set(names))or len(names)!=70:raise ValueError('70unique identities required')
  receipt={'schema':'jope.fresh-gp-offline-tests/v2','source_sha':os.environ.get('JOPE_SOURCE_SHA'),'collected':70,
   'executed':sum(r['executed']for r in rows),'passed':None if a.collect_only else all(r['passed']is True for r in rows),
   'identities':sorted(names),'families':rows,'execution_state':'not_run_collection_only'if a.collect_only else'executed',
   'scope':'70isolatedCPUfixtures only; no live128context qualification, GP, S3, broker launch or optimizer admission.'}
 else:
  source=SOURCE/a.family;modules,expected_count=FAMILIES[a.family]
  if any(n in sys.modules for n in modules):raise ValueError('clean isolated process required')
  sys.path[:0]=[str(source),str(source/'runtime')]
  suite=unittest.defaultTestLoader.loadTestsFromNames(modules);names=sorted(identities(suite))
  expected=json.loads((source/'test-identities.json').read_bytes())
  if names!=expected or len(names)!=expected_count:raise ValueError('exact family identities differ')
  for n in(*modules,'global_attempts','execution_boundary','jope_flywheel.program_worker'):
   mod=__import__('importlib').import_module(n)
   if mod is None or not Path(mod.__file__).resolve().is_relative_to(source):raise ValueError('isolated source origin differs: '+n)
  receipt={'schema':'jope.fresh-gp-family-tests/v2','family':a.family,'source_sha':os.environ.get('JOPE_SOURCE_SHA'),
   'started_utc_ns':time.time_ns(),'collected':len(names),'identities':names,
   'sources':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sorted(source.rglob('*'))if f.is_file()and'__pycache__'not in f.parts},
   'fixture_wall_unix':1791558000.0 if a.family in('revision2','revision3')else None,'production_global_deadline_unchanged':True,
   'native_fits':0,'s3_requests':0,'gpu_execution':False,'live_queue_mutations':0}
  if a.collect_only:receipt.update(executed=0,passed=None,execution_state='not_run_collection_only')
  else:
   with(a.output/'tests.log').open('x')as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
   receipt.update(executed=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
    passed=result.wasSuccessful()and result.testsRun==expected_count and not result.skipped and not result.expectedFailures,
    execution_state='executed',log_sha256=hashlib.sha256((a.output/'tests.log').read_bytes()).hexdigest())
  receipt['finished_utc_ns']=time.time_ns()
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:v for k,v in receipt.items()if k not in('identities','sources','families')}))
 if not a.collect_only and receipt['passed']is not True:raise SystemExit(1)
if __name__=='__main__':main()
