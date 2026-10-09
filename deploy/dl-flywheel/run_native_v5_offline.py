"""Isolated57 native-v5 CPU fixtures: original48 plus nine cold-build checks.

Only this test process uses a fixed fixture wallclock before source imports, so
clock-bound historical policies remain testable after their real deadline. Real
monotonic timers and real SQLite/filesystem concurrency remain in use. Production
source/deadline bytes are unchanged. Spawned test children execute this prefix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'pipelines/hpc_verified_v5'
FIXTURE_WALLCLOCK=1791550000.0
_REAL_TIME_NS=time.time_ns
_CLOCK=patch('time.time',return_value=FIXTURE_WALLCLOCK)
_CLOCK.start()
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(SOURCE))
MODULES=('test_native_successor','test_operations_census','test_operations_launch','test_native_build')


def identities(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from identities(item)
        else:yield item.id()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--collect-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    if any(n in sys.modules for n in MODULES):raise ValueError('v5 suite requires clean isolated imports')
    suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES);names=sorted(identities(suite))
    expected=json.loads((SOURCE/'test-identities.json').read_bytes())
    if names!=expected or len(set(names))!=len(names) or len(names)!=57:raise ValueError('exact57 native-v5 identities differ')
    for n in (*MODULES,'native_dispatch','native_build','native_global_guard','readonly_snapshot','operations_census','operations_launch','neural_outbox','jope_flywheel'):
        # Dispatcher is imported by its fixture methods, so force its inert import now.
        if n=='native_dispatch':__import__(n)
        mod=sys.modules.get(n)
        if mod is None or not Path(mod.__file__).resolve().is_relative_to(SOURCE):raise ValueError('isolated native-v5 source differs: '+n)
    receipt={'schema':'jope.native-v5-offline-tests/v1','source_sha':os.environ.get('JOPE_SOURCE_SHA'),
      'started_utc_ns':_REAL_TIME_NS(),'collected':len(names),'identities':names,'fixture_wallclock':FIXTURE_WALLCLOCK,
      'fixture_wallclock_scope':'Only isolated test process and spawned fixture children; production clock/global deadline unchanged.',
      'sources':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()for x in sorted(SOURCE.rglob('*'))if x.is_file()},
      'gp_executions':0,'s3_requests':0,'gpu_execution':False,'production_queue_mutations':0,
      'scope':'Local offline fixtures only; bounded operational templates require current inventory, review and live qualification separately.'}
    if a.collect_only:receipt.update(executed=0,passed=None,execution_state='not_run_collection_only')
    else:
        with(a.output/'tests.log').open('x')as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        receipt.update(executed=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
          passed=result.wasSuccessful()and result.testsRun==57 and not result.skipped and not result.expectedFailures,
          execution_state='executed',log_sha256=hashlib.sha256((a.output/'tests.log').read_bytes()).hexdigest())
    receipt['finished_utc_ns']=_REAL_TIME_NS();(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items()if k not in ('identities','sources')}))
    if not a.collect_only and receipt['passed']is not True:raise SystemExit(1)

if __name__=='__main__':main()
