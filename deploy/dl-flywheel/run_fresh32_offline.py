"""Run only the isolated37 fresh32 CPU fixtures, without deployment authority."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'pipelines/encoding/fresh32'
NAMES=('campaign','kumo_queue','kumo_controller','kumo_publish','kumo_intake','evergreen_kumos_packet','test_fresh32')


# Fixture-only wall clock (PR #283 independent review finding B2).  The frozen campaign modules cap every window at
# the historical global deadline 1791592146.380285 (2026-10-10T00:29:06Z) while these fixtures build their windows
# from time.time(), so the suite started failing as real time crossed that constant.  Inside this fixture process
# only, the wall clock is shifted to a fixed epoch at which the whole suite passed (2026-10-09T23:00:00Z); elapsed
# time, monotonic clocks and file times are untouched and the shift is recorded in the receipt.
FIXTURE_WALL_EPOCH = 1791586800.0


def shift_fixture_wall_clock():
    real_time, real_time_ns = time.time, time.time_ns
    shift_ns = int(round((FIXTURE_WALL_EPOCH - real_time()) * 1e9))
    time.time = lambda: real_time() + shift_ns / 1e9
    time.time_ns = lambda: real_time_ns() + shift_ns
    return {'fixture_wall_epoch': FIXTURE_WALL_EPOCH, 'fixture_wall_shift_ns': shift_ns}


def identities(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from identities(item)
        else:yield item.id()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--collect-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    if any(name in sys.modules for name in NAMES):
        raise ValueError('fresh32 requires an isolated process with no prior campaign imports')
    clock=shift_fixture_wall_clock()
    sys.path.insert(0,str(SOURCE))
    suite=unittest.defaultTestLoader.loadTestsFromName('test_fresh32')
    names=sorted(identities(suite))
    expected=json.loads((SOURCE/'test-identities.json').read_bytes())
    if names!=expected or len(names)!=len(set(names)) or len(names)!=37:
        raise ValueError('exact37 fresh32 fixture identities differ')
    for name in NAMES:
        module=(sys.modules['test_fresh32'].worker if name=='evergreen_kumos_packet' else sys.modules.get(name))
        if module is None or Path(module.__file__).resolve()!=SOURCE/(name+'.py'):
            raise ValueError('isolated source import differs: '+name)
    started=time.time_ns()
    receipt={'schema':'jope.fresh32-offline-tests/v1','source_sha':os.environ.get('JOPE_SOURCE_SHA'),
             'started_utc_ns':started,**clock,'collected':len(names),'identities':names,
             'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(SOURCE.rglob('*'))if p.is_file()},
             'gp_executions':0,'s3_requests':0,'gpu_execution':False,'production_queue_mutations':0,
             'scope':'Offline CPU fixtures only. Frozen configuration is dated deployment evidence, not portable defaults or launch permission.'}
    if args.collect_only:
        receipt.update(executed=0,passed=None,execution_state='not_run_collection_only')
    else:
        with(args.output/'tests.log').open('x')as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        receipt.update(executed=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                       skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
                       passed=result.wasSuccessful()and result.testsRun==37 and not result.skipped and not result.expectedFailures,
                       execution_state='executed',log_sha256=hashlib.sha256((args.output/'tests.log').read_bytes()).hexdigest())
    receipt['finished_utc_ns']=time.time_ns()
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items()if k not in('identities','sources')}))
    if not args.collect_only and receipt['passed']is not True:raise SystemExit(1)


if __name__=='__main__':main()
