"""Qualify only new variant bridge CPU fixtures; seal exact roster and source."""
import argparse,hashlib,json,os,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(ROOT),str(ROOT/'tests_native'),str(ROOT/'tests'),str(ROOT/'pipelines/continuous')]
    suite=unittest.defaultTestLoader.loadTestsFromName('test_variant_outbox.VariantTests')
    def ids(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite):yield from ids(item)
            else:yield item.id()
    identities=sorted(ids(suite));need=len(identities)==len(set(identities))==28
    if not need:raise ValueError('exact28 new fixture roster changed')
    started=time.time_ns()
    with(args.output/'tests.log').open('x')as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    paths=['jope_flywheel/variant_outbox.py','pipelines/variants/variant_dispatch.py','tests_native/test_variant_outbox.py','run_variant_offline.py']
    receipt={'schema':'jope.variant-outbox-offline/v1','started_utc_ns':started,'finished_utc_ns':time.time_ns(),
        'source_sha':os.environ.get('JOPE_SOURCE_SHA'),'collected':len(identities),'executed':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'identities':identities,'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in paths},
        'log_sha256':hashlib.sha256((args.output/'tests.log').read_bytes()).hexdigest(),
        'passed':result.wasSuccessful()and result.testsRun==28 and not result.skipped and not result.expectedFailures,
        'scope':'New CPU fixtures only; no S3 writes, native execution, live outbox mutations, GPU or optimizer.'}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items()if k not in('identities','sources')}))
    if not receipt['passed']:raise SystemExit(1)

if __name__=='__main__':main()
