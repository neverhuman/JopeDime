"""Bounded selected-result qualification; durable owned events precede any work.

A successful response is NOT a full cutoff, target export, or admission receipt.
Run inside the owner's reviewed CPU1/512MiB/swap0/noGPU service. This program
has no launch, credential configuration, GP, PUT or database mutation command.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
from pathlib import Path
import time
from .readonly_io import canonical, read_bound, need
from . import callbacks


def physical_envelope():
    need(sys.platform=='linux','qualified Linux host required')
    row=next(x.split(':',2)[2]for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::'))
    group=Path('/sys/fs/cgroup')/row.lstrip('/')
    need('..'not in group.parts,'cgroup path')
    quota,period=(group/'cpu.max').read_text().split()
    need(quota!='max' and int(quota)<=int(period) and int((group/'memory.max').read_text())<=512*1024**2
         and int((group/'memory.swap.max').read_text())==0,'CPU1/512MiB/swap0 maximum')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'no GPU/single BLAS')
    return {'cpu_stat':(group/'cpu.stat').read_text(),'memory_current':int((group/'memory.current').read_text()),'cgroup':str(group)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--config-sha',required=True)
    args=parser.parse_args();started=time.time();before=physical_envelope()
    raw=read_bound({'path':str(Path(args.config).absolute()),'sha256':args.config_sha},2*1024**2)
    journal=callbacks.start_evidence(raw,args.config_sha)
    try:
        config=json.loads(raw);need(config['mode']=='selected-qualification','this CLI only qualifies selected results; no full export')
        setup=callbacks.configure(raw,args.config_sha);result=callbacks.qualify_selected()
        result.update(setup=setup,started_unix=started,finished_unix=time.time(),cgroup_before=before,cgroup_after=physical_envelope())
        journal.emit('qualification.passed',{'config_sha256':args.config_sha,'selected_results':len(result['results'])})
        result=journal.finish(result)
        print(canonical(result).decode());return 0
    except BaseException as error:
        result=callbacks.failure_custody(error)
        print(canonical(result).decode())
        return 1

if __name__=='__main__':
    raise SystemExit(main())
