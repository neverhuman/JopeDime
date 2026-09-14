#!/usr/bin/env python3
"""Compare immutable group publication against an existing production reference."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

import experiment_runtime as runtime
from promotion_gate import immutable
from qualification_budget import Budget
from resume_expired_cycle import atomic,read,require,sha


def prepare_request(root,manifest,binary,output,ceiling,resume=None):
    if resume is not None:
        previous=read(resume/'complete.json')
        require(previous['status']=='inconclusive','only an inconclusive packing attempt may resume')
        require(previous['request_sha256']==sha(resume/'request.json'),'previous packing request changed')
        request=read(resume/'request.json')
        require(request['schema']=='parallel.teaching_store_qualification.v1','unknown packing request')
        require(request['executable_sha256']==sha(binary),'resume executable changed')
        require(request['contexts_sha256']==sha(root/request['contexts']),'resume contexts changed')
        require(request['reference_index_sha256']==sha(root/request['reference']/'index.json'),
                'resume reference changed')
        request['store']=request.get('store',str((resume/'store').relative_to(root)))
        request['resume']=dict(path=str((resume/'complete.json').relative_to(root)),
                               sha256=sha(resume/'complete.json'))
        request['ceiling_seconds']=ceiling
        request['driver_sha256']=sha(Path(__file__))
        return request
    m=read(manifest);state=read(root/m['output']/'state.json')
    reference=root/state['snapshots'][-1]['corpus']
    source_request=read(reference.parent/'snapshot-request.json')
    sources=list(source_request['sources'])
    return dict(schema='parallel.teaching_store_qualification.v1',manifest_sha256=sha(manifest),
        executable=str(binary.relative_to(root)),executable_sha256=sha(binary),reference=str(reference.relative_to(root)),
        reference_index_sha256=sha(reference/'index.json'),contexts=m['contexts'],contexts_sha256=m['contexts_sha256'],
        sources=sources,ceiling_seconds=ceiling,cpu_allocation=[60,61],final_tests_opened=False,
        store=str((output/'store').relative_to(root)),driver_sha256=sha(Path(__file__)))


def run(root,manifest,binary,output,budget_output,ceiling,resume=None):
    output.mkdir(parents=True,exist_ok=True)
    require(not (output/'request.json').exists(),'qualification attempt already frozen')
    request=prepare_request(root,manifest,binary,output,ceiling,resume)
    reference=root/request['reference'];sources=request['sources']
    immutable(output/'request.json',request)
    identity=output.name
    budget=Budget(budget_output);budget.reserve(identity,core_seconds=2*ceiling)
    start=time.monotonic();stages={};result=dict(status='inconclusive',final_tests_opened=False)
    try:
        with (root/'runs/experiment-spare-cpus.lock').open('a') as lease:
            while True:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    require(time.monotonic()-start<ceiling,'CPU handoff exceeded qualification ceiling')
                    time.sleep(1)
            runtime._lease_fds=(*runtime._lease_fds,lease.fileno())
            def stage(name,args):
                require(sha(binary)==request['executable_sha256'],'qualification executable changed')
                atomic(output/'progress.json',dict(stage=name,timestamp=time.time(),completed=stages))
                t=time.monotonic()
                with (output/(name+'.log')).open('xb') as log:
                    children=[]
                    with runtime.owned_ranks(children):
                        child=runtime.popen(['taskset','-c','60,61',str(binary),'--root',str(root),*map(str,args)],
                            stdout=log,stderr=log,env=runtime.child_env(OMP_NUM_THREADS='2',RAYON_NUM_THREADS='2',
                            LD_LIBRARY_PATH='/usr/lib/python3/dist-packages/torch/lib'))
                        children.append(child)
                        while child.poll() is None:
                            require(time.monotonic()-start<ceiling,'qualification ceiling exhausted')
                            time.sleep(.5)
                        require(child.returncode==0,'qualification stage failed: '+name)
                stages[name]=dict(seconds=time.monotonic()-t,log_sha256=sha(output/(name+'.log')))
            observations=output/'observations.json';immutable(observations,[str(root/s) for s in sources])
            common=['build-corpus','--contexts',root/request['contexts'],'--vocabulary',reference/'vocabulary.json',
                '--qualification','--observation-manifest',observations,'--teaching-store',root/request['store'],
                '--progress',output/'packing-progress.json']
            stage('cold',[*common,'--output',output/'cold'])
            stage('warm_unchanged',[*common,'--output',output/'warm'])
            require(sha(output/'cold/index.json')==sha(output/'warm/index.json'),'unchanged publication changed identity')
            require(read(output/'warm/packing-timing.json')['rebuilt_groups']==0,'unchanged groups were rebuilt')
            stage('export',['export-teaching-corpus','--corpus',output/'cold','--output',output/'export'])
            artifacts={}
            for name in ('records.bin','labels.jsonl','index.json','provenance.json','vocabulary.json'):
                actual,expected=sha(output/'export'/name),sha(reference/name)
                artifacts[name]=dict(actual=actual,expected=expected,equal=actual==expected)
            require(all(row['equal'] for row in artifacts.values()),'incremental export differs from reference')
            result.update(status='passed',artifacts=artifacts,records=read(reference/'index.json')['records'],
                cold=read(output/'cold/packing-timing.json'),warm=read(output/'warm/packing-timing.json'))
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        result['reason']=str(error)
    finally:
        result.update(seconds=time.monotonic()-start,stages=stages,request_sha256=sha(output/'request.json'))
        immutable(output/'complete.json',result);budget.complete(identity,output/'complete.json')
        atomic(output/'progress.json',result)
    require(result['status']=='passed',result.get('reason','qualification inconclusive'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','manifest','binary','output','budget-output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ceiling',type=int,default=1800)
    p.add_argument('--resume',type=Path,help='inconclusive attempt to continue with unchanged inputs and a new reservation')
    a=p.parse_args();run(a.root.resolve(),a.manifest.resolve(),a.binary.resolve(),a.output.resolve(),a.budget_output.resolve(),a.ceiling,
        a.resume.resolve() if a.resume else None)
