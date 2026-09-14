"""Freeze successive strict corpora from independently completed native searches."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import time

import experiment_runtime as runtime
from promotion_gate import immutable
from qualification_budget import Budget
from resume_expired_cycle import atomic,read,require,sha


def completed_sources(root,jobs,expected):
    """Headers are commit markers. Native admission verifies each payload digest."""
    sources=[]
    for destination in sorted({row['output'] for row in jobs}):
        path=runtime.inside(root,destination)/'search.json'
        if not path.exists():continue
        receipt=read(path)
        if receipt.get('evaluator_sha256')!=expected:continue
        require(receipt['schema']=='parallel.search.v1' and receipt.get('completed_at',0)>0
                and receipt.get('final_tests_opened') is False,'incomplete current evaluator ledger')
        require(receipt.get('observations_sha256') and receipt['completed_evaluations']>0,
                'current evaluator ledger lacks outcomes')
        sources.append(str((path.parent/'observations.json').relative_to(root)))
    return sources


def run(root,manifest,build,inventory,output,budget_output,minimum=104,snapshots=2):
    output.mkdir(parents=True,exist_ok=True);m=read(manifest);binary=build/'target-bin/jope-parallel-native'
    b=read(build/'target-build.json');require(b['status']=='passed' and sha(binary)==b['bindings']['jope-parallel-native'],
        'strict publication requires tested native admission')
    require(sha(root/m['cpu_executable'])==sha(binary),'strict publication evaluator is not the active GP version')
    context=root/m['contexts'];vocabulary=root/m['monitor_corpus']/'vocabulary.json'
    require(read(inventory)['context_index_sha256']==sha(context),'family inventory context changed')
    policy=dict(schema='parallel.label_admission.v2',evaluator=str(binary.relative_to(root)),
        evaluator_sha256=sha(binary),evaluator_protocol='featmath_training_fold_normalized_v1',
        runtime_sha256=m.get('fit_runtime_sha256'), numerical_contract_sha256=m.get('fit_numerical_contract_sha256'),
        authority_root=m.get('fit_authority_root'), authority_witness=m.get('fit_authority_witness'),
        authority_instance=m.get('fit_authority_instance'))
    require(all(isinstance(policy[key], str) and policy[key] for key in
                ('runtime_sha256', 'numerical_contract_sha256', 'authority_root', 'authority_witness', 'authority_instance')),
            'strict publication requires frozen numerical runtime and approved fit authority')
    immutable(output/'admission.json',policy)
    refs={str(p.relative_to(root)):sha(p) for p in (manifest,build/'target-build.json',binary,context,vocabulary,inventory,output/'admission.json',Path(__file__))}
    request=dict(schema='parallel.strict_corpus_collection.v1',bindings=refs,minimum_new_searches=minimum,
        snapshots=snapshots,label_policy='strict_midrank_v4',publication_ceiling_seconds=300,
        family_inventory=str(inventory.relative_to(root)),final_tests_opened=False)
    immutable(output/'request.json',request)
    budget=Budget(budget_output);sources=[]
    for number in range(snapshots):
        destination=output/f'snapshot-{number:06}';complete=destination/'complete.json'
        if complete.exists():
            result=read(complete);require(result['status']=='passed','inconclusive strict publication retained')
            sources=read(destination/'sources.json');continue
        pending=destination/'sources.json'
        if not pending.exists():
            while True:
                jobs=[]
                for path in (root/'manifests/generated/gp-expanded-vocabulary-jobs.json',root/'runs/gp-priority-jobs.json'):
                    if path.exists():jobs.extend(read(path)['jobs'])
                found=completed_sources(root,jobs,policy['evaluator_sha256'])
                new=[path for path in found if path not in set(sources)]
                atomic(output/'progress.json',dict(timestamp=time.time(),phase='awaiting independently committed searches',
                    snapshot=number,completed_sources=len(found),new_sources=len(new),minimum_new_searches=minimum))
                if len(new)>=minimum:
                    sources=[*sources,*new];destination.mkdir(exist_ok=True);immutable(pending,sources);break
                time.sleep(10)
        else:sources=read(pending)
        for path,digest in refs.items():require(sha(root/path)==digest,'strict collection input changed: '+path)
        key=output.name+f'-snapshot-{number}';result=dict(status='inconclusive');reserved=False;start=None
        try:
            with (root/'runs/experiment-spare-cpus.lock').open('a') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX);budget.reserve(key,core_seconds=600);reserved=True;start=time.monotonic()
                previous=runtime._lease_fds;runtime._lease_fds=(*previous,lease.fileno())
                try:
                    command=[str(binary),'--root',str(root),'build-corpus','--contexts',str(context),
                        '--vocabulary',str(vocabulary),'--observation-manifest',str(pending),
                        '--output',str(destination/'corpus'),'--teaching-store',str(output/'store'),
                        '--label-policy','strict_midrank_v4','--admission-manifest',str(output/'admission.json'),
                        '--qualification','--progress',str(destination/'packing-progress.json')]
                    children=[]
                    with (destination/'pack.log').open('ab') as log,runtime.owned_ranks(children):
                        child=runtime.popen(command,cwd=root,env=runtime.child_env(
                            LD_LIBRARY_PATH='/usr/lib/python3/dist-packages/torch/lib',OMP_NUM_THREADS='2',RAYON_NUM_THREADS='2'),
                            stdout=log,stderr=log);children.append(child);child.wait(timeout=300)
                        require(child.returncode==0,'strict corpus publication failed')
                    index=read(destination/'corpus/index.json')
                    require(index['admission_policy']==policy and index['label_policy']=='strict_midrank_v4'
                            and index['records']>0,'strict corpus has no admissible teaching records')
                    result.update(status='passed',index_sha256=sha(destination/'corpus/index.json'),
                        provenance_sha256=sha(destination/'corpus/provenance.json'),records=index['records'],groups=len(index['shards']))
                finally:runtime._lease_fds=previous
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:result['reason']=str(error)
        finally:
            result.update(request_sha256=sha(output/'request.json'),sources_sha256=sha(pending),source_count=len(sources),
                seconds=None if start is None else time.monotonic()-start,final_tests_opened=False)
            immutable(complete,result)
            if reserved:budget.complete(key,complete)
        require(result['status']=='passed',result.get('reason','strict publication inconclusive'))
    immutable(output/'complete.json',dict(status='passed',request_sha256=sha(output/'request.json'),
        snapshots=[dict(path=str((output/f'snapshot-{i:06}/complete.json').relative_to(root)),
            sha256=sha(output/f'snapshot-{i:06}/complete.json')) for i in range(snapshots)],final_tests_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','manifest','build','inventory','output','budget-output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minimum-new-searches',type=int,default=104);p.add_argument('--snapshots',type=int,default=2)
    a=p.parse_args();require(a.minimum_new_searches>0 and a.snapshots>=2,'strict freshness needs successive nonempty collections')
    run(a.root.resolve(),a.manifest.resolve(),a.build.resolve(),a.inventory.resolve(),a.output.resolve(),a.budget_output.resolve(),a.minimum_new_searches,a.snapshots)
