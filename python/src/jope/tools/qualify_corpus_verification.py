#!/usr/bin/env python3
"""Bounded shadow-controller qualification; production shards are read-only."""
import argparse
from pathlib import Path
import time

import continuous_learning as controller
import corpus_verification
import experiment_runtime as runtime
from promotion_gate import immutable
from qualification_budget import Budget
from resume_expired_cycle import read,require,sha


def shadow(root,output,request):
    supervisor=controller.Supervisor.__new__(controller.Supervisor)
    supervisor.root=root;supervisor.output=output/'controller';supervisor.output.mkdir(exist_ok=True)
    supervisor.state_path=supervisor.output/'state.json';supervisor.journal=supervisor.output/'journal.jsonl'
    supervisor.manifest_path=root/request['controller_manifest']
    require(sha(supervisor.manifest_path)==request['controller_manifest_sha256'],'production controller identity changed')
    supervisor.manifest_sha=request['controller_manifest_sha256']
    supervisor.m=read(supervisor.manifest_path)
    require(request['shadow_overrides']==dict(automatic_training=False,async_corpus_verification=True),
            'shadow configuration changed')
    supervisor.m.update(request['shadow_overrides'])
    supervisor.children={};supervisor.stopping=False
    if supervisor.state_path.exists():supervisor.s=read(supervisor.state_path)
    else:
        corpus=str((output/'snapshot/corpus').relative_to(root))
        supervisor.s=dict(manifest_sha256=supervisor.manifest_sha,tasks={},snapshots=[],covered_searches=[],
            last_snapshot_at=0,teaching_sources=[],generations=[],checkpoint='unused',serving_corpus='unused')
        supervisor.s['tasks']['pack']=dict(kind='pack',command=['already-completed-private-metadata-copy'],
            receipt=corpus+'/provenance.json',output=corpus,created_at=1,attempts=1,retry_at=0,
            covered_searches=[],sources=[])
        supervisor.save('private_fixture_created')
    return supervisor


def run(root,output,budget_output):
    request=read(output/'request.json');budget=Budget(budget_output)
    row=read(budget.path)['reservations'][output.name]
    require(row['core_seconds']==420 and row['node_seconds']==0 and not row['complete'],'qualification ceiling missing')
    require(request['source_bindings']==corpus_verification.source_bindings(),'qualification source changed')
    started=time.monotonic();supervisor=None;result=dict(status='inconclusive')
    try:
        corpus=output/'snapshot/corpus';corpus.mkdir(parents=True)
        for name,digest in request['metadata'].items():
            source=root/request['corpus']/name
            data=source.read_bytes();require(sha(source)==digest,'source metadata changed')
            runtime.atomic_bytes(corpus/name,data);require(sha(corpus/name)==digest,'metadata copy changed')
        supervisor=shadow(root,output,request)
        supervisor.finish_task('pack')
        require(supervisor.s['snapshots']==[] and 'pack_verify' in supervisor.s['tasks'],
                'pack committed before verification')
        intervals=[];last=time.monotonic();body_seconds=[];busy_polls=0
        while True:
            now=time.monotonic();intervals.append(now-last);last=now
            require(now-started<150,'shadow verification ceiling exhausted')
            require(set(supervisor.s['tasks'])<= {'pack','pack_verify'},'shadow acquired a non-verification task')
            body=time.monotonic();supervisor.start_tasks();body_seconds.append(time.monotonic()-body)
            entry=supervisor.children.get('pack_verify')
            if entry:
                child,log,lease,_=entry
                if child.poll() is not None:
                    require(child.returncode==0,'corpus verifier child failed')
                    log.close();lease.close();del supervisor.children['pack_verify']
                    break
                busy_polls+=1
            time.sleep(.25)
        task=supervisor.s['tasks']['pack_verify'];receipt=root/task['receipt']
        require(receipt.exists() and supervisor.s['snapshots']==[],'missing receipt or premature state commit')
        # Reconstruct solely from durable state after publication, before commit.
        supervisor=shadow(root,output,request)
        require(set(supervisor.s['tasks'])=={'pack','pack_verify'},'shadow recovery task set changed')
        supervisor.start_tasks();supervisor.start_tasks()
        require(len(supervisor.s['snapshots'])==1 and supervisor.s['tasks']=={},'restart duplicated or lost a commit')
        snapshot=supervisor.s['snapshots'][0];expected=read(corpus/'index.json')
        require(snapshot['records']==expected['records'] and snapshot['records_sha256']==expected['records_sha256'],
                'verified corpus differs from the reference manifest')
        require(busy_polls>=2,'verification did not overlap observable controller polls')
        require(max(body_seconds)<10,'a shadow scheduling call exceeded the polling target')
        result.update(status='passed',records=snapshot['records'],busy_polls=busy_polls,
            maximum_scheduling_call_seconds=max(body_seconds),maximum_poll_interval_seconds=max(intervals),
            verifier_sha256=sha(receipt),verifier=read(receipt),restart_committed_snapshots=1,
            controller_stage_timings=getattr(supervisor,'stage_timings',[]))
    except (OSError,RuntimeError,ValueError,KeyError) as error:result['reason']=str(error)
    finally:
        if supervisor is not None:
            runtime.stop([entry[0] for entry in supervisor.children.values()])
            for _,log,lease,_ in supervisor.children.values():
                if log:log.close()
                if lease:lease.close()
        result.update(request_sha256=sha(output/'request.json'),seconds=time.monotonic()-started,
            production_state_modified=False,gpu_work_dispatched=False,final_tests_opened=False)
        immutable(output/'complete.json',result);budget.complete(output.name,output/'complete.json')
    require(result['status']=='passed',result.get('reason','qualification incomplete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','budget-output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.output.resolve(),args.budget_output.resolve())
