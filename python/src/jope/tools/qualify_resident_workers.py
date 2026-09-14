#!/usr/bin/env python3
"""Matched-batch resident timing and fail-closed acceptance checks; recovery remains a separate gate."""
import argparse
from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import experiment_runtime as runtime
from promotion_gate import immutable
from qualification_budget import Budget
from qualify_regression_gain import gpu_handoff,require_collection
from resume_expired_cycle import atomic,read,require,sha
from resident_gpu_pool import process_identity,reconcile_tasks,verify_pool_ready


@dataclass
class ObservedTask:
    child: subprocess.Popen
    start_path: Path
    start_sha256: str
    process: dict
    command: list
    outcome_path: Path = None
    outcome_sha256: str = None
    log_path: Path = None

    @classmethod
    def capture(cls, child, command, path):
        require(isinstance(child,subprocess.Popen),'original task process unavailable')
        identity=process_identity(child.pid)
        deadline=time.monotonic()+1
        while child.poll() is None and time.monotonic()<deadline:
            if sha(Path(f'/proc/{child.pid}/exe'))==sha(Path(command[0])):break
            time.sleep(.01)
        require(identity is not None and process_identity(child.pid)==identity
                and sha(Path(f'/proc/{child.pid}/exe'))==sha(Path(command[0])),
                'original task loaded executable was not observed')
        immutable(path,dict(schema='parallel.resident_task_process_start.v1',process=identity,
                  command=command,executable_sha256=sha(Path(command[0])),timestamp=time.time(),
                  observer_process=process_identity(os.getpid()),observer_source_sha256=sha(Path(__file__)),
                  script_inputs={arg:sha(Path(arg)) for arg in command[1:]
                                 if Path(arg).suffix=='.py' and Path(arg).is_file()},
                  runtime_closure_qualified=False))
        return cls(child,path,sha(path),identity,command)

    def finish(self, path, log):
        require(self.child.poll() is not None,'original task has no observed exit')
        require(sha(self.start_path)==self.start_sha256,'original task process record changed')
        require(process_identity(self.child.pid)!=self.process,'original task still retains process custody')
        immutable(path,dict(schema='parallel.resident_task_process_exit.v1',start_sha256=self.start_sha256,
                  process=self.process,observed_exit=self.child.returncode,timestamp=time.time(),
                  log_sha256=sha(log),observer_source_sha256=sha(Path(__file__))))
        self.outcome_path=path
        self.outcome_sha256=sha(path);self.log_path=log

    def require_failed(self):
        require(self.outcome_path is not None and self.child.poll() not in (None,0),
                'original failed process observation required')
        require(sha(self.outcome_path)==self.outcome_sha256,'original process outcome identity changed')
        outcome=read(self.outcome_path)
        require(sha(self.start_path)==self.start_sha256 and outcome.get('start_sha256')==self.start_sha256
                and outcome.get('process')==self.process
                and outcome.get('observed_exit')==self.child.returncode
                and outcome.get('log_sha256')==sha(self.log_path),
                'original process outcome identity changed')


def verify_refused_attempt(destination, *, observed=None, root=None, pool=None, executable=None):
    require(isinstance(observed,ObservedTask),'original observed process required; refusal declarations cannot qualify')
    observed.require_failed()
    refusal=read(destination/'resident-reconciliation-required.json')
    launches=list(destination.glob('launch-*'))
    require(len(launches)==1 and refusal['automatic_retry_permitted'] is False,
            'uncertain accepted work was dispatched under a new launch')
    controller=read(launches[0]/'controller.json')
    require(controller.get('rank_failure_injected') is True and controller.get('passed') is False,
            'resident refusal did not exercise the injected accepted-task failure')
    require(Path(refusal['launch']).resolve()==launches[0].resolve() and refusal['attempt']==0,
            'resident refusal is not bound to its original launch')
    require(not (destination/'complete.json').exists(),'failed task published completion')
    require(root is not None and pool is not None and executable is not None,
            'original resident acceptance evidence required')
    reconciliation=reconcile_tasks(root,pool,executable)
    matching=[row for row in reconciliation['records']
              if row.get('request',{}).get('task',{}).get('kind')=='Train'
              and row['request']['task'].get('run')==str(destination)
              and row['request']['task'].get('launch')==str(launches[0])]
    require(matching and any(row.get('accepted_sha256') for row in matching),
            'original accepted resident task unavailable')
    require(all(row['status']!='committed_success' for row in matching),
            'failed original launch conflicts with successful resident completion')
    start=read(observed.start_path)
    command=start['command']
    require('--output' in command and command[command.index('--output')+1]==str(destination)
            and '--fail-rank' in command and start['executable_sha256']==executable,
            'original observed command does not bind the failed launch')
    return dict(status='observed_ambiguous_acceptance_refusal',
        receipt_sha256=sha(destination/'resident-reconciliation-required.json'),
        controller_sha256=sha(launches[0]/'controller.json'),
        process_start_sha256=observed.start_sha256,process_exit_sha256=sha(observed.outcome_path),
        original_tasks=matching,automatic_retry_permitted=False,
        rank_failure_injection_qualified=False,recovery_qualified=False,
        limitation='Native controller has no original per-rank fault-observer receipt; its injected Boolean is a declaration.')


def run(root,manifest,binary,output,budget_output,ceiling,corpus_override=None,reference_override=None):
    root=runtime.checked_path(root);manifest=runtime.inside(root,manifest);binary=runtime.inside(root,binary)
    output=runtime.output_root(root,output);budget_output=runtime.checked_path(budget_output)
    if corpus_override is not None:corpus_override=runtime.inside(root,corpus_override)
    if reference_override is not None:reference_override=runtime.inside(root,reference_override)
    output.mkdir(parents=True,exist_ok=True)
    require(not (output/'request.json').exists(),'worker qualification already frozen')
    m=read(manifest);state=read(root/m['output']/'state.json')
    corpus=corpus_override or root/state['corpus'];checkpoint=root/state['serving_checkpoint']
    request=dict(schema='parallel.resident_worker_qualification.v2',manifest_sha256=sha(manifest),
        executable=str(binary.relative_to(root)),executable_sha256=sha(binary),corpus=str(corpus.relative_to(root)),
        corpus_sha256=sha(corpus/'index.json'),checkpoint=str(checkpoint.relative_to(root)),
        checkpoint_sha256=sha(checkpoint/'state.json'),global_batch=2048,steps=128,worlds=[8,4,2],
        ceiling_seconds=ceiling,final_tests_opened=False,
        scope='timing_and_ambiguous_acceptance_refusal',
        full_recovery_qualification=False)
    request['reference_corpus']=None if reference_override is None else str(reference_override.relative_to(root))
    request['reference_corpus_sha256']=None if reference_override is None else sha(reference_override/'index.json')
    request['driver_sha256']=sha(Path(__file__))
    request['python_bindings']={str(path.relative_to(root)):sha(path)
        for path in sorted(Path(__file__).resolve().parent.glob('*.py'))}
    immutable(output/'request.json',request)
    budget=Budget(budget_output);stages={};result=dict(status='inconclusive',final_tests_opened=False)
    reserved=False;start=None
    try:
        with gpu_handoff(root,output) as leases:
            budget.reserve(output.name,node_seconds=ceiling,core_seconds=16*ceiling)
            reserved=True;start=time.monotonic()
            # The handoff excludes production. Each native task acquires and
            # transfers this same flock independently, exercising production IPC.
            fcntl.flock(leases[-1],fcntl.LOCK_UN)
            pool=output/'pool'
            logs=[];children=[]
            with runtime.owned_ranks(children):
                try:
                    log=(output/'pool.log').open('xb');logs.append(log)
                    supervisor=runtime.popen([sys.executable,'-B',str(Path(__file__).with_name('resident_gpu_pool.py')),
                        '--root',str(root),'--binary',str(binary),'--pool',str(pool),'--world','8',
                        '--qualification-owner',str(output.relative_to(root))],stdout=log,stderr=log,
                        env=runtime.child_env(LD_LIBRARY_PATH='/usr/lib/python3/dist-packages/torch/lib'))
                    children.append(supervisor)
                    supervisor_identity=process_identity(supervisor.pid)
                    def check():
                        require_collection(root)
                        require(time.monotonic()-start<ceiling,'resident qualification ceiling exhausted')
                        require(supervisor.poll() is None,'resident supervisor died')
                    while not (pool/'pool.json').is_file() or read(pool/'pool.json')['phase']!='running':
                        check();time.sleep(.1)
                    verify_pool_ready(pool,8,request['executable_sha256'],supervisor_identity)
                    arms=([('reference8',8,False)] if reference_override else [])+[('cold8',8,False),('warm8',8,False),('warm4',4,False),('warm2',2,False),('refuse-replay2',2,True)]
                    for name,world,fail in arms:
                        check();require(sha(binary)==request['executable_sha256'],'resident executable changed')
                        atomic(output/'progress.json',dict(stage=name,timestamp=time.time(),completed=stages))
                        destination=output/name
                        teaching=reference_override if name=='reference8' else corpus
                        command=[str(binary),'--root',str(root),'train','--corpus',str(teaching),'--output',str(destination),
                            '--initialize',str(checkpoint),'--world',str(world),'--microbatch','256','--steps','128',
                            '--checkpoint-steps','64']
                        if fail:command.extend(['--fail-rank','0'])
                        log=(output/(name+'.log')).open('xb');logs.append(log);t=time.monotonic()
                        task=runtime.popen(command,stdout=log,stderr=log,env=runtime.child_env(JOPE_GPU_POOL=str(pool),
                            LD_LIBRARY_PATH='/usr/lib/python3/dist-packages/torch/lib'))
                        children.append(task)
                        observed=ObservedTask.capture(task,command,output/(name+'-process-start.json'))
                        while task.poll() is None:check();time.sleep(.25)
                        log.flush()
                        observed.finish(output/(name+'-process-exit.json'),output/(name+'.log'))
                        if fail:
                            require(task.returncode!=0,'injected resident failure was automatically retried')
                            stages[name]=dict(world=world,**verify_refused_attempt(destination,observed=observed,
                                root=root,pool=pool,executable=request['executable_sha256']))
                            continue
                        require(task.returncode==0,'resident training failed: '+name)
                        complete=read(destination/'complete.json')
                        launches=[p for p in destination.glob('launch-*') if (p/'controller.json').is_file()
                                  and read(p/'controller.json')['passed']]
                        require(len(launches)==1,'missing or duplicate completed communicator')
                        receipts=[read(launches[0]/f'complete-{r}.json') for r in range(world)]
                        require(all(r['steps']==128 and r['corpus_sha256']==sha(teaching/'index.json') for r in receipts),
                                'resident rank result identity changed')
                        require(len({r['parameter_sha256'] for r in receipts})==1,'resident ranks disagree')
                        if name.startswith('warm'):
                            require(all(r['resident_tensor_cache_hit'] for r in receipts),'immutable device tensors were restaged')
                        stages[name]=dict(world=world,global_batch=2048,optimizer_steps=128,wall_seconds=time.monotonic()-t,
                            task_receipt_sha256=sha(destination/'complete.json'),rank_receipts=receipts,
                            checkpoint=read(destination/'latest.json'),attempts=complete['attempts'])
                    require(stages['cold8']['rank_receipts'][0]['parameter_sha256']==stages['warm8']['rank_receipts'][0]['parameter_sha256'],
                            'resident reuse changed identical training continuation')
                    if reference_override:
                        require(stages['reference8']['rank_receipts'][0]['parameter_sha256']==stages['cold8']['rank_receipts'][0]['parameter_sha256'],
                                'sharded training differs from byte-equivalent reference training')
                    baseline_paths=[root/m['output']/'monitoring'/(request['checkpoint_sha256']+'.json'),
                        root/'runs/continuous-learning-20260909/monitoring'/(request['checkpoint_sha256']+'.json')]
                    baseline=read(next(path for path in baseline_paths if path.is_file()))
                    require(baseline['binding']['checkpoint_sha256']==request['checkpoint_sha256'],'monitor reference identity changed')
                    for repeat in range(2):
                        check();destination=output/f'monitor-{repeat}.json'
                        command=[str(binary),'--root',str(root),'monitor-checkpoint','--corpus',str(root/m['monitor_corpus']),
                            '--checkpoint',str(checkpoint),'--panel',str(root/m['monitor_panel']),'--output',str(destination)]
                        log=(output/f'monitor-{repeat}.log').open('xb');logs.append(log)
                        task=runtime.popen(command,stdout=log,stderr=log,env=runtime.child_env(JOPE_GPU_POOL=str(pool),
                            LD_LIBRARY_PATH='/usr/lib/python3/dist-packages/torch/lib'));children.append(task)
                        while task.poll() is None:check();time.sleep(.25)
                        require(task.returncode==0,'resident monitoring failed')
                        observed=read(destination)
                        for metric in ('sequence_nll','ranking_bt','ranking_listwise','runtime_huber'):
                            require(math.isclose(observed[metric],baseline[metric],rel_tol=1e-6,abs_tol=1e-7),
                                    'resident transform changed the frozen diagnostic: '+metric)
                        stages[f'monitor-{repeat}']=dict(receipt_sha256=sha(destination),matched_frozen_reference=True)
                    result.update(status='inconclusive',scoped_timing_and_refusal_passed=True,
                        cache=read(pool/'teaching-cache.json'),
                        reason='Full resident recovery remains unqualified: all-rank optimizer agreement, '
                               'partial checkpoints and explicit interrupted-task reconciliation require separate evidence')
                finally:
                    for log in logs:log.close()
    except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as error:
        result['reason']=str(error)
    finally:
        result.update(stages=stages,seconds=None if start is None else time.monotonic()-start,
                      request_sha256=sha(output/'request.json'))
        immutable(output/'complete.json',result)
        if reserved:budget.complete(output.name,output/'complete.json')
        atomic(output/'progress.json',result)
    require(result['status']=='passed',result.get('reason','worker qualification inconclusive'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','manifest','binary','output','budget-output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ceiling',type=int,default=600)
    p.add_argument('--corpus',type=Path);p.add_argument('--reference-corpus',type=Path)
    a=p.parse_args();run(a.root.absolute(),a.manifest.absolute(),a.binary.absolute(),a.output.absolute(),a.budget_output.absolute(),a.ceiling,
        a.corpus.absolute() if a.corpus else None,a.reference_corpus.absolute() if a.reference_corpus else None)
