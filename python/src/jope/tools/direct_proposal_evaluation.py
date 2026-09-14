"""Evaluate frozen predicted-best programs under complete, reusable identities."""
import hashlib
import math
import csv
import struct
from pathlib import Path
import resource
import time

import experiment_runtime as runtime
import label_admission as admission
from promotion_gate import artifact,canonical,immutable
from resume_expired_cycle import read,require,sha


def identity(root,context,program,binary,vocabulary):
    prepared=runtime.inside(root,context['prepared']);metadata=read(prepared/'prepared.json')
    require(context['partition'] in ('meta_train','meta_validation') and metadata['meta_partition']==context['partition'],
            'sealed or conflicting proposal evaluation context')
    require(sha(prepared/'prepared.json')==context['prepared_sha256'],'prepared evaluation identity changed')
    require(metadata.get('task')=='regression','direct evaluation requires regression task semantics')
    cuts=[c for c in metadata['cuts'] if c['index']==context['cut']]
    require(len(cuts)==1,'missing or ambiguous direct evaluation cut');cut=cuts[0]
    directory=prepared/f"cut-{context['cut']}";preprocessing=read(directory/'preprocessing.json')
    files=[('preprocessing.json','preprocessing_sha256')]
    if metadata.get('storage') is None:files.extend([('train.csv','training_sha256'),('validation.csv','validation_sha256')])
    # Compact prepared views have no cut CSV files. Native load_view verifies
    # their reconstructed content against these frozen training/validation hashes.
    for name,key in files:
        require(sha(directory/name)==cut[key],'evaluation cut/preprocessing content changed')
    scale=preprocessing['target_scale'];require(admission.finite(scale) and scale>0,'invalid target scale')
    require(all(context.get(k,cut[k])==cut[k] for k in ('training_sha256','validation_sha256')),'proposal cut identity changed')
    fit_training=None
    if metadata.get('storage') is None:
        with (directory/'train.csv').open(newline='') as stream:rows=[[float(v) for v in row] for row in csv.reader(stream)]
        require(rows and len(rows[0])>=2 and all(len(row)==len(rows[0]) and all(math.isfinite(v) for v in row) for row in rows),
                'invalid numerical training rows')
        digest=hashlib.sha256(b'parallel.ordered_training.f64.v1\0')
        digest.update(struct.pack('<QQ',len(rows),len(rows[0])-1))
        for row in rows:digest.update(struct.pack('<'+'d'*len(row),*row))
        fit_training=digest.hexdigest()
    return dict(program=program,program_sha256=hashlib.sha256(program.encode()).hexdigest(),
        family=context['family'],partition=context['partition'],prepared_sha256=context['prepared_sha256'],cut=context['cut'],
        training_sha256=cut['training_sha256'],validation_sha256=cut['validation_sha256'],
        fit_training_sha256=fit_training,
        preprocessing_sha256=cut['preprocessing_sha256'],target_scale=scale,metric='rmse',units='original_target',
        evaluator_sha256=sha(binary),evaluator_protocol='featmath_training_fold_normalized_v1',
        vocabulary_sha256=sha(vocabulary),evaluation_seed=0,resource_protocol='single_candidate_30s_8gib_v1',
        feature_view_sha256=hashlib.sha256(canonical(dict(protocol='raw_training_fold_v1',
            training_sha256=cut['training_sha256'],preprocessing_sha256=cut['preprocessing_sha256']))).hexdigest())


def approved_policy(root,binary,policy):
    require(isinstance(policy,dict) and policy.get('schema')=='parallel.label_admission.v2'
            and policy.get('evaluator_protocol')=='featmath_training_fold_normalized_v1',
            'frozen native admission policy required before direct dispatch')
    for field in ('evaluator_sha256','runtime_sha256','numerical_contract_sha256','authority_instance'):
        require(admission.sha256_value(policy.get(field)),'invalid approved policy: '+field)
    require(runtime.inside(root,policy['evaluator'])==binary.resolve()
            and sha(binary)==policy['evaluator_sha256'],'approved direct evaluator changed')
    for field in ('authority_root','authority_witness'):
        require(isinstance(policy.get(field),str) and policy[field]
                and not Path(policy[field]).is_absolute(),'workspace-relative approved authority required')
        runtime.inside(root,policy[field])


def reference(root,path,expected=None):
    path=runtime.inside(root,path)
    digest=sha(path)
    require(expected is None or digest==expected,'original direct artifact changed')
    return dict(path=str(path.relative_to(root)),sha256=digest)


def finish(root,directory,bound,policy,attempt_ref,code,elapsed,timed_out,pid):
    """Original supervisor observation is immutable; no file can clear its deadline."""
    result=dict(schema='parallel.direct_program_evaluation.v2',identity=bound,status='unknown',
        outcome='unknown',rmse=None,seconds=elapsed,runtime_seconds=elapsed,numerical_runtime_seconds=None,
        reason='original completion unavailable',dispatched_requests=1,verified_physical_starts=0,
        unknown_consumption_requests=1,registry_reuse_requests=0,physical_evaluations=None,
        final_tests_opened=False,registered_query_receipt_qualified=False)
    completion=dict(schema='parallel.clean_completion.v3',attempt=attempt_ref,
        evaluation_identity_sha256=hashlib.sha256(canonical(bound)).hexdigest(),
        program_sha256=bound['program_sha256'],evaluator_sha256=bound['evaluator_sha256'],
        exit_observed=type(code) is int,exit_code=code if type(code) is int and code>=0 else None,
        signal=-code if type(code) is int and code<0 else None,process_pid=pid,
        deadline_seconds=30.,deadline_exceeded=timed_out,runtime_seconds=elapsed,
        outcome='timeout' if timed_out else ('success' if code==0 else 'infrastructure_failure'),output=None)
    metrics=None;error=None
    try:
        if (directory/'fit.json').exists():
            completion['output']=reference(root,directory/'fit.json');metrics=artifact(root,completion['output'])
            for field,source in (('fit_completion','fit_completion'),('model','fitted_model')):
                ref=metrics.get(source)
                if isinstance(ref,dict):completion[field]=reference(root,ref['path'],ref['sha256'])
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:error=type(exc).__name__+': '+str(exc)
    immutable(directory/'clean-completion.json',completion)
    result['clean_completion']=reference(root,directory/'clean-completion.json')
    if timed_out:
        result.update(status='timeout',outcome='timeout',reason='original direct deadline exceeded')
        return result
    if code!=0:
        result.update(status='failed',outcome='infrastructure_failure',reason='original evaluator exit/signal failed')
        return result
    try:
        require(error is None,error or 'invalid output')
        require(isinstance(metrics,dict),'original direct output missing')
        require(metrics.get('task')=='regression' and metrics.get('primary_metric')=='rmse'
                and metrics.get('units')=='cut_normalized_target','conflicting evaluator task/metric/units')
        require(admission.finite(metrics.get('rmse')) and metrics['rmse']>=0,'invalid original RMSE')
        context=dict(bound,**{key:policy[key] for key in ('evaluator','authority_root','authority_witness',
            'authority_instance','runtime_sha256','numerical_contract_sha256')})
        native=artifact(root,completion['fit_completion'])
        row=dict(context=context,program_sha256=bound['program_sha256'],
            evaluation_identity_sha256=completion['evaluation_identity_sha256'],runtime_seconds=elapsed,
            numerical_runtime_seconds=native.get('elapsed_seconds'),clean_completion=result['clean_completion'])
        _,native=admission.clean_completion(root,row,metrics)
        origin=metrics.get('evaluation_origin');current_task=str(directory/'fit.json')
        require(origin in ('execution','reuse'),'original execution origin unavailable')
        require(type(metrics.get('physical_fit_starts')) is int,'invalid physical-start count type')
        if origin=='execution':
            require(native.get('task_id')==current_task and metrics.get('physical_fit_starts')==1,
                    'original physical start is not attributable to this request')
            result['verified_physical_starts']=1
        else:
            require(native.get('task_id')!=current_task and metrics.get('physical_fit_starts')==0,
                    'reused fit conflicts with original task attribution')
            result['registry_reuse_requests']=1
        original=metrics['rmse']*bound['target_scale'];require(math.isfinite(original),'nonfinite original-unit RMSE')
        result.update(status='ok',outcome='success',rmse=original,reason=None,evaluation_origin=origin,
            numerical_runtime_seconds=native['elapsed_seconds'],native_fit_completion=completion['fit_completion'],
            native_receipt_sha256=completion['output']['sha256'],rmse_cut_normalized=metrics['rmse'],
            unknown_consumption_requests=0,physical_evaluations=result['verified_physical_starts'])
    except Exception as exc:
        # Verification errors and missing authority remain unavailable evidence;
        # retaining the original attempt never authorizes an automatic retry.
        result['reason']='unverified original evidence: '+type(exc).__name__+': '+str(exc)
    return result


def evaluate(root,cases,binary,vocabulary,output,cpus,budget,reservation,*,admission_policy=None):
    """Caller owns these GP slot leases; each CPU runs at most one fit."""
    require(0<len(cpus)<=16 and len(cpus)==len(set(cpus)),'invalid direct evaluation slot allocation')
    root=root.resolve();binary=binary.resolve();output=runtime.inside(root,output)
    approved_policy(root,binary,admission_policy)
    require(cases,'direct evaluation requires a nonempty frozen case list')
    output.mkdir(parents=True,exist_ok=True);unique={};mapping=[]
    budget.reserve(reservation+'-verification',core_seconds=60);verification_started=time.monotonic()
    for case in cases:
        bound=identity(root,case['context'],case['program'],binary,vocabulary)
        require(bound['fit_training_sha256'] is not None,
                'compact direct view needs a qualified native numerical-input/query receipt before dispatch')
        key=hashlib.sha256(canonical(bound)).hexdigest();unique.setdefault(key,dict(identity=bound,context=case['context']))
        mapping.append(key)
        require(time.monotonic()-verification_started<=60,'direct identity verification ceiling exceeded')
    immutable(output/'request.json',dict(schema='parallel.direct_evaluation_batch.v2',admission_policy=admission_policy,
        identities=unique,case_identities=mapping,cpus=list(cpus),requested_unique_evaluations=len(unique),
        logical_slots=len(cases),logical_evaluations=len(cases),fit_seconds=30,final_tests_opened=False))
    budget.reserve(reservation,core_seconds=90*len(unique),searches=len(cpus))
    results={};queue=list(unique);active={};children=[]
    try:
        with runtime.owned_ranks(children):
            while queue or active:
                for cpu in cpus:
                    if cpu in active or not queue:continue
                    key=queue.pop(0);case=unique[key];directory=output/key;directory.mkdir()
                    program=directory/'program.dsl';program.write_text(case['identity']['program'])
                    log=(directory/'fit.log').open('xb')
                    def limits():resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
                    require(sha(binary)==case['identity']['evaluator_sha256'],'direct evaluator changed')
                    started=time.monotonic();deadline_ns=time.monotonic_ns()+30_000_000_000
                    immutable(directory/'attempt.json',dict(schema='parallel.direct_attempt.v2',
                        evaluation_identity_sha256=key,program_sha256=case['identity']['program_sha256'],
                        evaluator_sha256=case['identity']['evaluator_sha256'],identity=case['identity'],
                        deadline_monotonic_ns=deadline_ns,deadline_seconds=30.,request_sha256=sha(output/'request.json')))
                    child=runtime.popen(['taskset','-c',str(cpu),str(binary),'--root',str(root),'evaluate-view',
                        '--prepared',str(root/case['context']['prepared']),'--cut',str(case['context']['cut']),
                        '--genome',str(program),'--output',str(directory/'fit.json')],cwd=root,
                        env=runtime.child_env(LD_LIBRARY_PATH='/usr/lib/python3/dist-packages/torch/lib',
                            JOPE_FIT_DEADLINE_MONOTONIC_NS=str(deadline_ns)),
                        preexec_fn=limits,stdout=log,stderr=log)
                    children.append(child);active[cpu]=(key,child,started,log)
                for cpu,(key,child,start,log) in list(active.items()):
                    elapsed=time.monotonic()-start;code=child.poll();timed_out=elapsed>=30
                    if code is None and not timed_out:continue
                    if timed_out and code is None:runtime.stop([child]);code=child.returncode
                    log.close();directory=output/key;bound=unique[key]['identity']
                    result=finish(root,directory,bound,admission_policy,reference(root,directory/'attempt.json'),
                        code,elapsed,timed_out,child.pid)
                    immutable(directory/'complete.json',result);results[key]=dict(path=str((directory/'complete.json').relative_to(root)),
                        sha256=sha(directory/'complete.json'));del active[cpu]
                if active:time.sleep(.05)
        references=[];seen=set()
        for key in mapping:
            row=artifact(root,results[key]);first=key not in seen
            references.append(dict(identity_sha256=key,receipt=results[key],evaluation_reused=key in seen,
                logical_slots=1,logical_evaluations=1,dispatched_requests=int(first),
                verified_physical_starts=row['verified_physical_starts'] if first else 0,
                unknown_consumption_requests=row['unknown_consumption_requests'] if first else 0));seen.add(key)
        rows=[artifact(root,ref) for ref in results.values()]
        unknown=sum(row['unknown_consumption_requests'] for row in rows)
        starts=sum(row['verified_physical_starts'] for row in rows)
        complete=dict(status='complete',request_sha256=sha(output/'request.json'),evaluations=references,
            physical_evaluations=None if unknown else starts,verified_physical_starts=starts,
            unknown_consumption_requests=unknown,dispatched_requests=len(results),logical_slots=len(cases),
            registry_reuse_requests=sum(row['registry_reuse_requests'] for row in rows),
            logical_evaluations=len(cases),final_tests_opened=False,registered_query_receipt_qualified=False)
        immutable(output/'complete.json',complete);budget.complete(reservation,output/'complete.json')
        budget.complete(reservation+'-verification',output/'complete.json')
        return complete
    finally:
        for _,_,_,log in active.values():log.close()
