"""One persistent resumed model; immutable parent, durable new epochs/outbox.

The supervised objective uses measured records only. Generated proposals are
durable work, never predictions substituted for reward. No automatic recovery.
"""
from dataclasses import asdict
from pathlib import Path
import json,os,sqlite3,time
import numpy as np
import torch
from dltrain import model as M
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.program_io import need,create,tick,sha
from jope_flywheel import timing
from neural_native_train import prepare_microbatches,run_microbatches,evaluate_full_validation,publish,OBJECTIVE,gpu_preflight
from neural_native_stream import BatchStream,Waiting
from neural_outbox import Outbox
from neural_decode import BatchedTrie
from continuous_state import restore,normalizer,Recovery,append


def readonly_outbox(path):
    """Parent consumer may advance outcomes; read one snapshot without writes."""
    q=Outbox.__new__(Outbox);q.db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,isolation_level=None);q.db.row_factory=sqlite3.Row
    try:
        q.db.execute('BEGIN');s=q.summary();q.db.execute('COMMIT');return s
    finally:q.db.close()


class Plateau:
    def __init__(self,patience,floor):self.patience=patience;self.floor=floor;self.best=None;self.bad=0;self.reductions=0
    def observe(self,value,opt):
        need(np.isfinite(value),'finite fixed-panel monitor')
        improved=self.best is None or value<self.best
        if improved:self.best=value;self.bad=0
        else:self.bad+=1
        reduced=False
        if self.bad>=self.patience:
            old=opt.param_groups[0]['lr'];new=max(self.floor,old*.5)
            for g in opt.param_groups:g['lr']=new
            self.bad=0;reduced=new<old;self.reductions+=int(reduced)
        return {'improved':improved,'best_since_continuation_loss':self.best,'nonimproving_validations':self.bad,'lr_reduced':reduced,'learning_rate':opt.param_groups[0]['lr'],'reductions':self.reductions,'policy':'continue shuffled epochs; retain best; no guarantee of improvement'}


def bytes_used(root):
    # Count allocated file identities once because recovery/best/epoch use hardlinks.
    seen=set();total=0
    for base,dirs,files in os.walk(root,followlinks=False):
        need(all(not(Path(base)/d).is_symlink()for d in dirs),'owned output symlink')
        for name in files:
            p=Path(base)/name;need(not p.is_symlink(),'owned output symlink');s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);total+=s.st_size
    return total


def run(native,snapshot,config,*,root,campaign_sha,ancestry,preprocessing,deadline,authority_check,adoption_check):
    root=Path(root);root.mkdir(exist_ok=False);n=config['neural'];device=torch.device(n['device'])
    authority_check(snapshot);M.configure_determinism()
    with timing.span('model.restore_checkpoint_and_optimizer'):
        net,opt,origin=restore(config['checkpoint_manifest'],config['checkpoint_state'],expected_parameter_sha256=config['parameter_sha256'],device=device)
    need(origin['step']==ancestry['optimizer_steps']==429,'registered absolute origin429')
    norm=normalizer(preprocessing,native);create(root/'preprocessing.json',preprocessing)
    create(root/'ancestry.json',canonical(ancestry));create(root/'initial-loaded-receipt.json',canonical(snapshot.receipt))
    recovery=Recovery(root);current_path,cp=recovery.save(net,opt,429,campaign_sha);recovery.committed(current_path)
    manifest=canonical(snapshot.batch_manifest);create(root/'initial-batch-manifest.json',manifest)
    planner=BatchStream((root/'batches.sqlite').resolve(),manifest,sha(manifest),seed=n['seed'],batch_size=n['batch_size'],initial_checkpoint_sha256=cp['parameter_sha256']);planner.inherit(ancestry)
    outbox=Outbox((root/'outbox.sqlite').resolve(),model_origin_sha256=cp['parameter_sha256'],max_pending=n['max_pending_emissions'],max_rows=100000,max_payload_bytes=256*1024**2)
    native_train=sorted(c for c,r in native.contexts.items()if r['partition']=='train');native_dev=sorted(c for c,r in native.contexts.items()if r['partition']=='validation')
    train_programs={native.winners[c].program_sha256 for c in native_train};catalog=[p for p in native.proposal_programs if p['program_sha256']in train_programs]
    trie=BatchedTrie(catalog,native.vocab,native.max_seq)
    train_panel=sorted(native_train,key=lambda x:digest({'seed':n['seed'],'panel':x}))[:n['evaluation_panel_contexts']]
    dev_panel=sorted(native_dev,key=lambda x:digest({'seed':n['seed'],'panel':x}))[:n['evaluation_panel_contexts']]
    panel_identity=digest({'native_snapshot':native.identity,'train':train_panel,'dev':dev_panel,'epoch':0,'batch_index':0,'objective':OBJECTIVE})
    panels={role:prepare_microbatches(native,ids,epoch=0,batch_index=0,normalizer=norm,max_padded_tokens=n['max_padded_tokens'])[:2]for role,ids in [('training',train_panel),('validation',dev_panel)]}
    # This qualification uses the original longest native teachers; variants
    # have no invented CE teacher and were independently measured/loaded.
    if device.type=='cuda':
        with timing.span('model.gpu_resume_preflight'):
            create(root/'gpu-preflight.json',canonical(gpu_preflight(native,net,norm,n,deadline)))
        torch.cuda.reset_peak_memory_stats(device)
    plateau=Plateau(n['plateau_patience'],n['learning_rate_floor']);steps=0;checkpoint_sha=cp['parameter_sha256'];current=None
    seen=set(ancestry['exposed_context_ids']);presentations=ancestry['presentations'];newseen=set();devseen=set(native_dev)
    curves=[];hist=[];proof_tail=[];state='running';started=time.monotonic();last_eval=-1;last_full=0;full_validation=None;extended_validation=None
    perf={'input_seconds':0.,'decode_seconds':0.,'optimizer_seconds':0.};used=bytes_used(root.parent);priorq=readonly_outbox(config['parent_outbox']);monitor=None;failure=None
    def progress():
        s=planner.summary();q=outbox.summary();combined=q['unresolved']+priorq['unresolved']
        return {'schema':'jope.flywheel-neural-progress/v1','run_id':n['run_id'],'config_sha256':campaign_sha,'at':time.time(),'state':state,
          'optimizer_steps':429+steps,'new_optimizer_steps':steps,'completed_batches':429+steps,'new_completed_batches':steps,
          'completed_epochs':ancestry['completed_epochs']+s['completed_epochs'],'epoch_or_pass':ancestry['completed_epochs']+max(1,s['epoch']),
          'unique_train_data_runs_seen':len(seen),'total_train_presentations':presentations,'new_run_unique_train_contexts_seen':len(newseen),
          'unique_validation_data_runs_scored':len(devseen),'model_generated_programs':priorq['observations']+q['observations'],
          'novel_programs_executed':priorq['states'].get('published_verified',0)+q['states'].get('published_verified',0),
          'cache_hits':priorq['states'].get('cached_verified',0)+q['states'].get('cached_verified',0),
          'published_results':priorq['states'].get('published_verified',0)+q['states'].get('published_verified',0),
          'failed_programs':priorq['states'].get('failure_custody_verified',0)+q['states'].get('failure_custody_verified',0),
          'unresolved_programs':combined,'outbox':q,'parent_outbox':priorq,'shared_unresolved_cap':n['max_pending_emissions'],
          'loss_history':curves[-1000:],'optimizer_history':hist[-2048:],'full_history_paths':{'loss':'loss-history.jsonl','optimizer':'optimizer-history.jsonl'},
          'comparison_games':0,'winrate_vs_random':None,'winrate_vs_winner':None,'elo':None,'prediction_outcomes_are_rewards':False,
          'performance':{**perf,'optimizer_steps_per_wall_second':steps/max(1e-9,time.monotonic()-started),
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device)if device.type=='cuda'else None,
             'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(device)if device.type=='cuda'else None},
          'latest_full_validation':full_validation,'extended_validation':extended_validation,'fixed_panel_sha256':panel_identity,'plateau':monitor,
          'planner':s,'checkpoint_sha256':checkpoint_sha,'original_teacher_inventory':native.receipt,'active_snapshot':snapshot.receipt,
          'storage':{'observed_unique_inode_bytes':used,'cap_bytes':n['output_cap_bytes'],'recovery_retention':'last2 + best2 + every completed epoch; historical batch proofs retained'},
          'resumed_from':{'config_sha256':config['training_config']['sha256'],'checkpoint_parameter_sha256':origin['parameter_sha256'],'absolute_optimizer_step':429,'planner_head':ancestry['head']},
          'proposal_catalog':{'sha256':trie.identity,'programs':len(trie.programs),'scope':'fixed original training teachers only'},
          'scope':'Same checkpoint/AdamW/RNG and fixed train-only preprocessing. Measured variant q/feas/runtime; no variant self-imitation CE. Pending generated outcomes never enter gradients.'}
    def control():
        value={'schema':'jope.continuous-optimizer-control/v1','checkpoint_parameter_sha256':checkpoint_sha,'checkpoint_state_sha256':cp['state_sha256'],'absolute_optimizer_step':429+steps,'group_learning_rates':[g['lr']for g in opt.param_groups],'plateau':{'best':plateau.best,'bad':plateau.bad,'reductions':plateau.reductions,'patience':plateau.patience,'floor':plateau.floor},'fixed_panel_sha256':panel_identity,'resume_requirement':'Verify and apply this control state together with the physical checkpoint; no automatic resume'}
        publish(root,'optimizer-control.json',value);return value
    def evaluate():
        nonlocal monitor,last_eval
        with timing.span('validation.fixed_native_panel',absolute_optimizer_step=429+steps):metrics={role:run_microbatches(net,*pack,deadline=deadline)for role,pack in panels.items()}
        monitor=plateau.observe(metrics['validation']['loss'],opt)
        row={'batch':429+steps,'new_batch':steps,'epoch':ancestry['completed_epochs']+planner.summary()['epoch'],'unique_data_runs_seen':len(seen),'train_loss':metrics['training']['loss'],'validation_loss':metrics['validation']['loss'],'at':time.time(),'dropout':False,'objective':OBJECTIVE['name'],'fixed_panel_sha256':panel_identity,'metrics':metrics,'plateau':monitor}
        append(root/'loss-history.jsonl',row);curves.append(row);last_eval=steps
        if monitor['improved']:recovery.retain(current_path,'best',f'{429+steps:09d}')
        control()
    try:
        evaluate();publish(root,'progress.json',progress())
        while steps<n['max_updates']:
            try:tick(deadline)
            except TimeoutError:state='global_deadline_reached';break
            # Pure capacity waits create no optimizer reservation and never drop work.
            priorq=readonly_outbox(config['parent_outbox']);q=outbox.summary()
            if steps%32==0:used=bytes_used(root.parent)
            if used>=n['output_cap_bytes']-64*1024**2:state='waiting_for_output_capacity'
            elif q['observations']+2>100000 or q['unresolved']+priorq['unresolved']+2>n['max_pending_emissions']or outbox.db.execute('SELECT coalesce(sum(length(body)),0)FROM emissions').fetchone()[0]+256*1024>256*1024**2:state='waiting_for_mandatory_gp_capacity'
            else:state='running'
            if state!='running':publish(root,'progress.json',progress());time.sleep(min(10,max(0,deadline-time.time())));continue
            authority_check(snapshot)
            current=planner.reserve(checkpoint_sha)
            if isinstance(current,Waiting):
                state='waiting_for_unique_batch';current=None;publish(root,'progress.json',progress());time.sleep(min(10,max(0,deadline-time.time())));continue
            ids=list(current.ordered_data_run_ids);absolute=430+steps;epoch=ancestry['completed_epochs']+current.epoch
            stage=time.monotonic()
            with timing.span('input.training_batch',batch_id=current.batch_id,absolute_optimizer_step=absolute,epoch=epoch):batches,total,selection=prepare_microbatches(snapshot,ids,epoch=epoch-1,batch_index=current.index,normalizer=norm,max_padded_tokens=n['max_padded_tokens'])
            perf['input_seconds']+=time.monotonic()-stage
            dev=sorted(c for c,r in snapshot.contexts.items()if r['partition']=='validation');emit=[ids[steps%len(ids)],dev[steps%len(dev)]]
            seeds=[int(digest({'seed':n['seed'],'step':absolute,'phase':phase,'context':cid})[:15],16)for phase,cid in zip(['training','validation'],emit)]
            stage=time.monotonic()
            with timing.span('model.decode_proposals',absolute_optimizer_step=absolute),timing.CUDAForwards(net)as cuda:
                z=torch.tensor(np.stack([norm(c,snapshot.encoding(c))for c in emit]),device=device);decoded,decode_info=trie.decode_many(net,z,seeds=seeds,temperature=n['decode_temperature'],deadline=deadline)
                decode_info['cuda_timing']=cuda.receipt()
            perf['decode_seconds']+=time.monotonic()-stage;emissions=[]
            for phase,cid,row in zip(['training','validation'],emit,decoded):
                raw=canonical(row);create(root/f'decoded-{absolute:09d}-{phase}.json',raw);ctx=snapshot.contexts[cid]
                emissions.append({'phase':phase,'data_run_id':cid,'context_id':cid,'parent_lineage_sha256':ctx['parent_lineage_sha256'],'input_descriptor':ctx,'raw_program_utf8':row['program']['dsl'],'raw_program_sha256':row['program']['program_sha256'],'decoder_seed':row['decoder_seed'],'decoder_protocol_sha256':row['batch_protocol_sha256'],'decode_artifact_sha256':sha(raw)})
            with timing.span('gp.durable_outbox_ack',absolute_optimizer_step=absolute):ack=outbox.record_batch({'batch_id':current.batch_id,'campaign_sha256':campaign_sha,'checkpoint_sha256':checkpoint_sha,'global_step':absolute,'manifest_sha256':planner.manifest_sha256,'train_data_run_ids':ids,'emitted_train_data_run_ids':[emit[0]],'validation_data_run_ids':[emit[1]]},emissions)
            before=M.parameter_sha256(net);need(before==checkpoint_sha,'actual pre-update checkpoint chain');stage=time.monotonic()
            with timing.span('model.forward_backward_optimizer',absolute_optimizer_step=absolute),timing.CUDAForwards(net)as cuda:
                objective=run_microbatches(net,batches,total,optimizer=opt,deadline=deadline);objective['cuda_forward_timing']=cuda.receipt()
            perf['optimizer_seconds']+=time.monotonic()-stage;after=M.parameter_sha256(net);need(after!=before,'actual parameter update')
            with timing.span('checkpoint.atomic_durable_physical_state',absolute_optimizer_step=absolute):newpath,newcp=recovery.save(net,opt,absolute,campaign_sha)
            proof={'schema':'jope.continuous-neural-batch-proof/v1','config_sha256':campaign_sha,'absolute_optimizer_step':absolute,'local_optimizer_step':steps+1,'batch':asdict(current),'selection':selection,'objective':objective,'decoder':decode_info,'emission_ack':ack,'checkpoint':newcp,'checkpoint_retention':'rolling; retained only if among last2/best2/epoch checkpoints','fresh_generated_rewards_used':False}
            raw=canonical(proof);create(root/f'batch-{absolute:09d}.json',raw)
            with timing.span('checkpoint.planner_commit',absolute_optimizer_step=absolute):planner.commit(current.batch_id,checkpoint_before=before,checkpoint_after=after,optimizer_step=steps+1,ordered_data_run_ids=ids,evidence={'origin':'native_optimizer','event_sha256':sha(raw)})
            current_path=newpath;cp=newcp;checkpoint_sha=after;steps+=1;seen.update(ids);newseen.update(ids);presentations+=len(ids);current=None
            # Once SQL commit succeeds, report the committed step even if a
            # later working-checkpoint cleanup fails. Never relabel it pending.
            recovery.committed(newpath);control()
            history={'batch':absolute,'new_batch':steps,'epoch':epoch,'context_count':len(ids),'train_loss':objective['loss'],'terms':objective['terms'],'support':objective['support'],'dropout':True,'at':time.time()};append(root/'optimizer-history.jsonl',history);hist.append(history);hist=hist[-2048:]
            proof_tail.append({'absolute_step':absolute,'proof_sha256':sha(raw),'parameter_sha256':after});proof_tail=proof_tail[-2:];current=None
            completed=planner.summary()['completed_epochs']
            if steps%n['evaluation_every_batches']==0:evaluate()
            if completed>last_full:
                with timing.span('validation.full_original_support',epoch=epoch):full=evaluate_full_validation(native,net,native_dev,epoch=0,normalizer=norm,config=n,deadline=deadline)
                full['actual_epoch']=epoch;create(root/f'validation-original-epoch-{epoch:06d}.json',canonical(full));full_validation={k:v for k,v in full.items()if k!='program_windows'}
                extra=sorted(set(dev)-set(native_dev))
                if extra:
                    with timing.span('validation.expanded_measured_support',epoch=epoch):extended=evaluate_full_validation(snapshot,net,extra,epoch=0,normalizer=norm,config=n,deadline=deadline)
                    extended['actual_epoch']=epoch;extended['comparable_to_fixed_native_monitor']=False;create(root/f'validation-variant-epoch-{epoch:06d}.json',canonical(extended));extended_validation={k:v for k,v in extended.items()if k!='program_windows'};devseen.update(extra)
                if last_eval!=steps:evaluate()
                recovery.retain(current_path,'epochs',f'{epoch:06d}');last_full=completed
                # The callback verifies a complete immutable candidate first.
                with timing.span('input.epoch_boundary_snapshot_scan',epoch=epoch):replacement=adoption_check(snapshot,root)
                if replacement is not None:
                    newmanifest=canonical(replacement.batch_manifest);planner.adopt_manifest(newmanifest,sha(newmanifest),checkpoint_sha256=checkpoint_sha);snapshot=replacement
                    create(root/f'adopted-snapshot-epoch-{epoch:06d}.json',canonical(snapshot.receipt))
            publish(root,'progress.json',progress())
        else:state='maximum_updates_reached'
    except BaseException as error:
        state='failed';failure=type(error).__name__+': '+str(error)[:4000]
        if current is not None:
            # An optimizer may have completed between an exception and commit.
            # Preserve the physical state; leave intent unresolved for review.
            try:
                from neural_storage import checkpoint
                checkpoint(net,opt,root/'uncertain-physical-state',step=429+steps+1,config_sha256=campaign_sha)
                planner.mark_uncertain(current.batch_id,failure)
            except BaseException as nested:append(root/'closure-errors.jsonl',{'error':str(nested)[:2000]})
        publish(root,'failure.json',{'error':failure,'at':time.time(),'pending_intent':planner.pending(),'automatic_retry':False});raise
    finally:
        summary=planner.summary();verification=planner.verify();priorq=readonly_outbox(config['parent_outbox'])
        publish(root,'progress.json',progress());publish(root,'result.json',{'schema':'jope.continuous-neural-result/v1','state':state,'error':failure,'config_sha256':campaign_sha,'resumed_from':origin,'new_optimizer_steps':steps,'absolute_optimizer_steps':429+steps,'planner':summary,'planner_verification':verification,'outbox':outbox.verify(),'parent_outbox':priorq,'current_parameter_sha256':M.parameter_sha256(net),'latest_checkpoint':cp,'latest_checkpoint_path':str(current_path),'proof_tail':proof_tail,'unique_train_contexts_seen':len(seen),'total_train_presentations':presentations,'fixed_panel_sha256':panel_identity,'plateau':monitor,'preprocessing_sha256':sha(preprocessing),'snapshot_sha256':snapshot.identity,'global_deadline_unix':config['global_deadline_unix'],'optimizer_control':control()});planner.close();outbox.close()
    return {'state':state,'new_optimizer_steps':steps,'absolute_optimizer_steps':429+steps}
