"""Batched sparse-support PolicyLM updates on a lazy full native corpus.

Fresh model version; frozen measured supervision only. Every optimizer update
requires durable generated-program outbox acknowledgement first. No unresolved
proposal is a measured reward. Context-complete microbatches preserve each term's
macro-batch denominator and never split pairwise/list comparisons.
"""
from pathlib import Path
from dataclasses import asdict
import math,time,os
import numpy as np
import torch
from dltrain import model as M
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.program_io import need,create,tick,sha
from neural_native_batch import collate,objective,OBJECTIVE
from neural_native_stream import BatchStream,Waiting
from neural_outbox import Outbox
from neural_decode import BatchedTrie
from neural_storage import checkpoint
from jope_flywheel import timing


def fit_preprocessing(snapshot):
 ids=sorted(c for c,r in snapshot.contexts.items()if r['partition']=='train');need(ids,'training contexts for preprocessing')
 mean=np.zeros(2051,dtype=np.float64);m2=mean.copy()
 for n,cid in enumerate(ids,1):
  x=np.asarray(snapshot.encoding(cid),dtype=np.float64);need(x.shape==(2051,)and np.isfinite(x).all(),'finite KumoS input')
  delta=x-mean;mean+=delta/n;m2+=delta*(x-mean)
 scale=np.sqrt(m2/len(ids));scale=np.where(scale>1e-12,scale,1.)
 receipt={'schema':'jope.native-kumos-preprocessing/v1','snapshot_sha256':snapshot.identity,'fit_context_ids':ids,'mean':mean.tolist(),'scale':scale.tolist(),'clip':[-10.,10.],'method':'float64 deterministic sorted-context Welford population variance','validation_used_for_fit':False}
 def transform(cid,x):
  need(cid in snapshot.contexts,'known context');return np.clip((np.asarray(x,dtype=np.float64)-mean)/scale,-10.,10.).astype(np.float32)
 return transform,receipt


def denominators(batch):
 out={k:0 for k in OBJECTIVE['coefficients']}
 for ci in range(len(batch.context_ids)):
  same=batch.seq_ctx==ci
  for k,mask in [('ce',batch.ce),('q',batch.quality),('feas',batch.feas),('rt',batch.runtime)]:out[k]+=int(bool((same&mask).any()))
  mask=same&batch.pairwise;out['list']+=int(mask.sum()>=3);grids=batch.q_grid[mask];values=batch.q_opt[mask];out['bt']+=int(bool(((grids[:,None]>grids[None,:])&((values[:,None]-values[None,:]).abs()>0)).any()))
 return out


def prepare_microbatches(snapshot,ids,*,epoch,batch_index,normalizer,max_padded_tokens):
 need(type(max_padded_tokens)is int and max_padded_tokens>=snapshot.max_seq,'bounded padded-token budget')
 native=snapshot.context_loader.get(ids,epoch=epoch,batch_index=batch_index);groups=[];group=[];seqs=0;width=0
 for cid in ids:
  rows=native.records[cid];need(rows,'context has no supported programs');w=max(len(r.tokens)for r in rows)
  need(len(rows)*w<=max_padded_tokens,'one full-context program window exceeds tensor budget; reduce registered window, never truncate')
  if group and (seqs+len(rows))*max(width,w)>max_padded_tokens:groups.append(group);group=[];seqs=0;width=0
  group.append(cid);seqs+=len(rows);width=max(width,w)
 if group:groups.append(group)
 batches=[collate(native,g,legal_mask_fn=snapshot.legal_mask_fn,max_seq=snapshot.max_seq,normalizer=normalizer)for g in groups]
 total={k:sum(denominators(b)[k]for b in batches)for k in OBJECTIVE['coefficients']}
 need(any(total[k]>0 and OBJECTIVE['coefficients'][k]>0 for k in total),'at least one actually supported measured objective term required')
 return batches,total,{'selection':native.selection_receipt,'support':native.support,'microbatches':[b.receipt for b in batches],'global_term_denominators':total}


def run_microbatches(net,batches,total,*,optimizer=None,deadline):
 previous=net.training;training=optimizer is not None;net.train(training);device=next(net.parameters()).device
 totals={k:0. for k in total};support={};forward_seconds=0.;backward_seconds=0.
 if training:optimizer.zero_grad(set_to_none=True)
 try:
  for cpu in batches:
   tick(deadline);b=cpu.to(device);start=time.monotonic()
   with torch.set_grad_enabled(training):
    with timing.span('model.forward_objective'):
     out=net(b.tokens,b.lengths,b.ctx_z,b.seq_ctx);_,report,parts=objective(out,b)
    if device.type=='cuda':torch.cuda.synchronize(device)
    forward_seconds+=time.monotonic()-start
    loss=sum(OBJECTIVE['coefficients'][k]*n/total[k] for k,(n,d)in parts.items()if total[k])
    need(torch.isfinite(loss).item(),'finite global-normalized microbatch loss')
    if training:
     start=time.monotonic()
     with timing.span('model.backward'):loss.backward()
     if device.type=='cuda':torch.cuda.synchronize(device)
     backward_seconds+=time.monotonic()-start
   for k,v in report['numerators'].items():totals[k]+=v
   for k,v in report['support'].items():support[k]=support.get(k,0)+v
   del out,b,parts,loss
  terms={k:totals[k]/total[k]if total[k]else 0. for k in total};result={'loss':sum(OBJECTIVE['coefficients'][k]*v for k,v in terms.items()),'terms':terms,'numerators':totals,'denominators':total,'support':support,'dropout':training,'forward_seconds':forward_seconds,'backward_seconds':backward_seconds}
  if training:
   need(all(p.grad is None or torch.isfinite(p.grad).all().item()for p in net.parameters()),'finite actual optimizer gradients')
   norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);need(torch.isfinite(norm).item(),'finite clipping norm');tick(deadline)
   with timing.span('model.optimizer_step'):
    optimizer.step()
    if device.type=='cuda':torch.cuda.synchronize(device)
   result['gradient_norm_before_clip']=float(norm)
  return result
 finally:net.train(previous)


def evaluate_full_validation(snapshot,net,ids,*,epoch,normalizer,config,deadline):
 totals={k:0. for k in OBJECTIVE['coefficients']};denoms={k:0 for k in totals};membership=[];support={}
 for start in range(0,len(ids),config['batch_size']):
  selected=ids[start:start+config['batch_size']];batches,den,receipt=prepare_microbatches(snapshot,selected,epoch=epoch,batch_index=start,normalizer=normalizer,max_padded_tokens=config['max_padded_tokens'])
  r=run_microbatches(net,batches,den,deadline=deadline)
  for k in totals:totals[k]+=r['numerators'][k];denoms[k]+=den[k]
  for k,v in r['support'].items():support[k]=support.get(k,0)+v
  membership.append(receipt['selection'])
 terms={k:totals[k]/denoms[k]if denoms[k]else 0. for k in totals}
 return {'schema':'jope.native-full-validation/v1','epoch':epoch+1,'context_ids':ids,'context_count':len(ids),'program_windows':membership,'loss':sum(OBJECTIVE['coefficients'][k]*terms[k]for k in terms),'terms':terms,'numerators':totals,'denominators':denoms,'support':support,'dropout':False,'new_program_rewards_used':False}


def publish(root,name,obj):
 root=Path(root);tmp=root/(name+'.pending');create(tmp,canonical(obj));os.replace(tmp,root/name)
 fd=os.open(root,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)


def gpu_preflight(snapshot,net,normalizer,config,deadline):
 """Actual longest training inputs, backward only; restore every RNG, no step."""
 device=next(net.parameters()).device;need(device.type=='cuda','actual CUDA preflight')
 ids=sorted((c for c,r in snapshot.contexts.items()if r['partition']=='train'),key=lambda c:(-len(snapshot.winners[c].tokens),c))[:config['batch_size']]
 start=time.monotonic();batches,total,selection=prepare_microbatches(snapshot,ids,epoch=0,batch_index=0,normalizer=normalizer,max_padded_tokens=config['max_padded_tokens']);loaded=time.monotonic()-start
 before=M.parameter_sha256(net);cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();mode=net.training;net.train();net.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(device);start=time.monotonic()
 try:
  for cpu in batches:
   tick(deadline);b=cpu.to(device);out=net(b.tokens,b.lengths,b.ctx_z,b.seq_ctx);_,report,parts=objective(out,b)
   loss=sum(OBJECTIVE['coefficients'][k]*n/total[k]for k,(n,d)in parts.items()if total[k]);need(torch.isfinite(loss).item(),'preflight finite measured teacher loss');loss.backward();del b,out,parts,loss
  torch.cuda.synchronize(device);tick(deadline);elapsed=time.monotonic()-start;need(all(p.grad is None or torch.isfinite(p.grad).all().item()for p in net.parameters()),'preflight finite backward')
  need(M.parameter_sha256(net)==before,'preflight must not update parameters')
  return {'schema':'jope.native-neural-gpu-preflight/v1','state':'passed','optimizer_updates':0,'parameter_sha256':before,'contexts':ids,'selection':selection,'context_batch_size':len(ids),'microbatches':len(batches),'input_seconds':loaded,'forward_backward_seconds':elapsed,'contexts_per_forward_backward_second':len(ids)/elapsed,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'larger_batches_benchmarked':False}
 finally:net.zero_grad(set_to_none=True);net.train(mode);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)


def run(snapshot,config,*,output,deadline,authority_check,campaign_sha256=None):
 """Caller pins code/config, enforces cgroup/GPU/deadline and current authority."""
 root=Path(output);root.mkdir(exist_ok=False);cfgsha=campaign_sha256 or digest(config)
 need(config['max_seq']==snapshot.max_seq and config['initialization']=='fresh'and config['input_dimension']==2051,'fresh exact full-corpus architecture')
 need(config['objective']==OBJECTIVE and config['max_updates']>=1 and config['max_epochs']>=1,'registered objective and run bounds')
 need(type(config['authority_check_every_batches'])is int and 1<=config['authority_check_every_batches']<=16,'bounded fresh-role cadence')
 authority_check();M.configure_determinism();torch.manual_seed(config['seed']);device=torch.device(config['device'])
 net=M.PolicyLM(M.PolicyConfig(282,2051,config['d_model'],config['layers'],config['heads'],config['d_ff'],snapshot.max_seq,dropout=config['dropout'])).to(device)
 need(config.get('expected_initial_parameter_sha256',M.parameter_sha256(net))==M.parameter_sha256(net),'registered fresh model initialization')
 opt=torch.optim.AdamW(M.param_groups(net,.01),lr=config['learning_rate'])
 norm,preprocess=fit_preprocessing(snapshot);create(root/'preprocessing.json',canonical(preprocess));create(root/'config.json',canonical(config));create(root/'loaded-receipt.json',canonical(snapshot.receipt))
 manifest=canonical(snapshot.batch_manifest);create(root/'batch-manifest.json',manifest)
 (root/'checkpoints').mkdir();initial=checkpoint(net,opt,root/'checkpoints'/'000000',step=0,config_sha256=cfgsha)
 if device.type=='cuda':
  preflight=gpu_preflight(snapshot,net,norm,config,deadline);create(root/'gpu-preflight.json',canonical(preflight));torch.cuda.reset_peak_memory_stats(device)
 planner=BatchStream((root/'batches.sqlite').resolve(),manifest,sha(manifest),seed=config['seed'],batch_size=config['batch_size'],initial_checkpoint_sha256=initial['parameter_sha256'])
 outbox=Outbox((root/'outbox.sqlite').resolve(),model_origin_sha256=initial['parameter_sha256'],max_pending=config['max_pending_emissions'])
 train_programs={snapshot.winners[c].program_sha256 for c,r in snapshot.contexts.items()if r['partition']=='train'}
 catalog=[p for p in snapshot.proposal_programs if p['program_sha256']in train_programs];trie=BatchedTrie(catalog,snapshot.vocab,snapshot.max_seq)
 train=sorted(c for c,r in snapshot.contexts.items()if r['partition']=='train');dev=sorted(c for c,r in snapshot.contexts.items()if r['partition']=='validation');need(train and dev,'nonempty train and validation support')
 train_panel=sorted(train,key=lambda x:digest({'seed':config['seed'],'panel':x}))[:config['evaluation_panel_contexts']];dev_panel=sorted(dev,key=lambda x:digest({'seed':config['seed'],'panel':x}))[:config['evaluation_panel_contexts']]
 panels={}
 for role,ids in [('training',train_panel),('validation',dev_panel)]:panels[role]=prepare_microbatches(snapshot,ids,epoch=0,batch_index=0,normalizer=norm,max_padded_tokens=config['max_padded_tokens'])[:2]
 curves=[];proofs=[];optimizer_history=[];checkpoint_sha=initial['parameter_sha256'];state='running';current=None;seen=set();devseen=set();presentations=0;last_summary=None;full_validation=None;started=time.monotonic();perf={'input_seconds':0.,'decode_seconds':0.,'optimizer_seconds':0.}
 def progress():
  summary=planner.summary();q=outbox.summary();return {'schema':'jope.flywheel-neural-progress/v1','run_id':config['run_id'],'config_sha256':cfgsha,'at':time.time(),'state':state,'optimizer_steps':len(proofs),'completed_batches':len(proofs),'completed_epochs':summary['completed_epochs'],'epoch_or_pass':summary['epoch'],'unique_train_data_runs_seen':len(seen),'total_train_presentations':presentations,'unique_validation_data_runs_scored':len(devseen),'model_generated_programs':q['observations'],'novel_programs_executed':q['states'].get('published_verified',0),'cache_hits':q['states'].get('cached_verified',0),'published_results':q['states'].get('published_verified',0),'failed_programs':q['states'].get('failure_custody_verified',0),'unresolved_programs':q['unresolved'],'outbox':q,'loss_history':curves,'optimizer_history':optimizer_history,'comparison_games':0,'winrate_vs_random':None,'winrate_vs_winner':None,'elo':None,'prediction_outcomes_are_rewards':False,'performance':{**perf,'optimizer_steps_per_wall_second':len(proofs)/max(1e-9,time.monotonic()-started),'optimizer_steps_per_compute_second':len(proofs)/perf['optimizer_seconds']if perf['optimizer_seconds']else None,'proposals_per_decode_second':q['observations']/perf['decode_seconds']if perf['decode_seconds']else None,'training_contexts_per_input_second':presentations/perf['input_seconds']if perf['input_seconds']else None,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device)if device.type=='cuda'else None,'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(device)if device.type=='cuda'else None},'latest_full_validation':full_validation,'native_teacher_inventory':snapshot.receipt,'planner':summary,'proposal_catalog':{'scope':'training-context teachers only; no development-only teacher candidates','all_teacher_programs':len(snapshot.proposal_programs),'programs':len(trie.programs),'nodes':len(trie.nodes),'sha256':trie.identity},'scope':'Fresh full-corpus sparse-support model; frozen measured teachers. Durable generated proposals are not scores.'}
 try:
  for step in range(config['max_updates']+1):
   tick(deadline)
   if step%config['evaluation_every_batches']==0 or step==config['max_updates'] or planner.summary()['completed_epochs']>=config['max_epochs']:
    metrics={role:run_microbatches(net,*pack,deadline=deadline)for role,pack in panels.items()};devseen.update(dev_panel)
    curves.append({'batch':len(proofs),'epoch':planner.summary()['epoch'],'unique_data_runs_seen':len(seen),'train_loss':metrics['training']['loss'],'validation_loss':metrics['validation']['loss'],'at':time.time(),'dropout':False,'objective':OBJECTIVE['name'],'fixed_panel':True,'panel_ids':{'training':train_panel,'validation':dev_panel},'metrics':metrics});publish(root,'progress.json',progress())
   if step==config['max_updates']or planner.summary()['completed_epochs']>=config['max_epochs']:state='completed';break
   if outbox.summary()['unresolved']+2>config['max_pending_emissions']:state='waiting_for_mandatory_gp_capacity';break
   if step%config['authority_check_every_batches']==0:authority_check()
   current=planner.reserve(checkpoint_sha)
   if isinstance(current,Waiting):state='waiting_for_unique_batch';current=None;break
   ids=list(current.ordered_data_run_ids);stage_start=time.monotonic();batches,total,selection=prepare_microbatches(snapshot,ids,epoch=current.epoch-1,batch_index=current.index,normalizer=norm,max_padded_tokens=config['max_padded_tokens']);perf['input_seconds']+=time.monotonic()-stage_start
   emit_train=ids[step%len(ids)];emit_dev=dev[step%len(dev)];emitted=[emit_train,emit_dev];z=torch.tensor(np.stack([norm(c,snapshot.encoding(c))for c in emitted]),device=device)
   seeds=[int(digest({'seed':config['seed'],'step':step+1,'phase':phase,'context':cid})[:15],16)for phase,cid in zip(['training','validation'],emitted)]
   stage_start=time.monotonic();decoded,decode_info=trie.decode_many(net,z,seeds=seeds,temperature=config['decode_temperature'],deadline=deadline);perf['decode_seconds']+=time.monotonic()-stage_start;emissions=[]
   for phase,cid,row in zip(['training','validation'],emitted,decoded):
    raw=canonical(row);name=f'decoded-{step+1:06d}-{phase}.json';create(root/name,raw);context=snapshot.contexts[cid]
    emissions.append({'phase':phase,'data_run_id':cid,'context_id':cid,'parent_lineage_sha256':context['parent_lineage_sha256'],'input_descriptor':context,'raw_program_utf8':row['program']['dsl'],'raw_program_sha256':row['program']['program_sha256'],'decoder_seed':row['decoder_seed'],'decoder_protocol_sha256':row['batch_protocol_sha256'],'decode_artifact_sha256':sha(raw)})
   ack=outbox.record_batch({'batch_id':current.batch_id,'campaign_sha256':cfgsha,'checkpoint_sha256':checkpoint_sha,'global_step':step+1,'manifest_sha256':sha(manifest),'train_data_run_ids':ids,'emitted_train_data_run_ids':[emit_train],'validation_data_run_ids':[emit_dev]},emissions)
   before=M.parameter_sha256(net);need(before==checkpoint_sha,'physical pre-update checkpoint chain')
   stage_start=time.monotonic();result=run_microbatches(net,batches,total,optimizer=opt,deadline=deadline);perf['optimizer_seconds']+=time.monotonic()-stage_start;after=M.parameter_sha256(net);need(after!=before,'real parameter change')
   cp=checkpoint(net,opt,root/'checkpoints'/f'{step+1:06d}',step=step+1,config_sha256=cfgsha)
   proof={'schema':'jope.native-neural-batch-proof/v1','config_sha256':cfgsha,'batch':asdict(current),'selection':selection,'objective':result,'decoder':decode_info,'emission_ack':ack,'checkpoint':cp,'fresh_generated_rewards_used':False};raw=canonical(proof);create(root/f'batch-{step+1:06d}.json',raw)
   planner.commit(current.batch_id,checkpoint_before=before,checkpoint_after=after,optimizer_step=step+1,ordered_data_run_ids=ids,evidence={'origin':'native_optimizer','event_sha256':sha(raw)})
   optimizer_history.append({'batch':step+1,'epoch':current.epoch,'train_loss':result['loss'],'dropout':True,'context_count':len(ids),'at':time.time(),'support':result['support']});proofs.append({'step':step+1,'proof_sha256':sha(raw),'checkpoint':cp});checkpoint_sha=after;seen.update(ids);presentations+=len(ids);current=None;publish(root,'progress.json',progress())
   completed=planner.summary()['completed_epochs']
   if completed and not (root/f'validation-epoch-{completed:04d}.json').exists():
    full=evaluate_full_validation(snapshot,net,dev,epoch=completed-1,normalizer=norm,config=config,deadline=deadline);full_validation={k:v for k,v in full.items()if k!='program_windows'};create(root/f'validation-epoch-{completed:04d}.json',canonical(full));devseen.update(dev);publish(root,'progress.json',progress())
 except BaseException as error:
  state='failed';publish(root,'failure.json',{'error':type(error).__name__+': '+str(error),'at':time.time(),'optimizer_steps_committed':len(proofs),'pending_intent':planner.pending(),'automatic_retry':False});raise
 finally:
  last_summary=planner.verify();publish(root,'progress.json',progress());publish(root,'result.json',{'schema':'jope.native-neural-result/v1','state':state,'config_sha256':cfgsha,'initial_checkpoint':initial,'proofs':proofs,'planner':planner.summary(),'planner_verification':last_summary,'outbox':outbox.verify(),'loss_history':curves,'preprocessing_sha256':digest(preprocess),'current_parameter_sha256':M.parameter_sha256(net)});planner.close();outbox.close()
 return {'state':state,'optimizer_steps':len(proofs),'output':str(root)}
