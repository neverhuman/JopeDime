"""Actual bounded PolicyLM loop: reserve -> decode -> custody -> update -> commit.

Novel measured outcomes are benchmarks only in this frozen run. They may become
future training labels only through a separately versioned snapshot admission.
Every optimizer batch has model-decoded programs, exact cache/evaluation closure,
actual changed model tensors, a physical checkpoint and a durable batch proof.
"""
from dataclasses import asdict
import json,math,os,random,time
from pathlib import Path
import torch
from dltrain import model as M, dsl_tokenizer as T
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.program_io import need,create,bound,sha,tick,remaining
from jope_flywheel.batch_stream import BatchStream,Waiting
from jope_flywheel.s3_library import BUCKET,PREFIX
from neural_core import ProposalTrie,context_batch,to_device,evaluation,optimizer_update,checkpoint,normalize_contexts


def publish_progress(root,obj):
    root=Path(root);tmp=root/'progress.pending';create(tmp,canonical(obj));os.replace(tmp,root/'progress.json')
    fd=os.open(root,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class MandatoryResolver:
    """Only a completed, independently read-back S3 result returns measured data."""
    def __init__(self,queue,observer,worker,inputs,store,deadline,bindings):
        self.queue,self.observer,self.worker,self.inputs,self.store,self.deadline=queue,observer,worker,inputs,store,deadline
        self.rows=[];self.bindings=bindings
    def resolve(self,row,decoded,checkpoint_sha256,step):
        variant=row['data_run']['variant_identity'];phase='training'if row['partition']=='train'else'validation'
        cid=row['data_run']['context_id'];binding=self.bindings[cid]
        # Use the pinned preflight descriptor to capture the emission first.
        # ProgramObserver then performs the fresh role/input check inside its
        # durable observed/error boundary, so role drift cannot erase attempts.
        req={'variant_identity':variant,'context':binding['context'],'descriptor_sha256':binding['descriptor_sha256'],
            'dsl':decoded['program']['dsl'],'phase':phase,'checkpoint_sha256':checkpoint_sha256,
            'step':step,'producer':'PolicyLM-autoregressive-finite-trie-v1'}
        result={'phase':phase,'data_run_id':row['data_run_id'],'context_id':cid,'variant_identity':variant,
                'program_sha256':decoded['program']['program_sha256'],'state':'observing','actual_execution':False,'published':False,'cache_hit':False}
        self.rows.append(result)
        try:
            first=self.observer.observe(req);result['observation']=first;result['evaluation_key']=first['evaluation_key']
            if first['state']=='ready':result['cache_hit']=True;last=first
            else:
                need(first['state']=='pending','only pending work or verified ready')
                ev=self.worker.once();result['evaluation_work']=ev
                need(ev['state']=='completed'and ev['stage']=='evaluate'and ev['job']=='gp-evaluate-'+first['evaluation_key'],'exact evaluation must complete')
                er=json.loads(bound(ev['receipt']));result['actual_execution']=er['state']=='executed'
                pub=self.worker.once();result['publication_work']=pub
                need(pub['state']=='completed'and pub['stage']=='publish'and pub['job']=='gp-publish-'+first['evaluation_key'],'exact publication must complete')
                pr=json.loads(bound(pub['receipt']));result['published']=pr['state']=='published'
                last=self.observer.observe(req);result['readback']=last
                need(last['state']=='ready','exact versioned S3 readback required before benchmark use')
            tick(self.deadline);entry=self.store.lookup(first['evaluation_key'])
            need(entry is not None,'authoritative result must remain readable')
            ref=self.store.client.head_object(Bucket=BUCKET,Key=PREFIX+first['evaluation_key']+'/record.json')
            record={'bucket':BUCKET,'key':PREFIX+first['evaluation_key']+'/record.json','version_id':ref.get('VersionId'),
                    'sha256':ref.get('Metadata',{}).get('sha256'),'bytes':ref['ContentLength']}
            body,_=self.store.get_version(record);need(json.loads(body)==entry,'exact wrapper reference matches verified result')
            result.update(state='ready',measurement=last['measurement'],record=record)
            return result
        except BaseException as error:
            result.update(state='failed',error=type(error).__name__+': '+str(error)[:1000],numeric_score=None)
            raise
    def counts(self):
        return {'model_generated_programs':len(self.rows),'novel_programs_executed':sum(r['actual_execution']for r in self.rows),
                'cache_hits':sum(r['cache_hit']for r in self.rows),'published_results':sum(r['published']for r in self.rows),
                'failed_programs':sum(r['state']=='failed'for r in self.rows)}


def measured_games(result,pool,seed):
    """Actual same-context geometric RMSE comparisons, no model predictions."""
    need(result['state']=='ready','only verified ready results have games')
    cid=result['context_id'];candidates=sorted(pool[cid],key=lambda x:x['program_sha256'])
    need(candidates,'frozen completed fixed-panel comparison pool')
    def metric(row):
        values=row['measurement']['inner_fold_rmse'];need(len(values)==2 and all(math.isfinite(v)and v>=0 for v in values),'two actual RMSEs')
        # log form avoids product overflow; zeros are represented exactly as -inf.
        return -math.inf if min(values)==0 else sum(math.log(v)for v in values)/2
    random_pick=random.Random(seed).choice(candidates);best=min(candidates,key=lambda x:(metric(x),x['program_sha256']))
    own=metric(result);games=[]
    for kind,other in [('random_fixed_panel',random_pick),('best_fixed_panel',best)]:
        theirs=metric(other);win=0.5 if own==theirs else float(own<theirs)
        games.append({'context_id':cid,'data_run_id':result['data_run_id'],'phase':result['phase'],'opponent':kind,
             'program_sha256':result['program_sha256'],'opponent_program_sha256':other['program_sha256'],
             'outcome':win,'metric':'geometric mean of two measured inner-fold RMSEs; lower is better',
             'rmse':result['measurement']['inner_fold_rmse'],'opponent_rmse':other['measurement']['inner_fold_rmse'],
             'result_ref':result['record'],'opponent_ref':other['record'],'comparison_seed':seed})
    return games


def game_summary(games):
    answer={}
    for opponent in ('random_fixed_panel','best_fixed_panel'):
        phases={}
        for phase in ('training','validation'):
            rows=[r for r in games if r['opponent']==opponent and r['phase']==phase]
            rating=1500.
            for r in rows:rating+=16.*(r['outcome']-1./(1.+10.**((1500.-rating)/400.)))
            phases[phase]={'games':len(rows),'contexts':len({r['context_id']for r in rows}),
                'wins':sum(r['outcome']==1 for r in rows),'ties':sum(r['outcome']==.5 for r in rows),'losses':sum(r['outcome']==0 for r in rows),
                'score_rate':None if not rows else sum(r['outcome']for r in rows)/len(rows),
                'elo':None if not rows else rating,'elo_definition':'Policy starts1500; fixed opponent1500; K16; chronological games, qualification-only',
                'independent_test_claim':False}
        answer[opponent]=phases
    return answer


def continue_training(config,loaded,batch_manifest,pool,resolver,root,deadline,config_sha,*,resume,progress_extra=None,fresh_admission=None):
    """CPU/GPU callers must preflight their resource/source/snapshot envelope first."""
    root=Path(root);need(root.is_dir(),'fresh owned output root must exist')
    need(loaded.receipt['state']=='loaded'and loaded.receipt['ctx_dim']==2051,'actual KumoS-only admitted snapshot')
    need(config['max_batches']<=32 and config['batch_size']<=8,'bounded qualification')
    from neural_resume import restore,apply_frozen_normalization,require_committed_cursor
    need(resume['prior_result']['state']=='completed_bounded_qualification','only completed known prior session may continue')
    need(resume['optimizer_step']==resume['prior_result']['optimizer_steps']==len(resume['prior_result']['proofs']),'exact prior committed step count')
    data,application=apply_frozen_normalization(loaded.data,resume['preprocessing'])
    preprocessing=resume['preprocessing'];create(root/'preprocessing.json',canonical(preprocessing))
    create(root/'normalization-application.json',canonical(application))
    manifest_body=canonical(batch_manifest);create(root/'batch-manifest.json',manifest_body)
    seed=config['seed'];run_seed=seed.to_bytes(32,'big')
    device=torch.device(config['device']);M.configure_determinism(config['torch_threads'])
    net,opt,restored=restore(resume['checkpoint_path'],manifest_sha256=resume['checkpoint_manifest_sha256'],
        config_sha256=resume['checkpoint_config_sha256'],parameter_sha256=resume['parameter_sha256'],
        optimizer_step=resume['optimizer_step'],model_config=config['model'],device=device)
    need(net.cfg.ctx_dim==2051 and net.cfg.vocab_size==data.vocab_size,'exact model input and vocab')
    need(opt.param_groups[0]['lr']==config['learning_rate'],'unchanged optimizer learning rate')
    vocab=T.Vocab.from_names({});trie=ProposalTrie(config['proposal_programs'],vocab,net.cfg.max_seq)
    planner=BatchStream(Path(resume['batch_stream_path']),manifest_body,sha(manifest_body),seed=seed,batch_size=config['batch_size'],initial_checkpoint_sha256=resume['origin_parameter_sha256'])
    cursor=require_committed_cursor(planner,parameter_sha256=resume['parameter_sha256'],optimizer_step=resume['optimizer_step'])
    create(root/'resume-proof.json',canonical({'restore':restored,'cursor':cursor,'prior_result_sha256':digest(resume['prior_result']),
        'origin_parameter_sha256':resume['origin_parameter_sha256'],'reset_optimizer':False,'reset_rng':False}))
    train_ids=sorted(data.fit.pi);dev_ids=sorted(c for c,r in data.corpus.contexts.items()if r['split_role']=='dev'and data.corpus.records[c])
    need(train_ids and dev_ids,'comparable train and development support required')
    # Fixed members and targets for every evaluation point (dropout off).
    train_eval,_=context_batch(data,train_ids,step=0,run_seed=run_seed,training=True)
    dev_eval,_=context_batch(data,dev_ids,step=0,run_seed=run_seed,training=False)
    train_eval=to_device(train_eval,device);dev_eval=to_device(dev_eval,device)
    dev_rows=[r for r in batch_manifest['rows']if r['partition']=='validation'and r['status']=='eligible']
    need({r['payload']['context_sha256']for r in dev_rows}==set(dev_ids),'exact dev manifest membership')
    prior=resume['prior_result'];steps=resume['optimizer_step'];session_start_step=steps
    seen={r[0]for r in planner.db.execute('SELECT DISTINCT run FROM members WHERE batch IS NOT NULL')}
    presentations=planner.db.execute('SELECT count(*)FROM members WHERE batch IS NOT NULL').fetchone()[0]
    games=list(prior['games']);history=list(prior['loss_history']);proofs=list(prior['proofs'])
    need(not resolver.rows,'new session resolver starts empty')
    resolver.rows.extend(prior['resolver_outcomes'])
    dev_contexts={r['context_id']for r in games if r['phase']=='validation'}
    dev_seen={r['data_run_id']for r in batch_manifest['rows']if r['payload']['context_sha256']in dev_contexts}
    epoch=planner.summary()['epoch'];state='restored';error=None;planner_summary=planner.summary();planner_closed=False
    def progress():
        nonlocal planner_summary
        if not planner_closed:planner_summary=planner.summary()
        summary=game_summary(games)
        random_val=summary['random_fixed_panel']['validation'];best_val=summary['best_fixed_panel']['validation']
        publish_progress(root,{'schema':'jope.flywheel-neural-progress/v1','run_id':config['run_id'],'config_sha256':config_sha,
            'at':time.time(),'state':state,'optimizer_steps':steps,'completed_batches':len(proofs),'epoch_or_pass':epoch,
            'completed_epochs':planner_summary['completed_epochs'], 'batch_stream':planner_summary,
            'session_start_optimizer_step':session_start_step,'session_optimizer_steps':steps-session_start_step,
            'origin_parameter_sha256':resume['origin_parameter_sha256'],'resumed':True,
            'unique_train_data_runs_seen':len(seen),'total_train_presentations':presentations,
            'unique_validation_data_runs_scored':len(dev_seen),**resolver.counts(),'loss_history':history,
            'resolver_outcomes':[{k:r[k]for k in('variant_identity','program_sha256','phase','state','measurement',
                'actual_execution','published','cache_hit','error')if k in r}for r in resolver.rows[-12:]],
            'winrate_vs_random':random_val['score_rate'],'winrate_vs_winner':best_val['score_rate'],
            'elo':best_val['elo'],'comparison_games':random_val['games']+best_val['games'],
            'comparison_summary':summary,'comparison_scalar_scope':'validation only; ties count half; Elo versus best fixed panel',
            'comparators':'uniform random and best of preregistered completed fixed panel; not global HWM',
            'model_parameters':M.count_parameters(net),'checkpoint_sha256':M.parameter_sha256(net),
            'training_data':{'snapshot_sha256':loaded.receipt['snapshot_sha256'],'ctx_dim':2051,'encoder':'KumoS',
                'train_data_sha256':data.content_sha256,'eligible_train_contexts':len(train_ids),'dev_contexts':len(dev_ids),
                'catalog_rows':len(batch_manifest['rows']),'plans':loaded.receipt['counts']},
            'generation_scope':'finite preregistered native-valid autoregressive trie','online_reward_update':False,
            'new_measurements_admission':'next separately reviewed frozen snapshot only','error':error,
            **(progress_extra()if progress_extra else {})})
    def readout(batch_index):
        tr=evaluation(net,train_eval);va=evaluation(net,dev_eval)
        need(all(x['support']['teachers']>0 and x['support']['pairs']>0 for x in (tr,va)),
             'waiting for nonzero native CE teachers and ranking pairs in both training and development')
        history.append({'batch':batch_index,'epoch':epoch,'unique_data_runs_seen':len(seen),'train_loss':tr['loss'],
            'validation_loss':va['loss'],'at':time.time(),'train':tr,'validation':va,'fixed_support':True,
            'checkpoint_sha256':M.parameter_sha256(net)})
    try:
        # Fixed members are identical to the first snapshot's evaluation definition.
        before_history=len(history);readout(steps);initial_readout=history.pop()
        need(initial_readout['train_loss']==history[-1]['train_loss']and initial_readout['validation_loss']==history[-1]['validation_loss'],
             'restored model/frozen support loss must equal previous terminal point')
        create(root/'resume-readout.json',canonical(initial_readout));progress()
        for session_index in range(config['max_batches']):
            index=session_start_step+session_index
            if remaining(deadline)<config['minimum_batch_seconds']:state='partial_deadline';break
            tick(deadline);before=M.parameter_sha256(net);batch=planner.reserve(before)
            if isinstance(batch,Waiting):state='waiting_distinct_membership';break
            epoch=batch.epoch;state='decoding_and_measuring';progress();batch_dir=root/('batch-%04d'%(index+1));batch_dir.mkdir()
            required=list(batch.rows)+[dev_rows[index%len(dev_rows)]];resolved=[]
            for ordinal,row in enumerate(required):
                tick(deadline);cid=row['payload']['context_sha256']
                decode_seed=int(digest({'seed':seed,'batch_index':index+1,'data_run_id':row['data_run_id'],'ordinal':ordinal})[:15],16)
                z=torch.tensor(data.corpus.encodings[cid],device=device).reshape(1,-1)
                decoded=trie.decode(net,z,seed=decode_seed,temperature=config['temperature'],deadline=deadline)
                decoded.update(checkpoint_sha256=before,context_id=cid,data_run_id=row['data_run_id'])
                create(batch_dir/('decoded-%02d.json'%ordinal),canonical(decoded))
                result=resolver.resolve(row,decoded,before,index+1);create(batch_dir/('measured-%02d.json'%ordinal),canonical(result));resolved.append(result)
                comparison_seed=int(digest({'purpose':'fixed-panel-opponent','decoder_seed':decode_seed})[:15],16)
                games.extend(measured_games(result,pool,comparison_seed));
                if row['partition']=='validation':dev_seen.add(row['data_run_id'])
                progress()
            # Fail closed: every emitted context has a ready actual result before this update.
            need(len(resolved)==len(required)and all(r['state']=='ready'for r in resolved),'all generated outcomes gated')
            tick(deadline);admission=None if fresh_admission is None else fresh_admission()
            ids=[r['payload']['context_sha256']for r in batch.rows]
            tensors,draw=context_batch(data,ids,step=index+1,run_seed=run_seed,training=True)
            update=optimizer_update(net,opt,to_device(tensors,device))
            need(update['checkpoint_before']==before,'no hidden update before batch')
            saved=checkpoint(net,opt,root/('checkpoint-%04d'%(index+1)),step=index+1,config_sha256=config_sha)
            need(saved['parameter_sha256']==update['checkpoint_after'],'physical checkpoint equals updated model')
            proof={'schema':'jope.flywheel-neural-batch/v1','batch':asdict(batch),'draw':draw,'update':update,
                'physical_checkpoint':saved,'measurements':resolved,'fresh_research_admission':admission,'optimizer_targets':'frozen admitted snapshot only',
                'source_snapshot_sha256':loaded.receipt['snapshot_sha256'],'at':time.time()}
            proof_bytes=canonical(proof);create(batch_dir/'optimizer-evidence.json',proof_bytes)
            planner.commit(batch.batch_id,checkpoint_before=before,checkpoint_after=update['checkpoint_after'],
                           optimizer_step=index+1,ordered_data_run_ids=list(batch.ordered_data_run_ids),
                           evidence={'origin':'native_optimizer','event_sha256':sha(proof_bytes)})
            proofs.append({'batch_id':batch.batch_id,'evidence_sha256':sha(proof_bytes),'checkpoint_sha256':update['checkpoint_after']})
            seen.update(batch.ordered_data_run_ids);presentations+=len(batch.rows);steps+=1;readout(steps);state='training';progress()
        else:state='completed_bounded_qualification'
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)[:2000];state='failed_preserved_for_reconciliation';progress();raise
    finally:
        final={'schema':'jope.flywheel-neural-result/v1','state':state,'config_sha256':config_sha,'optimizer_steps':steps,
            'session_start_optimizer_step':session_start_step,'session_optimizer_steps':steps-session_start_step,
            'origin_parameter_sha256':resume['origin_parameter_sha256'],'parent_result_sha256':digest(prior),
            'proofs':proofs,'games':games,'loss_history':history,'resolver_outcomes':resolver.rows,'planner':planner.verify(),
            'observations':resolver.observer.journal.verify(),'queue':resolver.queue.verify(),'error':error}
        create(root/'result.json',canonical(final));planner_summary=planner.summary();planner.close();planner_closed=True;progress()
    return final
