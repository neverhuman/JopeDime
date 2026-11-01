#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, os, random, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", KMP_BLOCKTIME="0", OMP_WAIT_POLICY="PASSIVE", RAYON_NUM_THREADS="1", JE_ARROW_MALLOC_CONF="background_thread:false")
import numpy as np
import torch
CUSTODY=Path(os.environ.get('H133_SOURCE_ROOT','/home/ubuntu/jope-rescue-20260921/h133-optim-geometry'))
ROOT=Path(os.environ.get('H133_RUN_ROOT',str(CUSTODY))); RUNTIME=Path(os.environ.get('H133_RUNTIME_ROOT',str(CUSTODY/'runtime')))
EXPORT=CUSTODY/'train_only_export.json'; FOLD=CUSTODY/'fold_manifest_133100.json'; INPUT=Path(os.environ.get('H133_INPUT_ROOT','/home/ubuntu/jope-data/h50/input')); R882=Path(os.environ.get('H133_R882_PATH','/home/ubuntu/jope-data/h51/ckpt/round-882.pt'))
EXPECTED_R882_FAMILY_UTILITY_ROUNDED=1.1377942
R882_SHA='0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64'; DOPE_SHA='c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814'; EXPORT_SHA='ffe506f68d7aade13a7eef6f0cf55d6e79b200dd5e742f8f035dccd0ee27e3ff'; FOLD_SHA='322137e01ef0a6079284c0e907c8a5ce0db476d46eac1c6ea0fdce3315ea65fc'
SEEDS=(133,134,138); FIT_SEED=133100; UPDATES={'CONTROL':40,'LOWLR_LONG':160,'SGDM':40}; BATCH=16; SLATE=192; LR={'CONTROL':1e-4,'LOWLR_LONG':2.5e-5,'SGDM':0.003}; WD=1e-4; CLIP=1.0
MAX_RSS=6*1024**3; MIN_MEM=30*1024**3; MAX_LOAD=86.7; MAX_DISK=93.8; MAX_SCRATCH=2*1024**3; MAX_HOST_CPU=85.0

def load_module(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None: raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m
_torch_set_num_threads=torch.set_num_threads
torch.set_num_threads=lambda n: _torch_set_num_threads(min(int(n),1))
try:
 h=load_module('h133_h132_helpers',RUNTIME/'research/flywheel_selfplay/h132_ewc_bitfit/run_h132_cpu.py')
finally:
 torch.set_num_threads=_torch_set_num_threads
h.SEEDS=SEEDS; h.BOOTSTRAP_SEED=FIT_SEED; h.BOOTSTRAP_DRAWS=10_000
h48,h55,h53b,h59,h61=h.h48,h.h55,h.h53b,h.h59,h.h61
try:
 import threadpoolctl; _THREADPOOL_LIMITER=threadpoolctl.threadpool_limits(limits=1)
except ImportError: _THREADPOOL_LIMITER=None
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass

def sha(p):
 d=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): d.update(b)
 return d.hexdigest()
def rss():
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmRSS:'): return int(line.split()[1])*1024
 return 0
def memavail():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
 return 0
def cpu_pct_sample(interval=.35):
 def read():
  vals=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:])); return sum(vals), vals[3]+vals[4]
 a=read(); time.sleep(interval); b=read(); total=b[0]-a[0]; idle=b[1]-a[1]
 return 100.0*(total-idle)/max(total,1)
def guard(stage):
 fs=os.statvfs('/'); used=fs.f_blocks-fs.f_bfree; avail=fs.f_bavail; disk=100*used/max(used+avail,1); l1,l5,_=os.getloadavg(); rs=rss(); scratch=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()); cpu=cpu_pct_sample()
 tasks=len(list(Path('/proc/self/task').iterdir())); nice=os.getpriority(os.PRIO_PROCESS,0)
 snap={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),'nice':nice,'process_threads':tasks,'torch_threads':torch.get_num_threads(),'rss_bytes':rs,'mem_available_bytes':memavail(),'load1':l1,'disk_used_pct':disk,'scratch_bytes':scratch,'host_cpu_pct':cpu}
 if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or bool(getattr(torch.cuda, "_initialized", False)): raise RuntimeError(f'{stage}: CUDA visible or initialized')
 if os.environ.get("JE_ARROW_MALLOC_CONF") != "background_thread:false": raise RuntimeError(f'{stage}: Arrow allocator thread cap missing')
 if nice!=19 or tasks>4 or torch.get_num_threads()>4:
  tids=list(Path('/proc/self/task').iterdir()); names=[(q.name,(q/'comm').read_text().strip()) for q in tids]
  raise RuntimeError(f'{stage}: nice/thread guard failed: {nice}/{tasks}/{torch.get_num_threads()} tasks={names}')
 supervised=os.environ.get('H133_MANAGED_PAUSE')=='1' and os.environ.get('H133_LAUNCHER_PID')==str(os.getppid())
 if not supervised and (snap['mem_available_bytes']<MIN_MEM or l1>MAX_LOAD): raise RuntimeError(f'{stage}: load/memory guard requires A6 supervisor: {snap}')
 if rs>MAX_RSS or disk>=MAX_DISK or scratch>=MAX_SCRATCH or cpu>MAX_HOST_CPU: raise RuntimeError(f'{stage}: hard resource guard failed: {snap}')
 return snap
def persist(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n'); os.replace(tmp,path)
def make_tokens(net,tables,device):
 toks={}; masks={}; ws=h48.base.FusedLandmarkWorkspace(num_threads=1)
 for i,(cid,(x,y)) in enumerate(tables.items(),1):
  if i%16==0: guard(f'dope_tokens_{i}'); h48.base.headroom()
  yd=y.astype(np.float64); yn=((yd-yd.mean())/max(float(yd.std()),1e-8)).astype(np.float32); vec,cols,_=ws.fit_transform_and_extract_bundle_high(h48.base.landmark_float32(x),np.ascontiguousarray(yn))
  buf=np.zeros((64,17),np.float32); n=min(len(cols),64); buf[:n,:min(cols.shape[1],17)]=cols[:n,:17]
  with torch.no_grad():
   d,lm=net._run_dope_mlc_v3_and_lm48(torch.as_tensor(vec,device=device).reshape(1,256),torch.as_tensor(buf,device=device).reshape(1,64,17),use_landmarks=True); tok,mask,_=net._project_stage1b_tokens_batched_gpu(lm,d)
  toks[cid],masks[cid]=tok[0].detach(),mask[0].detach()
 return toks,masks
def build_plan(cids,pools,orders,seed,n):
 full=tuple(sorted(cids)); sampler=seed*1_000_003+10_007; plan=[]; state=np.random.get_state()
 try:
  for rnd in range(1,81):
   round_seed=int((sampler+rnd*104729)%(2**32)); np.random.seed(round_seed); rot=rnd%len(full); eligible=full[rot:]+full[:rot]
   for step in range(2):
    stream=h.paired_stream_seed(seed,rnd,step); cs=tuple(random.Random(stream).choices(eligible,k=BATCH)); ix=tuple(np.asarray(h53b.cached_training_indices(pools[c],orders[c],SLATE),dtype=np.int64).copy() for c in cs)
    if len(plan)<n: plan.append({'stream_seed':stream,'context_ids':cs,'candidate_indices':ix})
 finally: np.random.set_state(state)
 if len(plan)!=n: raise RuntimeError(f'plan count {len(plan)} != {n}')
 return plan
def load_model(label):
 ck=ROOT/'scratch'/f'r882_{label}.pt'; ck.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(R882,ck)
 try:
  if sha(ck)!=R882_SHA: raise RuntimeError('staged r882 hash mismatch')
  net,train,gg,_=h55.build_model(INPUT,ck,torch.device('cpu'),h48)
 finally: ck.unlink(missing_ok=True)
 if len(train)!=47 or sum(p.numel() for p in train)!=29211: raise RuntimeError('adapter trainable inventory mismatch')
 if any(not n.startswith('adapter_10tok.') or '.anchor.' in n for n,p in net.named_parameters() if p.requires_grad): raise RuntimeError('non-adapter trainable parameter')
 return net,train,gg
def nontrainable_sha(net):
 hsh=hashlib.sha256()
 for name,parameter in sorted(net.named_parameters()):
  if not parameter.requires_grad:
   hsh.update(name.encode())
   hsh.update(parameter.detach().cpu().contiguous().numpy().tobytes())
 return hsh.hexdigest()
def run_regression(fit_cids,assess_cids,all_pools,tokens,masks,fit_pools,fit_orders):
 device=torch.device('cpu'); neta,traina,ga=load_model('reg_a'); netb,trainb,gb=load_model('reg_b'); da=ga.snapshot_dope_weights(neta).composite_sha256; db=gb.snapshot_dope_weights(netb).composite_sha256
 if da!=DOPE_SHA or db!=DOPE_SHA: raise RuntimeError('regression DOPE mismatch')
 aa=h53b.adapter_sha(neta); ab=h53b.adapter_sha(netb)
 if aa!=ab: raise RuntimeError('baseline adapter copies differ')
 pa=h48.make_predictor(neta,all_pools,tokens,masks,device); pb=h48.make_predictor(netb,all_pools,tokens,masks,device)
 sa=h.predict_pool_scores_fast(assess_cids,all_pools,pa,64); sb=h.predict_pool_scores_fast(assess_cids,all_pools,pb,64); pd=max(float(np.max(np.abs(sa[c]-sb[c]))) for c in assess_cids)
 ea=h61.evaluate_pool_predictions(assess_cids,all_pools,sa,eval_mode='unrev'); eb=h61.evaluate_pool_predictions(assess_cids,all_pools,sb,eval_mode='unrev'); ud=abs(float(ea['mean'])-float(eb['mean']))
 plan=build_plan(fit_cids,fit_pools,fit_orders,133,40); b=plan[0]; cs=list(b['context_ids']); ix=list(b['candidate_indices']); tg=h59.preregistered_targets(torch.as_tensor(np.stack([fit_pools[c]['fitness'][j] for c,j in zip(cs,ix)]),dtype=torch.float32),torch.as_tensor(np.stack([fit_pools[c]['gate'][j] for c,j in zip(cs,ix)]),dtype=torch.bool))
 pred_a=h48.make_predictor(neta,fit_pools,tokens,masks,device); pred_b=h48.make_predictor(netb,fit_pools,tokens,masks,device); la=h.per_context_losses(pred_a(cs,ix),tg).mean(); lb=h.per_context_losses(pred_b(cs,ix),tg).mean(); la.backward(); lb.backward()
 ga_vec=torch.cat([p.grad.detach().reshape(-1).clone() if p.grad is not None else torch.zeros(p.numel()) for p in traina]); gb_vec=torch.cat([p.grad.detach().reshape(-1).clone() if p.grad is not None else torch.zeros(p.numel()) for p in trainb]); ld=abs(float(la.detach())-float(lb.detach())); gd=float(torch.max(torch.abs(ga_vec-gb_vec)).item())
 if max(pd,ud,ld,gd)!=0.0: raise RuntimeError(f'r882 regression not exact: pred={pd} utility={ud} loss={ld} grad={gd}')
 expected_match=round(float(ea['mean']),7)==EXPECTED_R882_FAMILY_UTILITY_ROUNDED
 rec={'schema':'jope.h133.regression_receipt/v1','status':'PASS' if expected_match else 'BASELINE_UTILITY_MISMATCH','utc':datetime.now(timezone.utc).isoformat(),'r882_sha256':R882_SHA,'dope_sha256':DOPE_SHA,'adapter_sha256':aa,'fold_seed':FIT_SEED,'fold_manifest_sha256':FOLD_SHA,'assessment_families':41,'assessment_contexts':len(assess_cids),'fit_families':164,'fit_contexts':len(fit_cids),'two_copy_fold0_prediction_max_abs_diff':pd,'two_copy_fold0_utility_abs_diff':ud,'control_fit_loss_abs_diff':ld,'control_fit_gradient_max_abs_diff':gd,'r882_fold0_unrev_family_utility':ea['mean'],'r882_fold0_unrev_context_utility':ea['context_mean'],'r882_fold0_per_family_utility':ea['per_family'],'r882_fold0_per_context_utility':ea['per_context'],'expected_family_utility_rounded':EXPECTED_R882_FAMILY_UTILITY_ROUNDED,'observed_family_utility_rounded':round(float(ea['mean']),7),'expected_utility_match':expected_match,'optimizer_updates':0}
 persist(ROOT/'regression_receipt.json',rec)
 if not rec['expected_utility_match']: raise RuntimeError(f"r882 baseline utility mismatch: expected {EXPECTED_R882_FAMILY_UTILITY_ROUNDED}, observed {rec['observed_family_utility_rounded']}")
 del neta,netb,pa,pb,pred_a,pred_b,traina,trainb; return rec,ea
def run_cell(arm,seed,plan,fit_pools,all_pools,tokens,masks,assess_cids,r882_eval):
 label=f'{arm}_seed{seed}'; net,train,gg=load_model(label); dopepre=gg.snapshot_dope_weights(net).composite_sha256
 if dopepre!=DOPE_SHA: raise RuntimeError(f'{label}: DOPE pre mismatch')
 frozenpre=nontrainable_sha(net); init=h53b.adapter_sha(net); init_params=[p.detach().clone() for _,p in sorted(net.adapter_10tok.named_parameters()) if not _.startswith('anchor.')]; pred=h48.make_predictor(net,fit_pools,tokens,masks,torch.device('cpu'))
 if arm=='SGDM': opt=torch.optim.SGD(train,lr=LR[arm],momentum=0.9,nesterov=True,weight_decay=WD)
 else: opt=torch.optim.AdamW(train,lr=LR[arm],weight_decay=WD)
 losses=[]; norms=[]; started=time.monotonic()
 for i,b in enumerate(plan,1):
  if i%4==1 or i==len(plan): guard(f'{label}_update_{i}')
  torch.manual_seed(b['stream_seed']); cs=list(b['context_ids']); ix=list(b['candidate_indices']); targets=h59.preregistered_targets(torch.as_tensor(np.stack([fit_pools[c]['fitness'][j] for c,j in zip(cs,ix)]),dtype=torch.float32),torch.as_tensor(np.stack([fit_pools[c]['gate'][j] for c,j in zip(cs,ix)]),dtype=torch.bool))
  opt.zero_grad(set_to_none=True); out=pred(cs,ix); loss=h.per_context_losses(out,targets).mean()
  if not bool(torch.isfinite(loss)): raise RuntimeError(f'{label}: nonfinite training loss')
  loss.backward(); norms.append(float(torch.nn.utils.clip_grad_norm_(train,CLIP,error_if_nonfinite=True).item())); opt.step(); losses.append(float(loss.detach().item())); del targets,out,loss
 dopepost=gg.snapshot_dope_weights(net).composite_sha256; frozenpost=nontrainable_sha(net)
 if dopepost!=DOPE_SHA or frozenpost!=frozenpre: raise RuntimeError(f'{label}: frozen hashes changed')
 params=[p.detach() for _,p in sorted(net.adapter_10tok.named_parameters()) if not _.startswith('anchor.')]; drift=float(torch.linalg.vector_norm(torch.cat([(a-b).reshape(-1).to(torch.float64) for a,b in zip(params,init_params)])).item()); final=h53b.adapter_sha(net)
 eval_pred=h48.make_predictor(net,all_pools,tokens,masks,torch.device('cpu')); scores=h.predict_pool_scores_fast(assess_cids,all_pools,eval_pred,64); ev=h61.evaluate_pool_predictions(assess_cids,all_pools,scores,eval_mode='unrev'); maxdiff=max(abs(float(ev['per_context'][c])-float(r882_eval['per_context'][c])) for c in assess_cids); snap=guard(f'{label}_complete'); wall=time.monotonic()-started
 doc={'schema':'jope.h133.cell_receipt/v1','wave':'H133','arm':arm,'seed':seed,'updates':len(plan),'optimizer':'SGDM' if arm=='SGDM' else 'AdamW','lr':LR[arm],'momentum':0.9 if arm=='SGDM' else None,'nesterov':arm=='SGDM','weight_decay':WD,'grad_clip':CLIP,'dope_sha256_pre':dopepre,'dope_sha256_post':dopepost,'nontrainable_sha256_pre':frozenpre,'nontrainable_sha256_post':frozenpost,'initial_adapter_sha256':init,'final_adapter_sha256':final,'adapter_l2_drift_from_r882':drift,'max_abs_context_utility_diff_from_r882':maxdiff,'mean_step_loss':float(np.mean(losses)),'median_preclip_grad_norm':float(np.median(norms)),'fold0_family_mean_utility':ev['mean'],'fold0_context_mean_utility':ev['context_mean'],'fold0_per_family_utilities':ev['per_family'],'fold0_per_context_utilities':ev['per_context'],'elapsed_seconds':round(wall,3),'guard_complete':snap}
 path=ROOT/'cells'/f'{label}.json'; path.parent.mkdir(parents=True,exist_ok=True); persist(path,doc); del net,train,gg,pred,opt; return doc
def main():
 start=time.monotonic(); guard('wave_start')
 for p,d in [(R882,R882_SHA),(EXPORT,EXPORT_SHA),(FOLD,FOLD_SHA)]:
  if sha(p)!=d: raise RuntimeError(f'input sha mismatch {p}')
 ex=json.loads(EXPORT.read_bytes()); rows=ex.get('rows')
 if ex.get('schema')!='jope.h133.dl-train-only-metadata/v1' or len(rows)!=1219 or any(r.get('dl_split')!='DL-Train' for r in rows): raise RuntimeError('train-only export closure mismatch')
 folds=json.loads(FOLD.read_bytes()); famset=set().union(*(set(x) for x in folds))
 if len(folds)!=5 or famset!={r['family_id'] for r in rows}: raise RuntimeError('fold manifest mismatch')
 fam_by_cid={r['context_sha256']:r['family_id'] for r in rows}; fit_fams=set(f for fold in folds[1:] for f in fold); assess_fams=set(folds[0]); fit_rows=[r for r in rows if r['family_id'] in fit_fams]; assess_rows=[r for r in rows if r['family_id'] in assess_fams]
 fit_cids=sorted(r['context_sha256'] for r in fit_rows); assess_cids=sorted(r['context_sha256'] for r in assess_rows)
 if (len(fit_fams),len(assess_fams),len(fit_cids),len(assess_cids))!=(164,41,974,245): raise RuntimeError('fold roster counts invalid')
 all_pools={}; fit_pools={}; orders={}; tables={}; w1=h48.base.evaluator.get_w1_sign_matrix()/np.sqrt(412.0); fit_cid_set=set(fit_cids)
 for i,row in enumerate(rows,1):
  if row.get('dl_split')!='DL-Train': raise RuntimeError('non-train row')
  cid=row['context_sha256']; pp=INPUT/'pools'/f'{cid}.npz'; tp=INPUT/'tables'/f'{cid}.jtt1'
  if sha(pp)!=row['pool_sha256']: raise RuntimeError(f'pool hash mismatch {cid}')
  raw=tp.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=row['train_only_jtt1_sha256']: raise RuntimeError(f'table hash mismatch {cid}')
  nt,nq,width=h48.base.struct.unpack_from('<III',raw,4)
  if raw[:4]!=b'JTT1' or nq!=0 or raw[16]!=2: raise RuntimeError(f'non-train JTT1 {cid}')
  vals=np.frombuffer(raw,dtype='<f8',offset=17+nt).reshape(nt,width+1).copy(); tables[cid]=(vals[:,:-1],vals[:,-1])
  with np.load(pp,allow_pickle=False) as z:
   programs=z['program'].astype(np.float32); fitness=z['fitness'].astype(np.float64); gate=z['gate'].astype(bool); pids=[v.decode() if isinstance(v,bytes) else str(v) for v in z['program_sha256'].tolist()]
  item={'rec':row,'pids':pids,'program':programs,'proj16':(programs.astype(np.float64)@w1).astype(np.float32),'fitness':fitness,'gate':gate}; all_pools[cid]=item
  if cid in fit_cid_set: fit_pools[cid]=item; orders[cid]=np.argsort(fitness)
  if i%32==0: guard(f'load_train_payloads_{i}')
 del rows,ex
 if set(fit_pools)!=set(fit_cids) or set(all_pools)!=(set(fit_cids)|set(assess_cids)): raise RuntimeError('loaded row roster mismatch')
 net,train,gg=load_model('tokens'); tokens,masks=make_tokens(net,tables,torch.device('cpu')); dope=gg.snapshot_dope_weights(net).composite_sha256
 if dope!=DOPE_SHA or set(tokens)!=set(all_pools): raise RuntimeError('token/DOPE closure mismatch')
 del net,train,gg,tables
 regression,r882_eval=run_regression(fit_cids,assess_cids,all_pools,tokens,masks,fit_pools,orders); print(json.dumps({'phase':'REGRESSION_PASS','family_utility':r882_eval['mean']}),flush=True)
 cells={}
 for seed in SEEDS:
  for arm in ('CONTROL','LOWLR_LONG','SGDM'):
   path=ROOT/'cells'/f'{arm}_seed{seed}.json'
   if path.exists(): raise RuntimeError('existing cell detected; do not resume/retry')
   plan=build_plan(fit_cids,fit_pools,orders,seed,UPDATES[arm]); doc=run_cell(arm,seed,plan,fit_pools,all_pools,tokens,masks,assess_cids,r882_eval); cells[(arm,seed)]=doc
   print(json.dumps({'phase':'CELL_COMPLETE','arm':arm,'seed':seed,'updates':doc['updates'],'drift':doc['adapter_l2_drift_from_r882'],'fold0_family_utility':doc['fold0_family_mean_utility'],'elapsed':doc['elapsed_seconds']}),flush=True)
 fam={a:{s:cells[(a,s)]['fold0_per_family_utilities'] for s in SEEDS} for a in ('CONTROL','LOWLR_LONG','SGDM')}; ctx={a:{s:cells[(a,s)]['fold0_per_context_utilities'] for s in SEEDS} for a in ('CONTROL','LOWLR_LONG','SGDM')}; r_fam={s:r882_eval['per_family'] for s in SEEDS}; r_ctx={s:r882_eval['per_context'] for s in SEEDS}
 vs={a:h.evaluate_arm_vs_comparator(a,'R882',fam[a],r_fam,ctx[a],r_ctx,sorted(assess_fams),assess_cids) for a in ('CONTROL','LOWLR_LONG','SGDM')}; arms=sorted(('LOWLR_LONG','SGDM'),key=lambda a:vs[a]['p_value_one_sided']); holm={}; prior=True
 for rank,a in enumerate(arms):
  st=vs[a]; alpha=.05/(2-rank); lcb=st['lcb975_one_sided'] if rank==0 else st['lcb95_one_sided']; passed=prior and st['p_value_one_sided']<=alpha and lcb>0; prior=passed; gates={'median_seed_gain_gt_0_001':st['median_seed_gain']>0.001,'holm_lcb_gt_0':passed,'family_harm_le_10pct':st['pooled_family_harm_fraction']<=0.10,'all_three_seeds_positive':st['all_three_seeds_positive']}; st.update({'holm_alpha':alpha,'holm_lcb':lcb,'holm_pass':passed,'gates':gates,'admitted':all(gates.values())}); holm[a]={'rank':rank+1,'alpha':alpha,'holm_lcb':lcb,'p_value':st['p_value_one_sided'],'gates':gates,'admitted':all(gates.values())}
 verdict='ADMITTED' if any(x['admitted'] for x in holm.values()) else 'NO_ADMISSION'
 summary={'schema':'jope.h133.metrics/v1','wave':'H133','status':'COMPLETE','complete':True,'verdict':verdict,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':round(time.monotonic()-start,2),'gpu_seconds':0.0,'assessment_payload_reads':245,'assessment_outcome_metrics_read':1,'dl_val_reads':0,'outer_test_reads':0,'sealed_holdout_reads':0,'live_loop_reads':0,'optimizer_updates':sum(UPDATES[a]*len(SEEDS) for a in UPDATES),'cells_completed':len(cells),'provenance':{'merged_prereq_commit':'dbd88c910f30742a93e839289597beab452c2879','r882_sha256':R882_SHA,'dope_sha256':DOPE_SHA,'train_only_export_sha256':EXPORT_SHA,'fold_manifest_sha256':FOLD_SHA,'fold_seed':FIT_SEED,'seeds':list(SEEDS),'fit_families':164,'fit_contexts':len(fit_cids),'assessment_families':41,'assessment_contexts':len(assess_cids),'fit_folds':[1,2,3,4],'assessment_fold':0,'runtime_source_commit':'30be5edb79b0ce1406aa0367d68019c9a3a3c1e1','h132_helper_sha256':sha(RUNTIME/'research/flywheel_selfplay/h132_ewc_bitfit/run_h132_cpu.py'),'h133_runner_sha256':sha(Path(__file__))},'regression_gate':regression,'primary_vs_r882':vs,'holm':holm,'cells':{f'{a}_seed{s}':{'arm':a,'seed':s,'updates':cells[(a,s)]['updates'],'adapter_l2_drift_from_r882':cells[(a,s)]['adapter_l2_drift_from_r882'],'final_adapter_sha256':cells[(a,s)]['final_adapter_sha256'],'fold0_family_mean_utility':cells[(a,s)]['fold0_family_mean_utility'],'elapsed_seconds':cells[(a,s)]['elapsed_seconds']} for s in SEEDS for a in UPDATES},'guards_final':guard('wave_complete')}
 persist(ROOT/'metrics.json',summary); print(json.dumps({'phase':'WAVE_COMPLETE','verdict':verdict,'receipt':str(ROOT/'metrics.json'),'primary':{a:{'median':vs[a]['median_seed_gain'],'lcb':vs[a]['lcb95_one_sided'],'harm':vs[a]['pooled_family_harm_fraction'],'all_seeds_positive':vs[a]['all_three_seeds_positive'],'admitted':vs[a].get('admitted',False)} for a in ('LOWLR_LONG','SGDM')}},sort_keys=True),flush=True)
if __name__=='__main__': main()
