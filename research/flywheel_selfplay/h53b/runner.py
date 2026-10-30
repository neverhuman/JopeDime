#!/usr/bin/env python3
"""Run one preregistered H53b fresh-adapter cell on the immutable offline packet."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, random, sys, time
from pathlib import Path
from types import MethodType
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
H48_PATH = ROOT / "research/flywheel_selfplay/h48/run.py"
H49_SCORE_PATH = ROOT / "research/flywheel_selfplay/h49/score.py"
FUSED_PATH = ROOT / "research/flywheel_selfplay/run_fused_closed_loop.py"
DEV_PATH = ROOT / "research/flywheel_selfplay/dev_incumbent.py"
TRAINING_PATH = ROOT / "research/flywheel_selfplay/automatic_promotion.py"
INPUT_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
TRAINABLE_PARAMS = 29_211
K_STEPS, LEARNING_RATE, WEIGHT_DECAY = 2, 1e-4, 1e-4
BATCH_SIZE, SLATE_SIZE = 1219, 192
EVAL_EVERY = 25
TARGETS = (1.1536152769912449, 1.1377942)

def import_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

def sha256(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): d.update(block)
    return d.hexdigest()

def adapter_sha(net):
    d=hashlib.sha256()
    for name,p in sorted(net.adapter_10tok.named_parameters()):
        if not name.startswith('anchor.'):
            d.update(name.encode()); d.update(p.detach().contiguous().cpu().numpy().tobytes())
    return d.hexdigest()

def reset_adapter(net, scale, seed):
    old=net.adapter_10tok; cls=type(old)
    torch.manual_seed(seed); np.random.seed(seed)
    fresh=cls(frozen_anchor=old.anchor, prog_mean=old.prog_mean.detach().cpu().numpy(),
              prog_std=old.prog_std.detach().cpu().numpy(), d_token=32, n_modalities=10,
              modality_dropout_p=0.10).to(next(old.parameters()).device)
    for method_name in ('encode_multimodal_context_and_candidates','predict_parent_child_delta'):
        bound=getattr(old,method_name); setattr(fresh,method_name,MethodType(bound.__func__,fresh))
    net.adapter_10tok=fresh
    for p in net.parameters(): p.requires_grad_(False)
    for name,p in net.adapter_10tok.named_parameters():
        if not name.startswith('anchor.'):
            p.requires_grad_(True)
            if scale != 1.0:
                with torch.no_grad(): p.mul_(scale)
    trainable=[p for p in net.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable)!=TRAINABLE_PARAMS: raise RuntimeError('fresh adapter parameter inventory mismatch')
    if any(name.startswith('anchor.') and p.requires_grad for name,p in net.adapter_10tok.named_parameters()): raise RuntimeError('frozen anchor trainable')
    return trainable

def bootstrap(gains, seed):
    values=np.asarray(list(gains.values()),dtype=np.float64)
    if values.size!=19: raise RuntimeError(f'expected 19 families, got {values.size}')
    rng=np.random.default_rng(seed); ix=rng.integers(0,values.size,size=(10000,values.size))
    means=values[ix].mean(axis=1); medians=np.median(values[ix],axis=1)
    return {'family_count':19,'mean_gain':float(values.mean()),'median_gain':float(np.median(values)),
            'mean_lcb_one_sided_95':float(np.quantile(means,.05)),
            'median_lcb_two_sided_95':float(np.quantile(medians,.025)),
            'median_ucb_two_sided_95':float(np.quantile(medians,.975))}

def live_objective(prediction, pools, cids, indexes):
    target_np=np.stack([pools[cid]['fitness'][ix] for cid,ix in zip(cids,indexes,strict=True)])
    gate_np=np.stack([pools[cid]['gate'][ix] for cid,ix in zip(cids,indexes,strict=True)])
    target=torch.as_tensor(target_np,dtype=torch.float32,device=prediction.device)
    gates=torch.as_tensor(gate_np,dtype=torch.bool,device=prediction.device)
    target=torch.where(gates,target,torch.full_like(target,2.0))
    row_huber=F.smooth_l1_loss(prediction,target,reduction='none').mean(dim=1)
    row_kl=F.kl_div(F.log_softmax(-prediction/.2,dim=-1),F.softmax(-target/.2,dim=-1),reduction='none').sum(dim=1)
    return (row_huber+.1*row_kl).mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,choices=(12,13,14),required=True)
    ap.add_argument('--scale',type=float,choices=(1.0,.1),required=True)
    ap.add_argument('--max-rounds',type=int,default=300)
    ap.add_argument('--wall-seconds',type=int,default=1200)
    ap.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    ap.add_argument('--input',type=Path,default=Path.home()/'jope-data/h50/input')
    ap.add_argument('--checkpoint',type=Path,default=Path.home()/'jope-data/h51/ckpt/round-882.pt')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--lock-held',action='store_true')
    a=ap.parse_args()
    if a.max_rounds!=300 or not 1<=a.wall_seconds<=1200: raise RuntimeError('cell budget violates preregistration')
    if a.device=='cuda:0' and not a.lock_held: raise RuntimeError('CUDA cell requires qualification.lock')
    if sha256(a.input/'source_manifest.json')!=INPUT_SHA or sha256(a.checkpoint)!=R882_SHA: raise RuntimeError('pinned input identity mismatch')
    torch.set_num_threads(4); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device(a.device)
    h48=import_module('h53b_h48',H48_PATH); h48.CHECKPOINT_SHA=R882_SHA
    h49=import_module('h53b_h49_score',H49_SCORE_PATH)
    dev=import_module('h53b_dev_incumbent',DEV_PATH)
    training=import_module('h53b_training_indices',TRAINING_PATH)
    selected,train_pools,train_ids,family_by_cid,net,trainable,tokens,masks,guard=h48.build_model(a.input,a.checkpoint,device)
    if len(train_ids)!=1219 or len({family_by_cid[c] for c in train_ids})!=205: raise RuntimeError('pinned train roster mismatch')
    selected_val,val_pools,val_tables=h49.load_validation_packet(a.input)
    if selected_val!=selected['dev']: raise RuntimeError('DL-Val roster mismatch')
    val_tokens,val_masks=h48.h46.make_tokens(net,val_tables,device)
    val_ids=[row['context_sha256'] for row in selected_val]
    val_families={row['context_sha256']:row['family_id'] for row in selected_val}
    val_predict=h48.make_predictor(net,val_pools,val_tokens,val_masks,device)
    baseline=h48.evaluate_utility(val_ids,val_pools,val_predict)
    baseline_context,baseline_family=baseline['per_context'],baseline['per_family']
    r882_adapter_sha=adapter_sha(net)
    trainable=reset_adapter(net,a.scale,a.seed); init_sha=adapter_sha(net)
    init_params={n:p.detach().clone() for n,p in net.named_parameters() if p.requires_grad}
    dope_before=h48.h46.checkpoint_guard(guard,net,expected=DOPE_SHA)
    train_predict=h48.make_predictor(net,train_pools,tokens,masks,device)
    optimizer=torch.optim.AdamW(trainable,lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    h48.h46.base.guard.assert_optimizer_excludes_dope(optimizer,net)
    started=time.monotonic(); deadline=started+a.wall_seconds
    net.eval(); initial=h48.evaluate_utility(val_ids,val_pools,val_predict)
    trajectory=[{'round':0,'family_utility':initial['mean'],'context_utility':initial['context_mean'],
                 'per_family_utility':initial['per_family'],'per_context_utility':initial['per_context'],
                 'adapter_sha256':init_sha,'diagnostic_only':True}]
    completed=0; stop_reason='round_cap'
    for round_idx in range(1,a.max_rounds+1):
        if time.monotonic()>=deadline: stop_reason='wall_cap'; break
        round_seed=a.seed+round_idx*104729
        random.seed(round_seed); np.random.seed(round_seed%(2**32)); torch.manual_seed(round_seed)
        rotation=round_idx%len(train_ids); round_train=train_ids[rotation:]+train_ids[:rotation]
        for step in range(K_STEPS):
            if time.monotonic()>=deadline: stop_reason='wall_cap_mid_unit'; break
            step_seed=dev.batch_seed(a.seed,round_idx,step)
            cids=random.Random(step_seed).choices(round_train,k=BATCH_SIZE)
            np.random.seed(round_seed%(2**32))
            indexes=[training.training_indices(train_pools[cid],SLATE_SIZE) for cid in cids]
            optimizer.zero_grad(set_to_none=True)
            loss=live_objective(train_predict(cids,indexes),train_pools,cids,indexes)
            loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(trainable,1.0)); optimizer.step()
            if h48.h46.checkpoint_guard(guard,net,expected=DOPE_SHA)!=DOPE_SHA: raise RuntimeError('DOPE changed during cell')
        if stop_reason=='wall_cap_mid_unit': break
        completed=round_idx
        if completed%EVAL_EVERY==0 or completed==a.max_rounds or time.monotonic()>=deadline:
            net.eval(); val=h48.evaluate_utility(val_ids,val_pools,val_predict)
            digest=adapter_sha(net)
            drift=float(torch.sqrt(sum(torch.sum((p.detach()-init_params[n])**2) for n,p in net.named_parameters() if p.requires_grad)))
            trajectory.append({'round':completed,'family_utility':val['mean'],'context_utility':val['context_mean'],
                'per_family_utility':val['per_family'],'per_context_utility':val['per_context'],
                'adapter_sha256':digest,'update_l2_from_init':drift,'elapsed_seconds':time.monotonic()-started,'diagnostic_only':True})
            net.train()
        if time.monotonic()>=deadline: stop_reason='wall_cap'; break
    if completed==0: raise RuntimeError('cell ended before a complete live update unit')
    if trajectory[-1]['round']!=completed:
        net.eval(); val=h48.evaluate_utility(val_ids,val_pools,val_predict); digest=adapter_sha(net)
        drift=float(torch.sqrt(sum(torch.sum((p.detach()-init_params[n])**2) for n,p in net.named_parameters() if p.requires_grad)))
        trajectory.append({'round':completed,'family_utility':val['mean'],'context_utility':val['context_mean'],
            'per_family_utility':val['per_family'],'per_context_utility':val['per_context'],'adapter_sha256':digest,
            'update_l2_from_init':drift,'elapsed_seconds':time.monotonic()-started,'diagnostic_only':True})
    final=trajectory[-1]; gains={f:float(final['per_family_utility'][f])-float(baseline_family[f]) for f in baseline_family}
    stats=bootstrap(gains,53000+a.seed+int(a.scale*10))
    elo=import_module('h53b_elo',ROOT/'python/src/jope/north_star/elo_tracker.py')
    records=[elo.DatasetEvalRecord(dataset_id=cid,family_id=val_families[cid],task_type='regression',
        scores_by_contender={'r882':float(baseline_context[cid]),'H53b':float(final['per_context_utility'][cid])}) for cid in val_ids]
    elo_result=elo.compute_family_grouped_bradley_terry_elo(records,anchor_contender='r882',anchor_elo=1500.0)
    dope_after=h48.h46.checkpoint_guard(guard,net,expected=DOPE_SHA)
    if dope_after!=DOPE_SHA or sha256(a.input/'source_manifest.json')!=INPUT_SHA or sha256(a.checkpoint)!=R882_SHA: raise RuntimeError('integrity identity changed')
    crossings={str(target):next((r['round'] for r in trajectory if r['family_utility']>=target),None) for target in TARGETS}
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (FUSED_PATH,DEV_PATH,TRAINING_PATH,H48_PATH,H49_SCORE_PATH,Path(__file__).resolve(),ROOT/'research/flywheel_selfplay/h53b/run_wave.py',ROOT/'research/flywheel_selfplay/h53b/audit.py')}
    payload={'schema':'jope.h53b.cell/v1','seed':a.seed,'initialization_scale':a.scale,'rounds_completed':completed,
        'optimizer_updates_completed':completed*K_STEPS,'k_steps_per_round':K_STEPS,'learning_rate':LEARNING_RATE,
        'weight_decay':WEIGHT_DECAY,'batch_size':BATCH_SIZE,'slate_size':SLATE_SIZE,'batch_sampler':'dev_incumbent.batch_seed + random.Random.choices(round-rotated-train-roster); automatic_promotion.training_indices',
        'objective':'mean(SmoothL1 + 0.1*KL(softmax(-prediction/0.2), softmax(-target/0.2))); gated target=2.0',
        'device':str(device),'train_contexts':len(train_ids),'train_families':205,'validation_contexts':len(val_ids),'validation_families':19,
        'manifest_identity':MANIFEST_ID,'manifest_file_sha256':INPUT_SHA,'dope_sha256_before':dope_before,'dope_sha256_after':dope_after,
        'r882_checkpoint_sha256_comparator_only':R882_SHA,'r882_adapter_sha256_scored_then_discarded':r882_adapter_sha,
        'fresh_adapter_sha256_before_training':init_sha,'fresh_adapter_sha256_after_training':adapter_sha(net),
        'trainable_adapter_update_l2_from_init':final.get('update_l2_from_init'),'r882_adapter_transfer_to_candidate':False,
        'family_gain_vs_r882_diagnostic':gains,'family_bootstrap':stats,'elo_vs_r882_diagnostic':elo_result,
        'r882_harness_comparator_utility':1.1536152769912449,'live_loop_target_utility':1.1377942,
        'post_r882_candidate_ceiling':{'n':543,'mean':1.13393,'sd':.00212,'p50':1.13443,'p99_max':1.13771},
        'rounds_to_cross_at_diagnostic_reads':crossings,'best_observed_diagnostic_utility':max(r['family_utility'] for r in trajectory),
        'final_round':completed,'final_family_utility':final['family_utility'],'final_context_utility':final['context_utility'],
        'family_utility_gap_vs_harness_r882':final['family_utility']-1.1536152769912449,
        'family_utility_gap_vs_live_target':final['family_utility']-1.1377942,'stop_reason':stop_reason,
        'wall_seconds':time.monotonic()-started,'trajectory':trajectory,'validation_diagnostic_only':True,
        'holdout_payloads_opened':0,'outer_rows_delivered':0,'source_sha256':source_hashes}
    a.out.mkdir(parents=True,exist_ok=True); tmp=a.out/'summary.json.new'; tmp.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n'); tmp.replace(a.out/'summary.json')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('trajectory','family_gain_vs_r882_diagnostic')},allow_nan=False),flush=True)

if __name__=='__main__': main()
