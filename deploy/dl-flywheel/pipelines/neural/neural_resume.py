"""Exact checkpoint restoration; no launch, authorization or unknown replay.
Caller must hold exclusive run ownership, stopped-service proof, committed physical
checkpoint chain and fresh research admission. First immutable runtime stays intact.
"""
from pathlib import Path
import hashlib,json,math
import torch
from safetensors.torch import load as load_tensors
from dltrain import model as M

MAX_STATE_BYTES=64*1024**2
MANIFEST_KEYS={'schema','step','config_sha256','parameter_sha256','state_sha256','state_bytes',
 'model_config','parameters','optimizer','learning_rate','weight_decay','resume'}
def need(ok,message):
 if not ok:raise ValueError(message)
def read(path,cap):
 path=Path(path)
 need(path.is_absolute()and'..'not in path.parts and all(not p.is_symlink()for p in(path,*path.parents)), 'absolute nonsymlink checkpoint path')
 need(path.is_file()and 0<path.stat().st_size<=cap,'bounded checkpoint file')
 with path.open('rb')as f:body=f.read(cap+1)
 need(len(body)<=cap,'checkpoint grew beyond cap');return body
def sha(body):return hashlib.sha256(body).hexdigest()

def restore(path,*,manifest_sha256,config_sha256,parameter_sha256,optimizer_step,model_config,device):
 """Expected identities come from verified committed evidence, not this file.
 Restore parameters, AdamW named moments/scalar steps, CPU and single-CUDA RNG.
 AdamW settings explicitly match original Torch2.7 default recipe.
 """
 path=Path(path);body=read(path/'manifest.json',1024**2)
 need(sha(body)==manifest_sha256,'checkpoint manifest SHA mismatch');manifest=json.loads(body)
 need(set(manifest)==MANIFEST_KEYS and manifest['schema']=='jope.flywheel-neural-checkpoint/v1','closed original checkpoint schema')
 need(type(optimizer_step)is int and optimizer_step>=0 and manifest['step']==optimizer_step,'committed optimizer step mismatch')
 need(manifest['config_sha256']==config_sha256 and manifest['parameter_sha256']==parameter_sha256,'checkpoint/config committed identity mismatch')
 need(manifest['model_config']==model_config and manifest['optimizer']=='AdamW'and manifest['weight_decay']==.01 and math.isfinite(manifest['learning_rate'])and manifest['learning_rate']>0,'unchanged model/optimizer recipe required')
 raw=read(path/'state.safetensors',MAX_STATE_BYTES)
 need(sha(raw)==manifest['state_sha256']and len(raw)==manifest['state_bytes'],'physical state hash/size mismatch')
 state=load_tensors(raw)
 need(all(isinstance(t,torch.Tensor)and torch.isfinite(t).all().item()for t in state.values()),'finite tensor-only physical state')
 device=torch.device(device);net=M.PolicyLM(M.PolicyConfig.from_json(model_config)).to(device)
 need(M.count_parameters(net)==manifest['parameters'],'model parameter count mismatch')
 params=dict(net.named_parameters());model_keys={'model.'+n for n in params}
 need({k for k in state if k.startswith('model.')}==model_keys,'exact native named parameter set')
 with torch.no_grad():
  for name,p in params.items():
   tensor=state['model.'+name];need(tensor.shape==p.shape and tensor.dtype==p.dtype,'parameter shape/dtype mismatch');p.copy_(tensor)
 need(M.parameter_sha256(net)==parameter_sha256,'physical model parameter digest mismatch')
 opt=torch.optim.AdamW(M.param_groups(net,.01),lr=manifest['learning_rate'],betas=(.9,.999),eps=1e-8,amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None)
 expected=set(model_keys)|{'rng.torch_cpu'}
 for name,p in params.items():
  prefix='optimizer.'+name+'.';keys={k for k in state if k.startswith(prefix)}
  if not keys:continue
  names={prefix+k for k in('step','exp_avg','exp_avg_sq')}
  need(keys==names and optimizer_step>0,'complete AdamW moment triplet required')
  step=state[prefix+'step'];need(step.shape==torch.Size([])and step.dtype==torch.float32 and float(step).is_integer()and 1<=float(step)<=optimizer_step,'bounded original AdamW scalar step')
  restored={'step':step.clone()}
  for key in('exp_avg','exp_avg_sq'):
   tensor=state[prefix+key];need(tensor.shape==p.shape and tensor.dtype==p.dtype,'AdamW moment shape/dtype mismatch')
   if key=='exp_avg_sq':need((tensor>=0).all().item(),'nonnegative second moment')
   restored[key]=tensor.to(device).clone()
  opt.state[p]=restored;expected.update(names)
 need(optimizer_step==0 or len(opt.state)>0,'trained checkpoint cannot reset all optimizer moments')
 cpu_rng=state.get('rng.torch_cpu');need(cpu_rng is not None and cpu_rng.dtype==torch.uint8 and cpu_rng.ndim==1,'exact CPU RNG tensor')
 cuda_keys={k for k in state if k.startswith('rng.torch_cuda.')}
 if device.type=='cuda':
  need(torch.cuda.device_count()==1 and cuda_keys=={'rng.torch_cuda.0'},'original single CUDA RNG required')
  rng=state['rng.torch_cuda.0'];need(rng.dtype==torch.uint8 and rng.ndim==1,'exact CUDA RNG tensor');expected.update(cuda_keys)
 else:need(not cuda_keys,'CUDA checkpoint cannot silently resume as CPU')
 need(set(state)==expected,'unknown checkpoint tensors refused')
 torch.set_rng_state(cpu_rng)
 if device.type=='cuda':torch.cuda.set_rng_state_all([state['rng.torch_cuda.0']])
 return net,opt,{'schema':'jope.flywheel-neural-resume-proof/v1','manifest_sha256':manifest_sha256,'origin_config_sha256':config_sha256,'state_sha256':manifest['state_sha256'],'parameter_sha256':parameter_sha256,'optimizer_step':optimizer_step,'optimizer_parameter_states':len(opt.state),'cpu_rng_restored':True,'cuda_rng_restored':device.type=='cuda','automatic_unknown_replay':False}

def require_committed_cursor(planner,*,parameter_sha256,optimizer_step):
 chain=planner.verify();summary=planner.summary()
 need(planner.pending()is None,'pending optimizer batch requires explicit reconciliation')
 need(summary['checkpoint_sha256']==parameter_sha256 and summary['optimizer_steps_reported']==optimizer_step,'physical checkpoint and durable cursor disagree')
 return {'chain':chain,'summary':summary}

def adopt_at_epoch_boundary(planner,raw,expected_sha256,*,parameter_sha256):
 need(planner.pending()is None,'snapshot change cannot bypass pending optimizer work')
 summary=planner.summary();epoch=summary['epoch'];row=planner.db.execute('SELECT state FROM epochs WHERE id=?',(epoch,)).fetchone()
 need(epoch==0 or(row is not None and row['state']=='complete'),'snapshot adoption only after completed epoch')
 planner.adopt_manifest(raw,expected_sha256,checkpoint_sha256=parameter_sha256);return planner.verify()


def apply_frozen_normalization(data,receipt):
 """Keep original learned input coordinate system across resumed snapshots."""
 import numpy as np
 from dataclasses import replace
 from jope_flywheel.protocol import digest
 need(receipt['schema']=='jope.flywheel-context-preprocessing/v1'and receipt['encoder']=='kumo_tabular_s'and
      receipt['dimension']==2051 and receipt['clip']==[-10.,10.]and receipt['validation_used_for_fit']is False,
      'original KumoS-only train-fit normalization required')
 original={k:v for k,v in receipt.items()if k not in('statistics_identity_sha256','input_train_data_sha256','output_train_data_sha256')}
 need(digest(original)==receipt['statistics_identity_sha256'],'frozen normalizer identity mismatch')
 need(set(receipt['original_float64_hashes'])<=set(data.corpus.encodings),'snapshot must retain original normalized contexts')
 need(all(data.corpus.contexts[c]['split_role']=='train_fit'for c in receipt['fit_contexts']), 'original normalization fit roles unchanged')
 mean=np.asarray(receipt['mean'],dtype=np.float64);scale=np.asarray(receipt['scale'],dtype=np.float64)
 need(mean.shape==scale.shape==(2051,)and np.isfinite(mean).all()and np.isfinite(scale).all()and(scale>0).all(), 'finite fixed normalization statistics')
 transformed={}
 for context,value in data.corpus.encodings.items():
  z=np.asarray(value,dtype=np.float64);need(z.shape==(2051,)and np.isfinite(z).all(),'KumoS2051 context')
  old=receipt['original_float64_hashes'].get(context)
  need(old is None or sha(z.astype('<f8').tobytes())==old,'previous context encoding changed under frozen normalization')
  transformed[context]=np.clip((z-mean)/scale,-10.,10.).astype(np.float32)
 refs=dict(data.refs or {});refs['normalizers']='flywheel-preprocess:'+receipt['statistics_identity_sha256']
 updated=replace(data,corpus=replace(data.corpus,encodings=transformed),refs=refs)
 return updated,{'schema':'jope.flywheel-frozen-normalization-application/v1','statistics_identity_sha256':receipt['statistics_identity_sha256'],
  'input_train_data_sha256':data.content_sha256,'output_train_data_sha256':updated.content_sha256,'refitted':False,
  'new_contexts':sorted(set(transformed)-set(receipt['original_float64_hashes']))}
