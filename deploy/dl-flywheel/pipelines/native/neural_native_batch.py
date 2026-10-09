"""Direct supported tensors for native contexts; never manufacture missing y or q."""
from dataclasses import dataclass,fields,replace
import math
import numpy as np
import torch
from dltrain import labels as L,objective as O
from jope_flywheel.program_io import need
from jope_flywheel.protocol import digest

@dataclass
class NativeTensors:
 tokens:torch.Tensor
 lengths:torch.Tensor
 legal:torch.Tensor
 ctx_z:torch.Tensor
 seq_ctx:torch.Tensor
 ce:torch.Tensor
 quality:torch.Tensor
 pairwise:torch.Tensor
 feas:torch.Tensor
 runtime:torch.Tensor
 q_opt:torch.Tensor
 q_grid:torch.Tensor
 midrank_pct:torch.Tensor
 feas_class:torch.Tensor
 rt_target:torch.Tensor
 rt_censored:torch.Tensor
 context_ids:list
 member_ids:list
 receipt:dict
 def to(self,device):return replace(self,**{f.name:getattr(self,f.name).to(device)for f in fields(self)if isinstance(getattr(self,f.name),torch.Tensor)})

def collate(native,context_ids,*,legal_mask_fn,max_seq=128,normalizer=None):
 need(context_ids and len(context_ids)==len(set(context_ids)),'unique ordered context batch')
 records=[];members=[];seqctx=[];vectors=[];percentiles=[]
 for ci,cid in enumerate(context_ids):
  need(cid in native.contexts and cid in native.encodings and cid in native.records,'complete selected context')
  z=np.asarray(native.encodings[cid],dtype=np.float32);need(z.shape==(2051,)and np.isfinite(z).all(),'KumoS2051 finite only')
  if normalizer is not None:z=normalizer(cid,z)
  need(z.shape==(2051,)and np.isfinite(z).all(),'qualified preprocessing');vectors.append(z)
  rows=list(native.records[cid]);need(rows and len(rows)<=256,'bounded selected program rows')
  need(len({r.program_sha256 for r in rows})==len(rows),'no duplicated teacher/program rows')
  known=[r for r in rows if r.quality_eligible];pcts=dict(zip([r.program_sha256 for r in known],L.midrank_pcts([r.q_grid for r in known])))
  for r in rows:
   need(type(r.tokens)in(list,tuple)and 2<=len(r.tokens)<=max_seq and r.tokens[0]==1 and r.tokens[-1]==2 and all(type(t)is int and 0<=t<282 for t in r.tokens),'native vocabulary and bounded complete program')
   need(all(type(getattr(r,k))is bool for k in('ce_eligible','quality_eligible','pairwise_eligible','runtime_eligible','runtime_censored')),'typed support masks')
   need(not r.quality_eligible or(r.q is not None and r.q_opt is not None and math.isfinite(r.q)and math.isfinite(r.q_opt)and type(r.q_grid)is int),'measured quality required, never fitness substitution')
   need(not r.pairwise_eligible or r.quality_eligible,'pairwise requires actual quality')
   need(r.feasibility is None or(type(r.feasibility)is int and r.feasibility in(0,1,2)),'known native feasibility class orNone')
   need(not r.runtime_eligible or(r.runtime_seconds is not None and math.isfinite(r.runtime_seconds)and r.runtime_seconds>=0),'measured runtime or right-censor bound')
   need(r.ce_eligible or r.quality_eligible or r.feasibility is not None or r.runtime_eligible,'at least one explicit supported target')
   records.append(r);members.append((cid,r.program_sha256));seqctx.append(ci);percentiles.append(pcts.get(r.program_sha256,0.))
 n=len(records);width=max(len(r.tokens)for r in records);tokens=np.zeros((n,width),dtype=np.int64);legal=np.zeros((n,width-1,282),dtype=bool)
 for i,r in enumerate(records):
  tokens[i,:len(r.tokens)]=r.tokens;mask=np.asarray(legal_mask_fn(tuple(r.tokens)));need(mask.dtype==bool and mask.shape==(len(r.tokens)-1,282),'exact grammar mask')
  need(mask[np.arange(len(r.tokens)-1),r.tokens[1:]].all(),'every teacher token legal');legal[i,:len(r.tokens)-1]=mask
 flag=lambda k:torch.tensor([bool(getattr(r,k))for r in records],dtype=torch.bool)
 receipt={'schema':'jope.flywheel-native-tensor-batch/v1','context_ids':list(context_ids),'member_ids':members,'selection':native.selection_receipt,'q_missing_is_masked':True,'synthetic_y_created':False,'sequences':n,'padded_tokens':n*width,'real_tokens':sum(len(r.tokens)-1 for r in records),'legal_bytes':legal.nbytes}
 return NativeTensors(torch.from_numpy(tokens),torch.tensor([len(r.tokens)for r in records]),torch.from_numpy(legal),torch.tensor(np.stack(vectors)),torch.tensor(seqctx),flag('ce_eligible'),flag('quality_eligible'),flag('pairwise_eligible'),torch.tensor([r.feasibility is not None for r in records]),flag('runtime_eligible'),torch.tensor([r.q_opt if r.quality_eligible else 0. for r in records]),torch.tensor([r.q_grid if r.quality_eligible else 0 for r in records]),torch.tensor(percentiles),torch.tensor([r.feasibility if r.feasibility is not None else 0 for r in records]),torch.tensor([math.log1p(r.runtime_seconds)if r.runtime_eligible else 0. for r in records]),flag('runtime_censored'),list(context_ids),members,receipt)

OBJECTIVE={'schema':'jope.flywheel-native-support-objective/v1','name':'native-support-mask-v1','coefficients':dict(O.COEF_SUCCESSOR),'ce':'uniform mean of grammar-masked sequence CE over actual ce_eligible rows, then mean of supported contexts','q':'native q_opt Huber delta1; per-context mean then supported-context mean','bt':'native min(abs(dq_opt),1) weighted softplus, boost2if max native midrank>=.9; distinct q_grid only, then supported-context mean','list':'native midrank targets and list tau, at least3 pairwise-eligible measured programs','feas':'three-class CE only where recorded; context mean then supported-context mean','rt':'Huber log1p measured seconds; right-censored squared ReLU(target-prediction); supported-context mean','missing':'connected zero, numerator0, denominator0; no pseudo labels','context_weight':'one per supported context; no claim of native hierarchical sampling','group_claim':'explicit sparse support, not native four-teacher eligibility'}

def objective(out,b):
 import torch.nn.functional as F
 c=len(b.context_ids);member=torch.arange(c,device=b.seq_ctx.device).unsqueeze(1)==b.seq_ctx.unsqueeze(0);q=out.q_hat.float();rt=out.rt_hat.float()
 # Finite consumed values only; all zero-support heads remain graph-connected.
 zero=(torch.nan_to_num(out.logits).sum()+torch.nan_to_num(q).sum()+torch.nan_to_num(rt).sum()+torch.nan_to_num(out.feas_logits).sum())*0.;parts={};support={}
 need(torch.isfinite(q[b.quality|b.pairwise]).all().item()and torch.isfinite(rt[b.runtime]).all().item()and torch.isfinite(out.feas_logits[b.feas]).all().item(),'finite consumed heads')
 q=torch.where(b.quality|b.pairwise,q,torch.zeros_like(q));rt=torch.where(b.runtime,rt,torch.zeros_like(rt))
 def reduce(name,mask,values):
  count=(member&mask.unsqueeze(0)).sum(1);has=count>0;per=torch.where(member&mask.unsqueeze(0),values.unsqueeze(0),torch.zeros((),device=q.device)).sum(1)/count.clamp(min=1)
  numerator=per[has].sum()if has.any()else zero;den=int(has.sum());need(torch.isfinite(numerator).item(),'finite supported '+name);parts[name]=(numerator,den);support[name+'_members']=int(mask.sum());support[name+'_contexts']=den
 if b.ce.any():
  seq=O.sequence_ce(torch.where(b.ce[:,None,None],out.logits,torch.zeros_like(out.logits)),b.tokens[:,1:],b.legal,O.valid_positions(b.lengths,b.tokens.shape[1]));reduce('ce',b.ce,seq)
 else:parts['ce']=(zero,0);support.update(ce_members=0,ce_contexts=0)
 reduce('q',b.quality,F.huber_loss(q,b.q_opt,reduction='none',delta=1.))
 reduce('feas',b.feas,F.cross_entropy(torch.where(b.feas[:,None],out.feas_logits.float(),torch.zeros_like(out.feas_logits.float())),b.feas_class,reduction='none'))
 perrt=torch.where(b.rt_censored,torch.relu(b.rt_target-rt)**2,F.huber_loss(rt,b.rt_target,reduction='none',delta=1.));reduce('rt',b.runtime,perrt)
 same=b.seq_ctx[:,None]==b.seq_ctx[None,:];pair=same&b.pairwise[:,None]&b.pairwise[None,:]&(b.q_grid[:,None]>b.q_grid[None,:])
 weights=(b.q_opt[:,None]-b.q_opt[None,:]).abs().clamp(max=1.)*torch.where(torch.maximum(b.midrank_pct[:,None],b.midrank_pct[None,:])>=.9,2.,1.);weights=torch.where(pair,weights,torch.zeros_like(weights));losses=F.softplus(q[None,:]-q[:,None]);num=O._ctx_sum(member,(weights*losses).sum(1));den=O._ctx_sum(member,weights.sum(1));has=den>0
 parts['bt']=((num[has]/den[has]).sum()if has.any()else zero,int(has.sum()));support.update(bt_pairs=int(pair.sum()),bt_contexts=int(has.sum()))
 known=member&b.pairwise.unsqueeze(0);has=known.sum(1)>=3
 if has.any():
  values=O._list_term(known&has.unsqueeze(1),has,b.midrank_pct,q);parts['list']=(values*has.sum(),int(has.sum()))
 else:parts['list']=(zero,0)
 support['list_contexts']=int(has.sum());terms={k:(n/d if d else zero)for k,(n,d)in parts.items()};loss=sum(OBJECTIVE['coefficients'][k]*v for k,v in terms.items());need(torch.isfinite(loss).item(),'finite supported objective')
 return loss,{'objective':OBJECTIVE['name'],'identity_sha256':digest(OBJECTIVE),'terms':{k:float(v.detach())for k,v in terms.items()},'support':support,'numerators':{k:float(n.detach())for k,(n,d)in parts.items()},'denominators':{k:d for k,(n,d)in parts.items()}},parts
