"""Actual PolicyLM optimization and bounded model decoding, with mandatory custody.

The immutable native sampler, label rules and successor_v1 loss remain intact.
The finite proposal trie is a declared qualification grammar, not unrestricted
topology discovery. Generated measurements are retained for a later explicitly
admitted snapshot; they never silently alter the current frozen teacher corpus.
"""
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import save_file
from dltrain import model as M, sampler as S, objective as O, dsl_tokenizer as T
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.program_io import need,create,bound,sha,tick

class ProposalTrie:
    def __init__(self,programs,vocab,max_seq):
        need(type(programs)is list and 1<=len(programs)<=64,'bounded preregistered native programs')
        self.vocab,self.max_seq=vocab,max_seq;self.next={};self.programs={}
        for program in programs:
            text=program['dsl'];tokens=tuple(T.tokenize(text,vocab))
            need(len(tokens)<=max_seq and tokens[0]==T.BOS and tokens[-1]==T.EOS and
                 T.detokenize(tokens,vocab)==text and sha(text.encode())==program['program_sha256'],'exact trie program/token identity')
            need(tokens not in self.programs,'duplicate trie program');self.programs[tokens]=program
            for i in range(1,len(tokens)):self.next.setdefault(tokens[:i],set()).add(tokens[i])
        self.identity=digest({'schema':'jope.flywheel-bounded-proposal-trie/v1','vocabulary_sha256':vocab.sha256,
                              'max_seq':max_seq,'programs':programs})
    def decode(self,net,ctx_z,*,seed,temperature,deadline):
        need(type(seed)is int and 0<=seed<2**63 and 0.1<=temperature<=2,'fixed sampling parameters')
        need(ctx_z.shape==(1,net.cfg.ctx_dim),'one exact context to decode')
        generator=torch.Generator(device='cpu');generator.manual_seed(seed)
        previous_mode=net.training;net.eval();tokens=(T.BOS,);trace=[]
        try:
            with torch.no_grad():
                while tokens[-1]!=T.EOS:
                    tick(deadline);allowed=sorted(self.next.get(tokens,()))
                    need(allowed and len(tokens)<self.max_seq,'proposal prefix must finish within bound')
                    # Placeholder is to the right of the predicting position.
                    # Causal attention means it cannot influence that prediction.
                    padded=torch.tensor([tokens+(T.PAD,)],dtype=torch.long,device=ctx_z.device)
                    lengths=torch.tensor([len(tokens)+1],dtype=torch.long,device=ctx_z.device)
                    output=net(padded,lengths,ctx_z,torch.zeros(1,dtype=torch.long,device=ctx_z.device))
                    logits=output.logits[0,len(tokens)-1,allowed].detach().double().cpu()/temperature
                    need(torch.isfinite(logits).all().item(),'finite model decoder logits')
                    probabilities=torch.softmax(logits,dim=0)
                    selected=int(torch.multinomial(probabilities,1,generator=generator).item());token=allowed[selected]
                    trace.append({'position':len(tokens),'allowed':allowed,'chosen':token,
                                  'chosen_probability':float(probabilities[selected])})
                    tokens+= (token,)
        finally:net.train(previous_mode)
        need(tokens in self.programs,'decoded terminal is registered native program')
        return {'schema':'jope.flywheel-model-decoded-program/v1','program':self.programs[tokens],
            'tokens':list(tokens),'trace':trace,'decoder_seed':seed,'temperature':temperature,
            'trie_sha256':self.identity,'model_generated':True,
            'generation_scope':'Autoregressive model draws constrained to a preregistered finite native-valid program trie.'}

def normalize_contexts(data):
    """Fit context-vector preprocessing on training contexts only, with receipt."""
    ids=sorted(c for c,r in data.corpus.contexts.items()if r['split_role']=='train_fit')
    need(ids,'training contexts required for preprocessing')
    matrix=np.stack([np.asarray(data.corpus.encodings[c],dtype=np.float64)for c in ids])
    need(matrix.shape[1]==2051 and np.isfinite(matrix).all(),'KumoS-only2051 finite input')
    mean=matrix.mean(axis=0);scale=matrix.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
    original={};transformed={}
    for context,value in data.corpus.encodings.items():
        z=np.asarray(value,dtype=np.float64);need(z.shape==(2051,)and np.isfinite(z).all(),'context representation contract')
        original[context]=sha(z.astype('<f8').tobytes())
        transformed[context]=np.clip((z-mean)/scale,-10.,10.).astype(np.float32)
    receipt={'schema':'jope.flywheel-context-preprocessing/v1','encoder':'kumo_tabular_s','dimension':2051,
        'fit_contexts':ids,'mean':mean.tolist(),'scale':scale.tolist(),'clip':[-10.,10.],
        'original_float64_hashes':original,'validation_used_for_fit':False}
    from dataclasses import replace
    corpus=replace(data.corpus,encodings=transformed)
    normalizer_identity=digest(receipt)
    refs=dict(data.refs or {});refs['normalizers']='flywheel-preprocess:'+normalizer_identity
    receipt['statistics_identity_sha256']=normalizer_identity
    updated=replace(data,corpus=corpus,refs=refs)
    receipt['input_train_data_sha256']=data.content_sha256
    receipt['output_train_data_sha256']=updated.content_sha256
    return updated,receipt

def context_batch(data,context_ids,*,step,run_seed,training):
    need(context_ids and len(context_ids)==len(set(context_ids)),'unique contexts per batch')
    slots=[]
    for slot,context in enumerate(context_ids):
        row=data.corpus.contexts[context]
        need(row['split_role']==('train_fit'if training else'dev'),'optimizer/validation role boundary')
        plan=S.context_plan(data.corpus,context,data.config)
        stream,members,roles=S._members(plan,data.config,run_seed,step,slot)
        need(members,'admitted context must have labelled members')
        indexes={p:i for i,p in enumerate(plan.labelled)};counts=[0,0,0];feas=[0,0,0]
        for member in members:
            i=indexes[member];counts[plan.labelled_classes[i]]+=1
            if plan.labelled_feas_valid[i]:feas[plan.labelled_classes[i]]+=1
        slots.append(S.SlotDraw(slot=slot,family_id=row['family_id'],dataset_hash=row['dataset_hash'],
            context_sha256=context,pi_c=1/len(context_ids),stream=stream,members=members,roles=roles,
            class_counts=tuple(counts),feas_class_counts=tuple(feas)))
    draw=S.StepDraw(step,tuple(slots));batch=S.collate(draw,data)
    need(batch.ctx_ids==list(context_ids),'actual tensor context order matches reserved batch')
    journal=draw.journal_record()
    journal['selection']='exhaustive distinct-lineage BatchStream; not native hierarchical random draw'
    journal['pi_c_semantics']='legacy field records 1 / current batch context count for equal averaging; not a sampling inclusion probability'
    return batch,journal

def to_device(batch,device):
    from dataclasses import fields,replace
    return replace(batch,**{f.name:getattr(batch,f.name).to(device)for f in fields(batch)
                             if isinstance(getattr(batch,f.name),torch.Tensor)})

def objective(net,batch):
    output=net(batch.tokens,batch.lengths,batch.ctx_z,batch.seq_ctx)
    return O.successor_v1(output,batch)

def evaluation(net,batch):
    previous=net.training;net.eval()
    try:
        with torch.no_grad():result=objective(net,batch)
    finally:net.train(previous)
    return {'loss':float(result.loss.detach().cpu()),'terms':result.terms,'weighted':result.weighted,'support':result.support,
            'objective':'successor-v1','dropout':False,'teacher_members_frozen':True,
            'evaluated_context_ids':list(batch.ctx_ids),'member_ids_sha256':digest([list(x)for x in batch.member_ids])}

def optimizer_update(net,opt,batch):
    before=M.parameter_sha256(net);net.train();opt.zero_grad(set_to_none=True)
    result=objective(net,batch);need(torch.isfinite(result.loss).item(),'finite actual loss')
    result.loss.backward()
    need(any(p.grad is not None for p in net.parameters()),'real backward gradients')
    need(all(p.grad is None or torch.isfinite(p.grad).all().item()for p in net.parameters()),'finite gradients')
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);need(torch.isfinite(norm).item(),'finite gradient norm')
    opt.step();after=M.parameter_sha256(net);need(before!=after,'actual model parameters must change')
    return {'checkpoint_before':before,'checkpoint_after':after,'loss':float(result.loss.detach().cpu()),
            'terms':result.terms,'weighted':result.weighted,'support':result.support,'gradient_norm_before_clip':float(norm),'clip_norm':1.}

def checkpoint(net,opt,path,*,step,config_sha256):
    """Create-only safetensors, including named AdamW moments; no pickle."""
    path=Path(path);path.mkdir(exist_ok=False);params=dict(net.named_parameters())
    tensors={'model.'+name:p.detach().cpu().contiguous()for name,p in params.items()}
    for name,p in params.items():
        for key,value in opt.state.get(p,{}).items():
            need(isinstance(value,torch.Tensor),'optimizer state is numeric tensor only')
            tensors['optimizer.'+name+'.'+key]=value.detach().cpu().contiguous()
    tensors['rng.torch_cpu']=torch.get_rng_state()
    if next(net.parameters()).is_cuda:
        for index,state in enumerate(torch.cuda.get_rng_state_all()):tensors['rng.torch_cuda.'+str(index)]=state
    save_file(tensors,str(path/'state.safetensors'))
    with (path/'state.safetensors').open('rb')as stream:os.fsync(stream.fileno())
    body=(path/'state.safetensors').read_bytes()
    manifest={'schema':'jope.flywheel-neural-checkpoint/v1','step':step,'config_sha256':config_sha256,
        'parameter_sha256':M.parameter_sha256(net),'state_sha256':sha(body),'state_bytes':len(body),
        'model_config':net.cfg.to_json(),'parameters':M.count_parameters(net),
        'optimizer':'AdamW','learning_rate':opt.param_groups[0]['lr'],'weight_decay':0.01,
        'resume':'No automatic replay. Verify checkpoint, planner reservation, and GP journal before explicit recovery.'}
    create(path/'manifest.json',canonical(manifest));return manifest
