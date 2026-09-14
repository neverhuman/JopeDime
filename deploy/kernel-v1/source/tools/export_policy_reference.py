#!/usr/bin/env python3
"""One-time parity fixture export from the preserved Python policy reference.

This tool never reads datasets or accepted model weights. It creates fresh seeded
reference parameters, exports them losslessly, and records deterministic forwards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_source))
    from jope.policy.model_v3 import JopePolicyModelV3
    from jope.policy.losses import masked_sequence_nll, self_play_bt_loss, rank_pct_listwise_terms
    torch.set_num_threads(2)
    torch.manual_seed(20260907)
    config = dict(dataset_dim=1536,target_dim=512,target_keep_dim=512,width=512,layers=6,heads=4,ffn=2048,vocab_size=37,embedding_rows=293,max_seq=128,dropout=0.1,dataset_rms=1.,target_rms=1.)
    model = JopePolicyModelV3(SimpleNamespace(size=config['vocab_size']), dataset_dim=1536,target_dim=512,d_model=512,n_layers=6,n_heads=4,ffn_dim=2048,dropout=.1,target_keep_dim=512,split_reward_context=True,input_block_norm=True,dataset_block_rms=1.,target_block_rms=1.).eval()
    dataset=torch.randn(3,1536);target=torch.randn(3,512)
    tokens=torch.randint(3,37,(3,9));tokens[:,0]=1;tokens[:,-1]=2;tokens[1,6:]=0
    pad=tokens.ne(0)
    with torch.no_grad():
        z=model.context(dataset,target);rz=model.reward_context(dataset,target)
        fixtures=dict(dataset=dataset,target=target,tokens=tokens,pad=pad,logits=model._full_logits(z,tokens,pad),reward=model.reward_all(rz,tokens,pad),runtime=model.runtime_pred(rz,tokens,pad),stability=model.instability_logit(tokens,pad,rz),ablation=model.reward_ablation(tokens,pad))
        legal=torch.ones(3,8,37,dtype=torch.bool)
        legal[:,:,3:10]=False
        legal.scatter_(-1,tokens[:,1:].unsqueeze(-1),True)
        weights=torch.tensor([1.,.2,.5])
        fixtures.update(legal=legal,seq_weights=weights,seq_nll=masked_sequence_nll(fixtures['logits'][:,1:9],tokens[:,1:],legal,pad[:,1:],row_weights=weights))
        reward=fixtures['reward'].mean(-1)
        values=torch.tensor([.2,.8,-.1]);pct=torch.tensor([.5,1.,0.]);groups=['same-cut']*3
        fixtures.update(ranking_values=values,ranking_percentiles=pct,bt=self_play_bt_loss(reward,values,groups,rank_pct=pct),listwise=rank_pct_listwise_terms(reward,pct,groups,tau=.25)[0])
    param=torch.nn.Parameter(torch.tensor([.2,-.1,.7,-.3]));fixtures['adam_initial']=param.detach().clone()
    opt=torch.optim.AdamW([param],lr=.003,weight_decay=.01,foreach=False)
    grads=torch.arange(20,dtype=torch.float32).view(5,4)*.03-.2
    for grad in grads:
        param.grad=grad.clone();opt.step();opt.zero_grad()
    fixtures.update(adam_gradients=grads,adam_after_5=param.detach().clone(),adam_m=opt.state[param]['exp_avg'].clone(),adam_v=opt.state[param]['exp_avg_sq'].clone())
    args.output.mkdir(parents=True,exist_ok=False)
    save_file({k:v.contiguous() for k,v in model.state_dict().items()}, str(args.output/'reference.safetensors'))
    save_file({k:v.contiguous() for k,v in fixtures.items()}, str(args.output/'fixtures.safetensors'))
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    source=args.reference_source/'jope/policy/model_v3.py'
    (args.output/'reference.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch=torch.__version__,seed=20260907,fresh_parameters=True),indent=2)+'\n')

if __name__=='__main__': main()
