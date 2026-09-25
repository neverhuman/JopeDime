#!/usr/bin/env python3
"""Verify the accepted safetensors header against the documented native tensor geometry."""
import hashlib,json,math,re,struct
from pathlib import Path
root=Path(__file__).resolve().parents[1];repo=root.parents[2]
record=json.loads((root/'data/tensors.json').read_text());path=Path(record['path'])
raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==record['sha256']
n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);header.pop('__metadata__',None)
actual={name:entry['shape'] for name,entry in header.items()}
assert all(entry['dtype']=='F32' for entry in header.values())
expected={};w=512
def dense(name,i,o):expected[name+'.weight']=[o,i];expected[name+'.bias']=[o]
def norm(name):expected[name+'.weight']=[w];expected[name+'.bias']=[w]
for name in ('encoder','reward_encoder'):
 dense(name+'.0',2048,w);norm(name+'.2');dense(name+'.4',w,w);norm(name+'.6')
dense('ctx_proj',w,w);expected['token_emb.weight']=[657,w];expected['pos_emb.weight']=[129,w]
for i in range(6):
 p=f'decoder.layers.{i}';expected[p+'.self_attn.in_proj_weight']=[3*w,w];expected[p+'.self_attn.in_proj_bias']=[3*w];dense(p+'.self_attn.out_proj',w,w);dense(p+'.linear1',w,2048);dense(p+'.linear2',2048,w);norm(p+'.norm1');norm(p+'.norm2')
norm('out_norm');expected['reward_token_emb.weight']=[657,w];norm('reward_norm');dense('z_gamma.0',w,w);dense('z_proj',w,w)
for i in range(4):dense(f'reward_heads.{i}.0',3*w,w);dense(f'reward_heads.{i}.3',w,1)
for name,inputs,index in [('runtime_head',2*w,2),('stability_head',3*w,2),('reward_head_ablation',w,3)]:dense(name+'.0',inputs,w);dense(name+f'.{index}',w,1)
source=(repo/'crates/jope-policy/src/model.rs').read_text();block=source.split('const LOSS_LOG_VARS:',1)[1].split('];',1)[0]
for name in re.findall(r'"([a-z_]+)"',block):expected['loss_log_vars.'+name]=[]
assert actual==expected, 'header differs from documented expected_inventory geometry'
assert {t['name']:t['shape'] for t in record['tensors']}==actual
assert sum(math.prod(shape) for shape in actual.values())==record['parameters']==27796494
assert len(actual)==record['tensor_count']==136
print(json.dumps({'tensor_count':136,'parameters':27796494,'dtype':'F32','asset_sha256':record['sha256'],'all_names_and_shapes_match':True,'scope':'documented mirror of model.rs expected_inventory, not a forward-pass qualification'}))
