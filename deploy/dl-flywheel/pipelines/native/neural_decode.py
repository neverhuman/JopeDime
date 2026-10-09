"""Batched autoregressive native-program qualification; no score prediction as reward."""
import torch
from dltrain import dsl_tokenizer as T
from jope_flywheel.program_io import need,tick
from jope_flywheel.protocol import digest

class BatchedTrie:
 def __init__(self,programs,vocab,max_seq):
  need(type(programs)is list and 1<=len(programs)<=100000,'bounded full native program catalog')
  self.vocab,self.max_seq=vocab,max_seq;self.nodes=[{}];self.programs={};self.paths={}
  for program in programs:
   text=program['dsl'];tokens=tuple(T.tokenize(text,vocab))
   from jope_flywheel.program_io import sha
   need(len(tokens)<=max_seq and tokens[0]==T.BOS and tokens[-1]==T.EOS and T.detokenize(tokens,vocab)==text and sha(text.encode())==program['program_sha256'],'exact complete catalog program identity')
   need(tokens not in self.programs,'duplicate catalog program');self.programs[tokens]=program;node=0
   for token in tokens[1:]:
    if token not in self.nodes[node]:self.nodes[node][token]=len(self.nodes);self.nodes.append({})
    node=self.nodes[node][token]
   need(len(self.nodes)<=2000000,'bounded compact catalog trie nodes')
  self.identity=digest({'schema':'jope.flywheel-full-catalog-trie/v1','vocabulary_sha256':vocab.sha256,'max_seq':max_seq,'programs':programs})
  self.next=self
 def get(self,prefix,default=()):
  node=0
  for token in prefix[1:]:
   if token not in self.nodes[node]:return default
   node=self.nodes[node][token]
  return self.nodes[node]

 def decode_many(self,net,ctx_z,*,seeds,temperature,deadline):
  count=ctx_z.shape[0];need(ctx_z.ndim==2 and ctx_z.shape[1]==net.cfg.ctx_dim and 1<=count<=1024 and len(seeds)==count,'bounded context matrix/seeds')
  need(all(type(s)is int and 0<=s<2**63 for s in seeds)and 0.1<=temperature<=2,'fixed decoder parameters')
  generators=[torch.Generator(device='cpu').manual_seed(s)for s in seeds];prefixes=[(T.BOS,)for _ in seeds];traces=[[]for _ in seeds];cursors=[0 for _ in seeds];previous=net.training;net.eval();forwards=0
  try:
   with torch.no_grad():
    while True:
     active=[i for i,p in enumerate(prefixes)if p[-1]!=T.EOS]
     if not active:break
     tick(deadline);allowed=[sorted(self.nodes[cursors[i]])for i in active]
     need(all(a and len(prefixes[i])<self.max_seq for i,a in zip(active,allowed)),'bounded legal terminal prefix')
     lengths=torch.tensor([len(prefixes[i])+1 for i in active],dtype=torch.long,device=ctx_z.device)
     width=int(lengths.max());tokens=torch.full((len(active),width),T.PAD,dtype=torch.long,device=ctx_z.device)
     for row,i in enumerate(active):tokens[row,:len(prefixes[i])]=torch.tensor(prefixes[i],dtype=torch.long,device=ctx_z.device)
     output=net(tokens,lengths,ctx_z,torch.tensor(active,dtype=torch.long,device=ctx_z.device));forwards+=1
     # One transfer per token step across all still-active rows, not per context.
     latest=output.logits[torch.arange(len(active),device=ctx_z.device),lengths-2].detach().double().cpu()/temperature
     for row,(i,choices)in enumerate(zip(active,allowed)):
      logits=latest[row,choices];need(torch.isfinite(logits).all().item(),'finite supported logits');probabilities=torch.softmax(logits,dim=0)
      selected=int(torch.multinomial(probabilities,1,generator=generators[i]).item());token=choices[selected]
      traces[i].append({'position':len(prefixes[i]),'allowed':choices,'chosen':token,'chosen_probability':float(probabilities[selected])});prefixes[i]+=(token,);cursors[i]=self.nodes[cursors[i]][token]
  finally:net.train(previous)
  protocol={'schema':'jope.flywheel-batched-trie-decoder/v1','trie_sha256':self.identity,'batch_rows':count,'temperature':temperature,'device_type':ctx_z.device.type,'floating_point_scope':'Batched kernels may differ from serial rounding; fixed batching protocol and row seeds are bound.'}
  rows=[]
  for i,tokens in enumerate(prefixes):
   need(tokens in self.programs,'native-valid terminal');rows.append({'schema':'jope.flywheel-model-decoded-program/v1','program':self.programs[tokens],'tokens':list(tokens),'trace':traces[i],'decoder_seed':seeds[i],'temperature':temperature,'trie_sha256':self.identity,'model_generated':True,'generation_scope':'Autoregressive model draws constrained to a preregistered finite native-valid program trie.','batch_protocol_sha256':digest(protocol)})
  return rows,{'protocol':protocol,'forward_calls':forwards,'rows':count,'tokens_generated':sum(len(x)-1 for x in prefixes)}
