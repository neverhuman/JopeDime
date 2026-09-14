#!/usr/bin/env python3
"""Qualify larger independent dataset batches and report measured GPU memory."""
import torch,json,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];torch.set_num_threads(2);torch._C._set_graph_executor_optimize(False);torch.backends.cuda.matmul.allow_tf32=False
a=r/'assets/foundation-export-v1';m=torch.jit.load(str(a/'foundation.pt'),map_location='cuda:0').eval();old=json.loads((a/'parity.json').read_text());shapes=sorted({(x['context'],x['query'],x['width']) for x in old['tests']});tests=[]
with torch.no_grad():
 for nc,nq,width in shapes:
  for batch in [8,32]:
   torch.cuda.reset_peak_memory_stats();x=torch.rand(nc+nq,batch,width,device='cuda');y=torch.rand(nc,batch,device='cuda');ref=torch.cat([m(x[:,i:i+1],y[:,i:i+1]) for i in range(batch)],0);torch.cuda.synchronize();start=time.monotonic();result=m(x,y);torch.cuda.synchronize();seconds=time.monotonic()-start;err=float((ref-result).abs().max());t=dict(context=nc,query=nq,width=width,batch=batch,max_abs_error=err,passed=err<3e-4,seconds=seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30);tests.append(t);print(json.dumps(t),flush=True)
report=dict(schema='parallel.foundation_batch_parity.v1',module_sha256=old['module_sha256'],sampling_sha256=old['sampling_sha256'],base_parity_sha256=__import__('hashlib').sha256((a/'parity.json').read_bytes()).hexdigest(),passed=all(t['passed'] for t in tests),tolerance=3e-4,tests=old['tests']+tests);(a/'batch-parity.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed']
