#!/usr/bin/env python3
"""Export the frozen Starlight hidden encoder with dynamic dataset shapes.
The wrapper removes chunking control flow and the unused prediction head. Its
output is checked against the original forward for every admitted cut shape.
"""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn

class HiddenGraph(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,y):
        m=self.model;n=y.shape[0];b=x.shape[1]
        grouped,ycol,_=m._preprocess_and_group(x,y,n,None)
        embedded=m.x_embed(grouped);embedded[:,:n]=embedded[:,:n]+ycol.unsqueeze(2)
        embedded,_=m.feature_distribution_embedder(x_BRiCE=embedded,num_train_rows=n,cached_hidden=None,save_peak_memory_factor=None,force_recompute_layer=False,return_hidden=False)
        row=m.column_aggregator(x_BRiCE=embedded,save_peak_memory_factor=None,force_recompute_layer=False).flatten(-2)
        row[:,:n]=row[:,:n]+m._embed_icl_y(m._prepare_y(y,n,b))
        for block in m.icl_blocks:row,_=block(row,n,None)
        hidden=m.output_norm(row)[:,n:].float()
        return torch.cat([hidden.mean(1),hidden.std(1,unbiased=False),torch.quantile(hidden,.1,dim=1),torch.quantile(hidden,.9,dim=1)],1)

def main():
    r=Path(__file__).resolve().parents[1];src=r/'assets/source/fable_god';sys.path.insert(0,str(src));os.environ['FABLE_REGRESSOR_SOURCE']=str(src/'provenance/regressor_source')
    from python.elo_finetune.model_io import load_regressor
    from python.elo_finetune.paths import import_regressor_train
    torch._C._set_graph_executor_optimize(False)
    # The original helper leaves traced shape scalars on CPU. Move that scalar
    # explicitly to the query device; eager behavior is numerically identical.
    def safe_log(n,device,dtype):
        if isinstance(n,torch.Tensor):return n.to(device=device,dtype=torch.float32).clamp(min=1).log().to(dtype)
        return (torch.ones((),device=device,dtype=torch.float32)*n).clamp(min=1).log().to(dtype)
    import_regressor_train()._safe_log_seqlen=safe_log
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=r/'assets/models/starlight';model=load_regressor(weights/'starlight_core_regressor.safetensors',weights/'starlight_config.json',device='cuda')
    for m in model.modules():
        if hasattr(m,'num_kv_heads_test'):m.num_kv_heads_test=None
    graph=HiddenGraph(model).eval();out=r/'assets/foundation-export-v1';out.mkdir(parents=True,exist_ok=True)
    recipes={};shapes=set()
    for p in (r/'data/prepared/views-expanded-20').glob('*/prepared.json'):
        m=json.loads(p.read_text());width=len(m['storage']['feature_columns'])
        for c in m['cuts']:
            n=len(c['train_indices']);idx=np.random.default_rng(20260707).permutation(n);nc=min(768,max(8,3*n//4));nq=min(256,n-nc);ctx=idx[:nc];qry=idx[max(0,n-8):] if nq<4 else idx[nc:nc+nq];recipes[str(n)]=dict(context=ctx.tolist(),query=qry.tolist());shapes.add((len(ctx),len(qry),width))
    (out/'sampling.json').write_text(json.dumps(dict(schema='parallel.foundation_sampling.v1',seed=20260707,numpy_version=np.__version__,recipes=recipes),indent=2)+'\n')
    shapes=sorted(shapes);torch.manual_seed(42);inputs=(torch.rand(80,1,8,device='cuda'),torch.rand(60,1,device='cuda'))
    with torch.no_grad():
        traced=torch.jit.trace(graph,inputs,check_trace=False);traced.save(str(out/'foundation.pt'))
        results=[]
        for nc,nq,width in shapes:
            # Batch tests cover independent contexts sharing the batch dimension.
            for batch in [1,2]:
                x=torch.rand(nc+nq,batch,width,device='cuda');y=torch.rand(nc,batch,device='cuda')
                ref=model(x,y,only_return_standard_out=False,performance_options=import_regressor_train().PerformanceOptions(use_chunkwise_inference=False))['test_embeddings'].float().transpose(0,1)
                expected=torch.cat([ref.mean(1),ref.std(1,unbiased=False),torch.quantile(ref,.1,dim=1),torch.quantile(ref,.9,dim=1)],1)
                actual=traced(x,y);error=float((actual-expected).abs().max());results.append(dict(context=nc,query=nq,width=width,batch=batch,max_abs_error=error,passed=error<3e-4));print(json.dumps(results[-1]),flush=True)
        report=dict(schema='parallel.foundation_export.v1',module_sha256=hashlib.sha256((out/'foundation.pt').read_bytes()).hexdigest(),sampling_sha256=hashlib.sha256((out/'sampling.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256((src/'provenance/regressor_source/starlight_core_regressor_train.py').read_bytes()).hexdigest(),weights_sha256=hashlib.sha256((weights/'starlight_core_regressor.safetensors').read_bytes()).hexdigest(),export_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numerical_backend='LibTorch 2.7 CUDA sm80 FP32',tolerance=3e-4,passed=all(v['passed'] for v in results),output_dimensions=2048,tests=results)
        (out/'parity.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed']
if __name__=='__main__':main()
