#!/usr/bin/env python3
"""One-time frozen target bundle conversion and numerical reference checks.
The saved graph contains all statistics and both frozen neural encoders. Runtime
sampling uses exported row-index recipes; no Python is needed during encoding.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class WindowStats(nn.Module):
    def forward(self, values, lengths, task):
        x=values.double().clamp(0,1); sx=x.sort(1).values; n=32
        feats=[]
        def add(v):
            if not torch.is_tensor(v): v=torch.full_like(x[:,0],v)
            feats.append(v)
        def div(a,b): return torch.where(torch.abs(b)>1e-12,a/b,torch.zeros_like(a))
        def q(v,p): return torch.quantile(v,p,dim=1)
        def entropy(p): return -(torch.where(p>0,p*torch.log(p.clamp_min(1e-300)),torch.zeros_like(p))).sum(1)/math.log(max(2,p.shape[1]))
        mean=x.mean(1); var=x.var(1,unbiased=False); std=var.sqrt(); centered=x-mean[:,None]
        median=q(x,.5); mad=q((x-median[:,None]).abs(),.5)
        add((task==0).double()); add((task!=0).double()); add(torch.log1p(lengths.double())); add(torch.log1p(lengths.double().clamp_max(n))); add((1-lengths.double()/n).clamp_min(0)); add(float(n));add(1.)
        for v in [mean,std,var,std/math.sqrt(n),sx[:,0],sx[:,-1],sx[:,-1]-sx[:,0],median,mean-median,mad,(x-mean[:,None]).abs().mean(1),div(std,mean.abs()+1e-6),torch.log((mean+1e-6)/(1-mean+1e-6))]: add(v)
        for power in range(2,13):add((x**power).mean(1))
        for power in range(2,11):
            central=(centered**power).mean(1);add(central)
            if power>=3:add(div(central,(std+1e-6)**power))
        probs=[0.,.001,.005,.01,.02,.05,.1,.125,.2,.25,.333,.375,.5,.625,.667,.75,.8,.875,.9,.95,.98,.99,.995,.999,1.]
        quant={p:q(x,p) for p in probs}
        for p in probs:add(quant[p])
        for l,r in [(.25,.75),(.1,.9),(.05,.95),(.01,.99),(.001,.999)]:add(quant[r]-quant[l])
        for l,r in zip(probs,probs[1:]):add(quant[r]-quant[l])
        iqr=quant[.75]-quant[.25]
        for v in [iqr,div(quant[.75]+quant[.25]-2*quant[.5],iqr+1e-6),div(quant[.9]+quant[.1]-2*quant[.5],quant[.9]-quant[.1]+1e-6),div(quant[.875]-quant[.625]+quant[.375]-quant[.125],iqr+1e-6)]:add(v)
        for trim in [.01,.05,.1,.2,.25]:
            lo=int(n*trim);add(sx[:,lo:n-lo].mean(1));add(torch.minimum(torch.maximum(x,q(x,trim)[:,None]),q(x,1-trim)[:,None]).mean(1))
        for threshold in [0.,.001,.005,.01,.02,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,.99,.995,.999,1.]:add((x<=threshold).double().mean(1));add((x>=threshold).double().mean(1))
        for cond in [x<=1e-12,x>=1-1e-12,(x>=.4)&(x<=.6),(x<=.05)|(x>=.95)]:add(cond.double().mean(1))
        hist64=None
        for bins in [4,8,16,32,64]:
            indices=(x*bins).long().clamp_max(bins-1)
            hist=F.one_hot(indices,bins).double().mean(1)
            for v in [entropy(hist),1-(hist*hist).sum(1),hist.max(1).values,(hist<=0).double().mean(1)]:add(v)
            for i in range(bins):add(hist[:,i])
            if bins==64:hist64=hist
        fft=torch.fft.rfft(hist64-hist64.mean(1,keepdim=True),dim=1)
        for i in range(1,17):add(fft[:,i].real);add(fft[:,i].imag);add(fft[:,i].abs())
        gaps=torch.diff(torch.cat([torch.zeros_like(sx[:,:1]),sx,torch.ones_like(sx[:,:1])],1),dim=1)
        for v in [gaps.mean(1),gaps.std(1,unbiased=False),gaps.min(1).values,gaps.max(1).values,sx[:,0],1-sx[:,-1]]:add(v)
        for p in [.1,.25,.5,.75,.9,.99]:add(q(gaps,p))
        idx=torch.arange(1,n+1,device=x.device,dtype=torch.float64)[None,:]
        total=x.sum(1);add(torch.where(total>1e-12,2*(idx*sx).sum(1)/(n*total)-(n+1.)/n,torch.zeros_like(total)))
        positive=x.clamp_min(1e-12);pm=positive.mean(1);ratio=positive/pm[:,None];add(torch.where(pm>1e-12,(ratio*ratio.log()).mean(1),torch.zeros_like(pm)))
        add((x[:,:,None]-x[:,None,:]).abs().mean((1,2)))
        add(torch.maximum((idx/n-sx).abs(),(sx-(idx-1)/n).abs()).max(1).values)
        add(1./(12*n)+((sx-(2*idx-1)/(2*n))**2).sum(1));add((sx-(idx-.5)/n).abs().mean(1))
        for decimals in [0,1,2,3,4,6]:
            rounded=torch.round(x*(10**decimals))/(10**decimals); count=(rounded[:,:,None]==rounded[:,None,:]).double().sum(2)
            add((1/count).sum(1)/n);add(count.max(1).values/n)
            unique=(1/count).sum(1); ent=-(torch.log(count/n)/n).sum(1)/torch.log(unique.clamp_min(2));add(ent)
        y=2*x-1;p0=torch.ones_like(y);p1=y
        for degree in range(1,13):
            if degree==1:v=p1
            else:v=((2*degree-1)*y*p1-(degree-1)*p0)/degree;p0,p1=p1,v
            add(v.mean(1))
        for freq in range(1,17):
            angles=2*math.pi*freq*x;s=angles.sin().mean(1);c=angles.cos().mean(1);add(s);add(c);add(torch.hypot(s,c))
        for bw in [.025,.05,.1,.2]:
            for i in range(33):add(torch.exp(-((x-i/32)**2)/(2*bw*bw)).mean(1))
        rate=(x>=.5).double().mean(1);be=-(rate*rate.clamp_min(1e-300).log()+(1-rate)*(1-rate).clamp_min(1e-300).log())/math.log(2)
        add(rate);add(1-(2*rate-1).abs());add(be);add(rate*(1-rate))
        return torch.nan_to_num(torch.stack(feats,1).float(),nan=0.,posinf=0.,neginf=0.)

class FusedGraph(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.best=encoder.best_model;self.final=encoder.final_model;self.stats=WindowStats()
        self.register_buffer('center',torch.from_numpy(encoder.vectorizer['scaler'].mean_.copy()))
        self.register_buffer('scale',torch.from_numpy(encoder.vectorizer['scaler'].scale_.copy()))
        self.register_buffer('projection',torch.from_numpy(encoder.vectorizer['reducer'].components_.copy()))
        self.weights=encoder.vector_weights
    def forward(self, classical, neural, lengths, task):
        # Exactly two windows per component. A sole neural window is repeated;
        # its normalized mean is unchanged by repetition.
        batch=classical.shape[0]; lens=lengths[:,None].expand(-1,2).reshape(-1);tasks=task[:,None].expand(-1,2).reshape(-1)
        stats=self.stats(classical.reshape(-1,32),lens,tasks).reshape(batch,2,-1)
        signature=torch.cat([stats.mean(1),stats.std(1,unbiased=False),stats.min(1).values,stats.max(1).values],1)
        scaled=(signature.double()-self.center).float();scaled=(scaled.double()/self.scale).float();component=scaled @ self.projection.t();component=F.pad(component,(0,384-component.shape[1]));component=F.normalize(component,dim=1)
        values=neural.reshape(-1,32);effective=lens.clamp_max(32);replacement=(1-lens/32).clamp_min(0)
        best=self.best(values,tasks,lens,effective,mask=torch.ones_like(values),replacement_fraction=replacement,return_aux=False).reshape(batch,2,-1).mean(1)
        final=self.final(values,tasks,lens,effective,mask=torch.ones_like(values),replacement_fraction=replacement,return_aux=False).reshape(batch,2,-1).mean(1)
        return F.normalize(torch.cat([F.normalize(best,dim=1)*self.weights[0],F.normalize(final,dim=1)*self.weights[1],component*self.weights[2]],1),dim=1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.source))
    from target_maker.classical import summarize_target_window
    from target_maker.data import sample_indices, stable_seed
    from target_maker.encode import _window_indices
    from target_maker.fused import FusedTargetEncoder
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    enc=FusedTargetEncoder(a.bundle,device='cpu');assert enc.window_size==32 and enc.signature_views==2 and not enc.include_cache_profile
    graph=FusedGraph(enc).eval();a.output.mkdir(parents=True,exist_ok=True)
    lengths={7,32,48,64,91,725}
    for p in a.prepared.glob('*/prepared.json'):
        m=json.loads(p.read_text());lengths.update(len(c['train_indices']) for c in m['cuts'])
    recipes={}
    for task in [0,1]:
        for n in sorted(lengths):
            classical=[sample_indices(n,32,np.random.default_rng(stable_seed('fusion-signature-inference',17,task,n,i,32)))[0].tolist() for i in range(2)]
            neural=[x.tolist() for x in _window_indices(n,32,32,2,17)];neural=(neural+neural)[:2]
            recipes[f'{task}:{n}']=dict(classical=classical,neural=neural)
    (a.output/'sampling.json').write_text(json.dumps(dict(schema='parallel.target_sampling.v1',seed=17,numpy_version=np.__version__,window=32,views=2,recipes=recipes),indent=2)+'\n')
    cases=[]; inputs=[]; refs=[];stats_error=0.
    for n in [7,32,48,64,91,725]:
        for task in [0,1]:
            for kind in ['random','constant','endpoints']:
                arr=np.random.default_rng(n).random(n).astype('float32') if kind=='random' else (np.full(n,.5,dtype='float32') if kind=='constant' else (np.arange(n)%2).astype('float32'))
                if task==0:arr=(arr>=.5).astype('float32')
                recipe=recipes[f'{task}:{n}'];c=arr[recipe['classical']];v=arr[recipe['neural']]
                actual=graph.stats(torch.from_numpy(c),torch.tensor([n,n]),torch.tensor([task,task])).detach().numpy()
                expected=np.stack([summarize_target_window(row,task=task,original_length=n,effective_count=min(n,32),replacement_fraction=max(0,1-n/32),window_size=32)[0] for row in c]);stats_error=max(stats_error,float(np.abs(actual-expected).max()))
                inputs.append((c,v,n,task));refs.append(enc.encode_values(arr,task));cases.append(dict(n=n,task=task,kind=kind))
    tensors=(torch.from_numpy(np.stack([i[0] for i in inputs])),torch.from_numpy(np.stack([i[1] for i in inputs])),torch.tensor([i[2] for i in inputs],dtype=torch.float32),torch.tensor([i[3] for i in inputs]))
    with torch.no_grad():
        traced=torch.jit.trace(graph,tuple(x[:2] for x in tensors),check_trace=False);traced=torch.jit.freeze(traced);traced.save(str(a.output/'target.pt'))
        out=traced(*tensors).numpy();diff=np.abs(out-np.stack(refs));dynamic=[float(np.abs(traced(*(x[:n] for x in tensors)).numpy()-out[:n]).max()) for n in [1,7,36]]
    sha=hashlib.sha256((a.output/'target.pt').read_bytes()).hexdigest();report=dict(schema='parallel.target_export.v1',module_sha256=sha,stats_max_abs_error=stats_error,max_abs_error=float(diff.max()),mean_abs_error=float(diff.mean()),dynamic_batch_errors=dynamic,reference_cases=cases,tolerance=3e-4,passed=bool(diff.max()<3e-4 and stats_error<3e-4),sampling_sha256=hashlib.sha256((a.output/'sampling.json').read_bytes()).hexdigest(),bundle_manifest_sha256=hashlib.sha256((a.bundle/'manifest.json').read_bytes()).hexdigest(),export_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),output_dimensions=512,backend='LibTorch 2.7 CPU reference; CUDA parity pending')
    (a.output/'parity.json').write_text(json.dumps(report,indent=2)+'\n');np.savez(a.output/'parity-inputs.npz',classical=tensors[0].numpy(),neural=tensors[1].numpy(),lengths=tensors[2].numpy(),task=tensors[3].numpy(),reference=np.stack(refs));print(json.dumps({k:v for k,v in report.items() if k!='reference_cases'}));assert report['passed']
if __name__=='__main__': main()
