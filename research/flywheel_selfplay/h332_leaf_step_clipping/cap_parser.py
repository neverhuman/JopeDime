"""Read archived native ranker dumps; no model fitting or predictions."""
import re
import numpy as np

def raw_leaf_magnitudes(blob):
 text=blob.decode();params=dict(re.findall(r'^\[([^:]+): (.*)\]$',text.split('parameters:',1)[-1],re.M))
 assert params['objective']=='lambdarank' and params['extra_trees']=='0'
 assert float(params.get('max_delta_step',params.get('max_leaf_output','0')))==0
 gains=[int(x) for x in params.get('label_gain','').split(',') if x.strip()]
 assert not gains or gains[:8]==[0,1,3,7,15,31,63,127]
 lr=float(params['learning_rate']);assert np.isfinite(lr) and lr>0
 chunks=[];expected=[]
 for line in text.splitlines():
  if line.startswith('num_leaves='):expected.append(int(line.split('=',1)[1]))
  if line.startswith('leaf_value='):
   a=np.fromstring(line.split('=',1)[1],sep=' ',dtype=np.float64);assert len(a)>0 and np.isfinite(a).all();chunks.append(a)
 assert chunks and len(chunks)==len(expected) and [len(x) for x in chunks]==expected
 leaves=np.concatenate(chunks);return np.abs(leaves/lr),lr,len(chunks)

def fixed_cap(values):
 a=np.asarray(values,dtype=np.float64);assert a.ndim==1 and len(a)>0 and np.isfinite(a).all() and (a>=0).all()
 c=float(np.quantile(a,.90,method='linear'));assert np.isfinite(c) and c>0
 return c
