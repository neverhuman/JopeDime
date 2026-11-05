"""Observe fresh sklearn-created Dataset binning without changing inputs or parameters."""
from contextlib import contextmanager
import numpy as np

def validate_rows(rows,kind):
 assert rows,'No Dataset constructed by actual fit'
 target=255
 for row in rows:
  assert row['fresh_handle'] and row['raw_numpy_input'] and row['same_fit_reference']
  assert row['constructor_max_bin']==row['effective_max_bin']==target
  assert row['constructed'] and row['feature_num_bins'] and max(row['feature_num_bins'])<=target
 return True

@contextmanager
def audit_fit(model,args,kwargs,kind):
 import lightgbm as lgb
 data=args[0] if args else kwargs['X'];assert isinstance(data,np.ndarray),'INCOMPLETE_SPEC_MISMATCH prebuilt Dataset input'
 target=255
 assert model.get_params().get('max_bin',255)==target
 old_init=lgb.Dataset.__init__;old_construct=lgb.Dataset.construct;rows=[];owned={}
 def init(ds,*a,**k):
  raw=a[0] if a else k['data'];params=k.get('params') or {};ref=k.get('reference')
  assert isinstance(raw,np.ndarray),'INCOMPLETE_SPEC_MISMATCH non-raw Dataset'
  assert int(params.get('max_bin',255))==target,'INCOMPLETE_SPEC_MISMATCH Dataset constructor ignores max_bin'
  assert ref is None or id(ref) in owned,'INCOMPLETE_SPEC_MISMATCH prebuilt reference'
  old_init(ds,*a,**k)
  row=dict(fresh_handle=ds._handle is None,raw_numpy_input=True,same_fit_reference=True,constructor_max_bin=int(params.get('max_bin',255)),shape=list(raw.shape),constructed=False)
  assert row['fresh_handle'];owned[id(ds)]=row;rows.append(row)
 def construct(ds):
  assert id(ds) in owned,'INCOMPLETE_SPEC_MISMATCH prebuilt bins reused'
  row=owned[id(ds)];result=old_construct(ds)
  if not row['constructed']:
   row.update(constructed=True,effective_max_bin=int(ds.get_params().get('max_bin',255)),feature_num_bins=[ds.feature_num_bin(i) for i in range(ds.num_feature())])
   assert row['effective_max_bin']==target and max(row['feature_num_bins'])<=target,'INCOMPLETE_SPEC_MISMATCH effective Dataset bins'
  return result
 lgb.Dataset.__init__=init;lgb.Dataset.construct=construct
 try:yield rows;validate_rows(rows,kind)
 finally:lgb.Dataset.__init__=old_init;lgb.Dataset.construct=old_construct
