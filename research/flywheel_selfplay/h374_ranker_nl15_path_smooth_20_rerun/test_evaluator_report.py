"""Data-free complete evaluation flow: no fitting or model scoring."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import evaluate_h374 as e
import run_h374 as r

class ReportFlowTest(unittest.TestCase):
 def test_all_cells_config_agreement_and_named_raw_arrays(self):
  families=[f'F{i:02}' for i in range(13)]
  def panel(G,picks):
   u=np.zeros((3,13,2));u[:,:,1]=.001*np.arange(13)[None,:]+picks*.001
   return u,np.zeros_like(u,dtype=bool)
  def summarize(u,invalid,families,picks,boot_seed):
   return dict(per_seed_ci95=[[0,1]]*3,elo_by_seed=[1500]*3,elo_ci95_by_seed=[[1400,1600]]*3)
  fake=SimpleNamespace(panel=panel)
  model=SimpleNamespace(G=dict(train_rows=[dict(family_id=f) for f in families],a302=SimpleNamespace(summarize_three_seeds=summarize)))
  parent=Path(__file__).resolve().parent.parent/'report-test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as root:
   d=Path(root);(d/'regression.json').write_text(json.dumps(dict(status='PASS')));(d/'nl15-prepared.json').write_text('{}')
   for p in e.PARTS:
    (d/'base').mkdir(exist_ok=True)
    for prefix in ['base/','nl15/','']:
     for fold in range(4):
      for seed in e.SEEDS:
       td=d/prefix/p/f'fold_{fold}'/f'root_{seed}';td.mkdir(parents=True)
       (td/'fit_lock.json').write_text(json.dumps(dict(chosen=dict(tau=.05,k=5,a=2))))
    for td,arm in [(d/'base','BASE'),(d/'nl15','NL15'),(d,'PS20')]:
     (td/f'{arm}_{p}_locks.json').write_text('{}')
    np.save(d/'nl15'/f'NL15_{p}.npy',np.zeros((3,13),dtype=np.int32))
   base={p:np.zeros((3,13),dtype=np.int32) for p in e.PARTS};dart={p:np.ones((3,13),dtype=np.int32) for p in e.PARTS}
   with patch.object(e,'P',d),patch.dict(sys.modules,{'evaluate_h303':fake}),patch.dict(r.R,{'loader_counters':{}},clear=True),contextlib.redirect_stdout(io.StringIO()):
    e.evaluate(model,base,dart,d)
   m=json.loads((d/'metrics.json').read_text())
   self.assertEqual(len(m['cells']),27);self.assertEqual(len(m['chosen_configs']),36)
   self.assertEqual(m['config_equal_count'],36)
   self.assertIn('path_smooth0to20',m['intervention']);self.assertEqual(m['primary_comparator'],'NL15');self.assertEqual(m['secondary_comparator'],'BASE');self.assertNotIn('reg_lambda1->5',m['intervention'])
   self.assertTrue(all(set(x)=={'partition','fold','root_seed','BASE','NL15','PS20'} for x in m['chosen_configs']))
   self.assertTrue(all(x['config_agreement_folds']==4 for x in m['cells']))
   with np.load(d/'raw_family_gains.npz') as z:self.assertEqual(set(z.files),{'BASE','NL15','PS20'})

if __name__=='__main__':unittest.main()
