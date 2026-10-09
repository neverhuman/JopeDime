"""Portable synthetic records for native optimizer tests; no production artifact reads."""
import torch
import test_training_snapshot_kumos as fixtures
from jope_flywheel.protocol import canonical
from neural_native_train import OBJECTIVE
import hashlib
TEXTS=['(pipe X (ridge))','(pipe X (standardize) (ridge lambda=1))','(pipe X (robust_scale) (ridge lambda=0.1))','(pipe X (rank) (ridge lambda=0.1))','(pipe X (rff_cos append=false gamma=0.7 n=16 seed=5) (ridge lambda=0.1))','(pipe X (rand_proj append=false k=8 seed=7) (ridge lambda=0.1))','(pipe X (row_stats append=false) (ridge lambda=0.1))','(pipe X (elasticnet alpha=0.01 l1_ratio=0.3 sweeps=20))']
class NeuralTests:
 @classmethod
 def setUpClass(cls):fixtures.SnapshotTests.setUpClass();torch.set_num_threads(1)
 def fixture(self):
  f=fixtures.SnapshotTests();f.setUp();f.task(programs=8,program_texts=TEXTS);f.task('validation',programs=8,program_texts=TEXTS);return f.load()
def fixture_permit():
 sha=lambda x:hashlib.sha256(canonical(x)).hexdigest()
 s={'authority_pins':{'synthetic_authority':{'path':'/fixture/authority.json','sha256':'a'*64,'bytes':1}}}
 c={'snapshot':{'sha256':'b'*64},'neural':{'objective':OBJECTIVE,'max_updates':512,'max_epochs':1}}
 c['research_optimizer_admission']={'schema':'jope.native-research-optimizer-admission/v1','approved_for_bounded_research':True,'production_admission':False,'snapshot_sha256':'b'*64,'objective_sha256':sha(OBJECTIVE),'source_authority_sha256':sha(s['authority_pins']),'input_roster':{'kumo_tabular_s':2051},'maximum_updates':512,'maximum_epochs':1,'model_initialization':'fresh','generated_outcomes_used_as_reward':False,'native_hwm_promotion':False}
 return c,s
