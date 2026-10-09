import copy,json,unittest
from pathlib import Path
from neural_native_cli import validate_permit
from native_test_fixture import fixture_permit
P=Path(__file__).resolve().parent
class NativePermitTests(unittest.TestCase):
 def fixture(self):
  return fixture_permit()
 def test_actual_snapshot_closed_research_permit(self):
  c,s=self.fixture();p=validate_permit(c,s);self.assertFalse(p['production_admission']);self.assertFalse(p['native_hwm_promotion']);self.assertFalse(p['generated_outcomes_used_as_reward'])
 def test_no_unmeasured_reward_promotion_or_scope_growth(self):
  for field,value in [('production_admission',True),('native_hwm_promotion',True),('generated_outcomes_used_as_reward',True),('maximum_updates',513),('maximum_epochs',2),('model_initialization','resume'),('snapshot_sha256','0'*64),('objective_sha256','0'*64),('source_authority_sha256','0'*64)]:
   c,s=self.fixture();c['research_optimizer_admission'][field]=value
   with self.assertRaises(ValueError,msg=field):validate_permit(c,s)
 def test_no_unknown_permit_fields_or_changed_roster(self):
  c,s=self.fixture();c['research_optimizer_admission']['input_roster']['a0']=88
  with self.assertRaises(ValueError):validate_permit(c,s)
  c,s=self.fixture();c['research_optimizer_admission']['backpressure_release']=True
  with self.assertRaises(ValueError):validate_permit(c,s)
if __name__=='__main__':unittest.main()
