import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from spec_guard import evaluate,require_spec_pass
import feature_patch
class SpecTests(unittest.TestCase):
 def test_actual_native_one_rejects_before_probe_or_fit(self):
  x=evaluate([1.0],972);self.assertEqual(x['status'],'INCOMPLETE_SPEC_MISMATCH');self.assertEqual(x['capacity_probes'],0);self.assertEqual(x['treatment_fits'],0)
 def test_any_nonzero_rejects_even_mixed_values(self):self.assertEqual(evaluate([0.,1.],972)['status'],'INCOMPLETE_SPEC_MISMATCH')
 def test_owner_hypothetical_zero_gate_is_distinct_from_actual_one(self):self.assertEqual(evaluate([0.],972)['status'],'PASS')
 def test_missing_cohort_count_is_not_a_pass(self):
  with self.assertRaises(AssertionError):evaluate([1.],971)
 def test_install_denied_without_touching_models(self):
  with patch('spec_guard.require_spec_pass',side_effect=RuntimeError('INCOMPLETE_SPEC_MISMATCH')):
   with self.assertRaises(RuntimeError):feature_patch.install({})
 def test_rejected_receipt_blocks_pipeline(self):
  root=Path(__file__).parent.parent/'spec-fixtures';root.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=root) as d:
   p=Path(d)/'SPEC.json';p.write_text(json.dumps(evaluate([1.],972)))
   with self.assertRaises(RuntimeError):require_spec_pass(p)
