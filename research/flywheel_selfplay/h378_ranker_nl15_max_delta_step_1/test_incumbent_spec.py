import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import fetch_nl15 as f
class IncumbentSpecTests(unittest.TestCase):
 def test_exact_archived_zero_accepted(self):
  f.require_nl15_max_delta_step({'max_delta_step':'0'},'OFFLINE_FIXTURE')
 def test_nonzero_max_delta_step_stops_before_fits_and_records_spec_mismatch(self):
  parent=f.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as directory:
   root=Path(directory)
   with patch.object(f,'P',root/'evidence'),patch.object(f,'D',root/'scratch'),self.assertRaises(RuntimeError):
    f.require_nl15_max_delta_step({'max_delta_step':'1'},'OFFLINE_FIXTURE')
   for part in ('evidence','scratch'):
    record=json.loads((root/part/'INCOMPLETE_SPEC_MISMATCH.json').read_text())
    self.assertEqual(record['status'],'INCOMPLETE_SPEC_MISMATCH');self.assertEqual(record['fits'],0)

 def test_missing_or_fractional_baseline_rejected_before_fits(self):
  parent=f.P/'test-fixtures';parent.mkdir(exist_ok=True)
  for native in [{},{'max_delta_step':'0.1'},{'max_delta_step':'-1'},{'max_delta_step':'invalid'},{'max_delta_step':None}]:
   with self.subTest(native=native),tempfile.TemporaryDirectory(dir=parent) as directory:
    root=Path(directory)
    with patch.object(f,'P',root/'evidence'),patch.object(f,'D',root/'scratch'),self.assertRaises(RuntimeError):f.require_nl15_max_delta_step(native,'OFFLINE_FIXTURE')
    self.assertEqual(json.loads((root/'evidence/INCOMPLETE_SPEC_MISMATCH.json').read_text())['fits'],0)
