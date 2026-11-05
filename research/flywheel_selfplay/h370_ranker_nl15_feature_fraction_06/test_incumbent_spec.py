import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import fetch_nl15 as f
class IncumbentSpecTests(unittest.TestCase):
 def test_exact_archived_point8_accepted(self):
  f.require_nl15_feature_fraction({'feature_fraction':'0.8'},'OFFLINE_FIXTURE')
 def test_other_fraction_stops_before_fits_and_records_spec_mismatch(self):
  parent=f.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as directory:
   root=Path(directory)
   with patch.object(f,'P',root/'evidence'),patch.object(f,'D',root/'scratch'),self.assertRaises(RuntimeError):
    f.require_nl15_feature_fraction({'feature_fraction':'0.7'},'OFFLINE_FIXTURE')
   for part in ('evidence','scratch'):
    record=json.loads((root/part/'INCOMPLETE_SPEC_MISMATCH.json').read_text())
    self.assertEqual(record['status'],'INCOMPLETE_SPEC_MISMATCH');self.assertEqual(record['fits'],0)
