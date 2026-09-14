import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from inventory_legacy_fits import inventory


class LegacyInventoryTests(unittest.TestCase):
    def test_valid_looking_success_is_seen_but_never_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'observations.json'
            source.write_text(json.dumps([{'genome':'(pipe X (ridge))','metrics':{'rmse':0},'exit_success':True}]))
            state = root/'state.json';state.write_text(json.dumps({'teaching_sources':['observations.json']}))
            frozen = root/'frozen.json';frozen.write_text('{"artifacts":[]}')
            fence = root/'complete.json';fence.write_text(json.dumps({'legacy_dispatch_fenced':True,
                'descendants_reconciled':True,'frozen_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()}))
            result = inventory(root,state,fence,root/'inventory',10)
            self.assertEqual(result['classifications'],{'seen_execution_result_unavailable':1})
            self.assertEqual(result['verified_reusable_completions'],0)
            self.assertFalse(result['registry_import_authorized'])
            self.assertFalse(result['full_namespace_coverage'])
            result = inventory(root,state,fence,root/'inventory',10)
            self.assertEqual(result['classifications']['seen_execution_result_unavailable'],1)
            state.write_text('{"teaching_sources":[]}')
            with self.assertRaisesRegex(RuntimeError,'binding changed'):
                inventory(root,state,fence,root/'inventory',10)

    def test_torn_journal_keeps_containing_scope_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'observations.jsonl').write_bytes(b'{"genome":"(pipe X (ridge))"}')
            state = root/'state.json';state.write_text('{"teaching_sources":["observations.jsonl"]}')
            frozen = root/'frozen.json';frozen.write_text('{"artifacts":[]}')
            fence = root/'complete.json';fence.write_text(json.dumps({'legacy_dispatch_fenced':True,
                'descendants_reconciled':True,'frozen_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()}))
            result = inventory(root,state,fence,root/'inventory',10)
            self.assertEqual(result['unresolved_files'],1)
            self.assertFalse(result['registry_import_authorized'])
