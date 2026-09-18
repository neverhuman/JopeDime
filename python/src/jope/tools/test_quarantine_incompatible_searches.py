import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import quarantine_incompatible_searches as recovery
from promotion_gate import immutable


class RecoveryTests(unittest.TestCase):
    def test_interrupted_move_retains_bytes_and_never_overwrites_a_new_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);original=root/'search';original.mkdir();output=root/'recovery'
            immutable(original/'candidate-0.attempt.json',dict(evaluator_sha256='old'))
            (original/'observations.jsonl').write_text('')
            request=dict(retired_evaluator_sha256='old',searches=[dict(path='search',files=recovery.tree(original))])
            with patch.object(recovery,'immutable',side_effect=OSError('power loss')):
                with self.assertRaises(OSError):recovery.move(root,output,request)
            self.assertFalse(original.exists())
            original.mkdir();immutable(original/'candidate-0.attempt.json',dict(evaluator_sha256='new'))
            fresh=recovery.tree(original)
            recovery.move(root,output,request)
            self.assertEqual(recovery.tree(original),fresh)
            self.assertEqual(recovery.tree(output/'searches/000'),request['searches'][0]['files'])

    def test_completed_mixed_or_tampered_search_is_never_moved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);original=root/'search';original.mkdir()
            immutable(original/'candidate-0.attempt.json',dict(evaluator_sha256='old'))
            recovery.eligible(original,'old')
            with self.assertRaises(RuntimeError):recovery.eligible(original,'different')
            request=dict(retired_evaluator_sha256='old',searches=[dict(path='search',files=recovery.tree(original))])
            (original/'observations.jsonl').write_text('{"evaluator_sha256":"different"}\n')
            with self.assertRaises(RuntimeError):recovery.eligible(original,'old')
            with self.assertRaises(RuntimeError):recovery.move(root,root/'recovery',request)
            immutable(original/'search.json',dict(completed_evaluations=1))
            with self.assertRaises(RuntimeError):recovery.eligible(original,'old')
            self.assertTrue(original.exists())
