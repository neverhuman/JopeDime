import tempfile
import unittest
from pathlib import Path

from collect_admitted_corpus import completed_sources
from resume_expired_cycle import atomic


class CollectionTests(unittest.TestCase):
    def test_partial_and_historical_searches_do_not_create_fresh_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('old','new','partial'):(root/name).mkdir()
            receipt=dict(schema='parallel.search.v1',evaluator_sha256='new-exe',completed_at=1,
                         final_tests_opened=False,observations_sha256='bound-ledger',completed_evaluations=8)
            atomic(root/'old/search.json',dict(receipt,evaluator_sha256='old-exe'))
            atomic(root/'new/search.json',receipt)
            atomic(root/'partial/observations.json',[])
            jobs=[dict(output=name) for name in ('old','new','partial','new')]
            self.assertEqual(completed_sources(root,jobs,'new-exe'),['new/observations.json'])
            atomic(root/'new/search.json',dict(receipt,final_tests_opened=True))
            with self.assertRaisesRegex(RuntimeError,'incomplete current evaluator'):
                completed_sources(root,jobs,'new-exe')
