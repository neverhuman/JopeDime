import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_findings import ROOT, verify


class FindingsTests(unittest.TestCase):
    def test_original_inventory(self):
        result = verify()
        self.assertEqual((result['gap'], result['improvement']), (97, 98))

    def test_closure_and_dependency_clauses_are_preserved(self):
        record = json.loads((ROOT / 'register.json').read_text())
        entries = {entry['id']: entry for entry in record['entries']}
        # These independently inventoried source labels must not disappear when
        # the extraction vocabulary changes. Do not derive expectations from its regex.
        for shard, prefix, count, label in [
            (5, 'G', 8, '**Close with:**'),
            (5, 'I', 8, '**Completion evidence:**'),
            (10, 'K', 10, '**Stop / dependency:**'),
        ]:
            for number in range(1, count + 1):
                with self.subTest(shard=shard, number=number, label=label):
                    clauses = entries[f'S{shard:02d}-{prefix}{number}']['closure_route']['source_predicates']
                    self.assertEqual(sum(c.startswith(label) for c in clauses), 1)

    def test_mutations_are_rejected(self):
        # No builds, clones or worktrees here; small disposable fixture copies only.
        def remove_clause(record, entry_id, label):
            entry = next(e for e in record['entries'] if e['id'] == entry_id)
            clauses = entry['closure_route']['source_predicates']
            clauses[:] = [c for c in clauses if not c.startswith(label)]

        mutations = {
            'omission': lambda r: r['entries'].pop(),
            'duplicate': lambda r: r['entries'].__setitem__(0, r['entries'][1]),
            'renumber': lambda r: r['entries'][0].update(original_id='G01'),
            'source substitution': lambda r: r['sources'][0].update(sha256='0' * 64),
            'section substitution': lambda r: r['entries'][0].update(section_sha256='0' * 64),
            'acceptance missing': lambda r: r['entries'][0]['closure_route'].update(acceptance=''),
            'evidence traversal': lambda r: r['entries'][0]['closure_route'].update(evidence_location='../pass.json'),
            'closed without evidence': lambda r: r['entries'][0].update(disposition='closed'),
            'synthetic promotion': lambda r: r['entries'][0]['attestations'].update(promotion=['PASS']),
            'lost closure clause': lambda r: remove_clause(r, 'S05-G1', '**Close with:**'),
            'lost completion evidence': lambda r: remove_clause(r, 'S05-I1', '**Completion evidence:**'),
            'lost stop dependency': lambda r: remove_clause(r, 'S10-K1', '**Stop / dependency:**'),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copytree(ROOT / 'sources', root / 'sources')
                record = json.loads((ROOT / 'register.json').read_text())
                mutate(record)
                (root / 'register.json').write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    verify(root)


if __name__ == '__main__':
    unittest.main()
