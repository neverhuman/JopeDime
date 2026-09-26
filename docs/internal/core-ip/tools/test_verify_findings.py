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

    def test_mutations_are_rejected(self):
        # No builds, clones or worktrees here; small disposable fixture copies only.
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
