import json
import os
from pathlib import Path
import tempfile
import unittest

import node_mirror_pull as nm
from resume_expired_cycle import sha


class FakeNode:
    """An in-memory node: files -> bytes; census/transfer/rehash are the tool's seams."""

    def __init__(self, files):
        self.files = dict(files)
        self.transfers = 0

    def census(self, node, key, spec):
        rows = [dict(path=p, bytes=len(b), mtime=1.0, sha256=nm.hashlib.sha256(b).hexdigest()) for p, b in sorted(self.files.items())]
        return dict(rows=rows, skipped_meta_test=1, root=spec['root'])

    def transfer(self, node, key, base, paths, destination, bwlimit):
        self.transfers += 1
        for p in paths:
            target = Path(destination) / p
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.files[p])

    def rehash(self, node, key, base, paths):
        return {p: nm.hashlib.sha256(self.files[p]).hexdigest() if p in self.files else None for p in paths}


class NodeMirrorPull(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'mirror'
        self.node = FakeNode({'runs/champions/a/event-1.json': b'{"e":1}', 'runs/champions/a/current.json': b'{"c":1}', 'runs/champions/b/event-2.json': b'{"e":2}'})

    def pull(self):
        return nm.pull(self.root, 'champions', census_fn=self.node.census, transfer_fn=self.node.transfer, rehash_fn=self.node.rehash, tool_path=nm.__file__)

    def test_pull_is_idempotent_and_receipted(self):
        first = self.pull()
        self.assertEqual((first['pulled'], first['unchanged'], first['drift_refused'], first['node_rehash_verified'], first['census']['skipped_meta_test']), (3, 0, [], True, 1))
        self.assertEqual((self.root / 'champions/runs/champions/a/event-1.json').read_bytes(), b'{"e":1}')
        receipt = json.loads(Path(first['receipt_path']).read_bytes())
        self.assertEqual((receipt['schema'], receipt['class'], receipt['pulled']), (nm.SCHEMA, 'champions', 3))
        self.assertFalse(list(self.root.glob('.incoming-*')))
        second = self.pull()
        self.assertEqual((second['pulled'], second['unchanged'], self.node.transfers), (0, 3, 1))

    def test_mutable_refreshes_and_immutable_drift_is_refused(self):
        self.pull()
        self.node.files['runs/champions/a/current.json'] = b'{"c":2}'      # mutable: refreshed
        self.node.files['runs/champions/a/event-1.json'] = b'{"e":9}'        # immutable changed on the node: refused, kept
        receipt = self.pull()
        self.assertEqual((receipt['pulled'], [d['path'] for d in receipt['drift_refused']]), (1, ['runs/champions/a/event-1.json']))
        self.assertEqual((self.root / 'champions/runs/champions/a/event-1.json').read_bytes(), b'{"e":1}')
        self.assertEqual((self.root / 'champions/runs/champions/a/current.json').read_bytes(), b'{"c":2}')

    def test_change_during_transfer_is_not_installed_as_verified(self):
        original_rehash = self.node.rehash
        self.node.rehash = lambda node, key, base, paths: {p: ('f' * 64 if p.endswith('event-2.json') else v) for p, v in original_rehash(node, key, base, paths).items()}
        receipt = self.pull()
        self.assertEqual((receipt['pulled'], receipt['node_changed_during_pull'], receipt['node_rehash_verified']), (2, ['runs/champions/b/event-2.json'], False))
        self.assertFalse((self.root / 'champions/runs/champions/b/event-2.json').exists())
        index = json.loads((self.root / 'index/champions.json').read_bytes())
        self.assertNotIn('runs/champions/b/event-2.json', index)

    def test_mutable_rule(self):
        self.assertTrue(nm.mutable('x/repo/.git/refs/heads/main') and nm.mutable('x/current.json') and nm.mutable('a/.champion-git-state.json'))
        self.assertFalse(nm.mutable('x/repo/.git/objects/ab/cdef') or nm.mutable('runs/champions/a/event-1.json'))


if __name__ == '__main__':
    unittest.main()


import node_mirror_pull


class CensusOnly(unittest.TestCase):
    def test_census_only_indexes_without_transfer(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'mirror'
            rows = [dict(path='runs/big/a.bin', bytes=3, mtime=1.0, sha256='a' * 64)]
            calls = []
            node_mirror_pull.SPEC['census-test'] = dict(base='/node', root='runs', kind='roots', roots=('big',), census_only=True)
            try:
                receipt = node_mirror_pull.pull(root, 'census-test', census_fn=lambda n, k, s: dict(rows=rows), transfer_fn=lambda *a: calls.append(a), rehash_fn=lambda *a: {}, tool_path=node_mirror_pull.__file__)
            finally:
                del node_mirror_pull.SPEC['census-test']
            self.assertEqual(calls, [])
            self.assertTrue(receipt['census_only'])
            self.assertEqual((receipt['pulled'], receipt['indexed']), (0, 1))
            self.assertFalse((root / 'census-test' / 'runs/big/a.bin').exists())

