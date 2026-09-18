import json
import tempfile
import unittest
from pathlib import Path

import mirror_learning_receipts as m

RUN = 'runs/r'


class MirrorLearningReceipts(unittest.TestCase):
    """The node is simulated by an in-memory file map; every ssh call is monkeypatched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mirror = Path(self.tmp.name) / 'mirror'
        self.node = {RUN + '/learning/monitoring/a.json': b'{"a": 1}', RUN + '/successor/scoreboard/latest.json': b'{"live": 1}',
                     RUN + '/successor/scoreboard/champion-git.json': b'{"feed": 1}', RUN + '/learning/cycle-000001/train-2/complete.json': b'{"steps": 3000}'}
        self.mtime = {k: 1.0 for k in self.node}
        m.remote_listing = lambda node, key, root, run: [dict(path=p, bytes=len(d), mtime=self.mtime[p]) for p, d in sorted(self.node.items())]
        m.fetch = lambda node, key, root, rel, tail=None, size=None: (self.node[rel], None)
        m.fetch_stub = lambda node, key, root, rel: (b'{"schema": "parallel.receipt_stub.v1"}', 'f' * 64)
        m.node_hashes = lambda node, key, root, rels: {r: (m.sha256(self.node[r]) if r in self.node else None) for r in rels}

    def pull(self, receipt=None):
        return m.pull('n', 'k', '/root', RUN, self.mirror, receipt)

    def test_mutable_names_refresh_and_immutable_change_is_refused(self):
        first = self.pull(receipt=self.mirror / 'last-pull.json')
        self.assertEqual((first['pulled'], first['refused']), (4, []))
        receipt = json.loads((self.mirror / 'last-pull.json').read_bytes())
        self.assertEqual((receipt['schema'], receipt['pulled'], receipt['mutable_files'], receipt['refused']), (m.PULL_RECEIPT_SCHEMA, 4, 2, []))
        self.assertEqual(receipt['manifest_sha256'], m.sha256((self.mirror / 'learning-receipts/MANIFEST.json').read_bytes()))
        # a live feed may change between pulls: refreshed without refusal
        self.node[RUN + '/successor/scoreboard/latest.json'] = b'{"live": 2}'; self.mtime[RUN + '/successor/scoreboard/latest.json'] = 2.0
        second = self.pull()
        self.assertEqual((second['pulled'], second['unchanged'], second['refused']), (1, 3, []))
        self.assertEqual((self.mirror / 'learning-receipts/successor/scoreboard/latest.json').read_bytes(), b'{"live": 2}')
        # an immutable receipt that changed on the node is a defect: refused, not overwritten
        self.node[RUN + '/learning/monitoring/a.json'] = b'{"a": 2}'; self.mtime[RUN + '/learning/monitoring/a.json'] = 3.0
        with self.assertRaisesRegex(RuntimeError, 'immutable receipt changed'):
            self.pull()
        self.assertEqual((self.mirror / 'learning-receipts/learning/monitoring/a.json').read_bytes(), b'{"a": 1}')

    def test_node_verify_reports_immutable_mismatch_mutable_change_and_missing(self):
        self.pull()
        self.node[RUN + '/learning/monitoring/a.json'] = b'{"a": 9}'           # immutable drifted on the node
        self.node[RUN + '/successor/scoreboard/latest.json'] = b'{"live": 3}'  # mutable moved on
        del self.node[RUN + '/learning/cycle-000001/train-2/complete.json']     # vanished
        result = m.verify(self.mirror, 'n', 'k', '/root', node_verify=True)
        self.assertEqual(result['mismatched'], [])   # the local mirror itself is intact
        self.assertEqual([r['path'] for r in result['node_mismatched']], [RUN + '/learning/monitoring/a.json'])
        self.assertEqual([r['path'] for r in result['node_changed_mutable']], [RUN + '/successor/scoreboard/latest.json'])
        self.assertEqual(result['node_missing'], [RUN + '/learning/cycle-000001/train-2/complete.json'])
        self.assertEqual((result['node_checked'], result['node_verified']), (4, 1))

    def test_run_globs_cover_the_research_receipts(self):
        for needed in ('successor/scoreboard/headline/*.json', 'successor/verdicts/*.json', 'successor/acceptance/*.json', 'health/summary.json'):
            self.assertIn(needed, m.RUN_GLOBS)
        self.assertTrue(m.mutable('x/successor/scoreboard/champion-git.json') and m.mutable('x/successor/decisions/f11-rollback-status.json') and not m.mutable('x/learning/monitoring/a.json'))


if __name__ == '__main__':
    unittest.main()
