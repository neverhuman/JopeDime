import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import champion_git as cg

EVALUATOR = 'a' * 64
VOCABULARY = 'b' * 64


def identity(dataset, cut, partition='meta_train', evaluator=EVALUATOR):
    return dict(dataset=dataset, family='family-' + dataset, partition=partition, cut=cut, prepared_sha256='c' * 64, training_sha256='d' * 64,
                validation_sha256='e' * 64, evaluator_sha256=evaluator, vocabulary_sha256=VOCABULARY)


def event(ident, number, rmse, previous=None, stamp=1789300000):
    genome = 'genome %s %d' % (ident['dataset'], number)
    champion = dict(schema=cg.SCHEMA, identity=ident, observation=dict(genome=genome, program_sha256=hashlib.sha256(genome.encode()).hexdigest(), metrics=dict(rmse=rmse)),
                    observation_sha256='f' * 64, source='runs/x/learning/generation-000001/neural/%s/cut-%d/search/observations.json' % (ident['dataset'], ident['cut']),
                    rmse_original_units=rmse, previous_event_sha256=previous, previous_rmse_original_units=None, improvement_number=number, timestamp=stamp + number)
    compact = json.dumps(champion, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(compact).hexdigest(), json.dumps(champion, indent=2).encode(), champion


def store_with(root, identities):
    store = root / 'runs/champions'
    for ident, rmses in identities:
        key = hashlib.sha256(json.dumps(ident, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        directory = store / key
        directory.mkdir(parents=True)
        previous, current = None, None
        for number, rmse in enumerate(rmses):
            sha, pretty, champion = event(ident, number, rmse, previous)
            (directory / ('event-%s.json' % sha)).write_bytes(pretty)
            previous, current = sha, champion
        (directory / 'current.json').write_bytes(json.dumps(current, indent=2).encode())
    return store


def manifest(root):
    path = root / 'manifest.json'
    path.write_text(json.dumps(dict(cpu_executable='bin/gp', bindings={'bin/gp': EVALUATOR}, vocabulary_sha256=VOCABULARY)))
    return 'manifest.json'


class ChampionGitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'parallel'
        self.store = store_with(self.root, [(identity('ds1', 0), [0.9, 0.8, 0.7]), (identity('ds2', 1, 'meta_validation'), [1.5]),
                                            (identity('ds3', 0, 'meta_test'), [0.1]), (identity('ds4', 0, evaluator='9' * 64), [0.2, 0.1])])
        self.manifest = manifest(self.root)
        self.repo = self.root / 'runs/x/successor/champion-git/repo'

    def test_status_lists_live_publishable_identities_only(self):
        rows = cg.main(['status', '--root', str(self.root), '--manifest', self.manifest, '--json'])
        self.assertEqual(sorted((r['dataset'], r['events'], r['uncommitted_events']) for r in rows), [('ds1', 3, 3), ('ds2', 1, 1)])
        everything = cg.main(['status', '--root', str(self.root), '--all-evaluators', '--json'])
        self.assertEqual(sorted(r['dataset'] for r in everything), ['ds1', 'ds2', 'ds4'])   # meta_test never listed

    def test_sync_commits_every_event_once_in_order_and_is_idempotent(self):
        receipt = cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo), '--receipts', str(self.repo.parent / 'receipts')])
        self.assertEqual((receipt['committed'], receipt['events_total'], receipt['identities'], receipt['invalid_count'], receipt['push']['status']), (4, 4, 2, 0, 'no_remote'))
        log = subprocess.run(['git', '-C', str(self.repo), 'log', '--format=%s', '--reverse'], capture_output=True, text=True).stdout.strip().splitlines()
        self.assertEqual(len(log), 5)   # 4 events + 1 index
        ds1 = [line for line in log if 'family-ds1/ds1' in line]
        self.assertEqual([line.split('(#')[1][0] for line in ds1], ['0', '1', '2'])   # chronological per identity
        self.assertTrue(log[-1].startswith('index: 2 identities, 4 events'))
        again = cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo)])
        self.assertEqual(again['committed'], 0)
        self.assertEqual(len(subprocess.run(['git', '-C', str(self.repo), 'log', '--format=%H'], capture_output=True, text=True).stdout.split()), 5)
        rows = cg.main(['status', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo), '--json'])
        self.assertEqual({r['dataset']: r['uncommitted_events'] for r in rows}, {'ds1': 0, 'ds2': 0})
        layout = sorted(str(p.relative_to(self.repo)) for p in self.repo.rglob('champion.json'))
        self.assertEqual(len(layout), 4)
        self.assertTrue(all(p.startswith('champions/ds') and '/events/' in p for p in layout))
        current = json.loads(next(self.repo.glob('champions/ds1/cut-0/*/current.json')).read_bytes())
        self.assertEqual((current['improvement_number'], current['rmse_original_units'], current['claim']), (2, 0.7, cg.CLAIM))
        receipts = list((self.repo.parent / 'receipts').glob('champion-git-*.json'))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(json.loads(receipts[0].read_bytes())['schema'], cg.RECEIPT_SCHEMA)

    def test_new_event_after_sync_becomes_one_new_commit(self):
        cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo)])
        ident = identity('ds2', 1, 'meta_validation')
        directory = next(p for p in self.store.iterdir() if json.loads((p / 'current.json').read_bytes())['identity']['dataset'] == 'ds2')
        previous = next(directory.glob('event-*.json')).name[6:-5]
        sha, pretty, champion = event(ident, 1, 1.2, previous)
        (directory / ('event-%s.json' % sha)).write_bytes(pretty)
        (directory / 'current.json').write_bytes(json.dumps(champion, indent=2).encode())
        receipt = cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo)])
        self.assertEqual(receipt['committed'], 1)
        head = subprocess.run(['git', '-C', str(self.repo), 'log', '-1', '--format=%s', 'HEAD~1'], capture_output=True, text=True).stdout.strip()
        self.assertIn('family-ds2/ds2 cut 1 rmse 1.2 (#1)', head)

    def test_tampered_event_is_refused_not_committed(self):
        directory = next(p for p in self.store.iterdir() if json.loads((p / 'current.json').read_bytes())['identity']['dataset'] == 'ds2')
        path = next(directory.glob('event-*.json'))
        doc = json.loads(path.read_bytes())
        doc['rmse_original_units'] = 0.01
        path.write_bytes(json.dumps(doc, indent=2).encode())
        receipt = cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo)])
        self.assertEqual((receipt['committed'], receipt['invalid_count']), (3, 1))
        self.assertIn('do not hash to the file name', receipt['invalid_events'][0]['error'])

    def test_push_states_without_remote_or_credentials(self):
        cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo)])
        state = cg.load_state(self.repo)
        self.assertEqual(cg.push(self.repo, state, None, 'main', None)['status'], 'no_remote')
        self.assertEqual(cg.push(self.repo, state, 'https://git.example/x.git', 'main', None)['status'], 'no_credentials')
        loose = self.root / 'creds'
        loose.write_text('https://u:t@git.example')
        os.chmod(loose, 0o644)
        self.assertEqual(cg.push(self.repo, state, 'https://git.example/x.git', 'main', str(loose))['status'], 'no_credentials')
        self.assertIn('0600', cg.push(self.repo, state, 'https://git.example/x.git', 'main', str(loose))['last_error'])
        self.assertEqual(cg.scrub('fatal: https://user:token@git.neverhuman.org/x 403'), 'fatal: https://<redacted>@git.neverhuman.org/x 403')


if __name__ == '__main__':
    unittest.main()
