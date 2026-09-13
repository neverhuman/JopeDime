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

    def test_export_binary_output_is_committed_beside_the_event(self):
        # the source ledger's generation jobs.json resolves the prepared view; a fake exporter writes the artifacts
        generation = self.root / 'runs/x/learning/generation-000001'
        generation.mkdir(parents=True)
        jobs = [dict(output='runs/x/learning/generation-000001/neural/%s/cut-%d/search' % (d, c), prepared='data/prepared/views/%s' % d, prepared_sha256='c' * 64)
                for d, c in (('ds1', 0), ('ds2', 1), ('ds4', 0))]
        (generation / 'jobs.json').write_text(json.dumps(dict(jobs=jobs)))
        fake = self.root / 'fake-export.sh'
        fake.write_text('#!/bin/sh\nout=""\nwhile [ $# -gt 0 ]; do if [ "$1" = "--output" ]; then out="$2"; fi; shift; done\n'
                        'mkdir -p "$out" && echo "print(1)" > "$out/predict.py" && echo \'{"schema":"parallel.champion_export.v1"}\' > "$out/export.json" '
                        '&& echo \'{"status":"pass"}\' > "$out/parity.json" && exit 0\n')
        os.chmod(fake, 0o755)
        receipt = cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo), '--export-binary', str(fake), '--export-limit', '2'])
        self.assertEqual(receipt['committed'], 4)
        statuses = [e['status'] for e in receipt['exports']]
        self.assertEqual(statuses.count('exported'), 2)   # export limit honoured; the other events are still committed
        tracked = subprocess.run(['git', '-C', str(self.repo), 'ls-files', 'programs'], capture_output=True, text=True).stdout.split()
        self.assertEqual(len([t for t in tracked if t.endswith('predict.py')]), 2)
        self.assertTrue(all(t.startswith('programs/') for t in tracked))
        state = cg.load_state(self.repo)
        self.assertEqual(sorted(str(v.get('export')) for v in state['published'].values()), ['None', 'None', 'exported', 'exported'])
        # a later run exports the remaining events once and never re-exports a present one
        again = cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo), '--export-binary', str(fake)])
        self.assertEqual((again['committed'], len(again['exports'])), (0, 0))   # exports only accompany new event commits
        self.assertIsNone(cg.prepared_view_for(self.root, dict(value=dict(source='elsewhere/observations.json', identity={}))))

    def test_feed_reports_per_identity_git_state_for_the_console(self):
        feed = self.root / 'runs/x/successor/scoreboard/champion-git.json'
        cg.main(['status', '--root', str(self.root), '--manifest', self.manifest, '--feed', str(feed), '--json'])
        before = json.loads(feed.read_bytes())
        self.assertEqual((before['schema'], before['identities'], before['events'], before['committed_events'], before['head']), (cg.FEED_SCHEMA, 2, 4, 0, None))
        cg.main(['sync', '--root', str(self.root), '--manifest', self.manifest, '--repo', str(self.repo), '--feed', str(feed)])
        after = json.loads(feed.read_bytes())
        self.assertEqual((after['committed_events'], after['uncommitted_events'], after['last_sync']['committed'], after['last_sync']['push']), (4, 0, 4, 'no_remote'))
        ds1 = next(r for r in after['rows'] if r['dataset'] == 'ds1')
        self.assertEqual((ds1['events'], ds1['committed_events'], ds1['cut'], ds1['partition'], ds1['pushed'], len(ds1['identity_key'])), (3, 3, 0, 'meta_train', 'no', 64))
        log = subprocess.run(['git', '-C', str(self.repo), 'log', '--format=%H %s'], capture_output=True, text=True).stdout.splitlines()
        self.assertIn('family-ds1/ds1 cut 0 rmse 0.7 (#2)', next(line for line in log if line.startswith(ds1['last_commit'])))
        self.assertEqual(ds1['current_event'], cg.load_state(self.repo)['identities'][ds1['identity_key']]['current'])
        self.assertEqual([r['dataset'] for r in after['rows']], ['ds1', 'ds2'])   # sorted by family, dataset, cut
        self.assertFalse(list(feed.parent.glob('.*.partial')))
        # exact provenance per event: the producing search output, observation and program digests, and the commit
        self.assertEqual(len(ds1['history']), 3)
        self.assertEqual([h['improvement_number'] for h in ds1['history']], [0, 1, 2])
        self.assertTrue(all(h['source'] == 'runs/x/learning/generation-000001/neural/ds1/cut-0/search/observations.json' and h['commit'] and h['invalid'] is None for h in ds1['history']))
        self.assertEqual(ds1['current']['program_sha256'], hashlib.sha256(b'genome ds1 2').hexdigest())
        self.assertEqual((ds1['current']['sha256'], ds1['current']['rmse_original_units'], ds1['current']['commit']), (ds1['current_event'], 0.7, ds1['last_commit']))
        self.assertIn('UNVERIFIED', after['join_rule'])
        self.assertIsNone(before['rows'][0]['current']['commit'])   # before any sync nothing is committed

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
