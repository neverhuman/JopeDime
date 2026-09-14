import copy
import json
from pathlib import Path
import tempfile
import unittest

import flywheel_acceptance as fa
from resume_expired_cycle import sha

RUN = 'runs/r'
NOW = 1_800_000_000.0
TARGET = 't' * 64


def files(*rows):
    return {rel: dict(sha256='s' * 64, bytes=100, mtime=NOW - 10, json=value) for rel, value in rows}


def passing_evidence(tmp):
    """Everything green: reparented, complete generations, fresh audit within targets, no claims, published, mirrored, console live."""
    receipt_path = f'{RUN}/successor/scoreboard/headline/generation-000050.json'
    node_files = files(
        (f'{RUN}/learning/progress.json', dict(automatic_training=True, idle_explanation=None)),
        ('runs/gp-service.json', dict(active=104, evaluations_per_second=200.0)),
        (f'{RUN}/review/measured-performance-audit-latest.json', dict(rolling_full_hour=dict(allocated_core_utilization_while_ready=0.93, occupancy=0.95, mean_gpu_utilization_missing_zero=0.85, handoff_p95_sampled_upper_gap_seconds=12.0))),
        (f'{RUN}/review/flywheel-acceptance.json', dict(connected_transitions=2)),
        (f'{RUN}/successor/decisions/f11-first-monitor-readout.json', dict(outcome='rollback_required', reparent_to=dict(checkpoint_sha256=TARGET))),
        (f'{RUN}/successor/decisions/f11-rollback-status.json', dict(status='reparented', deviations=[], manifest=dict(automatic_training=True))),
        (f'{RUN}/successor/scoreboard/latest.json', dict(claims=dict(independent_improvement='not_established'), receipt=dict(path=receipt_path, sha256='h' * 64), rows=dict(high_marks=dict(per_hour_24h=8.0, all_time=600)))),
        (receipt_path, dict(generation=50)),
        (f'{RUN}/successor/scoreboard/champion-git.json', dict(uncommitted_events=0, last_sync=dict(at=NOW - 60), at=NOW - 60, remote=dict(status='verified'))),
        (f'{RUN}/successor/publication/gp-publication.json', dict(scope='live', lag_p95_24h=120)),
        (f'{RUN}/successor/verdicts/generation-000049.json', dict(schema='parallel.flywheel_verdict.v1', outcome='inconclusive_underpowered')),
    )
    node_files[receipt_path]['sha256'] = 'h' * 64
    events = [dict(timestamp=NOW - 4000, event='generation_frozen'), dict(timestamp=NOW - 3000, event='generation_frozen'), dict(timestamp=NOW - 2500, event='stage_committed', kind='train'),
              dict(timestamp=NOW - 2400, event='diagnostic_accepted', checkpoint_sha256='n' * 64), dict(timestamp=NOW - 2000, event='generation_frozen')]
    gens = [dict(number=n, stage=4, checkpoint_sha256=TARGET, jobs=420, searches=420, report=True, frozen_at=NOW - 1800 * (51 - n)) for n in range(45, 51)]
    gens[-1]['searches'], gens[-1]['report'] = 100, False   # the newest generation may be in flight
    node = dict(now=NOW, files=node_files, state=dict(manifest_sha256='m' * 64, learning_checkpoint='l/step-60000', learning_checkpoint_state_sha256=TARGET, training_diagnostic=dict(ranking_bt=0.505, baseline_ranking_bt=0.5056, held=False),
                                                       generations=gens, generation_count=51),
                journal=dict(bytes=1000, window=1000, events=events), champions=dict(newest_event_mtime=NOW - 120, events=600), rehash={'a.json': 'a' * 64},
                console=dict(unit='ActiveState=active\nMainPID=1\nInvocationID=x', panes='jope-recovery\t0\t123\n', frame='RECOVERY\nFrame generated at epoch %.3f\nHeld-out: 8W/0T/8L [CI90 -0.03, 0.04]\nLocal records: 2 / 1h\nSlot Dataset Git status\n' % (NOW - 3)))
    mirror_dir = Path(tmp) / 'mirror'
    local = dict(mirror=dict(files=3, mismatched=[], manifest_sha256='q' * 64, rehash_expected={'a.json': 'a' * 64}), pull=dict(at=NOW - 100, refused=[]),
                 node_mirror=dict(**{k: dict(path='/m/receipts/mirror/x-%s-y.json' % k, mtime=NOW - 100, value=dict(drift_refused=[], node_rehash_verified=True)) for k in ('receipts', 'champion-git', 'champions')},
                                  mirror=dict(path='/m/receipts/mirror/x-receipts-y.json', mtime=NOW - 100, value=dict(drift_refused=[])),
                                  restore=dict(path='/m/receipts/restore/x.json', mtime=NOW - 1000, value=dict(passed=True, scope='forge'))),
                 git=dict(head='abc', remotes=['forge'], remote_head='abc'), credentials=dict(exists=True, mode_0600=True, owner_ok=True), files={str(mirror_dir / 'learning-receipts/MANIFEST.json'): dict(host='xbabe2', sha256='q' * 64, bytes=1, mtime=NOW)})
    return dict(node=node, local=local, node_root='/root', node_name='n', mirror_dir=str(mirror_dir))


class Acceptance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ev = passing_evidence(self.tmp.name)

    def test_all_gates_pass_gives_approved(self):
        receipt = fa.evaluate(self.ev, RUN, since_generation=44)
        self.assertEqual((receipt['verdict'], receipt['failing']), ('approved', []))
        self.assertEqual({g: receipt['gates'][g]['status'] for g in fa.GATES}, {g: 'pass' for g in fa.GATES})
        self.assertTrue(receipt['screaming']['met'])
        self.assertFalse(any(p.endswith('.md') for p in receipt['evidence']))
        self.assertIn(f'{RUN}/learning/progress.json', receipt['evidence'])

    def flip(self, mutate):
        ev = copy.deepcopy(self.ev)
        mutate(ev)
        return fa.evaluate(ev, RUN, since_generation=44)

    def test_each_gate_fails_by_name(self):
        cases = {
            'LOOP': lambda ev: ev['node']['files'][f'{RUN}/successor/decisions/f11-rollback-status.json']['json'].update(status='training_disabled'),
            'CADENCE': lambda ev: ev['node']['state']['generations'][0].update(report=False),
            'THROUGHPUT': lambda ev: ev['node']['files'][f'{RUN}/review/measured-performance-audit-latest.json']['json']['rolling_full_hour'].update(handoff_p95_sampled_upper_gap_seconds=149.0),
            'SCIENCE': lambda ev: ev['node']['state']['training_diagnostic'].update(ranking_bt=0.5225),
            'PUBLICATION': lambda ev: ev['local']['git'].update(remote_head='def'),
            'SYNC_TRUST': lambda ev: ev['node']['rehash'].update({'a.json': 'b' * 64}),
            'TUI': lambda ev: ev['node']['console'].update(panes='jope-recovery\t0\t1\njope-recovery\t0\t2\n'),
        }
        for name, mutate in cases.items():
            receipt = self.flip(mutate)
            self.assertEqual((receipt['verdict'], receipt['failing']), ('not_approved', [name]), name)

    def test_pending_is_never_pass(self):
        receipt = self.flip(lambda ev: ev['local'].update(node_mirror={}))
        self.assertEqual(receipt['gates']['SYNC_TRUST']['status'], 'pending')
        self.assertIn('SYNC_TRUST', receipt['failing'])
        self.assertEqual(receipt['verdict'], 'not_approved')
        unset = fa.evaluate(self.ev, RUN, since_generation=None)
        self.assertEqual(unset['gates']['SCIENCE']['status'], 'pending')

    def test_gpu_targets_not_applicable_only_when_training_is_disabled_and_explained(self):
        def disabled(ev):
            ev['node']['files'][f'{RUN}/successor/decisions/f11-rollback-status.json']['json']['manifest']['automatic_training'] = False
            ev['node']['files'][f'{RUN}/learning/progress.json']['json'].update(automatic_training=False, idle_explanation='automatic training awaits qualified replay')
            ev['node']['files'][f'{RUN}/review/measured-performance-audit-latest.json']['json']['rolling_full_hour'].update(occupancy=0.0)
        receipt = self.flip(disabled)
        throughput = receipt['gates']['THROUGHPUT']
        self.assertEqual(throughput['status'], 'pass')
        self.assertEqual(next(c for c in throughput['checks'] if c['name'] == 'gpu_targets')['status'], 'not_applicable')
        self.assertEqual(receipt['gates']['LOOP']['status'], 'pending')   # training disabled: the loop cannot be closed

    def test_improvement_claim_requires_a_backing_verdict(self):
        def claim(ev):
            ev['node']['files'][f'{RUN}/successor/scoreboard/latest.json']['json']['claims']['independent_improvement'] = 'established_by_verdict:' + 'v' * 64
        receipt = self.flip(claim)
        self.assertIn('SCIENCE', receipt['failing'])

    def test_receipt_is_immutable_and_latest_binds_it(self):
        receipt = fa.evaluate(self.ev, RUN, since_generation=44)
        output = Path(self.tmp.name) / 'acceptance'
        path, digest = fa.write_receipt(output, receipt)
        self.assertEqual(sha(path), digest)
        latest = json.loads((output / 'latest.json').read_bytes())
        self.assertEqual((latest['schema'], latest['receipt']['sha256'], latest['verdict']), (fa.LATEST_SCHEMA, digest, 'approved'))
        again = dict(receipt, verdict='not_approved')
        with self.assertRaises(RuntimeError):
            fa.write_receipt(output, again)   # same timestamp -> same path -> different bytes refused


if __name__ == '__main__':
    unittest.main()


class MigrationBlock(unittest.TestCase):
    def test_migration_block_reports_mirror_and_lane_receipts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ev = passing_evidence(Path(tmp))
            ev['local']['node_mirror']['prepared'] = dict(path='/m/receipts/mirror/x-prepared-y.json', mtime=NOW - 50, value=dict(drift_refused=[], node_rehash_verified=True, census=dict(files=10, bytes=1000), at=NOW - 60))
            ev['local']['node_mirror']['runs-census'] = dict(path='/m/receipts/mirror/x-runs-census-y.json', mtime=NOW - 50, value=dict(drift_refused=[], node_rehash_verified=True, census_only=True, census=dict(files=5, bytes=5000), at=NOW - 60))
            ev['local']['lane'] = dict(prepared=dict(path='/m/receipts/lane/x-prepared.json', mtime=NOW - 40, value=dict(units=3, counts=dict(captured=2, already_bound=1), captured_bytes=700, at=NOW - 45)))
            receipt = fa.evaluate(ev, RUN, since_generation=44)
            m = receipt['migration']
            self.assertEqual((m['mirror_classes']['prepared']['verified'], m['mirror_classes']['runs-census']['census_only']), (True, True))
            self.assertNotIn(5000, [m['mirror_verified_bytes']])   # census-only bytes are never counted as mirrored
            self.assertEqual((m['lane_units_captured'], m['lane_failed'], m['lane_classes']['prepared']['captured_bytes']), (3, 0, 700))
            self.assertIn('code_published', m)
            self.assertEqual(receipt['verdict'], 'approved')   # informational block never changes the gates


class RestoreScope(unittest.TestCase):
    def test_local_store_drill_stays_pending(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ev = passing_evidence(Path(tmp))
            ev['local']['node_mirror']['restore']['value'] = dict(passed=True, scope='local_store', sample=12, acknowledged=100, mismatches=0)
            receipt = fa.evaluate(ev, RUN, since_generation=44)   # local_store drill stays pending: SYNC_TRUST cannot pass on it
            names = {c['name']: c for c in receipt['gates']['SYNC_TRUST']['checks']}
            self.assertEqual(names['restore_drill']['status'], 'pending')
            self.assertIn('SYNC_TRUST', receipt['failing'])
            self.assertEqual(receipt['migration']['restore_drill']['scope'], 'local_store')


class PullReceiptRule(unittest.TestCase):
    def test_rust_receipts_class_counts_as_pull(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ev = passing_evidence(Path(tmp))
            ev['local']['pull'] = dict(at=NOW - 100000, refused=[])   # stale Python pull receipt
            ev['local']['node_mirror']['receipts']['value'].update(at=NOW - 60, node_rehash_verified=True, drift_refused=[])
            receipt = fa.evaluate(ev, RUN, since_generation=44)
            names = {c['name']: c for c in receipt['gates']['SYNC_TRUST']['checks']}
            self.assertEqual(names['pull_receipt_fresh']['status'], 'pass')
            ev['local']['node_mirror']['receipts']['value'].update(drift_refused=[dict(path='x')])   # a drifted Rust receipt never substitutes
            receipt = fa.evaluate(ev, RUN, since_generation=44)
            names = {c['name']: c for c in receipt['gates']['SYNC_TRUST']['checks']}
            self.assertEqual(names['pull_receipt_fresh']['status'], 'fail')


class ChampionStreamReceipt(unittest.TestCase):
    def test_champion_stream_receipt_is_a_lane_class(self):
        import tempfile, json, os
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp) / 'receipts/lane'; lane.mkdir(parents=True)
            (lane / '20260914T100000Z-champion-events-abc.json').write_text(json.dumps(dict(schema='jain.champion_capture_receipt.v1', identities=5, captured=3, already_bound=1, refused_partition=0, failed=1, captured_bytes=99, at=1.0)))
            (lane / '20260914T100001Z-prepared-abc.json').write_text(json.dumps({'class': 'prepared', 'units': 2, 'counts': {'captured': 2}, 'captured_bytes': 10, 'at': 2.0}))
            local = fa.collect_local(Path(tmp) / 'root', Path(tmp) / 'mirror', Path(tmp), Path(tmp) / 'norepo', None)
            self.assertEqual(sorted(local['lane']), ['champion-events', 'prepared'])
            m = fa.migration(dict(local=local))
            self.assertEqual((m['lane_classes']['champion-events']['units'], m['lane_classes']['champion-events']['captured'], m['lane_failed']), (5, 3, 1))

