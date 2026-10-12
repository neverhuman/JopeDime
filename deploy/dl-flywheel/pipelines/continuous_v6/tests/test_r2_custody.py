"""v2build r2 counterexamples for Codex's six findings on the v2 BUILD (F1-F6).  Private CPU fixtures only.

Each test either proves the required refusal on a tampered/incomplete state or the positive custody it guards.
"""
import json
import os
import shutil
import sqlite3
import stat
import time
import unittest
from pathlib import Path
from unittest import mock

from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import sha
from pipelines.continuous_v6 import cli_v6 as CLI
from pipelines.continuous_v6 import custody_v6 as CU
from pipelines.continuous_v6 import genesis as G
from pipelines.continuous_v6 import resume_v6 as RV
from pipelines.continuous_v6 import runner_v6 as R6
from pipelines.final_capture_v1 import capture as FC
from pipelines.continuous_v6.tests.test_runner_v6 import EXEC, CAMPAIGN, ClosingStream, FakeBoundary, V6Fixture, parity_receipt


def _ro(path):
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True, isolation_level=None)
    db.row_factory = sqlite3.Row
    return db


class R2CustodyTests(V6Fixture, unittest.TestCase):
    """W10 idle world + genesis/run helpers from V6Fixture."""

    def resume(self, root):
        return RV.v2_resume(root, execution_config_sha256=EXEC)

    def copy(self, root, name):
        dest = self.base / name; shutil.copytree(root, dest, symlinks=True); return dest

    def audit(self, root):
        g = json.loads((root / 'genesis.json').read_bytes()); t = root / 'training'
        pdb, odb = _ro(t / 'batches.sqlite'), _ro(t / 'outbox.sqlite')
        try:
            return CU.audit(t, pdb, odb, g, campaign_sha256=CAMPAIGN, execution_sha256=EXEC)
        finally:
            pdb.close(); odb.close()

    # ------------------------------------------------------------------ F2 terminal custody
    def test_f2_clean_closure_resumes_and_binds_every_file(self):
        root = self.completed(); t = root / 'training'
        r = self.resume(root)
        self.assertEqual((r['local_step'], r['basis']['kind'], r['watermark']['local_step']), (2, 'closure', 2))
        seg = json.loads((t / 'segment.json').read_bytes())
        self.assertEqual(r['basis']['segment_id'], seg['segment_id'])
        mark, report = self.audit(root)
        self.assertEqual((mark['local_step'], report['joined_steps']), (2, 2), 'full audit from genesis re-joins every step')
        self.assertEqual(mark['proof_chain_sha256'], r['watermark']['proof_chain_sha256'])

    def test_f2_missing_or_interrupted_verification_refuses(self):
        root = self.completed()
        for drop in (('result-verification.json',), ('result.json',), ('result.json', 'result-verification.json')):
            c = self.copy(root, 'drop-' + '-'.join(drop))
            for f in drop: (c / 'training' / f).unlink()
            with self.assertRaisesRegex(ValueError, 'complete terminal closure'):
                self.resume(c)

    def test_f2_failed_closure_writes_failure_and_no_clean_result(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        real = CU.extend
        def failing(*a, **k):
            if k.get('expect_no_suffix'): return real(*a, **k)
            raise ValueError('injected closure verification failure')
        with mock.patch.object(R6.CU, 'extend', side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, 'terminal closure failed'):
                self.run_v6(root, self.config(max_new_updates=2), G.genesis_resume(root, execution_config_sha256=EXEC))
        self.assertTrue((t / 'terminal-intent.json').exists())
        self.assertFalse((t / 'result.json').exists()); self.assertFalse((t / 'result-verification.json').exists())
        self.assertIn('injected', json.loads((t / 'closure-failure.json').read_bytes())['error'])
        with self.assertRaisesRegex(ValueError, 'review required'):
            self.resume(root)

    def test_f2_stale_closure_of_an_earlier_segment_refuses(self):
        root = self.completed(); t = root / 'training'
        seg = json.loads((t / 'segment.json').read_bytes()); seg['segment_id'] = digest({'later': 'segment that never closed'})
        os.chmod(t / 'segment.json', stat.S_IWUSR | stat.S_IRUSR); (t / 'segment.json').write_bytes(canonical(seg))
        with self.assertRaisesRegex(ValueError, 'latest started segment'):
            self.resume(root)

    def test_f2_rebound_or_failed_verification_refuses(self):
        root = self.completed()
        c = self.copy(root, 'failed'); v = json.loads((c / 'training' / 'result-verification.json').read_bytes()); v['passed'] = False
        (c / 'training' / 'result-verification.json').write_bytes(canonical(v))
        with self.assertRaisesRegex(ValueError, 'passed verification'):
            self.resume(c)
        c = self.copy(root, 'restate'); r = json.loads((c / 'training' / 'result.json').read_bytes()); r['state'] = 'maximum_updates_reached '
        (c / 'training' / 'result.json').write_bytes(canonical(r))
        with self.assertRaisesRegex(ValueError, 'clean terminal state'):
            self.resume(c)

    def test_f2_fatal_and_markers_refuse_and_runner_rechecks_basis(self):
        root = self.completed()
        c = self.copy(root, 'fatal'); (c / f'fatal-{time.time_ns()}.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'fatal'):
            self.resume(c)
        c = self.copy(root, 'basis'); r = self.resume(c)
        res = json.loads((c / 'training' / 'result.json').read_bytes()); res['error'] = 'rewritten after resume'
        (c / 'training' / 'result.json').write_bytes(canonical(res))
        with self.assertRaisesRegex(ValueError, 'latest verified closure'):
            self.run_v6(c, self.config(max_new_updates=3), r)
        with self.assertRaisesRegex(ValueError, 'no segment has ever started'):
            G.genesis_resume(root, execution_config_sha256=EXEC)
        self.genesis('marked'); m = self.base / 'marked'; (m / 'training' / 'closure-failure.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'no segment has ever started'):
            G.genesis_resume(m, execution_config_sha256=EXEC)

    # ------------------------------------------------------------------ F3 joined suffix
    def test_f3_missing_or_tampered_proof_refuses(self):
        root = self.completed(); origin = json.loads((root / 'genesis.json').read_bytes())['origin_absolute_step']
        c = self.copy(root, 'missing'); (c / 'training' / f'batch-{origin + 1:09d}.json').unlink()
        with self.assertRaisesRegex(ValueError, 'committed proof present'):
            self.audit(c)
        with self.assertRaisesRegex(ValueError, 'proof inventory'):
            self.resume(c)
        c = self.copy(root, 'tampered'); p = c / 'training' / f'batch-{origin + 1:09d}.json'
        os.chmod(p, stat.S_IWUSR | stat.S_IRUSR); p.write_bytes(p.read_bytes().replace(b'"fresh_generated_rewards_used":false', b'"fresh_generated_rewards_used":true'))
        with self.assertRaisesRegex(ValueError, 'exactly the proof bytes'):
            self.audit(c)
        c = self.copy(root, 'last'); p = c / 'training' / f'batch-{origin + 2:09d}.json'
        os.chmod(p, stat.S_IWUSR | stat.S_IRUSR); p.write_bytes(p.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'current proof bytes equal the committed evidence|last committed proof'):
            self.resume(c)

    def test_f3_wrong_ack_and_old_outbox_refuse(self):
        root = self.completed()
        c = self.copy(root, 'ack'); db = sqlite3.connect(c / 'training' / 'outbox.sqlite', isolation_level=None)
        bid, ack = db.execute('SELECT id, ack FROM batches ORDER BY rowid LIMIT 1').fetchone()
        a = json.loads(ack); a['numeric_outcomes_required_for_this_optimizer'] = True
        db.execute('UPDATE batches SET ack=? WHERE id=?', (canonical(a), bid)); db.close()
        with self.assertRaisesRegex(ValueError, 'durable outbox ACK row'):
            self.audit(c)
        self.genesis('fresh'); c = self.copy(root, 'old-outbox')
        for suffix in ('', '-wal', '-shm'):
            q = c / 'training' / ('outbox.sqlite' + suffix)
            if q.exists(): q.unlink()
        shutil.copyfile(self.base / 'fresh' / 'training' / 'outbox.sqlite', c / 'training' / 'outbox.sqlite')
        with self.assertRaisesRegex(ValueError, 'outbox prefix head|durable outbox ACK row'):
            self.resume(c)
        with self.assertRaisesRegex(ValueError, 'durable outbox ACK row'):
            self.audit(c)

    def test_f3_truncated_suffix_orphan_and_prefix_tamper_refuse(self):
        root = self.completed(); origin = json.loads((root / 'genesis.json').read_bytes())['origin_absolute_step']
        c = self.copy(root, 'truncated'); db = sqlite3.connect(c / 'training' / 'batches.sqlite', isolation_level=None)
        top = db.execute('SELECT max(seq) FROM events').fetchone()[0]; db.execute('DELETE FROM events WHERE seq=?', (top,)); db.close()
        with self.assertRaisesRegex(ValueError, 'cursor state tampered|prefix head re-joins'):
            self.resume(c)
        for name, extra in (('orphan', f'batch-{origin + 3:09d}.json'), ('gapped', f'batch-{origin + 4:09d}.json'),
                            ('malformed', 'batch-17.json'), ('pending', f'batch-{origin + 3:09d}.json.pending')):
            c = self.copy(root, name); (c / 'training' / extra).write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'proof inventory|malformed or non-regular proof entry'):
                self.resume(c)
            with self.assertRaisesRegex(ValueError, 'proof inventory|malformed or non-regular proof entry'):
                self.audit(c)
        c = self.copy(root, 'prefix'); db = sqlite3.connect(c / 'training' / 'batches.sqlite', isolation_level=None)
        seq, body = db.execute("SELECT seq, body FROM events WHERE kind='batch.committed' ORDER BY seq LIMIT 1").fetchone()
        db.execute('UPDATE events SET body=? WHERE seq=?', (body.replace('native_optimizer', 'native_optimizeR'), seq)); db.close()
        with self.assertRaisesRegex(ValueError, 'event chain tampered'):
            self.resume(c)

    def test_f3_changed_earlier_proof_after_watermark_refuses(self):
        """CX-REV finding 1: a verified watermark does not certify today's bytes; resume re-hashes the whole prefix."""
        root = self.completed(); origin = json.loads((root / 'genesis.json').read_bytes())['origin_absolute_step']
        c = self.copy(root, 'changed-first'); p = c / 'training' / f'batch-{origin + 1:09d}.json'
        os.chmod(p, stat.S_IWUSR | stat.S_IRUSR); p.write_bytes(b'{"corrupted":true}')
        with self.assertRaisesRegex(ValueError, 'current proof bytes equal the committed evidence'):
            self.resume(c)
        v = json.loads((root / 'training' / 'result-verification.json').read_bytes())
        self.assertEqual((v['prefix_integrity']['proofs'], v['prefix_integrity']['bytes'] > 0), (2, True))

    def test_f3_closure_rehashes_a_proof_changed_after_its_epoch_watermark(self):
        """Mutation survivor a02: only prefix_integrity (not the suffix joins) can see this change before closure."""
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        origin = json.loads((root / 'genesis.json').read_bytes())['origin_absolute_step']
        real = CU.extend
        def corrupt_then_extend(*a, **k):
            mark = a[3]
            if not k.get('expect_no_suffix') and mark['local_step'] == 2 and (t / 'verified-watermark.json').exists():
                p = t / f'batch-{origin + 1:09d}.json'; os.chmod(p, stat.S_IWUSR | stat.S_IRUSR); p.write_bytes(b'{"corrupted":true}')
            return real(*a, **k)
        ClosingStream.commits = 0
        with mock.patch.object(R6.CU, 'extend', side_effect=corrupt_then_extend):
            with self.assertRaisesRegex(RuntimeError, 'terminal closure failed'):
                self.run_v6(root, self.config(max_new_updates=3), G.genesis_resume(root, execution_config_sha256=EXEC),
                            planner_factory=ClosingStream)
        self.assertEqual(json.loads((t / 'verified-watermark.json').read_bytes())['watermark']['local_step'], 2)
        self.assertIn('current proof bytes equal the committed evidence (step 1)',
                      json.loads((t / 'closure-failure.json').read_bytes())['error'])
        self.assertFalse((t / 'result.json').exists())

    def test_f3_symlinked_proof_refuses_even_with_identical_bytes(self):
        """Mutation survivor a04: a symlink to a byte-identical copy passes the re-hash; the inventory must refuse it."""
        root = self.completed(); origin = json.loads((root / 'genesis.json').read_bytes())['origin_absolute_step']
        c = self.copy(root, 'symlink'); p = c / 'training' / f'batch-{origin + 1:09d}.json'
        shutil.copyfile(p, self.base / 'outside.json'); p.unlink(); p.symlink_to(self.base / 'outside.json')
        with self.assertRaisesRegex(ValueError, 'malformed or non-regular proof entry'):
            self.resume(c)

    def test_g32_learning_guard_holds_adoptions_and_is_sticky(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        ClosingStream.commits = 0
        cfg = self.config(max_new_updates=2, regression_guard={'reference_loss': 1e-9, 'ratio': 1.02, 'consecutive_epochs': 1})
        out, boundary = self.run_v6(root, cfg, G.genesis_resume(root, execution_config_sha256=EXEC), planner_factory=ClosingStream)
        self.assertEqual(out['state'], 'maximum_updates_reached')
        guard = json.loads((t / 'learning-guard.json').read_bytes())
        self.assertTrue(guard['tripped']); self.assertIn('owner review', guard['alert'])
        self.assertEqual(boundary.calls, [], 'no adoption while the learning guard is tripped')
        self.assertEqual(json.loads((t / 'boundary-status.json').read_bytes())['state'], 'held_by_learning_guard')
        # Segment 2 reloads the persisted trip.  The 4-context fixture world has no further unique batch after this
        # resume (it waits until its wall deadline), so this segment shows reload + stickiness + no adoption only.
        r = self.resume(root); later = FakeBoundary()
        with mock.patch.object(R6, 'remaining', lambda d: d - time.time()), \
             mock.patch.object(R6, 'planner_epoch_complete', lambda planner: True):
            out2, _ = self.run_v6(root, self.config(max_new_updates=3, regression_guard=cfg['neural']['regression_guard']), r,
                                  boundary=later, planner_factory=ClosingStream, deadline=time.time() + 15)
        self.assertEqual(out2['state'], 'global_deadline_reached')
        reloaded = json.loads((t / 'learning-guard.json').read_bytes())
        self.assertTrue(reloaded['tripped']); self.assertEqual(reloaded['tripped_at'], guard['tripped_at'])
        self.assertEqual(later.calls, [], 'no adoption in a segment that starts with a tripped guard')
        with self.assertRaisesRegex(ValueError, 'policy unchanged'):
            self.run_v6(root, self.config(max_new_updates=5), self.resume(root))

    def test_g32_guard_state_is_sealed_in_the_closure(self):
        """CX-REV 6b1ee581: after a tripped HOLD, deleting/resetting/corrupting the sidecar must refuse every restart."""
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        ClosingStream.commits = 0
        cfg = self.config(max_new_updates=2, regression_guard={'reference_loss': 1e-9, 'ratio': 1.02, 'consecutive_epochs': 1})
        self.run_v6(root, cfg, G.genesis_resume(root, execution_config_sha256=EXEC), planner_factory=ClosingStream)
        v = json.loads((t / 'result-verification.json').read_bytes())
        self.assertTrue(v['learning_guard']['tripped'])
        self.assertEqual(v['learning_guard_sha256'], sha((t / 'learning-guard.json').read_bytes()))
        reset = dict(v['learning_guard'], tripped=False, over=0); reset.pop('tripped_at', None); reset.pop('alert', None)
        for name, change in (('deleted', lambda p: p.unlink()), ('reset', lambda p: p.write_bytes(canonical(reset))),
                             ('malformed', lambda p: p.write_bytes(b'{not json')),
                             ('symlink', lambda p: (p.unlink(), p.symlink_to(self.base / 'campaign' / 'training' / 'learning-guard.json')))):
            c = self.copy(root, 'guard-' + name); p = c / 'training' / 'learning-guard.json'; os.chmod(p, stat.S_IWUSR | stat.S_IRUSR)
            change(p)
            with self.assertRaisesRegex(ValueError, 'closure-sealed state'):
                self.resume(c)
        c = self.copy(root, 'guard-after-resume'); r = self.resume(c)
        p = c / 'training' / 'learning-guard.json'; os.chmod(p, stat.S_IWUSR | stat.S_IRUSR); p.write_bytes(canonical(reset))
        with self.assertRaisesRegex(ValueError, 'closure-sealed state'):
            self.run_v6(c, self.config(max_new_updates=3, regression_guard=cfg['neural']['regression_guard']), r)
        r = self.resume(root)
        self.assertTrue(r['learning_guard']['tripped'], 'intact restart recovers the sealed trip')
        self.genesis('g2'); (self.base / 'g2' / 'training' / 'learning-guard.json').write_bytes(canonical(reset))
        with self.assertRaisesRegex(ValueError, 'no segment has ever started'):
            G.genesis_resume(self.base / 'g2', execution_config_sha256=EXEC)

    def deferred_segment(self, root, guard_cfg):
        """Segment 1 closes an epoch with too little boundary time: the boundary work (G32 included) is deferred."""
        ClosingStream.commits = 0
        out, b = self.run_v6(root, self.config(max_new_updates=2, epoch_boundary_budget_seconds=1e6, regression_guard=guard_cfg),
                             G.genesis_resume(root, execution_config_sha256=EXEC), planner_factory=ClosingStream)
        t = root / 'training'; res = json.loads((t / 'result.json').read_bytes())
        self.assertEqual((out['state'], b.calls), ('global_deadline_reached', []))
        self.assertIsNotNone(res['deferred_boundary']); self.assertFalse(list(t.glob('validation-original-epoch-*.json')))
        self.assertNotIn('last_checked_epoch', json.loads((t / 'learning-guard.json').read_bytes()))
        return res['deferred_boundary']['epoch']

    def restart(self, root, guard_cfg, *, budget=0.001, deadline=15, updates=2, **kw):
        later = FakeBoundary()
        with mock.patch.object(R6, 'remaining', lambda d: d - time.time()), \
             mock.patch.object(R6, 'planner_epoch_complete', lambda planner: True):
            out, _ = self.run_v6(root, self.config(max_new_updates=updates, epoch_boundary_budget_seconds=budget, regression_guard=guard_cfg),
                                 self.resume(root), boundary=later, deadline=time.time() + deadline, **kw)
        return out, later, json.loads((root / 'training' / 'learning-guard.json').read_bytes())

    def test_g32_deferred_epoch_is_checked_before_adoption_and_trips(self):
        """CX-REV 6e4b209f: a deferred closed epoch must be guard-checked before any restart adoption."""
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        trip = {'reference_loss': 1e-9, 'ratio': 1.02, 'consecutive_epochs': 1}
        epoch = self.deferred_segment(root, trip)
        out, later, guard = self.restart(root, trip)
        self.assertEqual(later.calls, [], 'no adoption: the deferred check trips the guard first')
        self.assertTrue(guard['tripped']); self.assertEqual(guard['last_checked_epoch'], epoch)
        self.assertEqual([h['epoch'] for h in guard['history']], [epoch])
        self.assertTrue((t / f'validation-original-epoch-{epoch:06d}.json').exists())
        v = json.loads((t / 'result-verification.json').read_bytes()); self.assertEqual(v['learning_guard'], guard)
        out3, later3, guard3 = self.restart(root, trip)            # sticky at start: already checked and tripped
        self.assertEqual((later3.calls, guard3['history'], guard3['tripped']), ([], guard['history'], True))

    def test_g32_deferred_epoch_checked_once_then_adopts(self):
        self.genesis(); root = self.base / 'campaign'
        ok = {'reference_loss': 1e9, 'ratio': 1.02, 'consecutive_epochs': 1}
        epoch = self.deferred_segment(root, ok)
        out, later, guard = self.restart(root, ok)
        self.assertEqual((len(later.calls), guard['tripped'], guard['last_checked_epoch'], len(guard['history'])), (1, False, epoch, 1))
        out3, later3, guard3 = self.restart(root, ok)
        self.assertEqual((len(later3.calls), len(guard3['history'])), (1, 1), 'no double count; adoption retry only')

    def test_g32_deferred_epoch_with_too_little_time_defers_again_without_reservation(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        ok = {'reference_loss': 1e9, 'ratio': 1.02, 'consecutive_epochs': 1}
        epoch = self.deferred_segment(root, ok)
        db = sqlite3.connect(t / 'batches.sqlite'); before = db.execute('SELECT count(*) FROM batches').fetchone()[0]; db.close()
        calls = []
        class Counting(R6.BatchStream):   # the fixture's reserve() can return Waiting without writing a row (mutant g27)
            def reserve(self, *a, **k):
                calls.append(1); return super().reserve(*a, **k)
        # updates=3 > the resumed local step (2), so a missing start-deferral guard WOULD enter the loop (mutant g27)
        out, later, guard = self.restart(root, ok, budget=1e6, updates=3, planner_factory=Counting)
        self.assertEqual(calls, [], 'no reservation attempt while a closed epoch is still unchecked')
        db = sqlite3.connect(t / 'batches.sqlite'); after = db.execute('SELECT count(*) FROM batches').fetchone()[0]; db.close()
        self.assertEqual((out['state'], later.calls, before), ('global_deadline_reached', [], after))
        self.assertNotIn('last_checked_epoch', guard)
        self.assertEqual(json.loads((t / 'result.json').read_bytes())['deferred_boundary']['epoch'], epoch)

    def test_g32_guard_update_transitions(self):
        """Mutation survivors g13/g14/g15: the pure transition, table-tested."""
        rg = {'reference_loss': 1.0, 'ratio': 1.02, 'consecutive_epochs': 2}
        g0 = {'schema': 'jope.v2-learning-guard/v1', **rg, 'over': 0, 'tripped': False, 'history': []}
        at = R6.guard_update(g0, rg, epoch=1, at_step=10, loss=1.02, now=1.0)
        self.assertEqual((at['over'], at['tripped']), (0, False), 'exactly ratio x reference is not a regression')
        self.assertEqual(g0['history'], [], 'input state is not modified')
        g = R6.guard_update(g0, rg, epoch=1, at_step=10, loss=1.0201, now=1.0); self.assertEqual((g['over'], g['tripped']), (1, False))
        g = R6.guard_update(g, rg, epoch=2, at_step=20, loss=1.0, now=2.0); self.assertEqual(g['over'], 0, 'a good epoch resets the count')
        g = R6.guard_update(g, rg, epoch=3, at_step=30, loss=1.5, now=3.0); self.assertFalse(g['tripped'])
        g = R6.guard_update(g, rg, epoch=4, at_step=40, loss=1.5, now=4.0)
        self.assertEqual((g['over'], g['tripped'], g['tripped_at']), (2, True, {'epoch': 4, 'at_step': 40, 'at': 4.0}))
        g = R6.guard_update(g, rg, epoch=5, at_step=50, loss=0.5, now=5.0)
        g = R6.guard_update(g, rg, epoch=6, at_step=60, loss=9.0, now=6.0)
        g = R6.guard_update(g, rg, epoch=7, at_step=70, loss=9.0, now=7.0)
        self.assertEqual((g['tripped'], g['tripped_at']['epoch'], g['alert'], g['last_checked_epoch']), (True, 4, R6.GUARD_ALERT, 7),
                         'sticky: a later re-trip keeps the first trip identity')
        self.assertEqual([h['epoch'] for h in g['history']], [1, 2, 3, 4, 5, 6, 7])
        for i in range(60): g = R6.guard_update(g, rg, epoch=8 + i, at_step=80 + i, loss=0.1, now=8.0)
        self.assertEqual(len(g['history']), 50)
        with self.assertRaisesRegex(ValueError, 'finite'):
            R6.guard_update(g0, rg, epoch=1, at_step=1, loss=float('nan'), now=1.0)

    def test_g32_runner_refuses_malformed_policy_and_genesis_sidecar(self):
        """Mutation survivors g05/g08."""
        self.genesis(); root = self.base / 'campaign'
        bad = ({'reference_loss': 1.0, 'ratio': 1.02}, {'reference_loss': 1.0, 'ratio': 0.5, 'consecutive_epochs': 3},
               {'reference_loss': 1.0, 'ratio': 1.02, 'consecutive_epochs': 0}, {'reference_loss': -1.0, 'ratio': 1.02, 'consecutive_epochs': 3})
        for rg in bad:
            with self.assertRaisesRegex(ValueError, 'closed learning-guard policy'):
                self.run_v6(root, self.config(regression_guard=rg), G.genesis_resume(root, execution_config_sha256=EXEC))
        r = G.genesis_resume(root, execution_config_sha256=EXEC)
        (root / 'training' / 'learning-guard.json').write_bytes(canonical({'schema': 'jope.v2-learning-guard/v1'}))
        with self.assertRaisesRegex(ValueError, 'genesis has no learning-guard state'):
            self.run_v6(root, self.config(), r)

    def rebind(self, c, *, intent=None, verification=None, result=None):
        """Rewrite terminal files with every sha binding recomputed, so only the edited clause can refuse."""
        t = c / 'training'; load = lambda f: json.loads((t / f).read_bytes())
        i, v, r = load('terminal-intent.json'), load('result-verification.json'), load('result.json')
        i.update(intent or {}); v.update(verification or {}); r.update(result or {})
        for f in ('terminal-intent.json', 'result-verification.json', 'result.json'): os.chmod(t / f, stat.S_IWUSR | stat.S_IRUSR)
        (t / 'terminal-intent.json').write_bytes(canonical(i)); v['intent_sha256'] = sha(canonical(i))
        (t / 'result-verification.json').write_bytes(canonical(v))
        r.update(intent_sha256=sha(canonical(i)), verification_sha256=sha(canonical(v))); (t / 'result.json').write_bytes(canonical(r))

    def test_f2_fully_rebound_unpassed_or_unclean_closure_refuses(self):
        """Mutation survivors x02/x03: with every binding consistent, the passed / CLEAN clauses alone must refuse."""
        root = self.completed()
        c = self.copy(root, 'unpassed'); self.rebind(c, verification={'passed': False})
        with self.assertRaisesRegex(ValueError, 'passed verification'):
            self.resume(c)
        c = self.copy(root, 'unclean'); self.rebind(c, intent={'state': 'waiting_for_memory'}, verification={'state': 'waiting_for_memory'},
                                                    result={'state': 'waiting_for_memory'})
        with self.assertRaisesRegex(ValueError, 'clean terminal state'):
            self.resume(c)
        c = self.copy(root, 'resealed-guard')   # mutant g18: verification's sealed guard replaced, sidecar and its sha unchanged
        reset = dict(json.loads((c / 'training' / 'result-verification.json').read_bytes())['learning_guard'], tripped=True, over=99)
        self.rebind(c, verification={'learning_guard': reset})
        with self.assertRaisesRegex(ValueError, 'closure-sealed state'):
            self.resume(c)
        c = self.copy(root, 'rebound-ok'); self.rebind(c)
        self.assertEqual(self.resume(c)['local_step'], 2, 'a faithful rebind of unchanged content still resumes')

    def test_f3_stale_control_or_checkpoint_refuses(self):
        root = self.completed()
        c = self.copy(root, 'control'); t = c / 'training'
        v1 = (t / 'v1-final-optimizer-control.json').read_bytes(); os.chmod(t / 'optimizer-control.json', stat.S_IWUSR | stat.S_IRUSR)
        (t / 'optimizer-control.json').write_bytes(v1)
        with self.assertRaisesRegex(ValueError, 'verified control'):
            self.resume(c)
        c = self.copy(root, 'slot'); slots = sorted((c / 'training' / 'rolling').iterdir())
        shutil.rmtree(slots[-1])
        with self.assertRaisesRegex(ValueError, 'newest rolling slot'):
            self.resume(c)

    def test_f3_ack_without_commit_refuses_closure(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        original = R6.BatchStream.commit
        with mock.patch.object(R6.BatchStream, 'commit', lambda self, *a, **k: None):
            with self.assertRaisesRegex(RuntimeError, 'terminal closure failed'):
                self.run_v6(root, self.config(max_new_updates=1), G.genesis_resume(root, execution_config_sha256=EXEC))
        self.assertIs(R6.BatchStream.commit, original)
        self.assertIn('no ACK without a commit', json.loads((t / 'closure-failure.json').read_bytes())['error'])

    # ------------------------------------------------------------------ F4 capture/ancestry joins
    def test_f4_runner_refuses_ancestry_not_stored_by_the_planner(self):
        self.genesis(); root = self.base / 'campaign'
        _, _, ancestry = FC.load_capture(self.capture)
        shifted = {**ancestry, 'completed_epochs': ancestry['completed_epochs'] + 1}
        with self.assertRaisesRegex(ValueError, 'stored ancestry'):
            self.run_v6(root, self.config(), G.genesis_resume(root, execution_config_sha256=EXEC), ancestry=shifted)

    def test_f4_cli_joins_capture_and_ancestry_to_genesis(self):
        self.genesis(); g = json.loads((self.base / 'campaign' / 'genesis.json').read_bytes())
        c = {'capture_dir': str(self.capture), 'identity': {'parents': {'final_capture_sha256': g['capture']['final_capture_sha256']}}}
        ancestry = CLI.join_capture(c, g, FC.load_capture)
        self.assertEqual(digest(ancestry), g['ancestry_sha256'])
        swapped = self.base / 'capture-swap'; shutil.copytree(self.capture, swapped)
        f = swapped / 'final-capture.json'; os.chmod(f, stat.S_IWUSR | stat.S_IRUSR); f.write_bytes(f.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'pinned capture'):
            CLI.join_capture({**c, 'capture_dir': str(swapped)}, g, FC.load_capture)
        same_checkpoint = lambda cap: (None, None, {**ancestry, 'completed_epochs': ancestry['completed_epochs'] + 1})
        with self.assertRaisesRegex(ValueError, 'genesis-pinned ancestry'):
            CLI.join_capture(c, g, same_checkpoint)
        same_summary = lambda cap: (None, None, {**ancestry, 'fingerprints': list(ancestry['fingerprints'])[:-1]})
        with self.assertRaisesRegex(ValueError, 'genesis-pinned ancestry'):
            CLI.join_capture(c, g, same_summary)   # mutation survivor x06: only the digest covers fingerprints

    # ------------------------------------------------------------------ F1/F5 genesis gates and accounting
    def test_f1_r14_requires_exact_cuda_next_update_parity(self):
        for name, over in (('cpu', {'device': 'cpu'}), ('ctl', {'v1_control_sha256': '0' * 64}),
                           ('cap', {'capture_sha256': '1' * 64}), ('applied', {'control_applied': 'post-R14'})):
            with self.assertRaisesRegex(ValueError, 'next-update parity'):
                self.genesis('bad-' + name, next_update_parity=parity_receipt(self.capture, **over))
        bad = parity_receipt(self.capture); bad['comparisons']['A_vs_B1']['bitwise_equal'] = False
        with self.assertRaisesRegex(ValueError, 'next-update parity'):
            self.genesis('bad-bitwise', next_update_parity=bad)
        with self.assertRaises((ValueError, TypeError)):
            self.genesis('none', next_update_parity=None)
        g = self.genesis()
        event = json.loads((self.base / 'campaign' / 'training' / 'control-event-r14.json').read_bytes())
        self.assertEqual(set(event['parity_gate']), {'cpu', 'cuda', 'next_update'})
        self.assertEqual(g['r14']['event_sha256'], sha((self.base / 'campaign' / 'training' / 'control-event-r14.json').read_bytes()))

    def test_f1_probe_root_keeps_v1_control_and_has_no_r14(self):
        g = self.genesis('probe', probe=True); t = self.base / 'probe' / 'training'
        self.assertTrue(g['probe']); self.assertIsNone(g['r14'])
        self.assertFalse((t / 'control-event-r14.json').exists())
        self.assertEqual((t / 'optimizer-control.json').read_bytes(), (t / 'v1-final-optimizer-control.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'precedes the parity receipt'):
            self.genesis('probe2', probe=True, next_update_parity=parity_receipt(self.capture))

    def test_f5_genesis_records_v1_epoch_and_fresh_schedule(self):
        g = self.genesis(); acc = g['epoch_accounting']
        body = json.loads((self.capture / 'final-capture.json').read_bytes())
        self.assertEqual({k: acc['v1_open_epoch'][k] for k in body['planner']['current_epoch']}, body['planner']['current_epoch'])
        self.assertEqual(acc['v2_schedule']['first_v2_epoch_lineage_number'], g['ancestry_summary']['completed_epochs'] + 1)
        self.assertIn('fresh', acc['v2_schedule']['policy']); self.assertIn('does not schedule B*', acc['parity_relation'])
        fake = {'planner': {**body['planner'], 'current_epoch': {**body['planner']['current_epoch'], 'state': 'open', 'remaining': 0}}}
        _, _, ancestry = FC.load_capture(self.capture)
        with self.assertRaisesRegex(ValueError, 'captured v1 current epoch'):
            G.epoch_accounting(fake, ancestry)


class R2ClockAndOwnerTests(unittest.TestCase):
    """F6: paired lifetime clocks and physical-ownership checks (no systemd/GPU needed: injected readers)."""
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.base = Path(self.tmp.name)
        now = CLI.clocks(); self.window = {'created': now, 'window_seconds': 72 * 3600}

    def at(self, boot_s, wall_s, boot_id=None):
        c0 = self.window['created']
        return {'boot_id': boot_id or c0['boot_id'], 'boot_ns': c0['boot_ns'] + int(boot_s * 1e9),
                'wall_ns': c0['wall_ns'] + int(wall_s * 1e9), 'mono_ns': c0['mono_ns'] + int(boot_s * 1e9)}

    def test_f6_window_uses_boot_clock_and_fails_closed(self):
        left, el = CLI.window_remaining(self.window, self.at(3600, 3600.5))
        self.assertAlmostEqual(left, 71 * 3600, places=3)
        CLI.window_remaining(self.window, self.at(10 * 3600, 10 * 3600 + 30))           # within NTP slew
        for when, msg in ((self.at(3600, 3600, boot_id='other-boot'), 'reboot'), (self.at(3600, 3600 - 600), 'wall rollback'),
                          (self.at(3600, 3600 + 900), 'wall rollback or jump'), (self.at(72 * 3600 + 1, 72 * 3600 + 1), 'inside the fixed'),
                          (self.at(-5, -5), 'paired wall/boot')):
            with self.assertRaisesRegex(ValueError, msg):
                CLI.window_remaining(self.window, when)

    def fake_cgroup(self, unit, slice_name, mem, quota):
        owner = self.base / 'sys' / slice_name; group = owner / (unit + '.service'); group.mkdir(parents=True)
        (owner / 'memory.max').write_text(str(mem)); (owner / 'cpu.max').write_text(f'{quota} 100000')
        return group

    def test_f6_unit_membership_and_inclusive_allocation(self):
        c = {'unit': 'jopedime-neural-v2-20261010', 'envelope': {'slice': 'jopedime-flywheel.slice', 'inclusive_cpus': 24,
                                                                 'inclusive_memory_bytes': 16 * 1024**3}}
        ok = self.fake_cgroup(c['unit'], 'jopedime-flywheel.slice', 16 * 1024**3, 2400000)
        self.assertEqual(CLI.check_unit(ok, c)['owner_memory_max'], 16 * 1024**3)
        with self.assertRaisesRegex(ValueError, 'exactly the campaign unit'):
            CLI.check_unit(self.fake_cgroup('run-u42', 'jopedime-flywheel.slice/x', 1, 1), c)
        with self.assertRaisesRegex(ValueError, 'inclusive owner slice'):
            CLI.check_unit(self.fake_cgroup(c['unit'], 'app.slice', 1, 1), c)
        with self.assertRaisesRegex(ValueError, 'inclusive owner allocation'):
            CLI.check_unit(self.fake_cgroup(c['unit'], 'big/jopedime-flywheel.slice', 32 * 1024**3, 2400000), c)

    def script(self, name, body):
        p = self.base / name; p.write_text('#!/bin/sh\n' + body); p.chmod(0o755); return str(p)

    def test_f6_gpu_exclusive_and_pinned(self):
        uuid = 'GPU-fixture'
        free = self.script('smi-free', 'case "$1" in --query-gpu*) echo GPU-fixture;; *) echo -n "";; esac\n')
        self.assertEqual(CLI.check_gpu(uuid, smi=free)['compute_apps_on_gpu'], 0)
        busy = self.script('smi-busy', 'case "$1" in --query-gpu*) echo GPU-fixture;; *) echo "4242, GPU-fixture";; esac\n')
        with self.assertRaisesRegex(ValueError, 'no other compute process'):
            CLI.check_gpu(uuid, smi=busy)
        with self.assertRaisesRegex(ValueError, 'pinned GPU0'):
            CLI.check_gpu('GPU-other', smi=free)

    def test_f6_stopped_owners_must_be_closed(self):
        stopped = self.script('sc-stopped', 'echo ActiveState=failed; echo SubState=failed; echo MainPID=0; echo ControlGroup=; echo LoadState=loaded; echo InvocationID=abc\n')
        self.assertEqual(CLI.check_stopped_owners(['v1-unit'], systemctl=(stopped,))[0]['MainPID'], '0')
        cg = self.base / 'cg'; (cg / 'u.service').mkdir(parents=True); (cg / 'u.service' / 'cgroup.procs').write_text('4242\n')
        (cg / 'e.service').mkdir(); (cg / 'e.service' / 'cgroup.procs').write_text('')
        cases = {'active-pid0': 'echo ActiveState=active; echo MainPID=0; echo ControlGroup=',
                 'inactive-pid': 'echo ActiveState=inactive; echo MainPID=77; echo ControlGroup=',
                 'inactive-populated': 'echo ActiveState=inactive; echo MainPID=0; echo ControlGroup=/u.service'}
        for name, body in cases.items():   # mutation survivors x10-x12: each clause alone
            with self.assertRaisesRegex(ValueError, 'stopped owner is closed'):
                CLI.check_stopped_owners(['v1-unit'], systemctl=(self.script('sc-' + name, body + '\n'),), cgroup_root=cg)
        empty = self.script('sc-empty', 'echo ActiveState=inactive; echo MainPID=0; echo ControlGroup=/e.service\n')
        self.assertEqual(CLI.check_stopped_owners(['v1-unit'], systemctl=(empty,), cgroup_root=cg)[0]['cgroup_procs'], 0)


if __name__ == '__main__':
    unittest.main()
