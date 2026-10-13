"""GB-10 runner residuals: actual SIGKILL, durable bytes, then separately reopened consumers.

Private synthetic CPU worlds only. BoundaryDriver is exercised with the real
overlay/planner harness; the update faults use the complete runner_v6. These
fixtures qualify process death, not power loss, live execution or recovery authority.
"""
import copy
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import types
import unittest
from unittest import mock

from safetensors.torch import save

import neural_storage as Storage
from neural_native_stream import BatchStream
from pipelines.boundary_admission_v2 import boundary as Bd
from pipelines.boundary_admission_v2.tests import test_overlay_components as BC
from pipelines.boundary_admission_v2.tests.harness import Executor, CAMPAIGN as B_CAMPAIGN
from pipelines.continuous_v6 import custody_v6 as CU
from pipelines.continuous_v6 import genesis as G
from pipelines.continuous_v6 import resume_v6 as RV
from pipelines.continuous_v6 import runner_v6 as R6
from pipelines.continuous_v6.tests.test_runner_v6 import V6Fixture, EXEC
from pipelines.final_capture_v1.tests import world as W10
from pipelines.native_result_overlay_v1 import packet as PK


def die():
    os.kill(os.getpid(), signal.SIGKILL)


def files(root):
    """SQLite's reader-lock sidecar is transient; preserve every other surviving byte."""
    return {str(p.relative_to(root)): p.read_bytes() for p in Path(root).rglob('*')
            if p.is_file() and not p.name.endswith('-shm')}


class ProcessFixture:
    def child(self, call, *, killed=False):
        pid = os.fork()
        if pid == 0:
            try:
                call()
            except BaseException:
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        deadline = time.monotonic() + 30
        while True:
            ended, status = os.waitpid(pid, os.WNOHANG)
            if ended:
                if killed:
                    self.assertTrue(os.WIFSIGNALED(status), 'requested death point was reached')
                    self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
                else:
                    self.assertTrue(os.WIFEXITED(status))
                    self.assertEqual(os.WEXITSTATUS(status), 0)
                return
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                self.fail('private fault/readback child exceeded 30 seconds')
            time.sleep(.01)


class RunnerProcessDeathTests(ProcessFixture, V6Fixture, unittest.TestCase):
    def start(self):
        g = self.genesis()
        root = self.base / 'campaign'
        return root, g, G.genesis_resume(root, execution_config_sha256=EXEC)

    def kill_runner(self, root, resume, mode):
        """Torch's existing thread pools must not be inherited by a training child."""
        params = self.base / 'worker-params.json'
        params.write_text(json.dumps({'root': str(root), 'capture': str(self.capture),
                                      'config': self.config(), 'resume': resume}))
        paths = [str(Path(p).resolve()) for p in sys.path if p and 'dl-flywheel' in p]
        bootstrap = ('import sys,json;sys.path[:0]=json.loads(sys.argv[1]);'
                     'from pipelines.continuous_v6.tests.test_fault_process_death import runner_child;'
                     'runner_child(sys.argv[2],sys.argv[3])')
        log = self.base / 'worker.log'
        with log.open('wb') as stream:
            process = subprocess.Popen([sys.executable, '-I', '-B', '-c', bootstrap,
                                        json.dumps(paths), str(params), mode], stdout=stream, stderr=stream)
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
                self.fail('private runner child exceeded 30 seconds')
        self.assertEqual(code, -signal.SIGKILL, log.read_text(errors='replace')[-5000:])

    def planner(self, root):
        t = root / 'training'
        raw, active = Bd.active_manifest(t / 'batches.sqlite')
        g = json.loads((root / 'genesis.json').read_bytes())
        return BatchStream(t / 'batches.sqlite', raw, active,
                           seed=self.w.config['neural']['seed'],
                           batch_size=self.w.config['neural']['batch_size'],
                           initial_checkpoint_sha256=g['ancestry_summary']['checkpoint_sha256'])

    def refused_restart(self, root, resume):
        t = root / 'training'
        self.assertTrue((t / 'segment.json').is_file())
        self.assertFalse((t / 'terminal-intent.json').exists(), 'SIGKILL bypassed finally')
        before = files(root)

        def reopen():
            with self.assertRaisesRegex(ValueError, 'complete terminal closure'):
                RV.v2_resume(root, execution_config_sha256=EXEC)
            with self.assertRaisesRegex(ValueError, 'no segment has ever started'):
                G.genesis_resume(root, execution_config_sha256=EXEC)
            with self.assertRaisesRegex(ValueError, 'genesis start: no earlier segment'):
                self.run_v6(root, self.config(), resume)

        self.child(reopen)
        self.assertEqual(files(root), before, 'refusal cannot replay or rewrite custody')

    def assert_cursor(self, root, *, committed, unresolved, acknowledgements=1):
        t = root / 'training'
        with sqlite3.connect('file:' + str(t / 'batches.sqlite') + '?mode=ro', uri=True) as db:
            states = dict(db.execute('SELECT state,count(*) FROM batches GROUP BY state'))
            self.assertEqual(states.get('committed', 0), committed)
            self.assertEqual(sum(n for state, n in states.items() if state != 'committed'), unresolved)
        with sqlite3.connect('file:' + str(t / 'outbox.sqlite') + '?mode=ro', uri=True) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM batches').fetchone()[0], acknowledgements)
            self.assertEqual(db.execute('SELECT count(*) FROM emissions').fetchone()[0], acknowledgements)
            self.assertEqual(dict(db.execute('SELECT state,count(*) FROM emissions GROUP BY state')),
                             {'recorded': acknowledgements}, 'unknown outcomes stay unresolved')

    def assert_no_reservation_retry(self, root):
        def reopen():
            planner = self.planner(root)
            try:
                before = planner.verify()
                with self.assertRaisesRegex(ValueError, 'unresolved batch exists'):
                    planner.reserve(planner.summary()['checkpoint_sha256'])
                self.assertEqual(planner.verify(), before)
            finally:
                planner.close()
        self.child(reopen)

    def test_death_after_outbox_ack_refuses_resume_and_duplicate_reservation(self):
        root, g, resume = self.start()
        self.kill_runner(root, resume, 'ack')
        self.assert_cursor(root, committed=0, unresolved=1)
        self.assertEqual(list((root / 'training').glob('batch-*.json')), [])
        self.assertEqual(len(list((root / 'training' / 'rolling').iterdir())), 1)
        self.assert_no_reservation_retry(root)
        self.refused_restart(root, resume)

    def test_death_after_proof_before_commit_never_authenticates_exposure(self):
        root, g, resume = self.start()
        self.kill_runner(root, resume, 'proof')
        self.assert_cursor(root, committed=0, unresolved=1)
        self.assertTrue((root / 'training' / f'batch-{g["origin_absolute_step"] + 1:09d}.json').is_file())
        self.assert_no_reservation_retry(root)
        self.refused_restart(root, resume)

    def test_death_after_commit_keeps_one_exposure_but_requires_clean_closure(self):
        root, g, resume = self.start()
        self.kill_runner(root, resume, 'commit')
        self.assert_cursor(root, committed=1, unresolved=0)

        def readback():
            planner = self.planner(root)
            try:
                self.assertEqual(planner.verify()['committed_batches'], 1)
                CU.prefix_integrity(root / 'training', planner.db, g['origin_absolute_step'], 1)
                self.assertEqual(planner.db.execute('SELECT count(*) FROM events WHERE kind="batch.committed"')
                                 .fetchone()[0], 1)
            finally:
                planner.close()

        self.child(readback)
        self.refused_restart(root, resume)
        self.assert_cursor(root, committed=1, unresolved=0)

    def test_partial_checkpoint_after_optimizer_never_commits_or_resumes(self):
        root, g, resume = self.start()
        partial = root / 'training' / 'rolling' / f'{g["origin_absolute_step"] + 1:09d}'
        self.kill_runner(root, resume, 'partial-checkpoint')
        self.assertGreater((partial / 'state.safetensors').stat().st_size, 0)
        self.assertFalse((partial / 'manifest.json').exists())
        self.assert_cursor(root, committed=0, unresolved=1)
        self.assertEqual(list((root / 'training').glob('batch-*.json')), [])
        self.assert_no_reservation_retry(root)
        self.refused_restart(root, resume)


class BoundaryProcessDeathTests(ProcessFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BC.OverlayTests.setUpClass()

    def setUp(self):
        self.f = BC.OverlayTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def driver(self, ex, inbox):
        return R6.BoundaryDriver(inbox, self.f.w.native, root=ex.root,
                                 campaign_sha256=B_CAMPAIGN,
                                 preprocessing_sha256=self.f.w.A['preprocessing_sha256'],
                                 registry_sha256=self.f.w.A['parent_registry_sha256'],
                                 fresh_authority=self.f.w.auth, authority_check=self.f.w.role_check)

    def test_runner_boundary_death_during_cas_holds_after_cas_reconciles_once(self):
        for point in ('during-cas', 'after-cas'):
            with self.subTest(point=point):
                ex, inbox, _ = self.f.lane(point, adopt=0)
                self.f.publish(inbox, self.f.p1)
                ex.run_epoch()
                previous = ex.planner.verify()
                step = ex.step
                ex.close()

                def run():
                    fresh = Bd.Inbox(inbox.root, overlay=BC.OVERLAY)
                    reopened = Executor.restart(self.f.w, ex.root, fresh, step=step)
                    original = reopened.planner._event

                    def event(kind, body):
                        result = original(kind, body)
                        if point == 'during-cas' and kind == 'manifest.adopted': die()
                        return result

                    fresh.fault = lambda name: die() if name == 'after_cas' else None
                    with mock.patch.object(reopened.planner, '_event', side_effect=event):
                        self.driver(reopened, fresh).step(reopened.planner, reopened.snapshot,
                                                          reopened.ckpt, at_step=step)

                self.child(run, killed=True)
                self.assertFalse((ex.root / 'boundary-status.json').exists(), 'no returned adoption status')

                def readback():
                    fresh = Bd.Inbox(inbox.root, overlay=BC.OVERLAY)
                    reopened = Executor.restart(self.f.w, ex.root, fresh, step=step)
                    try:
                        driver = self.driver(reopened, fresh)
                        state = fresh.reconcile(reopened.planner)
                        events = [e for e in state['events'] if e['kind'] == 'manifest.adopted']
                        if point == 'during-cas':
                            self.assertEqual((state['records'], events), ([], []))
                            self.assertEqual([h['reason'] for h in state['holds']], ['INTENT_WITHOUT_PLANNER_CAS'])
                            self.assertEqual(reopened.planner.verify(), previous)
                            driver.step(reopened.planner, reopened.snapshot, reopened.ckpt, at_step=step)
                            self.assertEqual(driver.status['state'], 'held')
                        else:
                            self.assertEqual((len(state['records']), len(events)), (1, 1))
                            self.assertTrue(state['records'][0]['reconciled_after_crash'])
                            self.assertEqual(driver.seq, 1)
                            self.assertEqual(fresh.reconcile(reopened.planner), state)
                            before = reopened.planner.verify()
                            self.assertIs(driver.step(reopened.planner, reopened.snapshot,
                                                      reopened.ckpt, at_step=step), reopened.snapshot)
                            self.assertEqual(reopened.planner.verify(), before)
                            self.assertEqual(len(fresh.adoptions()), 1)
                    finally:
                        reopened.close()

                self.child(readback)

    def test_partial_overlay_pack_refuses_before_runner_adoption(self):
        ex, inbox, _ = self.f.lane('partial-pack', adopt=0)
        self.f.publish(inbox, self.f.p1)
        ex.run_epoch()
        step = ex.step
        ex.close()
        path = Path(self.f.p1['packet']['path'])
        original = path.read_bytes()

        def truncate():
            os.chmod(path, 0o600)  # fixture-only interrupted/corrupt referenced pack
            with path.open('wb') as out:
                out.write(original[:len(original) // 2]); out.flush(); os.fsync(out.fileno())
            die()

        self.child(truncate, killed=True)
        before = files(self.f.base)

        def readback():
            fresh = Bd.Inbox(inbox.root, overlay=BC.OVERLAY)
            reopened = Executor.restart(self.f.w, ex.root, fresh, step=step)
            try:
                prior = reopened.planner.verify()
                with self.assertRaises(ValueError): PK.completed_packet(path.parent, predecessor_head=None)
                with self.assertRaises(ValueError):
                    PK.validate_chain([self.f.p1['packet']], expected_head=self.f.p1['head_sha256'])
                with self.assertRaises(ValueError):
                    self.driver(reopened, fresh).step(reopened.planner, reopened.snapshot,
                                                      reopened.ckpt, at_step=step)
                self.assertEqual(reopened.planner.verify(), prior)
                self.assertEqual((fresh.adoptions(), fresh.holds()), ([], []))
                self.assertEqual(list((fresh.root / 'intents').iterdir()), [])
            finally:
                reopened.close()

        self.child(readback)
        self.assertEqual(files(self.f.base), before, 'partial pack kept; no adoption or rewrite')

    def test_death_at_first_selected_overlay_commit_refuses_or_counts_once(self):
        for after in (False, True):
            with self.subTest(after_commit=after):
                ex, inbox, _ = self.f.lane('first-selected-' + str(after), adopt=0)
                self.f.publish(inbox, self.f.p1)
                ex.run_epoch()
                ex.snapshot = self.driver(ex, inbox).step(ex.planner, ex.snapshot, ex.ckpt, at_step=ex.step)
                step = ex.step
                wanted = self.f.overlay_keys(1)
                ex.close()

                def run():
                    fresh = Bd.Inbox(inbox.root, overlay=BC.OVERLAY)
                    reopened = Executor.restart(self.f.w, ex.root, fresh, step=step)
                    original = reopened.planner.commit

                    def commit(*args, **kw):
                        proof = json.loads((ex.root / f'batch-{reopened.step:09d}.json').read_bytes())
                        selected = any((m['context_id'], row['program_sha256']) in wanted
                                       for m in proof['selection']['selection']['members']
                                       for row in m.get('selected', ()))
                        if selected and not after: die()
                        result = original(*args, **kw)
                        if selected: die()
                        return result

                    with mock.patch.object(reopened.planner, 'commit', side_effect=commit):
                        for _ in range(10): reopened.update()
                    reopened.close()  # no selected exposure => requested death was not reached

                self.child(run, killed=True)

                def readback():
                    raw, active = Bd.active_manifest(ex.root / 'batches.sqlite')
                    initial = json.loads((ex.root / 'ckpt/000000/manifest.json').read_bytes())
                    planner = BatchStream(ex.root / 'batches.sqlite', raw, active, seed=4, batch_size=2,
                                          initial_checkpoint_sha256=initial['parameter_sha256'])
                    try:
                        fresh = Bd.Inbox(inbox.root, overlay=BC.OVERLAY)
                        snapshots = {self.f.w.genesis.identity: self.f.w.genesis}
                        snapshots.update({r['snapshot_sha256']: self.f.w.load(fresh, r) for r in fresh.adoptions()})
                        proofs = sorted(ex.root.glob('batch-*.json'))
                        before = planner.verify()
                        if not after:
                            with self.assertRaisesRegex(ValueError, 'proof joins a committed planner batch'):
                                Bd.first_exposures(planner, fresh, proofs, snapshots)
                            with self.assertRaisesRegex(ValueError, 'unresolved batch exists'):
                                planner.reserve(planner.summary()['checkpoint_sha256'])
                        else:
                            exposure = Bd.first_exposures(planner, fresh, proofs, snapshots)
                            observed = [e for e in exposure['exposures'] if e['status'] == 'first']
                            self.assertGreater(len(observed), 0)
                            self.assertEqual(len({(e['context_id'], e['program_sha256']) for e in observed}), len(observed))
                            self.assertEqual(Bd.first_exposures(planner, fresh, proofs * 2, snapshots), exposure)
                            self.assertTrue(exposure['coverage']['complete'])
                        self.assertEqual(planner.verify(), before)
                        self.assertEqual(len(fresh.adoptions()), 1)
                    finally:
                        planner.close()

                self.child(readback)


def runner_child(params_path, mode):
    """Fresh interpreter entry point; private world inputs, genuine runner, no graceful closure."""
    params = json.loads(Path(params_path).read_bytes())
    fixture = V6Fixture()
    fixture.base = Path(params['root']).parent
    fixture.capture = Path(params['capture'])
    R6.torch.set_num_threads(1)  # same initialization as V6Fixture; run() applies its real controls
    # Rehydrate the same deterministic native fixture; the parent's captured
    # optimizer/RNG state is restored by the genuine runner. Rebuilding a v1
    # ancestor here only repeats unrelated training and hundreds of fsyncs.
    native, _ = W10.ContinuousTests().fixture()
    native.extension_raw = R6.canonical({'tasks': []})
    raw, identity = Bd.active_manifest(Path(params['root']) / 'training/batches.sqlite')
    manifest = json.loads(raw)
    assert {c['context_id']: c for c in manifest['contexts']} == native.contexts
    active = copy.copy(native)
    active.identity = manifest['snapshot_sha256']
    active.batch_manifest = manifest
    pre = (Path(params['root']) / 'training/preprocessing.json').read_bytes()
    assert json.loads(pre)['snapshot_sha256'] == native.identity
    assert R6.sha(R6.canonical(active.batch_manifest)) == identity
    fixture.w = types.SimpleNamespace(native=native, active=active, pre_raw=pre)
    if mode == 'partial-checkpoint':
        def action(tensors, path, *args, **kw):
            raw = save(tensors)
            with Path(path).open('xb') as out:
                out.write(raw[:len(raw) // 2]); out.flush(); os.fsync(out.fileno())
            die()
        patch = mock.patch.object(Storage, 'save_file', side_effect=action)
    else:
        cls, name = (R6.Outbox, 'record_batch') if mode == 'ack' else (R6.BatchStream, 'commit')
        original = getattr(cls, name)

        def action(*args, **kw):
            if mode != 'proof': original(*args, **kw)
            die()

        patch = mock.patch.object(cls, name, side_effect=action, autospec=True)
    with patch:
        fixture.run_v6(Path(params['root']), params['config'], params['resume'])
