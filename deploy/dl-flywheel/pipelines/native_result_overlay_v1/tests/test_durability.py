"""W03 private crash fixtures. Real publication, planner CAS and toy CPU checkpoints; no live qualification.

Children terminate with os._exit at the named boundary. The parent reads surviving bytes and real consumers;
it never replays an uncertain batch or silently republishes a failed root. These are process-death/I/O fixtures,
not power-loss or storage-hardware qualification.
"""
import json
import os
from pathlib import Path
import signal
import stat
import time
import traceback
import unittest
from unittest import mock

from pipelines.native_result_overlay_v1 import build as B
from pipelines.native_result_overlay_v1 import common as C
from pipelines.native_result_overlay_v1 import packet as PK
from pipelines.native_result_overlay_v1.tests import test_overlay as O
from pipelines.boundary_admission_v2 import boundary as Bd
from pipelines.boundary_admission_v2.tests import test_overlay_components as BC
from pipelines.boundary_admission_v2.tests.harness import Executor
import neural_storage as Storage


class CrashWorld(unittest.TestCase):
    def die(self, call):
        """One private child, no graceful cleanup, no retry; bound hangs and retain their failure."""
        pid = os.fork()
        if pid == 0:
            try:
                call()
            except BaseException:
                traceback.print_exc()
                os._exit(1)
            os._exit(2)  # the requested boundary must actually have been reached
        deadline = time.monotonic() + 30
        while True:
            ended, status = os.waitpid(pid, os.WNOHANG)
            if ended:
                self.assertTrue(os.WIFEXITED(status))
                self.assertEqual(os.WEXITSTATUS(status), 86)
                return
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                self.fail('private crash child exceeded 30 seconds')
            time.sleep(.01)


class PublicationDurabilityTests(CrashWorld):
    def setUp(self):
        self.f = O.OverlayTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def returned(self, name):
        row = self.f.w.row('B', self.f.w.result('B', '(pipe X (ridge alpha=321.0))'))
        r = self.f.w.build([row], name, accept=False)
        return r, Path(r['packet']['path']).parent

    def accept(self, r, root):
        return PK.accept_publication(root, physical={'kind': 'in_process_return',
                                                    'returned_sha256': r['marker']['sha256']})

    def refused(self, root):
        self.assertEqual({k: v[0] for k, v in self.f.consumers(root).items()},
                         {'completed_packet': False, 'validate_chain': False})

    def custody(self, root):
        paths = list(root.rglob('*')) + list(root.parent.glob('*' + root.name + '*'))
        return {str(p): p.read_bytes() for p in paths if p.is_file()}

    def no_retry(self, r, root):
        before = self.custody(root)
        with self.assertRaisesRegex(ValueError, 'create-only acceptance'):
            self.accept(r, root)
        self.assertEqual(self.custody(root), before)

    def test_complete_accepted_envelope_and_failed_caller_parent_fsync_stays_inadmissible(self):
        r, root = self.returned('caller-final-fsync')
        original = self.custody(root)
        events = []

        def fail(path):
            self.assertEqual(Path(path), root.parent)
            raw = C.acceptance_path(root).read_bytes()
            self.assertEqual(json.loads(raw)['verdict'], 'accepted')
            self.assertEqual(json.loads(raw)['marker'], r['marker'])
            events.append(C.sha(raw))
            raise OSError('actual caller parent fsync failed after complete accepted envelope')

        with mock.patch.object(PK, 'fsync_dir', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'actual caller parent fsync'):
                self.accept(r, root)
        self.assertEqual(len(events), 1)
        self.assertFalse(PK.completion_path(root).exists())
        self.refused(root)
        self.assertTrue(all(self.custody(root)[p] == raw for p, raw in original.items()))
        self.no_retry(r, root)

    def test_pending_completion_file_fsync_failure_is_not_a_completion(self):
        r, root = self.returned('marker-file-fsync')
        real = PK.create_only

        def create(path, raw):
            if str(path).endswith('.pending'):
                with mock.patch.object(C.os, 'fsync', side_effect=OSError('completion file fsync')):
                    return real(path, raw)
            return real(path, raw)

        with mock.patch.object(PK, 'create_only', side_effect=create):
            with self.assertRaisesRegex(OSError, 'completion file fsync'):
                self.accept(r, root)
        self.refused(root)
        self.assertEqual(len(list(root.parent.glob('.*acceptance-complete*.pending'))), 1)
        self.no_retry(r, root)

    def test_pending_completion_directory_fsync_failure_is_not_a_completion(self):
        r, root = self.returned('marker-dir-fsync')
        real, calls = PK.fsync_dir, []

        def fsync(path):
            calls.append(path)
            if len(calls) == 2:
                raise OSError('pending completion directory fsync')
            real(path)

        with mock.patch.object(PK, 'fsync_dir', side_effect=fsync):
            with self.assertRaisesRegex(OSError, 'pending completion directory fsync'):
                self.accept(r, root)
        self.assertEqual(len(calls), 2)
        self.refused(root)
        self.no_retry(r, root)

    def test_reader_during_every_caller_write_and_fsync_refuses_until_atomic_completion(self):
        r, root = self.returned('interleaved-reader')
        real_create, real_sync, real_link = PK.create_only, PK.fsync_dir, PK.os.link
        events = []

        def create(path, raw):
            self.refused(root)
            ref = real_create(path, raw)
            self.refused(root)
            events.append('create:' + Path(path).name)
            return ref

        def fsync(path):
            self.refused(root)
            real_sync(path)
            self.refused(root)
            events.append('directory-fsync')

        def link(src, dst, **kw):
            self.refused(root)
            self.assertEqual(stat.S_IMODE(Path(src).stat().st_mode), 0o444)
            real_link(src, dst, **kw)
            self.assertEqual(self.f.consumers(root), {'completed_packet': (True, 1), 'validate_chain': (True, 1)})
            events.append('atomic-completion')

        with mock.patch.object(PK, 'create_only', side_effect=create), \
                mock.patch.object(PK, 'fsync_dir', side_effect=fsync), mock.patch.object(PK.os, 'link', side_effect=link):
            self.accept(r, root)
        self.assertEqual(len(events), 5)
        self.assertEqual(events[-1], 'atomic-completion')

    def test_process_death_before_and_after_every_caller_publication_boundary(self):
        for boundary in ('envelope', 'envelope-dir', 'completion-file', 'completion-dir', 'before-link', 'after-link'):
            with self.subTest(boundary=boundary):
                r, root = self.returned('death-' + boundary)
                original = self.custody(root)

                def child():
                    create0, sync0, link0 = PK.create_only, PK.fsync_dir, PK.os.link
                    syncs = []

                    def create(path, body):
                        ref = create0(path, body)
                        if boundary == ('completion-file' if str(path).endswith('.pending') else 'envelope'):
                            os._exit(86)
                        return ref

                    def sync(path):
                        sync0(path)
                        syncs.append(path)
                        if boundary == ('envelope-dir' if len(syncs) == 1 else 'completion-dir'):
                            os._exit(86)

                    def link(src, dst, **kw):
                        if boundary == 'before-link':
                            os._exit(86)
                        link0(src, dst, **kw)
                        if boundary == 'after-link':
                            os._exit(86)

                    with mock.patch.object(PK, 'create_only', side_effect=create), \
                            mock.patch.object(PK, 'fsync_dir', side_effect=sync), \
                            mock.patch.object(PK.os, 'link', side_effect=link):
                        self.accept(r, root)

                self.die(child)
                self.assertTrue(all(self.custody(root)[p] == raw for p, raw in original.items()))
                if boundary == 'after-link':
                    self.assertEqual(self.f.consumers(root), {'completed_packet': (True, 1), 'validate_chain': (True, 1)})
                else:
                    self.refused(root)
                self.no_retry(r, root)

    def test_process_death_before_builder_publication_and_partial_pack_keep_custody(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                name = 'partial-pack' if partial else 'before-builder-publication'
                root = self.f.root / name
                row = self.f.w.row('B', self.f.w.result('B', '(pipe X (ridge alpha=322.0))'))

                def child():
                    def write(path, body):
                        self.assertEqual(Path(path), root / 'packet.json')
                        if partial:
                            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                            os.write(fd, body[:len(body) // 2])
                        os._exit(86)
                    with mock.patch.object(B, 'create_only', side_effect=write):
                        self.f.w.build([row], name, accept=False)
                self.die(child)
                self.assertTrue(root.is_dir())
                self.refused(root)
                before = self.custody(root)
                with self.assertRaises(ValueError):
                    self.f.w.build([row], name, accept=False)
                self.assertEqual(self.custody(root), before)
                self.assertEqual((root / 'packet.json').exists(), partial)

    def test_completion_binding_tampering_and_duplicate_delivery_are_refused_without_rewrite(self):
        r, root = self.returned('completion-binding')
        self.accept(r, root)
        self.no_retry(r, root)
        path = PK.completion_path(root)
        body = path.read_bytes()
        for key, value in (('root', str(root.parent)), ('acceptance', r['marker']), ('bound', {}),
                           ('schema', 'unqualified'), ('completion_sample', {})):
            with self.subTest(key=key):
                doc = json.loads(body)
                doc[key] = value
                os.unlink(path)  # fixture tamper only; production never changes original bytes
                path.write_bytes(C.canonical(doc))
                self.refused(root)

    def test_deadline_crossed_during_completion_fsync_retains_original_bound_and_refuses(self):
        r, root = self.returned('late-completion')
        bound = json.loads(Path(r['marker']['path']).read_bytes())['bound']
        real, samples = PK.clock_sample, []

        def clock():
            samples.append(1)
            return real() if len(samples) < 3 else {**real(), 'monotonic': bound['monotonic_deadline']}
        with mock.patch.object(PK, 'clock_sample', side_effect=clock):
            with self.assertRaisesRegex(ValueError, 'completion outside the original bound'):
                self.accept(r, root)
        self.assertEqual(json.loads(C.acceptance_path(root).read_bytes())['bound'], bound)
        self.refused(root)
        self.no_retry(r, root)


class LearningBoundaryDurabilityTests(CrashWorld):
    @classmethod
    def setUpClass(cls):
        BC.OverlayTests.setUpClass()

    def setUp(self):
        self.f = BC.OverlayTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def lane(self, name, adopt=False):
        ex, inbox, _ = self.f.lane(name, adopt=int(adopt))
        if not adopt:
            self.f.publish(inbox, self.f.p1)
        return ex, inbox

    def exposures(self, ex, inbox, proofs=None):
        snapshots = {self.f.w.genesis.identity: self.f.w.genesis}
        snapshots.update({r['snapshot_sha256']: self.f.w.load(inbox, r) for r in inbox.adoptions()})
        return Bd.first_exposures(ex.planner, inbox, ex.proofs() if proofs is None else proofs, snapshots)

    def test_death_after_claim_before_and_after_checkpoint_fsync_and_after_manifest_write_never_commits(self):
        for boundary in ('claim', 'before-checkpoint-fsync', 'after-checkpoint-fsync', 'manifest-write'):
            with self.subTest(boundary=boundary):
                ex, inbox = self.lane('checkpoint-' + boundary)
                prior = ex.planner.summary()

                def child():
                    reserve0, fsync0, create0 = ex.planner.reserve, Storage.os.fsync, Storage.create

                    def reserve(*a, **kw):
                        out = reserve0(*a, **kw)
                        if boundary == 'claim':
                            os._exit(86)
                        return out

                    def fsync(fd):
                        target = os.readlink('/proc/self/fd/' + str(fd))
                        checkpoint = target.endswith('/000001/state.safetensors')
                        if checkpoint and boundary == 'before-checkpoint-fsync':
                            os._exit(86)
                        fsync0(fd)
                        if checkpoint and boundary == 'after-checkpoint-fsync':
                            os._exit(86)

                    def create(path, body):
                        out = create0(path, body)
                        if boundary == 'manifest-write' and Path(path).name == 'manifest.json':
                            os._exit(86)
                        return out

                    with mock.patch.object(ex.planner, 'reserve', side_effect=reserve), \
                            mock.patch.object(Storage.os, 'fsync', side_effect=fsync), \
                            mock.patch.object(Storage, 'create', side_effect=create):
                        ex.update()
                self.die(child)
                current = ex.planner.summary()
                self.assertEqual(current['checkpoint_sha256'], prior['checkpoint_sha256'])
                self.assertEqual((current['batch_count'], current['unique_exposed_data_runs']), (0, 0))
                self.assertIsNotNone(current['pending_batch'])
                self.assertEqual(ex.proofs(), [])
                self.assertEqual(inbox.adoptions(), [])
                with self.assertRaisesRegex(ValueError, 'unresolved|pending|reconcile'):
                    ex.planner.reserve(ex.ckpt)
                if boundary != 'claim':
                    self.assertTrue((ex.root / 'ckpt/000001/state.safetensors').is_file())
                if boundary == 'manifest-write':
                    self.assertTrue((ex.root / 'ckpt/000001/manifest.json').is_file())

    def test_death_during_cas_rolls_back_and_after_cas_reconciles_exactly_once(self):
        for boundary in ('during-cas', 'after-cas'):
            with self.subTest(boundary=boundary):
                ex, inbox = self.lane(boundary)
                ex.run_epoch()
                original_manifest = ex.planner.summary()['manifest_sha256']

                def child():
                    event0 = ex.planner._event
                    def event(kind, body):
                        out = event0(kind, body)
                        if kind == 'manifest.adopted' and boundary == 'during-cas':
                            os._exit(86)
                        return out
                    inbox.fault = lambda point: os._exit(86) if point == 'after_cas' else None
                    with mock.patch.object(ex.planner, '_event', side_effect=event):
                        ex.boundary()
                self.die(child)
                state = inbox.reconcile(ex.planner)
                self.assertEqual(len(list((inbox.root / 'intents').iterdir())), 1)
                if boundary == 'during-cas':
                    self.assertEqual(ex.planner.summary()['manifest_sha256'], original_manifest)
                    self.assertEqual(state['records'], [])
                    self.assertEqual(len(state['holds']), 1)
                    with self.assertRaises(Bd.Held):
                        ex.boundary()
                else:
                    self.assertEqual(len(state['records']), 1)
                    self.assertTrue(state['records'][0]['reconciled_after_crash'])
                    self.assertEqual(inbox.reconcile(ex.planner)['records'], state['records'])
                    self.assertEqual(len([e for e in Bd.planner_events(ex.planner)[1]
                                          if e['kind'] == 'manifest.adopted']), 1)

    def test_death_before_first_exposure_has_no_commit_and_after_first_exposure_is_deduplicated(self):
        for after in (False, True):
            with self.subTest(after=after):
                ex, inbox = self.lane('exposure-' + str(after), adopt=True)
                prior = ex.planner.summary()['batch_count']
                def child():
                    commit0 = ex.planner.commit
                    def commit(*a, **kw):
                        if not after:
                            os._exit(86)
                        return commit0(*a, **kw)
                    with mock.patch.object(ex.planner, 'commit', side_effect=commit):
                        for _ in range(10):
                            ex.update()
                            out = self.exposures(ex, inbox)
                            if any(e['status'] == 'first' for e in out['exposures']):
                                os._exit(86)
                self.die(child)
                if not after:
                    self.assertEqual(ex.planner.summary()['batch_count'], prior)
                    with self.assertRaises(ValueError):
                        self.exposures(ex, inbox)
                    # The uncommitted proof stays as custody; it is not an exposure or a replay permit.
                    self.assertTrue((ex.root / f'batch-{ex.step + 1:09d}.json').exists())
                else:
                    out = self.exposures(ex, inbox)
                    self.assertTrue(any(e['status'] == 'first' for e in out['exposures']))
                    self.assertEqual(self.exposures(ex, inbox, ex.proofs() * 2), out)
