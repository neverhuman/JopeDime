"""C10b revision 6 private CPU fixtures: native-result overlay components through the boundary inbox.

A candidate may append W3 overlay packets to the active chain, publish a new variant root, or both; the adopted
snapshot is the composite (variant base E, overlay head H).  ``OverlayTests`` drives real W3 revision-2 packets built
on W3's synthetic native world (byte-exact copy of W3 DELIVERY-r2 in this tree) through real BatchStream planners,
PolicyLM/AdamW updates and physical checkpoints: overlay-only adoptions at two closed epochs, first exposures joined
to the overlay window receipts, restart parity and zero-mutation refusals.  ``MixedTests`` exercises the variant +
overlay state machine on the C10a fixture world with a stub chain interface (real head folding, empty rows), because
the two fixture worlds do not share a native base.  A W3 root whose final publication crossed its original deadline
(kept for custody with its refusal record) is refused at publication, adoption and restart.  No live data, outbox, S3,
GPU or program execution.
"""
import copy
import json
import os
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load

from jope_flywheel.program_io import sha
from jope_flywheel.protocol import canonical, digest
from pipelines.boundary_admission_v2 import boundary as Bd
from pipelines.boundary_admission_v2.tests.harness import CAMPAIGN, Executor, World, new_inbox
from pipelines.generated_target_preparation_v2.tests import world as CW
from pipelines.native_result_overlay_v1 import build as OB
from pipelines.native_result_overlay_v1 import loader as OL
from pipelines.native_result_overlay_v1 import packet as PK
from pipelines.native_result_overlay_v1.tests import world as W3
import test_native_variant_snapshot as variant

OVERLAY = types.SimpleNamespace(validate_chain=PK.validate_chain, head_from_sha=PK.head_from_sha,
                                overlay_snapshot=OL.overlay_snapshot)
B_PROGRAM = ('(pipe X (ridge alpha=3.0))', .005)
C_PROGRAM = ('(pipe X (ridge alpha=5.0))', .004)


def replace(path, body):
    """Fixture-only tamper: swap a create-only file for new bytes."""
    path = Path(path)
    os.unlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    os.write(fd, body)
    os.close(fd)


def after(*receipts):
    return {'predecessor_chain': [r['packet'] for r in receipts], 'predecessor_head': receipts[-1]['head_sha256']}


def files(root):
    return sorted((str(p.relative_to(root)), sha(p.read_bytes())) for p in Path(root).rglob('*.json'))


class OverlayWorld:
    """Harness-Executor world over W3's synthetic native corpus; the native snapshot is the genesis variant base E0.

    The fixture role check accepts only the native spec bytes under a proof younger than 60 s (the real authority
    belongs to the runner; C10 r2-r5 fixtures cover it on the variant world).  W3's context D carries a deliberately
    disagreeing shard baseline (W3's BASELINE_DISAGREEMENT fixture) that the frozen training loader refuses; this
    world anchors D's shard baseline exactly so every train context can be trained."""

    def __init__(self, base):
        (Path(base) / 'w3').mkdir()
        specs = [(n, role, rows, 'exact' if n == 'D' else mode, winner) for n, role, rows, mode, winner in W3.SPECS]
        with mock.patch.object(W3, 'SPECS', specs):
            self.w3 = W3.World(Path(base) / 'w3')
        self.native = self.genesis = self.w3.nat
        self.genesis.extension_raw = self.w3.spec_bytes
        self.A = {'preprocessing_sha256': sha(self.w3.pre), 'parent_registry_sha256': sha(self.w3.registry)}
        self.contract = CW.contract(self.w3.pre)

    def auth(self, *, at=None):
        return canonical({'schema': 'jope.c10-r6-fixture-role-proof/v1',
                          'checked_at_unix': time.time() if at is None else at})

    def role_check(self, raw, proof=None):
        doc = json.loads(proof or self.auth())
        if raw != self.w3.spec_bytes:
            raise ValueError('fixture role check: extension bytes are not the native spec')
        if not 0 <= time.time() - doc['checked_at_unix'] <= 60:
            raise ValueError('fixture role check: stale authority proof')

    def load(self, inbox, record):
        return Bd.load_adopted(inbox, record, self.native, genesis=self.genesis,
                               preprocessing_sha256=self.A['preprocessing_sha256'],
                               registry_sha256=self.A['parent_registry_sha256'],
                               authority_check=lambda raw: self.role_check(raw))

    def close(self):
        self.w3.outbox.close()


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.w = OverlayWorld(self.base)
        self.addCleanup(self.w.close)
        w3 = self.w.w3
        A, Bc = w3.ctx['A'], w3.ctx['B']
        a = [w3.result('A', f'(pipe X (ridge alpha={700 + j}.0))', [b * (0.40 + 0.03 * j) for b in A['recon']])
             for j in range(5)]
        b = [w3.result('B', f'(pipe X (ridge alpha={800 + j}.0))', [v * (0.50 + 0.03 * j) for v in Bc['recon']])
             for j in range(3)]
        self.rows1 = [w3.row('A', r) for r in a[:3]] + [w3.row('B', r) for r in b[:2]]
        self.rows2 = [w3.row('A', r) for r in a[3:]] + [w3.row('B', b[2])]
        self.p1 = w3.build(self.rows1, 'packet-1')
        self.p2 = w3.build(self.rows2, 'packet-2', **after(self.p1))

    def inbox(self, name, **kw):
        return new_inbox(self.w, self.base / (name + '-inbox'), overlay=kw.pop('overlay', OVERLAY), **kw)

    def publish(self, inbox, receipt):
        return inbox.publish(overlay_packets=[receipt['packet']], overlay_receipt_sha256=receipt['receipt']['sha256'])

    def lane(self, name, *, adopt=2):
        inbox = self.inbox(name)
        ex = Executor(self.w, self.base / name, inbox)
        self.addCleanup(ex.close)
        adopted = []
        for receipt in (self.p1, self.p2)[:adopt]:
            ex.run_epoch()
            self.publish(inbox, receipt)
            adopted.append(ex.boundary())
        return ex, inbox, adopted

    def overlay_keys(self, seq):
        """(context, program) keys of the rows packet ``seq`` adds to the chain."""
        receipts = (self.p1, self.p2)[:seq]
        view = OVERLAY.validate_chain([r['packet'] for r in receipts], expected_head=receipts[-1]['head_sha256'])
        return {(r.context_id, r.record.program_sha256) for r in view.rows if r.packet_seq == seq}

    # ------------------------------------------------------------------ overlay-only adoptions end to end
    def test_r6_overlay_only_adoptions_compose_and_first_exposures(self):
        ex, inbox, (s1, s2) = self.lane('main')
        nat = self.w.genesis
        for s, receipt, seq in ((s1, self.p1, 1), (s2, self.p2, 2)):
            self.assertEqual(s.identity, Bd.composite_identity(nat.identity, receipt['head_sha256']))
            self.assertEqual(s.overlay_head_sha256, receipt['head_sha256'])
            self.assertEqual(s.overlay_packet_seq, seq)
            self.assertIs(s.base_variant, nat)
            self.assertEqual(s.batch_manifest, {**nat.batch_manifest, 'snapshot_sha256': s.identity})
            self.assertEqual(s.contexts, nat.contexts)
        env2 = inbox.candidate(2)[1]
        self.assertIsNone(env2['components']['variant'])
        self.assertEqual(env2['state'], {'variant_extension_sha256': nat.identity, 'variant_source_seq': 0,
                                         'overlay_head_sha256': self.p2['head_sha256'],
                                         'overlay_packets': [self.p1['packet'], self.p2['packet']]})
        self.assertEqual(env2['components']['native_overlay']['predecessor_head_sha256'], self.p1['head_sha256'])
        raw, active = Bd.active_manifest(ex.root / 'batches.sqlite')
        self.assertEqual(active, inbox.adoptions()[-1]['manifest_after'])
        self.assertEqual(json.loads(raw)['snapshot_sha256'], s2.identity)
        ex.run_epoch()
        snapshots = {nat.identity: nat}
        for record in inbox.adoptions():
            loaded = self.w.load(inbox, record)
            self.assertEqual(loaded.identity, record['snapshot_sha256'])
            snapshots[loaded.identity] = loaded
        self.assertEqual(snapshots[s2.identity].overlay_records, s2.overlay_records)
        out = Bd.first_exposures(ex.planner, inbox, ex.proofs(), snapshots)
        want1, want2 = self.overlay_keys(1), self.overlay_keys(2)
        self.assertEqual((len(want1), len(want2)), (5, 3))
        got = {(e['context_id'], e['program_sha256']): e for e in out['exposures']}
        self.assertEqual(set(got), want1 | want2)
        self.assertEqual(out['pending'], [])
        self.assertTrue(out['coverage']['complete'])
        for key, e in got.items():
            seq = 1 if key in want1 else 2
            self.assertEqual((e['adoption_seq'], e['exposed_under_seq'], e['status'], e['component']),
                             (seq, seq, 'first', 'native_overlay'))
            self.assertEqual(e['result_ref_sha256'], OL.overlay_ref_digest(snapshots[e['snapshot_sha256']], *key))
            self.assertIn('q', e['terms'])
            self.assertNotIn('ce', e['terms'])
            proof = json.loads(next(p for p in ex.proofs()
                                    if sha(p.read_bytes()) == e['proof_sha256']).read_bytes())
            member = next(m for m in proof['selection']['selection']['members'] if m['context_id'] == key[0])
            self.assertIn(key[1], {x['program_sha256'] for x in member['selected']})

    def test_r6_overlay_restart_reloads_composite_and_next_update_is_bitwise_identical(self):
        uninterrupted, _, _ = self.lane('branch-1')
        step = uninterrupted.step
        uninterrupted.update()
        interrupted, inbox, _ = self.lane('branch-2')
        interrupted.close()
        restarted = Executor.restart(self.w, interrupted.root, Bd.Inbox(inbox.root, overlay=OVERLAY), step=step)
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.snapshot.identity, uninterrupted.snapshot.identity)
        self.assertEqual(restarted.snapshot.overlay_head_sha256, self.p2['head_sha256'])
        restarted.update()
        left = load((uninterrupted.root / 'ckpt' / f'{step + 1:06d}' / 'state.safetensors').read_bytes())
        right = load((restarted.root / 'ckpt' / f'{step + 1:06d}' / 'state.safetensors').read_bytes())
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)
        a = json.loads((uninterrupted.root / f'batch-{step + 1:09d}.json').read_bytes())
        b = json.loads((restarted.root / f'batch-{step + 1:09d}.json').read_bytes())
        self.assertEqual(a['selection'], b['selection'])
        self.assertTrue(any(m['selected'] for m in a['selection']['selection']['members']))
        restarted.close()
        with self.assertRaisesRegex(ValueError, 'overlay package required'):
            Executor.restart(self.w, interrupted.root, Bd.Inbox(inbox.root), step=step)

    def test_r6_overlay_refusals_mutate_nothing(self):
        bare = self.inbox('bare', overlay=None)
        with self.assertRaisesRegex(ValueError, 'overlay package required'):
            self.publish(bare, self.p1)
        inbox = self.inbox('refuse')
        before = files(inbox.root)
        with self.assertRaisesRegex(ValueError, 'at least one new component'):
            inbox.publish()
        with self.assertRaisesRegex(ValueError, 'chain sequence and predecessor head link'):
            self.publish(inbox, self.p2)                          # does not extend the (empty) active chain
        self.assertEqual(files(inbox.root), before)
        ex = Executor(self.w, self.base / 'refuse', inbox)
        self.addCleanup(ex.close)
        ex.run_epoch()
        self.publish(inbox, self.p1)
        with self.assertRaisesRegex(ValueError, 'chain sequence and predecessor head link'):
            self.publish(inbox, self.p1)                          # the same packet twice
        candidate = inbox.root / 'candidates' / '00000001.json'
        good = candidate.read_bytes()
        env = json.loads(good)
        planner_before = ex.planner.verify()
        forged = [('candidate result snapshot is its composite identity',
                   lambda e: e.update(result_snapshot_sha256=e['state']['variant_extension_sha256'])),
                  ('overlay chain head and cumulative packets',
                   lambda e: e['state'].update(overlay_head_sha256=digest('forged-head'))),
                  ('overlay component shape and predecessor head',
                   lambda e: e['components']['native_overlay'].update(predecessor_head_sha256=digest('x')))]
        for message, mutate in forged:
            doc = copy.deepcopy(env)
            mutate(doc)
            replace(candidate, canonical(doc))
            with self.assertRaisesRegex(ValueError, message):
                ex.boundary()
        replace(candidate, good)
        packet = Path(self.p1['packet']['path'])
        packet_raw = packet.read_bytes()
        replace(packet, packet_raw.replace(b'"seq":1', b'"seq":2', 1))
        with self.assertRaisesRegex(ValueError, 'pinned size|sha256|exact'):
            ex.boundary()
        replace(packet, packet_raw)
        self.assertEqual(ex.planner.verify(), planner_before)
        self.assertEqual(sorted(p.name for p in (inbox.root / 'intents').iterdir()), [])
        self.assertEqual(inbox.holds(), [])
        stale = lambda: self.w.auth(at=time.time() - 3600)
        with self.assertRaisesRegex(ValueError, 'stale authority proof'):
            ex.boundary(fresh_authority=stale)
        self.assertEqual(ex.planner.verify(), planner_before)
        self.assertIsNotNone(ex.boundary())                       # the untouched candidate still adopts
        self.assertEqual(ex.snapshot.overlay_head_sha256, self.p1['head_sha256'])

    def test_r6_overlay_inbox_requires_in_time_terminal_acceptance(self):
        """CX-REV PR292 P2 at the inbox: a late W3 root is refused at publication; an in-time packet whose root later
        gains a refusal record (its acceptance publication was late) is refused at adoption and restart."""
        w3 = self.w.w3
        row = w3.row('A', w3.result('A', '(pipe X (ridge alpha=790.0))', [b * 0.47 for b in w3.ctx['A']['recon']]))
        wall = [time.time()]
        deadline = wall[0] + 600
        real_fsync = OB.fsync_dir

        def fsync(path):                     # the REAL directory fsync, then the clock reaches the original deadline
            real_fsync(path)
            wall[0] = deadline
        with mock.patch.object(OB.time, 'time', side_effect=lambda: wall[0]), \
                mock.patch.object(OB, 'fsync_dir', side_effect=fsync):
            with self.assertRaisesRegex(ValueError, 'deadline crossed after durable marker'):
                w3.build([row], 'late-1', deadline=deadline)
        marker = json.loads((w3.root / 'late-1' / 'overlay-complete.json').read_bytes())
        late = {'packet': marker['packet'], 'receipt': marker['receipt'], 'head_sha256': marker['head_sha256']}
        inbox = self.inbox('late')
        before = files(inbox.root)
        with self.assertRaisesRegex(ValueError, 'late terminal record kept for custody'):
            self.publish(inbox, late)
        self.assertEqual(files(inbox.root), before)
        ex = Executor(self.w, self.base / 'late', inbox)
        self.addCleanup(ex.close)
        ex.run_epoch()
        self.publish(inbox, self.p1)                              # in time: publishes
        refused = Path(self.p1['packet']['path']).parent / 'overlay-refused.json'
        refused.write_bytes(canonical({'fixture': 'late acceptance publication'}))
        planner_before = ex.planner.verify()
        with self.assertRaisesRegex(ValueError, 'late terminal record kept for custody'):
            ex.boundary()
        self.assertEqual(ex.planner.verify(), planner_before)
        self.assertEqual(sorted(p.name for p in (inbox.root / 'intents').iterdir()), [])
        self.assertEqual(inbox.holds(), [])
        os.unlink(refused)
        adopted = ex.boundary()                                   # the in-time terminal acceptance adopts
        self.assertEqual(adopted.overlay_head_sha256, self.p1['head_sha256'])
        record = inbox.adoptions()[-1]
        self.assertEqual(self.w.load(inbox, record).identity, adopted.identity)
        refused.write_bytes(canonical({'fixture': 'late acceptance publication'}))
        with self.assertRaisesRegex(ValueError, 'late terminal record kept for custody'):
            self.w.load(inbox, record)                            # restart rehydration
        os.unlink(refused)
        self.assertEqual(self.w.load(inbox, record).identity, adopted.identity)


class StubChain:
    """Stub of the overlay chain interface for the variant + overlay state machine: real head folding over
    {path, sha256, bytes} refs, an empty cumulative row set, a shallow composite copy of the variant base."""
    head_from_sha = staticmethod(PK.head_from_sha)

    @staticmethod
    def validate_chain(refs, *, expected_head):
        head = None
        for ref in refs:
            head = PK.head_from_sha(ref['sha256'], head)
        if head != expected_head:
            raise ValueError('stub chain head mismatch')
        return types.SimpleNamespace(rows=[], head_sha256=head, seq=len(refs), packets=list(refs))

    @staticmethod
    def overlay_snapshot(base, view, *, max_padded_tokens):
        s = copy.copy(base)
        s.overlay_records, s.overlay_head_sha256, s.overlay_packet_seq = {}, view.head_sha256, view.seq
        return s


def stub_ref(name):
    body = canonical({'stub-overlay-packet': name})
    return {'path': '/nonexistent/stub/' + name, 'sha256': sha(body), 'bytes': len(body)}


class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        variant.VariantTests.setUpClass()
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.w = World(self.base)
        self.Bspec, refs_b = self.w.append(self.w.A, B_PROGRAM)
        self.rb = self.w.root(self.w.A, self.Bspec, refs_b, 'prep-b')
        self.Cspec, self.refs_c = self.w.append(self.Bspec, C_PROGRAM)
        self.rc = self.w.root(self.Bspec, self.Cspec, self.refs_c, 'prep-c')

    def test_r6_variant_then_overlay_then_both_keep_each_component(self):
        inbox = new_inbox(self.w, self.base / 'mixed-inbox', overlay=StubChain)
        e1 = inbox.publish(**self.w.variant(self.rb), predecessor_extension_raw=canonical(self.w.A))
        q1, q2 = stub_ref('p1'), stub_ref('p2')
        e2 = inbox.publish(overlay_packets=[q1], overlay_receipt_sha256=digest('r1'))
        with self.assertRaisesRegex(ValueError, 'candidate predecessor is the previous published snapshot'):
            inbox.publish(**self.w.variant(self.rc), predecessor_extension_raw=canonical(self.w.A),
                          overlay_packets=[q2], overlay_receipt_sha256=digest('r2'))
        e3 = inbox.publish(**self.w.variant(self.rc), predecessor_extension_raw=canonical(self.Bspec),
                           overlay_packets=[q2], overlay_receipt_sha256=digest('r2'))
        h1 = PK.head_from_sha(q1['sha256'], None)
        h2 = PK.head_from_sha(q2['sha256'], h1)
        eb, ec = e1['state']['variant_extension_sha256'], e3['state']['variant_extension_sha256']
        self.assertEqual(e1['result_snapshot_sha256'], eb)
        self.assertEqual((e2['state']['variant_source_seq'], e2['result_snapshot_sha256']),
                         (1, Bd.composite_identity(eb, h1)))
        self.assertEqual((e3['state']['variant_source_seq'], e3['state']['overlay_packets'], e3['result_snapshot_sha256']),
                         (3, [q1, q2], Bd.composite_identity(ec, h2)))
        self.assertEqual(e3['predecessor_snapshot_sha256'], e2['result_snapshot_sha256'])
        ex = Executor(self.w, self.base / 'mixed', inbox)
        self.addCleanup(ex.close)
        snaps = []
        for _ in range(3):
            ex.run_epoch()
            snaps.append(ex.boundary())
        s1, s2, s3 = snaps
        self.assertEqual(s1.identity, eb)
        self.assertEqual((s2.identity, s2.base_variant.identity), (Bd.composite_identity(eb, h1), eb))
        self.assertEqual((s3.identity, s3.base_variant.identity), (Bd.composite_identity(ec, h2), ec))
        self.assertEqual(s3.batch_manifest, {**s3.base_variant.batch_manifest, 'snapshot_sha256': s3.identity})
        ex.run_epoch()
        records = inbox.adoptions()
        snapshots = {self.w.genesis.identity: self.w.genesis}
        for record in records:
            loaded = self.w.load(inbox, record)
            self.assertEqual(loaded.identity, record['snapshot_sha256'])
            snapshots[loaded.identity] = loaded
        self.assertEqual(snapshots[s3.identity].base_variant.identity, ec)
        out = Bd.first_exposures(ex.planner, inbox, ex.proofs(), snapshots)
        got = {(e['context_id'], e['program_sha256']): e for e in out['exposures']}
        c_key = (self.rc['rows'][0]['context_id'], self.rc['rows'][0]['program_sha256'])
        self.assertEqual((got[c_key]['adoption_seq'], got[c_key]['component']), (3, 'variant'))
        self.assertTrue(all(e['component'] == 'variant' for e in got.values()))
        restarted_inbox = Bd.Inbox(inbox.root, overlay=StubChain)
        ex.close()
        restarted = Executor.restart(self.w, ex.root, restarted_inbox, step=ex.step)
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.snapshot.identity, s3.identity)


if __name__ == '__main__':
    unittest.main()
