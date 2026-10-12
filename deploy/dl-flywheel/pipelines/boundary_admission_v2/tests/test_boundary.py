"""C10b revision 2 private CPU fixtures: repeatable closed-epoch adoption, refusals, crash reconciliation, restart.

Ports the v1 behaviour fixtures to the revision-2 API and adds one regression family per independent finding:
B1 (restart hydration through the strict root validator), B2 (full chain replay; HOLD or refuse, never retry),
B3 (receipt-derived exposures, verified proof joins, descendant adoptions) and B4 (second fresh proof at the CAS).
Uses real BatchStream planners, PolicyLM/AdamW updates, physical checkpoints and the frozen prepared_admission
candidate/adopt functions on a portable fixture world.  No live data, outbox, S3, GPU or GP execution.
"""
import copy
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load

from jope_flywheel.program_io import sha
from jope_flywheel.protocol import canonical
import prepared_snapshot as P
from pipelines.boundary_admission_v2 import boundary as Bd
from pipelines.boundary_admission_v2.tests.harness import CAMPAIGN, Executor, World, new_inbox
import test_native_variant_snapshot as variant

B_PROGRAM = ('(pipe X (ridge alpha=3.0))', .005)
C_PROGRAM = ('(pipe X (ridge alpha=5.0))', .004)


class Fault(Exception):
    pass


def replace(path, body):
    """Fixture-only tamper: swap a create-only 0444 file for new bytes."""
    path = Path(path)
    os.unlink(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    os.write(fd, body)
    os.close(fd)


def planner_head(db):
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        return con.execute('SELECT count(*), max(seq), (SELECT sha FROM events ORDER BY seq DESC LIMIT 1) FROM events').fetchone()
    finally:
        con.close()


def inbox_files(inbox):
    return sorted((str(p.relative_to(inbox.root)), sha(p.read_bytes())) for p in inbox.root.rglob('*.json'))


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        variant.VariantTests.setUpClass()
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.w = World(self.base)
        self.Bspec, self.refs_b = self.w.append(self.w.A, B_PROGRAM)
        self.rb = self.w.root(self.w.A, self.Bspec, self.refs_b, 'prep-b')
        self.Cspec, self.refs_c = self.w.append(self.Bspec, C_PROGRAM)
        self.rc = self.w.root(self.Bspec, self.Cspec, self.refs_c, 'prep-c')

    def lane(self, name, fault=None, publish=('b', 'c')):
        inbox = new_inbox(self.w, self.base / (name + '-inbox'), fault=fault)
        if 'b' in publish:
            inbox.publish(**self.w.variant(self.rb), predecessor_extension_raw=canonical(self.w.A))
        if 'c' in publish:
            inbox.publish(**self.w.variant(self.rc), predecessor_extension_raw=canonical(self.Bspec))
        ex = Executor(self.w, self.base / name, inbox)
        self.addCleanup(ex.close)
        return ex, inbox

    def adopt_kwargs(self, ex, **over):
        kw = dict(campaign_sha256=CAMPAIGN, preprocessing_sha256=self.w.A['preprocessing_sha256'],
                  registry_sha256=self.w.A['parent_registry_sha256'], checkpoint_sha256=ex.ckpt,
                  fresh_authority=self.w.auth, authority_check=self.w.role_check)
        kw.update(over)
        return kw

    def snapshots(self, ex, inbox):
        out = {self.w.genesis.identity: self.w.genesis}
        for record in inbox.adoptions():
            out[record['snapshot_sha256']] = self.w.load(inbox, record)
        return out

    def key(self, receipt):
        return (receipt['rows'][0]['context_id'], receipt['rows'][0]['program_sha256'])

    # ------------------------------------------------------------------ v1 behaviour ported to revision 2
    def test_two_successive_adoptions_and_first_actual_exposures(self):
        ex, inbox = self.lane('main')
        ex.run_epoch()
        self.assertIsNotNone(ex.boundary())
        self.assertEqual(ex.snapshot.identity, self.rb['extension_sha256'])
        first_epoch_proofs = len(ex.proofs())
        ex.run_epoch()
        self.assertIsNotNone(ex.boundary())
        self.assertEqual(ex.snapshot.identity, self.rc['extension_sha256'])
        ex.run_epoch()
        records = inbox.adoptions()
        self.assertEqual([r['seq'] for r in records], [1, 2])
        self.assertNotEqual(records[0]['closed_epoch'], records[1]['closed_epoch'])
        self.assertEqual(records[1]['previous_snapshot_sha256'], records[0]['snapshot_sha256'])
        self.assertNotEqual(records[0]['hydration_authority_sha256'], records[0]['cas_authority_sha256'])
        events = [e for e in Bd.planner_events(ex.planner)[1] if e['kind'] == 'manifest.adopted']
        self.assertEqual([e['body']['new'] for e in events], [r['manifest_after'] for r in records])
        out = Bd.first_exposures(ex.planner, inbox, ex.proofs(), self.snapshots(ex, inbox))
        self.assertEqual(out['pending'], [])
        found = {(e['context_id'], e['program_sha256']): e for e in out['exposures']}
        eb, ec = found[self.key(self.rb)], found[self.key(self.rc)]
        self.assertEqual((eb['adoption_seq'], eb['exposed_under_seq']), (1, 1))
        self.assertEqual((ec['adoption_seq'], ec['exposed_under_seq']), (2, 2))
        self.assertTrue(out['coverage']['complete'])
        for e in (eb, ec):
            self.assertEqual(e['status'], 'first')
            self.assertIn('q', e['terms'])
            self.assertNotIn('ce', e['terms'])
            self.assertTrue(e['planner_commit_verified'])
            self.assertNotEqual(e['checkpoint_before'], e['checkpoint_after'])
        self.assertGreater(eb['absolute_optimizer_step'], first_epoch_proofs)
        self.assertLess(eb['absolute_optimizer_step'], ec['absolute_optimizer_step'])

    def test_one_adoption_per_completed_epoch_without_mutation(self):
        ex, inbox = self.lane('double')
        ex.run_epoch()
        ex.boundary()
        before = ex.planner.verify()
        with self.assertRaisesRegex(ValueError, 'closed boundary|one adoption per completed epoch'):
            ex.boundary()
        self.assertEqual(ex.planner.verify(), before)
        self.assertEqual(len(list((inbox.root / 'intents').iterdir())), 1)

    def test_refusals_happen_before_any_intent(self):
        ex, inbox = self.lane('refuse', publish=('b',))
        with self.subTest('open epoch'):
            ex.update()
            self.assertFalse(ex.epoch_complete())
            with self.assertRaisesRegex(ValueError, 'closed epoch required'):
                inbox.adopt_next(ex.planner, ex.snapshot, self.w.native, **self.adopt_kwargs(ex))
        while not ex.epoch_complete():
            ex.update()
        with self.subTest('stale hydration role proof'):
            stale = canonical({**json.loads(self.w.auth()), 'checked_at_unix': 0})
            before = ex.planner.verify()
            with self.assertRaisesRegex(ValueError, 'fresh role'):
                inbox.adopt_next(ex.planner, ex.snapshot, self.w.native,
                                 **self.adopt_kwargs(ex, fresh_authority=lambda: stale))
            self.assertEqual(ex.planner.verify(), before)
        with self.subTest('wrong campaign permit'):
            with self.assertRaisesRegex(ValueError, 'campaign'):
                inbox.adopt_next(ex.planner, ex.snapshot, self.w.native, **self.adopt_kwargs(ex, campaign_sha256='f' * 64))
        with self.subTest('normalizer drift'):
            with self.assertRaises(ValueError):
                inbox.adopt_next(ex.planner, ex.snapshot, self.w.native,
                                 **self.adopt_kwargs(ex, preprocessing_sha256='0' * 64))
        with self.subTest('checkpoint not joined'):
            with self.assertRaisesRegex(ValueError, 'physical checkpoint'):
                inbox.adopt_next(ex.planner, ex.snapshot, self.w.native, **self.adopt_kwargs(ex, checkpoint_sha256='0' * 64))
        self.assertEqual(list((inbox.root / 'intents').iterdir()), [])
        self.assertEqual(list((inbox.root / 'holds').iterdir()), [])
        with self.subTest('unresolved reservation'):
            ex.planner.reserve(ex.ckpt)
            with self.assertRaisesRegex(ValueError, 'closed epoch required|unresolved optimizer reservation'):
                inbox.adopt_next(ex.planner, ex.snapshot, self.w.native, **self.adopt_kwargs(ex))
        self.assertEqual(list((inbox.root / 'intents').iterdir()), [])

    def test_publication_chain_and_prefix_refusals(self):
        inbox = new_inbox(self.w, self.base / 'chain-inbox')
        with self.assertRaisesRegex(ValueError, 'previous published snapshot'):
            inbox.publish(**self.w.variant(self.rc), predecessor_extension_raw=canonical(self.Bspec))
        inbox.publish(**self.w.variant(self.rb), predecessor_extension_raw=canonical(self.w.A))
        other, refs = self.w.append(self.w.A, ('(pipe X (ridge alpha=7.0))', .003))
        r_other = self.w.root(self.w.A, other, refs, 'prep-other')
        with self.assertRaises(ValueError):
            inbox.publish(**self.w.variant(r_other), predecessor_extension_raw=canonical(self.Bspec))
        foreign = Bd.Inbox.create(self.base / 'foreign-inbox', genesis_snapshot_sha256=self.w.genesis.identity,
                                  genesis_manifest_sha256=sha(canonical(self.w.genesis.batch_manifest)),
                                  campaign_sha256=CAMPAIGN, contract={**self.w.contract, 'x1_source_packet_sha256': 'a' * 64})
        with self.assertRaisesRegex(ValueError, 'campaign target contract'):
            foreign.publish(**self.w.variant(self.rb), predecessor_extension_raw=canonical(self.w.A))
        self.assertEqual(len(list((inbox.root / 'candidates').iterdir())), 1)

    def crash_and_restart(self, point, name=None):
        def fault(p):
            if p == point:
                raise Fault(p)
        ex, inbox = self.lane(name or 'crash-' + point, fault=fault)
        ex.run_epoch()
        with self.assertRaises(Fault):
            ex.boundary()
        step = ex.step
        ex.close()
        return ex.root, inbox, step

    def restart(self, root, inbox, step):
        ex = Executor.restart(self.w, root, Bd.Inbox(inbox.root), step=step)
        self.addCleanup(ex.close)
        return ex, ex.inbox

    def test_crash_points_reconcile_to_exactly_one_adoption_or_hold(self):
        for point in ('before_intent', 'after_intent', 'after_cas', 'after_record'):
            with self.subTest(point=point):
                ex, inbox = self.restart(*self.crash_and_restart(point))
                events = [e for e in Bd.planner_events(ex.planner)[1] if e['kind'] == 'manifest.adopted']
                if point == 'before_intent':
                    self.assertEqual((events, inbox.adoptions(), inbox.holds()), ([], [], []))
                    self.assertIsNotNone(ex.boundary())
                    self.assertEqual(len(inbox.adoptions()), 1)
                elif point == 'after_intent':
                    self.assertEqual(events, [])
                    self.assertEqual([h['reason'] for h in inbox.holds()], ['INTENT_WITHOUT_PLANNER_CAS'])
                    with self.assertRaises(Bd.Held):
                        ex.boundary()
                else:
                    self.assertEqual(len(events), 1)
                    records = inbox.adoptions()
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]['reconciled_after_crash'], point == 'after_cas')
                    self.assertEqual(ex.snapshot.identity, self.rb['extension_sha256'])
                    self.assertEqual(inbox.holds(), [])
                    ex.run_epoch()
                    self.assertIsNotNone(ex.boundary())
                    self.assertEqual([r['seq'] for r in inbox.adoptions()], [1, 2])

    def test_restart_next_update_is_bitwise_identical(self):
        uninterrupted, _ = self.lane('branch-1')
        uninterrupted.run_epoch()
        uninterrupted.boundary()
        resumed_step = uninterrupted.step
        uninterrupted.update()
        interrupted, inbox = self.lane('branch-2')
        interrupted.run_epoch()
        interrupted.boundary()
        interrupted.close()
        restarted, _ = self.restart(interrupted.root, inbox, resumed_step)
        self.assertEqual(restarted.snapshot.identity, uninterrupted.snapshot.identity)
        restarted.update()
        left = load((uninterrupted.root / 'ckpt' / f'{resumed_step + 1:06d}' / 'state.safetensors').read_bytes())
        right = load((restarted.root / 'ckpt' / f'{resumed_step + 1:06d}' / 'state.safetensors').read_bytes())
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)
        a = json.loads((uninterrupted.root / f'batch-{resumed_step + 1:09d}.json').read_bytes())
        b = json.loads((restarted.root / f'batch-{resumed_step + 1:09d}.json').read_bytes())
        self.assertEqual(a['selection'], b['selection'])
        self.assertEqual(a['batch']['ordered_data_run_ids'], b['batch']['ordered_data_run_ids'])

    def test_adoption_retains_old_contexts_roles_and_validation(self):
        ex, inbox = self.lane('roles')
        before = dict(ex.snapshot.contexts)
        ex.run_epoch()
        ex.boundary()
        for cid, descriptor in before.items():
            self.assertEqual(ex.snapshot.contexts[cid], descriptor)
        self.assertEqual({c for c, r in ex.snapshot.contexts.items() if r['partition'] == 'validation'},
                         {c for c, r in before.items() if r['partition'] == 'validation'})
        raw, active = Bd.active_manifest(ex.root / 'batches.sqlite')
        self.assertEqual(active, inbox.adoptions()[-1]['manifest_after'])
        self.assertEqual(sha(raw), active)

    # ------------------------------------------------------------------ B1: strict restart hydration
    def adopted_once(self, name):
        ex, inbox = self.lane(name, publish=('b',))
        ex.run_epoch()
        ex.boundary()
        step = ex.step
        ex.close()
        return ex.root, inbox, step

    def test_b1_restart_refuses_tampered_numeric_target_with_matching_status(self):
        root, inbox, step = self.adopted_once('b1-numeric')
        record = inbox.adoptions()[0]
        prep = Path(self.rb['root'])
        prepared = json.loads((prep / 'prepared.json').read_bytes())
        cid, ph = self.key(self.rb)
        rows = [r for r in prepared['variant_records'][cid] if r['program_sha256'] == ph]
        self.assertEqual(len(rows), 1)
        rows[0]['q'] += .5
        rows[0]['q_opt'] += .5
        rows[0]['q_grid'] = round(rows[0]['q'] * 1e6)
        forged = canonical(prepared)
        status = json.loads((prep / 'status.json').read_bytes())
        status['prepared'] = {**status['prepared'], 'sha256': sha(forged), 'bytes': len(forged)}
        # Premise of the finding: the frozen loader alone accepts a self-consistent forged pair, identities unchanged.
        forged_snapshot = P.load(forged, self.w.native, receipt=status, expected_extension_sha256=self.rb['extension_sha256'],
                                 expected_preprocessing_sha256=self.w.A['preprocessing_sha256'],
                                 expected_parent_registry_sha256=self.w.A['parent_registry_sha256'],
                                 authority_check=lambda raw: self.w.role_check(raw))
        self.assertEqual(forged_snapshot.identity, record['snapshot_sha256'])
        self.assertEqual(sha(canonical(forged_snapshot.batch_manifest)), record['manifest_after'])
        replace(prep / 'prepared.json', forged)
        replace(prep / 'status.json', canonical(status))
        head, files = planner_head(root / 'batches.sqlite'), inbox_files(inbox)
        with self.assertRaisesRegex(ValueError, 'source SHA mismatch'):
            Executor.restart(self.w, root, Bd.Inbox(inbox.root), step=step)
        self.assertEqual((planner_head(root / 'batches.sqlite'), inbox_files(inbox)), (head, files), 'zero mutation')

    def test_b1_restart_refuses_tampered_status_config_source_extension_receipt_or_symlink(self):
        root, inbox, step = self.adopted_once('b1-files')
        prep = Path(self.rb['root'])
        head, files = planner_head(root / 'batches.sqlite'), inbox_files(inbox)

        def status(raw):
            d = json.loads(raw)
            d['seconds'] = d['seconds'] + 1
            return canonical(d)
        cases = {'status.json': status, 'config.json': lambda raw: raw + b' ',
                 'prepared_snapshot.py': lambda raw: raw + b'\n# changed\n', 'extension.json': lambda raw: raw + b' ',
                 'c10a-receipt.json': lambda raw: raw + b' ', 'preparation.py': lambda raw: raw + b'\n'}
        for name, edit in cases.items():
            with self.subTest(file=name):
                original = (prep / name).read_bytes()
                replace(prep / name, edit(original))
                try:
                    with self.assertRaises(ValueError):
                        Executor.restart(self.w, root, Bd.Inbox(inbox.root), step=step)
                finally:
                    replace(prep / name, original)
        with self.subTest(file='status.json via symlink'):
            original = (prep / 'status.json').read_bytes()
            elsewhere = self.base / 'status-copy.json'
            elsewhere.write_bytes(original)
            os.unlink(prep / 'status.json')
            os.symlink(elsewhere, prep / 'status.json')
            try:
                with self.assertRaises((ValueError, OSError)):
                    Executor.restart(self.w, root, Bd.Inbox(inbox.root), step=step)
            finally:
                os.unlink(prep / 'status.json')
                replace_target = prep / 'status.json'
                fd = os.open(replace_target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                os.write(fd, original)
                os.close(fd)
        self.assertEqual((planner_head(root / 'batches.sqlite'), inbox_files(inbox)), (head, files), 'zero mutation')
        restarted, _ = self.restart(root, inbox, step)
        self.assertEqual(restarted.snapshot.identity, self.rb['extension_sha256'])

    def test_b1r5_restart_requires_a_qualified_terminal_marker(self):
        root, inbox, step = self.adopted_once('b1r5-terminal')
        path = Path(self.rb['root']) / 'c10a-terminal.json'
        original = path.read_bytes()
        os.unlink(path)
        try:
            with self.assertRaises((ValueError, OSError)):
                Executor.restart(self.w, root, Bd.Inbox(inbox.root), step=step)
            late = json.loads(original)
            late['qualified_after_durable_status']['wall_unix'] = late['bound']['wall_deadline_unix'] + 1
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            os.write(fd, canonical(late))
            os.close(fd)
            with self.assertRaisesRegex(ValueError, 'qualified inside its original bound'):
                Executor.restart(self.w, root, Bd.Inbox(inbox.root), step=step)
        finally:
            replace(path, original) if path.exists() else None
            if not path.exists():
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                os.write(fd, original)
                os.close(fd)
        restarted, _ = self.restart(root, inbox, step)
        self.assertEqual(restarted.snapshot.identity, self.rb['extension_sha256'])

    # ------------------------------------------------------------------ B2: full chain replay
    def test_b2_intent_join_tamper_after_cas_holds_without_record(self):
        edits = {'checkpoint_sha256': lambda i: i.update(checkpoint_sha256='0' * 64),
                 'closed_epoch': lambda i: i.update(closed_epoch=i['closed_epoch'] + 1),
                 'planner_head_before': lambda i: i.update(planner_head_before={**i['planner_head_before'],
                                                                                'head_sha256': '0' * 64}),
                 'candidate_sha256': lambda i: i.update(candidate_sha256='0' * 64)}
        for field, edit in edits.items():
            with self.subTest(field=field):
                root, inbox, step = self.crash_and_restart('after_cas', name='b2-' + field)
                path = inbox.root / 'intents' / '00000001.json'
                intent = json.loads(path.read_bytes())
                edit(intent)
                replace(path, canonical(intent))
                ex, inbox2 = self.restart(root, inbox, step)
                self.assertEqual(inbox2.adoptions(), [])
                self.assertEqual([h['reason'] for h in inbox2.holds()], ['INTENT_JOIN_FAILED'])
                with self.assertRaises(Bd.Held):
                    ex.boundary()
                self.assertEqual(inbox2.adoptions(), [])

    def test_b2_broken_existing_adoption_links_refuse_restart(self):
        root, inbox, step = self.adopted_once('b2-links')
        files = {'record': inbox.root / 'adoptions' / '00000001.json', 'intent': inbox.root / 'intents' / '00000001.json',
                 'candidate': inbox.root / 'candidates' / '00000001.json'}
        edits = {'record planner event': ('record', lambda d: d['planner_event'].update(sha='0' * 64)),
                 'record snapshot': ('record', lambda d: d.update(snapshot_sha256='0' * 64)),
                 'record epoch': ('record', lambda d: d.update(closed_epoch=d['closed_epoch'] + 1)),
                 'intent bytes': ('intent', lambda d: d.update(at_utc_ns=d['at_utc_ns'] + 1)),
                 'candidate bytes': ('candidate', lambda d: d.update(created_utc_ns=d['created_utc_ns'] + 1))}
        for why, (which, edit) in edits.items():
            with self.subTest(why=why):
                original = files[which].read_bytes()
                doc = json.loads(original)
                edit(doc)
                replace(files[which], canonical(doc))
                try:
                    with self.assertRaises(Bd.Refused):
                        Executor.restart(self.w, root, Bd.Inbox(inbox.root), step=step)
                finally:
                    replace(files[which], original)
        self.assertEqual(Bd.Inbox(inbox.root).holds(), [])
        restarted, _ = self.restart(root, inbox, step)
        self.assertEqual(restarted.snapshot.identity, self.rb['extension_sha256'])

    def test_b2_intent_without_cas_holds_and_never_attempts_cas(self):
        root, inbox, step = self.crash_and_restart('after_intent', name='b2-nocas')
        ex, inbox2 = self.restart(root, inbox, step)
        head = planner_head(root / 'batches.sqlite')
        for _ in range(3):
            inbox2.reconcile(ex.planner)
            with self.assertRaises(Bd.Held):
                ex.boundary()
        self.assertEqual(planner_head(root / 'batches.sqlite'), head)
        self.assertEqual([h['reason'] for h in inbox2.holds()], ['INTENT_WITHOUT_PLANNER_CAS'])
        self.assertEqual([e for e in Bd.planner_events(ex.planner)[1] if e['kind'] == 'manifest.adopted'], [])

    # ------------------------------------------------------------------ B3: exposures
    def two_adoptions(self, name):
        ex, inbox = self.lane(name)
        ex.run_epoch()
        ex.boundary()
        ex.run_epoch()
        ex.boundary()
        ex.run_epoch()
        return ex, inbox

    def test_b3_first_exposure_counts_under_descendant_adoption(self):
        ex, inbox = self.two_adoptions('b3-desc')
        records = inbox.adoptions()
        snaps = self.snapshots(ex, inbox)
        c_era = [p for p in ex.proofs() if json.loads(p.read_bytes())['batch']['manifest_sha256'] == records[1]['manifest_after']]
        self.assertTrue(c_era)
        out = Bd.first_exposures(ex.planner, inbox, c_era, snaps)
        b = [e for e in out['exposures'] if (e['context_id'], e['program_sha256']) == self.key(self.rb)]
        self.assertEqual(len(b), 1, 'B observed only under C counts exactly once')
        self.assertEqual((b[0]['adoption_seq'], b[0]['exposed_under_seq']), (1, 2))
        self.assertEqual(b[0]['status'], 'earliest_observed_partial', 'B-era batches were not supplied')
        self.assertFalse(out['coverage']['complete'])
        self.assertEqual(out['pending'], [])
        full = Bd.first_exposures(ex.planner, inbox, ex.proofs(), snaps)
        bf = [e for e in full['exposures'] if (e['context_id'], e['program_sha256']) == self.key(self.rb)]
        self.assertEqual((bf[0]['adoption_seq'], bf[0]['exposed_under_seq'], bf[0]['status']), (1, 1, 'first'))
        self.assertLess(bf[0]['absolute_optimizer_step'], b[0]['absolute_optimizer_step'])

    def test_b3_exposure_evidence_tamper_refused_or_deduplicated(self):
        ex, inbox = self.two_adoptions('b3-tamper')
        snaps = self.snapshots(ex, inbox)
        proofs = ex.proofs()
        base = Bd.first_exposures(ex.planner, inbox, proofs, snaps)
        with self.subTest('reversed order'):
            self.assertEqual(Bd.first_exposures(ex.planner, inbox, list(reversed(proofs)), snaps), base)
        with self.subTest('duplicate paths'):
            self.assertEqual(Bd.first_exposures(ex.planner, inbox, proofs + proofs, snaps), base)
        target = proofs[len(proofs) // 2]
        for why, edit in {'checkpoint': lambda d: d['checkpoint'].update(parameter_sha256='0' * 64),
                          'step': lambda d: d.update(absolute_optimizer_step=d['absolute_optimizer_step'] + 1),
                          'selection': lambda d: d['selection']['selection']['members'].reverse()}.items():
            doc = json.loads(target.read_bytes())
            edit(doc)
            forged = self.base / f'forged-{why}.json'
            forged.write_bytes(canonical(doc))
            with self.subTest(why='altered ' + why):
                with self.assertRaisesRegex(ValueError, 'conflicting proofs|committed evidence'):
                    Bd.first_exposures(ex.planner, inbox, [p if p != target else forged for p in proofs], snaps)
                with self.assertRaisesRegex(ValueError, 'conflicting proofs'):
                    Bd.first_exposures(ex.planner, inbox, proofs + [forged], snaps)
        with self.subTest('wrong admitted reference version'):
            record = inbox.adoptions()[0]
            bad = copy.copy(snaps[record['snapshot_sha256']])
            bad.receipt = copy.deepcopy(bad.receipt)
            cid, ph = self.key(self.rb)
            for x in bad.receipt['variants'][cid]['records']:
                if x['program_sha256'] == ph:
                    x['result']['version_id'] = 'v-forged'
            with self.assertRaisesRegex(ValueError, 'admitted reference'):
                Bd.first_exposures(ex.planner, inbox, proofs, {**snaps, record['snapshot_sha256']: bad})

    def test_b3r3_omitted_earlier_proof_never_promotes_a_later_first(self):
        ex, inbox = self.two_adoptions('b3r3-omit')
        snaps = self.snapshots(ex, inbox)
        proofs = ex.proofs()
        full = Bd.first_exposures(ex.planner, inbox, proofs, snaps)
        first_b = next(e for e in full['exposures'] if (e['context_id'], e['program_sha256']) == self.key(self.rb))
        self.assertEqual(first_b['status'], 'first')
        dropped = [p for p in proofs if Bd.sha(p.read_bytes()) != first_b['proof_sha256']]
        out = Bd.first_exposures(ex.planner, inbox, dropped, snaps)
        self.assertEqual(out['coverage']['missing'], 1)
        later = [e for e in out['exposures'] if (e['context_id'], e['program_sha256']) == self.key(self.rb)]
        self.assertTrue(later, 'B is observed again in a later supplied batch')
        self.assertGreater(later[0]['absolute_optimizer_step'], first_b['absolute_optimizer_step'])
        self.assertEqual(later[0]['status'], 'earliest_observed_partial', 'never promoted to first')
        c_only = [p for p in proofs if json.loads(p.read_bytes())['batch']['manifest_sha256'] == inbox.genesis['manifest_sha256']]
        out = Bd.first_exposures(ex.planner, inbox, c_only, snaps)
        self.assertEqual(sorted(x[-1] for x in out['pending']), ['unobserved_partial', 'unobserved_partial'])

    # ------------------------------------------------------------------ B4: second fresh proof at the CAS
    def test_b4_stale_authority_at_boundary_holds_without_cas(self):
        ex, inbox = self.lane('b4-before', publish=('b',))
        ex.run_epoch()
        head = planner_head(ex.root / 'batches.sqlite')
        proofs = iter([self.w.auth(), canonical({**json.loads(self.w.auth()), 'checked_at_unix': 0})])
        with self.assertRaises(Bd.Held):
            ex.boundary(fresh_authority=lambda: next(proofs))
        self.assertEqual(list((inbox.root / 'intents').iterdir()), [])
        self.assertEqual([h['reason'] for h in inbox.holds()], ['AUTHORITY_REFUSED_BEFORE_CAS'])
        self.assertEqual(planner_head(ex.root / 'batches.sqlite'), head)
        with self.assertRaises(Bd.Held):
            ex.boundary()

    def test_b4_authority_refused_inside_cas_retains_intent_and_holds(self):
        ex, inbox = self.lane('b4-cas', publish=('b',))
        ex.run_epoch()
        head = planner_head(ex.root / 'batches.sqlite')
        issued = [canonical({**json.loads(self.w.auth()), 'nonce': i}) for i in range(2)]
        supply, counts = iter(issued), {}

        def check(raw, proof):
            counts[proof] = counts.get(proof, 0) + 1
            if proof == issued[1] and counts[proof] == 2:
                raise ValueError('role drift between the pre-CAS proof and the CAS')
            self.w.role_check(raw, proof)
        with self.assertRaises(Bd.Held):
            ex.boundary(fresh_authority=lambda: next(supply), authority_check=check)
        self.assertEqual(len(list((inbox.root / 'intents').iterdir())), 1)
        self.assertEqual([h['reason'] for h in inbox.holds()], ['AUTHORITY_REFUSED_AT_CAS'])
        self.assertEqual(planner_head(ex.root / 'batches.sqlite'), head)
        intent = json.loads((inbox.root / 'intents' / '00000001.json').read_bytes())
        self.assertEqual(intent['cas_authority_sha256'], sha(issued[1]))
        self.assertEqual(intent['hydration_authority_sha256'], sha(issued[0]))
        inbox.reconcile(ex.planner)
        self.assertEqual([h['reason'] for h in inbox.holds()], ['AUTHORITY_REFUSED_AT_CAS'])
        with self.assertRaises(Bd.Held):
            ex.boundary()


    # ------------------------------------------------------------------ r7: the accepted envelope at every consumer
    def test_b1r7_publication_and_restart_require_the_accepted_envelope(self):
        inbox = new_inbox(self.w, self.base / 'r7-inbox')
        descriptor, pred = self.w.descriptor(self.rb), canonical(self.w.A)
        before = inbox_files(inbox)
        with self.assertRaisesRegex(ValueError, 'exact bounded reference'):
            inbox.publish(descriptor=descriptor, acceptance=None, predecessor_extension_raw=pred)
        with self.assertRaisesRegex(ValueError, 'accepted bounded terminal envelope for this subject'):
            inbox.publish(descriptor=descriptor, acceptance=self.w.acceptances[self.rc['root']],
                          predecessor_extension_raw=pred)
        self.assertEqual(inbox_files(inbox), before)
        envelope = inbox.publish(**self.w.variant(self.rb), predecessor_extension_raw=pred)
        self.assertEqual(envelope['components']['variant']['preparation_acceptance'], self.w.acceptances[self.rb['root']])
        root, adopted, step = self.adopted_once('r7-restart')
        path = Path(self.w.acceptances[self.rb['root']]['path'])
        original = path.read_bytes()
        doc = json.loads(original)
        doc['bound']['wall_deadline_unix'] += 3600
        replace(path, canonical(doc))
        try:
            with self.assertRaisesRegex(ValueError, 'exact pinned bytes'):
                Executor.restart(self.w, root, Bd.Inbox(adopted.root), step=step)
        finally:
            replace(path, original)
        restarted, _ = self.restart(root, adopted, step)
        self.assertEqual(restarted.snapshot.identity, self.rb['extension_sha256'])


if __name__ == '__main__':
    unittest.main()
