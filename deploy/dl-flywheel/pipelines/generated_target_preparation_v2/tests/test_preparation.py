"""C10a revision 2 private CPU fixtures: verified A -> B -> C train-result appends become exact prepared roots.

Ports the v1 behaviour fixtures to the revision-2 API and adds one regression family per independent finding:
A1 (appended reference <-> row bijection), A2 (production contract binding) and A3 (single work clock and audited
zero-network store).  No live data, S3, GP execution, planner, optimizer step or GPU.
"""
import copy
import json
import math
import os
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np
import torch

import test_native_variant_snapshot as variant
from dltrain import labels
from dltrain import model as M
from jope_flywheel import native_variant_snapshot as V
from jope_flywheel import training_snapshot_kumos as S
from jope_flywheel.protocol import canonical
import prepared_admission as PA
from neural_native_train import prepare_microbatches
from neural_native_batch import OBJECTIVE, objective
from pipelines.generated_target_admission_v1.common import canonical as xcanonical, sha
from pipelines.generated_target_preparation_v2 import preparation as C
from pipelines.generated_target_preparation_v2.tests import world as W


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        variant.VariantTests.setUpClass()
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        f, n, spec, nsraw, registry, pre, auth = variant.VariantTests().fixture()
        self.f, self.native, self.A, self.nsraw, self.registry, self.pre = f, n, spec, nsraw, registry, pre
        self.store = W.AuditedStore(f.store)
        self.auth = lambda: canonical({**json.loads(auth()), 'checked_at_unix': time.time()})
        self.parents = {p['parent_lineage_sha256']: p for p in map(json.loads, registry.splitlines())}
        (self.base / 'acceptances').mkdir()
        self.acceptances = {}

    def accept(self, receipt, *, physical=None, name=None):
        """Operator acceptance after the in-process worker returned (r7): pins the returned sealed receipt."""
        root = receipt['root']
        ref = C.accept_root(root, acceptance_path=self.base / 'acceptances' / ((name or Path(root).name) + '.json'),
                            physical=physical or {'kind': 'in_process_return',
                                                  'returned_sha256': sha(xcanonical(receipt))})
        self.acceptances.setdefault(root, ref)
        return ref

    def append(self, spec, *programs):
        new = copy.deepcopy(spec)
        refs = [W.add_program(self.f, new['tasks'][0], dsl, rmse) for dsl, rmse in programs]
        new['tasks'][0]['program_results'].extend(refs)
        return new, refs

    def manifest(self, old, new, refs, **kw):
        return W.x1_manifest(self.f, old, new, refs, pre=self.pre, **kw)

    def prepare(self, old, new, refs, name='root', manifest=None, predecessor=None, auth=None, **over):
        raw, pred = self.manifest(old, new, refs) if manifest is None else (manifest, predecessor)
        args = dict(expected_manifest_sha256=sha(raw), predecessor_raw=pred, native=self.native, store=self.store,
                    native_spec_bytes=self.nsraw, registry_bytes=self.registry, preprocessing_bytes=self.pre,
                    objective_bytes=W.OBJECTIVE_BYTES, contract=W.contract(self.pre), export_provenance=dict(W.PROVENANCE),
                    authority_callback=auth or self.auth, root=self.base / name, deadline=time.time() + 60,
                    labels=labels)
        args.update(over)
        receipt = C.prepare_root(raw, **args)
        self.accept(receipt)
        return receipt, raw

    def composed(self, spec):
        b = canonical(spec)
        s = V.compose(self.native, b, V.sha(b), self.f.store, native_spec_bytes=self.nsraw,
                      registry_bytes=self.registry, preprocessing_bytes=self.pre, authority_callback=self.auth,
                      deadline=time.time() + 30)
        s.extension_raw = b
        return s

    def role_check(self, raw):
        V.check_authority(json.loads(raw), json.loads(self.nsraw), self.auth(), self.parents, time.time())

    def candidate(self, root, current):
        descriptor = C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=self.acceptances[str(root)])
        return PA.candidate(descriptor, current, self.native, campaign_sha256='e' * 64,
                            preprocessing_sha256=self.A['preprocessing_sha256'], registry_sha256=self.A['parent_registry_sha256'],
                            authority_check=self.role_check)

    # ------------------------------------------------------------------ v1 behaviour ported to revision 2
    def test_a_to_b_root_is_consumed_by_frozen_candidate(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .005))
        receipt, _ = self.prepare(self.A, B, refs)
        self.assertEqual(receipt['dispositions'], {'record': 1})
        replacement = self.candidate(receipt['root'], self.composed(self.A))
        cid = receipt['rows'][0]['context_id']
        programs = {r.program_sha256: r for r in replacement.variant_records[cid]}
        self.assertEqual(len(programs), 3)
        new = programs[receipt['rows'][0]['program_sha256']]
        self.assertFalse(new.ce_eligible)
        self.assertTrue(new.quality_eligible and new.pairwise_eligible)
        self.assertEqual(replacement.best_observed[cid].program_sha256, new.program_sha256)
        self.assertEqual(replacement.identity, receipt['extension_sha256'])
        self.assertEqual({p.name for p in Path(receipt['root']).iterdir()},
                         {'c10a-receipt.json', 'c10a-terminal.json', 'config.json', 'extension.json', 'prepared.json',
                          'prepared_snapshot.py', 'preparation.py', 'status.json'})
        self.assertEqual(receipt['bijection'], {'appended_references': 1, 'rows': 1})
        self.assertEqual(receipt['contract'], W.contract(self.pre))
        self.assertEqual(receipt['store_audit']['delta']['network_gets'], 0)
        self.assertGreater(receipt['store_audit']['delta']['local_reads'], 0)

    def test_composed_labels_equal_direct_recomputation(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015), ('(pipe X (ridge alpha=4.0))', .02))
        receipt, _ = self.prepare(self.A, B, refs)
        for row in receipt['rows']:
            d, r = row['direct'], row['record']
            self.assertEqual((r['q'], r['q_opt'], r['q_grid']), (d['q'], d['q_opt'], d['q_grid']))
            self.assertEqual(d['q_grid'], round(d['q'] * 1e6))
            self.assertEqual(d['q'], labels.q_from_detail(d['rmse_j'], d['baseline_rmse_j'], d['epsilon']))
        s = self.composed(B)
        cid = receipt['rows'][0]['context_id']
        task = B['tasks'][0]
        rows = []
        for rec in s.receipt['variants'][cid]['records']:
            entry = self.f.store.doc(rec['result'])
            ev = self.f.store.doc(entry['artifacts']['evaluation'])
            skeleton = next(r.skeleton_sha256 for r in s.variant_records[cid] if r.program_sha256 == rec['program_sha256'])
            rows.append({'context_sha256': cid, 'program_sha256': rec['program_sha256'], 'outcome': 'completed',
                         'backend': 'local', 'skeleton_sha256': skeleton, 'shared_seed': False,
                         'rmse_j': [x['metrics']['rmse'] for x in ev['inner_cv']],
                         'baseline_rmse_j': [x['baseline_metrics']['rmse'] for x in ev['inner_cv']],
                         'wall_us': sum(x['wall_us'] for x in ev['inner_cv'])})
        y = S.decode_input(self.f.raws[task['variant_identity']])[0].tolist()
        _, labs = labels.label_context({'context_sha256': cid, 'dataset_hash': task['variant_identity'],
                                        'family_id': task['parent_lineage_sha256'], 'split_role': 'train_fit',
                                        'y_train': y}, rows)
        self.assertEqual(len(labs), 4)
        records = {r.program_sha256: r for r in s.variant_records[cid]}
        for lab in labs:
            r = records[lab.program_sha256]
            self.assertEqual((lab.q, lab.q_opt, lab.q_grid), (r.q, r.q_opt, r.q_grid))

    def test_b_to_c_second_append_keeps_ordered_prefix(self):
        B, refs_b = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        rb, _ = self.prepare(self.A, B, refs_b, name='b')
        current = self.candidate(rb['root'], self.composed(self.A))
        Cspec, refs_c = self.append(B, ('(pipe X (ridge alpha=5.0))', .004))
        rc, _ = self.prepare(B, Cspec, refs_c, name='c')
        replacement = self.candidate(rc['root'], current)
        cid = rc['rows'][0]['context_id']
        self.assertEqual(len(replacement.variant_records[cid]), 4)
        self.assertEqual(json.loads(replacement.extension_raw)['tasks'][0]['program_results'][:3],
                         B['tasks'][0]['program_results'])
        raw, _ = self.manifest(B, Cspec, refs_c)
        with self.assertRaisesRegex(ValueError, 'actual current predecessor'):
            self.prepare(B, Cspec, refs_c, name='stale', manifest=raw, predecessor=xcanonical(self.A))

    def test_token_refusal_is_reconciled_not_dropped(self):
        long = '(pipe X ' + '(standardize append=true) ' * 160 + '(ridge alpha=1.0))'
        B, refs = self.append(self.A, (long, .01), ('(pipe X (ridge alpha=3.0))', .015))
        receipt, _ = self.prepare(self.A, B, refs)
        self.assertEqual(receipt['dispositions'], {'record': 1, 'refused:over_model_max_seq': 1})
        refused = [r for r in receipt['rows'] if r['disposition'].startswith('refused:')]
        self.assertNotIn('record', refused[0])

    def test_exact_fit_reaches_floor_cap_and_grid_uses_raw_q(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=9.0))', 0.0))
        receipt, _ = self.prepare(self.A, B, refs)
        d, r = receipt['rows'][0]['direct'], receipt['rows'][0]['record']
        self.assertEqual(d['q'], labels.q_cap(d['baseline_rmse_j'], d['epsilon']))
        self.assertGreater(d['q'], math.log(1000))
        self.assertEqual(r['q_opt'], labels.clip_q(d['q']))
        self.assertEqual(r['q_grid'], round(d['q'] * 1e6))

    def test_manifest_tamper_refusals(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, pred = self.manifest(self.A, B, refs)
        base = json.loads(raw)
        edits = {
            'extension digest': lambda m: m.update(extension_sha256='0' * 64),
            'references only': lambda m: m.update(numeric_targets_materialized=True),
            'train-only': lambda m: m['rows'][0].update(partition='validation'),
            'five-class census': lambda m: m['counts'].update(appended=2),
            'emission provenance': lambda m: m['rows'][0].update(source_emission_ids=[]),
            'closed X1 manifest': lambda m: m.update(unreviewed=True),
            'CE-false': lambda m: m['rows'][0].update(ce_eligible=True),
        }
        for why, edit in edits.items():
            m = copy.deepcopy(base)
            edit(m)
            bad = xcanonical(m)
            with self.subTest(why=why), self.assertRaises(ValueError):
                C.parse_manifest(bad, expected_sha256=sha(bad))
        with self.assertRaisesRegex(ValueError, 'pinned raw X1'):
            C.parse_manifest(raw, expected_sha256='0' * 64)

    def test_reordered_or_removed_old_refs_refused(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        reordered = copy.deepcopy(B)
        r = reordered['tasks'][0]['program_results']
        r[0], r[1] = r[1], r[0]
        raw, pred = self.manifest(self.A, reordered, refs)
        with self.assertRaises(ValueError):
            self.prepare(self.A, reordered, refs, name='reordered', manifest=raw, predecessor=pred)
        self.assertFalse((self.base / 'reordered').exists())

    def test_root_is_new_exclusive_and_read_only(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        (self.base / 'taken').mkdir()
        with self.assertRaisesRegex(ValueError, 'new absolute preparation root'):
            self.prepare(self.A, B, refs, name='taken')
        receipt, _ = self.prepare(self.A, B, refs, name='fresh')
        for path in Path(receipt['root']).iterdir():
            self.assertEqual(path.stat().st_mode & 0o222, 0, path.name)
        config = json.loads((Path(receipt['root']) / 'config.json').read_bytes())
        self.assertEqual(config['origin']['x1_source_packet_sha256'], '9' * 64)
        self.assertNotEqual(config['origin']['x1_manifest_sha256'], receipt['extension_sha256'])
        self.assertEqual(json.loads((Path(receipt['root']) / 'status.json').read_bytes())['state'], 'passed')

    def test_same_inputs_same_prepared_bytes(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, pred = self.manifest(self.A, B, refs)
        proof = self.auth()                     # prepared bytes bind the composing role proof digest
        r1, _ = self.prepare(self.A, B, refs, name='one', manifest=raw, predecessor=pred, auth=lambda: proof)
        r2, _ = self.prepare(self.A, B, refs, name='two', manifest=raw, predecessor=pred, auth=lambda: proof)
        self.assertEqual(r1['prepared']['sha256'], r2['prepared']['sha256'])
        self.assertEqual(r1['rows'], r2['rows'])

    def test_stale_role_proof_refused_before_any_root(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        stale = canonical({**json.loads(self.auth()), 'checked_at_unix': 0})
        with self.assertRaisesRegex(ValueError, 'fresh role'):
            self.prepare(self.A, B, refs, name='stale', auth=lambda: stale)
        self.assertFalse((self.base / 'stale').exists())

    def test_new_record_reaches_shared_trunk_with_zero_ce_support(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        receipt, _ = self.prepare(self.A, B, refs)
        snapshot = self.candidate(receipt['root'], self.composed(self.A))
        cid = receipt['rows'][0]['context_id']
        norm = lambda c, z: np.clip(np.asarray(z, dtype=np.float64), -10, 10).astype(np.float32)
        batches, total, selection = prepare_microbatches(snapshot, [cid], epoch=0, batch_index=0, normalizer=norm,
                                                         max_padded_tokens=4096)
        members = selection['selection']['members'][0]['program_ids']
        self.assertIn(receipt['rows'][0]['program_sha256'], members)
        self.assertEqual(total['ce'], 0)
        self.assertGreater(total['q'], 0)
        torch.manual_seed(4)
        net = M.PolicyLM(M.PolicyConfig(282, 2051, 32, 1, 4, 64, 640, dropout=.2))
        net.eval()
        loss = 0
        for b in batches:
            out = net(b.tokens, b.lengths, b.ctx_z, b.seq_ctx)
            _, report, parts = objective(out, b)
            loss = loss + sum(OBJECTIVE['coefficients'][k] * n / total[k] for k, (n, d) in parts.items() if total[k])
        loss.backward()
        grads = {name: p.grad for name, p in net.named_parameters()}
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for name, g in grads.items() if 'q_head' in name))
        shared = [name for name in grads if 'tok_emb' in name]
        self.assertTrue(shared and all(grads[name] is not None and grads[name].abs().sum() > 0 for name in shared))

    # ------------------------------------------------------------------ A1: appended reference <-> row bijection
    def test_a1_extra_appended_reference_without_row_refused_before_root(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015), ('(pipe X (ridge alpha=4.0))', .02))
        raw, pred = self.manifest(self.A, B, refs, rows_for=refs[:1])
        self.assertEqual(len(json.loads(raw)['rows']), 1)
        with self.assertRaisesRegex(ValueError, 'exact bijection'):
            self.prepare(self.A, B, refs, name='orphan-tail', manifest=raw, predecessor=pred)
        self.assertFalse((self.base / 'orphan-tail').exists())
        self.assertEqual(self.store.reads, 0, 'refused before any artifact read or composition')

    def test_a1_duplicate_orphan_and_mismatched_rows_refused(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, pred = self.manifest(self.A, B, refs)
        base = json.loads(raw)
        old_ref = self.A['tasks'][0]['program_results'][0]

        def dup(m):
            row = copy.deepcopy(m['rows'][0])
            row['evaluation_key'] = 'f' * 64
            m['rows'].append(row)
            m['census']['appended'].append({'evaluation_key': 'f' * 64})
            m['counts']['appended'] += 1
        cases = {
            'duplicate row for one appended reference': (dup, 'duplicate row'),
            'orphan row (old reference)': (lambda m: m['rows'][0].update(result_ref=old_ref), 'exact bijection'),
            'mismatched context': (lambda m: m['rows'][0].update(context_id='c' * 64), 'exact bijection'),
            'mismatched evaluation key': (lambda m: m['rows'][0].update(evaluation_key='0' * 64), 'wrapper identity'),
            'mismatched program': (lambda m: m['rows'][0].update(program_sha256='0' * 64), 'wrapper identity'),
        }
        for why, (edit, message) in cases.items():
            m = copy.deepcopy(base)
            edit(m)
            m['rows'].sort(key=lambda r: r['evaluation_key'])
            bad = xcanonical(m)
            name = 'bad-' + str(abs(hash(why)))
            with self.subTest(why=why):
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare(self.A, B, refs, name=name, manifest=bad, predecessor=pred)
                self.assertFalse((self.base / name).exists())

    # ------------------------------------------------------------------ A2: production contract binding
    def test_a2_contract_tamper_refused_before_root(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, pred = self.manifest(self.A, B, refs)
        base = json.loads(raw)

        def remanifest(edit):
            m = copy.deepcopy(base)
            edit(m)
            return xcanonical(m)
        good = W.contract(self.pre)
        cases = {
            'objective bytes': dict(objective_bytes=W.OBJECTIVE_BYTES + b' '),
            'objective coefficients': dict(objective_bytes=canonical({**json.loads(W.OBJECTIVE_BYTES),
                                                                      'coefficients': {'ce': 1.}}),
                                           contract={**good, 'objective_sha256': sha(canonical(
                                               {**json.loads(W.OBJECTIVE_BYTES), 'coefficients': {'ce': 1.}}))}),
            'manifest objective': dict(manifest=remanifest(lambda m: m.update(objective_sha256='4' * 64))),
            'normalizer bytes': dict(preprocessing_bytes=self.pre + b' '),
            'roster weight': dict(manifest=remanifest(lambda m: m['roster'].update(weight_sha256='8' * 64))),
            'roster dimension': dict(contract={**good, 'roster': {**good['roster'], 'dimension': 2052}}),
            'labels source pin': dict(manifest=remanifest(
                lambda m: m['construction_sources']['labels'].update(sha256='6' * 64))),
            'compose source bytes': dict(contract={**good, 'construction_sources': {**good['construction_sources'],
                                                                                    'compose': '7' * 64}}),
            'exporter source packet': dict(export_provenance={**W.PROVENANCE, 'x1_source_packet_sha256': 'a' * 64}),
            'X1 request provenance': dict(export_provenance={**W.PROVENANCE, 'request_sha256': '0' * 64}),
            'authority proof contract': dict(manifest=remanifest(
                lambda m: m['authority_proofs'][1].update(objective_sha256='4' * 64))),
            'role drift across export': dict(manifest=remanifest(
                lambda m: m['authority_proofs'][1].update(parent_roles={}))),
        }
        for why, over in cases.items():
            name = 'contract-' + str(abs(hash(why)))
            manifest = over.pop('manifest', raw)
            with self.subTest(why=why):
                with self.assertRaises(ValueError):
                    self.prepare(self.A, B, refs, name=name, manifest=manifest, predecessor=pred, **over)
                self.assertFalse((self.base / name).exists())

    def test_a2_seven_field_predecessor_refs_accepted_with_unchanged_bytes(self):
        annotated = copy.deepcopy(self.A)
        annotated['tasks'][0]['program_results'] = [W.annotate(r) for r in annotated['tasks'][0]['program_results']]
        B, refs = self.append(annotated, ('(pipe X (ridge alpha=3.0))', .015))
        receipt, _ = self.prepare(annotated, B, refs, name='annotated')
        extension = json.loads((Path(receipt['root']) / 'extension.json').read_bytes())
        self.assertEqual(extension['tasks'][0]['program_results'][:2], annotated['tasks'][0]['program_results'])
        self.assertEqual(set(extension['tasks'][0]['program_results'][0]), set(W.annotate(refs[0])))
        self.assertEqual(receipt['dispositions'], {'record': 1})
        duplicate = copy.deepcopy(B)
        duplicate['tasks'][0]['program_results'].append(W.annotate(refs[0]))
        raw, pred = self.manifest(annotated, duplicate, refs + [W.annotate(refs[0])], rows_for=refs)
        with self.assertRaisesRegex(ValueError, 'duplicate appended reference'):
            self.prepare(annotated, duplicate, refs, name='alias', manifest=raw, predecessor=pred)

    # ------------------------------------------------------------------ A3: single clock and audited store
    def test_a3_deadline_expiry_after_compose_or_between_rows_leaves_no_status(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015), ('(pipe X (ridge alpha=4.0))', .02))
        raw, pred = self.manifest(self.A, B, refs)
        real_time, real_compose, real_create, real_recompute = time.time, V.compose, C._create, C.recompute_label
        for stage in ('after_compose', 'between_rows', 'before_status'):
            with self.subTest(stage=stage):
                deadline = real_time() + 60
                state = {'expired': False, 'rows': 0}
                # Only the preparation module's wall clock jumps past the deadline once the trigger fires.
                C.time = types.SimpleNamespace(time=lambda: deadline + 1 if state['expired'] else real_time(),
                                               monotonic=time.monotonic, time_ns=time.time_ns)

                def compose(*a, **k):
                    out = real_compose(*a, **k)
                    state['expired'] |= stage == 'after_compose'
                    return out

                def recompute(labels_, reader, row):
                    out = real_recompute(labels_, reader, row)
                    state['rows'] += 1
                    state['expired'] |= stage == 'between_rows' and state['rows'] == 1
                    return out

                def create(path, body):
                    out = real_create(path, body)
                    state['expired'] |= stage == 'before_status' and Path(path).name == 'c10a-receipt.json'
                    return out
                V.compose, C.recompute_label, C._create = compose, recompute, create
                name = 'clock-' + stage
                try:
                    with self.assertRaisesRegex(ValueError, 'C10a work deadline expired'):
                        self.prepare(self.A, B, refs, name=name, manifest=raw, predecessor=pred, deadline=deadline)
                finally:
                    C.time, V.compose, C.recompute_label, C._create = time, real_compose, real_recompute, real_create
                root = self.base / name
                self.assertFalse((root / 'status.json').exists(), 'no successful status after expiry')
                if stage == 'before_status':
                    self.assertTrue((root / 'c10a-receipt.json').exists())
                    with self.assertRaises(OSError):
                        C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=self.acceptances.get(str(root)))
                else:
                    self.assertFalse(root.exists())

    def test_a3_network_attempting_or_unaudited_store_cannot_claim_zero_network(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        with self.assertRaisesRegex(ValueError, 'network-attempting store'):
            self.prepare(self.A, B, refs, name='network', store=W.AuditedStore(self.f.store, network=True))
        self.assertFalse((self.base / 'network').exists())
        with self.assertRaisesRegex(ValueError, 'must attest closed exact-version local reads'):
            self.prepare(self.A, B, refs, name='unaudited', store=self.f.store)
        self.assertFalse((self.base / 'unaudited').exists())
        with self.assertRaisesRegex(ValueError, 'shared read budget|read budget'):
            self.prepare(self.A, B, refs, name='budget', read_budget_bytes=4096)
        self.assertFalse((self.base / 'budget').exists())

    # ------------------------------------------------------------------ r3: paired clock and pre-read budget
    def test_a3r3_monotonic_or_boot_change_expires_the_paired_clock(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, pred = self.manifest(self.A, B, refs)
        real_compose = V.compose
        for case in ('monotonic', 'boot'):
            with self.subTest(case=case):
                state = {'fired': False}
                real_mono, real_boot = time.monotonic, C._boot_id
                C.time = types.SimpleNamespace(time=time.time, time_ns=time.time_ns,
                                               monotonic=lambda: real_mono() + (10**6 if state['fired'] else 0))
                if case == 'boot':
                    C.time = types.SimpleNamespace(time=time.time, time_ns=time.time_ns, monotonic=real_mono)
                    C._boot_id = lambda: 'rebooted' if state['fired'] else real_boot()

                def compose(*a, **k):
                    out = real_compose(*a, **k)
                    state['fired'] = True               # wall clock stays valid; only mono / boot change
                    return out
                V.compose = compose
                name = 'r3-clock-' + case
                try:
                    with self.assertRaisesRegex(ValueError, 'C10a work deadline expired'):
                        self.prepare(self.A, B, refs, name=name, manifest=raw, predecessor=pred)
                finally:
                    C.time, V.compose, C._boot_id = time, real_compose, real_boot
                self.assertFalse((self.base / name).exists())

    def test_a3r3_combined_budget_refuses_before_any_overshoot(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, pred = self.manifest(self.A, B, refs)
        full, _ = self.prepare(self.A, B, refs, name='r3-budget-full', manifest=raw, predecessor=pred)
        need_bytes = full['read_budget']['charged']
        self.assertGreater(need_bytes, 0)
        for budget in (need_bytes - 1, need_bytes // 2, 1):
            with self.subTest(budget=budget):
                store = W.AuditedStore(self.f.store)
                name = 'r3-budget-' + str(budget)
                # Either pre-read guard may fire first (the combined budget or the frozen reader's own limit); both
                # refuse BEFORE the read, which is the property under test.
                with self.assertRaisesRegex(ValueError, 'combined read budget would be exceeded|artifact read budget'):
                    self.prepare(self.A, B, refs, name=name, manifest=raw, predecessor=pred, store=store,
                                 read_budget_bytes=budget)
                self.assertLessEqual(store.read_bytes, budget, 'refused before any overshoot')
                self.assertFalse((self.base / name).exists())

    # ------------------------------------------------------------------ r5: anchor order and post-durable qualification
    def test_a3r5_clock_anchors_boot_and_monotonic_before_wall(self):
        order, real_mono, real_boot = [], time.monotonic, C._boot_id
        C.time = types.SimpleNamespace(time=lambda: order.append('wall') or time.time(), time_ns=time.time_ns,
                                       monotonic=lambda: order.append('mono') or real_mono())
        C._boot_id = lambda: order.append('boot') or real_boot()
        try:
            C.Clock(time.time() + 60)
        finally:
            C.time, C._boot_id = time, real_boot
        self.assertEqual(order[:3], ['boot', 'mono', 'wall'])

    def test_a3r5_expiry_after_durable_status_leaves_root_unqualified(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, pred = self.manifest(self.A, B, refs)
        deadline, state = time.time() + 60, {'expired': False}
        real_time, real_create = time.time, C._create
        C.time = types.SimpleNamespace(time=lambda: deadline + 1 if state['expired'] else real_time(),
                                       monotonic=time.monotonic, time_ns=time.time_ns)

        def create(path, body):
            out = real_create(path, body)
            state['expired'] |= Path(path).name == 'status.json'
            return out
        C._create = create
        try:
            with self.assertRaisesRegex(ValueError, 'after durable status'):
                self.prepare(self.A, B, refs, name='r5-late', manifest=raw, predecessor=pred, deadline=deadline)
        finally:
            C.time, C._create = time, real_create
        root = self.base / 'r5-late'
        self.assertTrue((root / 'status.json').exists())
        self.assertFalse((root / 'c10a-terminal.json').exists(), 'no qualified success')
        with self.assertRaises(OSError):
            C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=self.acceptances.get(str(root)))

    def test_a3r5_terminal_outside_its_bound_is_refused(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        receipt, _ = self.prepare(self.A, B, refs, name='r5-ok')
        root = Path(receipt['root'])
        status_raw, receipt_raw = (root / 'status.json').read_bytes(), (root / 'c10a-receipt.json').read_bytes()
        good = json.loads((root / 'c10a-terminal.json').read_bytes())
        C.check_terminal(canonical(good), status_raw=status_raw, receipt_raw=receipt_raw)
        for why, edit in {'late wall': lambda t: t['qualified_after_durable_status'].update(
                              wall_unix=t['bound']['wall_deadline_unix'] + 1),
                          'late monotonic': lambda t: t['qualified_after_durable_status'].update(
                              monotonic=t['bound']['monotonic_deadline'] + 1),
                          'other boot': lambda t: t['qualified_after_durable_status'].update(boot_id='other'),
                          'other status': lambda t: t.update(status_sha256='0' * 64)}.items():
            bad = json.loads(canonical(good))
            edit(bad)
            with self.subTest(why=why), self.assertRaises(ValueError):
                C.check_terminal(canonical(bad), status_raw=status_raw, receipt_raw=receipt_raw)


    # ------------------------------------------------------------------ r7: bound authority, final publication, last sample
    def swap(self, path, body):
        os.unlink(path)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        os.write(fd, body)
        os.close(fd)

    def test_a3r7_replaced_marker_bound_and_sample_refused(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        receipt, _ = self.prepare(self.A, B, refs, name='r7-bound')
        root, accepted = Path(receipt['root']), self.acceptances[receipt['root']]
        status_raw, receipt_raw = (root / 'status.json').read_bytes(), (root / 'c10a-receipt.json').read_bytes()
        terminal_path = root / 'c10a-terminal.json'
        original = terminal_path.read_bytes()
        good = json.loads(original)
        b = good['bound']
        # (a) bound AND sample replaced together, original status/receipt bytes and hashes retained
        extended = json.loads(original)
        extended['bound'] = {**b, 'wall_deadline_unix': b['wall_deadline_unix'] + 3600,
                             'monotonic_deadline': b['monotonic_deadline'] + 3600}
        extended['qualified_after_durable_status'] = {'boot_id': b['boot_id'], 'wall_unix': b['wall_deadline_unix'] + 10,
                                                      'monotonic': b['monotonic_deadline'] + 10}
        with self.assertRaisesRegex(ValueError, 'terminal bound is the sealed receipt clock bound'):
            C.check_terminal(xcanonical(extended), status_raw=status_raw, receipt_raw=receipt_raw)
        self.swap(terminal_path, xcanonical(extended))
        with self.assertRaisesRegex(ValueError, 'terminal bound is the sealed receipt clock bound'):
            C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=accepted)
        refused_path = self.base / 'acceptances' / 'r7-bound-replaced.json'
        with self.assertRaisesRegex(ValueError, 'acceptance refused'):
            C.accept_root(root, acceptance_path=refused_path,
                          physical={'kind': 'in_process_return', 'returned_sha256': sha(receipt_raw)})
        custody = json.loads(refused_path.read_bytes())
        self.assertEqual((custody['verdict'], custody['marker']['sha256']), ('refused', sha(xcanonical(extended))))
        self.assertEqual((root / 'status.json').read_bytes(), status_raw)
        # (b) a different in-bound sample under the ORIGINAL bound: the acceptance pins the exact marker bytes
        other = json.loads(original)
        other['qualified_after_durable_status']['wall_unix'] -= 1e-3
        C.check_terminal(xcanonical(other), status_raw=status_raw, receipt_raw=receipt_raw)
        self.swap(terminal_path, xcanonical(other))
        with self.assertRaisesRegex(ValueError, 'acceptance pins the exact final marker bytes'):
            C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=accepted)
        # (c) original-bound tamper: receipt clock extended and the marker re-sealed consistently
        forged_receipt = json.loads(receipt_raw)
        forged_receipt['clock']['wall_deadline_unix'] += 3600
        forged_receipt['clock']['monotonic_deadline'] += 3600
        forged_receipt_raw = xcanonical(forged_receipt)
        resealed = {**good, 'receipt_sha256': sha(forged_receipt_raw), 'bound': C.sealed_bound(forged_receipt['clock'])}
        C.check_terminal(xcanonical(resealed), status_raw=status_raw, receipt_raw=forged_receipt_raw)
        self.swap(root / 'c10a-receipt.json', forged_receipt_raw)
        self.swap(terminal_path, xcanonical(resealed))
        with self.assertRaisesRegex(ValueError, 'acceptance pins the exact'):
            C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=accepted)
        self.swap(root / 'c10a-receipt.json', receipt_raw)
        self.swap(terminal_path, original)
        self.assertEqual(C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=accepted)['prepared'],
                         receipt['prepared'])

    def test_a3r7_expiry_during_final_terminal_publication_is_refused(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, pred = self.manifest(self.A, B, refs)
        deadline, state = time.time() + 60, {'expired': False}
        real_time, real_create = time.time, C._create
        C.time = types.SimpleNamespace(time=lambda: deadline + 1 if state['expired'] else real_time(),
                                       monotonic=time.monotonic, time_ns=time.time_ns)

        def create(path, body):
            out = real_create(path, body)
            state['expired'] |= Path(path).name == 'c10a-terminal.json'   # the deadline passes during the final write
            return out
        C._create = create
        root = self.base / 'r7-late-final'
        try:
            with self.assertRaisesRegex(ValueError, 'after final terminal publication'):
                self.prepare(self.A, B, refs, name='r7-late-final', manifest=raw, predecessor=pred, deadline=deadline)
            receipt_raw = (root / 'c10a-receipt.json').read_bytes()
            # the operator samples AFTER the late publication: refused, kept as custody
            for tag, physical in (('returned', {'kind': 'in_process_return', 'returned_sha256': sha(receipt_raw)}),
                                  ('raised', {'kind': 'in_process_return', 'returned_sha256': None})):
                with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'acceptance refused'):
                    C.accept_root(root, acceptance_path=self.base / 'acceptances' / ('late-' + tag + '.json'),
                                  physical=physical)
        finally:
            C.time, C._create = time, real_create
        terminal = json.loads((root / 'c10a-terminal.json').read_bytes())     # the late marker is preserved
        b = terminal['bound']
        late_unit = {'kind': 'systemd_unit', 'unit': 'c10a-prepare.service', 'result': 'success', 'exec_main_status': 0,
                     'exit_monotonic_usec': int((b['monotonic_deadline'] + 1) * 1e6), 'boot_id': b['boot_id']}
        for tag, edit in (('exit after the bound', {}), ('failed unit', {'result': 'exit-code'}),
                          ('nonzero status', {'exec_main_status': 1}), ('other boot', {'boot_id': 'other'})):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'acceptance refused'):
                C.accept_root(root, acceptance_path=self.base / 'acceptances' / ('late-unit-' + tag.replace(' ', '-') + '.json'),
                              physical={**late_unit, **edit})
        refused = self.base / 'acceptances' / 'late-returned.json'
        ref = {'path': str(refused), 'sha256': sha(refused.read_bytes()), 'bytes': len(refused.read_bytes())}
        self.assertEqual(json.loads(refused.read_bytes())['verdict'], 'refused')
        with self.assertRaisesRegex(ValueError, 'accepted bounded terminal envelope'):
            C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=ref)
        self.assertNotIn(str(root), self.acceptances)

    def test_a3r7_systemd_unit_in_time_success_is_accepted(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        receipt, _ = self.prepare(self.A, B, refs, name='r7-unit')
        root = Path(receipt['root'])
        b = json.loads((root / 'c10a-terminal.json').read_bytes())['bound']
        unit = {'kind': 'systemd_unit', 'unit': 'c10a-prepare.service', 'result': 'success', 'exec_main_status': 0,
                'exit_monotonic_usec': int(time.monotonic() * 1e6), 'boot_id': b['boot_id']}
        ref = C.accept_root(root, acceptance_path=self.base / 'acceptances' / 'r7-unit-systemd.json', physical=unit)
        self.assertEqual(C.candidate_descriptor(root, campaign_sha256='e' * 64, acceptance=ref)['prepared'],
                         receipt['prepared'])
        with self.assertRaisesRegex(ValueError, 'create-only acceptance outside the subject root'):
            C.accept_root(root, acceptance_path=root / 'acceptance.json', physical=unit)

    def test_a3r7_qualify_rechecks_monotonic_and_boot_after_wall(self):
        clock = C.Clock(time.time() + 60)
        self.assertTrue(C.in_bound(clock.qualify('baseline'), clock.bound()))
        real_time, real_mono, real_boot = time.time, time.monotonic, C._boot_id
        for tag in ('monotonic stall behind a backward wall', 'reboot while sampling the wall'):
            state = {'stalled': False}

            def wall():
                state['stalled'] = True                  # the wall read stalls past the monotonic bound...
                return real_time() - 100                 # ...and the wall clock was stepped backwards meanwhile
            mono = (lambda: real_mono() + (10 ** 6 if state['stalled'] else 0)) if tag.startswith('monotonic') else real_mono
            C.time = types.SimpleNamespace(time=wall, monotonic=mono, time_ns=time.time_ns)
            if tag.startswith('reboot'):
                C._boot_id = lambda: 'rebooted' if state['stalled'] else real_boot()
            try:
                with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'C10a work deadline expired at late'):
                    clock.qualify('late')
            finally:
                C.time, C._boot_id = time, real_boot


if __name__ == '__main__':
    unittest.main()
