"""W3 native_result_overlay_v1 revision 2 private CPU fixtures (synthetic world in world.py; not live evidence).

Each fixture names the plan v2.0 §5 W3 acceptance item it covers; revision 2 adapts the revision-1 fixtures to the
DELTA chain API and adds targeted regressions (``test_r2_*``) for the delta chain, SHARD_RMSE_DISAGREEMENT and the
exact-bytes emission body. No S3, GPU, GP/program execution, planner, optimizer step or live outbox: the outbox is a
fixture-local frozen Outbox under TMPDIR.
"""
import base64
import copy
import json
import math
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dltrain import labels as L
from jope_flywheel import native_variant_snapshot as VS
from jope_flywheel import training_snapshot_native as N
from jope_flywheel.protocol import canonical, digest
from neural_native_train import prepare_microbatches

from pipelines.native_result_overlay_v1 import build as B
from pipelines.native_result_overlay_v1 import common as C
from pipelines.native_result_overlay_v1 import packet as PK
from pipelines.native_result_overlay_v1 import verify as V
from pipelines.native_result_overlay_v1 import (completed_packet, head_sha256, overlay_ref_digest, overlay_snapshot,
                                                validate_chain, validate_packet)
from pipelines.native_result_overlay_v1.labels_overlay import label_row, record_json
from pipelines.native_result_overlay_v1.loader import normalizer_from
from pipelines.native_result_overlay_v1.tests import world as W


def scenario1(w):
    """Packet 1 source rows (31): every census class and every dedup/refusal path, in emission order."""
    A, Bc = w.ctx['A'], w.ctx['B']
    out = {'a_new': [w.result('A', f'(pipe X (ridge alpha={100 + j}.0))', [b * (0.40 + 0.02 * j) for b in A['recon']])
                     for j in range(10)]}
    rows = [w.row('A', r) for r in out['a_new']]
    rows.append(w.row('A', out['a_new'][0], raw='  (pipe X   (ridge alpha=100.0))  '))
    shard = A['shard']['doc']['records']
    out['dup_agree'] = w.result('A', '(pipe X (ridge alpha=3.0))', shard[2]['rmse_j'])
    out['dup_differ'] = w.result('A', '(pipe X (ridge alpha=4.0))', [v * 1.5 for v in shard[3]['rmse_j']])
    rows += [w.row('A', out['dup_agree']), w.row('A', out['dup_differ'])]
    out['c1'] = w.result('A', '(pipe X (ridge alpha=120.0))')
    out['c2'] = w.result('A', '(pipe X (ridge alpha=120.0))', wrapper_key='gp/native/conflicting-copy/record.json')
    rows += [w.row('A', out['c1']), w.row('A', out['c2'])]
    out['bad_eval'] = w.result('A', '(pipe X (ridge alpha=121.0))',
                               evaluation_baseline=[A['recon'][0] * (1 + 1e-7), A['recon'][1]])
    rows.append(w.row('A', out['bad_eval']))
    rows.append(w.row('A', w.result('A', W.LONG_PROGRAM)))
    rows.append(w.row('A', program='(pipe X (ridge alpha=122.0))', state='failure_custody_verified'))
    for i, state in enumerate(('recorded', 'leased', 'unknown_attempt')):
        rows.append(w.row('A', program=f'(pipe X (ridge alpha={130 + i}.0))', state=state))
    rows.append(w.row('A', w.result('A', '(pipe X (ridge alpha=140.0))'), campaign=W.OTHER_CAMPAIGN))
    out['b_new'] = [w.result('B', f'(pipe X (ridge alpha={200 + j}.0))', [b * (0.5 + 0.05 * j) for b in Bc['recon']])
                    for j in range(3)] + [w.result('B', Bc['winner'], [b * 0.3 for b in Bc['recon']])]
    rows += [w.row('B', r) for r in out['b_new']]
    rows.append(w.row('C', w.result('C', '(pipe X (ridge alpha=300.0))'), phase='validation'))
    rows.append(w.row('C', w.result('C', '(pipe X (ridge alpha=301.0))'), phase='validation'))
    rows.append(w.row('C', w.result('C', '(pipe X (ridge alpha=302.0))'), phase='training'))
    out['d'] = w.result('D', '(pipe X (ridge alpha=400.0))')
    rows.append(w.row('D', out['d']))
    rows.append(w.variant_row())
    return rows, out


EXPECTED_CLASSES = {'TRAIN_VERIFIED': 16, 'EVALUATION_ONLY': 3, 'ROUTED_TO_X1': 1, 'REFUSED': 8, 'PENDING': 3}
EXPECTED_REASONS = {('TRAIN_VERIFIED', 'NEW_OVERLAY_ROW'): 14, ('TRAIN_VERIFIED', 'ALIAS'): 1,
                    ('TRAIN_VERIFIED', 'ALREADY_IN_SHARD'): 1, ('REFUSED', 'SHARD_RMSE_DISAGREEMENT'): 1,
                    ('EVALUATION_ONLY', 'VALIDATION_PHASE'): 2,
                    ('EVALUATION_ONLY', 'VALIDATION_PARTITION'): 1, ('ROUTED_TO_X1', 'VARIANT_SOURCE'): 1,
                    ('REFUSED', 'CONFLICTING_REFS'): 2, ('REFUSED', 'BASELINE_DISAGREEMENT'): 2,
                    ('REFUSED', 'TOKEN_REFUSAL'): 1, ('REFUSED', 'PARITY_UNQUALIFIED'): 1,
                    ('REFUSED', 'CAMPAIGN_NOT_ALLOWED'): 1, ('PENDING', 'UNRESOLVED_STATE'): 3}


def packet_of(receipt):
    return validate_packet(receipt['packet'], predecessor_head=receipt['predecessor_overlay_head_sha256'])


def view_of(*receipts):
    """The validated cumulative view of a chain given its build receipts in order."""
    return validate_chain([r['packet'] for r in receipts], expected_head=receipts[-1]['head_sha256'])


def after(*receipts):
    """build_packet keyword arguments appending to the chain whose receipts are given in order."""
    return {'predecessor_chain': [r['packet'] for r in receipts], 'predecessor_head': receipts[-1]['head_sha256']}


def forge(path, packet, mutate):
    """Re-seal ``packet`` after ``mutate`` (packet-internal consistency is the mutator's job); returns its ref."""
    doc = copy.deepcopy(packet)
    mutate(doc)
    return write(path, canonical(PK.validate_doc(doc)))


def write(path, body):
    Path(path).write_bytes(body)
    return {'path': str(path), 'sha256': C.sha(body), 'bytes': len(body)}


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.w = W.World(self.root)
        self.addCleanup(self.w.outbox.close)

    def packet1(self, **over):
        rows, out = scenario1(self.w)
        receipt = self.w.build(rows, 'packet-1', **over)
        self.view1 = view_of(receipt)
        return receipt, packet_of(receipt), out

    def entries(self, packet, **match):
        return [e for e in packet['census']['entries'] if all(e[k] == v for k, v in match.items())]

    # ------------------------------------------------------------------ labels
    def test_direct_recomputation_equality(self):
        receipt, p, _ = self.packet1()
        d = receipt['direct_recomputation']
        self.assertEqual((d['rows'], d['all_equal']), (14, True))
        for row in p['rows']:
            cid = row['context_id']
            y = (self.w.ctx['A'] if cid == self.w.ctx['A']['cid'] else self.w.ctx['B'])['y'].tolist()
            s_c = L.robust_scale(y)
            eps = L.epsilon(y, s_c)
            q = L.q_from_detail(row['inner_fold_rmse'], row['baseline']['anchored'], eps)
            rec = row['record']
            self.assertEqual(rec['q_grid'], L.q_grid(q))
            self.assertIs(type(rec['q_grid']), int)
            self.assertTrue(math.isclose(rec['q'], q, rel_tol=1e-12, abs_tol=0.))
            self.assertTrue(math.isclose(rec['q_opt'], L.clip_q(q), rel_tol=1e-12, abs_tol=0.))
            self.assertEqual(rec['feasibility'], 0)
            wall = row['wall_us']
            self.assertTrue(math.isclose(rec['runtime_seconds'], math.expm1(math.log1p(wall / 1e6)), rel_tol=1e-12))
            self.assertEqual((rec['quality_eligible'], rec['pairwise_eligible'], rec['runtime_eligible'],
                              rec['ce_eligible'], rec['source_kind']), (True, True, True, False, C.SOURCE_KIND))
        # The loader serves exactly the packet records.
        s = overlay_snapshot(self.w.nat, self.view1, max_padded_tokens=8192)
        b = s.context_loader.get([self.w.ctx['A']['cid']], epoch=0, batch_index=0)
        served = {r.program_sha256: record_json(r) for r in b.records[self.w.ctx['A']['cid']]
                  if r.source_kind == C.SOURCE_KIND}
        self.assertEqual(len(served), 8)
        packet_records = {r['program_sha256']: r['record'] for r in p['rows']}
        self.assertTrue(all(packet_records[k] == v for k, v in served.items()))

    def test_old_shard_rows_byte_unchanged_and_old_labels_unchanged(self):
        A = self.w.ctx['A']
        shard_path = '/fixture/tails/' + A['shard']['rel']
        before_bytes = self.w.store.b[shard_path]
        base = self.w.nat.context_loader.get([A['cid']], epoch=3, batch_index=1)
        receipt, p, _ = self.packet1()
        self.assertEqual(self.w.store.b[shard_path], before_bytes)
        self.assertEqual(C.sha(before_bytes), A['shard']['ref']['sha256'])
        self.assertEqual(receipt['old_labels_unchanged'][A['cid']], {'shard_rows': 12, 'union_equals_shard_only': True})
        self.assertTrue(receipt['frozen_baseline_verified'][A['cid']])
        # Union labels of every shard row equal the frozen shard-only labels on every NativeRecord field.
        doc = A['shard']['doc']
        _, shard_only = L.label_context(doc['context'], doc['records'])
        extra = [label_row(r['context_id'], r['program_sha256'], r['record']['tokens'], r['record']['skeleton_sha256'],
                           r['inner_fold_rmse'], r['baseline']['anchored'], r['wall_us'])
                 for r in p['rows'] if r['context_id'] == A['cid']]
        _, union = L.label_context(doc['context'], doc['records'] + extra)
        union = {x.program_sha256: x for x in union}
        for lab in shard_only:
            for f in ('q', 'q_opt', 'q_grid', 'q_status', 'feas_class', 'rt_target', 'rt_valid', 'rt_censored'):
                self.assertEqual(getattr(lab, f), getattr(union[lab.program_sha256], f))
        # The frozen base window is unchanged inside the overlay batch (same records, same order, nested receipt).
        s = overlay_snapshot(self.w.nat, self.view1, max_padded_tokens=8192)
        ob = s.context_loader.get([A['cid']], epoch=3, batch_index=1)
        n = len(base.records[A['cid']])
        self.assertEqual(ob.records[A['cid']][:n], base.records[A['cid']])
        self.assertEqual(ob.selection_receipt['base'], base.selection_receipt)
        self.assertEqual(ob.selection_receipt['schema'], C.WINDOW_SCHEMA)

    def test_anchoring_to_the_exact_shard_baseline_pair(self):
        receipt, p, _ = self.packet1()
        A = self.w.ctx['A']
        pair = A['shard']['pair']
        self.assertNotEqual(pair, A['recon'])
        self.assertTrue(V.agree(pair, A['recon']))
        self.assertEqual(p['contexts'][A['cid']]['anchored_baseline'], pair)
        self.assertEqual(p['contexts'][A['cid']]['baseline_source'], 'shard')
        a_rows = [r for r in p['rows'] if r['context_id'] == A['cid']]
        self.assertEqual(len(a_rows), 10)
        for r in a_rows:
            self.assertEqual(r['baseline'], {'reconstructed': A['recon'], 'evaluation': A['recon'], 'anchored': pair,
                                             'source': 'shard'})
            self.assertEqual(r['shard_ref'], A['shard']['ref'])
        # Without anchoring the frozen bit-equality rule of label_context refuses the union.
        doc = A['shard']['doc']
        loose = [label_row(A['cid'], r['program_sha256'], r['record']['tokens'], r['record']['skeleton_sha256'],
                           r['inner_fold_rmse'], A['recon'], r['wall_us']) for r in a_rows]
        with self.assertRaises(L.LabelError):
            L.label_context(doc['context'], doc['records'] + loose)

    def test_baseline_disagreement_refused(self):
        _, p, out = self.packet1()
        bad = self.entries(p, evaluation_key=out['bad_eval']['key'])
        d = self.entries(p, evaluation_key=out['d']['key'])
        self.assertEqual([(e['class'], e['reason']) for e in bad + d], [('REFUSED', 'BASELINE_DISAGREEMENT')] * 2)
        self.assertIn('evaluation baseline', bad[0]['detail'])
        self.assertIn('shard baseline', d[0]['detail'])
        self.assertNotIn(self.w.ctx['D']['cid'], p['contexts'])
        self.assertFalse(any(r['evaluation_key'] in (out['bad_eval']['key'], out['d']['key']) for r in p['rows']))

    def test_no_shard_context_path_and_chain_anchor(self):
        r1, p1, _ = self.packet1()
        Bc = self.w.ctx['B']
        c = p1['contexts'][Bc['cid']]
        self.assertEqual((c['baseline_source'], c['shard_ref'], c['anchored_baseline']), ('reconstructed', None, Bc['recon']))
        self.assertEqual(c['label_context'], {'context_sha256': Bc['cid'], 'dataset_hash': Bc['dataset'],
                                              'family_id': Bc['lineage'], 'split_role': 'train_fit'})
        self.assertEqual(c['y_train_sha256'], C.sha(Bc['y'].tobytes()))
        res = self.w.result('B', '(pipe X (ridge alpha=250.0))')
        r2 = self.w.build([self.w.row('B', res)], 'packet-2', **after(r1))
        p2 = validate_packet(r2['packet'], predecessor_head=r1['head_sha256'])
        new = p2['rows'][-1]
        self.assertEqual((new['baseline']['source'], new['baseline']['anchored']), ('chain', p1['contexts'][Bc['cid']]['anchored_baseline']))
        self.assertNotIn(Bc['cid'], p2['contexts'])          # delta packet: the first anchor stays in packet 1
        self.assertEqual(view_of(r1, r2).contexts[Bc['cid']], c)

    # ------------------------------------------------------------------ roles and dedup
    def test_validation_rows_are_evaluation_only(self):
        _, p, _ = self.packet1()
        Cc = self.w.ctx['C']
        es = self.entries(p, context_id=Cc['cid'])
        self.assertEqual(sorted((e['class'], e['reason']) for e in es),
                         [('EVALUATION_ONLY', 'VALIDATION_PARTITION')] + [('EVALUATION_ONLY', 'VALIDATION_PHASE')] * 2)
        self.assertNotIn(Cc['cid'], p['contexts'])
        s = overlay_snapshot(self.w.nat, self.view1, max_padded_tokens=8192)
        self.assertNotIn(Cc['cid'], s.overlay_records)
        # R6 roster: a train lineage in the pre-registered dev roster is evaluation-only, never a target.
        A = self.w.ctx['A']
        roster = canonical(W.World.r6_doc([], [A['lineage']]))
        r = self.w.build([self.w.row('A', self.w.result('A', '(pipe X (ridge alpha=150.0))'))], 'r6-root',
                         r6_roster_bytes=roster, r6_roster_sha256=C.sha(roster))
        pr = packet_of(r)
        self.assertEqual([(e['class'], e['reason']) for e in pr['census']['entries']], [('EVALUATION_ONLY', 'R6_ROSTER')])
        self.assertEqual(pr['rows'], [])

    def test_alias_dedup_keeps_provenance(self):
        _, p, out = self.packet1()
        first = out['a_new'][0]
        row = next(r for r in p['rows'] if r['evaluation_key'] == first['key'])
        es = self.entries(p, evaluation_key=first['key'])
        self.assertEqual([e['reason'] for e in es], ['NEW_OVERLAY_ROW', 'ALIAS'])
        self.assertEqual(row['source_emission_ids'], [e['emission_id'] for e in es])
        self.assertEqual(row['source_observations'], [e['observation_sha256'] for e in es])
        self.assertEqual({e['row'] for e in es}, {p['rows'].index(row)})

    def test_already_in_shard_and_conflicting_refs(self):
        receipt, p, out = self.packet1()
        agree = self.entries(p, evaluation_key=out['dup_agree']['key'])[0]
        self.assertEqual((agree['reason'], json.loads(agree['detail'])['agrees']), ('ALREADY_IN_SHARD', True))
        self.assertEqual(len(receipt['shard_rmse_crosschecks']), 2)
        conflict = self.entries(p, evaluation_key=out['c1']['key'])
        self.assertEqual([(e['class'], e['reason']) for e in conflict], [('REFUSED', 'CONFLICTING_REFS')] * 2)
        keys = {r['evaluation_key'] for r in p['rows']}
        self.assertFalse({out['c1']['key'], out['dup_agree']['key'], out['dup_differ']['key']} & keys)

    def test_census_five_classes_sum_exactly(self):
        receipt, p, _ = self.packet1()
        census = p['census']
        self.assertEqual(census['classes'], EXPECTED_CLASSES)
        self.assertEqual(sum(census['classes'].values()), census['source_rows'])
        self.assertEqual(census['source_rows'], 31)
        self.assertEqual(len(census['entries']), 31)
        got = {}
        for e in census['entries']:
            got[(e['class'], e['reason'])] = got.get((e['class'], e['reason']), 0) + 1
        self.assertEqual(got, EXPECTED_REASONS)
        pending = [e['detail'] for e in census['entries'] if e['class'] == 'PENDING']
        self.assertEqual(pending, ['recorded', 'leased', 'unknown_attempt'])
        self.assertEqual(p['sources'][0]['rows'], 31)
        self.assertEqual((p['new_ce_teachers'], p['optimizer_updates'], p['admission_committed']), (0, 0, False))
        # Pending/failed rows never carry a numeric target anywhere in the packet.
        refs = {e['evaluation_key'] for e in census['entries'] if e['class'] in ('PENDING', 'REFUSED')}
        self.assertFalse(refs & {r['evaluation_key'] for r in p['rows']})

    # ------------------------------------------------------------------ outer test never decoded
    def test_outer_test_values_never_decoded(self):
        A = self.w.ctx['A']
        self.assertIn(W.SENTINEL, A['masked'])
        self.assertIn(W.SENTINEL, A['raw'])
        with W.Tripwire() as trip:
            with self.assertRaises(W.OuterTestDecoded):
                np.frombuffer(A['masked'], dtype='<f8', offset=17 + W.N_TRAIN)
        rows, _ = scenario1(self.w)
        with W.Tripwire() as trip:
            receipt = self.w.build(rows, 'tripwire-root')
        self.assertGreater(trip.calls, 0)
        self.assertEqual((receipt['new_rows'], receipt['chain']['rows']), (14, 14))
        self.assertFalse(receipt['outer_test_values_decoded'])
        p = packet_of(receipt)
        self.assertTrue(all(r['original_score_context']['input_sha256'] == C.sha(self.w.ctx['A' if r['context_id'] ==
                            A['cid'] else 'B']['masked']) for r in p['rows']))

    # ------------------------------------------------------------------ loader
    def test_token_budget_stop(self):
        _, p, _ = self.packet1()
        A = self.w.ctx['A']
        base = self.w.nat.context_loader.get([A['cid']], epoch=0, batch_index=0).records[A['cid']]
        overlay_len = max(len(r['record']['tokens']) for r in p['rows'] if r['context_id'] == A['cid'])
        widest = max(max(len(r.tokens) for r in base), overlay_len)
        budget = max(W.MAX_SEQ, (len(base) + 2) * widest)
        self.assertGreater((len(base) + C.OVERLAY_WINDOW) * widest, budget)
        s = overlay_snapshot(self.w.nat, self.view1, max_padded_tokens=budget)
        b = s.context_loader.get([A['cid']], epoch=0, batch_index=0)
        m = b.selection_receipt['members'][0]
        rows = b.records[A['cid']]
        self.assertTrue(m['budget_stopped'])
        self.assertLess(len(m['selected']), C.OVERLAY_WINDOW)
        self.assertLessEqual(len(rows) * max(len(r.tokens) for r in rows), budget)
        nxt = next(r for r in s.context_loader.overlay[A['cid']][m['start']:] + s.context_loader.overlay[A['cid']]
                   if r.program_sha256 not in m['program_ids'])
        self.assertGreater((len(rows) + 1) * max(max(len(r.tokens) for r in rows), len(nxt.tokens)), budget)
        batches, _, receipt = prepare_microbatches(s, [A['cid']], epoch=0, batch_index=0, normalizer=None,
                                                   max_padded_tokens=budget)
        self.assertTrue(all(x.receipt['padded_tokens'] <= budget for x in batches))
        full = overlay_snapshot(self.w.nat, self.view1, max_padded_tokens=8192).context_loader.get([A['cid']], epoch=0,
                                                                                          batch_index=0)
        self.assertEqual(len(full.selection_receipt['members'][0]['selected']), C.OVERLAY_WINDOW)

    def test_winner_merge_keeps_one_ce_record(self):
        _, p, out = self.packet1()
        Bc = self.w.ctx['B']
        winner = self.w.nat.winners[Bc['cid']]
        base = self.w.nat.context_loader.get([Bc['cid']], epoch=0, batch_index=0).records[Bc['cid']]
        self.assertEqual(base, (winner,))
        s = overlay_snapshot(self.w.nat, self.view1, max_padded_tokens=8192)
        b = s.context_loader.get([Bc['cid']], epoch=0, batch_index=0)
        recs = b.records[Bc['cid']]
        self.assertEqual(len(recs), 4)
        self.assertEqual(sum(r.ce_eligible for r in recs), 1)
        merged = [r for r in recs if r.program_sha256 == winner.program_sha256]
        self.assertEqual(len(merged), 1)
        m = merged[0]
        row = next(r for r in p['rows'] if r['program_sha256'] == winner.program_sha256)
        self.assertEqual((m.ce_eligible, m.tokens, m.source_kind), (True, winner.tokens, C.MERGED_SOURCE_KIND))
        self.assertEqual((m.q, m.q_grid, m.feasibility, m.runtime_seconds),
                         (row['record']['q'], row['record']['q_grid'], 0, row['record']['runtime_seconds']))
        self.assertFalse(row['record']['ce_eligible'])
        self.assertEqual(sum(r['record']['ce_eligible'] for r in p['rows']), 0)
        self.assertTrue([x for x in b.selection_receipt['members'][0]['selected'] if x['merged_winner']])
        batches, total, _ = prepare_microbatches(s, [Bc['cid']], epoch=0, batch_index=0, normalizer=None,
                                                 max_padded_tokens=8192)
        self.assertEqual(total['ce'], 1)
        self.assertEqual(total['q'], 1)

    def test_dry_run_every_offset(self):
        receipt, p, _ = self.packet1()
        dry = receipt['dry_run']
        self.assertEqual((dry['passed'], dry['expected_runs'], dry['runs'], dry['contexts']), (True, 6, 6, 2))
        self.assertEqual(dry['never_selected'], {})
        self.assertEqual(sorted({(d['context_id'], d['start']) for d in dry['detail']}),
                         sorted([(self.w.ctx['A']['cid'], s) for s in (0, 8, 6, 4, 2)] + [(self.w.ctx['B']['cid'], 0)]))
        # Independent replay through the real prepare_microbatches + collate at every offset.
        s = overlay_snapshot(self.w.nat, self.view1, max_padded_tokens=8192)
        norm = normalizer_from(json.loads(self.w.pre))
        for d in dry['detail']:
            _, total, rec = prepare_microbatches(s, [d['context_id']], epoch=d['epoch'], batch_index=0, normalizer=norm,
                                                 max_padded_tokens=8192)
            self.assertEqual([x['program_sha256'] for x in rec['selection']['members'][0]['selected']], d['selected'])
            self.assertEqual(total, d['denominators'])

    def test_overlay_snapshot_identity_membership_and_mixed_base(self):
        _, p, _ = self.packet1()
        nat = self.w.nat
        panel_ids = sorted(c for c, r in nat.contexts.items() if r['partition'] == 'train' and c != self.w.ctx['D']['cid'])
        panel_before = prepare_microbatches(nat, panel_ids, epoch=0, batch_index=0, normalizer=None,
                                            max_padded_tokens=8192)[2]
        s = overlay_snapshot(nat, self.view1, max_padded_tokens=8192)
        self.assertIs(s.overlay_base_snapshot, nat)
        panel_after = prepare_microbatches(s.overlay_base_snapshot, panel_ids, epoch=0, batch_index=0, normalizer=None,
                                           max_padded_tokens=8192)[2]
        self.assertEqual(panel_after, panel_before)      # fixed panels prepared from the frozen base are unchanged
        self.assertEqual(s.identity, nat.identity)
        self.assertIs(s.batch_manifest, nat.batch_manifest)
        self.assertEqual(s.contexts, nat.contexts)
        self.assertIs(nat.context_loader.__class__, N.NativeContextLoader)
        self.assertEqual(sorted(s.overlay_records), sorted([self.w.ctx['A']['cid'], self.w.ctx['B']['cid']]))
        first = s.overlay_records[self.w.ctx['A']['cid']][0]
        self.assertEqual(set(first), {'program_sha256', 'result', 'evaluation_key', 'packet_seq'})
        self.assertEqual(set(first['result']), set(C.REF_KEYS))
        # Any base loader with get(ids, epoch=, batch_index=): a MixedLoader over the same native base.
        m = VS.MixedSnapshot()
        m.native, m.identity, m.max_seq, m.program_window = nat, 'e' * 64, nat.max_seq, nat.program_window
        m.contexts, m.winners, m.variant_records, m.variant_vectors, m.best_observed = dict(nat.contexts), dict(nat.winners), {}, {}, {}
        m.batch_manifest = {'schema': 'jope.mixed-corpus-batches/v1', 'snapshot_sha256': m.identity}
        m.context_loader = VS.MixedLoader(m)
        sm = overlay_snapshot(m, self.view1, max_padded_tokens=8192)
        self.assertEqual((sm.identity, sm.batch_manifest), (m.identity, m.batch_manifest))
        ids = [self.w.ctx['A']['cid'], self.w.ctx['B']['cid']]
        batches, _, rec = prepare_microbatches(sm, ids, epoch=1, batch_index=0, normalizer=None, max_padded_tokens=8192)
        self.assertEqual(rec['selection']['base']['schema'], 'jope.mixed-program-window/v1')
        self.assertEqual(rec['selection']['base']['native_window']['schema'], 'jope.native-program-window/v1')
        with self.assertRaisesRegex(ValueError, 'frozen native base'):
            bad = copy.copy(self.view1)
            bad.pins = {**self.view1.pins, 'native_spec_sha256': '0' * 64}
            overlay_snapshot(m, bad, max_padded_tokens=8192)
        with self.assertRaisesRegex(ValueError, 'validated overlay chain view'):
            overlay_snapshot(m, p, max_padded_tokens=8192)

    # ------------------------------------------------------------------ chain, packet and root
    def chain3(self):
        """Delta chain: P1 = A×3 + B×2 (both first seen), P2 = B×2, P3 = A×2 + a re-emitted alias of P1's first row."""
        w = self.w
        A, Bc = w.ctx['A'], w.ctx['B']
        res = {'a': [w.result('A', f'(pipe X (ridge alpha={500 + j}.0))', [b * (0.40 + 0.03 * j) for b in A['recon']])
                     for j in range(5)],
               'b': [w.result('B', f'(pipe X (ridge alpha={600 + j}.0))', [b * (0.50 + 0.03 * j) for b in Bc['recon']])
                     for j in range(4)]}
        rows1 = [w.row('A', r) for r in res['a'][:3]] + [w.row('B', r) for r in res['b'][:2]]
        rows2 = [w.row('B', r) for r in res['b'][2:]]
        rows3 = [w.row('A', r) for r in res['a'][3:]] + [w.row('A', res['a'][0], raw=' (pipe X (ridge alpha=500.0)) ')]
        r1 = w.build(rows1, 'chain-1')
        r2 = w.build(rows2, 'chain-2', **after(r1))
        r3 = w.build(rows3, 'chain-3', **after(r1, r2))
        return (r1, r2, r3), (rows1, rows2, rows3), res

    def test_r2_three_packet_chain_first_anchor_carried(self):
        (r1, r2, r3), _, res = self.chain3()
        A, Bc, nat = self.w.ctx['A']['cid'], self.w.ctx['B']['cid'], self.w.nat
        p1, p2, p3 = (packet_of(r) for r in (r1, r2, r3))
        self.assertEqual([len(p['rows']) for p in (p1, p2, p3)], [5, 2, 2])
        self.assertEqual((sorted(p1['contexts']), p2['contexts'], p3['contexts']), (sorted([A, Bc]), {}, {}))
        self.assertEqual((p2['chain'], p3['chain']), ({'packets_before': 1, 'rows_before': 5},
                                                      {'packets_before': 2, 'rows_before': 7}))
        self.assertEqual(p3['label_unions'], {A: {'shard_rows': 12, 'earlier_rows': 3, 'packet_rows': 2,
                         'overlay_programs_sha256': PK.overlay_programs_digest([r['program_sha256'] for r in res['a']])}})
        self.assertEqual(p2['label_unions'][Bc]['earlier_rows'], 2)
        for p, cid in ((p3, A), (p2, Bc)):            # the FIRST anchor (packet 1) is carried bit-for-bit
            self.assertTrue(all(r['baseline']['anchored'] == p1['contexts'][cid]['anchored_baseline'] for r in p['rows']))
        self.assertEqual({r['baseline']['source'] for r in p2['rows']}, {'chain'})
        self.assertEqual([(e['class'], e['reason'], e['row']) for e in p3['census']['entries']],
                         [('TRAIN_VERIFIED', 'NEW_OVERLAY_ROW', 0), ('TRAIN_VERIFIED', 'NEW_OVERLAY_ROW', 1),
                          ('TRAIN_VERIFIED', 'ALREADY_IN_OVERLAY', [1, 0])])
        raws = [Path(r['packet']['path']).read_bytes() for r in (r1, r2, r3)]
        head = None
        for r, raw in zip((r1, r2, r3), raws):
            head = digest({'schema': C.HEAD_SCHEMA, 'predecessor': head, 'packet_sha256': C.sha(raw)})
            self.assertEqual((r['head_sha256'], head_sha256(raw, r['predecessor_overlay_head_sha256'])), (head, head))
        view = view_of(r1, r2, r3)
        self.assertEqual((view.seq, len(view.rows), view.head_sha256, sorted(view.contexts)), (3, 9, head, sorted([A, Bc])))
        self.assertEqual([r.packet_seq for r in view.context_rows(A)], [1, 1, 1, 3, 3])
        self.assertEqual([p['seq'] for p in view.packets], [1, 2, 3])
        self.assertTrue(all(isinstance(r, PK.ChainRow) for r in view.rows))
        self.assertFalse(any(isinstance(v, (bytes, bytearray)) or (isinstance(v, dict) and 'census' in v)
                             for v in vars(view).values()))              # no raw bytes or parsed packets retained
        self.assertEqual((r3['chain']['packets'], r3['chain']['rows'], r3['dry_run']['contexts']), (3, 9, 1))
        s = overlay_snapshot(nat, view, max_padded_tokens=8192)
        m = s.context_loader.get([A], epoch=0, batch_index=0).selection_receipt['members'][0]
        self.assertEqual((m['k'], len(m['selected']), {x['packet_seq'] for x in m['selected']}), (5, 5, {1, 3}))

    def test_r2_replay_chain_view_equals_one_pass_view(self):
        (r1, r2, r3), (rows1, rows2, rows3), _ = self.chain3()
        chain, again = view_of(r1, r2, r3), view_of(r1, r2, r3)
        self.assertEqual((chain.head_sha256, [record_json(r.record) for r in chain.rows]),
                         (again.head_sha256, [record_json(r.record) for r in again.rows]))
        single = view_of(self.w.build(rows1 + rows2 + rows3, 'one-pass'))

        def content(v):
            return {(r.context_id, r.program_sha256): (record_json(r.record), r.result_ref, r.evaluation_key,
                                                       r.anchored, r.rmse, r.wall_us) for r in v.rows}
        self.assertEqual(content(chain), content(single))
        self.assertEqual(chain.contexts, single.contexts)
        sc = overlay_snapshot(self.w.nat, chain, max_padded_tokens=8192)
        ss = overlay_snapshot(self.w.nat, single, max_padded_tokens=8192)
        strip = lambda m: [{k: v for k, v in x.items() if k != 'packet_seq'} for x in m['selected']]  # noqa: E731
        for cid in (self.w.ctx['A']['cid'], self.w.ctx['B']['cid']):
            for epoch in range(6):
                bc = sc.context_loader.get([cid], epoch=epoch, batch_index=0)
                bs = ss.context_loader.get([cid], epoch=epoch, batch_index=0)
                self.assertEqual(bc.records[cid], bs.records[cid])
                self.assertEqual(strip(bc.selection_receipt['members'][0]), strip(bs.selection_receipt['members'][0]))

    def test_r2_chain_tampered_middle_packet_refused(self):
        (r1, r2, r3), _, _ = self.chain3()
        t = self.root / 'tamper'
        t.mkdir()
        p2, raw2 = packet_of(r2), Path(r2['packet']['path']).read_bytes()
        resealed = forge(t / 'p2.json', p2,
                         lambda d: d['rows'][0]['inner_fold_rmse'].__setitem__(0, d['rows'][0]['inner_fold_rmse'][0] * 1.01))
        with self.assertRaisesRegex(ValueError, 'predecessor head link'):
            validate_chain([r1['packet'], resealed, r3['packet']], expected_head=r3['head_sha256'])
        flipped = {**write(t / 'flip.json', raw2[:-2] + b'0}'), 'sha256': r2['packet']['sha256']}
        with self.assertRaisesRegex(ValueError, 'exact pinned'):
            validate_chain([r1['packet'], flipped, r3['packet']], expected_head=r3['head_sha256'])
        with self.assertRaisesRegex(ValueError, 'expected head'):
            validate_chain([r1['packet'], resealed], expected_head=r2['head_sha256'])
        with self.assertRaisesRegex(ValueError, 'predecessor head link'):
            self.w.build([], 'after-tamper', predecessor_chain=[r1['packet'], resealed, r3['packet']],
                         predecessor_head=r3['head_sha256'])
        self.assertFalse((self.root / 'after-tamper').exists())

    def test_r2_chain_reordered_packets_refused(self):
        (r1, r2, r3), _, _ = self.chain3()
        for order in ((r1, r3, r2), (r2, r1, r3), (r1, r1, r2)):
            with self.subTest([r['seq'] for r in order]):
                with self.assertRaisesRegex(ValueError, 'chain sequence'):
                    validate_chain([r['packet'] for r in order], expected_head=r3['head_sha256'])
        for refs, head in (((r1, r2), r3['head_sha256']), ((r1, r2, r3), r2['head_sha256'])):
            with self.assertRaisesRegex(ValueError, 'expected head'):
                validate_chain([r['packet'] for r in refs], expected_head=head)

    def test_r2_chain_duplicate_across_packets_refused(self):
        (r1, r2, r3), _, _ = self.chain3()
        A = self.w.ctx['A']['cid']
        p1 = packet_of(r1)

        def mutate(d):
            """Append a copy of packet 1's first row to packet 3 and repair every packet-internal invariant."""
            dup = copy.deepcopy(p1['rows'][0])
            dup['packet_seq'] = 3
            d['rows'].append(dup)
            d['census']['entries'].append({'observation_sha256': dup['source_observations'][0], 'outbox_kind': 'v1',
                                           'emission_id': dup['source_emission_ids'][0], 'context_id': A,
                                           'phase': 'training', 'state': 'published_verified',
                                           'class': 'TRAIN_VERIFIED', 'reason': 'NEW_OVERLAY_ROW',
                                           'evaluation_key': dup['evaluation_key'], 'row': len(d['rows']) - 1,
                                           'detail': None})
            d['census']['source_rows'] += 1
            d['census']['classes']['TRAIN_VERIFIED'] += 1
            d['sources'][0]['rows'] += 1
            u = d['label_unions'][A]
            u['packet_rows'] += 1
            u['overlay_programs_sha256'] = PK.overlay_programs_digest(
                [r['program_sha256'] for r in p1['rows'] if r['context_id'] == A] +
                [r['program_sha256'] for r in d['rows']])
        ref = forge(self.root / 'dup.json', packet_of(r3), mutate)
        head = head_sha256(Path(ref['path']).read_bytes(), r2['head_sha256'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_chain([r1['packet'], r2['packet'], ref], expected_head=head)

    def test_r2_chain_anchor_drift_refused(self):
        (r1, r2, r3), _, _ = self.chain3()
        A = self.w.ctx['A']['cid']
        p1, p3 = packet_of(r1), packet_of(r3)
        anchor = p1['contexts'][A]['anchored_baseline']
        drifted = [math.nextafter(anchor[0], math.inf), anchor[1]]
        drift = forge(self.root / 'drift.json', p3, lambda d: d['rows'][0]['baseline'].update(anchored=drifted))

        def redeclare(d):
            d['contexts'][A] = {**p1['contexts'][A], 'first_seq': 3, 'anchored_baseline': drifted}
            for r in d['rows']:
                r['baseline']['anchored'] = drifted
            d['label_unions'][A].update(earlier_rows=0, overlay_programs_sha256=PK.overlay_programs_digest(
                [r['program_sha256'] for r in d['rows']]))
        again = forge(self.root / 'redeclare.json', p3, redeclare)
        for ref, why in ((drift, 'drifts from the first anchor'), (again, 'declared once')):
            head = head_sha256(Path(ref['path']).read_bytes(), r2['head_sha256'])
            with self.subTest(why):
                with self.assertRaisesRegex(ValueError, why):
                    validate_chain([r1['packet'], r2['packet'], ref], expected_head=head)

    def test_r2_chain_label_union_must_be_shard_plus_all_earlier_rows(self):
        (r1, r2, r3), _, _ = self.chain3()
        A = self.w.ctx['A']['cid']
        p3 = packet_of(r3)
        own = [r['program_sha256'] for r in p3['rows']]
        for name, change in (('earlier', {'earlier_rows': 2}), ('shard', {'shard_rows': 11}),
                             ('programs', {'overlay_programs_sha256': PK.overlay_programs_digest(own)})):
            ref = forge(self.root / (name + '.json'), p3, lambda d, c=change: d['label_unions'][A].update(c))
            head = head_sha256(Path(ref['path']).read_bytes(), r2['head_sha256'])
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, 'label union'):
                    validate_chain([r1['packet'], r2['packet'], ref], expected_head=head)

    def test_r2_shard_rmse_disagreement_refused(self):
        _, p, out = self.packet1()
        differ = self.entries(p, evaluation_key=out['dup_differ']['key'])
        self.assertEqual([(e['class'], e['reason']) for e in differ], [('REFUSED', 'SHARD_RMSE_DISAGREEMENT')])
        detail = json.loads(differ[0]['detail'])
        self.assertFalse(detail['agrees'])
        self.assertNotIn(out['dup_differ']['key'], {r['evaluation_key'] for r in p['rows']})
        shard = self.w.ctx['A']['shard']['doc']['records']
        near = self.w.result('A', '(pipe X (ridge alpha=5.0))', [v * (1 + 1e-8) for v in shard[4]['rmse_j']])
        r = self.w.build([self.w.row('A', near)], 'rmse-near')
        self.assertEqual([(e['class'], e['reason']) for e in packet_of(r)['census']['entries']],
                         [('REFUSED', 'SHARD_RMSE_DISAGREEMENT')])

    def test_r2_emission_body_b64_exact_bytes(self):
        w = self.w
        rows = [w.row('A', w.result('A', f'(pipe X (ridge alpha={170 + j}.0))')) for j in range(5)]

        def seal(row, raw):
            row['emission_body_b64'] = base64.b64encode(raw).decode('ascii')
            row['emission_body_sha256'] = row['observation']['body_sha256'] = C.sha(raw)
        stored = base64.b64decode(rows[1]['emission_body_b64'])
        seal(rows[1], json.dumps(json.loads(stored), indent=1).encode())          # same JSON, not the stored bytes
        rows[2]['emission_body_b64'] += '\n'                                         # not strict canonical base64
        rows[3]['emission_body_sha256'] = '0' * 64                                   # digest mismatch
        body = json.loads(base64.b64decode(rows[4]['emission_body_b64']))
        body['ordinal'] += 1                                                         # id no longer re-derives
        seal(rows[4], canonical(body))
        es = packet_of(w.build(rows, 'emission'))['census']['entries']
        self.assertEqual([(e['class'], e['reason']) for e in es], [('TRAIN_VERIFIED', 'NEW_OVERLAY_ROW')] +
                         [('REFUSED', 'EMISSION_IDENTITY')] * 4)
        for e, why in zip(es[1:], ('canonical JSON', 'base64', 'digest', 'emission id')):
            self.assertIn(why, e['detail'])
        legacy = w.row('A', w.result('A', '(pipe X (ridge alpha=176.0))'))
        legacy['emission_body'] = json.loads(base64.b64decode(legacy.pop('emission_body_b64')))
        with self.assertRaisesRegex(ValueError, 'closed uniform'):
            w.build([legacy], 'legacy-row')

    def test_r2_row_cap_defers_rows_to_the_next_packet(self):
        w = self.w
        rows = ([w.row('A', w.result('A', f'(pipe X (ridge alpha={180 + j}.0))')) for j in range(5)] +
                [w.row('B', w.result('B', f'(pipe X (ridge alpha={280 + j}.0))')) for j in range(2)])
        it = iter(rows)
        with mock.patch.object(B, 'MAX_ROWS', 3):
            r1 = w.build(it, 'cap-1')
            r2 = w.build(it, 'cap-2', **after(r1))
            r3 = w.build(it, 'cap-3', **after(r1, r2))
        ps = [packet_of(r) for r in (r1, r2, r3)]
        self.assertEqual([(len(p['rows']), p['census']['source_rows'], p['stopped']) for p in ps],
                         [(3, 3, 'NEW_ROW_CAP'), (3, 3, 'NEW_ROW_CAP'), (1, 1, None)])
        self.assertEqual(sorted(ps[1]['contexts']), [w.ctx['B']['cid']])
        view = view_of(r1, r2, r3)
        self.assertEqual(sorted(r.evaluation_key for r in view.rows), sorted(r['evaluation_key'] for r in rows))
        with mock.patch.object(PK, 'MAX_CHAIN_PACKETS', 2):
            with self.assertRaisesRegex(ValueError, 'bounded'):
                view_of(r1, r2, r3)
        with mock.patch.object(PK, 'MAX_CHAIN_ROWS', 6):
            with self.assertRaisesRegex(ValueError, 'bounded chain row count'):
                view_of(r1, r2, r3)

    def test_validate_packet_tamper_refusals(self):
        r1, p1, _ = self.packet1()
        raw = Path(r1['packet']['path']).read_bytes()
        t = self.root / 'tamper'
        t.mkdir()
        Bc = self.w.ctx['B']['cid']

        def reseal(name, mutate):
            doc = copy.deepcopy(p1)
            mutate(doc)
            return write(t / name, canonical(doc))
        cases = {
            'byte flip': ({**write(t / 'flip.json', raw[:-2] + b'0}'), 'sha256': r1['packet']['sha256']}, 'exact pinned'),
            'size': ({**r1['packet'], 'bytes': r1['packet']['bytes'] + 1}, 'pinned size'),
            'pretty': (write(t / 'pretty.json', json.dumps(p1, indent=1, sort_keys=True).encode()), 'non-canonical'),
            'duplicate key': (write(t / 'dup.json', b'{"seq":1,' + raw[1:]), 'duplicate JSON key'),
            'nan': (write(t / 'nan.json', raw.replace(b'"optimizer_updates":0', b'"optimizer_updates":NaN')), 'non-finite'),
            'ce teacher': (reseal('ce.json', lambda d: d['rows'][-1]['record'].update(ce_eligible=True)), 'CE teacher'),
            'new ce flag': (reseal('flag.json', lambda d: d.update(new_ce_teachers=1)), 'packet flags'),
            'census sum': (reseal('census.json', lambda d: d['census']['classes'].update(PENDING=1)), 'census'),
            'anchor': (reseal('anchor.json', lambda d: d['rows'][-1]['baseline'].update(anchored=[1.0, 1.0])), 'anchor'),
            'partition': (reseal('part.json', lambda d: d['rows'][-1].update(partition='validation')), 'train context'),
            'limits': (reseal('limits.json', lambda d: d['limits'].update(max_rows=4096)), 'limits'),
            'chain position': (reseal('chain.json', lambda d: d['chain'].update(rows_before=1)), 'chain position'),
            'label union': (reseal('union.json', lambda d: d['label_unions'][Bc].update(packet_rows=9)), 'label union'),
        }
        for name, (ref, why) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, why):
                    validate_packet(ref, predecessor_head=None)
        with self.assertRaisesRegex(ValueError, 'predecessor head'):
            validate_packet(r1['packet'], predecessor_head='f' * 64)
        link = t / 'link.json'
        link.symlink_to(Path(r1['packet']['path']))
        with self.assertRaisesRegex(ValueError, 'symlink'):
            validate_packet({**r1['packet'], 'path': str(link)}, predecessor_head=None)
        self.assertEqual(validate_packet(r1['packet'], predecessor_head=None), p1)
        got, ref, head = completed_packet(Path(r1['packet']['path']).parent, predecessor_head=None)
        self.assertEqual((got, ref, head), (p1, r1['packet'], r1['head_sha256']))

    def test_root_create_only_read_only_and_marker_last(self):
        receipt, p, _ = self.packet1()
        root = Path(receipt['packet']['path']).parent
        for name in ('packet.json', 'receipt.json', 'overlay-complete.json'):
            self.assertEqual(stat.S_IMODE((root / name).stat().st_mode), 0o444)
        marker = json.loads((root / 'overlay-complete.json').read_bytes())
        self.assertEqual((marker['packet'], marker['receipt'], marker['head_sha256']),
                         (receipt['packet'], receipt['receipt'], receipt['head_sha256']))
        with self.assertRaisesRegex(ValueError, 'create-only'):
            self.w.build([], 'packet-1')
        # A failure before the marker leaves no success marker (and the root is not accepted).
        real = B.create_only

        def failing(path, body):
            if Path(path).name == 'overlay-complete.json':
                raise OSError('injected failure before the success marker')
            return real(path, body)
        with mock.patch.object(B, 'create_only', side_effect=failing):
            with self.assertRaises(OSError):
                self.w.build([self.w.row('B', self.w.result('B', '(pipe X (ridge alpha=260.0))'))], 'no-marker')
        self.assertTrue((self.root / 'no-marker' / 'packet.json').exists())
        self.assertFalse((self.root / 'no-marker' / 'overlay-complete.json').exists())
        with self.assertRaises((ValueError, OSError)):
            completed_packet(self.root / 'no-marker', predecessor_head=None)
        # An expired build window refuses before any root exists.
        with self.assertRaisesRegex(ValueError, 'bounded build window'):
            self.w.build([], 'late', deadline=time.time() - 1)
        self.assertFalse((self.root / 'late').exists())

    def test_stale_authority_and_pin_drift_refused_before_any_root(self):
        cases = {
            'stale authority': ({'authority': lambda: self.w.authority(at=time.time() - 120)}, 'fresh native authority'),
            'source drift': ({'source_pins': {**self.w.source_pins, 'dltrain.labels': '0' * 64}}, 'source drift'),
            'normalizer': ({'preprocessing_bytes': self.w.pre + b' '}, 'normalizer|JSON'),
            'objective': ({'objective_sha256': '0' * 64}, 'objective'),
            'r6 pin': ({'r6_roster_sha256': '0' * 64}, 'R6'),
            'registry': ({'registry_bytes': self.w.registry + b'\n'}, 'parent registry'),
            'native spec': ({'native_spec_bytes': self.w.spec_bytes + b' '}, 'native spec'),
            'window': ({'deadline': time.time() + C.MAX_SECONDS + 60}, 'bounded build window'),
        }
        for name, (over, why) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, why):
                    self.w.build([], 'refused-' + name.replace(' ', '-'), **over)
                self.assertFalse((self.root / ('refused-' + name.replace(' ', '-'))).exists())
        self.assertEqual(digest(B.NB.OBJECTIVE), C.PRODUCTION_OBJECTIVE_SHA256)
        # Protected at build time: a train row on a protected context is refused, never a target.
        self.w.protected = ([self.w.ctx['A']['cid']], [])
        r = self.w.build([self.w.row('A', self.w.result('A', '(pipe X (ridge alpha=151.0))'))], 'protected-root')
        self.assertEqual([(e['class'], e['reason']) for e in packet_of(r)['census']['entries']],
                         [('REFUSED', 'PROTECTED_CONTEXT')])

    def test_verification_refusals_are_recorded_not_admitted(self):
        A = self.w.ctx['A']
        tol = self.w.result('A', '(pipe X (ridge alpha=160.0))', tolerance=1e-6)
        wrong_ref = self.w.result('A', '(pipe X (ridge alpha=161.0))')
        wrong_ref = {**wrong_ref, 'artifact_refs': {**wrong_ref['artifact_refs'],
                                                    'evaluation': tol['artifact_refs']['evaluation']}}
        rows = [self.w.row('A', tol), self.w.row('A', wrong_ref)]
        tampered = self.w.row('A', self.w.result('A', '(pipe X (ridge alpha=162.0))'))
        body = json.loads(base64.b64decode(tampered['emission_body_b64']))
        body['emission']['raw_program_utf8'] = '(pipe X (ridge alpha=163.0))'
        tampered['emission_body_b64'] = base64.b64encode(canonical(body)).decode('ascii')
        rows.append(tampered)
        r = self.w.build(rows, 'refusals')
        es = packet_of(r)['census']['entries']
        self.assertEqual([(e['class'], e['reason']) for e in es], [('REFUSED', 'VERIFICATION_FAILED')] * 2 +
                         [('REFUSED', 'EMISSION_IDENTITY')])
        self.assertIn('1e-9', es[0]['detail'])
        self.assertEqual(packet_of(r)['rows'], [])
        self.w.s3.wrong_version = True
        r2 = self.w.build([self.w.row('A', self.w.result('A', '(pipe X (ridge alpha=164.0))'))], 'wrong-version')
        self.assertEqual(packet_of(r2)['census']['entries'][0]['reason'], 'VERIFICATION_FAILED')
        self.assertIn('VersionId', packet_of(r2)['census']['entries'][0]['detail'])
        self.assertEqual(A['cid'], packet_of(r2)['census']['entries'][0]['context_id'])

    def test_c10_r4_boundary_composite_identity_and_exposure_join(self):
        """The C10 r4 boundary_admission_v2 API: composite identity and the first_exposures ref/term joins."""
        from pipelines.boundary_admission_v2 import boundary as BA
        from jope_flywheel import training_snapshot_kumos as K
        r1, p1, _ = self.packet1()
        nat, head = self.w.nat, r1['head_sha256']
        self.assertEqual(BA.composite_identity(nat.identity, None), nat.identity)
        self.assertEqual(BA.composite_identity(nat.identity, head),
                         digest({'schema': BA.COMPOSITE, 'variant_extension_sha256': nat.identity,
                                 'native_overlay_head_sha256': head}))
        s = overlay_snapshot(nat, self.view1, max_padded_tokens=8192)
        self.assertEqual(s.identity, nat.identity)      # the boundary, not the overlay, assigns the composite identity
        wanted = {(r['context_id'], r['program_sha256']): digest(K.normalized_ref(r['result_ref'])) for r in p1['rows']}
        ids = sorted({r['context_id'] for r in p1['rows']})
        _, _, rec = prepare_microbatches(s, ids, epoch=2, batch_index=0, normalizer=None, max_padded_tokens=8192)
        proof = {'selection': rec}                       # the shape neural_native_train writes into batch proofs
        joined = 0
        for m in proof['selection']['selection']['members']:
            selected = {x['program_sha256']: x for x in m['selected']}
            records = s.context_loader.get([m['context_id']], epoch=2, batch_index=0).records[m['context_id']]
            for ph in m['program_ids']:
                key = (m['context_id'], ph)
                if key not in wanted:
                    continue
                joined += 1
                self.assertEqual(overlay_ref_digest(s, *key), wanted[key])
                self.assertEqual(selected[ph]['result_ref_sha256'], wanted[key])
                terms = BA.term_support(records, m['program_ids'], ph)
                self.assertIn('q', terms)
                self.assertEqual('ce' in terms, selected[ph]['merged_winner'])
        self.assertEqual(joined, 8 + 4)
        self.assertIsNone(overlay_ref_digest(s, self.w.ctx['C']['cid'], '0' * 64))

    def test_real_r6_preregistration_parses_under_its_pin(self):
        raw = Path(os.environ['JOPE_W3_R6_PATH']).read_bytes()
        contexts, lineages = V.r6_roster(raw, C.R6_PREREGISTRATION_SHA256)
        self.assertEqual((len(contexts), len(lineages)), (53, 53))
        with self.assertRaisesRegex(ValueError, 'R6'):
            V.r6_roster(raw + b' ', C.R6_PREREGISTRATION_SHA256)


    # ------------------------------------------------------------------ terminal publication (CX-REV PR292 P2)
    # The builder never certifies itself.  Admission is the CALLER's bounded terminal acceptance envelope, written only
    # after build_packet RETURNED (``World.build`` is that caller; ``accept=False`` stops right after the return).
    # CX-REV's five publication cases (reviews of 4293d771 and 053c2a97) are ported below; cases 4 and 5 sweep EVERY
    # builder publication operation, so they do not depend on which record a builder revision happens to write last.
    def consumers(self, root):
        """Both actual consumers of a genesis root, as {consumer: (admitted, detail)}; never raises."""
        root = Path(root)
        if not (root / 'overlay-complete.json').exists():
            return {'completed_packet': (False, 'no marker'), 'validate_chain': (False, 'no marker')}
        marker = json.loads((root / 'overlay-complete.json').read_bytes())
        out = {}
        for name, call in (
                ('completed_packet', lambda: len(completed_packet(root, predecessor_head=None)[0]['rows'])),
                ('validate_chain', lambda: len(validate_chain([marker['packet']],
                                                              expected_head=marker['head_sha256']).rows))):
            try:
                out[name] = (True, call())
            except (ValueError, OSError) as error:
                out[name] = (False, str(error))
        return out

    def publication(self, name, *, cross_at=None, monotonic=False, refusal_fails=False, probe_at=(), accept=True,
                    program='(pipe X (ridge alpha=261.0))'):
        """One build through the fixture caller under CX-REV's substitution: the REAL ``create_only`` / ``fsync_dir``
        run; right after builder publication operation ``cross_at`` (0-based, call order) returns, only the fixture
        clock moves -- the wall clock to the ORIGINAL deadline, or the monotonic clock past its paired bound.
        ``refusal_fails`` injects one OSError when the builder creates its refusal record; ``probe_at`` calls both
        actual consumers right after those operations, i.e. before the build returned."""
        wall, shift, real_mono = [time.time()], [0.0], time.monotonic
        deadline = wall[0] + 600
        real_create, real_fsync = B.create_only, B.fsync_dir
        root, ops, probes = self.root / name, [], {}

        def after(kind, path):
            i = len(ops)
            ops.append((kind, Path(path).name if kind == 'create' else
                        'root' if Path(path) == root else 'parent' if Path(path) == root.parent else Path(path).name))
            if i == cross_at:
                if monotonic:
                    shift[0] = 2.0 * C.MAX_SECONDS
                else:
                    wall[0] = deadline
            if i in probe_at:
                probes[i] = self.consumers(root)

        def create(path, body):
            if refusal_fails and Path(path).name == 'overlay-refused.json':
                raise OSError('private fixture: refusal storage unavailable')
            ref = real_create(path, body)
            after('create', path)
            return ref

        def fsync(path):
            real_fsync(path)
            after('directory-fsync', path)
        row = self.w.row('B', self.w.result('B', program))
        out = {'build': 'returned', 'error': None, 'receipt': None}
        with mock.patch.object(B.time, 'time', side_effect=lambda: wall[0]), \
                mock.patch.object(B.time, 'monotonic', side_effect=lambda: real_mono() + shift[0]), \
                mock.patch.object(B, 'create_only', side_effect=create), \
                mock.patch.object(B, 'fsync_dir', side_effect=fsync):
            try:
                out['receipt'] = self.w.build([row], name, deadline=deadline, **({} if accept else {'accept': False}))
            except ValueError as error:
                out['build'], out['error'] = 'raised', str(error)
        out.update(deadline=deadline, wall=wall[0], ops=ops, probes=probes, root=root, final=self.consumers(root),
                   envelope=(root.parent / (name + '.acceptance.json')).exists(),
                   refusal=(root / 'overlay-refused.json').exists())
        return out

    def assert_inadmissible(self, consumers, why=None):
        self.assertEqual({k: v[0] for k, v in consumers.items()}, {'completed_packet': False, 'validate_chain': False},
                         consumers)
        if why is not None:
            for _, detail in consumers.values():
                self.assertRegex(detail, why)

    def test_publication_deadline_in_time_control_is_accepted_end_to_end(self):
        out = self.publication('control')                                       # CX-REV case 1
        self.assertEqual(out['build'], 'returned', out['error'])
        self.assertLess(out['wall'], out['deadline'])
        self.assertEqual(out['final'], {'completed_packet': (True, 1), 'validate_chain': (True, 1)})

    def test_publication_deadline_crossed_during_final_marker_write_is_refused_at_build_and_consume(self):
        out = self.publication('marker-write', cross_at=2)                      # CX-REV case 2: create marker
        self.assertEqual(out['ops'][2], ('create', 'overlay-complete.json'))
        self.assertEqual((out['build'], out['wall']), ('raised', out['deadline']))
        self.assertIn('deadline crossed after durable marker', out['error'])
        self.assert_inadmissible(out['final'])
        self.assertTrue(out['refusal'])
        self.assertFalse(out['envelope'])

    def test_publication_deadline_crossed_during_final_directory_fsync_is_refused_at_build_and_consume(self):
        out = self.publication('directory-fsync', cross_at=3)                   # CX-REV case 3: fsync the root
        self.assertEqual(out['ops'][3], ('directory-fsync', 'root'))
        self.assertEqual((out['build'], out['wall']), ('raised', out['deadline']))
        self.assertIn('deadline crossed after durable marker', out['error'])
        self.assert_inadmissible(out['final'])
        self.assertTrue(out['refusal'])
        self.assertFalse(out['envelope'])

    def test_publication_monotonic_bound_crossed_during_final_directory_fsync_is_refused(self):
        out = self.publication('monotonic-directory-fsync', cross_at=3, monotonic=True)
        self.assertEqual(out['build'], 'raised')
        self.assertLess(out['wall'], out['deadline'])                           # the wall clock alone says in time
        self.assertIn('deadline crossed after durable marker', out['error'])
        self.assert_inadmissible(out['final'])
        refusal = json.loads((out['root'] / 'overlay-refused.json').read_bytes())
        marker = json.loads((out['root'] / 'overlay-complete.json').read_bytes())
        self.assertEqual((refusal['verdict'], refusal['stage'], refusal['bound']),
                         ('refused', 'after_durable_marker', marker['bound']))
        self.assertGreaterEqual(refusal['sample']['monotonic'], refusal['bound']['monotonic_deadline'])
        self.assertEqual(stat.S_IMODE((out['root'] / 'overlay-refused.json').stat().st_mode), 0o444)
        self.assertFalse(out['envelope'])

    def test_publication_late_refusal_storage_failure_stays_inadmissible(self):
        """CX-REV case 4 (053c2a97): the clock reaches the original deadline right after a REAL publication operation
        and the refusal record cannot be written -- swept over every operation the builder performs."""
        self.maxDiff = None                                                     # name every admitting operation
        ops = self.publication('ops-4')['ops']
        self.assertEqual(ops[:3], [('create', 'packet.json'), ('create', 'receipt.json'),
                                   ('create', 'overlay-complete.json')])
        self.assertIn(('directory-fsync', 'parent'), ops)
        seen = {}
        for i, op in enumerate(ops):
            out = self.publication('refusal-fails-%d' % i, cross_at=i, refusal_fails=True)
            seen[i] = (op, out['build'], {k: v[0] for k, v in out['final'].items()}, out['envelope'], out['refusal'])
        refused = {'completed_packet': False, 'validate_chain': False}
        self.assertEqual(seen, {i: (op, 'raised', refused, False, False) for i, op in enumerate(ops)})
        # Caller side: the build RETURNED in time, the clock crossed before the caller's acceptance, and even the
        # caller's refused envelope cannot be written: no envelope, nothing admissible.
        r = self.w.build([self.w.row('B', self.w.result('B', '(pipe X (ridge alpha=262.0))'))], 'returned-4',
                         accept=False)
        root = Path(r['packet']['path']).parent
        late = json.loads(Path(r['marker']['path']).read_bytes())['bound']['wall_deadline_unix']
        with mock.patch.object(B.time, 'time', side_effect=lambda: late), \
                mock.patch.object(PK, 'create_only',
                                  side_effect=OSError('private fixture: envelope storage unavailable')):
            with self.assertRaisesRegex(OSError, 'envelope storage unavailable'):
                PK.accept_publication(root, physical={'kind': 'in_process_return',
                                                      'returned_sha256': r['marker']['sha256']})
        self.assertFalse(C.acceptance_path(root).exists())
        self.assert_inadmissible(self.consumers(root), 'no caller terminal acceptance')

    def test_consumers_cannot_admit_before_the_caller_acceptance(self):
        """CX-REV case 5 (053c2a97): both actual consumers called mid-build, right after each REAL publication
        operation (in time, and with the clock at the original deadline), refuse; so does a returned build before
        its caller acceptance.  Only the caller's in-time envelope admits."""
        self.maxDiff = None                                                     # name every admitting operation
        ops = self.publication('ops-5')['ops']
        self.assertIn(('create', 'overlay-complete.json'), ops)
        every = tuple(range(len(ops)))
        in_time = self.publication('probe-in-time', probe_at=every)
        seen = {('in-time', i): {k: v[0] for k, v in in_time['probes'][i].items()} for i in every}
        for i in every:
            late = self.publication('probe-late-%d' % i, cross_at=i, probe_at=(i,))
            self.assertEqual(late['build'], 'raised')
            self.assert_inadmissible(late['final'])
            seen[('late', i)] = {k: v[0] for k, v in late['probes'][i].items()}
        refused = {'completed_packet': False, 'validate_chain': False}
        self.assertEqual(seen, {key: refused for key in seen})
        self.assertEqual(in_time['final'], {'completed_packet': (True, 1), 'validate_chain': (True, 1)})
        # A RETURNED build is still inadmissible until its caller seals the acceptance.
        r = self.w.build([self.w.row('B', self.w.result('B', '(pipe X (ridge alpha=263.0))'))], 'returned-5',
                         accept=False)
        root = Path(r['packet']['path']).parent
        self.assert_inadmissible(self.consumers(root), 'no caller terminal acceptance')
        ref = PK.accept_publication(root, physical={'kind': 'in_process_return',
                                                    'returned_sha256': r['marker']['sha256']})
        self.assertEqual(ref['path'], str(root.parent / 'returned-5.acceptance.json'))
        self.assertEqual(self.consumers(root), {'completed_packet': (True, 1), 'validate_chain': (True, 1)})

    def test_caller_acceptance_is_bounded_by_the_original_deadline(self):
        w = self.w

        def returned(name, alpha):
            r = w.build([w.row('B', w.result('B', '(pipe X (ridge alpha=%d.0))' % alpha))], name, accept=False)
            root = Path(r['packet']['path']).parent
            return r, root, json.loads(Path(r['marker']['path']).read_bytes())['bound']

        def ret(r):
            return {'kind': 'in_process_return', 'returned_sha256': r['marker']['sha256']}
        # Returned in time, but the caller's post-return sample is at the deadline (wall) or past it (monotonic).
        for name, alpha, patch in (('late-wall', 264, lambda b: mock.patch.object(
                                        B.time, 'time', side_effect=lambda: b['wall_deadline_unix'])),
                                   ('late-monotonic', 265, lambda b: mock.patch.object(
                                        B.time, 'monotonic', side_effect=lambda: b['monotonic_deadline']))):
            with self.subTest(name):
                r, root, bound = returned(name, alpha)
                with patch(bound):
                    with self.assertRaisesRegex(ValueError, 'acceptance refused .kept as custody.*physical'):
                        PK.accept_publication(root, physical=ret(r))
                env = json.loads(C.acceptance_path(root).read_bytes())
                self.assertEqual((env['schema'], env['verdict'], env['bound']),
                                 (C.ACCEPTANCE_SCHEMA, 'refused', bound))
                self.assertFalse(C.in_bound(env['post_publication_sample'], bound))
                self.assertEqual(stat.S_IMODE(C.acceptance_path(root).stat().st_mode), 0o444)
                self.assert_inadmissible(self.consumers(root), 'accepted bounded terminal envelope')
                with self.assertRaisesRegex(ValueError, 'create-only acceptance'):   # never re-accepted in time
                    PK.accept_publication(root, physical=ret(r))
        # Physical evidence of a build unit (C10a r7 ``systemd_unit``): success strictly before the monotonic deadline.
        for name, alpha, change, ok in (('unit-in-time', 266, {}, True),
                                        ('unit-at-deadline', 267, {'exit_monotonic_usec': None}, False),
                                        ('unit-failed', 268, {'result': 'exit-code', 'exec_main_status': 1}, False),
                                        ('unit-other-boot', 269, {'boot_id': 'other-boot'}, False)):
            with self.subTest(name):
                r, root, bound = returned(name, alpha)
                unit = {'kind': 'systemd_unit', 'unit': 'overlay-build.service', 'result': 'success',
                        'exec_main_status': 0, 'exit_monotonic_usec': int((bound['monotonic_deadline'] - 1) * 1e6),
                        'boot_id': bound['boot_id'], **change}
                if unit['exit_monotonic_usec'] is None:
                    unit['exit_monotonic_usec'] = math.ceil(bound['monotonic_deadline'] * 1e6)
                if ok:
                    PK.accept_publication(root, physical=unit)
                    self.assertEqual(self.consumers(root), {'completed_packet': (True, 1), 'validate_chain': (True, 1)})
                else:
                    with self.assertRaisesRegex(ValueError, 'acceptance refused'):
                        PK.accept_publication(root, physical=unit)
                    self.assert_inadmissible(self.consumers(root), 'accepted bounded terminal envelope')
        # A misbehaving caller that accepts after the build RAISED late (its refusal record unwritable) is refused by
        # the post-return sample; one that accepts a root carrying the builder's refusal is refused by that record.
        for name, kw in (('raised-refusal-unwritten', {'refusal_fails': True}), ('raised-refusal-written', {})):
            with self.subTest(name):
                out = self.publication(name, cross_at=3, **kw)
                self.assertEqual(out['build'], 'raised')
                marker = json.loads((out['root'] / 'overlay-complete.json').read_bytes())
                with mock.patch.object(B.time, 'time', side_effect=lambda: out['deadline']):
                    with self.assertRaisesRegex(ValueError, 'acceptance refused'):
                        PK.accept_publication(out['root'], physical={
                            'kind': 'in_process_return',
                            'returned_sha256': C.sha((out['root'] / 'overlay-complete.json').read_bytes())})
                self.assertEqual(json.loads(C.acceptance_path(out['root']).read_bytes())['verdict'], 'refused')
                self.assert_inadmissible(self.consumers(out['root']))
                self.assertEqual(marker['bound']['wall_deadline_unix'], out['deadline'])

    def test_consumers_require_the_exact_in_time_terminal_acceptance(self):
        r1 = self.w.build([self.w.row('A', self.w.result('A', '(pipe X (ridge alpha=262.0))'))], 'accepted-1')
        r2 = self.w.build([self.w.row('A', self.w.result('A', '(pipe X (ridge alpha=263.0))'))], 'accepted-2',
                          **after(r1))
        root = Path(r1['packet']['path']).parent
        env_path = C.acceptance_path(root)
        self.assertEqual(env_path, root.parent / 'accepted-1.acceptance.json')
        self.assertFalse((root / 'overlay-accepted.json').exists())             # the builder never certifies itself
        marker_raw, env_raw = (root / 'overlay-complete.json').read_bytes(), env_path.read_bytes()
        marker, env = json.loads(marker_raw), json.loads(env_raw)
        self.assertEqual(r1['acceptance'], {'path': str(env_path), 'sha256': C.sha(env_raw), 'bytes': len(env_raw)})
        self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o444)
        self.assertEqual((marker['schema'], set(marker['bound'])),
                         (C.MARKER_SCHEMA, {'wall_deadline_unix', 'monotonic_deadline', 'boot_id'}))
        self.assertEqual((env['schema'], env['subject'], env['verdict'], env['root'], env['bound'], env['physical']),
                         (C.ACCEPTANCE_SCHEMA, 'native_overlay_root', 'accepted', str(root), marker['bound'],
                          {'kind': 'in_process_return', 'returned_sha256': C.sha(marker_raw)}))
        self.assertEqual(env['marker'], {'path': str(root / 'overlay-complete.json'), 'sha256': C.sha(marker_raw),
                                         'bytes': len(marker_raw)})
        self.assertEqual(env['artifacts'], {'packet.json': r1['packet'], 'receipt.json': r1['receipt']})
        self.assertTrue(C.in_bound(env['post_publication_sample'], env['bound']))
        self.assertEqual(len(view_of(r1, r2).rows), 2)

        def put(path, body):
            if os.path.lexists(path):
                os.unlink(path)
            Path(path).write_bytes(body)
        bound = marker['bound']
        post = env['post_publication_sample']
        extended = {**bound, 'wall_deadline_unix': bound['wall_deadline_unix'] + 3600,
                    'monotonic_deadline': bound['monotonic_deadline'] + 3600}
        unit = {'kind': 'systemd_unit', 'unit': 'overlay-build.service', 'result': 'success', 'exec_main_status': 0,
                'exit_monotonic_usec': math.ceil(bound['monotonic_deadline'] * 1e6), 'boot_id': bound['boot_id']}
        env2_raw = C.acceptance_path(Path(r2['packet']['path']).parent).read_bytes()
        body = lambda **change: canonical({**env, **change})                    # noqa: E731
        forged = {
            'missing envelope': (None, 'no caller terminal acceptance'),
            'refused verdict': (body(verdict='refused'), 'accepted bounded terminal envelope'),
            'other subject': (body(subject='prepared_root'), 'accepted bounded terminal envelope'),
            'other root envelope': (env2_raw, 'accepted bounded terminal envelope'),
            'late wall sample': (body(post_publication_sample={**post, 'wall_unix': bound['wall_deadline_unix']}),
                                 'physical terminal evidence'),
            'late monotonic sample': (body(post_publication_sample={**post, 'monotonic': bound['monotonic_deadline']}),
                                      'physical terminal evidence'),
            'other boot': (body(post_publication_sample={**post, 'boot_id': 'x'}), 'physical terminal evidence'),
            'other returned marker': (body(physical={'kind': 'in_process_return', 'returned_sha256': '0' * 64}),
                                      'physical terminal evidence'),
            'unit exit at the deadline': (body(physical=unit), 'physical terminal evidence'),
            'open physical evidence': (body(physical={**env['physical'], 'note': 1}), 'closed physical'),
            'extended bound': (body(bound=extended, post_publication_sample={**post, 'wall_unix':
                                                                             bound['wall_deadline_unix']}),
                               'unchanged original bound'),
            'other marker bytes': (body(marker={**env['marker'], 'sha256': '0' * 64}), 'exact success marker bytes'),
            'other artifacts': (body(artifacts={**env['artifacts'], 'receipt.json': r2['receipt']}),
                                'sealed artifacts'),
            'non-canonical': (json.dumps(env, indent=1).encode(), 'non-canonical'),
        }
        for case, (raw, why) in forged.items():
            with self.subTest(case):
                if raw is None:
                    os.unlink(env_path)
                else:
                    put(env_path, raw)
                with self.assertRaisesRegex(ValueError, why):
                    completed_packet(root, predecessor_head=None)
                with self.assertRaisesRegex(ValueError, why):
                    view_of(r1, r2)
                name = 'after-' + case.replace(' ', '-')
                with self.assertRaisesRegex(ValueError, why):
                    self.w.build([], name, **after(r1))
                self.assertFalse((self.root / name).exists())
                put(env_path, env_raw)
        with self.subTest('symlinked envelope'):
            os.unlink(env_path)
            (self.root / 'elsewhere.json').write_bytes(env_raw)
            env_path.symlink_to(self.root / 'elsewhere.json')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                view_of(r1, r2)
            os.unlink(env_path)
            put(env_path, env_raw)
        self.assertEqual(len(view_of(r1, r2).rows), 2)
        # A builder self-certification (the r2.6 overlay-accepted.json) never stands in for the caller's envelope.
        os.unlink(env_path)
        (root / 'overlay-accepted.json').write_bytes(canonical({'verdict': 'accepted', 'fixture': 'self-certified'}))
        with self.assertRaisesRegex(ValueError, 'no caller terminal acceptance'):
            view_of(r1, r2)
        os.unlink(root / 'overlay-accepted.json')
        put(env_path, env_raw)
        # A builder refusal record beside a valid envelope refuses every consumer.
        (root / 'overlay-refused.json').write_bytes(canonical({'fixture': 'late builder refusal'}))
        for consume in (lambda: completed_packet(root, predecessor_head=None), lambda: view_of(r1, r2),
                        lambda: self.w.build([], 'after-refusal', **after(r1))):
            with self.assertRaisesRegex(ValueError, 'late terminal record kept for custody'):
                consume()
        self.assertFalse((self.root / 'after-refusal').exists())
        os.unlink(root / 'overlay-refused.json')
        # A pre-terminal v1 marker (no sealed bound) is refused even with an envelope re-pinned to its bytes.
        v1 = canonical({**{k: v for k, v in marker.items() if k != 'bound'},
                        'schema': 'jope.native-result-overlay-complete/v1'})
        put(root / 'overlay-complete.json', v1)
        put(env_path, body(marker={**env['marker'], 'sha256': C.sha(v1), 'bytes': len(v1)},
                           physical={'kind': 'in_process_return', 'returned_sha256': C.sha(v1)}))
        for consume in (lambda: completed_packet(root, predecessor_head=None), lambda: view_of(r1, r2)):
            with self.assertRaisesRegex(ValueError, 'overlay success marker'):
                consume()
        put(root / 'overlay-complete.json', marker_raw)
        put(env_path, env_raw)
        self.assertEqual(completed_packet(root, predecessor_head=None)[2], r1['head_sha256'])

if __name__ == '__main__':
    unittest.main()
