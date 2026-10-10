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
    def publication_boundary(self, mode):
        """CX-REV's publication-boundary reproducer (review of 4293d771), ported: the REAL ``create_only`` /
        ``fsync_dir`` run, then only the fixture clock moves -- the wall clock to the ORIGINAL deadline, or (``monotonic-``
        modes) the monotonic clock past its paired bound -- before the selected final publication operation returns.
        One synthetic verified B row; every consumer verdict is collected before anything is asserted."""
        wall, mono_shift, real_mono = [time.time()], [0.0], time.monotonic
        deadline = wall[0] + 600
        real_create, real_fsync = B.create_only, B.fsync_dir
        events = []

        def cross():
            if mode.startswith('monotonic-'):
                mono_shift[0] = 2.0 * C.MAX_SECONDS
            else:
                wall[0] = deadline

        def create(path, body):
            ref = real_create(path, body)
            events.append(('create', Path(path).name, wall[0]))
            if (mode, Path(path).name) in (('marker-write', 'overlay-complete.json'),
                                           ('acceptance-write', 'overlay-accepted.json')):
                cross()
            return ref

        def fsync(path):
            real_fsync(path)
            events.append(('directory-fsync', Path(path).name, wall[0]))
            if mode in ('directory-fsync', 'monotonic-directory-fsync'):
                cross()
        name = 'publication-' + mode
        row = self.w.row('B', self.w.result('B', '(pipe X (ridge alpha=261.0))'))
        out = {'build': 'accepted', 'error': None}
        with mock.patch.object(B.time, 'time', side_effect=lambda: wall[0]), \
                mock.patch.object(B.time, 'monotonic', side_effect=lambda: real_mono() + mono_shift[0]), \
                mock.patch.object(B, 'create_only', side_effect=create), \
                mock.patch.object(B, 'fsync_dir', side_effect=fsync):
            try:
                self.w.build([row], name, deadline=deadline)
            except ValueError as error:
                out['build'], out['error'] = 'refused', str(error)
        out.update(returned_wall=wall[0], deadline=deadline, events=events, root=self.root / name)
        marker = json.loads((self.root / name / 'overlay-complete.json').read_bytes())
        out['marker'] = marker
        for consumer, call in (
                ('completed_packet', lambda: completed_packet(self.root / name, predecessor_head=None)),
                ('validate_chain', lambda: len(validate_chain([marker['packet']],
                                                              expected_head=marker['head_sha256']).rows))):
            try:
                out[consumer + '_result'] = call()
                out[consumer] = 'accepted'
            except ValueError as error:
                out[consumer], out[consumer + '_error'] = 'refused', str(error)
        return out

    def assert_late_publication_refused(self, out, stage):
        self.assertEqual({k: out[k] for k in ('build', 'completed_packet', 'validate_chain')},
                         {'build': 'refused', 'completed_packet': 'refused', 'validate_chain': 'refused'}, out['events'])
        self.assertIn('deadline crossed ' + stage, out['error'])
        for consumer in ('completed_packet', 'validate_chain'):
            self.assertIn('late terminal record kept for custody', out[consumer + '_error'])
        root = out['root']
        for name in ('packet.json', 'receipt.json', 'overlay-complete.json', 'overlay-refused.json'):
            self.assertTrue((root / name).is_file(), name)                       # the late root is kept for custody
            self.assertEqual(stat.S_IMODE((root / name).stat().st_mode), 0o444)
        refusal = json.loads((root / 'overlay-refused.json').read_bytes())
        marker_raw = (root / 'overlay-complete.json').read_bytes()
        self.assertEqual((refusal['verdict'], refusal['stage']), ('refused', stage.replace(' ', '_')))
        self.assertEqual(refusal['marker'], {'path': str(root / 'overlay-complete.json'), 'sha256': C.sha(marker_raw),
                                             'bytes': len(marker_raw)})
        self.assertEqual(refusal['bound'], out['marker']['bound'])
        self.assertEqual(refusal['bound']['wall_deadline_unix'], out['deadline'])        # the UNCHANGED original bound
        self.assertFalse(C.in_bound(refusal['sample'], refusal['bound']))
        return refusal

    def test_publication_deadline_in_time_control_is_accepted_end_to_end(self):
        out = self.publication_boundary('control')
        self.assertEqual({k: out[k] for k in ('build', 'completed_packet', 'validate_chain')},
                         {'build': 'accepted', 'completed_packet': 'accepted', 'validate_chain': 'accepted'})
        self.assertLess(out['returned_wall'], out['deadline'])
        self.assertEqual(out['validate_chain_result'], 1)
        self.assertEqual(len(out['completed_packet_result'][0]['rows']), 1)

    def test_publication_deadline_crossed_during_final_marker_write_is_refused_at_build_and_consume(self):
        out = self.publication_boundary('marker-write')
        self.assertEqual(out['returned_wall'], out['deadline'])
        self.assert_late_publication_refused(out, 'after durable marker')
        self.assertFalse((out['root'] / 'overlay-accepted.json').exists())

    def test_publication_deadline_crossed_during_final_directory_fsync_is_refused_at_build_and_consume(self):
        out = self.publication_boundary('directory-fsync')
        self.assertEqual(out['returned_wall'], out['deadline'])
        self.assert_late_publication_refused(out, 'after durable marker')
        self.assertFalse((out['root'] / 'overlay-accepted.json').exists())

    def test_publication_monotonic_bound_crossed_during_final_directory_fsync_is_refused(self):
        out = self.publication_boundary('monotonic-directory-fsync')
        self.assertLess(out['returned_wall'], out['deadline'])                   # the wall clock alone says in time
        refusal = self.assert_late_publication_refused(out, 'after durable marker')
        self.assertGreaterEqual(refusal['sample']['monotonic'], refusal['bound']['monotonic_deadline'])
        self.assertFalse((out['root'] / 'overlay-accepted.json').exists())

    def test_publication_deadline_crossed_during_acceptance_write_is_refused_at_build_and_consume(self):
        out = self.publication_boundary('acceptance-write')
        refusal = self.assert_late_publication_refused(out, 'after durable acceptance')
        accepted_raw = (out['root'] / 'overlay-accepted.json').read_bytes()            # written in time, then late
        self.assertEqual(refusal['acceptance'], {'path': str(out['root'] / 'overlay-accepted.json'),
                                                 'sha256': C.sha(accepted_raw), 'bytes': len(accepted_raw)})

    def test_consumers_require_the_exact_in_time_terminal_acceptance(self):
        r1 = self.w.build([self.w.row('A', self.w.result('A', '(pipe X (ridge alpha=262.0))'))], 'accepted-1')
        r2 = self.w.build([self.w.row('A', self.w.result('A', '(pipe X (ridge alpha=263.0))'))], 'accepted-2', **after(r1))
        root = Path(r1['packet']['path']).parent
        marker_raw = (root / 'overlay-complete.json').read_bytes()
        accepted_raw = (root / 'overlay-accepted.json').read_bytes()
        marker, accepted = json.loads(marker_raw), json.loads(accepted_raw)
        self.assertEqual(r1['acceptance'], {'path': str(root / 'overlay-accepted.json'), 'sha256': C.sha(accepted_raw),
                                            'bytes': len(accepted_raw)})
        self.assertEqual(stat.S_IMODE((root / 'overlay-accepted.json').stat().st_mode), 0o444)
        self.assertFalse((root / 'overlay-refused.json').exists())
        self.assertEqual((marker['schema'], set(marker['bound'])),
                         (C.MARKER_SCHEMA, {'wall_deadline_unix', 'monotonic_deadline', 'boot_id'}))
        self.assertEqual((accepted['verdict'], accepted['stage'], accepted['bound'], accepted['packet'],
                          accepted['head_sha256'], accepted['seq']),
                         ('accepted', 'after_durable_marker', marker['bound'], r1['packet'], r1['head_sha256'], 1))
        self.assertEqual(accepted['marker'], {'path': str(root / 'overlay-complete.json'), 'sha256': C.sha(marker_raw),
                                              'bytes': len(marker_raw)})
        self.assertTrue(C.in_bound(accepted['sample'], accepted['bound']))
        self.assertEqual(len(view_of(r1, r2).rows), 2)

        def put(name, body):
            path = root / name
            if path.exists():
                os.unlink(path)
            path.write_bytes(body)
        bound = marker['bound']
        late = {**accepted['sample'], 'wall_unix': bound['wall_deadline_unix']}
        mono_late = {**accepted['sample'], 'monotonic': bound['monotonic_deadline']}
        extended = {**bound, 'wall_deadline_unix': bound['wall_deadline_unix'] + 3600,
                    'monotonic_deadline': bound['monotonic_deadline'] + 3600}
        r2_accepted = json.loads((Path(r2['packet']['path']).parent / 'overlay-accepted.json').read_bytes())
        forged = {
            'missing acceptance': (None, 'no success marker and in-time terminal acceptance'),
            'late wall sample': (canonical({**accepted, 'sample': late}), 'qualified inside its original bound'),
            'late monotonic sample': (canonical({**accepted, 'sample': mono_late}), 'qualified inside its original bound'),
            'other boot': (canonical({**accepted, 'sample': {**accepted['sample'], 'boot_id': 'x'}}),
                           'qualified inside its original bound'),
            'extended bound': (canonical({**accepted, 'bound': extended, 'sample': late}), 'unchanged original bound'),
            'other marker bytes': (canonical({**accepted, 'marker': {**accepted['marker'], 'sha256': '0' * 64}}),
                                   'exact success marker bytes'),
            'refusal renamed': (canonical({**accepted, 'verdict': 'refused'}), 'accepted overlay terminal record'),
            'other root acceptance': (canonical(r2_accepted), 'acceptance binds this packet'),
            'non-canonical': (json.dumps(accepted, indent=1).encode(), 'non-canonical'),
        }
        for case, (body, why) in forged.items():
            with self.subTest(case):
                if body is None:
                    os.unlink(root / 'overlay-accepted.json')
                else:
                    put('overlay-accepted.json', body)
                with self.assertRaisesRegex(ValueError, why):
                    completed_packet(root, predecessor_head=None)
                with self.assertRaisesRegex(ValueError, why):
                    view_of(r1, r2)
                with self.assertRaisesRegex(ValueError, why):
                    self.w.build([], 'after-' + case.replace(' ', '-'), **after(r1))
                self.assertFalse((self.root / ('after-' + case.replace(' ', '-'))).exists())
                put('overlay-accepted.json', accepted_raw)
        self.assertEqual(len(view_of(r1, r2).rows), 2)
        # A refusal beside a valid acceptance (the acceptance publication itself was late) refuses every consumer.
        (root / 'overlay-refused.json').write_bytes(canonical({'fixture': 'late acceptance publication'}))
        for consume in (lambda: completed_packet(root, predecessor_head=None), lambda: view_of(r1, r2),
                        lambda: self.w.build([], 'after-refusal', **after(r1))):
            with self.assertRaisesRegex(ValueError, 'late terminal record kept for custody'):
                consume()
        self.assertFalse((self.root / 'after-refusal').exists())
        os.unlink(root / 'overlay-refused.json')
        # A pre-terminal v1 marker (no sealed bound) is refused even with an acceptance re-pinned to its bytes.
        v1 = canonical({**{k: v for k, v in marker.items() if k != 'bound'},
                        'schema': 'jope.native-result-overlay-complete/v1'})
        put('overlay-complete.json', v1)
        put('overlay-accepted.json', canonical({**accepted, 'marker': {**accepted['marker'], 'sha256': C.sha(v1),
                                                                       'bytes': len(v1)}}))
        for consume in (lambda: completed_packet(root, predecessor_head=None), lambda: view_of(r1, r2)):
            with self.assertRaisesRegex(ValueError, 'overlay success marker'):
                consume()
        put('overlay-complete.json', marker_raw)
        put('overlay-accepted.json', accepted_raw)
        self.assertEqual(completed_packet(root, predecessor_head=None)[2], r1['head_sha256'])

if __name__ == '__main__':
    unittest.main()
