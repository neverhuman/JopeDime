"""Revision 2: DELTA ``jope.native-result-overlay-packet/v2`` packets, strict validation and the head chain.

Invariants:
- A packet is canonical JSON of at most 32 MiB holding at most 2,048 NEW rows (never earlier rows), the anchors of
  contexts first seen in it, and per-context ``label_unions`` that record exactly which rows its labels were computed
  on (shard rows + every EARLIER chain row of the context + its own rows, with a digest of the overlay programs).
- ``head_sha256(k) = digest({schema: jope.native-overlay-head/v1, predecessor: H(k-1) or None, packet_sha256})``.
- ``validate_chain(refs_in_order, expected_head=)`` strictly validates every packet in order, recomputes the head chain
  to ``expected_head`` and refuses, across the WHOLE chain: a broken predecessor link or chain position, a changed pin,
  a duplicate (context, program), evaluation key or result ref, an anchor redeclared or drifting from the context's
  first anchor (bit-for-bit), a label union that is not shard + all earlier chain rows + own rows, and more than
  4,096 packets or 500,000 rows. Raw packet bytes and parsed packets are dropped after each packet is accumulated;
  the cumulative view keeps one compact ``ChainRow`` per overlay row.
- Census: five closed classes with closed reasons summing exactly to the consumed rows of that packet's build.
- Fixed flags: ``new_ce_teachers == 0``, ``optimizer_updates == 0``, ``admission_committed is False``; every overlay
  record is CE-false with ``source_kind == native_overlay_verified_s1`` on a train partition.
- Admission requires the CALLER's bounded terminal acceptance envelope (the C10a r7 design; the builder never
  certifies itself): ``validate_chain`` and ``completed_packet`` accept a packet only from its own build root
  (``<root>/packet.json``) whose sibling ``<root>.acceptance.json`` -- written by ``accept_publication`` only after
  ``build_packet`` RETURNED -- is ``accepted`` and pins the exact success-marker bytes, the sealed packet and
  receipt, the UNCHANGED original bound sealed in the marker, and physical in-time evidence (an in-process return of
  that marker with a post-return sample strictly inside the bound, or the build unit's successful exit before the
  monotonic deadline); a builder refusal record also refuses. A build that raised, a build still running and a
  returned build not yet accepted are never admissible. Admission also requires the caller's exact completion
  marker, file-fsynced privately only after the envelope parent fsync succeeds, then atomically made visible.
  An envelope or pending completion left by failure stays custody only. ``validate_packet`` is structural only.
"""
import math
import os
from array import array
from collections import Counter

from jope_flywheel.training_snapshot_native import NativeRecord

from .common import (ACCEPTANCE_KEYS, ACCEPTANCE_SCHEMA, ACCEPTANCE_SUBJECT, CLASSES, HEAD_SCHEMA, MARKER_KEYS,
                     MARKER_NAME, MARKER_SCHEMA, MAX_CHAIN_PACKETS, MAX_CHAIN_ROWS, MAX_CONTEXTS, MAX_PACKET_BYTES,
                     MAX_ROWS, MAX_RSS_BYTES, MAX_SECONDS, MAX_SOURCE_ROWS, MAX_TERMINAL_BYTES, OVERLAY_WINDOW,
                     PACKET_SCHEMA, REASONS, REF_KEYS, REFUSED_NAME, ROTATION_DOMAIN, SOURCE_KIND, STOPS,
                     acceptance_path, artifact_refs, canonical, check_bound, clock_sample, create_only, digest,
                     exact_canonical, fsync_dir, hash64, in_bound, need, physical_ok, read_exact, read_small, ref5,
                     safe_path, sha)
from .labels_overlay import RECORD_FIELDS, record_from_json

PACKET_KEYS = frozenset({'schema', 'seq', 'predecessor_overlay_head_sha256', 'chain', 'pins', 'campaigns', 'sources',
                         'authority_proofs', 'rows', 'contexts', 'label_unions', 'census', 'limits', 'stopped',
                         'new_ce_teachers', 'optimizer_updates', 'admission_committed'})
PIN_KEYS = frozenset({'native_spec_sha256', 'preprocessing_sha256', 'parent_registry_sha256', 'objective_sha256',
                      'grammar_sha256', 'vocab_sha256', 'authority_pins_sha256', 'input_lists', 'source_sha256',
                      'builder_source_sha256', 'worker_sha256', 'r6_roster_sha256', 'overlay_window',
                      'rotation_domain'})
ROW_FIELDS = frozenset({'context_id', 'parent_lineage_sha256', 'partition', 'evaluation_key', 'program',
                        'program_sha256', 'result_ref', 'artifact_refs', 'original_score_context', 'inner_fold_rmse',
                        'wall_us', 'baseline', 'record', 'shard_ref', 'source_emission_ids', 'source_observations',
                        'packet_seq'})
CONTEXT_FIELDS = frozenset({'anchored_baseline', 'baseline_source', 'y_train_sha256', 'split_sha256', 'label_context',
                            'shard_ref', 'shard_rows', 'first_seq', 'parent_lineage_sha256'})
UNION_FIELDS = frozenset({'shard_rows', 'earlier_rows', 'packet_rows', 'overlay_programs_sha256'})
ENTRY_FIELDS = frozenset({'observation_sha256', 'outbox_kind', 'emission_id', 'context_id', 'phase', 'state',
                          'class', 'reason', 'evaluation_key', 'row', 'detail'})
LIMITS = {'max_rows': MAX_ROWS, 'max_contexts': MAX_CONTEXTS, 'max_packet_bytes': MAX_PACKET_BYTES,
          'max_seconds': MAX_SECONDS, 'max_rss_bytes': MAX_RSS_BYTES, 'max_source_rows': MAX_SOURCE_ROWS,
          'max_chain_packets': MAX_CHAIN_PACKETS, 'max_chain_rows': MAX_CHAIN_ROWS}
SCORE_CONTEXT_KEYS = frozenset({'identity', 'dataset_hash', 'cut_id', 'split_sha256', 'input_sha256', 'target_sha256',
                                'evaluator_sha256'})


def head_from_sha(packet_sha256, predecessor_head):
    need(hash64(packet_sha256) and (predecessor_head is None or hash64(predecessor_head)), 'head inputs')
    return digest({'schema': HEAD_SCHEMA, 'predecessor': predecessor_head, 'packet_sha256': packet_sha256})


def head_sha256(packet_raw, predecessor_head):
    """H(k) = digest({'schema': jope.native-overlay-head/v1, 'predecessor': H(k-1) or None, 'packet_sha256'})."""
    need(type(packet_raw) is bytes, 'head inputs')
    return head_from_sha(sha(packet_raw), predecessor_head)


def overlay_programs_digest(programs):
    """Digest of the sorted overlay program SHAs a label union used (earlier chain rows + own rows)."""
    return digest(sorted(programs))


def _pair(x):
    return type(x) is list and len(x) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in x)


def _record(doc, row):
    need(type(doc) is dict and set(doc) == set(RECORD_FIELDS), 'closed NativeRecord fields')
    t = doc['tokens']
    need(type(t) is list and 2 <= len(t) and all(type(v) is int and v >= 0 for v in t) and t[0] == 1 and t[-1] == 2,
         'complete overlay token sequence')
    need(doc['program_sha256'] == row['program_sha256'] and doc['outcome'] == 'completed', 'record program identity')
    # CE guard: an overlay record is never a CE teacher (L6: CE only on frozen native HWM winners).
    need(doc['ce_eligible'] is False and doc['source_kind'] == SOURCE_KIND, 'overlay never creates a CE teacher')
    for k in ('quality_eligible', 'pairwise_eligible', 'runtime_eligible', 'runtime_censored'):
        need(type(doc[k]) is bool, 'typed support mask')
    q = doc['quality_eligible']
    need(doc['pairwise_eligible'] == q and (not q or (type(doc['q']) is float and type(doc['q_opt']) is float and
         math.isfinite(doc['q']) and math.isfinite(doc['q_opt']) and type(doc['q_grid']) is int)) and
         (q or (doc['q'] is None and doc['q_opt'] is None and doc['q_grid'] is None)), 'measured quality or none')
    need(doc['feasibility'] == 0, 'verified completed feasibility only (R5 failures never enter)')
    need(doc['runtime_eligible'] == (doc['runtime_seconds'] is not None) and
         (doc['runtime_seconds'] is None or (type(doc['runtime_seconds']) in (int, float) and
                                             math.isfinite(doc['runtime_seconds']) and doc['runtime_seconds'] >= 0)),
         'measured runtime or none')


def _anchor(cid, c, seq):
    need(hash64(cid) and type(c) is dict and set(c) == CONTEXT_FIELDS and _pair(c['anchored_baseline']) and
         c['baseline_source'] in ('shard', 'reconstructed') and hash64(c['y_train_sha256']) and
         hash64(c['split_sha256']) and hash64(c['parent_lineage_sha256']) and c['first_seq'] == seq and
         type(c['shard_rows']) is int and c['shard_rows'] >= 0, 'closed first-seen context anchor')
    lc = c['label_context']
    need(type(lc) is dict and set(lc) == {'context_sha256', 'dataset_hash', 'family_id', 'split_role'} and
         lc['context_sha256'] == cid and all(type(lc[k]) is str and lc[k] for k in lc) and
         (c['shard_ref'] is not None or lc['split_role'] == 'train_fit'), 'label context identity')
    need((c['shard_ref'] is None) == (c['shard_rows'] == 0) and
         (c['shard_ref'] is None or (type(c['shard_ref']) is dict and set(c['shard_ref']) == {'path', 'sha256', 'bytes'})),
         'shard join ref and shard row count')
    need(c['baseline_source'] != 'shard' or c['shard_ref'] is not None, 'shard anchor has a shard join')


def validate_doc(p):
    """Closed validation of one parsed delta packet on its own (chain properties: ``ChainAccumulator.add``)."""
    need(type(p) is dict and set(p) == PACKET_KEYS and p['schema'] == PACKET_SCHEMA, 'closed overlay packet schema')
    seq = p['seq']
    need(type(seq) is int and 1 <= seq <= MAX_CHAIN_PACKETS, 'packet sequence')
    pred = p['predecessor_overlay_head_sha256']
    need((seq == 1) == (pred is None) and (pred is None or hash64(pred)), 'genesis iff no predecessor head')
    ch = p['chain']
    need(type(ch) is dict and set(ch) == {'packets_before', 'rows_before'} and ch['packets_before'] == seq - 1 and
         type(ch['rows_before']) is int and 0 <= ch['rows_before'] <= MAX_CHAIN_ROWS and
         (seq > 1 or ch['rows_before'] == 0), 'chain position')
    need(p['limits'] == LIMITS and p['stopped'] in STOPS, 'fixed packet limits and stop reason')
    need(p['new_ce_teachers'] == 0 and p['optimizer_updates'] == 0 and p['admission_committed'] is False,
         'packet flags: no CE teacher, no optimizer update, no admission')
    pins = p['pins']
    need(type(pins) is dict and set(pins) == PIN_KEYS and pins['overlay_window'] == OVERLAY_WINDOW and
         pins['rotation_domain'] == ROTATION_DOMAIN, 'closed packet pins')
    for k in PIN_KEYS - {'input_lists', 'source_sha256', 'builder_source_sha256', 'overlay_window', 'rotation_domain'}:
        need(hash64(pins[k]), 'pin ' + k)
    need(type(pins['input_lists']) is list and pins['input_lists'] and
         all(type(r) is dict and set(r) == {'path', 'sha256', 'bytes'} and hash64(r['sha256']) for r in pins['input_lists']),
         'pinned input lists')
    for k in ('source_sha256', 'builder_source_sha256'):
        need(type(pins[k]) is dict and pins[k] and all(hash64(v) for v in pins[k].values()), 'source pins ' + k)
    need(type(p['campaigns']) is list and p['campaigns'] and all(hash64(c) for c in p['campaigns']) and
         p['campaigns'] == sorted(set(p['campaigns'])), 'campaign allow-list')
    kinds = [s.get('outbox_kind') for s in p['sources']] if type(p['sources']) is list else []
    need(kinds and len(kinds) == len(set(kinds)) and all(
        type(s) is dict and set(s) == {'outbox_kind', 'backup_sha256', 'event_head', 'rows'} and hash64(s['backup_sha256'])
        and type(s['event_head']) is dict and set(s['event_head']) == {'events', 'head_sha256'} and
        type(s['event_head']['events']) is int and hash64(s['event_head']['head_sha256']) and type(s['rows']) is int
        for s in p['sources']), 'declared outbox sources')
    proofs = p['authority_proofs']
    need(type(proofs) is dict and set(proofs) == {'first', 'final'} and all(
        type(v) is dict and set(v) == {'sha256', 'checked_at_unix', 'observed_unix'} and hash64(v['sha256'])
        for v in proofs.values()), 'first/final authority proofs')
    rows, contexts, unions = p['rows'], p['contexts'], p['label_unions']
    need(type(rows) is list and len(rows) <= MAX_ROWS and type(contexts) is dict and type(unions) is dict,
         'delta packet row limit')
    touched = {r.get('context_id') for r in rows if type(r) is dict}
    need(len(touched | set(contexts)) <= MAX_CONTEXTS and set(contexts) <= touched and set(unions) == touched,
         'packet context limit; anchors and label unions exactly for the touched contexts')
    for cid, c in contexts.items():
        _anchor(cid, c, seq)
    keys, programs = set(), set()
    for row in rows:
        need(type(row) is dict and set(row) == ROW_FIELDS, 'closed overlay row')
        cid = row['context_id']
        need(row['partition'] == 'train' and hash64(row['parent_lineage_sha256']) and row['packet_seq'] == seq,
             'overlay row on a train context, sequenced in this packet')
        need(hash64(row['evaluation_key']) and row['evaluation_key'] not in keys and
             (cid, row['program_sha256']) not in programs, 'unique evaluation key and context/program in the packet')
        keys.add(row['evaluation_key'])
        programs.add((cid, row['program_sha256']))
        need(type(row['program']) is str and sha(row['program'].encode()) == row['program_sha256'], 'program bytes')
        ref5(row['result_ref'])
        artifact_refs(row['artifact_refs'])
        need(set(row['original_score_context']) == SCORE_CONTEXT_KEYS and
             row['original_score_context']['identity'] == cid, 'typed original score context')
        need(_pair(row['inner_fold_rmse']) and (row['wall_us'] is None or type(row['wall_us']) is int),
             'verified inner-fold RMSE')
        b = row['baseline']
        need(type(b) is dict and set(b) == {'reconstructed', 'evaluation', 'anchored', 'source'} and
             _pair(b['reconstructed']) and _pair(b['anchored']) and (b['evaluation'] is None or _pair(b['evaluation'])) and
             b['source'] in ('shard', 'chain', 'reconstructed'), 'row baseline record')
        if cid in contexts:
            c = contexts[cid]
            need(b['anchored'] == c['anchored_baseline'] and row['shard_ref'] == c['shard_ref'] and
                 row['parent_lineage_sha256'] == c['parent_lineage_sha256'], 'row anchored baseline equals the context anchor')
        _record(row['record'], row)
        ids = row['source_emission_ids']
        need(type(ids) is list and ids and len(ids) == len(set(ids)) and all(hash64(x) for x in ids) and
             type(row['source_observations']) is list and len(row['source_observations']) == len(ids) and
             all(hash64(x) for x in row['source_observations']), 'emission provenance (earliest first, aliases kept)')
    for cid, u in unions.items():
        own = sum(r['context_id'] == cid for r in rows)
        need(type(u) is dict and set(u) == UNION_FIELDS and all(type(u[k]) is int and u[k] >= 0 for k in
             ('shard_rows', 'earlier_rows', 'packet_rows')) and u['packet_rows'] == own and
             hash64(u['overlay_programs_sha256']), 'closed per-context label union record')
        if cid in contexts:
            need(u['earlier_rows'] == 0 and u['shard_rows'] == contexts[cid]['shard_rows'] and
                 u['overlay_programs_sha256'] == overlay_programs_digest(
                     [r['program_sha256'] for r in rows if r['context_id'] == cid]), 'first-seen label union')
    c = p['census']
    need(type(c) is dict and set(c) == {'source_rows', 'classes', 'entries'} and type(c['entries']) is list and
         len(c['entries']) == c['source_rows'] <= MAX_SOURCE_ROWS, 'census entries')
    counts, rows_by_new = Counter(), {}
    for e in c['entries']:
        need(type(e) is dict and set(e) == ENTRY_FIELDS and e['class'] in CLASSES and e['reason'] in REASONS[e['class']],
             'closed census entry')
        need(hash64(e['observation_sha256']) and type(e['outbox_kind']) is str and
             (e['detail'] is None or (type(e['detail']) is str and len(e['detail']) <= 500)), 'census entry identity')
        counts[e['class']] += 1
        if e['class'] == 'TRAIN_VERIFIED' and e['reason'] in ('NEW_OVERLAY_ROW', 'ALIAS'):
            need(type(e['row']) is int and 0 <= e['row'] < len(rows), 'new row census join')
            rows_by_new.setdefault(e['row'], []).append(e)
        elif e['reason'] == 'ALREADY_IN_OVERLAY':
            need(type(e['row']) is list and len(e['row']) == 2 and all(type(v) is int for v in e['row']) and
                 1 <= e['row'][0] < seq and e['row'][1] >= 0 and hash64(e['evaluation_key']), 'earlier chain row join')
        else:
            need(e['row'] is None, 'non-row census entry')
    need(c['classes'] == {k: counts.get(k, 0) for k in CLASSES} and sum(c['classes'].values()) == c['source_rows'],
         'census of five classes sums exactly to the source rows')
    need(set(rows_by_new) == set(range(len(rows))), 'every new row is censused')
    for i, entries in rows_by_new.items():
        need([e['reason'] for e in entries][0] == 'NEW_OVERLAY_ROW' and
             sum(e['reason'] == 'NEW_OVERLAY_ROW' for e in entries) == 1 and
             [e['emission_id'] for e in entries] == rows[i]['source_emission_ids'] and
             [e['observation_sha256'] for e in entries] == rows[i]['source_observations'],
             'earliest emission wins and aliases are kept in census order')
    need(sum(s['rows'] for s in p['sources']) == c['source_rows'], 'source row counts sum to the census')
    return p


_PROGRAM = RECORD_FIELDS.index('program_sha256')
_TOKENS = RECORD_FIELDS.index('tokens')


class ChainRow:
    """One compact overlay row of the cumulative view (no raw bytes, no census, no artifact refs or bodies).

    The record is kept as a field tuple with tokens packed as uint16 and the result ref as a 5-tuple; ``record`` and
    ``result_ref`` materialize a fresh ``NativeRecord`` / dict on access. Repeated strings are shared via the view.
    """
    __slots__ = ('context_id', 'evaluation_key', '_ref', '_record', '_tokens', 'rmse', 'wall_us', 'packet_seq',
                 'index', 'lineage', 'anchored')

    def __init__(self, row, index, anchored, strings):
        rec = record_from_json(row['record'])
        share = strings.setdefault
        self.context_id = share(row['context_id'], row['context_id'])
        self.lineage = share(row['parent_lineage_sha256'], row['parent_lineage_sha256'])
        self.evaluation_key = row['evaluation_key']
        r = ref5(row['result_ref'])
        self._ref = (share(r['bucket'], r['bucket']), r['key'], r['version_id'], r['sha256'], r['bytes'])
        self._tokens = array('H', rec.tokens).tobytes()
        self._record = tuple(None if k == 'tokens' else share(v, v) if k == 'source_kind' else v
                             for k, v in zip(RECORD_FIELDS, (getattr(rec, f) for f in RECORD_FIELDS)))
        self.rmse, self.wall_us = tuple(row['inner_fold_rmse']), row['wall_us']
        self.packet_seq, self.index, self.anchored = row['packet_seq'], index, anchored

    @property
    def program_sha256(self):
        return self._record[_PROGRAM]

    @property
    def ref_tuple(self):
        return self._ref

    @property
    def result_ref(self):
        return dict(zip(REF_KEYS, self._ref))

    @property
    def record(self):
        values = list(self._record)
        values[_TOKENS] = tuple(array('H', self._tokens))
        return NativeRecord(*values)


class ChainView:
    """Cumulative view of a validated overlay chain (what ``overlay_snapshot`` and ``build_packet`` consume)."""

    def __init__(self):
        self.head_sha256, self.seq, self.packets, self.pins = None, 0, [], None
        self.contexts, self.rows, self.offsets = {}, [], []
        self.keys, self.programs, self.refs, self.by_context, self.strings = {}, set(), set(), {}, {}

    def context_rows(self, cid):
        return [self.rows[i] for i in self.by_context.get(cid, ())]

    def row_at(self, seq, index):
        need(1 <= seq <= self.seq, 'chain row sequence')
        start = self.offsets[seq - 1]
        end = self.offsets[seq] if seq < self.seq else len(self.rows)
        need(0 <= index < end - start, 'chain row index')
        return self.rows[start + index]

    def add(self, p, *, packet_sha256, packet_bytes, path):
        """Accumulate one validated packet; refuses every whole-chain violation before mutating the view."""
        need(self.seq < MAX_CHAIN_PACKETS, 'bounded chain packet count')
        # Chain-order guard: exact sequence, predecessor head link and chain position.
        need(p['seq'] == self.seq + 1 and p['predecessor_overlay_head_sha256'] == self.head_sha256,
             'chain sequence and predecessor head link')
        need(p['chain'] == {'packets_before': self.seq, 'rows_before': len(self.rows)}, 'chain position')
        need(self.pins is None or p['pins'] == self.pins, 'identical pins across the chain')
        need(len(self.rows) + len(p['rows']) <= MAX_CHAIN_ROWS, 'bounded chain row count')
        for cid in p['contexts']:
            need(cid not in self.contexts, 'a context anchor is declared once (its first-seen packet only)')
        anchors = {**self.contexts, **p['contexts']}
        keys, programs, refs = set(), set(), set()
        for r in p['rows']:
            cid = r['context_id']
            a = anchors.get(cid)
            need(a is not None, 'row context has a first anchor in the chain')
            # Anchor-drift guard: every row of a context carries its FIRST anchor bit-for-bit.
            need(r['baseline']['anchored'] == a['anchored_baseline'] and r['shard_ref'] == a['shard_ref'] and
                 r['parent_lineage_sha256'] == a['parent_lineage_sha256'], 'anchored baseline drifts from the first anchor')
            # Whole-chain duplicate guard: (context, program), evaluation key and result ref are unique.
            ref_digest = tuple(ref5(r['result_ref'])[k] for k in REF_KEYS)
            need((cid, r['program_sha256']) not in self.programs and r['evaluation_key'] not in self.keys and
                 ref_digest not in self.refs, 'duplicate (context, program), evaluation key or result ref across the chain')
            keys.add(r['evaluation_key'])
            programs.add((cid, r['program_sha256']))
            refs.add(ref_digest)
        # Label-union guard: shard rows + ALL earlier chain rows + own rows of each touched context.
        for cid, u in p['label_unions'].items():
            prior = [self.rows[i].program_sha256 for i in self.by_context.get(cid, ())]
            own = [r['program_sha256'] for r in p['rows'] if r['context_id'] == cid]
            need(u['earlier_rows'] == len(prior) and u['packet_rows'] == len(own) and
                 u['shard_rows'] == anchors[cid]['shard_rows'] and
                 u['overlay_programs_sha256'] == overlay_programs_digest(prior + own),
                 'label union is shard rows + all earlier chain rows + own rows')
        for e in p['census']['entries']:
            if e['reason'] == 'ALREADY_IN_OVERLAY':
                need(self.row_at(*e['row']).evaluation_key == e['evaluation_key'], 'ALREADY_IN_OVERLAY joins a chain row')
        self.offsets.append(len(self.rows))
        for i, r in enumerate(p['rows']):
            row = ChainRow(r, i, anchors[r['context_id']]['anchored_baseline'], self.strings)
            self.by_context.setdefault(row.context_id, []).append(len(self.rows))
            self.rows.append(row)
            self.keys[row.evaluation_key] = (row.packet_seq, i, digest({'record': ref5(r['result_ref']),
                                                                         'artifacts': artifact_refs(r['artifact_refs'])}))
        self.programs |= programs
        self.refs |= refs
        self.contexts.update(p['contexts'])
        self.pins = p['pins']
        self.head_sha256 = head_from_sha(packet_sha256, self.head_sha256)
        self.seq += 1
        self.packets.append({'path': path, 'sha256': packet_sha256, 'bytes': packet_bytes, 'seq': self.seq})
        return self


def validate_packet(packet_ref, *, predecessor_head):
    """Strict read-only STRUCTURAL validation of ONE delta packet (no whole-chain properties, no admission: an
    admissible packet additionally needs its in-time terminal acceptance, see ``accepted_publication``)."""
    raw = read_exact(packet_ref, MAX_PACKET_BYTES)
    p = exact_canonical(raw)
    validate_doc(p)
    need(p['predecessor_overlay_head_sha256'] == predecessor_head, 'packet predecessor head')
    return p


LATE_REFUSAL = 'overlay publication refused: late terminal record kept for custody'
NO_ACCEPTANCE = 'overlay publication has no caller terminal acceptance (the build has not returned in time)'
ARTIFACTS = (('packet.json', 'packet'), ('receipt.json', 'receipt'))
COMPLETION_SCHEMA = 'jope.native-overlay-acceptance-complete/v1'
COMPLETION_KEYS = frozenset({'schema', 'root', 'acceptance', 'bound', 'completion_sample'})


def completion_path(root):
    """Caller completion, outside the root; a pending file is never an admission marker."""
    out = acceptance_path(root)
    return out.with_name(root.name + '.acceptance-complete.json')


def _ref(path, raw):
    return {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}


def _marker(root):
    """The closed v2 success marker of a build root (its sealed original bound is strict and finite)."""
    raw = read_small(root / MARKER_NAME, MAX_TERMINAL_BYTES)
    m = exact_canonical(raw)
    need(type(m) is dict and set(m) == MARKER_KEYS and m['schema'] == MARKER_SCHEMA and
         all(type(m[k]) is dict and m[k].get('path') == str(root / name) for name, k in ARTIFACTS),
         'overlay success marker of this root')
    check_bound(m['bound'])
    return raw, m


def accept_publication(root, *, physical):
    """CALLER/OPERATOR-ONLY, and only after ``build_packet`` RETURNED successfully (``in_process_return`` of the
    returned marker) or its unit terminated (``systemd_unit``): seal the ONE create-only bounded terminal acceptance
    envelope at ``acceptance_path(root)`` (outside the root; C10a r7 ``accept_subject``). The post-return clock sample
    is taken FIRST, before anything is read. It pins the exact marker bytes, the sealed packet and receipt (re-read
    exactly), the unchanged original bound and the physical evidence. A failed predicate is written as a ``refused``
    envelope (custody only; nothing is deleted, no window is reset) and then raises; if even that write fails, there
    is no qualified completion and nothing is admissible. Never call it after ``build_packet`` raised.

    Accepted bytes alone are custody, not completion. Only after their parent fsync succeeds do we create and
    file-fsync a private completion, fsync its directory, and expose it with a create-only atomic hard link.
    There is no fallible I/O after that visibility point. Concurrent readers never see a pending file; any earlier
    failure leaves all bytes in place and cannot be retried. A crash that loses the new link fails closed; surviving
    links always name the already-fsynced completion of the durable envelope. No negative record is needed.
    """
    root = safe_path(root)
    out = acceptance_path(root)
    complete = completion_path(root)
    pending = complete.with_name('.' + complete.name + '.pending')
    need(root.is_dir() and out.parent.is_dir() and
         not any(os.path.lexists(p) for p in (out, complete, pending)), 'create-only acceptance outside the root')
    post = clock_sample()
    doc = {'schema': ACCEPTANCE_SCHEMA, 'subject': ACCEPTANCE_SUBJECT, 'root': str(root), 'artifacts': {},
           'marker': None, 'bound': None, 'physical': physical, 'post_publication_sample': post, 'verdict': 'refused',
           'reason': None}
    try:
        need(not os.path.lexists(root / REFUSED_NAME), LATE_REFUSAL)
        marker_raw, marker = _marker(root)
        doc['marker'] = _ref(root / MARKER_NAME, marker_raw)
        for name, key in ARTIFACTS:
            read_exact(marker[key], MAX_PACKET_BYTES)
            doc['artifacts'][name] = marker[key]
        doc['bound'] = marker['bound']
        need(physical_ok(physical, doc['bound'], post, sha(marker_raw)),
             'physical terminal evidence is not an in-time success inside the original bound')
        doc['verdict'], doc['reason'] = 'accepted', 'in-time physical success after the build returned'
    except (ValueError, OSError, KeyError, TypeError) as exc:
        doc['reason'] = (type(exc).__name__ + ': ' + str(exc))[:300]
    ref = create_only(out, canonical(doc))
    fsync_dir(out.parent)
    need(doc['verdict'] == 'accepted', 'overlay acceptance refused (kept as custody): ' + doc['reason'])
    sample = clock_sample()
    need(in_bound(sample, doc['bound']), 'overlay acceptance completion outside the original bound (kept as custody)')
    completion = {'schema': COMPLETION_SCHEMA, 'root': str(root), 'acceptance': ref,
                  'bound': doc['bound'], 'completion_sample': sample}
    create_only(pending, canonical(completion))
    fsync_dir(out.parent)
    need(in_bound(clock_sample(), doc['bound']),
         'overlay acceptance completion outside the original bound (kept as custody)')
    os.link(pending, complete, follow_symlinks=False)
    return ref


def accepted_publication(packet_ref, *, head_sha256, predecessor_head, seq):
    """Consumer gate: the caller's ``accepted`` envelope AND qualified completion for this build root, no refusal.

    The envelope must pin the exact success-marker bytes (whose marker binds this packet ref, head, predecessor head
    and sequence), the sealed packet and receipt, the UNCHANGED original bound sealed in the marker, and physical
    in-time evidence of the returned build. Returns the envelope."""
    need(type(packet_ref) is dict and type(packet_ref.get('path')) is str, 'packet ref')
    path = safe_path(packet_ref['path'])
    root = path.parent
    need(path == root / 'packet.json', 'packet is the packet.json of its build root')
    out = acceptance_path(root)
    need(not os.path.lexists(root / REFUSED_NAME), LATE_REFUSAL)
    need(os.path.lexists(root / MARKER_NAME) and os.path.lexists(out), NO_ACCEPTANCE)
    marker_raw, marker = _marker(root)
    need(marker['packet'] == packet_ref and marker['head_sha256'] == head_sha256 and
         marker['predecessor_overlay_head_sha256'] == predecessor_head and marker['seq'] == seq,
         'overlay success marker binds this packet, head and sequence')
    a = exact_canonical(read_small(out, MAX_TERMINAL_BYTES))
    need(type(a) is dict and set(a) == ACCEPTANCE_KEYS and a['schema'] == ACCEPTANCE_SCHEMA and
         a['subject'] == ACCEPTANCE_SUBJECT and a['verdict'] == 'accepted' and a['root'] == str(root),
         'accepted bounded terminal envelope for this overlay root')
    need(a['marker'] == _ref(root / MARKER_NAME, marker_raw), 'acceptance pins the exact success marker bytes')
    need(a['artifacts'] == {name: marker[key] for name, key in ARTIFACTS}, 'acceptance pins the sealed artifacts')
    need(a['bound'] == marker['bound'], 'acceptance carries the unchanged original bound sealed in the marker')
    need(physical_ok(a['physical'], a['bound'], a['post_publication_sample'], sha(marker_raw)),
         'acceptance physical terminal evidence inside the original bound')
    complete = completion_path(root)
    need(os.path.lexists(complete), 'overlay acceptance has no durable caller completion')
    c = exact_canonical(read_small(complete, MAX_TERMINAL_BYTES))
    need(type(c) is dict and set(c) == COMPLETION_KEYS and c['schema'] == COMPLETION_SCHEMA and
         c['root'] == str(root) and c['acceptance'] == _ref(out, canonical(a)) and c['bound'] == a['bound'],
         'durable caller completion pins the exact acceptance and original bound')
    need(in_bound(c['completion_sample'], c['bound']) and
         c['completion_sample']['monotonic'] >= a['post_publication_sample']['monotonic'],
         'durable caller completion is in time after the acceptance sample')
    need(not os.path.lexists(root / REFUSED_NAME), LATE_REFUSAL)
    return a


def validate_chain(packet_refs_in_order, *, expected_head):
    """Strictly validate packets 1..n in order and return the cumulative ``ChainView`` ending at ``expected_head``.

    Every packet must also carry the caller's accepted bounded terminal envelope (``accepted_publication``); the gate
    runs after the structural and whole-chain checks, so it never relaxes them."""
    need(type(packet_refs_in_order) in (list, tuple) and 0 < len(packet_refs_in_order) <= MAX_CHAIN_PACKETS and
         hash64(expected_head), 'ordered bounded packet refs and an expected head')
    view, links = ChainView(), []
    for ref in packet_refs_in_order:
        raw = read_exact(ref, MAX_PACKET_BYTES)
        p = validate_doc(exact_canonical(raw))
        before = view.head_sha256
        view.add(p, packet_sha256=sha(raw), packet_bytes=len(raw), path=ref['path'])
        links.append((ref, view.head_sha256, before, view.seq))
        del raw, p                                        # no raw bytes or parsed packets are retained
    need(view.head_sha256 == expected_head, 'chain head recomputes to the expected head')
    for ref, head, before, seq in links:
        accepted_publication(ref, head_sha256=head, predecessor_head=before, seq=seq)
    return view


def completed_packet(root, *, predecessor_head):
    """Read a build root ACCEPTED by its caller, through its success marker; returns (packet, packet_ref, head)."""
    root = safe_path(root)
    marker = exact_canonical(read_small(root / MARKER_NAME, MAX_TERMINAL_BYTES))
    need(type(marker) is dict and marker.get('schema') == MARKER_SCHEMA and type(marker.get('packet')) is dict and
         marker['packet'].get('path') == str(root / 'packet.json'), 'overlay success marker')
    p = validate_packet(marker['packet'], predecessor_head=predecessor_head)
    head = head_sha256(canonical(p), predecessor_head)
    need(head == marker['head_sha256'] and p['seq'] == marker['seq'], 'marker head/sequence')
    accepted_publication(marker['packet'], head_sha256=head, predecessor_head=predecessor_head, seq=p['seq'])
    return p, marker['packet'], head


__all__ = ['ChainView', 'ChainRow', 'head_sha256', 'head_from_sha', 'validate_doc', 'validate_packet',
           'validate_chain', 'completed_packet', 'accept_publication', 'accepted_publication',
           'overlay_programs_digest']
