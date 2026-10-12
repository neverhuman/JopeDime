"""OverlayLoader and overlay_snapshot: append verified overlay records to a FROZEN base window (no frozen edits).

Invariants:
- The base loader (NativeContextLoader or MixedLoader, anything with ``get(ids, epoch=, batch_index=)``) is called
  unchanged and its window (records and order) is kept, except the single winner merge below.
- Per context with overlay rows O_c: up to k_c = min(|O_c|, 8) records are appended in the
  ``digest{native-overlay-rotation/v1, context, program}`` order starting at ``(epoch*k_c) % |O_c|``; programs already
  in the window are skipped; selection stops before ``(len(window)+1) * max_len > max_padded_tokens``.
- Winner merge: an overlay row whose program is the frozen HWM winner, while the window holds that CE-only winner
  record, replaces it by ONE record that keeps the frozen ``ce_eligible`` and takes q/feasibility/runtime from the
  overlay. The overlay never creates a CE teacher: every window has exactly as many CE records as its base window.
- ``overlay_snapshot`` takes the cumulative view of a validated DELTA chain (revision 2) and returns a shallow copy
  whose identity, batch_manifest and context membership are unchanged.
"""
import copy
import math
import time
from dataclasses import replace

import numpy as np

from jope_flywheel import training_snapshot_native as N

from .common import (DRY_RUN_SCHEMA, MERGED_SOURCE_KIND, OVERLAY_WINDOW, ROTATION_DOMAIN, SOURCE_KIND, WINDOW_SCHEMA,
                     digest, need, ref5, sha)
from .packet import ChainView


def rotation_key(cid, program_sha256):
    return digest({'domain': ROTATION_DOMAIN, 'context': cid, 'program': program_sha256})


class OverlayLoader:
    def __init__(self, base_loader, *, winners, overlay, provenance, head, seq, identity, max_padded_tokens):
        need(callable(getattr(base_loader, 'get', None)), 'base loader with get(ids, epoch=, batch_index=)')
        need(type(max_padded_tokens) is int and max_padded_tokens > 0, 'bounded padded-token budget')
        self.base, self.winners, self.overlay, self.provenance = base_loader, winners, overlay, provenance
        self.head, self.seq, self.identity, self.max_padded_tokens = head, seq, identity, max_padded_tokens

    def _member(self, record, cid, merged):
        p = self.provenance[(cid, record.program_sha256)]
        return {'program_sha256': record.program_sha256, 'evaluation_key': p['evaluation_key'],
                'result_ref_sha256': digest(p['result']), 'packet_seq': p['packet_seq'], 'merged_winner': merged}

    def get(self, context_ids, *, epoch, batch_index):
        ids = list(context_ids)
        b = self.base.get(ids, epoch=epoch, batch_index=batch_index)
        records, support, members = dict(b.records), {c: dict(v) for c, v in b.support.items()}, []
        for cid in ids:
            window = list(b.records[cid])
            base_ce = sum(r.ce_eligible for r in window)
            rows = self.overlay.get(cid, ())
            m = {'context_id': cid, 'base_program_ids': [r.program_sha256 for r in window], 'overlay_rows': len(rows),
                 'k': 0, 'start': 0, 'selected': [], 'skipped_present': [], 'budget_stopped': False}
            if rows:
                n = len(rows)
                k = min(n, OVERLAY_WINDOW)
                start = (epoch * k) % n
                m.update(k=k, start=start)
                position = {r.program_sha256: i for i, r in enumerate(window)}
                winner = self.winners.get(cid)
                width = max((len(r.tokens) for r in window), default=0)
                taken = 0
                for j in range(n):
                    if taken == k:
                        break
                    r = rows[(start + j) % n]
                    if r.program_sha256 in position:
                        i = position[r.program_sha256]
                        w = window[i]
                        if winner is not None and r.program_sha256 == winner.program_sha256 and w == winner:
                            need(tuple(r.tokens) == tuple(w.tokens), 'winner merge token identity')
                            window[i] = replace(w, q=r.q, q_opt=r.q_opt, q_grid=r.q_grid, feasibility=r.feasibility,
                                                runtime_seconds=r.runtime_seconds, runtime_censored=r.runtime_censored,
                                                quality_eligible=r.quality_eligible,
                                                pairwise_eligible=r.pairwise_eligible,
                                                runtime_eligible=r.runtime_eligible, source_kind=MERGED_SOURCE_KIND)
                            m['selected'].append(self._member(r, cid, True))
                            taken += 1
                        else:
                            m['skipped_present'].append(r.program_sha256)
                        continue
                    wide = max(width, len(r.tokens))
                    # Token-budget guard: never let the context window exceed the padded-token budget.
                    if (len(window) + 1) * wide > self.max_padded_tokens:
                        m['budget_stopped'] = True
                        break
                    window.append(r)
                    position[r.program_sha256] = len(window) - 1
                    width = wide
                    m['selected'].append(self._member(r, cid, False))
                    taken += 1
            # CE guard: the overlay never creates a CE teacher.
            need(sum(r.ce_eligible for r in window) == base_ce and
                 not any(r.ce_eligible for r in window if r.source_kind == SOURCE_KIND),
                 'overlay never creates a CE teacher')
            records[cid] = tuple(window)
            m['program_ids'] = [r.program_sha256 for r in window]
            m['padded_tokens'] = len(window) * max((len(r.tokens) for r in window), default=0)
            support.setdefault(cid, {})
            support[cid] = {**support[cid], 'overlay': {'rows': len(rows), 'selected': len(m['selected']),
                                                        'budget_stopped': m['budget_stopped'],
                                                        'packet_seq': self.seq}}
            members.append(m)
        receipt = {'schema': WINDOW_SCHEMA, 'snapshot_sha256': self.identity, 'overlay_head_sha256': self.head,
                   'overlay_packet_seq': self.seq, 'epoch': epoch, 'batch_index': batch_index,
                   'base': b.selection_receipt, 'members': members, 'membership_sha256': digest(members)}
        return N.NativeBatch(b.contexts, b.encodings, records, support, receipt, b.data)


def overlay_snapshot(base_snapshot, chain_view, *, max_padded_tokens, contexts=None):
    """Shallow copy of ``base_snapshot`` whose ``context_loader`` is an OverlayLoader over the CUMULATIVE union of a
    validated chain (``validate_chain``); identity, batch_manifest and membership are unchanged. The per-context
    rotation runs over ALL overlay rows of the context, whichever packets they came from. ``contexts`` (pre-publication
    dry run only) restricts the overlay to those contexts' complete row sets; the learner always uses ``None``."""
    need(isinstance(chain_view, ChainView) and chain_view.seq >= 1 and chain_view.head_sha256 is not None,
         'a validated overlay chain view')
    nat = getattr(base_snapshot, 'native', base_snapshot)
    need(nat.identity == chain_view.pins['native_spec_sha256'], 'chain pins the frozen native base snapshot')
    need(type(max_padded_tokens) is int and max_padded_tokens >= base_snapshot.max_seq, 'bounded padded-token budget')
    overlay, provenance, records = {}, {}, {}
    rows = chain_view.rows if contexts is None else [r for c in sorted(set(contexts)) for r in chain_view.context_rows(c)]
    for row in rows:
        cid, rec, result = row.context_id, row.record, row.result_ref
        c = base_snapshot.contexts.get(cid)
        need(c is not None and c == nat.contexts.get(cid) and c['partition'] == 'train' and
             c['source_kind'] == 'native_s1_context' and c['parent_lineage_sha256'] == row.lineage,
             'overlay rows only on frozen native train contexts')
        need(not rec.ce_eligible and rec.source_kind == SOURCE_KIND, 'overlay never creates a CE teacher')
        need(nat.token_refusal(rec.tokens) is None and
             sha(nat.tokenizer.detokenize(rec.tokens, nat.vocab).encode()) == rec.program_sha256,
             'overlay record tokens reproduce the program')
        overlay.setdefault(cid, []).append(rec)
        provenance[(cid, rec.program_sha256)] = {'evaluation_key': row.evaluation_key, 'result': result,
                                                 'packet_seq': row.packet_seq}
        records.setdefault(cid, []).append({'program_sha256': rec.program_sha256, 'result': dict(result),
                                            'evaluation_key': row.evaluation_key, 'packet_seq': row.packet_seq})
    ordered = {cid: tuple(sorted(rs, key=lambda r, c=cid: rotation_key(c, r.program_sha256)))
               for cid, rs in overlay.items()}
    s = copy.copy(base_snapshot)
    s.context_loader = OverlayLoader(base_snapshot.context_loader, winners=dict(nat.winners), overlay=ordered,
                                     provenance=provenance, head=chain_view.head_sha256, seq=chain_view.seq,
                                     identity=base_snapshot.identity, max_padded_tokens=max_padded_tokens)
    s.overlay_records = records
    s.overlay_base_snapshot = base_snapshot      # fixed panels stay on the frozen base windows (root integration)
    s.overlay_head_sha256 = chain_view.head_sha256
    s.overlay_packet_seq = chain_view.seq
    s.overlay_packets = [dict(p) for p in chain_view.packets]
    return s


def overlay_ref_digest(snapshot, context_id, program_sha256):
    """The exact normalized result-ref digest an overlay snapshot holds for one record, or None.

    Same form as the C10 r4 boundary ``_ref_in`` (``digest(normalized 5-field ref)``), so ``first_exposures`` can join
    an overlay row exactly like a variant row; the window receipt carries the same digest as ``result_ref_sha256``.
    """
    found = [x for x in getattr(snapshot, 'overlay_records', {}).get(context_id, ())
             if x['program_sha256'] == program_sha256]
    need(len(found) <= 1, 'one overlay record per program')
    return digest(ref5(found[0]['result'])) if found else None


def normalizer_from(preprocessing):
    """The preserved train-only normalizer exactly as ``neural_native_train.fit_preprocessing``'s transform."""
    mean = np.asarray(preprocessing['mean'], dtype=np.float64)
    scale = np.asarray(preprocessing['scale'], dtype=np.float64)
    lo, hi = preprocessing['clip']

    def transform(cid, x):
        return np.clip((np.asarray(x, dtype=np.float64) - mean) / scale, lo, hi).astype(np.float32)
    return transform


def dry_run(s, *, normalizer, max_padded_tokens, deadline, contexts=None):
    """Every overlay context (default all; a build passes the contexts its packet touched, whose overlay sets changed)
    at every rotation offset of its CUMULATIVE overlay set, through the real prepare_microbatches + collate (CPU)."""
    from neural_native_train import prepare_microbatches
    runs, expected, seen = [], 0, {}
    targets = sorted(s.overlay_records if contexts is None else contexts)
    need(set(targets) <= set(s.overlay_records), 'dry-run contexts carry overlay rows')
    for cid in targets:
        n = len(s.overlay_records[cid])
        k = min(n, OVERLAY_WINDOW)
        offsets = n // math.gcd(n, k)
        expected += offsets
        seen[cid] = set()
        for epoch in range(offsets):
            need(time.time() < deadline, 'overlay dry-run deadline')
            batches, total, receipt = prepare_microbatches(s, [cid], epoch=epoch, batch_index=0, normalizer=normalizer,
                                                           max_padded_tokens=max_padded_tokens)
            m = receipt['selection']['members'][0]
            padded = [b.receipt['padded_tokens'] for b in batches]
            need(all(v <= max_padded_tokens for v in padded), 'dry-run padded tokens within budget')
            seen[cid].update(x['program_sha256'] for x in m['selected'])
            runs.append({'context_id': cid, 'epoch': epoch, 'start': m['start'], 'k': m['k'],
                         'selected': [x['program_sha256'] for x in m['selected']], 'budget_stopped': m['budget_stopped'],
                         'skipped_present': m['skipped_present'], 'padded_tokens': padded, 'microbatches': len(batches),
                         'denominators': total})
    ids = targets
    combined = []
    for i in range(0, len(ids), N.MAX_BATCH_CONTEXTS):
        batches, total, _ = prepare_microbatches(s, ids[i:i + N.MAX_BATCH_CONTEXTS], epoch=0, batch_index=i,
                                                 normalizer=normalizer, max_padded_tokens=max_padded_tokens)
        combined.append({'contexts': len(ids[i:i + N.MAX_BATCH_CONTEXTS]), 'microbatches': len(batches),
                         'denominators': total})
    never = {cid: sorted({r['program_sha256'] for r in s.overlay_records[cid]} - seen[cid]) for cid in ids}
    return {'schema': DRY_RUN_SCHEMA, 'max_padded_tokens': max_padded_tokens, 'contexts': len(ids),
            'expected_runs': expected, 'runs': len(runs), 'passed': len(runs) == expected,
            'never_selected': {c: v for c, v in never.items() if v}, 'combined_batches': combined, 'detail': runs}
