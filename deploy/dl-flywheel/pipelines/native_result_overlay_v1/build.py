"""build_packet (revision 2): verified native results -> one create-only DELTA packet appended to a validated chain.

Invariants:
- Input is an injected iterator of uniform verified-result rows (W4 ``verified_results``). Every consumed row lands in
  exactly one census class (TRAIN_VERIFIED, EVALUATION_ONLY, ROUTED_TO_X1, REFUSED, PENDING) and the census sums to
  the consumed rows. Unknown, leased, recorded and blocked rows stay PENDING and are never zero-filled; failure custody
  is R5 PARITY_UNQUALIFIED (no q, no feasibility); variant rows are routed to X1; validation/R6 rows are evaluation-only.
- The emission identity is re-derived from W4's ``emission_body_b64`` (exact stored bytes) and its sha256.
- Train rows must be on a frozen native train context inside the normalizer fit roster, outside the R6 roster and
  outside the current protected roles, and must pass every step of ``verify``. Earliest emission wins, aliases are
  kept as provenance, conflicting refs for one key are refused, ALREADY_IN_SHARD / ALREADY_IN_OVERLAY (anywhere in the
  chain) are marked, and a shard duplicate whose per-fold RMSE disagrees is refused (SHARD_RMSE_DISAGREEMENT).
- A packet holds at most 2,048 NEW rows and touches at most 1,024 contexts: the build stops PULLING rows from the
  iterator when either cap is reached (``stopped``), so the caller passes the same iterator to the next build and no
  row is lost or double-consumed. The chain itself is bounded at 4,096 packets / 500,000 rows.
- Labels come from the frozen ``label_context`` on the union of the shard rows, ALL earlier chain rows and this
  packet's rows of each touched context (recorded in ``label_unions``); every overlay row carries its context's FIRST
  anchor; old shard labels and every earlier chain record must be unchanged; every new record equals its direct
  recomputation.
- Exact original objective, normalizer, lineage roles, native base and fixed panels are pinned and never changed.
  The packet is never a CE teacher, never an optimizer update and never an admission.
- The root is new and create-only: packet.json, receipt.json and then overlay-complete.json (0444, O_EXCL, fsync).
  The dry run (every touched context at every rotation offset of its cumulative overlay set, through the real
  prepare_microbatches + collate) and the deadline / RSS limits are checked before any byte is written.
- Terminal qualification (the C10a r7 design): the build fixes ONE paired original bound when it starts (wall
  ``deadline``, the matching monotonic deadline, the boot id; sealed in the marker) and never resets it. The build
  NEVER certifies itself: it writes no positive terminal record. After the marker and both directories are durable it
  samples the bound once more (defense in depth): a crossed bound writes the create-only refusal record
  overlay-refused.json (best effort), keeps the late root for custody and raises. Admission requires the CALLER's
  bounded terminal acceptance envelope (``packet.accept_publication``), written outside the root only after this
  function RETURNED, from a post-return clock sample inside the original bound. Any exception (late, OSError after
  any internal write, refusal not writable) therefore leaves no envelope and no admissible packet, and a read while
  the build is running finds no envelope.
"""
import importlib
import os
import time
from collections import Counter, OrderedDict
from dataclasses import asdict

import native_inputs as NI
import neural_native_batch as NB
from dltrain import labels as L
from jope_flywheel import training_snapshot_native as N

from . import verify as V
from .common import (AUTHORITY_EVERY, CLASSES, DEPENDENCIES, FAILURE_STATES, MARKER_NAME, MARKER_SCHEMA, MAX_CONTEXTS,
                     MAX_PACKET_BYTES, MAX_READ_BYTES, MAX_ROWS, MAX_RSS_BYTES, MAX_SECONDS, MAX_SOURCE_ROWS,
                     OVERLAY_WINDOW, PACKET_SCHEMA, PENDING_STATES, RECEIPT_SCHEMA, REFUSED_NAME, ROTATION_DOMAIN,
                     TERMINAL_SCHEMA, VERIFIED_STATES, WORKER_SHA_PIN, artifact_refs, boot_id, canonical, clock_sample,
                     create_only, digest, exact_canonical, fsync_dir, hash64, in_bound, max_rss_bytes, module_sha, need,
                     ref5, safe_path, sha)
from .labels_overlay import (direct_recomputation, label_row, old_labels_equal, overlay_record, record_json,
                             union_labels)
from .loader import dry_run, normalizer_from, overlay_snapshot
from .packet import LIMITS, ChainView, head_sha256, overlay_programs_digest, validate_chain, validate_doc

BUILDER_MODULES = ('common', 'verify', 'labels_overlay', 'loader', 'packet', 'build')
_END = object()


def builder_pins():
    pkg = __name__.rsplit('.', 1)[0]
    return {name: module_sha(importlib.import_module(pkg + '.' + name)) for name in BUILDER_MODULES}


def _sources(sources):
    need(type(sources) is list and sources, 'declared outbox sources')
    out = OrderedDict()
    for s in sources:
        need(type(s) is dict and set(s) == {'outbox_kind', 'backup_sha256', 'event_head'} and
             type(s['outbox_kind']) is str and s['outbox_kind'] and s['outbox_kind'] not in out and
             hash64(s['backup_sha256']) and type(s['event_head']) is dict and
             set(s['event_head']) == {'events', 'head_sha256'} and type(s['event_head']['events']) is int and
             hash64(s['event_head']['head_sha256']), 'closed declared source {outbox_kind, backup_sha256, event_head}')
        out[s['outbox_kind']] = s
    return out


def _refuse(root, terminal, stage, sample):
    """The original bound was crossed during the final publication: keep the late root for custody, write the
    create-only refusal record (best effort; consumers honour it, but admission never depends on it), and raise."""
    why = ('overlay build deadline crossed ' + stage.replace('_', ' ') +
           ' (late publication refused; root kept for custody)')
    try:
        create_only(root / REFUSED_NAME,
                    canonical({**terminal, 'verdict': 'refused', 'stage': stage, 'sample': sample}))
        fsync_dir(root)
    except OSError as error:
        raise ValueError(why + '; refusal record not written: ' + str(error)) from error
    raise ValueError(why)


def build_packet(rows_iter, *, native, store, authority, inputs, canonicalize, native_spec_bytes, preprocessing_bytes,
                 preprocessing_sha256, registry_bytes, objective_sha256, source_pins, r6_roster_bytes, r6_roster_sha256,
                 campaigns, sources, max_padded_tokens, root, deadline, predecessor_chain=None, predecessor_head=None,
                 max_read_bytes=MAX_READ_BYTES):
    """Verify, label, dry-run and publish delta packet k against the validated chain 1..k-1; returns the receipt.

    ``predecessor_chain`` is None (genesis) or the ordered packet refs ``[{path, sha256, bytes}, ...]`` of the
    chain ending at ``predecessor_head``; it is re-validated in full (``validate_chain``) before any row is read.
    ``preprocessing_sha256`` and ``objective_sha256`` are the campaign contract's normalizer/objective pins;
    ``native`` is the base snapshot the learner will wrap (NativeSnapshot or a composite exposing ``.native``);
    ``store.get_version(ref) -> (bytes, version_id)`` with ``store.nativecanonical_json``; ``authority() -> bytes``;
    ``inputs.describe(descriptor, phase)`` is NativeInputs or an equivalent; ``canonicalize(raw) -> canonical DSL``.
    """
    boot0, mono0 = boot_id(), time.monotonic()   # BEFORE wall: a delay can only shrink the monotonic allowance
    started, cpu0 = time.time(), os.times()
    need(type(deadline) in (int, float) and started < deadline <= started + MAX_SECONDS, 'bounded build window (<= 1800 s)')
    # The ONE original paired bound of this build (never reset); sealed in the marker and the terminal record.
    bound = {'wall_deadline_unix': deadline, 'monotonic_deadline': mono0 + (deadline - started), 'boot_id': boot0}
    root = safe_path(root)
    need(root.parent.is_dir() and not root.exists(), 'new create-only overlay root')
    nat = getattr(native, 'native', native)
    # ---- pins: exact original native base, normalizer, objective, lineage registry and source closure
    need(sha(native_spec_bytes) == nat.identity, 'native spec bytes are the loaded base snapshot')
    spec = N.document(native_spec_bytes)
    need(type(source_pins) is dict and set(source_pins) == set(DEPENDENCIES), 'closed dependency source pins')
    for name in DEPENDENCIES:
        need(module_sha(importlib.import_module(name)) == source_pins[name], 'dependency source drift: ' + name)
    for name in N.MODULES:
        need(spec['consumer_source_sha256'][name] == source_pins[name], 'native consumer pins equal: ' + name)
    need(NI.WORKER_SHA == WORKER_SHA_PIN, 'pinned native worker sha')
    need(digest(NB.OBJECTIVE) == objective_sha256, 'registered L6 objective identity unchanged')
    need(hash64(preprocessing_sha256) and sha(preprocessing_bytes) == preprocessing_sha256,
         'preserved native train-only normalizer bytes (campaign normalizer_sha256)')
    pre = N.document(preprocessing_bytes)
    train_ids = sorted(c for c, r in nat.contexts.items() if r['partition'] == 'train')
    need(pre.get('schema') == 'jope.native-kumos-preprocessing/v1' and pre.get('snapshot_sha256') == nat.identity and
         pre.get('fit_context_ids') == train_ids and pre.get('validation_used_for_fit') is False and
         len(pre.get('mean', ())) == len(pre.get('scale', ())) == N.DIM and pre.get('clip') == [-10., 10.],
         'preserved native train-only normalizer identity')
    fit_roster = frozenset(pre['fit_context_ids'])
    need(sha(registry_bytes) == spec['parent_registry']['sha256'], 'same full parent registry')
    parents = {}
    for line in registry_bytes.splitlines():
        if line.strip():
            p = N.document(line)
            parents[p['parent_lineage_sha256']] = p
    r6_contexts, r6_lineages = V.r6_roster(r6_roster_bytes, r6_roster_sha256)
    need(type(campaigns) in (list, tuple, set, frozenset) and campaigns and all(hash64(c) for c in campaigns),
         'campaign allow-list')
    campaign_set = frozenset(campaigns)
    srcmap = _sources(sources)
    pins = {'native_spec_sha256': nat.identity, 'preprocessing_sha256': sha(preprocessing_bytes),
            'parent_registry_sha256': sha(registry_bytes), 'objective_sha256': objective_sha256,
            'grammar_sha256': spec['grammar']['sha256'], 'vocab_sha256': spec['vocab_sha256'],
            'authority_pins_sha256': digest(spec['authority_pins']), 'input_lists': spec['input_lists'],
            'source_sha256': dict(source_pins), 'builder_source_sha256': builder_pins(),
            'worker_sha256': WORKER_SHA_PIN, 'r6_roster_sha256': r6_roster_sha256, 'overlay_window': OVERLAY_WINDOW,
            'rotation_domain': ROTATION_DOMAIN}
    # ---- predecessor chain 1..k-1, re-validated in full
    if predecessor_chain is None:
        need(predecessor_head is None, 'genesis has no predecessor head')
        view = ChainView()
    else:
        view = validate_chain(predecessor_chain, expected_head=predecessor_head)
        need(view.pins == pins, 'predecessor chain pins identical (same base, normalizer, objective, roles, sources)')
    prev_head, seq, carried = view.head_sha256, view.seq + 1, view.contexts
    # ---- authority, reader, frozen loader
    first_proof, first = V.check_authority(authority(), spec['authority_pins'], time.time())
    protected = (set(first_proof.get('protected_context_ids', [])), set(first_proof.get('protected_views', [])))
    reader = V.ExactReader(store, limit=max_read_bytes, deadline=deadline)
    loader = N.NativeContextLoader(nat)
    entries, stage, candidates, cache, shards, frozen, build_anchor = [], {}, [], {}, {}, {}, {}
    per_source, crosschecks, candidate_keys, candidate_contexts = Counter(), [], set(), set()
    stopped = None

    def put(entry, cls, reason, detail=None, row=None):
        entry.update({'class': cls, 'reason': reason, 'detail': None if detail is None else str(detail)[:500],
                      'row': row})

    iterator = iter(rows_iter)
    while True:
        # Per-packet caps: stop PULLING rows (the caller hands the same iterator to the next build).
        if len(entries) >= MAX_SOURCE_ROWS:
            stopped = 'SOURCE_ROW_CAP'
            break
        if len(candidate_keys) >= MAX_ROWS:
            stopped = 'NEW_ROW_CAP'
            break
        if len(candidate_contexts) >= MAX_CONTEXTS:
            stopped = 'CONTEXT_CAP'
            break
        row = next(iterator, _END)
        if row is _END:
            break
        need(time.time() < deadline, 'overlay build deadline')
        if entries and len(entries) % AUTHORITY_EVERY == 0:
            proof, _ = V.check_authority(authority(), spec['authority_pins'], time.time())
            protected = (set(proof.get('protected_context_ids', [])), set(proof.get('protected_views', [])))
        V.row_shape(row)
        need(row['outbox_kind'] in srcmap, 'row from a declared outbox source')
        obs = row['observation']
        if 'backup_sha256' in obs:
            need(obs['backup_sha256'] == srcmap[row['outbox_kind']]['backup_sha256'], 'row backup join')
        per_source[row['outbox_kind']] += 1
        entry = {'observation_sha256': digest(obs), 'outbox_kind': row['outbox_kind'],
                 'emission_id': obs.get('emission_id') if hash64(obs.get('emission_id')) else None,
                 'context_id': row['context_id'], 'phase': row['phase'], 'state': row['state'],
                 'evaluation_key': row['evaluation_key'] if hash64(row['evaluation_key']) else None}
        entries.append(entry)
        try:
            body = V.emission_body(row)
        except V.Refusal as r:
            put(entry, 'REFUSED', r.code, r.detail)
            continue
        kind = V.descriptor_of(body).get('source_kind')
        if kind == 'flywheel_variant':
            put(entry, 'ROUTED_TO_X1', 'VARIANT_SOURCE')
            continue
        if kind != 'native_s1_context':
            put(entry, 'REFUSED', 'UNSUPPORTED_SOURCE_KIND', repr(kind))
            continue
        try:
            eid, desc = V.emission_identity(row, body, canonicalize=canonicalize)
        except V.Refusal as r:
            put(entry, 'REFUSED', r.code, r.detail)
            continue
        entry['emission_id'] = eid
        if row['campaign'] not in campaign_set:
            put(entry, 'REFUSED', 'CAMPAIGN_NOT_ALLOWED')
            continue
        state = row['state']
        if state in FAILURE_STATES:
            put(entry, 'REFUSED', 'PARITY_UNQUALIFIED', 'R5 failure custody: no q, no feasibility')
            continue
        if state in PENDING_STATES:
            put(entry, 'PENDING', 'UNRESOLVED_STATE', state)
            continue
        need(state in VERIFIED_STATES, 'known outbox/ledger state: ' + state)
        cid = row['context_id']
        ctx = nat.contexts.get(cid)
        if ctx is None:
            put(entry, 'REFUSED', 'OUTSIDE_SNAPSHOT')
            continue
        if desc != ctx:
            put(entry, 'REFUSED', 'EMISSION_IDENTITY', 'input descriptor differs from the frozen snapshot')
            continue
        # Validation-exclusion guard: validation phase/partition and R6 rows are evaluation-only, never targets.
        if row['phase'] != 'training':
            put(entry, 'EVALUATION_ONLY', 'VALIDATION_PHASE')
            continue
        if ctx['partition'] != 'train':
            put(entry, 'EVALUATION_ONLY', 'VALIDATION_PARTITION')
            continue
        if cid in r6_contexts or ctx['parent_lineage_sha256'] in r6_lineages:
            put(entry, 'EVALUATION_ONLY', 'R6_ROSTER')
            continue
        lineage = parents.get(ctx['parent_lineage_sha256'])
        if cid not in fit_roster or lineage is None or lineage['partition'] != 'train':
            put(entry, 'REFUSED', 'OUTSIDE_FIT_ROSTER')
            continue
        if cid in protected[0] or ctx['view_id'] in protected[1]:
            put(entry, 'REFUSED', 'PROTECTED_CONTEXT')
            continue
        try:
            sig = digest({'record': ref5(row['record_ref']), 'artifacts': artifact_refs(row['artifact_refs'])})
        except ValueError as error:
            put(entry, 'REFUSED', 'VERIFICATION_FAILED', error)
            continue
        key = row['evaluation_key']
        stage[len(entries) - 1] = (key, sig)
        try:
            if (key, sig) not in cache:
                try:
                    cache[(key, sig)] = V.verify_result(row, desc, nat=nat, inputs=inputs, reader=reader)
                except (V.Refusal, V.Pending) as error:
                    cache[(key, sig)] = error
            facts = cache[(key, sig)]
            if isinstance(facts, Exception):
                raise facts
            if cid not in shards:
                shards[cid] = V.shard_join(cid, nat=nat, loader=loader, facts=facts)
            shard = shards[cid]
            anchored, source = V.anchor(shard=shard, carried=carried.get(cid, build_anchor.get(cid)), facts=facts)
            if cid not in frozen:
                frozen[cid] = V.frozen_baseline(cid, loader=loader, shard=shard)
        except V.Refusal as r:
            put(entry, 'REFUSED', r.code, r.detail)
            continue
        except V.Pending as p:
            put(entry, 'PENDING', p.code, p.detail)
            continue
        if cid not in carried and cid not in build_anchor:
            build_anchor[cid] = {'anchored_baseline': anchored, 'y_train_sha256': facts['score'].target_sha256,
                                 'split_sha256': facts['score'].split_sha256}
        if shard is not None and facts['program_sha256'] in shard['programs']:
            old = next(r for r in shard['doc']['records'] if r['program_sha256'] == facts['program_sha256'])
            agrees = V.agree(old.get('rmse_j'), facts['rmse'])
            check = {'program_sha256': facts['program_sha256'], 'shard_rmse_j': old.get('rmse_j'),
                     'verified_rmse_j': facts['rmse'], 'agrees': agrees}
            crosschecks.append({'context_id': cid, **check})
            # Shard-RMSE guard: a shard duplicate must reproduce the shard row's per-fold RMSE (rel 1e-10, abs 1e-12).
            if agrees:
                put(entry, 'TRAIN_VERIFIED', 'ALREADY_IN_SHARD', canonical(check).decode())
            else:
                put(entry, 'REFUSED', 'SHARD_RMSE_DISAGREEMENT', canonical(check).decode())
            continue
        if key in view.keys:
            prev_seq, prev_index, prev_sig = view.keys[key]
            if prev_sig == sig:
                put(entry, 'TRAIN_VERIFIED', 'ALREADY_IN_OVERLAY', row=[prev_seq, prev_index])
            else:
                put(entry, 'REFUSED', 'CONFLICTING_REFS', 'different refs than the existing chain row')
            continue
        candidates.append({'entry': len(entries) - 1, 'cid': cid, 'key': key, 'sig': sig, 'facts': facts,
                           'anchored': anchored, 'source': source, 'shard': shard, 'program': row['program'],
                           'lineage': ctx['parent_lineage_sha256'], 'eid': eid, 'obs': entry['observation_sha256'],
                           'descriptor': ctx})
        candidate_keys.add(key)
        candidate_contexts.add(cid)
    # ---- final authority, conflicting refs, earliest emission wins
    final_proof, final = V.check_authority(authority(), spec['authority_pins'], time.time())
    final_protected = (set(final_proof.get('protected_context_ids', [])), set(final_proof.get('protected_views', [])))
    sigs = {}
    for key, sig in stage.values():
        sigs.setdefault(key, set()).add(sig)
    conflicted = {k for k, v in sigs.items() if len(v) > 1}
    for i, (key, _) in stage.items():
        if key in conflicted:
            put(entries[i], 'REFUSED', 'CONFLICTING_REFS', 'distinct result refs for one evaluation key')
    groups = OrderedDict()
    for c in candidates:
        if c['key'] in conflicted:
            continue
        if c['cid'] in final_protected[0] or c['descriptor']['view_id'] in final_protected[1]:
            put(entries[c['entry']], 'REFUSED', 'PROTECTED_CONTEXT', 'protected at the final authority proof')
            continue
        groups.setdefault(c['key'], []).append(c)
    # ---- labels on the union (shard + ALL earlier chain rows + own rows), old-label equality, direct recomputation
    by_context = OrderedDict()
    for key, members in groups.items():
        by_context.setdefault(members[0]['cid'], []).append(members)
    new_rows, contexts, unions, recomputation, old_label_checks = [], {}, {}, [], {}
    for cid, keyed in by_context.items():
        first_c = keyed[0][0]
        facts, shard, anchored = first_c['facts'], first_c['shard'], first_c['anchored']
        if shard is not None:
            label_doc, shard_rows = shard['doc']['context'], shard['doc']['records']
        else:
            label_doc = {'context_sha256': cid, 'y_train': facts['y'].tolist(),
                         'dataset_hash': first_c['descriptor']['dataset_hash'], 'family_id': first_c['lineage'],
                         'split_role': 'train_fit'}
            shard_rows = []
        earlier = view.context_rows(cid)
        earlier_label_rows = [label_row(cid, r.program_sha256, r.record.tokens, r.record.skeleton_sha256, r.rmse,
                                        r.anchored, r.wall_us) for r in earlier]
        fresh = [label_row(cid, m[0]['facts']['program_sha256'], m[0]['facts']['tokens'],
                           m[0]['facts']['skeleton_sha256'], m[0]['facts']['rmse'], m[0]['anchored'],
                           m[0]['facts']['wall_us']) for m in keyed]
        refusal = None
        try:
            context_label, union = union_labels(label_doc, shard_rows, earlier_label_rows + fresh)
            if shard is not None:
                same = old_labels_equal(shard['labels'], {p: union[p] for p in shard['labels']})
                old_label_checks[cid] = {'shard_rows': len(shard['labels']), 'union_equals_shard_only': same}
                if not same:
                    refusal = ('OLD_LABELS_WOULD_CHANGE', 'union labels would change shard-row labels')
        except L.LabelError as error:
            refusal = ('LABEL_ERROR', error)
        if refusal is not None:
            for members in keyed:
                for c in members:
                    put(entries[c['entry']], 'REFUSED', refusal[0], refusal[1])
            continue
        for r in earlier:
            lab = union[r.program_sha256]
            need(record_json(overlay_record(lab, r.record.tokens, r.record.skeleton_sha256)) == record_json(r.record),
                 'earlier chain record labels drifted')
        y_ctx = label_doc['y_train']
        for members in keyed:
            c = members[0]
            f = c['facts']
            rec = overlay_record(union[f['program_sha256']], f['tokens'], f['skeleton_sha256'])
            check = direct_recomputation(y_ctx, f['rmse'], c['anchored'], f['wall_us'], rec)
            need(check['equal'], 'overlay label differs from its direct recomputation')
            recomputation.append({'context_id': cid, **check})
            index = len(new_rows)
            for j, m in enumerate(members):
                put(entries[m['entry']], 'TRAIN_VERIFIED', 'NEW_OVERLAY_ROW' if j == 0 else 'ALIAS', row=index)
            new_rows.append({'context_id': cid, 'parent_lineage_sha256': c['lineage'], 'partition': 'train',
                             'evaluation_key': c['key'], 'program': c['program'], 'program_sha256': f['program_sha256'],
                             'result_ref': f['record_ref'], 'artifact_refs': f['artifact_refs'],
                             'original_score_context': asdict(f['score']), 'inner_fold_rmse': f['rmse'],
                             'wall_us': f['wall_us'],
                             'baseline': {'reconstructed': f['reconstructed'], 'evaluation': f['evaluation_baseline'],
                                          'anchored': c['anchored'], 'source': c['source']},
                             'record': record_json(rec), 'shard_ref': None if shard is None else shard['ref'],
                             'source_emission_ids': [m['eid'] for m in members],
                             'source_observations': [m['obs'] for m in members], 'packet_seq': seq})
        own = [m[0]['facts']['program_sha256'] for m in keyed]
        unions[cid] = {'shard_rows': len(shard_rows), 'earlier_rows': len(earlier), 'packet_rows': len(own),
                       'overlay_programs_sha256': overlay_programs_digest([r.program_sha256 for r in earlier] + own)}
        if cid not in carried:
            ctx_label = {k: label_doc[k] for k in ('context_sha256', 'dataset_hash', 'family_id', 'split_role')}
            contexts[cid] = {'anchored_baseline': anchored,
                             'baseline_source': 'shard' if shard is not None and shard['pair'] is not None
                             else 'reconstructed', 'y_train_sha256': facts['score'].target_sha256,
                             'split_sha256': facts['score'].split_sha256, 'label_context': ctx_label,
                             'shard_ref': None if shard is None else shard['ref'], 'shard_rows': len(shard_rows),
                             'first_seq': seq, 'parent_lineage_sha256': first_c['lineage']}
    need(len(new_rows) <= MAX_ROWS and len(unions) <= MAX_CONTEXTS, 'delta packet row/context limits')
    need(all('class' in e for e in entries), 'every consumed row is censused')
    counts = Counter(e['class'] for e in entries)
    census = {'source_rows': len(entries), 'classes': {k: counts.get(k, 0) for k in CLASSES}, 'entries': entries}
    new_ce = sum(r['record']['ce_eligible'] for r in new_rows)
    packet = {'schema': PACKET_SCHEMA, 'seq': seq, 'predecessor_overlay_head_sha256': prev_head,
              'chain': {'packets_before': seq - 1, 'rows_before': len(view.rows)}, 'pins': pins,
              'campaigns': sorted(campaign_set),
              'sources': [{**s, 'rows': per_source.get(k, 0)} for k, s in srcmap.items()],
              'authority_proofs': {'first': first, 'final': final}, 'rows': new_rows, 'contexts': contexts,
              'label_unions': unions, 'census': census, 'limits': dict(LIMITS), 'stopped': stopped,
              'new_ce_teachers': new_ce, 'optimizer_updates': 0, 'admission_committed': False}
    raw = canonical(packet)
    need(len(raw) <= MAX_PACKET_BYTES, 'overlay packet <= 32 MiB')
    parsed = validate_doc(exact_canonical(raw))
    head = head_sha256(raw, prev_head)
    # ---- the whole-chain checks of validate_chain, on the cumulative view k (the predecessor view is extended)
    view.add(parsed, packet_sha256=sha(raw), packet_bytes=len(raw), path=str(root / 'packet.json'))
    need(view.head_sha256 == head, 'extended chain head')
    # ---- pre-publication dry run through the real prepare_microbatches + collate (CPU)
    dry = {'schema': 'jope.native-overlay-dry-run/v1', 'contexts': 0, 'expected_runs': 0, 'runs': 0, 'passed': True,
           'never_selected': {}, 'combined_batches': [], 'detail': [], 'max_padded_tokens': max_padded_tokens}
    if unions:
        s = overlay_snapshot(native, view, max_padded_tokens=max_padded_tokens, contexts=sorted(unions))
        dry = dry_run(s, normalizer=normalizer_from(pre), max_padded_tokens=max_padded_tokens, deadline=deadline,
                      contexts=sorted(unions))
    need(dry['passed'], 'overlay dry run at every rotation offset')
    rss = max_rss_bytes()
    need(rss <= MAX_RSS_BYTES, 'overlay build RSS <= 2 GiB')
    need(in_bound(clock_sample(), bound), 'overlay build deadline before publication')
    # ---- create-only publication; the success marker is the last positive record the build writes
    root.mkdir(mode=0o755)
    packet_ref = create_only(root / 'packet.json', raw)
    cpu1 = os.times()
    receipt = {'schema': RECEIPT_SCHEMA, 'packet': packet_ref, 'head_sha256': head,
               'predecessor_overlay_head_sha256': prev_head, 'seq': seq, 'new_rows': len(new_rows),
               'chain': {'packets': view.seq, 'rows': len(view.rows), 'contexts': len(view.contexts),
                         'packets_refs': view.packets},
               'touched_contexts': len(unions), 'first_seen_contexts': len(contexts), 'stopped': stopped,
               'source_rows': len(entries), 'census_classes': census['classes'],
               'direct_recomputation': {'rows': len(recomputation), 'all_equal': all(x['equal'] for x in recomputation),
                                        'tolerance': 'exact ints and grids; floats <= 1e-12 relative',
                                        'detail': recomputation},
               'old_labels_unchanged': old_label_checks, 'shard_rmse_crosschecks': crosschecks,
               'frozen_baseline_verified': {c: v for c, v in frozen.items()}, 'dry_run': dry,
               'reads': {'exact_version_gets': reader.gets, 'bytes': reader.read_bytes},
               'authority_proofs': {'first': first, 'final': final}, 'started_unix': started,
               'finished_unix': time.time(), 'cpu_user_seconds': cpu1.user - cpu0.user,
               'cpu_system_seconds': cpu1.system - cpu0.system, 'max_rss_bytes': rss, 'limits': dict(LIMITS),
               'outer_test_values_decoded': False, 'new_ce_teachers': new_ce, 'optimizer_updates': 0,
               'admission_committed': False,
               'scope': 'Train-only measured overlay targets; no optimizer, planner, outbox, S3 write, GPU or GP.'}
    need(in_bound(clock_sample(), bound), 'overlay build deadline before receipt')
    receipt_ref = create_only(root / 'receipt.json', canonical(receipt))
    need(in_bound(clock_sample(), bound), 'overlay build deadline before success marker')
    marker_ref = create_only(root / MARKER_NAME,
                             canonical({'schema': MARKER_SCHEMA, 'packet': packet_ref, 'receipt': receipt_ref,
                                        'head_sha256': head, 'predecessor_overlay_head_sha256': prev_head,
                                        'seq': seq, 'bound': bound}))
    fsync_dir(root)
    fsync_dir(root.parent)
    # ---- post-durable recheck of the UNCHANGED original bound (defense in depth; NOT a certification -- nothing
    # positive is written after it, and only the caller's post-return acceptance envelope admits the packet)
    after_marker = clock_sample()
    if not in_bound(after_marker, bound):
        _refuse(root, {'schema': TERMINAL_SCHEMA, 'root': str(root), 'packet': packet_ref, 'receipt': receipt_ref,
                       'marker': marker_ref, 'head_sha256': head, 'predecessor_overlay_head_sha256': prev_head,
                       'seq': seq, 'bound': bound}, 'after_durable_marker', after_marker)
    return {**receipt, 'receipt': receipt_ref, 'marker': marker_ref}


__all__ = ['build_packet', 'builder_pins']
