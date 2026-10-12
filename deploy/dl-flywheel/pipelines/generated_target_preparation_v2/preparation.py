"""C10a revision 7 (review of r5: every marker bound is the sealed receipt clock bound, strict and finite; the worker
re-qualifies after its FINAL marker publication; an operator acceptance envelope written after the worker terminated
pins the exact marker bytes, the unchanged bound and physical in-time success, and every consumer requires it; a
qualification sample re-reads monotonic and boot after the wall clock).  Revision 5 (revision 3/4 + review fixes: monotonic and boot sampled before wall; success qualified after
the durable status by a terminal marker; revision 3: paired clock checked at every read; one combined read budget charged
before every read): verified X1 train-target references -> one immutable local prepared-candidate root.

Cold successor of the HELD C10 v1 packet (DELIVERY 3290153b; v1 bytes preserved unchanged).  It closes the
independent review findings A1-A3 (CODEX_C10_INDEPENDENT_FINDINGS_20261009_v1.md):

* A1  every newly appended extension reference maps to exactly one manifest row and every row to exactly one
      appended reference (a bijection; references are compared only after normalizing the two established
      five/seven-field shapes, while the original ordered bytes stay untouched), and every composed record or
      refusal for a new reference reconciles exactly once;
* A2  the X1 wire contract is bound to an exact expected production contract: objective bytes and coefficients,
      unchanged normalizer bytes, the KumoSmall 2051 roster and weight, the actually imported labels/compose
      source bytes, the exporter source packet, the accepted X1 request/cutoff and both authority proofs;
* A3  one caller work clock and one read budget span composition, direct recomputation, hydration and every root
      write; the store must attest closed exact-version local reads with audited counters, and only a measured
      zero network count can produce the frozen zero-network preparation status.

Input is the raw ``jope.generated-train-target-reference-manifest/v1`` bytes from
``generated_target_reference_export_v2.export_verified.export_targets`` (references only).  Output is exactly the
preparation root that ``prepared_admission.candidate`` reads (``config.json``, ``extension.json``,
``prepared.json``, ``status.json`` and byte copies of the owned hydration sources) plus a pinned
``c10a-receipt.json`` holding the contract, the bijection and the row-level label evidence.  ``status.json`` is
written last and is the only success marker.  Numeric records are created only by the frozen
``native_variant_snapshot.compose`` and serialized/hydrated only by ``prepared_snapshot.prepare``/``load``.
Nothing here touches an optimizer, planner, outbox, ready pointer, S3 or GPU, and nothing executes a program.
"""
import json
import math
import os
import time
from pathlib import Path

from pipelines.generated_target_admission_v1 import export_verified as X1
from pipelines.generated_target_admission_v1.common import canonical, digest, document, need, reference, sha
from pipelines.generated_target_admission_v1.transition import validate_extension_transition
from jope_flywheel import native_variant_snapshot as V
from jope_flywheel import training_snapshot_kumos as K
from jope_flywheel.protocol import Context
import prepared_snapshot as P

MANIFEST_SCHEMA = 'jope.generated-train-target-reference-manifest/v1'
MANIFEST_KEYS = frozenset({
    'schema', 'request_sha256', 'cutoff_sha256', 'cutoff_head', 'predecessor_sha256', 'extension',
    'extension_sha256', 'objective_sha256', 'normalizer_sha256', 'construction_sources', 'roster',
    'authority_proofs', 'rows', 'census', 'counts', 'artifact_reads', 'numeric_targets_materialized',
    'target_recipe', 'variant_ce_teachers', 'optimizer_updates', 'admission_committed', 's3_writes',
    'gp_executions', 'source_optimizer_commits_verified', 'state'})
MANIFEST_FIXED = {
    'numeric_targets_materialized': False, 'target_recipe': 'pinned-existing-compose-and-labels-only',
    'variant_ce_teachers': 0, 'optimizer_updates': 0, 'admission_committed': False, 's3_writes': 0,
    'gp_executions': 0, 'source_optimizer_commits_verified': False, 'state': 'verified_reference_export_only'}
ROW_KEYS = frozenset({
    'context_id', 'parent_lineage_sha256', 'partition', 'evaluation_key', 'program_sha256', 'result_ref',
    'artifact_refs', 'measurement', 'ce_eligible', 'source_emission_ids', 'source_batch',
    'source_optimizer_commit_verified'})
CENSUS = ('appended', 'duplicates', 'validation', 'pending', 'refused')
ARTIFACTS = ('input', 'program', 'evaluation', 'rust_source', 'rust_executable', 'rust_parity', 'score_records')
CONFIG_SCHEMA = 'jope.prepared-mixed-snapshot-config/v1'
STATUS_SCHEMA = 'jope.prepared-snapshot-receipt/v1'
RECEIPT_SCHEMA = 'jope.generated-train-target-preparation/v3'
CONTRACT_SCHEMA = 'jope.c10-production-target-contract/v1'
CONTRACT_KEYS = frozenset({'schema', 'objective_sha256', 'normalizer_sha256', 'roster', 'construction_sources',
                           'x1_source_packet_sha256'})
PROVENANCE_KEYS = frozenset({'request_sha256', 'cutoff_sha256', 'x1_source_packet_sha256'})
STORE_AUDIT = 'jope.closed-exact-version-store-audit/v1'
ROSTER = {'name': 'kumo-s-only-v1', 'dimension': 2051, 'weight_sha256': X1.KUMO_WEIGHT_SHA256}
OBJECTIVE_NAME = ('jope.flywheel-native-support-objective/v1', 'native-support-mask-v1')
MAX_MANIFEST = 32 * 1024**2
RESERVED = frozenset({'config.json', 'status.json', 'prepared.json', 'extension.json', 'c10a-receipt.json',
                      'c10a-terminal.json'})
TERMINAL_SCHEMA = 'jope.generated-train-target-preparation-terminal/v1'
TERMINAL_KEYS = frozenset({'schema', 'status_sha256', 'receipt_sha256', 'qualified_after_durable_status', 'bound'})
BOUND_KEYS = frozenset({'wall_deadline_unix', 'monotonic_deadline', 'boot_id'})
SAMPLE_KEYS = frozenset({'boot_id', 'monotonic', 'wall_unix'})
ACCEPTANCE_SCHEMA = 'jope.c10a-bounded-terminal-acceptance/v1'
ACCEPTANCE_KEYS = frozenset({'schema', 'subject', 'root', 'artifacts', 'marker', 'bound', 'physical',
                             'post_publication_sample', 'verdict', 'reason'})
PHYSICAL_KEYS = {'systemd_unit': frozenset({'kind', 'unit', 'result', 'exec_main_status', 'exit_monotonic_usec',
                                            'boot_id'}),
                 'in_process_return': frozenset({'kind', 'returned_sha256'})}
MAX_MARKER = 1024 * 1024


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def check_bound(b):
    """A strict finite paired (wall, monotonic, boot) bound."""
    need(type(b) is dict and set(b) == BOUND_KEYS and _finite(b['wall_deadline_unix']) and
         _finite(b['monotonic_deadline']) and type(b['boot_id']) is str and 0 < len(b['boot_id']) <= 64,
         'strict finite paired bound')
    return b


def sealed_bound(clock_doc):
    """The original bound sealed in a receipt/completion ``clock`` record (its stages are not part of the bound)."""
    need(type(clock_doc) is dict and BOUND_KEYS <= set(clock_doc), 'sealed clock record')
    return check_bound({k: clock_doc[k] for k in BOUND_KEYS})


def in_bound(sample, bound):
    """True iff a strict clock sample lies strictly inside the bound on the same boot."""
    need(type(sample) is dict and set(sample) == SAMPLE_KEYS and _finite(sample['monotonic']) and
         _finite(sample['wall_unix']) and type(sample['boot_id']) is str, 'strict clock sample')
    return (sample['boot_id'] == bound['boot_id'] and sample['wall_unix'] < bound['wall_deadline_unix'] and
            sample['monotonic'] < bound['monotonic_deadline'])


def check_terminal(terminal_raw, *, status_raw, receipt_raw):
    """A root is successful only with a terminal marker whose bound IS the sealed receipt clock bound and whose
    post-durable-status sample lies inside it."""
    t = json.loads(terminal_raw)
    need(type(t) is dict and set(t) == TERMINAL_KEYS and t['schema'] == TERMINAL_SCHEMA and
         t['status_sha256'] == sha(status_raw) and t['receipt_sha256'] == sha(receipt_raw),
         'terminal marker binds this status and receipt')
    need(check_bound(t['bound']) == sealed_bound(json.loads(receipt_raw).get('clock')),
         'terminal bound is the sealed receipt clock bound')
    need(in_bound(t['qualified_after_durable_status'], t['bound']),
         'preparation success qualified inside its original bound')
    return t


def _boot_id():
    with open('/proc/sys/kernel/random/boot_id') as f:
        return f.read().strip()


def sample_now():
    """Boot, monotonic, wall, then a FRESH monotonic and boot after the wall sample: a stall while sampling the wall
    clock (plus any backward wall adjustment) cannot hide an elapsed monotonic bound or a reboot."""
    boot0, mono0 = _boot_id(), time.monotonic()
    wall = time.time()
    mono, boot = time.monotonic(), _boot_id()
    return {'boot_id': boot if boot == boot0 else 'boot-changed-while-sampling', 'monotonic': max(mono, mono0),
            'wall_unix': wall}


class Clock:
    """One paired work clock: wall deadline, the matching monotonic deadline and the boot identity, all checked at
    every stage and before and after every artifact read against the same original bound (never reset)."""

    def __init__(self, deadline):
        boot = _boot_id()
        mono = time.monotonic()      # sampled BEFORE wall: any delay can only shrink the monotonic allowance
        wall = time.time()
        need(type(deadline) in (int, float) and math.isfinite(deadline) and deadline > wall, 'future C10a work deadline')
        self.deadline, self.mono_deadline, self.boot_id = deadline, mono + (deadline - wall), boot
        self.started, self.stages = mono, []

    def check(self, stage):
        need(time.time() < self.deadline and time.monotonic() < self.mono_deadline and _boot_id() == self.boot_id,
             'C10a work deadline expired at ' + stage)

    def tick(self, stage):
        self.check(stage)
        self.stages.append({'stage': stage, 'elapsed_seconds': time.monotonic() - self.started})

    def qualify(self, stage):
        """One conservative sample (``sample_now``) that must lie inside the original bound; returned as evidence."""
        sample = sample_now()
        need(in_bound(sample, self.bound()), 'C10a work deadline expired at ' + stage)
        return sample

    def bound(self):
        return {'wall_deadline_unix': self.deadline, 'monotonic_deadline': self.mono_deadline, 'boot_id': self.boot_id}


def _read_exact(ref, limit=MAX_MARKER):
    need(type(ref) is dict and set(ref) == {'path', 'sha256', 'bytes'} and type(ref['path']) is str and
         type(ref['bytes']) is int and 0 < ref['bytes'] <= limit, 'exact bounded reference')
    fd = os.open(ref['path'], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        body = os.read(fd, ref['bytes'] + 1)
    finally:
        os.close(fd)
    need(len(body) == ref['bytes'] and sha(body) == ref['sha256'], 'exact pinned bytes: ' + ref['path'])
    return body


def _file_ref(path):
    body = Path(path).read_bytes()
    return {'path': str(path), 'sha256': sha(body), 'bytes': len(body)}, body


def physical_ok(physical, bound, post, returned_sha256):
    """Physical in-time success of the terminated worker against the ORIGINAL bound.

    ``systemd_unit``: the unit's own terminal record (Result=success, ExecMainStatus=0, ExecMainExitTimestampMonotonic
    on the same boot strictly before the monotonic deadline).  ``in_process_return``: the call returned exactly the
    sealed result and the operator's post-return sample (taken after the final marker publication) is in bound."""
    need(type(physical) is dict and physical.get('kind') in PHYSICAL_KEYS and
         set(physical) == PHYSICAL_KEYS[physical['kind']], 'closed physical terminal evidence')
    if physical['kind'] == 'systemd_unit':
        usec = physical['exit_monotonic_usec']
        return (physical['result'] == 'success' and physical['exec_main_status'] == 0 and type(usec) is int and
                0 < usec and usec / 1e6 < bound['monotonic_deadline'] and physical['boot_id'] == bound['boot_id'])
    return physical['returned_sha256'] == returned_sha256 and in_bound(post, bound)


def accept_subject(subject, root, *, marker_name, artifact_names, marker_bound, returned, physical, acceptance_path):
    """OPERATOR-ONLY, after the worker has returned or its unit has terminated: seal one create-only acceptance
    envelope (outside the subject root) pinning the exact final marker and artifact bytes, the unchanged original
    bound, the physical terminal evidence and a fresh post-publication sample.  A failed predicate is written as a
    ``refused`` envelope (custody only; no artifact is deleted, no window is reset) and then raises."""
    root, out = Path(root), Path(acceptance_path)
    need(root.is_absolute() and out.is_absolute() and root != out and root not in out.parents and
         out.parent.is_dir() and not out.exists(), 'create-only acceptance outside the subject root')
    post = sample_now()
    doc = {'schema': ACCEPTANCE_SCHEMA, 'subject': subject, 'root': str(root), 'artifacts': {}, 'marker': None,
           'bound': None, 'physical': physical, 'post_publication_sample': post, 'verdict': 'refused', 'reason': None}
    try:
        raws = {}
        for name in artifact_names:
            doc['artifacts'][name], raws[name] = _file_ref(root / name)
        doc['marker'], marker_raw = _file_ref(root / marker_name)
        doc['bound'] = marker_bound(marker_raw, raws)
        need(physical_ok(physical, doc['bound'], post, returned(raws)),
             'physical terminal evidence is not an in-time success inside the original bound')
        doc['verdict'], doc['reason'] = 'accepted', 'in-time physical success after the final marker publication'
    except (ValueError, OSError, KeyError, TypeError) as exc:
        doc['reason'] = (type(exc).__name__ + ': ' + str(exc))[:300]
    raw = canonical(doc)
    ref = _create(out, raw)
    _fsync_dir(out.parent)
    need(doc['verdict'] == 'accepted', subject + ' acceptance refused (kept as custody): ' + doc['reason'])
    return ref


def check_acceptance(ref, *, subject, root, marker_name, marker_raw, artifacts, bound, returned_sha256):
    """Consumer side: the exact accepted envelope for this subject, its marker bytes, artifacts and original bound."""
    a = json.loads(_read_exact(ref))
    root = Path(root)
    need(type(a) is dict and set(a) == ACCEPTANCE_KEYS and a['schema'] == ACCEPTANCE_SCHEMA and
         a['verdict'] == 'accepted' and a['subject'] == subject and a['root'] == str(root),
         'accepted bounded terminal envelope for this subject')
    need(a['marker'] == {'path': str(root / marker_name), 'sha256': sha(marker_raw), 'bytes': len(marker_raw)},
         'acceptance pins the exact final marker bytes')
    need(a['artifacts'] == {n: {'path': str(root / n), 'sha256': sha(r), 'bytes': len(r)} for n, r in artifacts.items()},
         'acceptance pins the exact sealed artifacts')
    need(a['bound'] == bound, 'acceptance carries the unchanged original bound')
    need(physical_ok(a['physical'], bound, a['post_publication_sample'], returned_sha256),
         'acceptance physical terminal evidence inside the original bound')
    return a


def accept_root(root, *, acceptance_path, physical):
    """OPERATOR-ONLY: accept one finished preparation root (see ``accept_subject``)."""
    return accept_subject(
        'prepared_root', root, marker_name='c10a-terminal.json', artifact_names=('status.json', 'c10a-receipt.json'),
        marker_bound=lambda m, raws: check_terminal(m, status_raw=raws['status.json'],
                                                    receipt_raw=raws['c10a-receipt.json'])['bound'],
        returned=lambda raws: sha(raws['c10a-receipt.json']), physical=physical, acceptance_path=acceptance_path)


def check_root_acceptance(ref, *, root, status_raw, receipt_raw, terminal_raw):
    bound = check_terminal(terminal_raw, status_raw=status_raw, receipt_raw=receipt_raw)['bound']
    return check_acceptance(ref, subject='prepared_root', root=root, marker_name='c10a-terminal.json',
                            marker_raw=terminal_raw, artifacts={'status.json': status_raw, 'c10a-receipt.json': receipt_raw},
                            bound=bound, returned_sha256=sha(receipt_raw))


class BudgetStore:
    """Every artifact read of the whole preparation (identity, composition, recomputation) passes through here: the
    paired clock is checked and the declared object size is charged to ONE combined budget BEFORE the read, so the
    budget refuses before any overshoot; the clock is checked again after the read."""

    def __init__(self, inner, *, budget, clock):
        need(type(budget) is int and 0 < budget <= 8 * 1024**3, 'bounded read budget')
        self.inner, self.budget, self.clock, self.charged, self.reads = inner, budget, clock, 0, 0
        self.nativecanonical_json = getattr(inner, 'nativecanonical_json', None)

    def get_version(self, ref):
        r = K.normalized_ref(ref)
        self.clock.check('before read')
        need(self.charged + r['bytes'] <= self.budget, 'combined read budget would be exceeded')
        self.charged += r['bytes']
        self.reads += 1
        out = self.inner.get_version(ref)
        self.clock.check('after read')
        return out


def store_audit(store):
    """The store must attest closed exact-version local reads with monotone audited counters."""
    audit = getattr(store, 'audit', None)
    need(callable(audit), 'store must attest closed exact-version local reads (audit)')
    a = audit()
    need(type(a) is dict and a.get('schema') == STORE_AUDIT and a.get('closed_exact_version_local') is True and
         all(type(a.get(k)) is int and a[k] >= 0 for k in ('network_gets', 'local_reads', 'local_bytes')),
         'closed exact-version local store audit')
    return {k: a[k] for k in ('network_gets', 'local_reads', 'local_bytes')}


def parse_manifest(raw, *, expected_sha256):
    """Strictly re-validate the X1 wire contract; returns the decoded manifest."""
    need(type(raw) is bytes and len(raw) <= MAX_MANIFEST, 'bounded X1 manifest bytes')
    need(sha(raw) == expected_sha256, 'pinned raw X1 manifest digest')
    m = document(raw, MAX_MANIFEST)
    need(type(m) is dict and set(m) == MANIFEST_KEYS and m['schema'] == MANIFEST_SCHEMA, 'closed X1 manifest schema')
    need(all(m[k] == v and type(m[k]) is type(v) for k, v in MANIFEST_FIXED.items()), 'X1 remains references only')
    need(digest(m['extension']) == m['extension_sha256'], 'canonical extension digest')
    need(type(m['census']) is dict and set(m['census']) == set(CENSUS) and
         m['counts'] == {k: len(m['census'][k]) for k in CENSUS}, 'exact five-class census and counts')
    rows = m['rows']
    need(type(rows) is list and len(rows) == m['counts']['appended'], 'one row per appended canonical target')
    keys = [r.get('evaluation_key') for r in rows if type(r) is dict]
    need(len(keys) == len(rows) and keys == sorted(set(keys)), 'unique rows sorted by evaluation key')
    for r in rows:
        need(set(r) == ROW_KEYS and r['partition'] == 'train' and r['ce_eligible'] is False and
             r['source_optimizer_commit_verified'] is False, 'closed train-only CE-false reference row')
        reference(r['result_ref'])
        need(type(r['artifact_refs']) is dict and set(r['artifact_refs']) == set(ARTIFACTS), 'seven artifact refs')
        for ref in r['artifact_refs'].values():
            reference(ref)
        need(type(r['source_emission_ids']) is list and r['source_emission_ids'], 'emission provenance retained')
    return m


def check_contract(m, *, contract, export_provenance, preprocessing_bytes, objective_bytes, labels):
    """A2: bind the X1 manifest to the exact expected production contract and the actually imported sources."""
    need(type(contract) is dict and set(contract) == CONTRACT_KEYS and contract['schema'] == CONTRACT_SCHEMA,
         'closed production target contract')
    need(type(export_provenance) is dict and set(export_provenance) == PROVENANCE_KEYS and
         m['request_sha256'] == export_provenance['request_sha256'] and
         m['cutoff_sha256'] == export_provenance['cutoff_sha256'] and
         export_provenance['x1_source_packet_sha256'] == contract['x1_source_packet_sha256'],
         'X1 request, cutoff and exporter source packet are the accepted export')
    need(type(objective_bytes) is bytes and
         sha(objective_bytes) == contract['objective_sha256'] == m['objective_sha256'], 'pinned objective bytes')
    objective = document(objective_bytes, 65536)
    need((objective.get('schema'), objective.get('name')) == OBJECTIVE_NAME and
         objective.get('coefficients') == X1.COEFFICIENTS, 'unchanged supervised objective coefficients')
    need(type(preprocessing_bytes) is bytes and
         sha(preprocessing_bytes) == contract['normalizer_sha256'] == m['normalizer_sha256'] ==
         m['extension']['preprocessing_sha256'], 'unchanged normalizer bytes')
    need(m['roster'] == contract['roster'] == ROSTER, 'exact KumoSmall 2051 roster and campaign weight')
    sources = m['construction_sources']
    need(type(sources) is dict and set(sources) == {'labels', 'compose'} and
         all(type(v) is dict and set(v) == {'path', 'sha256'} for v in sources.values()) and
         {k: v['sha256'] for k, v in sources.items()} == contract['construction_sources'],
         'X1 construction sources are the contract sources')
    need(sha(Path(labels.__file__).resolve().read_bytes()) == contract['construction_sources']['labels'] and
         sha(Path(V.__file__).resolve().read_bytes()) == contract['construction_sources']['compose'],
         'actually imported labels/compose bytes are the pinned construction sources')
    proofs = m['authority_proofs']
    need(type(proofs) is list and len(proofs) == 2, 'first and final export authority proofs')
    for proof in proofs:
        need(type(proof) is dict and proof.get('objective_sha256') == contract['objective_sha256'] and
             proof.get('normalizer_sha256') == contract['normalizer_sha256'] and
             proof.get('roster') == contract['roster'], 'export authority proofs bind the target contract')
    need(proofs[0].get('parent_roles') == proofs[1].get('parent_roles'), 'no role drift across the export')


def _ref_key(ref):
    """Comparison key across the two established reference shapes; the original bytes are never rewritten."""
    return digest(K.normalized_ref(ref))


def check_transition(manifest, predecessor_raw):
    """The predecessor is the actual current one, old refs stay an exact ordered prefix, and the appended
    references and the manifest rows are in exact bijection (A1).  Returns (old extension, appended map)."""
    need(type(predecessor_raw) is bytes and sha(predecessor_raw) == manifest['predecessor_sha256'],
         'X1 was exported against the actual current predecessor')
    old = document(predecessor_raw, MAX_MANIFEST)
    validate_extension_transition(old, manifest['extension'])
    appended = {}
    for before, after in zip(old['tasks'], manifest['extension']['tasks']):
        cid = Context(**after['context']).identity
        need(after['program_results'][:len(before['program_results'])] == before['program_results'],
             'old references are an exact byte prefix')
        for ref in after['program_results'][len(before['program_results']):]:
            key = (cid, _ref_key(ref))
            need(key not in appended, 'duplicate appended reference')
            appended[key] = {'task': after, 'ref': ref}
    rows = {}
    for r in manifest['rows']:
        key = (r['context_id'], _ref_key(r['result_ref']))
        need(key not in rows, 'duplicate row for one appended reference')
        rows[key] = r
    need(set(rows) == set(appended), 'appended references and rows are not in exact bijection')
    for key, r in rows.items():
        task = appended[key]['task']
        need(task['parent_lineage_sha256'] == r['parent_lineage_sha256'], 'row joins its task lineage')
    return old, appended


def row_identity(reader, row):
    """A1: the row's context, evaluation key, program and artifacts are exactly its wrapper's (read by exact version)."""
    entry = document(reader.read(row['result_ref']), 1024 * 1024)
    need(entry['artifacts'] == row['artifact_refs'] and entry['evaluation_key'] == row['evaluation_key'] and
         entry['program_sha256'] == row['program_sha256'] and
         Context(**entry['context']).identity == row['context_id'], 'row is the exact wrapper identity')
    return entry


def _y_and_baselines(reader, row):
    """Independent read of the scored input and evaluation for one appended row (no compose internals)."""
    entry = row_identity(reader, row)
    raw = reader.read(entry['artifacts']['input'])
    evaluation = document(reader.read(entry['artifacts']['evaluation']), 4 * 1024**2)
    y, _, _ = K.decode_input(raw)
    rmse = [float(f['metrics']['rmse']) for f in evaluation['inner_cv']]
    base = [float(x) for x in K.mean_baselines(raw, evaluation)]
    walls = [f.get('wall_us') for f in evaluation['inner_cv']]
    return y, rmse, base, walls


def recompute_label(labels, reader, row):
    """Direct label recomputation with the frozen label primitives on the same verified bytes."""
    y, rmse, base, walls = _y_and_baselines(reader, row)
    need(rmse == [float(x) for x in row['measurement']['inner_fold_rmse']], 'X1 verified fold errors agree')
    s = labels.robust_scale(y)
    eps = labels.epsilon(y, s)
    out = {'rmse_j': rmse, 'baseline_rmse_j': base, 'epsilon': eps, 'robust_scale': s}
    if any(b < eps for b in base):
        out.update(q=None, q_opt=None, q_grid=None, status='DEGENERATE_BASELINE')
    else:
        q = labels.q_from_detail(rmse, base, eps)
        out.update(q=q, q_opt=labels.clip_q(q), q_grid=labels.q_grid(q), status='OK')
    out['runtime_seconds'] = (math.expm1(math.log1p(sum(walls) / 1e6))
                              if all(type(w) is int and w >= 0 for w in walls) else None)
    return out


def reconcile(snapshot, manifest, appended):
    """Account for every appended reference exactly once against the composed snapshot (A1)."""
    details = snapshot.receipt['variants']
    composed_new = {}
    for cid, detail in details.items():
        for x in detail.get('records', []):
            key = (cid, _ref_key(x['result']))
            if key in appended:
                need(key not in composed_new, 'one composed record per appended reference')
                composed_new[key] = x
    refused = {}
    for r in snapshot.receipt['refused']:
        if 'program_sha256' in r and 'context_id' in r:
            refused.setdefault((r['context_id'], r['program_sha256']), []).append(r['reason'])
    out, accepted_keys = [], set()
    for r in manifest['rows']:
        cid, ph = r['context_id'], r['program_sha256']
        key = (cid, _ref_key(r['result_ref']))
        if key in composed_new:
            x = composed_new[key]
            need(x['program_sha256'] == ph and x['evaluation_key'] == r['evaluation_key'],
                 'exact composed result identity')
            record = [y for y in snapshot.variant_records[cid] if y.program_sha256 == ph]
            need(len(record) == 1 and record[0].ce_eligible is False, 'one CE-false composed record')
            accepted_keys.add(key)
            out.append({'evaluation_key': r['evaluation_key'], 'context_id': cid, 'program_sha256': ph,
                        'disposition': 'record' if record[0].q is not None else 'record_masked_no_q',
                        'record': record[0]})
        else:
            reasons = refused.get((cid, ph), [])
            need(len(reasons) == 1, 'unaccounted appended row: ' + r['evaluation_key'])
            out.append({'evaluation_key': r['evaluation_key'], 'context_id': cid, 'program_sha256': ph,
                        'disposition': 'refused:' + reasons[0], 'record': None})
    need(set(composed_new) == accepted_keys, 'composed record without exactly one manifest row')
    need(len(out) == len(appended), 'every appended reference reconciled once')
    return out


def _create(path, body):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'path': str(path), 'sha256': sha(body), 'bytes': len(body)}


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _module_pins():
    """Exact bytes of the frozen modules this preparation actually imported (recorded, never trusted from env)."""
    import pipelines.generated_target_admission_v1.common as common
    import pipelines.generated_target_admission_v1.transition as transition
    out = {}
    for mod in (V, K, P, X1, common, transition):
        path = Path(mod.__file__).resolve()
        out[mod.__name__] = {'path': str(path), 'sha256': sha(path.read_bytes())}
    return out


def prepare_root(manifest_raw, *, expected_manifest_sha256, predecessor_raw, native, store, native_spec_bytes,
                 registry_bytes, preprocessing_bytes, objective_bytes, contract, export_provenance,
                 authority_callback, root, deadline, labels, read_budget_bytes=512 * 1024**2, extra_sources=None):
    """Create a new immutable preparation root; return the C10a receipt dict (also written to disk).

    ``contract`` is the campaign-invariant production contract (CONTRACT_KEYS) and ``export_provenance`` the
    accepted X1 export's request and cutoff digests.  ``deadline`` is the single absolute work clock for every
    stage; ``read_budget_bytes`` the single artifact read budget shared by composition and recomputation.
    ``extra_sources`` maps a flat file name to an absolute file copied into the root and pinned in
    ``owned_sources`` (for example the bounded worker itself).
    """
    clock = Clock(deadline)
    need(type(read_budget_bytes) is int and 0 < read_budget_bytes <= 8 * 1024**3, 'bounded read budget')
    cpu0 = os.times()
    audit0 = store_audit(store)
    budgeted = BudgetStore(store, budget=read_budget_bytes, clock=clock)
    m = parse_manifest(manifest_raw, expected_sha256=expected_manifest_sha256)
    check_contract(m, contract=contract, export_provenance=export_provenance, preprocessing_bytes=preprocessing_bytes,
                   objective_bytes=objective_bytes, labels=labels)
    _, appended = check_transition(m, predecessor_raw)
    extension = canonical(m['extension'])
    need(sha(extension) == m['extension_sha256'], 'extension bytes are the pinned canonical bytes')
    root = Path(root)
    need(root.is_absolute() and not root.exists() and root.parent.is_dir() and not root.parent.is_symlink(),
         'new absolute preparation root')
    native_spec = document(native_spec_bytes, MAX_MANIFEST)
    parents = {p['parent_lineage_sha256']: p for p in map(lambda line: document(line, 1024 * 1024),
                                                             registry_bytes.splitlines())}

    def role_check(_extension_bytes):
        V.check_authority(m['extension'], native_spec, authority_callback(), parents, time.time())

    reader = V.Reader(budgeted, read_budget_bytes, deadline)
    for row in m['rows']:
        row_identity(reader, row)
    clock.tick('before_compose')
    snapshot = V.compose(native, extension, m['extension_sha256'], budgeted, native_spec_bytes=native_spec_bytes,
                         registry_bytes=registry_bytes, preprocessing_bytes=preprocessing_bytes,
                         authority_callback=authority_callback, deadline=deadline)
    clock.tick('after_compose')
    rows = reconcile(snapshot, m, appended)
    evidence = []
    for row, x in zip(m['rows'], rows):
        clock.tick('recompute:' + row['evaluation_key'])
        direct = recompute_label(labels, reader, row)
        item = {k: x[k] for k in ('evaluation_key', 'context_id', 'program_sha256', 'disposition')}
        item.update(result_ref=row['result_ref'], source_emission_ids=row['source_emission_ids'], direct=direct)
        if x['record'] is not None:
            r = x['record']
            need((r.q, r.q_opt, r.q_grid) == (direct['q'], direct['q_opt'], direct['q_grid']),
                 'composed label equals direct recomputation: ' + row['evaluation_key'])
            item['record'] = {'q': r.q, 'q_opt': r.q_opt, 'q_grid': r.q_grid, 'feasibility': r.feasibility,
                              'runtime_seconds': r.runtime_seconds, 'quality_eligible': r.quality_eligible,
                              'pairwise_eligible': r.pairwise_eligible, 'ce_eligible': r.ce_eligible}
        evidence.append(item)
    clock.tick('before_prepare')
    prepared = P.prepare(snapshot, extension)
    audit1 = store_audit(store)
    delta = {k: audit1[k] - audit0[k] for k in audit0}
    need(all(v >= 0 for v in delta.values()), 'monotone store audit counters')
    need(delta['network_gets'] == 0, 'network-attempting store cannot claim CPU-only zero-network preparation')
    clock.tick('before_root')
    root.mkdir(mode=0o755)
    sources = {'prepared_snapshot.py': Path(P.__file__).resolve(), 'preparation.py': Path(__file__).resolve()}
    sources.update(extra_sources or {})
    owned = {}
    for rel, src in sorted(sources.items()):
        need('/' not in rel and rel not in RESERVED, 'flat owned source name')
        owned[rel] = _create(root / rel, Path(src).read_bytes())['sha256']
    extension_ref = _create(root / 'extension.json', extension)
    origin = {'x1_manifest_sha256': expected_manifest_sha256,
              'x1_source_packet_sha256': contract['x1_source_packet_sha256'],
              'predecessor_sha256': m['predecessor_sha256'], 'request_sha256': m['request_sha256'],
              'cutoff_sha256': m['cutoff_sha256'], 'contract_sha256': digest(contract)}
    config = {'schema': CONFIG_SCHEMA, 'owned_root': str(root), 'extension_sha256': m['extension_sha256'],
              'owned_sources': owned, 'network_gets': delta['network_gets'], 's3_writes': 0, 'optimizer_updates': 0,
              'planner_mutations': 0, 'origin': origin}
    config_raw = canonical(config)
    config_ref = _create(root / 'config.json', config_raw)
    prepared_ref = _create(root / 'prepared.json', prepared)
    clock.tick('before_hydration_round_trip')
    probe_receipt = {'schema': STATUS_SCHEMA, 'state': 'passed', 'optimizer_updates': 0, 'planner_mutations': 0,
                     's3_writes': 0, 'prepared': prepared_ref}
    hydrated = P.load(prepared, native, receipt=probe_receipt, expected_extension_sha256=m['extension_sha256'],
                      expected_preprocessing_sha256=m['extension']['preprocessing_sha256'],
                      expected_parent_registry_sha256=m['extension']['parent_registry_sha256'],
                      authority_check=role_check)
    need(hydrated.contexts == snapshot.contexts and hydrated.variant_records == snapshot.variant_records and
         hydrated.best_observed == snapshot.best_observed, 'exact prepared hydration round trip')
    for cid, z in snapshot.variant_vectors.items():
        need(bytes(memoryview(hydrated.variant_vectors[cid])) == bytes(memoryview(z.astype('<f4'))),
             'exact vector bytes after hydration')
    cpu1 = os.times()
    status = {'schema': STATUS_SCHEMA, 'state': 'passed', 'config_sha256': sha(config_raw), 'prepared': prepared_ref,
              'extension_sha256': m['extension_sha256'], 'network_gets': delta['network_gets'], 's3_writes': 0,
              'optimizer_updates': 0, 'planner_mutations': 0, 'admission_committed': False, 'variant_ce_teachers': 0,
              'prepared_source_sha256': owned['prepared_snapshot.py'],
              'source_adapter_sha256': sha(Path(V.__file__).resolve().read_bytes()),
              'counts': snapshot.receipt['counts'],
              'program_records': sum(len(v) for v in snapshot.variant_records.values()),
              'store_reads': delta['local_reads'], 'store_bytes': delta['local_bytes'],
              'seconds': time.monotonic() - clock.started,
              'cpu_user_seconds': cpu1.user - cpu0.user, 'cpu_system_seconds': cpu1.system - cpu0.system,
              'scope': 'CPU preparation only; unadmitted; consumer must reconcile and adopt at a closed epoch.'}
    status_raw = canonical(status)
    status_ref = {'path': str(root / 'status.json'), 'sha256': sha(status_raw), 'bytes': len(status_raw)}
    receipt = {'schema': RECEIPT_SCHEMA, 'root': str(root), 'x1_manifest_sha256': expected_manifest_sha256,
               'extension_sha256': m['extension_sha256'], 'predecessor_sha256': m['predecessor_sha256'],
               'contract': contract, 'export_provenance': export_provenance, 'origin': origin,
               'config': config_ref, 'status': status_ref, 'prepared': prepared_ref, 'extension': extension_ref,
               'census': m['census'], 'counts': m['counts'],
               'bijection': {'appended_references': len(appended), 'rows': len(m['rows'])},
               'rows': evidence,
               'dispositions': {d: sum(1 for e in evidence if e['disposition'] == d)
                                for d in sorted({e['disposition'] for e in evidence})},
               'store_audit': {'before': audit0, 'after': audit1, 'delta': delta},
               'clock': {**clock.bound(), 'stages': list(clock.stages)},
               'read_budget': {'bytes': read_budget_bytes, 'charged': budgeted.charged, 'reads': budgeted.reads},
               'module_pins': _module_pins(),
               'labels_rule': 'frozen dltrain.labels via native_variant_snapshot.compose; midranks are recomputed '
                              'per selected window by collate and are not persisted',
               'flags': {'numeric_targets_materialized': True, 'optimizer_updates': 0, 'admission_committed': False,
                         'variant_ce_teachers': 0, 'network_gets': delta['network_gets'], 's3_writes': 0,
                         'gp_executions': 0}}
    clock.tick('before_receipt')
    receipt_raw = canonical(receipt)
    _create(root / 'c10a-receipt.json', receipt_raw)
    clock.tick('before_status')
    _create(root / 'status.json', status_raw)
    _fsync_dir(root)
    # Success is qualified only AFTER the status is durable: the post-fsync sample must still be inside the bound.
    qualified = clock.qualify('after durable status')
    terminal = {'schema': TERMINAL_SCHEMA, 'status_sha256': sha(status_raw), 'receipt_sha256': sha(receipt_raw),
                'qualified_after_durable_status': qualified, 'bound': clock.bound()}
    _create(root / 'c10a-terminal.json', canonical(terminal))
    _fsync_dir(root)
    # The FINAL publication is bounded too: a deadline crossed while writing/fsyncing the marker never returns
    # success (the late marker stays on disk as custody; consumers additionally require the operator acceptance).
    clock.qualify('after final terminal publication')
    return receipt


def candidate_descriptor(root, *, campaign_sha256, acceptance):
    """OPERATOR-ONLY: the exact prepared_admission.candidate descriptor for a finished, ACCEPTED root.

    ``acceptance`` is the ``accept_root`` envelope reference; the (closed, frozen) descriptor itself does not carry it,
    so a boundary publication pins it next to the descriptor.

    Producing a descriptor admits nothing; publishing it to a boundary inbox is a separate operator decision.
    Refuses a root whose success marker, configuration, prepared bytes and C10a receipt do not bind each other.
    """
    root = Path(root)
    refs = {}
    for key, name in (('preparation_config', 'config.json'), ('preparation_status', 'status.json'),
                      ('prepared', 'prepared.json')):
        body = (root / name).read_bytes()
        refs[key] = {'path': str(root / name), 'sha256': sha(body), 'bytes': len(body)}
    status = json.loads((root / 'status.json').read_bytes())
    receipt_raw = (root / 'c10a-receipt.json').read_bytes()
    receipt = json.loads(receipt_raw)
    terminal_raw = (root / 'c10a-terminal.json').read_bytes()
    check_terminal(terminal_raw, status_raw=(root / 'status.json').read_bytes(), receipt_raw=receipt_raw)
    check_root_acceptance(acceptance, root=root, status_raw=(root / 'status.json').read_bytes(), receipt_raw=receipt_raw,
                          terminal_raw=terminal_raw)
    need(status['state'] == 'passed' and status['prepared'] == refs['prepared'] and
         receipt['schema'] == RECEIPT_SCHEMA and receipt['status'] == refs['preparation_status'] and
         receipt['config'] == refs['preparation_config'] and receipt['prepared'] == refs['prepared'],
         'success marker, configuration, prepared bytes and receipt bind one root')
    return {'schema': 'jope.prepared-epoch-candidate/v1', 'campaign_sha256': campaign_sha256,
            'approved_for_bounded_research': True, **refs}
