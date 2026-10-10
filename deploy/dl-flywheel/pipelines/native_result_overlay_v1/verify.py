"""Per-row verification of one uniform verified native result row (plan v2.0 §5 W3 steps 1-7).

Invariants:
- Outer-test values are never converted to numbers. The only decoders are the frozen ``NativeInputs.describe``
  (``masked_training`` decodes the registered outer-TRAIN prefix) and ``train_prefix`` below, which converts exactly the
  header, n fold bytes and n*(p+1) training cells of the masked S1 input. Outer-test bytes are only hashed.
- A row can become a training target only when, in order: (1) its emission identity re-derives exactly with the
  frozen Outbox identity formula, body digest, batch membership, frozen snapshot descriptor and canonical program;
  (2) the frozen ``describe(descriptor, 'training')`` returns the typed OriginalScoreContext and training rows;
  (3) ``request_identity(context, program)`` equals the row's evaluation key under the pinned WORKER_SHA;
  (4) the wrapper and all seven artifacts are exact-version GETs that pass the frozen ``verified_measurement`` and
  Rust parity was checked at the fixed 1e-9 tolerance on both folds; (5) the opposite-fold mean baselines rebuilt from
  the verified training rows exactly like ``NativeContextLoader._baseline`` agree with the evaluation's per-fold
  baselines (rel 1e-10, abs 1e-12); (6) the shard join (``shard_join``) holds; (7) the canonical program tokenizes
  with no refusal and detokenizes to the same SHA.
- Row-local failures raise ``Refusal(code)`` (census REFUSED) or ``Pending`` (census PENDING); infrastructure failures
  (deadline, read budget, stale authority, pin drift) raise ValueError and stop the whole build with no root.
"""
import base64
import binascii
import math
import struct
import time
from collections import OrderedDict

import numpy as np

import native_evaluate as NE
import native_inputs as NI
from dltrain import labels as L
from jope_flywheel import program_worker as PW
from jope_flywheel import training_snapshot_native as N
from jope_flywheel.program_gate import object_bytes

from .common import (ARTIFACT_ROLES, AUTHORITY_MAX_AGE, AUTHORITY_SCHEMA, BASELINE_ABS, BASELINE_REL,
                     EMISSION_SCHEMA, MAX_CACHE_BYTES, PARITY_TOLERANCE, R6_SCHEMA, REF_KEYS, RESULT_SCHEMA,
                     WORKER_SHA_PIN, artifact_refs, canonical, digest, hash64, need, parse, ref5, sha)

ROW_KEYS = frozenset({'outbox_kind', 'observation', 'emission_body_b64', 'emission_body_sha256', 'state', 'record_ref',
                      'artifact_refs', 'evaluation_key', 'context_id', 'phase', 'campaign', 'program'})
MAX_EMISSION_BYTES = 128 * 1024           # frozen Outbox MAX_RECORD
OBSERVATION_KEYS = frozenset({'emission_id', 'body_sha256', 'context_id', 'phase', 'campaign_sha256',
                              'canonical_program_sha256', 'evaluation_key'})
# Closed key sets of the frozen Outbox.record_batch (continuous/neural_outbox.py).
BATCH_KEYS = frozenset({'batch_id', 'campaign_sha256', 'checkpoint_sha256', 'global_step', 'manifest_sha256',
                        'train_data_run_ids', 'validation_data_run_ids', 'emitted_train_data_run_ids'})
EMISSION_KEYS = frozenset({'phase', 'data_run_id', 'context_id', 'parent_lineage_sha256', 'input_descriptor',
                           'raw_program_utf8', 'raw_program_sha256', 'decoder_seed', 'decoder_protocol_sha256',
                           'decode_artifact_sha256'})


class Refusal(Exception):
    """A row-local refusal with a closed census reason."""

    def __init__(self, code, detail=''):
        super().__init__(code + ': ' + str(detail)[:500])
        self.code, self.detail = code, str(detail)[:500]


class Pending(Exception):
    """A row that cannot be resolved now (never zero-filled, never a target)."""

    def __init__(self, code, detail=''):
        super().__init__(code + ': ' + str(detail)[:500])
        self.code, self.detail = code, str(detail)[:500]


def row_shape(row):
    """The uniform W4 ``verified_results`` row contract (closed keys); a violation stops the build."""
    need(type(row) is dict and set(row) == ROW_KEYS, 'closed uniform verified-result row')
    need(type(row['outbox_kind']) is str and row['outbox_kind'] and type(row['state']) is str and row['state'],
         'row source kind/state')
    need(hash64(row['context_id']) and row['phase'] in ('training', 'validation') and hash64(row['campaign']),
         'row context/phase/campaign')
    need(type(row['observation']) is dict and type(row['emission_body_b64']) is str and
         type(row['emission_body_sha256']) is str, 'row observation and exact stored emission body')
    return row


def emission_body(row):
    """Decode W4's ``emission_body_b64`` (base64 of the EXACT stored Outbox body bytes) and check every digest.

    The bytes must be strict canonical base64, hash to ``emission_body_sha256`` and to the observation's
    ``body_sha256``, parse strictly, and be exactly the canonical JSON the frozen ``Outbox.record_batch`` stores.
    """
    try:
        text = row['emission_body_b64']
        need(0 < len(text) <= 4 * MAX_EMISSION_BYTES, 'bounded base64 emission body')
        raw = base64.b64decode(text.encode('ascii'), validate=True)
        need(base64.b64encode(raw).decode('ascii') == text, 'canonical base64 encoding')
        need(0 < len(raw) <= MAX_EMISSION_BYTES and sha(raw) == row['emission_body_sha256'] and
             row['observation'].get('body_sha256') == row['emission_body_sha256'], 'exact stored emission body digest')
        body = parse(raw)
        need(type(body) is dict and canonical(body) == raw, 'stored body is the frozen Outbox canonical JSON')
    except (ValueError, UnicodeError, TypeError, binascii.Error) as error:
        raise Refusal('EMISSION_IDENTITY', error)
    return body


def descriptor_of(body):
    """The emission's input descriptor (routing only; identity is re-derived by ``emission_identity``)."""
    e = body.get('emission')
    d = e.get('input_descriptor') if type(e) is dict else None
    return d if type(d) is dict else {}


def emission_identity(row, body, *, canonicalize):
    """Step 1: re-derive the frozen Outbox emission identity and every join; a caller-supplied id never self-certifies.

    ``body`` is the decoded exact stored body from ``emission_body(row)``; the id is the frozen Outbox formula
    ``digest({'origin': model_origin_sha256, 'batch_id': batch_id, 'ordinal': ordinal})``.
    """
    obs = row['observation']
    try:
        need(set(body) == {'schema', 'model_origin_sha256', 'batch', 'ordinal', 'emission'} and
             body['schema'] == EMISSION_SCHEMA, 'stored emission body schema')
        b, e = body['batch'], body['emission']
        need(type(b) is dict and set(b) == BATCH_KEYS and type(e) is dict and set(e) == EMISSION_KEYS,
             'closed frozen batch/emission keys')
        need(hash64(body['model_origin_sha256']) and hash64(b['batch_id']) and type(body['ordinal']) is int and
             body['ordinal'] >= 0, 'emission origin/batch/ordinal')
        eid = digest({'origin': body['model_origin_sha256'], 'batch_id': b['batch_id'], 'ordinal': body['ordinal']})
        need(OBSERVATION_KEYS <= set(obs), 'observation identity keys')
        need(obs['emission_id'] == eid and obs['body_sha256'] == sha(canonical(body)) == row['emission_body_sha256'],
             'emission id re-derived from the exact stored body')
        d = e['input_descriptor']
        cid = row['context_id']
        need(type(d) is dict and e['context_id'] == e['data_run_id'] == cid == obs['context_id'] ==
             d.get('context_id'), 'emission context join')
        need(e['phase'] == row['phase'] == obs['phase'], 'emission phase join')
        need(b['campaign_sha256'] == row['campaign'] == obs['campaign_sha256'], 'emission campaign join')
        members = b['emitted_train_data_run_ids'] if e['phase'] == 'training' else b['validation_data_run_ids']
        need(type(members) is list and cid in members, 'emission batch membership')
        need(e['parent_lineage_sha256'] == d.get('parent_lineage_sha256'), 'emission lineage join')
        need(type(e['raw_program_utf8']) is str and sha(e['raw_program_utf8'].encode()) == e['raw_program_sha256'],
             'raw emission content hash')
        if row['program'] is not None:
            need(type(row['program']) is str and canonicalize(e['raw_program_utf8']) == row['program'] and
                 obs['canonical_program_sha256'] == sha(row['program'].encode()), 'canonical program join')
        need(obs['evaluation_key'] in (None, row['evaluation_key']), 'observation evaluation key join')
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise Refusal('EMISSION_IDENTITY', error)
    return eid, d


def check_authority(raw, authority_pins, now):
    """Current native role authority: exact pins, passed, and at most 60 s old."""
    need(type(raw) is bytes, 'authority callback returns canonical bytes')
    p = N.document(raw)
    need(p.get('schema') == AUTHORITY_SCHEMA and p.get('passed') is True, 'current native role authority')
    need(p.get('source_pins') == authority_pins, 'exact native authority source pins')
    need(type(p.get('checked_at_unix')) in (int, float) and 0 <= now - p['checked_at_unix'] <= AUTHORITY_MAX_AGE,
         'fresh native authority (<= 60 s)')
    need(type(p.get('protected_context_ids', [])) is list and type(p.get('protected_views', [])) is list,
         'protected role lists')
    return p, {'sha256': sha(raw), 'checked_at_unix': p['checked_at_unix'], 'observed_unix': now}


def r6_roster(raw, expected_sha256):
    """The pre-registered R6 D-tune/D-confirm roster (contexts and lineages never enter optimizer targets)."""
    need(type(raw) is bytes and sha(raw) == expected_sha256, 'pinned R6 preregistration bytes')
    doc = parse(raw)
    need(doc.get('schema') == R6_SCHEMA, 'R6 preregistration schema')
    contexts, lineages = set(), set()
    for part in ('d_tune', 'd_confirm'):
        block = doc[part]
        need(type(block) is dict and all(hash64(x) for x in block['contexts'] + block['lineages']), 'R6 roster ids')
        contexts.update(block['contexts'])
        lineages.update(block['lineages'])
    return frozenset(contexts), frozenset(lineages)


class ExactReader:
    """Exact-version GETs through the injected ``store.get_version(ref) -> (bytes, version_id)``.

    Every body must match the ref's VersionId, SHA256 and size; total transfer is bounded and checked against the
    build deadline; identical refs are served from a bounded LRU (a different ref is always fetched).
    """

    def __init__(self, store, *, limit, deadline):
        need(callable(getattr(store, 'get_version', None)) and callable(getattr(store, 'nativecanonical_json', None)),
             'exact-version store with the pinned native canonicalizer')
        need(type(limit) is int and 0 < limit, 'bounded read budget')
        self.store, self.limit, self.deadline = store, limit, deadline
        self.nativecanonical_json = store.nativecanonical_json
        self.cache, self.cache_bytes, self.gets, self.read_bytes = OrderedDict(), 0, 0, 0

    def get_version(self, ref):
        r = ref5(ref)
        key = tuple(r[k] for k in REF_KEYS)
        need(time.time() < self.deadline, 'overlay build deadline')
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key], r['version_id']
        need(self.read_bytes + r['bytes'] <= self.limit, 'overlay artifact read budget')
        body, version = self.store.get_version(r)
        self.gets += 1
        need(type(body) is bytes and version == r['version_id'] and len(body) == r['bytes'] and
             sha(body) == r['sha256'], 'exact VersionId/SHA256/bytes readback')
        self.read_bytes += len(body)
        need(time.time() < self.deadline, 'overlay build deadline after GET')
        self.cache[key] = body
        self.cache_bytes += len(body)
        while self.cache_bytes > MAX_CACHE_BYTES and len(self.cache) > 1:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= len(old)
        return body, version


def train_prefix(masked):
    """Decode ONLY the outer-TRAIN prefix of a masked S1 JTT1 (header, n fold bytes, n*(p+1) cells).

    Outer-test rows that follow the prefix are never converted to numbers; they are covered only by sha(masked).
    """
    need(type(masked) is bytes and len(masked) >= 17 and masked[:4] == b'JTT1', 'masked original JTT1')
    n, nt, p = struct.unpack_from('<III', masked, 4)
    stride = (p + 1) * 8
    need(masked[16] == 2 and n >= 50 and nt >= 10 and 1 <= p <= 512 and len(masked) == 17 + n + (n + nt) * stride,
         'qualified masked original shape')
    folds = np.frombuffer(masked, dtype=np.uint8, count=n, offset=17).copy()
    train = np.frombuffer(masked, dtype='<f8', count=n * (p + 1), offset=17 + n).reshape(n, p + 1)
    y = train[:, p].copy()
    need(np.isfinite(y).all() and set(folds.tolist()) == {0, 1}, 'finite two-fold training targets')
    return y, folds


def reconstruct_baseline(y, folds):
    """Opposite-fold mean baselines, exactly as the frozen ``NativeContextLoader._baseline`` computes them."""
    y = np.asarray(y, dtype='<f8')
    folds = np.asarray(folds, dtype=np.uint8)
    vals = []
    for j in range(2):
        tr, va = y[folds != j], y[folds == j]
        need(len(tr) > 0 and len(va) > 0, 'empty native inner fold')
        mean = math.fsum(map(float, tr)) / len(tr)
        vals.append(math.sqrt(math.fsum((float(v) - mean) ** 2 for v in va) / len(va)))
    return vals


def agree(got, want):
    return (type(got) in (list, tuple) and len(got) == 2 and
            all(type(a) in (int, float) and math.isfinite(a) and
                math.isclose(a, b, rel_tol=BASELINE_REL, abs_tol=BASELINE_ABS) for a, b in zip(got, want)))


def check_agreement(what, got, want):
    """Baseline agreement guard: anything outside rel 1e-10 / abs 1e-12 is a BASELINE_DISAGREEMENT refusal."""
    if not agree(got, want):
        raise Refusal('BASELINE_DISAGREEMENT', f'{what} baseline {got} disagrees with reconstruction {want}')


def verify_result(row, descriptor, *, nat, inputs, reader):
    """Steps 2-5 and 7 for one train row; returns verified facts (shard join and labels happen per context)."""
    cid, program = row['context_id'], row['program']
    try:
        need(type(program) is str and 0 < len(program.encode()) <= 65536 and hash64(row['evaluation_key']),
             'verified row carries a canonical program and evaluation key')
        record_ref, refs = ref5(row['record_ref']), artifact_refs(row['artifact_refs'])
    except ValueError as error:
        raise Refusal('VERIFICATION_FAILED', error)
    try:
        score, masked, _x, y, folds, info = inputs.describe(descriptor, 'training')
    except NI.Unsupported as error:
        raise Pending('NO_INPUT_LOCATOR', error)
    except (ValueError, KeyError, TypeError) as error:
        raise Refusal('VERIFICATION_FAILED', 'describe: ' + str(error))
    try:
        need(type(score) is NI.OriginalScoreContext and score.identity == cid, 'typed original score context')
        need(score.evaluator_sha256 == WORKER_SHA_PIN == NI.WORKER_SHA, 'pinned native WORKER_SHA evaluator')
        need(info.get('outer_test_values_decoded') is False and info.get('native_worker_sha256') == WORKER_SHA_PIN and
             info.get('inherited_snapshot_partition') == 'train', 'training-only describe receipt')
        need(type(masked) is bytes and sha(masked) == score.input_sha256, 'masked registered input identity')
        ty, tf = train_prefix(masked)
        need(sha(ty.astype('<f8').tobytes()) == score.target_sha256 ==
             sha(np.asarray(y, dtype='<f8').tobytes()), 'training target from the registered train prefix')
        need(sha(tf.tobytes()) == score.split_sha256 == sha(np.asarray(folds, dtype=np.uint8).tobytes()),
             'training split from the registered train prefix')
        key = NE.request_identity(score, program)
        need(key == row['evaluation_key'], 'evaluation key re-derived from the typed context and canonical program')
        entry = N.document(object_bytes(reader, record_ref))
        need(entry.get('schema') == RESULT_SCHEMA and entry.get('evaluation_key') == key and
             entry.get('program_sha256') == sha(program.encode()), 'exact result wrapper identity')
        need(artifact_refs(entry.get('artifacts')) == refs, 'row artifact refs equal the exact wrapper refs')
        measured = NE.verified_measurement(reader, score, program, entry)
        need(measured['evaluation_key'] == key and measured['context_id'] == cid and
             measured['program_sha256'] == sha(program.encode()), 'measurement identity')
        evaluation = parse(reader.get_version(refs['evaluation'])[0])
        parity = parse(reader.get_version(refs['rust_parity'])[0])
        checks = parity['native_provenance'].get('fold_checks')
        need(type(checks) is list and len(checks) == 2 and all(
            type(c) is dict and c.get('fold') == i and c.get('tolerance') == PARITY_TOLERANCE == PW.TOLERANCE and
            c.get('passed') is True for i, c in enumerate(checks)), 'Rust parity at the fixed 1e-9 on both folds')
        rmse = [float(v) for v in measured['inner_fold_rmse']]
        walls = [f.get('wall_us') for f in evaluation['inner_cv']]
        wall = sum(walls) if all(type(w) is int and w >= 0 for w in walls) else None
        ev = [f.get('baseline_metrics', {}).get('rmse') for f in evaluation['inner_cv']]
    except Refusal:
        raise
    except (ValueError, KeyError, TypeError, AttributeError, L.LabelError) as error:
        raise Refusal('VERIFICATION_FAILED', error)
    reconstructed = reconstruct_baseline(ty, tf)
    evaluation_baseline = None
    if any(v is not None for v in ev):
        check_agreement('evaluation', ev, reconstructed)
        evaluation_baseline = [float(v) for v in ev]
    tokens = tuple(nat.tokenizer.tokenize(program, nat.vocab))
    reason = nat.token_refusal(tokens)
    if reason:
        raise Refusal('TOKEN_REFUSAL', reason)
    if sha(nat.tokenizer.detokenize(tokens, nat.vocab).encode()) != sha(program.encode()):
        raise Refusal('TOKEN_REFUSAL', 'detokenize does not reproduce the program SHA')
    return {'score': score, 'info': info, 'y': np.asarray(y, dtype='<f8'), 'folds': np.asarray(folds, dtype=np.uint8),
            'key': key, 'record_ref': record_ref, 'artifact_refs': refs, 'rmse': rmse, 'wall_us': wall,
            'reconstructed': reconstructed, 'evaluation_baseline': evaluation_baseline, 'tokens': tokens,
            'program_sha256': sha(program.encode()), 'skeleton_sha256': nat.tokenizer.skeleton_sha256(program)}


def shard_join(cid, *, nat, loader, facts):
    """Step 6 (identity half): the frozen shard identity, the training-target join and the shard's exact baseline pair.

    Returns None when the context has no detail shard. The frozen ``_shard`` re-checks schema, context identity,
    protocol/vocab, record count and cross-context rows; ``label_context`` re-checks shard baseline agreement.
    """
    desc = nat.shards.get(cid)
    if desc is None:
        return None
    try:
        x = loader._shard(cid)
        need(sha(np.asarray(x['context']['y_train'], dtype='<f8').tobytes()) == facts['score'].target_sha256,
             'shard y_train equals the verified training target')
        native_input = desc.get('native_input')
        if native_input is not None:
            need(native_input['teacher']['sha256'] == facts['info']['teacher']['gz_sha256'] and
                 native_input['context_json']['sha256'] == facts['info']['context_json']['sha256'],
                 'shard native input equals the described registered input')
        context_label, shard_labels = L.label_context(x['context'], x['records'])
    except (ValueError, KeyError, TypeError, L.LabelError) as error:
        raise Refusal('VERIFICATION_FAILED', 'shard join: ' + str(error))
    pair = None if context_label.baseline_rmse_j is None else list(context_label.baseline_rmse_j)
    return {'doc': x, 'ref': dict(desc['shard']), 'pair': pair, 'teacher_registered': native_input is not None,
            'labels': {lab.program_sha256: lab for lab in shard_labels},
            'programs': frozenset(r['program_sha256'] for r in x['records'])}


def frozen_baseline(cid, *, loader, shard):
    """The frozen ``NativeContextLoader._baseline`` re-verification of the shard against the teacher train prefix."""
    if shard is None or not shard['teacher_registered']:
        return False
    try:
        need(loader._baseline(cid, shard['doc']) is True, 'frozen teacher-prefix baseline verification')
    except ValueError as error:
        raise Refusal('VERIFICATION_FAILED', 'frozen _baseline: ' + str(error))
    return True


def anchor(*, shard, carried, facts):
    """Step 6 (anchoring): the exact baseline pair every overlay row of this context must carry.

    A shard with an established baseline anchors to the shard's EXACT pair when it agrees with the reconstruction
    (both values are recorded); otherwise the first anchored pair carried in the packet chain is enforced bit-equal;
    otherwise the reconstruction itself becomes the first anchor. Any disagreement is BASELINE_DISAGREEMENT.
    """
    recon = facts['reconstructed']
    if carried is not None:
        need(carried['y_train_sha256'] == facts['score'].target_sha256 and
             carried['split_sha256'] == facts['score'].split_sha256, 'carried context target/split identity')
    if shard is not None and shard['pair'] is not None:
        check_agreement('shard', shard['pair'], recon)
        anchored, source = list(shard['pair']), 'shard'
        if carried is not None:
            need(carried['anchored_baseline'] == anchored, 'carried anchor equals the shard pair bit-for-bit')
    elif carried is not None:
        check_agreement('chain', carried['anchored_baseline'], recon)
        anchored, source = list(carried['anchored_baseline']), 'chain'
    else:
        anchored, source = list(recon), 'reconstructed'
    return anchored, source
