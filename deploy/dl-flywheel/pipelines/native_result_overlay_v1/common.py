"""Closed constants and byte-exact helpers for native_result_overlay_v1 (plan v2.0 §5 W3, X3).

Invariants:
- Canonical JSON is the frozen ``jope_flywheel.protocol.canonical`` (sorted keys, compact separators, no NaN/inf).
- Every file this package writes is create-only (O_EXCL|O_NOFOLLOW), mode 0444, fsynced, and its directory fsynced.
- Every packet read is an exact ``{path, sha256, bytes}`` read of a regular, non-symlink file with no symlink parent.
- Census classes are closed: every consumed source row lands in exactly one of five classes with one closed reason.
- Revision 2: packets are DELTAS (<= 2,048 new rows each) in a head chain of <= 4,096 packets / <= 500,000 rows.
- Terminal qualification: one paired original bound (wall deadline, the matching monotonic deadline, the boot id),
  fixed when the build starts and never reset; a clock sample is in time only strictly inside it on the same boot.
Nothing here touches S3, a GPU, an outbox, a planner, an optimizer or a GP/program executor.
"""
import hashlib
import json
import math
import os
import resource
import stat
import time
from pathlib import Path

from jope_flywheel.protocol import canonical, digest

PACKET_SCHEMA = 'jope.native-result-overlay-packet/v2'   # revision 2: DELTA packets (new rows only)
HEAD_SCHEMA = 'jope.native-overlay-head/v1'
WINDOW_SCHEMA = 'jope.native-overlay-window/v1'
RECEIPT_SCHEMA = 'jope.native-result-overlay-build/v1'
MARKER_SCHEMA = 'jope.native-result-overlay-complete/v2'  # v2: seals the original paired bound
TERMINAL_SCHEMA = 'jope.native-result-overlay-terminal/v1'
MARKER_NAME = 'overlay-complete.json'
ACCEPTED_NAME = 'overlay-accepted.json'   # in-time terminal acceptance; every consumer requires it
REFUSED_NAME = 'overlay-refused.json'     # late terminal refusal (root kept for custody); every consumer refuses
MARKER_KEYS = frozenset({'schema', 'packet', 'receipt', 'head_sha256', 'predecessor_overlay_head_sha256', 'seq',
                         'bound'})
TERMINAL_KEYS = frozenset({'schema', 'verdict', 'stage', 'root', 'packet', 'receipt', 'marker', 'head_sha256',
                           'predecessor_overlay_head_sha256', 'seq', 'bound', 'sample', 'acceptance'})
BOUND_KEYS = frozenset({'wall_deadline_unix', 'monotonic_deadline', 'boot_id'})
SAMPLE_KEYS = frozenset({'boot_id', 'monotonic', 'wall_unix'})
MAX_TERMINAL_BYTES = 1024 * 1024
DRY_RUN_SCHEMA = 'jope.native-overlay-dry-run/v1'
ROTATION_DOMAIN = 'native-overlay-rotation/v1'
SOURCE_KIND = 'native_overlay_verified_s1'
MERGED_SOURCE_KIND = 'hwm_winner_ce+native_overlay_verified_s1'
RESULT_SCHEMA = 'jope.native-original-program-result/v1'
AUTHORITY_SCHEMA = 'jope.native-corpus-current-authority/v1'
R6_SCHEMA = 'jope.rlfw-r6-dev-split-preregistration/v1'
EMISSION_SCHEMA = 'jope.flywheel-neural-emission/v1'

OVERLAY_WINDOW = 8                      # k_c = min(|O_c|, 8)
MAX_PACKET_BYTES = 32 * 1024**2
MAX_ROWS = 2048                         # NEW overlay rows in one delta packet
MAX_CONTEXTS = 1024                     # contexts touched or first declared in one packet
MAX_CHAIN_PACKETS = 4096
MAX_CHAIN_ROWS = 500000
STOPS = (None, 'NEW_ROW_CAP', 'CONTEXT_CAP', 'SOURCE_ROW_CAP')   # why a build stopped consuming its iterator
MAX_SECONDS = 1800
MAX_RSS_BYTES = 2 * 1024**3
MAX_SOURCE_ROWS = 32768                 # consumed source rows (census entries) in one build
MAX_READ_BYTES = 8 * 1024**3
MAX_CACHE_BYTES = 128 * 1024**2
AUTHORITY_MAX_AGE = 60
AUTHORITY_EVERY = 16

WORKER_SHA_PIN = 'e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8'
PRODUCTION_OBJECTIVE_SHA256 = 'bf02f87341bc52aa4b3c329492d1eae5621c3543a9e60c9b3f34aa7b407ece48'
R6_PREREGISTRATION_SHA256 = 'd228d41366e5e313da63fc64ea3c76bcbfebe6899c4e0bb015f9da22d741725a'
PARITY_TOLERANCE = 1e-9
BASELINE_REL, BASELINE_ABS = 1e-10, 1e-12
RECOMPUTE_REL = 1e-12

ARTIFACT_ROLES = ('input', 'program', 'evaluation', 'rust_source', 'rust_executable', 'rust_parity', 'score_records')
REF_KEYS = ('bucket', 'key', 'version_id', 'sha256', 'bytes')
VERIFIED_STATES = frozenset({'published_verified', 'cached_verified'})
FAILURE_STATES = frozenset({'failure_custody_verified'})
PENDING_STATES = frozenset({'recorded', 'leased', 'unknown_attempt', 'unknown', 'pending_unsupported', 'blocked'})
# Frozen module closure whose exact bytes the packet pins (the caller supplies the reviewed shas).
DEPENDENCIES = ('dltrain.labels', 'dltrain.dsl_tokenizer', 'dltrain.automaton', 'native_inputs', 'native_evaluate',
                'jope_flywheel.training_snapshot_native', 'jope_flywheel.program_gate', 'jope_flywheel.program_worker',
                'jope_flywheel.score_join', 'neural_native_train', 'neural_native_batch')

CLASSES = ('TRAIN_VERIFIED', 'EVALUATION_ONLY', 'ROUTED_TO_X1', 'REFUSED', 'PENDING')
REASONS = {
    'TRAIN_VERIFIED': frozenset({'NEW_OVERLAY_ROW', 'ALIAS', 'ALREADY_IN_SHARD', 'ALREADY_IN_OVERLAY'}),
    'EVALUATION_ONLY': frozenset({'VALIDATION_PHASE', 'VALIDATION_PARTITION', 'R6_ROSTER'}),
    'ROUTED_TO_X1': frozenset({'VARIANT_SOURCE'}),
    'REFUSED': frozenset({'PARITY_UNQUALIFIED', 'CAMPAIGN_NOT_ALLOWED', 'UNSUPPORTED_SOURCE_KIND', 'EMISSION_IDENTITY',
                          'OUTSIDE_SNAPSHOT', 'PROTECTED_CONTEXT', 'OUTSIDE_FIT_ROSTER', 'VERIFICATION_FAILED',
                          'BASELINE_DISAGREEMENT', 'TOKEN_REFUSAL', 'CONFLICTING_REFS', 'OLD_LABELS_WOULD_CHANGE',
                          'LABEL_ERROR', 'SHARD_RMSE_DISAGREEMENT'}),
    'PENDING': frozenset({'UNRESOLVED_STATE', 'NO_INPUT_LOCATOR'}),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def hash64(x):
    return type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def ref5(ref):
    """Exact versioned object reference -> its closed five-field form (the seven-field annotated form is accepted)."""
    need(type(ref) is dict and set(ref) in (set(REF_KEYS), set(REF_KEYS) | {'state', 'verification'}),
         'exact versioned object reference required')
    if 'verification' in ref:
        need(ref['verification'] == 'exact VersionId GET + SHA256', 'unverified custody reference')
    need(all(type(ref[k]) is str and ref[k] for k in ('bucket', 'key', 'version_id')) and ref['version_id'] != 'null'
         and hash64(ref['sha256']) and type(ref['bytes']) is int and ref['bytes'] > 0, 'exact five-field object ref')
    return {k: ref[k] for k in REF_KEYS}


def artifact_refs(refs):
    need(type(refs) is dict and set(refs) == set(ARTIFACT_ROLES), 'exactly seven artifact refs')
    return {k: ref5(refs[k]) for k in ARTIFACT_ROLES}


def parse(raw):
    """Strict JSON: duplicate keys and non-finite constants are refused."""
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, 'duplicate JSON key')
            out[k] = v
        return out

    def constant(value):
        raise ValueError('non-finite JSON constant ' + value)
    need(type(raw) is bytes, 'JSON bytes required')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def exact_canonical(raw):
    """Parse ``raw`` and require it to be byte-identical to its canonical serialization."""
    doc = parse(raw)
    need(canonical(doc) == raw, 'non-canonical JSON bytes')
    return doc


def safe_path(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'absolute nontraversing path')
    need(all(not p.is_symlink() for p in (path, *path.parents)), 'symlink in path or parents')
    return path


def read_exact(ref, limit):
    """Exact bytes of a regular non-symlink file named by {path, sha256, bytes}; no ambient discovery."""
    need(type(ref) is dict and set(ref) == {'path', 'sha256', 'bytes'} and type(ref['path']) is str and
         hash64(ref['sha256']) and type(ref['bytes']) is int and 0 < ref['bytes'] <= limit, 'exact bounded file ref')
    path = safe_path(ref['path'])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and st.st_size == ref['bytes'], 'regular file of the pinned size')
        chunks, left = [], ref['bytes'] + 1
        while left > 0:
            chunk = os.read(fd, left)
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        body = b''.join(chunks)
    finally:
        os.close(fd)
    need(len(body) == ref['bytes'] and sha(body) == ref['sha256'], 'exact pinned file bytes')
    return body


def read_small(path, limit):
    """Bytes of a small regular non-symlink file whose digest is not yet known (only the success marker)."""
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= limit, 'small regular file')
        body = os.read(fd, limit + 1)
    finally:
        os.close(fd)
    need(len(body) == st.st_size, 'complete small file read')
    return body


def create_only(path, body):
    """Create-only 0444 write with fsync; refuses an existing path or a symlink."""
    path = safe_path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'path': str(path), 'sha256': sha(body), 'bytes': len(body)}


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def boot_id():
    with open('/proc/sys/kernel/random/boot_id') as f:
        return f.read().strip()


def clock_sample():
    """Boot, monotonic, wall, then a FRESH monotonic and boot after the wall sample (the C10a ``sample_now`` rule):
    a stall while sampling the wall clock, or a backward wall step, cannot hide an elapsed monotonic bound or a
    reboot."""
    boot0, mono0 = boot_id(), time.monotonic()
    wall = time.time()
    mono, boot = time.monotonic(), boot_id()
    return {'boot_id': boot if boot == boot0 else 'boot-changed-while-sampling', 'monotonic': max(mono, mono0),
            'wall_unix': wall}


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def check_bound(b):
    """A strict finite paired (wall, monotonic, boot) original bound."""
    need(type(b) is dict and set(b) == BOUND_KEYS and _finite(b['wall_deadline_unix']) and
         _finite(b['monotonic_deadline']) and type(b['boot_id']) is str and 0 < len(b['boot_id']) <= 64,
         'strict finite paired bound')
    return b


def in_bound(sample, bound):
    """True iff a strict clock sample lies strictly inside the original bound on the same boot."""
    check_bound(bound)
    need(type(sample) is dict and set(sample) == SAMPLE_KEYS and _finite(sample['monotonic']) and
         _finite(sample['wall_unix']) and type(sample['boot_id']) is str, 'strict clock sample')
    return (sample['boot_id'] == bound['boot_id'] and sample['wall_unix'] < bound['wall_deadline_unix'] and
            sample['monotonic'] < bound['monotonic_deadline'])


def module_sha(module):
    return sha(Path(module.__file__).resolve().read_bytes())


def max_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


__all__ = ['canonical', 'digest']
