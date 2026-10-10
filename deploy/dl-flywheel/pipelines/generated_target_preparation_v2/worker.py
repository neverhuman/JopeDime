"""W2: bounded two-phase C10a worker -- GET-only exact-version prefetch, then closed zero-network preparation.

Revision 7: ``qualified.json``'s bound must be the sealed ``complete.json`` clock bound (strict, finite); prefetch
re-qualifies after its FINAL marker publication; ``accept_cache`` (operator, after the worker terminated) seals the
exact marker/artifact bytes, unchanged bound and physical in-time success, and ``ClosedCache`` requires that envelope.

Phase 1 ``prefetch`` re-validates the X1 manifest, contract and transition (no reads), then performs ONE recording
dry run of the frozen composition and of the row identity/label reads against the remote store.  Every exact
object it reads is verified (VersionId, length, SHA-256) and streamed create-only into a fresh closed cache whose
roster is exactly the recorded set (revision 4: bodies are never accumulated in memory; the cumulative declared
size is checked against the cache cap before each new fetch).  Objects already present in a previous complete cache are copied from it after
re-verification instead of being fetched again (chained caches).  ``complete.json`` is written last.

Phase 2 ``prepare`` opens the complete cache (manifest + completion proof + every file re-hashed), and runs
``preparation.prepare_root`` with it.  The cache attests ``closed_exact_version_local`` with audited counters, so a
successful root proves ``network_gets == 0`` during composition.  Any read outside the declared roster refuses.

Nothing here writes S3, touches a planner, outbox, optimizer or GPU, or executes a program.  The production remote
store and the namespace allow-list are supplied by the operator; fixtures use a counting fake.
"""
import argparse
import json
import os
import time
from pathlib import Path

from pipelines.generated_target_admission_v1.common import canonical, need, sha
from jope_flywheel import native_variant_snapshot as V
from jope_flywheel import training_snapshot_kumos as K
from pipelines.generated_target_preparation_v2 import preparation as C

CACHE_SCHEMA = 'jope.c10a-closed-exact-cache/v1'
COMPLETE_SCHEMA = 'jope.c10a-closed-exact-cache-complete/v1'
QUALIFIED_SCHEMA = 'jope.c10a-closed-exact-cache-qualified/v1'
QUALIFIED_KEYS = frozenset({'schema', 'complete_sha256', 'qualified_after_durable_completion', 'bound'})
PRODUCTION_NAMESPACE = ('veox-jopedime-learning-use2', 'experiments/gp-policy-dl/v2-trainonly/')
MAX_OBJECT = 32 * 1024**2
MAX_CACHE = 6 * 1024**3


def _identity(ref, namespace):
    r = K.normalized_ref(ref)
    bucket, prefix = namespace
    need(r['bucket'] == bucket and type(r['key']) is str and r['key'].startswith(prefix) and len(r['key']) <= 4096,
         'reference outside the allowed namespace')
    need(type(r['version_id']) is str and r['version_id'] not in ('', 'null'), 'exact VersionId')
    need(type(r['bytes']) is int and 0 < r['bytes'] <= MAX_OBJECT, 'bounded object')
    return r, sha(canonical(r))


def _create(path, body):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Recorder:
    """Remote store wrapper for the dry run: verifies every exact object at first read and streams it create-only
    into the (lazily created) cache root, keeping only its reference in memory.  The cumulative declared size is
    checked against the cache cap BEFORE each new fetch, so a bounded worker never overshoots (memory or disk)."""

    def __init__(self, remote, namespace, previous=None, *, root, max_bytes=MAX_CACHE):
        need(type(max_bytes) is int and 0 < max_bytes <= MAX_CACHE, 'bounded cache cap')
        self.remote, self.namespace, self.previous, self.root, self.max_bytes = remote, namespace, previous, Path(root), max_bytes
        self.objects, self.declared, self.network_gets, self.network_bytes, self.previous_hits = {}, 0, 0, 0, 0
        self.nativecanonical_json = getattr(remote, 'nativecanonical_json', None)

    def get_version(self, ref):
        r, key = _identity(ref, self.namespace)
        path = self.root / (key + '.bin')
        if key in self.objects:
            body = path.read_bytes()
            need(len(body) == r['bytes'] and sha(body) == r['sha256'], 'recorded bytes corrupted')
            return body, r['version_id']
        need(self.declared + r['bytes'] <= self.max_bytes, 'closed cache cap would be exceeded')
        if self.previous is not None and self.previous.has(key):
            body, version = self.previous.get_version(r)
            self.previous_hits += 1
        else:
            body, version = self.remote.get_version(r)
            self.network_gets += 1
            self.network_bytes += len(body) if type(body) is bytes else 0
        need(type(body) is bytes and version == r['version_id'] and len(body) == r['bytes'] and
             sha(body) == r['sha256'], 'first exact-VersionId GET verification')
        if not self.root.exists():
            self.root.mkdir(mode=0o755)
        _create(path, body)
        self.declared += r['bytes']
        self.objects[key] = r
        return body, r['version_id']


def check_qualified(qualified_raw, *, complete_raw):
    """The cache completion marker: strict schema, bound IS the sealed complete.json clock bound, sample inside it."""
    q = json.loads(qualified_raw)
    need(type(q) is dict and set(q) == QUALIFIED_KEYS and q['schema'] == QUALIFIED_SCHEMA and
         q['complete_sha256'] == sha(complete_raw), 'qualified marker binds this completion')
    need(C.check_bound(q['bound']) == C.sealed_bound(json.loads(complete_raw).get('clock')),
         'qualified bound is the sealed completion clock bound')
    need(C.in_bound(q['qualified_after_durable_completion'], q['bound']),
         'closed cache completion qualified inside its original bound')
    return q


def accept_cache(cache_root, *, acceptance_path, physical):
    """OPERATOR-ONLY, after the prefetch worker returned/terminated: seal the cache acceptance (see
    ``preparation.accept_subject``); ``in_process_return`` pins the returned completion proof."""
    return C.accept_subject(
        'closed_cache', cache_root, marker_name='qualified.json', artifact_names=('manifest.json', 'complete.json'),
        marker_bound=lambda m, raws: check_qualified(m, complete_raw=raws['complete.json'])['bound'],
        returned=lambda raws: sha(raws['complete.json']), physical=physical, acceptance_path=acceptance_path)


class ClosedCache:
    """A complete, closed, ACCEPTED exact-version local store; reads outside the roster refuse.  Attests audited
    counters.  ``acceptance`` is the ``accept_cache`` envelope reference."""

    def __init__(self, root, namespace=PRODUCTION_NAMESPACE, *, canonicalizer, acceptance):
        need(callable(canonicalizer), 'pinned native canonicalizer required')
        self.root, self.namespace, self.nativecanonical_json = Path(root), tuple(namespace), canonicalizer
        need(self.root.is_absolute() and all(not p.is_symlink() for p in (self.root, *self.root.parents)),
             'absolute non-symlink cache root')
        raw = (self.root / 'manifest.json').read_bytes()
        done_raw = (self.root / 'complete.json').read_bytes()
        done = json.loads(done_raw)
        q_raw = (self.root / 'qualified.json').read_bytes()
        q = check_qualified(q_raw, complete_raw=done_raw)
        C.check_acceptance(acceptance, subject='closed_cache', root=self.root, marker_name='qualified.json',
                           marker_raw=q_raw, artifacts={'manifest.json': raw, 'complete.json': done_raw},
                           bound=q['bound'], returned_sha256=sha(done_raw))
        self.manifest = json.loads(raw)
        need(self.manifest['schema'] == CACHE_SCHEMA and tuple(self.manifest['namespace']) == self.namespace,
             'closed cache manifest')
        need(done['schema'] == COMPLETE_SCHEMA and done['state'] == 'complete' and done['manifest_sha256'] == sha(raw),
             'complete closed cache')
        self.refs = {}
        for r in self.manifest['refs']:
            ref, key = _identity(r, self.namespace)
            need(key not in self.refs, 'duplicate cached reference')
            self.refs[key] = ref
        self.manifest_sha256, self.reads, self.read_bytes = sha(raw), 0, 0

    def has(self, key):
        return key in self.refs

    def get_version(self, ref):
        r, key = _identity(ref, self.namespace)
        need(key in self.refs and self.refs[key] == r, 'reference not in the closed exact roster')
        path = self.root / (key + '.bin')
        need(not path.is_symlink() and path.is_file() and path.stat().st_size == r['bytes'], 'cached object file')
        body = path.read_bytes()
        need(len(body) == r['bytes'] and sha(body) == r['sha256'], 'cached bytes corrupted')
        self.reads += 1
        self.read_bytes += len(body)
        return body, r['version_id']

    def audit(self):
        return {'schema': C.STORE_AUDIT, 'closed_exact_version_local': True, 'network_gets': 0,
                'local_reads': self.reads, 'local_bytes': self.read_bytes}


def _canonicalizer(remote):
    fn = getattr(remote, 'nativecanonical_json', None)
    need(callable(fn), 'pinned native canonicalizer required')
    return fn


def prefetch(manifest_raw, *, expected_manifest_sha256, predecessor_raw, native, remote, native_spec_bytes,
             registry_bytes, preprocessing_bytes, objective_bytes, contract, export_provenance, authority_callback,
             labels, cache_root, deadline, namespace=PRODUCTION_NAMESPACE, previous_cache=None,
             previous_cache_acceptance=None, read_budget_bytes=512 * 1024**2, cache_cap_bytes=1024**3):
    """Phase 1: one recording dry run, then a fresh closed cache of exactly the objects it read.

    The same paired wall/monotonic/boot clock and one combined pre-read budget as preparation bound every read."""
    started = time.monotonic()
    clock = C.Clock(deadline)
    m = C.parse_manifest(manifest_raw, expected_sha256=expected_manifest_sha256)
    C.check_contract(m, contract=contract, export_provenance=export_provenance, preprocessing_bytes=preprocessing_bytes,
                     objective_bytes=objective_bytes, labels=labels)
    C.check_transition(m, predecessor_raw)
    cache_root = Path(cache_root)
    need(cache_root.is_absolute() and not cache_root.exists() and cache_root.parent.is_dir(), 'fresh cache root')
    canonicalizer = _canonicalizer(remote)
    previous = (ClosedCache(previous_cache, namespace, canonicalizer=canonicalizer,
                            acceptance=previous_cache_acceptance) if previous_cache is not None else None)
    recorder = Recorder(remote, tuple(namespace), previous, root=cache_root, max_bytes=cache_cap_bytes)
    budgeted = C.BudgetStore(recorder, budget=read_budget_bytes, clock=clock)
    extension = canonical(m['extension'])
    reader = V.Reader(budgeted, read_budget_bytes, deadline)
    for row in m['rows']:
        C.row_identity(reader, row)
        clock.tick('prefetch identity')
    V.compose(native, extension, m['extension_sha256'], budgeted, native_spec_bytes=native_spec_bytes,
              registry_bytes=registry_bytes, preprocessing_bytes=preprocessing_bytes,
              authority_callback=authority_callback, deadline=deadline)
    for row in m['rows']:
        C._y_and_baselines(reader, row)
    clock.tick('prefetch before cache write')
    total = recorder.declared
    need(0 < len(recorder.objects) <= 100000 and total <= cache_cap_bytes, 'bounded closed cache')
    manifest = {'schema': CACHE_SCHEMA, 'namespace': list(namespace), 'x1_manifest_sha256': expected_manifest_sha256,
                'refs': [recorder.objects[k] for k in sorted(recorder.objects)], 'declared_bytes': total,
                'previous_cache_manifest_sha256': previous.manifest_sha256 if previous else None}
    manifest_raw_cache = canonical(manifest)
    _create(cache_root / 'manifest.json', manifest_raw_cache)
    clock.tick('prefetch before completion marker')
    proof = {'schema': COMPLETE_SCHEMA, 'state': 'complete', 'manifest_sha256': sha(manifest_raw_cache),
             'objects': len(recorder.objects), 'bytes': total, 'network_gets': recorder.network_gets,
             'network_bytes': recorder.network_bytes, 'previous_cache_hits': recorder.previous_hits,
             'wall_seconds': time.monotonic() - started, 's3_writes': 0, 'clock': clock.bound(),
             'read_budget': {'bytes': read_budget_bytes, 'charged': budgeted.charged},
             'verification': 'every object exact VersionId, length and SHA-256 at first read; local reads re-verify'}
    complete_raw = canonical(proof)
    _create(cache_root / 'complete.json', complete_raw)
    _fsync_dir(cache_root)
    qualified = clock.qualify('after durable completion marker')
    _create(cache_root / 'qualified.json', canonical({'schema': QUALIFIED_SCHEMA, 'complete_sha256': sha(complete_raw),
                                                      'qualified_after_durable_completion': qualified,
                                                      'bound': clock.bound()}))
    _fsync_dir(cache_root)
    clock.qualify('after final qualified publication')   # a late final publication never returns success
    return proof


def prepare(manifest_raw, *, cache_root, cache_acceptance, canonicalizer, namespace=PRODUCTION_NAMESPACE, **kwargs):
    """Phase 2: closed zero-network preparation from an ACCEPTED cache; ``kwargs`` are ``preparation.prepare_root``
    arguments minus store."""
    store = ClosedCache(cache_root, namespace, canonicalizer=canonicalizer, acceptance=cache_acceptance)
    receipt = C.prepare_root(manifest_raw, store=store, **kwargs)
    need(receipt['store_audit']['delta']['network_gets'] == 0, 'closed preparation performed no network GET')
    return receipt


def main(argv=None):  # pragma: no cover - operator entry point; production wiring is supplied by the owner
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('phase', choices=('prefetch', 'prepare'))
    parser.add_argument('--config', type=Path, required=True,
                        help='owner-reviewed JSON naming every pinned input path/sha, the cache root, the new root, '
                             'the deadline and the remote store adapter; not defined by this library')
    args = parser.parse_args(argv)
    raise SystemExit('operator wiring (remote store, native snapshot, authority) is supplied by the live owner; '
                     'this library exposes prefetch() and prepare() only: ' + str(args.config))


if __name__ == '__main__':  # pragma: no cover
    main()
