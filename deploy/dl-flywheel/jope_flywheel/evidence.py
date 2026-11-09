"""Bounded, transactional evidence; no trainer, network, environment or implicit path reads.

API::

    with Ledger('owned/run.sqlite') as ledger:
        ledger.register_run('run-1', {'source_sha256': source_digest, ...})
        ledger.append('step-1', 'optimizer.step', {'loss': 0.2}, 'run-1')
        ledger.append_imported('old-1', 'epoch.summary', payload, 'run-1', receipt_sha)
        seal = ledger.seal('run-1', {'checkpoint': 'model.pt'})
        ledger.verify(expected_head=seal['event_hash'])
        ledger.export_jsonl('events.jsonl')
        ledger.export_csv('events.csv')

The default owned root is the database parent. Export and seal paths must be under
that root, without symlinks. Payload paths are never opened. ``seal`` hashes only
explicit regular-file arguments, freezes the run, and binds exact bytes at read
time; it cannot prevent later external mutation of those files. ``verify`` checks
the database, not referenced files. ``verify_artifacts(run_id)`` explicitly reads
the sealed artifacts again. Export JSONL contains every hash-bearing event field.

UTC is observation time, never a backdated event claim. Monotonic nanoseconds are
relative to an explicit per-Ledger clock origin, not UTC and not comparable across
origins. Imported aggregates are labeled as such and cannot claim native batch
order. This is provenance, not proof that a caller's reported measurement is true.

SQLite WAL/FULL and BEGIN IMMEDIATE make each operation atomic across processes.
Retries of the same event ID and content return the original event; different
content is a collision. A trusted external head/count detects tail truncation;
a hash chain alone cannot detect an attacker rewriting the entire local store.
Canonical JSON here means sorted keys, compact UTF-8, finite JSON primitives;
it is a versioned Python representation, not a claim of RFC 8785 compliance.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
import time
import uuid

SCHEMA = 'jope.flywheel-evidence/v1'
ZERO = '0' * 64
MAX_JSON_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_DEPTH = 32
MAX_ARTIFACTS = 128
MAX_ARTIFACT_BYTES = 8 * 1024**3
MAX_EVENTS = 1_000_000
MAX_EXPORT_BYTES = 512 * 1024**2
_IDENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_INTERNAL = 'ledger:'
_ORDER_KEYS = frozenset({'batch_order', 'batch_order_sha256', 'ordered_batch_sha256',
                         'ordered_row_ids', 'ordered_program_ids', 'optimizer_step_native'})


class EvidenceError(ValueError):
    """Invalid evidence, identity, path or lifecycle request."""


class CollisionError(EvidenceError):
    """An immutable identity already names different content."""


class IntegrityError(EvidenceError):
    """Stored evidence fails its declared chain or manifest binding."""


class LimitError(EvidenceError):
    """An operation exceeds a declared read/write bound."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENT.fullmatch(value):
        raise EvidenceError('identifier must be 1..128 safe ASCII characters')
    return value


def _digest(value: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise EvidenceError('expected lowercase SHA-256')
    return value


def _tree(value, depth=0):
    if depth > MAX_DEPTH:
        raise LimitError('JSON nesting exceeds limit')
    if value is None or type(value) in (str, bool):
        return
    if type(value) is int:
        if value.bit_length() > 256:
            raise LimitError('JSON integer exceeds 256 bits')
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise EvidenceError('nonfinite number')
        return
    if type(value) is list:
        for x in value:
            _tree(x, depth + 1)
        return
    if type(value) is dict:
        for k, x in value.items():
            if type(k) is not str:
                raise EvidenceError('JSON object keys must be strings')
            _tree(x, depth + 1)
        return
    raise EvidenceError('only JSON primitives are accepted')


def canonical_json(value, *, limit=MAX_JSON_BYTES) -> str:
    """Validate strict finite JSON and return its bounded canonical text."""
    _tree(value)
    try:
        text = json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False)
        size = len(text.encode('utf-8'))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise EvidenceError('invalid JSON encoding') from exc
    if size > limit:
        raise LimitError('JSON bytes exceed limit')
    return text


def _load(text, *, limit=MAX_JSON_BYTES):
    if not isinstance(text, str) or len(text.encode('utf-8')) > limit:
        raise LimitError('stored JSON exceeds limit')
    try:
        value = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(EvidenceError('nonfinite JSON')))
    except (ValueError, RecursionError) as exc:
        raise IntegrityError('invalid stored JSON') from exc
    if canonical_json(value, limit=limit) != text:
        raise IntegrityError('stored JSON is not canonical')
    return value


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _check_utc(value):
    try:
        parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
        if not value.endswith('Z') or parsed.utcoffset() != dt.timedelta(0):
            raise ValueError()
    except (ValueError, TypeError, AttributeError) as exc:
        raise IntegrityError('invalid UTC observation time') from exc


def _no_imported_order(payload):
    if isinstance(payload, dict):
        if _ORDER_KEYS.intersection(payload):
            raise EvidenceError('an imported aggregate cannot prove native batch order')
        for v in payload.values():
            _no_imported_order(v)
    elif isinstance(payload, list):
        for v in payload:
            _no_imported_order(v)


class Ledger:
    """Owned SQLite evidence store. All public mutations are committed atomically.

    ``register_run`` requires a source_sha256 identifying the pinned source-file
    manifest or source bundle (a Git SHA may be included as a separate field).
    ``append`` accepts a JSON object payload, and returns the original full event
    on an idempotent retry. ``seal`` closes one run, not other runs in the ledger.
    ``verify`` returns the whole-ledger head/count and per-run state; pass an
    independently retained anchor when historical tail completeness matters.
    """

    def __init__(self, db, *, root=None, busy_timeout_ms=10_000):
        if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 60_000:
            raise EvidenceError('busy timeout out of range')
        path = Path(db).absolute()
        self._input_root = Path(root).absolute() if root is not None else path.parent
        self.root = self._input_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._closed = False
        self._db_relative = self._relative(path)
        path = self.root / self._db_relative
        self.db_path = path
        self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            parent_fd, filename = self._parent(self._db_relative)
            try:
                for candidate in (filename, filename + '-wal', filename + '-shm'):
                    try:
                        info = os.stat(candidate, dir_fd=parent_fd, follow_symlinks=False)
                        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                            raise EvidenceError('database and sidecars must be unshared regular files')
                    except FileNotFoundError:
                        pass
            finally:
                os.close(parent_fd)
            self._protected = {self._db_relative, self._db_relative + '-wal', self._db_relative + '-shm'}
            self._connection = sqlite3.connect(str(path), timeout=busy_timeout_ms / 1000,
                                               isolation_level=None, check_same_thread=False)
        except BaseException:
            os.close(self._root_fd)
            self._closed = True
            raise
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute('PRAGMA foreign_keys=ON')
            self._connection.execute('PRAGMA journal_mode=WAL')
            self._connection.execute('PRAGMA synchronous=FULL')
            self._connection.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')
            tables = {r[0] for r in self._connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if tables - {'metadata', 'runs', 'events'}:
                raise IntegrityError('database contains unrelated tables')
            self._connection.executescript(f'''
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL CHECK(length(value)<=64));
                CREATE TABLE IF NOT EXISTS runs(
                  run_id TEXT PRIMARY KEY CHECK(length(run_id)<=128),
                  manifest_json TEXT NOT NULL CHECK(length(CAST(manifest_json AS BLOB))<={MAX_MANIFEST_BYTES}),
                  manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256)=64),
                  source_sha256 TEXT NOT NULL CHECK(length(source_sha256)=64),
                  seal_event_id TEXT);
                CREATE TABLE IF NOT EXISTS events(
                  seq INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE CHECK(length(event_id)<=160),
                  run_id TEXT NOT NULL REFERENCES runs(run_id),
                  body_json TEXT NOT NULL CHECK(length(CAST(body_json AS BLOB))<={MAX_JSON_BYTES}),
                  event_hash TEXT NOT NULL UNIQUE CHECK(length(event_hash)=64));
            ''')
            self._connection.execute('BEGIN IMMEDIATE')
            meta = dict(self._connection.execute('SELECT key,value FROM metadata'))
            if not meta:
                if self._connection.execute('SELECT 1 FROM runs LIMIT 1').fetchone() or self._connection.execute('SELECT 1 FROM events LIMIT 1').fetchone():
                    raise IntegrityError('ledger metadata was removed')
                self._connection.execute('INSERT INTO metadata VALUES (?,?)', ('schema', SCHEMA))
                self._connection.execute('INSERT INTO metadata VALUES (?,?)', ('ledger_id', uuid.uuid4().hex))
                meta = dict(self._connection.execute('SELECT key,value FROM metadata'))
            if meta.get('schema') != SCHEMA or set(meta) != {'schema', 'ledger_id'}:
                raise IntegrityError('unsupported ledger metadata')
            self.ledger_id = meta['ledger_id']
            self._connection.commit()
            self.origin = {'id': uuid.uuid4().hex, 'kind': 'observer_session',
                           'utc': _utc(), 'monotonic_ns': time.monotonic_ns()}
            self.verify()
        except BaseException:
            self._connection.rollback()
            self._connection.close()
            os.close(self._root_fd)
            self._closed = True
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._mutex:
            if not self._closed:
                self._connection.close()
                os.close(self._root_fd)
                self._closed = True

    def _relative(self, path):
        path = Path(path)
        if path.is_absolute():
            try:
                path = path.relative_to(self._input_root)
            except ValueError as exc:
                try:
                    path = path.relative_to(self.root)
                except ValueError:
                    raise EvidenceError('path is outside owned root') from exc
        if not path.parts or any(p in ('.', '..') for p in path.parts):
            raise EvidenceError('path must name an owned file')
        return path.as_posix()

    def _parent(self, relative):
        parts = Path(relative).parts
        fd = os.dup(self._root_fd)
        try:
            for part in parts[:-1]:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = nxt
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise

    def _run(self, run_id):
        row = self._connection.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            raise EvidenceError('run must be registered first')
        return row

    def _decode(self, row):
        body = _load(row['body_json'])
        if _sha(row['body_json'].encode()) != row['event_hash']:
            raise IntegrityError('event content hash mismatch')
        if body.get('seq') != row['seq'] or body.get('event_id') != row['event_id'] or body.get('run_id') != row['run_id']:
            raise IntegrityError('event index mismatch')
        return dict(body, event_hash=row['event_hash'])

    def _append(self, event_id, event_type, payload, run_id, evidence_kind):
        # Caller owns BEGIN IMMEDIATE. Keep timestamps out of retry equivalence.
        payload_text = canonical_json(payload)
        old = self._connection.execute('SELECT * FROM events WHERE event_id=?', (event_id,)).fetchone()
        if old:
            event = self._decode(old)
            if (event['run_id'], event['event_type'], event['evidence_kind'], canonical_json(event['payload'])) != (run_id, event_type, evidence_kind, payload_text):
                raise CollisionError('event ID already names different content')
            return event
        run = self._run(run_id)
        if run['seal_event_id'] is not None:
            raise EvidenceError('run is sealed')
        tip = self._connection.execute('SELECT * FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        prev = self._decode(tip) if tip else None
        if prev and prev['seq'] >= MAX_EVENTS:
            raise LimitError('event count would exceed verification limit')
        body = {'schema': SCHEMA, 'ledger_id': self.ledger_id,
                'seq': prev['seq'] + 1 if prev else 1, 'event_id': event_id,
                'run_id': run_id, 'event_type': event_type, 'evidence_kind': evidence_kind,
                'manifest_sha256': run['manifest_sha256'], 'source_sha256': run['source_sha256'],
                'observed_at_utc': _utc(), 'time_semantics': 'observation_time',
                'clock_origin': self.origin, 'monotonic_ns': time.monotonic_ns(),
                'prev_hash': prev['event_hash'] if prev else ZERO, 'payload': payload}
        text = canonical_json(body)
        digest = _sha(text.encode())
        self._connection.execute('INSERT INTO events VALUES (?,?,?,?,?)',
                                 (body['seq'], event_id, run_id, text, digest))
        return dict(body, event_hash=digest)

    def register_run(self, run_id, manifest):
        """Atomically register immutable JSON manifest and its source SHA-256."""
        _identifier(run_id)
        if type(manifest) is not dict:
            raise EvidenceError('manifest must be an object')
        source = _digest(manifest.get('source_sha256'))
        raw = canonical_json(manifest, limit=MAX_MANIFEST_BYTES)
        with self._mutex:
            self._connection.execute('BEGIN IMMEDIATE')
            try:
                old = self._connection.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()
                if old and old['manifest_json'] != raw:
                    raise CollisionError('run manifest is immutable')
                if not old:
                    self._connection.execute('INSERT INTO runs VALUES (?,?,?,?,NULL)', (run_id, raw, _sha(raw.encode()), source))
                result = self._append('ledger:run:' + run_id, 'ledger.run_registered',
                                      {'manifest': manifest}, run_id, 'native')
                self._connection.commit()
                return result
            except BaseException:
                self._connection.rollback()
                raise

    def append(self, event_id, event_type, payload, run_id, *, evidence_kind='native'):
        """Append one observed event; exact retry returns its original timestamp/hash.

        ``native`` means recorded through this API at observation time, not an
        independent attestation. Use ``append_imported`` for historical summaries.
        Internal ledger event namespaces are reserved.
        """
        for x in (event_id, event_type, run_id):
            _identifier(x)
        if event_id.startswith(_INTERNAL) or event_type.startswith('ledger.'):
            raise EvidenceError('reserved ledger namespace')
        if type(payload) is not dict or evidence_kind not in ('native', 'imported_aggregate'):
            raise EvidenceError('invalid payload or evidence kind')
        canonical_json(payload)
        if evidence_kind == 'imported_aggregate':
            _digest(payload.get('source_receipt_sha256'))
            _no_imported_order(payload)
        with self._mutex:
            self._connection.execute('BEGIN IMMEDIATE')
            try:
                result = self._append(event_id, event_type, payload, run_id, evidence_kind)
                self._connection.commit()
                return result
            except BaseException:
                self._connection.rollback()
                raise

    def append_imported(self, event_id, event_type, payload, run_id, source_receipt_sha256):
        """Record a historical aggregate without inventing original batch order."""
        _digest(source_receipt_sha256)
        return self.append(event_id, event_type,
                           {'source_receipt_sha256': source_receipt_sha256, 'aggregate': payload},
                           run_id, evidence_kind='imported_aggregate')

    def _hash_artifact(self, artifact_id, path, *, byte_limit=MAX_ARTIFACT_BYTES):
        _identifier(artifact_id)
        relative = self._relative(path)
        if relative in self._protected:
            raise EvidenceError('ledger files cannot be sealed as artifacts')
        parent, filename = self._parent(relative)
        try:
            fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise EvidenceError('artifact must be a regular file')
            byte_limit = min(byte_limit, MAX_ARTIFACT_BYTES)
            if before.st_size > byte_limit:
                raise LimitError('artifact exceeds size limit')
            h = hashlib.sha256(); count = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > byte_limit:
                    raise LimitError('artifact exceeds size limit')
                h.update(chunk)
            after = os.fstat(fd)
            signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if signature(before) != signature(after) or count != before.st_size:
                raise IntegrityError('artifact changed during hashing')
            return {'artifact_id': artifact_id, 'path': relative, 'bytes': count,
                    'sha256': h.hexdigest(), 'verification': 'exact_stream_read'}
        finally:
            os.close(fd)

    def seal(self, run_id, artifacts):
        """Hash explicit owned files and atomically close a run with exact digests.

        At most 128 files and 8 GiB total. Files may change after this point;
        use ``verify_artifacts`` for a later explicit readback. An identical retry
        is allowed; different content under the same seal ID is refused.
        """
        _identifier(run_id)
        if type(artifacts) is not dict or not 0 < len(artifacts) <= MAX_ARTIFACTS:
            raise LimitError('seal requires 1..128 named artifacts')
        with self._mutex:
            self._connection.execute('BEGIN IMMEDIATE')
            try:
                self._verify()
                # This write transaction also serializes artifact hashing with
                # export replacement by other Ledger instances/processes.
                items = []; remaining = MAX_ARTIFACT_BYTES
                for key, value in sorted(artifacts.items()):
                    item = self._hash_artifact(key, value, byte_limit=remaining)
                    items.append(item); remaining -= item['bytes']
                payload = {'artifacts': items, 'artifacts_sha256': _sha(canonical_json(items).encode())}
                result = self._append('ledger:seal:' + run_id, 'ledger.run_sealed', payload, run_id, 'native')
                self._connection.execute('UPDATE runs SET seal_event_id=? WHERE run_id=?', (result['event_id'], run_id))
                self._connection.commit()
                return result
            except BaseException:
                self._connection.rollback()
                raise

    def _verify(self, *, expected_head=None, expected_count=None):
        if self._connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise IntegrityError('SQLite quick_check failed')
        if self._connection.execute('PRAGMA foreign_key_check').fetchone():
            raise IntegrityError('foreign key violation')
        meta = dict(self._connection.execute('SELECT key,value FROM metadata'))
        if meta != {'schema': SCHEMA, 'ledger_id': self.ledger_id}:
            raise IntegrityError('ledger metadata changed')
        runs = {}
        for row in self._connection.execute('SELECT * FROM runs'):
            if len(runs) >= MAX_EVENTS:
                raise LimitError('run count exceeds verification limit')
            _identifier(row['run_id'])
            manifest = _load(row['manifest_json'], limit=MAX_MANIFEST_BYTES)
            if _sha(row['manifest_json'].encode()) != row['manifest_sha256'] or manifest.get('source_sha256') != row['source_sha256']:
                raise IntegrityError('run manifest identity mismatch')
            _digest(row['source_sha256'])
            runs[row['run_id']] = {'row': row, 'registered': False, 'sealed': None}
        count = 0; head = ZERO; clocks = {}
        required = {'schema','ledger_id','seq','event_id','run_id','event_type','evidence_kind',
                    'manifest_sha256','source_sha256','observed_at_utc','time_semantics',
                    'clock_origin','monotonic_ns','prev_hash','payload','event_hash'}
        for row in self._connection.execute('SELECT * FROM events ORDER BY seq'):
            count += 1
            if count > MAX_EVENTS:
                raise LimitError('event count exceeds verification limit')
            e = self._decode(row)
            if set(e) != required or e['schema'] != SCHEMA or e['ledger_id'] != self.ledger_id or type(e['seq']) is not int or e['seq'] != count or e['prev_hash'] != head:
                raise IntegrityError('event schema, sequence or chain mismatch')
            run = runs[e['run_id']]
            if e['manifest_sha256'] != run['row']['manifest_sha256'] or e['source_sha256'] != run['row']['source_sha256']:
                raise IntegrityError('event manifest binding mismatch')
            _check_utc(e['observed_at_utc'])
            clock = e['clock_origin']
            if (not isinstance(clock, dict) or set(clock) != {'id','kind','utc','monotonic_ns'} or
                clock['kind'] != 'observer_session' or type(clock['monotonic_ns']) is not int or
                clock['monotonic_ns'] < 0 or type(e['monotonic_ns']) is not int or
                e['monotonic_ns'] < clock['monotonic_ns'] or e['time_semantics'] != 'observation_time'):
                raise IntegrityError('clock origin or observation semantics mismatch')
            _identifier(clock['id']); _check_utc(clock['utc'])
            old = clocks.get(clock['id'])
            if old and (old[0] != clock or e['monotonic_ns'] < old[1]):
                raise IntegrityError('monotonic clock changed within origin')
            clocks[clock['id']] = (clock, e['monotonic_ns'])
            if e['evidence_kind'] not in ('native','imported_aggregate') or type(e['payload']) is not dict:
                raise IntegrityError('invalid event provenance')
            if e['evidence_kind'] == 'imported_aggregate':
                _digest(e['payload'].get('source_receipt_sha256')); _no_imported_order(e['payload'])
            if run['sealed']:
                raise IntegrityError('event follows sealed run')
            if e['event_type'] == 'ledger.run_registered':
                if run['registered'] or e['event_id'] != 'ledger:run:' + e['run_id'] or e['evidence_kind'] != 'native' or canonical_json(e['payload'].get('manifest'), limit=MAX_MANIFEST_BYTES) != run['row']['manifest_json']:
                    raise IntegrityError('registration mismatch')
                run['registered'] = True
            elif not run['registered']:
                raise IntegrityError('event precedes registration')
            elif e['event_type'] == 'ledger.run_sealed':
                p = e['payload']; items = p.get('artifacts')
                if e['event_id'] != 'ledger:seal:' + e['run_id'] or e['evidence_kind'] != 'native' or type(items) is not list or not 0 < len(items) <= MAX_ARTIFACTS or _sha(canonical_json(items).encode()) != p.get('artifacts_sha256'):
                    raise IntegrityError('artifact seal mismatch')
                seen = set(); size = 0
                for item in items:
                    if set(item) != {'artifact_id','path','bytes','sha256','verification'} or item['verification'] != 'exact_stream_read' or type(item['bytes']) is not int or item['bytes'] < 0:
                        raise IntegrityError('invalid artifact descriptor')
                    _identifier(item['artifact_id']); _digest(item['sha256'])
                    if item['artifact_id'] in seen or self._relative(item['path']) in self._protected:
                        raise IntegrityError('duplicate or protected artifact')
                    seen.add(item['artifact_id']); size += item['bytes']
                if size > MAX_ARTIFACT_BYTES:
                    raise LimitError('sealed artifacts exceed limit')
                run['sealed'] = e['event_id']
            elif e['event_type'].startswith('ledger.') or e['event_id'].startswith(_INTERNAL):
                raise IntegrityError('unknown internal event')
            head = e['event_hash']
        for run in runs.values():
            if not run['registered'] or run['sealed'] != run['row']['seal_event_id']:
                raise IntegrityError('run lifecycle mismatch')
        if expected_head is not None and _digest(expected_head) != head:
            raise IntegrityError('external head anchor mismatch')
        if expected_count is not None and (type(expected_count) is not int or expected_count != count):
            raise IntegrityError('external count anchor mismatch')
        return {'schema': SCHEMA, 'ledger_id': self.ledger_id, 'event_count': count,
                'head_sha256': head, 'runs': {k: {'manifest_sha256': v['row']['manifest_sha256'],
                    'sealed': bool(v['sealed'])} for k,v in runs.items()}}

    def verify(self, *, expected_head=None, expected_count=None):
        """Verify a consistent read snapshot, optionally against external anchors."""
        with self._mutex:
            self._connection.execute('BEGIN')
            try:
                result = self._verify(expected_head=expected_head, expected_count=expected_count)
                self._connection.commit()
                return result
            except BaseException:
                self._connection.rollback()
                raise

    def verify_artifacts(self, run_id):
        """Explicitly reread only the run's owned, sealed regular-file paths."""
        with self._mutex:
            self.verify()
            run = self._run(run_id)
            if not run['seal_event_id']:
                raise EvidenceError('run is not sealed')
            event = self._decode(self._connection.execute('SELECT * FROM events WHERE event_id=?', (run['seal_event_id'],)).fetchone())
            for item in event['payload']['artifacts']:
                if self._hash_artifact(item['artifact_id'], item['path']) != item:
                    raise IntegrityError('sealed artifact bytes changed')
            return event['payload']

    def _export(self, path, *, csv_format):
        relative = self._relative(path)
        if relative in self._protected:
            raise EvidenceError('export cannot replace ledger files')
        parent, filename = self._parent(relative)
        temporary = '.' + filename + '.' + uuid.uuid4().hex + '.tmp'
        count_bytes = 0
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            with self._mutex, os.fdopen(fd, 'w', encoding='utf-8', newline='') as out:
                self._connection.execute('BEGIN IMMEDIATE')
                try:
                    anchor = self._verify()
                    for sealed in self._connection.execute('SELECT e.* FROM events e JOIN runs r ON r.seal_event_id=e.event_id'):
                        if any(a['path'] == relative for a in self._decode(sealed)['payload']['artifacts']):
                            raise EvidenceError('export cannot replace a sealed artifact')
                    if csv_format:
                        out.write('seq,event_id,run_id,event_type,evidence_kind,event_hash,event_json\n')
                    for row in self._connection.execute('SELECT * FROM events ORDER BY seq'):
                        event = self._decode(row); encoded = canonical_json(event, limit=MAX_JSON_BYTES + 128)
                        if csv_format:
                            buff = io.StringIO(newline=''); writer = csv.writer(buff, lineterminator='\n')
                            writer.writerow([event[k] for k in ('seq','event_id','run_id','event_type','evidence_kind','event_hash')] + [encoded])
                            line = buff.getvalue()
                        else:
                            line = encoded + '\n'
                        count_bytes += len(line.encode())
                        if count_bytes > MAX_EXPORT_BYTES:
                            raise LimitError('export exceeds byte ceiling')
                        out.write(line)
                    out.flush(); os.fsync(out.fileno())
                    # Keep the write lock through replacement: a concurrent
                    # seal must hash either the old protected file or this one.
                    os.replace(temporary, filename, src_dir_fd=parent, dst_dir_fd=parent)
                    os.fsync(parent)
                    self._connection.commit()
                except BaseException:
                    self._connection.rollback()
                    raise
            return dict(anchor, path=relative, format='csv' if csv_format else 'jsonl')
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    def export_jsonl(self, path):
        """Atomically export full verifiable events under the owned root."""
        return self._export(path, csv_format=False)

    def export_csv(self, path):
        """Atomically export indexed fields plus complete canonical event JSON."""
        return self._export(path, csv_format=True)
