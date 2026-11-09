"""Bounded, host-local evidence capture; never trains, loads data or copies weights.

Run with Python 3.10+ stdlib on POSIX under the campaign's external resource
envelope, for example ``python -m jope_flywheel.paper_archive --help``. The
caller supplies the trusted campaign digest. Source paths are the fixed
allowlist derived from that manifest; every path component refuses symlinks.

CAS bytes are source observations, not independent validation of their claims.
In particular, copied exposure/batch logs do not establish native batch order.
The ledger supplies import observation times; timestamps within copied files
remain source-reported times. The archive neither follows checkpoint references
nor accesses data arrays, credentials, environment variables or the network.

An interrupted import may leave an unreferenced immutable blob. Cursor recovery
uses committed ledger events, never a mutable offset cache. A 512 MiB total
archive ceiling and a smaller blob budget preserve room for bounded metadata.
Capacity/time limits produce an explicitly incomplete summary, never a claim
of complete scientific evidence. External service deadline must be the pinned
training service deadline + 900 seconds; this worker also enforces that limit.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
import uuid

from .evidence import Ledger, EvidenceError, IntegrityError, LimitError, canonical_json

MIB = 1024 * 1024
TOTAL_CAP = 512 * MIB
BLOB_CAP = 464 * MIB
SMALL_CAP = 4 * MIB
LARGE_CAP = 64 * MIB
CHUNK_CAP = MIB
CYCLE_READ_CAP = 512 * MIB
CYCLE_SECONDS = 40
EVENT_CAP = 4000
SCHEMA = 'jope.wave4-paper-archive/v1'
FOLDS = tuple(f'r{r}f{k}' for r in range(3) for k in range(5))
FAIL_STATES = frozenset(('failed', 'blocked', 'deadline', 'admission_deadline', 'service_deadline'))
FILES = frozenset(('archive.json', 'archive.lock', 'evidence.sqlite', 'evidence.sqlite-wal',
                   'evidence.sqlite-shm', 'events.jsonl', 'events.csv', 'evidence_summary.json'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    if type(value) is not str or not re.fullmatch(r'[a-f0-9]{64}', value):
        raise EvidenceError('invalid SHA-256')
    return value


def token(value):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', value):
        raise EvidenceError('invalid path identity')
    return value


def absolute(path):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or '\x00' in str(p):
        raise EvidenceError('absolute confined path required')
    return p


def open_dir(path):
    """Open every absolute directory component without following symlinks."""
    p = absolute(path)
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in p.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_file(path, limit):
    p = absolute(path); parent = open_dir(p.parent)
    try:
        fd = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
        os.close(fd)
        raise LimitError('source must be a regular single-link file within size bound')
    return fd, info


def json_value(raw):
    def reject(x):
        raise EvidenceError('non-finite JSON')
    def pairs(xs):
        result = {}
        for key, value in xs:
            if key in result:
                raise EvidenceError('duplicate JSON key')
            result[key] = value
        return result
    def number(x):
        result = float(x)
        if not math.isfinite(result):
            raise EvidenceError('non-finite JSON')
        return result
    try:
        return json.loads(raw, parse_constant=reject, parse_float=number, object_pairs_hook=pairs)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EvidenceError('invalid source JSON') from exc


def initial_read(path, limit):
    fd, before = open_file(path, limit)
    try:
        raw = bytearray()
        while True:
            chunk = os.read(fd, min(MIB, limit + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > limit:
                raise LimitError('source grew beyond limit')
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise IntegrityError('source changed while read; retry next cycle')
        return bytes(raw)
    finally:
        os.close(fd)


class Archive:
    """One exclusive writer for one pinned campaign/host and owned output root.

    ``cycle()`` captures bounded current evidence and atomically publishes the
    summary last. ``close()`` releases the archive lock. Resuming verifies the
    ledger and all referenced CAS bytes, then recovers log offsets from it.
    No source file is opened unless it is on the manifest-derived allowlist.
    """
    def __init__(self, campaign_manifest, manifest_sha, host, out):
        self.manifest_path = absolute(campaign_manifest)
        self.manifest_sha = digest(manifest_sha)
        self.raw_manifest = initial_read(self.manifest_path, SMALL_CAP)
        if sha(self.raw_manifest) != self.manifest_sha:
            raise IntegrityError('campaign manifest digest mismatch')
        self.m = json_value(self.raw_manifest)
        self.host = host
        if (self.m.get('schema') != 'jope.dl-wave4/v1' or self.m.get('wave') != 4
                or host not in ('xbabe2', 'xbabe3') or host not in self.m.get('hosts', {})):
            raise EvidenceError('unsupported campaign/host')
        self.campaign = token(self.m.get('campaign_id'))
        digest(self.m.get('bundle_sha')); digest(self.m.get('source_archive_sha256'))
        if not re.fullmatch(r'[a-f0-9]{40}', self.m.get('source_sha', '')):
            raise EvidenceError('source git identity missing')
        self.deadline = self.m.get('service_deadline')
        if type(self.deadline) not in (float, int) or not math.isfinite(self.deadline):
            raise EvidenceError('finite service deadline required')
        self.deadline += 900
        if time.time() >= self.deadline:
            raise LimitError('archive observation deadline already passed')
        self.root = absolute(self.m['hosts'][host]['root'])
        self.runtime = absolute(self.m['hosts'][host]['runtime'])
        self.jobs = [j for j in self.m.get('jobs', []) if j.get('host') == host]
        if not 1 <= len(self.jobs) <= 3:
            raise EvidenceError('same-host campaign must have 1..3 jobs')
        if len({token(j.get('run_id')) for j in self.jobs}) != len(self.jobs):
            raise EvidenceError('duplicate job')
        for j in self.jobs:
            if j.get('arm') != 'V1' or type(j.get('seed')) is not int or j['seed'] not in (3, 4, 5):
                raise EvidenceError('unexpected Wave4 job')
        arms = [a for a in self.m.get('arms', []) if a.get('host') == host and a.get('id') == 'V1']
        if len(arms) != 1 or type(arms[0].get('configuration')) is not dict:
            raise EvidenceError('unique pinned host arm required')
        self.arm = arms[0]
        self.qual_path = absolute(self.m['gpu_qualification']['path'])
        digest(self.m['gpu_qualification']['sha256'])
        self.out = absolute(out)
        parent = open_dir(self.out.parent)
        try:
            try:
                os.mkdir(self.out.name, 0o700, dir_fd=parent)
            except FileExistsError:
                pass
        finally:
            os.close(parent)
        self.fd = open_dir(self.out)
        self.ledger = None; self.lock_fd = None
        try:
            self.lock_fd = os.open('archive.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
            if not stat.S_ISREG(os.fstat(self.lock_fd).st_mode) or os.fstat(self.lock_fd).st_nlink != 1:
                raise EvidenceError('archive lock must be regular single-link file')
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker = {'schema': SCHEMA, 'manifest_sha256': self.manifest_sha, 'host': host}
            existing = set(os.listdir(self.fd))
            if 'archive.json' in existing:
                if json_value(initial_read(self.out / 'archive.json', SMALL_CAP)) != marker:
                    raise IntegrityError('archive is owned by a different campaign/host')
            elif existing != {'archive.lock'}:
                raise EvidenceError('output is not a new or marked archive')
            else:
                self._atomic('archive.json', canonical_json(marker).encode() + b'\n')
            # Only abandoned temporary names owned by this writer are removed.
            # A crash just after CAS link publication may leave a second hardlink
            # here; removing it restores the immutable blob's single-link state.
            for name in os.listdir(self.fd):
                if re.fullmatch(r'\.archive-[a-f0-9]{32}\.tmp', name):
                    if not stat.S_ISREG(os.stat(name, dir_fd=self.fd, follow_symlinks=False).st_mode):
                        raise IntegrityError('invalid abandoned temporary file')
                    os.unlink(name, dir_fd=self.fd)
            try:
                os.mkdir('blobs', 0o700, dir_fd=self.fd)
            except FileExistsError:
                pass
            self._usage()
            self.ledger = Ledger(self.out / 'evidence.sqlite')
            # Limit SQLite itself as well as checking archive-wide disk use.
            self.ledger._connection.execute('PRAGMA max_page_count=4096')
            self.run_id = 'archive-' + host + '-' + self.campaign
            self.ledger.register_run(self.run_id, {
                'source_sha256': self.m['source_archive_sha256'], 'schema': SCHEMA,
                'campaign_manifest_sha256': self.manifest_sha, 'host': host,
                'source_git_sha': self.m['source_sha'], 'bundle_sha256': self.m['bundle_sha'],
                'claims': 'imported source observations; no independent scientific validation'})
            self.read_bytes = 0; self.cycle_end = time.monotonic() + CYCLE_SECONDS
            self.records = []; self.cursors = {}; self._recover()
            self._capture_bytes('campaign/manifest.json', self.manifest_path, self.raw_manifest, 'snapshot')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.ledger is not None:
            self.ledger.close(); self.ledger = None
        if self.lock_fd is not None:
            os.close(self.lock_fd); self.lock_fd = None
        if getattr(self, 'fd', None) is not None:
            os.close(self.fd); self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _usage(self):
        total = 0; blobs = 0
        for name in os.listdir(self.fd):
            info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            if name == 'blobs':
                if not stat.S_ISDIR(info.st_mode):
                    raise IntegrityError('blob directory replaced')
                bfd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd)
                try:
                    for blob in os.listdir(bfd):
                        digest(blob)
                        item = os.stat(blob, dir_fd=bfd, follow_symlinks=False)
                        if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
                            raise IntegrityError('non-regular CAS member')
                        blobs += item.st_size
                finally:
                    os.close(bfd)
            elif name in FILES or re.fullmatch(r'\.archive-[a-f0-9]{32}\.tmp', name):
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise IntegrityError('non-regular owned file')
                total += info.st_size
            else:
                raise IntegrityError('unexpected member in owned archive')
        if total + blobs > TOTAL_CAP or blobs > BLOB_CAP:
            raise LimitError('archive exceeds capacity')
        return total + blobs, blobs

    def _budget(self, additional):
        if self._usage()[0] + additional > TOTAL_CAP:
            raise LimitError('archive capacity exhausted')

    def _atomic(self, name, raw):
        if name not in FILES:
            raise EvidenceError('unknown metadata filename')
        self._budget(len(raw))
        temp = '.archive-' + uuid.uuid4().hex + '.tmp'
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            os.fsync(self.fd)
        finally:
            try:
                os.unlink(temp, dir_fd=self.fd)
            except FileNotFoundError:
                pass

    def _read(self, path, limit):
        self._time()
        raw = initial_read(path, min(limit, max(0, CYCLE_READ_CAP - self.read_bytes)))
        self.read_bytes += len(raw)
        if self.read_bytes > CYCLE_READ_CAP:
            raise LimitError('cycle read capacity exhausted')
        self._time()
        return raw

    def _time(self):
        if time.monotonic() > self.cycle_end or time.time() >= self.deadline:
            raise LimitError('archive observation deadline reached')

    def _recover(self):
        verified = self.ledger.verify()
        if verified['event_count'] > EVENT_CAP:
            raise LimitError('archive event capacity exceeded')
        self.event_count = verified['event_count']
        checked_blobs = {}
        for row in self.ledger._connection.execute('SELECT body_json FROM events ORDER BY seq'):
            self._time()
            event = json.loads(row[0])
            if event['event_type'] != 'archive.source_observed':
                continue
            d = event['payload']['aggregate']
            if d['blob'] != 'blobs/' + digest(d['sha256']):
                raise IntegrityError('invalid CAS reference')
            if d['sha256'] not in checked_blobs:
                raw = self._read(self.out / d['blob'], LARGE_CAP)
                if sha(raw) != d['sha256']:
                    raise IntegrityError('CAS bytes changed')
                checked_blobs[d['sha256']] = len(raw)
            if checked_blobs[d['sha256']] != d['bytes']:
                raise IntegrityError('CAS bytes changed')
            self.records.append(d)
            if d['kind'] == 'jsonl_chunk':
                prev = self.cursors.get(d['label'])
                if d['start'] != (prev['end'] if prev else 0):
                    raise IntegrityError('non-contiguous committed log chunks')
                self.cursors[d['label']] = d
        self.seen = {sha(canonical_json(d).encode()) for d in self.records}

    def _capture_bytes(self, label, path, raw, kind, **extra):
        h = sha(raw)
        descriptor = {'label': label, 'source_path': str(path), 'kind': kind,
                      'sha256': h, 'bytes': len(raw), 'blob': 'blobs/' + h, **extra}
        identity = sha(canonical_json(descriptor).encode())
        if identity in self.seen:
            return descriptor
        if self.event_count >= EVENT_CAP:
            raise LimitError('archive event capacity exhausted')
        total, used = self._usage()
        bfd = os.open('blobs', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd)
        try:
            try:
                existing = self._read(self.out / 'blobs' / h, LARGE_CAP)
            except FileNotFoundError:
                if used + len(raw) > BLOB_CAP:
                    raise LimitError('blob capacity exhausted')
                self._budget(len(raw) + 128 * 1024)
                temp = '.archive-' + uuid.uuid4().hex + '.tmp'
                fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                    # Atomic create-only publication; a retry never overwrites a
                    # content-addressed file, including after process death.
                    os.link(temp, h, src_dir_fd=self.fd, dst_dir_fd=bfd, follow_symlinks=False)
                    os.fsync(bfd)
                finally:
                    os.unlink(temp, dir_fd=self.fd)
                    os.fsync(self.fd)
            else:
                if existing != raw:
                    raise IntegrityError('CAS hash collision or modified blob')
        finally:
            os.close(bfd)
        self._budget(128 * 1024)
        self.ledger.append_imported('source-' + identity, 'archive.source_observed', descriptor,
                                    self.run_id, h)
        self.event_count += 1; self.seen.add(identity); self.records.append(descriptor)
        return descriptor

    def _snapshot(self, label, path, *, large=False, binary=False):
        raw = self._read(path, LARGE_CAP if large else SMALL_CAP)
        value = None if binary else json_value(raw)
        self._capture_bytes(label, path, raw, 'snapshot')
        return value, sha(raw)

    def _chunk(self, label, path, *, limit=LARGE_CAP):
        """Commit one <=1 MiB complete-newline chunk after verifying old prefix."""
        self._time()
        fd, info = open_file(path, limit)
        old = self.cursors.get(label)
        offset = old['end'] if old else 0
        try:
            if old and (old['device'], old['inode']) != (info.st_dev, info.st_ino):
                raise IntegrityError('append-log inode replaced')
            if info.st_size < offset:
                raise IntegrityError('append-log truncated')
            h = hashlib.sha256(); remaining = offset
            while remaining:
                self._time()
                part = os.read(fd, min(MIB, remaining))
                if not part:
                    raise IntegrityError('append-log prefix truncated')
                remaining -= len(part); self.read_bytes += len(part); h.update(part)
                if self.read_bytes > CYCLE_READ_CAP:
                    raise LimitError('cycle read capacity exhausted')
            if old and h.hexdigest() != old['prefix_sha256']:
                raise IntegrityError('previously archived append-log prefix changed')
            part = os.read(fd, CHUNK_CAP)
            self.read_bytes += len(part)
            if self.read_bytes > CYCLE_READ_CAP:
                raise LimitError('cycle read capacity exhausted')
            stop = part.rfind(b'\n') + 1
            if not stop:
                if len(part) == CHUNK_CAP:
                    raise LimitError('JSONL record exceeds chunk bound')
                return
            raw = part[:stop]
            for line in raw.splitlines():
                if not line.strip():
                    raise EvidenceError('blank JSONL record')
                json_value(line)
            h.update(raw)
            result = self._capture_bytes(label, path, raw, 'jsonl_chunk', start=offset,
                        end=offset + len(raw), prefix_sha256=h.hexdigest(),
                        device=info.st_dev, inode=info.st_ino, lines=raw.count(b'\n'))
            self.cursors[label] = result
        finally:
            os.close(fd)

    def _identity(self, job, sidecar, summary, arm_sha):
        required = {'schema': 'jope.dl-wave4-run-identity/v1', 'wave': 4,
                    'model_version': self.m['model_version'], 'campaign_id': self.campaign,
                    'campaign_manifest_sha256': self.manifest_sha, 'job': job,
                    'source_sha': self.m['source_sha'], 'bundle_sha256': self.m['bundle_sha'],
                    'trainer_engine': 'dlhwm.wave3_train', 'trainer_summary_wave_field': 3,
                    'configuration': self.arm['configuration']}
        if type(sidecar) is not dict or any(sidecar.get(k) != v for k, v in required.items()):
            return False
        required = {'run_id': job['run_id'], 'seed': job['seed'], 'wave': 3,
                    'git_sha': self.m['source_sha'], 'data_manifest_sha256': self.m['bundle_sha'],
                    'arm': 'V1', 'qualification_only': False, 'arm_spec_sha256': arm_sha}
        return (type(summary) is dict and summary.get('qualification_only') is False
                and type(summary.get('seed')) is int and arm_sha is not None
                and all(summary.get(k) == v for k, v in required.items()))

    def cycle(self):
        """Read one bounded cycle; return an explicitly scoped observation summary."""
        self.read_bytes = 0; self.cycle_end = time.monotonic() + CYCLE_SECONDS
        errors = []; missing = []; views = {}; values = {}
        if sha(self._read(self.manifest_path, SMALL_CAP)) != self.manifest_sha:
            raise IntegrityError('pinned campaign manifest changed')
        def capture(label, path, **kwargs):
            try:
                result = self._snapshot(label, path, **kwargs)
                values[label] = result
                return result[0]
            except FileNotFoundError:
                missing.append(label)
            except (EvidenceError, OSError) as exc:
                errors.append({'label': label, 'error': type(exc).__name__, 'reason': str(exc)[:240]})
            return None
        runtime_manifest = capture('runtime/manifest.json', self.runtime / 'manifest.json')
        if runtime_manifest is not None and values['runtime/manifest.json'][1] != self.manifest_sha:
            errors.append({'label': 'runtime/manifest.json', 'error': 'IdentityMismatch'})
        arm = capture('runtime/arm_V1.json', self.runtime / 'arm_V1.json')
        arm_sha = values.get('runtime/arm_V1.json', (None, None))[1] if arm == self.arm else None
        preflight = capture('runtime/preflight.json', self.runtime / 'preflight.json')
        qualification = capture('runtime/gpu_qualification.json', self.qual_path)
        qualification_pin = values.get('runtime/gpu_qualification.json', (None, None))[1] == self.m['gpu_qualification']['sha256']
        status = capture('runtime/status.json', self.runtime / 'status.json')
        queue_bound = (type(status) is dict and status.get('host') == self.host
                       and status.get('campaign_sha') == self.manifest_sha)
        for label, path, limit in [('runtime/events.jsonl', self.runtime / 'events.jsonl', SMALL_CAP)]:
            try:
                self._chunk(label, path, limit=limit)
            except FileNotFoundError:
                missing.append(label)
            except (EvidenceError, OSError) as exc:
                errors.append({'label': label, 'error': type(exc).__name__, 'reason': str(exc)[:240]})
        for job in self.jobs:
            rid = job['run_id']; run = self.root / 'runs' / rid
            sidecar = capture(rid + '/identity.json', self.runtime / (rid + '.identity.json'))
            summary = capture(rid + '/summary.json', run / 'summary.json')
            bound = self._identity(job, sidecar, summary, arm_sha)
            completed = (bound and summary.get('state') == 'all_done'
                         and type(summary.get('folds')) is dict and set(summary['folds']) == set(FOLDS))
            view = {'identity_matches': bound, 'source_state': summary.get('state') if bound else 'unknown',
                    'complete_fold_key_closure': completed, 'epochs_observed': {},
                    'scientific_claims_independently_validated': False}
            for fold in FOLDS:
                hist = capture(rid + '/hist_' + fold + '.json', run / ('hist_' + fold + '.json'))
                if bound and isinstance(hist, list) and len(hist) <= 250:
                    view['epochs_observed'][fold] = len(hist)
            for filename in ('batch_metrics.jsonl', 'exposure.jsonl'):
                label = rid + '/' + filename
                try:
                    self._chunk(label, run / filename)
                except FileNotFoundError:
                    missing.append(label)
                except (EvidenceError, OSError) as exc:
                    errors.append({'label': label, 'error': type(exc).__name__, 'reason': str(exc)[:240]})
                if completed or (queue_bound and status.get('state') in FAIL_STATES):
                    # Whole JSONL copied as bytes: trailing partial record remains
                    # forensics, never promoted to a complete event.
                    capture(label + ':final_bytes', run / filename, large=True, binary=True)
            if completed or (queue_bound and status.get('state') in FAIL_STATES):
                for repeat in range(3):
                    for stem in ('oof_', 'oofraw_'):
                        filename = f'{stem}r{repeat}.npy'
                        capture(rid + '/' + filename, run / filename, large=True, binary=True)
            views[rid] = view
        self.ledger._connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        anchor = self.ledger.verify()
        # Export snapshots are reproducible ledger views, not immutable seals.
        # Bound temporary space before Ledger's atomic export operation.
        rows = [dict(json.loads(r['body_json']), event_hash=r['event_hash'])
                for r in self.ledger._connection.execute('SELECT * FROM events ORDER BY seq')]
        json_size = sum(len(canonical_json(r).encode()) + 1 for r in rows)
        csv_size = 128
        for row in rows:
            buf = io.StringIO(newline=''); csv.writer(buf).writerow(
                [row[k] for k in ('seq', 'event_id', 'run_id', 'event_type', 'evidence_kind', 'event_hash')]
                + [canonical_json(row)])
            csv_size += len(buf.getvalue().encode())
        self._budget(json_size + csv_size + max(json_size, csv_size) + SMALL_CAP)
        self.ledger.export_jsonl('events.jsonl'); self.ledger.export_csv('events.csv')
        summary = {'schema': SCHEMA, 'observed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'manifest_sha256': self.manifest_sha, 'host': self.host, 'campaign_id': self.campaign,
            'state': 'observed_with_gaps' if errors or missing else 'observed',
            'scope': 'same-host fixed allowlist; source observations, not scientific validation',
            'source_verification': {'declared_git_sha': self.m['source_sha'],
                'declared_source_files': self.m.get('source_files', {}),
                'preflight_receipt_captured': preflight is not None,
                'source_file_bytes_independently_verified': False,
                'qualification_receipt_matches_pinned_hash': qualification_pin},
            'queue_identity_matches': queue_bound, 'queue_state': status.get('state') if queue_bound else 'unknown',
            'runs': views, 'artifact_observations': len(self.records),
            'unique_blob_count': len({x['sha256'] for x in self.records}),
            'raw_batch_bytes_in_complete_chunks': sum(x['bytes'] for x in self.records
                if x['kind'] == 'jsonl_chunk' and x['label'].endswith('/batch_metrics.jsonl')),
            'native_order_proof': 'not established; imported source bytes only',
            'checkpoint_bytes_copied': 0, 'raw_dataset_bytes_read': 0,
            'ledger': anchor, 'read_bytes_this_cycle': self.read_bytes,
            'archive_bytes_before_summary': self._usage()[0], 'archive_cap_bytes': TOTAL_CAP,
            'missing': missing, 'errors': errors, 'observer_deadline_unix': self.deadline}
        self._atomic('evidence_summary.json', canonical_json(summary, limit=SMALL_CAP).encode() + b'\n')
        self._usage()
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign-manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--host', required=True, choices=('xbabe2', 'xbabe3'))
    parser.add_argument('--out', required=True, type=Path, help='new empty directory or matching owned archive')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--interval', type=int, default=60, help='continuous polling interval, at least 60 seconds')
    args = parser.parse_args(argv)
    if args.interval < 60:
        parser.error('--interval must be at least 60 seconds')
    with Archive(args.campaign_manifest, args.manifest_sha, args.host, args.out) as archive:
        while time.time() < archive.deadline:
            result = archive.cycle()
            print(json.dumps({'state': result['state'], 'manifest_sha256': archive.manifest_sha,
                              'artifact_observations': result['artifact_observations']}), flush=True)
            if args.once:
                return 0
            remaining = min(args.interval, archive.deadline - time.time())
            # Never block for over 60s; deadline checked between waits.
            while remaining > 0:
                delay = min(60, remaining); time.sleep(delay); remaining -= delay
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
