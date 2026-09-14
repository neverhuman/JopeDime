#!/usr/bin/env python3
"""Resumable indexed project inventory. Enumeration is never parity evidence.

Opaque content hashing does not parse observation values. Symlinks and external
references need explicit resolution; SQLite files need application snapshots.
"""
import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import time

import workspace_store as store


class Inventory:
    def __init__(self, path, roots, owner, policy):
        self.path = Path(path)
        store.directory(self.path.parent)
        if self.path.is_symlink():
            raise ValueError('Inventory database symlink')
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS configuration(identity TEXT PRIMARY KEY, document TEXT);
            CREATE TABLE IF NOT EXISTS roots(name TEXT PRIMARY KEY, path TEXT NOT NULL, owner TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS queue(root TEXT, path TEXT, visited INTEGER DEFAULT 0,
                PRIMARY KEY(root,path));
            CREATE INDEX IF NOT EXISTS unvisited ON queue(visited,root,path);
            CREATE TABLE IF NOT EXISTS directory_cursors(root TEXT, path TEXT, signature TEXT,
                position INTEGER, PRIMARY KEY(root,path));
            CREATE TABLE IF NOT EXISTS entries(root TEXT, path TEXT, kind TEXT, size INTEGER,
                mode INTEGER, signature TEXT, classification TEXT, reason TEXT,
                sha256 TEXT, observed_at REAL, PRIMARY KEY(root,path));
            CREATE INDEX IF NOT EXISTS content ON entries(sha256,size);
            CREATE INDEX IF NOT EXISTS unhashed ON entries(root,path)
                WHERE kind='file' AND classification='included' AND sha256 IS NULL;
            CREATE TABLE IF NOT EXISTS history(origin TEXT, path TEXT, version TEXT, document TEXT,
                PRIMARY KEY(origin,path,version));
            CREATE TABLE IF NOT EXISTS imports(sha256 TEXT PRIMARY KEY, path TEXT, imported_at REAL);
            CREATE TABLE IF NOT EXISTS mirror_versions(version TEXT PRIMARY KEY, document TEXT NOT NULL, size INTEGER NOT NULL,
                captured TEXT, receipt TEXT, attempted_at REAL NOT NULL DEFAULT 0, last_error TEXT,
                last_deferred_attempted INTEGER);
            CREATE INDEX IF NOT EXISTS mirror_pending ON mirror_versions(receipt,attempted_at,version);
            CREATE TABLE IF NOT EXISTS mirror_pages(sequence INTEGER PRIMARY KEY, identity TEXT UNIQUE,
                document TEXT NOT NULL, receipt TEXT);
            CREATE TABLE IF NOT EXISTS mirror_cursor(name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS mirror_directory_offsets(root TEXT, path TEXT, signature TEXT,
                cookie INTEGER, PRIMARY KEY(root,path));
        ''')
        if 'last_deferred_attempted' not in {row['name'] for row in self.db.execute('PRAGMA table_info(mirror_versions)')}:
            # Old rows cannot reveal whether their last deferral was attempted.
            # Preserve that uncertainty instead of guessing from older attempts.
            self.db.execute('ALTER TABLE mirror_versions ADD COLUMN last_deferred_attempted INTEGER')
        self.roots = {name: str(Path(os.path.abspath(path))) for name, path in roots.items()}
        self.owner, self.policy = owner, policy
        doc = dict(schema='workspace.index-config.v2', roots=self.roots, owner=owner, policy=policy)
        self.identity = store.identity(doc)
        old = self.db.execute('SELECT identity FROM configuration').fetchone()
        if old and old[0] != self.identity:
            self.db.close()
            raise ValueError('Inventory scope changed; use a successor database')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO configuration VALUES (?,?)',
                            (self.identity, store.encoded(doc).decode()))
            for name, path in self.roots.items():
                store.relative(name)
                if self.path.resolve().is_relative_to(Path(path)):
                    raise ValueError('Inventory output must be outside traversed roots')
                self.db.execute('INSERT OR IGNORE INTO roots VALUES (?,?,?)', (name, path, owner))
                self.db.execute('INSERT OR IGNORE INTO queue(root,path) VALUES (?,?)', (name, ''))

    def close(self):
        self.db.close()

    def _record(self, root, name, info, kind, classification, reason=None, digest=None):
        prior = self.db.execute('SELECT * FROM entries WHERE root=? AND path=?', (root, name)).fetchone()
        if prior and (prior['signature'], prior['kind'], prior['classification'], prior['reason']) != (
                json.dumps(store.signature(info)) if info else None, kind, classification, reason):
            original = dict(prior)
            self.db.execute('INSERT OR IGNORE INTO history VALUES (?,?,?,?)',
                            ('prior_observation', root + ':' + name, store.identity(original), store.encoded(original).decode()))
        self.db.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (root, name, kind, info.st_size if info else None,
                         stat.S_IMODE(info.st_mode) if info else None,
                         json.dumps(store.signature(info)) if info else None,
                         classification, reason, digest, time.time()))
        if kind == 'file' and classification == 'included' and reason == 'capture_bytes_only':
            record = dict(size=info.st_size, mode=stat.S_IMODE(info.st_mode),
                          executable=bool(info.st_mode & 0o111), uid=info.st_uid, gid=info.st_gid,
                          observed_signature=store.signature(info), class_='mutable_snapshot')
            record['class'] = record.pop('class_')
            document = dict(root=root, path=name, record=record)
            self.db.execute('INSERT OR IGNORE INTO mirror_versions(version,document,size) VALUES (?,?,?)',
                            (store.identity(dict(configuration=self.identity, **document)), store.encoded(document).decode(), info.st_size))

    def _visit(self, root, name, deadline):
        path = Path(self.roots[root]) / name
        info = path.lstat()
        # Reject symlink ancestors, including declared roots, before traversal.
        if any(p.is_symlink() for p in [path, *path.parents]):
            self._record(root, name, info, 'symlink', 'unresolved',
                         'symlink_reference:' + str(path.resolve(strict=False)))
        elif stat.S_ISDIR(info.st_mode):
            if path.name in store.PRUNED - {'.git'}:
                self._record(root, name, info, 'directory', 'excluded', 'machine_local_cache_or_credentials')
            elif store.under(name, self.policy.get('prune_roots', [])):
                self._record(root, name, info, 'directory', 'excluded', 'explicit_duplicate_or_disposable_build_output')
            else:
                self._record(root, name, info, 'directory', 'included')
                progress = self.db.execute('SELECT signature,position FROM directory_cursors WHERE root=? AND path=?',
                                           (root, name)).fetchone()
                signature = json.dumps(store.signature(info))
                if progress and progress['signature'] != signature:
                    raise ValueError('Directory changed during resumed traversal')
                position = progress['position'] if progress else 0
                with os.scandir(path) as entries:
                    for entry in entries:
                        child = (Path(name) / entry.name).as_posix()
                        inserted = self.db.execute('INSERT OR IGNORE INTO queue(root,path) VALUES (?,?)', (root, child))
                        if not inserted.rowcount:
                            if time.monotonic() >= deadline:
                                self.db.execute('INSERT OR REPLACE INTO directory_cursors VALUES (?,?,?,?)',
                                                (root, name, signature, position))
                                return False
                            continue
                        # Resume by durable child identity, never scandir ordinal:
                        # directory iteration order may differ on the next open.
                        position += 1
                        if time.monotonic() >= deadline:
                            self.db.execute('INSERT OR REPLACE INTO directory_cursors VALUES (?,?,?,?)',
                                            (root, name, signature, position))
                            return False
                if store.signature(path.lstat()) != store.signature(info):
                    raise ValueError('Directory changed while enumerating')
                self.db.execute('DELETE FROM directory_cursors WHERE root=? AND path=?', (root, name))
        elif stat.S_ISREG(info.st_mode):
            if Path(name).name in {'config', 'config.worktree'} and '.git' in Path(name).parts:
                self._record(root, name, info, 'file', 'excluded', 'machine_local_git_configuration')
            else:
                with store.opened(self.roots[root], name) as (stream, current):
                    prefix = stream.read(4096)
                    if store.signature(info) != store.signature(os.fstat(stream.fileno())):
                        raise ValueError('File changed during enumeration')
                kind, action = store.classify(name, prefix, self.policy)
                classification = ('excluded' if kind == 'machine_local' else
                                  'snapshot_required' if action == 'application_snapshot_required' else 'included')
                self._record(root, name, info, 'file', classification, action)
        else:
            self._record(root, name, info, 'special', 'excluded', 'machine_local_process_state')
        return True

    def step(self, seconds=15, revalidate=True, visit=None):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid inventory time budget')
        deadline = time.monotonic() + seconds
        processed = 0
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            while time.monotonic() < deadline:
                row = self.db.execute('SELECT root,path FROM queue WHERE visited=0 ORDER BY root,path LIMIT 1').fetchone()
                if row is None:
                    break
                root, name = row
                self.db.execute('SAVEPOINT visit')
                try:
                    complete = (visit or self._visit)(root, name, deadline)
                    if complete:
                        self.db.execute('UPDATE queue SET visited=1 WHERE root=? AND path=?', (root, name))
                    self.db.execute('RELEASE visit')
                except TimeoutError:
                    self.db.execute('ROLLBACK TO visit')
                    self.db.execute('RELEASE visit')
                    break
                except (OSError, ValueError) as error:
                    self.db.execute('ROLLBACK TO visit')
                    self.db.execute('RELEASE visit')
                    self._record(root, name, None, 'unknown', 'unresolved', str(error))
                    self.db.execute('UPDATE queue SET visited=-1 WHERE root=? AND path=?', (root, name))
                processed += 1
                if processed % 256 == 0:
                    self.db.commit()
                    self.db.execute('BEGIN IMMEDIATE')
        return self.summary(revalidate=revalidate)

    def hash_step(self, seconds=15):
        """Every first-boundary file gets an actual hash, independent of stat reuse."""
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid hash time budget')
        deadline = time.monotonic() + seconds
        processed = 0
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            while time.monotonic() < deadline:
                row = self.db.execute("SELECT * FROM entries WHERE kind='file' AND classification='included' AND sha256 IS NULL LIMIT 1").fetchone()
                if row is None:
                    break
                try:
                    actual = store.stable_hash(self.roots[row['root']], row['path'])
                    if actual['observed_signature'] != json.loads(row['signature']):
                        raise ValueError('File changed since inventory; successor observation required')
                    self.db.execute('UPDATE entries SET sha256=? WHERE root=? AND path=?',
                                        (actual['sha256'], row['root'], row['path']))
                except (OSError, ValueError) as error:
                    self.db.execute("UPDATE entries SET classification='unresolved',reason=? WHERE root=? AND path=?",
                                        (str(error), row['root'], row['path']))
                processed += 1
                if processed % 128 == 0:
                    self.db.commit()
                    self.db.execute("BEGIN IMMEDIATE")
        return self.summary()

    def import_history(self, path):
        """Retain every previous version; imports never assert current possession."""
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        doc = json.loads(raw)
        if doc.get('schema') != 'workspace.durable-index.v1':
            raise ValueError('Unsupported previous mirror history')
        with self.db:
            for key, value in doc['files'].items():
                self.db.execute('INSERT OR IGNORE INTO history VALUES (?,?,?,?)',
                                ('legacy_current', key, store.identity(value), store.encoded(value).decode()))
            for key, versions in doc['history'].items():
                for value in versions:
                    self.db.execute('INSERT OR IGNORE INTO history VALUES (?,?,?,?)',
                                    ('legacy_history', key, store.identity(value), store.encoded(value).decode()))
            self.db.execute('INSERT OR IGNORE INTO imports VALUES (?,?,?)', (digest, str(path), time.time()))
        return digest

    def summary(self, revalidate=True):
        unvisited = self.db.execute('SELECT COUNT(*) FROM queue WHERE visited=0').fetchone()[0]
        unresolved = self.db.execute("SELECT COUNT(*) FROM entries WHERE classification IN ('unresolved','snapshot_required')").fetchone()[0]
        unhashed = self.db.execute("SELECT COUNT(*) FROM entries WHERE kind='file' AND classification='included' AND sha256 IS NULL").fetchone()[0]
        files, size = self.db.execute("SELECT COUNT(*),COALESCE(SUM(size),0) FROM entries WHERE kind='file' AND classification!='excluded'").fetchone()
        changed = 0
        if revalidate and unvisited == unresolved == unhashed == 0:
            # A completed enumeration is not frozen while its live roots change.
            # Revalidate the entire captured namespace before any completeness claim.
            for row in self.db.execute("SELECT root,path,signature FROM entries WHERE classification='included'"):
                path = Path(self.roots[row['root']]) / row['path']
                try:
                    if any(p.is_symlink() for p in [path, *path.parents]) or store.signature(path.lstat()) != json.loads(row['signature']):
                        changed += 1
                except OSError:
                    changed += 1
        return dict(schema='workspace.index-status.v2', configuration_sha256=self.identity, owner=self.owner,
                    roots=self.roots, files=files, path_counted_bytes=size, unvisited_paths=unvisited,
                    unresolved_paths=unresolved, unhashed_files=unhashed, traversal_complete=unvisited == 0,
                    changed_since_capture=changed,
                    content_inventory_complete=revalidate and unvisited == unresolved == unhashed == changed == 0,
                    project_reference_closure='unverified', missing_unique_bytes=None, synchronized=False)

    def mirror_advance(self, seconds=2, refresh_limit=256):
        """Advance bounded enumeration without exporting or revalidating a catalog.

        Completed traversals revisit a bounded range on each subsequent call.
        Prior observations, missing paths and captured versions are never deleted.
        """
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 5:
            raise ValueError('Mirror enumeration budget must be at most five seconds')
        if not self.db.execute('SELECT 1 FROM queue WHERE visited=0 LIMIT 1').fetchone():
            old = self.db.execute("SELECT value FROM mirror_cursor WHERE name='refresh'").fetchone()
            cursor = json.loads(old[0]) if old else ['', '']
            rows = self.db.execute('SELECT root,path FROM queue WHERE (root,path)>(?,?) ORDER BY root,path LIMIT ?',
                                   (*cursor, refresh_limit)).fetchall()
            if not rows:
                rows = self.db.execute('SELECT root,path FROM queue ORDER BY root,path LIMIT ?', (refresh_limit,)).fetchall()
            with self.db:
                for row in rows:
                    self.db.execute('UPDATE queue SET visited=0 WHERE root=? AND path=?', tuple(row))
                if rows:
                    self.db.execute("INSERT OR REPLACE INTO mirror_cursor VALUES ('refresh',?)", (json.dumps(list(rows[-1])),))
        return self.step(seconds, revalidate=False, visit=self._mirror_visit)

    def _mirror_visit(self, root, name, deadline):
        """Linux directory cookies avoid repeatedly rescanning a growing prefix.

        Cookies are used only while the complete directory signature matches.
        Changed directories restart enumeration; already observed children stay.
        Legacy scandir ordinal cursors are never interpreted as these cookies.
        """
        path = Path(self.roots[root]) / name
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or any(p.is_symlink() for p in [path, *path.parents])
                or path.name in store.PRUNED - {'.git'} or store.under(name, self.policy.get('prune_roots', []))):
            return self._visit(root, name, deadline)
        if os.uname().sysname != 'Linux' or ctypes.sizeof(ctypes.c_long) != 8:
            raise ValueError('Bounded directory enumeration requires the supported Linux ABI')
        class Dirent(ctypes.Structure):
            _fields_ = [('ino', ctypes.c_ulong), ('offset', ctypes.c_long),
                        ('length', ctypes.c_ushort), ('kind', ctypes.c_ubyte), ('name', ctypes.c_char * 256)]
        libc = ctypes.CDLL(None, use_errno=True)
        libc.fdopendir.argtypes, libc.fdopendir.restype = [ctypes.c_int], ctypes.c_void_p
        libc.readdir.argtypes, libc.readdir.restype = [ctypes.c_void_p], ctypes.POINTER(Dirent)
        libc.telldir.argtypes, libc.telldir.restype = [ctypes.c_void_p], ctypes.c_long
        libc.seekdir.argtypes, libc.seekdir.restype = [ctypes.c_void_p, ctypes.c_long], None
        libc.closedir.argtypes, libc.closedir.restype = [ctypes.c_void_p], ctypes.c_int
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if store.signature(os.fstat(fd)) != store.signature(info):
            os.close(fd)
            raise ValueError('Directory identity changed before enumeration')
        handle = libc.fdopendir(fd)
        if not handle:
            os.close(fd)
            raise OSError(ctypes.get_errno(), 'fdopendir failed')
        signature = json.dumps(store.signature(info))
        previous = self.db.execute('SELECT signature,cookie FROM mirror_directory_offsets WHERE root=? AND path=?', (root, name)).fetchone()
        self._record(root, name, info, 'directory', 'included')
        if previous and previous['signature'] == signature:
            libc.seekdir(handle, previous['cookie'])
        elif previous:
            evidence = dict(root=root, path=name, original_signature=previous['signature'], cookie=previous['cookie'])
            self.db.execute('INSERT OR IGNORE INTO history VALUES (?,?,?,?)',
                            ('invalidated_directory_cookie', name, store.identity(evidence), store.encoded(evidence).decode()))
        try:
            while True:
                ctypes.set_errno(0)
                entry = libc.readdir(handle)
                if not entry:
                    if ctypes.get_errno():
                        raise OSError(ctypes.get_errno(), 'readdir failed')
                    if store.signature(path.lstat()) != store.signature(info):
                        self.db.execute('DELETE FROM mirror_directory_offsets WHERE root=? AND path=?', (root, name))
                        return False
                    self.db.execute('DELETE FROM mirror_directory_offsets WHERE root=? AND path=?', (root, name))
                    return True
                child_name = os.fsdecode(entry.contents.name)
                if child_name not in ('.', '..'):
                    child = (Path(name) / child_name).as_posix()
                    self.db.execute('INSERT OR IGNORE INTO queue(root,path) VALUES (?,?)', (root, child))
                if time.monotonic() >= deadline:
                    cookie = libc.telldir(handle)
                    if cookie < 0:
                        raise OSError(ctypes.get_errno(), 'telldir failed')
                    self.db.execute('INSERT OR REPLACE INTO mirror_directory_offsets VALUES (?,?,?,?)',
                                    (root, name, signature, cookie))
                    return False
        finally:
            libc.closedir(handle)

    def mirror_summary(self):
        result = self.summary(revalidate=False)
        count, size = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(size),0) FROM mirror_versions WHERE receipt IS NULL").fetchone()
        def deferred_count(condition):
            return self.db.execute('SELECT COUNT(*) FROM mirror_versions WHERE receipt IS NULL AND last_error IS NOT NULL AND ' + condition).fetchone()[0]
        attempted = deferred_count('last_deferred_attempted=1')
        result.update(observed_unacknowledged_versions=count, observed_unacknowledged_bytes=size,
                      acknowledged_versions=self.db.execute('SELECT COUNT(*) FROM mirror_versions WHERE receipt IS NOT NULL').fetchone()[0],
                      failed_capture_versions=attempted,
                      attempted_deferred_versions=attempted,
                      unattempted_deferred_versions=deferred_count('last_deferred_attempted=0'),
                      deferred_attempt_status_unknown_versions=deferred_count('last_deferred_attempted IS NULL'),
                      files_remaining=None, bytes_remaining=None, backlog_coverage='live namespace and legacy custody not fully reconciled',
                      legacy_custody='retained unchanged; not imported by the bounded mirror')
        return result

    def mirror_page(self, limit=64, byte_limit=262144):
        if type(limit) is not int or not 1 <= limit <= 128 or type(byte_limit) is not int or not 16384 <= byte_limit <= 262144:
            raise ValueError('Invalid bounded mirror page size')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            existing = self.db.execute('SELECT identity,document FROM mirror_pages WHERE receipt IS NULL ORDER BY sequence LIMIT 1').fetchone()
            if existing:
                page_id, document = existing
                page = json.loads(document)
            else:
                sequence = self.db.execute('SELECT COALESCE(MAX(sequence),0)+1 FROM mirror_pages').fetchone()[0]
                page = dict(schema='workspace.inventory-page.v1', configuration_sha256=self.identity,
                            sequence=sequence, items=[])
                query = ('SELECT version,document,attempted_at,last_error FROM mirror_versions '
                         'WHERE receipt IS NULL AND attempted_at {} 0 ORDER BY attempted_at,version LIMIT ?')
                fresh = self.db.execute(query.format('='), (limit,)).fetchall()
                retries = self.db.execute(query.format('>'), (limit,)).fetchall()
                # Continuous new observations cannot consume the entire page
                # forever. Reserve a retry share and rotate which share goes
                # first so short windows can advance both kinds of originals.
                retry_quota = (1 if sequence % 2 else 0) if limit == 1 else max(1, limit // 4)
                retry_rows = retries[:retry_quota]
                fresh_rows = fresh[:limit - len(retry_rows)]
                retry_rows += retries[len(retry_rows):limit - len(fresh_rows)]
                rows = retry_rows + fresh_rows if sequence % 2 == 0 else fresh_rows + retry_rows
                for row in rows:
                    item = dict(version=row['version'], **json.loads(row['document']))
                    if row['last_error'] is not None and row['attempted_at'] > 0:
                        # A corrupt object or failed batch must not repeatedly
                        # prevent unrelated originals from reaching custody.
                        # This is a retry scheduling hint, never an admission.
                        item['retry_individually'] = True
                    candidate = dict(page, items=page['items'] + [item])
                    if len(store.encoded(candidate)) > byte_limit:
                        if not page['items']:
                            raise ValueError('An inventory record exceeds the bounded page limit')
                        break
                    page = candidate
                page_id = store.identity(page)
                if page['items']:
                    self.db.execute('INSERT INTO mirror_pages VALUES (?,?,?,NULL)',
                                    (sequence, page_id, store.encoded(page).decode()))
        return dict(page_id=page_id, page=page, summary=self.mirror_summary())

    def _mirror_item(self, page_id, version):
        row = self.db.execute('SELECT document FROM mirror_pages WHERE identity=?', (page_id,)).fetchone()
        if row is None or not any(item['version'] == version for item in json.loads(row[0])['items']):
            raise ValueError('Capture is outside its original inventory page')
        return self.db.execute('SELECT * FROM mirror_versions WHERE version=?', (version,)).fetchone()

    def mirror_capture(self, state, page_id, version):
        row = self._mirror_item(page_id, version)
        if row['captured']:
            record = json.loads(row['captured'])
            store.verify_object(state, record)
            return record
        document = json.loads(row['document'])
        record = store.seal(self.roots[document['root']], document['path'], state, reject_live_database=True)
        if record['observed_signature'] != document['record']['observed_signature']:
            # The actual stable version still has byte custody and provenance;
            # it cannot acknowledge the originally observed version.
            evidence = dict(expected=document, captured_actual=record)
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO history VALUES (?,?,?,?)',
                                ('changed_before_capture', document['path'], store.identity(evidence), store.encoded(evidence).decode()))
                actual_document = dict(document, record={k: record[k] for k in document['record'] if k != 'class'})
                actual_document['record']['class'] = document['record']['class']
                actual_version = store.identity(dict(configuration=self.identity, **actual_document))
                self.db.execute('INSERT OR IGNORE INTO mirror_versions(version,document,size,captured) VALUES (?,?,?,?)',
                                (actual_version, store.encoded(actual_document).decode(), record['size'], store.encoded(record).decode()))
                self.db.execute('UPDATE mirror_versions SET captured=COALESCE(captured,?) WHERE version=?',
                                (store.encoded(record).decode(), actual_version))
            raise ValueError('File changed since inventory; original version remains pending')
        with self.db:
            self.db.execute('UPDATE mirror_versions SET captured=? WHERE version=?', (store.encoded(record).decode(), version))
        return record

    def mirror_acknowledge(self, state, page_id, receipt_sha256):
        """Accept the authenticated mirror controller's custody attestation.

        This verifies the original page and local objects. Counterparty checks
        are performed by Mirror before receipt transfer; this is not an
        independent witness or a full-project synchronization certificate.
        """
        receipt_path = store.object_path(state, receipt_sha256)
        if receipt_path.stat().st_size > 524288:
            raise ValueError('Oversized mirror acknowledgment')
        with store.opened(receipt_path.parent, receipt_path.name) as (stream, current):
            raw = stream.read(524289)
        if len(raw) > 524288 or hashlib.sha256(raw).hexdigest() != receipt_sha256:
            raise ValueError('Mirror acknowledgment object changed')
        receipt = json.loads(raw)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT document,receipt FROM mirror_pages WHERE identity=?', (page_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown inventory page')
            if row['receipt']:
                if row['receipt'] != receipt_sha256:
                    raise ValueError('Conflicting acknowledgment of original page')
                return self.mirror_summary()
            items = json.loads(row['document'])['items']
            outcomes = receipt.get('outcomes', [])
            if (receipt.get('schema') != 'workspace.inventory-page-custody.v1'
                    or receipt.get('configuration_sha256') != self.identity or receipt.get('page_id') != page_id
                    or [item['version'] for item in items] != [item.get('version') for item in outcomes]):
                raise ValueError('Acknowledgment does not cover the exact original page')
            for item, outcome in zip(items, outcomes):
                original = self._mirror_item(page_id, item['version'])
                if outcome.get('status') == 'verified_on_both_hosts':
                    if not original['captured'] or store.encoded(outcome.get('record')) != store.encoded(json.loads(original['captured'])):
                        raise ValueError('Acknowledgment changed captured original identity')
                    store.verify_object(state, outcome['record'])
                    self.db.execute('UPDATE mirror_versions SET receipt=?,last_error=NULL,last_deferred_attempted=NULL WHERE version=?',
                                     (receipt_sha256, item['version']))
                elif outcome.get('status') == 'deferred' and isinstance(outcome.get('reason'), str):
                    attempted = outcome.get('attempted')
                    if type(attempted) is not bool:
                        raise ValueError('Deferred inventory outcome requires explicit attempted status')
                    self.db.execute('UPDATE mirror_versions SET attempted_at=?,last_error=?,last_deferred_attempted=? WHERE version=?',
                                     (time.time() if outcome.get('attempted') is True else original['attempted_at'],
                                      outcome['reason'][:4096], int(attempted), item['version']))
                else:
                    raise ValueError('Invalid inventory page outcome')
            self.db.execute('UPDATE mirror_pages SET receipt=? WHERE identity=?', (receipt_sha256, page_id))
        return self.mirror_summary()

    def export(self, path):
        """Stream the entire indexed denominator, including exclusions and omissions."""
        path = Path(path)
        if path.exists():
            raise ValueError('Refusing to overwrite an inventory manifest')
        store.directory(path.parent)
        temporary = path.with_name(path.name + '.partial')
        with self.db:
            self.db.execute('BEGIN')
            with temporary.open('xb') as stream:
                stream.write(store.encoded(self.summary()) + b'\n')
                for row in self.db.execute('SELECT * FROM entries ORDER BY root,path'):
                    stream.write(store.encoded(dict(row, owner=self.owner)) + b'\n')
                for row in self.db.execute('SELECT root,path FROM queue WHERE visited=0 ORDER BY root,path'):
                    stream.write(store.encoded(dict(row, owner=self.owner, classification='unvisited')) + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
        # Link without replacing any concurrently published manifest.
        os.link(temporary, path)
        temporary.unlink()
        store.fsync_dir(path.parent)
        return store.stable_hash(path.parent, path.name)

    @staticmethod
    def _state_identity(connection, deadline=None):
        """Bind the complete traversal, observations, and imported history."""
        digest = hashlib.sha256()
        for table in ('configuration', 'roots', 'queue', 'directory_cursors',
                      'entries', 'history', 'imports', 'mirror_versions', 'mirror_pages', 'mirror_cursor', 'mirror_directory_offsets'):
            columns = connection.execute('PRAGMA table_info(' + table + ')').fetchall()
            digest.update(store.encoded([table, [tuple(row) for row in columns]]) + b'\n')
            order = ','.join(str(i + 1) for i in range(len(columns)))
            for row in connection.execute('SELECT * FROM ' + table + ' ORDER BY ' + order):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Inventory restoration verification deadline exceeded')
                digest.update(store.encoded(list(row)) + b'\n')
        return digest.hexdigest()

    def snapshot(self, state, seconds=60):
        """Back up this owning connection and verify the entire restored index.

        This verifies inventory application state only. Authority journals,
        independent witnesses, and model continuation are different applications.
        """
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 300:
            raise ValueError('Bounded inventory snapshot deadline required')
        deadline = time.monotonic() + seconds
        with self.db:
            self.db.execute('BEGIN')
            expected = self._state_identity(self.db, deadline)
            snapshot = store.sqlite_snapshot_connection(
                self.db, {'schema': 'workspace.index-config.v2',
                          'configuration_sha256': self.identity,
                          'state_sha256': expected}, state, max(0.001, deadline - time.monotonic()))
        path = store.object_path(state, snapshot['object']['sha256'])
        store.verify_object(state, snapshot['object'])
        with contextlib.closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as restored:
            if self._state_identity(restored, deadline) != expected:
                raise ValueError('Restored inventory state differs from the original connection')
        receipt = dict(schema='workspace.inventory-restoration.v1',
                       configuration_sha256=self.identity, state_sha256=expected,
                       snapshot=snapshot, inventory_state_restored=True,
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       verified_at=time.time(), full_project_boundary=False,
                       authority_recovery_verified=False, model_resume_verified=False,
                       activation_allowed=False)
        store.write_json(Path(state) / 'receipts' / (store.identity(receipt) + '.json'), receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host', choices=('local', 'remote'), required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--hash', action='store_true')
    parser.add_argument('--import-history', type=Path)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--snapshot-state', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('schema') != 'workspace.sync-config.v2':
        raise ValueError('Explicit v2 scope required')
    host = config[args.host]
    inventory = Inventory(args.database, host['roots'], host['name'], config['policy'])
    try:
        if args.import_history:
            inventory.import_history(args.import_history)
        result = inventory.hash_step(args.seconds) if args.hash else inventory.step(args.seconds)
        if args.export:
            result['export'] = inventory.export(args.export)
        if args.snapshot_state:
            result['snapshot'] = inventory.snapshot(args.snapshot_state)
        print(json.dumps(result, indent=2))
    finally:
        inventory.close()


if __name__ == '__main__':
    main()
