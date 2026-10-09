"""Global one-use native execution guard, with permanent uncertain custody.

All concurrent v5 workers MUST share one registry and the identical source set.
Legacy workers must be physically closed before the first packet. The guard does
not authorize restarting any old queue, unit, evaluation or expired claim.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import sqlite3
import time

from readonly_snapshot import source_path, verified_snapshot

SCHEMA = 'jope.native-global-execution-guard/v1'
GLOBAL_DEADLINE = 1791592146.380285
BLOCKED = {'leased', 'unknown_attempt', 'failure_custody_verified', 'pending_unsupported'}
SUCCESS = {'cached_verified', 'published_verified'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def hash64(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def exact_file(ref):
    path = source_path(ref['path'])
    need(path.stat().st_size == ref['bytes'], 'exact declared local artifact size')
    with path.open('rb') as stream:
        raw = stream.read(ref['bytes'] + 1)
    need(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'],
         'exact local artifact bytes/SHA')
    return raw


def key_for(context_id, program):
    need(hash64(context_id) and type(program) is str and 0 < len(program.encode()) <= 32768,
         'canonical context/program identity')
    return digest({'schema': 'jope.native-canonical-context-program/v1',
                   'context_id': context_id, 'program_sha256': hashlib.sha256(program.encode()).hexdigest()})


def queue_aliases(con):
    """Derive actual Context.identity; publication jobs carry only evaluation_key.

    Yield evaluator identity plus BOTH stage states. Unknown context shapes fail
    closed rather than disappearing from a JSON-path filter.
    """
    from jope_flywheel.protocol import Context
    evaluations={}; publications={}
    for row in con.execute("SELECT * FROM jobs WHERE stage IN ('evaluate','publish')"):
        body=json.loads(row['payload']); key=body.get('evaluation_key')
        need(hash64(key),'historical queue evaluation identity')
        target=evaluations if row['stage']=='evaluate' else publications
        need(key not in target,'unique historical queue stage/key')
        target[key]=(row,body)
    need(set(publications)<=set(evaluations),'publication has registered evaluation')
    for key,(row,body) in evaluations.items():
        context=body.get('context');need(type(context)is dict,'historical queue context')
        cid=context['identity'] if set(context)=={'identity'} else Context(**context).identity
        need(hash64(cid) and type(body.get('program'))is str,'queue canonical context/program fields')
        if 'program_sha256' in body:
            need(hashlib.sha256(body['program'].encode()).hexdigest()==body['program_sha256'],'historical queue raw program hash')
        rows=[row]+([publications[key][0]]if key in publications else [])
        states=[{'state':r['state'],'attempts':r['attempts']}for r in rows]
        yield {'context_id':cid,'program':body['program'],'evaluation_key':key,'states':states}


class BlockedKey(ValueError):
    """Known attempted/uncertain key; preserve recorded aliases without claiming."""


def no_prior_attempt(ok, message):
    if not ok:
        raise BlockedKey(message)


class CacheRequired(ValueError):
    """Prior completion exists: exact verified cache or explicit unresolved hold."""


class GlobalGuard:
    """Owns only the new registry; all declared source databases stay read-only.

    check() runs again after the outbox lease and immediately before execution.
    reserve/start commits are durable before invoking an external callback. A
    crash at any point permanently burns the key; no elapsed-clock retry exists.
    """
    def __init__(self, config, *, canonicalize, packet_sha256, deadline, now=time.time):
        need(set(config) == {'schema', 'registry_path', 'scope_sha256', 'canonicalizer_sha256',
                            'global_deadline_unix', 'sources', 'census', 'max_snapshot_bytes'},
             'closed global guard config')
        need(config['schema'] == SCHEMA and hash64(config['scope_sha256']) and
             hash64(config['canonicalizer_sha256']) and hash64(packet_sha256), 'guard source identities')
        need(type(deadline) in (int, float) and math.isfinite(deadline) and
             now() < deadline <= config['global_deadline_unix'] == GLOBAL_DEADLINE,
             'unchanged global clock and future packet clock')
        self.config, self.canonicalize, self.packet = config, canonicalize, packet_sha256
        self.deadline, self.now = deadline, now
        self.started = time.monotonic()
        self.duration = deadline - now()
        self.sources = config['sources']
        need(type(self.sources) is list and 1 <= len(self.sources) <= 128, 'complete bounded source registry')
        paths = []
        self.physical_sources = {}
        for source in self.sources:
            need(set(source) == {'path', 'kind', 'config'} and source['kind'] in ('outbox', 'queue'),
                 'closed source descriptor')
            p = source_path(source['path'])
            paths.append(str(p))
            st = p.stat()
            self.physical_sources[str(p)] = {'device': st.st_dev, 'inode': st.st_ino}
            need((source['kind'] == 'outbox' and type(source['config']) is dict) or
                 (source['kind'] == 'queue' and source['config'] is None), 'source configuration')
        need(paths == sorted(set(paths)), 'unique sorted complete source paths')
        need(len({(s['device'], s['inode']) for s in self.physical_sources.values()}) == len(paths),
             'source inode aliases must not duplicate custody')
        ref = config['census']
        need(set(ref) == {'path', 'sha256', 'bytes'} and hash64(ref['sha256']) and
             type(ref['bytes']) is int and 0 < ref['bytes'] <= 16 * 1024**2, 'bounded pinned census')
        raw = exact_file(ref)
        census = json.loads(raw)
        need(set(census) == {'schema', 'scope_sha256', 'canonicalizer_sha256', 'source_paths',
                            'coverage_complete', 'all_legacy_writers_stopped', 'closure_evidence', 'entries', 'source_identities'} and
             census['schema'] == 'jope.native-global-census/v1' and
             census['scope_sha256'] == config['scope_sha256'] and
             census['canonicalizer_sha256'] == config['canonicalizer_sha256'] and
             census['source_paths'] == paths and census['coverage_complete'] is True and
             census['all_legacy_writers_stopped'] is True and census['closure_evidence'],
             'explicit complete historical and stopped-writer custody')
        need(census['source_identities']==self.physical_sources,
             'source physical identity differs from pinned census before first registry')
        # Referenced closure evidence is exact local custody; never just an unbound assertion.
        need(type(census['closure_evidence']) is list and 1 <= len(census['closure_evidence']) <= 256,
             'bounded closure evidence list')
        evidence_bytes = 0
        for evidence in census['closure_evidence']:
            need(set(evidence) == {'path', 'sha256', 'bytes'} and hash64(evidence['sha256']) and
                 type(evidence['bytes']) is int and 0 < evidence['bytes'] <= 4 * 1024**2,
                 'closed bounded writer/census evidence')
            evidence_bytes += evidence['bytes']
            need(evidence_bytes <= 32 * 1024**2, 'total closure evidence ceiling')
            exact_file(evidence)
        self.historical = {}
        need(type(census['entries']) is list and len(census['entries']) <= 1000000, 'bounded history')
        for entry in census['entries']:
            need(set(entry) == {'context_id', 'canonical_program_sha256', 'evaluation_key', 'disposition', 'source'} and
                 all(hash64(entry[k]) for k in ('context_id', 'canonical_program_sha256', 'evaluation_key')) and
                 entry['disposition'] in ('blocked', 'completed_cache_only') and
                 entry['source'] in paths + [r['path'] for r in census['closure_evidence']],
                 'closed canonical history entry with pinned source custody')
            k = digest({'schema': 'jope.native-canonical-context-program/v1',
                        'context_id': entry['context_id'], 'program_sha256': entry['canonical_program_sha256']})
            old = self.historical.get(k)
            self.historical[k] = 'blocked' if entry['disposition'] == 'blocked' or old == 'blocked' else 'completed_cache_only'
        # Full frozen verifiers run once per packet on indexed private copies.
        self.preflight = []
        for source in self.sources:
            self.tick()
            db, proof = verified_snapshot(source['path'], deadline=deadline, kind=source['kind'],
                                          expected_config=source['config'], max_bytes=config['max_snapshot_bytes'])
            db.close()
            self._source(source['path'])
            self.preflight.append({'path': source['path'], 'kind': source['kind'], 'proof': proof})
        p = Path(config['registry_path'])
        need(p.is_absolute() and '..' not in p.parts and all(not x.is_symlink() for x in (p, *p.parents)),
             'absolute owned guard path')
        need(str(p) not in paths and p.parent.is_dir(), 'new registry separate from every source')
        for suffix in ('-wal', '-shm'):
            sidecar = Path(str(p) + suffix)
            need(not sidecar.is_symlink(), 'registry sidecar symlink')
            if sidecar.exists():
                st = sidecar.stat()
                need(p.exists() and sidecar.is_file() and st.st_nlink == 1 and
                     (st.st_dev,st.st_ino) not in {(s['device'],s['inode']) for s in self.physical_sources.values()},
                     'orphan or source-aliased registry sidecar')
        identity = canonical({'schema': SCHEMA, 'scope_sha256': config['scope_sha256'],
            'canonicalizer_sha256': config['canonicalizer_sha256'], 'sources': self.sources,
            'physical_sources': self.physical_sources,
            'census_sha256': ref['sha256'], 'global_deadline_unix': GLOBAL_DEADLINE})
        created = False
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            # Refuse foreign/incompatible databases BEFORE any writable connection,
            # PRAGMA or CREATE. In particular, path inequality is not inode inequality.
            st = p.stat()
            need(p.is_file() and st.st_nlink == 1 and
                 (st.st_dev, st.st_ino) not in {(s['device'],s['inode']) for s in self.physical_sources.values()},
                 'registry must not alias/hardlink any source')
            for suffix in ('-wal', '-shm'):
                sidecar = Path(str(p) + suffix)
                if sidecar.exists():
                    need(sidecar.stat().st_nlink == 1, 'registry sidecar hardlink')
            self.db = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            try:
                self.db.execute('PRAGMA query_only=ON')
                rows = self.db.execute('SELECT body FROM config').fetchall()
                need(len(rows) == 1 and bytes(rows[0][0]) == identity,
                     'same global registry/census/physical sources; no reset or replacement')
                need({r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")} ==
                     {'config','claims','events'}, 'exact existing registry schema')
                self.verify()
            finally:
                self.db.close()
        else:
            os.close(fd)
            created = True
        self.db = sqlite3.connect(p, timeout=2, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA max_page_count=16384')
            if created:
                self.db.executescript('''CREATE TABLE config(body BLOB NOT NULL);
                CREATE TABLE claims(key TEXT PRIMARY KEY, body BLOB NOT NULL, state TEXT NOT NULL, result BLOB);
                CREATE TABLE events(seq INTEGER PRIMARY KEY, body BLOB NOT NULL, previous TEXT NOT NULL, sha TEXT NOT NULL);''')
            self.db.execute('BEGIN IMMEDIATE')
            rows = self.db.execute('SELECT body FROM config').fetchall()
            if not rows:
                self.db.execute('INSERT INTO config VALUES(?)', (identity,))
            else:
                need(len(rows) == 1 and bytes(rows[0][0]) == identity, 'same global registry/census; no reset or source removal')
            self.db.execute('COMMIT')
            self.verify()
            self.tick()
        except BaseException:
            if self.db.in_transaction:
                self.db.execute('ROLLBACK')
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def tick(self):
        need(self.now() < self.deadline and time.monotonic() - self.started < self.duration,
             'guard fixed wall/monotonic deadline')

    def _source(self, path):
        p = source_path(path)
        st = p.stat()
        need({'device':st.st_dev,'inode':st.st_ino} == self.physical_sources[path],
             'source physical identity changed; preserve prior attempted custody')
        return p

    def _event(self, value):
        row = self.db.execute('SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        seq, previous = (row['seq'] + 1, row['sha']) if row else (1, '0' * 64)
        body = canonical(value)
        sha = digest({'seq': seq, 'body': value, 'previous': previous})
        self.db.execute('INSERT INTO events VALUES(?,?,?,?)', (seq, body, previous, sha))

    def check(self, context_id, program, *, current=None):
        """Return cache-only requirement; raise for every uncertain prior alias.

        current is the exact already-leased outbox row, not a blanket owner skip.
        The source query runs inside its own read-only transaction. All lease
        ages, including expired and future clocks, remain attempted and blocked.
        """
        self.tick()
        need(self.canonicalize(program) == program, 'canonical candidate required')
        key = key_for(context_id, program)
        no_prior_attempt(self.historical.get(key) != 'blocked', 'historical failed/unknown/attempted canonical key')
        cache_only = self.historical.get(key) == 'completed_cache_only'
        seen_current = current is None
        for source in self.sources:
            p = self._source(source['path'])
            con = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True, isolation_level=None, timeout=2)
            con.row_factory = sqlite3.Row
            try:
                con.set_progress_handler(lambda: int(self.now() >= self.deadline or
                    time.monotonic() - self.started >= self.duration), 1000)
                con.execute('PRAGMA query_only=ON'); con.execute('BEGIN')
                if source['kind'] == 'outbox':
                    need(json.loads(con.execute('SELECT body FROM config').fetchone()[0]) == source['config'], 'live source config drift')
                    rows = con.execute("SELECT * FROM emissions WHERE json_extract(body,'$.emission.context_id')=?", (context_id,))
                    for row in rows:
                        body = json.loads(row['body']); e = body['emission']
                        need(hashlib.sha256(e['raw_program_utf8'].encode()).hexdigest() == e['raw_program_sha256'], 'live raw program hash')
                        if current is not None and source['path'] == current['path'] and row['id'] == current['lease']['emission_id']:
                            need(row['state'] == 'leased' and json.loads(row['lease']) == current['lease'] and
                                 digest(body) == current['body_sha256'] and self.canonicalize(e['raw_program_utf8']) == program and
                                 current['lease']['started_at'] <= self.now() < current['lease']['expires_at'],
                                 'exact current live lease/body; no unknown-clock retry')
                            seen_current = True
                            continue
                        need(row['state'] in {'recorded'} | BLOCKED | SUCCESS, 'known outbox state')
                        if row['state'] == 'recorded':
                            continue
                        if self.canonicalize(e['raw_program_utf8']) == program:
                            no_prior_attempt(row['state'] in SUCCESS, 'dynamic failed/unknown/leased canonical alias')
                            cache_only = True
                else:
                    for alias in queue_aliases(con):
                        if alias['context_id'] != context_id or self.canonicalize(alias['program']) != program:
                            continue
                        for state in alias['states']:
                            if state['state'] == 'completed':cache_only = True
                            else:no_prior_attempt(state['state'] == 'pending' and state['attempts'] == 0,
                                 'dynamic historical queue attempt is never repeated')

            finally:
                con.close()
            self._source(source['path'])
            self.tick()
        need(seen_current, 'current lease must belong to declared global sources')
        row = self.db.execute('SELECT state FROM claims WHERE key=?', (key,)).fetchone()
        if row:
            if getattr(self, '_own_reservation', None) == key:
                need(row['state'] == 'reserved', 'postclaim owned reservation')
            else:
                no_prior_attempt(row['state'] == 'completed', 'global canonical key already reserved/attempted; never retry')
                cache_only = True
        return cache_only

    def execute_once(self, context_id, program, evaluation_key, current, callback):
        """Acquire one permanent global key; post-claim guard then external call.

        Successful execution alone is still uncertain until exact S3/outbox
        closure calls complete(). Returning or raising never releases the key.
        """
        need(hash64(evaluation_key) and callable(callback), 'qualified execution identity/callback')
        need(type(current) is dict and set(current) == {'path', 'lease', 'body_sha256'} and
             hash64(current['body_sha256']), 'exact leased outbox custody required for execution')
        key = key_for(context_id, program)
        claim = {'key': key, 'context_id': context_id, 'program_sha256': hashlib.sha256(program.encode()).hexdigest(),
                 'evaluation_key': evaluation_key, 'packet_sha256': self.packet,
                 'current': current, 'at': self.now()}
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.check(context_id, program, current=current):
                raise CacheRequired('completed key requires verified cache; absence cannot authorize execution')
            need(self.db.execute('SELECT count(*) FROM claims').fetchone()[0] < 1000000, 'global key capacity')
            self.db.execute('INSERT INTO claims VALUES(?,?,?,NULL)', (key, canonical(claim), 'reserved'))
            self._event({'kind': 'reserved', 'claim': claim})
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
        try:
            # check() sees our permanent reservation, so permit only this exact
            # row via a transaction-local state-independent alias scan below.
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT * FROM claims WHERE key=?', (key,)).fetchone()
            need(row['state'] == 'reserved' and json.loads(row['body']) == claim, 'one-use owned reservation')
            self._own_reservation = key
            try:
                cache_only = self._postclaim_check(context_id, program, current, key)
            finally:
                self._own_reservation = None
            need(not cache_only, 'completed alias appeared after claim; never execute')
            self.db.execute("UPDATE claims SET state='started' WHERE key=?", (key,))
            self._event({'kind': 'started', 'key': key, 'at': self.now()})
            self.db.execute('COMMIT')
            self.tick()
            self._execution_boundary(current)
            return callback()
        except BaseException as error:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            self._mark_uncertain(key, error)
            raise

    def _postclaim_check(self, context_id, program, current, own_key):
        # check supports one owned reservation; every other key still fails.
        return self.check(context_id, program, current=current)

    def _execution_boundary(self, current):
        """Last local operation before callback: exact physical live lease/body."""
        p = self._source(current['path'])
        db = sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,isolation_level=None,timeout=2)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
            row = db.execute('SELECT * FROM emissions WHERE id=?',(current['lease']['emission_id'],)).fetchone()
            need(row is not None and row['state']=='leased' and json.loads(row['lease'])==current['lease'] and
                 digest(json.loads(row['body']))==current['body_sha256'], 'exact lease/body at execution boundary')
        finally:
            db.close()
        self._source(current['path'])
        self.tick()
        need(current['lease']['started_at'] <= self.now() < current['lease']['expires_at'],
             'lease expired/unknown clock at execution boundary')

    def _mark_uncertain(self, key, error):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute("UPDATE claims SET state='unknown' WHERE key=?", (key,))
            self._event({'kind': 'unknown', 'key': key, 'error': type(error).__name__ + ': ' + str(error)[:1000], 'at': self.now()})
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise

    def complete(self, context_id, program, record_ref):
        """Caller invokes only after existing exact seven-artifact/outbox proof."""
        from neural_outbox import ref
        need(ref(record_ref), 'exact successful result reference')
        key = key_for(context_id, program)
        self.tick(); self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT * FROM claims WHERE key=?', (key,)).fetchone()
            need(row is not None and row['state'] == 'started' and json.loads(row['body'])['packet_sha256'] == self.packet,
                 'only owned started execution can complete')
            self.db.execute("UPDATE claims SET state='completed',result=? WHERE key=?", (canonical(record_ref), key))
            self._event({'kind': 'completed', 'key': key, 'record_ref': record_ref, 'at': self.now()})
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise

    def verify(self):
        self.db.execute('BEGIN')
        try:
            previous = '0' * 64; claims = {}; results = {}
            for seq, row in enumerate(self.db.execute('SELECT * FROM events ORDER BY seq'), 1):
                b = json.loads(row['body'])
                need(row['seq'] == seq and row['previous'] == previous and
                     row['sha'] == digest({'seq': seq, 'body': b, 'previous': previous}), 'global custody chain')
                previous = row['sha']; kind = b['kind']
                if kind == 'reserved':
                    c = b['claim']; need(c['key'] not in claims, 'global canonical key reused')
                    claims[c['key']] = [c, 'reserved']
                else:
                    k = b['key']; need(k in claims, 'unknown custody key')
                    allowed = {'started': {'reserved'}, 'unknown': {'reserved','started'}, 'completed': {'started'}}
                    need(kind in allowed and claims[k][1] in allowed[kind], 'global custody order')
                    claims[k][1] = kind
                    if kind == 'completed':results[k] = b['record_ref']
            rows = self.db.execute('SELECT * FROM claims').fetchall()
            need(len(rows) == len(claims), 'global custody materialization count')
            for row in rows:
                need([json.loads(row['body']), row['state']] == claims.get(row['key']) and
                     (json.loads(row['result']) if row['result'] else None) == results.get(row['key']),
                     'global custody materialization')
            self.db.execute('COMMIT')
            return {'claims': len(claims), 'head_sha256': previous}
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
