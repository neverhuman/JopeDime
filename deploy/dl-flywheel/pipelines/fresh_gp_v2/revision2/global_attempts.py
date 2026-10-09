"""Cold whole-context ownership and permanent no-retry canonical attempt holds.

This registry is new and local to xbabe3. It does not copy/modify the active
xbabe2 native scorer registry. A separately qualified cross-host ownership
certificate is mandatory; a local SQLite file alone is not global exclusivity.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import time
from jope_flywheel.program_gate import request_identity
from jope_flywheel.program_io import bound, need, sha
from jope_flywheel.protocol import Context, canonical, digest


def alias(record, native, variant_contexts):
    """Canonicalize *all* captured rows, including null-key/outbox aliases.

    The cold reader supplies the original body in `payload`, plus source row
    id/state/attempts. Unknown syntax or identity blocks qualification globally.
    An absent evaluation_key is not evidence that no native attempt occurred.
    """
    payload = record['payload']
    emission = payload.get('emission', payload)
    desc = emission.get('input_descriptor', {})
    ctx = payload.get('context', desc.get('context'))
    variant = emission.get('variant_identity', emission.get('data_run_id'))
    if ctx is not None:
        context = Context(**ctx)
    else:
        need(variant in variant_contexts, 'unresolved historical context alias blocks scope')
        context = Context(**variant_contexts[variant])
    program = payload.get('program', emission.get('raw_program_utf8'))
    need(type(program) is str, 'unresolved historical program alias blocks scope')
    program = native.canonical_program(program)
    key = request_identity(context, program)
    oldkey = payload.get('evaluation_key')
    need(oldkey is None or oldkey == key, 'historical key disagrees with canonical alias')
    return {'context_id': context.identity, 'program': program, 'evaluation_key': key,
            'state': record['state'], 'source_row_id': str(record['id'])}


class History:
    """Pinned immutable captures: rehash every source before each fit.

    Read-only snapshot creation/native queue verification is an external cold
    qualification step, never simulated by the metadata inventory. Required
    refs must include every legacy queue and outbox, not just selected hits.
    """
    def __init__(self, refs, native, variant_contexts):
        need(len(refs) >= 20 and len({x['path'] for x in refs}) == len(refs), 'full history source census required')
        self.refs, self.native, self.variant_contexts = refs, native, variant_contexts
        self.pin = digest(refs)

    def check(self):
        rows, kinds = {}, []
        for ref in self.refs:
            doc = json.loads(bound(ref, 64*1024**2))
            need(doc['schema'] == 'jope.fresh-gp-readonly-history/v1' and doc['physical_readonly'] is True and
                 doc['source_verified'] is True and doc['kind'] in ('variant_queue','original_outbox','continuation_outbox'),
                 'qualified physical readonly history required')
            need(doc['capture_complete'] is True and len(doc['records']) == doc['source_row_count'],
                 'historical source must not be filtered/truncated')
            kinds.append(doc['kind'])
            for row in doc['records']:
                a = alias(row, self.native, self.variant_contexts)
                # Any observed alias is conservatively held, including pending
                # or zero-attempt jobs; claim-free is not global no-work proof.
                rows.setdefault(a['evaluation_key'], []).append({**a, 'capture': ref})
        need(kinds.count('variant_queue') == 18 and 'original_outbox' in kinds and
             'continuation_outbox' in kinds, '18 queues plus original/continuation outboxes')
        return rows


def ownership(proof, config, descriptors, now):
    """Verify cold launch authority *and* the continuing cross-host broker lease.

    Source-body verification is delegated to its separately reviewed read-only
    producer. Config pins that producer/census; this module never mints proof.
    """
    need(proof['schema'] == 'jope.fresh-gp-context-ownership/v1' and proof['passed'] is True and
         proof['host'] == 'xbabe3' and proof['campaign_sha256'] == config['campaign_sha256'] and
         proof['producer_sha256'] == config['ownership_producer_sha256'] and
         proof['history_census_sha256'] == digest(config['history_sources']) and
         proof['selection_sha256'] == config['selection_sha256'], 'whole-context ownership/source identity')
    need(0 <= now-proof['checked_at_unix'] < 120 and now < proof['lease_expires_unix'] <= config['work_deadline_unix'] and
         proof['global_deadline_unix'] == config['global_deadline_unix'], 'live global ownership lease required')
    need(proof['legacy_variant_workers_closed'] is True and proof['legacy_launch_controllers_closed'] is True and
         proof['active_xbabe2_source_kind'] == 'native_s1_context' and proof['active_native_context_overlap'] == [] and
         proof['other_variant_context_overlap'] == [] and proof['all_legacy_units_physically_closed'] is True,
         'active or unresolved cross-host context overlap')
    need(proof['context_descriptors'] == descriptors and len(descriptors) == 128 and
         len({x['context_id'] for x in descriptors}) == 128 and
         len({x['variant_identity'] for x in descriptors}) == 128, 'exact whole-context ownership support')
    return proof


class Attempts:
    def __init__(self, path, config, *, create=False):
        self.path, self.config = Path(path), config
        p = self.path
        need(p.is_absolute() and all(not x.is_symlink() for x in (p,*p.parents)), 'owned nonsymlink registry')
        for tail in ('-wal','-shm','-journal'):
            need(not Path(str(p)+tail).is_symlink(), 'unsafe registry sidecar')
        if create:
            fd = os.open(p, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
            os.close(fd)
        else:
            need(p.is_file(), 'existing cold registry required')
        self.db = sqlite3.connect(p.as_uri()+'?mode=rw', uri=True, timeout=.5, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        if create:
            self.db.executescript('''
            CREATE TABLE binding(id INTEGER PRIMARY KEY CHECK(id=1),sha TEXT NOT NULL);
            CREATE TABLE contexts(context_id TEXT PRIMARY KEY,variant TEXT UNIQUE NOT NULL,
              owner TEXT NOT NULL,descriptor TEXT NOT NULL,partition TEXT NOT NULL,proof_sha TEXT NOT NULL);
            CREATE TABLE attempts(evaluation_key TEXT PRIMARY KEY,context_id TEXT NOT NULL,
              program TEXT NOT NULL,owner TEXT NOT NULL,state TEXT NOT NULL,job_id TEXT,lease_token TEXT,
              receipt_sha TEXT,history_sha TEXT NOT NULL,created REAL NOT NULL);
            CREATE TABLE events(seq INTEGER PRIMARY KEY,body TEXT NOT NULL,prev_sha TEXT NOT NULL,sha TEXT NOT NULL);
            ''')
            self.db.execute('INSERT INTO binding VALUES(1,?)',(digest(config),))
        need(self.db.execute('SELECT sha FROM binding WHERE id=1').fetchone()[0] == digest(config), 'one-use registry campaign binding')

    def close(self):
        self.db.close()

    def event(self, body):
        body = {**body, 'state_sha256': self.state_sha()}
        row = self.db.execute('SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        seq, prev = (row['seq']+1,row['sha']) if row else (1,'0'*64)
        self.db.execute('INSERT INTO events VALUES(?,?,?,?)', (seq, canonical(body).decode(),prev,
            digest({'seq':seq,'body':body,'prev_sha256':prev})))

    def state_sha(self):
        return digest({name:[dict(r) for r in self.db.execute('SELECT * FROM '+name+' ORDER BY 1')]
                       for name in ('binding','contexts','attempts')})

    def reserve_contexts(self, descriptors, proof, owner):
        # The immutable shard owner table is frozen before launch. A caller
        # cannot claim the same context under a different spelling/shard alias.
        rows = [r for r in descriptors if self.config['context_owners'].get(r['context_id']) == owner]
        need(rows and owner in self.config['shard_ids'] and all(r['partition'] in ('train','validation') for r in rows),
             'frozen whole-context shard allocation')
        ownership(proof, self.config, descriptors, time.time())
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify()
            for r in rows:
                self.db.execute('INSERT INTO contexts VALUES(?,?,?,?,?,?)',
                    (r['context_id'],r['variant_identity'],owner,canonical(r).decode(),r['partition'],digest(proof)))
            self.event({'kind':'contexts_reserved','owner':owner,'contexts':rows,'proof_sha256':digest(proof)})
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise

    def assert_owner(self, context, descriptor, owner):
        self.verify()
        r = self.db.execute('SELECT * FROM contexts WHERE context_id=?',(context.identity,)).fetchone()
        want = {'context_id':context.identity,'variant_identity':descriptor['variant_identity'],
                'descriptor_sha256':digest(descriptor),'partition':descriptor['partition']}
        need(r is not None and r['owner'] == owner and json.loads(r['descriptor']) == want,
             'whole context is not owned by this frozen shard')

    def begin_attempt(self, context, program, owner, job, history, history_pin):
        key = request_identity(context,program)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify()
            r = self.db.execute('SELECT owner FROM contexts WHERE context_id=?',(context.identity,)).fetchone()
            need(r is not None and r['owner'] == owner, 'unowned native context')
            if key in history:
                old = self.db.execute('SELECT * FROM attempts WHERE evaluation_key=?',(key,)).fetchone()
                if old is None:
                    self.db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (key,context.identity,program,owner,'prior_hold',None,None,None,history_pin,time.time()))
                    self.event({'kind':'historical_hold','key':key,'aliases':history[key]})
                self.db.execute('COMMIT')
                raise ValueError('historical attempted/completed/unknown alias; never re-execute')
            need(self.db.execute('SELECT 1 FROM attempts WHERE evaluation_key=?',(key,)).fetchone() is None,
                 'permanent attempt already reserved; no retry')
            self.db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)',
                (key,context.identity,program,owner,'started',job['id'],job['lease_token'],None,history_pin,time.time()))
            self.event({'kind':'native_intent','key':key,'context_id':context.identity,'program_sha256':sha(program.encode()),
                        'owner':owner,'job_id':job['id'],'lease_token':job['lease_token']})
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
        return key

    def require_started(self, context, program, owner):
        r = self.db.execute('SELECT * FROM attempts WHERE evaluation_key=?',(request_identity(context,program),)).fetchone()
        need(r is not None and r['owner'] == owner and r['state'] in ('started','completed_verified'),
             'no owned measured attempt')
        return dict(r)

    def complete(self, context, program, owner, receipt, verifier):
        # Caller must provide unchanged native verified_measurement + exact
        # VersionId artifact verification. A status boolean is not completion.
        need(verifier(context,program,receipt) is True, 'full exact-version measurement verification required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify()
            r = self.require_started(context,program,owner)
            need(r['state'] == 'started', 'completion already recorded')
            self.db.execute("UPDATE attempts SET state='completed_verified',receipt_sha=? WHERE evaluation_key=?",
                            (digest(receipt),r['evaluation_key']))
            self.event({'kind':'completed_verified','key':r['evaluation_key'],'receipt_sha256':digest(receipt)})
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise

    def verify(self):
        owned = not self.db.in_transaction
        if owned:self.db.execute("BEGIN")
        try:return self._verify()
        finally:
            if owned and self.db.in_transaction:self.db.execute("ROLLBACK")

    def _verify(self):
        prev='0'*64; count=0; tail=None
        for r in self.db.execute('SELECT * FROM events ORDER BY seq'):
            count+=1
            need(r['seq']==count and r['prev_sha']==prev and
                 digest({'seq':count,'body':json.loads(r['body']),'prev_sha256':prev})==r['sha'], 'registry event chain')
            prev=r['sha']
            tail=json.loads(r['body'])
        if tail is not None:
            need(tail['state_sha256']==self.state_sha(), 'registry state differs from durable events')
        else:
            need(self.db.execute('SELECT count(*) FROM contexts').fetchone()[0]==0 and
                 self.db.execute('SELECT count(*) FROM attempts').fetchone()[0]==0, 'unlogged registry mutation')
        return {'events':count,'head_sha256':prev,
                'contexts':self.db.execute('SELECT count(*) FROM contexts').fetchone()[0],
                'held_attempts':self.db.execute('SELECT count(*) FROM attempts').fetchone()[0]}
