"""Durable bounded DAG queue for the experimental flywheel.

Expired work becomes needs_reconcile, never a silent retry of a possibly completed
external evaluation or S3 write. This module never runs commands or reads datasets.
"""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
from pathlib import Path
from .protocol import canonical, digest, sha256

STAGES = ('augment','encode','baseline','propose','evaluate','publish','admit','train','compare')
STATES = ('pending','running','completed','failed','needs_reconcile','cancelled')


class Queue:
    def __init__(self, path, *, max_jobs=1000000, max_payload_bytes=65536):
        self.path = Path(path)
        if self.path.is_symlink(): raise ValueError('queue may not be a symlink')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if type(max_jobs) is not int or max_jobs<1 or type(max_payload_bytes) is not int or not 1<=max_payload_bytes<=1048576:
            raise ValueError('invalid queue bounds')
        self.max_jobs,self.max_payload_bytes=max_jobs,max_payload_bytes
        self.db=sqlite3.connect(self.path,timeout=20,isolation_level=None)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY, stage TEXT NOT NULL, plan_sha TEXT NOT NULL, payload TEXT NOT NULL,
          spec_sha TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
          lease_token TEXT, worker TEXT, lease_deadline REAL, attempts INTEGER NOT NULL DEFAULT 0,
          receipt_sha TEXT, reason TEXT);
        CREATE TABLE IF NOT EXISTS dependencies(job TEXT NOT NULL REFERENCES jobs(id),
          dependency TEXT NOT NULL REFERENCES jobs(id), PRIMARY KEY(job,dependency));
        CREATE TABLE IF NOT EXISTS transitions(seq INTEGER PRIMARY KEY, at REAL NOT NULL,
          job TEXT NOT NULL REFERENCES jobs(id), state TEXT NOT NULL, detail TEXT NOT NULL,
          prev_sha TEXT NOT NULL, sha TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state,stage,created);
        ''')
    def close(self): self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    @staticmethod
    def _name(v):
        if not isinstance(v,str) or not v or len(v)>200 or any(ord(c)<32 for c in v):raise ValueError('invalid identifier')
        return v
    def _event(self,job,state,at,detail):
        tail=self.db.execute('SELECT seq,sha FROM transitions ORDER BY seq DESC LIMIT 1').fetchone()
        seq=tail['seq']+1 if tail else 1;prev=tail['sha'] if tail else '0'*64
        rec={'seq':seq,'at':at,'job':job,'state':state,'detail':detail,'prev_sha':prev}
        self.db.execute('INSERT INTO transitions VALUES(?,?,?,?,?,?,?)',(seq,at,job,state,canonical(detail).decode(),prev,digest(rec)))
    def enqueue(self,job_id,stage,plan_sha256,payload,dependencies=()):
        return self.enqueue_many([{'job_id':job_id,'stage':stage,'plan_sha256':plan_sha256,
                                   'payload':payload,'dependencies':dependencies}])[0]
    def enqueue_many(self,jobs):
        """Atomically add an ordered DAG packet; return one created flag per job.

        Dependencies may reference existing jobs or earlier entries in this
        packet. Capacity, identity or dependency failure rolls back every new
        job and its transition, so evaluation cannot outlive a missing publish
        stage. Completed matching jobs remain idempotent even at capacity.
        """
        if type(jobs) not in (list,tuple) or not 1<=len(jobs)<=128:
            raise ValueError('bounded ordered job packet required')
        prepared=[];ids=set()
        for item in jobs:
            if type(item) is not dict or not {'job_id','stage','plan_sha256','payload'}<=set(item) or set(item)-{'job_id','stage','plan_sha256','payload','dependencies'}:
                raise ValueError('invalid job packet fields')
            job_id=item['job_id'];stage=item['stage'];plan=item['plan_sha256'];payload=item['payload']
            self._name(job_id);sha256(plan)
            if job_id in ids:raise ValueError('duplicate job in packet')
            ids.add(job_id)
            if stage not in STAGES or not isinstance(payload,dict):raise ValueError('invalid stage/payload')
            encoded=canonical(payload)
            if len(encoded)>self.max_payload_bytes:raise ValueError('payload limit')
            deps=sorted(set(item.get('dependencies',())))
            if len(deps)>128 or any(self._name(d)==job_id for d in deps):raise ValueError('invalid dependency')
            spec=digest({'stage':stage,'plan_sha256':plan,'payload':payload,'dependencies':deps})
            prepared.append((job_id,stage,plan,encoded,deps,spec))
        self.db.execute('BEGIN IMMEDIATE')
        try:
            count=self.db.execute('SELECT count(*) FROM jobs').fetchone()[0];created=[]
            for job_id,stage,plan,encoded,deps,spec in prepared:
                old=self.db.execute('SELECT spec_sha FROM jobs WHERE id=?',(job_id,)).fetchone()
                if old:
                    if old['spec_sha']!=spec:raise ValueError('job identity collision')
                    created.append(False);continue
                if count>=self.max_jobs:raise ValueError('queue capacity reached')
                for dep in deps:
                    r=self.db.execute('SELECT plan_sha FROM jobs WHERE id=?',(dep,)).fetchone()
                    if not r or r['plan_sha']!=plan:raise ValueError('dependency absent or different plan')
                now=time.time()
                self.db.execute('INSERT INTO jobs(id,stage,plan_sha,payload,spec_sha,state,created,updated) VALUES(?,?,?,?,?,?,?,?)',
                                (job_id,stage,plan,encoded.decode(),spec,'pending',now,now))
                self.db.executemany('INSERT INTO dependencies VALUES(?,?)',[(job_id,d) for d in deps])
                self._event(job_id,'pending',now,{'spec_sha256':spec,'dependencies':deps})
                count+=1;created.append(True)
            self.db.execute('COMMIT');return created
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def _expire(self,now):
        for row in self.db.execute("SELECT id FROM jobs WHERE state='running' AND lease_deadline<=?",(now,)).fetchall():
            self.db.execute("UPDATE jobs SET state='needs_reconcile',updated=?,reason='lease_expired_unknown_external_outcome' WHERE id=?",(now,row['id']))
            self._event(row['id'],'needs_reconcile',now,{'reason':'lease_expired_unknown_external_outcome'})
    def claim(self,worker,stages,*,lease_seconds=300,max_running=1,now=None):
        self._name(worker);stages=tuple(sorted(set(stages)))
        if not stages or any(s not in STAGES for s in stages) or type(max_running) is not int or not 1<=max_running<=64 or type(lease_seconds) is not int or not 1<=lease_seconds<=3600:
            raise ValueError('invalid admission bounds')
        now=time.time() if now is None else float(now)
        canonical(now)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._expire(now)
            if self.db.execute("SELECT count(*) FROM jobs WHERE state IN ('running','needs_reconcile')").fetchone()[0]>=max_running:
                self.db.execute('COMMIT');return None
            marks=','.join('?' for _ in stages)
            row=self.db.execute(f"""SELECT * FROM jobs j WHERE state='pending' AND stage IN ({marks})
              AND NOT EXISTS(SELECT 1 FROM dependencies d JOIN jobs p ON p.id=d.dependency
                             WHERE d.job=j.id AND p.state!='completed') ORDER BY created,id LIMIT 1""",stages).fetchone()
            if row is None:self.db.execute('COMMIT');return None
            token=uuid.uuid4().hex;deadline=now+lease_seconds
            self.db.execute("UPDATE jobs SET state='running',updated=?,worker=?,lease_token=?,lease_deadline=?,attempts=attempts+1 WHERE id=?",(now,worker,token,deadline,row['id']))
            self._event(row['id'],'running',now,{'worker':worker,'lease_token':token,'lease_deadline':deadline})
            self.db.execute('COMMIT')
            return dict(id=row['id'],stage=row['stage'],plan_sha256=row['plan_sha'],payload=json.loads(row['payload']),lease_token=token,lease_deadline=deadline)
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def finish(self,job_id,lease_token,receipt_sha256,*,success,reason=None,now=None):
        sha256(receipt_sha256)
        if type(success) is not bool:raise ValueError('success must be boolean')
        now=time.time() if now is None else float(now);canonical(now)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._expire(now)
            r=self.db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
            state='completed' if success else 'failed'
            if r and r['state']==state and r['receipt_sha']==receipt_sha256 and r['lease_token']==lease_token and r['reason']==reason:
                self.db.execute('COMMIT');return False
            if not r or r['state']!='running' or r['lease_token']!=lease_token:
                self.db.execute('COMMIT')  # Preserve expiration evidence even when finish is refused.
                raise ValueError('lease ownership/expiry mismatch; reconcile required')
            if reason is not None:self._name(reason)
            self.db.execute('UPDATE jobs SET state=?,updated=?,receipt_sha=?,reason=? WHERE id=?',(state,now,receipt_sha256,reason,job_id))
            self._event(job_id,state,now,{'receipt_sha256':receipt_sha256,'reason':reason})
            self.db.execute('COMMIT');return True
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def reconcile(self,job_id,receipt_sha256,*,completed,reason):
        """External exact-identity readback resolves expiry. No automatic retry path."""
        sha256(receipt_sha256);self._name(reason)
        if type(completed) is not bool:raise ValueError('completed must be boolean')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            r=self.db.execute('SELECT state FROM jobs WHERE id=?',(job_id,)).fetchone()
            if not r or r['state']!='needs_reconcile':raise ValueError('not awaiting reconciliation')
            now=time.time();state='completed' if completed else 'failed'
            self.db.execute('UPDATE jobs SET state=?,updated=?,receipt_sha=?,reason=? WHERE id=?',(state,now,receipt_sha256,reason,job_id))
            self._event(job_id,state,now,{'receipt_sha256':receipt_sha256,'reason':reason,'reconciled':True})
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def snapshot(self):
        counts={s:{v:0 for v in STATES} for s in STAGES}
        for r in self.db.execute('SELECT stage,state,count(*) n FROM jobs GROUP BY stage,state'):counts[r['stage']][r['state']]=r['n']
        return {'schema':'jope.flywheel-queue-snapshot/v1','at':time.time(),'counts':counts,
                'total_jobs':sum(sum(x.values()) for x in counts.values()),
                'failed_or_unknown':[dict(r) for r in self.db.execute("SELECT id,stage,state,reason FROM jobs WHERE state IN ('failed','needs_reconcile') ORDER BY updated DESC LIMIT 100")]}
    def verify(self):
        prev='0'*64;seq=0;latest={};specs={};leases={};attempts={}
        for r in self.db.execute('SELECT * FROM transitions ORDER BY seq'):
            seq+=1;obj={'seq':r['seq'],'at':r['at'],'job':r['job'],'state':r['state'],'detail':json.loads(r['detail']),'prev_sha':r['prev_sha']}
            if r['seq']!=seq or r['prev_sha']!=prev or digest(obj)!=r['sha']:raise ValueError('queue transition integrity failure')
            prev=r['sha'];latest[r['job']]=obj
            if r['state']=='running':
                leases[r['job']]=obj['detail'];attempts[r['job']]=attempts.get(r['job'],0)+1
            if r['state']=='pending':specs[r['job']]=obj['detail']['spec_sha256']
        jobs=list(self.db.execute('SELECT * FROM jobs'))
        if len(jobs)!=len(latest):raise ValueError('job/transition closure mismatch')
        for row in jobs:
            deps=[x[0] for x in self.db.execute('SELECT dependency FROM dependencies WHERE job=? ORDER BY dependency',(row['id'],))]
            spec=digest({'stage':row['stage'],'plan_sha256':row['plan_sha'],'payload':json.loads(row['payload']),'dependencies':deps})
            event=latest[row['id']];lease=leases.get(row['id'],{})
            if row['attempts']!=attempts.get(row['id'],0) or any(row[k]!=lease.get(k) for k in ('worker','lease_token','lease_deadline')):
                raise ValueError('job lease differs from transition history')
            if spec!=row['spec_sha'] or spec!=specs[row['id']] or event['state']!=row['state']:
                raise ValueError('job state/spec differs from immutable transition history')
            if 'receipt_sha256' in event['detail'] and event['detail']['receipt_sha256']!=row['receipt_sha']:
                raise ValueError('job receipt differs from transition')
        return {'events':seq,'head_sha256':prev}
