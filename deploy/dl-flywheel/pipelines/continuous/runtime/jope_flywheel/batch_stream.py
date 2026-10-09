"""Durable exhaustive, shuffled batches with distinct parent lineages.

This is a planner/exposure journal, not an optimizer or proof of model execution.
A manifest includes every eligible, pending and refused data_run; validation is
never reserved for training. Each epoch consumes every eligible training run once.
Fingerprint = hash(sorted actual data_run IDs): changing order/epoch/manifest
cannot disguise a repeated batch. Finite support can run out of unique batches;
Waiting reports that condition, never a fictional complete epoch or a redraw.

One reserved batch is allowed. Persist its intent before computing. A crash leaves
it pending and blocks new reservations. commit is idempotent only for the exact
same verified execution evidence; resume never silently retries an unknown batch.
The trainer owns model/checkpoint validation and must call commit only after real
optimizer execution and durable native batch/GP evidence. This module reads no
paths contained in payloads, accesses no network/GPU and performs no DB admission.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,itertools,json,math,os,sqlite3,time
from pathlib import Path
from .protocol import canonical,digest,sha256

SCHEMA='jope.flywheel-batch-manifest/v1'
RUN_SCHEMA='jope.flywheel-data-run/v1'
MAX_ROWS=100000
MAX_MANIFEST=64*1024**2
MAX_EVENT=16*1024**2
MAX_ATTEMPTS=4096
RUN_KEYS={'schema','variant_identity','context_id','dataset_sha256','target_sha256','split_sha256','protocol_sha256','cut_id'}


def need(ok,msg):
    if not ok:raise ValueError(msg)
def byte_sha(b):return hashlib.sha256(b).hexdigest()
def decode(raw):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    d=json.loads(raw,object_pairs_hook=pairs);canonical(d);return d

def data_run_id(run):
    need(type(run)is dict and set(run)==RUN_KEYS and run['schema']==RUN_SCHEMA,'exact data_run schema')
    for k in RUN_KEYS-{'schema','cut_id'}:sha256(run[k])
    need(type(run['cut_id'])is str and 0<len(run['cut_id'])<=256,'explicit cut identity')
    return digest(run)

def validate_manifest(raw,expected_sha256):
    need(type(raw)is bytes and len(raw)<=MAX_MANIFEST and byte_sha(raw)==sha256(expected_sha256),'manifest hash/size')
    m=decode(raw);need(set(m)=={'schema','snapshot_sha256','role_authority_sha256','rows','metadata'}and m['schema']==SCHEMA,'closed manifest schema')
    sha256(m['snapshot_sha256']);sha256(m['role_authority_sha256'])
    need(type(m['rows'])is list and 0<len(m['rows'])<=MAX_ROWS,'bounded full membership')
    rows={};roles={};variants={}
    for row in m['rows']:
        need(set(row)=={'data_run','data_run_id','parent_lineage_sha256','partition','status','reason','payload'},'closed data_run row')
        identity=data_run_id(row['data_run']);need(identity==row['data_run_id']and identity not in rows,'duplicate/forged data_run')
        parent=sha256(row['parent_lineage_sha256']);role=row['partition']
        need(role in('train','validation')and roles.setdefault(parent,role)==role,'parent lineage role conflict')
        # A single variant cannot be relabelled into another parent lineage.
        variant=row['data_run']['variant_identity'];need(variants.setdefault(variant,parent)==parent,'variant parent collision')
        need(row['status']in('eligible','pending','refused'),'explicit eligibility state')
        need((row['status']=='eligible'and row['reason']is None)or
             (row['status']!='eligible'and type(row['reason'])is str and 0<len(row['reason'])<=2000),'explicit refusal/pending reason')
        need(type(row['payload'])is dict and len(canonical(row['payload']))<=128*1024,'bounded opaque row payload')
        rows[identity]=row
    return m,rows


def fingerprint(ids):
    need(len(ids)==len(set(ids))and len(ids)>0,'nonempty unique batch members')
    return digest({'schema':'jope.flywheel-batch-membership/v1','data_run_ids':sorted(ids)})


@dataclass(frozen=True)
class Batch:
    batch_id:str
    fingerprint:str
    manifest_sha256:str
    epoch:int
    index:int
    checkpoint_before:str
    ordered_data_run_ids:tuple[str,...]
    rows:tuple[dict,...]


@dataclass(frozen=True)
class Waiting:
    reason:str
    remaining:int
    epoch:int
    search_attempts:int


class BatchStream:
    def __init__(self,db_path,manifest_bytes,expected_manifest_sha256,*,seed,batch_size,initial_checkpoint_sha256):
        need(type(seed)is int and 0<=seed<2**64,'uint64 seed')
        need(type(batch_size)is int and 1<=batch_size<=256,'batch size1..256')
        sha256(initial_checkpoint_sha256)
        m,rows=validate_manifest(manifest_bytes,expected_manifest_sha256)
        path=Path(db_path);need(path.is_absolute()and'..'not in path.parts and all(not p.is_symlink()for p in(path,*path.parents)),'absolute non-symlink DB path')
        for suffix in ('-wal','-shm'):need(not Path(str(path)+suffix).is_symlink(),'SQLite sidecar symlink')
        self.db=sqlite3.connect(path,timeout=30,isolation_level=None);self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA foreign_keys=ON')
        self.config={'schema':'jope.flywheel-batch-stream/v1','seed':seed,'batch_size':batch_size,'initial_checkpoint_sha256':initial_checkpoint_sha256}
        self.db.executescript('''CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS manifests(sha TEXT PRIMARY KEY,body BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS epochs(id INTEGER PRIMARY KEY,manifest TEXT NOT NULL,state TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS members(epoch INTEGER NOT NULL,run TEXT NOT NULL,batch TEXT,PRIMARY KEY(epoch,run));
          CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,fingerprint TEXT UNIQUE NOT NULL,manifest TEXT NOT NULL,
            epoch INTEGER NOT NULL,batch_index INTEGER NOT NULL,body TEXT NOT NULL,state TEXT NOT NULL,commit_body TEXT);
          CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,kind TEXT NOT NULL,body TEXT NOT NULL,previous TEXT NOT NULL,sha TEXT NOT NULL);
        ''')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old=self._get('config')
            if old is None:
                self._set('config',self.config);self._set('checkpoint',initial_checkpoint_sha256);self._set('optimizer_step',0)
                self.db.execute('INSERT INTO manifests VALUES(?,?)',(expected_manifest_sha256,manifest_bytes))
                self._set('manifest',expected_manifest_sha256);self._set('epoch',0)
                self._event('stream.created',{'config':self.config,'manifest_sha256':expected_manifest_sha256})
            else:
                need(old==self.config,'stream seed/batch/checkpoint-origin mismatch')
                need(self._get('manifest')==expected_manifest_sha256,'resume dataset manifest mismatch; explicit adopt_manifest required')
                stored=self.db.execute('SELECT body FROM manifests WHERE sha=?',(expected_manifest_sha256,)).fetchone()
                need(stored is not None and bytes(stored[0])==manifest_bytes,'stored manifest bytes differ')
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            self.db.close();raise
        self.manifest,self.rows=m,rows
        self.manifest_sha256=expected_manifest_sha256
        self.verify()

    def close(self):self.db.close()
    def _get(self,key):
        row=self.db.execute('SELECT value FROM state WHERE key=?',(key,)).fetchone()
        return None if row is None else decode(row[0])
    def _set(self,key,value):self.db.execute('INSERT OR REPLACE INTO state VALUES(?,?)',(key,canonical(value).decode()))
    def _event(self,kind,body):
        raw=canonical(body);need(len(raw)<=MAX_EVENT,'event bound')
        tail=self.db.execute('SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        seq,previous=(tail['seq']+1,tail['sha'])if tail else(1,'0'*64)
        event={'seq':seq,'kind':kind,'body':body,'previous':previous};value=digest(event)
        self.db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(seq,kind,raw.decode(),previous,value));return value
    def _epoch(self):
        need(self._get('manifest')==self.manifest_sha256,'stale planner manifest; reopen against active snapshot')
        epoch=self._get('epoch')
        old=self.db.execute('SELECT state FROM epochs WHERE id=?',(epoch,)).fetchone()
        if old is None or old['state']in('complete','superseded'):
            epoch+=1;manifest=self._get('manifest')
            ids=sorted(k for k,r in self.rows.items()if r['partition']=='train'and r['status']=='eligible')
            need(ids,'no eligible training data_runs')
            self.db.execute('INSERT INTO epochs VALUES(?,?,?)',(epoch,manifest,'open'))
            self.db.executemany('INSERT INTO members VALUES(?,?,NULL)',((epoch,k)for k in ids));self._set('epoch',epoch)
            self._event('epoch.opened',{'epoch':epoch,'manifest_sha256':manifest,'eligible_count':len(ids),'membership_sha256':digest(ids)})
        return epoch
    def _choose(self,epoch,remaining,seen):
        groups={}
        for identity in remaining:groups.setdefault(self.rows[identity]['parent_lineage_sha256'],[]).append(identity)
        families=sorted(groups);attempts=0;seed=self.config['seed']
        # Start at the largest permitted batch. Smaller unique tails are allowed;
        # neither order changes nor a new epoch makes repeated membership new.
        for size in range(min(self.config['batch_size'],len(families)),0,-1):
            for attempt in range(MAX_ATTEMPTS//min(self.config['batch_size'],len(families))):
                key={'seed':seed,'epoch':epoch,'size':size,'attempt':attempt}
                chosen=sorted(families,key=lambda f:digest({**key,'purpose':'parents','parent':f}))[:size]
                ids=tuple(min(groups[f],key=lambda x:digest({**key,'purpose':'run','id':x}))for f in chosen)
                attempts+=1
                if fingerprint(ids)not in seen:return ids,attempts
        return None,attempts
    def pending(self):
        row=self.db.execute("SELECT body,state FROM batches WHERE state!='committed'").fetchone()
        return None if row is None else {'state':row['state'],**decode(row['body'])}
    def reserve(self,checkpoint_sha256):
        sha256(checkpoint_sha256);self.db.execute('BEGIN IMMEDIATE')
        try:
            need(self._get('manifest')==self.manifest_sha256,'stale planner manifest; reopen against active snapshot')
            need(self._get('checkpoint')==checkpoint_sha256,'checkpoint/cursor resume mismatch')
            need(self.pending()is None,'unresolved batch exists; reconcile before new reservation')
            if not any(r['partition']=='train'and r['status']=='eligible'for r in self.rows.values()):
                self.db.execute('COMMIT');return Waiting('no_eligible_training_data_runs',0,self._get('epoch'),0)
            epoch=self._epoch();remaining=[r[0]for r in self.db.execute('SELECT run FROM members WHERE epoch=? AND batch IS NULL ORDER BY run',(epoch,))]
            seen={r[0]for r in self.db.execute('SELECT fingerprint FROM batches')}
            ids,attempts=self._choose(epoch,remaining,seen)
            if ids is None:
                self.db.execute('COMMIT');return Waiting('no_unique_batch_found_within_bound; await new support or explicit manifest transition',len(remaining),epoch,attempts)
            index=self.db.execute('SELECT count(*) FROM batches WHERE epoch=?',(epoch,)).fetchone()[0]
            body={'manifest_sha256':self._get('manifest'),'epoch':epoch,'index':index,'checkpoint_before':checkpoint_sha256,
                  'ordered_data_run_ids':list(ids),'fingerprint':fingerprint(ids)}
            bid=digest({'schema':'jope.flywheel-batch-intent/v1',**body});body['batch_id']=bid
            self.db.execute('INSERT INTO batches VALUES(?,?,?,?,?,?,?,NULL)',(bid,body['fingerprint'],body['manifest_sha256'],epoch,index,canonical(body).decode(),'reserved'))
            self._event('batch.reserved',body);self.db.execute('COMMIT')
            return Batch(bid,body['fingerprint'],body['manifest_sha256'],epoch,index,checkpoint_sha256,ids,
                         tuple(decode(canonical(self.rows[x]))for x in ids))
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def commit(self,batch_id,*,checkpoint_before,checkpoint_after,optimizer_step,ordered_data_run_ids,evidence):
        for x in(batch_id,checkpoint_before,checkpoint_after):sha256(x)
        need(checkpoint_after!=checkpoint_before,'unchanged checkpoint is not an optimizer update proof')
        need(type(optimizer_step)is int and optimizer_step>0,'positive optimizer step')
        need(type(evidence)is dict and evidence.get('origin')=='native_optimizer'and
             type(evidence.get('event_sha256'))is str,'native optimizer event reference required')
        sha256(evidence['event_sha256']);need(len(canonical(evidence))<=MIB,'bounded evidence')
        body={'batch_id':batch_id,'checkpoint_before':checkpoint_before,'checkpoint_after':checkpoint_after,
              'optimizer_step':optimizer_step,'ordered_data_run_ids':list(ordered_data_run_ids),'evidence':evidence}
        encoded=canonical(body).decode();self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT * FROM batches WHERE id=?',(batch_id,)).fetchone();need(row is not None,'unknown reserved batch')
            if row['state']=='committed':
                need(row['commit_body']==encoded,'commit retry differs from recorded execution');self.db.execute('COMMIT');return body
            intent=decode(row['body'])
            need(row['state']in('reserved','needs_reconcile')and self._get('checkpoint')==checkpoint_before==intent['checkpoint_before'],'checkpoint before mismatch')
            need(self._get('optimizer_step')+1==optimizer_step,'optimizer step gap/repeat')
            need(intent['ordered_data_run_ids']==list(ordered_data_run_ids),'actual ordered batch differs from durable intent')
            for identity in ordered_data_run_ids:
                changed=self.db.execute('UPDATE members SET batch=? WHERE epoch=? AND run=? AND batch IS NULL',(batch_id,row['epoch'],identity))
                need(changed.rowcount==1,'duplicate exposure within epoch')
            self.db.execute("UPDATE batches SET state='committed',commit_body=? WHERE id=?",(encoded,batch_id))
            self._set('checkpoint',checkpoint_after);self._set('optimizer_step',optimizer_step);self._event('batch.committed',body)
            left=self.db.execute('SELECT count(*) FROM members WHERE epoch=? AND batch IS NULL',(row['epoch'],)).fetchone()[0]
            if not left:
                self.db.execute("UPDATE epochs SET state='complete'WHERE id=?",(row['epoch'],))
                self._event('epoch.completed',{'epoch':row['epoch'],'manifest_sha256':row['manifest'],'eligible_count':self.db.execute('SELECT count(*)FROM members WHERE epoch=?',(row['epoch'],)).fetchone()[0]})
            self.db.execute('COMMIT');return body
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def mark_uncertain(self,batch_id,reason):
        need(type(reason)is str and 0<len(reason)<=2000,'explicit uncertainty reason');self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT state FROM batches WHERE id=?',(batch_id,)).fetchone();need(row is not None and row['state']=='reserved','reserved batch required')
            self.db.execute("UPDATE batches SET state='needs_reconcile'WHERE id=?",(batch_id,));self._event('batch.uncertain',{'batch_id':batch_id,'reason':reason});self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def adopt_manifest(self,raw,expected_sha256,*,checkpoint_sha256):
        m,rows=validate_manifest(raw,expected_sha256);self.db.execute('BEGIN IMMEDIATE')
        try:
            need(self._get('manifest')==self.manifest_sha256,'stale planner manifest; reopen against active snapshot')
            need(self.pending()is None and self._get('checkpoint')==checkpoint_sha256,'manifest transition needs reconciled checkpoint')
            need(set(self.rows)<=set(rows),'new manifest must retain old rows, including explicit refusals')
            for identity,old in self.rows.items():
                new=rows[identity];need(old['data_run']==new['data_run']and old['parent_lineage_sha256']==new['parent_lineage_sha256']and old['partition']==new['partition'],'data_run/parent/partition identity drift')
            oldsha=self._get('manifest');need(oldsha!=expected_sha256,'manifest already active')
            prior=self.db.execute('SELECT body FROM manifests WHERE sha=?',(expected_sha256,)).fetchone()
            need(prior is None,'cannot reactivate an older manifest')
            epoch=self._get('epoch');left=self.db.execute('SELECT run FROM members WHERE epoch=?AND batch IS NULL ORDER BY run',(epoch,)).fetchall()
            self.db.execute("UPDATE epochs SET state='superseded'WHERE id=?AND state='open'",(epoch,))
            self.db.execute('INSERT INTO manifests VALUES(?,?)',(expected_sha256,raw));self._set('manifest',expected_sha256)
            self._event('manifest.adopted',{'old':oldsha,'new':expected_sha256,'unfinished_epoch':epoch,'unconsumed_count':len(left),'unconsumed_sha256':digest([r[0]for r in left])})
            # The old partial epoch remains stored and explicitly uncompleted.
            self.db.execute('COMMIT');self.manifest,self.rows=m,rows;self.manifest_sha256=expected_sha256
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def summary(self):
        need(self._get('manifest')==self.manifest_sha256,'stale planner manifest; reopen against active snapshot')
        counts={}
        for r in self.rows.values():
            key=r['partition']+':'+r['status'];counts[key]=counts.get(key,0)+1
        return {'schema':'jope.flywheel-batch-stream-status/v1','manifest_sha256':self._get('manifest'),
            'epoch':self._get('epoch'),'checkpoint_sha256':self._get('checkpoint'),'optimizer_steps_reported':self._get('optimizer_step'),
            'membership_counts':counts,'pending_batch':self.pending(),'completed_epochs':self.db.execute("SELECT count(*)FROM epochs WHERE state='complete'").fetchone()[0],
            'unique_exposed_data_runs':self.db.execute('SELECT count(DISTINCT run)FROM members WHERE batch IS NOT NULL').fetchone()[0],
            'batch_count':self.db.execute("SELECT count(*)FROM batches WHERE state='committed'").fetchone()[0],
            'scope':'Caller-reported native execution evidence; planner does not execute or independently verify optimizer/GP artifacts.'}
    def verify(self):
        """Verify against one consistent SQLite snapshot while other readers/writers run."""
        need(not self.db.in_transaction,'verification requires an independent read transaction')
        self.db.execute('BEGIN')
        try:
            result=self._verify();self.db.execute('COMMIT');return result
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def _verify(self):
        """Verify event chain, manifests, intents/commits and recorded exposure."""
        previous='0'*64;seq=0;reserved={};committed={};checkpoint=self.config['initial_checkpoint_sha256'];step=0
        for row in self.db.execute('SELECT * FROM events ORDER BY seq'):
            seq+=1;body=decode(row['body']);event={'seq':seq,'kind':row['kind'],'body':body,'previous':previous}
            need(row['seq']==seq and row['previous']==previous and digest(event)==row['sha'],'event chain tampered')
            previous=row['sha']
            if row['kind']=='batch.reserved':reserved[body['batch_id']]=body
            elif row['kind']=='batch.committed':
                need(body['batch_id']in reserved and body['batch_id']not in committed,'commit sequence tampered')
                need(body['checkpoint_before']==checkpoint and body['optimizer_step']==step+1,'checkpoint chain tampered')
                checkpoint=body['checkpoint_after'];step+=1;committed[body['batch_id']]=body
        manifests={}
        for row in self.db.execute('SELECT sha,body FROM manifests'):
            manifests[row['sha']]=validate_manifest(bytes(row['body']),row['sha'])[1]
        for epoch in self.db.execute('SELECT *FROM epochs'):
            need(epoch['manifest']in manifests,'epoch manifest absent')
            expected={k for k,r in manifests[epoch['manifest']].items()if r['partition']=='train'and r['status']=='eligible'}
            actual={r['run']:r['batch']for r in self.db.execute('SELECT *FROM members WHERE epoch=?',(epoch['id'],))}
            need(set(actual)==expected,'epoch membership missing/extra/protected')
            need(epoch['state']!='complete'or all(x is not None for x in actual.values()),'false complete epoch')
        need(self._get('checkpoint')==checkpoint and self._get('optimizer_step')==step,'cursor state tampered')
        seen=set()
        for row in self.db.execute('SELECT *FROM batches'):
            b=decode(row['body']);need(row['state']in('reserved','needs_reconcile','committed')and row['manifest']==b['manifest_sha256']and row['epoch']==b['epoch']and row['batch_index']==b['index'],'batch columns tampered')
            need(reserved.get(row['id'])==b and row['fingerprint']==fingerprint(b['ordered_data_run_ids']),'batch intent tampered')
            need(row['fingerprint']not in seen,'batch membership repeated');seen.add(row['fingerprint'])
            need((row['state']=='committed')==(row['id']in committed),'batch state tampered')
            if row['state']=='committed':need(decode(row['commit_body'])==committed[row['id']],'commit body tampered')
            actual={x[0]for x in self.db.execute('SELECT run FROM members WHERE batch=?',(row['id'],))}
            need(actual==(set(b['ordered_data_run_ids'])if row['state']=='committed'else set()),'exposure table tampered')
        need(len(reserved)==self.db.execute('SELECT count(*)FROM batches').fetchone()[0],'missing batch row')
        return {'events':seq,'head_sha256':previous,'committed_batches':len(committed),'unresolved_batches':len(reserved)-len(committed)}

MIB=1024**2
