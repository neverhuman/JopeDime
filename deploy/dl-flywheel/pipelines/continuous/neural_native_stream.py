"""Native-identity planner v1, derived from frozen batch_stream 2c4be42a.
New journal/model lineage only. Uses native context IDs, never manufactured raw
variant/target hashes. Membership/exposure/atomicity unchanged. Adoption requires
an exhausted epoch; original frozen planner source and journals stay untouched.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,itertools,json,math,os,sqlite3,time
from pathlib import Path
from jope_flywheel.protocol import canonical,digest,sha256
from jope_flywheel.protocol import Context

SCHEMA='jope.native-corpus-batches/v1'
MAX_ROWS=100000
MAX_MANIFEST=64*1024**2
MAX_EVENT=16*1024**2
MAX_ATTEMPTS=4096
CONTEXT_KEYS={'context_id','context_sha256','view_id','dataset_hash','fold','cut_id','parent_lineage_sha256','partition','source_kind','observation_dispatch','gp_hwm_ref','kumo_source_sha256'}

def need(ok,msg):
    if not ok:raise ValueError(msg)
def byte_sha(b):return hashlib.sha256(b).hexdigest()
def decode(raw):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    d=json.loads(raw,object_pairs_hook=pairs);canonical(d);return d

def validate_manifest(raw,expected_sha256):
    need(type(raw)is bytes and len(raw)<=MAX_MANIFEST and byte_sha(raw)==sha256(expected_sha256),'manifest hash/size')
    m=decode(raw);need(set(m)=={'schema','snapshot_sha256','role_authority_sha256','contexts','pending','refused'}and m['schema']in(SCHEMA,'jope.mixed-corpus-batches/v1'),'closed native/mixed manifest schema')
    sha256(m['snapshot_sha256']);sha256(m['role_authority_sha256'])
    need(type(m['contexts'])is list and 0<len(m['contexts'])<=MAX_ROWS,'bounded full native membership')
    need(type(m['pending'])is list and type(m['refused'])is list and len(m['pending'])+len(m['refused'])<=MAX_ROWS,'bounded pending/refused inventory')
    rows={};roles={}
    for c in m['contexts']:
        if c.get('source_kind')=='flywheel_variant':
            need(m['schema']=='jope.mixed-corpus-batches/v1'and set(c)=={'context_id','source_kind','variant_identity','parent_lineage_sha256','partition','context','observation_dispatch','dataset_publication','encoder_publication'},'closed verified variant descriptor')
            cid=sha256(c['context_id']);ctx=Context(**c['context']);need(cid==ctx.identity and cid not in rows,'variant native Context identity')
            sha256(c['variant_identity']);lin=sha256(c['parent_lineage_sha256']);role=c['partition']
            need(role in('train','validation')and roles.setdefault(lin,role)==role and c['observation_dispatch']=='qualified_flywheel_worker_adapter','variant retained parent role/dispatcher')
            for key in ('dataset_publication','encoder_publication'):
                ref=c[key];need(set(ref)=={'bucket','key','version_id','sha256','bytes'}and type(ref['bytes'])is int and ref['bytes']>0 and all(type(ref[k])is str and ref[k]for k in('bucket','key','version_id')),'exact variant publication ref');sha256(ref['sha256'])
            rows[cid]={'data_run':c,'data_run_id':cid,'parent_lineage_sha256':lin,'partition':role,'status':'eligible','reason':None,'payload':c};continue
        need(type(c)is dict and set(c)==CONTEXT_KEYS,'closed native context descriptor')
        cid=sha256(c['context_id']);need(c['context_sha256']==cid and cid not in rows,'duplicate/forged native context')
        lin=sha256(c['parent_lineage_sha256']);sha256(c['kumo_source_sha256']);role=c['partition']
        need(role in('train','validation')and roles.setdefault(lin,role)==role,'parent lineage role conflict')
        need(c['source_kind']=='native_s1_context'and c['observation_dispatch']in('pending_native_worker_adapter','qualified_native_worker_adapter'),'native identity type')
        need(type(c['dataset_hash'])is str and len(c['dataset_hash'])==16 and all(x in '0123456789abcdef'for x in c['dataset_hash']),'native dataset16 version, not raw SHA')
        need(type(c['fold'])is int and c['fold']>=0 and all(type(c[k])is str and 0<len(c[k])<=512 for k in('view_id','cut_id')),'native cut/fold identity')
        need(type(c['gp_hwm_ref'])is dict and len(canonical(c))<=128*1024,'bounded native provenance')
        # This ID names the actual immutable native context. No fabricated raw target hash.
        rows[cid]={'data_run':c,'data_run_id':cid,'parent_lineage_sha256':lin,'partition':role,'status':'eligible','reason':None,'payload':c}
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
        self.config={'schema':'jope.native-batch-stream/v1','seed':seed,'batch_size':batch_size,'initial_checkpoint_sha256':initial_checkpoint_sha256}
        self.db.executescript('''CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS manifests(sha TEXT PRIMARY KEY,body BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS epochs(id INTEGER PRIMARY KEY,manifest TEXT NOT NULL,state TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS members(epoch INTEGER NOT NULL,run TEXT NOT NULL,batch TEXT,PRIMARY KEY(epoch,run));
          CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,fingerprint TEXT UNIQUE NOT NULL,manifest TEXT NOT NULL,
            epoch INTEGER NOT NULL,batch_index INTEGER NOT NULL,body TEXT NOT NULL,state TEXT NOT NULL,commit_body TEXT);
          CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,kind TEXT NOT NULL,body TEXT NOT NULL,previous TEXT NOT NULL,sha TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS inherited_fingerprints(fingerprint TEXT PRIMARY KEY);
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
        families=sorted(groups);attempts=0;seed=self.config['seed'];epoch+=(self._get('ancestry')or{}).get('completed_epochs',0)
        # Start at the largest permitted batch. Smaller unique tails are allowed;
        # neither order changes nor a new epoch makes repeated membership new.
        for size in range(min(self.config['batch_size'],len(families)),0,-1):
            for attempt in range(MAX_ATTEMPTS//min(self.config['batch_size'],len(families))):
                key={'seed':seed,'epoch':epoch,'size':size,'attempt':attempt}
                # Drain large lineages early so distinct-lineage tails cannot strand a full pass.
                chosen=sorted(families,key=lambda f:(-len(groups[f]),digest({**key,'purpose':'parents','parent':f})))[:size]
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
            seen={r[0]for r in self.db.execute('SELECT fingerprint FROM batches')}|{r[0]for r in self.db.execute('SELECT fingerprint FROM inherited_fingerprints')}
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
    def inherit(self,ancestry):
        """Fresh journal only: preserve old membership bans without editing it."""
        need(type(ancestry)is dict and set(ancestry)=={'planner_ref','head','checkpoint_sha256','optimizer_steps','completed_epochs','fingerprints','exposed_context_ids','presentations'},'closed verified ancestry')
        need(ancestry['checkpoint_sha256']==self.config['initial_checkpoint_sha256']and ancestry['head']['unresolved_batches']==0 and ancestry['optimizer_steps']>0 and ancestry['completed_epochs']>0,'completed physical checkpoint ancestry')
        values=ancestry['fingerprints'];need(len(values)==len(set(values))==ancestry['head']['committed_batches']and all(sha256(v)for v in values),'complete historical fingerprint set')
        need(len(ancestry['exposed_context_ids'])==len(set(ancestry['exposed_context_ids']))and all(sha256(v)for v in ancestry['exposed_context_ids']),'exact exposed identities')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old=self._get('ancestry')
            if old is not None:
                need(old==ancestry,'ancestry cannot change');self.db.execute('COMMIT');return
            need(self._get('optimizer_step')==0 and self._get('epoch')==0 and self.pending()is None,'only a fresh journal accepts ancestry')
            self.db.executemany('INSERT INTO inherited_fingerprints VALUES(?)',((v,)for v in values));self._set('ancestry',ancestry);self._event('stream.ancestry',ancestry);self.db.execute('COMMIT')
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
            epoch=self._get('epoch');need(self.db.execute('SELECT count(*) FROM members WHERE epoch=? AND batch IS NULL',(epoch,)).fetchone()[0]==0,'adoption only at fully consumed epoch boundary')
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
        return {'schema':'jope.native-batch-stream-status/v1','manifest_sha256':self._get('manifest'),
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
        previous='0'*64;seq=0;reserved={};committed={};checkpoint=self.config['initial_checkpoint_sha256'];step=0;ancestry_events=[]
        for row in self.db.execute('SELECT * FROM events ORDER BY seq'):
            seq+=1;body=decode(row['body']);event={'seq':seq,'kind':row['kind'],'body':body,'previous':previous}
            need(row['seq']==seq and row['previous']==previous and digest(event)==row['sha'],'event chain tampered')
            previous=row['sha']
            if row['kind']=='stream.ancestry':ancestry_events.append(body)
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
        ancestry=self._get('ancestry');inherited=set()
        exists=self.db.execute("SELECT count(*)FROM sqlite_master WHERE type='table'AND name='inherited_fingerprints'").fetchone()[0]
        table={r[0]for r in self.db.execute('SELECT fingerprint FROM inherited_fingerprints')}if exists else set()
        need((ancestry is not None and len(ancestry_events)==1)or(ancestry is None and not ancestry_events and not table),'ancestry state/event/table must be present together')
        if ancestry is not None:
            need(ancestry_events==[ancestry],'ancestry event mismatch');inherited={r[0]for r in self.db.execute('SELECT fingerprint FROM inherited_fingerprints')};need(inherited==set(ancestry['fingerprints']),'historical fingerprints changed')
        seen=set()
        for row in self.db.execute('SELECT *FROM batches'):
            b=decode(row['body']);need(row['state']in('reserved','needs_reconcile','committed')and row['manifest']==b['manifest_sha256']and row['epoch']==b['epoch']and row['batch_index']==b['index'],'batch columns tampered')
            need(reserved.get(row['id'])==b and row['fingerprint']==fingerprint(b['ordered_data_run_ids']),'batch intent tampered')
            need(row['fingerprint']not in seen,'batch membership repeated');seen.add(row['fingerprint'])
            need(row['fingerprint']not in inherited,'batch repeats ancestral membership')
            need((row['state']=='committed')==(row['id']in committed),'batch state tampered')
            if row['state']=='committed':need(decode(row['commit_body'])==committed[row['id']],'commit body tampered')
            actual={x[0]for x in self.db.execute('SELECT run FROM members WHERE batch=?',(row['id'],))}
            need(actual==(set(b['ordered_data_run_ids'])if row['state']=='committed'else set()),'exposure table tampered')
        need(len(reserved)==self.db.execute('SELECT count(*)FROM batches').fetchone()[0],'missing batch row')
        return {'events':seq,'head_sha256':previous,'committed_batches':len(committed),'unresolved_batches':len(reserved)-len(committed)}

MIB=1024**2
