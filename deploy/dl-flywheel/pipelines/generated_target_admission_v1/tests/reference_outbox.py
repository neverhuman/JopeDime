"""Durable mandatory GP emissions, independent of frozen supervised gradients.

No network, evaluator, score synthesis or optimizer exists here. Successful
outcome refs require the caller's pinned exact-context native/S3 verifier.
Unknown leased work never retries automatically. All SQL lives on local disk.
"""
import hashlib,json,os,sqlite3,time
from pathlib import Path

TERMINAL={'cached_verified','published_verified','failure_custody_verified'}
STATES=TERMINAL|{'recorded','leased','pending_unsupported','unknown_attempt'}
MAX_RECORD=128*1024

def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(value):return sha(canonical(value))
def hash64(x):return type(x)is str and len(x)==64 and all(c in '0123456789abcdef'for c in x)
def ref(x):
 return type(x)is dict and set(x)=={'bucket','key','version_id','sha256','bytes'}and all(type(x[k])is str and x[k]for k in('bucket','key','version_id'))and hash64(x['sha256'])and type(x['bytes'])is int and x['bytes']>0

class Outbox:
 def __init__(self,path,*,model_origin_sha256,max_pending=512,max_rows=100000,max_payload_bytes=64*1024**2,now=time.time):
  path=Path(path);need(path.is_absolute()and'..'not in path.parts and all(not p.is_symlink()for p in(path,*path.parents)),'absolute owned non-symlink outbox')
  for suffix in('-wal','-shm'):need(not Path(str(path)+suffix).is_symlink(),'outbox sidecar symlink')
  need(hash64(model_origin_sha256),'model origin');need(all(type(x)is int and x>0 for x in(max_pending,max_rows,max_payload_bytes))and max_pending<=max_rows<=1000000 and max_payload_bytes<=256*1024**2,'bounded outbox limits')
  self.now=now;self.db=sqlite3.connect(path,timeout=2,isolation_level=None);self.db.row_factory=sqlite3.Row
  self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA foreign_keys=ON');self.db.execute('PRAGMA wal_autocheckpoint=256');self.db.execute('PRAGMA max_page_count=262144')
  self.db.executescript("""CREATE TABLE IF NOT EXISTS config(body BLOB NOT NULL);
  CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,body BLOB NOT NULL,ack BLOB NOT NULL);
  CREATE TABLE IF NOT EXISTS emissions(id TEXT PRIMARY KEY,batch_id TEXT NOT NULL REFERENCES batches(id),ordinal INTEGER NOT NULL,body BLOB NOT NULL,state TEXT NOT NULL,lease BLOB,outcome BLOB);
  CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,kind TEXT NOT NULL,body BLOB NOT NULL,previous TEXT NOT NULL,sha TEXT NOT NULL UNIQUE);""")
  self.config={'schema':'jope.flywheel-neural-outbox/v1','model_origin_sha256':model_origin_sha256,'max_pending':max_pending,'max_rows':max_rows,'max_payload_bytes':max_payload_bytes}
  self.db.execute('BEGIN IMMEDIATE')
  try:
   rows=self.db.execute('SELECT body FROM config').fetchall()
   if not rows:self.db.execute('INSERT INTO config VALUES(?)',(canonical(self.config),))
   else:need(len(rows)==1 and bytes(rows[0][0])==canonical(self.config),'immutable outbox config')
   self.db.execute('COMMIT')
  except BaseException:self.db.execute('ROLLBACK');self.db.close();raise
 def close(self):self.db.close()
 def _event(self,kind,body):
  row=self.db.execute('SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1').fetchone();seq=1 if row is None else row['seq']+1;previous='0'*64 if row is None else row['sha'];raw=canonical(body);h=digest({'seq':seq,'kind':kind,'body':body,'previous':previous})
  self.db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(seq,kind,raw,previous,h));return {'seq':seq,'sha256':h}
 def record_batch(self,batch,emissions):
  keys={'batch_id','campaign_sha256','checkpoint_sha256','global_step','manifest_sha256','train_data_run_ids','validation_data_run_ids','emitted_train_data_run_ids'}
  need(type(batch)is dict and set(batch)==keys,'closed batch identity')
  need(all(hash64(batch[k])for k in('batch_id','campaign_sha256','checkpoint_sha256','manifest_sha256'))and type(batch['global_step'])is int and batch['global_step']>0,'batch hashes/step')
  train=batch['train_data_run_ids'];dev=batch['validation_data_run_ids'];need(type(train)is list and type(dev)is list and train and all(hash64(x)for x in train+dev)and len(set(train+dev))==len(train+dev),'unique explicit train/development membership')
  selected=batch['emitted_train_data_run_ids'];need(type(selected)is list and selected and len(set(selected))==len(selected)and set(selected)<=set(train),'explicit nonempty emitted training subset')
  expected=[('training',x)for x in selected]+[('validation',x)for x in dev]
  need(type(emissions)is list and 1<=len(emissions)<=1024 and len(emissions)==len(expected),'every declared proposal slot emits exactly once')
  prepared=[]
  ekeys={'phase','data_run_id','context_id','parent_lineage_sha256','input_descriptor','raw_program_utf8','raw_program_sha256','decoder_seed','decoder_protocol_sha256','decode_artifact_sha256'}
  for ordinal,(e,want)in enumerate(zip(emissions,expected)):
   need(type(e)is dict and set(e)==ekeys and(e['phase'],e['data_run_id'])==want,'ordered emission membership/phase')
   need(all(hash64(e[k])for k in('context_id','parent_lineage_sha256','decoder_protocol_sha256','decode_artifact_sha256')),'emission identities')
   need(type(e['raw_program_utf8'])is str and 0<len(e['raw_program_utf8'].encode())<=32768 and sha(e['raw_program_utf8'].encode())==e['raw_program_sha256'],'raw emission content hash')
   need(type(e['decoder_seed'])is int and 0<=e['decoder_seed']<2**63 and type(e['input_descriptor'])is dict,'decoder/input descriptor')
   body={'schema':'jope.flywheel-neural-emission/v1','model_origin_sha256':self.config['model_origin_sha256'],'batch':batch,'ordinal':ordinal,'emission':e};raw=canonical(body);need(len(raw)<=MAX_RECORD,'bounded emission payload')
   identity=digest({'origin':self.config['model_origin_sha256'],'batch_id':batch['batch_id'],'ordinal':ordinal});prepared.append((identity,raw))
  need(len({e['parent_lineage_sha256']for e in emissions[:len(selected)]})==len(selected),'distinct training parent lineages')
  body=canonical({'batch':batch,'emissions':[i for i,_ in prepared],'payload_sha256':[sha(b)for _,b in prepared]});self.db.execute('BEGIN IMMEDIATE')
  try:
   existing=self.db.execute('SELECT body,ack FROM batches WHERE id=?',(batch['batch_id'],)).fetchone()
   if existing:
    need(bytes(existing['body'])==body,'conflicting batch replay');ack=json.loads(existing['ack']);self.db.execute('COMMIT');return ack
   counts=self.db.execute("SELECT count(*) AS n,coalesce(sum(length(body)),0)AS bytes,sum(CASE WHEN state NOT IN('cached_verified','published_verified','failure_custody_verified')THEN 1 ELSE 0 END)AS pending FROM emissions").fetchone()
   need(counts['n']+len(prepared)<=self.config['max_rows']and(counts['pending']or 0)+len(prepared)<=self.config['max_pending']and counts['bytes']+sum(len(b)for _,b in prepared)<=self.config['max_payload_bytes'],'outbox backpressure capacity')
   self.db.execute('INSERT INTO batches VALUES(?,?,?)',(batch['batch_id'],body,b'{}'))
   for ordinal,(identity,raw)in enumerate(prepared):self.db.execute('INSERT INTO emissions VALUES(?,?,?,?,?,NULL,NULL)',(identity,batch['batch_id'],ordinal,raw,'recorded'))
   head=self._event('batch.emitted',{'batch_id':batch['batch_id'],'body_sha256':sha(body),'emission_ids':[i for i,_ in prepared]})
   ack={'schema':'jope.flywheel-neural-emission-ack/v1','batch_id':batch['batch_id'],'checkpoint_sha256':batch['checkpoint_sha256'],'global_step':batch['global_step'],'body_sha256':sha(body),'emission_ids':[i for i,_ in prepared],'head':head,'numeric_outcomes_required_for_this_optimizer':False}
   self.db.execute('UPDATE batches SET ack=?WHERE id=?',(canonical(ack),batch['batch_id']));self.db.execute('COMMIT');return ack
  except BaseException:
   if self.db.in_transaction:self.db.execute('ROLLBACK')
   raise
 def claim(self,owner,*,lease_seconds,deadline):
  now=self.now();need(type(owner)is str and 0<len(owner)<=128 and type(lease_seconds)is int and 1<=lease_seconds<=600 and now+lease_seconds<=deadline,'bounded lease within fixed campaign deadline')
  self.db.execute('BEGIN IMMEDIATE')
  try:
   row=self.db.execute("SELECT *FROM emissions WHERE state='recorded'ORDER BY rowid LIMIT 1").fetchone()
   if row is None:self.db.execute('COMMIT');return None
   lease={'emission_id':row['id'],'owner':owner,'started_at':now,'expires_at':now+lease_seconds,'token':digest({'id':row['id'],'owner':owner,'at':now})}
   self.db.execute("UPDATE emissions SET state='leased',lease=?WHERE id=?",(canonical(lease),row['id']));self._event('emission.leased',lease);self.db.execute('COMMIT');return {'lease':lease,'body':json.loads(row['body'])}
  except BaseException:
   if self.db.in_transaction:self.db.execute('ROLLBACK')
   raise
 def finish(self,lease,outcome,*,verify_outcome):
  need(type(outcome)is dict and set(outcome)=={'state','evaluation_key','record_ref','attempt_custody_ref','error'},'closed numeric-free outcome join')
  state=outcome['state'];need(state in TERMINAL|{'pending_unsupported','unknown_attempt'},'explicit terminal/pending outcome')
  success=state in{'cached_verified','published_verified'}
  need((success and hash64(outcome['evaluation_key'])and ref(outcome['record_ref'])and outcome['error']is None)or(not success and outcome['record_ref']is None and outcome['evaluation_key']is None and type(outcome['error'])is str and 0<len(outcome['error'])<=4000),'no numeric result or ready claim for failures')
  need(outcome['attempt_custody_ref']is None or ref(outcome['attempt_custody_ref']),'exact custody ref')
  need(state!='failure_custody_verified'or ref(outcome['attempt_custody_ref']),'failure requires verified nonnumeric custody')
  row=self.db.execute('SELECT body FROM emissions WHERE id=?',(lease['emission_id'],)).fetchone();need(row is not None,'known emission')
  need(verify_outcome(json.loads(row['body']),outcome)is True,'pinned external exact-context/S3 verifier required')
  self.db.execute('BEGIN IMMEDIATE')
  try:
   row=self.db.execute('SELECT *FROM emissions WHERE id=?',(lease['emission_id'],)).fetchone()
   need(row['state']=='leased'and json.loads(row['lease'])==lease and self.now()<=lease['expires_at'],'live owned lease required; unknown attempts never auto retry')
   self.db.execute('UPDATE emissions SET state=?,outcome=?WHERE id=?',(state,canonical(outcome),lease['emission_id']));self._event('emission.resolved',{'emission_id':lease['emission_id'],'outcome':outcome});self.db.execute('COMMIT')
  except BaseException:
   if self.db.in_transaction:self.db.execute('ROLLBACK')
   raise
 def summary(self):
  counts={r['state']:r['n']for r in self.db.execute('SELECT state,count(*)AS n FROM emissions GROUP BY state')};return {'schema':'jope.flywheel-neural-outbox-status/v1','observations':sum(counts.values()),'batches':self.db.execute('SELECT count(*)FROM batches').fetchone()[0],'states':counts,'unresolved':sum(n for s,n in counts.items()if s not in TERMINAL),'numeric_scores_synthesized':0}
 def verify(self):
  need(not self.db.in_transaction,'independent verifier transaction');self.db.execute('BEGIN')
  try:
   previous='0'*64;seq=0;registered={};states={};leases={};outcomes={};ack_heads={}
   for row in self.db.execute('SELECT *FROM events ORDER BY seq'):
    seq+=1;body=json.loads(row['body']);need(row['seq']==seq and row['previous']==previous and row['sha']==digest({'seq':seq,'kind':row['kind'],'body':body,'previous':previous}),'outbox event chain');previous=row['sha']
    if row['kind']=='batch.emitted':
     need(body['batch_id']not in registered,'duplicate batch event');registered[body['batch_id']]=body;ack_heads[body['batch_id']]={'seq':seq,'sha256':row['sha']}
     for identity in body['emission_ids']:need(identity not in states,'duplicate emission');states[identity]='recorded'
    elif row['kind']=='emission.leased':need(states.get(body['emission_id'])=='recorded','lease event order');states[body['emission_id']]='leased';leases[body['emission_id']]=body
    elif row['kind']=='emission.resolved':need(states.get(body['emission_id'])=='leased','outcome event order');states[body['emission_id']]=body['outcome']['state'];outcomes[body['emission_id']]=body['outcome']
    else:raise ValueError('unknown outbox event')
   for row in self.db.execute('SELECT *FROM batches'):
    need(row['id']in registered and sha(bytes(row['body']))==registered[row['id']]['body_sha256'],'batch payload changed');b=json.loads(row['body']);ack=json.loads(row['ack'])
    need(ack['head']==ack_heads[row['id']]and ack['body_sha256']==sha(bytes(row['body']))and ack['checkpoint_sha256']==b['batch']['checkpoint_sha256']and ack['global_step']==b['batch']['global_step'],'ACK payload changed')
    actual=self.db.execute('SELECT id,body FROM emissions WHERE batch_id=?ORDER BY ordinal',(row['id'],)).fetchall();need([x['id']for x in actual]==b['emissions']==ack['emission_ids']and[sha(bytes(x['body']))for x in actual]==b['payload_sha256'],'emission payload missing/changed')
   for row in self.db.execute('SELECT *FROM emissions'):
    need((None if row['lease']is None else json.loads(row['lease']))==leases.get(row['id'])and(None if row['outcome']is None else json.loads(row['outcome']))==outcomes.get(row['id']),'lease/outcome payload changed')
   actual={r['id']:r['state']for r in self.db.execute('SELECT id,state FROM emissions')};need(actual==states and len(registered)==self.db.execute('SELECT count(*)FROM batches').fetchone()[0],'outbox materialized state changed')
   result={**self.summary(),'events':seq,'head_sha256':previous};self.db.execute('COMMIT');return result
  except BaseException:
   if self.db.in_transaction:self.db.execute('ROLLBACK')
   raise
