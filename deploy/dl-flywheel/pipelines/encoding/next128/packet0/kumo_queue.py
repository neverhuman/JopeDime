"""Independent Kumo packet claims and verified stages; never mutates variants.

Leases do not silently expire back into the work queue: interrupted packets retain
ownership and need explicit reconciliation. S3 receipt verification precedes the
short DB transaction, which checks source identity again before committing.
"""
import hashlib,json,math,sqlite3,time
from pathlib import Path
from campaign import PACKET_ID,SELECTION_SHA,HOST,COUNT,GLOBAL_DEADLINE,PIPELINE_SHA,DB_CONFIG_SHA,MAX_TOTAL_OWNED_PACKETS
PACKETS='flywheel_kumo_packets_v1'
ITEMS='flywheel_kumo_packet_items_v1'
STAGES='flywheel_pipeline_stage_v1'
PUB='flywheel_s3_publications_v1'
STAGE='kumo_small_s3_verified'
CONFIG='cd9ee8f882735c42b49d694b99dbbc4187035804099bc49f0ca81f8ec77a9b0a'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(v):need(isinstance(v,str)and len(v)==64 and all(c in '0123456789abcdef'for c in v),'invalid digest');return v

def connect(path,expected_config):
 p=Path(path)
 need(p.is_absolute()and p.is_file()and all(not q.is_symlink()for q in(p,*p.parents)),'unsafe expansion DB')
 for suffix in ('-wal','-shm'):
  side=Path(str(p)+suffix);need(not side.is_symlink(),'unsafe expansion sidecar')
 db=sqlite3.connect('file:'+str(p)+'?mode=rw',uri=True,timeout=.5,isolation_level=None);db.row_factory=sqlite3.Row
 try:need(db.execute('SELECT sha FROM config WHERE id=1').fetchone()[0]==digest(expected_config),'immutable expansion DB config drift')
 except BaseException:db.close();raise
 return db

def ensure(db):
 db.execute('CREATE TABLE IF NOT EXISTS '+PACKETS+'(packet_id TEXT PRIMARY KEY,host TEXT NOT NULL,pipeline_sha TEXT NOT NULL,created REAL NOT NULL,deadline REAL NOT NULL,state TEXT NOT NULL,proof_json TEXT)')
 db.execute('CREATE TABLE IF NOT EXISTS '+ITEMS+'(identity TEXT PRIMARY KEY,packet_id TEXT NOT NULL,sequence INTEGER NOT NULL UNIQUE,row_json TEXT NOT NULL,publication_json TEXT NOT NULL)')
 db.execute('CREATE TABLE IF NOT EXISTS '+STAGES+'(identity TEXT NOT NULL,stage TEXT NOT NULL,receipt_json TEXT NOT NULL,receipt_sha TEXT NOT NULL,created REAL NOT NULL,PRIMARY KEY(identity,stage))')
 for table,cols in [(PACKETS,['packet_id','host','pipeline_sha','created','deadline','state','proof_json']),(ITEMS,['identity','packet_id','sequence','row_json','publication_json']),(STAGES,['identity','stage','receipt_json','receipt_sha','created'])]:
  need([r[1]for r in db.execute('PRAGMA table_info('+table+')')]==cols,'owned table schema differs')

def publication(row,pub,pipeline_sha):
 need(pub['identity']==row['identity']and pub['sequence']==row['sequence']and pub['content_sha']==row['content_sha']and pub['metadata_sha']==row['metadata_sha'],'publication row identity')
 raw=pub['publication_json'].encode();need(sha(raw)==pub['publication_sha'],'publication evidence hash')
 p=json.loads(raw)
 need(p['qualified']is True and p['variant_identity']==row['identity']and p['content_sha256']==row['content_sha']and p['metadata_sha256']==row['metadata_sha']and p['pipeline_config_sha256']==pipeline_sha and p['publication_helper_sha256']=='e11dd3cc829b4379212a33cb69c4ce0ae992777874836253b20b0ae780d10f34','qualified pipeline publication required')
 ref=p['s3_receipt'];digest(ref['sha256']);need(ref['bucket']=='veox-jopedime-learning-use2'and ref['key']=='experiments/gp-policy-dl/v2-trainonly/receipts/flywheel-datasets/v1/'+row['identity']+'/'+ref['sha256']+'.json'and ref['version_id']not in ('','null')and ref['verification']=='exact VersionId GET + SHA256','exact publication custody required')
 return p

def selected_contract(selection):
 rows=selection['rows'];need(selection['packet_id']==PACKET_ID and len(rows)==COUNT,'fixed fresh32 packet support')
 need(len({r['identity']for r in rows})==len({r['parent']for r in rows})==COUNT,'unique identities/lineages')
 need([r['sequence']for r in rows]==list(range(938,970)),'fixed sequence support')
 body=[{k:r[k]for k in ('sequence','identity','parent')}for r in rows]
 need(sha(canonical(body))==SELECTION_SHA==selection['selection_sha256'],'frozen selection identity')
 return rows

def claim(db,pipeline_sha,packet_id,count,deadline,selection,role_check,now=None):
 now=time.time()if now is None else now
 need(pipeline_sha==PIPELINE_SHA and packet_id==PACKET_ID and type(count)is int and count==COUNT,'fixed one-packet campaign')
 need(math.isfinite(deadline)and 0<deadline-now<=1800 and deadline<=GLOBAL_DEADLINE,'bounded original-global claim window')
 wanted=selected_contract(selection)
 db.execute('BEGIN IMMEDIATE')
 try:
  ensure(db)
  existing=db.execute('SELECT state FROM '+PACKETS+' WHERE packet_id=?',(packet_id,)).fetchone()
  if existing is not None:
   db.execute('COMMIT');return {'state':'packet_already_exists','packet_state':existing['state'],'packet_id':packet_id,'rows':[]}
  # Additional conservative all-history row ceiling; not the old process budget.
  charged=db.execute('SELECT count(*) FROM '+PACKETS).fetchone()[0]
  need(charged<MAX_TOTAL_OWNED_PACKETS,'additional all-history packet-row safety ceiling exhausted')
  blocking=[dict(r)for r in db.execute('SELECT packet_id,state,deadline FROM '+PACKETS+" WHERE host=? AND state='claimed'",(HOST,))]
  if blocking:
   db.execute('COMMIT');return {'state':'blocked_claim','blocking':blocking,'rows':[]}
  selected=[];missing=[];conflicts=[]
  for expected in wanted:
   found=db.execute('SELECT * FROM variants WHERE identity=?',(expected['identity'],)).fetchone()
   if found is None:missing.append(expected['identity']);continue
   row=dict(found);need(row['state']in('published','admitted'),'selected row not published')
   need({k:v for k,v in row.items()if k!='state'}=={k:v for k,v in expected.items()if k!='state'},'frozen source row changed')
   item=db.execute('SELECT packet_id FROM '+ITEMS+' WHERE identity=?',(row['identity'],)).fetchone()
   stage=db.execute('SELECT receipt_sha FROM '+STAGES+' WHERE identity=? AND stage=?',(row['identity'],STAGE)).fetchone()
   if item is not None or stage is not None:
    conflicts.append({'identity':row['identity'],'packet_id':item['packet_id']if item else None,'stage':bool(stage)});continue
   pub=db.execute('SELECT * FROM '+PUB+' WHERE identity=?',(row['identity'],)).fetchone();need(pub is not None,'published custody row absent')
   proof=publication(row,dict(pub),pipeline_sha)
   need(role_check(row) is True,'selected parent currently protected')
   selected.append({'row':row,'publication':proof})
  if conflicts:
   db.execute('COMMIT');return {'state':'blocked_claim','conflicts':conflicts,'rows':[]}
  if missing:
   db.execute('COMMIT');return {'state':'empty','missing_selected_identities':missing,'rows':[]}
  need(len(selected)==COUNT and deadline-time.time()>=660,'complete fixed support and660 seconds immediately before claim INSERT')
  db.execute('INSERT INTO '+PACKETS+' VALUES(?,?,?,?,?,?,NULL)',(packet_id,HOST,pipeline_sha,now,deadline,'claimed'))
  for item in selected:
   row=item['row'];db.execute('INSERT INTO '+ITEMS+' VALUES(?,?,?,?,?)',(row['identity'],packet_id,row['sequence'],canonical(row).decode(),canonical(item['publication']).decode()))
  db.execute('COMMIT');return {'state':'claimed','packet_id':packet_id,'observed_packet_rows_before':charged,'observed_packet_rows_after':charged+1,'rows':selected}
 except BaseException:
  if db.in_transaction:db.execute('ROLLBACK')
  raise

def finish(db,packet_id,rows,verifier,deadline,now=None):
 now=time.time()if now is None else now;need(packet_id==PACKET_ID and now<deadline<=GLOBAL_DEADLINE,'fixed stage deadline')
 # The verifier performs exact-Version GET and full encoding provenance checks;
 # a path or existence-only check is never a qualified stage.
 proof=verifier(rows);need(proof is True,'unverified encoding receipts')
 db.execute('BEGIN IMMEDIATE')
 try:
  ensure(db);packet=db.execute('SELECT * FROM '+PACKETS+' WHERE packet_id=?',(packet_id,)).fetchone();need(packet is not None and packet['state']=='claimed'and packet['host']==HOST and deadline==packet['deadline'],'packet state/deadline')
  expected={r['identity']:dict(r)for r in db.execute('SELECT * FROM '+ITEMS+' WHERE packet_id=?',(packet_id,))};need(len(rows)==len({r['variant_identity']for r in rows})==len(expected)and{r['variant_identity']for r in rows}==set(expected),'exact packet stage support')
  for r in rows:
   identity=r['variant_identity'];item=expected[identity];old=json.loads(item['row_json']);current=db.execute('SELECT * FROM variants WHERE identity=?',(identity,)).fetchone();need(current is not None,'source variant disappeared');current=dict(current)
   need(current['state']in('published','admitted'),'source variant not published')
   need({k:v for k,v in current.items()if k!='state'}=={k:v for k,v in old.items()if k!='state'},'source variant changed during encoding')
   need(r['content_sha256']==current['content_sha']and r['metadata_sha256']==current['metadata_sha']and r['required_encoders']==['kumo_tabular_s']and r['new_gp_labels']==0 and r['teaching_eligible']is False,'encoding stage identity/roster')
   body=canonical(r).decode();db.execute('INSERT INTO '+STAGES+' VALUES(?,?,?,?,?)',(identity,STAGE,body,sha(body.encode()),time.time()))
  body=canonical({'completed':len(rows),'receipts':rows}).decode();db.execute('UPDATE '+PACKETS+" SET state='complete',proof_json=? WHERE packet_id=? AND state='claimed'",(body,packet_id));need(db.total_changes>=len(rows)+1,'stage commit')
  need(time.time()<deadline,'deadline before stage commit');db.execute('COMMIT')
 except BaseException:
  if db.in_transaction:db.execute('ROLLBACK')
  raise

def fail(db,packet_id,reason):
 need(packet_id==PACKET_ID,'cannot mutate another campaign claim');body=canonical({'state':'needs_reconcile','reason':str(reason)[:1000],'at':time.time()}).decode()
 db.execute('UPDATE '+PACKETS+" SET state='needs_reconcile',proof_json=? WHERE packet_id=? AND state='claimed'",(body,packet_id))
