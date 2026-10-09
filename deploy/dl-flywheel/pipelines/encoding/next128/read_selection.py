"""Read one bounded128-row queue snapshot. No SQL writes, network or payload reads.

SQLite mode=ro/query_only preserves SQL/event data; normal WAL read coordination
may touch existing shared-memory metadata. Never use immutable=1 on a live WAL.
"""
import hashlib,json,socket,sqlite3,sys,time
from pathlib import Path
ROOT=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')
DB=ROOT/'flywheel_evergreen_20261009/data/expansion.sqlite'
PIPELINE=ROOT/'flywheel_growth_pipeline_v2_20261009_r2/config.json'
PIPELINE_SHA='b4eb3a3aa5e85ff9af0eab29101c21dec74018826e3244ee9b5103caa47d5e78'
DB_SHA='8fdd90611435beaaa9b29ced7239e6329ad9e0beea4065fe6c7173d21cae2241'
PUBLISHER_SHA='e11dd3cc829b4379212a33cb69c4ce0ae992777874836253b20b0ae780d10f34'
BUCKET='veox-jopedime-learning-use2'
PREFIX='experiments/gp-policy-dl/v2-trainonly/'
START,STOP=938,1066
CAP=4*1024**2

def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def guard(path):
 p=Path(path);need(p.is_absolute()and p.is_file()and all(not x.is_symlink()for x in(p,*p.parents)),'exact regular nonsymlink source required');return p

def main():
 need(socket.gethostname().split('.')[0]=='xbabe2','source host must be xbabe2')
 need(len(sys.argv)==2 and len(sys.argv[1])==64 and all(c in '0123456789abcdef'for c in sys.argv[1]),'caller must pin exact reader input SHA')
 source_path=Path(__file__);source_self_verified=source_path.is_file()
 if source_self_verified:need(sha(source_path.read_bytes())==sys.argv[1],'reader source drift')
 started=time.monotonic();guard(DB)
 for suffix in ('-wal','-shm'):need(not Path(str(DB)+suffix).is_symlink(),'unsafe SQLite sidecar')
 pipeline_bytes=guard(PIPELINE).read_bytes();need(len(pipeline_bytes)<=1024**2 and sha(pipeline_bytes)==PIPELINE_SHA,'fixed growth pipeline identity');pipeline=json.loads(pipeline_bytes)
 need(pipeline['required_encoders']==['kumo_tabular_s']and pipeline['publisher']['sha256']==PUBLISHER_SHA,'fixed source publisher/roster')
 connection=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True,timeout=.5,isolation_level=None);connection.row_factory=sqlite3.Row
 try:
  connection.execute('PRAGMA query_only=ON');connection.execute('BEGIN')
  identity=connection.execute('SELECT sha FROM config WHERE id=1').fetchone();need(identity is not None and identity[0]==DB_SHA,'DB identity differs')
  rows=[dict(r)for r in connection.execute('SELECT * FROM variants WHERE sequence>=? AND sequence<? ORDER BY sequence',(START,STOP))]
  need(len(rows)==STOP-START==128 and [r['sequence']for r in rows]==list(range(START,STOP)),'exact128 selected sequences required')
  need(len({r['identity']for r in rows})==128,'duplicate identity')
  publications=[]
  for row in rows:
   need(time.monotonic()-started<15,'reader duration bound')
   need(row['state']in('published','admitted')and row['error']is None,'source row not ready')
   need(connection.execute('SELECT 1 FROM flywheel_kumo_packet_items_v1 WHERE identity=?',(row['identity'],)).fetchone()is None,'selected identity already claimed; never replay')
   need(connection.execute("SELECT 1 FROM flywheel_pipeline_stage_v1 WHERE identity=? AND stage='kumo_small_s3_verified'",(row['identity'],)).fetchone()is None,'selected identity already encoded; never replay')
   selected=connection.execute('SELECT * FROM flywheel_s3_publications_v1 WHERE identity=?',(row['identity'],)).fetchone();need(selected is not None,'dataset custody row missing');pub=dict(selected)
   need(all(pub[k]==row[k]for k in('identity','sequence','content_sha','metadata_sha')),'publication row differs')
   raw=pub['publication_json'].encode();need(len(raw)<=32768 and sha(raw)==pub['publication_sha'],'publication body hash/cap')
   body=json.loads(raw);need(body['qualified']is True and body['variant_identity']==row['identity']and body['content_sha256']==row['content_sha']and body['metadata_sha256']==row['metadata_sha']and body['pipeline_config_sha256']==PIPELINE_SHA and body['publication_helper_sha256']==PUBLISHER_SHA,'qualified dataset publication identity')
   r=body['s3_receipt'];need(r['bucket']==BUCKET and r['key']==PREFIX+'receipts/flywheel-datasets/v1/'+row['identity']+'/'+r['sha256']+'.json'and type(r['version_id'])is str and r['version_id']not in('','null')and r['verification']=='exact VersionId GET + SHA256'and type(r['bytes'])is int and 0<r['bytes']<=8*1024**2,'exact recorded publication ref')
   proof=body['qualification'];need(proof['variant_identity']==row['identity']and proof['parent_lineage_sha256']==row['parent']and proof['content_sha256']==row['content_sha']and proof['metadata_sha256']==row['metadata_sha']and proof['deterministic_replay_exact']is True and proof['outer_test_rows']==0 and proof['new_gp_labels']==0 and proof['teaching_eligible']is False,'recorded dataset qualification')
   need(type(proof['rows'])is int and 50<=proof['rows']<2000 and type(proof['features'])is int and 1<=proof['features']<=512 and len(proof['inner_counts'])==2 and min(proof['inner_counts'])>=25 and sum(proof['inner_counts'])==proof['rows'],'recorded dimensions/inner folds')
   publications.append(pub)
  packets=[]
  for index in range(4):
   selection=rows[index*32:(index+1)*32];need(len({r['parent']for r in selection})==32,'packet must have distinct32 parent lineages')
   support=[{k:r[k]for k in('sequence','identity','parent')}for r in selection]
   packets.append({'ordinal':index,'sequence_first':selection[0]['sequence'],'sequence_last':selection[-1]['sequence'],'count':32,'support_sha256':sha(canonical(support)),'row_sha256':sha(canonical(selection))})
  observation={'packet_state_counts':{r[0]:r[1]for r in connection.execute('SELECT state,count(*) FROM flywheel_kumo_packets_v1 GROUP BY state')},'existing_packet_rows':connection.execute('SELECT count(*) FROM flywheel_kumo_packets_v1').fetchone()[0],'kumo_stage_rows':connection.execute("SELECT count(*) FROM flywheel_pipeline_stage_v1 WHERE stage='kumo_small_s3_verified'").fetchone()[0]}
  connection.execute('ROLLBACK')
 finally:connection.close()
 result={'schema':'jope.kumo-next128-readonly-selection/v1','state':'metadata_ready_current_roles_pending','checked_at_utc_ns':time.time_ns(),'host':'xbabe2','reader_sha256':sys.argv[1],'reader_source_self_verified':source_self_verified,'database':str(DB),'database_config_sha256':DB_SHA,'pipeline':{'path':str(PIPELINE),'sha256':PIPELINE_SHA,'bytes':len(pipeline_bytes)},'publisher_sha256':PUBLISHER_SHA,'rows':rows,'publications':publications,'packets':packets,'observation':observation,'global_deadline_unix':pipeline['deadline_unix'],'capabilities':{'sql_writes':0,'network_requests':0,'raw_payload_reads':0,'current_role_check':False,'fresh_s3_readback':False,'gpu_execution':False},'scope':'Consistent SQLite metadata and recorded custody only. Current roles, source replay, exact-version GET, ownership and resource admission must be rechecked before each later claim.'}
 raw=canonical(result);need(len(raw)<=CAP and time.monotonic()-started<15,'output/duration cap');sys.stdout.buffer.write(raw+b'\n')
if __name__=='__main__':main()
