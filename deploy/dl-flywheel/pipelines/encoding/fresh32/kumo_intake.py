"""xbabe2 bounded rolling-packet preparation from the independently published queue."""
import argparse,hashlib,importlib.util,json,os,socket,sys,time,threading
from pathlib import Path
sys.dont_write_bytecode=True
import kumo_queue as queue
from campaign import PACKET_ID,ROOT,HOST,COUNT,GLOBAL_DEADLINE,DB_CONFIG_SHA
W=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')
DATA=W/'flywheel_evergreen_20261009/data'
PIPELINE_SHA='b4eb3a3aa5e85ff9af0eab29101c21dec74018826e3244ee9b5103caa47d5e78'
PIPELINE=W/'flywheel_growth_pipeline_v2_20261009_r2/config.json'
BASE_SHA='63b0a13aa76369c08553e9cfb360a6998492ac43e8c1a152e9e08c841b9bc965'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return queue.canonical(x)
def sha(b):return queue.sha(b)
def read(ref,cap=16*1024**2):
 p=Path(ref['path']);need(p.is_absolute()and all(not q.is_symlink()for q in(p,*p.parents))and p.is_file()and p.stat().st_size<=cap,'safe bounded input')
 b=p.read_bytes();need(sha(b)==ref['sha256']and len(b)==ref.get('bytes',len(b)),'input SHA/size');return b

def save(root,rel,b):
 p=root/rel;need(not Path(rel).is_absolute()and'..'not in Path(rel).parts and all(not q.is_symlink()for q in(p,*p.parents)),'owned packet path')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':sha(b),'bytes':len(b)}

def get(client,ref,deadline):
 need(time.time()<deadline,'intake deadline');need(ref['bucket']=='veox-jopedime-learning-use2'and ref['key'].startswith('experiments/gp-policy-dl/v2-trainonly/')and ref['version_id']not in('','null')and ref['verification']=='exact VersionId GET + SHA256'and 0<ref['bytes']<=8*1024**2,'exact bounded prior S3 ref')
 o=client.get_object(Bucket=ref['bucket'],Key=ref['key'],VersionId=ref['version_id']);stream=o['Body']
 try:b=stream.read(ref['bytes']+1);need(not stream.read(1),'S3 grew')
 finally:stream.close()
 need(time.time()<deadline and o.get('VersionId')==ref['version_id']and o['ContentLength']==len(b)==ref['bytes']and sha(b)==ref['sha256'],'exact S3 GET mismatch/deadline');return b

def prepare(config,config_sha,packet_id,deadline,client):
 need(socket.gethostname().split('.')[0]=='xbabe2','source host');need(packet_id==PACKET_ID and config['packet_size']==COUNT and config['created_unix']<=time.time()<deadline==config['deadline_unix']<=GLOBAL_DEADLINE and 0<deadline-config['created_unix']<=1800,'fixed single packet/global window')
 pbody=read({'path':str(PIPELINE),'sha256':PIPELINE_SHA});pipeline=json.loads(pbody);need(deadline<=pipeline['deadline_unix']and pipeline['required_encoders']==['kumo_tabular_s'],'unchanged pipeline deadline/roster')
 for ref in config['source_pins']:read(ref)
 need(config['pipeline_sha256']==PIPELINE_SHA,'pipeline identity')
 template=json.loads(read(config['template']));oldroot=Path(template['owned_root']);root=Path(ROOT);need(packet_id==PACKET_ID,'fixed packet ID');need(not root.exists(),'fresh packet root')
 # Current roles precede claims and any prepared-parent payload read.
 pubspec=__import__('importlib.util',fromlist=['x']).spec_from_file_location('intake_original_publication',pipeline['publisher']['path']);read(pipeline['publisher']);publisher=__import__('importlib.util',fromlist=['x']).module_from_spec(pubspec);pubspec.loader.exec_module(publisher)
 base,g,original=publisher.load_modules(pipeline['base']['path']);eligible,excluded,excluded_views=publisher.check_authority(base,original)
 original_parents=[json.loads(l)for l in read(original['parent_registry'],32*1024**2).splitlines()];parent_map={p['parent_lineage_sha256']:p for p in original_parents};expected_db=g.digest({'config_sha':queue.CONFIG,'parents':original_parents,'target':original['target'],'backlog':original['backlog_limit']})
 db=queue.connect(DATA/'expansion.sqlite',expected_db)
 need(expected_db==DB_CONFIG_SHA,'original immutable DB identity')
 def role_check(row):
  parent=parent_map[row['parent']]
  return parent['view_id']in eligible and parent['view_id']not in excluded_views and not(set(parent['retained_lineage_context_sha256'])&excluded)
 try:claim=queue.claim(db,PIPELINE_SHA,packet_id,config['packet_size'],deadline,json.loads(read(config['selection'])),role_check)
 except BaseException:db.close();raise
 if claim['state']!='claimed':db.close();return {**claim,'pipeline_config_sha256':PIPELINE_SHA}
 selected=claim['rows']
 root.mkdir(mode=0o700);allrefs=[];refs_lock=threading.Lock()
 def write(rel,b):
  ref=save(root,rel,b)
  with refs_lock:allrefs.append(ref)
  return ref
 def copy(ref):return write(str(Path(ref['path']).relative_to(oldroot)),read(ref,32*1024**2))
 try:
  m=dict(template);m.update(schema='jope.flywheel-evergreen-kumos-packet/v1',owned_root=str(root),campaign_id='evergreen-kumos-fresh32-xbabe3-'+packet_id[:16],worker_path=str(root/'evergreen_kumos_packet.py'),worker_sha256=config['worker']['sha256'],authority_helper_sha256=config['authority']['sha256'],approved_for_qualification=True,created_unix=time.time(),deadline_unix=min(deadline-600,time.time()+1200),total_variants=len(selected),packet_id=packet_id,queue_deadline_unix=deadline,expansion_db_config_sha256=expected_db,pipeline_config=write('sources/growth-pipeline.json',pbody),rolling_config_sha256=config_sha)
  m['deadline_unix']=min(deadline-600,m['created_unix']+1200);m['scientific_scope']='New prepared variants from independently published queue. Kumo Small2051 only; parent development partitions, task-training label exposure and unknown pretraining overlap inherited. No GP/admission claim.'
  m['generator_config']=copy(template['generator_config']);m['parent_registry']=copy(template['parent_registry']);m['source_pins']=[copy(r)for r in template['source_pins']];m['helpers']={k:copy(v)for k,v in template['helpers'].items()}
  m['io_helper']=write('kumo_io.py',read(config['io_helper']));m['telemetry']=write('kumo_timing.py',read(config['telemetry']));write('evergreen_kumos_packet.py',read(config['worker']));write('evergreen_kumos_packet_authority.py',read(config['authority']))
  cfg={**template['hosts']['xbabe3'],'host':'xbabe3','out':str(root/'runs/xbabe3'),'cpu_reservations':[],'variants':[]};cfg['ofm']=dict(cfg['ofm']);cfg['ofm']['runtime_inventory']=write('sources/runtime_kumos_xbabe3.json',read(config['runtime']));cfg['ofm']['license_pins']=[copy(r)for r in cfg['ofm']['license_pins']]
  parents={p['parent_lineage_sha256']:p for p in (json.loads(l)for l in read(m['parent_registry'],32*1024**2).splitlines())}
  pub={'schema':'jope.flywheel-generated-packet-publication-complete/v1','state':'complete','pipeline_config_sha256':PIPELINE_SHA,'publisher_sha256':pipeline['publisher']['sha256'],'new_gp_labels':0,'teaching_eligible':False,'metadata':[],'variants':[],'object_receipts':[]}
  def prepare_one(selected_row):
   row=selected_row['row'];identity=row['identity']
   if hasattr(client,'logger'):event=client.logger.start('queue_wait_observation',variant_identity=identity,queue_source_created_unix=row['created'],queue_wait_wall_seconds=max(0,time.time()-row['created']));client.logger.end(event,state='observed',precision='source DB creation uses floating-point UTC seconds; elapsed work uses monotonic_ns separately')
   publication=selected_row['publication'];ref=publication['s3_receipt'];doc=json.loads(get(client,ref,deadline));need(doc['schema']=='jope.flywheel-evergreen-dataset-publication/v1' and doc['publication_helper_sha256']=='e11dd3cc829b4379212a33cb69c4ce0ae992777874836253b20b0ae780d10f34' and doc['variant_identity']==identity and doc['pipeline_config_sha256']==PIPELINE_SHA and doc['metadata_sha256']==row['metadata_sha']and doc['content_sha256']==row['content_sha']and doc['required_encoders']==['kumo_tabular_s']and doc['new_gp_labels']==0 and doc['teaching_eligible']is False,'published dataset identity')
   objects={r['key'].split('/')[-1]:r for r in doc['objects']};need(len(objects)==len(doc['objects'])==3 and set(objects)=={'encoder.jtt1.gz','metadata.json','generation.json'},'three input objects')
   need(all(r['key']=='experiments/gp-policy-dl/v2-trainonly/datasets/flywheel/v1/'+identity+'/'+name for name,r in objects.items()),'exact per-variant object namespace')
   payloads={name:get(client,r,deadline)for name,r in objects.items()}
   need(sha(payloads['metadata.json'])==row['metadata_sha'],'metadata DB binding');meta=json.loads(payloads['metadata.json']);parent=parents[row['parent']];need(meta['parent']==parent and meta['variant_identity']==identity,'registered parent identity')
   files={name:write('inputs/'+identity+'/'+name,b)for name,b in payloads.items()};files['parent_encoder.jtt1.gz']=write('inputs/'+identity+'/parent_encoder.jtt1.gz',read(parent['encoder'],8*1024**2))
   variant=({'identity':identity,'sequence':row['sequence'],'lineage':row['parent'],'partition':meta['partition'],'n':meta['rows'],'p':meta['features'],'changed':meta['target_sha256']!=meta['source_target_sha256'],'files':files,'database_row':row})
   return variant,{**meta,'sequence':row['sequence']},{'variant_identity':identity,'sequence':row['sequence'],'receipt':ref},[*doc['objects'],ref]
  import kumo_io
  prepared=[];kumo_io.parallel_each(selected,prepare_one,prepared.append)
  for variant,meta,receipt,refs in sorted(prepared,key=lambda row:row[0]['sequence']):cfg['variants'].append(variant);pub['metadata'].append(meta);pub['variants'].append(receipt);pub['object_receipts'].extend(refs)
  m['hosts']={'xbabe3':cfg};m['dataset_publication']=write('sources/dataset-publication.json',canonical(pub));mref=write('manifest.json',canonical(m));index={'schema':'jope.kumo-rolling-stage-index/v1','root':str(root),'packet_id':packet_id,'files':sorted(allrefs,key=lambda r:r['path']),'bytes':sum(r['bytes']for r in allrefs)};need(index['bytes']<=512*1024**2,'packet stage cap');indexref=save(root,'stage-index.json',canonical(index));return {'state':'prepared','packet_id':packet_id,'manifest':mref,'index':indexref,'variants':len(selected),'additional_row_ceiling_observation':{k:claim[k]for k in('observed_packet_rows_before','observed_packet_rows_after')},'queue_deadline_unix':deadline}
 except BaseException as e:queue.fail(db,packet_id,e);raise
 finally:db.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--packet-id',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args();c=json.loads(read({'path':a.config,'sha256':a.config_sha}));need(sha(Path(__file__).read_bytes())==c['intake']['sha256'],'intake source')
 import boto3
 from botocore.config import Config
 client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=4))
 import kumo_timing as timing
 read(c['io_helper']);import kumo_io
 need(Path(kumo_io.__file__).resolve()==Path(c['io_helper']['path']).resolve(),'bound I/O helper origin')
 gate=kumo_io.Gate(a.deadline,4*c['packet_size'],0)
 log=timing.Logger(Path(c['owned_runtime'])/('intake-timing-'+a.packet_id+'.jsonl'),{'packet_id':a.packet_id,'config_sha256':a.config_sha});client=kumo_io.Client(timing.Client(client,log),gate)
 try:
  with log.span('packet_prepare_total'):result=prepare(c,a.config_sha,a.packet_id,a.deadline,client)
  print(json.dumps(result,sort_keys=True))
 finally:log.close()
if __name__=='__main__':main()
