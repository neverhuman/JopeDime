"""Exact S3 verification and fresh-role check before additive encoder-stage CAS."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import kumo_queue as q
from kumo_intake import read,get,DATA

def load(ref,name):
 read(ref);s=importlib.util.spec_from_file_location(name,ref['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def validate(config,manifest,status,client,deadline,base):
 q.need(status['schema']=='jope.flywheel-kumos-fresh32-publication/v1'and status['state']=='complete'and status['host']=='xbabe3'and status['completed_variants']==status['vectors_verified']==manifest['total_variants']and status['verified_objects']==2*manifest['total_variants'],'complete exact packet publication')
 variants={v['identity']:v for v in manifest['hosts']['xbabe3']['variants']};q.need(len(status['variants'])==len(variants)and{r['variant_identity']for r in status['variants']}==set(variants),'complete stage support');rows=[]
 def verify_one(v):
  identity=v['variant_identity'];expected=variants[identity];ref=v['receipt'];body=get(client,ref,deadline);r=json.loads(body)
  q.need(ref['key']=='experiments/gp-policy-dl/v2-trainonly/receipts/flywheel-datasets/v1/'+identity+'/'+q.sha(body)+'.json','completion namespace')
  q.need(r['schema']=='jope.flywheel-kumos-fresh32-publication/v1'and r['variant_identity']==identity and r['publisher_sha256']==config['publisher']['sha256']and r['required_encoders']==r['completed_encoders']==['kumo_tabular_s']and r['missing_encoders']==[]and r['new_gp_labels']==0 and r['teaching_eligible']is False,'qualified encoder receipt')
  q.need(r['metadata_sha256']==expected['files']['metadata.json']['sha256']and r['content_sha256']==expected['database_row']['content_sha']and r['partition']==expected['partition']and r['parent_lineage_sha256']==expected['lineage'],'stage source identity')
  proof=r['source_proof'];q.need(proof['manifest']==config['manifest']and proof['source_sha256']==manifest['worker_sha256']and proof['producer_phase_sealed']is True and proof['finalization_custody']is None,'encoder source proof')
  for anchor in [proof['coordinator_ledger'],proof['phase_ledgers']['ofm']]:q.need(len(anchor['runs'])==1 and all(x['sealed']is True for x in anchor['runs'].values()),'sealed source anchor')
  q.need(len(r['objects'])==1 and set(r['encoder_records'])=={'kumo_tabular_s'},'one Kumo vector')
  vector=r['objects'][0];q.need(vector['key']=='experiments/gp-policy-dl/v2-trainonly/datasets/flywheel/v1/'+identity+'/encodings/kumo_tabular_s.npy','Kumo namespace');raw=get(client,vector,deadline);base.numeric_vector(raw,2051)
  record=r['encoder_records']['kumo_tabular_s'];qualification=record['qualification'];qb=q.canonical(qualification['body']);q.need(qualification['body']['passed']is True and q.sha(qb)==qualification['ref']['sha256'],'pinned qualification')
  q.need(record['vector_ref']['sha256']==vector['sha256']and record['record']['variant_identity']==identity and record['record']['source_input']==expected['files']['encoder.jtt1.gz']and record['record']['target_sha256']==r['target_sha256'],'vector record identity')
  return {'variant_identity':identity,'content_sha256':r['content_sha256'],'metadata_sha256':r['metadata_sha256'],'parent_lineage_sha256':r['parent_lineage_sha256'],'partition':r['partition'],'target_sha256':r['target_sha256'],'required_encoders':['kumo_tabular_s'],'new_gp_labels':0,'teaching_eligible':False,'receipt':ref,'vector':vector,'packet_id':manifest['packet_id'],'manifest':config['manifest']}
 import kumo_io
 kumo_io.parallel_each(status['variants'],verify_one,rows.append);rows.sort(key=lambda r:variants[r['variant_identity']]['sequence']);return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);a=p.parse_args();c=json.loads(read({'path':a.config,'sha256':a.config_sha}));m=json.loads(read(c['manifest']));status=json.loads(read(c['status']));deadline=m['queue_deadline_unix'];q.need(time.time()<deadline,'commit packet deadline')
 for ref in c['source_pins']:read(ref)
 base=load(c['base'],'commit_s3_base');auth=load(c['authority'],'commit_current_authority');role=auth.check(m,c['manifest']['sha256']);q.need(role['state']=='passed','current roles')
 import boto3
 from botocore.config import Config
 client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=4))
 import kumo_io
 client=kumo_io.Client(client,kumo_io.Gate(deadline,2*m['total_variants'],0))
 rows=validate(c,m,status,client,deadline,base);role=auth.check(m,c['manifest']['sha256']);q.need(role['state']=='passed' and 0<=time.time()-role['checked_at_unix']<30 and time.time()<deadline,'fresh roles immediately before stage CAS');db=q.connect(DATA/'expansion.sqlite',m['expansion_db_config_sha256'])
 try:q.finish(db,m['packet_id'],rows,lambda rs:rs==rows,deadline)
 finally:db.close()
 print(json.dumps({'state':'stage_committed','packet_id':m['packet_id'],'variants':len(rows),'stage':q.STAGE,'rows':rows,'fresh_role_at':role['checked_at_unix']}))
if __name__=='__main__':main()
