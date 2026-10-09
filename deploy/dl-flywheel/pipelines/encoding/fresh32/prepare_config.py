"""Prepare a local unapproved template or freeze once after explicit review.

This tool performs no SSH, queue mutation, GPU work or S3 operation. A frozen
configuration starts the sole1800-second clock; resource/role gates still apply.
"""
import argparse,hashlib,json,time
from pathlib import Path
import campaign as c
HERE=Path(__file__).resolve().parent
COMMON='/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-kumo-xbabe3-fresh32-20261009/runtime'
FILES=['campaign.py','kumo_queue.py','kumo_intake.py','kumo_commit.py','kumo_controller.py','kumo_transport.py','kumo_io.py','kumo_timing.py','evergreen_kumos_packet_authority.py','evergreen_kumos_packet.py','generic_worker_dfba7d52.py','template.json','runtime_kumos_xbabe3.json','pilot_publish_core.py','kumo_publish.py','selection.json']
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def ref(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
def template():
 local=[ref(HERE/f)for f in FILES];remote=[{**r,'path':COMMON+'/'+Path(r['path']).name}for r in local];by={Path(r['path']).name:r for r in remote}
 value={'schema':'jope.kumos-xbabe3-fresh32-controller/v1','approved_for_launch':False,'host':'xbabe3','packet_id':c.PACKET_ID,'packet_size':32,'max_packets':1,'created_unix':None,'deadline_unix':None,'global_deadline_unix':c.GLOBAL_DEADLINE,'pipeline_sha256':c.PIPELINE_SHA,'database_config_sha256':c.DB_CONFIG_SHA,'owned_runtime':COMMON,'controller_sha256':by['kumo_controller.py']['sha256'],'local_source_pins':local,'source_pins':remote,'scope_boundary':{'additional_all_history_packet_row_ceiling':128,'old_process_budget_reset':False,'old_controller_changes':False,'new_packets':1,'new_variants':32},'qualification':'Full six-test/four-encoder-call qualification per variant; no reused xbabe1 baseline','local_publisher':str(HERE/'kumo_publish.py'),'local_base':str(HERE/'pilot_publish_core.py'),'no_automatic_retry':True}
 for key,name in {'worker':'evergreen_kumos_packet.py','authority':'evergreen_kumos_packet_authority.py','intake':'kumo_intake.py','runtime':'runtime_kumos_xbabe3.json','publisher':'kumo_publish.py','base':'pilot_publish_core.py','io_helper':'kumo_io.py','telemetry':'kumo_timing.py','template':'template.json','selection':'selection.json'}.items():value[key]=by[name]
 return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--review');p.add_argument('--review-sha');a=p.parse_args();value=template()
 if a.freeze:
  if not a.review or not a.review_sha:raise ValueError('explicit pinned independent review required')
  raw=Path(a.review).read_bytes();review=json.loads(raw)
  if sha(raw)!=a.review_sha or review.get('state')!='passed' or review.get('template_sha256')!=sha(canonical(value)):raise ValueError('review must bind these exact source/config bytes')
  now=time.time()
  if now+1800>c.GLOBAL_DEADLINE:raise ValueError('full fixed window unavailable before original global deadline')
  value.update(approved_for_launch=True,created_unix=now,deadline_unix=now+1800,review=ref(Path(a.review).resolve()))
  target=HERE/'config.frozen.json'
 else:target=HERE/'config.template.json'
 with target.open('xb')as f:f.write(canonical(value))
 print(json.dumps(ref(target)))
if __name__=='__main__':main()
