"""Local-only source/selection packaging; does not stage or launch anything."""
import ast,copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent/'kumo-xbabe3-fresh32'
SELECT_SHA='947ff1dd0269b82dfa2cf4e3a53ceb052fa4c498b327429410529012499fbd5b'
WORKER_SHA='0883dd4085e01cc0e1a3353396fd7004854c4ae2e68868f1eb79a053a6cb2a42'
SOURCE='/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-kumo-xbabe3-next128-20261009/revision2'
GLOBAL=1791592146.380285
SCHEMA='jope.kumos-next128-controller/v1'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def ref(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
def write(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:f.write(b)
def main():
 raw=(HERE/'selection-readonly.json').read_bytes();need(sha(raw)==SELECT_SHA,'exact root-collected128 snapshot');selected=json.loads(raw)
 need(selected['state']=='metadata_ready_current_roles_pending'and selected['global_deadline_unix']==GLOBAL and len(selected['rows'])==len(selected['publications'])==128,'selected scope')
 old=json.loads((BASE/'config.template.json').read_bytes())
 for r in old['local_source_pins']:need(ref(Path(r['path']))==r,'frozen first32 source drift')
 need(ref(BASE/'evergreen_kumos_packet.py')['sha256']==WORKER_SHA,'qualified numerical source')
 children=[];origins={}
 for ordinal,packet in enumerate(selected['packets']):
  need(packet['ordinal']==ordinal and packet['count']==32,'packet order/count')
  subset=selected['rows'][32*ordinal:32*(ordinal+1)];support=[{k:r[k]for k in('sequence','identity','parent')}for r in subset]
  need(sha(canonical(support))==packet['support_sha256'],'packet support')
  pid=sha(canonical({'schema':SCHEMA,'selection_snapshot_sha256':SELECT_SHA,'ordinal':ordinal,'support_sha256':packet['support_sha256']}))
  d=HERE/('packet'+str(ordinal));need(not d.exists(),'packet source already exists');d.mkdir()
  for r in old['local_source_pins']:
   src=Path(r['path']);write(d/src.name,src.read_bytes());origins[str((d/src.name).relative_to(HERE))]=r
  # Explicit fixed identity and source roots; no mutation of previous campaign.
  f=d/'campaign.py';s=f.read_text().replace("PACKET_ID='c3c9e02f299e9bd0837fbe935520898136af1e61d230d564b31037a3d4f39a13'","PACKET_ID='"+pid+"'")
  s=s.replace("SELECTION_SHA='1947113aa2d5a39c3f55e557bebe56e24737ac364f1e69bc700c6702cc82a12c'","SELECTION_SHA='"+packet['support_sha256']+"'")
  oldroot='/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_kumos_packet_fresh32_xbabe3_c3c9e02f299e9bd0'
  newroot='/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_kumos_packet_next128_xbabe3_'+pid[:16]
  s=s.replace(oldroot,newroot);f.write_text(s)
  f=d/'kumo_queue.py';s=f.read_text().replace('list(range(906,938))',f"list(range({packet['sequence_first']},{packet['sequence_last']+1}))");f.write_text(s)
  f=d/'kumo_controller.py';s=f.read_text().replace('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-kumo-xbabe3-fresh32-20261009/runtime',SOURCE+'/packet'+str(ordinal)+'/runtime')
  s=s.replace("mono=time.monotonic()+deadline-time.time()","mono=config['shared_monotonic_deadline'];need(time.monotonic()<mono,'unchanged shared monotonic deadline')")
  s=s.replace("need(c['controller_sha256']==sha(Path(__file__).read_bytes()),'controller source')", "need(type(c['shared_monotonic_deadline'])in(int,float)and 0<c['shared_monotonic_deadline']-time.monotonic()<=1800,'shared monotonic deadline');need(c['shared_scope']=='next128-four-fixed32-v1','shared scope');need(c['controller_sha256']==sha(Path(__file__).read_bytes()),'controller source')")
  f.write_text(s)
  selection={'schema':'jope.kumos-fresh32-selection/v1','selection_sha256':packet['support_sha256'],'packet_id':pid,'rows':subset};(d/'selection.json').write_bytes(canonical(selection))
  # Rebind every local/remote source hash after only declared operational changes.
  c=copy.deepcopy(old);local=[ref(d/Path(r['path']).name)for r in old['local_source_pins']];common=SOURCE+'/packet'+str(ordinal)+'/runtime';remote=[{**r,'path':common+'/'+Path(r['path']).name}for r in local];by={Path(r['path']).name:r for r in remote}
  c.update(packet_id=pid,owned_runtime=common,controller_sha256=by['kumo_controller.py']['sha256'],local_source_pins=local,source_pins=remote,shared_scope='next128-four-fixed32-v1',shared_monotonic_deadline=None)
  for key in ['worker','authority','intake','runtime','publisher','base','io_helper','telemetry','template','selection']:c[key]=by[Path(c[key]['path']).name]
  c.update(local_publisher=str(d/'kumo_publish.py'),local_base=str(d/'pilot_publish_core.py'))
  write(d/'config.template.json',canonical(c))
  need(ref(d/'evergreen_kumos_packet.py')['sha256']==WORKER_SHA,'scientific worker changed')
  children.append({'ordinal':ordinal,'packet_id':pid,'sequence_first':packet['sequence_first'],'sequence_last':packet['sequence_last'],'template':ref(d/'config.template.json'),'worker_sha256':WORKER_SHA,'controller':ref(d/'kumo_controller.py')})
 for d in HERE.glob('packet[0-3]'):
  for f in d.glob('*.py'):ast.parse(f.read_bytes(),filename=str(f))
 proposal={'schema':SCHEMA,'approved_for_launch':False,'shared_scope':'next128-four-fixed32-v1','host':'xbabe3','max_packets':4,'max_new_variants':128,'created_unix':None,'deadline_unix':None,'shared_monotonic_deadline':None,'global_deadline_unix':GLOBAL,'selection':ref(HERE/'selection-readonly.json'),'packets':children,'worker_sha256':WORKER_SHA,'no_retries':True,'old_scopes_unchanged':True,'source_root':SOURCE}
 write(HERE/'config.template.json',canonical(proposal));write(HERE/'origin-pins.json',canonical(origins));print(json.dumps(ref(HERE/'config.template.json')))
if __name__=='__main__':main()
