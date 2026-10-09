"""Read pinned live stage progress and host counters every15seconds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

def collect_compute(host):
    """Read-only compute view; does not attribute foreign training to this policy."""
    assert host == 'xbabe1'
    code = r'''import hashlib,json,os,subprocess,time
from pathlib import Path
out={'host':'xbabe1','at':time.time(),'load':os.getloadavg(),'cpu_count':os.cpu_count(),'archives':[],'errors':[],
 'workload_note':'GPU counters cover all jobs on this host. Only the current policy’s optimizer receipts affect its training totals.'}
cpu0=[int(v)for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
mem={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
out['memory']={k:mem[k] for k in ('MemTotal','MemAvailable','SwapTotal','SwapFree')}
try:
 r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
 out['gpus']=[dict(zip(['uuid','name','utilization_percent','memory_used_mib','memory_total_mib'],[x.strip()for x in line.split(',')]))for line in r.stdout.splitlines()]
except Exception as e:out['errors'].append('GPU counters: '+type(e).__name__)
out['memory_pressure']=Path('/proc/pressure/memory').read_text()[:1024]
out['gp_expansion_campaigns']=[]
try:
 root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_xbabe1_gp16_20261009')
 raw=(root/'campaign.json').read_bytes()
 if len(raw)>2097152 or hashlib.sha256(raw).hexdigest()!='70c1f9560076740ece80edb773be14919614dcce4b1460579ef784e2cf23ae6c':raise ValueError('xb1 campaign identity')
 p=root/'shard0/config.json';raw=p.read_bytes();pin='97d0bf6a516c992e14ec31659ed810b8795ebea5ee8df570cda94dbb7032b0c3'
 if p.is_symlink()or len(raw)>2097152 or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('xb1 config identity')
 c=json.loads(raw);slots=len(c['variants'])*len(c['programs']);v={'state':'staged','counts':{},'outcomes':[],'scored_context_ids':[]}
 p=root/'shard0/progress.json'
 if p.exists():
  if p.is_symlink()or p.stat().st_size>8388608:raise ValueError('xb1 progress bounds')
  v=json.loads(p.read_bytes())
  if v['config_sha256']!=pin:raise ValueError('xb1 progress identity')
  v['scored_context_ids']=sorted({r['variant_identity']for r in v.get('outcomes',[])if r.get('state')=='ready'})
  v['outcomes']=[{k:r.get(k)for k in ('slot','variant_identity','partition','phase','program_name','program_sha256','state','measurement','error')}for r in v.get('outcomes',[])[-12:]]
 p=root/'shard0/status.json'
 if p.exists():
  if p.is_symlink()or p.stat().st_size>16777216:raise ValueError('xb1 status bounds')
  terminal=json.loads(p.read_bytes())
  if terminal['config_sha256']!=pin:raise ValueError('xb1 terminal identity')
  v.update(state=terminal['state'],terminal_at=terminal['at'])
 v.update(label='xbabe1 / CPU qualification',expected_slots=slots,data_runs_with_scores=len(v['scored_context_ids']))
 out['gp_expansion_campaigns'].append(v)
except FileNotFoundError:pass
except Exception as e:out['errors'].append('xb1 GP scoring: '+type(e).__name__)


try:
 root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/hpc_trace_20261009_v1')
 if hashlib.sha256((root/'host_trace.py').read_bytes()).hexdigest()!='b181c4881521ca1630ca29ccbad69ce3a23470940424708dd56e5807375c31ed':raise ValueError('trace source')
 for field,filename,schema in [('hpc_trace','logs/status.json','jope.hpc-host-status/v1'),('hpc_archive','logs/archive-v1/status.json','jope.hpc-archive-status/v1')]:
  p=root/filename
  if not p.exists():continue
  raw=p.read_bytes()
  if p.is_symlink()or len(raw)>65536:raise ValueError('HPC status bounds')
  v=json.loads(raw)
  if v['schema']!=schema or v['host']!=out['host']:raise ValueError('HPC status host/schema')
  out[field]=v
except FileNotFoundError:pass
except Exception as e:out['errors'].append('HPC telemetry: '+type(e).__name__)

cpu1=[int(v)for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
delta=sum(cpu1)-sum(cpu0);idle=(cpu1[3]+cpu1[4])-(cpu0[3]+cpu0[4]);out['cpu_utilization_percent']=max(0,min(100,100*(delta-idle)/delta))if delta>0 else None
print(json.dumps(out,allow_nan=False))
'''
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,'python3 -c '+shlex.quote(code)],text=True,capture_output=True,timeout=12,check=True)
    if len(r.stdout)>1048576:raise ValueError('compute response cap')
    return json.loads(r.stdout)

def collect(host, entries):
    code = '''import hashlib,json,os,subprocess,time
from pathlib import Path
out={'host':HOST,'at':time.time(),'load':os.getloadavg(),'cpu_count':os.cpu_count(),'archives':[],'errors':[]}
cpu0=[int(v)for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
mem={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
out['memory']={k:mem[k] for k in ('MemTotal','MemAvailable')}
if HOST=='xbabe3':
 try:
  p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/wave4_full_20261008/baseline_full7120_20261008/status.json')
  if p.is_symlink() or p.stat().st_size>1048576:raise ValueError('baseline size')
  b=json.loads(p.read_bytes())
  if b['manifest_sha256']!='0a64c5a6ff35531aea0b854cdaeeea22952dfe8e3483a78580a64489662bbecf':raise ValueError('baseline manifest')
  out['baseline']={'state':b['state'],'completed':len(b['completed']),'expected':b['tasks_expected'],'at':b.get('at',b['started']),'deadline':b['deadline'],'cpu_cores_limit':32,'error':b.get('error')}
 except FileNotFoundError:out['baseline']={'state':'not_observed'}
 except Exception as e:out['errors'].append('Baseline: '+type(e).__name__)
try:
 root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_pilot_20261008')
 p=root/'runs/recovery1'/HOST/'progress.json'
 if p.is_symlink() or p.stat().st_size>1048576:raise ValueError('pilot size')
 out['pilot']=json.loads(p.read_bytes())
 if out['pilot']['host']!=HOST or out['pilot']['manifest_sha256']!='800f7c11ef9c2d55c18742de2c7c8ac10c3ac2cae58b737e7e44ec7c09fb7ea3':raise ValueError('pilot identity')
 out['pilot']['attempt']='recovery1'
except FileNotFoundError:
 out['pilot']={'state':'qualifying','attempt':'recovery1','completed':0,'expected':50}
except Exception as e:out.pop('pilot',None);out['errors'].append('Pilot: '+type(e).__name__)
try:
 r=subprocess.run(['systemctl','--user','show','jopedime-flywheel-pilot100-recovery1-'+HOST,'--property=ActiveState,SubState,Result,ExecMainStatus'],text=True,capture_output=True,timeout=5,check=True)
 unit=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
 out.setdefault('pilot',{})['service']=unit
 if unit.get('Result') not in ('success','') or unit.get('ExecMainStatus') not in ('0',''):
  out['pilot']['state']='failed';out['pilot']['observer_failure']=unit
 out['pilot']['prior_attempt']={'state':'failed_before_cuda','reason':'strict runtime version-string logging mismatch','retry':'explicit recovery1; original absolute deadline retained'}
except Exception as e:out['errors'].append('Pilot service: '+type(e).__name__)
try:
 p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_ofm_pilot_20261009/runs')/HOST/'progress.json'
 if p.is_symlink() or p.stat().st_size>1048576:raise ValueError('OFM progress size')
 v=json.loads(p.read_bytes())
 if v['host']!=HOST or v['manifest_sha256']!='3e434ef8e5280709c013de565bc66eb79851b4ed9ff7c74ca008e9c762f0d655':raise ValueError('OFM identity')
 out['ofm']=v
except FileNotFoundError:pass
except Exception as e:out['errors'].append('OFM: '+type(e).__name__)
try:
 p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_pilot_20261008/publication_v1')/HOST/'status.json'
 if p.is_symlink() or p.stat().st_size>1048576:raise ValueError('publication size')
 v=json.loads(p.read_bytes())
 pins={'xbabe2':'0584cfdaa8a2b8c8d71383d0bb153f1468e7541826015a6f4d660f0c9e13dee8','xbabe3':'18f63e95be411944affa32d7e2be17b554181cc0fe137d9fca4175a4501359a6'}
 if v.get('host')!=HOST or hashlib.sha256(p.read_bytes()).hexdigest()!=pins[HOST]:raise ValueError('publication identity')
 out['variant_publication']={k:v[k]for k in ('at','state','completed_variants','expected_variants','verified_objects','expected_objects','parent_lineages','new_gp_labels')}
except FileNotFoundError:pass
except Exception as e:out['errors'].append('Publication: '+type(e).__name__)
try:
 p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_dope_pilot_20261009/runs')/HOST/'progress.json'
 if p.is_symlink() or p.stat().st_size>1048576:raise ValueError('DOPE progress size')
 v=json.loads(p.read_bytes())
 if v['host']!=HOST or v['manifest_sha256']!='717b9302f94da71ba6690a12780b612a5785d48433e6725ca4240594a3eef709':raise ValueError('DOPE identity')
 out['dope']=v
except FileNotFoundError:pass
except Exception as e:out['errors'].append('DOPE: '+type(e).__name__)
if HOST=='xbabe2':
 for name,filename,pin in [('full_corpus', 'DL_FLYWHEEL_FULL_CORPUS_INVENTORY.json', '1f27e0455f14a5f9c1e2644d73e1a9c8d4362d6ac5b34452ae7b1ab8b1f362a3'), ('parents', 'DL_FLYWHEEL_PARENT_REGISTRY_V2.json', '74a405cf5ef9d5cc28a3ebe7fa1e4c72439b701973f1fd86344bd01d3c1651e9'), ('gp_qualification', 'DL_FLYWHEEL_S1_EMIT_QUALIFICATION_VERIFICATION.json', '2aeab5f5efffdd4e935b4ae5f21a71c5807a30de4e05302e6deba5a07becc80c'), ('paper_custody', 'DL_PAPER_S3_CUSTODY.json', '8b37c9a8f6d6ac32707942e32d8b88bc25ffbb5e5a67cf0121919490688ceb32'), ('gp_bridge', 'DL_FLYWHEEL_PROGRAM_BRIDGE_VERIFICATION.json', '8ebf8112183712e7473703475c760ee291e634bf7b500db14bbd91138f1c97bd'), ('gp_attempt_custody', 'DL_FLYWHEEL_PROGRAM_ATTEMPT_ARCHIVE_VERIFICATION.json', '30c7d77a51010c924e2bcb1b266f2f74fb9f37af08195d1563dcd1ccff226f37')]:
  try:
   p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/paper_observer_addfba0176a7')/filename
   b=p.read_bytes()
   if p.is_symlink() or len(b)>2097152 or hashlib.sha256(b).hexdigest()!=pin:raise ValueError('proof identity')
   v=json.loads(b)
   if name=='parents':v={k:v[k]for k in ('parents','partitions','rows_range','features_range','registry_sha256')}
   if name=='gp_qualification':v={**{k:v[k]for k in ('qualified_program_count','actual_inner_fit_count','prediction_rows_verified','prediction_rows_bit_exact','max_absolute_prediction_error','canonical_S3_publication_completed','teaching_admitted')},'programs':[{'program_sha256':r['program']['program_sha256'],'operator':r['program']['operator'],'rmse':r['rmse_per_inner_fold'],'baseline_rmse':r['baseline_rmse_per_inner_fold'],'variant_id':r['variant']['variant_id'],'train_rows':r['variant']['train_rows'],'target_units':r['variant']['target_units']}for r in v['result']['candidate_results']]}
   if name=='gp_bridge':v={'status':v['status'],'actual_evaluations':v['actual_proof']['actual_evaluations'],'canonical_score_rows':v['actual_proof']['canonical_score_rows'],'measurement':v['actual_proof']['after'][0]['measurement'],'observations':v['actual_proof']['observations'],'live_trainer_hook_exercised':v['actual_proof']['live_trainer_hook_exercised'],'proof_sha256':pin}
   if name=='gp_attempt_custody':v={'status':v['status'],'counts':v['service_status']['result']['counts'],'objects_verified':len(v['objects']),'proof_sha256':pin}
   if name=='paper_custody':v={'state':v.get('state'),'archive_count':len(v.get('archives',[])),'proof_sha256':pin}
   out[name]=v
  except Exception as e:out['errors'].append(name+': '+type(e).__name__)
try:
 p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_pilot_20261008/publication_v2')/HOST/'status.json'
 pins={'xbabe2':'c0227a71f3baaeca22d0b8cbb14118f61cf9ffc1e448e263259c51d112c3dc6f','xbabe3':'f2b6ee381c5faa6cdf1ac4fb58b8d3c6d14ccf48b3553b5b0a6c62137ad6510d'}
 raw=p.read_bytes()
 if p.is_symlink() or len(raw)>1048576 or hashlib.sha256(raw).hexdigest()!=pins[HOST]:raise ValueError('encoder publication identity')
 v=json.loads(raw);out['encoding_publication']={k:v[k]for k in ('at','state','completed_variants','new_vectors_verified','verified_objects','completed_encoders')}
except FileNotFoundError:pass
except Exception as e:out['errors'].append('Encoder publication: '+type(e).__name__)
if HOST=='xbabe2':
 for name,filename,pin in [('evergreen','flywheel_evergreen_20261009/data/progress.json','cd9ee8f882735c42b49d694b99dbbc4187035804099bc49f0ca81f8ec77a9b0a'),('gp_publication','flywheel_s1_publication_20261009_v1/progress.json','26e457d461a4629a51c913c79e8fd1593b7f71f593618a6bb3dcd3537a8ae1ed')]:
  try:
   p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')/filename
   raw=p.read_bytes()
   if p.is_symlink() or len(raw)>1048576:raise ValueError('live status bounds')
   v=json.loads(raw)
   if v['config_sha256']!=pin:raise ValueError('live status identity')
   out[name]=v
  except FileNotFoundError:pass
  except Exception as e:out['errors'].append(name+': '+type(e).__name__)
try:
 p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_encode16_20261009/publication_v1')/HOST/'status.json'
 pins={'xbabe2':'4bb7c0ba1e733ea4f59bd3db920011842e617c019aecfd780a891974b5e9a3a0','xbabe3':'94944b4eb7026881359e94e7672348f1160ab057b11d10d3e4aaa49fdf2aa77e'}
 raw=p.read_bytes()
 if p.is_symlink() or len(raw)>1048576 or hashlib.sha256(raw).hexdigest()!=pins[HOST]:raise ValueError('evergreen encoding publication identity')
 v=json.loads(raw)
 out['evergreen_encoding_publication']={**{k:v[k]for k in ('at','state','completed_variants','verified_objects','completed_encoders')},'new_vectors_verified':v['vectors_verified']}
except FileNotFoundError:pass
except Exception as e:out['errors'].append('Evergreen encoder publication: '+type(e).__name__)
if HOST=='xbabe2':
 try:
  root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_label16_20261009_v1')
  p=root/'progress.json';raw=p.read_bytes()
  if p.is_symlink() or len(raw)>4194304:raise ValueError('GP label progress cap')
  v=json.loads(raw)
  if v['config_sha256']!='c26ea0088211b23ba18407a8136efb8fbcebc89dfadb57c622ab18429ae19e3c':raise ValueError('GP label source identity')
  v['scored_context_ids']=sorted({r['variant_identity']for r in v.get('outcomes',[])if r.get('state')=='ready'});v['data_runs_with_scores']=len(v['scored_context_ids'])
  v['outcomes']=[{k:r.get(k)for k in ('slot','round','variant_identity','partition','phase','program_name','program_sha256','state','measurement','error')}for r in v.get('outcomes',[])[-12:]]
  if (root/'status.json').is_file():
   terminal=json.loads((root/'status.json').read_bytes())
   if terminal['config_sha256']!=v['config_sha256']:raise ValueError('GP terminal identity')
   v['state']=terminal['state'];v['terminal_at']=terminal['at']
  out['gp_label_campaign']=v
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('GP labeling: '+type(e).__name__)
if HOST=='xbabe2':
 try:
  root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_fill3_20261009_v1')
  p=root/'progress.json';raw=p.read_bytes()
  if p.is_symlink() or len(raw)>4194304:raise ValueError('GP fill progress cap')
  v=json.loads(raw)
  if v['config_sha256']!='70de9a40a16887f246db2dca83c54d6f5e535a513a228accd9b3bb7de7edd931':raise ValueError('GP fill identity')
  v['outcomes']=[{k:r.get(k)for k in ('slot','variant_identity','partition','phase','program_name','program_sha256','state','measurement','error')}for r in v.get('outcomes',[])]
  if (root/'status.json').is_file():
   terminal=json.loads((root/'status.json').read_bytes())
   if terminal['config_sha256']!=v['config_sha256']:raise ValueError('GP fill terminal identity')
   v['state']=terminal['state'];v['terminal_at']=terminal['at']
  out['gp_fill_campaign']=v
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('GP fill: '+type(e).__name__)
 for neural_name,neural_pin,neural_unit in [
  ('flywheel_neural_20261009_v1','ce55dfe5f887d78e50937545d294850625fc036da5305cb3b1070205a3699cf5','jopedime-flywheel-neural-20261009-v1'),
  ('flywheel_neural_continuation1_20261009','e94ee45bcc4fc4fa089d5e0e42f73d809cc4e48322f24a148434d069af1c955e','jopedime-flywheel-neural-continuation1-20261009'),
  ('flywheel_neural_continuation1_recovery1_20261009','822cb18a250e1f3e163172e744b5680aca04a9e776cfc5bb32552d854b98c74c','jopedime-flywheel-neural-continuation1-recovery1-20261009')]:
  try:
   root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')/neural_name
   if not(root/'config.json').exists():continue
   cfg=root/'config.json';raw=cfg.read_bytes()
   if cfg.is_symlink() or len(raw)>2097152 or hashlib.sha256(raw).hexdigest()!=neural_pin:raise ValueError('Neural source identity')
   config=json.loads(raw)
   out.pop('neural_terminal',None);out.pop('neural_unfinished_stop',None)
   if neural_name!='flywheel_neural_20261009_v1':
    out['neural_transition']={'state':'checking_resume','config_sha256':neural_pin,'from_completed_batches':out.get('neural_training',{}).get('completed_batches',0)}
   p=root/'training/progress.json'
   if p.exists():
    raw=p.read_bytes()
    if p.is_symlink() or len(raw)>4194304:raise ValueError('Neural progress cap')
    v=json.loads(raw)
    if v['schema']!='jope.flywheel-neural-progress/v1' or v['config_sha256']!=neural_pin:raise ValueError('Neural progress identity')
    out['neural_training']=v;out.pop('neural_transition',None)
   service=subprocess.run(['systemctl','--user','show',neural_unit,'--property=ActiveState,SubState,Result,ExecMainStatus'],capture_output=True,text=True,timeout=5,check=True)
   state=dict(line.split('=',1)for line in service.stdout.splitlines()if '='in line)
   out['neural_service']=state
   for filename in ('status.json','fatal-status.json'):
    p=root/filename
    if not p.exists():continue
    raw=p.read_bytes()
    if p.is_symlink() or len(raw)>4194304:raise ValueError('Neural terminal cap')
    t=json.loads(raw)
    if filename=='status.json'and t['config_sha256']!=neural_pin:raise ValueError('Neural terminal identity')
    out['neural_terminal']={k:t.get(k)for k in ('state','at','optimizer_steps','error','attempt_custody')}
   if state.get('ActiveState')not in ('active','activating') and 'neural_terminal'not in out:
    out['neural_unfinished_stop']=True
  except FileNotFoundError:pass
  except Exception as e:out['errors'].append('Neural training: '+type(e).__name__)
if HOST=='xbabe2':
 try:
  root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_growth_pipeline_v2_20261009_r2')
  pin='b4eb3a3aa5e85ff9af0eab29101c21dec74018826e3244ee9b5103caa47d5e78'
  raw=(root/'config.json').read_bytes()
  if (root/'config.json').is_symlink()or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('growth config identity')
  p=root/'progress.json';raw=p.read_bytes()
  if p.is_symlink()or len(raw)>1048576:raise ValueError('growth progress cap')
  v=json.loads(raw)
  if v['schema']!='jope.flywheel-growth-pipeline/v2'or v['config_sha256']!=pin:raise ValueError('growth identity')
  out['evergreen']=v
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('Growth pipeline: '+type(e).__name__)
 try:
  p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/paper_observer_addfba0176a7/DL_FLYWHEEL_GROWTH10_KUMOS_PUBLICATION_RECOVERY1_COMPLETE.json');raw=p.read_bytes()
  if p.is_symlink()or hashlib.sha256(raw).hexdigest()!='11811a380e6c61cb580ac7510c61a9ce244cae84c87833d1debcc887e3607520':raise ValueError('growth10 Kumo closure')
  v=json.loads(raw)
  if v['state']!='complete'or v['completed_variants']!=10 or v['verified_objects']!=20:raise ValueError('growth10 counts')
  out['growth10_kumos_publication']={k:v[k]for k in ('at','state','completed_variants','verified_vectors','verified_objects')}
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('Growth10 Kumo: '+type(e).__name__)
if HOST=='xbabe2':
 try:
  root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_neural_native_full_20261009_v1')
  pin='3725119c62edb81f7e1be7a92b307a10a71812900e4f80d428c37fc60cb0fda4'
  raw=(root/'config.json').read_bytes()
  if (root/'config.json').is_symlink() or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('native neural config identity')
  raw=(root/'training/progress.json').read_bytes()
  if (root/'training/progress.json').is_symlink() or len(raw)>8388608:raise ValueError('native neural progress cap')
  v=json.loads(raw)
  if v['schema']!='jope.flywheel-neural-progress/v1' or v['config_sha256']!=pin:raise ValueError('native neural progress identity')
  inv=v.get('native_teacher_inventory',{});v['native_teacher_inventory']={k:x for k,x in inv.items() if k in ('schema','counts','spec_sha256','at')}
  if out.get('neural_training'):out['neural_previous_run']={k:x for k,x in out['neural_training'].items() if k not in ('loss_history','optimizer_history','resolver_outcomes')}
  out['neural_training']=v;out['neural_native_corpus']=True
  for key in ('neural_terminal','neural_unfinished_stop','neural_transition'):out.pop(key,None)
  service=subprocess.run(['systemctl','--user','show','jopedime-neural-native-full-20261009-v1','--property=ActiveState,SubState,Result,ExecMainStatus'],capture_output=True,text=True,timeout=5,check=True)
  state=dict(line.split('=',1)for line in service.stdout.splitlines()if '='in line);out['neural_service']=state
  for filename in ('status.json','fatal.json'):
   p=root/filename
   if not p.is_file():continue
   raw=p.read_bytes()
   if p.is_symlink() or len(raw)>8388608:raise ValueError('native neural terminal cap')
   t=json.loads(raw)
   if t.get('config_sha256',pin)!=pin:raise ValueError('native neural terminal identity')
   out['neural_terminal']={k:t.get(k)for k in ('state','at','optimizer_steps','error')}
  if state.get('ActiveState')not in ('active','activating')and 'neural_terminal'not in out:out['neural_unfinished_stop']=True
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('Neural training: native '+type(e).__name__)
try:
 root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_kumos112_20261009')
 p=root/'runs'/HOST/'progress.json';raw=p.read_bytes()
 if p.is_symlink() or len(raw)>1048576:raise ValueError('Kumo progress cap')
 v=json.loads(raw)
 if v['host']!=HOST or v['manifest_sha256']!='d7c48defd7252361beba4555a5178e47e1f2d5094ace0c623980cd6e6ea43f29' or v['roster_version']!='kumo-s-only-v1':raise ValueError('Kumo source identity')
 out['kumos_encoding']=v
except FileNotFoundError:pass
except Exception as e:out['errors'].append('Kumo encoding: '+type(e).__name__)
if HOST=='xbabe2':
 try:
  p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_kumos112_20261009/finalization1/xbabe2/status.json')
  raw=p.read_bytes()
  if p.is_symlink() or len(raw)>1048576 or hashlib.sha256(raw).hexdigest()!='9497a4d79b030d02b68371262fbfef2973aed0fe11f3bc890a894453bf278e39':raise ValueError('Kumo finalization identity')
  v=json.loads(raw)
  if v['state']!='complete' or v['original_producer_sealed'] is not False or v['original_coordinator_state']!='failed':raise ValueError('Kumo finalization state')
  out['kumos_encoding_prior_attempt']=out.get('kumos_encoding')
  out['kumos_encoding']={'host':HOST,'state':'verified_after_finalization','at':v['completed_at_unix'],
      'completed_variant_components':v['completed_variant_components'],'completed_variants':v['completed_variants'],
      'original_coordinator_state':'failed','original_producer_sealed':False,'recomputed_vectors':0,
      'manifest_sha256':v['manifest_sha256'],'roster_version':'kumo-s-only-v1','required_components':['kumo_tabular_s']}
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('Kumo finalization: '+type(e).__name__)
try:
 pin={'xbabe2': {'config_path': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_kumos112_20261009/publisher_runtime_de56f9ce6265/config_xbabe2.json', 'config_sha256': '40e7cdef76aa543791cb49ae85ad11fed56d726ddc07af1ac49443d9d5e474a3', 'objects': 112, 'schema': 'jope.flywheel-evergreen-kumos-publication/v1', 'state': 'complete', 'status_path': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_kumos112_20261009/publication_kumos_v1/xbabe2/status.json', 'variants': 56, 'vectors': 56}, 'xbabe3': {'config_path': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_kumos112_20261009/publisher_runtime_de56f9ce6265/config_xbabe3.json', 'config_sha256': 'f6f042cd61fae4e783012dcf01c41b42afb00e058ea00ae159385c841ce33629', 'objects': 112, 'schema': 'jope.flywheel-evergreen-kumos-publication/v1', 'state': 'complete', 'status_path': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_kumos112_20261009/publication_kumos_v1/xbabe3/status.json', 'variants': 56, 'vectors': 56}}[HOST]
 p=Path(pin['config_path']);raw=p.read_bytes()
 if p.is_symlink() or len(raw)>1048576 or hashlib.sha256(raw).hexdigest()!=pin['config_sha256']:raise ValueError('Kumo publication config identity')
 p=Path(pin['status_path']);raw=p.read_bytes()
 if p.is_symlink() or len(raw)>16777216:raise ValueError('Kumo publication cap')
 v=json.loads(raw)
 if v['schema']!=pin['schema'] or v['host']!=HOST or v['state']!='complete' or v['completed_variants']!=56 or v['vectors_verified']!=56 or v['verified_objects']!=112:raise ValueError('Kumo publication closure')
 out['kumos_publication']={k:v[k]for k in ('state','at','completed_variants','vectors_verified','verified_objects')}
except FileNotFoundError:pass
except Exception as e:out['errors'].append('Kumo publication: '+type(e).__name__)

if HOST=='xbabe2':
 out['gp_expansion_campaigns']=[]
 try:
  root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_hpc_growth10_20261009')
  p=root/'dispatch-manifest.json';raw=p.read_bytes()
  if p.is_symlink()or len(raw)>1048576 or hashlib.sha256(raw).hexdigest()!='dd25d0e41372bdec02418379c3b6dacd7176b4764ac9158a0f2966efebffc76f':raise ValueError('GP80 manifest identity')
  manifest=json.loads(raw)
  for job in manifest['jobs']:
   p=Path(job['config']['path']);raw=p.read_bytes()
   if p.is_symlink()or len(raw)!=job['config']['bytes']or hashlib.sha256(raw).hexdigest()!=job['config']['sha256']:raise ValueError('GP80 config identity')
   cfg=json.loads(raw);slots=len(cfg['variants'])*len(cfg['programs'])
   if slots!=cfg['maximum_actual_evaluations']:raise ValueError('GP80 planned slots')
   v={'state':'staged','config_sha256':job['config']['sha256'],'counts':{},'outcomes':[],'scored_context_ids':[]}
   p=Path(job['root'])/'progress.json'
   if p.exists():
    raw=p.read_bytes()
    if p.is_symlink()or len(raw)>8388608:raise ValueError('GP80 progress bounds')
    v=json.loads(raw)
    if v['config_sha256']!=job['config']['sha256']:raise ValueError('GP80 progress identity')
    v['scored_context_ids']=sorted({r['variant_identity']for r in v.get('outcomes',[])if r.get('state')=='ready'})
    v['outcomes']=[{k:r.get(k)for k in ('slot','variant_identity','partition','phase','program_name','program_sha256','state','measurement','error')}for r in v.get('outcomes',[])[-12:]]
   p=Path(job['root'])/'status.json'
   if p.exists():
    raw=p.read_bytes()
    if p.is_symlink()or len(raw)>16777216:raise ValueError('GP80 terminal bounds')
    terminal=json.loads(raw)
    if terminal['config_sha256']!=job['config']['sha256']:raise ValueError('GP80 terminal identity')
    v.update(state=terminal['state'],terminal_at=terminal['at'])
   v.update(label='xbabe2 / '+job['id'],expected_slots=slots,data_runs_with_scores=len(v['scored_context_ids']))
   out['gp_expansion_campaigns'].append(v)
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('GP80: '+type(e).__name__)

if HOST=='xbabe3':
 out['gp_expansion_campaigns']=[]
 for index,pin in {'0': {'sha256': '1e332c5de94fe4356ddcb913af31affe9fd2b5ad9dfce193946b71abb2d88b13', 'path': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_label112_20261009_v1/recovery1/shard0/config.json', 'progress': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_label112_20261009_v1/recovery1/shard0/progress.json', 'status': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_label112_20261009_v1/recovery1/shard0/status.json'}, '1': {'sha256': 'c0fe5cf52c5e9b8365ed06026f7be58009a65041d71e063920759fa72adfb04c', 'path': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_label112_20261009_v1/recovery1/shard1/config.json', 'progress': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_label112_20261009_v1/recovery1/shard1/progress.json', 'status': '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_label112_20261009_v1/recovery1/shard1/status.json'}}.items():
  try:
   root=Path(pin['path']).parent
   cfg=Path(pin['path'])
   if not cfg.exists():continue
   raw=cfg.read_bytes()
   if cfg.is_symlink() or len(raw)>2097152 or hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('GP112 config identity')
   p=Path(pin['progress'])
   v={'state':'staged','config_sha256':pin['sha256'],'counts':{},'outcomes':[],'data_runs_with_scores':0}
   if p.exists():
    raw=p.read_bytes()
    if p.is_symlink() or len(raw)>8388608:raise ValueError('GP112 progress bounds')
    v=json.loads(raw)
    if v['config_sha256']!=pin['sha256']:raise ValueError('GP112 progress identity')
    v['scored_context_ids']=sorted({r['variant_identity']for r in v.get('outcomes',[])if r.get('state')=='ready'});v['data_runs_with_scores']=len(v['scored_context_ids'])
    v['outcomes']=[{k:r.get(k)for k in ('slot','variant_identity','partition','phase','program_name','program_sha256','state','measurement','error')}for r in v.get('outcomes',[])[-12:]]
   p=Path(pin['status'])
   if p.exists():
    raw=p.read_bytes()
    if p.is_symlink() or len(raw)>16777216:raise ValueError('GP112 terminal bounds')
    t=json.loads(raw)
    if t['config_sha256']!=pin['sha256']:raise ValueError('GP112 terminal identity')
    v.update(state=t['state'],terminal_at=t['at'])
   p=root/'fatal-status.json'
   if p.exists():
    raw=p.read_bytes()
    if p.is_symlink()or len(raw)>1048576:raise ValueError('GP112 failure bounds')
    t=json.loads(raw);v.update(state='failed',error=t.get('error'),terminal_at=t.get('at'))
   service=subprocess.run(['systemctl','--user','show','jopedime-gp112r1-shard'+index+'-20261009-v1','--property=ActiveState,SubState,Result,ExecMainStatus'],capture_output=True,text=True,timeout=5,check=True)
   v['service']=dict(line.split('=',1)for line in service.stdout.splitlines()if '='in line)
   v['shard']=index;v['host']=HOST;v['expected_slots']=448
   out['gp_expansion_campaigns'].append(v)
  except Exception as e:out['errors'].append('GP112 shard'+index+': '+type(e).__name__)

if HOST=='xbabe3':

 try:
  p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_label112_20261009_v1/recovery1/original-zero-attempt-proof.json')
  raw=p.read_bytes()
  if p.is_symlink()or len(raw)>2097152 or hashlib.sha256(raw).hexdigest()!='60476a05cef4c599da911b62388a02ff6cc5b19adb064cab80c2f2553b9268a7':raise ValueError('old GP attempt proof')
  t=json.loads(raw)
  if t['closed']is not True or t['all_native_attempts']!=0 or t['all_result_puts']!=0:raise ValueError('old GP zero-attempt proof')
  out['gp_expansion_previous_attempt']={'state':'closed_before_execution','worker_guard_refusals':895,'native_attempts':0,'result_puts':0,'custody_verified':True}
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('GP112 previous attempt: '+type(e).__name__)

if HOST=='xbabe2':
 for key,name,pin in [('hpc_profile','DL_FLYWHEEL_HPC_BOTTLENECK_PROFILE.json','89762b2d867ee0acd67bd6527a8046a104bcafe1e832b7f34bb1804164742894'),('inference_first','DL_FLYWHEEL_INFERENCE_FIRST_COMPLETE.json','3898f8a83b12fd4f8a5df9d466a6ae054708378427a00094f7c1a0543d4c0373')]:
  try:
   p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/paper_observer_addfba0176a7')/name;raw=p.read_bytes()
   if p.is_symlink()or len(raw)>2097152 or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('evidence pin')
   v=json.loads(raw)
   if key=='inference_first':v={k:v[k]for k in ('at','status','timing_events','timing_stage_summary','decode','all_expected_record_versions_and_seven_artifacts_verified')}
   out[key]=v
  except Exception as e:out['errors'].append(key+': '+type(e).__name__)

attempts=[('initial','runs/'+HOST,'e561d42d83fa655e1d3391747183b87587ee20be0291ef04db6d6116d26dabc9')]
if HOST=='xbabe2':attempts.append(('recovery1','runs/recovery1/'+HOST,'b3d448a20668f6ae16c5369e589600ff1ffb9f1f642a3c43e5d179b0f72e35ee'))
for label,folder,pin in attempts:
 try:
  p=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_encode16_20261009')/folder/'progress.json'
  raw=p.read_bytes()
  if p.is_symlink() or len(raw)>1048576:raise ValueError('evergreen encoder progress cap')
  v=json.loads(raw)
  if v['host']!=HOST or v['manifest_sha256']!=pin:raise ValueError('evergreen encoder identity')
  v['attempt']=label
  out.setdefault('evergreen_encoding_attempts',[]).append(v)
  out['evergreen_encoding']=v
 except FileNotFoundError:pass
 except Exception as e:out['errors'].append('Evergreen encoding '+label+': '+type(e).__name__)
try:
 r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10,check=True)
 out['gpus']=[dict(zip(['uuid','name','utilization_percent','memory_used_mib','memory_total_mib'],[x.strip() for x in line.split(',')])) for line in r.stdout.splitlines()]
except Exception as e:out['errors'].append('GPU counters: '+type(e).__name__)
for x in ENTRIES:
 try:
  p=Path(x['archive'])/'evidence_summary.json'
  if p.is_symlink() or p.stat().st_size>4194304:raise ValueError('summary shape')
  s=json.loads(p.read_bytes())
  if s['host']!=HOST or s['manifest_sha256']!=x['manifest_sha256']:raise ValueError('identity mismatch')
  s['archive_root']=x['archive'];s['observer_unit']=x['unit'];out['archives'].append(s)
 except Exception as e:out['errors'].append(x['label']+': '+type(e).__name__)

try:
 root=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/hpc_trace_20261009_v1')
 if hashlib.sha256((root/'host_trace.py').read_bytes()).hexdigest()!='b181c4881521ca1630ca29ccbad69ce3a23470940424708dd56e5807375c31ed':raise ValueError('trace source')
 for field,filename,schema in [('hpc_trace','logs/status.json','jope.hpc-host-status/v1'),('hpc_archive','logs/archive-v1/status.json','jope.hpc-archive-status/v1')]:
  p=root/filename
  if not p.exists():continue
  raw=p.read_bytes()
  if p.is_symlink()or len(raw)>65536:raise ValueError('HPC status bounds')
  v=json.loads(raw)
  if v['schema']!=schema or v['host']!=out['host']:raise ValueError('HPC status host/schema')
  out[field]=v
except FileNotFoundError:pass
except Exception as e:out['errors'].append('HPC telemetry: '+type(e).__name__)

cpu1=[int(v)for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
delta=sum(cpu1)-sum(cpu0);idle=(cpu1[3]+cpu1[4])-(cpu0[3]+cpu0[4]);out['cpu_utilization_percent']=max(0,min(100,100*(delta-idle)/delta))if delta>0 else None
print(json.dumps(out,allow_nan=False))
'''.replace('HOST', repr(host)).replace('ENTRIES', repr(entries))
    cmd = ['/usr/bin/python3', '-c', code] if host == 'xbabe2' else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, 'python3 -c ' + shlex.quote(code)]
    r = subprocess.run(cmd, text=True, capture_output=True, timeout=25, check=True)
    if len(r.stdout) > 4 * 1024 * 1024:
        raise ValueError('publisher response too large')
    return json.loads(r.stdout)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    entries = json.loads(a.deployment.read_text())
    assert len(entries) == 3 and {x['host'] for x in entries} == {'xbabe2', 'xbabe3'}
    deadline = min(x['deadline'] for x in entries)
    while time.time() < deadline:
        s = {'schema': 'jope.paper-observatory/v1', 'at': time.time(), 'archives': [], 'hosts': [], 'errors': []}
        for host in ('xbabe1', 'xbabe2', 'xbabe3'):
            try:
                h = collect_compute(host) if host=='xbabe1' else collect(host, [x for x in entries if x['host'] == host])
                s['archives'].extend(h.pop('archives'));s['hosts'].append(h)
            except Exception as exc:
                s['errors'].append({'host': host, 'error': type(exc).__name__, 'message': str(exc)[:300]})
        host2 = next((h for h in s['hosts'] if h['host']=='xbabe2'), {})
        g = host2.get('evergreen', {}); gp = host2.get('gp_label_campaign', {}).get('counts', {}); fill = host2.get('gp_fill_campaign', {}).get('counts', {}); neural = host2.get('neural_training', {})
        expanded = sum(x.get('counts',{}).get('completed_publications',0)for h in s['hosts']for x in h.get('gp_expansion_campaigns',[]))
        point = {'at': s['at'], 'data_runs_published': g.get('published',0),
                 'gp_results_published': host2.get('gp_publication',{}).get('published_programs',0)+host2.get('gp_bridge',{}).get('actual_evaluations',0)+gp.get('completed_publications',0)+fill.get('completed_publications',0)+neural.get('published_results',0)+host2.get('neural_previous_run',{}).get('published_results',0)+expanded,
                 'unique_data_runs_seen': host2.get('neural_training',{}).get('unique_train_data_runs_seen',0),
                 'completed_batches': host2.get('neural_training',{}).get('completed_batches',0),'neural_run_id':neural.get('run_id',neural.get('config_sha256'))}
        history_path = a.deployment.parent/'flywheel_history_v1.jsonl'
        history = []
        if history_path.exists():
            if history_path.is_symlink() or history_path.stat().st_size>8*1024**2:raise ValueError('history cap')
            previous = '0'*64
            for line in history_path.read_bytes().splitlines():
                event = json.loads(line); expected = event.pop('sha256')
                body = json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                if event['previous']!=previous or hashlib.sha256(body).hexdigest()!=expected:raise ValueError('history integrity')
                previous=expected;history.append(event['point'])
        else:previous='0'*64
        event={'point':point,'previous':previous}
        event['sha256']=hashlib.sha256(json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if not a.once:
            with history_path.open('ab')as stream:
                stream.write((json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
        history.append(point);s['flywheel_history']=history[-400:]
        tmp = a.out.with_suffix('.new')
        tmp.write_text(json.dumps(s, allow_nan=False) + '\n');os.replace(tmp, a.out)
        print(json.dumps({'at': s['at'], 'archives': len(s['archives']), 'errors': s['errors']}), flush=True)
        if a.once:
            return
        time.sleep(max(0, min(15, deadline - time.time())))

if __name__ == '__main__':
    main()
