"""NEW synthetic capture-reuse scenarios; no old qualification is replayed."""
from copy import deepcopy
from pathlib import Path
import os
from monitor_capture_reuse_v1.vendor.common import canonical,digest,ZERO
from monitor_capture_reuse_v1.reuse import POLICY,ExternalSeal,authenticate_history
from monitor_capture_reuse_v1.vendor.phase_verifier import PhaseBinding

CLOCK={'boot_id':'ce7634f6-3038-4875-85bd-03fb277dbcfd','created_unix':1000.0,'deadline_unix':4600.0,'created_monotonic_ns':1000*10**9,'deadline_monotonic_ns':4600*10**9}
def pin(w):return {'sha256':digest(w),'bytes':len(w)}
def physical(path,wire,inode,immutable=False):
    d={'path':str(path),**pin(wire),'device':2,'inode':inode,'mtime_ns':10}
    if immutable:d.update(uid=os.getuid(),mode=0o400,nlink=1,file_type='regular')
    return d

def phase():
    config=dict(CLOCK,source_manifest_sha256=digest(b'source'));cw=canonical(config);cs=digest(cw)
    snapshot=canonical({'standalone_monitor':{'config_sha256':cs,'source_manifest_sha256':config['source_manifest_sha256'],'attempt':1}})
    sr={'path':'/actual/published/snapshot.json',**pin(snapshot)};rows=[];head=ZERO
    events=[('monitor.started',{'legacy_history_readonly':None}),('collection.attempt',{'attempt':1}),('publication.attempt',{'attempt':1,'qualification_only':False}),('publication.verified',{'attempt':1,'snapshot':sr,'errors':[]})]
    for i,(kind,body)in enumerate(events):
        line=canonical({'seq':i+1,'previous_sha256':head,'kind':kind,'utc_ns':(4550+i)*10**9,'monotonic_ns':(4550+i)*10**9,'config_sha256':cs,'body':body});rows.append(line);head=digest(line)
    wire=b'\n'.join(rows)+b'\n'
    result={'schema':'jope.paper-monitor-result/v1','state':'window_closed_before_next_collection','at':4554.0,'config_sha256':cs,'attempts':1,'last_verified_snapshot':sr,'last_publication_custody':{'attempt':1,'phase':'verified','snapshot_may_have_been_replaced':True,'attempted_snapshot':sr},'event_count':4,'event_head_sha256':head,'optimizer_updates':0,'GP_executions':0,'S3_PUT':0,'old_windows_mutated':False,'retry_allowed':False}
    binding=PhaseBinding(cs,config['source_manifest_sha256'],canonical(CLOCK),sr['path'],b'null',((2,digest(b'\n'.join(rows[:2])+b'\n')),))
    return cw,wire,canonical(result),snapshot,binding

def scenario(tmp):
    tmp=Path(tmp).resolve();evidence={}
    def ev(name):
        w=canonical({'synthetic_source_evidence':name});r=pin(w);evidence[r['sha256']]=w;return r
    ns={'id':'old14','root':'/historical/old14','source_manifest_sha256':digest(b'oldsource'),'config_sha256':digest(b'oldconfig'),'clock':deepcopy(CLOCK),'status':'normal_closed','window_state':'frozen','members':[],'member_manifest_sha256':'','acceptance':None}
    for i,(name,role,wire)in enumerate([('config.json','config',b'oldconfig'),('events.jsonl','events',b'old exact events\n'),('result.json','result',b'old result'),('source.json','source',b'oldsource')]):
        ns['members'].append({'relative_path':name,'role':role,'original':physical(ns['root']+'/'+name,wire,100+i),'custody':physical(tmp/'custody'/name,wire,200+i,True)})
    ns['member_manifest_sha256']=digest(canonical(ns['members']))
    acc={'schema':'codex.monitor-capture-reuse-acceptance/v1','namespace':ns['id'],'member_manifest_sha256':ns['member_manifest_sha256'],'source_manifest_sha256':ns['source_manifest_sha256'],'config_sha256':ns['config_sha256'],'clock_sha256':digest(canonical(CLOCK)),'status':ns['status'],'window_state':ns['window_state'],'previous_acceptance_sha256':ZERO,'capture':ev('actual capture all four bodies'),'verifier_source':ev('source reviewed complete verifier'),'independent_review':ev('separate independent acceptance'),'source_inventory_sha256':digest(canonical([m for m in ns['members']if m['role']=='source'])),'terminal_capture':{'captured_unix':4590.0,'completed_monotonic_ns':4590*10**9,'boot_id':CLOCK['boot_id'],'fsync_receipt':ev('capture body fsync before endpoint'),'physical_closure_receipt':ev('three exact units closed before and after'),'event_bytes_pin':pin(b'old exact events\n'),'result_pin':pin(b'old result')}}
    aw=canonical(acc);ns['acceptance']=pin(aw);g={'schema':'codex.monitor-accepted-custody-graph/v1','generation_index':14,'namespaces':[ns],'acceptance_head':digest(aw)}
    root=tmp/'current';root.mkdir();fresh=[]
    for name,wire in [('source.json',b'current source'),('config.json',b'current config'),('snapshot.json',b'current snapshot')]:
        p=root/name;p.write_bytes(wire);st=p.stat();fresh.append({'path':str(p),**pin(wire),'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns})
    cc={**CLOCK,'created_unix':5000.0,'deadline_unix':8600.0,'created_monotonic_ns':5000*10**9,'deadline_monotonic_ns':8600*10**9}
    current={'namespace':'new15','root':str(root),'source_manifest_sha256':digest(b'current source'),'config_sha256':digest(b'current config'),'clock':cc,'snapshot_path':str(root/'snapshot.json'),'inventory_sha256':digest(canonical(sorted(fresh,key=lambda r:r['path']))),'guards':{k:ev('fresh guard '+k)for k in POLICY['fresh_guards']}}
    seal={'schema':'codex.monitor-capture-reuse-root-seal/v1','generation':15,'graph':pin(canonical(g)),'acceptance_head':digest(aw),'policy':deepcopy(POLICY),'current':current,'evidence':{},'producer_source':ev('producer source'),'producer_review':ev('producer review'),'root_commitment':ev('external operator commitment')};seal['evidence']={k:pin(v)for k,v in evidence.items()}
    f={'graph':g,'acceptance':acc,'seal':seal,'evidence':evidence,'fresh':sorted(fresh,key=lambda x:x['path']),'seal_path':tmp/'root-seal.json'};reseal(f);return f

def reseal(f):
    aw=canonical(f['acceptance']);f['graph']['namespaces'][0]['acceptance']=pin(aw);f['graph']['acceptance_head']=digest(aw)
    f['graph_wire']=canonical(f['graph']);s=f['seal'];s['graph']=pin(f['graph_wire']);s['acceptance_head']=digest(aw);f['aw']=(aw,);f['seal_wire']=canonical(s)
    p=f['seal_path']
    if p.exists():p.chmod(0o600)
    p.write_bytes(f['seal_wire']);p.chmod(0o400);st=p.stat();f['seal_pin']={'path':str(p),**pin(f['seal_wire']),'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns,'uid':st.st_uid,'mode':0o400,'nlink':1,'file_type':'regular'}
    f['external']=ExternalSeal(canonical(f['seal_pin']),s['generation'],digest(f['graph_wire']),s['acceptance_head'],digest(canonical(s['policy'])))

def verified(f):return authenticate_history(f['seal_wire'],f['external'],f['seal_pin'],dict(f['seal_pin']),f['graph_wire'],f['aw'],f['evidence'])
def request(f):return {'schema':'codex.monitor-fresh-capture-plan/v1','seal_sha256':digest(f['seal_wire']),'generation':15,'predecessor_graph_sha256':digest(f['graph_wire']),'predecessor_acceptance_head':f['seal']['acceptance_head'],'current':deepcopy(f['seal']['current']),'members':deepcopy(f['fresh'])}
