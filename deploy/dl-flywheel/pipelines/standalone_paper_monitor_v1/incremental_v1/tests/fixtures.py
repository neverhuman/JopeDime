"""Small synthetic closed-evidence fixtures. No file, clock, process or network IO."""
from copy import deepcopy
from monitor_incremental_custody_v1.common import canonical, digest, ZERO
from monitor_incremental_custody_v1.authority import (CAPS, LIMITS, FRESH_GUARDS, ExternalAnchor, verify_authority)
from monitor_incremental_custody_v1.custody_manifest import BodyWitness, verify_accepted_history
from monitor_incremental_custody_v1.phase_verifier import PhaseBinding

CLOCK = {'boot_id':'ce7634f6-3038-4875-85bd-03fb277dbcfd', 'created_unix':1000.0,
         'deadline_unix':4600.0, 'created_monotonic_ns':10**12, 'deadline_monotonic_ns':4600*10**9}


def pin(wire): return {'sha256':digest(wire), 'bytes':len(wire)}


def physical(path, wire, inode, immutable=False):
    result=dict(path=path, **pin(wire), device=7, inode=inode, mtime_ns=900000000000)
    if immutable: result.update(uid=1000, mode=0o400, nlink=1, file_type='regular')
    return result


def namespace(name, base_inode, status):
    bodies={'source.json':canonical({'source':name}), 'config.json':canonical({'config':name})}
    roles={'source.json':'source','config.json':'config'}
    if status in ('failed','unknown'):
        bodies[status+'.json']=canonical({'state':status,'retry_allowed':False}); roles[status+'.json']=status if status=='unknown' else 'failure'
    if status == 'held_source_only':
        del bodies['config.json']; del roles['config.json']
    members=[]; witnesses=[]
    for i, (path, wire) in enumerate(sorted(bodies.items())):
        original=physical('/original/'+name+'/'+path, wire, base_inode+i)
        custody=physical('/custody/'+name+'/'+path, wire, base_inode+100+i, True)
        members.append({'relative_path':path,'role':roles[path],'original':original,'custody':custody})
        witnesses.append(BodyWitness(name,path,wire,canonical(custody),canonical(custody)))
    result={'id':name,'root':'/original/'+name,'source_manifest_sha256':digest(bodies['source.json']),
            'config_sha256':digest(bodies['config.json']) if 'config.json' in bodies else None,
            'clock':deepcopy(CLOCK) if 'config.json' in bodies else None,
            'window_state':'frozen' if 'config.json' in bodies else 'not_created','status':status,
            'members':members,'member_manifest_sha256':digest(canonical(members)),'acceptance':None}
    return result, witnesses


def family(statuses=('failed','unknown','normal_closed')):
    old=[]; witnesses=[]; receipts=[]; previous=ZERO
    for i, status in enumerate(statuses):
        ns, ws=namespace('old'+str(i),1000+1000*i,status)
        acc={'schema':'codex.monitor-custody-acceptance/v1','namespace':ns['id'],
             'member_manifest_sha256':ns['member_manifest_sha256'],
             'source_manifest_sha256':ns['source_manifest_sha256'],'config_sha256':ns['config_sha256'],
             'clock_sha256':digest(canonical(ns['clock'])),'status':status,'window_state':ns['window_state'],
             'previous_acceptance_sha256':previous,'capture':pin(canonical({'captured':ns['id']})),
             'verifier_source':pin(b'synthetic independently reviewed verifier source')}
        wire=canonical(acc); ns['acceptance']=pin(wire); old.append(ns); witnesses.extend(ws)
        receipts.append(wire); previous=digest(wire)
    graph={'schema':'codex.monitor-accepted-custody-graph/v1','generation_index':3,
           'namespaces':old,'acceptance_head':previous}
    graph_wire=canonical(graph)
    new,new_witnesses=namespace('new4',8000,'pending_verification')
    policy={'caps':deepcopy(CAPS),'resource_roster_sha256':'1'*64,'device_policy_sha256':'2'*64,
            'foreign_boundaries_sha256':'3'*64,'activation_chain_sha256':'4'*64,'boundary_count':5}
    authority={'schema':'codex.monitor-incremental-authority/v1','generation_index':4,
               'predecessor_graph_sha256':digest(graph_wire),'predecessor_acceptance_sha256':previous,
               'engine_source':pin(b'synthetic new source'),'collector_source':pin(b'synthetic frozen collector'),
               'policy':policy,'limits':deepcopy(LIMITS),'fresh_guards':list(FRESH_GUARDS),
               'new_namespace':{k:new[k] for k in ('id','root','source_manifest_sha256','config_sha256')},
               'clock':deepcopy(CLOCK)}
    aw=canonical(authority); seal=physical('/external/root/authority.json',aw,50000,True)
    anchor=ExternalAnchor(canonical(seal),4,digest(graph_wire),previous,digest(canonical(policy)))
    return {'graph':graph,'graph_wire':graph_wire,'acceptances':tuple(receipts),'witnesses':tuple(witnesses),
            'new':new,'new_witnesses':tuple(new_witnesses),'authority':authority,'aw':aw,'seal':seal,'anchor':anchor}


def authenticated(f): return verify_authority(f['aw'],f['anchor'],f['seal'],deepcopy(f['seal']))


def verified(f):
    a=authenticated(f)
    h=verify_accepted_history(f['graph_wire'],f['acceptances'],f['witnesses'],a)
    return a,h


def candidate(f,a,h):
    d={'schema':'codex.monitor-pending-custody-extension/v1','generation_index':4,
       'authority_sha256':a.sha256,'predecessor_graph_sha256':h.graph_sha256,
       'predecessor_acceptance_sha256':h.acceptance_head,
       'namespaces':deepcopy(f['graph']['namespaces'])+[deepcopy(f['new'])]}
    wire=canonical(d); fresh=sum(len(w.wire) for w in f['new_witnesses'])
    counters={'fresh_read_bytes':fresh,'decoded_witness_bytes':h.witness_bytes+fresh,
              'transport_bytes':len(wire)+h.witness_bytes+fresh}
    return d,counters


def reseal_authority(f):
    # Test-author-only creation of an alternative externally sealed scenario.
    f['aw']=canonical(f['authority']); f['seal']=physical('/external/root/authority.json',f['aw'],50000,True)
    f['anchor']=ExternalAnchor(canonical(f['seal']),f['authority']['generation_index'],
        f['authority']['predecessor_graph_sha256'],f['authority']['predecessor_acceptance_sha256'],
        digest(canonical(f['authority']['policy'])))


def reseal_graph(f):
    f['graph_wire']=canonical(f['graph']); f['authority']['predecessor_graph_sha256']=digest(f['graph_wire'])
    reseal_authority(f)


def phase_case(phase='live'):
    config=dict(CLOCK,source_manifest_sha256='8'*64)
    cw=canonical(config); cs=digest(cw)
    snapshot=canonical({'hosts':[{'state':'fixture'}], 'standalone_monitor':{
        'config_sha256':cs,'source_manifest_sha256':'8'*64,'attempt':1}})
    snapshotref={'path':'/original/normal/dry-snapshot.json' if phase=='qualification' else '/published/snapshot.json',**pin(snapshot)}
    raw=[]; previous=ZERO
    bodies=[('monitor.started',{'legacy_history_readonly':None}),('collection.attempt',{'attempt':1}),
            ('publication.attempt',{'attempt':1,'qualification_only':phase=='qualification'}),
            ('publication.verified',{'attempt':1,'snapshot':snapshotref,'errors':[]})]
    for i,(kind,body) in enumerate(bodies):
        e={'seq':i+1,'previous_sha256':previous,'kind':kind,'utc_ns':(4500+i)*10**9,
           'monotonic_ns':(4500+i)*10**9,'config_sha256':cs,'body':body}
        wire=canonical(e); raw.append(wire); previous=digest(wire)
    events=b'\n'.join(raw)+b'\n'
    result={'schema':'jope.paper-monitor-result/v1','state':'qualification_pass' if phase=='qualification' else 'window_closed_before_next_collection',
            'at':4504.0,'config_sha256':cs,'attempts':1,'last_verified_snapshot':snapshotref,
            'last_publication_custody':{'attempt':1,'phase':'verified','snapshot_may_have_been_replaced':True,'attempted_snapshot':snapshotref},
            'event_count':4,'event_head_sha256':previous,'optimizer_updates':0,'GP_executions':0,'S3_PUT':0,
            'old_windows_mutated':False,'retry_allowed':False}
    binding=PhaseBinding(cs,'8'*64,canonical(CLOCK),snapshotref['path'],b'null',((2,digest(b'\n'.join(raw[:2])+b'\n')),))
    return {'config':cw,'events':events,'result':result,'snapshot':snapshot,'binding':binding,'phase':phase}


def rechain(events):
    previous=ZERO; wires=[]
    for i,e in enumerate(events):
        e['seq']=i+1; e['previous_sha256']=previous; line=canonical(e); previous=digest(line); wires.append(line)
    return b'\n'.join(wires)+b'\n'
