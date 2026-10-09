"""Root-host proof relay; fixed read-only SSH producers, owned proof installs.

Run before broker with --physical-only, then continuously after broker starts.
No model/data files, GP commands, arbitrary destination paths, or S3 writes.
"""
import json,shlex,time
from pathlib import Path
from common import *
from history_collect import closed_unit,closed_controller,command

def probe(spec_ref,producer_sha):
    inv=obj(spec_ref);host(inv['host']);clock=Clock(min(time.time()+60,GLOBAL),60)
    return {'schema':'jope.fresh128-physical-probe/v2','inventory':spec_ref,'host':inv['host'],
            'producer_sha256':producer_sha,'units':[closed_unit(u,clock)for u in inv['legacy_units']],
            'controllers':[closed_controller(r)for r in inv['legacy_controllers']], 'checked_at_unix':time.time()}

def install_physical(epoch_ref,bundle):
    host('xbabe2');e=obj(epoch_ref)
    need(bundle['schema']=='jope.fresh128-physical-bundle/v2'and bundle['epoch_sha256']==epoch_ref['sha256'],'same epoch physical bundle')
    checked=max([bundle['checked_at_unix']]+[p['checked_at_unix']for p in bundle['probes']]);now=time.time()
    need(checked-now<=1,'physical proof clock skew exceeds qualified wait')
    if checked+.1>now:time.sleep(checked+.1-now)
    need(time.time()<e['work_deadline_unix'],'physical proof cannot extend epoch')
    root=Path(e['physical_pointer']).parent;root.mkdir(exist_ok=True);(root/'proofs').mkdir(exist_ok=True)
    raw=canonical(bundle);r=create(root/'proofs'/(digest(bundle)+'.json'),raw);atom(root/'current.json',canonical(r));return r

def export_proof(epoch_ref):
    host('xbabe2');e=obj(epoch_ref);r=json.loads(read(Path(e['broker_out'])/'current.json',4096))
    p=Path(r['path']);need(p.parent==Path(e['broker_out'])/'proofs'and p.name==r['sha256']+'.json','owned broker proof')
    return obj(r,4*MIB)

def install_proof(epoch_ref,bundle):
    host('xbabe3');e=obj(epoch_ref);need(bundle['schema']=='jope.fresh128-broker-proof-bundle/v2'and bundle['epoch_sha256']==epoch_ref['sha256'],'exact proof bundle epoch')
    now=time.time();checked=max(bundle['ownership']['checked_at_unix'],bundle['roles']['checked_at_unix'])
    need(checked-now<=1,'cross-host clock exceeds qualified wait')
    if checked+.1>now:time.sleep(checked+.1-now)
    need(time.time()<min(bundle['ownership']['lease_expires_unix'],e['work_deadline_unix']),'live proof within fixed window')
    refs={}
    for kind,body in [('ownership',bundle['ownership']),('roles',bundle['roles'])]:
        root=Path(e['relay_root'])/kind;(root/'proofs').mkdir(parents=True,exist_ok=True)
        p=root/'proofs'/(digest(body)+'.json')
        if p.exists():need(read(p)==canonical(body),'immutable proof bytes')
        else:create(p,canonical(body))
        refs[kind]=ref(p);atom(root/'current.json',canonical(refs[kind]))
    return refs

def relay(request_ref,*,once=False,physical_only=False):
    r=obj(request_ref);need(r['schema']=='jope.fresh128-proof-relay/v2','relay request')
    e=obj(r['epoch']);clock=Clock(e['work_deadline_unix'],1500)
    need(set(r['hosts'])>={'xbabe2','xbabe3'}and set(r['hosts'])<={'xbabe1','xbabe2','xbabe3'},'explicit remote host set')
    def call(hostname,action,extra,body=None):
        remote=r['hosts'][hostname]
        need(remote['python'].startswith('/')and remote['cli'].startswith(ROOT_PREFIX),'owned pinned remote source invocation')
        argv=[remote['python'],remote['cli'],'--source-manifest',remote['source_manifest']['path'],'--source-sha',remote['source_manifest']['sha256'],action]+extra
        raw_body=None
        if body is not None:
            raw_body=canonical(body);need(len(raw_body)<=1024*1024,'bounded proof transport')
            argv+=['--body-stdin']
        raw=command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',hostname,shlex.join(argv)],clock,input_body=raw_body)
        return json.loads(raw)
    while clock.remaining()>0:
        probes=[]
        for invref in e['history_inventories']:
            inv=obj(invref);probes.append(call(inv['host'],'probe',['--ref',invref['path'],'--sha',invref['sha256']]))
        bundle={'schema':'jope.fresh128-physical-bundle/v2','epoch_sha256':r['epoch']['sha256'],
                'checked_at_unix':time.time(),'probes':probes}
        args=['--ref',r['epoch']['path'],'--sha',r['epoch']['sha256']]
        call('xbabe2','install-physical',args,bundle)
        if not physical_only:
            proof=call('xbabe2','export-proof',args);call('xbabe3','install-proof',args,proof)
        if once or physical_only:break
        time.sleep(min(15,max(0,clock.remaining())))
    return {'state':'relay_closed','epoch_sha256':r['epoch']['sha256'],'clock_not_extended':True}
