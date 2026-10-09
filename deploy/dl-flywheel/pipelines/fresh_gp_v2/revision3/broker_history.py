"""Cold permanent-owner inventory plus fresh prior worker queues and outcomes."""
import json,sqlite3,time
from pathlib import Path
from common import *
from permanent_broker import Registry
from history_collect import snapshot,closed_unit
from global_attempts import Attempts
from jope_flywheel.program_io import NativeTools

def inventory(request_ref,out,*,hostname=None,fixture_root=None):
    if hostname is None:host('xbabe2')
    else:need(hostname=='xbabe2','permanent inventory authority host')
    request=obj(request_ref);need(request['schema']=='jope.permanent-broker-inventory-request/v3','explicit latest externally retained broker head required')
    bootstrap_ref=request['bootstrap'];head_anchor=request['known_head']
    # The next epoch must use an independently retained exact prior seal. A
    # mutable local commits directory is not its own rollback authority.
    broker_root=Path(obj(bootstrap_ref)['root']);anchor_path=Path(head_anchor['path'])
    need(anchor_path.is_absolute()and not anchor_path.is_relative_to(broker_root),'externally retained sealed head required')
    reg=Registry(bootstrap_ref,head_anchor,readonly=True,fixture_root=fixture_root)
    try:
        reg.db.execute('BEGIN');head=reg.verify();epochs=[json.loads(r['body'])for r in reg.db.execute('SELECT body FROM epochs ORDER BY sha')]
        rows=[dict(r)for r in reg.db.execute('SELECT *FROM contexts ORDER BY context_id')];reg.db.execute('ROLLBACK')
        return put(Path(out).parent,Path(out).name,{'schema':'jope.permanent-broker-inventory/v3','bootstrap':bootstrap_ref,
          'producer_sha256':ref(__file__)['sha256'],'head_anchor':head_anchor,'registry_head':head,'epochs':epochs,'contexts':rows,
          'checked_at_unix':time.time(),'all_prior_contexts_permanently_held':True})
    finally:reg.close()

def attempt_snapshot(path,clock):
    # Immutable read transaction copied to private memory; no source constructor.
    from readonly_snapshot import source_path
    p=source_path(path);source=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,isolation_level=None,timeout=.5)
    db=sqlite3.connect(':memory:',isolation_level=None);db.row_factory=sqlite3.Row
    try:
        source.execute('PRAGMA query_only=ON');source.execute('BEGIN');source.execute('SELECT *FROM binding').fetchone()
        need(source.execute('PRAGMA page_count').fetchone()[0]*source.execute('PRAGMA page_size').fetchone()[0]<=64*MIB,'bounded attempt registry')
        source.backup(db,pages=256,progress=lambda *_:clock.tick(),sleep=.01);checker=Attempts.__new__(Attempts);checker.db=db
        proof=checker.verify();rows={n:[dict(r)for r in db.execute('SELECT *FROM '+n+' ORDER BY 1')]for n in('binding','contexts','attempts')}
        clock.tick();return proof,rows
    finally:source.close();db.close()

def workers(request_ref,out,*,hostname=None,native_factory=NativeTools,unit_reader=closed_unit):
    if hostname is None:host('xbabe3')
    else:need(hostname=='xbabe3','prior worker history host')
    request=obj(request_ref);need(request['schema']=='jope.prior-broker-worker-history-request/v3','worker census request')
    inv=obj(request['inventory']);need(inv['schema']=='jope.permanent-broker-inventory/v3','permanent census source')
    clock=Clock(min(time.time()+600,GLOBAL),600);root=fresh(out);native=native_factory(request['canonicalizer'],clock.deadline)
    docs=[];closed=[]
    for ep in inv['epochs']:
        clock.tick();before=[unit_reader(s['unit'],clock)for s in ep['shards']]
        binding_path=Path(ep['epoch_root'])/'whole-context-registry-binding.json'
        binding=json.loads(read(binding_path,8*MIB));need(binding['epoch_sha256']==digest(ep)and digest(binding['config'])==binding['config_sha256'],'prior actual host registry binding')
        proof,state=attempt_snapshot(Path(ep['epoch_root'])/'whole-context-attempts.sqlite',clock)
        need(len(state['binding'])==1 and state['binding'][0]['sha']==binding['config_sha256'],'prior host registry config')
        known={r['context_id']:r for r in ep['context_descriptors']}
        for r in state['contexts']:
            need(r['context_id']in known and json.loads(r['descriptor'])==known[r['context_id']]and r['owner']==ep['context_owners'][r['context_id']],'prior owned context changed')
        for shard in ep['shards']:
            owned=[r for r in state['contexts']if r['owner']==shard['id']];p=Path(shard['owned_root']);q=p/'data/queue.sqlite'
            need(len(owned)in(0,8),'partial old whole-context reservation')
            if owned:
                # A prior started shard cannot become apparently fresh by
                # deleting its queue. Refuse; held contexts remain permanent.
                need(q.is_file(),'prior reserved worker queue missing; manual custody reconciliation required')
                d=snapshot({'kind':'queue','path':str(q)},native,clock)
                need({a['context_id']for a in d['aliases']}<=set(shard['context_ids']),'prior queue context escaped frozen shard')
                d['outcomes']={};d['outcome_sources']={}
                for name in('status.json','progress.json','label-results.json','early-failure.json'):
                    if(p/name).is_file():
                        original=ref(p/name);raw=bound(original,8*MIB)
                        d['outcome_sources'][name]=original
                        d['outcomes'][name]=create(root/('outcome-%04d-'%len(docs)+name),raw)
                d.update(epoch_sha256=digest(ep),shard=shard['id'],attempt_registry=proof,execution_state='reserved_all_outcomes_held')
            else:
                need(not q.exists(),'prior queue exists without permanent host reservation')
                need(not p.exists(),'prior unreserved shard has unresolved output; manual custody required')
                d={'epoch_sha256':digest(ep),'shard':shard['id'],'execution_state':'no_host_reservation_or_output',
                   'contexts_still_permanently_held':True,'attempt_registry':proof,'aliases':[],'outcomes':{}}
            docs.append(put(root,'worker-%04d.json'%len(docs),d))
        after=[unit_reader(s['unit'],clock)for s in ep['shards']];closed.extend(after)
        need([v['unit']for v in before]==[v['unit']for v in after],'exact prior closure')
    return put(root,'workers.json',{'schema':'jope.prior-broker-worker-history/v3','inventory':request['inventory'],
        'producer_sha256':ref(__file__)['sha256'],'host':'xbabe3','sources':docs,'closed_units':closed,
        'checked_at_unix':time.time(),'all_prior_shards_accounted':True,'source_SQL_mutations':0})

def validate(reference,bootstrap_ref,*,live=False,fixture_root=None):
    h=obj(reference);need(h['schema']=='jope.permanent-broker-history/v3'and h['bootstrap']==bootstrap_ref,'permanent history identity')
    inv=obj(h['inventory']);w=obj(h['workers'])
    need(inv['schema']=='jope.permanent-broker-inventory/v3'and inv['bootstrap']==bootstrap_ref and inv['all_prior_contexts_permanently_held']is True,'permanent history inventory')
    need(w['schema']=='jope.prior-broker-worker-history/v3'and w['inventory']==h['inventory']and w['host']=='xbabe3'and
         w['all_prior_shards_accounted']is True and w['source_SQL_mutations']==0,'all prior worker history')
    expected={(digest(e),s['id'])for e in inv['epochs']for s in e['shards']};docs=[obj(r)for r in w['sources']]
    need({(d['epoch_sha256'],d['shard'])for d in docs}==expected and len(docs)==len(expected),'no prior shard omitted')
    need(len(w['closed_units'])==len(expected)and all(u['systemd']['MainPID']=='0'and u['systemd']['ActiveState']in('inactive','failed')and
         all(g['exists']is False or(g['pids']==[]and g['populated']==0)for g in u['cgroups'])for u in w['closed_units']),'prior writer physical closure')
    aliases={(c['context_id'],None):[{'source':'permanent-broker','epoch_sha256':c['epoch'],'disposition':'whole_context_hold'}]for c in inv['contexts']}
    for d in docs:
        if d['execution_state']=='reserved_all_outcomes_held':
            need(d['full_source_verified']is True and 'head_sha256'in d['verifier'],'prior full queue verifier')
            need(set(d['outcomes'])==set(d['outcome_sources']),'all prior outcome snapshots')
            for name,r in d['outcomes'].items():
                bound(r,8*MIB);need(all(r[k]==d['outcome_sources'][name][k]for k in('sha256','bytes')),'prior exact outcome bytes')
            for a in d['aliases']:aliases.setdefault((a['context_id'],a['canonical_program_sha256']),[]).append(a)
        else:need(d['execution_state']=='no_host_reservation_or_output'and d['contexts_still_permanently_held']is True,'unknown prior history state')
    if live:
        need(all(0<=time.time()-d['checked_at_unix']<600 for d in(inv,w)),'fresh permanent history required')
        reg=Registry(bootstrap_ref,inv['head_anchor'],readonly=True,fixture_root=fixture_root)
        try:need(reg.verify()==inv['registry_head'],'permanent broker advanced or rewound since capture')
        finally:reg.close()
    return inv,aliases

def combine(request_ref,out):
    host('xbabe2');r=obj(request_ref);need(r['schema']=='jope.permanent-broker-history-join/v3','history join request')
    h={'schema':'jope.permanent-broker-history/v3','bootstrap':r['bootstrap'],'inventory':r['inventory'],'workers':r['workers'],'created_unix':time.time()}
    result=put(Path(out).parent,Path(out).name,h);validate(result,r['bootstrap'],live=True);return result
