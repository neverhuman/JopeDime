"""Freeze one new whole-context scoring epoch after all cold inputs qualify."""
import json,time
from pathlib import Path
from common import *
from history_collect import validate_collections

BROKER_ROOT=Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-fresh-context-broker-20261009-v1')
DISPATCH_SHA='e6f37b7c8d8998a0c78128726b129e284f974fe4690e3304892b74083197a998'

def descriptors(prepared):
    rows=obj(prepared['contexts'])
    need(len(rows)==len({r['context_id']for r in rows})==128,'full typed context closure')
    return [{k:r[k]for k in('context_id','variant_identity','descriptor_sha256','partition')}for r in rows]

def handoff(reference,prepared,history_refs):
    h=obj(reference)
    need(h['schema']=='jope.fresh128-variant-writer-fence/v2'and h['candidate_inventory_sha256']==
         obj(prepared['request'])['candidate_inventory']['sha256']and h['global_deadline_unix']==GLOBAL,
         'explicit root cross-host fencing handoff')
    need(h['foreign_variant_writers_released']is True and h['all_future_variant_launchers_require_broker']is True and
         h['broker_root']==str(BROKER_ROOT)and h['broker_lock']==str(BROKER_ROOT/'epoch.lock')and
         h['history_collection_sha256']==[r['sha256']for r in history_refs] and
         0<=time.time()-h['authorized_at_unix']<=600,'current complete launch-coordination fence required')
    need(type(h['root_authorization'])is str and len(h['root_authorization'])>=20,'explicit authorization evidence')
    for writer in h['compatible_native_writers']:
        need(writer['dispatch_source']['sha256']==DISPATCH_SHA,'only reviewed native-only dispatcher may coexist')
        text=bound(writer['dispatch_source']).decode();c=obj(writer['config'])
        need("get('source_kind')=='native_s1_context'"in text or "source_kind')='native_s1_context'"in text,
             'exact pinned native predicate')
        need(c['outbox_path']==writer['outbox_path']and c['outbox_config']==writer['outbox_config'],'native worker original authority')
    return h

def freeze(request_ref,out):
    host('xbabe2');r=obj(request_ref);p=obj(r['prepared']);source_closure(p['source_manifest'])
    need(r['schema']=='jope.fresh128-epoch-request/v2'and type(r['authorized_concurrency'])is int and 1<=r['authorized_concurrency']<=16,'bounded CPU1 worker concurrency')
    aliases,native_ids,units,inventory_refs=validate_collections(r['history_collections'],r['required_outboxes'],600)
    pins={Path(x['path']).name:x['sha256']for x in obj(p['source_manifest'])['files']}
    need(r['broker_producer_sha256']==pins['broker.py']and r['physical_producer_sha256']==pins['transport.py'],'exact ownership and physical producer source')
    h=handoff(r['fencing_handoff'],p,r['history_collections']);ds=descriptors(p);rows=obj(p['contexts'])
    need(not({d['context_id']for d in ds}&native_ids),'prepared/native context overlap')
    # No silent rerun or changed denominator: a held canonical key stops the
    # fresh128 epoch before ownership or any worker queue can be created.
    for d in ds:
        need((d['context_id'],None)not in aliases,'unresolved canonical historical context held')
        for program in p['programs']:
            need((d['context_id'],program['program_sha256'])not in aliases,'historical canonical key is held; select another cold scope')
    root=fresh(out);created=time.time();need(created+1800<=GLOBAL,'full new1800s fits original global')
    shards=[];owners={}
    for index in range(16):
        selected=ds[8*index:8*(index+1)];sid='shard%02d'%index
        shards.append({'id':sid,'context_ids':[v['context_id']for v in selected],
                       'variant_ids':[v['variant_identity']for v in selected],
                       'planned_slots':64,'max_put_attempts':514,'owned_root':str(root/sid),
                       'unit':'jope-fresh128-'+digest(r)[:12]+'-'+sid+'.service'})
        owners.update({v['context_id']:sid for v in selected})
    c={'schema':'jope.fresh128-epoch/v2','prepared':r['prepared'],'fencing_handoff':r['fencing_handoff'],
       'host':'xbabe3','source_manifest':p['source_manifest'],'history_sources':r['history_collections'],
       'required_outboxes':r['required_outboxes'],'history_inventories':inventory_refs,
       'legacy_units':units,'context_descriptors':ds,'context_owners':owners,'shards':shards,
       'shard_ids':[s['id']for s in shards],'selection':p['selection'],'selection_sha256':p['selection_sha256'],
       'created_unix':created,'work_deadline_unix':created+1500,'deadline_unix':created+1800,'global_deadline_unix':GLOBAL,
       'broker_root':str(BROKER_ROOT),'broker_lock':str(BROKER_ROOT/'epoch.lock'),
       'broker_producer_sha256':r['broker_producer_sha256'],'physical_producer_sha256':r['physical_producer_sha256'],'relay_root':str(root/'relay'),
       'broker_out':str(root/'broker'),'physical_pointer':str(root/'physical/current.json'),
       'epoch_root':str(root),'authorized_concurrency':r['authorized_concurrency'],'resource_reservations':r['resource_reservations'],'maximum_actual_evaluations_per_shard':64,
       'resources':{'CPUQuota':'100%','MemoryMax':4*1024**3,'MemorySwapMax':0,'TasksMax':64,'Nice':10,'CUDA_VISIBLE_DEVICES':'','BLAS_THREADS':1},
       'native_context_overlap':[],'teaching_eligible':False,'optimizer_reward_allowed':False}
    return put(root,'epoch.json',c)

def execution_config(epoch_ref,shard=None):
    e=obj(epoch_ref);p=obj(e['prepared']);c={**p['config_base']}
    c.update({k:e[k]for k in('context_descriptors','context_owners','shard_ids','history_sources','work_deadline_unix','global_deadline_unix','selection','selection_sha256')})
    c.update(schema='jope.fresh-variant-gp-campaign/v1',campaign_sha256=epoch_ref['sha256'],
        ownership_producer_sha256=e['broker_producer_sha256'],authority_deadline_unix=e['work_deadline_unix'],
        authority_root=str(Path(e['relay_root'])/'roles'),authority_pointer=str(Path(e['relay_root'])/'roles/current.json'),
        registry=p['registry'],source_pins=obj(e['source_manifest'])['files'],programs=p['programs'],
        required_encoders=['kumo_tabular_s'],research_only=True,teaching_eligible=False,resources=e['resources'])
    if shard is not None:
        s=next((x for x in e['shards']if x['id']==shard),None);need(s is not None,'declared shard')
        c.update(owned_root=s['owned_root'],maximum_actual_evaluations=64)
    return c
