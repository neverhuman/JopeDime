"""Cold campaign gates. No remote access, launch, claims, or native fitting."""
import math
from pathlib import Path
from jope_flywheel.program_io import need, bound, sha
from jope_flywheel.protocol import digest

GLOBAL = 1791592146.380285
SCHEMA = 'jope.fresh-variant-gp-campaign/v1'


def validate(c, now):
    need(c['schema']==SCHEMA and c['host']=='xbabe3' and c['approved_for_execution'] is True,
         'cold packet is not approved for execution')
    need(c['global_deadline_unix']==GLOBAL and c['created_unix']<=now<c['work_deadline_unix']<c['deadline_unix']<=GLOBAL and
         c['deadline_unix']-c['created_unix']==1800 and c['deadline_unix']-c['work_deadline_unix']==300,
         'fixed1800s window with300s custody under original global')
    need(c['required_encoders']==['kumo_tabular_s'] and c['research_only'] is True and
         c['teaching_eligible'] is False and c['optimizer_reward_allowed'] is False,
         'research measurements preserve development roles')
    selection=c['selection'];descriptors=c['context_descriptors'];shards=c['shards']
    need(len(selection)==len({r['identity']for r in selection})==128 and
         len({r['parent_lineage_sha256']for r in selection})==128 and
         c['selection_sha256']==digest(selection), 'exact128 selected source')
    need(len(descriptors)==len({r['context_id']for r in descriptors})==128 and
         {r['variant_identity']for r in descriptors}=={r['identity']for r in selection}, 'native context qualification required')
    need(len(shards)==8 and len({s['id']for s in shards})==8 and
         len({s['owned_root']for s in shards})==8 and len({s['unit']for s in shards})==8,
         'eight disjoint whole-context shards')
    ids=[cid for s in shards for cid in s['context_ids']]
    need(len(ids)==len(set(ids))==128 and set(ids)=={r['context_id']for r in descriptors},'complete nonoverlapping shard union')
    for s in shards:
        need(len(s['context_ids'])==16 and s['planned_slots']==128 and s['max_put_attempts']==1026 and
             all(c['context_owners'][v]==s['id']for v in s['context_ids']), 'ordinary128-evaluation ceiling')
    need(c['initial_concurrency']==1 and c['maximum_qualified_concurrency']<=8 and
         c['resources']=={'CPUQuota':'100%','MemoryMax':4*1024**3,'MemorySwapMax':0,'TasksMax':64,'Nice':10,
                           'CUDA_VISIBLE_DEVICES':'','BLAS_THREADS':1}, 'CPU1/RAM4GiB initial worker envelope')
    need(len(c['programs'])==len({p['program_sha256']for p in c['programs']})==8 and
         all(sha(p['dsl'].encode())==p['program_sha256']for p in c['programs']), 'pinned eight fixed programs')
    need(c['source_build_plan_sha256']=='4bb10409dde5a686da91fb96501113a0e5dd1efcde6429cab1024d4270302632',
         'prepared native protocol identity')
    need(c['source_pins'] and len(c['history_sources'])>=20 and
         c['ownership_producer_sha256'] and c['cold_qualification'] is not None,
         'live history/ownership/source qualification missing')
    for ref in c['source_pins']:bound(ref,64*1024**2)
    return True


def resource_gate(mem_available, owned_reservations, proposed_workers=1):
    """Charge authorized residual reservations, not just current RSS.

    Caller must pin fresh cgroup/source observations, including encoder12GiB
    and any active scorer/learner reservations. Missing observations are not0.
    """
    need(type(mem_available)is int and mem_available>=0 and type(proposed_workers)is int and
         1<=proposed_workers<=8, 'finite observed host memory and bounded worker count')
    residual=0
    for row in owned_reservations:
        need(type(row['memory_max'])is int and type(row['memory_current'])is int and
             row['memory_max']>0 and row['memory_current']>=0 and row['observed'] is True,
             'all authorized owned reservations must be measured')
        residual+=max(0,row['memory_max']-row['memory_current'])
    required=24*1024**3+residual+proposed_workers*4*1024**3
    need(mem_available>=required,'24GiB foreign reserve plus residual owned allocations and new4GiB workers')
    return {'required_available_bytes':required,'residual_owned_bytes':residual,
            'new_workers':proposed_workers,'new_worker_bytes':proposed_workers*4*1024**3}
