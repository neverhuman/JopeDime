"""Fresh original xbabe2 role authority for all128 selected descendants."""
import json,time
from common import *
from jope_flywheel.program_inputs import current_roles

def produce(c):
    generator=obj(c['generator_config']);parents=[json.loads(x)for x in bound(generator['parent_registry']).splitlines()]
    by={x['parent_lineage_sha256']:x for x in parents}
    need(len(by)==len(parents)==1068 and generator['parent_registry']['sha256']==c['parent_registry_sha256'] and
         generator['authority_pins']==c['authority_pins'],'full pinned parent authority')
    for r in generator['source_pins']:bound(r)
    eligible,cids,views=current_roles(generator);selection=c['selection']
    need(len(selection)==len({x['identity']for x in selection})==128 and digest(selection)==c['selection_sha256'],'exact selected128')
    support=[]
    for v in selection:
        p=by[v['parent_lineage_sha256']]
        need(p['view_id']in eligible and p['view_id']not in views and not(set(p['retained_lineage_context_sha256'])&cids), 'current protected role excludes parent')
        need(p['partition']==v['partition']and p['partition']in('train','validation'),'inherited partition')
        support.append({'identity':v['identity'],'parent_lineage_sha256':v['parent_lineage_sha256'],
                        'parent_record_sha256':digest(p),'view_id':p['view_id'],'partition':p['partition'],'protected_overlap':0})
    need(time.time()<c['authority_deadline_unix']<=GLOBAL,'authority work clock')
    return {'schema':'jope.fresh-variant-gp-authority/v1','passed':True,'host':'xbabe2',
            'campaign_sha256':c['campaign_sha256'],'selection_sha256':c['selection_sha256'],
            'generator_config_sha256':c['generator_config']['sha256'],'parent_registry_sha256':c['parent_registry_sha256'],
            'authority_pins':c['authority_pins'],'checked_at_unix':time.time(),
            'deadline_unix':c['authority_deadline_unix'],'support':support}
