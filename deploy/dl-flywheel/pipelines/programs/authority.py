"""Fresh authoritative xbabe2 role proof for the fixed112 campaign; no mutations."""
import argparse,json,socket,time,sys
from pathlib import Path
sys.dont_write_bytecode=True

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--sha',required=True);v=a.parse_args()
    from jope_flywheel.program_io import bound,need,tick
    from jope_flywheel.program_inputs import current_roles
    from jope_flywheel.protocol import canonical,digest
    c=json.loads(bound({'path':v.config,'sha256':v.sha}))
    need(socket.gethostname().split('.')[0]=='xbabe2','authoritative host required')
    tick(c['deadline_unix'])
    generator=json.loads(bound(c['generator_config']));parents=[json.loads(x)for x in bound(generator['parent_registry']).splitlines()]
    by={x['parent_lineage_sha256']:x for x in parents};need(len(by)==len(parents)==1068,'full authoritative parent registry')
    need(generator['parent_registry']['sha256']==c['parent_registry_sha256'] and generator['authority_pins']==c['authority_pins'],'original authority identity')
    for ref in generator['source_pins']:bound(ref)
    eligible,cids,views=current_roles(generator);support=[]
    need(len(c['selection'])==len({v['identity']for v in c['selection']})==112,'exact112 role selection')
    need(digest(c['selection'])==c['selection_sha256'],'selection immutable')
    for v in c['selection']:
        p=by[v['parent_lineage_sha256']]
        need(p['view_id']in eligible and p['view_id']not in views and not(set(p['retained_lineage_context_sha256'])&cids),'new protected role excludes parent')
        need(p['partition']==v['partition']=='train','inherited training partition')
        support.append({'identity':v['identity'],'parent_lineage_sha256':v['parent_lineage_sha256'],
            'parent_record_sha256':digest(p),'view_id':p['view_id'],'partition':p['partition'],'protected_overlap':0})
    tick(c['deadline_unix'])
    proof={'schema':'jope.flywheel-program112-authority/v1','passed':True,'host':'xbabe2','campaign_sha256':c['campaign_sha256'],
        'selection_sha256':c['selection_sha256'],'generator_config_sha256':c['generator_config']['sha256'],
        'parent_registry_sha256':c['parent_registry_sha256'],'authority_pins':c['authority_pins'],
        'checked_at_unix':time.time(),'deadline_unix':c['deadline_unix'],'support':support}
    sys.stdout.buffer.write(canonical(proof))
if __name__=='__main__':main()
