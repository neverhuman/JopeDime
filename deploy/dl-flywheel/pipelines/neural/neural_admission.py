"""Explicit bounded research-optimizer permit, separate from production admission."""
import time
from dltrain import objective
from jope_flywheel.protocol import digest
from jope_flywheel.program_io import need,bound
from jope_flywheel.program_inputs import current_roles

KEYS={'schema','approved_for_bounded_research','snapshot_sha256','train_data_content_sha256','all_membership_sha256',
      'roster_audit_sha256','objective_sha256','role_support_sha256','role_authority_pins_sha256','context_roles',
      'production_admission','release_backpressure','generated_outcomes_as_optimizer_reward','scope'}

def validate(permit,loaded,authority):
    need(type(permit)is dict and set(permit)==KEYS,'closed explicit optimizer admission permit')
    need(permit['schema']=='jope.flywheel-research-optimizer-admission/v1'and permit['approved_for_bounded_research']is True,'research optimizer approval required')
    need(all(permit[k]is False for k in('production_admission','release_backpressure','generated_outcomes_as_optimizer_reward')),'research permit scope cannot widen')
    need(permit['scope']=='One immutable KumoS-only bounded research PolicyLM qualification; original publication flags remain unchanged.', 'exact research-only scope')
    receipt=loaded.receipt
    for key in('snapshot_sha256','train_data_content_sha256','all_membership_sha256','roster_audit_sha256'):
        need(permit[key]==receipt[key],'permit exact admitted snapshot/corpus/roster identity')
    need(permit['objective_sha256']==objective.objective_sha256('successor-v1'),'unchanged native objective')
    roles={cid:{'variant_identity':d['variant_identity'],'parent_lineage_sha256':d['parent_lineage_sha256'],
                 'partition':d['partition'],'split_role':d['split_role']}for cid,d in receipt['contexts'].items()}
    need(permit['context_roles']==roles,'permit exact context/lineage/role membership')
    need(permit['role_support_sha256']==digest(authority['support'])and permit['role_authority_pins_sha256']==digest(authority['authority_pins']), 'permit current role closure')
    need(authority['state']=='passed'and 0<=time.time()-authority['checked_at_unix']<120,'fresh research role authority')
    return {'schema':'jope.flywheel-research-optimizer-permit-verification/v1','permit_sha256':digest(permit),
            'snapshot_sha256':receipt['snapshot_sha256'],'approved_for_bounded_research':True,'production_admission':False,
            'release_backpressure':False,'generated_outcomes_as_optimizer_reward':False,'verified_at':time.time()}

def recheck_roles(inputs,permit):
    """Every optimizer step rechecks immutable source pins and latest local roles."""
    generator=inputs.generator
    for ref in generator['source_pins']:bound(ref)
    bound(inputs.config['generator_config']);bound(generator['parent_registry'])
    eligible,cids,views=current_roles(generator)
    for row in permit['context_roles'].values():
        parent=inputs.parents[row['parent_lineage_sha256']]
        need(parent['partition']==row['partition']and parent['view_id']in eligible and parent['view_id']not in views and
             not(set(parent['retained_lineage_context_sha256'])&cids),'research optimizer current role changed')
    return {'permit_sha256':digest(permit),'checked_at':time.time(),'contexts_checked':len(permit['context_roles'])}
