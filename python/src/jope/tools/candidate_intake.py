"""Validate an externally qualified replay candidate before controller intake."""
from pathlib import Path

from promotion_gate import artifact
from resume_expired_cycle import require


def validate(root,submission,vocabulary_sha):
    require(submission['schema']=='parallel.candidate_submission.v1','unknown candidate submission')
    trial=artifact(root,submission['trial']);training=artifact(root,submission['training'])
    nomination=artifact(root,submission['nomination']);state=artifact(root,submission['checkpoint'])
    nomination_request=artifact(root,submission['nomination_request'])
    training_result=artifact(root,submission['training_result'])
    run=artifact(root,submission['run'])
    corpus=artifact(root,submission['corpus']);arm=submission['arm']
    require(trial['schema']=='parallel.regression_gain_trial.v1' and trial['final_tests_opened'] is False,
            'candidate trial is not the frozen replay qualification')
    require(nomination_request['trial']==submission['trial']
            and nomination_request['training_result']==submission['training_result']
            and nomination['request_sha256']==submission['nomination_request']['sha256']
            and nomination_request['arms'][arm]==submission['checkpoint'],
            'nomination is from another trial or candidate')
    require(training_result['trial_sha256']==submission['trial']['sha256']
            and arm in training_result['eligible_challengers']
            and training_result['results'][arm]['checkpoint_sha256']==submission['checkpoint']['sha256']
            and training_result['results'][arm]['diagnostic_pass'] is True,
            'candidate lacks the frozen diagnostic guard')
    require(training['status']=='trained' and training['checkpoint_sha256']==submission['checkpoint']['sha256']
            and training['checkpoint']==str(Path(submission['checkpoint']['path']).parent),'candidate training identity changed')
    require(nomination['status']=='nominated' and nomination['winner']==arm
            and nomination['checkpoint_sha256']==submission['checkpoint']['sha256']
            and nomination['training_only'] is True and nomination['validation_results_used'] is False
            and nomination['final_tests_opened'] is False,'candidate lacks training-only direct nomination')
    require(submission['corpus']==trial['corpus_index'] and state['corpus_sha256']==submission['corpus']['sha256']
            and state['vocabulary_sha256']==vocabulary_sha and corpus['label_policy']=='strict_midrank_v4',
            'candidate corpus/vocabulary/admission changed')
    policy_ref=trial['policies'][arm];policy=artifact(root,policy_ref)
    require(run['trial_sha256']==submission['trial']['sha256'] and run['policy']==policy_ref
            and run['corpus']==submission['corpus'] and run['executable']==trial['training_executable']
            and run['steps']==trial['steps_per_variant']==1000,
            'candidate training run is from another trial, policy or executable')
    ranks=[artifact(root,ref) for ref in training['rank_receipts']]
    require(len(ranks)==8 and {row['rank'] for row in ranks}==set(range(8))
            and all(row['world']==8 and row['local_steps']==trial['steps_per_variant']
            and row['corpus_sha256']==submission['corpus']['sha256']
            and row['steps']==state['step'] and row['parameter_sha256']==state['parameter_sha256']
            and row['replay_policy_sha256']==policy_ref['sha256'] for row in ranks)
            and len({row['parameter_sha256'] for row in ranks})==1,'candidate rank/policy completion mismatch')
    if policy['fresh_fraction']:
        require(sum(row['actual_fresh_comparisons_this_rank_attempt'] for row in ranks)>0
            and all(row['actual_fresh_programs_this_rank_attempt']>=row['actual_fresh_comparisons_this_rank_attempt']
                    for row in ranks),'candidate never consumed fresh programs')
    return dict(checkpoint=training['checkpoint'],checkpoint_sha256=submission['checkpoint']['sha256'],
        corpus=str(Path(submission['corpus']['path']).parent),corpus_sha256=submission['corpus']['sha256'],
        parent_checkpoint=str(Path(trial['recovery']['path']).parent),parent_checkpoint_sha256=trial['recovery']['sha256'],
        training_receipt=submission['training']['path'],training_receipt_sha256=submission['training']['sha256'],
        vocabulary_sha256=vocabulary_sha,proposal_executable_sha256=trial['proposal_executable']['sha256'],
        qualification_status='inconclusive',reason='awaiting identity-valid matched qualification',
        nomination=submission['nomination'],trial=submission['trial'])
