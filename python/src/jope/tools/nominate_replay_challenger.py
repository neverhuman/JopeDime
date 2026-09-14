"""Nominate a replay arm using frozen training-family predicted-best fits."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import time

from direct_proposal_evaluation import evaluate
from promotion_gate import artifact,immutable
from qualification_budget import Budget
from qualification_slots import acquire
from qualify_regression_gain import gpu_handoff,ranks,ref,verify_trial_inputs
from qualify_pure_proposals import validate_ranking
from resume_expired_cycle import read,require,sha


def select(scores,challengers,baselines):
    """Failures stay in the panel; no available-case deletion or target tuning."""
    require(scores and len({len(rows) for rows in scores.values()})==1,'incomplete nomination panel')
    if any(row['status']!='ok' for name in baselines for row in scores[name]):
        return None,{},'a required training reference fit failed'
    means={}
    for name in challengers:
        if any(row['status']!='ok' for row in scores[name]):continue
        means[name]={base:statistics.fmean(row['rmse']/max(reference['rmse'],1e-12)
            for row,reference in zip(scores[name],scores[base])) for base in baselines}
    eligible=[name for name in means if all(value<1 for value in means[name].values())]
    winner=min(eligible,key=lambda name:(max(means[name].values()),name)) if eligible else None
    return winner,means,None if winner else 'no replay arm improved pure predicted-best RMSE against both training references'


def run(root,trial_path,output,budget_output):
    output.mkdir(parents=True,exist_ok=True);trial=read(trial_path);verify_trial_inputs(root,trial)
    trained=read(trial_path.parent/'training-policy-result.json')
    require(trained['trial_sha256']==sha(trial_path) and trained['final_tests_opened'] is False,'training trial changed')
    challengers=trained['eligible_challengers'];require(challengers,'no replay policy passed the frozen diagnostic guard')
    contexts=trial['training_contexts'];require(len(contexts)==12 and len({c['family'] for c in contexts})==12
        and all(c['partition']=='meta_train' for c in contexts),'nomination requires twelve frozen training families')
    arms={name:ref(root,root/trained['results'][name]['checkpoint']/'state.json') for name in challengers}
    arms.update(recovery=trial['recovery'],serving=trial['serving'])
    unique={};aliases={}
    for name,binding in arms.items():
        primary=unique.setdefault(binding['sha256'],name);aliases[name]=primary
    request=dict(schema='parallel.training_nomination.v1',trial=ref(root,trial_path),
        training_result=ref(root,trial_path.parent/'training-policy-result.json'),arms=arms,aliases=aliases,
        training_contexts=contexts,proposal_executable=trial['proposal_executable'],
        evaluator=trial['admission']['admission_policy'],proposal_ceiling_seconds=60,
        proposal_cpus_per_rank=1,proposal_cpu_ceiling=9,
        rule='minimum worst family-mean relative RMSE versus recovery and serving; strictly improve both; failed fits remain disqualifying; lexical arm tie-break',
        training_only=True,validation_results_used=False,frozen_diagnostic_guard_used=True,
        independent_evaluation=False,final_tests_opened=False,driver=ref(root,Path(__file__)))
    immutable(output/'request.json',request);budget=Budget(budget_output)
    vocabulary=root/trial['vocabulary']['path'];cases=[];case_names=[]
    with gpu_handoff(root,output/'gpu-handoff'):
        for name in unique.values():
            directory=output/name;directory.mkdir();checkpoint=(root/arms[name]['path']).parent
            config=read(checkpoint/'config.json')
            proposal=dict(schema='parallel.qualification_proposals.v1',checkpoint=str(checkpoint.relative_to(root)),
                output=str(directory.relative_to(root)),vocabulary=str(vocabulary.relative_to(root)),contexts=contexts,
                artifacts=[arms[name],trial['vocabulary'],trial['proposal_executable'],ref(root,output/'request.json')],final_tests_opened=False)
            path=directory/'request.json';immutable(path,proposal)
            key=output.name+'-'+name+'-proposals';budget.reserve(key,node_seconds=60,core_seconds=9*60)
            seconds=ranks(root,[[str(root/trial['proposal_executable']['path']),str(root),str(path),str(rank)]
                for rank in range(8)],directory/'logs',60,cpus_per_rank=1)
            for number,context in enumerate(contexts):
                p=directory/f'context-{number:03}';best=read(p/'pure-best-predicted.json');pool=read(p/'pure-ranking.json')
                validate_ranking(pool['ranking'],arms[name]['sha256'],context['conditioning_sha256'],sha(vocabulary),config['max_seq'])
                require(best['context']==context and best['manifest_sha256']==sha(path)
                    and best['pure_ranking_sha256']==sha(p/'pure-ranking.json') and best['prediction']==pool['ranking']['ranked'][0],
                    'pure predicted-best selection changed')
                require(best['teacher_population_consulted'] is False and best['gp_scores_used_for_prediction'] is False
                    and best['heldout_targets_used_for_prediction'] is False and best['final_tests_opened'] is False,
                    'pure proposal consulted forbidden outcomes')
                cases.append(dict(context=context,program=best['prediction']['program'],proposal=ref(root,p/'pure-best-predicted.json')))
                case_names.append(name)
            immutable(directory/'complete.json',dict(status='passed',request_sha256=sha(path),seconds=seconds,
                pure_proposals=[case['proposal'] for case,arm in zip(cases,case_names) if arm==name]))
            budget.complete(key,directory/'complete.json')
    manifest=output/'request.json'
    with acquire(root,manifest,output/'cpu-slots') as cpus:
        fits=evaluate(root,cases,root/request['evaluator']['evaluator'],vocabulary,output/'fits',cpus,budget,output.name+'-direct-fits',admission_policy=request['evaluator'])
    physical={name:[] for name in unique.values()}
    for name,evaluation in zip(case_names,fits['evaluations']):physical[name].append(artifact(root,evaluation['receipt']))
    scores={name:physical[aliases[name]] for name in arms}
    winner,means,reason=select(scores,challengers,['recovery','serving'])
    result=dict(schema='parallel.training_nomination_result.v1',status='nominated' if winner else 'rejected',
        request_sha256=sha(manifest),training_only=True,validation_results_used=False,final_tests_opened=False,
        winner=winner,checkpoint_sha256=arms[winner]['sha256'] if winner else None,
        checkpoint=str((root/arms[winner]['path']).parent.relative_to(root)) if winner else None,
        relative_rmse=means,reason=reason,fit_receipt=ref(root,output/'fits/complete.json'),
        selected_origins={name:dict(Counter(artifact(root,case['proposal'])['origin']
            for case,arm in zip(cases,case_names) if arm==aliases[name])) for name in arms},
        frozen_diagnostic_guard_used=True,independent_evaluation=False,
        failure_counts={name:sum(row['status']!='ok' for row in rows) for name,rows in scores.items()},
        timestamp=time.time(),serving_checkpoint_unchanged=True)
    immutable(output/'complete.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','trial','output','budget-output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.trial.resolve(),a.output.resolve(),a.budget_output.resolve())
