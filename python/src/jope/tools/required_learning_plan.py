"""Prepare the recovered support/replay study; perform no training or outcome reads."""
import argparse
import copy
from pathlib import Path

from original_learning_selection import custody_json, recover_selection, require
from scientific_metadata import canonical, digest, immutable, inside, sha256_value


def unsigned(value):
    return type(value) is int and 0 <= value < 2**64


def metadata_reference(root, reference):
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'}
            and sha256_value(reference['sha256']), 'exact metadata reference required')
    path = inside(root, reference['path'])
    require(path.suffix == '.json'
            and not {'data', 'raw', 'sealed'} & {part.lower() for part in path.parts},
            'raw, sealed or nonmetadata reference rejected before open')
    value, observed = custody_json(path, 16 * 1024 * 1024)
    require(observed == reference['sha256'], 'required study metadata changed')
    return value


def support_metadata(root, specification, parent_sha, vocabulary_sha):
    require(isinstance(specification, dict) and set(specification) == {'corpus_index', 'replay_policy'},
            'corpus index and original frozen replay policy required')
    index = metadata_reference(root, specification['corpus_index'])
    require(index.get('schema') == 'parallel.teaching_shards.v1'
            and index.get('label_policy') == 'strict_midrank_v4'
            and index.get('vocabulary_sha256') == vocabulary_sha
            and sha256_value(index.get('context_index_sha256')), 'strict support index required')
    admission = index.get('admission_policy', {})
    require(admission.get('schema') == 'parallel.label_admission.v2'
            and admission.get('evaluator_protocol') == 'featmath_training_fold_normalized_v1'
            and all(sha256_value(admission.get(key)) for key in
                    ('evaluator_sha256', 'runtime_sha256', 'numerical_contract_sha256', 'authority_instance')),
            'support lacks original admission metadata')
    contexts, shards = index.get('contexts'), index.get('shards')
    require(isinstance(contexts, list) and isinstance(shards, list) and shards,
            'support context/shard metadata missing')
    groups, ranges, offset = set(), {}, 0
    for number, shard in enumerate(shards):
        context_index, count = shard.get('context'), shard.get('records')
        require(unsigned(context_index) and context_index < len(contexts) and unsigned(count),
                'invalid support shard coordinates')
        row = contexts[context_index]
        require(row.get('partition') in ('meta_train', 'meta_validation'), 'sealed support context rejected')
        require(unsigned(row.get('cut')) and sha256_value(row.get('prepared_sha256'))
                and isinstance(row.get('family'), str) and row['family'], 'support context identity missing')
        require(all(sha256_value(shard.get(key)) for key in ('sha256', 'records_sha256', 'labels_sha256')),
                'support shard byte identities missing')
        if row['partition'] == 'meta_train' and count >= 8:
            key = digest({key: row[key] for key in ('prepared_sha256', 'cut', 'family', 'partition')})
            require(key not in groups, 'duplicate training support context')
            groups.add(key)
            ranges[number] = (offset, offset + count)
        offset += count
    require(unsigned(index.get('records')) and offset == index['records'] and groups,
            'support counts disagree or no eligible training group exists')
    policy = metadata_reference(root, specification['replay_policy'])
    require(policy.get('schema') == 'parallel.replay_policy.v2'
            and policy.get('optimizer') == 'carry'
            and policy.get('parent_checkpoint_sha256') == parent_sha
            and policy.get('corpus_sha256') == specification['corpus_index']['sha256']
            and policy.get('vocabulary_sha256') == vocabulary_sha
            and policy.get('final_tests_opened') is False,
            'replay policy parent, optimizer or corpus differs')
    require(policy.get('eligible') is True and policy.get('blocked_reasons') == [],
            'original replay policy declares unavailable dependencies')
    fresh = policy.get('fresh_records')
    require(isinstance(fresh, dict), 'original fresh record membership missing')
    fresh_groups = []
    for key, records in fresh.items():
        require(isinstance(key, str) and key.isdecimal() and str(int(key)) == key,
                'canonical fresh group identity required')
        number = int(key)
        require(number in ranges and isinstance(records, list) and records
                and all(unsigned(record) for record in records)
                and len(set(records)) == len(records), 'invalid fresh training records')
        low, high = ranges[number]
        require(all(low <= record < high for record in records), 'fresh records escape original support group')
        fresh_groups.append(number)
    require(isinstance(policy.get('fresh_groups'), list)
            and all(unsigned(group) for group in policy['fresh_groups'])
            and policy['fresh_groups'] == sorted(fresh_groups), 'fresh group identity disagreement')
    return dict(index=index, groups=groups, policy=policy, specification=specification,
                fresh_present=bool(fresh_groups))


def prepare(root, custody_root, inputs_path, output):
    root = Path(root).resolve(strict=True)
    selected = recover_selection(custody_root)
    inputs_path = inside(root, inputs_path)
    require(inputs_path.suffix == '.json'
            and not {'data', 'raw', 'sealed'} & {part.lower() for part in inputs_path.parts},
            'study input must be unsealed metadata')
    inputs, inputs_sha = custody_json(inputs_path, 1024 * 1024)
    require(set(inputs) == {'schema', 'continuation_template', 'narrow', 'broader'}
            and inputs['schema'] == 'parallel.required_learning_plan_inputs.v1', 'unknown study input contract')
    template = metadata_reference(root, inputs['continuation_template'])
    allowed = {'schema', 'mode', 'optimizer', 'exact_resume', 'sampling_seed',
        'parent_checkpoint_sha256', 'corpus_sha256', 'vocabulary_sha256', 'execution_partition',
        'requested_step_increment', 'original_run', 'original_completion', 'original_initialization',
        'parent_state', 'replay_policy_ref'}
    require(isinstance(template, dict) and not set(template) - allowed
            and allowed - {'replay_policy_ref'} <= set(template)
            and template.get('schema') == 'parallel.checkpoint_continuation.v1'
            and template.get('mode') == 'optimizer_carry_new_window'
            and template.get('optimizer') == 'carry' and template.get('exact_resume') is False
            and template.get('requested_step_increment') == 1000,
            'compatible carry continuation template required')
    require(unsigned(template['requested_step_increment']), 'invalid new-step increment')
    partition = template.get('execution_partition', {})
    require(partition.get('schema') == 'parallel.training_partition.v1'
            and all(unsigned(partition.get(key)) and partition[key] == value for key, value in
                    (('world', 8), ('microbatch', 256), ('global_batch', 2048), ('accumulation', 1))),
            'original eight-rank execution partition required')
    parent = metadata_reference(root, template['parent_state'])
    require(template.get('parent_checkpoint_sha256') == template['parent_state']['sha256']
            and unsigned(parent.get('step')) and parent['step'] + 1000 < 2**64
            and sha256_value(parent.get('vocabulary_sha256')), 'parent baseline identity unavailable')
    # Reopen the original metadata closure, never model tensors, corpus labels or query targets.
    for name in ('original_run', 'original_completion', 'original_initialization'):
        metadata_reference(root, template[name])
    supports = {'narrow': support_metadata(root, inputs['narrow'], template['parent_checkpoint_sha256'],
                                         parent['vocabulary_sha256'])}
    if inputs['broader'] is not None:
        supports['broader'] = support_metadata(root, inputs['broader'], template['parent_checkpoint_sha256'],
                                              parent['vocabulary_sha256'])
        require(supports['narrow']['groups'] <= supports['broader']['groups'],
                'broader support must retain every narrow training context')
        require(supports['narrow']['index']['admission_policy'] == supports['broader']['index']['admission_policy']
                and supports['narrow']['index']['context_index_sha256'] == supports['broader']['index']['context_index_sha256'],
                'support contrast changes the frozen admission or context protocol')
    output = inside(root, output)
    drafts, branches = [], []
    for condition in selected['conditions']:
        support = supports.get(condition['support'])
        reason = None
        if support is None or (condition['support'] == 'broader'
                               and support['groups'] == supports['narrow']['groups']):
            reason = 'no distinct eligible broader support established'
        elif condition['replay'] == 'fresh_balanced' and not support['fresh_present']:
            reason = 'no original fresh membership; 50/50 would collapse to broad replay'
        for seed in selected['seeds']:
            branch = dict(condition=condition['identity'], training_seed=seed,
                          preparation_status='unavailable' if reason else 'metadata_prepared',
                          unavailable_reason=reason, launch_allowed=False)
            if reason is None:
                directory = output / condition['identity'] / ('seed-' + str(seed))
                policy = copy.deepcopy(support['policy'])
                policy.update(sampling_seed=seed, replay=condition['replay'],
                              fresh_fraction=.5 if condition['replay'] == 'fresh_balanced' else 0,
                              fallback=None, block_identity=digest(dict(inputs_sha256=inputs_sha,
                                  condition=condition['identity'], training_seed=seed)))
                continuation = copy.deepcopy(template)
                continuation.update(sampling_seed=seed,
                    corpus_sha256=support['specification']['corpus_index']['sha256'],
                    vocabulary_sha256=parent['vocabulary_sha256'])
                branch.update(corpus_index=support['specification']['corpus_index'],
                    original_replay_policy=support['specification']['replay_policy'],
                    optimizer_parent_step=parent['step'], new_committed_steps=1000,
                    target_optimizer_step=parent['step'] + 1000, training_ceiling_seconds=300,
                    output=str(directory.relative_to(root)))
                drafts.append((branch, directory, policy, continuation))
            branches.append(branch)
    # Finish every metadata check before creating any branch files.
    for branch, directory, policy, continuation in drafts:
        policy_path = directory / 'replay-policy.json'
        continuation['replay_policy_ref'] = dict(path=str(policy_path.relative_to(root)),
                                                sha256=immutable(policy_path, policy))
        path = directory / 'continuation.json'
        branch['continuation'] = dict(path=str(path.relative_to(root)), sha256=immutable(path, continuation))
    result = dict(schema='parallel.required_learning_plan.v1', original_selection=selected,
        inputs=dict(path=str(inputs_path.relative_to(root)), sha256=inputs_sha), branches=branches,
        support_admission_independently_verified=False, committed_exposure_verified=False,
        continuation_numerically_qualified=False, funding_verified=False, launch_allowed=False,
        references_opened=False, numerical_work_started=False,
        required_before_launch=['native original admission and complete input verification',
            'compatible continuation and 50/50 fresh/broad comparison groups where selected; '
            'report actual committed and aborted program/comparison exposure',
            'original grant, branch reservation, absolute deadline and exclusive execution custody',
            'unchanged objective/architecture/labels and frozen monitoring/nomination'])
    immutable(output / 'plan.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'custody-root', 'inputs', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.root, args.custody_root, args.inputs, args.output)
    print(canonical(dict(schema=result['schema'], branches=len(result['branches']), launch_allowed=False)).decode())
