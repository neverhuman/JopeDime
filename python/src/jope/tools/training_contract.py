"""Durable training lineage and single-use exceptions to a retained training hold.

Funding remains in qualification_budget's canonical domain. These contracts do
not issue production funding, activate residents, or qualify a model for serving.
"""
import fcntl
import hashlib
import math
import os
from pathlib import Path

from promotion_gate import canonical, immutable
from resume_expired_cycle import read, require, sha


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def checked_ref(root, ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'artifact reference required')
    path = (root / ref['path']).resolve()
    require(path.is_relative_to(root.resolve()) and sha(path) == ref['sha256'], 'training artifact changed')
    return path


def checkpoint_binding(root, checkpoint):
    path = (root / checkpoint).resolve()
    require(path.is_relative_to(root.resolve()), 'training checkpoint outside workspace')
    state = read(path / 'state.json')
    require(type(state.get('step')) is int and state['step'] >= 0, 'checkpoint step unavailable')
    for name, key in [('model.ot', 'model_sha256'), ('optimizer.ot', 'optimizer_sha256'),
                      ('optimizer.json', 'optimizer_metadata_sha256'), ('config.json', 'config_sha256')]:
        require(sha(path / name) == state.get(key), 'training checkpoint payload changed: ' + name)
    require(read(path / 'optimizer.json').get('step') == state['step'], 'optimizer/model step mismatch')
    for field in ('corpus_sha256', 'vocabulary_sha256', 'sampler_id', 'ranking_backward', 'runtime_objective', 'rng'):
        require(isinstance(state.get(field), str) and state[field], 'checkpoint continuity identity absent: ' + field)
    require(type(state.get('seed')) is int and type(state.get('global_batch')) is int
            and state['global_batch'] > 0, 'checkpoint RNG or exposure identity unavailable')
    return state, sha(path / 'state.json')


def bind_task(root, task, steps, head=None):
    require(type(steps) is int and steps > 0, 'requested new step increment must be positive')
    state, parent_sha = checkpoint_binding(root, task['checkpoint'])
    mode = task.get('initialization_mode', 'experiment_initialize')
    require(mode in ('experiment_initialize', 'optimizer_carry', 'exact_resume'), 'unknown initialization mode')
    if mode == 'exact_resume':
        # Resume belongs to the original durable task window, never a new grant
        # with an old checkpoint's absolute step masquerading as fresh exposure.
        raise RuntimeError('exact resume must reconcile the original task and its unchanged step window')
    if mode == 'optimizer_carry':
        require(isinstance(head, dict) and head.get('checkpoint_sha256') == parent_sha
                and head.get('checkpoint') == task['checkpoint'], 'optimizer carry requires the experiment training head')
        require(task.get('expected_parent_revision') == digest(head), 'stale experiment training parent')
        checked_ref(root, head['original_completion'])
        policy_path = checked_ref(root, task.get('replay_policy'))
        policy = read(policy_path)
        require(policy.get('optimizer') == 'carry' and policy.get('eligible') is True
                and policy.get('parent_checkpoint_sha256') == parent_sha
                and policy.get('corpus_sha256') == sha(root / task['corpus'] / 'index.json'),
                'optimizer carry policy changed')
    initial = state['step'] if mode == 'optimizer_carry' else 0
    return dict(schema='parallel.training_execution.v1', initialization_mode=mode,
                expected_parent_revision=digest(head) if head else None,
                parent_checkpoint_sha256=parent_sha,
                parent_model_sha256=state['model_sha256'],
                parent_optimizer_sha256=state['optimizer_sha256'],
                parent_optimizer_step=state['step'],
                initial_optimizer_step=initial, requested_step_increment=steps,
                target_optimizer_step=initial + steps,
                corpus_sha256=sha(root / task['corpus'] / 'index.json'),
                global_batch=2048, parent_original_completion=head.get('original_completion') if head else None)


def completed_head(root, task, checkpoint, checkpoint_sha, original_receipt, previous=None):
    execution = task.get('execution')
    require(isinstance(execution, dict), 'training task lacks an original increment contract')
    require(execution['expected_parent_revision'] == (digest(previous) if previous else None),
            'experiment training head advanced before task completion')
    state, current_sha = checkpoint_binding(root, checkpoint)
    completion = read(original_receipt)
    window = dict(schema='parallel.training_step_window.v1', **{key: execution[key] for key in
                  ('initial_optimizer_step', 'requested_step_increment', 'target_optimizer_step')})
    require(completion.get('schema') == 'parallel.training_complete.v2'
            and completion.get('step_window') == window
            and completion.get('committed_new_steps') == execution['requested_step_increment']
            and completion.get('latest', {}).get('state_sha256') == checkpoint_sha == current_sha
            and completion.get('latest', {}).get('checkpoint') == checkpoint
            and state['step'] == execution['target_optimizer_step']
            and state['corpus_sha256'] == execution['corpus_sha256']
            and state['global_batch'] == execution['global_batch'], 'training completion increment or checkpoint changed')
    initialization = state.get('initialization', {})
    require(initialization.get('state_sha256') == execution['parent_checkpoint_sha256'],
            'trained checkpoint has a different original parent')
    require(initialization.get('optimizer_parent_step', 0) == execution['initial_optimizer_step'],
            'trained checkpoint optimizer initialization changed')
    prior_exposure = previous['committed_optimizer_examples'] if previous and execution['initialization_mode'] == 'optimizer_carry' else 0
    return dict(schema='parallel.experiment_training_head.v1', checkpoint=checkpoint,
                checkpoint_sha256=current_sha, model_sha256=state['model_sha256'],
                optimizer_sha256=state['optimizer_sha256'], optimizer_metadata_sha256=state['optimizer_metadata_sha256'],
                committed_step=state['step'], corpus=task['corpus'], corpus_sha256=state['corpus_sha256'],
                objective=dict(ranking_backward=state['ranking_backward'], runtime_objective=state['runtime_objective'],
                               replay_policy_sha256=state.get('replay_policy_sha256')),
                rng=dict(seed=state['seed'], protocol=state['rng']),
                sampler=dict(identity=state['sampler_id'], next_example=(state['step'] - initialization.get('optimizer_parent_step', 0)) * state['global_batch']),
                committed_optimizer_examples=prior_exposure + execution['requested_step_increment'] * execution['global_batch'],
                parent_revision=execution['expected_parent_revision'], execution=execution,
                original_completion=dict(path=str(original_receipt.relative_to(root)), sha256=sha(original_receipt)))


def verify_permit(root, output, ref, *, hold_sha256, task_sha256, release_sha256,
                  execution, reservation, deadline, now, consume=False):
    path = checked_ref(root, ref)
    permit = read(path)
    expected = dict(schema='parallel.held_training_permit.v1', hold_sha256=hold_sha256,
                    task_sha256=task_sha256, release_sha256=release_sha256,
                    corpus_sha256=execution['corpus_sha256'],
                    parent_checkpoint_sha256=execution['parent_checkpoint_sha256'],
                    expected_parent_revision=execution['expected_parent_revision'], reservation=reservation,
                    deadline=deadline)
    require(permit == expected, 'held training permit binding changed')
    require(type(deadline) in (int, float) and math.isfinite(deadline) and now < deadline,
            'held training permit expired')
    directory = output / 'training-permit-consumption'
    directory.mkdir(exist_ok=True)
    with (directory / 'permit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        used = directory / digest(permit)
        require(not used.exists(), 'held training permit consumed or acceptance ambiguous; reconcile original task')
        if consume:
            # Even an interrupted first receipt is a permanent no-retry tombstone.
            used.mkdir()
            parent = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            immutable(used / 'acceptance.json', dict(permit_sha256=sha(path), task_sha256=task_sha256,
                      reservation=reservation, accepted_at=now, acceptance='launch_not_yet_observed'))
    return permit
