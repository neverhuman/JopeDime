"""Bounded CUDA historical-development experiments; no launcher or CPU fallback."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

# -I deliberately excludes cwd. This is the exact pinned source capsule only.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import canonical, digest, file_hash, publish, validate_plan, validate_result
from admission import check
from acquisition import parent_bootstrap_weights, thompson_pick


def main(args):
    plan = validate_plan(json.loads(args.plan.read_text()))
    plan_hash = file_hash(args.plan)
    source = HERE.parent
    for name, sha in plan['source_sha256'].items():
        p = (source / name).resolve()
        if not p.is_relative_to(source) or file_hash(p) != sha:
            raise ValueError('source drift: ' + name)
    for name, sha in plan['input_pins'].items():
        p = (args.inputs / name).resolve()
        if not p.is_relative_to(args.inputs.resolve()) or file_hash(p) != sha:
            raise ValueError('input drift: ' + name)
    if args.out.exists():
        raise ValueError('output already exists; retain original attempt, no implicit resume')
    # No torch, device, model, archive outcomes or output root before admission.
    grant = check(args.grant, plan_sha256=plan_hash, source_commit=plan['source_commit'], expected_node=plan['assigned_node'])
    started = time.monotonic()
    import numpy as np
    import torch
    from losses import context_loss
    sys.path.insert(0, str(source))
    from feedback import FeedbackLedger, ValueAdapter, parameter_hash, selection_metric, paired_interval
    from prepare import open_archive
    from run import checkpoint
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one admitted visible CUDA GPU required')
    device = torch.device('cuda:0')
    args.out.mkdir()
    publish(args.out / 'preregistration.json', plan)
    updates = 0
    program_examples = 0
    context_examples = 0
    ledger_slots = 0
    train_log = None
    acquisition_log = None
    parameter_hashes = {}
    try:
        original = json.loads((args.inputs / 'preregistration.json').read_text())
        if original['checkpoint_parameters'] != plan['native_checkpoint_parameters']:
            raise ValueError('native checkpoint ancestry mismatch')
        receipt = json.loads((args.inputs / 'feature-receipt.json').read_text())
        if receipt['features_sha256'] != file_hash(args.inputs / 'features.npz'):
            raise ValueError('native feature cache mismatch')
        archive, decoded = open_archive(args.inputs / 'inputs.tar.zst')
        if decoded != receipt['input_archive_decoded_sha256']:
            raise ValueError('decoded S3 archive mismatch')
        import hashlib
        details = json.loads(archive.extractfile('DETAILS_MANIFEST.json').read())
        for item in details['files']:
            data = archive.extractfile(item['path']).read()
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('S3 member restore mismatch: ' + item['path'])
        split = original['split']
        if digest(split) != original['split_sha256']:
            raise ValueError('split mismatch')
        train_ids = sorted(k for k, v in split.items() if v['partition'] == 'train')
        eval_ids = sorted(k for k, v in split.items() if v['partition'] == 'calibration')
        # The training ledger receives TRAIN rows only. Old assessment rows
        # are not retained or evaluated. Calibration labels are used at the end.
        selected_rows = {}
        for raw in archive.extractfile('inputs/pools.jsonl'):
            row = json.loads(raw)
            if row['context_id'] in train_ids or row['context_id'] in eval_ids:
                selected_rows[row['context_id']] = row
        archive.close()
        if set(selected_rows) != set(train_ids + eval_ids):
            raise ValueError('missing development/train contexts')
        train_rows = [selected_rows[c] for c in train_ids]
        groups = {c: split[c]['group'] for c in eval_ids}
        if set(groups.values()) & {split[c]['group'] for c in train_ids}:
            raise ValueError('parent leakage')
        arrays = np.load(args.inputs / 'features.npz', allow_pickle=False)
        features = {c: torch.tensor(arrays[c], device=device) for c in train_ids + eval_ids}
        arrays.close()
        width = receipt['feature_width']
        for c, f in features.items():
            if f.ndim != 2 or f.shape[1] != width or not torch.isfinite(f).all():
                raise ValueError('invalid frozen feature tensor')
        controls = json.loads((args.inputs / 'controls.json').read_text())
        rng = np.random.default_rng(8102026)
        initial = FeedbackLedger(split, train_rows)
        acquisition_log = gzip.open(args.out / 'acquisition.jsonl.gz', 'wt')

        def acquire(ledger, cid, member, *, wave, propensity, guide=None, scores=None, policy='uniform', posterior_member=None):
            nonlocal ledger_slots
            ledger_slots += 1
            acquisition_log.write(json.dumps({'state': 'charged_attempt', 'context_id': cid, 'member': member,
                'wave': wave, 'propensity': propensity, 'guide_checkpoint_sha256': guide,
                'acquisition_scores_sha256': digest(scores.tolist()) if scores is not None else None,
                'policy': policy, 'posterior_member': posterior_member, 'fresh_gp_fits': 0}) + '\n')
            acquisition_log.flush()
            target = ledger.acquire(cid, member, wave=wave, propensity=propensity)
            ledger.events[-1].update(guide_checkpoint_sha256=guide,
                acquisition_scores_sha256=digest(scores.tolist()) if scores is not None else None, policy=policy)
            acquisition_log.write(json.dumps({'state': 'measured' if target else 'unsupported',
                                             'target': vars(target) if target else None}) + '\n')
            acquisition_log.flush()
            return target

        for cid in train_ids:
            choices = rng.choice(len(selected_rows[cid]['members']), 2, replace=False)
            for member in choices:
                acquire(initial, cid, int(member), wave=0, propensity=1 / len(selected_rows[cid]['members']))
        publish(args.out / 'initial-targets.json', [vars(t) for _, t in sorted(initial.targets.items())])
        models, optimizers = [], []
        for seed in plan['seeds']:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = ValueAdapter(width).to(device)
            models.append(model)
            optimizers.append(torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01))
        torch.cuda.reset_peak_memory_stats()
        train_log = gzip.open(args.out / 'training.jsonl.gz', 'wt')
        last_guard = 0.

        def status(state, **extra):
            value = {'state': state, 'run_id': plan['run_id'], 'pid': os.getpid(),
                     'preregistration_sha256': plan_hash, 'optimizer_updates': updates,
                     'context_examples': context_examples, 'program_examples': program_examples,
                     'elapsed_seconds': time.monotonic() - started, **extra}
            temp = args.out / '.STATUS.tmp'
            temp.write_bytes(canonical(value))
            os.replace(temp, args.out / 'STATUS.json')

        def guard():
            nonlocal last_guard
            now = time.monotonic()
            if now - started >= plan['wall_seconds']:
                raise RuntimeError('original scientific wall budget exhausted')
            if now - last_guard >= .5:
                check(args.grant, plan_sha256=plan_hash, source_commit=plan['source_commit'], expected_node=plan['assigned_node'])
                last_guard = now
            free, _ = torch.cuda.mem_get_info()
            if free < plan['vram_reserve_bytes']:
                raise RuntimeError('explicit VRAM reserve breached')

        def train(model, optimizer, ledger, steps, seed, label, member_no):
            nonlocal updates, context_examples, program_examples
            contexts = ledger.by_context()
            ids = sorted(c for c, ts in contexts.items() if len(ts) >= 2)
            if not ids:
                raise ValueError('no supported TRAIN targets')
            random = np.random.default_rng(seed)
            before = parameter_hash(model)
            target_snapshot = ledger.identity()
            # Context-parent bootstrap is frozen per member. Multiple contexts
            # in one parent share weight; members are not independent samples.
            parent_weights = parent_bootstrap_weights([split[c]['group'] for c in ids], plan['seeds'][member_no])
            if plan['concept'] != 'bootstrap_thompson':
                parent_weights = {p: 1 for p in parent_weights}
            if not any(parent_weights.values()):
                raise ValueError('empty frozen parent bootstrap; retain failure')
            model.train()
            for step in range(steps):
                guard()
                chosen = random.choice(ids, min(plan['effective_batch_contexts'], len(ids)), replace=False).tolist()
                batch_digest = digest([(c, [t.outcome_sha256 for t in contexts[c]]) for c in chosen])
                optimizer.zero_grad(set_to_none=True)
                denominator = sum(parent_weights[split[c]['group']] for c in chosen)
                if denominator == 0:
                    raise ValueError('zero supported bootstrap batch; no silent resampling')
                loss_value = 0.
                for start in range(0, len(chosen), plan['microbatch_contexts']):
                    guard()
                    batch = chosen[start:start + plan['microbatch_contexts']]
                    maximum = max(len(contexts[c]) for c in batch)
                    x = torch.zeros(len(batch), maximum, width, device=device)
                    q = torch.zeros(len(batch), maximum, device=device)
                    mask = torch.zeros(len(batch), maximum, dtype=torch.bool, device=device)
                    weights = torch.tensor([parent_weights[split[c]['group']] for c in batch], device=device)
                    for i, c in enumerate(batch):
                        ts = contexts[c]
                        x[i, :len(ts)] = features[c][[t.member for t in ts]]
                        q[i, :len(ts)] = torch.tensor([t.quality for t in ts], device=device)
                        mask[i, :len(ts)] = True
                    losses = context_loss(model(x), q, mask, plan['concept'])
                    loss = (losses * weights).sum() / denominator
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite objective')
                    loss.backward()
                    loss_value += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                context_examples += len(chosen)
                program_examples += sum(len(contexts[c]) for c in chosen)
                train_log.write(json.dumps({'arm': label, 'member': member_no, 'step': step + 1,
                    'global_updates': updates, 'loss_composite': loss_value, 'unclipped_grad_norm': float(norm),
                    'contexts': chosen, 'supported_weight_denominator': denominator,
                    'supported_contexts': len(chosen), 'supported_programs': sum(len(contexts[c]) for c in chosen),
                    'consumed_batch_sha256': batch_digest, 'target_snapshot_sha256': target_snapshot}) + '\n')
                if step % 16 == 0:
                    train_log.flush()
                    status('running', arm=label, member=member_no,
                           parameter_sha256=parameter_hash(model),
                           cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            after = parameter_hash(model)
            if before == after:
                raise ValueError('optimizer made no parameter change')
            parameter_hashes[label + ':' + str(member_no)] = {'before': before, 'after': after}

        config = {'feature_width': width, 'ensemble_seeds': plan['seeds'],
                  'checkpoint_parameters': plan['native_checkpoint_parameters']}
        for i, (m, o, seed) in enumerate(zip(models, optimizers, plan['seeds'])):
            train(m, o, initial, plan['initial_updates'], seed, 'initial', i)
        initial_child = checkpoint(args.out / 'initial', models, optimizers, ledger=initial, feature_receipt=receipt, config=config, wave=0)
        arms = {name: {'models': copy.deepcopy(models), 'optimizers': [], 'ledger': copy.deepcopy(initial)} for name in plan['arms']}
        for arm in arms.values():
            arm['last_child'] = initial_child['state_sha256']
            for i, m in enumerate(arm['models']):
                o = torch.optim.AdamW(m.parameters(), lr=.0005, weight_decay=.01)
                o.load_state_dict(copy.deepcopy(optimizers[i].state_dict()))
                arm['optimizers'].append(o)

        @torch.no_grad()
        def predict(ms, cid):
            values = np.stack([m.eval()(features[cid]).detach().cpu().numpy() for m in ms])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite acquisition/inference scores')
            return values

        ancestry = []
        for wave in range(1, plan['waves'] + 1):
            for name, arm in arms.items():
                if name != 'continuation':
                    random = np.random.default_rng(3000 + wave)
                    acquired = arm['ledger'].by_context()
                    for cid in train_ids:
                        guard()
                        available = sorted(set(range(len(selected_rows[cid]['members']))) -
                                           {t.member for t in acquired.get(cid, [])})
                        scores = predict(arm['models'], cid) if name == 'targeted_feedback' else None
                        for _ in range(min(2, len(available))):
                            posterior_member = None
                            if name == 'uniform_feedback':
                                picked = int(random.choice(available)); propensity = 1 / len(available)
                            else:
                                if plan['concept'] == 'bootstrap_thompson':
                                    picked, propensity, posterior_member = thompson_pick(available, scores.tolist(), random)
                                else:
                                    values = scores.mean(0) + scores.std(0)
                                    priority = max(available, key=lambda k: (float(values[k]), -k))
                                    picked = priority if random.random() < .75 else int(random.choice(available))
                                    propensity = .25 / len(available) + (.75 if picked == priority else 0.)
                            acquire(arm['ledger'], cid, picked, wave=wave, propensity=propensity,
                                guide=arm['last_child'] if name == 'targeted_feedback' else None,
                                scores=scores, policy='parent_bootstrap_thompson_mix'
                                if plan['concept'] == 'bootstrap_thompson' and name == 'targeted_feedback' else name,
                                posterior_member=posterior_member)
                            available.remove(picked)
                before_target = arm['ledger'].identity()
                for i, (m, o, seed) in enumerate(zip(arm['models'], arm['optimizers'], plan['seeds'])):
                    train(m, o, arm['ledger'], plan['updates_per_wave'], seed + wave, name + '-wave' + str(wave), i)
                child = checkpoint(args.out / (name + '-wave' + str(wave)), arm['models'], arm['optimizers'],
                                   ledger=arm['ledger'], feature_receipt=receipt, config=config, wave=wave)
                arm['last_child'] = child['state_sha256']
                publish(args.out / (name + '-wave' + str(wave) + '-targets.json'),
                        {'targets': [vars(t) for _, t in sorted(arm['ledger'].targets.items())], 'events': arm['ledger'].events})
                ancestry.append({'arm': name, 'wave': wave, 'consumed_target_sha256': before_target,
                                 'child_parameters': child['parameters'], 'checkpoint_sha256': child['state_sha256']})
                publish(args.out / (name + '-wave' + str(wave) + '-progress.json'),
                        {'state': 'wave_complete', 'optimizer_updates': updates, 'ancestry': ancestry[-1],
                         'elapsed_seconds': time.monotonic() - started})
        train_log.close(); train_log = None
        acquisition_log.close(); acquisition_log = None
        guard()
        per_context = {}
        for cid in eval_ids:
            guard()
            row = selected_rows[cid]
            scores = {name: predict(arm['models'], cid).mean(0) for name, arm in arms.items()}
            for native in ('native17773', 'native429'):
                for head in ('trie', 'qhead'):
                    scores[native + '_' + head] = controls[cid][native + '.safetensors_' + head]
            per_context[cid] = {name: selection_metric(row, value, 4) for name, value in scores.items()}
        names = list(next(iter(per_context.values())))
        metrics = {}
        for name in names:
            parent = {}
            for cid in eval_ids:
                parent.setdefault(groups[cid], []).append(per_context[cid][name])
            metrics[name] = {
                'parent_mean_regret': float(np.mean([np.mean([x['regret'] for x in xs]) for xs in parent.values()])),
                'parent_mean_quality': float(np.mean([np.mean([x['quality'] for x in xs]) for xs in parent.values()])),
                'parent_mean_beats_incumbent': float(np.mean([np.mean([x['beats_incumbent'] for x in xs]) for xs in parent.values()]))}
        primary = paired_interval({c: per_context[c]['uniform_feedback']['normalized_quality'] for c in eval_ids},
                                  {c: per_context[c]['continuation']['normalized_quality'] for c in eval_ids}, groups)
        primary['practical_margin'] = .005
        primary['decision'] = 'positive' if primary['lower'] > .005 else ('negative' if primary['upper'] < 0 else 'inconclusive')
        identity = plan['comparison_identity']
        expected_slots = len(initial.events) + sum(len(a['ledger'].events) - len(initial.events) for a in arms.values())
        if ledger_slots != expected_slots:
            raise ValueError('charged acquisition accounting mismatch')
        evaluation_rows = sum(len(selected_rows[c]['members']) for c in eval_ids)
        result = {'schema': 'jope.research-result/v1', 'state': 'complete', 'run_id': plan['run_id'],
                  'branch': plan['branch'], 'source_commit': plan['source_commit'], 'preregistration_sha256': plan_hash,
                  'concept': plan['concept'], 'scope': 'historical_replay', 'comparison_identity': identity,
                  'learned_component': 'value_guide_adapter', 'native_backbone_optimizer_updates': 0,
                  'comparison_identity_sha256': digest(identity), 'metrics': metrics, 'primary': primary,
                  'parameter_hashes': parameter_hashes, 'ancestry': ancestry,
                  'counts': {'optimizer_updates': updates, 'fresh_gp_fits': 0, 'historical_replay_slots': ledger_slots,
                             'cache_reads': ledger_slots + evaluation_rows, 'failures': 0}, 'hwm_promotion': False,
                  'counter_definition': 'Shared initial acquisitions charged once; additional arm acquisitions plus once-used development outcomes. Cache reads count logical cached-outcome requests, not S3 GETs or JSON deserialization; repeated batch consumption is program_examples.',
                  'cost': {'wall_seconds': time.monotonic() - started, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'oom_count': 0,
                           'context_examples': context_examples, 'program_examples': program_examples,
                           'program_examples_per_second': program_examples / (time.monotonic() - started),
                           'native_decoder_tokens_optimized': 0,
                           'cost_scope': 'research process; external owner must append whole-job/control/bootstrap costs'},
                  'limitations': ['Historical development only; both backbones previously saw all contexts as TRAIN.',
                                  'No new raw-data GP fits/native backbone updates/HWM/promotion.',
                                  'Teacher CE is unchanged; training objective here is not teacher CE.',
                                  'Equal update budgets do not imply equal measured wall/GPU costs.']}
        validate_result(result)
        publish(args.out / 'per-context.json', {'groups': groups, 'metrics': per_context})
        publish(args.out / 'result.json', result)
        status('complete', result_sha256=file_hash(args.out / 'result.json'))
    except BaseException as exc:
        if train_log is not None:
            train_log.close()
        if acquisition_log is not None:
            acquisition_log.close()
        publish(args.out / 'failure.json', {'state': 'failed', 'exception': type(exc).__name__, 'message': str(exc),
                'optimizer_updates': updates, 'fresh_gp_fits': 0, 'elapsed_seconds': time.monotonic() - started,
                'oom_count': int(isinstance(exc, torch.cuda.OutOfMemoryError)),
                'preregistration_sha256': plan_hash, 'retry': 'no automatic retry; preserve charged attempt'})
        failed = {'schema': 'jope.research-result/v1', 'state': 'failed', 'run_id': plan['run_id'],
                  'branch': plan['branch'], 'source_commit': plan['source_commit'], 'preregistration_sha256': plan_hash,
                  'concept': plan['concept'], 'scope': 'historical_replay', 'comparison_identity': plan['comparison_identity'],
                  'comparison_identity_sha256': digest(plan['comparison_identity']), 'metrics': {},
                  'counts': {'optimizer_updates': updates, 'fresh_gp_fits': 0, 'historical_replay_slots': ledger_slots,
                             'cache_reads': ledger_slots, 'failures': 1}, 'hwm_promotion': False,
                  'failure': {'kind': type(exc).__name__, 'message': str(exc)},
                  'cost': {'wall_seconds': time.monotonic() - started, 'program_examples': program_examples,
                           'remaining_costs': 'External bootstrap/whole-job custody must append actual costs, including failed setup.'}}
        validate_result(failed)
        if not (args.out / 'result.json').exists():
            publish(args.out / 'result.json', failed)
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('plan', 'inputs', 'out', 'grant'):
        ap.add_argument('--' + name, type=Path, required=True)
    main(ap.parse_args())
