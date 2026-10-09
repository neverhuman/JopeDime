"""Finite train-only exact-result reference export. No GP, SQL or write API.

Numeric targets are NOT materialized here: the pinned existing compose/labels
contract constructs them in a later, separately qualified preparation packet.
"""
import copy
import math
import time
from .common import Budget, CheckedCallback, canonical, digest, document, hash64, need, reference, safe_file, sha
from .cutoff import SUCCESS, STATES
from .transition import validate_extension_transition

ARTIFACTS = ('input', 'program', 'evaluation', 'rust_source', 'rust_executable', 'rust_parity', 'score_records')
CAPS = dict(zip(ARTIFACTS, (8*1024**2, 65536, 4*1024**2, 4*1024**2, 16*1024**2, 4*1024**2, 2*1024**2)))
KUMO_WEIGHT_SHA256 = 'fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423'
COEFFICIENTS = {'ce': 1., 'bt': .25, 'list': .10, 'q': .25, 'feas': .10, 'rt': .05}
CONTEXT_KEYS = {'task', 'dataset_sha256', 'target_sha256', 'cut_id', 'evaluator_sha256',
                'vocabulary_sha256', 'protocol_sha256', 'split_sha256', 'metric', 'units', 'direction'}
BATCH_KEYS = {'batch_id', 'campaign_sha256', 'checkpoint_sha256', 'global_step', 'manifest_sha256',
              'train_data_run_ids', 'validation_data_run_ids', 'emitted_train_data_run_ids'}
EMISSION_KEYS = {'phase', 'data_run_id', 'context_id', 'parent_lineage_sha256', 'input_descriptor',
                 'raw_program_utf8', 'raw_program_sha256', 'decoder_seed', 'decoder_protocol_sha256', 'decode_artifact_sha256'}


def context_identity(ctx):
    need(type(ctx) is dict and set(ctx) == CONTEXT_KEYS, 'closed Context identity')
    need(all(hash64(ctx[k]) for k in CONTEXT_KEYS if k.endswith('_sha256')) and
         all(type(ctx[k]) is str and 0 < len(ctx[k]) <= 256 for k in CONTEXT_KEYS if not k.endswith('_sha256')) and
         ctx['task'] == 'regression' and ctx['metric'] == 'rmse' and ctx['direction'] == 'minimize',
         'exact regression/RMSE context')
    return digest(ctx)


def evaluation_key(ctx, program):
    return digest({'schema': 'jope.program-evaluation-key/v1', 'context': ctx, 'program_sha256': sha(program.encode())})


class ExactReader:
    def __init__(self, get_version, limits, budget):
        self.get_version, self.limits, self.budget = get_version, limits, budget
        self.calls = self.bytes = 0
        self.bindings = {}

    def read(self, ref, cap):
        reference(ref)
        need(ref['bytes'] <= cap, 'artifact role bound')
        key = (ref['bucket'], ref['key'], ref['version_id'])
        bound = (ref['sha256'], ref['bytes'])
        need(self.bindings.setdefault(key, bound) == bound, 'conflicting same-version pins')
        self.budget.tick()
        need(self.calls < self.limits['max_reads'] and self.bytes + ref['bytes'] <= self.limits['max_read_bytes'],
             'exact artifact read budget exhausted')
        self.calls += 1
        self.bytes += ref['bytes']  # Charge attempted reads; no retries here.
        raw, version = self.get_version(copy.deepcopy(ref))
        self.budget.tick()
        need(type(raw) is bytes and version == ref['version_id'] and len(raw) == ref['bytes'] and
             sha(raw) == ref['sha256'], 'exact artifact version/length/SHA mismatch')
        return raw


def _emission(row, config):
    need(set(row) == {'id', 'body', 'state', 'outcome'} and hash64(row['id']) and row['state'] in STATES,
         'closed cutoff row')
    body = row['body']
    need(set(body) == {'schema', 'model_origin_sha256', 'batch', 'ordinal', 'emission'} and
         body['schema'] == 'jope.flywheel-neural-emission/v1' and
         body['model_origin_sha256'] == config['model_origin_sha256'], 'closed emission ancestry')
    b, e = body['batch'], body['emission']
    need(set(b) == BATCH_KEYS and set(e) == EMISSION_KEYS and type(body['ordinal']) is int and body['ordinal'] >= 0,
         'closed original batch/emission schema')
    need(all(hash64(b[k]) for k in ('batch_id', 'campaign_sha256', 'checkpoint_sha256', 'manifest_sha256')) and
         type(b['global_step']) is int and b['global_step'] > 0, 'batch identity')
    train, dev, selected = b['train_data_run_ids'], b['validation_data_run_ids'], b['emitted_train_data_run_ids']
    need(all(type(x) is list for x in (train, dev, selected)) and train and selected and
         all(hash64(x) for x in train + dev + selected) and len(set(train + dev)) == len(train + dev) and
         len(set(selected)) == len(selected) and set(selected) <= set(train), 'phase membership')
    expected = [('training', x) for x in selected] + [('validation', x) for x in dev]
    need(body['ordinal'] < len(expected) and (e['phase'], e['data_run_id']) == expected[body['ordinal']] and
         e['data_run_id'] == e['context_id'] and
         row['id'] == digest({'origin': config['model_origin_sha256'], 'batch_id': b['batch_id'], 'ordinal': body['ordinal']}),
         'exact ordered emission identity')
    need(all(hash64(e[k]) for k in ('context_id', 'parent_lineage_sha256', 'raw_program_sha256', 'decoder_protocol_sha256', 'decode_artifact_sha256')) and
         type(e['raw_program_utf8']) is str and 0 < len(e['raw_program_utf8'].encode()) <= 32768 and
         sha(e['raw_program_utf8'].encode()) == e['raw_program_sha256'] and
         type(e['decoder_seed']) is int and 0 <= e['decoder_seed'] < 2**63 and type(e['input_descriptor']) is dict,
         'emission raw program/decoder identity')
    out = row['outcome']
    if row['state'] in {'recorded', 'leased'}:
        need(out is None, 'unresolved outcome must be absent')
    else:
        need(type(out) is dict and set(out) == {'state', 'evaluation_key', 'record_ref', 'attempt_custody_ref', 'error'} and
             out['state'] == row['state'], 'closed terminal outcome')
        if row['state'] in SUCCESS:
            need(hash64(out['evaluation_key']) and out['error'] is None, 'successful outcome identity')
            reference(out['record_ref'])
        else:
            need(out['evaluation_key'] is None and out['record_ref'] is None and
                 type(out['error']) is str and 0 < len(out['error']) <= 4000, 'nonnumeric unresolved/failure custody')
        if out['attempt_custody_ref'] is not None:
            reference(out['attempt_custody_ref'])
        need(row['state'] != 'failure_custody_verified' or out['attempt_custody_ref'] is not None, 'failure custody ref')
    return b, e


def export_targets(request_raw, *, expected_request_sha256, cutoff_raw, predecessor_raw,
                   objective_raw, normalizer_raw, get_version, canonicalize, verify, authority):
    """Return canonical immutable reference manifest bytes, never publish/admit.

    Callbacks must be reviewed, pinned read-only adapters. verify(ctx,dsl,entry,
    bodies) must call the qualified current gate and return its structured result.
    authority(request) must actually recheck roles/source closures; not echo them.
    This first source packet includes no production callback/runtime qualification.
    """
    need(sha(request_raw) == expected_request_sha256, 'request SHA')
    r = document(request_raw, 16*1024**2)
    keys = {'schema', 'cutoff_sha256', 'predecessor_sha256', 'objective_sha256', 'normalizer_sha256',
            'construction_sources', 'roster', 'callbacks', 'authority_source_pins', 'admitted_descriptors',
            'allowed_campaign_sha256', 'native_train_context_ids_sha256', 'limits'}
    need(set(r) == keys and r['schema'] == 'jope.generated-train-target-export-request/v1', 'closed export request')
    for name, raw in [('cutoff', cutoff_raw), ('predecessor', predecessor_raw), ('objective', objective_raw), ('normalizer', normalizer_raw)]:
        need(sha(raw) == r[name+'_sha256'], 'pinned '+name+' mismatch')
    limits = r['limits']
    need(set(limits) == {'max_contexts', 'max_new_rows', 'max_reads', 'max_read_bytes', 'seconds'} and
         all(type(limits[k]) is int and limits[k] > 0 for k in limits) and
         limits['max_contexts'] <= 16 and limits['max_new_rows'] <= 64 and limits['max_reads'] <= 4096 and
         limits['max_read_bytes'] <= 512*1024**2 and limits['seconds'] <= 120, 'finite first-packet bounds')
    budget = Budget(limits['seconds'])
    need(set(r['callbacks']) == {'read_version', 'canonicalize', 'verify', 'authority'}, 'exact callback closure')
    callbacks = {name: CheckedCallback(fn, r['callbacks'][name]) for name, fn in
                 [('read_version', get_version), ('canonicalize', canonicalize), ('verify', verify), ('authority', authority)]}
    need(set(r['construction_sources']) == {'labels', 'compose'}, 'existing target construction source pins')
    def check_sources():
        for ref in r['construction_sources'].values():
            need(set(ref) == {'path', 'sha256'} and hash64(ref['sha256']) and
                 sha(safe_file(ref['path'], 2*1024**2)) == ref['sha256'], 'target construction source changed')
    check_sources()
    objective = document(objective_raw, 65536)
    need(objective.get('schema') == 'jope.flywheel-native-support-objective/v1' and
         objective.get('name') == 'native-support-mask-v1' and objective.get('coefficients') == COEFFICIENTS,
         'unchanged supervised objective required')
    normalizer = document(normalizer_raw, 2*1024**2)
    need(normalizer.get('schema') == 'jope.native-kumos-preprocessing/v1' and
         normalizer.get('validation_used_for_fit') is False and normalizer.get('clip') == [-10., 10.] and
         all(type(normalizer.get(k)) is list and len(normalizer[k]) == 2051 and
             all(type(x) in (int, float) and math.isfinite(x) for x in normalizer[k]) for k in ('mean', 'scale')) and
         all(x > 0 for x in normalizer['scale']), 'unchanged finite train-only Kumo normalizer')
    fit_ids = normalizer.get('fit_context_ids')
    need(type(fit_ids) is list and fit_ids and fit_ids == sorted(set(fit_ids)) and
         all(hash64(x) for x in fit_ids) and digest(fit_ids) == r['native_train_context_ids_sha256'],
         'exact original native training normalization roster')
    need(set(r['roster']) == {'name', 'dimension', 'weight_sha256'} and
         r['roster']['name'] == 'kumo-s-only-v1' and r['roster']['dimension'] == 2051 and
         r['roster']['weight_sha256'] == KUMO_WEIGHT_SHA256, 'required exact campaign KumoSmall weight')
    old = document(predecessor_raw, 16*1024**2)
    need(set(old) == {'schema', 'native_snapshot_sha256', 'parent_registry_sha256', 'generator_config_sha256',
                     'authority_helper_sha256', 'dependency_sha256', 'dataset_publishers', 'encoder_releases',
                     'tasks', 'max_read_bytes', 'preprocessing_sha256', 'teacher_rule'} and
         old.get('schema') == 'jope.native-variant-extension/v1' and old.get('teacher_rule') == 'disabled_for_unpromoted_variants/v1' and
         old.get('preprocessing_sha256') == r['normalizer_sha256'] and
         old.get('native_snapshot_sha256') == normalizer.get('snapshot_sha256') and
         type(old.get('tasks')) is list and len(old['tasks']) <= 1000, 'compatible pinned predecessor extension')
    tasks = {context_identity(t['context']): t for t in old['tasks']}
    need(len(tasks) == len(old['tasks']), 'unique prior contexts')
    for task in tasks.values():
        need(set(task) == {'variant_identity', 'parent_lineage_sha256', 'context', 'dataset_publication', 'encoder_publication', 'program_results'} and
             hash64(task['variant_identity']) and hash64(task['parent_lineage_sha256']) and
             type(task['program_results']) is list and len(task['program_results']) <= 256, 'closed prior task')
        reference(task['dataset_publication']);reference(task['encoder_publication'])
        need(len({digest(ref) for ref in task['program_results']}) == len(task['program_results']), 'duplicate old result reference')
        for ref in task['program_results']:
            reference(ref)
    need(type(r['admitted_descriptors']) is dict and set(r['admitted_descriptors']) == set(tasks), 'complete prior descriptor roster')
    need(type(r['authority_source_pins']) is list and 1 <= len(r['authority_source_pins']) <= 64 and
         all(type(p) is dict and set(p) == {'path', 'sha256'} and type(p['path']) is str and p['path'].startswith('/') and hash64(p['sha256'])
             for p in r['authority_source_pins']), 'pinned authoritative role sources')
    for cid, task in tasks.items():
        d = r['admitted_descriptors'][cid]
        need(type(d) is dict and set(d) == {'context_id', 'source_kind', 'variant_identity', 'parent_lineage_sha256',
             'partition', 'context', 'observation_dispatch', 'dataset_publication', 'encoder_publication'} and
             d['context_id'] == cid and d['source_kind'] == 'flywheel_variant' and
             d['variant_identity'] == task['variant_identity'] and d['parent_lineage_sha256'] == task['parent_lineage_sha256'] and
             d['context'] == task['context'] and d['partition'] in ('train', 'validation') and
             d['dataset_publication'] == task['dataset_publication'] and d['encoder_publication'] == task['encoder_publication'] and
             d['observation_dispatch'] == 'qualified_flywheel_worker_adapter', 'pinned descriptor identity')
    need(type(r['allowed_campaign_sha256']) is list and r['allowed_campaign_sha256'] and
         all(hash64(x) for x in r['allowed_campaign_sha256']), 'pinned source campaigns')
    def fresh():
        budget.tick()
        proof = callbacks['authority'](copy.deepcopy(r))
        need(type(proof) is dict and set(proof) == {'schema', 'state', 'request_sha256', 'checked_at_unix',
             'source_pins', 'descriptors', 'parent_roles', 'objective_sha256', 'normalizer_sha256', 'native_train_context_ids_sha256', 'roster'}, 'structured role proof')
        need(proof['schema'] == 'jope.generated-target-current-authority/v1' and proof['state'] == 'passed' and
             proof['request_sha256'] == expected_request_sha256 and type(proof['checked_at_unix']) in (int, float) and
             0 <= time.time()-proof['checked_at_unix'] <= 60 and proof['source_pins'] == r['authority_source_pins'] and
             proof['descriptors'] == r['admitted_descriptors'] and proof['objective_sha256'] == r['objective_sha256'] and
             proof['normalizer_sha256'] == r['normalizer_sha256'] and
             proof['native_train_context_ids_sha256'] == r['native_train_context_ids_sha256'] and proof['roster'] == r['roster'], 'fresh bound authority/target contract')
        need(type(proof['parent_roles']) is dict and
             set(proof['parent_roles']) == {t['parent_lineage_sha256'] for t in tasks.values()} and
             all(proof['parent_roles'][d['parent_lineage_sha256']] == d['partition'] for d in r['admitted_descriptors'].values()),
             'complete authoritative inherited parent roles')
        budget.tick()
        return proof
    first = fresh()
    cutoff = document(cutoff_raw, 256*1024**2)
    need(set(cutoff) == {'schema', 'source_sha256', 'source_bytes', 'config', 'head', 'records', 'source_optimizer_commit_verified'} and
         cutoff['schema'] == 'jope.generated-target-outbox-cutoff/v1' and cutoff['source_optimizer_commit_verified'] is False and
         hash64(cutoff['source_sha256']) and type(cutoff['source_bytes']) is int and 0 < cutoff['source_bytes'] <= 256*1024**2 and
         set(cutoff['head']) == {'events', 'head_sha256', 'observations', 'batches'} and hash64(cutoff['head']['head_sha256']) and
         all(type(cutoff['head'][k]) is int and cutoff['head'][k] >= 0 for k in ('events', 'observations', 'batches')) and
         type(cutoff['records']) is list and len(cutoff['records']) <= 100000 and len(cutoff['records']) == cutoff['head']['observations'], 'qualified finite cutoff required')
    reader = ExactReader(callbacks['read_version'], limits, budget)
    new = copy.deepcopy(old)
    new_tasks = {context_identity(t['context']): t for t in new['tasks']}
    census = {k: [] for k in ('appended', 'duplicates', 'validation', 'pending', 'refused')}
    accepted, prior_keys, selected_contexts = {}, {}, set()
    def prior(cid, ctx):
        if cid not in prior_keys:
            prior_keys[cid] = {}
            for ref in tasks[cid]['program_results']:
                wrapper = document(reader.read(ref, 1024*1024), 1024*1024)
                need(wrapper.get('schema') == 'jope.flywheel-program-result/v1' and wrapper.get('context') == ctx and
                     wrapper.get('status') == 'completed' and wrapper.get('measurement') == 'executed', 'prior result context/measurement')
                dsl = reader.read(wrapper['artifacts']['program'], 65536).decode()
                key = evaluation_key(ctx, dsl)
                need(callbacks['canonicalize'](dsl) == dsl and wrapper['program_sha256'] == sha(dsl.encode()) and wrapper['evaluation_key'] == key and
                     key not in prior_keys[cid], 'prior canonical duplicate or changed identity')
                prior_keys[cid][key] = ref
        return prior_keys[cid]
    seen_ids = set()
    for row in cutoff['records']:
        budget.tick()
        need(row['id'] not in seen_ids, 'duplicate cutoff emission');seen_ids.add(row['id'])
        b, e = _emission(row, cutoff['config'])
        note = {'emission_id': row['id'], 'state': row['state'], 'phase': e['phase'], 'context_id': e['context_id']}
        if row['state'] not in SUCCESS:
            bucket = 'refused' if row['state'] == 'failure_custody_verified' else 'pending'
            census[bucket].append({**note, 'reason': 'NONNUMERIC_'+row['state']});continue
        descriptor = e['input_descriptor']
        if descriptor.get('source_kind') != 'flywheel_variant':
            census['pending'].append({**note, 'reason': 'UNSUPPORTED_NATIVE_OVERLAY'});continue
        cid = e['context_id']
        if cid not in tasks:
            census['refused'].append({**note, 'reason': 'CONTEXT_NOT_IN_PINNED_PREDECESSOR'});continue
        task = tasks[cid]
        need(descriptor == r['admitted_descriptors'][cid] and descriptor['context'] == task['context'] and
             context_identity(descriptor['context']) == cid and descriptor['context_id'] == cid and
             descriptor['variant_identity'] == task['variant_identity'] and
             descriptor['parent_lineage_sha256'] == task['parent_lineage_sha256'] == e['parent_lineage_sha256'] and
             descriptor['dataset_publication'] == task['dataset_publication'] and
             descriptor['encoder_publication'] == task['encoder_publication'] and
             descriptor['observation_dispatch'] == 'qualified_flywheel_worker_adapter', 'exact admitted input/parent closure')
        role = first['parent_roles'].get(task['parent_lineage_sha256'])
        need(role in ('train', 'validation') and role == descriptor['partition'] and
             e['phase'] == ('training' if role == 'train' else 'validation'), 'authoritative inherited role/phase mismatch')
        need(b['campaign_sha256'] in r['allowed_campaign_sha256'], 'source campaign changed')
        if role == 'validation':
            census['validation'].append({**note, 'reason': 'EVALUATION_ONLY_NO_OPTIMIZER_TARGET'});continue
        ctx = task['context']
        dsl = callbacks['canonicalize'](e['raw_program_utf8'])
        need(type(dsl) is str and 0 < len(dsl.encode()) <= 65536, 'bounded canonical DSL')
        key = evaluation_key(ctx, dsl);ph = sha(dsl.encode());out = row['outcome'];ref = out['record_ref']
        need(out['evaluation_key'] == key, 'exact canonical evaluation key')
        if key in accepted:
            need(accepted[key]['result_ref'] == ref, 'canonical alias has conflicting result reference')
            accepted[key]['source_emission_ids'].append(row['id'])
            census['duplicates'].append({**note, 'evaluation_key': key, 'reason': 'SAME_CANONICAL_TARGET'});continue
        prior_map = prior(cid, ctx)
        if key in prior_map:
            need(prior_map[key] == ref, 'existing canonical result reference cannot be replaced')
            census['duplicates'].append({**note, 'evaluation_key': key, 'reason': 'ALREADY_IN_PREDECESSOR'});continue
        if len(accepted) >= limits['max_new_rows'] or (cid not in selected_contexts and len(selected_contexts) >= limits['max_contexts']):
            census['pending'].append({**note, 'reason': 'FINITE_EXPORT_SELECTION_BOUND'});continue
        wrapper_raw = reader.read(ref, 1024*1024);entry = document(wrapper_raw, 1024*1024)
        need(entry.get('schema') == 'jope.flywheel-program-result/v1' and entry.get('context') == ctx and
             entry.get('evaluation_key') == key and entry.get('program_sha256') == ph and
             entry.get('status') == 'completed' and entry.get('measurement') == 'executed' and
             set(entry.get('artifacts', {})) == set(ARTIFACTS), 'exact completed wrapper identity/closure')
        bodies = {name: reader.read(entry['artifacts'][name], CAPS[name]) for name in ARTIFACTS}
        need(bodies['program'] == dsl.encode(), 'canonical program bytes differ')
        measured = callbacks['verify'](copy.deepcopy(ctx), dsl, copy.deepcopy(entry), dict(bodies))
        need(type(measured) is dict and set(measured) == {'evaluation_key', 'program_sha256', 'context_id', 'measurement',
             'status', 'teaching_eligible', 'evaluation_sha256', 'artifact_versions', 'inner_fold_rmse', 'canonical_score_row_sha256'} and measured.get('evaluation_key') == key and measured.get('program_sha256') == ph and
             measured.get('context_id') == cid and measured.get('measurement') == 'executed' and measured.get('status') == 'completed' and
             measured.get('teaching_eligible') is False and measured.get('evaluation_sha256') == sha(bodies['evaluation']) and
             measured.get('artifact_versions') == {k: entry['artifacts'][k]['version_id'] for k in ARTIFACTS} and
             type(measured.get('inner_fold_rmse')) is list and len(measured['inner_fold_rmse']) == 2 and
             all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in measured['inner_fold_rmse']) and
             type(measured.get('canonical_score_row_sha256')) is list and len(measured['canonical_score_row_sha256']) == 2 and
             all(hash64(x) for x in measured['canonical_score_row_sha256']), 'structured exact verifier result required')
        need(len(new_tasks[cid]['program_results']) < 256, 'per-context immutable record bound')
        new_tasks[cid]['program_results'].append(copy.deepcopy(ref));selected_contexts.add(cid)
        accepted[key] = {'context_id': cid, 'parent_lineage_sha256': task['parent_lineage_sha256'],
                         'partition': 'train', 'evaluation_key': key, 'program_sha256': ph, 'result_ref': ref,
                         'artifact_refs': entry['artifacts'], 'measurement': measured, 'ce_eligible': False,
                         'source_emission_ids': [row['id']], 'source_batch': b,
                         'source_optimizer_commit_verified': False}
        census['appended'].append({**note, 'evaluation_key': key})
    final = fresh()
    need(first['parent_roles'] == final['parent_roles'], 'role drift before final closure')
    check_sources()
    need(sum(len(v) for v in census.values()) == cutoff['head']['observations'], 'complete denominator census')
    validate_extension_transition(old, new)
    result = {'schema': 'jope.generated-train-target-reference-manifest/v1', 'request_sha256': expected_request_sha256,
              'cutoff_sha256': r['cutoff_sha256'], 'cutoff_head': cutoff['head'],
              'predecessor_sha256': r['predecessor_sha256'], 'extension': new, 'extension_sha256': digest(new),
              'objective_sha256': r['objective_sha256'], 'normalizer_sha256': r['normalizer_sha256'],
              'construction_sources': r['construction_sources'], 'roster': r['roster'],
              'authority_proofs': [first, final], 'rows': [accepted[k] for k in sorted(accepted)],
              'census': census, 'counts': {k: len(v) for k, v in census.items()},
              'artifact_reads': {'calls': reader.calls, 'attempted_bytes': reader.bytes},
              'numeric_targets_materialized': False, 'target_recipe': 'pinned-existing-compose-and-labels-only',
              'variant_ce_teachers': 0, 'optimizer_updates': 0, 'admission_committed': False,
              's3_writes': 0, 'gp_executions': 0, 'source_optimizer_commits_verified': False,
              'state': 'verified_reference_export_only'}
    raw = canonical(result);need(len(raw) <= 32*1024**2, 'manifest output cap');budget.tick()
    return raw
