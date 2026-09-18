#!/usr/bin/env python3
"""Continue verified learning checkpoints with durable GP, snapshot and training stages."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import candidate_intake
import corpus_verification
import experiment_runtime as runtime
import gpu_occupancy
import promotion_gate
import teaching_manifest
import training_contract
from qualification_budget import Budget
from resume_expired_cycle import atomic, read, require, sha


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=fields[19],
                    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, IndexError):
        return None


class SurvivingProcess:
    """Reconcile an owned process without confusing a reused PID with the task."""
    def __init__(self, identity):
        self.identity, self.pid, self.returncode = identity, identity['pid'], None

    def poll(self):
        if process_identity(self.pid) != self.identity:
            self.returncode = -1
        return self.returncode

    def wait(self, timeout=10):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            require(time.monotonic() < deadline, 'surviving task did not exit')
            time.sleep(.05)
        return self.returncode


def meaningful_progress(root, task, packing_path):
    kind = task['kind']
    if kind == 'pack':
        paths = [packing_path]
    elif kind=='pack_verify':
        paths=[(root/task['receipt']).with_name('progress.json')]
    elif kind == 'train':
        paths = [root / 'runs/neural-service.json']
    elif kind == 'proposal':
        paths = list((root / task['receipt']).parent.glob('rank-*.json'))
    else:
        paths = []
    values = []
    for path in paths:
        if not path.is_file():
            continue
        row = read(path)
        if kind == 'train' and row.get('run') != task.get('output'):
            continue
        value = {key: row[key] for key in ('step', 'checkpoint_step', 'completed', 'generated', 'jobs',
                    'work', 'replay_comparisons_this_rank_attempt') if key in row}
        if value:
            values.append(value)
    return values or None


def bound_path(root, value):
    path = (root / value).resolve()
    require(path.is_relative_to(root), 'path outside learning workspace: ' + str(path))
    return path


def verify_checkpoint(root, checkpoint, expected=None):
    path = bound_path(root, checkpoint)
    state = read(path / 'state.json')
    if expected:
        require(sha(path / 'state.json') == expected, 'checkpoint state changed')
    for name, key in [('model.ot', 'model_sha256'), ('optimizer.ot', 'optimizer_sha256'),
                      ('optimizer.json', 'optimizer_metadata_sha256'), ('config.json', 'config_sha256')]:
        require(sha(path / name) == state[key], 'checkpoint payload changed: ' + name)
    require(not any((parent / 'invalidated.json').exists() for parent in [path, *path.parents] if parent.is_relative_to(root)),
            'withdrawn checkpoint cannot initialize learning')
    return state


def verify_corpus(root, corpus, contexts_sha, vocabulary_sha,progress=None):
    path = bound_path(root, corpus)
    index = read(path / 'index.json')
    provenance = read(path / 'provenance.json')
    require(index['schema'] in ('parallel.teaching_corpus.v1', 'parallel.teaching_shards.v1') and index['qualification'] is True,
            'unrecognized teaching corpus')
    require(index['context_index_sha256'] == contexts_sha and index['vocabulary_sha256'] == vocabulary_sha,
            'corpus cohort or vocabulary changed')
    require(sha(path / 'index.json') == provenance['index_sha256'], 'corpus index changed')
    require(sha(path / 'vocabulary.json') == vocabulary_sha, 'corpus vocabulary changed')
    if index['schema'] == 'parallel.teaching_shards.v1':
        index,_ = teaching_manifest.load(root,path,progress=progress)
    else:
        require(sha(path / 'records.bin') == index['records_sha256'], 'corpus records changed')
        require(sha(path / 'labels.jsonl') == provenance['labels_sha256'], 'corpus labels changed')
        require((path / 'records.bin').stat().st_size == index['records'] * index['record_bytes'], 'truncated corpus')
    require(all(g['partition'] in ('meta_train', 'meta_validation') for g in index['groups']), 'sealed supervision')
    return index


def reference_training_growth(before,after):
    """Reference first-label semantics can only add programs, never rescore them."""
    if after.get('schema')!='parallel.teaching_shards.v1' or after.get('label_policy')!='reference_v1':
        return None
    def sizes(index):
        if index.get('schema')=='parallel.teaching_shards.v1':
            return {s['context']:s['records'] for s in index['shards']
                    if index['contexts'][s['context']]['partition']=='meta_train'}
        return {g['context']:len(g['records']) for g in index['groups'] if g['partition']=='meta_train'}
    old,new=sizes(before),sizes(after)
    return sum(max(0,size-old.get(context,0)) for context,size in new.items())


def snapshot_due(new_searches, changed, last_snapshot, now):
    return changed and (new_searches >= 104 or now - last_snapshot >= 600)


def verified_search(root, job):
    """A legacy service must never turn a generation into an apparently valid snapshot."""
    output = bound_path(root, job['output'])
    require(not any((parent / 'invalidated.json').exists()
                    for parent in [output, *output.parents] if parent.is_relative_to(root)),
            'invalidated GP generation cannot teach')
    if not (output / 'search.json').exists():
        return False
    receipt = read(output / 'search.json')
    require(receipt.get('rng_seed') == job['rng_seed'], 'GP result RNG identity changed')
    require(receipt.get('seeds_sha256') == job['seeds_sha256'], 'GP result population changed')
    binding = read(output / 'job-binding.json')
    for key in ('generation', 'rng_seed', 'parent_checkpoint', 'parent_checkpoint_sha256',
                'prepared', 'prepared_sha256', 'cut', 'seeds', 'seeds_sha256', 'output'):
        require(binding.get(key) == job[key], 'GP dispatch identity changed: ' + key)
    require((output / 'observations.json').is_file(), 'committed GP ledger missing')
    return True


def proposal_input_inventory(root, generations, manifest, previous=None):
    """Advance a bounded metadata census; usable inputs and authority stay unknown."""
    reserve = manifest['gp_workers'] * 3
    result = dict(schema='parallel.proposal_input_inventory.v2', observed_at=time.time(),
        planned_reserve=reserve, refill_priority_at=manifest['gp_workers'] * 2,
        scanned_generations=0, scanned_jobs=0, planned_unfinished=0, validated_proposal_metadata=0,
        completion_present=0, ambiguous_outputs=0, ineligible_inputs=0,
        invalid_inputs=0, duplicate_identities=0, unverified_manifests=0,
        scan_complete=False, verified_proposal_inputs=None, eligible_jobs=None, runnable_jobs=None,
        unavailable=['current coverage eligibility', 'original funding grant',
                     'authorized parent', 'original task deadline', 'prepared cut payload verification',
                     'native seed grammar validation'],
        dispatch_authority='unavailable', scope='newest completed generations; stop at reserve',
        input_check='metadata census only; prepared cut payloads are not opened or verified')
    binding = [len(generations), manifest['contexts'], manifest['contexts_sha256']]
    cursor = dict(binding=binding, generation=len(generations)-1, job=0, identities=[], outputs=[])
    counters = ('scanned_generations', 'scanned_jobs', 'planned_unfinished', 'validated_proposal_metadata',
                'completion_present', 'ambiguous_outputs', 'ineligible_inputs', 'invalid_inputs',
                'duplicate_identities', 'unverified_manifests')
    if (isinstance(previous, dict) and previous.get('schema') == result['schema']
            and isinstance(previous.get('cursor'), dict) and previous['cursor'].get('binding') == binding):
        cursor = dict(previous['cursor'])
        result.update({key: previous[key] for key in counters})
    result['scan_started_at'] = previous['scan_started_at'] if cursor.get('continued') else result['observed_at']
    result['cursor'] = cursor
    cursor['continued'] = True
    cache, seen, seen_outputs, bytes_read = {}, set(cursor['identities']), set(cursor['outputs']), 0
    scanned_generations = scanned_jobs = 0

    def integer(value, maximum=(1 << 64)-1):
        require(type(value) is int and 0 <= value <= maximum, 'native unsigned integer required')
        return value

    def digest(value):
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'invalid metadata digest')
        return value

    def allowed(name):
        path = Path(name)
        require(not path.is_absolute() and not any(part in ('..', 'data', 'sealed', 'meta_test', 'raw')
                or part.startswith(('sealed-', 'sealed_')) for part in path.parts),
                'raw or sealed queue input path rejected')
        return runtime.inside(root, name)

    def finish(limited=False):
        cursor['identities'], cursor['outputs'] = sorted(seen), sorted(seen_outputs)
        if limited:
            result['scan_limit_reached'] = True
        return result

    class WindowEnded(Exception):
        pass

    def observed(name, parse=True):
        nonlocal bytes_read
        path = allowed(name)
        key = (path, parse)
        if key not in cache:
            if bytes_read >= 64 * 1024 * 1024:
                raise WindowEnded()
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                require(stat.S_ISREG(info.st_mode), 'queue input is not regular')
                require(info.st_size <= 16 * 1024 * 1024, 'queue metadata file limit')
                size = min(16 * 1024 * 1024, 64 * 1024 * 1024 - bytes_read)
                if info.st_size > size:
                    raise WindowEnded()
                data = stream.read(size + 1)
            bytes_read += len(data)
            require(len(data) <= size, 'queue metadata grew during read')
            def unique(pairs):
                value = {}
                for name, item in pairs:
                    require(name not in value, 'duplicate queue metadata key')
                    value[name] = item
                return value
            value = json.loads(data, object_pairs_hook=unique) if parse else None
            if parse:
                json.dumps(value, allow_nan=False)
            cache[key] = (value, hashlib.sha256(data).hexdigest(), len(data))
        return cache[key]

    try:
        contexts, contexts_sha, _ = observed(manifest['contexts'])
        require(contexts_sha == manifest['contexts_sha256']
                and contexts['schema'] == 'parallel.context_index.v1'
                and integer(contexts['dimensions']) == 2048
                and type(contexts['qualification']) is bool, 'queue context index changed')
        digest(contexts['assembly_sha256'])
        lookup = {}
        for row in contexts['encodings']:
            key = (row['prepared'], integer(row['cut']))
            require(key not in lookup, 'ambiguous queue context identity')
            lookup[key] = row
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError):
        result['unverified_manifests'] += 1
        result['cursor'] = None
        return result
    for generation_index in range(cursor['generation'], -1, -1):
        if scanned_generations >= 256:
            return finish(limited=True)
        scanned_generations += 1
        generation = generations[generation_index]
        start_job = cursor['job'] if cursor['generation'] == generation_index else 0
        cursor.update(generation=generation_index, job=start_job)
        if start_job == 0:
            result['scanned_generations'] += 1
        try:
            integer(generation['number'])
            if integer(generation['stage'], 4) != 4:
                cursor.update(generation=generation_index-1, job=0)
                continue
            jobs, jobs_sha, _ = observed(str(Path(generation['path']) / 'jobs.json'))
            if start_job and cursor.get('jobs_sha256') != jobs_sha:
                result.update(cursor=None, restart_required=True)
                return result
            cursor['jobs_sha256'] = jobs_sha
            receipt, _, _ = observed('runs/queue-receipts/' + jobs_sha + '.json')
            require(jobs['schema'] == 'parallel.gp_jobs.v1'
                    and type(jobs['qualification']) is bool
                    and (jobs['qualification'] or contexts['qualification'] is False)
                    and receipt['schema'] == 'parallel.encoded_gp_admission.v1'
                    and receipt['jobs_sha256'] == jobs_sha
                    and receipt['contexts_sha256'] == contexts_sha
                    and receipt['campaign_contexts_sha256'] == contexts_sha
                    and receipt['campaign_bound'] is True
                    and receipt['inputs_rechecked_before_merge'] is True
                    and type(receipt['qualification']) is bool
                    and receipt['qualification'] == jobs['qualification']
                    and integer(receipt['verified_contexts']) == len(jobs['jobs']), 'queue admission binding changed')
        except WindowEnded:
            return finish(limited=True)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError):
            result['unverified_manifests'] += 1
            cursor.update(generation=generation_index-1, job=0)
            continue
        for job_index in range(start_job, len(jobs['jobs'])):
            if scanned_jobs >= 4096:
                return finish(limited=True)
            scanned_jobs += 1
            job = jobs['jobs'][job_index]
            cursor.update(generation=generation_index, job=job_index)
            before_job = {key: result[key] for key in counters}
            result['scanned_jobs'] += 1
            try:
                for key in ('generation', 'rng_seed', 'cut'):
                    integer(job[key])
                output = allowed(job['output'])
                if (output / 'search.json').exists():
                    result['completion_present'] += 1  # Presence is not verified completion.
                    continue
                result['planned_unfinished'] += 1
                if any((p / 'invalidated.json').exists() for p in [output, *output.parents]
                       if p.is_relative_to(root)):
                    result['ineligible_inputs'] += 1
                    continue
                if output.exists() and (not output.is_dir() or next(output.iterdir(), None) is not None):
                    result['ambiguous_outputs'] += 1
                    continue
                row = lookup[(job['prepared'], job['cut'])]
                if row['partition'] not in ('meta_train', 'meta_validation'):
                    result['ineligible_inputs'] += 1
                    continue
                require(job['generation'] == generation['number']
                        and job['parent_checkpoint'] == generation['checkpoint']
                        and job['parent_checkpoint_sha256'] == generation['checkpoint_sha256'],
                        'queue generation parent changed')
                parent = runtime.inside(root, job['parent_checkpoint'])
                if any((p / 'invalidated.json').exists() for p in [parent, *parent.parents]
                       if p.is_relative_to(root)):
                    result['ineligible_inputs'] += 1
                    continue
                _, parent_sha, _ = observed(str(Path(job['parent_checkpoint']) / 'state.json'))
                require(parent_sha == job['parent_checkpoint_sha256'], 'queue parent metadata changed')
                prepared, prepared_sha, _ = observed(str(Path(job['prepared']) / 'prepared.json'))
                qualification = jobs['qualification']
                require(all(type(prepared[key]) is bool for key in
                            ('family_partition_frozen', 'clean_evaluation', 'qualification_only')),
                        'native preparation booleans required')
                require(prepared_sha == job['prepared_sha256'] == row['prepared_sha256']
                        and prepared['schema'] == 'parallel.prepared.v1'
                        and prepared['dataset'] == row['dataset']
                        and prepared['family'] == row['family']
                        and prepared['meta_partition'] == row['partition'], 'queue preparation changed')
                require(prepared['task'] == 'regression'
                        and (prepared['meta_partition'] == 'meta_train' or qualification)
                        and (prepared['family_partition_frozen'] or qualification)
                        and (prepared['clean_evaluation'] or (qualification and prepared['qualification_only']
                            and prepared['clean_status'] == 'raw_training_reconstructed_qualification_only')),
                        'native preparation policy rejected')
                cuts = {integer(cut['index']): cut for cut in prepared['cuts']}
                require(len(cuts) == len(prepared['cuts']), 'ambiguous prepared cut')
                cut = cuts[job['cut']]
                for key in ('training_sha256', 'validation_sha256', 'preprocessing_sha256'):
                    digest(cut[key])
                for index in cut['train_indices'] + cut['validation_indices']:
                    integer(index)
                training_rows_sha = hashlib.sha256(json.dumps(cut['train_indices'], separators=(',', ':')).encode()).hexdigest()
                # Validate metadata before opening the referenced encoding bytes.
                allowed(row['path'])
                encoding, receipt_sha, _ = observed(str(Path(row['path']).with_suffix('.json')))
                require((not row.get('receipt_sha256') or receipt_sha == row['receipt_sha256'])
                        and encoding['schema'] == 'parallel.encoding.v1'
                        and encoding['output_sha256'] == digest(row['sha256'])
                        and encoding['encoder_spec_sha256'] == contexts['assembly_sha256']
                        and encoding['training_sha256'] == cut['training_sha256']
                        and encoding['preprocessing_sha256'] == cut['preprocessing_sha256']
                        and encoding['selected_training_rows_sha256'] == training_rows_sha
                        and encoding['prepared_sha256'] == prepared_sha
                        and encoding['dataset'] == row['dataset'] and integer(encoding['cut']) == job['cut']
                        and encoding['family'] == row['family'] and encoding['partition'] == row['partition'],
                        'queue context receipt changed')
                require(integer(encoding['dataset_dimensions']) == 1536
                        and integer(encoding['target_dimensions']) == 512, 'encoding dimensions changed')
                if row.get('components_sha256') is not None:
                    require(encoding['components_sha256'] == digest(row['components_sha256']),
                            'encoding component binding changed')
                seeds, seeds_sha, _ = observed(job['seeds'])
                require(seeds_sha == job['seeds_sha256'], 'queue population changed')
                require(seeds['schema'] == 'parallel.seed_population.v1'
                        and seeds['prepared_sha256'] == prepared_sha and integer(seeds['cut']) == job['cut']
                        and seeds['training_sha256'] == cut['training_sha256']
                        and len(seeds['candidates']) == 80, 'seed population metadata changed')
                digest(seeds['context_sha256']); digest(seeds['checkpoint_sha256'])
                require(all(isinstance(c['genome'], str) and c['genome']
                            and isinstance(c['origin'], str) and c['origin'] for c in seeds['candidates'])
                        and len({c['genome'] for c in seeds['candidates']}) == 80,
                        'invalid or duplicate seed metadata')
                _, context_sha, context_size = observed(row['path'], parse=False)
                require(context_sha == row['sha256'] and context_size == 8192, 'queue context changed')
                identity = hashlib.sha256(json.dumps([job['generation'], job['rng_seed'], parent_sha,
                    prepared_sha, job['cut'], seeds_sha]).encode()).hexdigest()
                output_identity = hashlib.sha256(str(output).encode()).hexdigest()
                if identity in seen or output_identity in seen_outputs:
                    result['duplicate_identities'] += 1
                    continue
                seen.add(identity); seen_outputs.add(output_identity)
                result['validated_proposal_metadata'] += 1
                if result['validated_proposal_metadata'] >= reserve:
                    result['cursor'] = None
                    return result
            except WindowEnded:
                # A row larger than one whole pass must not pin the cursor forever.
                position = [generation_index, job_index]
                if cursor.get('window_position') != position:
                    cursor['window_position'] = position
                    result.update(before_job)
                    return finish(limited=True)
                result['invalid_inputs'] += 1
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError):
                result['invalid_inputs'] += 1
            finally:
                # A window retry resumes this row; other outcomes consume it.
                if not result.get('scan_limit_reached'):
                    cursor.update(generation=generation_index, job=job_index+1)
        cursor.update(generation=generation_index-1, job=0)
    result['scan_complete'] = result['unverified_manifests'] == 0
    result['census_finished'] = True
    result['cursor'] = None
    return result


def idle_status(seconds, eligible, explained):
    if not eligible or explained:
        return None
    return 'incident' if seconds >= 60 else 'warning' if seconds >= 30 else None


def idle_explanation(tasks, children, gp, now):
    if any(kind in tasks for kind in ('train', 'proposal', 'monitor')):
        return None
    if 'pack' in children:
        return 'packing fresh targets'
    if (gp.get('phase') == 'running' and 0 <= now - gp.get('timestamp', 0) <= 30
            and gp.get('active', 0) > 0):
        return 'waiting for fresh GP results; snapshot at 104 completions or 10 minutes'
    return None


def validate_manifest(root, path):
    manifest = read(path)
    require(manifest['schema'] == 'parallel.continuous_learning.v1', 'bad continuous manifest')
    require(manifest['run_until_paused'] is True, 'continuous authorization missing')
    require(manifest['world'] == 8 and manifest['microbatch'] == 256, 'unqualified recovery batch')
    require(manifest['gp_workers'] == 104 and manifest['steps'] > 0, 'invalid bounded block')
    for name, digest in manifest['bindings'].items():
        require(sha(bound_path(root, name)) == digest, 'manifest input changed: ' + name)
    require(manifest['bindings'][manifest['contexts']] == manifest['contexts_sha256'], 'unbound cohort')
    contexts = read(bound_path(root, manifest['contexts']))
    require(contexts['schema'] == 'parallel.context_index.v1', 'invalid context index')
    require(not set(manifest['withdrawn_families']) & {c['family'] for c in contexts['encodings']}, 'withdrawn cohort')
    return manifest


def child_environment(root, manifest):
    name = manifest['cpu_executable']
    executable = bound_path(root, name)
    require(manifest.get('bindings', {}).get(name) == sha(executable),
            'CPU evaluator is not bound to the reviewed release')
    return dict(os.environ, JOPE_GP_EXECUTABLE=str(executable), JOPE_CPU_EVALUATOR=str(executable))


RECOVERY_PARENT = 'c2f229c9b8c582e11e0e62ec011a9ccb066820fbcee5e2da9d0ad5d63780160e'


def assemble_jobs(root, generation, spec, arms):
    """Bind each new identity; matched arms use the same search RNG and budgets."""
    names = ['neural', 'generic']
    if 'shuffled' in arms:
        names.append('shuffled')
    loaded = [read(root / arms[name] / 'jobs.json') for name in names]
    require(len({len(block['jobs']) for block in loaded}) == 1, 'unmatched tournament arms')
    result = dict(schema='parallel.gp_jobs.v1', qualification=True, vocabulary=loaded[0]['vocabulary'], jobs=[])
    cohort = read(root / spec['contexts'])['encodings']
    eligible = {(c['prepared'], c['cut']): c for c in cohort if c['partition'] in ('meta_train', 'meta_validation')}
    for index, rows in enumerate(zip(*(block['jobs'] for block in loaded))):
        key = (rows[0]['prepared'], rows[0]['cut'])
        require(key in eligible, 'tournament cut outside frozen eligible cohort')
        seed = int.from_bytes(hashlib.sha256(f"{spec['rng_seed']}:{generation}:{key}".encode()).digest()[:8], 'little')
        for arm, original in zip(names, rows):
            job = dict(original)
            require((job['prepared'], job['cut']) == key and job['prepared_sha256'] == eligible[key]['prepared_sha256'],
                    'comparison arms or context identity differ')
            require(sha(root / job['seeds']) == job['seeds_sha256'], 'population changed')
            require(bound_path(root, job['output']).is_relative_to(root / arms[arm]), 'reused tournament output')
            job.update(generation=generation, rng_seed=seed, parent_checkpoint=spec['checkpoint'],
                       parent_checkpoint_sha256=spec['checkpoint_sha256'])
            result['jobs'].append(job)
    require(len({j['output'] for j in result['jobs']}) == len(result['jobs']), 'duplicate tournament identity')
    return result


class Supervisor:
    def __init__(self, root, manifest_path):
        self.root = root.resolve()
        self.manifest_path = manifest_path
        self.m = validate_manifest(root, manifest_path)
        self.output = bound_path(root, self.m['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest_sha = sha(manifest_path)
        self.children = {}
        self.stopping = False
        self.idle_since = None
        self.last_tick = time.time()
        self.journal = self.output / 'journal.jsonl'
        runtime.journal(self.journal, repair=True)
        self.state_path = self.output / 'state.json'
        if self.state_path.exists():
            self.s = read(self.state_path)
            require(self.s['manifest_sha256'] == self.manifest_sha, 'controller manifest changed')
        else:
            cycle = root / self.m['bootstrap_cycle']
            complete = read(cycle / 'complete.json')
            require(complete['dl_gp_dl_complete'] is True, 'recovered learning cycle must finish first')
            checkpoint = complete['second_checkpoint']
            verify_checkpoint(root, checkpoint)
            corpus = str((cycle / 'corpus-2').relative_to(root))
            verify_corpus(root, corpus, self.m['contexts_sha256'], self.m['vocabulary_sha256'])
            reconciliation = read(root / self.m['reconciliation'])
            require(reconciliation['passed'] is True and reconciliation['corpus_sha256'] == complete['teaching_corpus_sha256'],
                    'retained ledger/label reconciliation missing')
            self.s = dict(manifest_sha256=self.manifest_sha, checkpoint=checkpoint, corpus=corpus,
                          generations=[], snapshots=[], trained_snapshots=[], tasks={},
                          last_snapshot_at=time.time(), covered_searches=[], completed_blocks=0,
                          paid_gpu_idle_seconds=0.0)
            self.save('bootstrap', checkpoint=checkpoint, corpus=corpus)
        self.migrate_serving_state()

    def migrate_serving_state(self):
        if 'serving_checkpoint' not in self.s:
            checkpoint = self.s['checkpoint']
            verify_checkpoint(self.root, checkpoint)
            self.s.update(serving_checkpoint=checkpoint,
                          serving_checkpoint_sha256=sha(self.root / checkpoint / 'state.json'),
                          serving_status='provisional_unqualified', candidate_checkpoint=None,
                          candidates=[], promotion_decisions=[],
                          consumed_training_snapshots=list(self.s['trained_snapshots']))
            self.save('serving_migrated_provisionally', checkpoint=checkpoint,
                      qualification='not established; historical advancement was structural only')
        require(self.s['checkpoint'] == self.s['serving_checkpoint'], 'serving compatibility alias changed')
        require(sha(self.root / self.s['serving_checkpoint'] / 'state.json') == self.s['serving_checkpoint_sha256'],
                'serving checkpoint identity changed')
        if 'serving_corpus' not in self.s:
            corpus_sha = read(self.root / self.s['serving_checkpoint'] / 'state.json')['corpus_sha256']
            corpora = [self.s['corpus'], *[s['corpus'] for s in self.s['snapshots']]]
            serving_corpus = next((c for c in corpora if sha(self.root / c / 'index.json') == corpus_sha), None)
            require(serving_corpus is not None, 'serving checkpoint teaching corpus unavailable')
            self.s['serving_corpus'] = serving_corpus
            self.save('serving_corpus_bound', corpus=serving_corpus, corpus_sha256=corpus_sha)

    def intake_candidates(self):
        """A durable inbox connects isolated qualification to controller state."""
        processed=self.s.setdefault('candidate_submissions',{})
        for path in sorted((self.output/'candidate-inbox').glob('*.json')):
            digest=sha(path)
            if digest in processed:continue
            try:
                candidate=candidate_intake.validate(self.root,read(path),self.m['vocabulary_sha256'])
                verify_checkpoint(self.root,candidate['checkpoint'],candidate['checkpoint_sha256'])
                verify_corpus(self.root,candidate['corpus'],self.m['contexts_sha256'],self.m['vocabulary_sha256'])
            except (OSError,KeyError,ValueError,RuntimeError) as error:
                processed[digest]=dict(status='rejected',reason=str(error),path=str(path.relative_to(self.root)))
                self.save('candidate_submission_rejected',submission_sha256=digest,reason=str(error))
                continue
            promotion_gate.immutable(self.output/'candidates'/(candidate['checkpoint_sha256']+'.json'),candidate)
            if not any(row['checkpoint_sha256']==candidate['checkpoint_sha256'] for row in self.s['candidates']):
                self.s['candidates'].append(candidate)
            for key in ('trained_snapshots','consumed_training_snapshots'):
                if candidate['corpus'] not in self.s[key]:self.s[key].append(candidate['corpus'])
            self.s['candidate_checkpoint']=candidate['checkpoint']
            processed[digest]=dict(status='registered',checkpoint_sha256=candidate['checkpoint_sha256'])
            self.save('external_candidate_registered',submission_sha256=digest,checkpoint_sha256=candidate['checkpoint_sha256'])

    def promotion_work(self):
        """Commit immutable decisions before atomically projecting the serving pointer."""
        for candidate in self.s['candidates']:
            if candidate.get('qualification_status')=='accepted' and candidate['checkpoint_sha256']==self.s['serving_checkpoint_sha256']:
                continue
            directory = self.output / 'qualification' / candidate['checkpoint_sha256']
            processed={row['request_sha256'] for row in self.s['promotion_decisions']}
            requests=sorted(directory.glob('request*.json'),key=lambda path:(path.name!='request.json',path.name))
            request=next((path for path in requests if sha(path) not in processed),None)
            if request is None:
                continue
            request_sha = sha(request)
            if any(d['request_sha256'] == request_sha for d in self.s['promotion_decisions']):
                continue
            value = read(request)
            decision_path = directory / ('decision-' + request_sha + '.json')
            if decision_path.exists():
                decision = read(decision_path)
            else:
                decision = promotion_gate.evaluate(self.root, value.get('manifest'), value.get('evidence'),
                                                   candidate, self.s['serving_checkpoint_sha256'])
                promotion_gate.immutable(decision_path, decision)
            require(decision['checkpoint_sha256'] == candidate['checkpoint_sha256'], 'decision candidate changed')
            # A completed stale branch cannot overwrite a newer serving decision.
            if decision['outcome'] == 'accepted':
                require(decision['serving_before_sha256'] == self.s['serving_checkpoint_sha256'],
                        'promotion was compared against a different serving checkpoint')
                verified = promotion_gate.evaluate(self.root, value.get('manifest'), value.get('evidence'),
                                                   candidate, self.s['serving_checkpoint_sha256'])
                require(verified == decision, 'promotion evidence changed after decision publication')
                verify_checkpoint(self.root, candidate['checkpoint'], candidate['checkpoint_sha256'])
                self.s.update(checkpoint=candidate['checkpoint'], serving_checkpoint=candidate['checkpoint'],
                              serving_corpus=candidate['corpus'],
                              serving_checkpoint_sha256=candidate['checkpoint_sha256'], serving_status='qualified')
            candidate['qualification_status'] = decision['outcome']
            self.s['promotion_decisions'].append(dict(request_sha256=request_sha,
                receipt=str(decision_path.relative_to(self.root)), receipt_sha256=sha(decision_path),
                outcome=decision['outcome'], checkpoint_sha256=candidate['checkpoint_sha256']))
            self.save('promotion_committed', decision=str(decision_path.relative_to(self.root)),
                      decision_sha256=sha(decision_path), outcome=decision['outcome'])

    def save(self, event, **details):
        runtime.append(self.journal, dict(timestamp=time.time(), event=event, manifest_sha256=self.manifest_sha, **details))
        atomic(self.state_path, self.s)

    def paused(self):
        path = self.root / 'runs/continuous-control.json'
        return self.stopping or (path.exists() and read(path)['paused'])

    def qualification_handoff(self):
        path = self.root / 'runs/qualification-control.json'
        if not path.exists():
            return False
        request = read(path)
        if not request.get('active'):
            return False
        if (request.get('schema') != 'parallel.qualification_control.v1'
                or type(request.get('deadline')) not in (float, int)
                or type(request.get('requested_at')) not in (float, int)
                or not 0 < request['deadline'] - request['requested_at'] <= 7200):
            atomic(self.root / 'runs/qualification-handoff.json', dict(
                schema='parallel.qualification_handoff.v1', timestamp=time.time(),
                request_sha256=sha(path), ready=False, error='invalid or excessive qualification handoff'))
            return False
        active = request['active'] and request['requested_at'] <= time.time() < request['deadline']
        if active:
            pool_ready = True
            if self.m.get('gpu_pool'):
                pool_path = self.root / self.m['gpu_pool'] / 'pool.json'
                pool = read(pool_path) if pool_path.is_file() else {}
                pool_ready = pool.get('phase') == 'suspended' and 0 <= time.time() - pool.get('timestamp', 0) < 10
            atomic(self.root / 'runs/qualification-handoff.json', dict(
                schema='parallel.qualification_handoff.v1', timestamp=time.time(),
                request_sha256=sha(path), ready=pool_ready and not any(k in self.children for k in ('train', 'proposal', 'monitor')),
                serving_checkpoint=self.s['serving_checkpoint'],
                serving_checkpoint_sha256=self.s['serving_checkpoint_sha256']))
        return active

    def command(self, executable, *args):
        return [str(self.root / self.m[executable]), '--root', str(self.root), *map(str, args)]

    def new_task(self, kind, command, receipt, **details):
        require(kind not in self.s['tasks'], 'stage already scheduled')
        task = dict(kind=kind, command=command, receipt=receipt, attempts=0, retry_at=0,
                    created_at=time.time(), **details)
        if kind == 'train':
            task['execution'] = training_contract.bind_task(self.root, task, self.m['steps'],
                                                            self.s.get('experiment_training_head'))
            task['reservation'] = self.m.get('funded_task')
            row = Budget(self.output).verify(task['reservation'], self.training_task_identity(task))
            task['deadline'] = row['deadline']
            self.verify_training_hold(task)
            self.s['training_funding_status'] = 'verified reservation'
        self.s['tasks'][kind] = task
        self.save('stage_scheduled', task=task)

    def training_task_identity(self, task):
        execution = training_contract.bind_task(self.root, task, self.m['steps'],
                                                self.s.get('experiment_training_head'))
        require(task.get('execution', execution) == execution, 'training execution parent or increment changed')
        contract = dict(schema='parallel.funded_training_task.v2', capability='regression_training',
            controller_configuration_sha256=hashlib.sha256(promotion_gate.canonical(
                {key: value for key, value in self.m.items() if key not in ('funded_task', 'qualified_task_permit')})).hexdigest(),
            command=task['command'], receipt=task['receipt'],
            output=task['output'], corpus=task['corpus'], checkpoint=task['checkpoint'],
            corpus_sha256=sha(bound_path(self.root, task['corpus']) / 'index.json'),
            checkpoint_sha256=sha(bound_path(self.root, task['checkpoint']) / 'state.json'),
            world=8, microbatch=256, steps=self.m['steps'], execution=execution)
        return hashlib.sha256(promotion_gate.canonical(contract)).hexdigest()

    def verify_training_hold(self, task, consume=False):
        if not self.automatic_training_hold():
            require(not task.get('hold_permit'), 'original training hold disappeared; reconciliation required')
            return
        hold = dict(held=True, reason=self.s.get('training_hold_reason'),
                    diagnostics=self.s.get('diagnostic_journal_index', {}).get('seen', []))
        permit = task.get('hold_permit', self.m.get('qualified_task_permit'))
        executable = self.m.get('training_executable', self.m['neural_executable'])
        training_contract.verify_permit(self.root, self.output, permit,
            hold_sha256=training_contract.digest(hold), task_sha256=self.training_task_identity(task),
            release_sha256=self.m['bindings'][executable], execution=task['execution'],
            reservation=task['reservation'], deadline=task['deadline'], now=time.time(), consume=consume)
        task['hold_permit'] = permit

    def start_tasks(self):
        for kind, task in list(self.s['tasks'].items()):
            if self.stopping:
                return
            if kind not in self.s['tasks'] or task.get('quarantined'):
                continue
            if kind=='pack_verify' and self.quarantine_failed_verification(task):
                continue
            if kind in self.children or time.time() < task['retry_at']:
                continue
            identity = task.get('process')
            if identity and process_identity(identity['pid']) == identity:
                require(os.getpgid(identity['pid']) == identity['pid'], 'saved task process group changed')
                self.children[kind] = (SurvivingProcess(identity), None, None, task['started_at'])
                continue
            if (self.root / task['receipt']).exists():
                if kind == 'train' and not task.get('original_completion'):
                    task.update(quarantined=True, failure_reason='training exit unknown; original completion reconciliation required')
                    self.save('training_completion_unknown', task=task)
                    continue
                self.finish_task(kind)
                continue
            if kind == 'train':
                try:
                    Budget(self.output).verify(task.get('reservation'), self.training_task_identity(task))
                    self.verify_training_hold(task)
                except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
                    task.update(quarantined=True, failure_reason=str(error))
                    self.save('training_funding_denied', task=task)
                    continue
            if kind in ('train', 'proposal', 'monitor'):
                if self.qualification_handoff():
                    continue
                if self.m.get('gpu_pool'):
                    path=self.root/self.m['gpu_pool']/'pool.json'
                    pool=read(path) if path.is_file() else {}
                    if pool.get('phase')!='running' or not 0 <= time.time()-pool.get('timestamp',0)<10:
                        continue
            lease = None
            if kind in ('pack', 'pack_verify', 'export'):
                name = 'continuous-export.lock' if kind == 'export' else 'experiment-spare-cpus.lock'
                lease = (self.root / 'runs' / name).open('a')
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
            self.timed_stage('launch_input_verification',lambda:validate_manifest(self.root,self.manifest_path))
            env = child_environment(self.root, self.m)
            if kind in ('train', 'proposal', 'monitor'):
                # Freeze once in the durable task, rather than renewing its
                # ceiling after an interrupted launch or controller restart.
                task.setdefault('execution_deadline', min(
                    task.get('deadline', float('inf')),
                    task['created_at'] + self.m.get('stage_timeout_seconds', 7200)))
                require(time.time() < task['execution_deadline'], 'original GPU task deadline elapsed')
                env['JOPE_GPU_TASK_DEADLINE_MS'] = str(int(task['execution_deadline'] * 1000))
            command = list(task['command'])
            if kind == 'train':
                # Consume once before process creation. An ambiguous launch is
                # reconciled from original evidence, never automatically retried.
                self.verify_training_hold(task, consume=True)
                Budget(self.output).verify(task['reservation'], self.training_task_identity(task), consume=True)
                self.s['training_funding_status'] = 'reservation consumed by task; no new allowance'
            task['attempts'] += 1
            log_path = self.output / f"{kind}-{int(task['created_at'] * 1e6)}-attempt-{task['attempts']}.log"
            self.save('stage_started', kind=kind, command=command, log=str(log_path))
            log = log_path.open('xb')
            kwargs = {}
            if lease:
                kwargs.update(pass_fds=(lease.fileno(),), preexec_fn=lambda: os.sched_setaffinity(0, {60, 61}))
            if self.m.get('gpu_pool') and kind in ('train', 'proposal', 'monitor'):
                env['JOPE_GPU_POOL'] = str(self.root / self.m['gpu_pool'])
            child = runtime.popen(command, cwd=self.root, env=env, stdout=log, stderr=log, **kwargs)
            started = time.time()
            task.update(process=process_identity(child.pid), started_at=started,
                        last_progress_at=started, progress_value=None)
            self.children[kind] = (child, log, lease, started)
            self.save('stage_process_bound', kind=kind, process=task['process'])

    def poll_tasks(self):
        for kind, (child, log, lease, started) in list(self.children.items()):
            task = self.s['tasks'][kind]
            progress = meaningful_progress(self.root, task, self.output / 'packing-progress.json')
            if progress is not None and progress != task.get('progress_value'):
                task.update(progress_value=progress, last_progress_at=time.time())
            timeout = self.m.get('stage_timeout_seconds', 7200)
            expired = time.time() - started >= timeout
            if kind == 'train':
                expired = expired or time.time() >= task.get('deadline', 0)
            stalled = (kind in ('train', 'proposal', 'pack', 'pack_verify') and time.time() - task.get('last_progress_at', started)
                       >= self.m.get('progress_timeout_seconds', 600))
            expired = expired or stalled
            if child.poll() is None and not expired:
                continue
            if expired:
                runtime.stop([child])
            if log:
                log.close()
            if lease:
                lease.close()
            del self.children[kind]
            task = self.s['tasks'][kind]
            if kind in ('train', 'monitor'):
                task['original_completion'] = dict(process=task.get('process'), exit_code=child.returncode,
                    expired=expired, observed_at=time.time(), receipt_sha256=sha(self.root / task['receipt'])
                    if (self.root / task['receipt']).is_file() else None)
                self.save('original_process_completion', kind=kind, completion=task['original_completion'])
                if child.returncode != 0 or expired or isinstance(child, SurvivingProcess):
                    task.update(quarantined=True, failure_reason='failed or unknown original process completion')
                    self.save('original_completion_quarantined', task=task)
                    continue
            if kind=='pack_verify':
                failure=(self.root/task['receipt']).with_name('failure.json')
                unknown_exit=isinstance(child,SurvivingProcess) and child.returncode==-1
                if failure.exists() or (child.returncode!=0 and not unknown_exit and (self.root/task['receipt']).exists()):
                    task.update(quarantined=True,failure_receipt=str(failure.relative_to(self.root))
                                if failure.exists() else None,exit_code=child.returncode)
                    self.save('corpus_verification_quarantined',task=task)
                    continue
            if (self.root / task['receipt']).exists():
                self.finish_task(kind)
            else:
                task['retry_at'] = time.time() + min(300, 5 * 2 ** min(task['attempts'], 6))
                self.save('stage_failed', kind=kind, exit_code=child.returncode, timeout=expired,
                          stalled=stalled, retry_at=task['retry_at'], attempt=task['attempts'])

    def schedule_corpus_verification(self,task):
        if 'verification_request' not in task:
            task['verification_request']=corpus_verification.prepare(self.root,task,self.manifest_sha,
                self.m['contexts_sha256'],self.m['vocabulary_sha256'])
            task['verification_manifest_sha256']=self.manifest_sha
            self.save('corpus_verification_intent',output=task['output'],verification_manifest_sha256=self.manifest_sha)
        output=self.root/task['output']
        directory=output.parent/'verification'
        request_path=directory/'request.json';receipt=directory/'complete.json'
        request=task['verification_request']
        require(request['pack_identity']==corpus_verification.pack_identity(task)
                and request['controller_manifest_sha256']==task['verification_manifest_sha256'],
                'pending verification belongs to another pack')
        require(request['contexts_sha256']==self.m['contexts_sha256']
                and request['vocabulary_sha256']==self.m['vocabulary_sha256'],
                'verification cohort or vocabulary is incompatible with this controller')
        promotion_gate.immutable(request_path,request)
        if 'pack_verify' not in self.s['tasks']:
            self.new_task('pack_verify',['/usr/bin/python3','-B',request['verifier_path'],
                '--root',str(self.root),'--request',str(request_path),'--receipt',str(receipt),
                '--request-sha256',sha(request_path)],
                str(receipt.relative_to(self.root)),request=str(request_path.relative_to(self.root)),
                request_sha256=sha(request_path),output=task['output'])

    def quarantine_failed_verification(self,task):
        failure=(self.root/task['receipt']).with_name('failure.json')
        if not failure.exists():return False
        require(read(failure)['request_sha256']==sha(self.root/task['request']),
                'verification failure belongs to another request')
        task.update(quarantined=True,failure_receipt=str(failure.relative_to(self.root)))
        self.save('corpus_verification_quarantined',task=task)
        return True

    def commit_pack(self,task,index):
        require(not any(row['corpus']==task['output'] for row in self.s['snapshots']),
                'pack snapshot is already committed')
        incremental=self.m.get('teaching_store') and '--teaching-store' in task['command'] and 'sources' in task
        if incremental:
            supplied=task['command'][task['command'].index('--teaching-store')+1]
            require(Path(supplied).resolve()==(self.root/self.m['teaching_store']).resolve(),
                    'completed pack used another teaching store')
        self.s['snapshots'].append(dict(corpus=task['output'],index_sha256=sha(self.root/task['output']/'index.json'),
             records_sha256=index['records_sha256'],records=index['records'],timestamp=time.time()))
        self.s['covered_searches']=task['covered_searches']
        self.s['last_snapshot_at']=time.time()
        if incremental:
            self.s['teaching_sources']=task['sources']

    def finish_task(self, kind):
        task = self.s['tasks'][kind]
        receipt = self.root / task['receipt']
        if kind == 'pack':
            # A migrated controller must finish an already frozen verification
            # even if its fallback disables verification for subsequent packs.
            asynchronous='verification_request' in task or (self.m.get('async_corpus_verification',False)
                and read(self.root/task['output']/'index.json').get('schema')=='parallel.teaching_shards.v1')
            if asynchronous:
                self.schedule_corpus_verification(task)
                return
            index = verify_corpus(self.root, task['output'], self.m['contexts_sha256'], self.m['vocabulary_sha256'])
            self.commit_pack(task,index)
        elif kind=='pack_verify':
            require(not task.get('quarantined'),'failed corpus verification cannot commit')
            pack=self.s['tasks']['pack']
            request=pack['verification_request']
            require(request['contexts_sha256']==self.m['contexts_sha256']
                    and request['vocabulary_sha256']==self.m['vocabulary_sha256'],
                    'verification cohort or vocabulary is incompatible with this controller')
            index=corpus_verification.result(self.root,pack,pack['verification_manifest_sha256'],
                                             self.root/task['request'],receipt,task['request_sha256'])
            self.commit_pack(pack,index)
            del self.s['tasks']['pack']
        elif kind == 'export':
            self.s.setdefault('exported_snapshots', []).append(task['corpus'])
            self.s['last_export_at']=time.time()
        elif kind == 'report':
            self.s['generations'][task['generation']]['reported'] = True
        elif kind == 'monitor':
            value = read(receipt)
            require(value['binding']['checkpoint_sha256'] == task['checkpoint_sha256'], 'monitor checkpoint changed')
            self.s.setdefault('monitored', []).append(task['checkpoint_sha256'])
            if 'monitor_policy' in task:
                self.s.setdefault('monitored_receipts',[]).append(task['monitor_policy']+':'+task['checkpoint_sha256'])
            atomic(self.root / 'runs/frozen-monitoring.json', value)
            self.record_diagnostic(task, receipt)
        elif kind == 'train':
            completion = task.get('original_completion') or {}
            require(completion.get('exit_code') == 0 and completion.get('expired') is False
                    and completion.get('receipt_sha256') == sha(receipt), 'original training completion unavailable')
            latest = read(self.root / task['output'] / 'latest.json')
            state = verify_checkpoint(self.root, latest['checkpoint'], latest['state_sha256'])
            require(state['corpus_sha256'] == sha(self.root / task['corpus'] / 'index.json'), 'trained corpus changed')
            head = training_contract.completed_head(self.root, task, latest['checkpoint'],
                latest['state_sha256'], receipt, self.s.get('experiment_training_head'))
            promotion_gate.immutable(self.output / 'training-heads' / (training_contract.digest(head) + '.json'), head)
            candidate = dict(checkpoint=latest['checkpoint'], checkpoint_sha256=latest['state_sha256'],
                corpus=task['corpus'], corpus_sha256=state['corpus_sha256'],
                parent_checkpoint=task['checkpoint'],
                parent_checkpoint_sha256=sha(self.root / task['checkpoint'] / 'state.json'),
                training_receipt=task['receipt'], training_receipt_sha256=sha(receipt),
                vocabulary_sha256=self.m['vocabulary_sha256'],
                proposal_executable_sha256=self.m['bindings'][self.m['neural_executable']],
                qualification_status='inconclusive', reason='awaiting identity-valid matched qualification')
            candidate_path = self.output / 'candidates' / (latest['state_sha256'] + '.json')
            promotion_gate.immutable(candidate_path, candidate)
            parent_sha = sha(self.root / task['checkpoint'] / 'state.json')
            promotion_gate.immutable(self.output / 'qualification' / latest['state_sha256'] / 'request.json',
                promotion_gate.promotion_request(latest['state_sha256'],
                    promotion_gate.default_blocked_by(parent_sha)))
            self.s['candidates'].append(candidate)
            self.s.update(candidate_checkpoint=latest['checkpoint'], corpus=task['corpus'],
                          completed_blocks=self.s['completed_blocks'] + 1, experiment_training_head=head)
            self.s['trained_snapshots'].extend(s['corpus'] for s in self.s['snapshots']
                if s['timestamp'] <= next(x['timestamp'] for x in self.s['snapshots'] if x['corpus'] == task['corpus'])
                and s['corpus'] not in self.s['trained_snapshots'])
            self.s['consumed_training_snapshots'] = list(self.s['trained_snapshots'])
            Budget(self.output).complete(task['reservation']['identity'], receipt)
            self.s['training_funding_status'] = 'reservation completed; a new verified reservation is required'
        else:
            generation = self.s['generations'][task['generation']]
            if generation['stage'] < 2:
                spec = read(receipt.parent / 'run.json')
                require(spec['checkpoint_sha256'] == generation['checkpoint_sha256']
                        and spec['corpus_sha256'] == sha(self.root / generation['corpus'] / 'index.json'),
                        'proposal receipt binding changed')
            generation['stage'] += 1
        del self.s['tasks'][kind]
        self.save('stage_committed', kind=kind, receipt=task['receipt'], receipt_sha256=sha(receipt))

    def generation_work(self, top_up=True, refresh_inventory=True):
        if any(k in self.s['tasks'] for k in ('train', 'proposal', 'monitor')):
            return
        pending = next((g for g in self.s['generations'] if g['stage'] < 4), None)
        if pending is None and self.generation_mint_blocked():
            return
        if pending is None:
            # A preparation reserve is distinct from eligible or funded work.
            if refresh_inventory or 'proposal_input_inventory' not in self.s:
                self.s['proposal_input_inventory'] = proposal_input_inventory(
                    self.root, self.s['generations'], self.m, self.s.get('proposal_input_inventory'))
            inventory = self.s['proposal_input_inventory']
            if inventory['validated_proposal_metadata'] >= inventory['planned_reserve']:
                return
            if inventory.get('scan_limit_reached') or inventory.get('restart_required'):
                return  # The next tick resumes the metadata census.
            if not top_up and inventory['validated_proposal_metadata'] >= inventory['refill_priority_at']:
                return  # Two waves remain planned; let other stages schedule first.
            number = len(self.s['generations'])
            path = self.output / f'generation-{number:06d}'
            path.mkdir(exist_ok=True)
            checkpoint_sha = sha(self.root / self.s['checkpoint'] / 'state.json')
            verify_checkpoint(self.root, self.s['checkpoint'], checkpoint_sha)
            pending = dict(path=str(path.relative_to(self.root)), number=number, stage=0,
                 checkpoint=self.s['checkpoint'], checkpoint_sha256=checkpoint_sha, corpus=self.s['serving_corpus'],
                 contexts=self.m['contexts'], contexts_sha256=self.m['contexts_sha256'], rng_seed=self.m['rng_seed'],
                 cuts=self.m['cuts'], cut_start=self.m['cut_start'], evaluations=1000, seconds=600,
                 omit_shuffled=True, coverage_selection_protocol='family_dataset_cut_strict_v1')
            spec_path = path / 'generation.json'
            if spec_path.exists():
                require(read(spec_path) == pending, 'generation identity changed after interruption')
            else:
                atomic(spec_path, pending)
            self.s['generations'].append(pending)
            self.save('generation_frozen', generation=pending)
        number, path, stage = pending['number'], self.root / pending['path'], pending['stage']
        if stage == 1 and pending.get('omit_shuffled'):
            pending['stage'] = 2
            self.save('shuffled_arm_omitted', generation=number)
            stage = 2
        if stage < 2:
            arm = ('neural', 'shuffled')[stage]
            output = path / arm
            command = self.command('neural_executable', 'propose', '--corpus', self.root / pending['corpus'],
                 '--checkpoint', self.root / pending['checkpoint'], '--output', output,
                 '--cuts', pending['cuts'], '--cut-start', pending['cut_start'])
            if arm == 'shuffled':
                command.append('--shuffled')
            if (self.m.get('proposal_generation_protocol') == 'generation_seeded_v1'
                    or pending.get('coverage_selection_protocol')):
                command.extend(['--generation-manifest', str(path / 'generation.json')])
            receipt = output / 'complete.json'
        elif stage == 2:
            output = path / 'generic'
            command = self.command('cpu_executable', 'comparison-baseline', '--jobs', path / 'neural/jobs.json',
                  '--vocabulary', self.root / pending['corpus'] / 'vocabulary.json', '--output', output)
            receipt = output / 'jobs.json'
        else:
            arm_names = ('neural', 'generic') if pending.get('omit_shuffled') else ('neural', 'generic', 'shuffled')
            jobs = assemble_jobs(self.root, number, pending,
                 {arm: str((path / arm).relative_to(self.root)) for arm in arm_names})
            jobs_path = path / 'jobs.json'
            if jobs_path.exists():
                require(read(jobs_path) == jobs, 'generation jobs changed')
            else:
                atomic(jobs_path, jobs)
            command = self.command('cpu_executable', 'enqueue', '--jobs', jobs_path,
                 '--contexts', self.root / self.m['contexts'], '--campaign-contexts', self.root / self.m['contexts'])
            receipt = self.root / 'runs/queue-receipts' / (sha(jobs_path) + '.json')
        self.new_task('proposal', command, str(receipt.relative_to(self.root)), generation=number)

    def snapshot_work(self):
        if 'pack' in self.s['tasks']:
            return
        completed = {}
        for generation in self.s['generations']:
            if generation['stage'] != 4:
                continue
            for job in read(self.root / generation['path'] / 'jobs.json')['jobs']:
                path = self.root / job['output']
                if verified_search(self.root, job):
                    completed[job['output']] = path / 'observations.json'
        fresh = set(completed) - set(self.s['covered_searches'])
        due = (bool(fresh) and time.time() - self.s['last_snapshot_at'] >= 10
               if self.m.get('teaching_store') else
               snapshot_due(len(fresh), bool(fresh), self.s['last_snapshot_at'], time.time()))
        if not due:
            return
        number = len(self.s['snapshots'])
        output = self.output / f'snapshot-{number:06d}' / 'corpus'
        output.parent.mkdir(exist_ok=True)
        # Freeze the exact completed ledgers; no moving tournament barrier enters this pack.
        base = read(self.root / self.m['bootstrap_cycle'] / 'corpus-2/provenance.json')
        sources = list(dict.fromkeys(row['path'] for row in base['sources']))
        sources.extend(str(completed[key].relative_to(self.root)) for key in sorted(completed))
        if self.m.get('teaching_store'):
            sources = list(dict.fromkeys([*self.s.get('teaching_sources', []), *sources]))
        request = dict(sources=sources, covered_searches=sorted(completed),
                       new_completed_searches=len(fresh), contexts_sha256=self.m['contexts_sha256'])
        request_path = output.parent / 'snapshot-request.json'
        if request_path.exists():
            request = read(request_path)
            sources = request['sources']
        else:
            atomic(request_path, request)
        command = self.command('packing_executable' if 'packing_executable' in self.m else 'cpu_executable',
             'build-corpus', '--contexts', self.root / self.m['contexts'],
             '--vocabulary', self.root / self.s['corpus'] / 'vocabulary.json', '--output', output,
             '--progress', self.output / 'packing-progress.json', '--qualification')
        if self.m.get('teaching_store'):
            command.extend(['--teaching-store', str(self.root / self.m['teaching_store'])])
        else:
            command.extend(['--scan-cache', str(self.output / 'scan-cache')])
        atomic(output.parent / 'observations.json', [str(self.root / source) for source in sources])
        command.extend(['--observation-manifest', str(output.parent / 'observations.json')])
        self.new_task('pack', command, str((output / 'provenance.json').relative_to(self.root)),
                      output=str(output.relative_to(self.root)), covered_searches=request['covered_searches'], sources=sources)

    def export_work(self):
        if (not self.m.get('teaching_store') or 'export' in self.s['tasks'] or 'pack' in self.s['tasks']
                or time.time()-self.s.get('last_export_at',0)<self.m.get('legacy_export_interval_seconds',3600)):
            return
        pending = [s for s in self.s['snapshots'] if s['corpus'] not in self.s.get('exported_snapshots', [])
                   and read(self.root / s['corpus'] / 'index.json')['schema'] == 'parallel.teaching_shards.v1']
        if pending:
            corpus = pending[-1]['corpus']
            output = str(Path(corpus).with_name('legacy-export'))
            self.new_task('export', self.command('packing_executable' if 'packing_executable' in self.m else 'cpu_executable', 'export-teaching-corpus',
                '--corpus', self.root / corpus, '--output', self.root / output),
                output + '/provenance.json', corpus=corpus, output=output)

    def record_diagnostic(self, task, receipt):
        """Produce a guard event from an observed monitor of a nominated model."""
        candidate = next((row for row in self.s.get('candidates', [])
            if row['checkpoint_sha256'] == task['checkpoint_sha256']), None)
        if not candidate or not candidate.get('nomination') or not self.m.get('diagnostic_reference'):
            self.s['diagnostic_status'] = 'original nomination or frozen recovery monitor unavailable'
            return
        nomination = promotion_gate.artifact(self.root, candidate['nomination'])
        require(nomination.get('schema') == 'parallel.training_nomination_result.v1'
                and nomination.get('status') == 'nominated' and nomination.get('training_only') is True
                and nomination.get('checkpoint_sha256') == task['checkpoint_sha256'], 'diagnostic nomination changed')
        completion = task.get('original_completion') or {}
        require(completion.get('exit_code') == 0 and completion.get('expired') is False
                and completion.get('receipt_sha256') == sha(receipt), 'original monitor completion unavailable')
        value = read(receipt)
        reference = promotion_gate.artifact(self.root, self.m['diagnostic_reference'])
        require(value.get('schema') == reference.get('schema') == 'parallel.frozen_monitoring.v1', 'unsupported diagnostic monitor')
        for key in ('executable_sha256', 'corpus_sha256', 'panel_sha256'):
            require(value['binding'].get(key) and value['binding'][key] == reference['binding'].get(key),
                    'diagnostic monitor policy changed: ' + key)
        metrics = (('sequence_nll', .005), ('ranking_bt', .01), ('ranking_listwise', .01))
        require(all(promotion_gate.finite(value.get(key)) and promotion_gate.finite(reference.get(key))
                    and reference[key] > 0 for key, _ in metrics), 'missing or nonfinite diagnostic evidence')
        rejected = [key for key, tolerance in metrics if value[key] > reference[key] * (1 + tolerance)]
        evidence = dict(schema='parallel.training_diagnostic.v1', checkpoint_sha256=task['checkpoint_sha256'],
            outcome='rejected' if rejected else 'accepted', reasons=rejected,
            nomination=candidate['nomination'], reference=self.m['diagnostic_reference'],
            monitor=dict(path=str(receipt.relative_to(self.root)), sha256=sha(receipt)),
            original_completion=completion, policy=dict(metrics))
        path = self.output / 'diagnostics' / (task['checkpoint_sha256'] + '.json')
        promotion_gate.immutable(path, evidence)
        self.save('diagnostic_' + evidence['outcome'], checkpoint_sha256=task['checkpoint_sha256'],
            evidence=dict(path=str(path.relative_to(self.root)), sha256=sha(path)))

    def replay_diagnostics(self):
        """Read every committed event once per process; persist an audit index.

        A restart revalidates the original journal. Truncation, replacement,
        malformed records and anonymous old diagnostics require reconciliation.
        """
        if not self.journal.exists():
            return
        stat = self.journal.stat()
        index = getattr(self, '_diagnostic_index', None)
        if index is None:
            index = dict(offset=0, identity=[stat.st_dev, stat.st_ino], seen={}, rejects=[], records=0,
                         original_completions={})
        require(index['identity'] == [stat.st_dev, stat.st_ino] and stat.st_size >= index['offset'],
                'diagnostic journal replaced or truncated')
        if stat.st_size == index['offset']:
            require(index.get('mtime_ns', stat.st_mtime_ns) == stat.st_mtime_ns, 'diagnostic journal changed in place')
            return
        with self.journal.open('rb') as stream:
            stream.seek(index['offset'])
            for line in stream:
                require(line.endswith(b'\n'), 'incomplete diagnostic journal; reconciliation required')
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise RuntimeError('corrupt diagnostic journal; reconciliation required') from error
                require(isinstance(row, dict), 'invalid diagnostic journal record')
                event = row.get('event')
                if event == 'original_process_completion' and row.get('kind') == 'monitor':
                    completion = row.get('completion')
                    require(isinstance(completion, dict), 'invalid original monitor completion')
                    index['original_completions'][hashlib.sha256(promotion_gate.canonical(completion)).hexdigest()] = completion
                if event in ('diagnostic_rejected', 'diagnostic_accepted'):
                    checkpoint = row.get('checkpoint_sha256')
                    require(isinstance(checkpoint, str) and len(checkpoint) == 64 and row.get('evidence'),
                            'diagnostic checkpoint or original evidence missing')
                    evidence = promotion_gate.artifact(self.root, row['evidence'])
                    require(evidence.get('schema') == 'parallel.training_diagnostic.v1'
                            and evidence.get('checkpoint_sha256') == checkpoint
                            and event == 'diagnostic_' + evidence.get('outcome', ''), 'diagnostic evidence changed')
                    completion = evidence.get('original_completion')
                    require(isinstance(completion, dict) and completion.get('exit_code') == 0
                            and completion.get('expired') is False
                            and hashlib.sha256(promotion_gate.canonical(completion)).hexdigest() in index['original_completions'],
                            'diagnostic lacks original committed monitor completion')
                    nomination = promotion_gate.artifact(self.root, evidence['nomination'])
                    monitor = promotion_gate.artifact(self.root, evidence['monitor'])
                    reference = promotion_gate.artifact(self.root, evidence['reference'])
                    require(nomination.get('schema') == 'parallel.training_nomination_result.v1'
                            and nomination.get('status') == 'nominated' and nomination.get('training_only') is True
                            and nomination.get('checkpoint_sha256') == checkpoint
                            and monitor.get('schema') == reference.get('schema') == 'parallel.frozen_monitoring.v1'
                            and monitor['binding'].get('checkpoint_sha256') == checkpoint
                            and completion.get('receipt_sha256') == evidence['monitor']['sha256'],
                            'diagnostic nomination or monitor identity changed')
                    for key in ('executable_sha256', 'corpus_sha256', 'panel_sha256'):
                        require(monitor['binding'].get(key) and monitor['binding'][key] == reference['binding'].get(key),
                                'diagnostic policy binding changed')
                    metrics = (('sequence_nll', .005), ('ranking_bt', .01), ('ranking_listwise', .01))
                    require(all(promotion_gate.finite(monitor.get(key)) and promotion_gate.finite(reference.get(key))
                                and reference[key] > 0 for key, _ in metrics), 'invalid original diagnostic metric')
                    rejected = [key for key, tolerance in metrics if monitor[key] > reference[key] * (1 + tolerance)]
                    require(evidence.get('reasons') == rejected and evidence.get('policy') == dict(metrics)
                            and evidence['outcome'] == ('rejected' if rejected else 'accepted'),
                            'diagnostic outcome differs from original measurements')
                    previous = index['seen'].get(checkpoint)
                    require(previous is None or previous == row['evidence']['sha256'], 'conflicting checkpoint diagnostics')
                    if previous is None:
                        index['seen'][checkpoint] = row['evidence']['sha256']
                        if event == 'diagnostic_accepted':
                            index['rejects'] = []
                        else:
                            index['rejects'].append(checkpoint)
                            if len(index['rejects']) >= 3:
                                self.s['automatic_training_hold'] = True
                                self.s['training_hold_reason'] = 'three distinct nominated checkpoint diagnostic rejections'
                index['offset'] += len(line)
                index['records'] += 1
        index['mtime_ns'] = stat.st_mtime_ns
        self._diagnostic_index = index
        self.s['diagnostic_journal_index'] = dict(index)

    def generation_mint_blocked(self):
        """Hold latches new generations. In-flight searches may still finish."""
        if self.s.get('automatic_training_hold'):
            return True
        progress = self.root / 'runs/continuous-progress.json'
        if progress.exists():
            hold = read(progress).get('automatic_training_hold')
            if isinstance(hold, dict) and hold.get('status') == 'held' and hold.get('block_new_training'):
                return True
        return False

    def automatic_training_hold(self):
        """Diagnostic holds do not substitute for a verified task reservation."""
        self.replay_diagnostics()
        if self.s.get('automatic_training_hold'):
            return True
        progress = self.root / 'runs/continuous-progress.json'
        if progress.exists():
            hold = read(progress).get('automatic_training_hold')
            if isinstance(hold, dict) and hold.get('status') == 'held' and hold.get('block_new_training'):
                self.s['automatic_training_hold'] = True
                return True
        return False

    def training_work(self):
        if self.automatic_training_hold():
            # A retained hold can allow one prebound task. It never enables
            # automatic snapshot training or creates a funding reservation.
            task = self.m.get('qualified_training_task')
            if isinstance(task, dict) and not any(k in self.s['tasks'] for k in ('train', 'proposal', 'monitor')):
                try:
                    self.new_task('train', **task)
                except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
                    self.s['training_funding_status'] = str(error)
            return
        if not self.m.get('automatic_training', False):
            # Keep discoveries pending until the explicitly qualified replay
            # experiment can consume them. Waiting is not a training exposure.
            return
        if any(k in self.s['tasks'] for k in ('train', 'proposal', 'monitor')):
            return
        if not isinstance(self.m.get('funded_task'), dict):
            self.s['training_funding_status'] = 'valid task reservation required'
            return
        snapshots = [s for s in self.s['snapshots'] if s['corpus'] not in self.s['consumed_training_snapshots']]
        if not snapshots:
            return
        newest = snapshots[-1]
        current = read(self.root / self.s['corpus'] / 'index.json')
        next_index=read(self.root/newest['corpus']/'index.json')
        growth=reference_training_growth(current,next_index)
        if newest['records_sha256'] == current['records_sha256'] or growth==0:
            self.s['trained_snapshots'].extend(s['corpus'] for s in snapshots)
            self.s['consumed_training_snapshots'] = list(self.s['trained_snapshots'])
            self.save('unchanged_teaching_labels', snapshots=[s['corpus'] for s in snapshots],
                      newly_admitted_training_programs=growth)
            return
        output = self.output / f"cycle-{self.s['completed_blocks']:06d}" / 'train-2'
        command = self.command('training_executable' if 'training_executable' in self.m else 'neural_executable',
             'train', '--corpus', self.root / newest['corpus'],
             '--output', output, '--initialize', self.root / self.s['checkpoint'], '--world', 8,
             '--microbatch', 256, '--steps', self.m['steps'], '--checkpoint-steps', 1000)
        try:
            self.new_task('train', command, str((output / 'complete.json').relative_to(self.root)),
                          output=str(output.relative_to(self.root)), corpus=newest['corpus'], checkpoint=self.s['checkpoint'])
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
            self.s['training_funding_status'] = str(error)

    def monitoring_work(self):
        if any(k in self.s['tasks'] for k in ('train', 'proposal', 'monitor')):
            return
        checkpoints = [self.s['serving_checkpoint']] + [c['checkpoint'] for c in self.s['candidates']]
        policy=hashlib.sha256((self.m['bindings'][self.m['monitor_executable']]+':'+
            sha(self.root/self.m['monitor_panel'])+':'+sha(self.root/self.m['monitor_corpus']/'index.json')).encode()).hexdigest()
        checkpoint = next((cp for cp in checkpoints
                           if policy+':'+sha(self.root / cp / 'state.json') not in self.s.get('monitored_receipts', [])), None)
        if checkpoint is None:
            return
        checkpoint_sha = sha(self.root / checkpoint / 'state.json')
        output = self.output / 'monitoring' / (policy+'-'+checkpoint_sha + '.json')
        output.parent.mkdir(exist_ok=True)
        command = self.command('monitor_executable', 'monitor-checkpoint', '--corpus',
             self.root / self.m['monitor_corpus'], '--checkpoint', self.root / checkpoint,
             '--panel', self.root / self.m['monitor_panel'], '--output', output)
        self.new_task('monitor', command, str(output.relative_to(self.root)), checkpoint_sha256=checkpoint_sha,monitor_policy=policy)

    def reporting_work(self):
        if 'report' in self.s['tasks']:
            return

        for generation in self.s['generations']:
            if generation['stage'] != 4 or generation.get('reported'):
                continue
            path = self.root / generation['path']
            jobs = read(path / 'jobs.json')['jobs']
            if not all(verified_search(self.root, j) for j in jobs):
                continue
            self.new_task('report', ['/usr/bin/python3', str(Path(__file__).with_name('continuous_quality_report.py')),
                 '--root', str(self.root), '--generation', str(path)],
                 str((path / 'quality-report.json').relative_to(self.root)), generation=generation['number'])
            return

    def gpu_work(self):
        # A pending generation must reach enqueue without an intervening
        # training block. Each scheduler respects an already-owned GPU stage.
        self.generation_work(top_up=False)
        self.monitoring_work()
        self.training_work()
        # Top up two-to-three planned waves only if no other GPU stage scheduled.
        # These stages retain their own admission checks; metadata is not authority.
        self.generation_work(top_up=True, refresh_inventory=False)

    def progress(self):
        now = time.time()
        try:
            output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                             '--format=csv,noheader,nounits'], text=True, timeout=3)
            owners = [line.strip() for line in output.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            owners = None
        eligible = bool(self.s['tasks'] or self.s['snapshots'] or self.s['generations'])
        ranks=None if owners is None else 0
        if self.m.get('gpu_pool'):
            path=self.root/self.m['gpu_pool']/'pool.json'
            pool=read(path) if path.is_file() else {}
            if 0 <= now-pool.get('timestamp',0)<10:
                ranks=0
                for rank,pid in pool.get('pids',{}).items():
                    path=self.root/self.m['gpu_pool']/f'rank-{rank}.json'
                    row=read(path) if path.is_file() else {}
                    ranks+=int(row.get('pid')==pid and row.get('status')=='busy')
                legacy=any(k in self.children and self.s['tasks'][k]['command'][0]
                    !=str(self.root/self.m['training_executable']) for k in ('train','proposal','monitor'))
                if legacy and owners:
                    pool_pids=set(pool.get('pids',{}).values())
                    external={int(line.split(',',1)[0]) for line in owners if line.split(',',1)[0].strip().isdigit()}-pool_pids
                    ranks=min(8,ranks+len(external))
            else:ranks=None
        elif owners:
            if any(k in self.children for k in ('train','proposal')):ranks=8
            elif 'monitor' in self.children:ranks=1
        try:
            values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3)
            kernel=[float(v) for v in values.splitlines() if v.strip()]
            kernel_mean=sum(kernel)/len(kernel) if kernel else None
        except (OSError,ValueError,subprocess.SubprocessError):kernel_mean=None
        if ranks == 0 and not self.paused():
            self.idle_since = self.idle_since or now
            self.s['paid_gpu_idle_seconds'] += max(0, now - self.last_tick) * 8
        else:
            self.idle_since = None
        idle = now - self.idle_since if self.idle_since else 0
        gp_path = self.root / 'runs/gp-service.json'
        gp = read(gp_path) if gp_path.exists() else {}
        explanation = idle_explanation(self.s['tasks'], self.children, gp, now)
        if not self.m.get('automatic_training',False) and not any(k in self.s['tasks'] for k in ('train','proposal','monitor','pack')):
            explanation = 'automatic training awaits qualified replay; GP collection continues'
        if self.qualification_handoff():
            explanation = ('bounded qualification owns the GPU handoff; '
                           f"GP active={gp.get('active', 0)}, queued={gp.get('queued', 0)}")
        occupancy=gpu_occupancy.update(self.s.setdefault('gpu_occupancy_history',{}),now,ranks,
            eligible and not self.paused() and not self.qualification_handoff(),kernel_mean,explanation)
        inactivity=idle_status(idle,eligible,explanation is not None)
        if inactivity=='incident' and not self.s.get('open_idle_incident'):
            self.s['open_idle_incident']=dict(started_at=self.idle_since,detected_at=now)
            self.save('unexplained_gpu_idle_incident',**self.s['open_idle_incident'])
        elif ranks and self.s.pop('open_idle_incident',None):
            self.save('gpu_idle_incident_resolved',timestamp_resolved=now)
        serving_sha = self.s.get('serving_checkpoint_sha256')
        same_parent = serving_sha == RECOVERY_PARENT
        data = dict(schema='parallel.continuous_progress.v1', timestamp=int(now), pid=os.getpid(),
             phase='paused' if self.paused() else 'running', complete_learning_cycle=False,
             output=str(self.output.relative_to(self.root)), packing_progress=str((self.output / 'packing-progress.json').relative_to(self.root)),
             last_committed_checkpoint=self.s['checkpoint'], completed_training_blocks=self.s['completed_blocks'],
             serving_checkpoint=self.s['serving_checkpoint'], serving_status=self.s['serving_status'],
             parent_checkpoint_sha256=serving_sha,
             recovery_parent_sha256=RECOVERY_PARENT,
             training_blocks_from_same_parent=self.s['completed_blocks'] if same_parent else None,
             checkpoint_ever_advanced=bool(serving_sha and serving_sha != RECOVERY_PARENT),
             promotion_producer=False,
             hold_is_latch=True,
             rmse_scale='raw_cut_normalized_target',
             claimed_savings_745=False,
             restart_774452=False,
             automatic_training=(self.m.get('automatic_training',False) and not self.s.get('automatic_training_hold')
                                  and isinstance(self.m.get('funded_task'),dict)
                                  and self.s.get('training_funding_status') == 'verified reservation'),
             training_hold_reason=(self.s.get('training_hold_reason', 'consecutive diagnostic rejects')
                                  if self.s.get('automatic_training_hold')
                                  else self.s.get('training_funding_status', 'valid task reservation required')
                                  if self.m.get('automatic_training',False)
                                  else 'automatic training is held'),
             training_funding_status=self.s.get('training_funding_status', 'unverified'),
             training_task_reservation=self.s['tasks'].get('train',{}).get('reservation'),
             training_task_deadline=self.s['tasks'].get('train',{}).get('deadline'),
             diagnostic_status=self.s.get('diagnostic_status', 'original diagnostic evidence unavailable'),
             candidate_checkpoint=self.s['candidate_checkpoint'],
             promotion_decisions=len(self.s['promotion_decisions']),
             next_scheduled_work=list(self.s['tasks']), active_gpu_owners=owners,
             gpu_idle_seconds=int(idle), paid_gpu_idle_seconds=int(self.s['paid_gpu_idle_seconds']),
             idle_explanation=explanation,gpu_task_ranks=ranks,gpu_occupancy=occupancy,
             inactivity=idle_status(idle, eligible, explanation is not None), manifest_sha256=self.manifest_sha)
        data['quarantined_tasks']={kind:task.get('failure_receipt') for kind,task in self.s['tasks'].items()
                                   if task.get('quarantined')}
        data['controller_stage_timings']=getattr(self,'stage_timings',[])
        data['proposal_input_inventory'] = self.s.get('proposal_input_inventory')
        atomic(self.root / 'runs/continuous-progress.json', data)
        self.last_tick = now
        atomic(self.state_path, self.s)

    def timed_stage(self,name,operation):
        """Attribute polling gaps without treating a heartbeat as task progress."""
        started=time.monotonic()
        row=dict(stage=name,started_at=time.time(),pid=os.getpid(),manifest_sha256=self.manifest_sha)
        path=self.output/'controller-stage.json'
        atomic(path,dict(row,status='running'))
        try:
            value=operation()
        except BaseException as error:
            atomic(path,dict(row,status='failed',seconds=time.monotonic()-started,error=str(error)))
            raise
        row.update(status='complete',seconds=time.monotonic()-started)
        self.stage_timings=[*getattr(self,'stage_timings',[]),row][-32:]
        atomic(path,row)
        return value

    def run(self):
        def stop(signum, frame):
            self.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while not self.stopping:
                loop_started = time.monotonic()
                self.timed_stage('poll_tasks',self.poll_tasks)
                self.timed_stage('candidate_intake',self.intake_candidates)
                self.timed_stage('promotion_evidence',self.promotion_work)
                if not self.paused():
                    self.timed_stage('snapshot_discovery',self.snapshot_work)
                    self.timed_stage('legacy_export',self.export_work)
                    self.timed_stage('reporting',self.reporting_work)
                    if not self.qualification_handoff():
                        self.timed_stage('gpu_scheduling',self.gpu_work)
                    self.timed_stage('start_tasks',self.start_tasks)
                elif self.children:
                    runtime.stop([v[0] for v in self.children.values()])
                self.progress()
                time.sleep(max(0, 10 - (time.monotonic() - loop_started)))
        finally:
            runtime.stop([v[0] for v in self.children.values()])
            self.save('controller_stopped', resumable=True)
            self.progress()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--pause', action='store_true')
    actions.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    m = validate_manifest(args.root, args.manifest)
    if args.check:
        print(json.dumps(dict(valid=True, manifest_sha256=sha(args.manifest), bootstrap_cycle=m['bootstrap_cycle'])))
        return
    output = bound_path(args.root, m['output'])
    if args.pause or args.resume:
        output.mkdir(parents=True, exist_ok=True)
        atomic(args.root / 'runs/continuous-control.json', dict(schema='parallel.continuous_control.v1', paused=args.pause, timestamp=time.time(), manifest_sha256=sha(args.manifest)))
        return
    with (args.root / 'runs/continuous-learning.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime._controller_fds = (lock.fileno(),)
        Supervisor(args.root, args.manifest).run()


if __name__ == '__main__':
    main()
