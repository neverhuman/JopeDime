#!/usr/bin/env python3
"""Continue verified learning checkpoints with durable GP, snapshot and training stages."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import struct
import teaching_manifest

import experiment_runtime as runtime
import promotion_gate
import gpu_occupancy
import candidate_intake
import corpus_verification
from resume_expired_cycle import atomic, read, require, sha
from flywheel_plan import consecutive_diagnostic_hold


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


def assemble_jobs(root, generation, spec, arms):
    """Bind each new identity; matched arms use the same search RNG and budgets."""
    neural, generic, shuffled = [read(root / arms[name] / 'jobs.json') for name in ('neural', 'generic', 'shuffled')]
    require(len(neural['jobs']) == len(generic['jobs']) == len(shuffled['jobs']), 'unmatched tournament arms')
    result = dict(schema='parallel.gp_jobs.v1', qualification=True, vocabulary=neural['vocabulary'], jobs=[])
    cohort = read(root / spec['contexts'])['encodings']
    eligible = {(c['prepared'], c['cut']): c for c in cohort if c['partition'] in ('meta_train', 'meta_validation')}
    for index, triple in enumerate(zip(neural['jobs'], generic['jobs'], shuffled['jobs'])):
        key = (triple[0]['prepared'], triple[0]['cut'])
        require(key in eligible, 'tournament cut outside frozen eligible cohort')
        seed = int.from_bytes(hashlib.sha256(f"{spec['rng_seed']}:{generation}:{key}".encode()).digest()[:8], 'little')
        for arm, original in zip(('neural', 'generic', 'shuffled'), triple):
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
        self.s['tasks'][kind] = task
        self.save('stage_scheduled', task=task)

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
                self.finish_task(kind)
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
            command = list(task['command'])
            if kind == 'train' and (self.root / task['output'] / 'run.json').exists():
                command.append('--resume')
            task['attempts'] += 1
            log_path = self.output / f"{kind}-{int(task['created_at'] * 1e6)}-attempt-{task['attempts']}.log"
            self.save('stage_started', kind=kind, command=command, log=str(log_path))
            log = log_path.open('xb')
            kwargs = {}
            if lease:
                kwargs.update(pass_fds=(lease.fileno(),), preexec_fn=lambda: os.sched_setaffinity(0, {60, 61}))
            env = dict(os.environ, JOPE_GP_EXECUTABLE=str(self.root / self.m['cpu_executable']))
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
        elif kind == 'train':
            latest = read(self.root / task['output'] / 'latest.json')
            state = verify_checkpoint(self.root, latest['checkpoint'], latest['state_sha256'])
            require(state['corpus_sha256'] == sha(self.root / task['corpus'] / 'index.json'), 'trained corpus changed')
            require(state['step'] >= self.m['steps'], 'training block incomplete')
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
            promotion_gate.immutable(self.output / 'qualification' / latest['state_sha256'] / 'request.json',
                                     dict(manifest=None, evidence=None, checkpoint_sha256=latest['state_sha256']))
            self.s['candidates'].append(candidate)
            self.s.update(candidate_checkpoint=latest['checkpoint'], corpus=task['corpus'],
                          completed_blocks=self.s['completed_blocks'] + 1)
            self.s['trained_snapshots'].extend(s['corpus'] for s in self.s['snapshots']
                if s['timestamp'] <= next(x['timestamp'] for x in self.s['snapshots'] if x['corpus'] == task['corpus'])
                and s['corpus'] not in self.s['trained_snapshots'])
            self.s['consumed_training_snapshots'] = list(self.s['trained_snapshots'])
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

    def generation_work(self):
        if any(k in self.s['tasks'] for k in ('train', 'proposal', 'monitor')):
            return
        pending = next((g for g in self.s['generations'] if g['stage'] < 4), None)
        if pending is None:
            # Keep three worker waves available while training and proposals
            # share the GPUs. Refill before another training block can drain it.
            reserve = self.m['gp_workers'] * (1 if 'pack' in self.s['tasks'] else 3)
            remaining = 0
            # Only the reserve threshold affects this decision. Recent waves
            # usually satisfy it without reopening the whole search history.
            for generation in reversed(self.s['generations']):
                if generation['stage'] != 4:
                    continue
                for job in read(self.root / generation['path'] / 'jobs.json')['jobs']:
                    remaining += not (self.root / job['output'] / 'search.json').exists()
                    if remaining >= reserve:
                        return
            number = len(self.s['generations'])
            path = self.output / f'generation-{number:06d}'
            path.mkdir(exist_ok=True)
            checkpoint_sha = sha(self.root / self.s['checkpoint'] / 'state.json')
            verify_checkpoint(self.root, self.s['checkpoint'], checkpoint_sha)
            pending = dict(path=str(path.relative_to(self.root)), number=number, stage=0,
                 checkpoint=self.s['checkpoint'], checkpoint_sha256=checkpoint_sha, corpus=self.s['serving_corpus'],
                 contexts=self.m['contexts'], contexts_sha256=self.m['contexts_sha256'], rng_seed=self.m['rng_seed'],
                 cuts=self.m['cuts'], cut_start=self.m['cut_start'], evaluations=1000, seconds=600)
            spec_path = path / 'generation.json'
            if spec_path.exists():
                require(read(spec_path) == pending, 'generation identity changed after interruption')
            else:
                atomic(spec_path, pending)
            self.s['generations'].append(pending)
            self.save('generation_frozen', generation=pending)
        number, path, stage = pending['number'], self.root / pending['path'], pending['stage']
        if stage < 2:
            arm = ('neural', 'shuffled')[stage]
            output = path / arm
            command = self.command('neural_executable', 'propose', '--corpus', self.root / pending['corpus'],
                 '--checkpoint', self.root / pending['checkpoint'], '--output', output,
                 '--cuts', self.m['cuts'], '--cut-start', self.m['cut_start'])
            if arm == 'shuffled':
                command.append('--shuffled')
            if self.m.get('proposal_generation_protocol') == 'generation_seeded_v1':
                command.extend(['--generation-manifest', str(path / 'generation.json')])
            receipt = output / 'complete.json'
        elif stage == 2:
            output = path / 'generic'
            command = self.command('cpu_executable', 'comparison-baseline', '--jobs', path / 'neural/jobs.json',
                  '--vocabulary', self.root / pending['corpus'] / 'vocabulary.json', '--output', output)
            receipt = output / 'jobs.json'
        else:
            jobs = assemble_jobs(self.root, number, pending,
                 {arm: str((path / arm).relative_to(self.root)) for arm in ('neural', 'generic', 'shuffled')})
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

    def automatic_training_hold(self):
        """Deployed diagnostic hold: three distinct frozen-guard rejects stop trains."""
        if self.s.get('automatic_training_hold'):
            return True
        if self.m.get('funded_task') is False:
            return True
        progress = self.root / 'runs/continuous-progress.json'
        if progress.exists():
            hold = read(progress).get('automatic_training_hold')
            if isinstance(hold, dict) and hold.get('status') == 'held' and hold.get('block_new_training'):
                self.s['automatic_training_hold'] = True
                return True
        journal = getattr(self, 'journal', None)
        events = []
        if journal and Path(journal).exists():
            for line in Path(journal).read_text().splitlines()[-256:]:
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if consecutive_diagnostic_hold(events):
            self.s['automatic_training_hold'] = True
            return True
        return False

    def training_work(self):
        if self.automatic_training_hold():
            return
        if not self.m.get('automatic_training', True):
            # Keep discoveries pending until the explicitly qualified replay
            # experiment can consume them. Waiting is not a training exposure.
            return
        if any(k in self.s['tasks'] for k in ('train', 'proposal', 'monitor')):
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
        self.new_task('train', command, str((output / 'complete.json').relative_to(self.root)),
                      output=str(output.relative_to(self.root)), corpus=newest['corpus'], checkpoint=self.s['checkpoint'])

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
        self.generation_work()
        self.monitoring_work()
        self.training_work()

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
        if not self.m.get('automatic_training',True) and not any(k in self.s['tasks'] for k in ('train','proposal','monitor','pack')):
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
        data = dict(schema='parallel.continuous_progress.v1', timestamp=int(now), pid=os.getpid(),
             phase='paused' if self.paused() else 'running', complete_learning_cycle=False,
             output=str(self.output.relative_to(self.root)), packing_progress=str((self.output / 'packing-progress.json').relative_to(self.root)),
             last_committed_checkpoint=self.s['checkpoint'], completed_training_blocks=self.s['completed_blocks'],
             serving_checkpoint=self.s['serving_checkpoint'], serving_status=self.s['serving_status'],
             automatic_training=self.m.get('automatic_training',True) and not self.s.get('automatic_training_hold'),
             training_hold_reason=('consecutive diagnostic rejects' if self.s.get('automatic_training_hold')
                                  else None if self.m.get('automatic_training',True)
                                  else 'strict fresh replay has not been qualified'),
             candidate_checkpoint=self.s['candidate_checkpoint'],
             promotion_decisions=len(self.s['promotion_decisions']),
             next_scheduled_work=list(self.s['tasks']), active_gpu_owners=owners,
             gpu_idle_seconds=int(idle), paid_gpu_idle_seconds=int(self.s['paid_gpu_idle_seconds']),
             idle_explanation=explanation,gpu_task_ranks=ranks,gpu_occupancy=occupancy,
             inactivity=idle_status(idle, eligible, explanation is not None), manifest_sha256=self.manifest_sha)
        data['quarantined_tasks']={kind:task.get('failure_receipt') for kind,task in self.s['tasks'].items()
                                   if task.get('quarantined')}
        data['controller_stage_timings']=getattr(self,'stage_timings',[])
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
