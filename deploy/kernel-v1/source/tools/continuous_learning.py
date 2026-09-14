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

import experiment_runtime as runtime
from resume_expired_cycle import atomic, read, require, sha


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


def verify_corpus(root, corpus, contexts_sha, vocabulary_sha):
    path = bound_path(root, corpus)
    index = read(path / 'index.json')
    provenance = read(path / 'provenance.json')
    require(index['schema'] == 'parallel.teaching_corpus.v1' and index['qualification'] is True,
            'unrecognized teaching corpus')
    require(index['context_index_sha256'] == contexts_sha and index['vocabulary_sha256'] == vocabulary_sha,
            'corpus cohort or vocabulary changed')
    require(sha(path / 'index.json') == provenance['index_sha256'], 'corpus index changed')
    require(sha(path / 'records.bin') == index['records_sha256'], 'corpus records changed')
    require(sha(path / 'labels.jsonl') == provenance['labels_sha256'], 'corpus labels changed')
    require(sha(path / 'vocabulary.json') == vocabulary_sha, 'corpus vocabulary changed')
    require((path / 'records.bin').stat().st_size == index['records'] * index['record_bytes'], 'truncated corpus')
    require(all(g['partition'] in ('meta_train', 'meta_validation') for g in index['groups']), 'sealed supervision')
    return index


def snapshot_due(new_searches, changed, last_snapshot, now):
    return changed and (new_searches >= 104 or now - last_snapshot >= 600)


def idle_status(seconds, eligible, explained):
    if not eligible or explained:
        return None
    return 'incident' if seconds >= 300 else 'warning' if seconds >= 120 else None


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

    def save(self, event, **details):
        runtime.append(self.journal, dict(timestamp=time.time(), event=event, manifest_sha256=self.manifest_sha, **details))
        atomic(self.state_path, self.s)

    def paused(self):
        path = self.root / 'runs/continuous-control.json'
        return self.stopping or (path.exists() and read(path)['paused'])

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
            if kind in self.children or time.time() < task['retry_at']:
                continue
            if (self.root / task['receipt']).exists():
                self.finish_task(kind)
                continue
            lease = None
            if kind == 'pack':
                lease = (self.root / 'runs/experiment-spare-cpus.lock').open('a')
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
            validate_manifest(self.root, self.manifest_path)
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
            child = runtime.popen(command, cwd=self.root, env=env, stdout=log, stderr=log, **kwargs)
            self.children[kind] = (child, log, lease, time.time())

    def poll_tasks(self):
        for kind, (child, log, lease, started) in list(self.children.items()):
            timeout = self.m.get('stage_timeout_seconds', 7200)
            expired = time.time() - started >= timeout
            if child.poll() is None and not expired:
                continue
            if expired:
                runtime.stop([child])
            log.close()
            if lease:
                lease.close()
            del self.children[kind]
            task = self.s['tasks'][kind]
            if child.returncode == 0 and (self.root / task['receipt']).exists():
                self.finish_task(kind)
            else:
                task['retry_at'] = time.time() + min(300, 5 * 2 ** min(task['attempts'], 6))
                self.save('stage_failed', kind=kind, exit_code=child.returncode, timeout=expired,
                          retry_at=task['retry_at'], attempt=task['attempts'])

    def finish_task(self, kind):
        task = self.s['tasks'][kind]
        receipt = self.root / task['receipt']
        if kind == 'pack':
            index = verify_corpus(self.root, task['output'], self.m['contexts_sha256'], self.m['vocabulary_sha256'])
            self.s['snapshots'].append(dict(corpus=task['output'], index_sha256=sha(self.root / task['output'] / 'index.json'),
                 records_sha256=index['records_sha256'], records=index['records'], timestamp=time.time()))
            self.s['covered_searches'] = task['covered_searches']
            self.s['last_snapshot_at'] = time.time()
        elif kind == 'report':
            self.s['generations'][task['generation']]['reported'] = True
        elif kind == 'monitor':
            value = read(receipt)
            require(value['binding']['checkpoint_sha256'] == task['checkpoint_sha256'], 'monitor checkpoint changed')
            self.s.setdefault('monitored', []).append(task['checkpoint_sha256'])
            atomic(self.root / 'runs/frozen-monitoring.json', value)
        elif kind == 'train':
            latest = read(self.root / task['output'] / 'latest.json')
            state = verify_checkpoint(self.root, latest['checkpoint'], latest['state_sha256'])
            require(state['corpus_sha256'] == sha(self.root / task['corpus'] / 'index.json'), 'trained corpus changed')
            require(state['step'] >= self.m['steps'], 'training block incomplete')
            self.s.update(checkpoint=latest['checkpoint'], corpus=task['corpus'], completed_blocks=self.s['completed_blocks'] + 1)
            self.s['trained_snapshots'].extend(s['corpus'] for s in self.s['snapshots']
                if s['timestamp'] <= next(x['timestamp'] for x in self.s['snapshots'] if x['corpus'] == task['corpus'])
                and s['corpus'] not in self.s['trained_snapshots'])
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
        remaining = sum(not (self.root / j['output'] / 'search.json').exists()
                        for g in self.s['generations'] if g['stage'] == 4
                        for j in read(self.root / g['path'] / 'jobs.json')['jobs'])
        if pending is None:
            if remaining >= 104:
                return
            number = len(self.s['generations'])
            path = self.output / f'generation-{number:06d}'
            path.mkdir(exist_ok=True)
            checkpoint_sha = sha(self.root / self.s['checkpoint'] / 'state.json')
            verify_checkpoint(self.root, self.s['checkpoint'], checkpoint_sha)
            pending = dict(path=str(path.relative_to(self.root)), number=number, stage=0,
                 checkpoint=self.s['checkpoint'], checkpoint_sha256=checkpoint_sha, corpus=self.s['corpus'],
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
                if (path / 'search.json').exists():
                    completed[job['output']] = path / 'observations.json'
        fresh = set(completed) - set(self.s['covered_searches'])
        if not snapshot_due(len(fresh), bool(fresh), self.s['last_snapshot_at'], time.time()):
            return
        number = len(self.s['snapshots'])
        output = self.output / f'snapshot-{number:06d}' / 'corpus'
        output.parent.mkdir(exist_ok=True)
        # Freeze the exact completed ledgers; no moving tournament barrier enters this pack.
        base = read(self.root / self.m['bootstrap_cycle'] / 'corpus-2/provenance.json')
        sources = list(dict.fromkeys(row['path'] for row in base['sources']))
        sources.extend(str(completed[key].relative_to(self.root)) for key in sorted(completed))
        request = dict(sources=sources, covered_searches=sorted(completed),
                       new_completed_searches=len(fresh), contexts_sha256=self.m['contexts_sha256'])
        request_path = output.parent / 'snapshot-request.json'
        if request_path.exists():
            request = read(request_path)
            sources = request['sources']
        else:
            atomic(request_path, request)
        command = self.command('cpu_executable', 'build-corpus', '--contexts', self.root / self.m['contexts'],
             '--vocabulary', self.root / self.s['corpus'] / 'vocabulary.json', '--output', output,
             '--scan-cache', self.output / 'scan-cache', '--progress', self.output / 'packing-progress.json', '--qualification')
        atomic(output.parent / 'observations.json', [str(self.root / source) for source in sources])
        command.extend(['--observation-manifest', str(output.parent / 'observations.json')])
        self.new_task('pack', command, str((output / 'provenance.json').relative_to(self.root)),
                      output=str(output.relative_to(self.root)), covered_searches=request['covered_searches'])

    def training_work(self):
        if any(k in self.s['tasks'] for k in ('train', 'proposal', 'monitor')):
            return
        snapshots = [s for s in self.s['snapshots'] if s['corpus'] not in self.s['trained_snapshots']]
        if not snapshots:
            return
        newest = snapshots[-1]
        current = read(self.root / self.s['corpus'] / 'index.json')
        if newest['records_sha256'] == current['records_sha256']:
            self.s['trained_snapshots'].extend(s['corpus'] for s in snapshots)
            self.save('unchanged_teaching_labels', snapshots=[s['corpus'] for s in snapshots])
            return
        output = self.output / f"cycle-{self.s['completed_blocks']:06d}" / 'train-2'
        command = self.command('neural_executable', 'train', '--corpus', self.root / newest['corpus'],
             '--output', output, '--initialize', self.root / self.s['checkpoint'], '--world', 8,
             '--microbatch', 256, '--steps', self.m['steps'], '--checkpoint-steps', 1000)
        self.new_task('train', command, str((output / 'complete.json').relative_to(self.root)),
                      output=str(output.relative_to(self.root)), corpus=newest['corpus'], checkpoint=self.s['checkpoint'])

    def monitoring_work(self):
        if any(k in self.s['tasks'] for k in ('train', 'proposal', 'monitor')):
            return
        checkpoint_sha = sha(self.root / self.s['checkpoint'] / 'state.json')
        if checkpoint_sha in self.s.get('monitored', []):
            return
        output = self.output / 'monitoring' / (checkpoint_sha + '.json')
        output.parent.mkdir(exist_ok=True)
        command = self.command('monitor_executable', 'monitor-checkpoint', '--corpus',
             self.root / self.m['monitor_corpus'], '--checkpoint', self.root / self.s['checkpoint'],
             '--panel', self.root / self.m['monitor_panel'], '--output', output)
        self.new_task('monitor', command, str(output.relative_to(self.root)), checkpoint_sha256=checkpoint_sha)

    def reporting_work(self):
        if 'report' in self.s['tasks']:
            return
        for generation in self.s['generations']:
            if generation['stage'] != 4 or generation.get('reported'):
                continue
            path = self.root / generation['path']
            jobs = read(path / 'jobs.json')['jobs']
            if not all((self.root / j['output'] / 'search.json').exists() for j in jobs):
                continue
            self.new_task('report', ['/usr/bin/python3', str(Path(__file__).with_name('continuous_quality_report.py')),
                 '--root', str(self.root), '--generation', str(path)],
                 str((path / 'quality-report.json').relative_to(self.root)), generation=generation['number'])
            return

    def progress(self):
        now = time.time()
        try:
            output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                             '--format=csv,noheader,nounits'], text=True, timeout=3)
            owners = [line.strip() for line in output.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            owners = None
        eligible = bool(self.s['tasks'] or self.s['snapshots'] or self.s['generations'])
        if owners == [] and not self.paused():
            self.idle_since = self.idle_since or now
            self.s['paid_gpu_idle_seconds'] += max(0, now - self.last_tick) * 8
        else:
            self.idle_since = None
        idle = now - self.idle_since if self.idle_since else 0
        explained = 'pack' in self.children and 'train' not in self.s['tasks']
        data = dict(schema='parallel.continuous_progress.v1', timestamp=int(now), pid=os.getpid(),
             phase='paused' if self.paused() else 'running', complete_learning_cycle=False,
             output=str(self.output.relative_to(self.root)), packing_progress=str((self.output / 'packing-progress.json').relative_to(self.root)),
             last_committed_checkpoint=self.s['checkpoint'], completed_training_blocks=self.s['completed_blocks'],
             next_scheduled_work=list(self.s['tasks']), active_gpu_owners=owners,
             gpu_idle_seconds=int(idle), paid_gpu_idle_seconds=int(self.s['paid_gpu_idle_seconds']),
             idle_explanation='packing fresh targets' if explained else None,
             inactivity=idle_status(idle, eligible, explained), manifest_sha256=self.manifest_sha)
        atomic(self.root / 'runs/continuous-progress.json', data)
        self.last_tick = now
        atomic(self.state_path, self.s)

    def run(self):
        def stop(signum, frame):
            self.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while not self.stopping:
                self.poll_tasks()
                if not self.paused():
                    self.snapshot_work()
                    self.reporting_work()
                    self.monitoring_work()
                    self.training_work()
                    self.generation_work()
                    self.start_tasks()
                elif self.children:
                    runtime.stop([v[0] for v in self.children.values()])
                self.progress()
                time.sleep(2)
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
