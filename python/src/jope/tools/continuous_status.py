#!/usr/bin/env python3
"""Observe continuous learning from saved receipts; never schedule or promote work."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


METRICS = ('sequence_nll', 'ranking_bt', 'ranking_listwise', 'runtime_huber')
GP_UNIT = 'jope-parallel-gp.service'
CONTROLLER_UNIT = 'jope-parallel-campaign.service'
MIGRATION_OVERRIDE = Path('/etc/systemd/system') / (GP_UNIT + '.d') / 'zzzzzzzz-registry-migration-fence.conf'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def systemd_properties(unit):
    fields=('MainPID','ControlPID','ActiveState','SubState','ControlGroup','Restart','DropInPaths','Requires')
    command=['systemctl','show',unit]
    for field in fields:command.extend(['-p',field])
    return dict(line.split('=',1) for line in subprocess.check_output(command,text=True,timeout=5).splitlines())


def systemd_conditions(unit):
    # systemctl renders this property as [unprintable]; use its structured D-Bus value.
    name=''.join(character if character.isascii() and character.isalnum() else '_'+format(ord(character),'02x')
                 for character in unit)
    result=json.loads(subprocess.check_output(['busctl','--json=short','get-property',
        'org.freedesktop.systemd1','/org/freedesktop/systemd1/unit/'+name,
        'org.freedesktop.systemd1.Unit','Conditions'],text=True,timeout=5))
    require(result['type']=='a(sbbsi)','unknown systemd condition representation')
    return result['data']


def process_identity(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,birth=fields[19],state=fields[0])
    except FileNotFoundError:
        return None


def cgroup_pids(group):
    require(group.startswith('/system.slice/') and '..' not in Path(group).parts,'unsafe service cgroup')
    directory=Path('/sys/fs/cgroup')/group.lstrip('/')
    return sorted({int(pid) for path in directory.rglob('cgroup.procs') for pid in path.read_text().split()})


def boot_identity():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def observed_gp_status(receipt, now):
    """Expose current activity only while its original publisher lease is valid."""
    result = dict(receipt)
    binding = receipt.get('status_binding') or {}
    if not isinstance(binding, dict):
        binding = {}
    timestamp, expires = receipt.get('timestamp'), receipt.get('expires_at')
    fresh = (type(timestamp) is int and type(expires) is int
             and timestamp <= now < expires and expires == timestamp + 10)
    progress = receipt.get('last_progress_at')
    activity_fresh = type(progress) is int and progress <= now < progress + 10
    alive = False
    try:
        original = binding['process']
        pid = original['pid']
        require(type(pid) is int and pid > 0, 'invalid publisher PID')
        current = process_identity(pid)
        metadata = (Path('/proc') / str(pid) / 'exe').stat()
        executable = dict(device=metadata.st_dev, inode=metadata.st_ino, bytes=metadata.st_size,
            mtime=metadata.st_mtime_ns // 1_000_000_000, mtime_ns=metadata.st_mtime_ns % 1_000_000_000,
            ctime=metadata.st_ctime_ns // 1_000_000_000, ctime_ns=metadata.st_ctime_ns % 1_000_000_000)
        alive = (current is not None and current['state'] not in ('Z', 'X')
                 and int(current['birth']) == original['birth_ticks']
                 and boot_identity() == original['boot_id']
                 and executable == binding['executable'])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    bound = (binding.get('schema') == 'parallel.status_binding.v1'
             and all(isinstance(binding.get(key), str) and len(binding[key]) == 64
                     for key in ('controller_generation', 'executable_sha256'))
             and receipt.get('executable_sha256', binding.get('executable_sha256')) == binding.get('executable_sha256'))
    activity = ('terminal' if receipt.get('terminal') is True else
                'current' if fresh and activity_fresh and alive and bound else 'stale')
    result['current_activity'] = activity
    if activity != 'current':
        result.setdefault('last_recorded_activity', {key: receipt.get(key) for key in
            ('phase', 'active', 'running_evaluations', 'evaluations_per_second', 'active_rows')})
        if activity == 'stale':
            result['phase'] = 'stale'
        result.update(active=None, running_evaluations=None, evaluations_per_second=None, active_rows=[])
    return result


def verified_migration_fence(root,now,override=MIGRATION_OVERRIDE):
    """A saved fence assertion never overrides evidence of a live dispatcher."""
    if not override.exists():return None
    try:
        content=override.read_text()
        condition=[line.split('=',1)[1] for line in content.splitlines() if line.startswith('ConditionPathExists=')]
        require(len(condition)==1,'migration override has no unique start condition')
        permit=within(root,condition[0]);directory=permit.parent
        require(permit.name=='REGISTRY_HANDOFF_REQUIRED' and directory.is_relative_to(root/'runs'),
                'migration start condition is outside its private run')
        require(not permit.exists() and not permit.is_symlink(),'migration start condition is satisfied or redirected')
        completion_path=directory/'complete.json';request_path=directory/'request.json';frozen_path=directory/'frozen.json'
        completion=read(completion_path);request=read(request_path);snapshot=read(frozen_path)
        require(completion['schema']=='parallel.legacy_fit_fence_completion.v1'
            and completion['legacy_dispatch_fenced'] is True and completion['descendants_reconciled'] is True
            and completion['registry_dispatch_enabled'] is False,'migration completion is incomplete or superseded')
        require(completion['request_sha256']==digest(request_path) and completion['frozen_sha256']==digest(frozen_path)
            and snapshot['request_sha256']==digest(request_path)
            and read(directory/'kill-intent.json')['frozen_sha256']==digest(frozen_path),
            'migration original evidence binding changed')
        require(request['schema']=='parallel.legacy_fit_fence.v1' and request['unit']==GP_UNIT
            and Path(request['root']).resolve()==root and request['boot']==boot_identity(),
            'migration service, workspace or boot identity changed')
        require(Path(completion['permanent_override'])==override and completion['override_sha256']==digest(override)
            and content==f'# Registry migration fence: {directory}\n[Unit]\nConditionPathExists={permit}\n[Service]\nRestart=no\n',
            'migration override changed')
        require(finite(completion['timestamp']) and completion['timestamp']<=now+5,'migration completion time is invalid')
        group='/system.slice/'+GP_UNIT
        require(request['properties']['ControlGroup']==snapshot['cgroup']==group and bool(snapshot['processes']),
            'migration original process scope is absent')
        require(any(p['pid']==request['pid'] and p['birth']==request['process']['birth'] for p in snapshot['processes']),
            'migration original dispatcher is absent from frozen capture')
        for original in snapshot['processes']:
            current=process_identity(original['pid'])
            require(current is None or current['birth']!=original['birth'] or current['state'] in ('Z','X'),
                    'an original fenced descendant is still alive')
        current=systemd_properties(GP_UNIT)
        require(current['MainPID']=='0' and current['ControlPID']=='0'
            and current['ActiveState'] in ('inactive','failed') and current['Restart']=='no'
            and current['ControlGroup'] in ('',group),'legacy dispatcher is active or restartable')
        require(str(override) in current['DropInPaths'].split(),'migration override is not loaded by systemd')
        require(any(len(row)==5 and row[:4]==['ConditionPathExists',False,False,str(permit)]
                    for row in systemd_conditions(GP_UNIT)),
                'systemd does not enforce the original non-negated start condition')
        require(not cgroup_pids(group),'legacy cgroup has surviving processes')
        controller=systemd_properties(CONTROLLER_UNIT)
        controller_group='/system.slice/'+CONTROLLER_UNIT
        controller_stopped=(GP_UNIT in controller['Requires'].split() and controller['MainPID']=='0'
            and controller['ControlPID']=='0' and controller['ActiveState'] in ('inactive','failed')
            and controller['ControlGroup'] in ('',controller_group) and not cgroup_pids(controller_group))
        return dict(verified=True,phase='planned_migration',unit=GP_UNIT,receipt=str(completion_path.relative_to(root)),
            receipt_sha256=digest(completion_path),fenced_at=completion['timestamp'],
            fence_age_seconds=max(0,now-completion['timestamp']),gp_properties=current,
            controller_properties=controller,controller_stopped_by_dependency=controller_stopped,
            historical_import_complete=completion['historical_import_complete'],registry_dispatch_enabled=False,
            collection_active=False,useful_gpu_work_established=False)
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        return dict(verified=False,phase='migration_fence_unverified',reason=str(error))


def progress_ages(progress,gp,teacher,reporter,now):
    def age(value):return max(0,now-value) if finite(value) and value<=now+5 else None
    pending=reporter.get('pending_searches')
    return dict(controller_heartbeat_seconds=age(progress.get('timestamp')),
        gp_heartbeat_seconds=age(gp.get('timestamp')),teacher_receipt_seconds=age(teacher.get('timestamp')),
        teacher_reporter_heartbeat_seconds=age(reporter.get('timestamp')),
        oldest_unprocessed_receipt_seconds=0 if pending==0 else age(reporter.get('oldest_pending_completed_at')),
        seconds_since_meaningful_improvement=age(teacher.get('last_meaningful_improvement_at')))


def within(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root), 'receipt path outside workspace')
    return path


def last_metric(path):
    """Ignore only an unfinished final write, never a malformed committed line."""
    if not path.exists():
        return None
    with path.open('rb') as stream:
        offset = max(0, os.fstat(stream.fileno()).st_size - 65536)
        stream.seek(offset)
        lines = stream.read().split(b'\n')[:-1]
    if offset:
        lines = lines[1:]
    return json.loads(lines[-1]) if lines else None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def better(a, b):
    return a < b and not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def teacher_health(teacher, manifest_sha, now):
    if (teacher.get('schema') != 'parallel.continuous_teacher_progress.v1'
            or teacher.get('manifest_sha256') != manifest_sha
            or not -5 <= now - teacher.get('timestamp', 0) <= 180):
        return 'teacher_reporter_stale', 'continuous teacher receipt missing, outdated or bound to another run'
    if teacher.get('source_stale'):
        return 'teacher_source_stale', 'completed generations have not received quality reports for over 180 seconds'
    return None


def gp_queue_health(gp, paused, now, previous):
    if paused or gp.get('active', 0) or gp.get('queued', 0):
        return None, None
    since = previous.get('gp_queue_empty_since')
    since = since if since is not None and since <= now else now
    seconds = max(0, now - since)
    return since, dict(code='gp_queue_empty',
        severity='incident' if seconds >= 120 else 'warning',
        detail=f'no running or queued searches for {seconds:.0f}s; collection is idle')


def compare_monitoring(baseline, receipts):
    keys = ('panel_sha256', 'corpus_sha256', 'executable_sha256')
    identity = {key: baseline['binding'][key] for key in keys}
    for receipt in [baseline, *receipts]:
        require(receipt['schema'] == 'parallel.frozen_monitoring.v1', 'unknown monitor schema')
        require(all(receipt['binding'][k] == identity[k] for k in keys), 'incomparable monitor bindings')
        require(receipt['sampled_examples'] == baseline['sampled_examples']
                and receipt['family_balanced'] is True, 'incomparable monitor sampling')
        require(receipt['final_tests_opened'] is False and receipt['independent_evaluation'] is False
                and receipt['training_selection_used'] is False, 'monitor evaluation scope changed')
        require(all(finite(receipt[k]) for k in METRICS), 'nonfinite monitoring loss')
    require(bool(receipts), 'no completed continuous monitoring receipts')
    receipts = sorted(receipts, key=lambda r: (r['timestamp'], r['checkpoint']))
    def compact(row):
        return {k: row[k] for k in ('checkpoint', 'timestamp', *METRICS)}
    latest = receipts[-1]
    return dict(binding=identity, baseline=compact(baseline), latest=compact(latest),
                best_nll=compact(min(receipts, key=lambda r: r['sequence_nll'])),
                compared_checkpoints=len(receipts),
                delta_from_baseline={k: latest[k] - baseline[k] for k in METRICS},
                regressed_metrics=[k for k in METRICS if better(baseline[k], latest[k])],
                history=[compact(r) for r in receipts],
                independent_evaluation=False, training_selection_used=False, final_tests_opened=False)


def summarize_quality(report, jobs):
    require(report['schema'] == 'parallel.continuous_quality.v1', 'unknown quality schema')
    require(report['final_tests_opened'] is False and report['independent_evaluation'] is False,
            'quality evaluation scope changed')
    expected = {j['output']: j for j in jobs['jobs']}
    rows = report['rows']
    require(len(expected) == len(jobs['jobs']) == len(rows)
            and {r['output'] for r in rows} == set(expected), 'missing or duplicate quality rows')
    groups = {}
    for row in rows:
        require(row['rng_seed'] == expected[row['output']]['rng_seed'], 'quality RNG binding changed')
        require(row['partition'] in ('meta_train', 'meta_validation')
                and row['metric'] == 'rmse' and row['units'] == 'original_target', 'quality scope changed')
        for key in ('best_final_gp', 'best_retained_teacher', 'best_initial_neural_seed'):
            require(row[key] is None or finite(row[key]), 'invalid quality metric')
        key = (row['dataset'], row['family'], row['partition'], row['cut'])
        arms = groups.setdefault(key, {})
        require(row['arm'] not in arms, 'duplicate comparison arm')
        arms[row['arm']] = row
    summary = {}
    for (_, _, partition, _), arms in groups.items():
        require(set(arms) == {'neural', 'generic', 'shuffled'}, 'incomplete matched triplet')
        require(len({r['rng_seed'] for r in arms.values()}) == 1, 'unmatched comparison RNG')
        counts = summary.setdefault(partition, dict(triplets=0, comparable=0, missing_final=0,
            neural_exclusive_wins=0, neural_best_ties=0, neural_losses=0,
            direct_seed_teacher_pairs=0, direct_seed_beats_teacher=0,
            final_teacher_pairs=0, neural_final_beats_teacher=0))
        counts['triplets'] += 1
        neural = arms['neural']
        n, g, s = [arms[a]['best_final_gp'] for a in ('neural', 'generic', 'shuffled')]
        if all(v is not None for v in (n, g, s)):
            counts['comparable'] += 1
            control = min(g, s)
            key = ('neural_exclusive_wins' if better(n, control) else
                   'neural_losses' if better(control, n) else 'neural_best_ties')
            counts[key] += 1
        else:
            counts['missing_final'] += 1
        teacher = neural['best_retained_teacher']
        direct = neural['best_initial_neural_seed']
        if teacher is not None and direct is not None:
            counts['direct_seed_teacher_pairs'] += 1
            counts['direct_seed_beats_teacher'] += better(direct, teacher)
        if teacher is not None and n is not None:
            counts['final_teacher_pairs'] += 1
            counts['neural_final_beats_teacher'] += better(n, teacher)
    return summary


def monitoring_series(baseline, receipts):
    """Retain old executable series without comparing their losses across policies."""
    current=[];archived={}
    for row in receipts:
        require(all(row['binding'][key]==baseline['binding'][key] for key in ('panel_sha256','corpus_sha256')),
                'incomparable monitoring panel/corpus')
        if row['binding']['executable_sha256']==baseline['binding']['executable_sha256']:
            current.append(row)
        else:archived.setdefault(row['binding']['executable_sha256'],[]).append(row)
    result=compare_monitoring(baseline,current or [baseline])
    result['archived_policies']=[compare_monitoring(min(rows,key=lambda row:row['timestamp']),rows)
                                for _,rows in sorted(archived.items())]
    return result


def observe(root, manifest_path, baseline_path, previous):
    now = time.time()
    manifest = read(manifest_path)
    manifest_sha = digest(manifest_path)
    run = within(root, manifest['output'])
    state = read(run / 'state.json')
    progress = read(root / 'runs/continuous-progress.json')
    gp = read(root / 'runs/gp-service.json')
    current_gp = observed_gp_status(gp, now)
    control = read(root / 'runs/continuous-control.json')
    require(state['manifest_sha256'] == progress['manifest_sha256'] == control['manifest_sha256']
            == manifest_sha, 'active manifest binding mismatch')
    require(progress['output'] == manifest['output'], 'progress belongs to another run')
    migration=verified_migration_fence(root,now)
    planned_migration=bool(migration and migration['verified'])
    controller_stopped=planned_migration and migration['controller_stopped_by_dependency']
    alerts = []
    def alert(code, detail, severity='warning'):
        alerts.append(dict(code=code, severity=severity, detail=detail))
    teacher_path = root / 'runs/teacher-progress.json'
    teacher = read(teacher_path) if teacher_path.exists() else {}
    reporter_path=root/'runs/teacher-reporter.json'
    reporter=read(reporter_path) if reporter_path.exists() else {}
    ages=progress_ages(progress,gp,teacher,reporter,now)
    if migration and not planned_migration:
        alert('migration_fence_unverified',migration['reason'],'incident')
    elif planned_migration:
        alert('planned_registry_migration','legacy fitting is fenced; collection is stopped and learning progress is not established','info')
    teacher_alert = teacher_health(teacher, manifest_sha, now)
    if teacher_alert:
        alert(*teacher_alert)
    for name, receipt in (('controller', progress), ('gp', gp)):
        age = now - receipt['timestamp']
        if age >= (10 if name == 'gp' else 30) or age < 0:
            expected_stop=age>30 and ((name=='gp' and planned_migration) or (name=='controller' and controller_stopped))
            alert(name + ('_receipt_before_migration' if expected_stop else '_heartbeat'),
                  f'last receipt age {age:.1f}s'+('; service is verifiably stopped' if expected_stop else ''),
                  'info' if expected_stop else 'incident')
    if not planned_migration and current_gp['current_activity'] == 'stale':
        alert('gp_activity_unknown', 'GP publisher is missing, expired or belongs to another process/release', 'incident')
    if not planned_migration and current_gp['current_activity'] == 'current' and (gp['phase'] not in ('running', 'waiting') or gp['failed_views'] or gp.get('retry_pending', 0)):
        alert('gp_health', f"phase={gp['phase']}, failures={gp['failed_views']}, retries={gp.get('retry_pending', 0)}", 'incident')
    if gp['executable_sha256'] != manifest['bindings'][manifest['cpu_executable']]:
        alert('gp_executable_changed', 'GP is not using the manifest-bound executable', 'incident')
    if control['paused']:
        alert('paused', 'continuous control is durably paused')
    if progress.get('inactivity') and not controller_stopped:
        alert('controller_inactivity', str(progress['inactivity']), progress['inactivity'])
    tasks = {k: {f: v.get(f) for f in ('attempts', 'created_at', 'retry_at', 'output', 'receipt')}
             for k, v in state['tasks'].items()}
    for kind, task in tasks.items():
        if task['attempts'] > 1:
            alert('stage_retry', f"{kind} attempt {task['attempts']}")
        if now - task['created_at'] > manifest.get('stage_timeout_seconds', 7200):
            alert('stage_overdue', kind, 'incident')
    train = None
    if 'train' in tasks:
        path = within(root, tasks['train']['output']) / 'metrics.jsonl'
        value = last_metric(path)
        if value:
            require(value['run'] == tasks['train']['output'], 'training telemetry belongs to another run')
            train = {k: value.get(k) for k in ('run', 'step', 'steps', 'timestamp', 'world', 'microbatch')}
            if now - value['timestamp'] > 300:
                alert('training_stalled', 'no completed metric line for over 300s', 'incident')
        elif now - tasks['train']['created_at'] > 300:
            alert('training_startup_stalled', 'no training metric after 300s', 'incident')
    packing = read(run / 'packing-progress.json') if (run / 'packing-progress.json').exists() else None
    if packing and 'pack' in tasks and not packing.get('complete') and now - packing['last_progress_at'] > 300:
        alert('packing_stalled', 'no recorded packing progress for over 300s')
    baseline = read(baseline_path)
    panel = read(within(root, manifest['monitor_panel']))
    require(baseline['binding']['panel_sha256'] == panel['panel_sha256']
            and baseline['binding']['corpus_sha256'] == panel['corpus_sha256']
            and baseline['binding']['executable_sha256'] == manifest['bindings'][manifest['monitor_executable']],
            'baseline differs from active monitoring policy')
    receipts = [read(p) for p in (run / 'monitoring').glob('*.json')]
    monitoring = monitoring_series(baseline, receipts)
    for row in [baseline, *receipts]:
        require(digest(within(root, row['checkpoint']) / 'state.json') == row['binding']['checkpoint_sha256'],
                'monitor checkpoint state binding changed')
    if monitoring['regressed_metrics']:
        alert('fixed_panel_regression', ', '.join(monitoring['regressed_metrics']))
    if (monitoring['latest']['checkpoint'] != (state.get('candidate_checkpoint') or state['checkpoint']) and 'monitor' not in tasks
            and now - monitoring['latest']['timestamp'] > 900):
        alert('monitoring_lag', 'committed checkpoint lacks a recent fixed-panel result')
    quality = []
    for generation in state['generations']:
        if not generation.get('reported'):
            continue
        directory = within(root, generation['path'])
        report = read(directory / 'quality-report.json')
        require(report['jobs_sha256'] == digest(directory / 'jobs.json'), 'quality jobs binding changed')
        quality.append(dict(generation=generation['number'], checkpoint=generation['checkpoint'],
                            partitions=summarize_quality(report, read(directory / 'jobs.json'))))
    gpu_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                          '--format=csv,noheader,nounits'], text=True, timeout=5)
    gpus = []
    for line in gpu_output.splitlines():
        index, utilization, memory = (int(v.strip()) for v in line.split(','))
        gpus.append(dict(index=index, utilization_percent=utilization, memory_mib=memory))
    if len(gpus) != manifest['world']:
        alert('gpu_count', f"expected {manifest['world']}, observed {len(gpus)}", 'incident')
    comparable = previous.get('manifest_sha256') == manifest_sha and previous.get('schema') == 'parallel.continuous_status.v1'
    queue_empty_since, queue_alert = gp_queue_health(gp, control['paused'] or planned_migration
        or current_gp['current_activity'] != 'current', now, previous if comparable else {})
    if queue_alert:
        alerts.append(queue_alert)
    before = previous.get('gp', {}) if comparable else {}
    last_completion = (previous.get('gp_last_completion_at', now)
                       if before.get('completed_views') == gp['completed_views'] else now)
    if not control['paused'] and not planned_migration and current_gp.get('active') and now - last_completion > 1200:
        alert('gp_no_completions', 'workers active but no view completed for over 20 minutes', 'incident')
    elapsed = now - previous.get('timestamp', now) if comparable else 0
    delta = None
    if elapsed > 0:
        delta = dict(seconds=elapsed, completed_views=gp['completed_views'] - before.get('completed_views', gp['completed_views']),
                     training_blocks=state['completed_blocks'] - previous.get('completed_training_blocks', state['completed_blocks']))
    latest = state['snapshots'][-1] if state['snapshots'] else None
    gp_status={k:current_gp.get(k) for k in ('timestamp','phase','active','queued','completed_views',
        'completed_evaluations','failed_views','executable_sha256','current_activity',
        'last_recorded_activity','status_binding','expires_at','terminal','termination_reason')}
    if planned_migration:
        gp_status.update(last_receipt=dict(gp),phase='planned_migration',active=0,
            counter_scope='last recorded values; queue and failures are historical during migration')
    incidents=any(a['severity']=='incident' for a in alerts)
    return dict(schema='parallel.continuous_status.v1', timestamp=now, manifest_sha256=manifest_sha,
        gp_queue_empty_since=queue_empty_since,
        health='incident' if incidents else 'planned_migration' if planned_migration else 'warning' if alerts else 'healthy',
        alerts=alerts, controller_pid=0 if controller_stopped else progress['pid'], paused=control['paused'],
        controller_last_recorded_pid=progress['pid'],migration=migration,progress_ages=ages,
        completed_training_blocks=state['completed_blocks'], checkpoint=state['checkpoint'], tasks=tasks,
        serving_checkpoint=state.get('serving_checkpoint', state['checkpoint']),
        serving_status=state.get('serving_status', 'legacy_unqualified'),
        candidate_checkpoint=state.get('candidate_checkpoint'),
        promotion=dict(decisions=state.get('promotion_decisions', []),
                       pending=sum(c['qualification_status'] == 'inconclusive' for c in state.get('candidates', []))),
        teacher_progress=teacher,
        scientific_status='qualified_proposer' if state.get('serving_status') == 'qualified' else 'transfer_gain_not_established',
        operational_status='planned_migration' if planned_migration else 'incident' if incidents else 'running',
        learning_status='controller_stopped_for_migration' if controller_stopped else 'not_established_by_health_receipt',
        useful_gpu_work_established=False,
        training=train, gp=gp_status, gp_last_completion_at=last_completion,
        delta_since_previous=delta, gpus=gpus, packing=packing, latest_snapshot=latest,
        snapshots=len(state['snapshots']), trained_snapshots=len(state['trained_snapshots']),
        new_records_since_first_snapshot=latest['records'] - state['snapshots'][0]['records'] if latest else 0,
        monitoring=monitoring, quality=quality, reported_generations=len(quality),
        pending_generations=len(state['generations']) - len(quality),
        comparison_tolerance=dict(relative=1e-9, absolute=1e-12),
        scope='Saved adaptive evaluation receipts; no fits, promotion, queue or controller writes.',
        final_tests_opened=False)


def publish(output, value):
    temporary = output / f'.latest-{os.getpid()}.json'
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output / 'latest.json')
    compact = {k: v for k, v in value.items() if k not in ('monitoring', 'quality', 'packing', 'tasks')}
    compact['latest_losses'] = value.get('monitoring', {}).get('latest')
    with (output / 'history.jsonl').open('a') as stream:
        stream.write(json.dumps(compact, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(args.output / 'latest.json') if (args.output / 'latest.json').exists() else {}
        try:
            value = observe(args.root.resolve(), args.manifest, args.baseline, previous)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            value = dict(schema='parallel.continuous_status_error.v1', timestamp=time.time(), health='incident',
                         alerts=[dict(code='observation_failed', severity='incident', detail=str(error))])
        publish(args.output, value)
        print(json.dumps({k: value.get(k) for k in ('timestamp', 'health', 'alerts', 'completed_training_blocks')}))
        return 1 if value['schema'] == 'parallel.continuous_status_error.v1' else 0


if __name__ == '__main__':
    raise SystemExit(main())
