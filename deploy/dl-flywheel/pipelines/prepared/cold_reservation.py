"""Certify one pinned reservation that never passed the durable optimizer gate."""
import json, sqlite3, subprocess
from pathlib import Path
from jope_flywheel.program_io import bound, read, need, sha
from jope_flywheel.protocol import canonical

EXECUTION = 'fff71ac0903aad2b3fb2327db95da53595d797e368dd895d03af069c678c70db'
TRACE = '64dc72040a782d2899b00a6930cab3f6d95935db1eb717e0ea4dedf45d2322fb'
RUNNER = '7b43bae91cd8301f5f00fe2da61e331fe8b421f91c0500dd98958c19b951878c'
BATCH = '61ee8cf44f949bead8c88e82821cd8cbb104f7a0fa13265da4ebcbd8fe61b331'
STEP = 3298
EVENTS = 119
UNITS = {'jopedime-neural-continuous-20261009-v1', 'jopedime-neural-continuous-resume3-20261009'}


def stopped(owner):
    need(owner['unit'] in UNITS and owner['original_pid'] > 0, 'own failed writers only')
    actual = dict(x.split('=', 1) for x in subprocess.check_output(
        ['systemctl', '--user', 'show', owner['unit'], '--property=MainPID,ActiveState,InvocationID'], text=True, timeout=10).splitlines())
    need(actual['MainPID'] == '0' and actual['ActiveState'] in ('failed', 'inactive'), 'physical writer stopped')
    path=Path(owner['control_group'])
    need(path.is_absolute()and '..'not in path.parts and str(path).startswith('/user.slice/'),'absolute pinned user cgroup path')
    group = Path('/sys/fs/cgroup') / owner['control_group'].lstrip('/')
    need(owner['control_group'].endswith('/'+owner['unit']+'.service') and not Path('/proc',str(owner['original_pid'])).exists(), 'pinned unit cgroup and original process absent')
    need(not group.exists() or not (group / 'cgroup.procs').read_text().strip(), 'pinned writer cgroup empty')
    return actual


def prove(training_root, permit, owner_check=stopped):
    need(set(permit) == {'schema', 'execution_config', 'trace', 'stopped_owners'} and
         permit['schema'] == 'jope.exact-unstarted-reservation-permit/v1', 'closed exact recovery permit')
    raw = bound(permit['execution_config']); need(sha(raw) == EXECUTION, 'exact failed execution')
    execution = json.loads(raw); old_root = Path(permit['execution_config']['path']).parent
    need(execution['owned_sources']['continuous_runner.py'] == RUNNER, 'qualified mandatory ACK ordering')
    for name, pin in execution['owned_sources'].items(): need(sha(read(old_root / name)) == pin, 'failed source closure')
    campaign = json.loads(bound(execution['campaign_config']))
    campaign_root = Path(execution['campaign_root'])
    for name, pin in campaign['owned_sources'].items(): need(sha(read(campaign_root / name)) == pin, 'qualified import closure')
    need(Path(training_root) == campaign_root / 'training', 'same owned training journal')
    need({o['unit'] for o in permit['stopped_owners']} == UNITS and len(permit['stopped_owners']) == 2, 'both original writers accounted for')
    old=next(o for o in permit['stopped_owners']if o['unit'].endswith('20261009-v1'))
    new=next(o for o in permit['stopped_owners']if o['unit'].endswith('resume3-20261009'))
    need(old['original_pid']==execution['frozen_owner']['pid'] and old['control_group']==execution['frozen_owner']['control_group'] and
         new['control_group']==old['control_group'].replace(old['unit'],new['unit']), 'original pinned writer cgroup identities')
    owners = [owner_check(o) for o in permit['stopped_owners']]
    trace = bound(permit['trace'], 2 * 1024**2); need(sha(trace) == TRACE, 'sealed complete failed trace')
    rows = [json.loads(line) for line in trace.splitlines()]
    need(len(rows) == EVENTS and [r['sequence'] for r in rows] == list(range(1, EVENTS + 1)), 'complete contiguous trace')
    need(all(r['config_sha256'] == EXECUTION and r['host'] == 'xbabe2' for r in rows), 'trace execution identity')
    need(all(a['monotonic_ns'] < b['monotonic_ns'] for a, b in zip(rows, rows[1:])), 'trace monotonic order')
    pid = next(o['original_pid'] for o in permit['stopped_owners'] if o['unit'].endswith('resume3-20261009'))
    need(all(r['pid'] == pid for r in rows), 'same stopped writer trace')
    tail = rows[-1]
    need(tail['kind'] == 'start' and tail['stage'] == 'input.training_batch' and
         tail['absolute_optimizer_step'] == STEP and tail['batch_id'] == BATCH, 'last durable event is uncompleted CPU input')
    need(not any(r is not tail and r.get('absolute_optimizer_step') == STEP for r in rows), 'no subsequent stage for certified batch')
    root = Path(training_root)
    for pattern in (f'decoded-{STEP:09d}-*', f'batch-{STEP:09d}.json'):
        need(not list(root.glob(pattern)), 'no decode or optimizer evidence for unstarted reservation')
    need(not (root / 'rolling' / f'{STEP:09d}').exists(), 'no newer physical state')
    db = sqlite3.connect((root / 'outbox.sqlite').as_uri() + '?mode=ro', uri=True, isolation_level=None)
    try:
        db.execute('BEGIN')
        need(db.execute('SELECT count(*)FROM batches WHERE id=?', (BATCH,)).fetchone()[0] == 0 and
             db.execute('SELECT count(*)FROM emissions WHERE batch_id=?', (BATCH,)).fetchone()[0] == 0,
             'mandatory durable ACK has never existed')
        # Immutable event history also must not mention this batch. Outcome consumers
        # may advance other rows but never remove optimizer ACK custody.
        need(not any(BATCH.encode() in (r[0].encode() if isinstance(r[0],str) else r[0]) for r in db.execute('SELECT body FROM events')), 'no emitted event for certified batch')
    finally:
        db.execute('ROLLBACK'); db.close()
    return {'schema': 'jope.certified-unstarted-reservation/v1', 'batch_id': BATCH,
            'absolute_step': STEP, 'trace': permit['trace'], 'failed_execution': permit['execution_config'],
            'stopped_owners': permit['stopped_owners'], 'owner_readbacks': owners,
            'optimizer_started': False, 'retry_optimizer': False, 'reservation_must_remain_in_place': True,
            'source_runner_sha256': RUNNER, 'permit_sha256': sha(canonical(permit))}
