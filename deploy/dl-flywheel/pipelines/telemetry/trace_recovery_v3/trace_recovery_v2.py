"""Cold, once-only continuation after physically retired trace recovery1.

Reconstructs all physical custody and the full sealed local event chain before
publishing fresh parts. Attempted but unreceipted parts are never PUT again.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE_SHA = 'c8a13033c6a93bb97d14d3ccb4de0d2fae01be0928a75f654cfd79f32d835c69'
OLD_CONFIG_SHA = '142aede3c9186516cd85ca0942f077cd3421c0f07d6c2987105611719b6bc3b6'
OLD_ROOT = Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v1')
ROOT = Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v2')
OLD_UNIT = 'jopedime-hpc-archive-recovery-20261009-v1'
NEW_UNIT = 'jopedime-hpc-archive-recovery-20261009-v2'
SCHEMA = 'jope.trace-archive-recovery/v2'
base_path = HERE / 'trace_archive_v1_c8a13033.py'
if not base_path.is_file() or any(p.is_symlink() for p in (base_path, *base_path.parents)) or hashlib.sha256(base_path.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError('pinned prior primitives before import')
spec = importlib.util.spec_from_file_location('trace_archive_v1_c8a13033', base_path)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def show_closed(unit):
    r = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'ControlGroup', '-p', 'ActiveState', '-p', 'Result'], capture_output=True, text=True, timeout=5)
    a.need(r.returncode == 0 and len(r.stdout) < 16384, 'bounded systemd read')
    state = dict(line.split('=', 1) for line in r.stdout.splitlines() if '=' in line)
    a.need(state.get('MainPID') == '0' and state.get('ControlGroup') == '' and state.get('ActiveState') in ('inactive', 'failed'), 'prior archive physically active')
    return state


def ordered(entries, prefix):
    found = {}
    for name, value in entries.items():
        if name.startswith(prefix):
            match = re.fullmatch(re.escape(prefix) + r'(\d{5})\.json', name)
            a.need(match is not None, 'malformed custody artifact')
            found[int(match[1])] = value
    return found


def cold_prefix(capture, loader=a.pinned, segment_reader=a.segment):
    """Pure verification plus physical reads; no S3, source edits or old clocks."""
    a.need(capture['schema'] == 'jope.trace-recovery1-closed-readonly-capture/v1' and capture['read_only'] is True and capture['network_S3'] == capture['mutations'] == 0, 'root cold capture')
    service = dict(line.split('=', 1) for line in capture['unit'].splitlines() if '=' in line)
    a.need(service.get('MainPID') == '0' and service.get('ControlGroup') == '' and service.get('ActiveState') in ('inactive', 'failed'), 'captured prior closure')
    entries = {}
    for path, item in capture['files'].items():
        p = Path(path)
        a.need(p == Path(item['path']) and OLD_ROOT in p.parents, 'prior capture path scope')
        name = p.relative_to(OLD_ROOT).as_posix()
        ref = {k: item[k] for k in ('path', 'sha256', 'bytes')}
        raw = loader(ref, 64 * 1024**2)
        a.need(len(raw) == ref['bytes'] and a.digest(raw) == ref['sha256'], 'prior physical artifact drift')
        entries[name] = raw
    a.need(a.digest(entries['config.json']) == OLD_CONFIG_SHA and a.digest(entries['trace_archive.py']) == BASE_SHA, 'exact retired source/config')
    old = json.loads(entries['config.json'])
    a.need(old['schema'] == 'jope.trace-archive-recovery/v1' and old['out'] == str(OLD_ROOT) and old['logs_root'] == a.LOGS and old['global_deadline_unix'] == a.GLOBAL and old['maximum_segments'] == 2500 and old['maximum_put_attempts'] == 4692, 'old immutable envelope')
    index = json.loads(loader(old['prior_index']))
    gap = json.loads(loader(old['gap_reconciliation']))
    execution = json.loads(entries['execution-started.json'])
    a.need(execution['config_sha256'] == OLD_CONFIG_SHA and execution['automatic_retry'] is False and old['created_unix'] * 1e9 <= execution['utc_ns'] < old['deadline_unix'] * 1e9, 'old execution/window binding')
    progress = json.loads(entries['progress.json'])
    a.need(progress['schema'] == 'jope.trace-recovery-progress/v1' and progress['deadline_unix'] == old['deadline_unix'] and progress['state'] in ('completed_bounded_window_with_preserved_gap', 'stopped_unknown_or_integrity_no_retry', 'running'), 'final old progress')
    if progress['state'] == 'running':
        a.need(capture['at'] >= old['deadline_unix'], 'unfinished old window')
    receipts = ordered(entries, 'receipts/')
    attempts = ordered(entries, 'put-attempt-')
    replies = ordered(entries, 'put-reply-')
    verified = ordered(entries, 'verified-object-')
    unknown = ordered(entries, 'unknown-put-')
    count, puts = len(receipts), len(attempts)
    a.need(sorted(receipts) == list(range(154, 154 + count)) and sorted(attempts) == list(range(1, puts + 1)), 'contiguous prior custody numbering')
    a.need(2 * count <= puts <= 2 * count + 2 and puts <= 4692 and set(replies) <= set(attempts) and set(verified) <= set(replies) and set(unknown) <= set(attempts), 'prior attempt/receipt closure')
    a.need(len(index['receipts']) == 153 and index['host'] == 'xbabe3' and index['logs_root'] == a.LOGS, 'original153 authority')
    original_sha = a.digest(loader(index['original_archive_source']))
    loader(index['trace_writer_source'])
    previous, sequence, archived_events, verified_bytes = '0' * 64, 0, 0, 0
    states, gaps = {}, [153]
    last_part = 153 + count + int(puts > 2 * count)
    a.need(last_part < 2500, 'global part ceiling')
    for part in range(last_part + 1):
        body, sealed, head, end = segment_reader(Path(a.LOGS), part, previous, sequence)
        states[part] = {'first_seq': sequence + 1, 'last_seq': end, 'event_head': head, 'events': {'sha256': a.digest(body), 'bytes': len(body)}, 'seal': {'sha256': a.digest(sealed), 'bytes': len(sealed)}}
        receipt = None
        if part < 153:
            item = index['receipts'][part]
            receipt = json.loads(loader({k: item[k] for k in ('path', 'sha256', 'bytes')}))
            a.need(receipt == item['receipt'] and item['path'] == str(Path(a.LOGS) / 'archive-v1' / f'{part:05d}.json') and receipt['schema'] == 'jope.hpc-host-archive/v1' and receipt['source_sha256'] == original_sha, 'original physical receipt')
        elif part in receipts:
            receipt = json.loads(receipts[part])
            a.need(receipt['schema'] == 'jope.hpc-host-archive-recovery-receipt/v1' and receipt['source_sha256'] == BASE_SHA and receipt['config_sha256'] == OLD_CONFIG_SHA and receipt['gap_parts'] == [153], 'recovery1 source/config receipt')
        else:
            if part != 153:
                gaps.append(part)
        if receipt is not None:
            a.need(receipt['host'] == 'xbabe3' and receipt['part'] == part and receipt['first_seq'] == sequence + 1 and receipt['last_seq'] == end and receipt['event_head'] == head, 'receipt/event chain join')
            a.object_ref(receipt['segment'], part, (body, sealed), 'events.jsonl')
            a.object_ref(receipt['seal'], part, (body, sealed), 'seal.json')
            archived_events += end - sequence
            verified_bytes += len(body) + len(sealed)
        if part == 153:
            a.need(gap['schema'] == 'jope.trace-unknown-object-reconciliation/v1' and gap['state'] == 'absent_preserve_unknown_no_put' and gap['s3_error_code'] == '404' and gap['bucket'] == a.BUCKET and gap['key'] == a.PREFIX + '00153/' + a.digest(body) + '/events.jsonl' and gap['expected_sha256'] == a.digest(body) and gap['expected_bytes'] == len(body) and gap['s3_puts'] == gap['gp_executions'] == 0 and gap['local_chain_verified'] is True, 'old153 unknown custody')
        previous, sequence = head, end
    attempted_keys = []
    for number, raw in attempts.items():
        attempt = json.loads(raw)
        part = 154 + (number - 1) // 2
        name = 'events.jsonl' if number % 2 else 'seal.json'
        record = states[part]['events' if number % 2 else 'seal']
        key = a.PREFIX + f'{part:05d}/' + states[part]['events']['sha256'] + '/' + name
        a.need(attempt['schema'] == 'jope.trace-upload-attempt/v1' and attempt['attempt'] == number and attempt['bucket'] == a.BUCKET and attempt['key'] == key and attempt['sha256'] == record['sha256'] and attempt['bytes'] == record['bytes'] and attempt['automatic_retry'] is False and old['created_unix'] * 1e9 <= attempt['utc_ns'] < old['deadline_unix'] * 1e9, 'prior durable attempt identity/window')
        attempted_keys.append(key)
        if number in replies:
            reply = json.loads(replies[number])
            a.need(reply['key'] == key, 'attempt reply key')
        if number in verified:
            ref = json.loads(verified[number])
            a.need(ref['bucket'] == a.BUCKET and ref['key'] == key and ref['bytes'] == record['bytes'] and ref['sha256'] == record['sha256'] and ref['version_id'] == reply['version_id'] and ref['version_id'] not in (None, '', 'null'), 'prior exact-version object closure')
            if part in receipts:
                receipt = json.loads(receipts[part])
                a.need(ref == receipt['segment' if number % 2 else 'seal'], 'receipt/verified object must agree exactly')
        if number in unknown:
            failure = json.loads(unknown[number])
            a.need(failure['state'] == 'unknown_no_retry' and all(failure[k] == v for k, v in attempt.items()), 'unknown attempt custody')
    a.need(set(range(1, 2 * count + 1)) <= set(verified), 'complete segment without two verified objects')
    a.need(progress['unresolved_gap_parts'] == [153] and progress['cloud_contiguous_verified_through_part'] == 152 and 0 <= count - progress['new_verified_segments'] <= 1 and 0 <= puts - progress['put_attempts'] <= 2, 'progress cannot overstate physical custody')
    if progress['state'] == 'completed_bounded_window_with_preserved_gap':
        a.need(puts == 2 * count == len(verified) and progress['new_verified_segments'] == count and progress['put_attempts'] == puts and progress['next_fresh_part'] == last_part + 1 and progress['local_chain_head'] == previous and progress['local_chain_last_seq'] == sequence and progress['archived_events'] == archived_events and progress['verified_bytes'] == verified_bytes, 'completed prior report must match physical closure')
    timing = entries['stage-timing.jsonl'].splitlines()
    for number, line in enumerate(timing, 1):
        event = json.loads(line)
        a.need(event['sequence'] == number and event['config_sha256'] == OLD_CONFIG_SHA and event['host'] == 'xbabe3' and event['gpu_used'] is False, 'prior detailed timing custody')
    a.need(progress['stage_events'] <= len(timing), 'progress timing overstatement')
    return {'previous': previous, 'sequence': sequence, 'fresh_first_part': last_part + 1, 'verified_segments': 153 + count, 'verified_objects': 306 + len(verified), 'archived_events': archived_events, 'verified_bytes': verified_bytes, 'prior_global_put_attempts': 307 + puts, 'remaining_put_attempts': 5000 - 307 - puts, 'gap_parts': gaps, 'gap_records': [gap], 'attempted_keys': attempted_keys, 'new_prior_unknown_parts': gaps[1:], 'prior_progress': progress}


def preflight(config):
    a.need(set(config) == {'schema', 'host', 'logs_root', 'out', 'created_unix', 'deadline_unix', 'global_deadline_unix', 'maximum_segments', 'maximum_global_put_attempts', 'maximum_put_attempts', 'prior_capture', 'source', 'base'}, 'closed v2 config')
    a.need(config['schema'] == SCHEMA and config['host'] == 'xbabe3' and config['logs_root'] == a.LOGS and config['out'] == str(ROOT), 'fixed v2 scope')
    a.need(config['created_unix'] <= time.time() < config['deadline_unix'] <= config['global_deadline_unix'] == a.GLOBAL and 0 < config['deadline_unix'] - config['created_unix'] <= 3600, 'new bounded window/original global')
    a.need(config['source']['path'] == str(ROOT / 'trace_recovery_v2.py') and config['base']['path'] == str(ROOT / 'trace_archive_v1_c8a13033.py') and config['base']['sha256'] == BASE_SHA, 'source authority')
    a.need(a.digest(Path(__file__).read_bytes()) == config['source']['sha256'], 'actual executing source differs')
    a.pinned(config['source']); a.pinned(config['base'])
    capture = json.loads(a.pinned(config['prior_capture'], 16 * 1024**2))
    show_closed('jopedime-hpc-archive-20261009-v1'); show_closed(OLD_UNIT)
    prefix = cold_prefix(capture)
    # Refuse extra post-capture attempt/receipt files before any new cloud work.
    actual = {str(p) for p in OLD_ROOT.glob('*.json')} | {str(p) for p in (OLD_ROOT / 'receipts').glob('*.json')}
    expected = {p for p in capture['files'] if p.endswith('.json')}
    a.need(actual == expected, 'old custody changed after capture')
    a.need(config['maximum_segments'] == 2500 and config['maximum_global_put_attempts'] == 5000 and config['maximum_put_attempts'] == prefix['remaining_put_attempts'] and prefix['remaining_put_attempts'] >= 2, 'global attempt budget conservation')
    show_closed(OLD_UNIT)
    return prefix


class Archive(a.Archive):
    def __init__(self, config, client, timing, prefix):
        self.c, self.client, self.timing = config, client, timing
        self.out, self.logs = Path(config['out']), Path(config['logs_root'])
        self.prefix = prefix
        self.previous, self.sequence = prefix['previous'], prefix['sequence']
        self.archived_events, self.verified_bytes = prefix['archived_events'], prefix['verified_bytes']
        self.part, self.new_segments = prefix['fresh_first_part'], 0
        self.puts = self.gets = self.heads = 0
        self.monotonic_deadline = time.monotonic() + config['deadline_unix'] - time.time()
        a.create(self.out / 'cold-prefix.json', a.canonical(prefix))

    def tick(self):
        a.need(time.time() < self.c['deadline_unix'] and time.monotonic() < self.monotonic_deadline, 'fixed v2 wall/monotonic deadline')

    def publish(self, key, raw):
        a.need(self.part >= self.prefix['fresh_first_part'] and key not in self.prefix['attempted_keys'], 'never replay prior attempted key')
        return super().publish(key, raw)

    def once(self):
        self.tick()
        a.need(self.part < 2500 and self.puts + 2 <= self.c['maximum_put_attempts'], 'global segment/pair attempt ceiling')
        if not (self.logs / f'events-{self.part:05d}.sealed.json').exists():
            return False
        with self.timing.span('local.sealed_chain', part=self.part):
            body, sealed, head, end = a.segment(self.logs, self.part, self.previous, self.sequence)
        key = a.PREFIX + f'{self.part:05d}/' + a.digest(body) + '/'
        first = self.publish(key + 'events.jsonl', body)
        second = self.publish(key + 'seal.json', sealed)
        self.tick()
        receipt = {'schema': 'jope.hpc-host-archive-recovery-receipt/v2', 'host': 'xbabe3', 'part': self.part, 'segment': first, 'seal': second, 'first_seq': self.sequence + 1, 'last_seq': end, 'event_head': head, 'gap_parts': self.prefix['gap_parts'], 'source_sha256': self.c['source']['sha256'], 'config_sha256': self.timing.config_sha, 'prior_capture': self.c['prior_capture'], 'global_put_attempts_through': self.prefix['prior_global_put_attempts'] + self.puts}
        a.create(self.out / 'receipts' / f'{self.part:05d}.json', a.canonical(receipt))
        self.archived_events += end - self.sequence
        self.verified_bytes += len(body) + len(sealed)
        self.previous, self.sequence = head, end
        self.part += 1
        self.new_segments += 1
        return True

    def report(self, state):
        value = {'schema': 'jope.trace-recovery-progress/v2', 'state': state, 'utc_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), 'config_sha256': self.timing.config_sha, 'deadline_unix': self.c['deadline_unix'], 'verified_segments': self.prefix['verified_segments'] + self.new_segments, 'verified_objects': self.prefix['verified_objects'] + 2 * self.new_segments, 'new_verified_segments': self.new_segments, 'new_verified_objects': 2 * self.new_segments, 'archived_events': self.archived_events, 'verified_bytes': self.verified_bytes, 'local_chain_last_seq': self.sequence, 'local_chain_head': self.previous, 'cloud_contiguous_verified_through_part': 152, 'unresolved_gap_parts': self.prefix['gap_parts'], 'next_fresh_part': self.part, 'put_attempts': self.puts, 'global_put_attempts': self.prefix['prior_global_put_attempts'] + self.puts, 'get_attempts': self.gets, 'head_attempts': self.heads, 'stage_events': self.timing.sequence, 'optimizer_updates': 0, 'gp_executions': 0, 'gpu_used': False, 'cuda_execution_time_ns': None}
        temp = self.out / 'progress.new'
        with temp.open('wb') as stream:
            stream.write(a.canonical(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, self.out / 'progress.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    raw = a.read(args.config, 1024**2)
    a.need(a.digest(raw) == args.config_sha and os.uname().nodename == 'xbabe3', 'config/host')
    config = json.loads(raw)
    checked_start, checked_mono, cpu_before = time.time_ns(), time.monotonic_ns(), a.cpu()
    prefix = preflight(config)
    cpu_after = a.cpu()
    checked = {'started_utc_ns': checked_start, 'ended_utc_ns': time.time_ns(), 'duration_monotonic_ns': time.monotonic_ns() - checked_mono, 'cpu_delta': {k: cpu_after[k] - cpu_before[k] for k in cpu_before}}
    if args.preflight_only:
        print(json.dumps({'state': 'passed', 'config_sha256': args.config_sha, 'fresh_first_part': prefix['fresh_first_part'], 'prior_verified_segments': prefix['verified_segments'], 'prior_global_put_attempts': prefix['prior_global_put_attempts'], 'remaining_put_attempts': prefix['remaining_put_attempts'], 'gap_parts': prefix['gap_parts'], 's3_requests': 0, 'qualification_timing': checked})); return
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    cg = Path('/sys/fs/cgroup') / group.lstrip('/')
    quota, period = (cg / 'cpu.max').read_text().split()
    a.need(quota != 'max' and int(quota) <= int(period) and int((cg / 'memory.max').read_text()) <= 512 * 1024**2 and int((cg / 'memory.swap.max').read_text()) == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU1/512MiB/swap0/noGPU')
    lock = os.open(Path(a.LOGS) / 'archive-v1', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    show_closed(OLD_UNIT)
    a.create(ROOT / 'execution-started.json', a.canonical({'config_sha256': args.config_sha, 'utc_ns': time.time_ns(), 'automatic_retry': False, 'prior_capture': config['prior_capture']}))
    (ROOT / 'receipts').mkdir()
    timing = a.Timing(ROOT / 'stage-timing.jsonl', args.config_sha)
    timing.event({'kind': 'verified', 'stage': 'local.cold_prefix', **checked})
    import boto3
    from botocore.config import Config
    client = boto3.Session(profile_name='veox-jepson-prod').client('s3', config=Config(connect_timeout=5, read_timeout=20, retries={'max_attempts': 0}))
    archive = Archive(config, client, timing, prefix)
    state = 'running'
    try:
        archive.report(state)
        while min(config['deadline_unix'] - time.time(), archive.monotonic_deadline - time.monotonic()) > 60 and archive.part < 2500 and archive.puts + 2 <= config['maximum_put_attempts']:
            if not archive.once():
                time.sleep(min(15, max(0, min(config['deadline_unix'] - time.time(), archive.monotonic_deadline - time.monotonic()) - 60)))
            archive.report(state)
        state = 'completed_bounded_window_with_preserved_gaps'
    except BaseException as error:
        state = 'stopped_unknown_or_integrity_no_retry'
        a.create(ROOT / 'fatal.json', a.canonical({'utc_ns': time.time_ns(), 'error': repr(error), 'automatic_retry': False, 'unknown_attempts_preserved': True}))
        raise
    finally:
        archive.report(state)
        os.close(lock)


if __name__ == '__main__':
    main()
