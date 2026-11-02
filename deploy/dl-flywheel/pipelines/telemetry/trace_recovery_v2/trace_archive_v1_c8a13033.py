"""Fresh telemetry custody after an unknown upload; never replay that upload."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

BUCKET = 'veox-jopedime-learning-use2'
PREFIX = 'experiments/gp-policy-dl/v2-trainonly/paper-evidence/20261009/hpc-trace-v1/xbabe3/'
GLOBAL = 1791592146.380285
LOGS = '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/hpc_trace_20261009_v1/logs'
SCHEMA = 'jope.trace-archive-recovery/v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path, ceiling):
    path = Path(path)
    need(path.is_absolute() and all(not p.is_symlink() for p in (path, *path.parents)), 'physical path')
    need(path.is_file() and 0 < path.stat().st_size <= ceiling, 'physical byte bounds')
    raw = path.read_bytes()
    need(len(raw) <= ceiling, 'read byte bounds')
    return raw


def pinned(ref, ceiling=1024**2):
    need(set(ref) == {'path', 'sha256', 'bytes'}, 'closed physical pin')
    raw = read(ref['path'], ceiling)
    need(len(raw) == ref['bytes'] and digest(raw) == ref['sha256'], 'physical pin integrity')
    return raw


def create(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def error_code(error):
    response = getattr(error, 'response', None)
    detail = response.get('Error') if isinstance(response, dict) else None
    return detail.get('Code') if isinstance(detail, dict) else None


def cpu():
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'user_seconds': own.ru_utime, 'system_seconds': own.ru_stime,
            'child_user_seconds': child.ru_utime, 'child_system_seconds': child.ru_stime}


class Timing:
    def __init__(self, path, config_sha):
        self.path, self.config_sha, self.sequence = path, config_sha, 0

    def event(self, value):
        self.sequence += 1
        value.update(schema='jope.trace-archive-stage/v1', sequence=self.sequence,
                     config_sha256=self.config_sha, utc_ns=time.time_ns(),
                     monotonic_ns=time.monotonic_ns(), host=os.uname().nodename,
                     pid=os.getpid(), gpu_used=False, cuda_execution_time_ns=None)
        with self.path.open('ab') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())

    @contextlib.contextmanager
    def span(self, stage, **details):
        start, before = time.monotonic_ns(), cpu()
        self.event({'kind': 'start', 'stage': stage, 'cpu': before, **details})
        error = None
        try:
            yield
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            after = cpu()
            self.event({'kind': 'end', 'stage': stage, 'duration_monotonic_ns': time.monotonic_ns() - start,
                        'cpu': after, 'cpu_delta': {k: after[k] - before[k] for k in before},
                        'error': error, **details})


def segment(root, part, previous, sequence):
    seal_raw = read(root / f'events-{part:05d}.sealed.json', 4096)
    seal = json.loads(seal_raw)
    need(seal['schema'] == 'jope.hpc-host-log-segment/v1' and
         seal['path'] == str(root / f'events-{part:05d}.jsonl'), 'closed segment authority')
    raw = read(seal['path'], 10*1024**2)
    need(len(raw) == seal['bytes'] and digest(raw) == seal['sha256'], 'segment bytes')
    first = sequence + 1
    for line in raw.splitlines():
        event = json.loads(line)
        claimed = event.pop('sha256')
        sequence += 1
        need(event['host'] == 'xbabe3' and event['seq'] == sequence and
             event['previous'] == previous and digest(canonical(event)) == claimed, 'event chain')
        previous = claimed
    need(sequence >= first and sequence == seal['last_seq'] and previous == seal['event_head'], 'sealed head')
    return raw, seal_raw, previous, sequence


def object_ref(ref, part, body, name):
    key = PREFIX + f'{part:05d}/' + digest(body[0]) + '/' + name
    actual = body[0] if name == 'events.jsonl' else body[1]
    need(ref['bucket'] == BUCKET and ref['key'] == key and
         ref['version_id'] not in (None, '', 'null') and
         ref['sha256'] == digest(actual) and ref['bytes'] == len(actual), 'historical exact-version authority')


def preflight(config):
    need(set(config) == {'schema', 'host', 'logs_root', 'out', 'created_unix', 'deadline_unix',
                        'global_deadline_unix', 'maximum_segments', 'maximum_put_attempts',
                        'prior_index', 'gap_reconciliation', 'source'}, 'closed recovery config')
    need(config['schema'] == SCHEMA and config['host'] == 'xbabe3' and config['logs_root'] == LOGS,
         'single qualified host and trace root')
    need(0 < config['deadline_unix']-config['created_unix'] <= 3600 and
         config['deadline_unix'] <= config['global_deadline_unix'] == GLOBAL and
         config['created_unix'] <= time.time() < config['deadline_unix'], 'fixed recovery clock')
    need(config['maximum_segments'] == 2500 and config['maximum_put_attempts'] == 4692,
         'original 5000 attempts including306 verified +1 unknown; stricter remaining budget')
    root = Path(config['logs_root'])
    index = json.loads(pinned(config['prior_index']))
    gap = json.loads(pinned(config['gap_reconciliation']))
    need(config['source']['path'] == str(Path(config['out'])/'trace_archive.py'), 'executor source authority')
    pinned(config['source'])
    need(index['schema'] == 'jope.trace-prior-archive-index/v1' and index['host'] == 'xbabe3' and
         index['logs_root'] == str(root) and len(index['receipts']) == 153, 'exact prior prefix')
    need(index['original_archive_source']['path'] == str(root.parent/'archive_trace.py') and
         index['trace_writer_source']['path'] == str(root.parent/'host_trace.py'), 'original runtime authority')
    original = digest(pinned(index['original_archive_source']))
    pinned(index['trace_writer_source'])
    previous, sequence, total = '0'*64, 0, 0
    for part, item in enumerate(index['receipts']):
        receipt = json.loads(pinned({k: item[k] for k in ('path', 'sha256', 'bytes')}))
        need(receipt == item['receipt'] and item['path'] == str(root/'archive-v1'/f'{part:05d}.json'),
             'unchanged physical prior receipt')
        body, seal, head, end = segment(root, part, previous, sequence)
        need(receipt['schema'] == 'jope.hpc-host-archive/v1' and receipt['host'] == 'xbabe3' and
             receipt['part'] == part and receipt['first_seq'] == sequence+1 and
             receipt['last_seq'] == end and receipt['event_head'] == head and
             receipt['source_sha256'] == original, 'prior receipt chain and source')
        object_ref(receipt['segment'], part, (body, seal), 'events.jsonl')
        object_ref(receipt['seal'], part, (body, seal), 'seal.json')
        total += len(body) + len(seal)
        previous, sequence = head, end
    prior_events = sequence
    body, seal, previous, sequence = segment(root, 153, previous, sequence)
    need(gap['schema'] == 'jope.trace-unknown-object-reconciliation/v1' and
         gap['state'] == 'absent_preserve_unknown_no_put' and gap['s3_error_code'] == '404' and
         gap['bucket'] == BUCKET and gap['key'] == PREFIX+'00153/'+digest(body)+'/events.jsonl' and
         gap['expected_sha256'] == digest(body) and gap['expected_bytes'] == len(body) and
         gap['s3_puts'] == gap['gp_executions'] == 0 and gap['local_chain_verified'] is True,
         'preserved absent unknown gap, never republish it')
    return {'previous': previous, 'sequence': sequence, 'prior_events': prior_events,
            'verified_bytes': total, 'gap': gap, 'gap_event_count': sequence-prior_events}


class Archive:
    def __init__(self, config, client, timing, prefix):
        self.c, self.client, self.timing = config, client, timing
        self.out, self.logs = Path(config['out']), Path(config['logs_root'])
        self.previous, self.sequence = prefix['previous'], prefix['sequence']
        self.archived_events, self.verified_bytes = prefix['prior_events'], prefix['verified_bytes']
        self.part, self.new_segments = 154, 0
        self.puts, self.gets, self.heads = 0, 0, 0
        create(self.out/'preserved-gap-00153.json', canonical(prefix['gap']))

    def tick(self):
        need(time.time() < self.c['deadline_unix'], 'archive deadline')

    def verify(self, key, raw, version):
        self.tick()
        need(version not in (None, '', 'null'), 'versioning required')
        self.gets += 1
        with self.timing.span('s3.GET_headers', key=key, version_id=version):
            response = self.client.get_object(Bucket=BUCKET, Key=key, VersionId=version)
        need(response.get('VersionId') == version and response['ContentLength'] == len(raw), 'exact version and bytes')
        with response['Body'] as stream, self.timing.span('s3.GET_body', key=key):
            actual = stream.read(len(raw)+1)
        self.tick()
        need(actual == raw, 'exact S3 readback')
        return {'bucket': BUCKET, 'key': key, 'version_id': version, 'bytes': len(raw),
                'sha256': digest(raw), 'verified_at_unix_ns': time.time_ns()}

    def publish(self, key, raw):
        self.tick()
        need(key.startswith(PREFIX+f'{self.part:05d}/') and self.part >= 154 and
             len(raw) <= 10*1024**2 and self.puts < self.c['maximum_put_attempts'], 'fresh key and attempt budget')
        self.puts += 1
        attempt = {'schema': 'jope.trace-upload-attempt/v1', 'attempt': self.puts,
                   'utc_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
                   'bucket': BUCKET, 'key': key, 'sha256': digest(raw), 'bytes': len(raw),
                   'automatic_retry': False}
        create(self.out/f'put-attempt-{self.puts:05d}.json', canonical(attempt))
        try:
            with self.timing.span('s3.PUT', key=key, bytes=len(raw)):
                response = self.client.put_object(Bucket=BUCKET, Key=key, Body=raw,
                    ContentLength=len(raw), IfNoneMatch='*', Metadata={'sha256': digest(raw)})
        except Exception as error:
            if error_code(error) not in ('412', 'PreconditionFailed', 'ConditionalRequestConflict'):
                create(self.out/f'unknown-put-{self.puts:05d}.json', canonical({
                    **attempt, 'state': 'unknown_no_retry', 'error': repr(error), 'response_code': error_code(error)}))
                raise
            self.tick()
            self.heads += 1
            with self.timing.span('s3.HEAD', key=key):
                response = self.client.head_object(Bucket=BUCKET, Key=key)
        create(self.out/f'put-reply-{self.puts:05d}.json', canonical({
            'key': key, 'version_id': response.get('VersionId'), 'utc_ns': time.time_ns()}))
        # A readback failure leaves the fsynced attempted key/reply in custody. Never retry it.
        ref = self.verify(key, raw, response.get('VersionId'))
        create(self.out/f'verified-object-{self.puts:05d}.json', canonical(ref))
        return ref

    def once(self):
        self.tick()
        need(self.part < self.c['maximum_segments'], 'fixed segment ceiling')
        path = self.logs/f'events-{self.part:05d}.sealed.json'
        if not path.exists():
            return False
        with self.timing.span('local.sealed_chain', part=self.part):
            body, sealed, head, end = segment(self.logs, self.part, self.previous, self.sequence)
        prefix = PREFIX+f'{self.part:05d}/'+digest(body)+'/'
        event_ref = self.publish(prefix+'events.jsonl', body)
        seal_ref = self.publish(prefix+'seal.json', sealed)
        receipt = {'schema': 'jope.hpc-host-archive-recovery-receipt/v1', 'host': 'xbabe3',
                   'part': self.part, 'segment': event_ref, 'seal': seal_ref,
                   'first_seq': self.sequence+1, 'last_seq': end, 'event_head': head,
                   'gap_parts': [153], 'source_sha256': self.c['source']['sha256'],
                   'config_sha256': self.timing.config_sha}
        create(self.out/'receipts'/f'{self.part:05d}.json', canonical(receipt))
        self.archived_events += end-self.sequence
        self.verified_bytes += len(body)+len(sealed)
        self.previous, self.sequence = head, end
        self.part += 1
        self.new_segments += 1
        return True

    def report(self, state):
        value = {'schema': 'jope.trace-recovery-progress/v1', 'state': state, 'utc_ns': time.time_ns(),
                 'deadline_unix': self.c['deadline_unix'], 'verified_segments': 153+self.new_segments,
                 'new_verified_segments': self.new_segments, 'verified_objects': 306+2*self.new_segments,
                 'new_verified_objects': 2*self.new_segments, 'archived_events': self.archived_events,
                 'local_chain_last_seq': self.sequence, 'local_chain_head': self.previous,
                 'cloud_contiguous_verified_through_part': 152, 'unresolved_gap_parts': [153],
                 'verified_bytes': self.verified_bytes, 'next_fresh_part': self.part,
                 'put_attempts': self.puts, 'get_attempts': self.gets, 'head_attempts': self.heads,
                 'stage_events': self.timing.sequence, 'optimizer_updates': 0, 'gp_executions': 0,
                 'gpu_used': False, 'cuda_execution_time_ns': None}
        raw = canonical(value)
        temp = self.out/'progress.new'
        with temp.open('wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, self.out/'progress.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    raw = read(args.config, 1024**2)
    need(digest(raw) == args.config_sha, 'config SHA')
    c = json.loads(raw)
    need(os.uname().nodename == 'xbabe3', 'executor host')
    out = Path(c['out'])
    need(str(out).startswith('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-') and
         all(not p.is_symlink() for p in (out, *out.parents)), 'owned receipt root')
    prefix = preflight(c)
    if args.preflight_only:
        print(json.dumps({'state': 'passed', 'prior_segments': 153, 'gap_preserved': 153,
                          'fresh_first_part': 154, 's3_requests': 0, 'optimizer_updates': 0})); return
    group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    cg = Path('/sys/fs/cgroup')/group.lstrip('/')
    quota, period = (cg/'cpu.max').read_text().split()
    need(quota != 'max' and int(quota) <= int(period) and int((cg/'memory.max').read_text()) <= 512*1024**2 and
         int((cg/'memory.swap.max').read_text()) == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU1/512MiB/swap0/noGPU')
    old = subprocess.run(['systemctl', '--user', 'show', 'jopedime-hpc-archive-20261009-v1',
                          '-p', 'MainPID', '-p', 'ControlGroup'], capture_output=True, text=True).stdout
    need('MainPID=0' in old and 'ControlGroup=\n' in old, 'old archiver physically retired')
    # Lock an existing directory without writing historical log or receipt files.
    lock = os.open(Path(c['logs_root'])/'archive-v1', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    create(out/'execution-started.json', canonical({'config_sha256': args.config_sha,
           'utc_ns': time.time_ns(), 'automatic_retry': False}))
    (out/'receipts').mkdir(exist_ok=False)
    timing = Timing(out/'stage-timing.jsonl', args.config_sha)
    import boto3
    from botocore.config import Config
    client = boto3.Session(profile_name='veox-jepson-prod').client('s3', config=Config(
        connect_timeout=5, read_timeout=20, retries={'max_attempts': 0}))
    archive = Archive(c, client, timing, prefix)
    state = 'running'
    try:
        archive.report(state)
        while time.time() < c['deadline_unix']-60 and archive.part < c['maximum_segments']:
            if not archive.once():
                time.sleep(min(15, max(0, c['deadline_unix']-time.time()-60)))
            archive.report(state)
        state = 'completed_bounded_window_with_preserved_gap'
    except BaseException as error:
        state = 'stopped_unknown_or_integrity_no_retry'
        create(out/'fatal.json', canonical({'utc_ns': time.time_ns(), 'error': repr(error),
               'automatic_retry': False, 'unknown_attempts_preserved': True}))
        raise
    finally:
        archive.report(state)
        os.close(lock)


if __name__ == '__main__':
    main()
