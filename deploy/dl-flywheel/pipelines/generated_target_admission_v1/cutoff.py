"""Verify one pre-captured, sealed SQLite outbox, without constructors or writes.

The caller supplies a physical SQLite backup made at one read transaction by a
separately qualified owner. Never point this at a live WAL database. Immutable
SQLite ignores WAL, so any sidecar is refused before opening. A hash and exact
event head/config are mandatory. This verifies ACK custody, not optimizer commit.
"""
import hashlib
import math
from pathlib import Path
import sqlite3
from .common import Budget, canonical, digest, document, hash64, need, sha

STATES = {'recorded', 'leased', 'unknown_attempt', 'pending_unsupported',
          'failure_custody_verified', 'published_verified', 'cached_verified'}
SUCCESS = {'published_verified', 'cached_verified'}


def _blob(value, cap):
    need(type(value) is bytes, 'SQLite body must be an immutable BLOB')
    return document(value, cap)


def _file_hash(path, limit, budget):
    need(path.is_absolute() and all(not x.is_symlink() for x in (path, *path.parents))
         and path.is_file(), 'absolute sealed regular snapshot')
    stat = path.stat()
    need(0 < stat.st_size <= limit, 'database byte bound')
    for suffix in ('-wal', '-shm', '-journal'):
        need(not Path(str(path) + suffix).exists() and not Path(str(path) + suffix).is_symlink(),
             'sealed backup cannot have SQLite sidecars')
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(65536):
            budget.tick()
            h.update(chunk)
    now = path.stat()
    need((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) ==
         (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns), 'snapshot changed while hashing')
    return h.hexdigest(), stat.st_size


def capture_cutoff(path, *, expected_sha256, expected_config, expected_head,
                   max_database_bytes=128 * 1024**2, max_rows=100000,
                   max_events=400000, max_payload_bytes=128 * 1024**2, seconds=30):
    """Return canonical verified cutoff bytes; create no output file or DB row."""
    need(hash64(expected_sha256), 'snapshot SHA required')
    need(type(max_database_bytes) is int and 0 < max_database_bytes <= 256 * 1024**2 and
         type(max_payload_bytes) is int and 0 < max_payload_bytes <= 256 * 1024**2 and
         type(max_rows) is int and 0 < max_rows <= 100000 and
         type(max_events) is int and 0 < max_events <= 400000, 'finite cutoff bounds')
    need(set(expected_head) == {'events', 'head_sha256', 'observations', 'batches'} and
         hash64(expected_head['head_sha256']) and
         all(type(expected_head[k]) is int and expected_head[k] >= 0 for k in ('events', 'observations', 'batches')),
         'explicit finite event/count cutoff')
    need(expected_head['events'] <= max_events and expected_head['observations'] <= max_rows,
         'head exceeds cutoff bounds')
    budget = Budget(seconds)
    path = Path(path)
    identity = _file_hash(path, max_database_bytes, budget)
    need(identity[0] == expected_sha256, 'snapshot hash mismatch')
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON')
        db.set_progress_handler(lambda: int(__import__('time').monotonic() >= budget.deadline), 10000)
        db.execute('BEGIN')
        configs = db.execute('SELECT body FROM config').fetchall()
        need(len(configs) == 1 and _blob(configs[0][0], 4096) == expected_config,
             'exact singleton outbox configuration')
        need(set(expected_config) == {'schema', 'model_origin_sha256', 'max_pending', 'max_rows', 'max_payload_bytes'}
             and expected_config['schema'] == 'jope.flywheel-neural-outbox/v1'
             and hash64(expected_config['model_origin_sha256']) and
             all(type(expected_config[k]) is int and expected_config[k] > 0 for k in ('max_pending', 'max_rows', 'max_payload_bytes')) and
             expected_config['max_pending'] <= expected_config['max_rows'] <= 1000000 and
             expected_config['max_payload_bytes'] <= 256*1024**2, 'closed outbox schema')
        for table, bound in [('events', max_events), ('emissions', max_rows), ('batches', max_rows)]:
            row = db.execute(f'SELECT count(*),coalesce(sum(length(body)),0) FROM {table}').fetchone()
            need(row[0] <= bound and row[1] <= max_payload_bytes, 'cutoff row/payload bound')
        previous = '0' * 64
        seq = 0
        registered, heads, states, leases, outcomes = {}, {}, {}, {}, {}
        for row in db.execute('SELECT * FROM events ORDER BY seq'):
            budget.tick()
            seq += 1
            body = _blob(row['body'], 128 * 1024)
            need(row['seq'] == seq and row['previous'] == previous and
                 row['sha'] == digest({'seq': seq, 'kind': row['kind'], 'body': body, 'previous': previous}),
                 'event chain mismatch')
            previous = row['sha']
            if row['kind'] == 'batch.emitted':
                need(set(body) == {'batch_id', 'body_sha256', 'emission_ids'} and
                     body['batch_id'] not in registered, 'closed unique batch event')
                registered[body['batch_id']] = body
                heads[body['batch_id']] = {'seq': seq, 'sha256': previous}
                for eid in body['emission_ids']:
                    need(hash64(eid) and eid not in states, 'unique emission')
                    states[eid] = 'recorded'
            elif row['kind'] == 'emission.leased':
                need(set(body) == {'emission_id', 'owner', 'started_at', 'expires_at', 'token'} and
                     type(body['owner']) is str and 0 < len(body['owner']) <= 128 and hash64(body['token']) and
                     all(type(body[k]) in (int, float) and math.isfinite(body[k]) for k in ('started_at', 'expires_at')) and
                     0 < body['expires_at'] - body['started_at'] <= 600 and
                     body['token'] == digest({'id': body['emission_id'], 'owner': body['owner'], 'at': body['started_at']}),
                     'closed lease identity')
                eid = body['emission_id']
                need(states.get(eid) == 'recorded', 'lease event order')
                states[eid], leases[eid] = 'leased', body
            elif row['kind'] == 'emission.resolved':
                need(set(body) == {'emission_id', 'outcome'}, 'closed resolved event')
                eid = body['emission_id']
                need(states.get(eid) == 'leased' and body['outcome']['state'] in STATES - {'recorded', 'leased'},
                     'outcome event order/state')
                states[eid], outcomes[eid] = body['outcome']['state'], body['outcome']
            else:
                raise ValueError('unknown outbox event kind')
        need(seq == expected_head['events'] and previous == expected_head['head_sha256'], 'exact event head drift')
        batches = {}
        for row in db.execute('SELECT * FROM batches ORDER BY id'):
            budget.tick()
            body, ack = _blob(row['body'], 128 * 1024), _blob(row['ack'], 128 * 1024)
            need(set(ack) == {'schema', 'batch_id', 'checkpoint_sha256', 'global_step', 'body_sha256', 'emission_ids', 'head', 'numeric_outcomes_required_for_this_optimizer'} and
                 ack['schema'] == 'jope.flywheel-neural-emission-ack/v1' and ack['batch_id'] == row['id'] and
                 ack['numeric_outcomes_required_for_this_optimizer'] is False, 'closed ACK semantics')
            need(set(body) == {'batch', 'emissions', 'payload_sha256'} and row['id'] in registered and
                 row['id'] == body['batch']['batch_id'] and
                 sha(row['body']) == registered[row['id']]['body_sha256'] == ack['body_sha256'] and
                 ack['head'] == heads[row['id']] and ack['emission_ids'] == body['emissions'] == registered[row['id']]['emission_ids'] and
                 ack['checkpoint_sha256'] == body['batch']['checkpoint_sha256'] and
                 ack['global_step'] == body['batch']['global_step'], 'batch/ACK exact closure')
            actual = db.execute('SELECT id,body,ordinal FROM emissions WHERE batch_id=? ORDER BY ordinal', (row['id'],)).fetchall()
            need([x['id'] for x in actual] == body['emissions'] and
                 [sha(x['body']) for x in actual] == body['payload_sha256'] and
                 [x['ordinal'] for x in actual] == list(range(len(actual))), 'emission membership/payload/ordinal closure')
            batches[row['id']] = body['batch']
        need(len(batches) == len(registered) == expected_head['batches'], 'batch count')
        records = []
        for row in db.execute('SELECT * FROM emissions ORDER BY id'):
            budget.tick()
            body = _blob(row['body'], 128 * 1024)
            lease = None if row['lease'] is None else _blob(row['lease'], 4096)
            outcome = None if row['outcome'] is None else _blob(row['outcome'], 16384)
            need(row['state'] == states.get(row['id']) and lease == leases.get(row['id']) and
                 outcome == outcomes.get(row['id']), 'materialized state/lease/outcome mismatch')
            need(set(body) == {'schema', 'model_origin_sha256', 'batch', 'ordinal', 'emission'} and
                 body['schema'] == 'jope.flywheel-neural-emission/v1' and
                 body['model_origin_sha256'] == expected_config['model_origin_sha256'] and
                 body['batch'] == batches[row['batch_id']] and body['ordinal'] == row['ordinal'] and
                 row['id'] == digest({'origin': expected_config['model_origin_sha256'],
                                     'batch_id': row['batch_id'], 'ordinal': row['ordinal']}), 'emission identity/ancestry')
            records.append({'id': row['id'], 'body': body, 'state': row['state'], 'outcome': outcome})
        need(len(records) == len(states) == expected_head['observations'], 'complete emission census')
        db.execute('COMMIT')
    finally:
        db.close()
    need(identity == _file_hash(path, max_database_bytes, budget), 'snapshot changed during verification')
    result = canonical({'schema': 'jope.generated-target-outbox-cutoff/v1', 'source_sha256': identity[0],
                        'source_bytes': identity[1], 'config': expected_config, 'head': expected_head,
                        'records': records, 'source_optimizer_commit_verified': False})
    need(len(result) <= max_payload_bytes, 'cutoff output byte bound')
    budget.tick()
    return result
