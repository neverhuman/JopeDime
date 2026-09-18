#!/usr/bin/env python3
"""Audit and garbage-collect the GP priority queue under the service's own queue lock.

The deployed GP service (recovery source/parallel/src/service.rs:186-230) re-reads
runs/gp-priority-jobs.json every loop, admits only outputs it has not seen, and
keeps every admitted job in an in-memory `known` map plus an in-memory retry map
(backoff 5*2^min(attempts,6) s, capped at 320 s, retried forever; :156-157,
:289-296, :306-313). A job whose observation ledger cannot be resumed fails on every
attempt and cycles a slot until the service restarts. Removing such a job from the
file therefore takes effect ONLY at the next GP restart; while the service runs, the
file-level GC is inert but safe (the service never removes or re-validates outputs
it already knows).

Resume predicates mirrored from the deployed search (source/parallel/src/gp.rs):
  * :649-653  observations.jsonl is read whenever it exists; observations.json only
              when progress.json exists (resuming); otherwise the search starts fresh.
  * :656-659  a non-empty journal without progress.json: "observation ledger has no
              resume identity" -- permanent.
  * :705-714  progress.json prepared_sha256/cut/seeds_sha256/rng_seed must match the job.
  * :744-755  every ledger row must carry the job's identity AND the running binary's
              evaluator_sha256: "observation ledger identity mismatch" -- permanent when
              the ledger was written by a foreign evaluator.
Worker stderr goes to runs/gp-worker-N.log (service.rs:350-351), so the deployed
failure receipts (runs/gp-failures/*.json: timestamp, output, attempt, exit_code,
retry_after_seconds[, failure]) never carry that diagnosis text; the receipt-text
rule below exists for producers that do record one.

--audit classifies every job in the priority file and writes the dead-letter file
(parallel.gp_dead_letter.v1) plus a receipt. --apply re-reads the file under the
same flock queue::merge takes (queue.rs:295-300), drops completed + dead-lettered
jobs preserving order and the header, and rewrites the file atomically in the
serde pretty layout (2-space indent, no trailing newline). Jobs that cannot be
classified (malformed entries, unreadable ledgers or progress receipts) are never
removed. Unknown job keys (for example `continuation`) are preserved verbatim.
"""
import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from resume_expired_cycle import read, require, sha

DEAD_LETTER_SCHEMA = 'parallel.gp_dead_letter.v1'
RECEIPT_SCHEMA = 'parallel.gp_queue_gc_receipt.v1'
JOBS_SCHEMA = 'parallel.gp_jobs.v1'
SEARCH_SCHEMA = 'parallel.search.v1'
HEADER_KEYS = ('schema', 'qualification', 'vocabulary')
REASONS = ('ledger_evaluator_mismatch', 'identity_mismatch', 'attempts_exhausted')
IDENTITY_MISMATCH = 'identity mismatch'
IDENTITY_FIELDS = ('prepared_sha256', 'cut', 'seeds_sha256', 'rng_seed')
LEDGER_NAMES = ('observations.jsonl', 'observations.json',
                'search/observations.jsonl', 'search/observations.json')
NO_RESUME_IDENTITY = 'progress.json absent: observation ledger has no resume identity (gp.rs:656-659)'
RESTART_NOTE = ('effective only at GP restart: the deployed service keeps known/retry maps in memory '
                '(source/parallel/src/service.rs:138-157, 231-239)')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def job_sha256(job):
    return hashlib.sha256(canonical(job)).hexdigest()


def pretty(value):
    """Byte-compatible with serde_json::to_vec_pretty as used by manifest::write_json."""
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.partial')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(path, pretty(value) + b'\n')


def safe_relative(value):
    """Mirror manifest::relative: non-empty, no root, no '.'/'..' components (over-rejects 'a/./b')."""
    if not isinstance(value, str) or not value or value.startswith('/'):
        return False
    parts = [part for part in value.split('/') if part]
    return bool(parts) and all(part not in ('.', '..') for part in parts)


def lock_path_for(jobs_path):
    return Path(jobs_path).with_suffix('.lock')


def acquire_lock(path, timeout):
    """LOCK_EX on the queue lock file, polling LOCK_NB until `timeout` seconds elapse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, 'a+b')
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise RuntimeError(f'queue lock busy after {timeout}s: {path}')
            time.sleep(0.05)


def release_lock(handle):
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def expected_evaluator(root, manifest_path):
    """sha256 of root/manifest['cpu_executable'], cross-checked against manifest['bindings'] when bound."""
    manifest = read(manifest_path)
    executable = manifest.get('cpu_executable')
    require(safe_relative(executable), 'manifest cpu_executable missing or unsafe')
    path = root / executable
    require(path.is_file(), f'manifest cpu_executable absent: {executable}')
    digest = sha(path)
    bindings = manifest.get('bindings')
    bound = bindings.get(executable) if isinstance(bindings, dict) else None
    require(bound is None or bound == digest,
            f'cpu_executable on disk differs from the manifest binding: {executable}')
    return dict(manifest_sha256=sha(manifest_path), cpu_executable=executable,
                evaluator_sha256=digest, binding_checked=bound is not None)


def load_jobs(path):
    """Return (document, sha256 of the bytes that were parsed) -- one read, one identity."""
    path = Path(path)
    require(path.is_file(), f'priority file absent: {path}')
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError(f'priority file unreadable: {path}: {error}')
    require(isinstance(document, dict) and document.get('schema') == JOBS_SCHEMA,
            f'priority file is not {JOBS_SCHEMA}: {path}')
    require(isinstance(document.get('jobs'), list), 'priority file jobs is not a list')
    return document, hashlib.sha256(raw).hexdigest()


def header_of(document):
    return {key: document[key] for key in document if key != 'jobs'}


def first_ledger_row(directory):
    """Return (relative ledger name, first row | None) or (None, None) when no ledger exists.

    Raises ValueError when a ledger exists but its first row cannot be parsed.
    """
    for name in LEDGER_NAMES:
        path = directory / name
        if not path.is_file():
            continue
        try:
            if name.endswith('.jsonl'):
                with path.open('rb') as stream:
                    line = stream.readline()
                if not line.strip():
                    return name, None
                row = json.loads(line)
            else:
                rows = json.loads(path.read_bytes())
                if not isinstance(rows, list):
                    raise ValueError('ledger is not a list')
                if not rows:
                    return name, None
                row = rows[0]
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f'{name}: {error}')
        if not isinstance(row, dict):
            raise ValueError(f'{name}: first row is not an object')
        return name, row
    return None, None


def read_progress(directory):
    """progress.json as a dict, None when absent; ValueError when present but unreadable."""
    path = directory / 'progress.json'
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f'progress.json: {error}')
    if not isinstance(value, dict):
        raise ValueError('progress.json is not an object')
    return value


def identity_mismatches(job, row, progress):
    """Job-carried resume identity, as the deployed search checks it (gp.rs:705-714, :744-755).

    A field is reported only when both sides carry it and disagree; absent fields are
    never inferred (the service substitutes defaults this tool does not bind).
    """
    mismatched = []
    for name, source in (('progress.json', progress), ('ledger', row)):
        if not isinstance(source, dict):
            continue
        for field in IDENTITY_FIELDS:
            if field in job and field in source and job[field] != source[field]:
                mismatched.append(f'{name}.{field}')
    if isinstance(row, dict) and row.get('evaluation_index') not in (None, 0):
        mismatched.append('ledger.evaluation_index')
    return mismatched


def mentions_identity_mismatch(value):
    if isinstance(value, str):
        return IDENTITY_MISMATCH in value
    if isinstance(value, dict):
        return any(mentions_identity_mismatch(v) for v in value.values())
    if isinstance(value, list):
        return any(mentions_identity_mismatch(v) for v in value)
    return False


def no_failures():
    return dict(attempts=0, first_seen=None, last_seen=None, max_attempt=0, identity_mismatch=False)


def failure_receipts(directory):
    """Group runs/gp-failures/*.json receipts by output (service.rs:292-294, 309-311)."""
    grouped = {}
    directory = Path(directory)
    if not directory.is_dir():
        return grouped
    for path in sorted(directory.glob('*.json')):
        try:
            receipt = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(receipt, dict) or not isinstance(receipt.get('output'), str):
            continue
        entry = grouped.setdefault(receipt['output'], no_failures())
        entry['attempts'] += 1
        stamp = receipt.get('timestamp')
        if type(stamp) in (int, float):
            entry['first_seen'] = stamp if entry['first_seen'] is None else min(entry['first_seen'], stamp)
            entry['last_seen'] = stamp if entry['last_seen'] is None else max(entry['last_seen'], stamp)
        attempt = receipt.get('attempt')
        if type(attempt) is int:
            entry['max_attempt'] = max(entry['max_attempt'], attempt)
        if mentions_identity_mismatch(receipt):
            entry['identity_mismatch'] = True
    return grouped


def classify_job(root, job, evaluator_sha256, failures, max_attempts=None):
    """Return dict(state, reason, ...) with state in {completed, dead, pending, malformed}."""
    if not isinstance(job, dict):
        return dict(state='malformed', reason='job is not an object', output=None)
    output = job.get('output')
    if not safe_relative(output):
        return dict(state='malformed', reason='job output missing or unsafe', output=output)
    directory = root / output
    summary = directory / 'search.json'
    if summary.is_file():
        try:
            document = json.loads(summary.read_text())
        except (ValueError, UnicodeDecodeError) as error:
            return dict(state='malformed', reason=f'search.json unreadable: {error}', output=output)
        if not isinstance(document, dict) or document.get('schema') != SEARCH_SCHEMA:
            return dict(state='malformed', reason=f'search.json schema is not {SEARCH_SCHEMA}', output=output)
        return dict(state='completed', reason=None, output=output)
    try:
        ledger, row = first_ledger_row(directory)
        progress = read_progress(directory)
    except ValueError as error:
        return dict(state='malformed', reason=f'ledger unreadable: {error}', output=output)
    failure = failures.get(output, no_failures())
    base = dict(output=output, ledger=ledger, attempts=failure['attempts'],
                first_seen=failure['first_seen'], last_seen=failure['last_seen'],
                ledger_evaluator_sha256=None, mismatched_fields=[])
    # The service reads a journal unconditionally but a committed observations.json only
    # when progress.json marks the search as resuming; otherwise it starts fresh.
    resumable = row is not None and (ledger.endswith('.jsonl') or progress is not None)
    if row is not None:
        ledger_sha = row.get('evaluator_sha256')
        base['ledger_evaluator_sha256'] = ledger_sha if isinstance(ledger_sha, str) else None
        if resumable and ledger_sha != evaluator_sha256:
            return dict(base, state='dead', reason='ledger_evaluator_mismatch')
    if failure['identity_mismatch']:
        return dict(base, state='dead', reason='identity_mismatch', mismatched_fields=['gp-failures receipt'])
    if resumable:
        mismatched = identity_mismatches(job, row, progress)
        if mismatched:
            return dict(base, state='dead', reason='identity_mismatch', mismatched_fields=mismatched)
        # A first attempt may be between its first journal append and its first
        # progress receipt; only a job that has already failed once is certainly stuck.
        if progress is None and failure['attempts'] >= 1:
            return dict(base, state='dead', reason='identity_mismatch', mismatched_fields=[NO_RESUME_IDENTITY])
    if max_attempts is not None and failure['attempts'] >= max_attempts:
        return dict(base, state='dead', reason='attempts_exhausted')
    return dict(base, state='pending', reason=None, ledger_ignored=row is not None and not resumable)


def audit_jobs(root, document, evaluator_sha256, failures, max_attempts=None):
    """Classify every job; returns (classes, dead_entries, malformed_entries)."""
    classes = dict(completed=[], dead=[], pending=[], malformed=[])
    dead, malformed = [], []
    for index, job in enumerate(document['jobs']):
        result = classify_job(root, job, evaluator_sha256, failures, max_attempts)
        classes[result['state']].append(index)
        if result['state'] == 'dead':
            require(result['reason'] in REASONS, 'unknown dead-letter reason')
            dead.append(dict(output=result['output'], job_sha256=job_sha256(job), reason=result['reason'],
                             attempts=result['attempts'], first_seen=result['first_seen'],
                             last_seen=result['last_seen'],
                             ledger_evaluator_sha256=result['ledger_evaluator_sha256'],
                             mismatched_fields=result['mismatched_fields'],
                             ledger=result['ledger'], index=index))
        elif result['state'] == 'malformed':
            malformed.append(dict(index=index, output=result['output'], reason=result['reason'],
                                  job_sha256=job_sha256(job) if isinstance(job, dict) else None))
    return classes, dead, malformed


def receipt_body(mode, before, removed_completed, removed_dead, kept_pending, kept_malformed,
                 kept_dead_unaudited, lock_path, binding, jobs_path):
    after = before - removed_completed - removed_dead
    require(after == kept_pending + kept_malformed + kept_dead_unaudited, 'GC receipt counts do not add up')
    return dict(schema=RECEIPT_SCHEMA, mode=mode, before=before, removed_completed=removed_completed,
                removed_dead=removed_dead, after=after, kept_pending=kept_pending,
                kept_malformed=kept_malformed, kept_dead_unaudited=kept_dead_unaudited,
                lock_path=str(lock_path), lock_taken=mode == 'apply', priority_path=str(jobs_path),
                manifest_sha256=binding['manifest_sha256'], cpu_executable=binding['cpu_executable'],
                evaluator_sha256=binding['evaluator_sha256'], binding_checked=binding['binding_checked'],
                timestamp=time.time(), applied=mode == 'apply', note=RESTART_NOTE, final_tests_opened=False)


def audit(root, manifest_path, jobs_path, dead_letter_path, receipt_path, failures_dir=None,
          max_attempts=None):
    root = Path(root).resolve()
    jobs_path, dead_letter_path = Path(jobs_path), Path(dead_letter_path)
    binding = expected_evaluator(root, manifest_path)
    document, priority_sha256 = load_jobs(jobs_path)
    failures = failure_receipts(failures_dir or root / 'runs/gp-failures')
    classes, dead, malformed = audit_jobs(root, document, binding['evaluator_sha256'], failures, max_attempts)
    lock_path = lock_path_for(jobs_path)
    dead_letter = dict(schema=DEAD_LETTER_SCHEMA, timestamp=time.time(), priority_path=str(jobs_path),
                       priority_sha256=priority_sha256, manifest_sha256=binding['manifest_sha256'],
                       expected_evaluator_sha256=binding['evaluator_sha256'],
                       completed=[document['jobs'][i]['output'] for i in classes['completed']],
                       entries=dead, malformed=malformed, counts={k: len(v) for k, v in classes.items()},
                       note=RESTART_NOTE, final_tests_opened=False)
    atomic_json(dead_letter_path, dead_letter)
    receipt = receipt_body('audit', len(document['jobs']), len(classes['completed']), len(classes['dead']),
                           len(classes['pending']), len(classes['malformed']), 0, lock_path, binding, jobs_path)
    receipt.update(priority_sha256=priority_sha256, dead_letter_path=str(dead_letter_path),
                   dead_letter_sha256=sha(dead_letter_path))
    atomic_json(receipt_path, receipt)
    return receipt


def verify_audit(jobs_path, dead_letter_path, binding, priority_sha256):
    """Fail closed unless the dead-letter file was produced from the priority file as it is now."""
    jobs_path, dead_letter_path = Path(jobs_path), Path(dead_letter_path)
    require(dead_letter_path.is_file(), f'no audit: run --audit first ({dead_letter_path} absent)')
    dead_letter = read(dead_letter_path)
    require(dead_letter.get('schema') == DEAD_LETTER_SCHEMA, 'dead-letter schema mismatch')
    require(isinstance(dead_letter.get('entries'), list), 'dead-letter file has no entries list')
    require(dead_letter_path.stat().st_mtime >= jobs_path.stat().st_mtime,
            'stale audit: dead-letter file is older than the priority file; re-run --audit')
    require(dead_letter.get('priority_sha256') == priority_sha256,
            'stale audit: priority file changed since --audit; re-run --audit')
    require(dead_letter.get('manifest_sha256') == binding['manifest_sha256']
            and dead_letter.get('expected_evaluator_sha256') == binding['evaluator_sha256'],
            'audit was bound to a different manifest or evaluator; re-run --audit')
    for entry in dead_letter['entries']:
        require(entry.get('reason') in REASONS, 'dead-letter entry has an unknown reason')
    return dead_letter


def apply(root, manifest_path, jobs_path, dead_letter_path, receipt_path, lock_timeout=60.0,
          failures_dir=None):
    root = Path(root).resolve()
    jobs_path, dead_letter_path = Path(jobs_path), Path(dead_letter_path)
    binding = expected_evaluator(root, manifest_path)
    lock_path = lock_path_for(jobs_path)
    handle = acquire_lock(lock_path, lock_timeout)
    try:
        document, priority_sha256 = load_jobs(jobs_path)
        dead_letter = verify_audit(jobs_path, dead_letter_path, binding, priority_sha256)
        dead_by_sha = {e['job_sha256'] for e in dead_letter['entries']}
        failures = failure_receipts(failures_dir or root / 'runs/gp-failures')
        classes, _, _ = audit_jobs(root, document, binding['evaluator_sha256'], failures)
        completed_now, malformed_now, dead_now = (set(classes[k]) for k in ('completed', 'malformed', 'dead'))
        # The sha check above guarantees the audited job bytes, so the audited dead set is honoured
        # as written; completion is re-derived live so a search that finished between audit and
        # apply is dropped by its current state. Anything else is kept untouched.
        kept, removed_completed, removed_dead = [], 0, 0
        kept_pending, kept_malformed, kept_dead_unaudited = 0, 0, 0
        for index, job in enumerate(document['jobs']):
            if index in completed_now:
                removed_completed += 1
            elif isinstance(job, dict) and job_sha256(job) in dead_by_sha:
                removed_dead += 1
            else:
                kept.append(job)
                if index in malformed_now:
                    kept_malformed += 1
                elif index in dead_now:
                    kept_dead_unaudited += 1
                else:
                    kept_pending += 1
        rewritten = dict(document)
        rewritten['jobs'] = kept
        require(header_of(rewritten) == header_of(document), 'GC changed the priority header')
        atomic_bytes(jobs_path, pretty(rewritten))
        receipt = receipt_body('apply', len(document['jobs']), removed_completed, removed_dead, kept_pending,
                               kept_malformed, kept_dead_unaudited, lock_path, binding, jobs_path)
        require(receipt['after'] == len(kept), 'GC receipt after-count disagrees with the rewritten file')
        receipt.update(priority_sha256_before=priority_sha256, priority_sha256_after=sha(jobs_path),
                       dead_letter_path=str(dead_letter_path), dead_letter_sha256=sha(dead_letter_path))
        atomic_json(receipt_path, receipt)
        return receipt
    finally:
        release_lock(handle)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--apply', action='store_true')
    p.add_argument('--jobs', type=Path, default=Path('runs/gp-priority-jobs.json'))
    p.add_argument('--dead-letter', type=Path, default=Path('runs/gp-dead-letter.json'))
    p.add_argument('--failures', type=Path, default=Path('runs/gp-failures'))
    p.add_argument('--receipt', type=Path)
    p.add_argument('--lock-timeout', type=float, default=60.0)
    p.add_argument('--max-attempts', type=int, default=None,
                   help='opt-in attempts_exhausted reason; the deployed service itself retries forever')
    args = p.parse_args(argv)
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    jobs, dead_letter, failures = resolve(args.jobs), resolve(args.dead_letter), resolve(args.failures)
    mode_name = 'audit' if args.audit else 'apply'
    receipt = resolve(args.receipt) if args.receipt else root / f'runs/gp-queue-gc-{mode_name}.json'
    if args.audit:
        result = audit(root, args.manifest.resolve(), jobs, dead_letter, receipt, failures, args.max_attempts)
    else:
        result = apply(root, args.manifest.resolve(), jobs, dead_letter, receipt, args.lock_timeout, failures)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
