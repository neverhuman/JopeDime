#!/usr/bin/env python3
"""Champion git flywheel (explicit one-shot administration): every new per-dataset-fold high mark becomes a commit.

The GP service's only high-mark writer is champions::record (parallel/src/champions.rs): one directory per
identity (dataset, family, partition, cut, prepared/training/validation/evaluator/vocabulary sha) under
runs/champions/<identity hash>/ holding current.json and one immutable event-<sha256(champion)>.json per strict
improvement (parallel.cut_champion.v1). This tool mirrors those events into a git repository so a new winner is a
commit within minutes and, when a remote and a credentials file exist, a push:

  status   read-only table of the live identities (evaluator = manifest bindings[cpu_executable] by default):
           family / dataset / cut / partition / best rmse / improvements / last event / committed? / pushed?
  sync     import every unpublished event into the repository (one commit per event, chronological per identity,
           layout below), then push when --remote is configured and the credentials file exists; a refused push
           (403, missing repository, network) is journaled with backoff and never blocks the next sync; local
           commits are the durable record either way.

Layout (dedicated small-text repository, no LFS):
  champions/<dataset>/cut-<cut>/<identity hash>/identity.json
  champions/<dataset>/cut-<cut>/<identity hash>/current.json      (event sha, program sha, rmse, improvement number)
  champions/<dataset>/cut-<cut>/<identity hash>/events/<event sha>/champion.json   (verbatim event bytes)
  index/identities.jsonl, index/README.md
The claim carried by every record: "local record high on the adaptive cut-heldout teacher; not a held-out win, not an
independent evaluation, not a serving promotion". meta_test identities are refused (never fitted, never published).
Sealed data is never read: the tool opens nothing under data/ and copies only champion event bytes.

Credentials: --credentials names a git-credentials file (https://<login>:<token>@host) owned by the user with mode
0600; the tool checks existence, owner and mode, passes it to git through `credential.helper=store --file=<path>`
with GIT_TERMINAL_PROMPT=0 and never reads, prints or journals its content.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

from resume_expired_cycle import require, sha

SCHEMA = 'parallel.cut_champion.v1'
STATE_SCHEMA = 'parallel.champion_git_state.v1'
RECEIPT_SCHEMA = 'parallel.champion_git_receipt.v1'
FEED_SCHEMA = 'parallel.champion_git_feed.v2'
FEED_JOIN_RULE = 'a running search is committed <commit> only when its output observations.json path equals an event source and the observation_sha256 matches; a coarse (dataset, cut, partition) join is UNVERIFIED'
CLAIM = 'local record high on the adaptive cut-heldout teacher; not a held-out win, not an independent evaluation, not a serving promotion'
PUBLISHABLE_PARTITIONS = ('meta_train', 'meta_validation')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
BACKOFF_INITIAL, BACKOFF_CAP = 300, 3600
AUTHOR = 'jope-champion-git <champions@jope-dime.local>'


# ---------------------------------------------------------------- champion store

def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_event(path):
    """One immutable event; its file name must be the sha256 of its bytes (champions.rs names events by content)."""
    data = path.read_bytes()
    value = json.loads(data)
    require(isinstance(value, dict) and value.get('schema') == SCHEMA, '%s: schema is not %s' % (path, SCHEMA))
    name = path.name[len('event-'):-len('.json')]
    # champions.rs names an event by sha256(serde_json::to_vec(champion)): compact json in struct field order, which the
    # pretty-printed file preserves; re-serializing the parsed document compactly in file order reproduces those bytes.
    ordered = json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()
    require(HEX64.match(name) and (digest(ordered) == name or digest(data) == name), '%s: event bytes do not hash to the file name' % path)
    identity = value.get('identity')
    require(isinstance(identity, dict) and all(isinstance(identity.get(k), (str, int)) for k in
                                                ('dataset', 'family', 'partition', 'cut', 'prepared_sha256', 'evaluator_sha256', 'vocabulary_sha256')),
            '%s: identity incomplete' % path)
    require(type(value.get('rmse_original_units')) in (int, float), '%s: rmse_original_units missing' % path)
    require(type(value.get('improvement_number')) is int, '%s: improvement_number missing' % path)
    return dict(sha256=name, bytes=data, value=value, path=path)


def identity_dirs(store):
    return sorted(p for p in Path(store).iterdir() if p.is_dir() and (p / 'current.json').is_file())


def identity_chain(directory):
    """Every event of one identity in chronological order (improvement_number ascending), verified by content hash."""
    events = []
    for path in directory.glob('event-*.json'):
        try:
            events.append(load_event(path))
        except (RuntimeError, ValueError, OSError) as error:
            events.append(dict(sha256=path.name[6:-5], error=str(error), path=path))
    valid = sorted((e for e in events if 'error' in e or True), key=lambda e: (e.get('value', {}).get('improvement_number', 0), e['sha256']))
    return valid


def live_evaluator(root, manifest):
    doc = json.loads((Path(root) / manifest).read_bytes())
    executable = doc.get('cpu_executable')
    bound = (doc.get('bindings') or {}).get(executable)
    require(isinstance(bound, str) and HEX64.match(bound), 'manifest does not bind cpu_executable')
    return bound, doc.get('vocabulary_sha256')


# ---------------------------------------------------------------- repository

def git(repo, *args, env=None, timeout=600):
    command = ['git', '-C', str(repo), '-c', 'user.name=jope-champion-git', '-c', 'user.email=champions@jope-dime.local', *args]
    merged = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    if env:
        merged.update(env)
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=merged)


def ensure_repo(repo):
    repo = Path(repo)
    if not (repo / '.git').is_dir():
        repo.mkdir(parents=True, exist_ok=True)
        result = git(repo, 'init', '-q', '-b', 'main')
        require(result.returncode == 0, 'git init failed: ' + result.stderr.strip())
    return repo


def credentials_ok(path):
    """The credentials file must exist, belong to the caller and be mode 0600; its content is never read."""
    if path is None:
        return False, 'no credentials file configured'
    try:
        info = os.stat(path)
    except OSError as error:
        return False, 'credentials file unavailable: ' + error.strerror
    if info.st_uid != os.getuid():
        return False, 'credentials file is not owned by the current user'
    if stat.S_IMODE(info.st_mode) != 0o600:
        return False, 'credentials file mode is not 0600'
    if info.st_size == 0:
        return False, 'credentials file is empty'
    return True, 'present'


def scrub(text):
    return re.sub(r'://[^@\s]+@', '://<redacted>@', text or '')


def state_path(repo):
    return Path(repo) / '.champion-git-state.json'


def load_state(repo):
    path = state_path(repo)
    if path.is_file():
        value = json.loads(path.read_bytes())
        require(value.get('schema') == STATE_SCHEMA, 'state schema changed')
        return value
    return dict(schema=STATE_SCHEMA, identities={}, published={}, remote=dict(status='never', backoff_seconds=BACKOFF_INITIAL, next_attempt=0.0, last_error=None),
                verified_head=None)


def save_state(repo, state):
    path = state_path(repo)
    temporary = path.with_suffix('.partial')
    temporary.write_bytes(json.dumps(state, sort_keys=True, indent=1).encode())
    os.replace(temporary, path)


def record_layout(event):
    value = event['value']
    identity = value['identity']
    key = digest(compact(identity))
    base = Path('champions') / str(identity['dataset']) / ('cut-%s' % identity['cut']) / key
    return key, base


def write_event(repo, event, key, base):
    value = event['value']
    identity = value['identity']
    directory = repo / base
    (directory / 'events' / event['sha256']).mkdir(parents=True, exist_ok=True)
    (directory / 'events' / event['sha256'] / 'champion.json').write_bytes(event['bytes'])
    (directory / 'identity.json').write_bytes(json.dumps(dict(identity, identity_key=key, claim=CLAIM, final_tests_opened=False), sort_keys=True, indent=1).encode())
    observation = value.get('observation') or {}
    (directory / 'current.json').write_bytes(json.dumps(dict(
        event_sha256=event['sha256'], program_sha256=observation.get('program_sha256'), rmse_original_units=value['rmse_original_units'],
        improvement_number=value['improvement_number'], timestamp=value.get('timestamp'), source=value.get('source'),
        evaluator_sha256=identity.get('evaluator_sha256'), claim=CLAIM, final_tests_opened=False), sort_keys=True, indent=1).encode())
    return [str(base / 'events' / event['sha256'] / 'champion.json'), str(base / 'identity.json'), str(base / 'current.json')]


def prepared_view_for(root, event):
    """The prepared view directory of a champion, resolved through its source ledger's generation jobs.json (the job whose
    output directory prefixes the source path); None when the generation or job cannot be found."""
    source = str(event['value'].get('source') or '')
    marker = source.find('/generation-')
    if marker < 0:
        return None
    end = source.find('/', marker + 1)
    generation_dir = Path(root) / source[:end] if end > 0 else None
    jobs_path = generation_dir / 'jobs.json' if generation_dir else None
    if not jobs_path or not jobs_path.is_file():
        return None
    try:
        jobs = json.loads(jobs_path.read_bytes()).get('jobs') or []
    except (OSError, ValueError):
        return None
    for job in jobs:
        output = job.get('output')
        if isinstance(output, str) and source.startswith(output) and job.get('prepared_sha256') == event['value']['identity'].get('prepared_sha256'):
            return job.get('prepared')
    return None


def export_event(root, repo, event, export_binary, timeout=600):
    """Run the champion export binary for one event into programs/<program sha>/<event sha>/ inside the repository; returns the
    receipt summary (never raises: an export failure is recorded, the champion record is still committed)."""
    program = ((event['value'].get('observation') or {}).get('program_sha256')) or 'unknown'
    target = Path(repo) / 'programs' / program / event['sha256']
    if (target / 'export.json').is_file():
        return dict(status='present', path=str(target.relative_to(repo)))
    prepared = prepared_view_for(root, event)
    if prepared is None:
        return dict(status='unresolved_prepared_view')
    if target.exists():
        return dict(status='partial_export_present', path=str(target.relative_to(repo)))
    command = [str(export_binary), '--root', str(root), '--event', str(event['path']), '--prepared', prepared, '--output', str(target)]
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=dict(os.environ, OMP_NUM_THREADS='1'))
    except (OSError, subprocess.SubprocessError) as error:
        return dict(status='export_error', error=str(error)[:300])
    parity = None
    if (target / 'parity.json').is_file():
        try:
            parity = json.loads((target / 'parity.json').read_bytes()).get('status')
        except ValueError:
            parity = None
    return dict(status='exported' if run.returncode == 0 else 'parity_failed', exit=run.returncode, parity=parity, path=str(target.relative_to(repo)),
                stderr=run.stderr[-300:] if run.returncode != 0 else None)


def commit_event(repo, event, key, base, export=None):
    paths = write_event(repo, event, key, base)
    if export and export.get('path') and export.get('status') in ('exported', 'parity_failed', 'present'):
        for name in ('predict.py', 'predict.rs', 'predict.cpp', 'predict.js', 'export.json', 'parity.json', 'artifact-manifest.json'):
            if (repo / export['path'] / name).is_file():
                paths.append(str(Path(export['path']) / name))
    add = git(repo, 'add', '--', *paths)
    require(add.returncode == 0, 'git add failed: ' + add.stderr.strip())
    identity = event['value']['identity']
    message = 'champion: %s/%s cut %s rmse %.6g (#%d) event %s' % (identity.get('family'), identity.get('dataset'), identity.get('cut'),
                                                                   event['value']['rmse_original_units'], event['value']['improvement_number'], event['sha256'][:12])
    result = git(repo, 'commit', '-q', '--author', AUTHOR, '-m', message, '--', *paths)
    require(result.returncode == 0, 'git commit failed: ' + result.stderr.strip())
    head = git(repo, 'rev-parse', 'HEAD')
    return head.stdout.strip()


def write_index(repo, rows):
    (repo / 'index').mkdir(exist_ok=True)
    lines = [json.dumps(r, sort_keys=True) for r in sorted(rows, key=lambda r: (r['family'], r['dataset'], r['cut'], r['identity_key']))]
    (repo / 'index' / 'identities.jsonl').write_text('\n'.join(lines) + ('\n' if lines else ''))
    readme = ['# jope champions', '', CLAIM + '.', '', 'identities: %d; events: %d' % (len(rows), sum(r['events'] for r in rows)), '',
              '| family | dataset | cut | partition | best rmse (original units) | improvements | evaluator |', '|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['family'], r['dataset'], r['cut'])):
        readme.append('| %s | %s | %s | %s | %.6g | %d | %s |' % (r['family'], r['dataset'], r['cut'], r['partition'], r['rmse_original_units'], r['events'], r['evaluator_sha256'][:8]))
    (repo / 'index' / 'README.md').write_text('\n'.join(readme) + '\n')
    add = git(repo, 'add', '--', 'index/identities.jsonl', 'index/README.md')
    require(add.returncode == 0, 'git add index failed: ' + add.stderr.strip())
    status = git(repo, 'status', '--porcelain', '--', 'index')
    if status.stdout.strip():
        result = git(repo, 'commit', '-q', '--author', AUTHOR, '-m', 'index: %d identities, %d events' % (len(rows), sum(r['events'] for r in rows)), '--', 'index')
        require(result.returncode == 0, 'git commit index failed: ' + result.stderr.strip())


# ---------------------------------------------------------------- push

def push(repo, state, remote, branch, credentials):
    now = time.time()
    entry = state['remote']
    if remote is None:
        entry.update(status='no_remote', last_error=None)
        return entry
    ok, why = credentials_ok(credentials)
    if not ok:
        entry.update(status='no_credentials', last_error=why)
        return entry
    if now < entry.get('next_attempt', 0.0):
        entry.update(status='backoff', last_error=entry.get('last_error'))
        return entry
    remotes = git(repo, 'remote')
    if 'origin' not in remotes.stdout.split():
        require(git(repo, 'remote', 'add', 'origin', remote).returncode == 0, 'git remote add failed')
    else:
        git(repo, 'remote', 'set-url', 'origin', remote)
    helper = ['-c', 'credential.helper=', '-c', 'credential.helper=store --file=%s' % credentials]
    result = git(repo, *helper, 'push', '-q', 'origin', 'HEAD:refs/heads/%s' % branch, timeout=300)
    if result.returncode != 0:
        text = scrub(result.stderr.strip())
        kind = 'refused_403' if '403' in text else 'repo_missing' if '404' in text or 'not found' in text.lower() else 'auth' if 'Authentication' in text or '401' in text else 'network'
        backoff = min(BACKOFF_CAP, max(BACKOFF_INITIAL, int(entry.get('backoff_seconds', BACKOFF_INITIAL)) * 2))
        entry.update(status=kind, last_error=text[-400:], backoff_seconds=backoff, next_attempt=now + backoff, attempted_at=now)
        return entry
    readback = git(repo, *helper, 'ls-remote', 'origin', 'refs/heads/%s' % branch, timeout=120)
    head = git(repo, 'rev-parse', 'HEAD').stdout.strip()
    remote_head = readback.stdout.split()[0] if readback.returncode == 0 and readback.stdout.split() else None
    verified = remote_head == head
    entry.update(status='verified' if verified else 'pushed_unverified', last_error=None if verified else scrub(readback.stderr.strip())[-200:],
                 backoff_seconds=BACKOFF_INITIAL, next_attempt=0.0, attempted_at=now, remote_head=remote_head)
    if verified:
        state['verified_head'] = head
    return entry


# ---------------------------------------------------------------- actions

def scan(store, evaluator=None, vocabulary=None, all_evaluators=False):
    """Identity rows of the store (live evaluator only unless all_evaluators), excluding meta_test."""
    rows = []
    for directory in identity_dirs(store):
        try:
            current = json.loads((directory / 'current.json').read_bytes())
        except (OSError, ValueError):
            continue
        identity = current.get('identity') or {}
        if identity.get('partition') not in PUBLISHABLE_PARTITIONS:
            continue
        if not all_evaluators and evaluator is not None and identity.get('evaluator_sha256') != evaluator:
            continue
        if not all_evaluators and vocabulary is not None and identity.get('vocabulary_sha256') != vocabulary:
            continue
        key = digest(compact(identity))
        events = identity_chain(directory)
        latest = events[-1] if events else None
        rows.append(dict(identity_key=key, directory=directory, dataset=identity.get('dataset'), family=identity.get('family'), partition=identity.get('partition'),
                         cut=identity.get('cut'), evaluator_sha256=identity.get('evaluator_sha256'), prepared_sha256=identity.get('prepared_sha256'),
                         rmse_original_units=current.get('rmse_original_units'),
                         improvement_number=current.get('improvement_number'), events=len(events), event_shas=[e['sha256'] for e in events],
                         last_event=latest['value'].get('timestamp') if latest and 'value' in latest else None, chain=events,
                         invalid=[e['sha256'] for e in events if 'error' in e]))
    return rows


def event_provenance(event, published):
    """The exact producer of one high mark: the search output that produced it, the observation and program digests, the commit."""
    value = event.get('value') or {}
    observation = value.get('observation') or {}
    return dict(sha256=event['sha256'], source=value.get('source'), observation_sha256=value.get('observation_sha256'), program_sha256=observation.get('program_sha256'),
                rmse_original_units=value.get('rmse_original_units'), improvement_number=value.get('improvement_number'), timestamp=value.get('timestamp'),
                commit=(published.get(event['sha256']) or {}).get('commit'), invalid=event.get('error'))


def status_table(rows, state, filters=(), provenance=False):
    published = state.get('published', {}) if state else {}
    verified_head = state.get('verified_head') if state else None
    out = []
    for r in rows:
        if filters and not any(f in (r['dataset'] or '') or f in (r['family'] or '') for f in filters):
            continue
        committed = sum(1 for s in r['event_shas'] if s in published)
        last_committed = next((published[s]['commit'] for s in reversed(r['event_shas']) if s in published), None)
        row = dict(identity_key=r['identity_key'], family=r['family'], dataset=r['dataset'], cut=r['cut'], partition=r['partition'], prepared_sha256=r.get('prepared_sha256'),
                   rmse=r['rmse_original_units'], improvements=r['improvement_number'], events=r['events'], committed_events=committed, uncommitted_events=r['events'] - committed,
                   last_event=r['last_event'], last_commit=last_committed, current_event=r['event_shas'][-1] if r['event_shas'] else None,
                   pushed='verified' if (verified_head and committed == r['events'] and state['remote'].get('status') == 'verified') else 'no',
                   invalid=len(r['invalid']))
        if provenance:
            history = [event_provenance(e, published) for e in r['chain']]
            row['history'] = history
            row['current'] = history[-1] if history else None
        out.append(row)
    return out


def write_feed(path, rows, state, evaluator, head, receipt=None):
    """Atomic console feed (parallel.champion_git_feed.v2): per identity the git state of its high marks with exact provenance.

    A (dataset, cut, partition) join proves nothing about a running search; each row therefore carries `history` — every
    high-mark event with the `source` observations.json that produced it, its observation and program digests and its
    champion-git commit — and `current`, the latest of them. The console labels a search `committed <commit>` only when
    the search's own output path equals an event's `source` and the observation digest matches; otherwise it is
    UNVERIFIED. Overwritten on every sync.
    """
    table = status_table(rows, state, provenance=True)
    feed = dict(schema=FEED_SCHEMA, at=time.time(), head=head, evaluator_sha256=evaluator, remote=(state or {}).get('remote'), last_sync=receipt,
                identities=len(table), events=sum(r['events'] for r in table), committed_events=sum(r['committed_events'] for r in table),
                uncommitted_events=sum(r['uncommitted_events'] for r in table), rows=sorted(table, key=lambda r: (r['family'] or '', r['dataset'] or '', r['cut'] if r['cut'] is not None else -1)),
                join_rule=FEED_JOIN_RULE, claim=CLAIM, final_tests_opened=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.%s.%d.partial' % (path.name, os.getpid()))
    temporary.write_bytes(json.dumps(feed, sort_keys=True).encode())
    os.replace(temporary, path)
    return feed


def sync(root, store, repo, remote, branch, credentials, evaluator, vocabulary, all_evaluators, limit, receipts, export_binary=None, export_limit=None, feed=None):
    repo = ensure_repo(repo)
    lock = open(repo / '.champion-git.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = load_state(repo)
    rows = scan(store, evaluator, vocabulary, all_evaluators)
    committed, skipped, errors, exports, exports_run = [], [], [], [], 0
    for r in rows:
        for event in r['chain']:
            if event['sha256'] in state['published']:
                continue
            if 'error' in event:
                errors.append(dict(identity=r['identity_key'], event=event['sha256'], error=event['error']))
                continue
            if limit is not None and len(committed) >= limit:
                skipped.append(event['sha256'])
                continue
            key, base = record_layout(event)
            export = None
            if export_binary and (export_limit is None or exports_run < export_limit):
                export = export_event(root, repo, event, export_binary)
                exports_run += export.get('status') in ('exported', 'parity_failed')
                exports.append(dict(event=event['sha256'], **{k: v for k, v in export.items() if k != 'stderr'}))
            head = commit_event(repo, event, key, base, export)
            state['published'][event['sha256']] = dict(commit=head, identity=key, at=time.time(), export=(export or {}).get('status'))
            state['identities'][key] = dict(events=state['identities'].get(key, {}).get('events', 0) + 1, current=event['sha256'])
            committed.append(dict(identity=key, event=event['sha256'], commit=head, rmse=event['value']['rmse_original_units']))
    index_rows = [dict(identity_key=r['identity_key'], family=r['family'], dataset=r['dataset'], cut=r['cut'], partition=r['partition'],
                       rmse_original_units=r['rmse_original_units'], events=r['events'], evaluator_sha256=r['evaluator_sha256'],
                       current_event=r['event_shas'][-1] if r['event_shas'] else None) for r in rows]
    if committed:
        write_index(repo, index_rows)
    remote_state = push(repo, state, remote, branch, credentials)
    save_state(repo, state)
    head = git(repo, 'rev-parse', 'HEAD').stdout.strip() or None
    receipt = dict(schema=RECEIPT_SCHEMA, at=time.time(), root=str(root), store=str(store), repo=str(repo), head=head, remote=remote, branch=branch,
                   credentials_path=str(credentials) if credentials else None, evaluator_sha256=None if all_evaluators else evaluator,
                   identities=len(rows), events_total=sum(r['events'] for r in rows), committed=len(committed), skipped_by_limit=len(skipped),
                   invalid_events=errors[:20], invalid_count=len(errors), exports=exports, export_binary=str(export_binary) if export_binary else None,
                   push=remote_state, claim=CLAIM, final_tests_opened=False)
    if receipts:
        Path(receipts).mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        (Path(receipts) / ('champion-git-%s-%s.json' % (stamp, (head or 'none')[:12]))).write_bytes(json.dumps(receipt, sort_keys=True, indent=1).encode())
    if feed:
        write_feed(feed, rows, state, evaluator, head, dict(at=receipt['at'], committed=len(committed), invalid_count=len(errors), push=remote_state.get('status')))
    fcntl.flock(lock, fcntl.LOCK_UN)
    return receipt, committed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('status', 'sync'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', help='continuous manifest within root; the live evaluator is bindings[cpu_executable]')
    parser.add_argument('--store', default='runs/champions', help='champion store within root')
    parser.add_argument('--repo', help='sync: git repository directory (created if absent)')
    parser.add_argument('--remote', help='sync: remote url (https://git.neverhuman.org/git/<owner>/<repo>.git)')
    parser.add_argument('--branch', default='main')
    parser.add_argument('--credentials', help='sync: git-credentials file (0600, never read by this tool)')
    parser.add_argument('--all-evaluators', action='store_true', help='include identities of every evaluator (default: the live one)')
    parser.add_argument('--limit', type=int, help='sync: at most this many new event commits per run')
    parser.add_argument('--receipts', help='sync: directory for immutable run receipts')
    parser.add_argument('--export-binary', help='sync: champion_export binary; emitted code + parity receipt are committed beside each new event')
    parser.add_argument('--export-limit', type=int, help='sync: at most this many exports per run (a parity run fits and executes Python)')
    parser.add_argument('--filter', action='append', default=[], help='status: dataset or family substring (repeatable)')
    parser.add_argument('--feed', help='atomic parallel.champion_git_feed.v1 for the console (written by status and by every sync)')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    store = root / args.store
    require(store.is_dir(), 'champion store missing: ' + str(store))
    evaluator = vocabulary = None
    if args.manifest and not args.all_evaluators:
        evaluator, vocabulary = live_evaluator(root, args.manifest)
    if args.action == 'status':
        state = load_state(args.repo) if args.repo and state_path(args.repo).is_file() else None
        scanned = scan(store, evaluator, vocabulary, args.all_evaluators)
        if args.feed:
            write_feed(args.feed, scanned, state, evaluator, git(Path(args.repo), 'rev-parse', 'HEAD').stdout.strip() or None if state else None)
        rows = status_table(scanned, state, args.filter)
        if args.json:
            print(json.dumps(dict(evaluator_sha256=evaluator, repo=args.repo, remote_status=(state or {}).get('remote'), rows=rows), sort_keys=True))
        else:
            print('%-34s %-18s %3s %-15s %12s %5s %6s %9s %7s %s' % ('family', 'dataset', 'cut', 'partition', 'best rmse', 'impr', 'events', 'uncommit', 'pushed', 'last event (UTC)'))
            for r in rows:
                when = datetime.datetime.fromtimestamp(r['last_event'], datetime.timezone.utc).strftime('%m-%d %H:%M') if r['last_event'] else '-'
                print('%-34s %-18s %3s %-15s %12.6g %5s %6d %9d %7s %s' % (r['family'][:34], r['dataset'][:18], r['cut'], r['partition'], r['rmse'] if r['rmse'] is not None else float('nan'),
                                                                        r['improvements'], r['events'], r['uncommitted_events'], r['pushed'], when))
            print('identities %d | events %d | uncommitted %d | remote %s' % (len(rows), sum(r['events'] for r in rows), sum(r['uncommitted_events'] for r in rows),
                                                                             (state or {}).get('remote', {}).get('status', 'no repository')))
        return rows
    require(args.repo, 'sync requires --repo')
    receipt, committed = sync(root, store, Path(args.repo), args.remote, args.branch, args.credentials, evaluator, vocabulary, args.all_evaluators, args.limit, args.receipts,
                              args.export_binary, args.export_limit, args.feed)
    print(json.dumps(dict(head=receipt['head'], committed=receipt['committed'], identities=receipt['identities'], events_total=receipt['events_total'],
                          invalid=receipt['invalid_count'], exports=len(receipt['exports']), push=receipt['push']['status'], push_error=receipt['push'].get('last_error')), sort_keys=True))
    return receipt


if __name__ == '__main__':
    main()
