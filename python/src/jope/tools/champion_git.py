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
    require(HEX64.match(name) and digest(data) == name, '%s: event bytes do not hash to the file name' % path)
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


def commit_event(repo, event, key, base):
    paths = write_event(repo, event, key, base)
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
                         cut=identity.get('cut'), evaluator_sha256=identity.get('evaluator_sha256'), rmse_original_units=current.get('rmse_original_units'),
                         improvement_number=current.get('improvement_number'), events=len(events), event_shas=[e['sha256'] for e in events],
                         last_event=latest['value'].get('timestamp') if latest and 'value' in latest else None, chain=events,
                         invalid=[e['sha256'] for e in events if 'error' in e]))
    return rows


def status_table(rows, state, filters=()):
    published = state.get('published', {}) if state else {}
    verified_head = state.get('verified_head') if state else None
    out = []
    for r in rows:
        if filters and not any(f in (r['dataset'] or '') or f in (r['family'] or '') for f in filters):
            continue
        committed = sum(1 for s in r['event_shas'] if s in published)
        out.append(dict(family=r['family'], dataset=r['dataset'], cut=r['cut'], partition=r['partition'], rmse=r['rmse_original_units'],
                        improvements=r['improvement_number'], events=r['events'], committed_events=committed, uncommitted_events=r['events'] - committed,
                        last_event=r['last_event'], pushed='verified' if (verified_head and committed == r['events'] and state['remote'].get('status') == 'verified') else 'no',
                        invalid=len(r['invalid'])))
    return out


def sync(root, store, repo, remote, branch, credentials, evaluator, vocabulary, all_evaluators, limit, receipts):
    repo = ensure_repo(repo)
    lock = open(repo / '.champion-git.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = load_state(repo)
    rows = scan(store, evaluator, vocabulary, all_evaluators)
    committed, skipped, errors = [], [], []
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
            head = commit_event(repo, event, key, base)
            state['published'][event['sha256']] = dict(commit=head, identity=key, at=time.time())
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
                   invalid_events=errors[:20], invalid_count=len(errors), push=remote_state, claim=CLAIM, final_tests_opened=False)
    if receipts:
        Path(receipts).mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        (Path(receipts) / ('champion-git-%s-%s.json' % (stamp, (head or 'none')[:12]))).write_bytes(json.dumps(receipt, sort_keys=True, indent=1).encode())
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
    parser.add_argument('--filter', action='append', default=[], help='status: dataset or family substring (repeatable)')
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
        rows = status_table(scan(store, evaluator, vocabulary, args.all_evaluators), state, args.filter)
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
    receipt, committed = sync(root, store, Path(args.repo), args.remote, args.branch, args.credentials, evaluator, vocabulary, args.all_evaluators, args.limit, args.receipts)
    print(json.dumps(dict(head=receipt['head'], committed=receipt['committed'], identities=receipt['identities'], events_total=receipt['events_total'],
                          invalid=receipt['invalid_count'], push=receipt['push']['status'], push_error=receipt['push'].get('last_error')), sort_keys=True))
    return receipt


if __name__ == '__main__':
    main()
