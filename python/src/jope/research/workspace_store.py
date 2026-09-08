#!/usr/bin/env python3
"""Filesystem primitives for workspace custody. Standalone for SSH execution.

Published objects are immutable byte snapshots, never assertions that a live
checkpoint or database is resumable. Only source generations are materialized.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')) or '\\' in name or '\0' in name:
        raise ValueError('Unsafe relative path: ' + repr(name))
    return p


def directory(path):
    """Create a private namespace, rejecting all existing symlink components."""
    path = Path(os.path.abspath(path))
    for p in reversed([path, *path.parents]):
        if p.is_symlink():
            raise ValueError('Directory symlink: ' + str(p))
        p.mkdir(exist_ok=True, mode=0o700)
        if not p.is_dir():
            raise ValueError('Not a directory: ' + str(p))
    return path


@contextlib.contextmanager
def opened(root, name):
    """Walk using directory descriptors: neither parent nor leaf links escape."""
    parts = relative(name).parts
    fds = []
    try:
        # Validate the root's parents too. Root must already exist.
        root = Path(os.path.abspath(root))
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        fds.append(fd)
        for part in root.parts[1:] + parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            fds.append(fd)
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('Not a regular file: ' + name)
            yield stream, lambda: os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
    finally:
        for fd in reversed(fds):
            os.close(fd)


def signature(s):
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode, s.st_uid, s.st_gid]


def secret(name, prefix=b''):
    base = Path(name).name.lower()
    return (base in {'.env', 'credentials', 'credentials.json', 'id_rsa', 'id_ed25519', '.netrc', '.npmrc', '.pypirc'}
            or base.startswith('.env.') and not base.endswith(('.example', '.sample'))
            or base.endswith(('.pem', '.key', '.p12', '.pfx', '.passphrase'))
            or 'passphrase' in base or base.endswith('-token.txt')
            or re.search(br'^-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----\r?$', prefix, re.M) is not None)


def stable_hash(root, name, copy_to=None):
    h = hashlib.sha256()
    with opened(root, name) as (stream, current):
        before = os.fstat(stream.fileno())
        first = True
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if first and secret(name, block):
                raise ValueError('Secret excluded: ' + name)
            first = False
            h.update(block)
            if copy_to is not None:
                copy_to.write(block)
        if signature(before) != signature(os.fstat(stream.fileno())) or signature(before) != signature(current()):
            raise ValueError('File changed during hashing: ' + name)
    return {'sha256': h.hexdigest(), 'size': before.st_size,
            'executable': bool(before.st_mode & 0o111), 'uid': before.st_uid, 'gid': before.st_gid,
            'observed_signature': signature(before)}


def write_json(path, value):
    parent = directory(Path(path).parent)
    fd, tmp = tempfile.mkstemp(prefix='.json-', dir=parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        if Path(path).is_symlink():
            raise ValueError('JSON destination symlink')
        os.replace(tmp, path)
        fsync_dir(parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def object_path(state, sha):
    if not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('Invalid object identity')
    return Path(state) / 'objects' / sha


def verify_object(state, record):
    path = object_path(state, record['sha256'])
    actual = stable_hash(path.parent, path.name)
    if any(actual[k] != record[k] for k in ('sha256', 'size')):
        raise ValueError('Object integrity failure: ' + path.name)
    return actual


def publish_object(state, incoming, record):
    """Validate a temporary file before linking it into the published namespace."""
    dest = object_path(state, record['sha256'])
    directory(dest.parent)
    actual = stable_hash(Path(incoming).parent, Path(incoming).name)
    if any(actual[k] != record[k] for k in ('sha256', 'size')):
        raise ValueError('Incoming object hash mismatch')
    if dest.exists() or dest.is_symlink():
        verify_object(state, record)
    else:
        os.chmod(incoming, 0o444)
        with open(incoming, 'rb') as stream:
            os.fsync(stream.fileno())
        try:
            os.link(incoming, dest, follow_symlinks=False)
        except FileExistsError:
            verify_object(state, record)
        fsync_dir(dest.parent)
    os.unlink(incoming)
    return str(dest)


def seal(root, name, state, expected=None):
    """Preserve one observed version; failure leaves no published object."""
    folder = directory(Path(state) / 'incoming')
    fd, temporary = tempfile.mkstemp(prefix='seal-', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as output:
            record = stable_hash(root, name, output)
            output.flush()
            os.fsync(output.fileno())
        if expected and any(record[k] != expected[k] for k in ('sha256', 'size')):
            raise ValueError('Source changed since inventory: ' + name)
        publish_object(state, temporary, record)
        return record
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def under(name, prefixes):
    return any(name == p or name.startswith(p.rstrip('/') + '/') for p in prefixes)


PRUNED = {'.git', '.ssh', '.workspace-mirror', '__pycache__', '.pytest_cache', '.ruff_cache',
          'node_modules', '.conversion-venv', '.venv', '.cache'}
CACHE_SUFFIXES = {'.pyc', '.pyo', '.o', '.rlib', '.rmeta', '.d'}
DATABASE_SUFFIXES = {'.db', '.sqlite', '.sqlite3', '.db-wal', '.db-shm', '.sqlite-wal', '.sqlite-shm'}


def classify(name, prefix, policy):
    p = Path(name)
    if secret(name, prefix):
        return 'machine_local', 'secret'
    if p.suffix in DATABASE_SUFFIXES or prefix.startswith(b'SQLite format 3\0'):
        return 'mutable_snapshot', 'application_snapshot_required'
    if p.suffix in {'.sock', '.lock', '.pid'} and p.name != 'Cargo.lock':
        return 'machine_local', 'lock_or_socket'
    if p.suffix in CACHE_SUFFIXES:
        return 'machine_local', 'disposable_cache'
    if under(name, policy.get('mutable_roots', [])):
        return 'mutable_snapshot', 'capture_bytes_only'
    if under(name, policy['source_roots']):
        return ('dependency' if under(name, policy.get('dependency_roots', [])) else 'source'), 'capture'
    # All remaining regular files are preserved by content snapshot. This includes
    # evidence under build directories. Their application consistency stays pending.
    return 'mutable_snapshot', 'capture_bytes_only'


def inventory(root, policy, source_only=False):
    root = Path(os.path.abspath(root))
    files, excluded, pending = {}, {}, {}
    bases = policy['source_roots'] if source_only else ['.']
    seen = set()
    for base in bases:
        if base != '.':
            relative(base)
        path = root / base
        if not os.path.lexists(path):
            continue
        if path.is_symlink():
            pending[base] = {'reason': 'symlink', 'target': os.readlink(path)}
            continue
        if path.is_file():
            walks = [(path.parent, [], [path.name])]
        else:
            walks = os.walk(path, followlinks=False)
        for parent, dirs, names in walks:
            kept = []
            for d in sorted(dirs):
                child = Path(parent) / d
                name = child.relative_to(root).as_posix()
                if child.is_symlink():
                    pending[name] = {'reason': 'symlink', 'target': os.readlink(child)}
                elif d in PRUNED:
                    excluded[name] = 'git_custody_required' if d == '.git' else 'machine_local_cache'
                elif under(name, policy.get('prune_roots', [])):
                    excluded[name] = 'explicit_duplicate_or_disposable_build_output'
                else:
                    kept.append(d)
            dirs[:] = kept
            for f in sorted(names):
                child = Path(parent) / f
                name = child.relative_to(root).as_posix()
                if name in seen:
                    continue
                seen.add(name)
                try:
                    if child.is_symlink():
                        pending[name] = {'reason': 'symlink', 'target': os.readlink(child)}
                        continue
                    with opened(root, name) as (stream, current):
                        info = os.fstat(stream.fileno())
                        prefix = stream.read(4096)
                    kind, action = classify(name, prefix, policy)
                    if kind == 'machine_local':
                        excluded[name] = action
                        continue
                    if source_only and kind not in ('source', 'dependency'):
                        continue
                    rec = {'class': kind, 'state': action, 'size': info.st_size,
                           'executable': bool(info.st_mode & 0o111), 'uid': info.st_uid, 'gid': info.st_gid,
                           'observed_signature': signature(info)}
                    if action == 'application_snapshot_required':
                        pending[name] = rec
                    elif source_only:
                        rec.update(stable_hash(root, name))
                        files[name] = rec
                    else:
                        files[name] = rec
                except (OSError, ValueError) as error:
                    pending[name] = {'reason': str(error)}
    return {'schema': 'workspace.inventory.v1', 'root': str(root), 'timestamp': time.time(),
            'files': files, 'excluded': excluded, 'pending': pending, 'source_only': source_only}


def source_snapshot(root, policy, state, owner):
    before = inventory(root, policy, source_only=True)
    if before['pending']:
        raise ValueError('Source inventory incomplete: ' + repr(before['pending']))
    records = {}
    for name, row in before['files'].items():
        if object_path(state, row['sha256']).exists():
            verify_object(state, row)
        else:
            seal(root, name, state, row)
        records[name] = {k: row[k] for k in ('sha256', 'size', 'executable', 'uid', 'gid', 'class')}
    after = inventory(root, policy, source_only=True)
    if before['files'] != after['files'] or before['excluded'] != after['excluded'] or after['pending']:
        raise ValueError('Source changed while snapshotting')
    manifest = {'schema': 'workspace.manifest.v1', 'kind': 'source_snapshot', 'owner': owner,
                'provenance': {'root': str(root)}, 'files': records, 'activation_allowed': False}
    manifest['generation'] = identity(manifest)
    target = Path(state) / 'manifests' / (manifest['generation'] + '.json')
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError('Manifest collision')
    else:
        write_json(target, manifest)
    return manifest


def inventory_step(root, policy, state, seconds=15):
    """Bounded, resumable metadata survey; incomplete coverage is explicit.

    A completed survey is retained while the next traversal runs. The durable
    index rechecks each file before copying, and source snapshots are independent.
    """
    root = Path(os.path.abspath(root))
    key = identity({'root': str(root), 'policy': policy})
    path = Path(state) / 'scans' / (key + '.json')
    progress = json.loads(path.read_text()) if path.exists() else None
    if progress is None or progress['complete']:
        progress = {'schema': 'workspace.inventory.v1', 'root': str(root),
                    'timestamp': time.time(), 'files': {}, 'pending': {}, 'excluded': {},
                    'source_only': False, 'queue': [''], 'complete': False}
    deadline = time.monotonic() + seconds
    while progress['queue'] and time.monotonic() < deadline:
        name = progress['queue'].pop()
        child = root / name
        try:
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode):
                progress['pending'][name] = {'reason': 'symlink', 'target': os.readlink(child)}
            elif stat.S_ISDIR(info.st_mode):
                if child.name in PRUNED:
                    progress['excluded'][name] = 'git_custody_required' if child.name == '.git' else 'machine_local_cache'
                elif under(name, policy.get('prune_roots', [])):
                    progress['excluded'][name] = 'explicit_duplicate_or_disposable_build_output'
                else:
                    with os.scandir(child) as entries:
                        progress['queue'].extend(sorted((e.path[len(str(root)) + 1:] for e in entries), reverse=True))
            else:
                with opened(root, name) as (stream, current):
                    info = os.fstat(stream.fileno())
                    prefix = stream.read(4096)
                kind, action = classify(name, prefix, policy)
                rec = {'class': kind, 'state': action, 'size': info.st_size,
                       'executable': bool(info.st_mode & 0o111), 'uid': info.st_uid, 'gid': info.st_gid,
                       'observed_signature': signature(info)}
                if kind == 'machine_local':
                    progress['excluded'][name] = action
                elif action == 'application_snapshot_required':
                    progress['pending'][name] = rec
                else:
                    progress['files'][name] = rec
        except (OSError, ValueError) as error:
            progress['pending'][name] = {'reason': str(error)}
    progress['complete'] = not progress['queue']
    progress['observed_at'] = time.time()
    write_json(path, progress)
    result = {k: v for k, v in progress.items() if k != 'queue'}
    result['unvisited_paths'] = len(progress['queue'])
    result['unvisited_bytes'] = None  # Unknown subtree sizes must never look like zero.
    return result


def validate_manifest(manifest):
    if (manifest.get('schema') != 'workspace.manifest.v1' or manifest.get('activation_allowed') is not False
            or manifest.get('kind') != 'source_snapshot' or not manifest.get('files')
            or identity({k: v for k, v in manifest.items() if k != 'generation'}) != manifest.get('generation')):
        raise ValueError('Manifest identity failure')
    for name, rec in manifest['files'].items():
        relative(name)
        object_path('.', rec['sha256'])
        if type(rec['size']) is not int or rec['size'] < 0 or type(rec['executable']) is not bool:
            raise ValueError('Invalid file record')


def verify_generation(state, manifest):
    validate_manifest(manifest)
    root = Path(state) / 'generations' / manifest['generation']
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Generation missing or indirect')
    actual = set()
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(parent) / name
            if p.is_symlink():
                raise ValueError('Generation symlink')
        actual.update((Path(parent) / f).relative_to(root).as_posix() for f in files)
    if actual != set(manifest['files']):
        raise ValueError('Generation path coverage differs')
    for name, rec in manifest['files'].items():
        observed = stable_hash(root, name)
        if any(observed[k] != rec[k] for k in ('sha256', 'size', 'executable')):
            raise ValueError('Generation integrity failure: ' + name)
    return {'generation': manifest['generation'], 'files': len(actual), 'verified': True}


def materialize(state, manifest):
    validate_manifest(manifest)
    final = Path(state) / 'generations' / manifest['generation']
    directory(final.parent)
    if final.exists() or final.is_symlink():
        return verify_generation(state, manifest)
    temporary = Path(tempfile.mkdtemp(prefix='.generation-', dir=final.parent))
    try:
        for name, rec in manifest['files'].items():
            verify_object(state, rec)
            dest = temporary / relative(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Separate inode: changing permissions must not mutate shared objects.
            shutil.copyfile(object_path(state, rec['sha256']), dest, follow_symlinks=False)
            dest.chmod(0o555 if rec['executable'] else 0o444)
            with dest.open('rb') as stream:
                os.fsync(stream.fileno())
        write_json(Path(state) / 'manifests' / (manifest['generation'] + '.json'), manifest)
        for p in sorted(temporary.rglob('*'), reverse=True):
            if p.is_dir():
                fsync_dir(p)
                p.chmod(0o555)
        fsync_dir(temporary)
        temporary.chmod(0o555)
        temporary.rename(final)
        fsync_dir(final.parent)
        return verify_generation(state, manifest)
    finally:
        if temporary.exists():
            temporary.chmod(0o700)
            for p in temporary.rglob('*'):
                if p.is_dir():
                    p.chmod(0o700)
            shutil.rmtree(temporary)


def dispatch(request):
    op = request['op']
    state = request['state']
    directory(state)
    if op == 'inventory':
        return inventory(request['root'], request['policy'], request.get('source_only', False))
    if op == 'inventory_step':
        return inventory_step(request['root'], request['policy'], state, request.get('seconds', 15))
    if op == 'snapshot':
        return source_snapshot(request['root'], request['policy'], state, request['owner'])
    if op == 'seal':
        return seal(request['root'], request['name'], state, request.get('expected'))
    if op == 'has':
        result = []
        for row in request['records']:
            if object_path(state, row['sha256']).exists():
                verify_object(state, row)
                result.append(row['sha256'])
        return result
    if op == 'prepare_transfer':
        folder = directory(Path(state) / 'transfer')
        for rec in request['records']:
            name = object_path(state, rec['sha256']).name
            if (folder / name).is_symlink():
                raise ValueError('Transfer symlink')
        if (folder / '.partial').is_symlink():
            raise ValueError('Partial directory symlink')
        return str(folder)
    if op == 'publish_many':
        for rec in request['records']:
            publish_object(state, Path(state) / 'transfer' / rec['sha256'], rec)
        return len(request['records'])
    if op == 'receipt':
        value = request['value']
        digest = identity(value)
        write_json(Path(state) / 'receipts' / (digest + '.json'), value)
        return digest
    if op == 'incoming':
        directory(Path(state) / 'incoming')
        p = Path(state) / 'incoming' / (request['sha256'] + '.partial')
        object_path(state, request['sha256'])
        if p.is_symlink():
            raise ValueError('Partial destination symlink')
        return str(p)
    if op == 'publish':
        p = Path(state) / 'incoming' / (request['record']['sha256'] + '.partial')
        return publish_object(state, p, request['record'])
    if op == 'materialize':
        return materialize(state, request['manifest'])
    if op == 'verify':
        return verify_generation(state, request['manifest'])
    raise ValueError('Unknown worker operation: ' + op)
