#!/usr/bin/env python3
"""Refuse source removal unless a bounded live tree still equals its custody inventory."""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def kind(mode):
    if stat.S_ISDIR(mode):
        return 'directory'
    if stat.S_ISREG(mode):
        return 'file'
    if stat.S_ISLNK(mode):
        return 'symlink'
    return 'special'


def main():
    if len(sys.argv) != 5:
        raise SystemExit('usage: jope-source-verify.py INVENTORY PREFIX ROOT RECEIPT')
    inventory, prefix, root, receipt = map(Path, sys.argv[1:])
    root = root.resolve(strict=True)
    prefix_bytes = os.fsencode(str(prefix))
    if not prefix_bytes or prefix_bytes.startswith(b'/') or any(
            part in (b'', b'.', b'..') for part in prefix_bytes.split(b'/')):
        raise ValueError('prefix must be a safe relative path')
    records = {}
    with inventory.open('rb') as stream:
        for raw in stream:
            row = json.loads(raw)
            if row.get('error'):
                raise ValueError('inventory has unreadable or changing entries')
            name = bytes.fromhex(row['path_hex'])
            if name == prefix_bytes or name.startswith(prefix_bytes + b'/'):
                if name in records or any(part in (b'', b'.', b'..') for part in name.split(b'/')):
                    raise ValueError('duplicate or unsafe inventory path')
                if row.get('source_root') != str(root):
                    raise ValueError('inventory belongs to another root')
                records[name] = row
    if prefix_bytes not in records:
        raise ValueError('inventory lacks the portion root')
    required = set(records)
    found = set()
    bytes_hashed = 0
    stack = [prefix_bytes]
    root_bytes = os.fsencode(root)
    while stack:
        name = stack.pop()
        if name not in required or name in found:
            raise ValueError('source contains extra or duplicate path: ' + repr(name[:200]))
        found.add(name)
        path = root_bytes + b'/' + name
        before = os.lstat(path)
        row = records[name]
        if (kind(before.st_mode) != row['type'] or before.st_size != row['size'] or
                before.st_mode != row['mode'] or before.st_uid != row['uid'] or
                before.st_gid != row['gid'] or
                before.st_mtime_ns != row['mtime_sec'] * 1000000000 + row['mtime_nsec']):
            raise ValueError('source metadata differs: ' + repr(name[:200]))
        if row['type'] == 'file':
            if sha256(path) != row['sha256']:
                raise ValueError('source SHA-256 differs: ' + repr(name[:200]))
            after = os.lstat(path)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('source changed during read: ' + repr(name[:200]))
            bytes_hashed += before.st_size
        elif row['type'] == 'symlink':
            if os.readlink(path).hex() != row.get('link_target_hex'):
                raise ValueError('source symlink target differs: ' + repr(name[:200]))
        elif row['type'] == 'directory':
            with os.scandir(path) as entries:
                for child in entries:
                    stack.append(name + b'/' + child.name)
    if found != required:
        raise ValueError(f'{len(required - found)} inventoried entries absent from source')
    document = {'schema': 'jope.source-custody-verify/v1', 'inventory': str(inventory),
                'inventory_sha256': sha256(inventory), 'prefix': str(prefix),
                'root': str(root), 'entries': len(found), 'verified_file_bytes': bytes_hashed,
                'all_paths_and_hashes_match': True, 'posix_metadata_match': True}
    with receipt.open('x') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'entries': len(found), 'verified_file_bytes': bytes_hashed}))


if __name__ == '__main__':
    main()
