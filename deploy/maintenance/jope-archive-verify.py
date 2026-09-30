#!/usr/bin/env python3
"""Compare every encrypted tar member with a complete independent source census."""
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from decimal import Decimal
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 6:
        raise SystemExit('usage: archive-verify.py INVENTORY PREFIX ARCHIVE KEY_FILE RECEIPT')
    inventory, prefix, archive, key_file, receipt = map(Path, sys.argv[1:])
    prefix_bytes = os.fsencode(str(prefix))
    required = {}
    with inventory.open('rb') as stream:
        for raw in stream:
            row = json.loads(raw)
            path_hex = row.get('path_hex')
            if path_hex is None:
                if row.get('error'):
                    raise ValueError('incomplete census has traversal errors')
                continue
            name = bytes.fromhex(path_hex)
            if name == prefix_bytes or name.startswith(prefix_bytes + b'/'):
                if row.get('error'):
                    raise ValueError('incomplete census entry under shard')
                required[name] = row
    if not required:
        raise ValueError('no census entries for shard')
    usb = json.loads(Path(str(archive) + '.receipt.json').read_text())
    if usb.get('archive') != str(archive) or not usb.get('full_readback') or usb.get('source_deleted'):
        raise ValueError('USB readback receipt is invalid')
    archive_sha = digest(archive)
    if not Path(str(archive) + '.sha256').read_text().startswith(archive_sha + ' '):
        raise ValueError('USB archive SHA-256 differs')
    inventory_sha = digest(inventory)
    gpg = subprocess.Popen(['gpg', '--batch', '--quiet', '--pinentry-mode', 'loopback',
                            '--passphrase-file', str(key_file), '--decrypt', '--output', '-',
                            str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    zstd = subprocess.Popen(['zstd', '-dc', '-q'], stdin=gpg.stdout,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gpg.stdout.close()
    seen = set()
    hardlinks = []
    total_bytes = 0
    counts = {'file': 0, 'directory': 0, 'symlink': 0, 'special': 0}
    try:
        with tarfile.open(fileobj=zstd.stdout, mode='r|') as tar:
            for member in tar:
                name = os.fsencode(member.name).rstrip(b'/')
                row = required.get(name)
                if row is None or name in seen:
                    raise ValueError('unexpected or duplicate tar member: ' + repr(name[:200]))
                seen.add(name)
                kind = row['type']
                if member.islnk():
                    if kind != 'file':
                        raise ValueError('hardlink type differs: ' + repr(name[:200]))
                    hardlinks.append((name, os.fsencode(member.linkname)))
                elif member.isfile():
                    if kind != 'file' or member.size != row['size']:
                        raise ValueError('file type or size differs: ' + repr(name[:200]))
                    body = tar.extractfile(member)
                    h = hashlib.sha256()
                    for block in iter(lambda: body.read(1024 * 1024), b''):
                        h.update(block)
                    if h.hexdigest() != row['sha256']:
                        raise ValueError('file SHA-256 differs: ' + repr(name[:200]))
                    total_bytes += member.size
                elif member.isdir():
                    if kind != 'directory':
                        raise ValueError('directory type differs: ' + repr(name[:200]))
                elif member.issym():
                    if kind != 'symlink' or os.fsencode(member.linkname).hex() != row.get('link_target_hex'):
                        raise ValueError('symlink target differs: ' + repr(name[:200]))
                elif kind != 'special':
                    raise ValueError('special file type differs: ' + repr(name[:200]))
                if stat.S_IMODE(row['mode']) != member.mode or row['uid'] != member.uid or row['gid'] != member.gid:
                    raise ValueError('POSIX metadata differs: ' + repr(name[:200]))
                mtime = Decimal(member.pax_headers.get('mtime', str(int(member.mtime))))
                expected_mtime = Decimal(row['mtime_sec']) + Decimal(row['mtime_nsec']) / Decimal(1000000000)
                if mtime != expected_mtime:
                    raise ValueError('mtime differs: ' + repr(name[:200]))
                counts[kind] += 1
        zstd.stdout.close()
        zstd_error = zstd.stderr.read(1000)
        gpg_error = gpg.stderr.read(1000)
        if zstd.wait() or gpg.wait():
            raise ValueError('decrypt/decompress failed: ' + repr(zstd_error + gpg_error))
        for name, target in hardlinks:
            if target not in seen or required[name]['sha256'] != required[target]['sha256'] or required[name]['size'] != required[target]['size']:
                raise ValueError('hardlink content binding differs: ' + repr(name[:200]))
        if seen != set(required):
            raise ValueError(f'{len(set(required) - seen)} census entries absent from shard')
    except Exception:
        zstd.kill()
        gpg.kill()
        zstd.wait()
        gpg.wait()
        raise
    document = {'schema': 'jope.usb-custody-verify/v1', 'inventory': str(inventory),
                'inventory_sha256': inventory_sha, 'prefix': str(prefix),
                'archive': str(archive), 'archive_sha256': archive_sha,
                'entries': len(seen), 'counts': counts, 'verified_file_bytes': total_bytes,
                'all_paths_and_hashes_match': True, 'posix_metadata_match': True}
    with receipt.open('x') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'entries': len(seen), 'verified_file_bytes': total_bytes,
                      'archive_sha256': archive_sha}))


if __name__ == '__main__':
    main()
