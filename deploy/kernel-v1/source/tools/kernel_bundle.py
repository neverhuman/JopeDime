#!/usr/bin/env python3
"""Freeze runtime source, verification instructions, probes and prior evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from kernel_release import digest, stage, verify_release

DOCUMENTS = (
    'docs/ALGORITHM_KERNEL_AUDIT_2026-09-08.md',
    'docs/ALGORITHM_KERNEL_IMPLEMENTATION_STATUS_2026-09-08.md',
    'docs/KRANTZ_KERNEL_ASSESSMENT_2026-09-08.md',
    'docs/ALGORITHM_KERNEL_VERIFICATION_ACTIONS_2026-09-08.md',
)


def files_under(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Expected a real directory: {root}')
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f'Bundle symlink rejected: {path}')
        for name in names:
            yield Path(parent) / name


def file_record(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular file: {path}')
    return {'sha256': digest(path), 'bytes': path.stat().st_size,
            'executable': bool(path.stat().st_mode & 0o111)}


def bundle_identity(files, release_id):
    body = json.dumps({'files': files, 'release_id': release_id}, sort_keys=True,
                      separators=(',', ':')).encode()
    return 'kernel-work-v1-' + hashlib.sha256(body).hexdigest()


def verify_bundle(bundle, require_name=True):
    bundle = Path(bundle)
    if bundle.is_symlink():
        raise ValueError('Bundle root symlink rejected')
    receipt = json.loads((bundle / 'bundle.json').read_text())
    files = {str(p.relative_to(bundle)): file_record(p) for p in files_under(bundle)
             if p != bundle / 'bundle.json'}
    if (receipt['schema'] != 'kernel.work-bundle.v1' or files != receipt['files']
            or bundle_identity(files, receipt['release_id']) != receipt['bundle_id']
            or (require_name and bundle.name != receipt['bundle_id'])
            or receipt.get('activation_allowed') is not False):
        raise ValueError('Bundle integrity failure')
    release = bundle / 'runtime' / receipt['release_id']
    if verify_release(release)['release_id'] != receipt['release_id']:
        raise ValueError('Nested release identity mismatch')
    return receipt


def pack(repository, destination):
    repository, destination = Path(repository).resolve(), Path(destination).resolve()
    release = stage(repository / 'parallel', repository / 'parallel/releases')
    sources = {str(Path('runtime') / release.name / p.relative_to(release)): p
               for p in files_under(release)}
    for name in (*DOCUMENTS, 'Cargo.toml', 'Cargo.lock'):
        sources['workspace/' + name] = repository / name
    for name in ('docs/audits/algorithm-kernel-upgrade-2026-09-08',
                 'research/algorithm-kernel-regimes'):
        for p in files_under(repository / name):
            if '__pycache__' not in p.parts:
                sources[str(Path('workspace') / p.relative_to(repository))] = p
    records = {name: file_record(p) for name, p in sorted(sources.items())}
    bundle_id = bundle_identity(records, release.name)
    final = destination / bundle_id
    if final.exists():
        verify_bundle(final)
        return final
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.packing-', dir=destination))
    try:
        for name, source in sources.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
        if records != {name: file_record(p) for name, p in sources.items()}:
            raise ValueError('Source changed while packaging')
        copied = {str(p.relative_to(temporary)): file_record(p) for p in files_under(temporary)}
        if records != copied:
            raise ValueError('Copied bytes differ')
        (temporary / 'bundle.json').write_text(json.dumps({
            'schema': 'kernel.work-bundle.v1', 'bundle_id': bundle_id,
            'release_id': release.name, 'files': records, 'activation_allowed': False,
            'scope': 'Source, test instructions and development evidence; no runtime cutover',
        }, indent=2) + '\n')
        for p in sorted(temporary.rglob('*'), reverse=True):
            p.chmod(0o555 if p.is_dir() or p.stat().st_mode & 0o111 else 0o444)
        temporary.chmod(0o555)
        temporary.rename(final)
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            for p in temporary.rglob('*'):
                if p.is_dir():
                    p.chmod(0o755)
            shutil.rmtree(temporary)
    verify_bundle(final)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('pack')
    create.add_argument('--repository', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('verify')
    check.add_argument('bundle', type=Path)
    args = parser.parse_args()
    if args.command == 'pack':
        print(pack(args.repository, args.output))
    else:
        print(verify_bundle(args.bundle)['bundle_id'])
