#!/usr/bin/env python3
"""Install a verified immutable controller and enable the xbabe2 mirror timer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

import workspace_store as store

ROOT = Path(__file__).resolve().parents[1]
FILES = ('scripts/workspace_sync.py', 'scripts/workspace_store.py', 'scripts/workspace_inventory.py',
         'parallel/tools/kernel_remote_state.py', 'parallel/tools/kernel_bundle.py',
         'parallel/tools/kernel_release.py', 'manifests/workspace-sync.v1.json', 'manifests/workspace-sync.v2.json')


def install():
    config = json.loads((ROOT / 'manifests/workspace-sync.v1.json').read_text())
    if socket.gethostname() != config['merge_host']:
        raise ValueError('Controller installation is restricted to the merge host')
    records = {name: store.stable_hash(ROOT, name) for name in FILES}
    hashes = {name: row['sha256'] for name, row in records.items()}
    release_id = store.identity(hashes)
    parent = store.directory(Path(config['local']['state']) / 'controller')
    final = parent / release_id
    if not final.exists():
        temporary = Path(tempfile.mkdtemp(prefix='.install-', dir=parent))
        try:
            for name in FILES:
                dest = temporary / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
                if store.stable_hash(temporary, name)['sha256'] != hashes[name]:
                    raise ValueError('Controller changed during installation')
                dest.chmod(0o444)
            store.write_json(temporary / 'controller.json', {'schema': 'workspace.controller.v1', 'identity': release_id, 'files': hashes})
            for p in temporary.rglob('*'):
                if p.is_dir():
                    p.chmod(0o555)
            temporary.chmod(0o555)
            temporary.rename(final)
        finally:
            if temporary.exists():
                temporary.chmod(0o700)
                for p in temporary.rglob('*'):
                    if p.is_dir():
                        p.chmod(0o700)
                shutil.rmtree(temporary)
    for name, digest in hashes.items():
        if store.stable_hash(final, name)['sha256'] != digest:
            raise ValueError('Installed controller integrity failure')
    units = [ROOT / 'systemd' / ('jope-workspace-sync.' + suffix) for suffix in ('service', 'timer')]
    subprocess.run(['systemd-analyze', 'verify', *map(str, units)], check=True)
    current = parent / 'current'
    if current.exists() and not current.is_symlink():
        raise ValueError('Refusing to replace an existing controller directory')
    link = parent / ('.current-' + release_id)
    link.symlink_to(final.name, target_is_directory=True)
    os.replace(link, current)
    for unit in units:
        subprocess.run(['sudo', '-n', 'install', '-m', '0644', str(unit), '/etc/systemd/system/' + unit.name], check=True)
    subprocess.run(['sudo', '-n', 'systemctl', 'daemon-reload'], check=True)
    subprocess.run(['sudo', '-n', 'systemctl', 'enable', '--now', 'jope-workspace-sync.timer'], check=True)
    print(json.dumps({'controller': str(final), 'identity': release_id, 'verified': True}))


if __name__ == '__main__':
    install()
