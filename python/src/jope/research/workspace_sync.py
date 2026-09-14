#!/usr/bin/env python3
"""Inventory, reconcile and mirror workspace snapshots without activating runtime.

Only xbabe2 publishes canonical generations. Both nodes retain incoming source
snapshots and versioned durable bytes. A timer never edits a working checkout.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import workspace_store as store

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'parallel/tools'))
from kernel_remote_state import compare, compare_health
from kernel_bundle import verify_bundle
from kernel_release import verify_release


class BulkWindowEnded(Exception):
    pass


class Mirror:
    def __init__(self, config):
        self.config = config
        self.repository = Path(config['local']['root'])
        self.state = store.directory(config['local']['state'])
        self.remote = config['remote']
        self.ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
                    '-i', str(Path(self.remote['key']).expanduser()), self.remote['host']]
        self.worker = Path(__file__).with_name('workspace_store.py').read_text()
        self.health_source = (ROOT / 'parallel/tools/kernel_remote_state.py').read_text()
        self.baseline = None

    def rpc(self, op, remote=False, **values):
        node = self.config['remote' if remote else 'local']
        request = dict(op=op, state=node['state'], root=node['root'],
                       policy=self.config['policy'], owner=node['name'])
        request.update(values)
        if not remote:
            return store.dispatch(request)
        code = self.worker + '\nprint(json.dumps(dispatch(json.loads(' + repr(json.dumps(request)) + '))))\n'
        result = subprocess.run(self.ssh + ['nice -n 15 ionice -c 3 python3 -'],
                                input=code, text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError('Remote worker failed: ' + result.stderr[-2000:])
        return json.loads(result.stdout)

    def health(self):
        code = self.health_source + '\nprint(json.dumps(capture(' + repr(self.remote['root'] + '/parallel') + ')))\n'
        result = subprocess.run(self.ssh + ['nice -n 15 ionice -c 3 python3 -'],
                                input=code, text=True, capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError('Live preservation observation unavailable: ' + result.stderr[-1000:])
        return json.loads(result.stdout)

    def healthy(self):
        current = self.health()
        before = self.baseline or current
        preservation, health = compare(before, current), compare_health(before, current)
        if not preservation['stable_identities_unchanged'] or health['status'] != 'passed':
            raise RuntimeError('Runtime preservation/health not verified: ' + json.dumps(dict(preservation=preservation, health=health)))
        hardware = current['telemetry']['hardware.json']
        for field in ('available_gb', 'disk_free_gb', 'cpu_total'):
            value = hardware.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise RuntimeError('Resource evidence unavailable: ' + field)
        if hardware['available_gb'] < 4 or hardware['disk_free_gb'] < 10:
            raise RuntimeError('Bulk paused: remote memory or disk reserve reached')
        if shutil.disk_usage(self.state).free < 10 * 1024**3:
            raise RuntimeError('Bulk paused: local disk reserve reached')
        return current

    def transfer(self, records, from_remote=False, deadline=None):
        """Sequential transfers provide one aggregate bandwidth limit in both directions."""
        present = set(self.rpc('has', remote=not from_remote, records=records))
        missing = {rec['sha256']: rec for rec in records if rec['sha256'] not in present}
        records = list(missing.values())
        if records:
            self.healthy()
            dest = self.rpc('prepare_transfer', remote=not from_remote, records=records) + '/'
            source_state = self.remote['state'] if from_remote else self.config['local']['state']
            source = str(Path(source_state) / 'objects') + '/'
            if from_remote:
                source = self.remote['host'] + ':' + source
            else:
                dest = self.remote['host'] + ':' + dest
            # The partial directory is private and never a published object. Rsync
            # resumes from it; publication independently verifies the complete hash.
            names = tempfile.NamedTemporaryFile(mode='w', dir=self.state, prefix='.transfer-list-', delete=False)
            with names:
                names.write('\n'.join(missing) + '\n')
            command = ['rsync', '-t', '--partial', '--partial-dir=.partial', '--protect-args',
                       '--checksum', '--files-from=' + names.name,
                       '--bwlimit=' + str(self.config['bandwidth_kib_s']),
                       '--rsync-path=nice -n 15 ionice -c 3 rsync',
                       '-e', shlex.join(self.ssh[:-1]), source, dest]
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            try:
                while True:
                    try:
                        _, error = process.communicate(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        self.healthy()
                        if deadline is not None and time.monotonic() >= deadline:
                            raise BulkWindowEnded('Partial transfer retained for the next timer run')
                if process.returncode:
                    raise RuntimeError('Transfer interrupted; partial retained: ' + error.decode()[-1000:])
                self.healthy()
                self.rpc('publish_many', remote=not from_remote, records=records)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, __import__('signal').SIGTERM)
                    try:
                        process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, __import__('signal').SIGKILL)
                        process.communicate()
                os.unlink(names.name)
        return sum(rec['size'] for rec in records)

    def snapshots(self):
        local = self.rpc('snapshot')
        remote = self.rpc('snapshot', remote=True)
        store.write_json(self.state / 'incoming-source.json', {'local': local, 'remote': remote, 'timestamp': time.time()})
        return local, remote

    def source_plan(self, local, remote):
        rows = []
        for path in sorted(local['files'].keys() | remote['files'].keys()):
            left, right = local['files'].get(path), remote['files'].get(path)
            def key(row):
                return None if row is None else (row['sha256'], row['size'], row['executable'])
            if key(left) != key(right):
                rows.append({'path': path, 'local': left, 'remote': right,
                             'status': 'conflict' if left and right else 'local_only' if left else 'remote_only'})
        return {'schema': 'workspace.plan.v1', 'merge_host': 'xbabe2',
                'local_snapshot': local['generation'], 'remote_snapshot': remote['generation'],
                'candidates': rows, 'automatic_resolution': False}

    def inventory(self, incremental=False):
        result = {}
        operation = 'inventory_step' if incremental else 'inventory'
        for node in ('local', 'remote'):
            result[node] = self.rpc(operation, remote=node == 'remote')
            store.write_json(self.state / ('inventory-' + node + '.json'), result[node])
            for label, spec in self.config[node].get('durable_roots', {}).items():
                key = node + '/' + label
                result[key] = self.rpc(operation, remote=node == 'remote', root=spec['path'],
                                       policy={'source_roots': [], 'prune_roots': spec.get('prune_roots', [])})
                store.write_json(self.state / ('inventory-' + node + '-' + label + '.json'), result[key])
        return result

    def durable(self, inventories, deadline):
        path = self.state / 'durable-index.json'
        index = json.loads(path.read_text()) if path.exists() else {'schema': 'workspace.durable-index.v1', 'files': {}, 'history': {}}
        todo = []
        for node, inv in inventories.items():
            for name, row in inv['files'].items():
                if row['class'] in ('source', 'dependency'):
                    continue
                key = node + ':' + name
                old = index['files'].get(key)
                if old and old['observed_signature'] == row['observed_signature']:
                    continue
                todo.append((row['size'], node, name, row))
        # Small manifests, reports and receipts precede large historical archives.
        todo.sort(key=lambda x: (x[0], x[1], x[2]))
        completed, moved = set(), 0
        for _, node, name, observed in todo:
            if time.monotonic() >= deadline:
                break
            self.healthy()
            from_remote = node.split('/')[0] == 'remote'
            try:
                rec = self.rpc('seal', remote=from_remote, name=name, root=inventories[node]['root'])
            except (OSError, ValueError, RuntimeError) as error:
                index.setdefault('pending', {})[node + ':' + name] = {'reason': str(error), 'observed_at': time.time()}
                store.write_json(path, index)
                continue
            # If an active file changed since discovery, retain the captured bytes
            # but defer any coverage assertion to the next inventory.
            if rec['observed_signature'] != observed['observed_signature']:
                continue
            try:
                moved += self.transfer([rec], from_remote=from_remote, deadline=deadline)
            except BulkWindowEnded:
                break
            key = node + ':' + name
            old = index['files'].get(key)
            if old:
                index['history'].setdefault(key, []).append(old)
            index['files'][key] = dict(rec, verified_at=time.time(),
                                      application_consistency='unqualified_byte_snapshot',
                                      resumable=False, provenance={'node': node, 'path': name})
            index.setdefault('pending', {}).pop(key, None)
            store.write_json(path, index)
            completed.add((node, name))
        remaining = [row for row in todo if (row[1], row[2]) not in completed]
        store.write_json(path, index)
        index_id = self.rpc('receipt', remote=True, value=index)
        if index_id != store.identity(index):
            raise ValueError('Durable provenance index mirror differs')
        store.write_json(self.state / 'receipts' / (index_id + '.json'), index)
        return {'files_remaining': len(remaining), 'bytes_remaining': sum(x[0] for x in remaining),
                'verified_versions': len(index['files']), 'transferred_bytes': moved,
                'index_sha256': index_id,
                'application_consistency': 'pending application receipts; byte snapshots are not resumable checkpoints',
                'pending_live_files': sum(len(v['pending']) for v in inventories.values()),
                'pending_capture_errors': len(index.get('pending', {})),
                'excluded_git_requires_custody': True}

    def sync(self, publish=None, bulk_seconds=180):
        receipt_dir = store.directory(self.state / 'receipts' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]))
        receipt = {'schema': 'workspace.sync-receipt.v1', 'started': time.time(),
                   'status': 'incomplete', 'activation_allowed': False, 'runtime_qualification': 'not_attempted',
                   'source_parity': 'unverified', 'source_conflicts': None,
                   'artifacts': {'files_remaining': None, 'bytes_remaining': None,
                                 'coverage': 'unknown until inventory completes'},
                   'runtime_version': 'unobserved'}
        pointer = self.state / 'canonical.json'
        if pointer.exists():
            receipt['canonical_generation'] = json.loads(pointer.read_text())['generation']
        try:
            self.baseline = self.healthy()
            store.write_json(receipt_dir / 'before.json', self.baseline)
            local, remote = self.snapshots()
            # Existing release identities and their verifiers remain authoritative.
            receipt['kernel_bundle_verification'] = []
            for folder, verifier in (('kernel-work', verify_bundle), ('releases', verify_release)):
                parent = self.repository / 'parallel' / folder
                if parent.exists():
                    for path in sorted(parent.iterdir()):
                        if path.name.startswith(('kernel-work-v1-', 'kernel-v1-')) and path.is_dir():
                            verifier(path)
                            receipt['kernel_bundle_verification'].append(str(path.relative_to(self.repository)))
            plan = self.source_plan(local, remote)
            store.write_json(self.state / 'plan.json', plan)
            receipt['source_candidates'] = len(plan['candidates'])
            receipt['source_conflicts'] = sum(x['status'] == 'conflict' for x in plan['candidates'])
            # Custody both source snapshots on both hosts, including remote edits.
            for manifest, origin_remote in ((remote, True), (local, False)):
                self.transfer(list(manifest['files'].values()), from_remote=origin_remote)
                self.rpc('materialize', manifest=manifest)
                self.rpc('materialize', remote=True, manifest=manifest)
            pointer = self.state / 'canonical.json'
            if publish:
                if publish != local['generation']:
                    raise ValueError('Publish requires the current reviewed local snapshot identity; run plan first')
                store.write_json(pointer, {'generation': publish, 'published_at': time.time(),
                                           'merge_host': 'xbabe2', 'manifest': local})
            canonical = json.loads(pointer.read_text()) if pointer.exists() else None
            if canonical:
                manifest = canonical['manifest']
                self.transfer(list(manifest['files'].values()))
                receipt['source_parity'] = {'local': self.rpc('verify', manifest=manifest),
                                            'remote': self.rpc('materialize', remote=True, manifest=manifest)}
                receipt['canonical_generation'] = canonical['generation']
            else:
                receipt['source_parity'] = 'pending explicit publication'
            inventories = self.inventory(incremental=True)
            receipt['artifacts'] = self.durable(inventories, time.monotonic() + bulk_seconds)
            receipt['inventory_coverage'] = {k: {'complete': v.get('complete', True),
                                                'unvisited_paths': v.get('unvisited_paths', 0),
                                                'unvisited_bytes': v.get('unvisited_bytes'),
                                                'survey_started': v['timestamp']}
                                             for k, v in inventories.items()}
            after = self.healthy()
            store.write_json(receipt_dir / 'after.json', after)
            receipt['preservation'] = compare(self.baseline, after)
            receipt['health'] = compare_health(self.baseline, after)
            receipt['runtime_version'] = {k: {'exe': v['exe'], 'sha256': v['exe_sha256'],
                                             'invocation': v['properties']['InvocationID']}
                                          for k, v in after['stable'].items() if k.startswith('jope-parallel-')}
            receipt['status'] = 'source_verified_backfill_pending' if canonical else 'snapshots_verified_publication_pending'
            # Completion is deliberately stricter than transport success. Application
            # snapshots, external roots and historical gitlinks have separate gates.
            receipt['synchronized'] = False
        except Exception as error:
            receipt.update(status='paused', synchronized=False, error=str(error))
            try:
                after = self.health()
                store.write_json(receipt_dir / 'after.json', after)
                receipt['runtime_version'] = {k: {'exe': v['exe'], 'sha256': v['exe_sha256'],
                                                  'invocation': v['properties']['InvocationID']}
                                              for k, v in after['stable'].items() if k.startswith('jope-parallel-')}
                if self.baseline:
                    receipt['preservation'] = compare(self.baseline, after)
                    receipt['health'] = compare_health(self.baseline, after)
            except Exception as observation:
                receipt['health'] = {'status': 'inconclusive', 'reason': str(observation)}
        receipt['finished'] = time.time()
        try:
            digest = self.rpc('receipt', remote=True, value=receipt)
            if digest != store.identity(receipt):
                raise ValueError('Remote receipt identity differs')
        except Exception as error:
            receipt.update(status='paused', synchronized=False, receipt_mirror_error=str(error))
        store.write_json(receipt_dir / 'receipt.json', receipt)
        store.write_json(self.state / 'status.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'manifests/workspace-sync.v1.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inventory')
    sub.add_parser('plan')
    sync = sub.add_parser('sync')
    sync.add_argument('--publish', help='Explicit, reviewed local source snapshot SHA-256')
    sync.add_argument('--bulk-seconds', type=int, default=180)
    verify = sub.add_parser('verify')
    verify.add_argument('--durable', action='store_true', help='Rehash all mirrored durable objects on both nodes')
    sub.add_parser('status')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('schema') != 'workspace.sync-config.v1' or config['bandwidth_kib_s'] != 8192:
        raise ValueError('Unknown policy or bandwidth drift')
    if socket.gethostname() != config['merge_host']:
        raise ValueError('Only the configured merge host may run synchronization')
    mirror = Mirror(config)
    if args.command == 'status':
        path = mirror.state / 'status.json'
        result = json.loads(path.read_text()) if path.exists() else {'status': 'never_run', 'synchronized': False}
        snapshots = mirror.state / 'incoming-source.json'
        result['snapshot_age_seconds'] = time.time() - json.loads(snapshots.read_text())['timestamp'] if snapshots.exists() else None
        result['last_attempt_age_seconds'] = time.time() - result['finished'] if 'finished' in result else None
        print(json.dumps(result, indent=2))
        return
    with (mirror.state / 'sync.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status': 'already_running', 'synchronized': False}))
            return
        if args.command == 'inventory':
            inventories = mirror.inventory()
            result = {k: {'files': len(v['files']), 'bytes': sum(x['size'] for x in v['files'].values()),
                          'pending': len(v['pending']), 'excluded': len(v['excluded'])} for k, v in inventories.items()}
        elif args.command == 'plan':
            mirror.baseline = mirror.healthy()
            snapshots = mirror.snapshots()
            result = mirror.source_plan(*snapshots)
            after = mirror.healthy()
            result['preservation'] = compare(mirror.baseline, after)
            result['health'] = compare_health(mirror.baseline, after)
            store.write_json(mirror.state / 'plan.json', result)
        elif args.command == 'verify':
            canonical = json.loads((mirror.state / 'canonical.json').read_text())['manifest']
            result = {k: mirror.rpc('verify', remote=k == 'remote', manifest=canonical) for k in ('local', 'remote')}
            result['durable_coverage'] = 'consult status; source verification alone does not establish durable parity'
            if args.durable:
                index = json.loads((mirror.state / 'durable-index.json').read_text())
                records = {r['sha256']: r for r in index['files'].values()}
                for history in index['history'].values():
                    for row in history:
                        records[row['sha256']] = row
                result['durable_objects_verified'] = {}
                for node in ('local', 'remote'):
                    found = mirror.rpc('has', remote=node == 'remote', records=list(records.values()))
                    if set(found) != set(records):
                        raise ValueError('Durable object coverage incomplete on ' + node)
                    result['durable_objects_verified'][node] = len(found)
        else:
            if args.bulk_seconds < 0:
                raise ValueError('Bulk duration must be nonnegative')
            if args.publish:
                # Local publication can complete while remote preservation evidence
                # is unavailable. The timer stages this exact generation when safe.
                manifest = mirror.rpc('snapshot')
                if manifest['generation'] != args.publish:
                    raise ValueError('Source changed since the reviewed publication identity')
                mirror.rpc('materialize', manifest=manifest)
                store.write_json(mirror.state / 'canonical.json',
                                 {'generation': args.publish, 'manifest': manifest,
                                  'merge_host': 'xbabe2', 'published_at': time.time()})
            result = mirror.sync(args.publish, args.bulk_seconds)
        print(json.dumps(result, indent=2))
        if result.get('status') == 'paused':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
