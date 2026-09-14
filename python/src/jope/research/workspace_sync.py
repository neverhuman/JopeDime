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
        self.inventory_worker = Path(__file__).with_name('workspace_inventory.py').read_text()
        self.health_source = (ROOT / 'parallel/tools/kernel_remote_state.py').read_text()
        self.baseline = None

    def rpc(self, op, remote=False, **values):
        node = self.config['remote' if remote else 'local']
        request = dict(op=op, state=node['state'], root=node['root'],
                       policy=self.config['policy'], owner=node['name'])
        request.update(values)
        if not remote:
            return store.dispatch(request)
        # Execute the exact controller-bound indexed implementation, never an
        # unrelated installed module on the remote host.
        code = 'import sys, types, json\n'
        for name, source in (('workspace_store', self.worker), ('workspace_inventory', self.inventory_worker)):
            code += ('module = types.ModuleType(' + repr(name) + ')\n'
                     'module.__file__ = ' + repr(self.remote['root'] + '/scripts/' + name + '.py') + '\n'
                     'sys.modules[' + repr(name) + '] = module\n'
                     'exec(' + repr(source) + ', module.__dict__)\n')
        code += 'print(json.dumps(sys.modules["workspace_store"].dispatch(json.loads(' + repr(json.dumps(request)) + '))))\n'
        result = subprocess.run(self.ssh + ['nice -n 15 ionice -c 3 python3 -'],
                                input=code, text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError('Remote worker failed: ' + result.stderr[-2000:])
        if op in ('inventory_page', 'inventory_capture', 'inventory_ack', 'inventory_status') and len(result.stdout.encode()) > 524288:
            raise RuntimeError('Indexed inventory RPC exceeded its bounded response')
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
        if hardware['available_gb'] < 4:
            raise RuntimeError('Bulk paused: remote memory or disk reserve reached')
        local_space = shutil.disk_usage(self.state)
        if local_space.free < max(250 * 1024**3, (local_space.total + 9) // 10):
            raise RuntimeError('Bulk paused: local disk reserve reached')
        return current

    def transfer(self, records, from_remote=False, deadline=None):
        """Sequential transfers provide one aggregate bandwidth limit in both directions."""
        if deadline is not None and time.monotonic() >= deadline:
            raise BulkWindowEnded('Backfill window ended before transfer preparation')
        present = set(self.rpc('has', remote=not from_remote, records=records))
        missing = {rec['sha256']: rec for rec in records if rec['sha256'] not in present}
        records = list(missing.values())
        if records:
            self.healthy()
            # This reservation covers this immutable transfer batch only. It
            # cannot establish capacity for an unknown full-project backlog.
            space = self.rpc('capacity', remote=not from_remote)
            required = space['required_reserve_bytes'] + 3 * sum(r['size'] for r in records)
            required += 2 * 1024**3  # Bounded batch metadata and growth headroom.
            if space['free_bytes'] < required:
                raise RuntimeError('Transfer batch capacity shortfall: ' + str(required - space['free_bytes']))
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
            process = None
            try:
                # Hash/capacity/health RPCs can consume the remaining window.
                # They must not authorize a new transport after its deadline.
                if deadline is not None and time.monotonic() >= deadline:
                    raise BulkWindowEnded('Backfill window ended before starting transfer')
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
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
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, __import__('signal').SIGTERM)
                    try:
                        process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, __import__('signal').SIGKILL)
                        process.communicate()
                os.unlink(names.name)
        return sum(rec['size'] for rec in records)

    def snapshots(self):
        local = self.rpc('snapshot_observed')
        remote = self.rpc('snapshot_observed', remote=True)
        store.write_json(self.state / 'incoming-source.json', {'local': local, 'remote': remote, 'timestamp': time.time()})
        return local, remote

    def source_plan(self, local, remote):
        rows = []
        for path in sorted(local['files'].keys() | remote['files'].keys()):
            left, right = local['files'].get(path), remote['files'].get(path)
            def key(row):
                return None if row is None else (row['sha256'], row['size'], row['executable'], row.get('mode'))
            if key(left) != key(right):
                rows.append({'path': path, 'local': left, 'remote': right,
                             'status': 'conflict' if left and right else 'local_only' if left else 'remote_only'})
        return {'schema': 'workspace.plan.v1', 'merge_host': 'xbabe2',
                'local_snapshot': local['generation'], 'remote_snapshot': remote['generation'],
                'source_coverage': {name: manifest.get('coverage', {'complete': False, 'pending': {'unknown': {'reason': 'legacy_snapshot'}}})
                                    for name, manifest in (('local', local), ('remote', remote))},
                'candidates': rows, 'automatic_resolution': False}

    def inventory(self, incremental=False):
        # Both CLI inventory and timer ticks use the indexed page boundary.
        # The old JSON catalogs remain untouched as historical custody.
        result = {}
        for node in ('local', 'remote'):
            scopes = [(node, {})]
            for label, spec in self.config[node].get('durable_roots', {}).items():
                scopes.append((node + '/' + label, dict(root=spec['path'],
                    policy={'source_roots': [], 'prune_roots': spec.get('prune_roots', [])})))
            for key, scope in scopes:
                page = self.rpc('inventory_page', remote=node == 'remote', **scope)
                result[key] = dict(page, rpc_scope=scope)
                store.write_json(self.state / ('inventory-page-' + key.replace('/', '-') + '.json'), page)
        return result

    @staticmethod
    def capture_batches(items):
        """Bound metadata groups; large originals retain a single-file attempt."""
        batch, size = [], 0
        for item in items:
            item_size = item.get('record', {}).get('size', 0)
            if batch and (item.get('retry_individually') or len(batch) == 8 or size + item_size > 8 * 1024**2):
                yield batch
                batch, size = [], 0
            if item.get('retry_individually'):
                yield [item]
                continue
            batch.append(item)
            size += item_size
        if batch:
            yield batch

    def capture_batch(self, items, page_id, scope, from_remote, deadline):
        outcomes = [dict(version=item['version'], status='deferred',
                         reason='bounded backfill window ended', attempted=False) for item in items]
        if time.monotonic() >= deadline:
            return 0, outcomes
        self.healthy()
        captured, moved = [], 0
        errors = (OSError, ValueError, RuntimeError, BulkWindowEnded, subprocess.TimeoutExpired)
        for item, outcome in zip(items, outcomes):
            if time.monotonic() >= deadline:
                break
            outcome['attempted'] = True
            try:
                record = self.rpc('inventory_capture', remote=from_remote,
                                  page_id=page_id, version=item['version'], **scope)
                captured.append((outcome, record))
            except errors as error:
                outcome['reason'] = str(error)[:1024]
        if captured and time.monotonic() < deadline:
            records = [record for _, record in captured]
            try:
                # One capacity reservation and rate-limited transfer serves the
                # group. Individual capture failures retain their own reason.
                moved = self.transfer(records, from_remote=from_remote, deadline=deadline)
                present = []
                expected = {record['sha256'] for record in records}
                for remote in (False, True):
                    response = self.rpc('has', remote=remote, records=records)
                    if (not isinstance(response, list) or len(response) > len(records)
                            or any(not isinstance(value, str) or value not in expected for value in response)):
                        raise ValueError('Invalid batched original object verification')
                    present.append(set(response))
                for outcome, record in captured:
                    if all(record['sha256'] in values for values in present):
                        outcome.update(status='verified_on_both_hosts', record=record)
                        outcome.pop('reason')
                    else:
                        outcome['reason'] = 'Original captured object is not verified on both hosts'
            except errors as error:
                for outcome, _ in captured:
                    outcome['reason'] = str(error)[:1024]
        # A failed group health observation prevents all page acknowledgment;
        # the original page and any sealed/partially transferred bytes survive.
        self.healthy()
        return moved, outcomes

    def durable(self, inventories, deadline):
        moved, verified, deferred = 0, 0, 0
        attempted_deferred, unattempted_deferred = 0, 0
        summaries, receipts = {}, []
        names = list(inventories)
        cursor_path = self.state / 'backfill-scope-cursor.json'
        cursor = json.loads(cursor_path.read_text()).get('next') if cursor_path.exists() else None
        if cursor in names:
            offset = names.index(cursor)
            names = names[offset:] + names[:offset]
        if names:
            # Rotation is scheduling state only: it never acknowledges a page.
            # Persist before capture so one slow origin cannot starve all others.
            store.write_json(cursor_path, {'schema': 'workspace.backfill-cursor.v1', 'next': names[1 % len(names)]})
        for node in names:
            envelope = inventories[node]
            page = envelope['page']
            page_id = envelope['page_id']
            if store.identity(page) != page_id:
                raise ValueError('Inventory page identity differs')
            scope = envelope['rpc_scope']
            from_remote = node.split('/')[0] == 'remote'
            outcomes = []
            for batch in self.capture_batches(page['items']):
                batch_moved, batch_outcomes = self.capture_batch(batch, page_id, scope, from_remote, deadline)
                moved += batch_moved
                verified += sum(item['status'] == 'verified_on_both_hosts' for item in batch_outcomes)
                deferred += sum(item['status'] == 'deferred' for item in batch_outcomes)
                attempted_deferred += sum(item['status'] == 'deferred' and item['attempted'] for item in batch_outcomes)
                unattempted_deferred += sum(item['status'] == 'deferred' and not item['attempted'] for item in batch_outcomes)
                outcomes.extend(batch_outcomes)
            if outcomes:
                receipt = dict(schema='workspace.inventory-page-custody.v1', page_id=page_id,
                               configuration_sha256=page['configuration_sha256'], origin=node,
                               outcomes=outcomes, application_consistency='unqualified_byte_snapshot',
                               resumable=False, activation_allowed=False)
                # Receipt bytes use the same object-transfer limiter and lock as
                # payload bytes. A lost acknowledgment leaves the original page
                # and captured objects reusable without resealing changing files.
                path = self.state / 'receipts' / ('page-' + store.identity(receipt) + '.json')
                if not path.exists():
                    store.write_json(path, receipt)
                elif json.loads(path.read_text()) != receipt:
                    raise ValueError('Page custody receipt collision')
                record = store.seal(self.state, path.relative_to(self.state).as_posix(), self.state)
                moved += self.transfer([record])
                summary = self.rpc('inventory_ack', remote=from_remote, page_id=page_id,
                                   receipt_sha256=record['sha256'], **scope)
                receipts.append(dict(origin=node, page_id=page_id, receipt_sha256=record['sha256']))
            else:
                summary = envelope['summary']
            summaries[node] = summary
        return dict(files_remaining=None, bytes_remaining=None,
                    backlog_coverage='unknown; page counts are not a project denominator',
                    verified_versions_this_pass=verified, deferred_versions_this_pass=deferred,
                    attempted_deferred_versions_this_pass=attempted_deferred,
                    unattempted_deferred_versions_this_pass=unattempted_deferred,
                    transferred_bytes=moved, page_receipts=receipts, indexed_status=summaries,
                    legacy_custody='existing scans and durable-index.json retained unchanged; not imported',
                    application_consistency='pending application receipts; byte snapshots are not resumable checkpoints',
                    excluded_git_requires_custody=True)

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
            receipt['source_coverage'] = plan['source_coverage']
            # Custody both source snapshots on both hosts, including remote edits.
            for manifest, origin_remote in ((remote, True), (local, False)):
                self.transfer(list(manifest['files'].values()), from_remote=origin_remote)
                self.rpc('materialize', manifest=manifest)
                self.rpc('materialize', remote=True, manifest=manifest)
            pointer = self.state / 'canonical.json'
            if publish:
                if not all(row['complete'] for row in plan['source_coverage'].values()):
                    raise ValueError('Canonical publication requires complete source coverage on both hosts')
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
            receipt['inventory_coverage'] = {k: dict(v['summary'], complete=False)
                                             for k, v in inventories.items()}
            after = self.healthy()
            store.write_json(receipt_dir / 'after.json', after)
            receipt['preservation'] = compare(self.baseline, after)
            receipt['health'] = compare_health(self.baseline, after)
            receipt['runtime_version'] = {k: {'exe': v['exe'], 'sha256': v['exe_sha256'],
                                             'invocation': v['properties']['InvocationID']}
                                          for k, v in after['stable'].items() if k.startswith('jope-parallel-') and 'exe' in v}
            receipt['status'] = 'source_verified_backfill_pending' if canonical else 'snapshots_verified_publication_pending'
            if not all(row['complete'] for row in plan['source_coverage'].values()):
                receipt['status'] = 'observed_sources_preserved_coverage_incomplete'
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
                                              for k, v in after['stable'].items() if k.startswith('jope-parallel-') and 'exe' in v}
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
            result = {k: v['summary'] for k, v in inventories.items()}
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
                # The legacy JSON denominator can be hundreds of megabytes.
                # Retain it without an unbounded read/RPC or a page-sized claim
                # that all historical durable objects have been reverified.
                result['durable_verification'] = dict(status='unavailable',
                    reason='complete legacy and indexed custody boundary has not been reconciled',
                    page_receipts='consult bounded mirror status', synchronized=False)
        else:
            if args.bulk_seconds < 0:
                raise ValueError('Bulk duration must be nonnegative')
            # Publication belongs to sync after both-host health, observed
            # coverage and custody checks. The CLI must not pre-publish it.
            result = mirror.sync(args.publish, args.bulk_seconds)
        print(json.dumps(result, indent=2))
        if result.get('status') == 'paused':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
