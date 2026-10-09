"""Four fixed fresh32 campaigns, one shared clock, stop on the first refusal.

This local controller launches only the reviewed child controllers. Their actual
current-role, resource, source/replay, full qualification, publication and DB CAS
gates remain mandatory. This is a new128-variant scope, not an old-budget reset.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SCHEMA = 'jope.kumos-next128-controller/v1'
SCOPE = 'next128-four-fixed32-v1'
GLOBAL = 1791592146.380285
WORKER_SHA = '0883dd4085e01cc0e1a3353396fd7004854c4ae2e68868f1eb79a053a6cb2a42'
SELECTION_SHA = '947ff1dd0269b82dfa2cf4e3a53ceb052fa4c498b327429410529012499fbd5b'
SOURCE_ROOT = '/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-kumo-xbabe3-next128-20261009/revision2'
MAX_LOG = 8 * 1024**2
# Intake reserves the last600 seconds for publication/commit. Do not claim a
# packet with less than60 seconds left in its GPU qualification portion.
MIN_START_SECONDS = 660


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(ref, cap=4 * 1024**2):
    path = Path(ref['path'])
    need(path.is_absolute() and path.is_file() and all(not p.is_symlink() for p in (path, *path.parents)), 'unsafe pinned file')
    need(type(ref['bytes']) is int and 0 < ref['bytes'] <= cap and path.stat().st_size == ref['bytes'], 'pinned byte size/cap')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    need(len(raw) == ref['bytes'] and sha(raw) == ref['sha256'], 'pinned bytes/hash drift')
    return raw


def file_ref(path, cap=MAX_LOG):
    path = Path(path).resolve()
    need(path.is_file() and path.stat().st_size <= cap, 'receipt size bound')
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha(raw)}


def save(path, value):
    raw = canonical(value)
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def remaining(config, wall=None, monotonic=None):
    wall = time.time() if wall is None else wall
    monotonic = time.monotonic() if monotonic is None else monotonic
    return min(config['deadline_unix'] - wall, config['shared_monotonic_deadline'] - monotonic)


def validate_template(config):
    need(config['schema'] == SCHEMA and config['approved_for_launch'] is False, 'unapproved exact template required')
    need(config['shared_scope'] == SCOPE and config['host'] == 'xbabe3' and config['max_packets'] == 4 and config['max_new_variants'] == 128, 'fixed four32 scope')
    need(config['no_retries'] is True and config['old_scopes_unchanged'] is True and config['global_deadline_unix'] == GLOBAL and config['source_root'] == SOURCE_ROOT, 'scope/global/source-root drift')
    need(all(config[k] is None for k in ('created_unix', 'deadline_unix', 'shared_monotonic_deadline')), 'template must not carry a clock')
    need(config['worker_sha256'] == WORKER_SHA and config['selection']['sha256'] == SELECTION_SHA, 'original numerical/snapshot identity')
    need(Path(config['orchestrator']['path']).resolve() == Path(__file__).resolve(), 'controller path')
    for ref in config['source_pins']:
        read(ref)
    need(config['orchestrator'] in config['source_pins'] and config['freezer'] in config['source_pins'], 'controller/freezer custody')
    read(config['orchestrator'])
    snapshot = json.loads(read(config['selection']))
    need(snapshot['state'] == 'metadata_ready_current_roles_pending' and len(snapshot['rows']) == 128 and len(snapshot['publications']) == 128, 'exact metadata inventory')
    need([r['sequence'] for r in snapshot['rows']] == list(range(938, 1066)), 'fresh fixed sequences')
    need(len({r['identity'] for r in snapshot['rows']}) == 128, 'duplicate selected identity')
    need(len(config['packets']) == 4 and [p['ordinal'] for p in config['packets']] == list(range(4)), 'exact packet order')
    packet_ids = set()
    for ordinal, descriptor in enumerate(config['packets']):
        child = json.loads(read(descriptor['template']))
        support = snapshot['rows'][ordinal * 32:(ordinal + 1) * 32]
        digest = sha(canonical([{k: r[k] for k in ('sequence', 'identity', 'parent')} for r in support]))
        expected_id = sha(canonical({'schema': SCHEMA, 'selection_snapshot_sha256': SELECTION_SHA, 'ordinal': ordinal, 'support_sha256': digest}))
        need(descriptor['packet_id'] == child['packet_id'] == expected_id and expected_id not in packet_ids, 'fixed distinct packet identity')
        packet_ids.add(expected_id)
        need(descriptor['sequence_first'] == 938 + 32 * ordinal and descriptor['sequence_last'] == 969 + 32 * ordinal, 'exact packet bounds')
        need(child['approved_for_launch'] is False and child['max_packets'] == 1 and child['packet_size'] == 32 and child['host'] == 'xbabe3', 'one child packet only')
        need(child['shared_scope'] == SCOPE and child['shared_monotonic_deadline'] is None, 'child shared clock contract')
        need(child['worker']['sha256'] == descriptor['worker_sha256'] == WORKER_SHA, 'scientific worker changed')
        need(child['owned_runtime'] == SOURCE_ROOT + '/packet' + str(ordinal) + '/runtime', 'child source root')
        need(child['controller_sha256'] == descriptor['controller']['sha256'], 'child controller identity')
        read(descriptor['controller'])
        names = set()
        for ref in child['local_source_pins']:
            read(ref)
            need(Path(ref['path']).name not in names, 'duplicate child source')
            names.add(Path(ref['path']).name)
        selections = [ref for ref in child['local_source_pins'] if Path(ref['path']).name == 'selection.json']
        need(len(selections) == 1, 'unique child selection')
        selected = json.loads(read(selections[0]))
        need(selected['rows'] == support and selected['selection_sha256'] == digest and selected['packet_id'] == expected_id, 'child support/row identity')
        need(len({r['parent'] for r in support}) == 32, 'distinct parent lineages within each packet')
    return snapshot


def validate_config(config):
    template = json.loads(read(config['template']))
    snapshot = validate_template(template)
    need(config['approved_for_launch'] is True and config['clock_host'] == socket.gethostname(), 'approved config/same local clock host')
    review = json.loads(read(config['review']))
    need(review.get('state') == 'passed' and review.get('template_sha256') == config['template']['sha256'], 'pinned review does not cover template')
    for key in ('created_unix', 'deadline_unix', 'shared_monotonic_deadline'):
        need(type(config[key]) in (int, float) and math.isfinite(config[key]), 'finite shared clocks')
    need(config['deadline_unix'] == config['created_unix'] + 1800 <= GLOBAL and config['created_unix'] <= time.time() and 0 < remaining(config) <= 1800, 'shared fixed wall/monotonic window expired')
    expected = copy.deepcopy(template)
    for key in ('approved_for_launch', 'created_unix', 'deadline_unix', 'shared_monotonic_deadline', 'clock_host', 'review', 'template'):
        expected[key] = config[key]
    expected['packets'] = []
    for ordinal, original in enumerate(template['packets']):
        frozen = config['packets'][ordinal]
        descriptor = {**original, 'config': frozen['config']}
        expected['packets'].append(descriptor)
        child = json.loads(read(frozen['config']))
        child_expected = json.loads(read(original['template']))
        child_expected.update(approved_for_launch=True, review=config['review'])
        for key in ('created_unix', 'deadline_unix', 'shared_monotonic_deadline'):
            child_expected[key] = config[key]
        need(child == child_expected, 'child config changed or reset its clock')
    need(config == expected, 'unreviewed frozen config fields')
    return snapshot


def terminate_owned_group(child):
    # A child may have exited while descendants retain pipes/SSH connections.
    # Always target the process group created by this controller, even then.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # Give already-exited descendants a bounded reaping interval; waiting on
    # their original parent alone returns immediately once it has exited.
    time.sleep(.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=2)


def child_process(descriptor, config, output):
    command = [sys.executable, '-B', descriptor['controller']['path'], '--config', descriptor['config']['path'], '--config-sha', descriptor['config']['sha256']]
    stdout_path, stderr_path = output / 'stdout.jsonl', output / 'stderr.log'
    save(output / 'intent.json', {'command': command, 'config': descriptor['config'], 'shared_deadline_unix': config['deadline_unix'], 'at_unix_ns': time.time_ns(), 'remaining_seconds': remaining(config)})
    child = None
    try:
        with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
            child = subprocess.Popen(command, cwd=HERE, stdout=stdout, stderr=stderr, start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            while child.poll() is None:
                need(remaining(config) > 0, 'shared deadline expired during child')
                need(stdout_path.stat().st_size <= MAX_LOG and stderr_path.stat().st_size <= MAX_LOG, 'bounded child output exceeded')
                time.sleep(min(.1, max(.001, remaining(config))))
            need(remaining(config) > 0, 'late child completion cannot count')
            need(child.returncode == 0, 'child refused/failed/unknown; no retry: ' + str(child.returncode))
        need(stdout_path.stat().st_size <= MAX_LOG and stderr_path.stat().st_size <= MAX_LOG, 'bounded final child output')
        lines = stdout_path.read_bytes().splitlines()
        need(bool(lines) and len(lines[-1]) <= 2 * 1024**2, 'bounded child result missing')
        return json.loads(lines[-1])
    except BaseException:
        if child is not None:
            terminate_owned_group(child)
        raise


def verify_completion(descriptor, result, support):
    need(result['state'] == 'stage_committed' and result['packet_id'] == descriptor['packet_id'] and result['variants'] == 32 and result['stage'] == 'kumo_small_s3_verified', 'child has not committed exact32')
    need(len(result['rows']) == 32 and {r['variant_identity'] for r in result['rows']} == {r['identity'] for r in support}, 'committed support mismatch')
    for row in result['rows']:
        need(row['packet_id'] == descriptor['packet_id'] and row['required_encoders'] == ['kumo_tabular_s'] and row['new_gp_labels'] == 0 and row['teaching_eligible'] is False, 'stage roster or eligibility differs')
    folder = Path(descriptor['controller']['path']).parent / 'runs' / descriptor['packet_id']
    complete_ref = file_ref(folder / 'complete.json')
    need(json.loads(read(complete_ref)) == result, 'stdout/committed local proof differs')
    publication_ref = file_ref(folder / 'publication.json')
    publication = json.loads(read(publication_ref))
    need(publication['state'] == 'complete' and publication['host'] == 'xbabe3' and publication['completed_variants'] == publication['vectors_verified'] == 32 and publication['verified_objects'] == 64, 'exact64 publication verification required')
    need({r['variant_identity'] for r in publication['variants']} == {r['identity'] for r in support}, 'publication support differs')
    return {'ordinal': descriptor['ordinal'], 'packet_id': descriptor['packet_id'], 'config': descriptor['config'], 'state': 'stage_committed', 'variants': 32, 'verified_objects': 64, 'complete': complete_ref, 'publication': publication_ref, 'manifest': file_ref(folder / 'manifest.json')}


def sequence(config, snapshot, output, invoke=child_process, verify=verify_completion):
    results = []
    for descriptor in config['packets']:
        need(remaining(config) >= MIN_START_SECONDS, 'insufficient shared window before next claim; no clock reset')
        ordinal = descriptor['ordinal']
        child_output = output / ('packet' + str(ordinal))
        child_output.mkdir()
        result = invoke(descriptor, config, child_output)
        need(remaining(config) > 0, 'expired after child; no aggregate success')
        proof = verify(descriptor, result, snapshot['rows'][ordinal * 32:(ordinal + 1) * 32])
        save(child_output / 'verified.json', proof)
        results.append(proof)
        save(output / ('progress-' + str(ordinal) + '.json'), {'state': 'progress', 'completed_packets': len(results), 'completed_variants': 32 * len(results), 'packets': results, 'deadline_unix': config['deadline_unix']})
    return {'schema': 'jope.kumos-next128-result/v1', 'state': 'complete', 'completed_packets': 4, 'completed_variants': 128, 'verified_objects': 256, 'packets': results, 'deadline_unix': config['deadline_unix'], 'new_gp_labels': 0, 'teaching_eligible': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha', required=True)
    args = parser.parse_args()
    path = Path(args.config).resolve()
    reference = file_ref(path)
    need(reference['sha256'] == args.config_sha, 'root config hash')
    config = json.loads(read(reference))
    snapshot = validate_config(config)
    output = HERE / 'run'
    # Exclusive directory is the persistent once-only attempt marker, including
    # stage/refusal failures. No restart/resume mode exists.
    output.mkdir()
    save(output / 'window.json', {'config': reference, 'created_unix': config['created_unix'], 'deadline_unix': config['deadline_unix'], 'shared_monotonic_deadline': config['shared_monotonic_deadline'], 'pid': os.getpid(), 'host': socket.gethostname()})
    try:
        result = sequence(config, snapshot, output)
        need(remaining(config) > 0, 'aggregate completed after shared deadline')
        result['config'] = reference
        save(output / 'complete.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        save(output / 'failure.json', {'schema': 'jope.kumos-next128-result/v1', 'state': 'stopped_requires_reconciliation', 'config': reference, 'error': str(error)[:1500], 'deadline_unix': config['deadline_unix'], 'at_unix_ns': time.time_ns(), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
