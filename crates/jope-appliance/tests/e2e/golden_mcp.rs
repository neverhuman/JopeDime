//! Native AG1 acceptance helper for the existing INT-01 e2e suite.
//!
//! Invoke `run` from the suite's existing real-surface test after importing this module.
//! Real MCP/daemon processes, immutable executable/input identities, TRAIN-only rows,
//! request retries, bounded status/event reads, two-language export, parity and prediction.
//! Unavailable parity is a failing acceptance result, never a skipped or passing test.

use std::{fs, io, path::Path, process::Command};

pub(super) fn run(
    executable: &Path,
    source: &str,
    train_csv: &Path,
    train_sha256: &str,
    evidence: &Path,
    runtime: &Path,
) -> io::Result<()> {
    let parent = runtime
        .parent()
        .ok_or_else(|| io::Error::other("AG1 runtime needs a parent"))?;
    fs::create_dir_all(parent)?;
    let probe = runtime.with_extension("probe.py");
    fs::write(&probe, PROBE)?;
    let output = Command::new("nice")
        .args(["-n", "10", "python3"])
        .arg(&probe)
        .arg("--binary")
        .arg(executable)
        .arg("--source")
        .arg(source)
        .arg("--csv")
        .arg(train_csv)
        .arg("--csv-sha256")
        .arg(train_sha256)
        .arg("--out")
        .arg(evidence)
        .arg("--home")
        .arg(runtime)
        .env("JOPE_INT01_QUALIFIER_TEST", std::env::current_exe()?)
        .output()?;
    fs::create_dir_all(evidence)?;
    fs::write(evidence.join("probe.stdout"), &output.stdout)?;
    fs::write(evidence.join("probe.stderr"), &output.stderr)?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "AG1 acceptance failed ({}); receipts {}\n{}\n{}",
            output.status,
            evidence.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )));
    }
    Ok(())
}

const PROBE: &str = r####"#!/usr/bin/env python3
"""Bounded AG1 proof over real stdio MCP and an owned native daemon."""
import argparse
import csv
import datetime
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import time
import traceback
import urllib.parse

p = argparse.ArgumentParser()
b = p.add_mutually_exclusive_group(required=True)
b.add_argument('--pin-manifest')
b.add_argument('--binary')
p.add_argument('--source', required=True)
p.add_argument('--csv', required=True)
p.add_argument('--csv-sha256', required=True)
p.add_argument('--out', required=True)
p.add_argument('--home', required=True)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
home = Path(a.home); home.mkdir(parents=True, mode=0o700, exist_ok=False)
assert '/tmp/' not in str(home.resolve())
host_cpus = int(os.environ.get('JOPE_INT01_HOST_CPUS', os.cpu_count() or 1))
assert 0 < host_cpus <= (os.cpu_count() or 1)
assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
assert os.getloadavg()[0] < .85 * host_cpus
assert shutil.disk_usage(home).free >= int(os.environ.get('JOPE_INT01_MIN_FREE_BYTES', 512 * 2**20))
if hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
pin = json.loads(Path(a.pin_manifest).read_text()) if a.pin_manifest else {'source': a.source, 'pin': a.binary}
assert pin['source'] == a.source
binary_file = open(pin['pin'], 'rb')
binary_hash = hashlib.sha256()
for chunk in iter(lambda: binary_file.read(1024 * 1024), b''):
    binary_hash.update(chunk)
actual_binary_sha256 = binary_hash.hexdigest()
if 'sha256' in pin:
    assert actual_binary_sha256 == pin['sha256']
pin['sha256'] = actual_binary_sha256
binary = f'/proc/{os.getpid()}/fd/{binary_file.fileno()}' if Path('/proc/self/fd').is_dir() else pin['pin']
def binary_signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
binary_identity = binary_signature(os.fstat(binary_file.fileno()))
def check_binary():
    assert binary_signature(os.fstat(binary_file.fileno())) == binary_identity, 'opened binary changed'
    assert binary_signature(os.stat(pin['pin'])) == binary_identity, 'binary path changed'
check_binary()
raw = Path(a.csv).read_bytes()
assert hashlib.sha256(raw).hexdigest() == a.csv_sha256
table = list(csv.reader(io.StringIO(raw.decode())))
assert len(table) - 1 >= 64 and 'target' in table[0]
train_rows = len(table) - 1
env = {**os.environ, 'JOPEDIME_HOME': str(home), 'CUDA_VISIBLE_DEVICES': '',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '4',
       'JOPEDIME_DEFAULTS_WORKERS': '1'}
env.pop('JOPEDIME_ENDPOINT', None)
for key in ['JOPE_STORE_FAILPOINT', 'JOPE_STORE_FAILPOINT_OCCURRENCE',
            'JOPE_STORE_FAILPOINT_PARK', 'JOPE_STORE_TRACE_BOUNDARIES']:
    env.pop(key, None)
report = {'source': a.source, 'binary_sha256': pin['sha256'], 'binary_original': pin['pin'],
          'binary_binding': 'held read-only inode, no binary/source/model copy',
          'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'run_languages': ['rust', 'python'], 'export_languages': ['python', 'rust'],
          'input_sha256': a.csv_sha256, 'input_rows': train_rows, 'input_scope': 'approved TRAIN only',
          'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'nice': os.getpriority(os.PRIO_PROCESS, 0), 'host_cpus': host_cpus,
          'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
          'host_capacity_source': 'JOPE_INT01_HOST_CPUS launcher pin or online logical CPUs', 'load_before': os.getloadavg()[0],
          'checks': {}, 'full_AG1_qualified': False}
frames = []; children = []; client = None; daemon = None

def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')

def check(name, condition):
    report['checks'][name] = bool(condition)

def command(argv, timeout=20):
    check_binary()
    result = subprocess.run([binary, *argv], env=env, capture_output=True,
                            text=True, timeout=timeout)
    assert result.returncode == 0, (argv[0], result.returncode, result.stderr)
    return json.loads(result.stdout)

class Rpc:
    def __init__(self):
        check_binary()
        with (out / f'mcp-{len(children)}.stderr').open('wb') as log:
            self.p = subprocess.Popen([binary, 'mcp'], env=env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=log, bufsize=0)
        children.append(self.p); self.next = 0; self.buffer = b''
        self.call('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                  'clientInfo': {'name': 'int01-obj02-ag1', 'version': '1'}})
        self.send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})

    def send(self, value):
        self.p.stdin.write((json.dumps(value) + '\n').encode()); self.p.stdin.flush()

    def call(self, method, params):
        check_binary()
        self.next += 1; request = {'jsonrpc': '2.0', 'id': self.next,
                                  'method': method, 'params': params}
        frames.append({'direction': 'request', 'message': request}); self.send(request)
        deadline = time.monotonic() + 40
        while True:
            while b'\n' not in self.buffer:
                wait = max(0, deadline - time.monotonic())
                ready, _, _ = select.select([self.p.stdout], [], [], wait)
                assert ready, f'MCP timeout: {method}'
                chunk = os.read(self.p.stdout.fileno(), 65536)
                assert chunk, f'MCP EOF: {method}'
                self.buffer += chunk
            line, self.buffer = self.buffer.split(b'\n', 1)
            value = json.loads(line)
            frames.append({'direction': 'response', 'message': value})
            if value.get('id') == self.next:
                assert 'error' not in value, value
                return value['result']

    def tool(self, name, args):
        value = self.call('tools/call', {'name': name, 'arguments': args})
        assert not value.get('isError'), (name, value)
        if 'structuredContent' in value:
            return value['structuredContent']
        return json.loads(''.join(x['text'] for x in value['content'] if x['type'] == 'text'))

    def close(self):
        self.p.stdin.close()
        try:
            assert self.p.wait(timeout=5) == 0
        finally:
            self.p.stdout.close()

def start_owned_daemon():
    global daemon
    with (out / 'engine.log').open('wb') as log:
        daemon = subprocess.Popen([binary, 'serve', '--idle-exit', '120s'],
                                  env=env, stdout=log, stderr=log)
    children.append(daemon)
    deadline = time.monotonic() + 20
    while True:
        assert daemon.poll() is None, 'owned native daemon exited before health'
        assert time.monotonic() < deadline, 'native daemon health deadline'
        connection = None
        try:
            descriptor = json.loads((home / 'daemon.json').read_text())
            if descriptor['pid'] != daemon.pid:
                time.sleep(.02); continue
            url = urllib.parse.urlsplit(descriptor['url']); assert url.hostname == '127.0.0.1'
            connection = http.client.HTTPConnection(url.hostname, url.port, timeout=.2)
            connection.request('GET', '/v1/system', headers={
                'Authorization': 'Bearer ' + descriptor['token']})
            response = connection.getresponse(); response.read()
            if response.status == 200:
                break
        except (OSError, ValueError, KeyError, http.client.HTTPException):
            pass
        finally:
            if connection is not None:
                connection.close()
        time.sleep(.02)
    return daemon

try:
    version = command(['--version', '--json'])
    assert version['source_sha40'] == a.source, version
    save('version.json', version)
    start_owned_daemon()
    client = Rpc()
    save('system-info.json', client.tool('system_info', {}))
    check('system_info', True)
    registration = {'path': a.csv, 'name': 'int01-original-train',
                    'idempotency_key': 'int01-obj02-ag1-register'}
    profile = client.tool('dataset_register', registration)
    check('dataset_register_replay', client.tool('dataset_register', registration) == profile)
    save('dataset.json', profile); dataset = profile['dataset_id']
    spec = {'dataset': dataset, 'target': 'target', 'task': 'regression', 'ignore_columns': [],
            'test_dataset': None, 'split': {'kind': 'random', 'holdout_fraction': .2,
            'group_column': None, 'time_column': None}, 'objective': 'rmse', 'effort': 'fast',
            'budget': {'evaluations': '1', 'wall_seconds': '120'}, 'landmarks': 'none',
            'guidance': 'auto', 'languages': ['rust', 'python'], 'require_all_languages': True,
            'device': 'cpu', 'seed': '73', 'dry_run': True, 'plan_digest': None,
            'idempotency_key': 'int01-obj02-ag1-plan'}
    plan = client.tool('run_start', spec); save('plan.json', plan)
    check('dry_run_plan_replay', client.tool('run_start', spec) == plan)
    real = {**spec, 'dry_run': False, 'plan_digest': plan['plan_digest'],
            'idempotency_key': 'int01-obj02-ag1-start'}
    initial_inventory = client.tool('run_status', {})
    check('initial_run_inventory_empty', initial_inventory['next_after'] is None
          and initial_inventory['runs'] == [])
    assert report['checks']['initial_run_inventory_empty']
    started = client.tool('run_start', real); rid = started['run_id']
    created_inventory = client.tool('run_status', {})
    check('one_run_between_create_and_retry', created_inventory['next_after'] is None
          and [r['run_id'] for r in created_inventory['runs']] == [rid])
    check('run_start_replay', client.tool('run_start', real) == started)
    replay_inventory = client.tool('run_status', {})
    check('one_run_after_create_retry', replay_inventory['next_after'] is None
          and [r['run_id'] for r in replay_inventory['runs']] == [rid])
    save('run-retry-inventories.json', [initial_inventory, created_inventory, replay_inventory])
    client.close(); client = Rpc()
    check('run_start_replay_after_mcp_reconnect', client.tool('run_start', real) == started)
    events = []; cursor = 0; deadline = time.monotonic() + 140
    while True:
        snapshot = client.tool('run_status', {'run_id': rid, 'wait_seconds': 3,
                                              'after_event': str(cursor)})
        watermark = int(snapshot['event_watermark'])
        while cursor < watermark:
            page = client.tool('run_events', {'run_id': rid, 'after': str(cursor), 'limit': 2})
            assert int(page['next_after']) > cursor
            cursor = int(page['next_after']); events.extend(page['events'])
        if snapshot['state'] in ['completed', 'failed', 'cancelled']:
            break
        assert time.monotonic() < deadline, 'bounded run did not finish'
    save('snapshot.json', snapshot); save('events.json', events)
    check('completed_with_real_champion', snapshot['state'] == 'completed' and snapshot.get('best') is not None)
    check('event_sequences', [int(e['sequence']) for e in events] == list(range(1, cursor + 1)))
    check('event_run_identity', all(e['run_id'] == rid for e in events))
    check('zero_resource_limits', events[-1]['body']['nondeterminism']['count'] == '0')
    check('one_run_after_retry', [r['run_id'] for r in command(['runs', 'list'])['runs']] == [rid])
    # Test-only host qualification: real engine/Python/Rust outputs, then the
    # existing trusted registrar. Production export never launches a compiler.
    client.close(); client = None
    stopped = subprocess.run([binary, 'serve', '--stop'], env=env, capture_output=True, timeout=10)
    assert stopped.returncode == 0, stopped.stderr
    assert daemon.wait(timeout=10) == 0
    qenv = os.environ.copy()
    qenv.update(JOPE_INT01_QUALIFY_HOME=str(home), JOPE_INT01_QUALIFY_RUN=rid,
                JOPE_INT01_QUALIFY_CSV=a.csv, JOPE_INT01_QUALIFY_SHA256=a.csv_sha256,
                JOPE_INT01_QUALIFY_OUT=str(out / 'measured64'))
    q = subprocess.run([os.environ['JOPE_INT01_QUALIFIER_TEST'],
                        '--exact', 'all_product_surfaces_share_the_bounded_run_projection',
                        '--nocapture', '--test-threads=1'], env=qenv, capture_output=True, timeout=90)
    (out / 'qualification.stdout').write_bytes(q.stdout)
    (out / 'qualification.stderr').write_bytes(q.stderr)
    assert q.returncode == 0 and b'1 passed' in q.stdout, (q.returncode, q.stderr.decode(errors='replace'))
    start_owned_daemon(); client = Rpc()
    manifest = client.tool('model_export', {'run_id': rid, 'kind': 'bundle',
        'languages': ['python', 'rust'], 'out_dir': str(home / 'export'),
        'idempotency_key': 'int01-obj02-ag1-export'})
    save('export-manifest.json', manifest)
    check('fit_holdout_export', manifest['refit'] == 'fit_holdout')
    check('both_languages_exported', {x.get('language') for x in manifest['languages']
                                      if x['status'] == 'exported'} == {'python', 'rust'})
    check('measured_parity_receipt', manifest['parity_status'] == 'passed' and manifest.get('parity_receipt') is not None)
    export_files = {str(f.relative_to(home / 'export')): {'bytes': f.stat().st_size,
                    'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
                    for f in (home / 'export').rglob('*') if f.is_file()}
    save('export-files.json', export_files)
    check('exported_sources_exist', 'manifest.json' in export_files and all(
        language['path'] in export_files for language in manifest['languages']
        if language['status'] == 'exported'))
    check('model_payload_exists', 'model.jope' in export_files or 'model/manifest.json' in export_files)
    check('parity_receipt_file_exists', 'parity_receipt.json' in export_files)
    check('usage_readme_exists', 'README.md' in export_files or all(
        'src/' + language['language'] + '/README.md' in export_files
        for language in manifest['languages'] if language['status'] == 'exported'))
    measured = json.loads((out / 'measured64' / 'observations.json').read_text())
    check('downloaded_sources_match_measured_bytes', all(
        export_files[language['path']]['sha256'] == measured['source_sha256'][language['language']]
        for language in manifest['languages'] if language['status'] == 'exported'))
    for language in ['python', 'rust']:
        directory = home / 'export' / 'src' / language
        readme = (directory / 'README.md').read_text()
        if language == 'python':
            assert 'python3 predictor.py --self-check' in readme
            command_line = ['python3', '-B', 'predictor.py', '--self-check']
        else:
            assert 'rustc --edition=2021 --test predictor.rs' in readme
            command_line = ['rustc', '--edition=2021', '--test', 'predictor.rs', '-o', str(out / 'downloaded-rust-self-check')]
        checked = subprocess.run(command_line, cwd=directory, env=env, capture_output=True, timeout=60)
        (out / (language + '-readme.stdout')).write_bytes(checked.stdout)
        (out / (language + '-readme.stderr')).write_bytes(checked.stderr)
        assert checked.returncode == 0, checked.stderr
        if language == 'rust':
            checked = subprocess.run([str(out / 'downloaded-rust-self-check'), '--test-threads=1'], env=env, capture_output=True, timeout=20)
            (out / 'rust-readme-execute.stdout').write_bytes(checked.stdout)
            (out / 'rust-readme-execute.stderr').write_bytes(checked.stderr)
            assert checked.returncode == 0 and b'1 passed' in checked.stdout
        check(language + '_downloaded_readme_self_check', True)
    repeated_export = client.tool('model_export', {'run_id': rid, 'kind': 'bundle',
        'languages': ['python', 'rust'], 'out_dir': str(home / 'export'),
        'idempotency_key': 'int01-obj02-ag1-export'})
    check('export_retry', repeated_export == manifest)

    prediction_request = {'model': {'artifact_id': manifest['artifact_id']},
        'input': {'dataset_id': dataset}, 'idempotency_key': 'int01-obj02-ag1-predict'}
    predictions = client.tool('model_predict', prediction_request)
    replay = client.tool('model_predict', prediction_request)
    check('prediction_retry', all(replay[k] == predictions[k] for k in ['rows', 'sha256', 'resource']))
    resource = client.call('resources/read', {'uri': predictions['resource']})
    text = ''.join(x['text'] for x in resource['contents'])
    check('prediction_resource_hash', hashlib.sha256(text.encode()).hexdigest() == predictions['sha256'])
    predicted_rows = list(csv.reader(io.StringIO(text)))
    check('prediction_rows', int(predictions['rows']) == train_rows and len(predicted_rows) - 1 == train_rows)
    check('finite_predictions', all(math.isfinite(float(row[0])) for row in predicted_rows[1:]))
    save('predictions.json', predictions); (out / 'predictions.csv').write_text(text)
    file_request = {'model': {'run_id': rid}, 'input': {'dataset_id': dataset},
        'out_path': str(home / 'predictions.csv'),
        'idempotency_key': 'int01-obj02-ag1-predict-file'}
    file_prediction = client.tool('model_predict', file_request)
    before = (home / 'predictions.csv').stat()
    file_replay = client.tool('model_predict', file_request)
    after = (home / 'predictions.csv').stat()
    check('prediction_file_retry', all(file_replay[k] == file_prediction[k]
                                      for k in ['rows', 'sha256', 'out_path']))
    check('prediction_file_retry_preserves_inode', (before.st_dev, before.st_ino, before.st_mtime_ns)
          == (after.st_dev, after.st_ino, after.st_mtime_ns))
    check('prediction_file_matches_resource', (home / 'predictions.csv').read_bytes() == text.encode()
          and file_prediction['sha256'] == predictions['sha256']
          and file_prediction['rows'] == str(train_rows))
    save('prediction-file-retry.json', {'arguments': file_request,
         'first': file_prediction, 'replay': file_replay,
         'file_sha256': hashlib.sha256((home / 'predictions.csv').read_bytes()).hexdigest()})
    check('prediction_did_not_mutate_run', client.tool('run_status', {'run_id': rid}) == snapshot)
    check_binary()
    report['full_AG1_qualified'] = all(report['checks'].values())
except Exception as error:
    report['error'] = str(error); report['traceback'] = traceback.format_exc()
finally:
    cleanup = []
    if client is not None:
        try:
            client.close()
        except (OSError, AssertionError, subprocess.TimeoutExpired) as error:
            cleanup.append(str(error))
    if daemon is not None:
        try:
            result = subprocess.run([binary, 'serve', '--stop'], env=env, capture_output=True,
                                    text=True, timeout=10)
            if result.returncode:
                cleanup.append('own serve-stop exit ' + str(result.returncode))
        except subprocess.TimeoutExpired:
            cleanup.append('own serve-stop timeout')
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(timeout=5)
        for stream in [child.stdin, child.stdout, child.stderr]:
            if stream is not None:
                stream.close()
    report['cleanup_errors'] = cleanup
    report['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['load_after'] = os.getloadavg()[0]
    save('frames.json', frames); save('report.json', report)
    binary_file.close()
print(json.dumps(report))
raise SystemExit(0 if report['full_AG1_qualified'] and not report['cleanup_errors'] else 1)
"####;
