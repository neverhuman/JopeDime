#!/usr/bin/env python3
"""Measure three explicit synthetic warm demos using the installed private CLI.

Run only when the host's exclusive numerical allocation is available. The CLI
starts its existing shared supervisor; this script never changes runner services.
No dataset downloads, sealed inputs, GPU tasks, or policy training occur here.
"""
import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_private(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_demo(folder, task, source_sha, binary_sha256):
    invocation = json.loads((folder / 'invocation-receipt.json').read_text())
    result = json.loads((folder / 'demo-result.json').read_text())
    plan = json.loads((folder / 'demo-plan.json').read_text())
    if sha256(folder / 'demo-plan.json') != invocation['plan_file_sha256']:
        raise ValueError('demo plan bytes changed')
    if sha256(folder / 'demo-result.json') != invocation['result_file_sha256'] or invocation['result'] != result:
        raise ValueError('demo result bytes changed')
    if invocation['program']['sha256'] != binary_sha256 or invocation['source_sha'] != source_sha or result['source_sha'] != source_sha:
        raise ValueError('demo executable or source identity differs')
    for name in ['predictions.json', 'baseline.json', 'execution.json']:
        if sha256(folder / name) != result['artifact_sha256'][name]:
            raise ValueError('demo artifact bytes changed: ' + name)
    execution = json.loads((folder / 'execution.json').read_text())
    if execution['command_status'] != 'completed' or execution['exit_code'] != 0:
        raise ValueError('demo worker did not complete')
    if len(execution['outputs']) != 1:
        raise ValueError('unexpected demo worker inventory')
    artifact = execution['outputs'][0]
    worker_path = Path(artifact['path'])
    if not worker_path.is_relative_to(folder / 'journal/jobs') or sha256(worker_path) != artifact['sha256']:
        raise ValueError('worker output bytes or location changed')
    worker = json.loads(worker_path.read_text())
    if worker['predictions'] != json.loads((folder / 'predictions.json').read_text()) or worker['artifact_sha256'] != result['fitted']['sha256']:
        raise ValueError('fitted or prediction output differs from worker')
    if invocation['schema'] != 'jope.warm-demo-invocation/v1' or not invocation['ok']:
        raise ValueError('demo invocation did not complete')
    if result['schema'] != 'jope.warm-demo/v1' or not result['ok']:
        raise ValueError('demo fit did not complete')
    if result['task'] != task or result['input'] != {'kind': 'synthetic', 'task': task, 'version': 'jope.synthetic-demo/v1'}:
        raise ValueError('demo did not use the declared synthetic input')
    if plan['experiment']['source_sha'] != source_sha or result['plan_sha256'] != invocation['plan_sha256']:
        raise ValueError('source or plan identity differs')
    if result['worker_fit_attempts'] != 1 or result['fitted_artifacts'] != 1 or result['neural_consumed']:
        raise ValueError('unexpected fitted or neural accounting')
    if result['sealed_inputs_opened'] or not result['fitted']['reload_verified']:
        raise ValueError('missing reload proof or inappropriate data access')
    fitted = folder / 'fitted.json'
    if Path(result['fitted']['path']) != fitted or sha256(fitted) != result['fitted']['sha256']:
        raise ValueError('retained fitted artifact digest differs')
    for name in ['model_score', 'baseline_score']:
        if not math.isfinite(result[name]['loss']) or not math.isfinite(result[name]['value']):
            raise ValueError('nonfinite demo comparison')
    seconds = invocation['warm_elapsed_seconds']
    if not math.isfinite(seconds) or seconds <= 0 or invocation['under_60_seconds'] != (seconds < 60):
        raise ValueError('invalid measured warm elapsed time')
    predictions = json.loads((folder / 'predictions.json').read_text())
    if task == 'regression':
        if predictions['kind'] != 'values' or len(predictions['data']) != 15:
            raise ValueError('missing regression predictions')
    elif predictions['kind'] != 'probabilities' or len(predictions['data']['values']) != 15 or predictions['data']['classes'] != result['classes']:
        raise ValueError('missing declared-order probability matrix')
    # Independent scalar scoring of the explicit fixture; no model or fit runs.
    baseline = json.loads((folder / 'baseline.json').read_text())
    model_score = fixture_score(task, result['classes'], predictions)
    baseline_score = fixture_score(task, result['classes'], baseline['predictions'])
    for expected, actual in [(model_score, result['model_score']), (baseline_score, result['baseline_score']), (baseline_score, baseline['score'])]:
        if expected['metric'] != actual['metric'] or any(not math.isclose(expected[key], actual[key], rel_tol=1e-12, abs_tol=1e-12) for key in ['value', 'loss']):
            raise ValueError('retained score differs from independently scored predictions')
    if task == 'regression':
        mean = sum(2.0 * (i / 8.0 - 4.0) + 1.0 for i in range(60)) / 60
        if baseline['state'] != {'kind': 'train_mean', 'value': mean} or baseline['predictions'] != {'kind':'values','data':[mean] * 15}:
            raise ValueError('baseline did not use only fitting labels')
    else:
        count = 2 if task == 'binary' else 3
        expected = {'kind':'probabilities','data':{'classes':result['classes'],'values':[[1.0/count] * count for _ in range(15)]}}
        if baseline['predictions'] != expected:
            raise ValueError('baseline did not use fitting class priors')
    return {'task': task, 'ok': True, 'warm_elapsed_seconds': seconds,
            'under_60_seconds': seconds < 60, 'result': result,
            'receipts': {name: sha256(folder / name) for name in ['invocation-receipt.json', 'demo-result.json', 'fitted.json', 'predictions.json', 'baseline.json', 'execution.json']}}


def fixture_score(task, classes, predictions):
    if task == 'regression':
        if classes != [] or predictions['kind'] != 'values':
            raise ValueError('incorrect regression contract')
        values = predictions['data']
        if len(values) != 15 or any(not math.isfinite(x) for x in values):
            raise ValueError('invalid regression values')
        value = math.sqrt(sum((2.0 * (i / 8.0 - 4.0) + 1.0 - p) ** 2 for i, p in zip(range(60, 75), values)) / 15)
        return {'metric':'rmse','value':value,'loss':value}
    declared = ['negative', 'positive'] if task == 'binary' else ['zebra', 'alpha', 'middle']
    if classes != declared or predictions['kind'] != 'probabilities' or predictions['data']['classes'] != classes:
        raise ValueError('incorrect frozen class ordering')
    values = predictions['data']['values']
    if len(values) != 15 or any(len(row) != len(classes) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row) or abs(sum(row)-1) > 1e-6 for row in values):
        raise ValueError('invalid class probabilities')
    labels = [i % len(classes) for i in range(60,75)]
    if task == 'binary':
        positive = [values[i][1] for i, y in enumerate(labels) if y == 1]
        negative = [values[i][1] for i, y in enumerate(labels) if y == 0]
        value = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in positive for b in negative) / (len(positive) * len(negative))
        return {'metric':'auroc','value':value,'loss':1.0-value}
    value = -sum(math.log(max(row[label],1e-15)) for row,label in zip(values,labels))/15
    return {'metric':'log_loss','value':value,'loss':value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-under-60', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.sha256) or not re.fullmatch('[0-9a-f]{40}', args.source_sha):
        parser.error('full lowercase binary SHA256 and source SHA required')
    binary = args.binary.resolve(strict=True)
    if sha256(binary) != args.sha256:
        parser.error('binary differs from pin; nothing executed')
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to('/tmp'):
        parser.error('output must be a fresh durable directory outside /tmp')
    output.mkdir(parents=True, mode=0o700)
    output = output.resolve()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('AWS_', 'GH_', 'GITHUB_')):
            env.pop(key)
    env.update({key: '1' for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']})
    env['JOPE_OFFLINE'] = '1'
    results = []
    for task in ['regression', 'binary', 'multiclass']:
        folder = output / task
        command = [str(binary), 'demo', '--synthetic', task, '--source-sha', args.source_sha, '--output', str(folder)]
        started = time.monotonic()
        error = None
        with (output / (task + '.stdout')).open('x') as stdout, (output / (task + '.stderr')).open('x') as stderr:
            os.chmod(stdout.name, 0o600)
            os.chmod(stderr.name, 0o600)
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = child.wait(timeout=70)
            except subprocess.TimeoutExpired:
                error = 'outer qualification deadline exceeded; requested supervisor cancellation'
                os.killpg(child.pid, signal.SIGINT)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
        try:
            if code != 0 or error:
                raise ValueError(error or ('demo exited ' + str(code)))
            result = verify_demo(folder, task, args.source_sha, args.sha256)
        except Exception as problem:
            result = {'task': task, 'ok': False, 'error': str(problem), 'under_60_seconds': None}
        result.update({'command': command, 'exit_code': code, 'outer_elapsed_seconds': time.monotonic() - started})
        results.append(result)
        write_private(output / (task + '-qualification.json'), result)
    ok = all(item['ok'] and (not args.require_under_60 or item['under_60_seconds']) for item in results)
    receipt = {'schema': 'jope.warm-demo-qualification/v1', 'ok': ok, 'binary': str(binary), 'binary_sha256': args.sha256,
               'source_sha': args.source_sha, 'synthetic': True, 'cold_install_included': False,
               'require_under_60': args.require_under_60, 'attempts': results}
    write_private(output / 'qualification.json', receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
