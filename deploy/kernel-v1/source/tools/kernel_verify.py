#!/usr/bin/env python3
"""Execute bounded, source-identified kernel checks without a runtime cutover."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True

from kernel_bundle import verify_bundle
from kernel_release import digest
import kernel_research_check as research
import experiment_runtime as runtime
import experiment_scheduler as scheduler


class Deferred(RuntimeError):
    pass


def monitor(child, deadline, node_root=None):
    """Kill only our process group, releasing resources before mandatory work."""
    while child.poll() is None:
        reason = scheduler.admission_reason(node_root) if node_root else None
        if reason or time.monotonic() >= deadline:
            runtime.stop([child])
            if reason:
                raise Deferred(reason)
            raise TimeoutError('Verification command exceeded its wall-time limit')
        time.sleep(0.5)
    return child.returncode


def supervised(args):
    root = args.node_root.resolve()
    output = runtime.output_root(root, args.output)
    if os.environ.get('JOPE_KERNEL_VERIFY_SERVICE') == '1':
        if 'jope-kernel-verify-' not in Path('/proc/self/cgroup').read_text():
            raise ValueError('Verification service cgroup missing')
        return
    if output.exists():
        raise ValueError('Use a fresh verification output directory')
    output.mkdir(parents=True)
    lock = root / 'runs/experiment-spare-cpus.lock'
    if not lock.is_file():
        raise ValueError('Existing node-wide spare-CPU lock required')
    unit = 'jope-kernel-verify-' + uuid.uuid4().hex
    command = ['sudo', '-n', 'systemd-run', '--quiet', '--wait', '--pipe', '--collect', '--unit=' + unit,
        '-p', 'User=' + str(os.getuid()), '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=5',
        '-p', 'RuntimeMaxSec=1800', '-p', 'CPUQuota=100%', '-p', 'AllowedCPUs=0-7 62-69',
        '-p', 'MemoryMax=6G', '-p', 'MemorySwapMax=0', '-p', 'TasksMax=128',
        '-p', 'Nice=15', '-p', 'IOSchedulingClass=idle', '-p', 'PrivateDevices=yes',
        '-p', 'ProtectSystem=strict', '-p', 'ProtectHome=read-only', '-p', 'PrivateTmp=yes',
        '-p', 'NoNewPrivileges=yes', '-p', 'UMask=0077',
        '-p', 'ReadOnlyPaths=' + str(root),
        '-p', 'ReadWritePaths=' + ' '.join(map(str, (output, lock, Path.home() / '.cargo'))),
        '--setenv=JOPE_KERNEL_VERIFY_SERVICE=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
        sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


def verify(args):
    metadata = verify_bundle(args.bundle)
    output = args.output.resolve()
    source = args.bundle.resolve() / 'runtime' / metadata['release_id'] / 'source'
    probes = args.bundle.resolve() / 'workspace/research/algorithm-kernel-regimes'
    if output == args.bundle.resolve() or output.is_relative_to(args.bundle.resolve()):
        raise ValueError('Build output must be outside immutable source')
    output.mkdir(parents=True, exist_ok=args.node_root is not None)
    if (output / 'receipt.json').exists():
        raise ValueError('Verification receipt already exists')
    env = {**os.environ, 'CARGO_TARGET_DIR': str(output / 'target'), 'CARGO_BUILD_JOBS': '1',
           'RUST_TEST_THREADS': '1', 'RAYON_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '',
           'PYTHONDONTWRITEBYTECODE': '1', 'PATH': str(Path.home() / '.cargo/bin') + ':' + os.environ['PATH']}
    checks = []
    begun = time.time()
    receipt = {'schema': 'kernel.verification.v1', 'bundle_id': metadata['bundle_id'],
               'release_id': metadata['release_id'], 'profile': args.profile,
               'gate_manifest_sha256': digest(probes / 'verification-gates-v1.json'),
               'hostname': os.uname().nodename, 'started': begun, 'checks': checks,
               'qualified_for_deployment': False, 'gpu_execution': False,
               'unqualified_gates': ['full search quality/performance', 'programmable algorithms and tasks',
                                    'neural migration', 'eight-GPU recovery', '24-hour campaign', 'cutover/rollback']}

    def run(name, command, stdout=None):
        path = output / (name + '.log')
        start = time.monotonic()
        row = {'gate': name, 'command': command, 'status': 'running'}
        checks.append(row)
        with path.open('x') as log:
            stream = (output / stdout).open('x') if stdout else log
            try:
                child = runtime.popen(command, cwd=source, env=env, stdout=stream, stderr=log)
                try:
                    result = monitor(child, start + args.command_timeout, args.node_root)
                finally:
                    runtime.stop([child])
            finally:
                if stream is not log:
                    stream.close()
        row.update(exit_code=result, seconds=time.monotonic() - start, log_sha256=digest(path),
                   status='passed' if result == 0 else 'failed')
        if result:
            raise RuntimeError(f'{name} failed with exit code {result}; inspect {path}')
        print(json.dumps({'gate': name, 'status': 'passed'}), flush=True)

    try:
        receipt['rustc'] = subprocess.check_output(['rustc', '--version', '--verbose'], env=env, text=True)
        lease = scheduler.spare_lease(args.node_root, metadata['bundle_id'], cpus=1, gpu=False) if args.node_root else nullcontext()
        with lease as allocation:
            if allocation:
                receipt['allocation'] = {k: v for k, v in allocation.items() if k != 'pass_fds'}
            flags = ['--locked'] + (['--offline'] if args.node_root else [])
            def cargo(*parts):
                return ['cargo', *parts, *flags]
            run('contracts', cargo('test', '-p', 'feat-math', '--test', 'kernel_codec',
                '--test', 'kernel_expression', '--test', 'kernel_pipeline', '--test', 'legacy_v2_freeze'))
            if args.profile in ('core', 'full'):
                run('library', cargo('test', '-p', 'feat-math', '--lib'))
                run('exports', cargo('test', '-p', 'feat-math', '--test', 'kernel_emission_safety',
                    '--test', 'advanced_v3_fitted_cpp', '--test', 'portable_parity'))
                run('parallel-library', cargo('check', '--lib'))
            if args.profile == 'full':
                run('golden', cargo('test', '-p', 'feat-math', '--test', 'golden'))
            run('assumptions', [sys.executable, str(probes / 'assumption_probe.py'),
                '--output', str(output / 'assumptions.json')])
            receipt['assumptions'] = research.assumptions(output / 'assumptions.json', probes / 'assumption-probe-v1.json')
            command = cargo('run', '-p', 'feat-math', '--example', 'screening_regime_probe')
            run('screening', command + ['--', str(probes / 'screening-manifest-v1.json')], 'screening.json')
            receipt['screening'] = research.screening(output / 'screening.json',
                probes / 'screening-manifest-v1.json', probes / 'screening-probe-v1.json')
            verify_bundle(args.bundle)
            receipt['status'] = 'passed'
    except (Deferred, BlockingIOError) as error:
        receipt.update(status='deferred', reason=str(error))
    except Exception as error:
        receipt.update(status='failed', reason=str(error))
    finally:
        for check in checks:
            if check['status'] == 'running':
                check['status'] = 'interrupted'
        receipt['seconds'] = time.time() - begun
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'core', 'full'), default='smoke')
    parser.add_argument('--command-timeout', type=int, default=1200)
    parser.add_argument('--node-root', type=Path, help='Use the live node resource scheduler and a restricted transient service')
    args = parser.parse_args()
    if not 1 <= args.command_timeout <= 1800:
        parser.error('command timeout must be in 1..1800 seconds')
    if args.node_root:
        supervised(args)
    result = verify(args)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 20 if result['status'] == 'deferred' else 1)
