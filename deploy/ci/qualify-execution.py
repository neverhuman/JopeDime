#!/usr/bin/env python3
"""Explicit synthetic host proof; unavailable controls are failures, never skips."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--test-binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    binary = args.test_binary.resolve(strict=True)
    output = args.output.resolve()
    if not output.is_absolute() or output.is_relative_to('/tmp'):
        raise ValueError('durable output outside /tmp required')
    output.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[:20]
    if len(cpus) != 20:
        raise RuntimeError('twenty available CPUs required for positive concurrency proof')
    source = hashlib.sha256(binary.read_bytes()).hexdigest()
    records = []
    for case in ('timeout', 'controller-stall-success', 'controller-stall-timeout', 'controller-sigkill', 'concurrency20', 'memory', 'graceful-sigterm', 'original-cutoff'):
        unit = f'jope-asm-qualification-{os.getpid()}-{case}.service'
        armed = output / 'sigkill-armed.json'
        command = ['sudo', '-n', 'systemd-run', '--wait', '--pipe', f'--unit={unit}',
                   '--property=Type=exec', '--property=ExitType=main',
                   '--property=User=ubuntu', '--property=Restart=no', '--property=RemainAfterExit=no',
                   '--property=OOMPolicy=continue',
                   '--property=KillMode=control-group', '--property=SendSIGKILL=yes',
                   '--property=FinalKillSignal=SIGKILL', '--property=TimeoutStopSec=1s',
                   f'--property=RuntimeMaxSec={7 if case == "original-cutoff" else 30}s', '--property=Delegate=cpu cpuset memory',
                   '--property=DelegateSubgroup=controller',
                   f'--property=CPUQuota={2000 if case == "concurrency20" else 200}%',
                   f'--property=AllowedCPUs={" ".join(map(str, cpus if case == "concurrency20" else cpus[:2]))}',
                   f'--property=MemoryMax={"1G" if case == "concurrency20" else "256M"}',
                   '--property=MemorySwapMax=0', f'--setenv=JOPE_EXECUTION_TEST_UNIT={unit}',
                   f'--setenv=JOPE_EXECUTION_TEST_CASE={case}']
        if case == 'controller-sigkill':
            command += [f'--setenv=JOPE_EXECUTION_SIGKILL_ARMED={armed}']
        command += [str(binary), '--exact', 'execution::tests::real_supervisor_resources_and_descendant_cleanup', '--ignored', '--nocapture']
        began = time.time()
        result = subprocess.run(command, capture_output=True, text=True, timeout=40)
        (output / f'{case}.stdout').write_text(result.stdout)
        (output / f'{case}.stderr').write_text(result.stderr)
        if case != 'controller-sigkill':
            receipts = [line.split('execution_host_receipt=', 1)[1] for line in result.stdout.splitlines() if 'execution_host_receipt=' in line]
            if result.returncode or len(receipts) != 1:
                raise RuntimeError(f'{case} proof did not execute and pass exactly one real host fixture')
            receipt = json.loads(receipts[0])
            if case == 'concurrency20':
                if (any(receipt.get(k) != v for k, v in dict(case='concurrency20', peak=20, completed=21, active_after=0).items()) or not receipt['all_fit_limits_read_back'] or receipt['fit_memory_allocation']['per_fit_bytes'] * 20 + receipt['fit_memory_allocation']['controller_reserve_bytes'] != 1 << 30):
                    raise RuntimeError('twenty-worker concurrency/cap proof failed')
            elif case in ('graceful-sigterm', 'original-cutoff'):
                expected = 'cancelled' if case == 'graceful-sigterm' else 'timed_out'
                if (receipt['case'] != case or receipt['outcome']['command_status'] != expected
                        or not receipt['descendants_drained'] or not receipt['resumed_without_new_gate']
                        or receipt['original_window'] != receipt['resumed_window']
                        or receipt['remaining_global_ms'] <= 0
                        or receipt['outcome']['consumption']['fit_memory_limit_bytes'] != receipt['fit_memory_allocation']['per_fit_bytes']):
                    raise RuntimeError('graceful cancellation/cutoff did not retain its original bounded journal')
                journal = Path(receipt['journal']).resolve(strict=True)
                retained_files = {'window.json': receipt['window_sha256'],
                                  'fit-memory-policy.json': receipt['allocation_file_sha256'],
                                  'finished/' + receipt['outcome']['job_sha256'] + '.json': receipt['finished_sha256']}
                for relative, digest in retained_files.items():
                    if hashlib.sha256((journal / relative).read_bytes()).hexdigest() != digest:
                        raise RuntimeError('retained original cancellation proof changed')
            elif case == 'controller-stall-success':
                if receipt['command_status'] != 'completed' or receipt['consumption']['wall_seconds'] >= 1.25:
                    raise RuntimeError('controller stall changed a successful fit result')
            elif case == 'memory':
                if receipt['command_status'] != 'failed' or not receipt['reason'].startswith('memory_limit:') or receipt['consumption']['cpu_seconds'] is None:
                    raise RuntimeError('forced memory-limit failure not observed')
            elif receipt['command_status'] != 'timed_out' or receipt['consumption']['cpu_seconds'] is None:
                raise RuntimeError('missing timed-out measurement')
        else:
            if result.returncode == 0 or not armed.is_file():
                raise RuntimeError('controller SIGKILL was not armed and observed')
            receipt = json.loads(armed.read_text())
            expected = Path('/sys/fs/cgroup/system.slice') / unit
            if receipt['unit'] != unit or Path(receipt['cgroup']).parent != expected:
                raise RuntimeError('foreign controller proof receipt')
            # systemd-run --wait has returned after unit stop; any surviving live
            # process in the unit tree invalidates the crash-cleanup proof.
            events = expected / 'cgroup.events'
            if events.exists() and 'populated 0' not in events.read_text().splitlines():
                raise RuntimeError('controller died but descendants survived')
            props = subprocess.run(['systemctl', 'show', unit, '-p', 'ExecMainCode,ExecMainStatus,Result'], capture_output=True, text=True, check=True).stdout
            if 'ExecMainStatus=9' not in props.splitlines() or 'Result=signal' not in props.splitlines():
                raise RuntimeError(f'controller did not exit by SIGKILL: {props}')
            descendant = Path('/proc') / str(receipt['descendant_pid'])
            if descendant.exists():
                state = (descendant / 'stat').read_text().rsplit(')', 1)[1].split()[0]
                if state not in ('Z', 'X'):
                    raise RuntimeError('controller SIGKILL left the recorded descendant alive')
                receipt['descendant_terminal_state'] = state
            receipt['systemd_exit'] = props
        subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', unit], capture_output=True, timeout=5)
        records.append(dict(case=case, test_binary_sha256=source, command=command,
                            exit_status=result.returncode, started_unix=began,
                            wall_seconds=time.time()-began, receipt=receipt))
        (output / 'receipt.json').write_text(json.dumps(dict(schema='jope.execution-host-proof/v1', cases=records), indent=2)+'\n')
    print(output / 'receipt.json')


if __name__ == '__main__':
    main()
