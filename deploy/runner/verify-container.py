#!/usr/bin/env python3
"""Verify observed cgroup caps and immutable host policy from inside a CPU job."""
import hashlib
import json
from pathlib import Path


def verify(policy, memory, cpu, swap, devices):
    role = (policy['host'], policy['lane'])
    if role in {('xbabe1', 'ci'), ('xbabe1', 'ci2')}:
        domain, gib = 'pr', 16
    elif role == ('xbabe2', 'full'):
        domain, gib = 'main', 48
    elif role == ('xbabe2', 'guard'):
        domain, gib = 'main', 8
    else:
        raise ValueError('unapproved host/lane')
    if policy != dict(host=role[0], lane=role[1], domain=domain, memory=f'{gib}g', cpus=16,
                      volume_prefix=f'jope-{domain}-{role[1]}'):
        raise ValueError('unexpected immutable launch policy')
    quota, period = cpu.split()
    if memory != str(gib * 1024**3) or quota == 'max' or int(quota) != 16 * int(period) or swap != '0':
        raise ValueError('cgroup limits do not equal the approved CPU budget')
    if devices:
        raise ValueError('CPU container exposes a GPU device or Docker socket')


def main():
    policy = json.loads(Path('/opt/jope-runner/policy.json').read_text())
    cgroup = Path('/sys/fs/cgroup')
    verify(policy, (cgroup/'memory.max').read_text().strip(), (cgroup/'cpu.max').read_text().strip(),
           (cgroup/'memory.swap.max').read_text().strip(),
           list(Path('/dev').glob('nvidia*')) + [p for p in (Path('/var/run/docker.sock'),) if p.exists()])
    # Same deploy-AFTER-merge one-way transition as runner-provenance: when this commit
    # carries the PR head hook and the host still has main's hook, accept. Reverse fails.
    # Remove with the provenance allowlist after post-merge provision.
    transition_base = {
        'job-policy.py': '3cc8cfac3b83a2746402c1b3f44351cb9746e208f23a19bf1ea653815306cbf6',
    }
    transition_head = {
        'job-policy.py': '6c634baf784ea5f8d487c2d7b6aec5b354e0d7f68122f50567a19f777dac5f9d',
    }
    for name in ('job-started.sh', 'job-policy.py'):
        actual = hashlib.sha256((Path('/opt/jope-runner') / name).read_bytes()).hexdigest()
        expected = hashlib.sha256((Path('deploy/runner') / name).read_bytes()).hexdigest()
        if actual == expected:
            continue
        if expected == transition_head.get(name) and actual == transition_base.get(name):
            continue
        raise ValueError(f'installed admission hook differs: {name}')
    print(json.dumps({'policy': policy, 'memory_max': (cgroup/'memory.max').read_text().strip(),
                      'cpu_max': (cgroup/'cpu.max').read_text().strip(), 'swap_max': '0'}, sort_keys=True))


if __name__ == '__main__':
    main()
