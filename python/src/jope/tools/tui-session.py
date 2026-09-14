#!/usr/bin/env python3
"""Supervise the hash-verified observer in the reconnectable jope tmux session."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/home/ubuntu/jope-dime/parallel')
BINARY = ROOT / 'runs/continuous-observer-v5-solid-20260909/jope-parallel-observer'
EXPECTED = 'fa23e153971a08903c7729ad9030d70b98fe84bd162f158b69e54eb75c9f7244'


def verify():
    if hashlib.sha256(BINARY.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError('reviewed full-history observer executable hash changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observer', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    verify()
    if args.check:
        print(EXPECTED)
        return
    if args.observer:
        os.execv(str(BINARY), [str(BINARY), '--root', str(ROOT), 'tui'])
    command = shlex.join(['/usr/bin/python3', str(Path(__file__).resolve()), '--observer'])
    while True:
        verify()
        exists = subprocess.run(['tmux', 'has-session', '-t', '=jope'], capture_output=True).returncode == 0
        if not exists:
            subprocess.run(['tmux', 'new-session', '-d', '-s', 'jope', '-x', '180', '-y', '50', command], check=True)
            subprocess.run(['tmux', 'set-option', '-t', 'jope', 'status', 'off'], check=True)
        time.sleep(2)


if __name__ == '__main__':
    main()
