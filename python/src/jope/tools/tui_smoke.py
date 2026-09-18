#!/usr/bin/env python3
"""Exercise the actual terminal UI without sending controls to any worker service."""
import argparse
import fcntl
import hashlib
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import termios
import time
from pathlib import Path

from experiment_runtime import output_root, require, write

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--experiment-manifest', type=Path)
args = p.parse_args()
args.output = output_root(args.root, args.output)
args.output.mkdir(parents=True, exist_ok=True)

def state(name):
    return json.loads((args.root / 'runs' / name).read_text())

before = state('gp-service.json')
hardware_before = state('hardware.json')['timestamp']
campaign = state('status.json')
context_index = json.loads((args.root / campaign.get('contexts_index', 'data/encodings/contexts-v1/index.json')).read_text())
context_count = len(context_index['encodings'])
planned=set()
for manifest in campaign.get('prepared_job_manifests',[]):
    for job in json.loads((args.root/manifest).read_text())['jobs']:
        planned.add((Path(job['prepared']).name,job['cut']))
context_total=len(planned) if planned else context_count
captures = []
frames = []
for session in range(2):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 55, 160, 0, 0))
    command = [str(args.binary), '--root', str(args.root), 'tui']
    if args.experiment_manifest:
        command += ['--experiment-manifest', str(args.experiment_manifest)]
    proc = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, env={**os.environ, 'TERM': 'xterm-256color'}, start_new_session=True)
    os.close(slave)
    chunks = []
    def drain(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], min(.1, max(0, deadline-time.monotonic())))
            if ready:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
    try:
        drain(1.5)
        if session == 0:
            frames.append(dict(view=1,offset=sum(map(len,chunks)),columns=160,rows=55,timestamp=time.time()))
            for key in b'23451':
                os.write(master, bytes([key])); drain(.8)
                if key != ord('1'):
                    frames.append(dict(view=int(chr(key)),offset=sum(map(len,chunks)),columns=160,rows=55,timestamp=time.time()))
            for height, width in [(15, 48), (55, 160)]:
                fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack('HHHH', height, width, 0, 0))
                os.kill(proc.pid, signal.SIGWINCH); drain(.8)
        os.write(master, b'q'); drain(.5)
        require(proc.wait(timeout=5) == 0, "TUI did not exit cleanly")
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=5)
        os.close(master)
    data = b''.join(chunks)
    path = args.output / f'session-{session}.ansi'
    path.write_bytes(data)
    captures.append({'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    if session == 0:
        (args.output/'frames.json').write_text(json.dumps(dict(schema='parallel.tui_terminal_frames.v1',ansi=path.name,ansi_sha256=hashlib.sha256(data).hexdigest(),frames=frames),indent=2)+'\n')
    rendered = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', data.decode('utf-8', errors='replace'))
    # Ratatui uses cursor jumps for unchanged spaces; check visible words independently of spacing.
    rendered = re.sub(r'\s+', '', rendered)
    require('JOPE/PARALLEL' in rendered, "TUI title missing")
    if session == 0:
        policy_title = 'PURE SEED QUALITY' if (args.root/'runs/seed-quality-experiment-v1/report.json').exists() else 'POLICY VALUE'
        expected = [f'{context_count} / {context_total}', 'All eight GPUs', 'Every logical CPU', 'Saved checkpoint', 'Adaptive GP teacher searches', 'Task board', policy_title, 'Sequence NLL']
        if policy_title == 'PURE SEED QUALITY':
            expected += ['Selected checkpoint', 'Predictor export', 'Corpus packing', 'Seed quality experiment', 'Binary qualification']
        if args.experiment_manifest:
            expected = ['All eight GPUs', 'Every logical CPU', 'Saved checkpoint', 'Task board', 'Production selected:', 'Experiment frozen:']
        for text in expected:
            require(re.sub(r'\s+', '', text) in rendered, f"TUI text missing: {text}")

after = state('gp-service.json')
hardware_after = state('hardware.json')['timestamp']
require(after['timestamp'] > before['timestamp'], "GP service receipt did not advance")
require(hardware_after > hardware_before, "hardware receipt did not advance")
receipt = {'schema':'parallel.tui_live_smoke.v3','passed':True,'timestamp':int(time.time()),'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),'five_views':True,'task_board':True,'learning_curves':True,'policy_comparisons':True,'resize':[[160,55],[48,15],[160,55]],'quit_detaches':True,'reconnect':True,'hardware_timestamps':[hardware_before,hardware_after],'gp_timestamps':[before['timestamp'],after['timestamp']],'gp_active_after':after['active'],'gp_completed_before':before['completed_views'],'gp_completed_after':after['completed_views'],'captures':captures}
receipt['experiment_manifest'] = str(args.experiment_manifest) if args.experiment_manifest else None
write(args.output / 'receipt.json', receipt)
print(json.dumps(receipt,indent=2))
