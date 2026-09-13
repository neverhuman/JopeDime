#!/usr/bin/env python3
"""Mirror learning-block receipts (never weights) from the node run into the git-tracked recovery tree.

Per training cycle `learning/cycle-NNNNNN/`: train-2/{complete.json, initialization.json, run.json, latest.json,
step-window.json}, the tail of metrics.jsonl (last 200 lines plus the sha of the whole file), every
checkpoints/step-*/state.json, plus the run-level learning/monitoring/*.json receipts and successor/decisions/*.json.
Files are pulled read-only over ssh/scp into <mirror>/learning-receipts/ with a MANIFEST.json (node path, sha256, bytes,
mtime) per cycle. A re-pull that differs from a mirrored receipt is a defect: the tool refuses and reports
`immutable receipt changed` instead of overwriting. Weights (model.ot, optimizer.ot), corpora and ledgers are never copied.

  mirror_learning_receipts.py pull --node ubuntu@host --key KEY --node-root /home/ubuntu/jope-dime/parallel --run runs/<run> --mirror parallel/recovery/<run-id>
  mirror_learning_receipts.py verify --mirror ...
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

MANIFEST_SCHEMA = 'parallel.learning_receipt_mirror.v1'
CYCLE_FILES = ('train-2/complete.json', 'train-2/initialization.json', 'train-2/run.json', 'train-2/latest.json', 'train-2/step-window.json')
METRICS_TAIL_LINES = 200
FORBIDDEN_SUFFIXES = ('.ot', '.bin', '.sqlite', '.sqlite-wal', '.sqlite-shm')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ssh(node, key, command, timeout=600, stdin=None):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-i', key, node, command], capture_output=True, text=True, timeout=timeout, input=stdin)


def remote_python(node, key, script, timeout=600):
    """Run a Python script on the node by feeding it over stdin (no shell quoting of the source)."""
    return ssh(node, key, 'python3 -', timeout=timeout, stdin=script)


def remote_listing(node, key, node_root, run):
    """Cycle receipts, checkpoint state files, monitoring receipts and decisions on the node (paths, sizes, mtimes)."""
    script = r'''
import json, os, glob
root = %r; run = %r
out = []
def add(path):
    try: st = os.stat(path)
    except OSError: return
    out.append(dict(path=os.path.relpath(path, root), bytes=st.st_size, mtime=st.st_mtime))
for cycle in sorted(glob.glob(os.path.join(root, run, 'learning/cycle-*'))):
    for name in %r: add(os.path.join(cycle, name))
    for state in glob.glob(os.path.join(cycle, 'train-2/checkpoints/step-*/state.json')): add(state)
    add(os.path.join(cycle, 'train-2/metrics.jsonl'))
for path in glob.glob(os.path.join(root, run, 'learning/monitoring/*.json')): add(path)
for path in glob.glob(os.path.join(root, run, 'successor/decisions/*.json')): add(path)
print(json.dumps(out))
''' % (node_root, run, CYCLE_FILES)
    result = remote_python(node, key, script)
    if result.returncode != 0:
        raise RuntimeError('remote listing failed: ' + result.stderr.strip()[-300:])
    return json.loads(result.stdout)


def fetch(node, key, node_root, relative, tail_lines=None):
    if tail_lines:
        script = 'import hashlib,sys; p=%r; d=open(p,"rb").read(); lines=d.splitlines(True); sys.stdout.buffer.write(b"".join(lines[-%d:])); sys.stderr.write(hashlib.sha256(d).hexdigest())' % (os.path.join(node_root, relative), tail_lines)
        result = remote_python(node, key, script)
        if result.returncode != 0:
            raise RuntimeError('tail fetch failed for %s: %s' % (relative, result.stderr[-200:]))
        return result.stdout.encode(), result.stderr.strip()
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-i', key, node, 'cat %s' % json.dumps(os.path.join(node_root, relative))], capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError('fetch failed for %s: %s' % (relative, result.stderr.decode(errors='replace')[-200:]))
    return result.stdout, None


def pull(node, key, node_root, run, mirror):
    mirror = Path(mirror) / 'learning-receipts'
    mirror.mkdir(parents=True, exist_ok=True)
    manifest_path = mirror / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_bytes()) if manifest_path.is_file() else dict(schema=MANIFEST_SCHEMA, run=run, files={})
    listing = remote_listing(node, key, node_root, run)
    pulled, unchanged, refused = [], [], []
    for entry in listing:
        relative = entry['path']
        if relative.endswith(FORBIDDEN_SUFFIXES):
            continue
        tail = METRICS_TAIL_LINES if relative.endswith('metrics.jsonl') else None
        known = manifest['files'].get(relative)
        if known and not tail and known.get('bytes') == entry['bytes'] and known.get('mtime') == entry['mtime']:
            unchanged.append(relative)
            continue
        data, full_sha = fetch(node, key, node_root, relative, tail)
        target = mirror / (relative + ('.tail' if tail else ''))
        digest = sha256(data)
        if known and known.get('sha256') != digest and not tail and not relative.endswith('latest.json') and not relative.endswith('run.json'):
            refused.append(dict(path=relative, mirrored=known.get('sha256'), node=digest))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest['files'][relative] = dict(sha256=digest, bytes=entry['bytes'], mtime=entry['mtime'], tail_lines=tail, full_sha256=full_sha, mirrored=str(target.relative_to(mirror)))
        pulled.append(relative)
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, indent=1).encode())
    if refused:
        raise RuntimeError('immutable receipt changed on the node: %s' % json.dumps(refused)[:600])
    return dict(pulled=len(pulled), unchanged=len(unchanged), files=len(manifest['files']), manifest=str(manifest_path))


def verify(mirror):
    mirror = Path(mirror) / 'learning-receipts'
    manifest = json.loads((mirror / 'MANIFEST.json').read_bytes())
    bad = []
    for relative, entry in manifest['files'].items():
        path = mirror / entry['mirrored']
        if not path.is_file() or sha256(path.read_bytes()) != entry['sha256']:
            bad.append(relative)
    return dict(files=len(manifest['files']), mismatched=bad)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('pull', 'verify'))
    parser.add_argument('--node', default='ubuntu@158.101.20.226')
    parser.add_argument('--key', default=str(Path.home() / '.ssh/lambda_cloud_a100_20260907_ed25519'))
    parser.add_argument('--node-root', default='/home/ubuntu/jope-dime/parallel')
    parser.add_argument('--run', default='runs/eight-a100-recovery-20260912')
    parser.add_argument('--mirror', required=True)
    args = parser.parse_args(argv)
    result = pull(args.node, args.key, args.node_root, args.run, args.mirror) if args.action == 'pull' else verify(args.mirror)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
