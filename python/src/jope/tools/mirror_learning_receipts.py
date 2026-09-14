#!/usr/bin/env python3
"""Mirror learning-block receipts (never weights) from the node run into the git-tracked recovery tree.

Per training cycle `learning/cycle-NNNNNN/`: train-2/{complete.json, initialization.json, run.json, latest.json,
step-window.json}, the tail of metrics.jsonl (last 200 lines plus the sha of the whole file), every
checkpoints/step-*/state.json, plus the run-level learning/monitoring/*.json receipts, successor/decisions/*.json, the
research receipts (successor/scoreboard/{headline/*.json, latest.json, champion-git.json, flywheel-series.jsonl},
successor/verdicts/*.json, successor/acceptance/*.json, successor/champion-git/receipts/*.json), the performance audit
(review/measured-performance-audit-latest.json) and health/summary.json.
Files are pulled read-only over ssh/scp into <mirror>/learning-receipts/ with a MANIFEST.json (node path, sha256, bytes,
mtime) per cycle. A re-pull that differs from a mirrored receipt is a defect: the tool refuses and reports
`immutable receipt changed` instead of overwriting; only the files named in MUTABLE_BASENAMES (live feeds and the
append-only series) are allowed to change. Weights (model.ot, optimizer.ot), corpora and ledgers are never copied.
`--receipt` writes a parallel.learning_receipt_mirror_pull.v1 pull receipt; `verify --node-verify` re-hashes every
mirrored small file on the node (read-only) and reports what changed there since the pull (both-host verification).

  mirror_learning_receipts.py pull --node ubuntu@host --key KEY --node-root /home/ubuntu/jope-dime/parallel --run runs/<run> --mirror parallel/recovery/<run-id> [--receipt PATH]
  mirror_learning_receipts.py verify --mirror ... [--node-verify]
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

MANIFEST_SCHEMA = 'parallel.learning_receipt_mirror.v1'
PULL_RECEIPT_SCHEMA = 'parallel.learning_receipt_mirror_pull.v1'
CYCLE_FILES = ('train-2/complete.json', 'train-2/initialization.json', 'train-2/run.json', 'train-2/latest.json', 'train-2/step-window.json')
RUN_GLOBS = ('learning/monitoring/*.json', 'successor/decisions/*.json', 'successor/scoreboard/headline/*.json', 'successor/scoreboard/latest.json',
             'successor/scoreboard/champion-git.json', 'successor/scoreboard/flywheel-series.jsonl', 'successor/verdicts/*.json', 'successor/acceptance/*.json',
             'successor/champion-git/receipts/*.json', 'review/measured-performance-audit-latest.json', 'health/summary.json')
# Live feeds and the append-only series may legitimately change between pulls; everything else is an immutable receipt.
MUTABLE_BASENAMES = ('latest.json', 'run.json', 'champion-git.json', 'summary.json', 'measured-performance-audit-latest.json', 'flywheel-series.jsonl',
                     'f11-rollback-status.json')   # the rollback status is a live verification feed, rewritten by hold_release_readout.py
NODE_VERIFY_LIMIT_BYTES = 1 << 20   # both-host verification re-hashes mirrored files up to this size on the node
METRICS_TAIL_LINES = 200
TAIL_WINDOW_BYTES = 4 << 20   # the tail is cut from at most this many trailing bytes
FULL_HASH_LIMIT_BYTES = 1 << 30   # larger files are not hashed whole (cycle-000000/train-2/metrics.jsonl is 69 GB on the node)
FRAGMENT_BYTES = 64 << 10   # kept when the tail window holds no complete line
STUB_ABOVE_BYTES = 1_000_000   # larger files (checkpoint state.json embeds the replay membership) are mirrored as digest stubs
STUB_KEYS = ('schema', 'step', 'world', 'global_batch', 'microbatch', 'model_sha256', 'optimizer_sha256', 'optimizer_metadata_sha256', 'config_sha256', 'parameter_sha256',
             'corpus_sha256', 'vocabulary_sha256', 'sampler_id', 'ranking_backward', 'runtime_objective', 'seed', 'kind', 'checkpoint', 'timestamp', 'final_tests_opened')
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
for pattern in %r:
    for path in sorted(glob.glob(os.path.join(root, run, pattern))): add(path)
print(json.dumps(out))
''' % (node_root, run, CYCLE_FILES, RUN_GLOBS)
    result = remote_python(node, key, script)
    if result.returncode != 0:
        raise RuntimeError('remote listing failed: ' + result.stderr.strip()[-300:])
    return json.loads(result.stdout)


def fetch_stub(node, key, node_root, relative):
    """Digest stub of a large JSON receipt: sha256, bytes and its small scalar fields, found by a bounded scan.

    The file is never parsed (a 59 MB checkpoint state.json costs ~10 GB and minutes to json.loads on the node); the
    scalar keys of interest are picked out with one regular-expression pass over the bytes and labelled `scanned`.
    """
    script = (
        'import hashlib, json, re, sys\n'
        'p = %r; keys = %r\n'
        'h = hashlib.sha256(); size = 0; head = b""\n'
        'with open(p, "rb") as f:\n'
        '    while True:\n'
        '        piece = f.read(1 << 20)\n'
        '        if not piece: break\n'
        '        h.update(piece); size += len(piece)\n'
        '        if len(head) < (64 << 20): head += piece\n'
        'stub = dict(schema="parallel.receipt_stub.v1", path=%r, sha256=h.hexdigest(), bytes=size, scanned=True)\n'
        'fields = {}\n'
        'for k in keys:\n'
        '    m = re.search(rb\'"%%s"\\s*:\\s*("(?:[^"\\\\]|\\\\.){0,300}"|-?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?|true|false|null)\' %% re.escape(k.encode()), head)\n'
        '    if m:\n'
        '        try: fields[k] = json.loads(m.group(1))\n'
        '        except ValueError: pass\n'
        'stub["fields"] = fields\n'
        'sys.stdout.write(json.dumps(stub, sort_keys=True, indent=1))\n') % (os.path.join(node_root, relative), STUB_KEYS, relative)
    result = remote_python(node, key, script)
    if result.returncode != 0:
        raise RuntimeError('stub fetch failed for %s: %s' % (relative, result.stderr[-200:]))
    return result.stdout.encode(), json.loads(result.stdout)['sha256']


def fetch(node, key, node_root, relative, tail_lines=None, size=None):
    if tail_lines:
        # The tail is cut from at most TAIL_WINDOW_BYTES trailing bytes (a metrics.jsonl can be tens of GB with very long
        # lines; nothing here loads it into memory). The whole file is hashed by sha256sum only below FULL_HASH_LIMIT_BYTES;
        # above it the manifest records full_sha256 null and the size, and the tail file is the evidence.
        quoted = json.dumps(os.path.join(node_root, relative))
        hashed = size is None or size <= FULL_HASH_LIMIT_BYTES
        command = ('sha256sum %s | cut -d" " -f1 >&2 && ' % quoted if hashed else '') + 'tail -c %d %s | tail -n %d' % (TAIL_WINDOW_BYTES, quoted, tail_lines)
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-i', key, node, command], capture_output=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError('tail fetch failed for %s: %s' % (relative, result.stderr.decode(errors='replace')[-200:]))
        return result.stdout, (result.stderr.decode().strip() or None) if hashed else None
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-i', key, node, 'cat %s' % json.dumps(os.path.join(node_root, relative))], capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError('fetch failed for %s: %s' % (relative, result.stderr.decode(errors='replace')[-200:]))
    return result.stdout, None


def mutable(relative):
    return os.path.basename(relative) in MUTABLE_BASENAMES


def pull(node, key, node_root, run, mirror, receipt=None):
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
        stub = entry['bytes'] > STUB_ABOVE_BYTES and not tail
        known = manifest['files'].get(relative)
        if known and not tail and known.get('bytes') == entry['bytes'] and known.get('mtime') == entry['mtime']:
            unchanged.append(relative)
            continue
        data, full_sha = fetch_stub(node, key, node_root, relative) if stub else fetch(node, key, node_root, relative, tail, entry['bytes'])
        fragment = bool(tail) and data.count(b'\n') < 2
        if fragment:
            data = data[-FRAGMENT_BYTES:]   # the window held no complete line (gigantic rows): keep a bounded fragment, labelled
        # Mirrored under the run-relative path (never a `runs/` component: the repository ignores that directory name).
        inside = relative[len(run) + 1:] if relative.startswith(run + '/') else relative.replace('runs/', 'run-', 1)
        target = mirror / (inside + ('.tail' if tail else '.stub.json' if stub else ''))
        digest = sha256(data)
        if known and known.get('sha256') != digest and not tail and not mutable(relative):
            refused.append(dict(path=relative, mirrored=known.get('sha256'), node=digest))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest['files'][relative] = dict(sha256=digest, bytes=entry['bytes'], mtime=entry['mtime'], tail_lines=tail, stub=stub, full_sha256=full_sha, mirrored=str(target.relative_to(mirror)),
                                           truncated_fragment=fragment, mutable=mutable(relative))
        pulled.append(relative)
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=1).encode()
    manifest_path.write_bytes(manifest_bytes)
    result = dict(pulled=len(pulled), unchanged=len(unchanged), files=len(manifest['files']), manifest=str(manifest_path), refused=refused)
    if receipt:
        # The pull receipt binds the manifest and the tool that produced it; refusals are recorded, never hidden.
        files = manifest['files']
        document = dict(schema=PULL_RECEIPT_SCHEMA, at=time.time(), node=node, run=run, pulled=len(pulled), unchanged=len(unchanged), files=len(files),
                        refused=refused, stubs=sum(1 for v in files.values() if v.get('stub')), tails=sum(1 for v in files.values() if v.get('tail_lines')),
                        mutable_files=sum(1 for v in files.values() if v.get('mutable')), manifest=str(manifest_path), manifest_sha256=sha256(manifest_bytes),
                        tool_sha256=sha256(Path(__file__).read_bytes()), pulled_paths=pulled)
        Path(receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt).write_bytes(json.dumps(document, sort_keys=True, indent=1).encode())
        result['receipt'] = str(receipt)
    if refused:
        raise RuntimeError('immutable receipt changed on the node: %s' % json.dumps(refused)[:600])
    return result


def node_hashes(node, key, node_root, relatives):
    """sha256 of each listed node file (null when absent), computed on the node in one read-only pass."""
    script = ('import hashlib, json, os, sys\n'
              'root = %r; out = {}\n'
              'for rel in %r:\n'
              '    p = os.path.join(root, rel)\n'
              '    try:\n'
              '        h = hashlib.sha256()\n'
              '        with open(p, "rb") as f:\n'
              '            for piece in iter(lambda: f.read(1 << 20), b""): h.update(piece)\n'
              '        out[rel] = h.hexdigest()\n'
              '    except OSError:\n'
              '        out[rel] = None\n'
              'sys.stdout.write(json.dumps(out))\n') % (node_root, list(relatives))
    result = remote_python(node, key, script, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError('node verification failed: ' + result.stderr.strip()[-300:])
    return json.loads(result.stdout)


def verify(mirror, node=None, key=None, node_root=None, node_verify=False):
    mirror = Path(mirror) / 'learning-receipts'
    manifest = json.loads((mirror / 'MANIFEST.json').read_bytes())
    bad = []
    for relative, entry in manifest['files'].items():
        path = mirror / entry['mirrored']
        if not path.is_file() or sha256(path.read_bytes()) != entry['sha256']:
            bad.append(relative)
    result = dict(files=len(manifest['files']), mismatched=bad)
    if node_verify:
        # Both-host check: every mirrored whole file small enough to re-hash cheaply is hashed again on the node and
        # compared with the manifest. Immutable receipts that differ are defects; mutable feeds that moved on are reported
        # separately. Nothing is overwritten either way.
        eligible = [relative for relative, entry in manifest['files'].items()
                    if not entry.get('stub') and not entry.get('tail_lines') and entry.get('bytes', 0) <= NODE_VERIFY_LIMIT_BYTES]
        remote = node_hashes(node, key, node_root, eligible) if eligible else {}
        immutable_mismatch, mutable_changed, missing = [], [], []
        for relative in eligible:
            expected = manifest['files'][relative]['sha256']
            actual = remote.get(relative)
            if actual is None:
                missing.append(relative)
            elif actual != expected:
                (mutable_changed if mutable(relative) else immutable_mismatch).append(dict(path=relative, mirrored=expected, node=actual))
        result.update(node=node, node_verified=len(eligible) - len(immutable_mismatch) - len(mutable_changed) - len(missing), node_checked=len(eligible),
                      node_mismatched=immutable_mismatch, node_changed_mutable=mutable_changed, node_missing=missing, node_skipped=len(manifest['files']) - len(eligible))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('pull', 'verify'))
    parser.add_argument('--node', default='ubuntu@158.101.20.226')
    parser.add_argument('--key', default=str(Path.home() / '.ssh/lambda_cloud_a100_20260907_ed25519'))
    parser.add_argument('--node-root', default='/home/ubuntu/jope-dime/parallel')
    parser.add_argument('--run', default='runs/eight-a100-recovery-20260912')
    parser.add_argument('--mirror', required=True)
    parser.add_argument('--receipt', help='pull: write a parallel.learning_receipt_mirror_pull.v1 receipt here')
    parser.add_argument('--node-verify', action='store_true', help='verify: also re-hash the mirrored small files on the node (read-only) and report drift')
    args = parser.parse_args(argv)
    if args.action == 'pull':
        result = pull(args.node, args.key, args.node_root, args.run, args.mirror, args.receipt)
    else:
        result = verify(args.mirror, args.node, args.key, args.node_root, args.node_verify)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
