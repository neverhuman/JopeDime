#!/usr/bin/env python3
"""Trusted node -> xbabe2 mirror (explicit one-shot administration; no daemon). Read-only on the node.

Per class in the spec (champions, champion-git, receipts, weights, dataset): one census on the node (size, mtime and sha256
of every regular file, computed there at nice 19; meta_test champion identities are skipped and counted), a delta against
the local index, `rsync --checksum --partial --files-from` into <root>/.incoming-<uuid>/, a local re-hash against the
census, refusal (never overwrite) when an immutable path already exists locally with a different digest, atomic install,
a second sha256 pass on the node for exactly the pulled paths (both-host: a file that changed during the transfer is
reported, not installed as verified), and an immutable receipt parallel.node_mirror_receipt.v1 under
<root>/receipts/mirror/. The champion-git class is additionally checked with `git fsck` and head == newest sync receipt.

  node_mirror_pull.py --class champions --class champion-git --class receipts [--class dataset --class weights] [--root /home/ubuntu/jope-node-mirror]
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import promotion_gate
from resume_expired_cycle import atomic, read, require, sha

SCHEMA = 'parallel.node_mirror_receipt.v1'
NODE = 'ubuntu@158.101.20.226'
KEY = str(Path.home() / '.ssh/lambda_cloud_a100_20260907_ed25519')
NODE_ROOT = '/home/ubuntu/jope-dime/parallel'
RUN = 'runs/eight-a100-recovery-20260912'
MUTABLE_BASENAMES = ('current.json', 'f11-rollback-status.json', 'latest.json', 'champion-git.json', '.champion-git-state.json', 'HEAD', 'ORIG_HEAD', 'FETCH_HEAD', 'COMMIT_EDITMSG', 'description', 'index', 'packed-refs', 'config', 'main')
# class -> (base directory on the node the paths are relative to, walk root under that base, rules)
SPEC = {
    'champions': dict(base=NODE_ROOT, root='runs/champions', kind='champions'),
    'champion-git': dict(base=NODE_ROOT, root=RUN + '/successor/champion-git', kind='tree', exclude=('/repo/programs/', '/repo/champions/', '/repo/index.json')),
    'receipts': dict(base=NODE_ROOT, root=RUN, kind='globs', globs=('learning/monitoring/*.json', 'successor/decisions/*.json', 'successor/scoreboard/headline/*.json', 'successor/verdicts/*.json', 'successor/acceptance/*.json')),
    'weights': dict(base=NODE_ROOT, root=RUN + '/learning', kind='globs', globs=('cycle-*/train-2/checkpoints/step-*/state.json', 'cycle-*/train-2/checkpoints/step-*/model.ot', 'cycle-*/train-2/checkpoints/step-*/optimizer.ot', 'cycle-*/train-2/checkpoints/step-*/optimizer.json', 'cycle-*/train-2/checkpoints/step-*/config.json')),
    'dataset': dict(base='/', root='home/ubuntu/quant', kind='tree'),
    # owner instruction 2026-09-14: encodings, splits and per-dataset/split GP results are migrated too
    'prepared': dict(base=NODE_ROOT, root='data/prepared', kind='tree'),
    'contexts': dict(base=NODE_ROOT, root='runs/dataset-inclusion-20260909/correction-final', kind='tree'),
    'teaching-corpus': dict(base=NODE_ROOT, root='runs/dataset-inclusion-20260909/teaching-corpus', kind='tree'),
    'results': dict(base=NODE_ROOT, root=RUN + '/learning', kind='globs', globs=('generation-*/generation.json', 'generation-*/jobs.json', 'generation-*/quality-report.json', 'generation-*/champion-snapshot.json',
                                                                            'generation-*/*/complete.json', 'generation-*/*/run.json', 'generation-*/*/jobs.json',
                                                                            'generation-*/*/*/cut-*/seeds.json', 'generation-*/*/*/cut-*/ranking.json', 'generation-*/*/*/cut-*/best-predicted.json', 'generation-*/*/*/cut-*/generation.json',
                                                                            'generation-*/*/*/cut-*/search/search.json', 'generation-*/*/*/cut-*/search/progress.json')),
    'observations': dict(base=NODE_ROOT, root=RUN + '/learning', kind='globs', globs=('generation-*/*/*/cut-*/search/observations.json',)),
    # code and stores that exist ONLY on the node (no git there): source, deployed executables/manifests, the dataset-lane store, home scripts
    'node-source': dict(base=NODE_ROOT, root='.', kind='roots', roots=('src', 'tools', 'tests', 'examples', 'docs', 'systemd', 'Cargo.toml', 'Cargo.lock', 'build.rs', 'README.md')),
    'node-release': dict(base=NODE_ROOT, root='.', kind='roots', roots=('bin', 'native', 'builds', 'releases', 'manifests'), exclude=('/target/',)),
    'node-home': dict(base='/home/ubuntu', root='.', kind='roots', roots=('prepare_native_v8/src', 'prepare_native_v8/Cargo.toml', 'prepare_native_v8/Cargo.lock'), globs=('*.py', '*.sh', '*.out', '*.log')),
    'kernel-mirror': dict(base='/', root='home/ubuntu/jope-dime-mirror', kind='tree'),   # Codex's kernel work mirror on the node (formerly mislabelled lane-store)
    'recovery-20260908': dict(base='/', root='home/ubuntu/jope-dime-recovery', kind='tree'),
    'corpus': dict(base=NODE_ROOT, root='corpus', kind='tree'),
    'assets': dict(base=NODE_ROOT, root='assets', kind='tree'),
    'kernel-work': dict(base=NODE_ROOT, root='.', kind='roots', roots=('kernel-work', 'kernel-work-evidence')),
    # census-only classes (sha256 index + receipt, no transfer): the run roots too large to copy without an owner decision
    'runs-census': dict(base=NODE_ROOT, root='runs', kind='roots', census_only=True,
                        roots=('continuous-learning-v2-20260909', 'flywheel-20260909', 'regression-campaign64-qualification-v1', 'train-check-8', 'registry-integration-20260910',
                               'all-gap-closure-20260911', 'continuous-health-20260909', 'gp-reviewed3', 'gp-expanded', 'parity', 'continuous-packer-qualification-20260909')),
    'recovery-run-census': dict(base=NODE_ROOT, root=RUN, kind='roots', census_only=True, roots=('learning/snapshot-*', 'learning/cycle-*')),   # packed corpora (99 GB) and training cycles incl. the 69 GB cycle-0 metrics ledger
    # owner decision D-H (2026-09-14): every run root <= 20 GB is copied; the live recovery run's review/experiments/proposal-compat trees too
    'runs-small': dict(base=NODE_ROOT, root='runs', kind='sized_roots', max_bytes=20 << 30, skip=('eight-a100-recovery-20260912', 'continuous-learning-v2-20260909', 'flywheel-20260909', 'regression-campaign64-qualification-v1', 'champions', 'dataset-inclusion-20260909'),
                       exclude=('/target/', '/.conversion-venv/', '/debug/deps/', '/release/deps/', '/incremental/', '/.fingerprint/', '/debug/build/', '/release/build/')),
    'recovery-review': dict(base=NODE_ROOT, root=RUN + '/review', kind='tree', exclude=('/target/', '/debug/deps/', '/release/deps/', '/incremental/', '/.fingerprint/', '/debug/build/', '/release/build/')),
    'recovery-experiments': dict(base=NODE_ROOT, root=RUN + '/experiments', kind='tree', exclude=('/target/', '/debug/deps/', '/release/deps/', '/incremental/', '/.fingerprint/', '/debug/build/', '/release/build/')),
    'proposal-compat': dict(base=NODE_ROOT, root=RUN + '/proposal-compat', kind='tree', exclude=('/target/',)),
}

CENSUS_SCRIPT = r'''
import glob, hashlib, json, os, sys
spec = %(spec)r
base = spec['base']; root = os.path.join(base, spec['root'])
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for piece in iter(lambda: f.read(1 << 20), b''): h.update(piece)
    return h.hexdigest()
paths = []; skipped = 0
kind = spec['kind']
if kind == 'champions':
    with os.scandir(root) as it:
        for d in it:
            if not d.is_dir(): continue
            try:
                part = json.load(open(os.path.join(d.path, 'current.json'))).get('identity', {}).get('partition')
            except Exception:
                part = None
            if part == 'meta_test':
                skipped += 1; continue
            with os.scandir(d.path) as inner:
                for f in inner:
                    if f.is_file(follow_symlinks=False) and (f.name == 'current.json' or f.name.startswith('event-')): paths.append(f.path)
elif kind == 'globs':
    for g in spec['globs']:
        paths.extend(p for p in glob.glob(os.path.join(root, g)) if os.path.isfile(p) and not os.path.islink(p))
elif kind == 'sized_roots':
    # every top-level entry under root whose total regular-file bytes are <= max_bytes (computed here), minus the skip list
    exclude = tuple(spec.get('exclude') or ()); skip = set(spec.get('skip') or ()); chosen = []
    with os.scandir(root) as it:
        entries = sorted(e.name for e in it if not e.is_symlink())
    for name in entries:
        if name in skip: continue
        start = os.path.join(root, name)
        if os.path.isfile(start):
            chosen.append(start); continue
        total = 0; found = []
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
            for f in filenames:
                p = os.path.join(dirpath, f)
                if os.path.islink(p) or any(x in p for x in exclude): continue
                try: total += os.stat(p).st_size
                except OSError: continue
                found.append(p)
                if total > spec['max_bytes']: break
            if total > spec['max_bytes']: break
        if total <= spec['max_bytes']: chosen.extend(found)
    paths.extend(chosen)
elif kind == 'roots':
    exclude = tuple(spec.get('exclude') or ())
    for g in spec.get('globs') or ():
        paths.extend(p for p in glob.glob(os.path.join(root, g)) if os.path.isfile(p) and not os.path.islink(p))
    for r in spec['roots']:
      for start in (sorted(glob.glob(os.path.join(root, r))) if any(c in r for c in '*?[') else [os.path.join(root, r)]):   # roots may be globs (learning/cycle-*)
        if os.path.isfile(start) and not os.path.islink(start):
            paths.append(start); continue
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
            for name in filenames:
                p = os.path.join(dirpath, name)
                if os.path.islink(p) or any(x in p for x in exclude): continue
                paths.append(p)
else:
    exclude = tuple(spec.get('exclude') or ())
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames:
            p = os.path.join(dirpath, name)
            if os.path.islink(p) or any(x in p for x in exclude): continue
            paths.append(p)
rows = []; unreadable = []
for p in sorted(paths):
    try:
        st = os.stat(p)
        if not os.path.isfile(p):
            unreadable.append(os.path.relpath(p, base)); continue   # sockets, fifos, devices: listed, never hashed or transferred
        h = digest(p)
    except OSError:
        unreadable.append(os.path.relpath(p, base)); continue   # root-owned artifacts (perf.data ...) are listed, never transferred
    rows.append(dict(path=os.path.relpath(p, base), bytes=st.st_size, mtime=st.st_mtime, sha256=h))
    st2 = os.stat(p)
    if (st2.st_size, st2.st_mtime) != (st.st_size, st.st_mtime): rows[-1]['unstable'] = True
sys.stdout.write(json.dumps(dict(rows=rows, skipped_meta_test=skipped, root=root, unreadable=unreadable)))
'''

REHASH_SCRIPT = r'''
import hashlib, json, os, sys
base = %(base)r; out = {}
for rel in %(paths)r:
    p = os.path.join(base, rel)
    try:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for piece in iter(lambda: f.read(1 << 20), b''): h.update(piece)
        out[rel] = h.hexdigest()
    except OSError:
        out[rel] = None
sys.stdout.write(json.dumps(out))
'''


def ssh_python(node, key, script, timeout):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-i', key, node, 'nice -n 19 python3 -'], capture_output=True, text=True, timeout=timeout, input=script)
    require(result.returncode == 0, 'node script failed: ' + result.stderr[-400:])
    return json.loads(result.stdout)


def census(node, key, spec):
    return ssh_python(node, key, CENSUS_SCRIPT % dict(spec=spec), timeout=7200)


def rehash_node(node, key, base, paths):
    out = {}
    for start in range(0, len(paths), 2000):
        out.update(ssh_python(node, key, REHASH_SCRIPT % dict(base=base, paths=paths[start:start + 2000]), timeout=3600))
    return out


def transfer(node, key, base, paths, destination, bwlimit):
    destination.mkdir(parents=True, exist_ok=True)
    listing = '\n'.join(paths) + '\n'
    # --chmod=Du+w: staged directories must stay writable so the atomic move out of .incoming works for read-only build trees (node-release builds/*)
    result = subprocess.run(['rsync', '-a', '--checksum', '--partial', '--chmod=Du+w', '--files-from=-', '--bwlimit=%d' % bwlimit, '-e', 'ssh -o BatchMode=yes -i ' + key,
                             '%s:%s/' % (node, base.rstrip('/') or '/'), str(destination) + '/'], input=listing, capture_output=True, text=True, timeout=6 * 3600)
    require(result.returncode == 0, 'rsync failed: ' + result.stderr[-400:])


def local_digest(path):
    return sha(path)


def mutable(rel):
    name = os.path.basename(rel)
    return name in MUTABLE_BASENAMES or '/repo/.git/refs/' in rel or '/repo/.git/logs/' in rel or '/repo/index/' in rel   # the champion index tree is regenerated by every sync


def pull(root, cls, node=NODE, key=KEY, bwlimit=8192, census_fn=census, transfer_fn=transfer, rehash_fn=rehash_node, tool_path=None):
    """Mirror one class; returns the receipt. Never deletes, never overwrites an immutable path with different bytes."""
    spec = SPEC[cls]
    root = Path(root)
    index_path = root / 'index' / (cls + '.json')
    index = read(index_path) if index_path.is_file() else {}
    started = time.time()
    snapshot = census_fn(node, key, spec)
    rows = [r for r in snapshot['rows'] if not r.get('unstable')]
    unstable = [r['path'] for r in snapshot['rows'] if r.get('unstable')]
    wanted, unchanged, drift = [], 0, []
    for r in rows:
        have = index.get(r['path'])
        target = root / cls / r['path']
        if have == r['sha256'] and (target.is_file() or spec.get('census_only')):
            unchanged += 1
            continue
        if have is not None and have != r['sha256'] and not mutable(r['path']):
            drift.append(dict(path=r['path'], mirrored=have, node=r['sha256']))
            continue
        wanted.append(r)
    incoming = root / ('.incoming-%s' % uuid.uuid4().hex[:12])
    pulled, transfer_mismatch, node_changed = [], [], []
    if spec.get('census_only'):
        for r in wanted:   # census-only: the index records the node digest without a local copy (receipt says so)
            index[r['path']] = r['sha256']
        wanted = []
    if wanted:
        transfer_fn(node, key, spec['base'], [r['path'] for r in wanted], incoming, bwlimit)
        verified_by_node = rehash_fn(node, key, spec['base'], [r['path'] for r in wanted])
        for r in wanted:
            source = incoming / r['path']
            if not source.is_file() or local_digest(source) != r['sha256']:
                transfer_mismatch.append(r['path'])
                continue
            if verified_by_node.get(r['path']) != r['sha256']:
                node_changed.append(r['path'])   # changed on the node during the pull: kept out of the index, re-pulled next run
                continue
            target = root / cls / r['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            index[r['path']] = r['sha256']
            pulled.append(r)
        for dirpath, dirnames, filenames in os.walk(incoming, topdown=False):   # leftovers (mismatches) are removed with the staging dir
            for name in filenames:
                os.unlink(os.path.join(dirpath, name))
            for name in dirnames:
                os.rmdir(os.path.join(dirpath, name))
        if incoming.exists():
            os.rmdir(incoming)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    atomic(index_path, index)
    newest = max((r['mtime'] for r in rows), default=None)
    receipt = dict(schema=SCHEMA, **{'class': cls}, node=node, base=spec['base'], root=spec['root'], at=time.time(), seconds=time.time() - started,
                   census=dict(files=len(rows), bytes=sum(r['bytes'] for r in rows), unstable=unstable, skipped_meta_test=snapshot.get('skipped_meta_test', 0), unreadable=snapshot.get('unreadable', [])),
                   pulled=len(pulled), pulled_bytes=sum(r['bytes'] for r in pulled), unchanged=unchanged, drift_refused=drift, transfer_mismatch=transfer_mismatch,
                   node_changed_during_pull=node_changed, node_rehash_verified=not node_changed and not transfer_mismatch, indexed=len(index), census_only=bool(spec.get('census_only')),
                   lag_seconds=(time.time() - newest) if newest else None, mirror_root=str(root / cls), tool_sha256=sha(tool_path or __file__), final_tests_opened=False)
    if cls == 'champion-git':
        receipt['git'] = verify_champion_git(root / cls / spec['root'])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = root / 'receipts' / 'mirror' / ('%s-%s-%s.json' % (stamp, cls, hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()[:12]))
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt['receipt_sha256'] = promotion_gate.immutable(out, receipt)
    receipt['receipt_path'] = str(out)
    return receipt


def verify_champion_git(mirror_dir):
    """fsck the mirrored .git and compare refs/heads/main with the head recorded in the newest node sync receipt."""
    repo = mirror_dir / 'repo'
    if not (repo / '.git').is_dir():
        return dict(status='no_repo')
    fsck = subprocess.run(['git', '-C', str(repo), 'fsck', '--full', '--connectivity-only'], capture_output=True, text=True, timeout=1800)
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'refs/heads/main'], capture_output=True, text=True, timeout=60).stdout.strip()
    receipts = sorted((mirror_dir / 'receipts').glob('champion-git-*.json')) if (mirror_dir / 'receipts').is_dir() else []
    expected = read(receipts[-1]).get('head') if receipts else None
    return dict(status='verified' if fsck.returncode == 0 and expected and head == expected else 'unverified', fsck_exit=fsck.returncode, fsck_tail=fsck.stderr[-300:] or fsck.stdout[-300:],
                head=head or None, receipt_head=expected, receipt=str(receipts[-1]) if receipts else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path('/home/ubuntu/jope-node-mirror'))   # real directory (symlink under the lane's materialized/)
    parser.add_argument('--class', dest='classes', action='append', choices=sorted(SPEC), required=True)
    parser.add_argument('--node', default=NODE)
    parser.add_argument('--key', default=KEY)
    parser.add_argument('--bwlimit', type=int, default=8192)
    args = parser.parse_args(argv)
    for cls in args.classes:
        receipt = pull(args.root, cls, args.node, args.key, args.bwlimit)
        summary = {k: receipt[k] for k in ('class', 'pulled', 'pulled_bytes', 'unchanged', 'drift_refused', 'transfer_mismatch', 'node_changed_during_pull', 'node_rehash_verified', 'seconds', 'receipt_path', 'census')}
        if 'git' in receipt:
            summary['git'] = receipt['git']
        print(json.dumps(summary, default=str))


if __name__ == '__main__':
    main()
