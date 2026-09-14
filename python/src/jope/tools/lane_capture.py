#!/usr/bin/env python3
"""Import a verified node-mirror class into the jain-datasets lane Store as captures (explicit one-shot administration).

Every capture unit (a prepared view, a quant dataset, one GP cut result, one champion identity, one node-only source root)
becomes one lane `Request`: evaluation = '<class>:<node-relative unit path>', source_folder = the node path, observation =
the unit's census rows (path, bytes, sha256 from the both-host verified mirror index), artifacts = every file of the unit
with its verified sha256 (role = path inside the unit). The Rust lane binary does the durable work: reserve -> capture ->
send | receive (loopback, xbabe2 is the receiver) -> acknowledge -> verify. Idempotent: an evaluation already bound in the
Store is skipped. Receipt parallel.lane_capture_receipt.v1 under <mirror>/receipts/lane/.

  lane_capture.py --class prepared [--store ~/jain-split/jain-datasets/.staging/a100-store] [--limit N] [--dry-run]
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import promotion_gate
from resume_expired_cycle import atomic, read, sha

SCHEMA = 'parallel.lane_capture_receipt.v1'
MIRROR = Path('/home/ubuntu/jope-node-mirror')   # real directory (symlink under the lane's materialized/)
STORE = Path('/home/ubuntu/jain-split/jain-datasets/.staging/a100-store')
BIN_DIR = MIRROR / 'bin'
NODE_ROOT = '/home/ubuntu/jope-dime/parallel'
# class -> (node-relative prefix the unit path is taken under, number of path components that form one unit; None = whole class)
UNITS = {
    'prepared': ('data/prepared/', 2),                       # data/prepared/<set>/<view> (cut-K subdirs inside)
    'dataset': ('home/ubuntu/quant/', 2),                     # quant/<category>/<dataset>
    'results': ('runs/eight-a100-recovery-20260912/learning/', 4),   # learning/generation-N/<arm>/<dataset>/cut-K (search/ inside)
    'observations': ('runs/eight-a100-recovery-20260912/learning/', 4),
    'champions': ('runs/champions/', 1),                      # champions/<identity>
    'contexts': ('runs/dataset-inclusion-20260909/correction-final/', 1),
    'teaching-corpus': ('runs/dataset-inclusion-20260909/teaching-corpus/', 1),
    'node-source': ('', 1),
    'node-home': ('', 1),
    'node-release': ('', 2),
    'kernel-work': ('', 2),
    'corpus': ('corpus/', 1),
    'assets': ('assets/', 1),
    'receipts': ('runs/eight-a100-recovery-20260912/', 2),
    # xbabe2-only: the >5 MB untracked binaries of the jope-dime checkout, indexed by the committed inventory (not in git by design)
    'xbabe2-binaries': ('', 5),
}
INVENTORY = Path('/home/ubuntu/jope-dime/parallel/recovery/eight-a100-20260912/inventory/xbabe2-untracked-binaries-20260914.json')


def binary():
    sums = (BIN_DIR / 'SHA256SUMS').read_text().split()
    names = [n for n in sums if n.startswith('jain-datasets-')]
    path = max((BIN_DIR / n for n in names), key=lambda q: q.stat().st_mtime)   # newest pinned build
    check = subprocess.run(['sha256sum', '--check', '--ignore-missing', '--quiet', 'SHA256SUMS'], cwd=BIN_DIR, capture_output=True, text=True)
    if check.returncode != 0:
        raise RuntimeError('lane binary digest mismatch: ' + check.stderr[-200:])
    return path


SHARDS = {'champions': 2}   # hash-named identity dirs are grouped into shard-<2 hex> units (256 instead of ~12,000)


def unit_folder(unit):
    parent, _, leaf = unit.rpartition('/')
    return parent if leaf.startswith('shard-') and parent else unit


def units(cls, index):
    prefix, depth = UNITS[cls]
    groups = {}
    for rel, digest in index.items():
        if prefix and not rel.startswith(prefix):
            continue
        tail = rel[len(prefix):]
        parts = tail.split('/')
        if cls in SHARDS and len(parts) > 1:
            unit = 'shard-' + parts[0][:SHARDS[cls]]
        else:
            unit = '/'.join(parts[:depth]) if len(parts) > depth else ('/'.join(parts[:-1]) or parts[0])
        groups.setdefault(prefix + unit, []).append((rel, digest))
    return groups


def run(bin_path, *args, stdin=None, stdout=None):
    result = subprocess.run([str(bin_path), *args], input=stdin, stdout=stdout or subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError('%s failed: %s' % (args[0], (result.stderr or b'')[-400:].decode('utf-8', 'replace')))
    return result.stdout


def bound(store, evaluation):
    return (store / 'evaluations' / hashlib.sha256(evaluation.encode()).hexdigest()).exists()


def capture_unit(bin_path, store, cls, unit, files, mirror_root, base, scratch):
    evaluation = '%s:%s' % (cls, unit)
    if bound(store, evaluation):
        return dict(evaluation=evaluation, status='already_bound')
    rows = []
    artifacts = []
    total = 0
    for rel, digest in sorted(files):
        local = mirror_root / rel
        st = os.stat(local)
        rows.append(dict(path=rel, bytes=st.st_size, sha256=digest))
        folder = unit_folder(unit)
        artifacts.append(dict(role=rel[len(folder) + 1:] if rel.startswith(folder + '/') else rel, path=str(local), sha256=digest))
        total += st.st_size
    observation = json.dumps(dict(schema='parallel.lane_capture_observation.v1', **{'class': cls}, unit=unit, node_base=base, files=rows), sort_keys=True, separators=(',', ':')).encode()
    if len(observation) > 64 << 20:
        return dict(evaluation=evaluation, status='observation_too_large', files=len(rows))
    request = dict(evaluation=evaluation, source_folder=os.path.join(base, unit_folder(unit)), observation=list(observation), artifacts=artifacts)
    req_path = scratch / 'request.json'
    atomic(req_path, request)
    run(bin_path, 'reserve', str(store), evaluation, str(total + len(observation) + 4096))
    envelope = json.loads(run(bin_path, 'capture', str(store), str(req_path)))
    env_path = scratch / 'envelope.json'
    atomic(env_path, envelope)
    frame = run(bin_path, 'send', str(store), envelope['producer'], str(envelope['sequence']))
    receipt = json.loads(run(bin_path, 'receive', str(store), '--authorized-producer', envelope['producer'], stdin=frame))
    rcpt_path = scratch / 'receipt.json'
    atomic(rcpt_path, receipt)
    run(bin_path, 'acknowledge', str(store), str(env_path), str(rcpt_path))
    verified = json.loads(run(bin_path, 'verify', str(store), str(env_path)))
    return dict(evaluation=evaluation, status='captured', producer=envelope['producer'], sequence=envelope['sequence'], request_sha256=envelope['request_sha256'],
                files=len(rows), bytes=total, verified=verified.get('verified') is True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--class', dest='cls', choices=sorted(UNITS), required=True)
    parser.add_argument('--mirror', type=Path, default=MIRROR)
    parser.add_argument('--store', type=Path, default=STORE)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    from node_mirror_pull import SPEC
    if args.cls == 'xbabe2-binaries':
        base = '/home/ubuntu/jope-dime'
        inventory = read(INVENTORY)
        index = {row['path']: row['sha256'] for row in inventory['files']}
        mirror_root = Path(base)
        index_path = INVENTORY
    else:
        base = SPEC[args.cls]['base']
        index_path = args.mirror / 'index' / (args.cls + '.json')
        index = read(index_path)
        mirror_root = args.mirror / args.cls
    groups = units(args.cls, index)
    if args.dry_run:
        print(json.dumps({'class': args.cls, 'units': len(groups), 'files': sum(len(v) for v in groups.values()), 'sample': sorted(groups)[:3]}))
        return 0
    bin_path = binary()
    args.store.mkdir(parents=True, exist_ok=True)
    scratch = args.mirror / '.lane-scratch' / args.cls
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    for n, unit in enumerate(sorted(groups)):
        if args.limit is not None and n >= args.limit:
            break
        try:
            results.append(capture_unit(bin_path, args.store, args.cls, unit, groups[unit], mirror_root, base, scratch))
        except Exception as error:   # the lane refuses loudly; record and continue with the next unit
            results.append(dict(evaluation='%s:%s' % (args.cls, unit), status='failed', error=str(error)[-300:]))
    counts = {}
    for r in results:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    receipt = dict(schema=SCHEMA, **{'class': args.cls}, store=str(args.store), binary=str(bin_path), binary_sha256=sha(bin_path), mirror_index_sha256=sha(index_path),
                   at=time.time(), seconds=time.time() - started, units=len(groups), counts=counts, captured_bytes=sum(r.get('bytes', 0) for r in results if r['status'] == 'captured'),
                   results=results, final_tests_opened=False)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.mirror / 'receipts' / 'lane' / ('%s-%s.json' % (stamp, args.cls))
    out.parent.mkdir(parents=True, exist_ok=True)
    promotion_gate.immutable(out, receipt)
    print(json.dumps(dict(receipt=str(out), units=len(groups), counts=counts, seconds=round(receipt['seconds'], 1))))
    return 0 if not counts.get('failed') else 1


if __name__ == '__main__':
    sys.exit(main())
