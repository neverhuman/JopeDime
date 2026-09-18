#!/usr/bin/env python3
"""Loop preflight census (plan v3 step 0.1): one read-only pass over the node's loop state, as an immutable receipt.

The collector is a pure function over a `Tree` (a tiny filesystem view), so it runs unchanged on the node (this file's
own text is fed to `python3 -` over ssh, with LOOP_PREFLIGHT_REMOTE=1 so `main()` does not run there) and in tests
over an in-memory tree. It reads: runs/gp-service.json, <run>/health/latest.json (gp block), <run>/learning/state.json
(loaded once; it is ~31 MB), <run>/learning/control.json, <run>/successor/decisions/f11-rollback-status.json, and for
every frozen generation without quality-report.json the per-job classification of its jobs.json outputs
(completed / pending_no_output / partial / quarantined) with each pending output's position in runs/gp-priority-jobs.json
(newest generation first). Nothing is written on the node; the receipt (parallel.loop_preflight.v1) is immutable.

  loop_preflight.py --output parallel/recovery/eight-a100-20260912/review/loop-slice-v1/preflight-<utc>.json
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'parallel.loop_preflight.v1'
GP_SERVICE_FIELDS = ('phase', 'active', 'queued', 'retry_pending', 'ordinary_ready', 'ordinary_active', 'continuation_ready', 'continuation_active',
                     'dispatch_policy', 'executable_sha256', 'manifest_sha256', 'completed_views', 'failed_views', 'evaluations_per_second')
STATE_FIELDS = ('manifest_sha256', 'learning_checkpoint', 'learning_corpus', 'serving_checkpoint', 'serving_status', 'completed_blocks',
                'gp_replenishing', 'training_diagnostic', 'training_growth', 'candidate_checkpoint')
CLASSES = ('completed', 'pending_no_output', 'partial', 'quarantined')


class Tree:
    """Filesystem view over a root: paths are root-relative strings."""

    def exists(self, path):
        raise NotImplementedError

    def is_dir(self, path):
        raise NotImplementedError

    def read_text(self, path):
        raise NotImplementedError

    def read_json(self, path):
        return json.loads(self.read_text(path))

    def sha256(self, path):
        raise NotImplementedError

    def listdir(self, path):
        raise NotImplementedError


class FsTree(Tree):
    def __init__(self, root):
        self.root = Path(root)

    def exists(self, path):
        return (self.root / path).exists()

    def is_dir(self, path):
        return (self.root / path).is_dir()

    def read_text(self, path):
        return (self.root / path).read_text()

    def sha256(self, path):
        digest = hashlib.sha256()
        with (self.root / path).open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        return digest.hexdigest()

    def listdir(self, path):
        return sorted(os.listdir(self.root / path))


class MemoryTree(Tree):
    """In-memory tree: {path: text}; directories are implied by their files."""

    def __init__(self, files):
        self.files = {k.strip('/'): (v if isinstance(v, str) else json.dumps(v)) for k, v in files.items()}

    def exists(self, path):
        path = path.strip('/')
        return path in self.files or self.is_dir(path)

    def is_dir(self, path):
        prefix = path.strip('/') + '/'
        return any(k.startswith(prefix) for k in self.files)

    def read_text(self, path):
        return self.files[path.strip('/')]

    def sha256(self, path):
        return hashlib.sha256(self.files[path.strip('/')].encode()).hexdigest()

    def listdir(self, path):
        prefix = path.strip('/') + '/'
        return sorted({k[len(prefix):].split('/', 1)[0] for k in self.files if k.startswith(prefix)})


def classify_output(tree, output):
    """One GP job output directory -> completed | quarantined | partial | pending_no_output."""
    if tree.exists(output + '/search.json'):
        return 'completed'
    if not tree.is_dir(output):
        return 'pending_no_output'
    if tree.exists(output + '/service-quarantine.json'):
        return 'quarantined'
    return 'partial'


def queue_positions(tree):
    """output -> index in runs/gp-priority-jobs.json (the file lists the newest generation first)."""
    if not tree.exists('runs/gp-priority-jobs.json'):
        return {}, 0
    queue = tree.read_json('runs/gp-priority-jobs.json')
    jobs = queue['jobs'] if isinstance(queue, dict) else queue
    return {job.get('output'): index for index, job in enumerate(jobs) if isinstance(job, dict)}, len(jobs)


def generation_census(tree, run, generation, positions):
    path = generation.get('path') or ''
    jobs_path = path + '/jobs.json'
    row = dict(number=generation.get('number'), path=path, stage=generation.get('stage'), checkpoint_sha256=generation.get('checkpoint_sha256'),
               report=tree.exists(path + '/quality-report.json'), jobs=None, counts={c: 0 for c in CLASSES}, pending=[])
    if row['report'] or not tree.exists(jobs_path):
        return row
    doc = tree.read_json(jobs_path)
    jobs = doc['jobs'] if isinstance(doc, dict) else doc
    row['jobs'] = len(jobs)
    for job in jobs:
        output = job.get('output')
        kind = classify_output(tree, output)
        row['counts'][kind] += 1
        if kind != 'completed':
            row['pending'].append(dict(output=output.split('/' + os.path.basename(path.rstrip('/')) + '/')[-1] if path else output, kind=kind,
                                       queue_index=positions.get(output), arm=output.split('/')[-4] if output.count('/') >= 4 else None))
    return row


def collect(tree, run):
    """Pure census over a Tree; returns the receipt body (without schema/timestamp)."""
    out = dict(run=run, errors=[])
    try:
        service = tree.read_json('runs/gp-service.json')
        out['gp_service'] = {k: service.get(k) for k in GP_SERVICE_FIELDS}
    except Exception as error:  # noqa: BLE001 - a census records what it cannot read
        out['gp_service'] = None
        out['errors'].append('gp-service.json: %s' % error)
    try:
        health = tree.read_json(run + '/health/latest.json')
        gp = health.get('gp') or {}
        out['health'] = dict(timestamp=health.get('timestamp'), gp={k: v for k, v in gp.items() if not isinstance(v, (dict, list))},
                             active_rows=len(gp.get('active_rows') or []), gpu=[g.get('utilization') for g in (health.get('gpu') or []) if isinstance(g, dict)])
    except Exception as error:  # noqa: BLE001
        out['health'] = None
        out['errors'].append('health/latest.json: %s' % error)
    state = tree.read_json(run + '/learning/state.json')
    learning = state.get('learning_checkpoint')
    out['state'] = {k: state.get(k) for k in STATE_FIELDS}
    out['state']['generations'] = len(state.get('generations') or [])
    out['state']['learning_checkpoint_state_sha256'] = tree.sha256(learning + '/state.json') if learning and tree.exists(learning + '/state.json') else None
    out['control'] = None
    if tree.exists(run + '/learning/control.json'):
        control = tree.read_json(run + '/learning/control.json')
        out['control'] = dict(paused=control.get('paused'), manifest_sha256=control.get('manifest_sha256'), keys=sorted(control))
    out['rollback_status'] = tree.read_json(run + '/successor/decisions/f11-rollback-status.json') if tree.exists(run + '/successor/decisions/f11-rollback-status.json') else None
    if isinstance(out['rollback_status'], dict):
        out['rollback_status'] = {k: out['rollback_status'].get(k) for k in ('status', 'learning_checkpoint', 'manifest', 'deviations', 'revisions_after_verdict')}
    positions, queue_length = queue_positions(tree)
    out['queue'] = dict(entries=queue_length)
    out['generations'] = [generation_census(tree, run, g, positions) for g in state.get('generations') or []]
    out['incomplete'] = [g for g in out['generations'] if not g['report'] and g['jobs']]
    out['totals'] = {c: sum(g['counts'][c] for g in out['incomplete']) for c in CLASSES}
    return out


def remote_script(root, run):
    """This module's own text plus a trailer; fed to `python3 -` on the node (LOOP_PREFLIGHT_REMOTE=1 disables main)."""
    source = Path(__file__).read_text()
    return source + '\nimport json as _j\nprint(_j.dumps(collect(FsTree(%r), %r), sort_keys=True))\n' % (root, run)


def collect_remote(node, key, root, run, timeout=900):
    import subprocess
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-i', key, node, 'LOOP_PREFLIGHT_REMOTE=1 python3 -'],
                            capture_output=True, text=True, timeout=timeout, input=remote_script(root, run))
    if result.returncode != 0:
        raise RuntimeError('remote census failed: ' + result.stderr.strip()[-600:])
    return json.loads(result.stdout)


def summary(body):
    gens = [dict(number=g['number'], jobs=g['jobs'], **g['counts']) for g in body['incomplete']]
    return dict(gp=body.get('gp_service') and {k: body['gp_service'].get(k) for k in ('phase', 'active', 'queued', 'ordinary_ready', 'continuation_ready', 'executable_sha256')},
                state={k: body['state'].get(k) for k in ('manifest_sha256', 'learning_checkpoint', 'learning_checkpoint_state_sha256', 'generations', 'gp_replenishing')},
                control=body.get('control'), rollback=(body.get('rollback_status') or {}).get('status'), queue=body['queue'], incomplete=gens, totals=body['totals'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--node', default='ubuntu@158.101.20.226')
    parser.add_argument('--key', default=str(Path.home() / '.ssh/lambda_cloud_a100_20260907_ed25519'))
    parser.add_argument('--node-root', default='/home/ubuntu/jope-dime/parallel')
    parser.add_argument('--run', default='runs/eight-a100-recovery-20260912')
    parser.add_argument('--local-root', type=Path, help='census a local tree instead of the node (tests, mirrors)')
    parser.add_argument('--output', type=Path, help='immutable receipt path (parallel.loop_preflight.v1)')
    args = parser.parse_args(argv)
    body = collect(FsTree(args.local_root), args.run) if args.local_root else collect_remote(args.node, args.key, args.node_root, args.run)
    receipt = dict(schema=SCHEMA, collected_at=datetime.datetime.now(datetime.UTC).isoformat(), node=None if args.local_root else args.node,
                   node_root=str(args.local_root) if args.local_root else args.node_root, final_tests_opened=False, **body)
    if args.output:
        import promotion_gate
        args.output.parent.mkdir(parents=True, exist_ok=True)
        receipt = dict(receipt, receipt_sha256=promotion_gate.immutable(args.output, receipt))
    print(json.dumps(dict(summary(body), receipt=str(args.output) if args.output else None, receipt_sha256=receipt.get('receipt_sha256')), sort_keys=True))
    return receipt


if __name__ == '__main__' and not os.environ.get('LOOP_PREFLIGHT_REMOTE'):
    main()
