#!/usr/bin/env python3
"""Mirror an immutable work bundle to an unused namespace; never activate it."""
import argparse
import json
import shlex
import subprocess
import uuid
from pathlib import Path

from kernel_bundle import verify_bundle
from kernel_release import digest
from kernel_remote_state import compare, compare_health

REMOTE_ROOT = '/home/ubuntu/jope-dime/parallel'


def mirror(bundle, host, key, receipt_dir):
    bundle = Path(bundle).resolve()
    receipt = verify_bundle(bundle)
    receipt_dir = Path(receipt_dir).resolve()
    receipt_dir.mkdir(parents=True, exist_ok=False)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-i', str(Path(key).expanduser()), host]
    root = REMOTE_ROOT
    final = root + '/kernel-work/' + receipt['bundle_id']
    incoming = root + '/kernel-work/.incoming-' + uuid.uuid4().hex
    capture_source = (Path(__file__).parent / 'kernel_remote_state.py').read_text()

    def python(source):
        return json.loads(subprocess.check_output(ssh + ['python3 -'], input=source.encode()))

    def state():
        return python(capture_source + '\nprint(json.dumps(capture(' + repr(root) + ')))\n')

    before = state()
    (receipt_dir / 'before.json').write_text(json.dumps(before, indent=2) + '\n')
    try:
        prepare = python('from pathlib import Path\nimport json\n' +
            f'final=Path({final!r}); incoming=Path({incoming!r})\n' +
            'for p in [final.parent, *final.parent.parents]:\n'
            ' if p.is_symlink(): raise ValueError("Remote namespace contains symlink")\n' +
            'if not final.exists(): incoming.mkdir(parents=True, exist_ok=False)\n' +
            'print(json.dumps({"exists":final.exists()}))\n')
        if not prepare['exists']:
            subprocess.run(['rsync', '-a', '--checksum', '--bwlimit=4096',
                            '-e', shlex.join(ssh[:-1]), str(bundle) + '/', host + ':' + incoming + '/'], check=True)
        publication_source = (Path(__file__).parent / 'kernel_remote_publish.py').read_text()
        publication = python(publication_source + '\nprint(json.dumps(publish(' +
                             repr(incoming) + ', ' + repr(final) + ', ' + repr(receipt) + ')))\n')
    except Exception as error:
        after = state()
        (receipt_dir / 'after.json').write_text(json.dumps(after, indent=2) + '\n')
        failure = {'schema': 'kernel.remote-mirror.v1', 'status': 'failed',
                   'reason': str(error), 'activation_allowed': False, **compare(before, after),
                   'operational_health': compare_health(before, after)}
        (receipt_dir / 'mirror.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise
    after = state()
    (receipt_dir / 'after.json').write_text(json.dumps(after, indent=2) + '\n')
    preservation = compare(before, after)
    result = {'schema': 'kernel.remote-mirror.v1', **publication, **preservation,
              'host': host, 'qualified_for_deployment': False,
              'operational_health': compare_health(before, after)}
    (receipt_dir / 'mirror.json').write_text(json.dumps(result, indent=2) + '\n')
    # Keep preservation evidence on both machines, outside the immutable bundle.
    remote_evidence = root + '/kernel-work-evidence/' + receipt['bundle_id'] + '/' + receipt_dir.name
    python('from pathlib import Path\nimport json\n' +
           f'p=Path({remote_evidence!r})\n' +
           'for part in [p, *p.parents]:\n'
           ' if part.is_symlink(): raise ValueError("Evidence symlink rejected")\n'
           'p.mkdir(parents=True, exist_ok=False)\nprint(json.dumps({"created":True}))\n')
    subprocess.run(['rsync', '-a', '--checksum', '-e', shlex.join(ssh[:-1]),
                    str(receipt_dir) + '/', host + ':' + remote_evidence + '/'], check=True)
    expected = {p.name: digest(p) for p in receipt_dir.iterdir() if p.is_file()}
    python('from pathlib import Path\nimport json,hashlib\n' +
           f'p=Path({remote_evidence!r}); expected={expected!r}\n' +
           'actual={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}\n'
           'if actual != expected: raise ValueError("Mirrored evidence differs")\n'
           'print(json.dumps({"verified":True}))\n')
    if not preservation['stable_identities_unchanged']:
        raise ValueError('Live drift detected; inspect receipt. No automatic reversal of other work.')
    if result['operational_health']['status'] != 'passed':
        raise ValueError('Remote health was not verified; inspect the mirrored receipt. No runtime was activated.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--host', default='ubuntu@158.101.20.226')
    parser.add_argument('--key', default='~/.ssh/lambda_cloud_a100_20260907_ed25519')
    parser.add_argument('--receipts', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(mirror(args.bundle, args.host, args.key, args.receipts), indent=2))
