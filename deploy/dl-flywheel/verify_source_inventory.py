"""Additional static source/test classification check; never replaces jope-records.

The native Rust verify-test-inventory command remains authoritative for complete
baseline-triage semantics. This helper checks its exact test-bearing key rule,
roles/subsystems and the separate non-test research-source hash inventory.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROLES = {'regression', 'regression-real-corpus', 'deferred-classification', 'historical', 'research', 'infrastructure'}
REQUIRED = {'preparation', 'encoders', 'policy', 'gp', 'fitting', 'teaching', 'training', 'storage', 'export'}


def verify(repo):
    repo = Path(repo).resolve()
    coverage = json.loads((repo / 'deploy/ci/coverage.json').read_bytes())
    inventory = json.loads((repo / 'docs/internal/core-ip/findings/test-inventory.json').read_bytes())
    sources = json.loads((repo / 'deploy/dl-flywheel/RESEARCH_SOURCE_CLASSIFICATIONS.json').read_bytes())
    if inventory['schema'] != 'jope.regression-test-inventory/v1':
        raise ValueError('test inventory schema')
    expected = {key for key, value in coverage['items'].items()
                if value['discovery']['kind'] == 'python-family'
                or value['discovery']['kind'] == 'target' and value['discovery'].get('test')}
    actual = set(inventory['items'])
    if expected != actual:
        raise ValueError({'unclassified': sorted(expected - actual), 'stale': sorted(actual - expected)})
    if not REQUIRED <= set(inventory['subsystems']):
        raise ValueError('required subsystem definitions')
    covered = set()
    for key, value in inventory['items'].items():
        if value['role'] not in ROLES:
            raise ValueError('unknown test role: ' + key)
        if value['role'] in ('regression', 'regression-real-corpus'):
            names = value.get('subsystems', [])
            if not names or len(set(names)) != len(names) or not set(names) <= set(inventory['subsystems']):
                raise ValueError('regression subsystems: ' + key)
            if value['role'] == 'regression' and any(
                lane != 'rust-check' and coverage['lanes'][lane]['stage'] == 'execution'
                for lane in coverage['items'][key]['lanes']
            ):
                covered.update(names)
        elif value.get('subsystems') or not value.get('reason'):
            raise ValueError('non-regression reason: ' + key)
    if not REQUIRED <= covered:
        raise ValueError('missing executing regression subsystem')
    if sources['schema'] != 'jope.flywheel-research-source-classification/v1':
        raise ValueError('research source schema')
    for key, value in sources['items'].items():
        if key in actual or not key.startswith('python:') or value['classification']['role'] != 'research':
            raise ValueError('source is not a distinct non-test research item: ' + key)
        relative = Path(key[len('python:'):])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('source path confinement')
        path = repo / relative
        if not path.resolve().is_relative_to(repo) or path.is_symlink():
            raise ValueError('source path confinement')
        raw = path.read_bytes()
        if len(raw) != value['bytes'] or hashlib.sha256(raw).hexdigest() != value['sha256']:
            raise ValueError('research source hash: ' + key)
        if not value['classification'].get('reason'):
            raise ValueError('research source reason')
    return {'schema': 'jope.flywheel-source-classification-check/v1',
            'state': 'passed', 'test_items': len(actual), 'research_sources': len(sources['items']),
            'scope': 'Static key/role/source-hash validation only; no tests executed. Native jope-records baseline-triage validation remains required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo), sort_keys=True))


if __name__ == '__main__':
    main()
