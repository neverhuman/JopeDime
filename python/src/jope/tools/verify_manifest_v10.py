#!/usr/bin/env python3
"""Verify a manifest v10 revision against the review packets it must bind (plan v3 step 1.3; research-lane one-shot).

Checks (each pass / fail / pending, never null-as-ok): parent chain to the re-parent manifest, training still disabled,
preregistration block digest, amendment digests (PROPOSAL-RNG.md, ACTIVATION-AMENDMENT.md), generation-seeded proposal
protocol, neural/GP executable bindings against the expected build digests, oldest-generation-first dispatch policy,
growth gate v2, the pre-registered diagnostic hold of 3, sequential world-8 population geometry. Writes an immutable
receipt parallel.manifest_v10_verification.v1.

  verify_manifest_v10.py --manifest PATH --preregistration PATH --amendment PATH ... [--parent-sha SHA] [--neural-sha SHA] [--gp-sha SHA] --output DIR
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import flywheel_verdict
import promotion_gate
from resume_expired_cycle import read, sha

SCHEMA = 'parallel.manifest_v10_verification.v1'
PROTOCOL = 'generation_seeded_v1'
ORDER = 'oldest_generation_first_v1'
GEOMETRY = 'sequential_world8'


def check(name, status, value=None, expected=None, note=None):
    return dict(name=name, status=status, value=value, expected=expected, note=note)


def verdict(ok):
    return 'pass' if ok else 'fail'


def parent_sha(manifest):
    parent = manifest.get('parent_manifest')
    if isinstance(parent, dict):
        return parent.get('sha256')
    if isinstance(parent, str) and len(parent) == 64:
        return parent
    return None


def verify(manifest, preregistration, amendments, parent=None, neural_sha=None, gp_sha=None):
    """Pure: manifest dict, preregistration read-out (flywheel_verdict.preregistration), amendments {path: sha256}."""
    checks = []
    found_parent = parent_sha(manifest)
    checks.append(check('parent_chain', verdict(found_parent == parent) if parent else 'pending', found_parent, parent, None if parent else 'no expected parent supplied'))
    checks.append(check('training_disabled', verdict(manifest.get('automatic_training') is False), manifest.get('automatic_training'), False))
    pre = manifest.get('preregistration_sha256')
    checks.append(check('preregistration_digest', verdict(pre in (preregistration['sha256'], preregistration.get('block_bytes_sha256'))), pre, preregistration['sha256'], preregistration.get('digest_rule')))
    bound = manifest.get('preregistration_amendments') or []
    bound_shas = {a.get('sha256') if isinstance(a, dict) else a for a in bound}
    missing = {p: s for p, s in amendments.items() if s not in bound_shas}
    checks.append(check('amendments_bound', verdict(not missing and bool(amendments)), sorted(bound_shas), amendments, 'missing: %s' % sorted(missing) if missing else None))
    checks.append(check('proposal_protocol', verdict(manifest.get('proposal_generation_protocol') == PROTOCOL), manifest.get('proposal_generation_protocol'), PROTOCOL))
    bindings = manifest.get('bindings') or {}
    neural = bindings.get(manifest.get('neural_executable'))
    checks.append(check('neural_binding', verdict(neural == neural_sha) if neural_sha else 'pending', neural, neural_sha, None if neural_sha else 'expected proposal-compat-v2 digest not supplied'))
    gp = bindings.get(manifest.get('gp_service_executable'))
    checks.append(check('gp_binding', verdict(gp == gp_sha) if gp_sha else 'pending', gp, gp_sha, None if gp_sha else 'expected gp-dispatch-order-v2 digest not supplied'))
    order = (manifest.get('gp_dispatch_policy') or {}).get('ordinary_order')
    checks.append(check('dispatch_order', verdict(order == ORDER), order, ORDER))
    growth = manifest.get('training_growth_gate') or {}
    checks.append(check('growth_gate_v2', verdict(growth.get('minimum_new_complete_generations') == 1 and (growth.get('minimum_new_verified_searches') or 0) <= 300),
                        dict(minimum_new_complete_generations=growth.get('minimum_new_complete_generations'), minimum_new_verified_searches=growth.get('minimum_new_verified_searches')),
                        dict(minimum_new_complete_generations=1, minimum_new_verified_searches='<= 300')))
    guard = manifest.get('training_diagnostic_guard') or {}
    checks.append(check('diagnostic_hold_3', verdict(guard.get('consecutive_rejections') == 3), guard.get('consecutive_rejections'), 3))
    arms = manifest.get('population_arms') or {}
    checks.append(check('population_geometry', verdict(arms.get('geometry') == GEOMETRY and arms.get('k') in (1, 2, 8)), arms, dict(geometry=GEOMETRY, k='1|2|8')))
    checks.append(check('release_revision', verdict(bool(manifest.get('release_revision'))), manifest.get('release_revision'), 'present'))
    failing = [c['name'] for c in checks if c['status'] == 'fail']
    pending = [c['name'] for c in checks if c['status'] == 'pending']
    return dict(schema=SCHEMA, verdict='verified' if not failing and not pending else 'not_verified', failing=failing, pending=pending, checks=checks, final_tests_opened=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, action='append', default=[])
    parser.add_argument('--parent-sha')
    parser.add_argument('--neural-sha')
    parser.add_argument('--gp-sha')
    parser.add_argument('--output', type=Path, required=True, help='directory for the immutable receipt')
    args = parser.parse_args(argv)
    manifest = read(args.manifest)
    receipt = verify(manifest, flywheel_verdict.preregistration(args.preregistration), {str(p): sha(p) for p in args.amendment}, args.parent_sha, args.neural_sha, args.gp_sha)
    receipt.update(manifest_path=str(args.manifest), manifest_sha256=sha(args.manifest), at=datetime.datetime.now(datetime.timezone.utc).timestamp())
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.output / ('manifest-v10-verification-%s.json' % stamp)
    args.output.mkdir(parents=True, exist_ok=True)
    promotion_gate.immutable(out, receipt)
    print(json.dumps(dict(verdict=receipt['verdict'], failing=receipt['failing'], pending=receipt['pending'], receipt=str(out))))
    return 0 if receipt['verdict'] == 'verified' else 1


if __name__ == '__main__':
    sys.exit(main())
