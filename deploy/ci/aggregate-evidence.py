#!/usr/bin/env python3
"""Verify run-scoped evidence before it enters the CI aggregate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def _receipt(root, pattern="receipt.json"):
    paths = list(root.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern}, found {len(paths)}")
    return paths[0]


def verify_publication(root, source_sha, checkout_sha, run_id, attempt, event):
    path = _receipt(root)
    receipt = json.loads(path.read_text())
    assert receipt['schema'] == 'jope.ci-receipt/v1' and receipt['lane'] == 'paper'
    assert receipt['source_sha'] == source_sha
    assert receipt['checkout_sha'] == checkout_sha
    assert receipt['run_id'] == run_id and type(receipt['attempt']) is int and 1 <= receipt['attempt'] <= int(attempt)
    assert receipt['event'] == event
    assert receipt['status'] == 'passed' and receipt['exit_status'] == 0 and not receipt.get('diagnostic')
    assert receipt['dirty_source'] == receipt['dirty_source_after'] == []
    assert receipt['command'] == ['make', '-C', 'docs/paper/innovationzero', 'check', 'self-test']
    required = {'command.log', 'evidence/source-identity.json'}
    assert required <= set(receipt['evidence'])
    actual = {str(member.relative_to(path.parent)) for member in path.parent.rglob('*') if member.is_file() and member != path}
    assert actual == set(receipt['evidence'])
    for name, digest in receipt['evidence'].items():
        member = path.parent / name
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert not member.is_symlink() and member.resolve().is_relative_to(path.parent.resolve())
        assert hashlib.sha256(member.read_bytes()).hexdigest() == digest
    identity = json.loads((path.parent / 'evidence/source-identity.json').read_text())
    assert identity['schema'] == 'jope.publication-source/v1'
    assert identity['source_sha'] == receipt['source_sha']
    assert identity['checkout_sha'] == identity['event_sha'] == receipt['checkout_sha']
    assert identity['run_id'] == receipt['run_id'] and identity['attempt'] == receipt['attempt'] and identity['event'] == receipt['event']
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert identity['source_tree'] == identity['checkout_tree'] == tree
    print('Exact-head publication source checks and refusal checks retained; PDF rendering is separate')


def verify_frontend(root, source_sha, run_id, attempt):
    receipt = json.loads(_receipt(root).read_text())
    assert receipt['schema'] == 'jope.observatory-ui-ci/v1'
    assert receipt['source_sha'] == source_sha
    assert receipt['run_id'] == run_id and isinstance(receipt['attempt'], str) and receipt['attempt'].isdecimal() and 1 <= int(receipt['attempt']) <= int(attempt)
    assert receipt['executed'] > 0 and receipt['passed'] == receipt['executed'] and receipt['failed'] == 0
    print('Exact-head frontend tests:', receipt['executed'], 'passed')


def verify_rendered(root, source_sha, run_id, attempt, event_sha):
    rendered = _receipt(root, 'rendered-receipt.json').parent
    ui_attempt = json.loads(_receipt(root).read_text())['attempt']
    assert isinstance(ui_attempt, str) and ui_attempt.isdecimal() and 1 <= int(ui_attempt) <= int(attempt)
    subprocess.run([
        'python3', 'deploy/ci/rendered-ux-receipt.py', 'verify', '--root', str(rendered),
        '--source-sha', source_sha, '--run-id', run_id, '--attempt', ui_attempt,
        '--checkout-sha', source_sha, '--event-sha', event_sha,
    ], check=True)


def verify_export_parity(root, source_sha, run_id, attempt, repository, event):
    path = _receipt(root, 'report.json')
    report = json.loads(path.read_text())
    assert report['schema'] == 'jope.export-parity/v1' and report['status'] == 'pass' and report['passed'] is True
    assert report['ci_executed'] is True and report['source_stable'] is True and not report['source_dirty']
    assert report['source_sha'] == report['source_end'] == source_sha
    assert report['harness_sha256'] == hashlib.sha256(Path('deploy/ci/export-parity.sh').read_bytes()).hexdigest()
    assert report['cargo_lock_sha256'] == hashlib.sha256(Path('Cargo.lock').read_bytes()).hexdigest()
    context = report['ci_context']
    assert context['GITHUB_REPOSITORY'] == repository == 'neverhuman/JopeDime'
    assert context['GITHUB_EVENT_NAME'] == event == 'workflow_dispatch'
    assert context['GITHUB_RUN_ID'] == run_id
    assert context['GITHUB_RUN_ATTEMPT'] == attempt
    assert report['fault_injected'] is False
    assert report['toolchains']['rustc'].startswith('rustc 1.97.1 ')
    assert report['toolchains']['python'] == 'Python 3.12.3' and report['toolchains']['node'] == 'v26.1.0'
    registry_path = path.parent / 'registry.json'
    fixtures_path = path.parent / 'fixtures.json'
    registry = json.loads(registry_path.read_text())
    assert hashlib.sha256(registry_path.read_bytes()).hexdigest() == report['registry_sha256']
    assert hashlib.sha256(fixtures_path.read_bytes()).hexdigest() == report['fixture_sha256']
    expected = {(entry['kind'].capitalize() + '/' + entry['name'], language): admitted
                for entry in registry['entries'] for language, admitted in entry['supported'].items()}
    cells = report['cells']
    supported = [cell for cell in cells if cell['supported']]
    assert {(cell['key'], cell['language']): cell['supported'] for cell in cells} == expected
    assert len({(cell['key'], cell['language']) for cell in cells}) == len(cells)
    assert len(supported) > 0 and all(cell['status'] == 'pass' for cell in supported)
    assert report['expected_supported_cells'] == report['executed_supported_cells'] == len(supported)
    assert all(cell['status'] == 'unsupported_registry' for cell in cells if not cell['supported'])
    cases = report['cases']
    assert report['expected_emitted_modules'] == report['executed_emitted_modules'] == len(cases) > 0
    assert all(case['status'] == 'pass' and case['self_check_rows'] == 64 for case in cases)
    assert {case['language'] for case in cases} == {'rust', 'python', 'javascript'}
    routing = [case for case in cases if case.get('decision_measurement')]
    assert routing and all(case['golden_flip_rate'] == 0 and case['randomized_flip_rate'] <= .001 for case in routing)
    assert len(report['raw_numeric_contract']) == 3 and all(item['status'] == 'pass' and item['rows'] > 0 for item in report['raw_numeric_contract'])
    commands = report['commands']
    assert commands and all(type(command['exit']) is int for command in commands)
    for command in commands:
        assert {'argv', 'exit', 'stdout', 'stderr', 'seconds'} <= command.keys()
        assert command['argv'] and command['stdout'] in report['files'] and command['stderr'] in report['files']
    actual = {str(member.relative_to(path.parent)) for member in path.parent.rglob('*')
              if member.is_file() and member != path and member != path.parent / 'MANIFEST.sha256'}
    assert actual == set(report['files']), 'complete generated predictor/model/subprocess inventory'
    assert {'harness.sh', 'generate.rs', 'registry.json', 'fixtures.json'} <= report['files'].keys()
    for extension in ('rs', 'py', 'mjs'):
        assert any(name.endswith('/predictor.' + extension) for name in report['files'])
    for name, item in report['files'].items():
        member = path.parent / name
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert not member.is_symlink() and member.resolve().is_relative_to(path.parent.resolve())
        assert member.stat().st_size == item['bytes']
        assert hashlib.sha256(member.read_bytes()).hexdigest() == item['sha256']
    print('Hosted export parity:', len(supported), 'supported cells;', len(cases),
          'emitted modules; source/run/attempt and every file verified')


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_carbon_live_e2e(root, source_sha, run_id, attempt, spec_root=None):
    path = _receipt(root, 'live-ci-4tier-docker.json')
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    manifest_path = path.parent / 'MANIFEST.sha256'
    _require(manifest_path.is_file(), 'missing MANIFEST.sha256 next to live-ci-4tier-docker.json')
    _require(
        f'{digest}  live-ci-4tier-docker.json' in manifest_path.read_text().splitlines(),
        'MANIFEST.sha256 digest mismatch for live-ci-4tier-docker.json',
    )
    receipt = json.loads(raw_bytes)
    _require(receipt.get('schema') == 'jopedime.carbon.live-ci-4tier-docker.v1', 'invalid carbon live schema')
    _require(receipt.get('all_passed') is True, 'carbon live receipt all_passed != True')
    _require(receipt.get('source_sha') == source_sha, 'carbon live source_sha mismatch')
    _require(receipt.get('github_run_id') == run_id, 'carbon live github_run_id mismatch')
    rcpt_attempt = receipt.get('github_run_attempt')
    _require(type(rcpt_attempt) is int and 1 <= rcpt_attempt <= int(attempt), 'carbon live attempt mismatch')
    mode = receipt.get('pro_secrets_mode')
    _require(
        mode in ('zero_secrets_hybrid_live_and_emulator', 'live_pro_secrets'),
        f'unexpected pro_secrets_mode: {mode}',
    )
    container = receipt.get('container', {})
    _require(container.get('base_image') == 'ubuntu:24.04', 'container base_image must be ubuntu:24.04')
    _require(container.get('zero_credentials_pre_install') is True, 'zero_credentials_pre_install required')
    _require(container.get('post_logout_credentials_absent') is True, 'post_logout_credentials_absent required')
    install = receipt.get('install', {})
    _require(install.get('self_verify_verified') is True, 'self_verify_verified required')
    _require(install.get('manifest_ecdsa_verified') is True, 'manifest_ecdsa_verified required')
    _require(int(install.get('kek_pins_verified', 0)) >= 6, 'kek_pins_verified >= 6 required')
    _require(install.get('doctor_healthy') is True, 'doctor_healthy required')
    _require(int(install.get('doctor_dev_healthy_regions', 0)) >= 1, 'doctor_dev_healthy_regions >= 1 required')
    _require(int(install.get('doctor_prod_healthy_regions', 0)) >= 5, 'doctor_prod_healthy_regions >= 5 required')
    _require(install.get('quickstart_passed') is True, 'quickstart_passed required')
    _require(install.get('live_anon_high_gate_status') == 403, 'live_anon_high_gate_status == 403 required')
    runs = receipt.get('runs', [])
    expected_pairs = {(c, e) for c in ('dev', 'prod') for e in ('fast', 'medium', 'high', 'ultra')}
    _require(len(runs) == 8, f'expected 8 dev+prod tier runs, got {len(runs)}')
    _require({(r.get('cloud'), r.get('effort')) for r in runs} == expected_pairs, 'missing dev/prod effort pairs')
    for entry in runs:
        _require(entry.get('final_state') == 'completed', f'incomplete tier run: {entry}')
        _require(entry.get('encrypted_cuen') is True and entry.get('shredded_after_run') is True, 'CUEN/shred required')
        _require(float(entry.get('holdout_r2', 0.0)) > 0.90, 'holdout_r2 > 0.90 required')
        _require(float(entry.get('total_wall_ms', 0.0)) > 0.0, 'total_wall_ms > 0 required')
        cost = entry.get('cost_estimate', {})
        _require(
            float(cost.get('total_estimated_usd', 0.0)) > 0.0 and float(cost.get('runs_per_1_usd', 0.0)) > 0.0,
            'positive itemized cost_estimate required',
        )
        target = entry.get('execution_target')
        if entry.get('effort') in ('fast', 'medium') or mode == 'live_pro_secrets':
            _require(target == 'aws_live', f'expected aws_live target for {entry}')
        else:
            _require(target == 'ci_emulator', f'expected ci_emulator target for {entry}')
    _require(
        int(receipt.get('emulator_cells', {}).get('total_residual_cas_blobs', 1)) == 0,
        'zero residual CAS blobs required',
    )
    if spec_root is not None:
        pubs = sorted(p for p in spec_root.iterdir() if p.is_dir() and '-e2e-smoke' in p.name)
        _require(len(pubs) == 1, f'expected 1 e2e-smoke publication in {spec_root}, found {len(pubs)}')
        spec_dir = pubs[0] / 'carbon-spec'
        bundle_mf = json.loads((spec_dir / 'spec-bundle-manifest.json').read_text())
        pdf_rcpt = json.loads((spec_dir / 'spec-pdf-receipt.json').read_text())
        embedded_sha = hashlib.sha256((spec_dir / 'live-ci-4tier-docker.json').read_bytes()).hexdigest()
        _require(embedded_sha == digest, 'spec-dir live-ci-4tier-docker.json SHA-256 mismatch')
        _require(bundle_mf.get('live_ci_qualification_file') == 'live-ci-4tier-docker.json', 'missing bundle live file')
        _require(bundle_mf.get('live_ci_qualification_sha256') == digest, 'bundle live SHA-256 mismatch')
        _require(pdf_rcpt.get('live_ci_qualification_sha256') == digest, 'spec-pdf-receipt live SHA-256 mismatch')
        cov_file = spec_dir / 'code-coverage-summary.json'
        _require(cov_file.is_file(), 'missing spec-dir code-coverage-summary.json')
        cov_digest = hashlib.sha256(cov_file.read_bytes()).hexdigest()
        _require(bundle_mf.get('code_coverage_file') == 'code-coverage-summary.json', 'missing bundle code_coverage_file')
        _require(bundle_mf.get('code_coverage_sha256') == cov_digest, 'bundle code_coverage_sha256 mismatch')
        _require(pdf_rcpt.get('code_coverage_sha256') == cov_digest, 'spec-pdf-receipt code_coverage_sha256 mismatch')
    print('Fresh-Docker Carbon install + 8 dev/prod 4-tier runs verified; sha256 =', digest)


def filter_needs(source, output, excluded):
    needs = json.loads(source.read_text())
    if not isinstance(needs, dict):
        raise ValueError('job result inventory must be an object')
    for name in excluded:
        if name not in needs:
            raise ValueError(f'missing expected workflow dependency: {name}')
        needs.pop(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(needs) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    publication = commands.add_parser('publication')
    frontend = commands.add_parser('frontend')
    rendered = commands.add_parser('rendered-ux')
    export = commands.add_parser('export-parity')
    carbon_live = commands.add_parser('carbon-live-e2e')
    for command in (publication, frontend, rendered, export, carbon_live):
        command.add_argument('--root', type=Path, required=True)
    publication.add_argument('--source-sha', required=True)
    publication.add_argument('--checkout-sha', required=True)
    publication.add_argument('--run-id', required=True)
    publication.add_argument('--attempt', required=True)
    publication.add_argument('--event', required=True)
    frontend.add_argument('--source-sha', required=True)
    frontend.add_argument('--run-id', required=True)
    frontend.add_argument('--attempt', required=True)
    rendered.add_argument('--source-sha', required=True)
    rendered.add_argument('--run-id', required=True)
    rendered.add_argument('--attempt', required=True)
    rendered.add_argument('--event-sha', required=True)
    export.add_argument('--source-sha', required=True)
    export.add_argument('--run-id', required=True)
    export.add_argument('--attempt', required=True)
    export.add_argument('--repository', required=True)
    export.add_argument('--event', required=True)
    carbon_live.add_argument('--source-sha', required=True)
    carbon_live.add_argument('--run-id', required=True)
    carbon_live.add_argument('--attempt', required=True)
    carbon_live.add_argument('--spec-root', type=Path, default=None)
    needs = commands.add_parser('filter-needs')
    needs.add_argument('--input', type=Path, required=True)
    needs.add_argument('--output', type=Path, required=True)
    needs.add_argument('--exclude', action='append', default=[])
    args = parser.parse_args()
    if args.command == 'publication':
        verify_publication(args.root, args.source_sha, args.checkout_sha, args.run_id, args.attempt, args.event)
    elif args.command == 'frontend':
        verify_frontend(args.root, args.source_sha, args.run_id, args.attempt)
    elif args.command == 'rendered-ux':
        verify_rendered(args.root, args.source_sha, args.run_id, args.attempt, args.event_sha)
    elif args.command == 'export-parity':
        verify_export_parity(args.root, args.source_sha, args.run_id, args.attempt, args.repository, args.event)
    elif args.command == 'carbon-live-e2e':
        verify_carbon_live_e2e(args.root, args.source_sha, args.run_id, args.attempt, args.spec_root)
    else:
        filter_needs(args.input, args.output, args.exclude)


if __name__ == '__main__':
    main()
