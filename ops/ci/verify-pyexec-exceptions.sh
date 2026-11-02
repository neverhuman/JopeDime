#!/usr/bin/env bash
set -euo pipefail
[[ $# == 5 ]] || { echo 'usage: verify-pyexec-exceptions.sh AUDITED_REPO AUDITOR REPORT_JSON CHECKOUT SOURCE_MANIFEST' >&2; exit 2; }
repo=$(realpath -- "$1")
auditor=$2
report=$(realpath -m -- "$3")
checkout=$(realpath -- "$4")
source_manifest=$(realpath -- "$5")
python3 - "$repo" "$auditor" "$report" "$checkout" "$source_manifest" <<'PY'
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

repo = Path(sys.argv[1]).resolve(strict=True)
auditor = sys.argv[2]
report = Path(sys.argv[3])
checkout = Path(sys.argv[4]).resolve(strict=True)
source_manifest_path = Path(sys.argv[5]).resolve(strict=True)
register_path = repo / 'docs/release/jankurai-pyexec-p37-dispositions-20261102.json'
register = json.loads(register_path.read_text())
if register.get('schema') != 'jopedime.jankurai-pyexec-dispositions/v1':
    raise SystemExit('P37 disposition register schema mismatch')
if dt.date.today().isoformat() > register.get('expires_on', ''):
    raise SystemExit('P37 source/fingerprint dispositions expired; review before audit')
policy = tomllib.loads((repo / 'agent/audit-policy.toml').read_text())
excluded = set(policy.get('scan', {}).get('extra_excluded_paths', []))
expected_paths = register.get('paths', {})
if not expected_paths or not set(expected_paths).issubset(excluded):
    raise SystemExit('P37 register paths must all have matching exact-path policy entries')
actual = {}
for rel, entry in sorted(expected_paths.items()):
    path = Path(rel)
    if path.is_absolute() or '..' in path.parts:
        raise SystemExit(f'unsafe P37 path in register: {rel}')
    source = (repo / path).resolve(strict=True)
    if repo not in source.parents or not source.is_file():
        raise SystemExit(f'P37 path escapes audited source: {rel}')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != entry.get('sha256'):
        raise SystemExit(f'P37 source digest drift: {rel} expected {entry.get("sha256")} got {digest}')
    cmd = [auditor, 'audit-file', '--path', rel, '--mode', 'advisory', str(repo), '--format', 'json']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise SystemExit(f'audit-file failed for {rel} rc={proc.returncode}: {proc.stderr[-500:]}')
    try:
        decision = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'audit-file returned invalid JSON for {rel}: {exc}')
    if decision.get('verdict') != 'pass' or decision.get('exit_code') != 0:
        raise SystemExit(f'audit-file advisory gate failed for {rel}: {decision.get("summary")}')
    blocking = decision.get('blocking', {})
    advisory = decision.get('advisory', {})
    if any(blocking.get(key) for key in ('new_hard_findings', 'worsened_findings', 'always_block_findings')) or advisory.get('new_soft_findings'):
        raise SystemExit(f'audit-file reported new or worsened findings for {rel}')
    findings = decision.get('preexisting_findings')
    if not isinstance(findings, list):
        raise SystemExit(f'audit-file omitted preexisting_findings for {rel}')
    actual_rows = sorted((f.get('rule_id') or f.get('check_id') or '', f.get('fingerprint') or '') for f in findings)
    expected_rows = sorted((f.get('rule_id') or '', f.get('fingerprint') or '') for f in entry.get('findings', []))
    if any(not rule or not fingerprint for rule, fingerprint in actual_rows + expected_rows):
        raise SystemExit(f'P37 finding inventory has missing identity: {rel}')
    if actual_rows != expected_rows:
        raise SystemExit(f'P37 finding inventory drift for {rel}: expected {len(expected_rows)}, got {len(actual_rows)}')
    actual[rel] = {'sha256': digest, 'findings': [{'rule_id': r, 'fingerprint': f} for r, f in actual_rows]}
cross_expected = register.get('cross_file_findings', [])
if not cross_expected:
    raise SystemExit('P37 cross-file inventory is empty')
manifest = json.loads(source_manifest_path.read_text())
head = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
if manifest.get('checkout_sha') != head:
    raise SystemExit('P37 cross-file audit checkout differs from the captured source head')
cross_paths = set(expected_paths)
for item in cross_expected:
    participants = item.get('participants', {})
    if not item.get('path') or not isinstance(participants, dict) or not (set(participants) & cross_paths):
        raise SystemExit('P37 cross-file identity must name a registered path and participants')
    for rel, expected_digest in participants.items():
        if rel not in expected_paths:
            raise SystemExit(f'P37 cross-file participant is not a registered path: {rel}')
        for root in (repo, checkout):
            source = (root / rel).resolve(strict=True)
            if root not in source.parents or hashlib.sha256(source.read_bytes()).hexdigest() != expected_digest:
                raise SystemExit(f'P37 cross-file participant digest drift: {rel}')

def cross_rows(findings):
    rows = []
    for finding in findings:
        evidence = '\n'.join(str(value) for value in finding.get('evidence', []))
        participants = {path for path in cross_paths if path in evidence}
        if finding.get('path') in cross_paths:
            participants.add(finding['path'])
        if not participants:
            continue
        rows.append({
            'path': finding.get('path') or '',
            'rule_id': finding.get('rule_id') or '',
            'check_id': finding.get('check_id') or '',
            'fingerprint': finding.get('fingerprint') or '',
            'line': finding.get('line'),
            'problem': finding.get('problem') or '',
            'reason': finding.get('reason') or '',
            'category': finding.get('category') or '',
            'evidence': finding.get('evidence', []),
            'participants': sorted(participants),
        })
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))

policy_path = checkout / 'agent/audit-policy.toml'
policy_text = policy_path.read_text()
policy = tomllib.loads(policy_text)
excluded_paths = policy.get('scan', {}).get('extra_excluded_paths', [])
for rel in expected_paths:
    if excluded_paths.count(rel) < 1:
        raise SystemExit(f'missing exact P37 policy entry before cross-file audit: {rel}')
for rel in sorted(cross_paths):
    line = f'  "{rel}",\n'
    if policy_text.count(line) < 1:
        raise SystemExit(f'missing literal exact-path policy line for {rel}')
    policy_text = policy_text.replace(line, '')
report.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='pyexec-audit-', dir=report.parent) as temp_dir:
    captured_repo = Path(temp_dir) / 'source'
    captured_repo.mkdir()
    tracked = subprocess.run(['git', '-C', str(checkout), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], capture_output=True, check=True).stdout.split(b'\0')
    for raw_path in tracked:
        if not raw_path:
            continue
        rel = os.fsdecode(raw_path)
        source = checkout / rel
        destination = captured_repo / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        elif source.is_file():
            os.link(source, destination)
    git_marker = checkout / '.git'
    if git_marker.is_file():
        (captured_repo / '.git').write_bytes(git_marker.read_bytes())
    else:
        raise SystemExit('P37 source capture expected a worktree .git file')
    captured_policy = captured_repo / 'agent/audit-policy.toml'
    captured_policy.unlink()
    captured_policy.write_text(policy_text)
    cross_json = report.with_name('pyexec-cross-file-audit.json')
    cross_md = report.with_name('pyexec-cross-file-audit.md')
    cmd = [auditor, 'audit', str(captured_repo), '--full', '--read-only', '--no-badge', '--no-score-history', '--json', str(cross_json), '--md', str(cross_md)]
    for rel in sorted(cross_paths):
        cmd.extend(['--changed', rel])
    proc = subprocess.run(cmd, cwd=captured_repo, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise SystemExit(f'P37 cross-file audit failed rc={proc.returncode}: {proc.stderr[-500:]}')
    try:
        cross_decision = json.loads(cross_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'P37 cross-file audit report unavailable or invalid: {exc}')
    actual_cross = cross_rows(cross_decision.get('findings', []))
    expected_cross = sorted((item.get('raw_identity') for item in cross_expected), key=lambda row: json.dumps(row, sort_keys=True))
    if any(not isinstance(row, dict) for row in expected_cross):
        raise SystemExit('P37 cross-file register lacks complete raw identities')
    if actual_cross != expected_cross:
        raise SystemExit(f'P37 cross-file finding drift: expected {len(expected_cross)} raw identities, got {len(actual_cross)}')

    copy_json = report.with_name('pyexec-copy-code-full.json')
    copy_md = report.with_name('pyexec-copy-code-full.md')
    cmd = [auditor, 'copy-code', str(captured_repo), '--include-tests', '--max-findings', '10000', '--json', str(copy_json), '--md', str(copy_md)]
    proc = subprocess.run(cmd, cwd=captured_repo, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise SystemExit(f'P37 full copy-code scan failed rc={proc.returncode}: {proc.stderr[-500:]}')
    try:
        copy_decision = json.loads(copy_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'P37 full copy-code report unavailable or invalid: {exc}')
    copy_classes = copy_decision.get('classes')
    copy_summary = copy_decision.get('summary', {})
    if not isinstance(copy_classes, list) or len(copy_classes) >= 10000 or len(copy_classes) != copy_summary.get('hard_classes', -1) + copy_summary.get('warning_classes', -1):
        raise SystemExit('P37 full copy-code inventory is truncated or inconsistent with its class totals')
    def raw_copy_rows(classes):
        return sorted((item for item in classes
                       if any(instance.get('path') in cross_paths for instance in item.get('instances', []))),
                      key=lambda row: json.dumps(row, sort_keys=True))

    actual_copy = raw_copy_rows(copy_classes)
    expected_copy = register.get('raw_copy_code_classes')
    if not isinstance(expected_copy, list) or actual_copy != sorted(expected_copy, key=lambda row: json.dumps(row, sort_keys=True)):
        raise SystemExit(f'P37 raw cross-file copy identity drift: expected {len(expected_copy) if isinstance(expected_copy, list) else "missing"}, got {len(actual_copy)}')

    # Negative control: a new cross finding on a second excluded path must be rejected,
    # without changing any pinned source bytes.
    second_path = next((path for path in sorted(cross_paths) if path != cross_expected[0]['path']), None)
    if second_path is None:
        raise SystemExit('P37 negative control needs a second registered path')
    source_hashes_before = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in expected_paths}
    injected = dict(cross_decision.get('findings', [])[0]) if cross_decision.get('findings') else {}
    injected.update({'path': second_path, 'fingerprint': 'sha256:negative-control-injected', 'line': 1,
                     'evidence': [f'instances={second_path}:1-2, synthetic/peer.py:1-2']})
    if cross_rows(cross_decision.get('findings', []) + [injected]) == actual_cross:
        raise SystemExit('P37 cross-file negative control failed to detect injected second-path finding')
    synthetic = {'id': 'copy-code-negative-control', 'fingerprint': 'negative-control',
                 'instances': [{'path': second_path, 'start_line': 1, 'end_line': 2}]}
    if raw_copy_rows(copy_classes + [synthetic]) == actual_copy:
        raise SystemExit('P37 raw copy-code negative control failed to detect injected second-path class')
    if source_hashes_before != {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in expected_paths}:
        raise SystemExit('P37 cross-file negative control changed pinned source bytes')
result = {
    'schema': 'jopedime.jankurai-pyexec-audit-check/v1',
    'register_sha256': hashlib.sha256(register_path.read_bytes()).hexdigest(),
    'audited_source': str(repo),
    'checkout_sha': head,
    'checked_paths': len(actual),
    'checked_findings': sum(len(v['findings']) for v in actual.values()) + len(actual_cross),
    'cross_file_audit': str(cross_json),
    'copy_code_full_audit': str(copy_json),
    'cross_file_findings': actual_cross,
    'result': 'pass',
    'paths': actual,
}
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(f"P37 exception integrity: PASS ({len(actual)} exact files; {result['checked_findings']} exact rule/fingerprint findings; cross-file audit and expiry enforced)")
PY
