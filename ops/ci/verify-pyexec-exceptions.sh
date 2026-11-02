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
import tempfile
import subprocess
import sys
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
if len(cross_expected) != 1:
    raise SystemExit('P37 cross-file inventory must contain exactly one registered finding')
manifest = json.loads(source_manifest_path.read_text())
head = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
if manifest.get('checkout_sha') != head:
    raise SystemExit('P37 cross-file audit checkout differs from the captured source head')
cross = cross_expected[0]
for rel, expected_digest in cross.get('participants', {}).items():
    for root in (repo, checkout):
        source = (root / rel).resolve(strict=True)
        if root not in source.parents or hashlib.sha256(source.read_bytes()).hexdigest() != expected_digest:
            raise SystemExit(f'P37 cross-file participant digest drift: {rel}')
cross_path = cross['path']
if cross_path not in expected_paths:
    raise SystemExit('P37 cross-file finding must be attributed to a registered path')
policy_path = checkout / 'agent/audit-policy.toml'
policy_bytes = policy_path.read_bytes()
policy_text = policy_bytes.decode()
for rel in expected_paths:
    line = f'  "{rel}",\n'
    if policy_text.count(line) != 1:
        raise SystemExit(f'expected one exact P37 policy entry before cross-file audit: {rel}')
    policy_text = policy_text.replace(line, '', 1)
cross_json = report.with_name('pyexec-cross-file-audit.json')
cross_md = report.with_name('pyexec-cross-file-audit.md')
# The auditor requires fresh reports outside its audited repository in read-only
# mode. Use the runner scratch (or a private home cache), then retain full bytes.
scratch = Path(os.environ.get('RUNNER_TEMP') or os.environ.get('TMPDIR')
               or (Path.home() / '.cache/jope/p37-cross')).resolve()
if scratch == checkout or checkout in scratch.parents:
    raise SystemExit('P37 read-only report scratch must be outside the audited checkout')
scratch.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='jope-p37-cross-', dir=scratch) as temporary:
    external_json = Path(temporary) / 'audit.json'
    external_md = Path(temporary) / 'audit.md'
    try:
        policy_path.write_text(policy_text)
        cmd = [auditor, 'audit', str(checkout), '--changed', cross_path, '--read-only', '--no-badge', '--no-score-history', '--json', str(external_json), '--md', str(external_md)]
        proc = subprocess.run(cmd, cwd=checkout, capture_output=True, text=True)
    finally:
        policy_path.write_bytes(policy_bytes)
    for original, retained in ((external_json, cross_json), (external_md, cross_md)):
        if original.is_file():
            retained.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive publication cannot replace evidence from an earlier attempt.
            with retained.open('xb') as output:
                output.write(original.read_bytes())
if proc.returncode not in (0, 1):
    raise SystemExit(f'P37 cross-file audit failed rc={proc.returncode}: {proc.stderr[-500:]}')
try:
    cross_decision = json.loads(cross_json.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f'P37 cross-file audit report unavailable or invalid: {exc}')
actual_cross = sorted((f.get('path') or '', f.get('rule_id') or '', f.get('check_id') or '', f.get('fingerprint') or '', f.get('line'))
                      for f in cross_decision.get('findings', [])
                      if f.get('path') == cross_path and f.get('rule_id') == cross.get('rule_id'))
expected_cross = [(cross_path, cross['rule_id'], cross['check_id'], cross['fingerprint'], cross['line'])]
if actual_cross != expected_cross:
    raise SystemExit(f'P37 cross-file finding drift: expected one exact finding, got {len(actual_cross)}')
result = {
    'schema': 'jopedime.jankurai-pyexec-audit-check/v1',
    'register_sha256': hashlib.sha256(register_path.read_bytes()).hexdigest(),
    'audited_source': str(repo),
    'checkout_sha': head,
    'checked_paths': len(actual),
    'checked_findings': sum(len(v['findings']) for v in actual.values()) + len(actual_cross),
    'cross_file_audit': str(cross_json),
    'cross_file_findings': [{'path': p, 'rule_id': r, 'check_id': c, 'fingerprint': f, 'line': line} for p, r, c, f, line in actual_cross],
    'result': 'pass',
    'paths': actual,
}
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(f"P37 exception integrity: PASS ({len(actual)} exact files; {result['checked_findings']} exact rule/fingerprint findings; cross-file audit and expiry enforced)")
PY
