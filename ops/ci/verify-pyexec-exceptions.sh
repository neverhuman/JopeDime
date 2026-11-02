#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: verify-pyexec-exceptions.sh AUDITED_REPO AUDITOR REPORT_JSON' >&2; exit 2; }
repo=$(realpath -- "$1")
auditor=$2
report=$(realpath -m -- "$3")
python3 - "$repo" "$auditor" "$report" <<'PY'
import collections
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

repo = Path(sys.argv[1]).resolve(strict=True)
auditor = sys.argv[2]
report = Path(sys.argv[3])
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
result = {
    'schema': 'jopedime.jankurai-pyexec-audit-check/v1',
    'register_sha256': hashlib.sha256(register_path.read_bytes()).hexdigest(),
    'audited_source': str(repo),
    'checked_paths': len(actual),
    'checked_findings': sum(len(v['findings']) for v in actual.values()),
    'result': 'pass',
    'paths': actual,
}
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(f"P37 exception integrity: PASS ({len(actual)} exact files; {result['checked_findings']} exact rule/fingerprint findings; source digests and expiry enforced)")
PY
