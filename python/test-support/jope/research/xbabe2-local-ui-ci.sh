#!/usr/bin/env bash
# Explicit xbabe2-only Jope dashboard proof. Never part of normal CI.
# Usage: XBABE2_LOCAL_UI_CI=1 scripts/xbabe2-local-ui-ci.sh --output NEW_DIRECTORY
# Bullet/Playwright and authenticated-provider checks have separate owners/runners.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 -B - "$script_dir/../parallel/runs/flywheel-implementation-20260910/jope-dashboard-rendered-v1" "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

if (Path('/proc/sys/kernel/hostname').read_text().strip() != 'xbabe2'
        or os.environ.get('XBABE2_LOCAL_UI_CI') != '1'
        or any(key in os.environ for key in (
            'CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'BUILDKITE', 'TF_BUILD', 'TEAMCITY_VERSION'))):
    raise SystemExit('Requires explicit xbabe2 local execution; other hosts and CI are refused.')

bundle = Path(sys.argv[1]).resolve()

def verified(path, expected):
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SystemExit(f'Missing local proof artifact: {path}: {error}')
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit(f'Local proof artifact changed: {path}')
    return raw

binding = json.loads(verified(bundle / 'source-binding.json',
    '1438af464da41946e44c9f61bad10fe6dcf77f95110d7e17e519a855400bf579'))
for name, digest in binding['files'].items():
    verified(bundle / name, digest)
original = json.loads(verified(bundle / 'attempt-1/request.json',
    'fc255e0ff67f4c1e62326965e1191093c05d92865738b869c5e81400cbc52149'))
verified(bundle / 'target/debug/jope-dashboard-rendered-check', original['runner_binary_sha256'])
print('Scope: actual Jope dashboard PTY fixture only; excludes browser/provider checks and all-CI certification.', flush=True)
os.execv('/usr/bin/python3', ['/usr/bin/python3', '-B', str(bundle / 'run_local.py'), *sys.argv[2:]])
PY
