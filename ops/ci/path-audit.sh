#!/usr/bin/env bash
# Fail if product source hard-codes host paths or the old forge.
# Historical citations are allowed only in ORIGIN.md and the port plan.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

allow='(^|/)(CURSOR_GITHUB_PORT_PLAN\.md|docs/ORIGIN\.md)$'
hits="$(
  find . \
    -path './.git' -prune -o \
    -path './target' -prune -o \
    -type f \( \
      -name '*.rs' -o -name '*.py' -o -name '*.toml' -o -name '*.sh' \
      -o -name '*.yml' -o -name '*.yaml' -o -name '*.md' -o -name 'Justfile' \
    \) -print \
  | grep -Ev "$allow" \
  | LC_ALL=C xargs -r grep -nE '/home/ubunt[u]|git\.neverhuman\.org|158\.101\.' \
    || true
)"

if [[ -n "$hits" ]]; then
  printf '%s\n' 'path-audit: host or forge paths in product source:'
  printf '%s\n' "$hits"
  exit 1
fi

printf '%s\n' 'path-audit: ok'
