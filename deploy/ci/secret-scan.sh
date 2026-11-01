#!/usr/bin/env bash
# Scan every ancestor of the checked-out source head, independent of unrelated local refs.
set -euo pipefail
[ "$#" -le 1 ] || { echo 'usage: secret-scan.sh [repository]' >&2; exit 2; }
repo=${1:-.}
repo=$(cd "$repo" && pwd -P)
config=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.gitleaks.toml
[ "$(git -C "$repo" rev-parse --is-shallow-repository)" = false ] || {
  echo 'secret scan requires full HEAD history; fetch complete history before scanning' >&2
  exit 2
}
head=$(git -C "$repo" rev-parse --verify HEAD)
count=$(git -C "$repo" rev-list --count HEAD)
printf 'Secret scan scope: HEAD=%s and all %s reachable commits; unrelated refs are excluded\n' "$head" "$count"
# Load exact reviewed finding fingerprints from the scanned checkout, regardless
# of the caller's working directory. New commits produce new fingerprints.
gitleaks detect --redact --no-banner --config "$config" --source "$repo" \
  --gitleaks-ignore-path "$repo" --log-opts=HEAD
