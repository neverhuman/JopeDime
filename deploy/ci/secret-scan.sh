#!/usr/bin/env bash
# Scan committed HEAD blobs, then every ancestor of HEAD when history is complete.
# The tree scan is what a shallow clone and a full clone share: a tracked .env
# cannot hide in an ancestor the shallow fetch did not download. gitignored
# working-tree env files are not on HEAD and are not scanned.
# History uses deploy/ci/gitleaks-history.toml, which allowlists only the
# removed parallel/recovery tree. That allowlist is not used for the tree scan.
set -euo pipefail
[ "$#" -le 1 ] || { echo 'usage: secret-scan.sh [repository]' >&2; exit 2; }
repo=${1:-.}
repo=$(cd "$repo" && pwd -P)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config=$root/.gitleaks.toml
history_config=$root/deploy/ci/gitleaks-history.toml
[ -f "$config" ] && [ -f "$history_config" ] || { echo 'secret scan config missing' >&2; exit 2; }

tree=$(mktemp -d)
history_runtime=$(mktemp --suffix=.toml)
trap 'rm -rf "$tree"; rm -f "$history_runtime"' EXIT
printf 'synthetic tree scan anchor\n' > "$tree/README"
env_count=0
while IFS= read -r -d '' path; do
  [ -n "$path" ] || continue
  if [[ "$path" =~ (^|/)[^/]*\.env$ ]]; then
    mkdir -p -- "$tree/$(dirname -- "$path")"
    git -C "$repo" show "HEAD:$path" > "$tree/$path"
    env_count=$((env_count + 1))
  fi
done < <(git -C "$repo" ls-tree -r -z --name-only HEAD)
printf 'Secret scan HEAD tree: %s tracked env files\n' "$env_count"
# Base config only. The history allowlist must not hide a restored path.
gitleaks detect --redact --no-banner --no-git --config "$config" \
  --enable-rule=jope-env-files --source "$tree"

[ "$(git -C "$repo" rev-parse --is-shallow-repository)" = false ] || {
  echo 'secret scan requires full HEAD history; fetch complete history before scanning' >&2
  exit 2
}
head=$(git -C "$repo" rev-parse --verify HEAD)
count=$(git -C "$repo" rev-list --count HEAD)
printf 'Secret scan history: HEAD=%s and all %s reachable commits; unrelated refs are excluded\n' "$head" "$count"
sed "s#^path = \"\\.gitleaks.toml\"#path = \"$config\"#" "$history_config" > "$history_runtime"
gitleaks detect --redact --no-banner --config "$history_runtime" --source "$repo" \
  --gitleaks-ignore-path "$repo" --log-opts=HEAD
