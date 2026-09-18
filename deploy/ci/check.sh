#!/usr/bin/env bash
# deploy/ci/check.sh — the `hygiene` lane: cheap, deterministic, no build.
set -euo pipefail
cd "$(dirname "$0")/../.."
fail=0
step(){ printf '\n== %s\n' "$*"; }
step "cargo metadata --locked"; cargo metadata --locked --format-version 1 --no-deps > /dev/null
step "shell syntax + shellcheck"; while IFS= read -r f; do bash -n "$f"; done < <(git ls-files '*.sh'); if command -v shellcheck >/dev/null; then git ls-files 'deploy/ci/*.sh' 'xtools/**/*.sh' | xargs -r shellcheck -S warning; fi
step "actionlint"; if command -v actionlint >/dev/null; then actionlint -color never; else echo "actionlint not installed (skipped locally; CI installs it)"; fi
step "gitleaks"; if command -v gitleaks >/dev/null; then gitleaks detect --redact --no-banner --config .gitleaks.toml --source . ; else echo "gitleaks not installed (skipped locally; CI installs it)"; fi
step "no family leftovers"; if git grep -nE 'git\.neverhuman\.org|127\.0\.0\.1:8787' -- crates python deploy .github ':!docs' ':!**/docs/**' ':!crates/jope-engine/assets/grammar/*' | grep -v 'deploy/kernel-v1/source/' | head -20; then echo "family remotes still referenced above"; fail=1; else echo "ok"; fi
step "no absolute home paths in code"; if git grep -nE '/home/[a-z]+/' -- crates/*/src crates/*/tests python/src deploy/systemd .github | head -20; then echo "absolute paths above (allowed only until WP4 completes)"; fi
step "size caps"; big=$(git ls-files -z | xargs -0 stat -c '%s %n' | awk '$1>5242880{print}' ); if [ -n "$big" ]; then echo "$big"; echo "files over 5 MiB"; fail=1; else echo "ok"; fi
exit $fail
