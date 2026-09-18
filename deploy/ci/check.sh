#!/usr/bin/env bash
# deploy/ci/check.sh — the `hygiene` lane: cheap, deterministic, no build.
set -euo pipefail
cd "$(dirname "$0")/../.."
fail=0
step(){ printf '\n== %s\n' "$*"; }
step "cargo metadata --locked"; cargo metadata --locked --format-version 1 --no-deps > /dev/null
step "shell syntax + shellcheck"; while IFS= read -r f; do bash -n "$f"; done < <(git ls-files '*.sh'); if command -v shellcheck >/dev/null; then git ls-files 'deploy/ci/*.sh' 'xtools/**/*.sh' | xargs -r shellcheck -S warning; fi
step "actionlint"; if command -v actionlint >/dev/null; then actionlint -no-color; else echo "actionlint not installed (skipped locally; CI installs it)"; fi
step "gitleaks"; if command -v gitleaks >/dev/null; then gitleaks detect --redact --no-banner --config .gitleaks.toml --source . ; else echo "gitleaks not installed (skipped locally; CI installs it)"; fi
step "no family leftovers"; leftovers=$(git grep -nE 'git\.neverhuman\.org|127\.0\.0\.1:8787' -- crates python deploy .github ':!docs' ':!**/docs/**' ':!crates/jope-engine/assets/grammar/*' | grep -v 'deploy/kernel-v1/source/' || true); if [ -n "$leftovers" ]; then head -20 <<<"$leftovers"; echo "family remotes still referenced above"; fail=1; else echo "ok"; fi
step "no absolute home paths in code"; abs=$(git grep -nE '/home/[a-z]+/' -- crates/*/src crates/*/tests python/src deploy/systemd .github || true); if [ -n "$abs" ]; then head -20 <<<"$abs"; echo "absolute paths above: $(wc -l <<<"$abs") hits (allowed only until WP4 completes)"; else echo "ok"; fi
step "datasets/test (jope-datasets verify --repo)"; if cargo run -q -p jope-datasets --locked -- verify --repo datasets/test; then echo "ok"; else echo "datasets/test verification failed"; fail=1; fi
step "manifests (models index ↔ bundle manifests ↔ schema; no weights downloaded)"; if bash deploy/ci/manifests.sh; then echo "ok"; else fail=1; fi
step "models/ (jope-models verify --manifests-only: schema + index + PINNED, no network)"; if cargo run -q -p jope-models --locked -- verify --manifests-only; then echo "ok"; else echo "models manifests verification failed"; fail=1; fi
step "size caps"; big=$(git ls-files -z | xargs -0 stat -c '%s %n' | awk '$1>5242880{print}' ); if [ -n "$big" ]; then echo "$big"; echo "files over 5 MiB"; fail=1; else echo "ok"; fi
exit $fail
