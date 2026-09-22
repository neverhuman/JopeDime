#!/usr/bin/env bash
# deploy/ci/check.sh — the `hygiene` lane: cheap, deterministic, no build.
set -euo pipefail
cd "$(dirname "$0")/../.."
fail=0
step(){ printf '\n== %s\n' "$*"; }
step "cargo metadata --locked"; cargo metadata --locked --format-version 1 --no-deps > /dev/null
step "exactly one rustls crypto provider in the workspace graph"
rustls_err=$(mktemp); rustls_tree=""
if rustls_tree=$(cargo tree --locked --workspace -e features -i rustls 2>"$rustls_err"); then
  providers=$(grep -oE 'rustls feature "(ring|aws_lc_rs)"' <<<"$rustls_tree" | sort -u)
  n=$(printf '%s\n' "$providers" | grep -c 'rustls feature' || true)
  if [ "$n" = 1 ]; then echo "ok: $providers"; else
    echo "${providers:-<none>}"
    echo "rustls must resolve exactly one crypto provider; $n found. Two (or zero) make CryptoProvider::from_crate_features() return None and every bare rustls::ClientConfig::builder() in the workspace panics (crates/smartcluster/jainnode/src/control.rs). Build the offending HTTP client without its provider feature (e.g. ureq's rustls-no-provider) instead of relaxing this check."
    fail=1
  fi
elif grep -q 'did not match any packages' "$rustls_err"; then
  # The one legitimate non-zero exit: rustls is genuinely absent from the graph.
  echo "rustls is not in the workspace graph; nothing to disambiguate"
else
  # Any other failure (most commonly a Cargo.toml edit whose Cargo.lock was not
  # regenerated: "cannot update the lock file ... because --locked was passed") means the
  # guard did not run. Fail closed -- a silent pass here is how the regression lands.
  cat "$rustls_err"
  echo "cargo tree failed (above); the rustls provider guard could not run. Fix the error -- do not treat this as a pass."
  fail=1
fi
rm -f "$rustls_err"
step "shell syntax + shellcheck"; while IFS= read -r f; do bash -n "$f"; done < <(git ls-files '*.sh'); if command -v shellcheck >/dev/null; then git ls-files 'deploy/ci/*.sh' 'xtools/**/*.sh' | xargs -r shellcheck -S warning; fi
step "actionlint"; if command -v actionlint >/dev/null; then actionlint -no-color; else echo "actionlint not installed (skipped locally; CI installs it)"; fi
step "gitleaks"; if command -v gitleaks >/dev/null; then gitleaks detect --redact --no-banner --config .gitleaks.toml --source . ; else echo "gitleaks not installed (skipped locally; CI installs it)"; fi
step "no family leftovers"; leftovers=$(git grep -nE 'git\.neverhuman\.org|127\.0\.0\.1:8787' -- crates python deploy .github ':!docs' ':!**/docs/**' ':!crates/jope-engine/assets/grammar/*' | grep -v 'deploy/kernel-v1/source/' || true); if [ -n "$leftovers" ]; then head -20 <<<"$leftovers"; echo "family remotes still referenced above"; fail=1; else echo "ok"; fi
step "capability parity matrix (docs/parity/capability-matrix.md; source-parsed subcommands, --help cross-check runs in the test lanes)"; if ! bash deploy/ci/parity-matrix.sh --no-build; then fail=1; fi
step "test inventory (docs/parity/test-inventory.json; Python static scan here, the nextest half runs where the workspace is built)"; if ! bash deploy/ci/test-inventory.sh --no-rust --static-python; then fail=1; fi
step "manifests (models index ↔ bundle manifests ↔ schema; no weights downloaded)"; if bash deploy/ci/manifests.sh; then echo "ok"; else fail=1; fi
step "datasets/test (jope-datasets verify --repo)"; if cargo run -q -p jope-datasets --locked -- verify --repo datasets/test; then echo "ok"; else echo "datasets/test verification failed"; fail=1; fi
step "models/ (jope-models verify --manifests-only: schema + index + PINNED, no network)"; if cargo run -q -p jope-models --locked -- verify --manifests-only; then echo "ok"; else echo "models manifests verification failed"; fail=1; fi
step "no absolute home paths in code"; abs=$(git grep -nE '/home/[a-z]+/' -- crates/*/src crates/*/tests deploy/systemd .github || true); if [ -n "$abs" ]; then head -20 <<<"$abs"; echo "absolute paths above: $(wc -l <<<"$abs") hits (ADR-0003: use the jope-paths resolvers / unit-template placeholders)"; fail=1; else echo "ok"; fi
step "no absolute home paths in python (informational until WP6 lands)"; pyabs=$(git grep -nE '/home/[a-z]+/' -- python/src || true); if [ -n "$pyabs" ]; then echo "$(wc -l <<<"$pyabs") hits in python/src (informational until WP6 lands)"; else echo "ok"; fi
step "rust-test scope partitions the workspace (deploy/ci/rust-test-scope.sh)"; if ! python3 deploy/ci/rust-test-scope-check.py; then fail=1; fi
step "README benchmark blocks match committed evidence (deploy/ci/readme-benchmarks.py)"; if ! python3 deploy/ci/readme-benchmarks.py --check; then fail=1; fi
step "size caps"; big=$(git ls-files -z | xargs -0 stat -c '%s %n' | awk '$1>5242880{print}' ); if [ -n "$big" ]; then echo "$big"; echo "files over 5 MiB"; fail=1; else echo "ok"; fi
exit $fail
