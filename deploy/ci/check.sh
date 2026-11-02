#!/usr/bin/env bash
# deploy/ci/check.sh — the `hygiene` lane: cheap, deterministic, no build.
set -euo pipefail
cd "$(dirname "$0")/../.."
fail=0
python3 -B deploy/ci/verify-vendor-security.py --root .
step(){ printf '\n== %s\n' "$*"; }
step "cargo metadata --locked"; cargo metadata --locked --format-version 1 --no-deps > /dev/null
step "Rust formatting"; if ! cargo fmt --all --check; then fail=1; fi
step "explicit coverage mapping (static, not collection or execution)"; if ! cargo run -q --locked -p jope-records -- coverage validate; then fail=1; fi
step "full-source audit evidence and trusted-main routing"; if ! python3 -B deploy/ci/test-audit-lane.py; then fail=1; fi
step "trusted compiler cache boundaries"; if ! python3 deploy/ci/test-full-main.py; then fail=1; fi
step "S1 fixed-equal attempt contracts (CPU only)"; if ! python3 deploy/lambda/training/teacher-v2/test_s1_budget.py; then fail=1; fi
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
step "DL train-only budget and output contracts (CPU only)"; if ! python3 deploy/dl-trainonly/test_contracts.py; then fail=1; fi
step "DL lane pinned source custody (stdlib, no staged imports)"; if ! python3 deploy/dl-trainonly/test_verify_lane.py; then fail=1; fi
step "DL offline Python audit refusals (stdlib, no traffic or jobs)"; if ! python3 deploy/dl-trainonly/test_offline_guard.py; then fail=1; fi
step "DL offline shard group shares one deadline (scaled, fake runner)"; if ! python3 deploy/ci/test-dl-offline-group.py; then fail=1; fi
step "shell syntax + shellcheck"; while IFS= read -r f; do bash -n "$f"; done < <(git ls-files '*.sh'); if command -v shellcheck >/dev/null; then git ls-files 'deploy/ci/*.sh' 'deploy/release/*.sh' 'xtools/**/*.sh' | xargs -r shellcheck -S warning; fi
step "security lane (workflow lint, full-history secrets, all-feature dependency advisories)"; if ! bash tools/security-lane.sh; then fail=1; fi
step "no family leftovers"; leftovers=$(git grep -nE 'git\.neverhuman\.org|127\.0\.0\.1:8787' -- crates python deploy .github ':!docs' ':!**/docs/**' ':!crates/jope-engine/assets/grammar/*' || true); if [ -n "$leftovers" ]; then head -20 <<<"$leftovers"; echo "family remotes still referenced above"; fail=1; else echo "ok"; fi
step "shipped-text rule in prose (deploy/ci/ip-scrub.sh; emitted code is covered by feat-math's ip_scrub test)"; if ! bash deploy/ci/ip-scrub.sh; then fail=1; fi
step "capability parity matrix (docs/parity/capability-matrix.md; source-parsed subcommands, --help cross-check runs in the test lanes)"; if ! bash deploy/ci/parity-matrix.sh --no-build; then fail=1; fi
step "test inventory (docs/parity/test-inventory.json; Python static scan here, the nextest half runs where the workspace is built)"; if ! bash deploy/ci/test-inventory.sh --no-rust --static-python; then fail=1; fi
step "manifests (models index ↔ bundle manifests ↔ schema; no weights downloaded)"; if bash deploy/ci/manifests.sh; then echo "ok"; else fail=1; fi
step "datasets/test (jope-datasets verify --repo)"; if cargo run -q -p jope-datasets --locked -- verify --repo datasets/test; then echo "ok"; else echo "datasets/test verification failed"; fail=1; fi
step "models/ (jope-models verify --manifests-only: schema + index + PINNED, no network)"; if cargo run -q -p jope-models --locked -- verify --manifests-only; then echo "ok"; else echo "models manifests verification failed"; fail=1; fi
step "no absolute home paths in code"; abs=$(git grep -nE '/home/[a-z]+/' -- crates/*/src crates/*/tests deploy/systemd .github ':(exclude)crates/jope-appliance/tests/e2e/INT-01-ACCEPTANCE.md' ':(exclude)crates/jope-appliance/tests/system/artifact-stage-linux.json' || true); if [ -n "$abs" ]; then head -20 <<<"$abs"; echo "absolute paths above: $(wc -l <<<"$abs") hits (ADR-0003: use the jope-paths resolvers / unit-template placeholders)"; fail=1; else echo "ok"; fi
step "no absolute home paths in python (informational until WP6 lands)"; pyabs=$(git grep -nE '/home/[a-z]+/' -- python/src || true); if [ -n "$pyabs" ]; then echo "$(wc -l <<<"$pyabs") hits in python/src (informational until WP6 lands)"; else echo "ok"; fi
step "rust-test scope partitions the workspace (deploy/ci/rust-test-scope.sh)"; if ! cargo run -q --locked -p jope-records -- rust-scope-check; then fail=1; fi
step "repository layout (jope-records repo-layout; exceptions in deploy/ci/repo-layout-allow.txt)"; if ! cargo run -q --locked -p jope-records -- repo-layout; then fail=1; fi
step "README benchmark blocks match committed evidence (jope-records readme-benchmarks)"; if ! cargo run -q --locked -p jope-records -- readme-benchmarks --check; then fail=1; fi
step "README current regression ranking and plot identities"; if ! bash assets/readme-performance/v1/render.sh --check; then fail=1; fi
step "paper and README numbers trace (jope-records paper-trace; docs/paper/NUMBERS_TRACE.md)"; if ! cargo run -q --locked -p jope-records -- paper-trace; then fail=1; fi
# A family shorter than one of its own tests can never pass: full-scratch was killed at 600 s
# inside feat-math::golden fast_profile_portfolio_head (~580-619 s). Budgets are argv[3] of each
# family's `timeout ... <N>s` command in the coverage map.
step "full-default/full-scratch budgets cover the slowest recorded feat-math::golden test"
slowest=$(jq '[.tests[] | select(.binary == "feat-math::golden") | .seconds] | max' docs/parity/test-timings.json)
if jq -e --argjson slowest "$slowest" '[.lanes["full-default", "full-scratch"].command[3] | rtrimstr("s") | tonumber] | length == 2 and all(. >= 1.25 * $slowest + 120)' deploy/ci/coverage.json > /dev/null; then echo "ok (slowest ${slowest}s)"; else echo "a full family budget is below 1.25 x ${slowest}s + 120s"; fail=1; fi
step "external custody verifier controls"; if ! python3 -B research/flywheel_selfplay/custody/verify_custody.py --self-test; then fail=1; fi
step "size caps"; big=$(git ls-files -z | xargs -0 stat -c '%s %n' | awk '$1>5242880{print}' ); if [ -n "$big" ]; then echo "$big"; echo "files over 5 MiB"; fail=1; else echo "ok"; fi
step "no large files or archives newly added under flywheel custody"; custody_base=${JOPE_CUSTODY_BASE:-${GITHUB_BASE_REF:+origin/$GITHUB_BASE_REF}}; if ! bash deploy/ci/check-custody-additions.sh "${custody_base:-origin/main}"; then fail=1; fi
# gitleaks' private-key rule needs a complete block and skips its allowlisted paths (docs/legacy/,
# SHA256SUMS, ...), and the gitleaks step above is skipped where gitleaks is not installed. This
# refuses any PEM private-key armour line (BEGIN or END; RSA, EC, OPENSSH, ENCRYPTED SIGSTORE, ...)
# in a tracked text file, so the cosign release key (docs/release/README.md, "Key custody") can
# never be committed next to its public half. .gitleaks.toml is excluded: a comment there quotes
# gitleaks' own synthetic fixture.
step "no private key blocks in tracked files"
pem='-----(BEGIN|END) ([A-Z0-9]+ )*PRIVATE KEY( BLOCK)?-----'
keys=$(git grep -nIE -e "$pem" -- . ':(exclude).gitleaks.toml' || true)
if [ -n "$keys" ]; then cut -c1-160 <<<"$keys" | head -20; echo "private key material is tracked above; remove it from the tree (and rotate the key)"; fail=1; else echo "ok"; fi
exit $fail
