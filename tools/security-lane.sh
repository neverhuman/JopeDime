#!/usr/bin/env bash
# Fail closed: security tools are required for the same local and CI entry point.
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in actionlint gitleaks cargo npm; do
  command -v "$tool" >/dev/null || { printf 'security lane requires %s\n' "$tool" >&2; exit 127; }
done
bash deploy/ci/actionlint-workflows.sh "$(pwd -P)"
bash deploy/ci/secret-scan.sh
set -euo pipefail
# Was EmbarkStudios/cargo-deny-action, a Docker container action: the ci lane is
# deliberately given no docker socket, and the binary is in the image anyway.
cargo deny --version | grep -qF 'cargo-deny 0.20.2'
# --all-features is NOT cosmetic: EmbarkStudios/cargo-deny-action defaulted to it,
# which overrides deny.toml's [graph] all-features = false. Measured on this image
# (cargo-deny 0.20.2): the graph is 329 crates without it and 367 with it, and
# torch-sys and cudarc are absent from the smaller one. safetensors is in both -- it is
# asserted below anyway as a canary, not as evidence that the scope changed.
#
# ORDER MATTERS: --all-features is a top-level cargo-deny option, not a `check`
# option. `cargo deny check --all-features ...` is a parse error ("unexpected
# argument '--all-features' found"), so an earlier attempt at this restore never ran
# the gate at all. The assertion below is here because a comment could not stop that.
# `--layout crate` prints one `name@version (n): licenses` line per crate, so the
# crate name is anchored with @, not with a space.
missing=""
graph="$(cargo deny --locked --all-features list --layout crate)"
for c in torch-sys cudarc safetensors; do
  grep -qE "^$c@" <<< "$graph" || missing="$missing $c"
done
if [ -n "$missing" ]; then
  echo "cargo-deny is not seeing the optional-feature graph; missing:$missing" >&2
  echo "check that --all-features is passed BEFORE the check subcommand" >&2
  exit 1
fi
echo "deny graph: $(wc -l <<< "$graph") crates, optional-feature deps present"
# Advisories are a blocking check too: fetch/update failure must remain failure,
# and the committed deny.toml has no ignored advisories.
cargo deny --locked --all-features check advisories bans licenses sources
# Scan the committed frontend lock graph, including build/test dependencies.
# This reads package-lock.json and does not require installing node_modules.
npm --prefix web/training-dashboard audit --package-lock-only --audit-level=high
