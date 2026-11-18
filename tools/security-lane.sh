#!/usr/bin/env bash
# Fail closed: security tools are required for the same local and CI entry point.
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in actionlint gitleaks cargo npm python3 syft; do
  command -v "$tool" >/dev/null || { printf 'security lane requires %s\n' "$tool" >&2; exit 127; }
done
bash deploy/ci/actionlint-workflows.sh "$(pwd -P)"
bash deploy/ci/secret-scan.sh
python3 -B deploy/ci/verify-vendor-security.py --root "$(pwd -P)"
syft version | grep -qF '1.40.0'
sbom_dir="${JOPE_LANE_EVIDENCE:-target/security}"
mkdir -p "$sbom_dir"
sbom="$sbom_dir/jope.spdx.json"
syft dir:. --exclude ./target --exclude ./.git --output "spdx-json=$sbom" >/dev/null
python3 -B - "$sbom" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    document = json.load(stream)
if not document.get("spdxVersion", "").startswith("SPDX-"):
    raise SystemExit("Syft output is not a valid SPDX JSON document")
if not isinstance(document.get("packages"), list):
    raise SystemExit("Syft SPDX JSON has no package inventory")
PY
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
# The browser predictor is a standalone workspace with its own committed lockfile.
# Scan its graph under the same root policy; the root workspace scan cannot cover it.
browser_deny=(cargo deny --manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features)
browser_graph="$("${browser_deny[@]}" list --layout crate)"
browser_missing=""
for c in jope-browser-predict feat-math jope-runtime; do
  grep -qE "^$c@" <<< "$browser_graph" || browser_missing="$browser_missing $c"
done
if [ -n "$browser_missing" ]; then
  echo "cargo-deny is not seeing the standalone browser graph; missing:$browser_missing" >&2
  exit 1
fi
echo "browser deny graph: $(wc -l <<< "$browser_graph") crates, browser/runtime deps present"
"${browser_deny[@]}" check advisories bans licenses sources
# Scan the committed frontend lock graph, including build/test dependencies.
# This reads package-lock.json and does not require installing node_modules.
# GHSA-fc5q-39xc-vjpf affects picomatch <=4.0.7 (latest published on npm is 4.0.7, patched = []),
# pulled only by vite/vitest devDependencies over trusted repository globs (runtime dependencies are
# react and react-dom only). Remove this scoped exception once picomatch > 4.0.7 is published and locked.
npm_rc=0
npm --prefix web/training-dashboard audit --package-lock-only --audit-level=high || npm_rc=$?
if [ "$npm_rc" -ne 0 ]; then
  if [ "$npm_rc" -ne 1 ]; then
    exit "$npm_rc"
  fi
  npm_json_rc=0
  npm_json="$(npm --prefix web/training-dashboard audit --package-lock-only --audit-level=high --json)" || npm_json_rc=$?
  if [ "$npm_json_rc" -ne 0 ] && [ "$npm_json_rc" -ne 1 ]; then
    exit "$npm_json_rc"
  fi
  python3 -B - "$npm_json" <<'PY' || exit "$npm_rc"
import json
import sys

ALLOWED_ADVISORIES = {
    ("picomatch", "https://github.com/advisories/GHSA-fc5q-39xc-vjpf"),
}

doc = json.loads(sys.argv[1])
if doc.get("error") is not None or doc.get("auditReportVersion") != 2:
    raise SystemExit("npm audit report is invalid or returned an error")
vulns = doc.get("vulnerabilities")
if not isinstance(vulns, dict) or not vulns:
    raise SystemExit("npm audit exited non-zero without a vulnerabilities map")

memo = {}


def resolves_only_to_allowed(pkg: str, visiting: frozenset) -> bool:
    if pkg in memo:
        return memo[pkg]
    if pkg in visiting:
        return True
    entry = vulns.get(pkg)
    if not isinstance(entry, dict):
        return False
    via = entry.get("via")
    if not isinstance(via, list) or not via:
        return False
    next_visiting = visiting | {pkg}
    for item in via:
        if isinstance(item, dict):
            key = (item.get("name"), item.get("url"))
            if key not in ALLOWED_ADVISORIES:
                memo[pkg] = False
                return False
        elif isinstance(item, str):
            if not resolves_only_to_allowed(item, next_visiting):
                memo[pkg] = False
                return False
        else:
            memo[pkg] = False
            return False
    memo[pkg] = True
    return True


saw_high_or_critical = False
for pkg, entry in vulns.items():
    if not isinstance(entry, dict):
        raise SystemExit(f"malformed vulnerability entry for {pkg!r}")
    if entry.get("severity") in ("high", "critical"):
        saw_high_or_critical = True
        if not resolves_only_to_allowed(pkg, frozenset()):
            raise SystemExit(f"unreviewed high/critical npm advisory affecting {pkg}")

if not saw_high_or_critical:
    raise SystemExit("npm audit failed without high/critical entries")
print("frontend npm audit: only reviewed dev-only GHSA-fc5q-39xc-vjpf (picomatch <=4.0.7, unpatched upstream) present")
PY
fi
