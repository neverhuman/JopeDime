#!/usr/bin/env bash
# deploy/ci/version-consistency.sh — one version everywhere (plan "Release engineering").
#
# Asserts that
#   * every product crate carries the workspace version ([workspace.package] version), whether by
#     `version.workspace = true` or by an explicit `version = "…"` (the vendored jain-smartcluster
#     crates under crates/smartcluster/ keep their upstream 8.x versions and are excluded);
#   * python/pyproject.toml [project] version equals the workspace version;
#   * when a tag is given (argument 1, or GITHUB_REF_NAME on a tag push), it is exactly v<version>.
#
# Usage: deploy/ci/version-consistency.sh [vX.Y.Z]
# It runs in no CI lane, and nothing calls it any more. release.yml, which passed it the pushed tag,
# is dispatch-only and fails at its first step; `just bump` no longer calls it. A release is gated by
# deploy/release/release.sh's own cargo-metadata check instead: jope-cli and every product crate
# equal the tag, with the smartcluster crates and python/ left out.
# python/pyproject.toml is intentionally not bumped for v0.2.0: no wheel ships (owner decision; the
# Python surface is being retired under ADR-0006), so the workspace is 0.2.0 while pyproject stays
# 0.1.0. Run by hand, this script therefore reports the known
#   MISMATCH python/pyproject.toml: 0.1.0 != workspace 0.2.0
# until the Rust port of this check lands. The Python below is not edited or extended (ADR-0006).
set -euo pipefail
cd "$(dirname "$0")/../.."

tag=${1:-}
if [ -z "$tag" ] && [ "${GITHUB_REF_TYPE:-}" = tag ]; then tag=${GITHUB_REF_NAME:-}; fi

cargo metadata --locked --format-version 1 --no-deps > target/version-metadata.json 2>/dev/null \
  || { mkdir -p target; cargo metadata --locked --format-version 1 --no-deps > target/version-metadata.json; }

python3 - "$tag" <<'PY'
import json, re, sys, tomllib

tag = sys.argv[1]
meta = json.load(open("target/version-metadata.json"))
root = meta["workspace_root"]
with open("Cargo.toml", "rb") as f:
    workspace = tomllib.load(f)["workspace"]["package"]["version"]

failures = []
checked = 0
for pkg in meta["packages"]:
    if pkg["id"] not in meta["workspace_members"]:
        continue
    rel = pkg["manifest_path"][len(root) + 1:]
    if rel.startswith("crates/smartcluster/"):
        continue  # vendored upstream workspace, versioned by jain-smartcluster
    checked += 1
    if pkg["version"] != workspace:
        failures.append(f"{rel}: {pkg['version']} != workspace {workspace}")

with open("python/pyproject.toml", "rb") as f:
    py = tomllib.load(f)["project"]["version"]
if py != workspace:
    failures.append(f"python/pyproject.toml: {py} != workspace {workspace}")

if not re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?", workspace):
    failures.append(f"workspace version {workspace!r} is not semver")
if tag and tag != f"v{workspace}":
    failures.append(f"tag {tag} != v{workspace}")

print(f"workspace version {workspace}: {checked} product crates, python {py}" + (f", tag {tag}" if tag else ""))
for line in failures:
    print("MISMATCH " + line)
sys.exit(1 if failures else 0)
PY
