#!/usr/bin/env bash
# deploy/release/bump.sh <X.Y.Z>: `just bump X.Y.Z`, the version step of a release PR
# (docs/release/README.md, "Cutting a release").
#
# It sets the Rust workspace version, the [workspace.package] version in Cargo.toml and the explicit
# versions of jope-engine and jope-campaigns, and regenerates Cargo.lock with
# `cargo update --workspace --offline`. The standalone Lambda workspaces under deploy/lambda/ have
# locks of their own, which pin the root crates they take by path (feat-math today) at the version
# they were locked with; deploy/ci/lambda-workspaces.sh sync-locks brings each such lock to the new
# version and records its new sha256 in deploy/lambda/SNAPSHOT.json. python/pyproject.toml is not
# bumped while no wheel ships.
#
# It then checks the result as the release and the hygiene lane will, and exits non-zero, saying
# which check failed, unless all of them pass: release.sh's version check on the root workspace, and
# cargo metadata --locked --offline on every Lambda workspace with the SNAPSHOT.json check
# (lambda-workspaces.sh verify). The Lambda checks also run first, before anything is changed, so a
# tree that is already inconsistent is refused as it is. A later refusal leaves the root bump in
# place, and a Lambda lock that could not be brought along as it was.
#
# The version check sources deploy/release/release.sh as release.sh requires, with bash -p (a plain
# bash is refused). The sourced shell looks commands up only in release.sh's read-only TRUSTED_PATH,
# which holds no cargo, so cargo metadata runs here and is piped in; and release.sh's jq wrapper keeps
# jq's HOME in a private directory under TMPDIR, here target/bump-tmp.
set -euo pipefail
cd "$(dirname "$0")/../.."
fail() { printf 'bump: %s\n' "$*" >&2; exit 1; }

(( $# == 1 )) || fail "usage: just bump X.Y.Z"
version=$1
[[ $version =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] \
  || fail "the version must be MAJOR.MINOR.PATCH (no leading zeros, no suffix), as release.sh's tag is: '$version'"

lambda=deploy/ci/lambda-workspaces.sh
bash "$lambda" verify \
  || fail "the Lambda workspaces under deploy/lambda/ are not consistent before the bump (above); nothing was changed"

old=$(grep -m1 -E '^version = "[^"]+"' Cargo.toml) || fail "Cargo.toml has no version line"
old=${old#version = \"}
old=${old%\"}
for manifest in Cargo.toml crates/jope-engine/Cargo.toml crates/jope-campaigns/Cargo.toml; do
  sed -i -E '0,/^version = "[^"]+"/s//version = "'"$version"'"/' "$manifest"
done
cargo update --workspace --offline

meta=$(cargo metadata --no-deps --format-version 1 --locked)
tmp=$(pwd -P)/target/bump-tmp
mkdir -p -- "$tmp"
chmod 0700 -- "$tmp"
# shellcheck disable=SC2016  # expanded by the bash -p that sources release.sh
verdict=$(printf '%s\n' "$meta" \
  | env TMPDIR="$tmp" bash -p -c 'source deploy/release/release.sh && jq -r --arg want "$1" "$VERSION_JQ"' bump "$version")
if [[ $verdict != OK ]]; then
  printf '%s\n' "$verdict" >&2
  fail "the root workspace is not at $version after the bump (above)"
fi
echo "ok: release.sh's version check: the root workspace is at $version"

bash "$lambda" sync-locks \
  || fail "the Lambda workspaces are not consistent with $version (above); the root workspace is bumped to $version, so fix them before this is committed"

if [[ $old == "$version" ]]; then
  echo "the workspace was already at $version; the Lambda workspaces are consistent with it"
  exit 0
fi
# The prose that names the old version. README.md's current-release lines (the `jope X.Y.Z` line
# among them) must follow; the rest are historical paths and examples to review.
mentions=$(git grep -n -F -e "$old" -- README.md docs/release/README.md | sed 's/^/    /' || true)
cat <<EOF
bumped the Rust workspace from $old to $version, and with it every Lambda workspace lock that pins a
root crate (deploy/lambda/SNAPSHOT.json records each one). This is a version change, not a
releasable state. The same reviewed PR must also carry:
  - README.md's current-release lines, the 'jope $version' line among them;
  - a CHANGELOG.md [$version] section (docs/release/README.md, 'Cutting a release').
These lines still name $old; historical paths such as the key-custody receipt stay:
${mentions:-    (none)}
After the merge, deploy/release/release.sh v$version re-checks everything and creates the tag (never
push one by hand).
EOF
