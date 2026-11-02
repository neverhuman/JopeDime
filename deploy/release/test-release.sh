#!/usr/bin/env bash
# deploy/release/test-release.sh: tests the guard and refusal paths of release.sh.
#
# It uses no docker, network, sudo or real repository. It builds a synthetic git repository, a local
# bare "origin" and a stub cargo that prints canned `cargo metadata`, all under
# $HOME/.cache/jope-test-tmp/release-selftest, the test root release.sh accepts
# (/home/ubuntu/.cache/jope-test-tmp on xbabe2, /home/runner/.cache/jope-test-tmp in the CI runner
# container). RELEASE_TEST_DIR may move them elsewhere under that root. The CI hygiene lane runs this
# script (deploy/ci/fast-lane.sh hygiene).
#
# release.sh is exercised in three ways:
#   - read statically, for its temporary files: /tmp is never a working path, TMPDIR is pinned
#     before any external command runs, every `docker run` goes through builder_container, every
#     env -i names TMPDIR, and root's sign script exports its own TMPDIR before it runs anything;
#   - run in its test mode, which stops after the guards and cannot publish (once with TMPDIR unset
#     and every command it names wrapped, to show that each one runs with the pinned TMPDIR and none
#     reads its input from /tmp, and once with a syft stand-in, which preflight refuses); the probe
#     runs a copy whose only change is TRUSTED_PATH, since release.sh ignores the PATH it is given;
#   - sourced, so that its helpers can be called directly: sums, notes, glibc, NEEDED, artifact
#     selection, manifest, the git archive source export and its verification, the shared cargo
#     home check, the syft and cosign pins (with stand-ins, never the real tools), the host-tool
#     checks (the read-only TRUSTED_PATH, the walk of every directory and symlink on the way to a
#     tool, and fixtures for a stat that the operator put first on PATH and for an operator-owned
#     directory holding bin -> /usr/bin), the root sign script (run by sh with its root paths moved
#     under the test dir, also with an empty environment as sudo leaves it), the build, binary
#     check and receipt stages (with docker and objdump stood in by functions that record every
#     container's arguments), the tampered-copy negative, and the readback (with cosign, GitHub and
#     origin stood in by functions);
#   - with hostile per-user configuration, which must not change what a check decides or what a
#     release holds: a ~/.jq, a ~/.gitconfig URL rewrite, a ~/.config/git/ignore, a URL rewrite, a
#     remote named after the given URL (url or pushurl), a TLS setting or push.followTags in the
#     repository's config, a replace ref, exported shell functions named after a command or a builtin (git, unset,
#     command, exit), bash's own variables (SHELLOPTS, BASHOPTS, BASH_COMPAT, TMOUT, FUNCNEST,
#     BASH_ENV), a bash started without -p, the loader's, tar's, objdump's, git's and docker's
#     variables, and cosign's environment. The hostile HOMEs are directories under the test dir; the
#     real ~ is never written.
#
# release.sh refuses a bash that was not started in privileged mode, so every run and every sourcing
# below uses bash -p, as its #! line does (and one case runs it through that line).
#
# Usage: deploy/release/test-release.sh   (exit 0 only when every case passes)
#
# The unit cases pass bash bodies in single quotes on purpose: they expand inside the sourced shell.
# shellcheck disable=SC2016
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
release=$here/release.sh
# The same root as release.sh's TEST_ROOT.
test_parent=$(realpath -m -- "${HOME:?HOME must be set}/.cache/jope-test-tmp")
work=$(realpath -m -- "${RELEASE_TEST_DIR:-$test_parent/release-selftest}")
case $work/ in
  "$test_parent"/?*) ;;
  *) printf 'test dir must be under %s/: %s\n' "$test_parent" "$work" >&2; exit 2 ;;
esac

# Start from a known environment: no inherited release-test overrides, isolated git config.
while IFS= read -r v; do unset "$v"; done < <(compgen -v JOPE_RELEASE_TEST || true)
repo=$work/repo origin=$work/origin.git stubs=$work/stubs evidence=$work/evidence
keys=$work/keys scratch=$work/scratch
mkdir -p -- "$work"
# Only the fixture paths this script creates are reset; nothing else under $work is touched.
rm -rf -- "$repo" "$origin" "$stubs" "$evidence" "$keys" "$scratch" "$work/tmp" "$work/keys-"* \
  "$work/evidence-"* "$work/meta-"*.json "$work/gitconfig" "$work/probe"
mkdir -p -- "$stubs" "$evidence" "$keys" "$scratch" "$work/tmp"
export TMPDIR=$work/tmp
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=$work/gitconfig
cat > "$GIT_CONFIG_GLOBAL" <<'EOF'
[user]
	name = Release Test
	email = release-test@example.invalid
[init]
	defaultBranch = main
[commit]
	gpgsign = false
[tag]
	gpgsign = false
[advice]
	detachedHead = false
EOF

# The real release public key (public material). Its sha256 is release.sh's PUBKEY_SHA256.
pubkey() {
  printf '%s\n' '-----BEGIN PUBLIC KEY-----' \
    'MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEMx44DTDE4ObFoh7otiKzAJA9fd8U' \
    'eEunkkDgvRCL74VyxIW9Q0wt49FMtTqv7aH7sj//B2uqMwPdm/MlBu45Nw==' \
    '-----END PUBLIC KEY-----'
}

changelog() {
  cat <<'EOF'
# Changelog

All notable changes to JopeDime.

## [0.2.0] - 2026-09-28

### CI, build and release

- release: fast signed host release (abcdef12)
- ci: something else (12345678)

## [0.1.0] - 2026-09-01

### Other changes

- initial import (0badc0de)

<!-- generated by git-cliff -->
EOF
}

# --- the synthetic repository and its bare origin --------------------------------------------------
git init -q "$repo"
(
  cd "$repo"
  mkdir -p crates/jope-cli/src crates/jope-engine docs/release models
  printf '[workspace]\nresolver = "2"\nmembers = ["crates/jope-cli", "crates/jope-engine"]\n\n[workspace.package]\nversion = "0.2.0"\n' > Cargo.toml
  printf '[package]\nname = "jope-cli"\nversion.workspace = true\n' > crates/jope-cli/Cargo.toml
  printf 'fn main() { println!("jope {}", env!("CARGO_PKG_VERSION")); }\n' > crates/jope-cli/src/main.rs
  printf '[package]\nname = "jope-parallel"\nversion = "0.2.0"\n' > crates/jope-engine/Cargo.toml
  printf 'target/\n' > .gitignore
  git add -- .gitignore Cargo.toml crates/jope-cli/Cargo.toml \
    crates/jope-cli/src/main.rs crates/jope-engine/Cargo.toml
  git commit -qm 'scaffold'
  printf '[toolchain]\nchannel = "1.97.1"\nprofile = "minimal"\n' > rust-toolchain.toml
  changelog > CHANGELOG.md
  pubkey > docs/release/signing.pub
  printf '{"models": []}\n' > models/manifest.json
  printf '# pinned bundles\n\nbundle-a\n  bundle-b  \n' > models/PINNED
  git add -- rust-toolchain.toml CHANGELOG.md docs/release/signing.pub \
    models/manifest.json models/PINNED
  git commit -qm 'release: 0.2.0'
  # An ignored build directory must not count as a dirty tree.
  mkdir -p target/release && printf 'cache' > target/release/cached
)
c0=$(git -C "$repo" rev-parse HEAD~1)
c1=$(git -C "$repo" rev-parse HEAD)
git init -q --bare "$origin"
git -C "$repo" push -q "$origin" main

# Key custody stand-in (test mode reads it without sudo).
printf 'not-a-real-key\n' > "$keys/signing.key"
printf 'not-a-real-password\n' > "$keys/signing.password"
pubkey > "$keys/signing.pub"

# Canned cargo metadata: meta <file> <jope-cli version> <jope-parallel version> [drop-cli]
meta() {
  jq -n --arg cli "$2" --arg par "$3" --arg root "$repo" --arg drop "${4:-}" '
    [ {name: "jope-cli", version: $cli, manifest_path: "\($root)/crates/jope-cli/Cargo.toml",
       id: "path+file://\($root)/crates/jope-cli#\($cli)"},
      {name: "jope-parallel", version: $par, manifest_path: "\($root)/crates/jope-engine/Cargo.toml",
       id: "path+file://\($root)/crates/jope-engine#jope-parallel@\($par)"},
      {name: "jain-smartcluster", version: "8.0.0",
       manifest_path: "\($root)/crates/smartcluster/jain-smartcluster/Cargo.toml",
       id: "path+file://\($root)/crates/smartcluster/jain-smartcluster#8.0.0"} ]
    | map(select($drop == "" or .name != "jope-cli")) as $packages
    | {packages: $packages, workspace_members: [$packages[].id], workspace_root: $root, version: 1}' > "$1"
}
meta "$work/meta-ok.json" 0.2.0 0.2.0
meta "$work/meta-cli-010.json" 0.1.0 0.2.0
meta "$work/meta-par-010.json" 0.2.0 0.1.0
meta "$work/meta-no-cli.json" 0.2.0 0.2.0 drop

cat > "$stubs/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$*" == "metadata --no-deps --format-version 1 --locked" ]] \
  || { printf 'stub cargo: unexpected arguments: %s\n' "$*" >&2; exit 97; }
cat -- "${FAKE_CARGO_METADATA:?}"
EOF
chmod +x "$stubs/cargo"

base=(JOPE_RELEASE_TEST=1 "JOPE_RELEASE_TEST_REMOTE=$origin" "JOPE_RELEASE_TEST_CARGO=$stubs/cargo"
  "JOPE_RELEASE_TEST_OUTROOT=$evidence" "JOPE_RELEASE_TEST_KEYDIR=$keys"
  JOPE_RELEASE_TEST_HOSTNAME=xbabe2 JOPE_RELEASE_TEST_STAMP=20260928T000000Z
  "FAKE_CARGO_METADATA=$work/meta-ok.json")
# run [VAR=value...] <command...>: the base test environment plus per-case overrides (the last wins).
run() { env "${base[@]}" "$@"; }
in_dir() { local d=$1; shift; (cd "$d" && "$@"); }
# unit '<bash body>' args...: sources release.sh and evaluates the body with the args as $1...
# $0 must differ from the sourced path, or release.sh's BASH_SOURCE[0] == $0 guard would run main.
unit() {
  local body=$1
  shift
  bash -p -c "set -euo pipefail; source \"\$1\"; shift; $body" release-unit "$release" "$@"
}
unit_with_fixture() {
  export RELEASE_TAG_TEST_FIXTURE=$1
  shift
  unit "$@"
}

pass=0
failures=()
# expect <name> <ok|refuse> <ERE the output must match> <command...>   (run from $repo)
expect() {
  local name=$1 want=$2 pattern=$3 out line rc=0 good=0
  shift 3
  out=$(cd "$repo" && "$@" 2>&1) || rc=$?
  if [[ $want == ok ]]; then
    if (( rc == 0 )); then good=1; fi
  elif (( rc != 0 )); then
    good=1
  fi
  if (( good )) && ! grep -Eq -- "$pattern" <<<"$out"; then good=0; fi
  if (( good )); then
    pass=$((pass + 1))
    printf 'ok    %s\n' "$name"
  else
    failures+=("$name")
    printf 'FAIL  %s (rc=%s, wanted %s matching /%s/)\n' "$name" "$rc" "$want" "$pattern"
    while IFS= read -r line; do printf '      | %s\n' "$line"; done <<<"$out"
  fi
}

R=(bash -p "$release")

echo "# release.sh guards (test mode; synthetic repo $repo)"
expect "baseline: every guard passes (ignored target/ is not dirt)" ok 'test mode: all guards passed' \
  run "${R[@]}" v0.2.0 "$c1"
expect "help prints usage" ok '^usage: deploy/release/release.sh' run "${R[@]}" --help
expect "no arguments" refuse 'expected <tag> <merge-commit-sha>' run "${R[@]}"
expect "one argument" refuse 'expected <tag> <merge-commit-sha>' run "${R[@]}" v0.2.0
for bad in v1.2 1.2.3 v1.2.3-rc1 v01.2.3 V1.2.3 'v1.2.3 ' v1.2.3.4 ''; do
  expect "bad tag format '$bad'" refuse 'tag must match' run "${R[@]}" "$bad" "$c1"
done
expect "short sha" refuse 'full 40-character' run "${R[@]}" v0.2.0 "${c1:0:12}"
expect "uppercase sha" refuse 'full 40-character' run "${R[@]}" v0.2.0 "${c1^^}"
expect "unknown option" refuse 'unknown option: --force' run "${R[@]}" v0.2.0 "$c1" --force
expect "test mode never publishes" refuse 'test mode never publishes' run "${R[@]}" v0.2.0 "$c1" --publish
expect "wrong host" refuse "must run on xbabe2 \\(hostname is 'xbabe1'\\)" \
  run JOPE_RELEASE_TEST_HOSTNAME=xbabe1 "${R[@]}" v0.2.0 "$c1"
expect "host xbabe20 is not xbabe2 (exact short-name match)" refuse "must run on xbabe2 \\(hostname is 'xbabe20'\\)" \
  run JOPE_RELEASE_TEST_HOSTNAME=xbabe20 "${R[@]}" v0.2.0 "$c1"
expect "host my-xbabe2 is not xbabe2" refuse "must run on xbabe2 \\(hostname is 'my-xbabe2'\\)" \
  run JOPE_RELEASE_TEST_HOSTNAME=my-xbabe2 "${R[@]}" v0.2.0 "$c1"
expect "host xbabe20.lan is not xbabe2" refuse "must run on xbabe2 \\(hostname is 'xbabe20\\.lan'\\)" \
  run JOPE_RELEASE_TEST_HOSTNAME=xbabe20.lan "${R[@]}" v0.2.0 "$c1"
expect "host xbabe2.lan (short name xbabe2) is accepted" ok 'ok: host xbabe2\.lan' \
  run JOPE_RELEASE_TEST_HOSTNAME=xbabe2.lan "${R[@]}" v0.2.0 "$c1"
expect "not the repository root" refuse 'run from the repository root' \
  in_dir "$repo/crates" run "${R[@]}" v0.2.0 "$c1"

echo "# clean tree"
printf '\n# edit\n' >> "$repo/Cargo.toml"
expect "dirty tracked file" refuse 'working tree is not clean' run "${R[@]}" v0.2.0 "$c1"
git -C "$repo" checkout -q -- Cargo.toml
printf 'new\n' > "$repo/untracked.txt"
expect "untracked file" refuse 'working tree is not clean.*' run "${R[@]}" v0.2.0 "$c1"
rm -f -- "$repo/untracked.txt"
mkdir -p "$repo/newdir/deeper" && printf 'x' > "$repo/newdir/deeper/file"
expect "untracked file in a new directory" refuse 'newdir/deeper/file' run "${R[@]}" v0.2.0 "$c1"
rm -rf -- "$repo/newdir"
printf 'staged\n' > "$repo/staged.txt" && git -C "$repo" add staged.txt
expect "staged change" refuse 'working tree is not clean' run "${R[@]}" v0.2.0 "$c1"
git -C "$repo" rm -q --cached staged.txt && rm -f -- "$repo/staged.txt"
expect "baseline again after restoring the tree" ok 'all guards passed' run "${R[@]}" v0.2.0 "$c1"

echo "# index flags that hide edits from git status"
git -C "$repo" update-index --assume-unchanged Cargo.toml
printf '\n[profile.release]\nopt-level = 0\n' >> "$repo/Cargo.toml"
expect "assume-unchanged file with a hidden edit" refuse 'REFUSED: index entries are marked assume-unchanged' \
  run "${R[@]}" v0.2.0 "$c1"
git -C "$repo" checkout -q -- Cargo.toml
expect "assume-unchanged flag even without an edit" refuse '^h Cargo\.toml$' run "${R[@]}" v0.2.0 "$c1"
git -C "$repo" update-index --no-assume-unchanged Cargo.toml
git -C "$repo" update-index --skip-worktree crates/jope-cli/src/main.rs
printf 'fn main() { println!("jope 0.2.0"); }\n' > "$repo/crates/jope-cli/src/main.rs"
expect "skip-worktree file with a hidden edit" refuse '^S crates/jope-cli/src/main\.rs$' run "${R[@]}" v0.2.0 "$c1"
git -C "$repo" update-index --no-skip-worktree crates/jope-cli/src/main.rs
git -C "$repo" checkout -q -- crates/jope-cli/src/main.rs
expect "baseline again after clearing the flags" ok 'all guards passed' run "${R[@]}" v0.2.0 "$c1"

echo "# HEAD, origin/main and tags"
expect "HEAD != given sha (parent commit)" refuse "HEAD is $c1, not the given merge commit $c0" \
  run "${R[@]}" v0.2.0 "$c0"
expect "HEAD != given sha (unknown sha)" refuse 'HEAD is .* not the given merge commit' \
  run "${R[@]}" v0.2.0 0123456789abcdef0123456789abcdef01234567
git -C "$repo" commit -q --allow-empty -m 'local only'
c2=$(git -C "$repo" rev-parse HEAD)
expect "HEAD ahead of origin/main" refuse "origin/main is $c1 after a fresh fetch, not $c2" \
  run "${R[@]}" v0.2.0 "$c2"
git -C "$repo" reset -q --hard "$c1"
git -C "$repo" push -q "$origin" "$c2:refs/heads/main"
expect "origin/main ahead of HEAD" refuse "origin/main is $c2 after a fresh fetch, not $c1" \
  run "${R[@]}" v0.2.0 "$c1"
git -C "$repo" push -q -f "$origin" "$c1:refs/heads/main"
git -C "$repo" tag v0.2.0 "$c1"
expect "tag exists locally" refuse 'tag v0.2.0 already exists locally' run "${R[@]}" v0.2.0 "$c1"
git -C "$repo" tag -d v0.2.0 >/dev/null
git -C "$repo" push -q "$origin" "$c1:refs/tags/v0.2.0"
expect "tag exists on origin only" refuse 'tag v0.2.0 already exists on' run "${R[@]}" v0.2.0 "$c1"
git -C "$repo" push -q "$origin" :refs/tags/v0.2.0
expect "origin unreachable" refuse 'fetch of main from .* failed' \
  run "JOPE_RELEASE_TEST_REMOTE=$work/no-such-origin.git" "${R[@]}" v0.2.0 "$c1"

echo "# versions (stub cargo metadata)"
expect "jope-cli version != tag" refuse 'MISMATCH jope-cli 0.1.0 != tag version 0.2.0' \
  run "FAKE_CARGO_METADATA=$work/meta-cli-010.json" "${R[@]}" v0.2.0 "$c1"
expect "another product crate != tag" refuse 'MISMATCH jope-parallel 0.1.0 != 0.2.0' \
  run "FAKE_CARGO_METADATA=$work/meta-par-010.json" "${R[@]}" v0.2.0 "$c1"
expect "no jope-cli workspace member" refuse 'ERROR expected one jope-cli workspace member, found 0' \
  run "FAKE_CARGO_METADATA=$work/meta-no-cli.json" "${R[@]}" v0.2.0 "$c1"
expect "cargo metadata fails" refuse 'cargo metadata --no-deps --format-version 1 --locked failed' \
  run "FAKE_CARGO_METADATA=$work/no-such-metadata.json" "${R[@]}" v0.2.0 "$c1"
expect "tag v0.2.1 against a 0.2.0 workspace" refuse 'MISMATCH jope-cli 0.2.0 != tag version 0.2.1' \
  run "${R[@]}" v0.2.1 "$c1"

echo "# committed release inputs (each case is a new commit pushed as origin/main)"
variant() {  # variant <name> <pattern> <shell snippet that edits the tree>
  local name=$1 pattern=$2 edit=$3 c
  # Stage only the tracked release inputs these cases are allowed to change.
  (
    cd "$repo"
    eval "$edit"
    for path in rust-toolchain.toml CHANGELOG.md docs/release/signing.pub; do
      if git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
        git add -u -- "$path"
      fi
    done
    git commit -qm "variant: $name"
  )
  c=$(git -C "$repo" rev-parse HEAD)
  if [[ $name == "toolchain channel != builder toolchain" ]]; then
    if git -C "$repo" cat-file -e "$c:unrelated.txt" 2>/dev/null; then
      echo "variant staged unrelated fixture content" >&2
      exit 1
    fi
    if ! git -C "$repo" status --porcelain --untracked-files=all | grep -Fq '?? unrelated.txt'; then
      echo "variant did not leave unrelated fixture content untracked" >&2
      exit 1
    fi
    rm -- "$repo/unrelated.txt"
  fi
  git -C "$repo" push -q -f "$origin" "$c:refs/heads/main"
  expect "$name" refuse "$pattern" run "${R[@]}" v0.2.0 "$c"
  git -C "$repo" reset -q --hard "$c1"
  git -C "$repo" push -q -f "$origin" "$c1:refs/heads/main"
}
variant "toolchain channel != builder toolchain" "rust-toolchain.toml channel '1.98.0' != builder toolchain 1.97.1" \
  "sed -i 's/1.97.1/1.98.0/' rust-toolchain.toml; printf 'must remain untracked\\n' > unrelated.txt"
variant "no rust-toolchain.toml" "rust-toolchain.toml channel '' != builder" "git rm -q rust-toolchain.toml"
variant "CHANGELOG.md missing" 'CHANGELOG.md is not committed' "git rm -q CHANGELOG.md"
variant "CHANGELOG.md without a [0.2.0] section" "no '## \\[0.2.0\\]' section" \
  "sed -i 's/^## \\[0.2.0\\]/## [0.2.0-rc1]/' CHANGELOG.md"
variant "CHANGELOG.md [0.2.0] section without entries" "no '## \\[0.2.0\\]' section" \
  "sed -i '/^- release: fast/d; /^- ci: something/d' CHANGELOG.md"
variant "signing.pub missing" 'docs/release/signing.pub is not committed' "git rm -q docs/release/signing.pub"
variant "signing.pub is a different key" 'not the custody fingerprint de84945e' \
  "printf 'x' >> docs/release/signing.pub"

echo "# key custody (test stand-in directory; the real run uses sudo -n)"
mkdir -p "$work/keys-empty" "$work/keys-nopass" "$work/keys-otherpub"
cp -- "$keys/signing.key" "$keys/signing.pub" "$work/keys-nopass/"
cp -- "$keys/signing.key" "$keys/signing.password" "$work/keys-otherpub/"
printf 'other public key\n' > "$work/keys-otherpub/signing.pub"
expect "signing key missing" refuse 'signing key .*/keys-empty/signing.key is missing' \
  run "JOPE_RELEASE_TEST_KEYDIR=$work/keys-empty" "${R[@]}" v0.2.0 "$c1"
expect "signing password missing" refuse 'signing.password is missing' \
  run "JOPE_RELEASE_TEST_KEYDIR=$work/keys-nopass" "${R[@]}" v0.2.0 "$c1"
expect "custody public key != committed key" refuse 'does not match .*/keys-otherpub/signing.pub' \
  run "JOPE_RELEASE_TEST_KEYDIR=$work/keys-otherpub" "${R[@]}" v0.2.0 "$c1"

echo "# output directory"
run_dir=$evidence/release-v0.2.0/dryrun-20260928T000000Z
mkdir -p "$evidence/release-v0.2.0/custody" && printf 'receipt' > "$evidence/release-v0.2.0/custody/r.txt"
expect "existing custody/ beside the run dir is fine" ok 'all guards passed' run "${R[@]}" v0.2.0 "$c1"
mkdir -p "$run_dir"
expect "existing empty run dir is accepted" ok 'all guards passed' run "${R[@]}" v0.2.0 "$c1"
printf 'old evidence' > "$run_dir/release.log"
expect "existing non-empty run dir is refused" refuse 'already exists and is not empty; evidence is never overwritten' \
  run "${R[@]}" v0.2.0 "$c1"
mkdir -p "$work/evidence-real" "$work/evidence-link"
ln -s "$work/evidence-real" "$work/evidence-link/release-v0.2.0"
expect "release dir is a symlink" refuse 'release-v0.2.0 is a symlink' \
  run "JOPE_RELEASE_TEST_OUTROOT=$work/evidence-link" "${R[@]}" v0.2.0 "$c1"
mkdir -p "$work/evidence-t/release-v0.2.0" && ln -s "$work/evidence-real" "$work/evidence-t/release-v0.2.0/target"
expect "shared target dir is a symlink" refuse 'target is a symlink' \
  run "JOPE_RELEASE_TEST_OUTROOT=$work/evidence-t" "${R[@]}" v0.2.0 "$c1"

echo "# the test overrides cannot leak into a real run"
expect "JOPE_RELEASE_TEST=1 outside the test root" refuse 'only honoured with JOPE_RELEASE_TEST=1 inside' \
  in_dir "$here" run "${R[@]}" v0.2.0 "$c1"
expect "override without JOPE_RELEASE_TEST=1" refuse 'only honoured with JOPE_RELEASE_TEST=1' \
  env JOPE_RELEASE_TEST_HOSTNAME=xbabe2 "${R[@]}" v0.2.0 "$c1"
expect "JOPE_RELEASE_TEST=0 with overrides" refuse 'only honoured with JOPE_RELEASE_TEST=1' \
  run JOPE_RELEASE_TEST=0 "${R[@]}" v0.2.0 "$c1"
expect "override path outside the test root" refuse 'JOPE_RELEASE_TEST_REMOTE must be under' \
  run "JOPE_RELEASE_TEST_REMOTE=$HOME/not-a-test-remote.git" "${R[@]}" v0.2.0 "$c1"
expect "override path escaping with .." refuse 'JOPE_RELEASE_TEST_OUTROOT must be under' \
  run "JOPE_RELEASE_TEST_OUTROOT=$test_parent/../escape" "${R[@]}" v0.2.0 "$c1"
expect "required override missing" refuse 'test mode requires JOPE_RELEASE_TEST_CARGO' \
  run JOPE_RELEASE_TEST_CARGO= "${R[@]}" v0.2.0 "$c1"
expect "malformed stamp override" refuse 'JOPE_RELEASE_TEST_STAMP must look like' \
  run JOPE_RELEASE_TEST_STAMP=../../x "${R[@]}" v0.2.0 "$c1"

echo "# sums (sourced write_sums / dist_names / digests_json)"
d=$scratch/dist-ok
mkdir -p "$d"
printf 'binary' > "$d/b.bin"; printf '{}' > "$d/a.json"; printf 'notes' > "$d/RELEASE-NOTES-v0.2.0.md"
printf 'stale' > "$d/SHA256SUMS-v0.2.0.txt"; printf 'other' > "$d/SHA256SUMS-v0.1.0.txt"
printf 'sig' > "$d/a.json.sig"; printf 'sig' > "$d/SHA256SUMS-v0.2.0.txt.sig"
expect "sums exclude SHA256SUMS-* and *.sig, sorted, sha256sum -c passes" ok '^names=RELEASE-NOTES-v0.2.0.md a.json b.bin ;check=ok$' \
  unit 'write_sums "$1" v0.2.0
        names=$(sed "s/^[0-9a-f]\{64\}  //" "$1/SHA256SUMS-v0.2.0.txt" | tr "\n" " ")
        (cd "$1" && sha256sum -c --strict SHA256SUMS-v0.2.0.txt >/dev/null)
        printf "names=%s;check=ok\n" "$names"' "$d"
expect "digests_json is sorted and matches sha256sum" ok '^ok$' \
  unit 'got=$(digests_json "$1" artifacts | jq -c .)
        want=$(cd "$1" && sha256sum RELEASE-NOTES-v0.2.0.md a.json b.bin | jq -Rnc "[inputs | split(\"  \") | {key: .[1], value: .[0]}] | from_entries")
        [[ $got == "$want" ]] && [[ $(jq -r "keys_unsorted | join(\",\")" <<<"$got") == "RELEASE-NOTES-v0.2.0.md,a.json,b.bin" ]] && echo ok' "$d"
mkdir -p "$scratch/dist-link" && printf 'x' > "$scratch/dist-link/a.json" && ln -s a.json "$scratch/dist-link/b.json"
expect "sums refuse a symlink" refuse 'refusing symlink in .*: b.json' unit 'write_sums "$1" v0.2.0' "$scratch/dist-link"
mkdir -p "$scratch/dist-dir/sub" && printf 'x' > "$scratch/dist-dir/a.json"
expect "sums refuse a subdirectory" refuse 'refusing non-regular file .*: sub' unit 'write_sums "$1" v0.2.0' "$scratch/dist-dir"
mkdir -p "$scratch/dist-name" && printf 'x' > "$scratch/dist-name/has space.txt"
expect "sums refuse an unsafe file name" refuse 'refusing file name outside' unit 'write_sums "$1" v0.2.0' "$scratch/dist-name"
mkdir -p "$scratch/dist-empty" && printf 'x' > "$scratch/dist-empty/SHA256SUMS-v0.2.0.txt" && printf 's' > "$scratch/dist-empty/x.sig"
expect "sums refuse a dist with nothing to sum" refuse 'no release files to checksum' unit 'write_sums "$1" v0.2.0' "$scratch/dist-empty"
ln -s "$d" "$scratch/dist-symlinked"
expect "sums refuse a symlinked dist directory" refuse 'not a real directory' unit 'write_sums "$1" v0.2.0' "$scratch/dist-symlinked"

echo "# release notes (sourced extract_notes)"
changelog > "$scratch/CHANGELOG.md"
expect "notes: the [0.2.0] section only, without the cliff footer" ok '^first=## \[0.2.0\] - 2026-09-28;entries=2;other=0;footer=0$' \
  unit 'n=$(extract_notes 0.2.0 < "$1")
        printf "first=%s;entries=%s;other=%s;footer=%s\n" "$(head -n1 <<<"$n")" "$(grep -c "^- " <<<"$n")" \
          "$(grep -c "initial import" <<<"$n" || true)" "$(grep -c "git-cliff" <<<"$n" || true)"' "$scratch/CHANGELOG.md"
expect "notes: the last section also works" ok '^entries=1$' \
  unit 'printf "entries=%s\n" "$(extract_notes 0.1.0 < "$1" | grep -c "^- ")"' "$scratch/CHANGELOG.md"
expect "notes: missing section" refuse '^rc=1$' unit 'rc=0; extract_notes 0.3.0 < "$1" >/dev/null || rc=$?; echo "rc=$rc"; exit "$rc"' "$scratch/CHANGELOG.md"
expect "notes: 0.2.0 does not match [10.2.0] or [0.2.0-rc1]" refuse '^rc=1$' \
  unit 'rc=0; printf "## [10.2.0] - x\n- a\n## [0.2.0-rc1] - y\n- b\n" | extract_notes 0.2.0 >/dev/null || rc=$?; echo "rc=$rc"; exit "$rc"'

echo "# binary checks (sourced max_glibc / version_le / needed_libs / gpu_libs / binary_prints_version)"
cat > "$scratch/objdump.txt" <<'EOF'
0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.2.5) memcpy
0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.34) __libc_start_main
0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.14) memcpy
0000000000000000      DO *UND*  0000000000000000 (GLIBC_PRIVATE) _rtld_global
  required from libc.so.6:
    0x06969194 0x00 05 GLIBC_2.14
EOF
expect "max_glibc picks 2.34 (ignores GLIBC_PRIVATE)" ok '^2\.34$' unit 'max_glibc < "$1"' "$scratch/objdump.txt"
expect "max_glibc is empty without versions" ok '^empty$' unit 'v=$(printf "no versions\n" | max_glibc); [[ -z $v ]] && echo empty'
expect "version_le 2.34 <= 2.36" ok '^yes$' unit 'version_le 2.34 2.36 && echo yes'
expect "version_le 2.36 <= 2.36" ok '^yes$' unit 'version_le 2.36 2.36 && echo yes'
expect "version_le 2.4 <= 2.36 (version order, not string order)" ok '^yes$' unit 'version_le 2.4 2.36 && echo yes'
expect "version_le 2.39 > 2.36" refuse '^no$' unit 'version_le 2.39 2.36 && echo yes || { echo no; exit 1; }'
cat > "$scratch/objdump-p.txt" <<'EOF'
Dynamic Section:
  NEEDED               libgcc_s.so.1
  NEEDED               libcurl.so.4
  NEEDED               libc.so.6
  NEEDED               LIBNVIDIA-ml.so.1
  NEEDED               libtorch_cpu.so
  NEEDED               libcudart.so.12
  RUNPATH              $ORIGIN
EOF
expect "NEEDED parsing and GPU detection (libcurl is not CUDA)" ok '^needed=6;gpu=LIBNVIDIA-ml.so.1,libtorch_cpu.so,libcudart.so.12$' \
  unit 'n=$(needed_libs < "$1"); g=$(gpu_libs <<<"$n" | paste -sd, -)
        printf "needed=%s;gpu=%s\n" "$(wc -l <<<"$n")" "$g"' "$scratch/objdump-p.txt"
expect "no GPU libraries in a plain glibc binary" ok '^none$' \
  unit 'g=$(printf "libc.so.6\nlibm.so.6\nlibgcc_s.so.1\n" | gpu_libs); [[ -z $g ]] && echo none'
mkdir -p "$scratch/run"
printf '#!/bin/sh\nprintf "jope 0.2.0\\nmore\\n"\n' > "$scratch/jope-good" && chmod +x "$scratch/jope-good"
printf '#!/bin/sh\nprintf "jope 0.2.0-dirty\\n"\n' > "$scratch/jope-bad" && chmod +x "$scratch/jope-bad"
printf '#!/bin/sh\nexit 3\n' > "$scratch/jope-fail" && chmod +x "$scratch/jope-fail"
expect "binary prints 'jope 0.2.0' first" ok '^jope 0\.2\.0$' unit 'binary_prints_version "$1" 0.2.0 "$2"' "$scratch/jope-good" "$scratch/run"
expect "binary printing 'jope 0.2.0-dirty' is refused" refuse 'jope 0\.2\.0-dirty' unit 'binary_prints_version "$1" 0.2.0 "$2"' "$scratch/jope-bad" "$scratch/run"
expect "binary exiting non-zero is refused" refuse '^rc=1$' unit 'rc=0; binary_prints_version "$1" 0.2.0 "$2" || rc=$?; echo "rc=$rc"; exit "$rc"' "$scratch/jope-fail" "$scratch/run"
printf '#!/bin/sh\necho "jope 0.2.0"\necho "tmpdir=${TMPDIR-unset}"\n' > "$scratch/jope-tmpdir" && chmod +x "$scratch/jope-tmpdir"
expect "env -i: the binary check hands the binary the pinned TMPDIR by name" ok "^tmpdir=$TMPDIR\$" \
  unit 'binary_prints_version "$1" 0.2.0 "$2"' "$scratch/jope-tmpdir" "$scratch/run"

echo "# cargo artifact selection (sourced select_executable / map_target_path)"
m=/src/crates/jope-cli/Cargo.toml
art() {  # art <manifest> <name> <executable|null> <test:true|false>
  jq -nc --arg m "$1" --arg n "$2" --argjson e "$3" --argjson t "$4" \
    '{reason: "compiler-artifact", manifest_path: $m, target: {name: $n, kind: ["bin"]},
      profile: {test: $t}, executable: $e, fresh: false}'
}
{
  printf 'warning: a non-JSON line\n'
  art /src/crates/feat-math/Cargo.toml feat-math null false
  art "$m" jope '"/target/release/jope"' false
  art "$m" jope '"/target/release/deps/jope-0123"' true
  jq -nc '{reason: "build-finished", success: true}'
} > "$scratch/ok.jsonl"
{ art "$m" jope '"/target/release/jope"' false; jq -nc '{reason: "build-finished", success: false}'; } > "$scratch/failed.jsonl"
{ art "$m" jope '"/target/release/jope"' false; art "$m" jope '"/target/x/jope"' false; jq -nc '{reason: "build-finished", success: true}'; } > "$scratch/two.jsonl"
art "$m" jope '"/target/release/jope"' false > "$scratch/unfinished.jsonl"
expect "select the one jope executable (test/other artifacts ignored)" ok '^/target/release/jope$' \
  unit 'select_executable "$1" /src/crates/jope-cli/Cargo.toml jope' "$scratch/ok.jsonl"
expect "select refuses a failed build" refuse 'successful completed build' \
  unit 'select_executable "$1" /src/crates/jope-cli/Cargo.toml jope' "$scratch/failed.jsonl"
expect "select refuses an unfinished build" refuse 'successful completed build' \
  unit 'select_executable "$1" /src/crates/jope-cli/Cargo.toml jope' "$scratch/unfinished.jsonl"
expect "select refuses two candidates" refuse 'found 2' \
  unit 'select_executable "$1" /src/crates/jope-cli/Cargo.toml jope' "$scratch/two.jsonl"
expect "select refuses another package's manifest" refuse 'found 0' \
  unit 'select_executable "$1" /src/crates/other/Cargo.toml jope' "$scratch/ok.jsonl"
expect "map /target path to the host target dir" ok '^/home/x/release-v0.2.0/target/release/jope$' \
  unit 'target_root=/home/x/release-v0.2.0/target; map_target_path /target/release/jope'
expect "map refuses paths outside /target" refuse 'not under /target' unit 'target_root=/x; map_target_path /work/release/jope'
expect "map refuses .. escapes" refuse 'not under /target' unit 'target_root=/x; map_target_path /target/../etc/passwd'

echo "# release manifest (sourced write_manifest, against the synthetic commit)"
md=$scratch/dist-manifest
mkdir -p "$md"
printf 'binary' > "$md/jope-v0.2.0-x86_64-unknown-linux-gnu"; printf 'notes' > "$md/RELEASE-NOTES-v0.2.0.md"
printf 'sig' > "$md/SHA256SUMS-v0.2.0.txt.sig"
models_sha=$(git -C "$repo" cat-file blob "$c1:models/manifest.json" | sha256sum | cut -d' ' -f1)
expect "manifest keys, sorted artifacts, null datasets, pinned models" ok '^ok$' \
  unit 'cd "$3" && write_manifest "$1" v0.2.0 "$2" 0.2.0 2.34
        jq -e --arg sha "$2" --arg models "$4" "
          .schema == \"jope.release-manifest/v1\" and .tag == \"v0.2.0\" and .git_sha == \$sha
          and .crate_version == \"0.2.0\" and .max_glibc == \"2.36\" and .observed_max_glibc == \"2.34\"
          and (.builder_image | startswith(\"rust@sha256:\"))
          and .models_manifest_sha256 == \$models and .datasets_manifest_sha256 == null
          and .models_pinned == [\"bundle-a\", \"bundle-b\"]
          and (.artifacts | keys_unsorted) == [\"RELEASE-NOTES-v0.2.0.md\", \"jope-v0.2.0-x86_64-unknown-linux-gnu\"]
          and (.artifacts | keys_unsorted | map(test(\"sig$|^SHA256SUMS\")) | any | not)
        " "$1/release-manifest-v0.2.0.json" >/dev/null && echo ok' "$md" "$c1" "$repo" "$models_sha"
expect "manifest refuses to overwrite itself" refuse 'already exists' \
  unit 'cd "$3" && write_manifest "$1" v0.2.0 "$2" 0.2.0 2.34' "$md" "$c1" "$repo"

echo "# source export (sourced export_source / verify_export; git archive and tar, no docker)"
# A commit with the release tree but an old committer date: without tar -m its files would carry
# 2001 mtimes and could look older than an earlier build in the shared target/.
c_old=$(GIT_COMMITTER_DATE='2001-01-01T00:00:00Z' GIT_AUTHOR_DATE='2001-01-01T00:00:00Z' \
  git -C "$repo" commit-tree "$c1^{tree}" -p "$c1" -m 'release tree, old date')
n_entries=$(git -C "$repo" ls-tree -r "$c1" | wc -l)
# The live checkout hides an assume-unchanged edit and an ignored .cargo/config.toml from git status.
exclude_file=$(cd "$repo" && realpath -m -- "$(git rev-parse --git-path info/exclude)")
mkdir -p -- "$(dirname "$exclude_file")"
cp -- "$exclude_file" "$scratch/exclude.bak" 2>/dev/null || : > "$scratch/exclude.bak"
git -C "$repo" update-index --assume-unchanged Cargo.toml
printf '\n[profile.release]\nopt-level = 0\n' >> "$repo/Cargo.toml"
mkdir -p "$repo/.cargo"
printf '[build]\nrustflags = ["-C", "target-cpu=native"]\n' > "$repo/.cargo/config.toml"
printf '.cargo/\n' >> "$exclude_file"
t0=$(date +%s)
expect "export is the commit's tree: hidden live edits, ignored files and old mtimes stay out" ok \
  "^entries=$n_entries;verified=true;cargo_toml=committed;dotcargo=absent;target=absent;fresh_mtime=yes\$" \
  unit 'run_dir=$1 sha=$2; mkdir -p "$run_dir/tmp"; export_source >/dev/null 2>&1
        s=$run_dir/src j=$run_dir/source-export.json
        if cmp -s "$s/Cargo.toml" <(git cat-file blob "$sha:Cargo.toml"); then ct=committed; else ct=live; fi
        if [[ -e $s/.cargo ]]; then dc=present; else dc=absent; fi
        if [[ -e $s/target ]]; then tg=present; else tg=absent; fi
        if (( $(stat -c %Y "$s/Cargo.toml") >= $3 )); then fm=yes; else fm=no; fi
        printf "entries=%s;verified=%s;cargo_toml=%s;dotcargo=%s;target=%s;fresh_mtime=%s\n" \
          "$(jq -r .entries "$j")" "$(jq -r .verified "$j")" "$ct" "$dc" "$tg" "$fm"' \
  "$scratch/export-ok" "$c_old" "$t0"
rm -rf -- "$repo/.cargo"
cp -- "$scratch/exclude.bak" "$exclude_file"
git -C "$repo" update-index --no-assume-unchanged Cargo.toml
git -C "$repo" checkout -q -- Cargo.toml
expect "an untouched copy of the export verifies" ok "^$n_entries\$" \
  unit 'verify_export "$1" "$2" "$3"' "$c_old" "$scratch/export-ok/src" "$scratch/export-ok.check"
tamper_n=0
tamper() {  # tamper <name> <pattern> <snippet run inside a copy of the verified export>
  local name=$1 pattern=$2 edit=$3 d
  tamper_n=$((tamper_n + 1))
  d=$scratch/tamper-$tamper_n
  cp -a -- "$scratch/export-ok/src" "$d"
  (cd "$d" && eval "$edit")
  expect "$name" refuse "$pattern" unit 'verify_export "$1" "$2" "$3"' "$c_old" "$d" "$d.check"
}
tamper "export check refuses a changed file" 'is not exactly the tree of' 'printf x >> Cargo.toml'
tamper "export check refuses an extra file (.cargo/config.toml)" 'is not exactly the tree of' \
  'mkdir .cargo && printf "[build]\n" > .cargo/config.toml'
tamper "export check refuses a missing file" 'is not exactly the tree of' 'rm rust-toolchain.toml'
tamper "export check refuses an added executable bit" 'is not exactly the tree of' 'chmod u+x Cargo.toml'
tamper "export check refuses a symlink in place of a file" 'is not exactly the tree of' \
  'rm CHANGELOG.md && ln -s Cargo.toml CHANGELOG.md'
tamper "export check refuses an empty directory" 'empty directories that are not in' 'mkdir -p .cargo'
tamper "export check refuses a FIFO" 'the export holds special files' 'mkfifo pipe'
blob_link=$(printf 'Cargo.toml' | git -C "$repo" hash-object -w --stdin)
blob_exec=$(printf '#!/bin/sh\necho hi\n' | git -C "$repo" hash-object -w --stdin)
tree_lx=$({ git -C "$repo" ls-tree "$c1"
  printf '120000 blob %s\tlink-to-cargo\n100755 blob %s\trun.sh\n' "$blob_link" "$blob_exec"; } | git -C "$repo" mktree)
c_lx=$(git -C "$repo" commit-tree "$tree_lx" -p "$c1" -m 'a symlink and an executable')
expect "export keeps a committed symlink and executable bit, and verifies them" ok '^link=Cargo.toml;exec=yes;verified=true$' \
  unit 'run_dir=$1 sha=$2; mkdir -p "$run_dir/tmp"; export_source >/dev/null 2>&1
        if [[ -x $run_dir/src/run.sh ]]; then x=yes; else x=no; fi
        printf "link=%s;exec=%s;verified=%s\n" "$(readlink "$run_dir/src/link-to-cargo")" "$x" \
          "$(jq -r .verified "$run_dir/source-export.json")"' "$scratch/export-lx" "$c_lx"
tree_sub=$({ git -C "$repo" ls-tree "$c1"; printf '160000 commit %s\tvendor-sub\n' "$c0"; } | git -C "$repo" mktree)
c_sub=$(git -C "$repo" commit-tree "$tree_sub" -p "$c1" -m 'a submodule')
expect "export check refuses a commit with a submodule" refuse "has a commit entry at 'vendor-sub'" \
  unit 'verify_export "$1" "$2" "$3"' "$c_sub" "$scratch/export-ok/src" "$scratch/export-sub.check"
# A new stamp: the output-directory cases above left evidence in the default one.
expect "guards pass again after the export cases restored the checkout" ok 'all guards passed' \
  run JOPE_RELEASE_TEST_STAMP=20260928T000001Z "${R[@]}" v0.2.0 "$c1"

echo "# temporary files: TMPDIR is pinned before any external command, and nothing goes through /tmp"
# With TMPDIR unset, bash writes a here-string or here-document larger than a pipe buffer to
# /tmp/sh-thd.* (the real cargo metadata is about 150 KB), and sort, mktemp, git and others use /tmp.
static_tmp_check() {
  local refs first pre top line seg kind
  local -a segs words
  # 1. /tmp is a path only in the refusal to run from a checkout under /tmp and in the name of the
  #    containers' /tmp, which builder_container covers with the run's own directory (comment lines
  #    aside; $run_dir/tmp and ./tmp are the run directory's own tmp/).
  refs=$(grep -nE '(^|[^[:alnum:]_.}/-])/tmp([^[:alnum:]_.-]|$)' "$release" | grep -vE '^[0-9]+:[[:space:]]*#' || true)
  if [[ $(grep -cvE '^[0-9]+:(  \[\[ \$here/ != /tmp/\* \]\] \|\| die "refusing to run from under /tmp: \$here"|readonly CONTAINER_SYSTEM_TMP=/tmp)$' <<<"$refs") -ne 0 \
    || $(wc -l <<<"$refs") -ne 2 ]]; then
    printf 'release.sh uses /tmp outside its refusal and the containers'"'"' /tmp:\n%s\n' "$refs"
    return 1
  fi
  # 2. main() pins TMPDIR before it does anything else.
  first=$(awk '/^main\(\) \{$/ { m = 1; next } m && NF && $1 !~ /^#/ { print; exit }' "$release")
  [[ $first =~ ^[[:space:]]*pin_tmpdir[[:space:]]*$ ]] || { printf 'main() starts with: %s\n' "$first"; return 1; }
  # 3. pin_tmpdir runs only builtins and keywords before it exports TMPDIR.
  pre=$(awk '/^pin_tmpdir\(\) \{$/ { p = 1; next }
    p && /^[[:space:]]*export TMPDIR=/ { found = 1; exit }
    p && /^\}$/ { exit }
    p { print }
    END { if (!found) print "NO-EXPORT" }' "$release")
  [[ $pre != *NO-EXPORT* ]] || { printf 'pin_tmpdir does not export TMPDIR\n'; return 1; }
  while IFS= read -r line; do
    [[ ! $line =~ ^[[:space:]]*(#|$) ]] || continue
    if [[ $line == *'$('* || $line == *'`'* || $line == *'<('* || $line == *'>('* || $line == *'|'* || $line == *'<<'* ]]; then
      printf 'pin_tmpdir runs a command before it exports TMPDIR: %s\n' "$line"
      return 1
    fi
    IFS=';' read -ra segs <<<"$line"
    for seg in "${segs[@]}"; do
      read -ra words <<<"$seg"
      while (( ${#words[@]} )) && [[ ${words[0]} =~ ^(if|then|else|elif|fi|while|until|do|done|!)$ ]]; do
        words=("${words[@]:1}")
      done
      (( ${#words[@]} )) || continue
      [[ ! ${words[0]} =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
      kind=$(type -t -- "${words[0]}" || true)
      [[ $kind == builtin || $kind == keyword ]] \
        || { printf 'pin_tmpdir runs %s (%s) before it exports TMPDIR\n' "${words[0]}" "${kind:-not found}"; return 1; }
    done
  done <<<"$pre"
  # 4. Outside function bodies nothing runs but builtins: no command or process substitution, and
  #    a here-document there feeds only the read builtin.
  top=$(awk -v q="'" '
    hd != "" { if ($0 == hd) hd = ""; next }
    { starts = ""; if (match($0, "<<" q "?[A-Z]+" q "?")) { starts = substr($0, RSTART + 2, RLENGTH - 2); gsub(q, "", starts) } }
    f == "" && /^[A-Za-z_][A-Za-z0-9_]*\(\) [{(]$/ { f = (substr($0, length($0)) == "{") ? "}" : ")"; hd = starts; next }
    f != "" { if ($0 == f) f = ""; hd = starts; next }
    /^[A-Za-z_][A-Za-z0-9_]*\(\) \{ .*\}$/ { next }
    /^[[:space:]]*#/ { next }
    { print; hd = starts }' "$release")
  while IFS= read -r line; do
    if [[ $line == *'$('* || $line == *'`'* || $line == *'<('* || $line == *'>('* ]]; then
      printf 'release.sh runs a command at the top level: %s\n' "$line"
      return 1
    fi
    if [[ $line == *"<<'"* && $line != 'IFS= read '* ]]; then
      printf 'a top-level here-document feeds something other than read: %s\n' "$line"
      return 1
    fi
  done <<<"$top"
  echo ok
}
expect "static: /tmp is never a working path; main() pins TMPDIR first; only builtins run before it" ok '^ok$' \
  static_tmp_check

# The dynamic probe: every word of release.sh that names a command on PATH runs through a wrapper
# that logs the command, its TMPDIR and what its standard input is (/proc/self/fd/0), then runs the
# real command. release.sh runs in test mode with TMPDIR unset and a cargo metadata document larger
# than a pipe buffer. release.sh looks commands up only in its read-only TRUSTED_PATH, whatever PATH
# it is given, so the probe runs a copy of it whose one difference is that line, with the wrappers'
# directory first; the wrappers run the real commands from the original TRUSTED_PATH.
jq '.packages[0].description = ("x" * 200000)' "$work/meta-ok.json" > "$work/meta-big.json"
probe=$work/probe
mkdir -p "$probe/bin"
trusted_path=$(sed -n 's/^readonly TRUSTED_PATH=\([^[:space:]]*\)$/\1/p' "$release")
sed "s|^readonly TRUSTED_PATH=.*\$|readonly TRUSTED_PATH=$probe/bin:$trusted_path|" "$release" > "$probe/release.sh"
expect "probe: the copy differs from release.sh only in TRUSTED_PATH, which puts the wrappers first" ok \
  "^changed=2;trusted=/usr/sbin:/usr/bin:/sbin:/bin\$" \
  bash -c 'printf "changed=%s;trusted=%s\n" "$(diff -- "$1" "$2" | grep -c "^[<>]")" "$3"' probe-diff "$release" "$probe/release.sh" "$trusted_path"
cat > "$probe/wrap" <<'EOF'
#!/bin/sh
name=${0##*/}
PATH=$PROBE_REAL_PATH
export PATH
fd0=$(readlink /proc/self/fd/0 2>/dev/null) || fd0=none
printf '%s\t%s\t%s\n' "$name" "${TMPDIR-<unset>}" "$fd0" >> "$PROBE_LOG"
exec "$name" "$@"
EOF
chmod 0755 "$probe/wrap"
while IFS= read -r word; do
  if type -P -- "$word" >/dev/null; then ln -s ../wrap "$probe/bin/$word"; fi
done < <(grep -oE '[a-z][a-z0-9_+-]*' "$release" | LC_ALL=C sort -u)
# The directory release.sh pins in test mode (a real run pins ~/jope-evidence/release-tmp).
pinned=$HOME/.cache/jope-test-tmp/release-tmp
probe_run() {
  local rc=0 log=$probe/log mode
  : > "$log"
  env -u TMPDIR "${base[@]}" "FAKE_CARGO_METADATA=$work/meta-big.json" JOPE_RELEASE_TEST_STAMP=20260928T000002Z \
    "PROBE_LOG=$log" "PROBE_REAL_PATH=$trusted_path" "$BASH" -p "$probe/release.sh" v0.2.0 "$c1" \
    </dev/null >"$probe/out" 2>&1 || rc=$?
  mode=$(stat -c %a -- "$pinned" 2>/dev/null || echo missing)
  awk -F '\t' -v want="$pinned" -v rc="$rc" -v mode="$mode" '
    { n++; if ($2 != want) other++; if ($3 ~ /^\/tmp\//) viatmp++
      if ($1 == "jq") { jq++; if ($3 !~ /^pipe:/) jqfile++ } }
    END { printf "rc=%s;commands=%s;tmpdir_not_pinned=%d;stdin_under_tmp=%d;jq=%s;jq_stdin_not_a_pipe=%d;pinned_mode=%s\n",
            rc, (n ? "logged" : "none"), other, viatmp, (jq ? "logged" : "none"), jqfile, mode }' "$log"
  # For a failure: what jq read, the first commands that did not see the pinned TMPDIR, the output.
  awk -F '\t' '$1 == "jq"' "$log" | head -n 3
  grep -vF -- "$pinned" "$log" | head -n 5 || true
  cat -- "$probe/out"
}
expect "TMPDIR unset: every command runs with the pinned 0700 TMPDIR, none reads from /tmp, jq reads a pipe" ok \
  '^rc=0;commands=logged;tmpdir_not_pinned=0;stdin_under_tmp=0;jq=logged;jq_stdin_not_a_pipe=0;pinned_mode=700$' \
  probe_run
# Every command the probe saw release.sh run is one that a real run checks (require_tools).
expect "every command release.sh ran in the probe is a checked host tool (HOST_TOOLS)" ok '^ran=[1-9][0-9]*;unlisted=none$' \
  unit 'mapfile -t ran < <(cut -f1 "$1" | LC_ALL=C sort -u)
        extra=()
        for c in "${ran[@]}"; do
          listed=0
          for t in "${HOST_TOOLS[@]}"; do [[ $c != "$t" ]] || listed=1; done
          (( listed )) || extra+=("$c")
        done
        printf "ran=%s;unlisted=%s\n" "${#ran[@]}" "${extra[*]:-none}"' "$probe/log"

echo "# shared cargo home (sourced check_cargo_home)"
ch=$scratch/cargo-home
mkdir -p "$ch/ok/registry/index" "$ch/ok/git/db" "$ch/empty"
: > "$ch/ok/.package-cache"; : > "$ch/ok/.package-cache-mutate"; : > "$ch/ok/.global-cache"
expect "cargo home holding only cargo's caches is accepted" ok '^ok$' unit 'check_cargo_home "$1" && echo ok' "$ch/ok"
expect "an empty cargo home is accepted" ok '^ok$' unit 'check_cargo_home "$1" && echo ok' "$ch/empty"
for bad in config.toml config credentials.toml credentials env .crates.toml; do
  mkdir -p "$ch/bad-$bad" && printf 'x\n' > "$ch/bad-$bad/$bad"
  expect "cargo home refuses $bad" refuse "unexpected entry in the shared cargo home: .*/$bad \\(" \
    unit 'check_cargo_home "$1"' "$ch/bad-$bad"
done
mkdir -p "$ch/bad-bin/bin"
expect "cargo home refuses bin/" refuse 'unexpected entry in the shared cargo home: .*/bin \(' \
  unit 'check_cargo_home "$1"' "$ch/bad-bin"
mkdir -p "$ch/bad-link" "$ch/real-registry" && ln -s "$ch/real-registry" "$ch/bad-link/registry"
expect "cargo home refuses a symlinked registry" refuse 'refusing a symlink in the shared cargo home' \
  unit 'check_cargo_home "$1"' "$ch/bad-link"
mkdir -p "$ch/bad-kind/.package-cache"
expect "cargo home refuses a directory where cargo keeps a lock file" refuse 'expected a regular file' \
  unit 'check_cargo_home "$1"' "$ch/bad-kind"

echo "# sign step (SIGN_SCRIPT run by sh without sudo; its /root and /etc/jope-release paths moved under the test dir)"
sd=$scratch/sign
mkdir -p "$sd/run" "$sd/keys" "$sd/dist"
printf 'not-a-real-key\n' > "$sd/keys/signing.key"
printf 'not-a-real-password\n' > "$sd/keys/signing.password"
printf 'sums\n' > "$sd/dist/SHA256SUMS-v0.2.0.txt"
cat > "$sd/fake-cosign" <<'EOF'
#!/bin/sh
# Records its argv, its working directory, its TMPDIR (and that directory's mode) and whether
# COSIGN_PASSWORD is the stand-in password.
log=${FAKE_COSIGN_LOG:?}
printf '%s\n' "$@" > "$log.argv"
pwd > "$log.cwd"
printf '%s\n' "${TMPDIR-unset}" > "$log.tmpdir"
if [ -d "${TMPDIR-}" ]; then stat -c %a "$TMPDIR"; else echo none; fi > "$log.tmpmode"
if [ "${COSIGN_PASSWORD-}" = not-a-real-password ]; then echo match; else echo mismatch; fi > "$log.password"
out=
while [ $# -gt 0 ]; do
  if [ "$1" = --output-signature ]; then out=$2; shift; fi
  shift
done
printf 'fake-signature\n' > "$out"
EOF
chmod 0755 "$sd/fake-cosign"
fake_sha=$(sha256sum < "$sd/fake-cosign")
fake_sha=${fake_sha%% *}
SIGN_PREP='s=${SIGN_SCRIPT//\/root\/jope-release-sign/$1/run/jope-release-sign}
  s=${s//\/etc\/jope-release/$1/keys}
  [[ $s != *"/etc/jope-release"* && $s != *" /root/"* ]] || { echo "path substitution failed"; exit 9; }
  rc=0'
expect "sign: the pinned copy runs in the key dir, the password is only in its environment" ok \
  '^rc=0;sig=fake-signature;password=match;argv_has_password=no;tlog_off=yes;cwd=keys;leftover=0$' \
  unit "$SIGN_PREP"'
        FAKE_COSIGN_LOG=$1/log sh -c "$s" sh "$1/dist/SHA256SUMS-v0.2.0.txt.sig" "$1/dist/SHA256SUMS-v0.2.0.txt" \
          "$1/fake-cosign" "$2" || rc=$?
        if grep -q -- not-a-real-password "$1/log.argv"; then a=yes; else a=no; fi
        if grep -qx -- --tlog-upload=false "$1/log.argv"; then t=yes; else t=no; fi
        printf "rc=%s;sig=%s;password=%s;argv_has_password=%s;tlog_off=%s;cwd=%s;leftover=%s\n" "$rc" \
          "$(cat "$1/dist/SHA256SUMS-v0.2.0.txt.sig")" "$(cat "$1/log.password")" "$a" "$t" \
          "$(basename "$(cat "$1/log.cwd")")" "$(find "$1/run" -mindepth 1 | wc -l)"' "$sd" "$fake_sha"
expect "sign: a cosign that does not match the pinned digest is never run" ok '^rc=3;ran=no;refused=yes;leftover=0$' \
  unit "$SIGN_PREP"'
        out=$(FAKE_COSIGN_LOG=$1/log2 sh -c "$s" sh "$1/dist/x.sig" "$1/dist/SHA256SUMS-v0.2.0.txt" \
          "$1/fake-cosign" "$COSIGN_SHA256" 2>&1) || rc=$?
        if [[ -e $1/log2.argv ]]; then r=yes; else r=no; fi
        if grep -q "not the pinned $COSIGN_SHA256; refusing to run it as root" <<<"$out"; then m=yes; else m=no; fi
        printf "rc=%s;ran=%s;refused=%s;leftover=%s\n" "$rc" "$r" "$m" "$(find "$1/run" -mindepth 1 | wc -l)"' \
  "$sd" "$fake_sha"
# sudo resets the environment, so root's shell inherits no TMPDIR; env -i stands in for that here.
expect "sign: with the environment reset (as sudo does), cosign gets a 0700 TMPDIR inside root's work directory" ok \
  '^rc=0;tmpdir=work-dir;mode=700;leftover=0$' \
  unit "$SIGN_PREP"'
        env -i PATH=/usr/bin:/bin FAKE_COSIGN_LOG="$1/log3" sh -c "$s" sh "$1/dist/y.sig" \
          "$1/dist/SHA256SUMS-v0.2.0.txt" "$1/fake-cosign" "$2" || rc=$?
        t=$(cat "$1/log3.tmpdir")
        case $t in "$1"/run/jope-release-sign.*/tmp) w=work-dir ;; *) w="other:$t" ;; esac
        printf "rc=%s;tmpdir=%s;mode=%s;leftover=%s\n" "$rc" "$w" "$(cat "$1/log3.tmpmode")" \
          "$(find "$1/run" -mindepth 1 | wc -l)"' "$sd" "$fake_sha"
# The same rule, read from the source: every sudo runs the checked absolute path (host_path) of test,
# cat, chown, chmod or sh, never a name that sudo would look up in its own secure_path; sh runs only
# the sign script; and the sign script names no /tmp, sets root's PATH to TRUSTED_PATH first, gives
# mktemp an absolute /root template and exports TMPDIR inside its work directory before it runs
# anything else.
expect "static: sudo runs only the checked paths of test, cat, chown, chmod and sh (the sign script, which sets PATH and TMPDIR first)" ok '^ok$' \
  unit 'code=$(grep -vE "^[[:space:]]*#" "$1")
        n_sudo=$(grep -oE "(^|[;&|(]|then |else )[[:space:]]*sudo -n " <<<"$code" | wc -l)
        sudos=$(grep -oE "(^|[;&|(]|then |else )[[:space:]]*sudo -n \"\\\$\{host_path\[[a-z]+\]:\?[^}]*\}\"" <<<"$code" \
          | sed -E "s/.*host_path\[([a-z]+)\].*/\1/" | LC_ALL=C sort | paste -sd, -)
        [[ $sudos == cat,chmod,chown,sh,test ]] || { echo "sudo runs: $sudos"; exit 1; }
        (( n_sudo == 5 )) || { echo "$n_sudo sudo commands, not 5: a sudo that names a command would search secure_path"; exit 1; }
        [[ $(grep -c "sudo -n \"\${host_path\[sh\]:?[^}]*}\" -c \"\$SIGN_SCRIPT\" sh" <<<"$code") == 1 ]] \
          || { echo "sudo runs sh with something other than SIGN_SCRIPT"; exit 1; }
        s=$SIGN_SCRIPT
        ! grep -nE "(^|[^[:alnum:]_.}/-])/tmp([^[:alnum:]_.-]|\$)" <<<"$s" || { echo "the sign script names /tmp"; exit 1; }
        mapfile -t l <<<"$s"
        [[ ${l[0]} == "set -eu" && ${l[1]} == "PATH=$TRUSTED_PATH" && ${l[2]} == "export PATH" \
          && ${l[3]} =~ ^work=\$\(mktemp\ -d\ /root/[A-Za-z0-9._-]*XXXXXX\)$ ]] \
          || { echo "the sign script does not start with set -eu, PATH=$TRUSTED_PATH and mktemp -d /root/...: ${l[*]:0:4}"; exit 1; }
        set_at=-1 export_at=-1 first=-1
        for i in "${!l[@]}"; do
          (( i > 3 )) || continue
          case ${l[i]} in
            "trap "*) ;;
            "TMPDIR=\$work/"*) (( set_at >= 0 )) || set_at=$i ;;
            "export TMPDIR") (( export_at >= 0 )) || export_at=$i ;;
            *) (( first >= 0 )) || first=$i ;;
          esac
        done
        (( set_at > 3 && export_at > set_at && first > export_at )) \
          || { echo "TMPDIR is not set and exported before the first command (set=$set_at export=$export_at first=$first)"; exit 1; }
        [[ $s == *"\"\$work/cosign\" sign-blob"* ]] || { echo "cosign does not run from the work directory"; exit 1; }
        echo ok' "$release"

echo "# tampered-copy negative (sourced tampered_rejected; cosign stood in by a function)"
# cosign_verify stand-in: a "signature" is "sig:<sha256 of the signed file>".
STUB_COSIGN='cosign_verify() {
    if [[ ! -f $2 || ! -f $3 ]]; then echo "Error: open: no such file or directory"; return 1; fi
    if [[ $(cat "$2") == "sig:$(sha256_of "$3")" ]]; then echo "Verified OK"; return 0; fi
    echo "Error: invalid signature when validating ASN.1 encoded signature"; return 1
  }
  fake_sign() { printf "sig:%s" "$(sha256_of "$1")" > "$1.sig"; }
  '
td=$scratch/tampered
mkdir -p "$td" && printf 'aaaa  b\n' > "$td/sums.txt"
expect "tampered negative holds for a good signature" ok '^held$' \
  unit "$STUB_COSIGN"'fake_sign "$1/sums.txt"; tampered_rejected pub "$1/sums.txt.sig" "$1/sums.txt" "$1/t1" >/dev/null && echo held' "$td"
expect "tampered negative fails, not passes, when the sums file is missing" refuse 'is missing' \
  unit "$STUB_COSIGN"'tampered_rejected pub "$1/sums.txt.sig" "$1/no-such-sums.txt" "$1/t2"' "$td"
expect "tampered negative fails when the untouched sums do not verify" refuse 'does not verify, so a rejection would prove nothing' \
  unit "$STUB_COSIGN"'printf bad > "$1/bad.sig"; tampered_rejected pub "$1/bad.sig" "$1/sums.txt" "$1/t3"' "$td"
expect "tampered negative fails when the verifier accepts anything" refuse 'a tampered copy of .* verified' \
  unit "$STUB_COSIGN"'cosign_verify() { echo "Verified OK"; }; tampered_rejected pub "$1/sums.txt.sig" "$1/sums.txt" "$1/t4"' "$td"

echo "# readback (sourced readback; cosign, GitHub and origin stood in by functions)"
pubkey > "$scratch/signing.pub"
tag_obj=1111111111111111111111111111111111111111
# $1 run dir, $2 released commit, $3 tag object, $4 public key, $5 another commit.
RB_SETUP='tag=v0.2.0 version=0.2.0 sha=$2 run_dir=$1 dist=$1/dist src_dir=$1/src other_commit=$5
  checks_file=$1/readback-checks.jsonl tag_object=$3 release_url=https://example.invalid/releases/v0.2.0
  mkdir -p "$dist" "$src_dir/docs/release"
  cp -- "$4" "$src_dir/docs/release/signing.pub"
  printf "#!/bin/sh\necho \"jope 0.2.0\"\n" > "$dist/jope-v0.2.0-x86_64-unknown-linux-gnu"
  chmod 0755 "$dist/jope-v0.2.0-x86_64-unknown-linux-gnu"
  printf "notes\n" > "$dist/RELEASE-NOTES-v0.2.0.md"
  jq -n --arg sha "$sha" "{tag: \"v0.2.0\", git_sha: \$sha, crate_version: \"0.2.0\"}" > "$dist/release-manifest-v0.2.0.json"
  write_sums "$dist" v0.2.0
  fake_sign "$dist/SHA256SUMS-v0.2.0.txt"
  release_view() { jq -n --arg t "$tag" "{tagName: \$t, isDraft: false}"; }
  release_download() { cp -p -- "$dist"/* "$1"/; }
  remote_tag_lines() { printf "%s\trefs/tags/%s\n%s\trefs/tags/%s^{}\n" "$tag_object" "$tag" "$sha" "$tag"; }
  evil_binary() { printf "#!/bin/sh\ntouch %s/marker\necho \"jope 0.2.0\"\n" "$run_dir" > "$1/jope-v0.2.0-x86_64-unknown-linux-gnu"; }
  '
# Runs readback in a background subshell (so errexit stays on inside it) and summarises readback.json.
RB_RUN='
  rc=0
  ( readback ) > "$run_dir/readback.out" 2>&1 &
  wait "$!" || rc=$?
  j=$run_dir/readback.json
  chk() { jq -r --arg n "$1" "[.checks[] | select(.name == \$n)
    | if .ok then \"ok\" elif .skipped then \"skipped\" else \"FAIL\" end] | join(\",\")" "$j"; }
  if [[ -e $run_dir/marker ]]; then mk=present; else mk=absent; fi
  printf "rc=%s;passed=%s;remote=%s;set=%s;blob=%s;tampered=%s;sums=%s;manifest=%s;binary=%s;marker=%s;origin_commit=%s\n" \
    "$rc" "$(jq -r .passed "$j")" "$(chk remote-tag)" "$(chk asset-set-equals-dist)" "$(chk verify-blob)" \
    "$(chk verify-blob-tampered)" "$(chk sha256sum-c)" "$(chk manifest-binds-tag)" "$(chk binary-version)" \
    "$mk" "$(jq -r .origin_tag.commit "$j")"'
rb_n=0
rb() {  # rb <case name> <expected summary ERE> <snippet that overrides a stand-in>
  rb_n=$((rb_n + 1))
  expect "$1" ok "$2" unit "$STUB_COSIGN$RB_SETUP$3$RB_RUN" "$scratch/rb-$rb_n" "$c1" "$tag_obj" "$scratch/signing.pub" "$c0"
}
rb "readback of an intact release passes and runs the binary" \
  "^rc=0;passed=true;remote=ok;set=ok;blob=ok;tampered=ok;sums=ok;manifest=ok;binary=ok;marker=absent;origin_commit=$c1\$" ''
rb "readback never runs a replaced binary asset" \
  "^rc=1;passed=false;remote=ok;set=FAIL;blob=ok;tampered=ok;sums=FAIL;manifest=ok;binary=skipped;marker=absent;origin_commit=$c1\$" \
  'release_download() { cp -p -- "$dist"/* "$1"/; evil_binary "$1"; }'
rb "readback never runs a binary whose sums were regenerated (signature fails)" \
  "^rc=1;passed=false;remote=ok;set=FAIL;blob=FAIL;tampered=FAIL;sums=ok;manifest=ok;binary=skipped;marker=absent;origin_commit=$c1\$" \
  'release_download() { cp -p -- "$dist"/* "$1"/; evil_binary "$1"; write_sums "$1" v0.2.0; }'
rb "readback with a failed download fails every check, the tampered negative included" \
  "^rc=1;passed=false;remote=ok;set=FAIL;blob=FAIL;tampered=FAIL;sums=FAIL;manifest=FAIL;binary=skipped;marker=absent;origin_commit=$c1\$" \
  'release_download() { return 1; }'
rb "readback catches an origin tag that moved after the release" \
  "^rc=1;passed=false;remote=FAIL;set=ok;blob=ok;tampered=ok;sums=ok;manifest=ok;binary=ok;marker=absent;origin_commit=$c0\$" \
  'remote_tag_lines() { printf "%s\trefs/tags/%s\n%s\trefs/tags/%s^{}\n" "$tag_object" "$tag" "$other_commit" "$tag"; }'

echo "# pinned tools (sourced check_syft / check_cosign / check_sboms / make_sbom; stand-ins only, never the real tools)"
# stand_in <path> <name> <reported version>: a syft or cosign stand-in. It records that it ran and
# its environment next to itself ($0.ran, $0.env), since syft_env hands it no variable of ours.
stand_in() {
  mkdir -p -- "${1%/*}"
  {
    printf '#!/bin/sh\n: > "$0.ran"\nenv > "$0.env"\n'
    case $2 in
      syft) printf 'if [ "$1" = version ]; then printf %s; fi\n' "'{\"application\":\"syft\",\"version\":\"$3\"}\\n'" ;;
      cosign) printf 'if [ "$1" = version ]; then printf %s; fi\n' "'{\"gitVersion\":\"$3\"}\\n'" ;;
    esac
  } > "$1"
  chmod 0755 "$1"
}
pt=$scratch/pinned
stand_in "$pt/syft-1.40.0/syft" syft 1.40.0
stand_in "$pt/syft-1.41.0/syft" syft 1.41.0
stand_in "$pt/cosign-v2.4.4/cosign" cosign v2.4.4
# ran <path>: whether a stand-in ran.
PT_RAN='ran() { if [[ -e $1.ran ]]; then echo yes; else echo no; fi; }
  '
expect "syft pin: a syft whose sha256 is not the pin is refused and never run" refuse \
  'has sha256 [0-9a-f]{64}, not the pinned syft 1\.40\.0 \(eb9714fb[0-9a-f]{56}\).*;ran=no$' \
  unit "$PT_RAN"'rc=0; out=$(check_syft "$1" "$SYFT_SHA256" "$SYFT_VERSION" "$2" 2>&1) || rc=$?
        printf "%s;ran=%s\n" "$out" "$(ran "$1")"; exit "$rc"' "$pt/syft-1.40.0/syft" "$pt/check-a"
expect "syft pin: the pinned digest but another reported version is refused" refuse \
  "reports version '1\\.41\\.0', not the pinned syft 1\\.40\\.0" \
  unit 'check_syft "$1" "$(sha256_of "$1")" "$SYFT_VERSION" "$2"' "$pt/syft-1.41.0/syft" "$pt/check-b"
expect "syft pin: digest and version match; syft runs with nothing inherited, TMPDIR, HOME and cache under its dir" ok \
  '^version=1\.40\.0;leak=no;tmpdir=ok;home=ok;cache=ok;update_check=off$' \
  unit 'd=$2; export LEAK_CANARY=1; out=$(check_syft "$1" "$(sha256_of "$1")" 1.40.0 "$d")
        e=$1.env
        if grep -q "^LEAK_CANARY=" "$e"; then l=yes; else l=no; fi
        printf "version=%s;leak=%s;tmpdir=%s;home=%s;cache=%s;update_check=%s\n" "$(jq -r .version <<<"$out")" "$l" \
          "$(grep -qx "TMPDIR=$d/tmp" "$e" && echo ok || echo bad)" "$(grep -qx "HOME=$d/home" "$e" && echo ok || echo bad)" \
          "$(grep -qx "SYFT_CACHE_DIR=$d/cache/syft" "$e" && echo ok || echo bad)" \
          "$(grep -qx "SYFT_CHECK_FOR_APP_UPDATE=false" "$e" && echo off || echo on)"' "$pt/syft-1.40.0/syft" "$pt/check-c"
rm -f -- "$pt/syft-1.40.0/syft.ran"
# The same refusal through release.sh itself: preflight checks a syft stand-in against the real pin.
preflight_syft() {
  local rc=0 out
  out=$(run "JOPE_RELEASE_TEST_SYFT=$pt/syft-1.40.0/syft" JOPE_RELEASE_TEST_STAMP=20260928T000003Z \
    "${R[@]}" v0.2.0 "$c1" 2>&1) || rc=$?
  printf '%s\n' "$out"
  printf 'refused=%s;ran=%s\n' "$(grep -q 'REFUSED: .*syft-1\.40\.0/syft has sha256 [0-9a-f]*, not the pinned syft 1\.40\.0' <<<"$out" && echo yes || echo no)" \
    "$([[ -e $pt/syft-1.40.0/syft.ran ]] && echo yes || echo no)"
  return "$rc"
}
expect "preflight refuses a replaced syft before anything is built or signed, without running it" refuse \
  '^refused=yes;ran=no$' preflight_syft
expect "make_sbom runs only a checked copy: a replaced syft is refused, never run, and writes no SBOM" refuse \
  '^rc=1;refused=yes;ran=no;copy_ran=no;sboms=0$' \
  unit "$PT_RAN"'tag=v0.2.0 sha=$2 run_dir=$1/run dist=$1/run/dist src_dir=$1/run/src syft_bin=$3
        mkdir -p "$run_dir/tmp" "$dist" "$src_dir"
        rc=0; ( make_sbom ) > "$1/out" 2>&1 || rc=$?
        if grep -q "tmp/syft/syft has sha256 [0-9a-f]*, not the pinned syft 1\.40\.0" "$1/out"; then r=yes; else r=no; fi
        printf "rc=%s;refused=%s;ran=%s;copy_ran=%s;sboms=%s\n" "$rc" "$r" "$(ran "$3")" "$(ran "$run_dir/tmp/syft/syft")" \
          "$(find "$dist" -type f | wc -l)"; exit "$rc"' "$pt/sbom" "$c1" "$pt/syft-1.40.0/syft"
jq -n '{spdxVersion: "SPDX-2.3", creationInfo: {creators: ["Organization: Anchore, Inc", "Tool: syft-1.40.0"]}}' > "$pt/good.spdx.json"
jq -n '{bomFormat: "CycloneDX", metadata: {tools: {components: [{type: "application", author: "anchore", name: "syft", version: "1.40.0"}]}}}' > "$pt/good.cdx.json"
jq '.creationInfo.creators[1] = "Tool: syft-1.41.0"' "$pt/good.spdx.json" > "$pt/other.spdx.json"
jq '.metadata.tools.components[0].version = "1.41.0"' "$pt/good.cdx.json" > "$pt/other.cdx.json"
expect "SBOMs that name the pinned syft as their creator are accepted" ok '^ok$' \
  unit 'check_sboms "$1" "$2" "$SYFT_VERSION" && echo ok' "$pt/good.spdx.json" "$pt/good.cdx.json"
expect "an SBOM made by another syft version is refused" refuse 'other\.cdx\.json is not a CycloneDX document made by syft 1\.40\.0' \
  unit 'check_sboms "$1" "$2" "$SYFT_VERSION"' "$pt/good.spdx.json" "$pt/other.cdx.json"
expect "cosign pin: a cosign whose sha256 is not the pin is refused and never run" refuse \
  'has sha256 [0-9a-f]{64}, not the pinned cosign v2\.4\.3 \(caaad125[0-9a-f]{56}\).*;ran=no$' \
  unit "$PT_RAN"'rc=0; out=$(check_cosign "$1" "$COSIGN_SHA256" "$COSIGN_VERSION" 2>&1) || rc=$?
        printf "%s;ran=%s\n" "$out" "$(ran "$1")"; exit "$rc"' "$pt/cosign-v2.4.4/cosign"
expect "cosign pin: the pinned digest but another reported version is refused" refuse \
  "reports version 'v2\\.4\\.4', not the pinned cosign v2\\.4\\.3" \
  unit 'check_cosign "$1" "$(sha256_of "$1")" "$COSIGN_VERSION"' "$pt/cosign-v2.4.4/cosign"

echo "# host tools (sourced check_host_tool and trusted_chain; both bypasses of review 5347008178)"
# with_path <dir> <command...>: runs the command with <dir> first on PATH, as the operator's PATH puts
# ~/.local/bin first. A sourced release.sh must look past it.
with_path() { local d=$1; shift; ( PATH=$d:$PATH; "$@" ); }
# The operator's fixture directories are group-writable (0775) as well as owned by whoever runs the
# tests, so they are refused by mode even when that is root.
ht=$scratch/host-tools
mkdir -p "$ht/user-bin" "$ht/shadow" "$ht/ancestor/replacement"
chmod 0775 "$ht/user-bin" "$ht/shadow" "$ht/ancestor"
printf '#!/bin/sh\nexec /usr/bin/jq "$@"\n' > "$ht/user-bin/jq" && chmod 0755 "$ht/user-bin/jq"
# Bypass 1: a stat that the operator put first on PATH and that vouches for anything. Whatever it is
# asked, it prints "0 755" (root-owned, 0755), and it leaves a marker when it runs. Its directory
# also holds an operator-owned cat.
printf '#!/bin/sh\n: > "$0.ran"\nprintf "0 755\\n"\n' > "$ht/shadow/stat"
printf '#!/bin/sh\necho SHADOW_CAT\n' > "$ht/shadow/cat"
chmod 0755 "$ht/shadow/stat" "$ht/shadow/cat"
# Bypass 2: an operator-owned directory holding bin -> /usr/bin. stat -L sees only the root-owned
# directories beyond the link, and the operator can later swap the link for one to replacement/.
ln -s /usr/bin "$ht/ancestor/bin"
printf '#!/bin/sh\necho REPLACEMENT_EXECUTED\n' > "$ht/ancestor/replacement/cat"
chmod 0755 "$ht/ancestor/replacement/cat"
printf 'real-cat\n' > "$ht/marker"
expect "host tools, positive control: sourced with an operator directory first on PATH, PATH becomes TRUSTED_PATH, and system tools, also reached through root-owned links (/bin -> usr/bin, sh -> dash), pass" ok \
  '^path=trusted;cat=/(usr/)?bin/cat;sh=/(usr/)?bin/sh;bin_sh=ok$' \
  with_path "$ht/user-bin" unit 'if [[ $PATH == "${TRUSTED_PATH-}" ]]; then p=trusted; else p=$PATH; fi
        c=$(check_host_tool cat); s=$(check_host_tool sh)
        b=no; if trusted_chain /bin/sh; then b=ok; fi
        printf "path=%s;cat=%s;sh=%s;bin_sh=%s\n" "$p" "$c" "$s" "$b"'
expect "host tools: a root-owned system tool is accepted" ok '^/(usr/)?bin/cat$' unit 'check_host_tool cat'
expect "host tools: a jq the operator can replace, first on the caller's PATH, is neither the one checked nor the one run" ok \
  '^checked=/(usr/)?bin/jq;runs=/(usr/)?bin/jq$' \
  with_path "$ht/user-bin" unit 'printf "checked=%s;runs=%s\n" "$(check_host_tool jq)" "$(type -P jq)"'
expect "host tools: a sourced release.sh keeps PATH read-only, so no later assignment can put a directory in front" refuse \
  'PATH: readonly variable' unit 'PATH=$1:$PATH; echo "PATH changed"' "$ht/user-bin"
rm -f -- "$ht/shadow/stat.ran"
expect "review bypass 1: a stat first on PATH that vouches for everything is never run, and stat and cat resolve to the system's" ok \
  '^stat=/(usr/)?bin/stat;cat=/(usr/)?bin/cat;shadow_ran=no$' \
  with_path "$ht/shadow" unit 'st=$(check_host_tool stat); c=$(check_host_tool cat)
        if [[ -e $1/stat.ran ]]; then r=yes; else r=no; fi
        printf "stat=%s;cat=%s;shadow_ran=%s\n" "$st" "$c" "$r"' "$ht/shadow"
# REFUSED_AT: runs trusted_chain on "$2" and says whether it refused at a directory on the fixture's
# own path "$1" (the first one the operator owns, from / down), not somewhere past a link.
REFUSED_AT='rc=0; out=$(trusted_chain "$2" 2>&1) || rc=$?
  d=$(sed -nE "s/.*: (\/[^ ]*) is not owned by root and closed to group and other writes \(uid [0-9]+, mode [0-7]+\)$/\1/p" <<<"$out")
  if [[ -n $d && $1/ == "$d"/* ]]; then w=fixture-path; else w="other: $out"; fi
  printf "rc=%s;refused_at=%s\n" "$rc" "$w"; exit "$rc"'
expect "review bypass 1: trusted_chain refuses the operator's stat at an operator-owned directory on its path" refuse \
  '^rc=1;refused_at=fixture-path$' unit "$REFUSED_AT" "$ht/shadow" "$ht/shadow/stat"
expect "review bypass 2: with <operator dir>/bin -> /usr/bin first on PATH, cat resolves to the system's, and swapping the link afterwards changes nothing that runs" ok \
  '^checked=/(usr/)?bin/cat;ran=real-cat$' \
  with_path "$ht/ancestor/bin" unit 'c=$(check_host_tool cat)
        ln -s "$1/replacement" "$1/next" && mv -T "$1/next" "$1/bin"
        printf "checked=%s;ran=%s\n" "$c" "$(cat "$2")"' "$ht/ancestor" "$ht/marker"
rm -f -- "$ht/ancestor/bin" && ln -s /usr/bin "$ht/ancestor/bin"
expect "review bypass 2: trusted_chain refuses <operator dir>/bin/cat at an operator-owned directory on its path, although bin -> /usr/bin leads only to root-owned directories" refuse \
  '^rc=1;refused_at=fixture-path$' unit "$REFUSED_AT" "$ht/ancestor" "$ht/ancestor/bin/cat"
md2=$scratch/dist-manifest-tools
mkdir -p "$md2" && printf 'binary' > "$md2/jope-v0.2.0-x86_64-unknown-linux-gnu"
expect "the release manifest records the pinned syft and cosign" ok '^ok$' \
  unit 'cd "$3" && write_manifest "$1" v0.2.0 "$2" 0.2.0 2.34
        jq -e --arg ss "$SYFT_SHA256" --arg cs "$COSIGN_SHA256" \
          ".tools == {syft: {version: \"1.40.0\", sha256: \$ss}, cosign: {version: \"v2.4.3\", sha256: \$cs}}" \
          "$1/release-manifest-v0.2.0.json" >/dev/null && echo ok' "$md2" "$c1" "$repo"

echo "# containers carry the run's temporary directory (sourced build_jope, check_binary, make_receipt; docker and objdump stood in by functions)"
{ art /src/crates/jope-cli/Cargo.toml jope '"/target/release/jope"' false; jq -nc '{reason: "build-finished", success: true}'; } \
  > "$scratch/dk-build.jsonl"
{ art /src/crates/jope-engine/Cargo.toml jope-build-receipt '"/target/release/jope-build-receipt"' false
  jq -nc '{reason: "build-finished", success: true}'; } > "$scratch/dk-receipt-tool.jsonl"
# $1 dir, $2 commit, $3 cargo metadata, $4 build messages, $5 receipt-tool build messages. The
# docker stand-in logs each argv (fields joined by \x1f) and answers as the builder would.
CONTAINERS=$(cat <<'EOF'
tag=v0.2.0 version=0.2.0 sha=$2 run_dir=$1/run target_root=$1/target
dist=$run_dir/dist src_dir=$run_dir/src log=$1/docker.log meta=$3 build_msgs=$4 receipt_msgs=$5
mkdir -p "$run_dir/tmp" "$run_dir/home" "$dist" "$src_dir" "$target_root/cargo-home" "$target_root/release"
printf '#!/bin/sh\necho "jope 0.2.0"\n' > "$target_root/release/jope"
chmod 0755 "$target_root/release/jope"
: > "$log"
docker() {
  local line i img=-1 out=''
  local -a args=("$@") cmd
  printf -v line '%s\x1f' "$@"
  printf '%s\n' "${line%$'\x1f'}" >> "$log"
  [[ $1 == run ]] || return 0
  # The first image argument is docker's; the receipt tool is later given it as --builder-image.
  for i in "${!args[@]}"; do if [[ ${args[i]} == "$BUILDER_IMAGE" ]]; then img=$i; break; fi; done
  (( img > 0 )) || { echo "docker stand-in: no image in: $*" >&2; return 99; }
  cmd=("${args[@]:img+1}")
  case ${cmd[0]} in
    rustc) printf 'rustc 1.97.1 (stand-in)\nrelease: 1.97.1\n' ;;
    ldd) printf 'ldd (Debian GLIBC 2.36-9+deb12u7) 2.36\n' ;;
    cargo)
      if [[ ${cmd[1]} == metadata ]]; then cat "$meta"
      elif [[ " ${cmd[*]} " == *" jope-build-receipt "* ]]; then cat "$receipt_msgs"
      else cat "$build_msgs"; fi ;;
    /dist/*) printf 'jope 0.2.0\n' ;;
    /target/release/jope-build-receipt)
      for i in "${!cmd[@]}"; do [[ ${cmd[i]} != --output ]] || out=${cmd[i+1]}; done
      jq -n --arg bin "$(sha256_of "$dist/jope-$tag-$TARGET_TRIPLE")" \
        --arg toml "$(blob_sha256 "$sha" Cargo.toml)" --arg lock "$(blob_sha256 "$sha" Cargo.lock)" '
        {bounds: {passed: true}, binary_sha256: $bin, links_cuda_or_torch: false,
         glibc_max_needed: "GLIBC_2.34", cargo_toml_sha256: $toml, cargo_lock_sha256: $lock, git: null}' \
        > "$run_dir/receipt/${out#/out/}" ;;
    *) echo "docker stand-in: unexpected command: ${cmd[*]}" >&2; return 98 ;;
  esac
}
objdump() {
  case $1 in
    -T) printf '0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.34) __libc_start_main\n' ;;
    -p) printf 'Dynamic Section:\n  NEEDED               libc.so.6\n' ;;
  esac
}
build_jope
check_binary
make_receipt
ctmp=$run_dir/tmp/container runs=0 carried=0 bounded=0 attributed=0
while IFS= read -r line; do
  [[ $line == run$'\x1f'* ]] || continue
  runs=$((runs + 1))
  s=$'\x1f'"$line"$'\x1f'
  if [[ $s == *$'\x1f-v\x1f'"$ctmp:/release-tmp"$'\x1f'* && $s == *$'\x1f-v\x1f'"$ctmp:/tmp"$'\x1f'* \
        && $s == *$'\x1f-e\x1fTMPDIR=/release-tmp\x1f'* ]]; then
    carried=$((carried + 1))
  fi
  if [[ $s == *$'\x1f--cpus\x1f4\x1f'* && $s == *$'\x1f--memory\x1f20g\x1f'* \
        && $s == *$'\x1f--memory-swap\x1f20g\x1f'* && $s == *$'\x1f-e\x1fCARGO_BUILD_JOBS=4\x1f'* ]]; then
    bounded=$((bounded + 1))
  fi
  if [[ $s == *$'\x1f--label\x1f'"ai.jopedime.release.run=$run_dir"$'\x1f'* \
        && $s == *$'\x1f--label\x1fai.jopedime.release.tag=v0.2.0\x1f'* ]]; then
    attributed=$((attributed + 1))
  fi
done < "$log"
printf 'runs=%s;carrying_tmp=%s;container_tmp_mode=%s;bounded=%s;attributed=%s\n' "$runs" "$carried" "$(stat -c %a "$ctmp" 2>/dev/null || echo missing)" "$bounded" "$attributed"
EOF
)
expect "every build/check/receipt container carries private TMPDIR, fixed CPU/RAM/Cargo limits and run attribution" ok \
  '^runs=7;carrying_tmp=7;container_tmp_mode=700;bounded=7;attributed=7$' \
  unit "$CONTAINERS" "$scratch/containers" "$c1" "$work/meta-ok.json" "$scratch/dk-build.jsonl" "$scratch/dk-receipt-tool.jsonl"
# The same rule, read from the source, so that a new docker run cannot bypass builder_container.
expect "static: every docker run in release.sh is in builder_container" ok '^sites=builder_container\(\)$' \
  unit 'sites=$(awk "/^[A-Za-z_][A-Za-z0-9_]*\\(\\) [{(]\$/ { f = \$1 }
          !/^[[:space:]]*#/ && /(^|[;&|(]|\\\$\\()[[:space:]]*docker[[:space:]]+run([[:space:]]|\$)/ { print f }" "$1" | paste -sd, -)
        printf "sites=%s\n" "$sites"' "$release"
expect "static: every env -i in release.sh names TMPDIR" ok '^env_i=[2-9];without_tmpdir=0$' \
  unit 'lines=$(grep -vE "^[[:space:]]*#" "$1" | grep -E "(^|[^[:alnum:]_-])env -i( |\$)")
        printf "env_i=%s;without_tmpdir=%s\n" "$(grep -c . <<<"$lines")" "$(grep -vc "TMPDIR=" <<<"$lines" || true)"' "$release"

echo "# the run's temporary directory (sourced open_run_dir; tools_json stood in)"
expect "the run's tmp/ is private (0700) under umask 022, and becomes TMPDIR" ok '^rc=0;tmp_mode=700;tmpdir=run$' \
  unit 'umask 022; tag=v0.2.0 sha=$2 mode=dryrun run_dir=$1/run target_root=$1/target metadata_json="{}"
        tools_json() { printf "{\"host\": []}\n"; }
        mkdir -p "$1"; rc=0
        ( open_run_dir; printf "%s\n" "$TMPDIR" > "$1/tmpdir.txt" ) > "$1/open.out" 2>&1 &
        wait "$!" || rc=$?
        if [[ $(cat "$1/tmpdir.txt" 2>/dev/null) == "$run_dir/tmp" ]]; then t=run; else t=other; fi
        printf "rc=%s;tmp_mode=%s;tmpdir=%s\n" "$rc" "$(stat -c %a "$run_dir/tmp")" "$t"' "$scratch/open-run-dir" "$c1"

echo "# the tools' own configuration (hostile HOMEs, repository config and variables, all under the test dir)"
tc=$scratch/toolconf
mkdir -p "$tc"
# fake_home <dir>: a HOME whose test root resolves to the real one (release.sh takes the realpath of
# $HOME/.cache/jope-test-tmp), so that release.sh runs in test mode with that HOME's files in reach.
fake_home() { mkdir -p -- "$1/.cache"; ln -sfn -- "$test_parent" "$1/.cache/jope-test-tmp"; }
# The cases below that run release.sh use a stamp whose run directory does not exist, so that a
# release.sh that let the hostile configuration through would pass every guard.
fresh=JOPE_RELEASE_TEST_STAMP=20260928T000005Z
# run_home <home> <command...>: the base environment with HOME=<home> and without this script's own
# git isolation, so that git would read <home>'s configuration if release.sh let it.
run_home() {
  local h=$1
  shift
  env -u GIT_CONFIG_GLOBAL -u GIT_CONFIG_NOSYSTEM -u XDG_CONFIG_HOME "${base[@]}" "$fresh" "HOME=$h" "$@"
}
fake_home "$tc/home-clean"
expect "control: with a fake HOME that holds no configuration, every guard passes" ok 'all guards passed' \
  run_home "$tc/home-clean" "${R[@]}" v0.2.0 "$c1"

# jq: every jq of release.sh runs with an empty private HOME, so a ~/.jq defines nothing.
mkdir -p "$tc/home-jq-sbom" "$tc/home-jq-manifest" "$tc/planted-tmp"
printf 'def startswith($s): true;\ndef any(g; c): true;\n' > "$tc/home-jq-sbom/.jq"
printf 'def from_entries: {};\ndef startswith($s): false;\n' > "$tc/home-jq-manifest/.jq"
jq -n '{spdxVersion: "NOT-SPDX", creationInfo: {creators: ["Tool: other"]}}' > "$tc/bad.spdx.json"
expect "jq: a ~/.jq that redefines startswith and any cannot make check_sboms accept a non-SPDX document" refuse \
  'bad\.spdx\.json is not an SPDX document created by syft 1\.40\.0' \
  unit 'HOME=$1; check_sboms "$2" "$3" "$SYFT_VERSION" && echo ACCEPTED' \
  "$tc/home-jq-sbom" "$tc/bad.spdx.json" "$pt/good.cdx.json"
md3=$scratch/dist-manifest-jq
mkdir -p "$md3" && printf 'binary' > "$md3/jope-v0.2.0-x86_64-unknown-linux-gnu" && printf 'notes' > "$md3/RELEASE-NOTES-v0.2.0.md"
expect "jq: a ~/.jq that redefines from_entries and startswith cannot change the release manifest" ok '^ok$' \
  unit 'cd "$3" && ( HOME=$4; write_manifest "$1" v0.2.0 "$2" 0.2.0 2.34 )
        jq -e ".artifacts | keys == [\"RELEASE-NOTES-v0.2.0.md\", \"jope-v0.2.0-x86_64-unknown-linux-gnu\"]" \
          "$1/release-manifest-v0.2.0.json" >/dev/null || { echo "artifacts: $(jq -c .artifacts "$1/release-manifest-v0.2.0.json")"; exit 1; }
        jq -e ".models_pinned == [\"bundle-a\", \"bundle-b\"]" "$1/release-manifest-v0.2.0.json" >/dev/null \
          || { echo "models_pinned: $(jq -c .models_pinned "$1/release-manifest-v0.2.0.json")"; exit 1; }
        echo ok' "$md3" "$c1" "$repo" "$tc/home-jq-manifest"
fake_home "$tc/home-jq-version"
printf 'def map(f): [.[] | f | .version = "0.2.0"];\ndef startswith($s): true;\n' > "$tc/home-jq-version/.jq"
expect "jq: through release.sh, a ~/.jq cannot turn a jope-cli 0.1.0 workspace into a pass for v0.2.0" refuse \
  'MISMATCH jope-cli 0\.1\.0 != tag version 0\.2\.0' \
  run_home "$tc/home-jq-version" "FAKE_CARGO_METADATA=$work/meta-cli-010.json" "${R[@]}" v0.2.0 "$c1"
expect "jq: its private HOME must stay empty (a .jq planted there is refused, jq never runs)" refuse \
  'jq-home is not empty' \
  unit 'export TMPDIR=$1; mkdir -p "$TMPDIR/jq-home"; printf "def x: 1;\n" > "$TMPDIR/jq-home/.jq"; jq -n 1' "$tc/planted-tmp"

# git: no /etc/gitconfig, ~/.gitconfig, per-user ignore file, replace refs, repository URL rewrites,
# remotes named after a URL, or TLS settings.
git clone -q --bare "$origin" "$tc/evil-origin.git"
git -C "$repo" push -q -f "$origin" "$c2:refs/heads/main"
fake_home "$tc/home-gitconfig"
printf '[url "%s"]\n\tinsteadOf = %s\n' "$tc/evil-origin.git" "$origin" > "$tc/home-gitconfig/.gitconfig"
expect "git: a URL rewrite in ~/.gitconfig cannot answer the fresh fetch from another origin" refuse \
  "origin/main is $c2 after a fresh fetch, not $c1" \
  run_home "$tc/home-gitconfig" "${R[@]}" v0.2.0 "$c1"
refused_git_key='the repository.s git config sets 1 key\(s\) that could send the fetch, the tag checks or the push somewhere other than'
git -C "$repo" config "url.$tc/evil-origin.git.insteadOf" "$origin"
expect "git: a URL rewrite in the repository's own config is refused" refuse \
  "$refused_git_key" run "$fresh" "${R[@]}" v0.2.0 "$c1"
git -C "$repo" config --remove-section "url.$tc/evil-origin.git"
# A remote named after the URL that git is given: git looks the URL up as a remote name first, so
# that remote's url sends the fetch, the tag checks and the push to another repository, and its
# pushurl the push alone.
# git never takes an absolute path such as $origin as a remote name, so these cases give release.sh
# the same origin as a file:// URL. origin's main is c2 here and evil-origin's is c1, so a release.sh
# that followed such a remote would pass every guard.
furl=file://$origin
git -C "$repo" config remote.origin.url "$tc/evil-origin.git"
git -C "$repo" config remote.origin.pushurl "$tc/evil-origin.git"
expect "git: a remote with another name (origin) passes the config check and redirects nothing" refuse \
  "origin/main is $c2 after a fresh fetch, not $c1" \
  run "$fresh" "JOPE_RELEASE_TEST_REMOTE=$furl" "${R[@]}" v0.2.0 "$c1"
git -C "$repo" config --remove-section remote.origin
git -C "$repo" config "remote.$furl.url" "$tc/evil-origin.git"
expect "git: remote.<URL>.url in this git answers ls-remote of that URL from another repository" ok \
  "^$c1"$'\t'"refs/heads/main\$" git ls-remote "$furl" refs/heads/main
expect "git: a remote named after the URL release.sh is given (remote.<URL>.url) is refused" refuse \
  "$refused_git_key" run "$fresh" "JOPE_RELEASE_TEST_REMOTE=$furl" "${R[@]}" v0.2.0 "$c1"
git -C "$repo" config --remove-section "remote.$furl"
git -C "$repo" config "remote.$furl.pushurl" "$tc/evil-origin.git"
expect "git: a pushurl alone, which would take only the push elsewhere, is refused" refuse \
  "$refused_git_key" run "$fresh" "JOPE_RELEASE_TEST_REMOTE=$furl" "${R[@]}" v0.2.0 "$c1"
git -C "$repo" config --remove-section "remote.$furl"
git -C "$repo" config remote.https://github.com/neverhuman/JopeDime.git.url "$tc/evil-origin.git"
expect "git: a remote named after any other URL (here the GitHub one) is refused" refuse \
  "$refused_git_key" run "$fresh" "JOPE_RELEASE_TEST_REMOTE=$furl" "${R[@]}" v0.2.0 "$c1"
git -C "$repo" config --remove-section remote.https://github.com/neverhuman/JopeDime.git
# A test remote given as a relative name is looked up as a remote name too, without any ://.
git -C "$repo" config remote.release-origin.url "$tc/evil-origin.git"
expect "git: a remote named exactly as the given remote, without ://, is refused" refuse \
  "$refused_git_key" run "$fresh" JOPE_RELEASE_TEST_REMOTE=release-origin "${R[@]}" v0.2.0 "$c1"
git -C "$repo" config --remove-section remote.release-origin
expect "git: with the remote sections removed, the file:// URL reaches origin itself again" refuse \
  "origin/main is $c2 after a fresh fetch, not $c1" \
  run "$fresh" "JOPE_RELEASE_TEST_REMOTE=$furl" "${R[@]}" v0.2.0 "$c1"
git -C "$repo" push -q -f "$origin" "$c1:refs/heads/main"
expect "git: a file:// URL as the remote passes every guard when nothing redirects it" ok 'all guards passed' \
  run "$fresh" "JOPE_RELEASE_TEST_REMOTE=$furl" "${R[@]}" v0.2.0 "$c1"
git -C "$repo" config http.sslVerify false
expect "git: a TLS setting in the repository's own config is refused" refuse \
  "$refused_git_key" run "$fresh" "${R[@]}" v0.2.0 "$c1"
git -C "$repo" config --unset http.sslVerify
fake_home "$tc/home-ignore"
mkdir -p "$tc/home-ignore/.config/git" && printf 'hidden-by-home.txt\n' > "$tc/home-ignore/.config/git/ignore"
printf 'x\n' > "$repo/hidden-by-home.txt"
expect "git: ~/.config/git/ignore cannot hide an untracked file from the clean-tree check" refuse \
  '^\?\? hidden-by-home\.txt$' run_home "$tc/home-ignore" "${R[@]}" v0.2.0 "$c1"
rm -f -- "$repo/hidden-by-home.txt"
c_rp=$(git -C "$repo" commit-tree "$c1^{tree}" -p "$c1" -m 'a release commit that a replace ref shadows')
rp_blob=$(printf 'replaced\n' | git -C "$repo" hash-object -w --stdin)
rp_tree=$(printf '100644 blob %s\treplaced.txt\n' "$rp_blob" | git -C "$repo" mktree)
git -C "$repo" replace "$c_rp" "$(git -C "$repo" commit-tree "$rp_tree" -m 'the replacement')"
expect "git: a replace ref cannot swap the tree that is exported and verified as the commit's" ok \
  '^tree=commit;cargo_toml=present;replaced_txt=absent;verified=true$' \
  unit 'run_dir=$1 sha=$2; mkdir -p "$run_dir/tmp"; export_source >/dev/null 2>&1
        real=$(git --no-replace-objects rev-parse "$sha^{tree}")
        if [[ $(jq -r .tree "$run_dir/source-export.json") == "$real" ]]; then t=commit; else t=replacement; fi
        if [[ -f $run_dir/src/Cargo.toml ]]; then c=present; else c=absent; fi
        if [[ -e $run_dir/src/replaced.txt ]]; then r=present; else r=absent; fi
        printf "tree=%s;cargo_toml=%s;replaced_txt=%s;verified=%s\n" "$t" "$c" "$r" "$(jq -r .verified "$run_dir/source-export.json")"' \
  "$scratch/export-replace" "$c_rp"
git -C "$repo" replace -d "$c_rp" >/dev/null
# The tag push sends the release tag and nothing else. With push.followTags in the repository's
# config, git would also push every local annotated tag in the commit's history that origin lacks.
# A clone of $repo and new bare origins keep $repo and $origin as they were. The tags are made here,
# because a sourced release.sh reads no ~/.gitconfig and so has no identity to make them with.
ft=$scratch/follow-tags
git clone -q "$repo" "$ft/repo"
git -C "$ft/repo" tag -a v0.1.9 "$c0" -m 'an older local annotated tag that origin lacks'
git -C "$ft/repo" tag -a v0.2.0 "$c1" -m 'JopeDime v0.2.0'
git -C "$ft/repo" config push.followTags true
for o in forced flag-only control; do git init -q --bare "$ft/origin-$o.git"; done
git -C "$ft/repo" tag -a v0.1.8 "$c0" -m 'the existing control tag points to the previous release commit'
git -C "$ft/repo" push -q "$ft/origin-control.git" refs/tags/v0.1.8:refs/tags/v0.2.0
expect "git: the tag push sends only the release tag, even with push.followTags set in the repository's config" ok \
  '^forced=refs/tags/v0\.2\.0;flag_only=refs/tags/v0\.2\.0;control=protected$' \
  unit_with_fixture "$ft" 'cd "$RELEASE_TAG_TEST_FIXTURE/repo"; tag=v0.2.0
        tags() { git -C "$RELEASE_TAG_TEST_FIXTURE/origin-$1.git" for-each-ref --format="%(refname)" refs/tags | tr "\n" " " | sed "s/ \$//"; }
        remote=$RELEASE_TAG_TEST_FIXTURE/origin-forced.git; push_release_tag
        # Without the forced configuration, --no-follow-tags alone still holds.
        ( export GIT_CONFIG_COUNT=0; remote=$RELEASE_TAG_TEST_FIXTURE/origin-flag-only.git; push_release_tag )
        # A conflicting existing release tag must remain pinned; an ordinary push cannot move it.
        before=$(git --git-dir="$RELEASE_TAG_TEST_FIXTURE/origin-control.git" rev-parse "refs/tags/$tag")
        if ( export GIT_CONFIG_COUNT=0; git push -q "$RELEASE_TAG_TEST_FIXTURE/origin-control.git" "refs/tags/$tag:refs/tags/$tag" ); then
          echo "unexpectedly replaced an existing release tag"; exit 1
        fi
        after=$(git --git-dir="$RELEASE_TAG_TEST_FIXTURE/origin-control.git" rev-parse "refs/tags/$tag")
        [[ $before == "$after" ]] || { echo "existing tag moved: $before -> $after"; exit 1; }
        printf "forced=%s;flag_only=%s;control=protected\n" "$(tags forced)" "$(tags flag-only)"' \
  "$ft"

# bash itself: release.sh runs only in privileged mode (bash -p), which imports no exported function
# and ignores BASH_ENV, SHELLOPTS and BASHOPTS. An imported function would stand in for the command
# or the builtin of its name: git hides a dirty tree from git status, unset keeps every variable that
# the isolation block drops, and command hands jq a hostile ~/.jq.
shadow_git() (
  # shellcheck disable=SC2317  # it runs only in the bash that imports it
  git() { if [[ $1 == status ]]; then return 0; fi; command git "$@"; }
  export -f git
  run "$fresh" "${R[@]}" v0.2.0 "$c1"
)
printf 'x\n' > "$repo/hidden-by-function.txt"
expect "an exported shell function named git cannot stand in for git" refuse '^\?\? hidden-by-function\.txt$' shadow_git
expect "exported unset and git functions (a builtin and a command) cannot hide an untracked file" refuse \
  '^\?\? hidden-by-function\.txt$' \
  run "$fresh" 'BASH_FUNC_unset%%=() { :; }' \
    'BASH_FUNC_git%%=() { if [[ $1 == status ]]; then return 0; fi; command git "$@"; }' "${R[@]}" v0.2.0 "$c1"
expect "the #! line runs bash -p: exported unset and git functions cannot hide an untracked file there either" refuse \
  '^\?\? hidden-by-function\.txt$' \
  run "$fresh" 'BASH_FUNC_unset%%=() { :; }' \
    'BASH_FUNC_git%%=() { if [[ $1 == status ]]; then return 0; fi; command git "$@"; }' "$release" v0.2.0 "$c1"
rm -f -- "$repo/hidden-by-function.txt"
jq_via_command="BASH_FUNC_command%%=() { HOME=$tc/home-jq-version builtin command \"\$@\"; }"
expect "an exported command function cannot hand jq a hostile ~/.jq (a jope-cli 0.1.0 workspace stays a mismatch)" refuse \
  'MISMATCH jope-cli 0\.1\.0 != tag version 0\.2\.0' \
  run "$fresh" "FAKE_CARGO_METADATA=$work/meta-cli-010.json" "$jq_via_command" "${R[@]}" v0.2.0 "$c1"
expect "the #! line runs bash -p: an exported command function cannot hand jq a hostile ~/.jq there either" refuse \
  'MISMATCH jope-cli 0\.1\.0 != tag version 0\.2\.0' \
  run "$fresh" "FAKE_CARGO_METADATA=$work/meta-cli-010.json" "$jq_via_command" "$release" v0.2.0 "$c1"
expect "the #! line runs bash -p: executed directly, every guard passes" ok 'all guards passed' \
  run "$fresh" "$release" v0.2.0 "$c1"
expect "a bash started without -p is refused" refuse '^release\.sh: REFUSED: bash was not started in privileged mode' \
  run "$fresh" bash "$release" v0.2.0 "$c1"
# Without -p, printf and exit may be imported functions too: exit then returns, and only the keyword
# test that guards main() stops the run, with status 1 and no guard passed.
expect "without -p, exported exit and unset functions cannot carry a run past the refusal" refuse \
  '^release\.sh: REFUSED: bash was not started in privileged mode' \
  run "$fresh" 'BASH_FUNC_exit%%=() { :; }' 'BASH_FUNC_unset%%=() { :; }' bash "$release" v0.2.0 "$c1"
expect "an exported SHELLOPTS is refused (a plain bash takes privileged mode from it)" refuse \
  '^release\.sh: REFUSED: SHELLOPTS is exported' \
  run "$fresh" SHELLOPTS=privileged 'BASH_FUNC_exit%%=() { :; }' bash "$release" v0.2.0 "$c1"
expect "SHELLOPTS=noexec through the #! line does not make a run exit 0 unchecked" refuse \
  '^release\.sh: REFUSED: SHELLOPTS is exported' \
  run "$fresh" SHELLOPTS=noexec "$release" v0.2.0 nothex --publish
expect "BASHOPTS=nocasematch cannot make [[ ]] accept an uppercase SHA" refuse 'full 40-character lowercase' \
  run "$fresh" BASHOPTS=nocasematch "${R[@]}" v0.2.0 "${c1^^}"
expect "BASHOPTS=nocasematch through the #! line cannot either" refuse 'full 40-character lowercase' \
  run "$fresh" BASHOPTS=nocasematch "$release" v0.2.0 "${c1^^}"
# bash -p still reads these; sourcing release.sh (as a run starts) unsets them before anything runs.
# TMOUT=0.1 would end extract_notes's loop at the pause in its input, FUNCNEST=1 would stop the
# f -> g call, and BASH_COMPAT=31 would make a quoted =~ operand a pattern.
bash_vars_case() {
  env BASH_COMPAT=31 TMOUT=0.1 FUNCNEST=1 GLOBSORT=-name "BASH_ENV=$tc/no-bash-env" LANG=C.UTF-8 \
    bash -p -c 'set -euo pipefail; source "$1"
      f() { g; }; g() { :; }; f
      if [[ abc =~ "a.c" ]]; then re=pattern; else re=literal; fi
      notes=$({ printf "## [0.2.0] - x\n\n- first\n"; sleep 0.5; printf -- "- second\n## [0.1.0]\n"; } | extract_notes 0.2.0)
      printf "compat=%s;tmout=%s;funcnest=%s;globsort=%s;bash_env=%s;lc_all=%s;quoted_regex=%s;notes=%s\n" \
        "${BASH_COMPAT-unset}" "${TMOUT-unset}" "${FUNCNEST-unset}" "${GLOBSORT-unset}" "${BASH_ENV-unset}" \
        "${LC_ALL-unset}" "$re" "$(printf "%s" "$notes" | tr "\n" "|")"' release-unit "$release"
}
expect "bash's own variables: BASH_COMPAT, TMOUT, FUNCNEST, GLOBSORT and BASH_ENV go, LC_ALL is C" ok \
  '^compat=unset;tmout=unset;funcnest=unset;globsort=unset;bash_env=unset;lc_all=C;quoted_regex=literal;notes=## \[0\.2\.0\] - x\|\|- first\|- second$' \
  bash_vars_case

# Variables: sourcing release.sh (as a run starts) drops those that change a tool's behaviour.
env_case() (
  # shellcheck disable=SC2317  # it runs only in the bash that imports it
  sha256sum() { echo 0000; }
  export -f sha256sum
  export LD_LIBRARY_PATH=$tc/no-libs TAR_OPTIONS=--exclude=Cargo.toml GNUTARGET=binary \
    GIT_DIR=$tc/no-git-dir GIT_CONFIG_PARAMETERS="'core.hookspath'='$tc/hooks'" \
    GIT_COMMITTER_NAME='Release Test' DOCKER_BUILDKIT=0 JQ_COLORS=1
  unit 'printf "fn=%s;ld=%s;tar=%s;gnutarget=%s;git_dir=%s;git_params=%s;committer=%s;docker=%s;jq=%s;" \
          "$(type -t sha256sum)" "${LD_LIBRARY_PATH-unset}" "${TAR_OPTIONS-unset}" "${GNUTARGET-unset}" "${GIT_DIR-unset}" \
          "${GIT_CONFIG_PARAMETERS-unset}" "${GIT_COMMITTER_NAME-unset}" "${DOCKER_BUILDKIT-unset}" "${JQ_COLORS-unset}"
        printf "global=%s;nosystem=%s;noreplace=%s;excludes=%s;attributes=%s;hooks=%s;fsmonitor=%s;gpgsign=%s;followtags=%s\n" \
          "${GIT_CONFIG_GLOBAL-unset}" "${GIT_CONFIG_NOSYSTEM-unset}" "${GIT_NO_REPLACE_OBJECTS-unset}" \
          "$(git config core.excludesFile)" "$(git config core.attributesFile)" "$(git config core.hooksPath)" \
          "$(git config core.fsmonitor)" "$(git config tag.gpgSign)" "$(git config push.followTags)"'
)
expect "variables: the loader's, tar's, objdump's, git's, docker's and jq's go, exported functions go, the committer stays" ok \
  '^fn=file;ld=unset;tar=unset;gnutarget=unset;git_dir=unset;git_params=unset;committer=Release Test;docker=unset;jq=unset;global=/dev/null;nosystem=1;noreplace=1;excludes=/dev/null;attributes=/dev/null;hooks=/dev/null;fsmonitor=false;gpgsign=false;followtags=false$' \
  env_case

# docker: an empty private DOCKER_CONFIG and the system socket, whatever ~/.docker and DOCKER_* say.
mkdir -p "$tc/docker-bin" "$tc/home-docker/.docker"
cat > "$tc/docker-bin/docker" <<'EOF'
#!/bin/sh
n=$(find "${DOCKER_CONFIG:-/nonexistent}" -mindepth 1 2>/dev/null | wc -l)
printf 'config=%s;entries=%s;host=%s;context=%s;platform=%s;args=%s\n' "${DOCKER_CONFIG-unset}" "$n" \
  "${DOCKER_HOST-unset}" "${DOCKER_CONTEXT-unset}" "${DOCKER_DEFAULT_PLATFORM-unset}" "$*"
EOF
chmod 0755 "$tc/docker-bin/docker"
printf '{"currentContext": "elsewhere", "proxies": {"default": {"httpProxy": "http://proxy.invalid:3128"}}}\n' \
  > "$tc/home-docker/.docker/config.json"
docker_case() (
  export HOME=$tc/home-docker DOCKER_CONTEXT=elsewhere DOCKER_HOST=tcp://docker.invalid:2375 \
    DOCKER_CONFIG=$tc/home-docker/.docker DOCKER_DEFAULT_PLATFORM=linux/arm64
  # PATH is read-only once release.sh is sourced, so the stand-in goes into bash's command table,
  # which `command docker` in the wrapper consults.
  unit 'hash -p "$1/docker" docker; out=$(docker version)
        want="config=$TMPDIR/docker-config;entries=0;host=unix:///var/run/docker.sock;context=unset;platform=unset;args=version"
        if [[ $out == "$want" ]]; then echo ok; else printf "%s\n" "$out"; exit 1; fi' "$tc/docker-bin"
)
expect "docker: every docker command gets an empty private DOCKER_CONFIG and the system socket" ok '^ok$' docker_case

# cosign: an empty environment on the host, with an empty private HOME.
stand_in "$pt/cosign-v2.4.3/cosign" cosign v2.4.3
expect "cosign: runs with nothing inherited (no COSIGN_*/SIGSTORE_* or other variable), HOME empty and private" ok \
  '^version=v2\.4\.3;leak=no;home=private;tmpdir=ok$' \
  unit 'export LEAK_CANARY=1 COSIGN_EXPERIMENTAL=1 SIGSTORE_ROOT_FILE=/nonexistent
        out=$(check_cosign "$1" "$(sha256_of "$1")" v2.4.3); e=$1.env
        if grep -qE "^(LEAK_CANARY|COSIGN_|SIGSTORE_)" "$e"; then l=yes; else l=no; fi
        printf "version=%s;leak=%s;home=%s;tmpdir=%s\n" "$(jq -r .gitVersion <<<"$out")" "$l" \
          "$(grep -qx "HOME=$TMPDIR/cosign-home" "$e" && [[ $(stat -c %a "$TMPDIR/cosign-home") == 700 ]] && echo private || echo bad)" \
          "$(grep -qx "TMPDIR=$TMPDIR" "$e" && echo ok || echo bad)"' "$pt/cosign-v2.4.3/cosign"

echo
total=$((pass + ${#failures[@]}))
printf '%s of %s cases passed\n' "$pass" "$total"
if (( ${#failures[@]} )); then
  printf 'failed: %s\n' "${failures[@]}"
  exit 1
fi
