#!/usr/bin/env bash
# deploy/ci/test-bump.sh [<commit>]: `just bump` leaves every workspace of the tree consistent, or
# refuses with a message and a non-zero exit. The hygiene lane runs it after deploy/ci/lambda.sh,
# whose cargo metadata and cargo test have the Lambda workspaces' dependencies in the cargo cache
# by then (the bump and every check below run --offline).
#
# It exports <commit> (default HEAD) with git archive into a throwaway directory, makes that a git
# repository of its own (the Lambda checks read git ls-files) and runs everything there; the
# checkout is never written. BUMP_TEST_DIR moves the directory (default
# $HOME/.cache/jope-test-tmp/bump-selftest; it must stay under $HOME/.cache/jope-test-tmp/). The checks are this script's own and
# do not depend on the commit under test: release.sh's version check (VERSION_JQ, from the export),
# and, from this script's tree, deploy/ci/lambda-workspaces.sh verify (cargo metadata --locked
# --offline for every nested Lambda workspace, then the SNAPSHOT.json check). The cases:
#   - the export is consistent: the Lambda checks pass, and the version check says OK for its version;
#   - `just bump <patch + 1>` succeeds; afterwards the version check says OK for the new version and
#     MISMATCH for another one, the Lambda checks pass, nothing but the root manifests and lock, the
#     version lines of the nested locks and SNAPSHOT.json's tree_sha256/tree_change fields changed,
#     at least one nested lock changed, and SNAPSHOT.json records each such lock at the new version;
#   - `just bump` refuses (exit non-zero, with its message) a malformed version; a Lambda file that
#     differs from SNAPSHOT.json, before it changes anything; a nested lock whose SNAPSHOT.json record
#     names another change; and a cargo update that changes more than the version lines (a cargo
#     stand-in appends to the lock), each time leaving the Lambda lock as it was;
#   - snapshot-refresh refuses the gzipped corpus, an untracked file, a tracked file with no record
#     and a changed file without a reason; it records a changed file's bytes and reason and nothing
#     else, and records the frozen bytes again without a tree_change.
# Run on an older commit, it shows what that commit's `just bump` leaves behind (the
# snapshot-refresh cases test this script's own helper, so they pass there too).
#
# Usage: deploy/ci/test-bump.sh [<commit>]   (exit 0 only when every case passes)
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd "$here/../.." && pwd -P)
helper=$here/lambda-workspaces.sh
commit=$(git -C "$root" rev-parse --verify "${1:-HEAD}^{commit}")
# Confined like test-release.sh's RELEASE_TEST_DIR: the directory is reset below, so it must be a
# subdirectory of $HOME/.cache/jope-test-tmp (never $HOME itself, never /tmp).
test_parent=$(realpath -m -- "${HOME:?HOME must be set}/.cache/jope-test-tmp")
work=$(realpath -m -- "${BUMP_TEST_DIR:-$test_parent/bump-selftest}")
case $work/ in
  "$test_parent"/?*) ;;
  *) printf 'test-bump.sh: test dir must be under %s/: %s\n' "$test_parent" "$work" >&2; exit 2 ;;
esac
tree=$work/tree stubs=$work/stubs
# Only this script's own fixtures are reset; nothing else under $work is touched.
rm -rf -- "$tree" "$stubs" "$work/tmp" "$work/gitconfig"
mkdir -p -- "$tree" "$stubs" "$work/tmp"
export TMPDIR=$work/tmp
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=$work/gitconfig
cat > "$GIT_CONFIG_GLOBAL" <<'EOF'
[user]
	name = Bump Test
	email = bump-test@example.invalid
[init]
	defaultBranch = main
[commit]
	gpgsign = false
EOF

echo "# export of $commit in $tree"
git -C "$root" archive --format=tar "$commit" | tar -xf - -C "$tree"
git -C "$tree" init -q
git -C "$tree" add -A -f
git -C "$tree" commit -q -m "export of $commit"
current=$(grep -m1 -E '^version = "[^"]+"' "$tree/Cargo.toml")
current=${current#version = \"}
current=${current%\"}
[[ $current =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || { echo "unexpected workspace version: $current" >&2; exit 2; }
scratch=${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))
other=${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 2))

# A cargo that runs the real one and then, after a `cargo update --manifest-path <M>` (the nested
# lock sync; the root update names no manifest), appends a package to <M>'s Cargo.lock.
real_cargo=$(command -v cargo)
cat > "$stubs/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
"${BUMP_TEST_REAL_CARGO:?}" "$@"
if [[ ${1-} == update ]]; then
  manifest='' prev=''
  for arg in "$@"; do
    if [[ $prev == --manifest-path ]]; then manifest=$arg; fi
    prev=$arg
  done
  if [[ -n $manifest ]]; then
    printf '\n[[package]]\nname = "injected-by-test-bump"\nversion = "0.0.0"\n' >> "${manifest%Cargo.toml}Cargo.lock"
  fi
fi
EOF
chmod +x "$stubs/cargo"

reset_tree() { git -C "$tree" reset -q --hard && git -C "$tree" clean -q -f -d -x; }
bump() { just --justfile "$tree/justfile" --working-directory "$tree" bump "$@"; }
# release.sh's version check on the export's root workspace, as bump.sh and release.sh run it.
version_check() {
  local meta tmp verdict
  meta=$(cd "$tree" && cargo metadata --no-deps --format-version 1 --locked)
  tmp=$tree/target/test-bump-tmp
  mkdir -p -- "$tmp" && chmod 0700 -- "$tmp"
  # shellcheck disable=SC2016  # expanded by the bash -p that sources release.sh
  verdict=$(cd "$tree" && printf '%s\n' "$meta" \
    | env TMPDIR="$tmp" bash -p -c 'source deploy/release/release.sh && jq -r --arg want "$1" "$VERSION_JQ"' check "$1")
  printf '%s\n' "$verdict"
  [[ $verdict == OK ]]
}
lambda_checks() { (cd "$tree" && bash "$helper" verify); }
# Nothing but version records changed: the root manifests and lock, the version lines of the nested
# Lambda locks, and the tree_sha256 and tree_change fields of SNAPSHOT.json.
only_version_records_changed() {
  local path bad='' lines records='del(.files[].tree_sha256, .files[].tree_change)'
  while IFS= read -r path; do
    case $path in
      Cargo.toml | Cargo.lock | crates/jope-engine/Cargo.toml | crates/jope-campaigns/Cargo.toml) ;;
      deploy/lambda/*/Cargo.lock)
        lines=$(git -C "$tree" diff -U0 -- "$path" | grep -E '^[-+]' | grep -vE '^(\+\+\+|---) ' || true)
        if grep -qvE '^[-+]version = "[^"]+"$' <<<"$lines"; then bad+=" $path"; fi
        ;;
      deploy/lambda/SNAPSHOT.json)
        cmp -s <(git -C "$tree" show "HEAD:$path" | jq "$records") <(jq "$records" "$tree/$path") || bad+=" $path"
        ;;
      *) bad+=" $path" ;;
    esac
  done < <(git -C "$tree" diff --name-only)
  git -C "$tree" diff --stat | tail -n1
  if [[ -n $bad ]]; then echo "changes beyond version records in:$bad"; return 1; fi
  echo ok
}
# The nested locks that the bump changed, for the refusal cases.
synced_locks() { git -C "$tree" diff --name-only -- 'deploy/lambda/*Cargo.lock'; }
# Each nested lock that the bump changed has a SNAPSHOT.json tree_change that names every package
# it moved at the version it now pins.
records_name_the_new_version() {
  local lock record prose name bad='' re='^cargo update( -p [A-Za-z0-9_-]+)+ --offline: (.*); no other package moves$'
  (( ${#locks[@]} )) || { echo "the bump changed no Lambda lock"; return 1; }
  for lock in "${locks[@]}"; do
    record=$(jq -r --arg p "$lock" '.files[] | select(.tree_path == $p) | .tree_change // ""' \
      "$tree/deploy/lambda/SNAPSHOT.json")
    printf '%s: %s\n' "$lock" "$record"
    if [[ ! $record =~ $re ]]; then bad+=" $lock"; continue; fi
    prose=${BASH_REMATCH[2]}
    while IFS= read -r name; do
      [[ $prose == *"the $name path entry reads $scratch,"* || $prose == *"$name $scratch"* ]] || bad+=" $lock($name)"
    done < <(grep -oE ' -p [A-Za-z0-9_-]+' <<<"${record%% --offline:*}" | cut -c5-)
  done
  if [[ -n $bad ]]; then echo "records that do not name the new version:$bad"; return 1; fi
  echo ok
}
# unchanged <pathspec>...: those paths are as exported, and no file was left behind under
# deploy/lambda (such as a temporary copy of a lock).
unchanged() {
  local left
  git -C "$tree" diff --exit-code --stat -- "$@" || return 1
  left=$(git -C "$tree" ls-files --others -- deploy/lambda)
  if [[ -n $left ]]; then printf 'left behind: %s\n' "$left"; return 1; fi
  echo unchanged
}

pass=0
failures=()
# expect <name> <ok|refuse> <ERE the output must match> <command...>
expect() {
  local name=$1 want=$2 pattern=$3 out line rc=0 good=0
  shift 3
  out=$("$@" 2>&1) || rc=$?
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

echo "# the export as it is (workspace $current)"
expect "the Lambda workspaces are consistent as exported" ok 'tracked files match SNAPSHOT\.json' lambda_checks
expect "release.sh's version check says OK for $current" ok '^OK$' version_check "$current"

echo "# just bump $scratch"
expect "just bump $scratch succeeds" ok 'bumped the Rust workspace' bump "$scratch"
expect "the version check says OK for $scratch" ok '^OK$' version_check "$scratch"
expect "the version check refuses $other" refuse "MISMATCH jope-cli $scratch != tag version $other" version_check "$other"
expect "every Lambda workspace: cargo metadata --locked --offline, then the SNAPSHOT.json check" ok \
  'tracked files match SNAPSHOT\.json' lambda_checks
expect "only version records changed" ok '^ok$' only_version_records_changed
mapfile -t locks < <(synced_locks)
expect "the bump brought at least one Lambda lock along" ok 'Cargo\.lock' synced_locks
expect "SNAPSHOT.json records each of them at $scratch" ok '^ok$' records_name_the_new_version

echo "# just bump refuses"
reset_tree
expect "a malformed version" refuse 'the version must be MAJOR\.MINOR\.PATCH' bump "${scratch%.*}"
expect "  and changes nothing" ok '^unchanged$' unchanged .
reset_tree
tampered=$(jq -r '[.files[] | select(.disposition == "tracked" and (.tree_path | endswith(".md")))][0].tree_path' \
  "$tree/deploy/lambda/SNAPSHOT.json")
printf '\ntampered\n' >> "$tree/$tampered"
expect "a Lambda file that differs from SNAPSHOT.json ($tampered)" refuse \
  'not consistent before the bump.*nothing was changed' bump "$scratch"
expect "  and changes nothing else" ok '^unchanged$' unchanged . ":(exclude)$tampered"
for lock in "${locks[@]}"; do
  reset_tree
  # A newly imported lock still equals its frozen bytes. Adding a change record alone
  # must fail the snapshot check before sync-locks is reached. Retain that refusal,
  # then make an actual synthetic non-version edit to exercise the sync fence too.
  frozen=$(jq -r --arg p "$lock" '.files[] | select(.tree_path == $p) | .sha256' \
    "$tree/deploy/lambda/SNAPSHOT.json")
  now=$(sha256sum "$tree/$lock"); now=${now%% *}
  record=$(jq --arg p "$lock" '(.files[] | select(.tree_path == $p) | .tree_change) = "hand edit: another change"' \
    "$tree/deploy/lambda/SNAPSHOT.json")
  printf '%s\n' "$record" > "$tree/deploy/lambda/SNAPSHOT.json"
  if [[ $frozen == "$now" ]]; then
    expect "SNAPSHOT.json invents a change to unchanged $lock" refuse \
      "a tree_change for an unchanged one: $lock" bump "$scratch"
    expect "  and the lock is as exported" ok '^unchanged$' unchanged "$lock"
    expect "  and no release files changed" ok '^unchanged$' unchanged . ':(exclude)deploy/lambda/SNAPSHOT.json'
    reset_tree
    printf '\n# synthetic non-version hand edit\n' >> "$tree/$lock"
    (cd "$tree" && bash "$helper" snapshot-refresh "$lock" 'hand edit: another change')
    fixture_hash=$(sha256sum "$tree/$lock"); fixture_hash=${fixture_hash%% *}
    expect "SNAPSHOT.json records another change to $lock" refuse \
      'records another change to .*The lock is unchanged' bump "$scratch"
    expect "  and the hand-edited lock is unchanged" ok 'OK$' \
      bash -c 'printf "%s  %s\n" "$1" "$2" | sha256sum -c --strict' check "$fixture_hash" "$tree/$lock"
  else
    expect "SNAPSHOT.json records another change to $lock" refuse 'records another change to .*The lock is unchanged' \
      bump "$scratch"
    expect "  and the lock is as exported" ok '^unchanged$' unchanged "$lock"
  fi
  reset_tree
  expect "cargo update changes $lock beyond its version lines" refuse \
    'beyond the version lines of .*the lock is unchanged' \
    env "PATH=$stubs:$PATH" "BUMP_TEST_REAL_CARGO=$real_cargo" \
    just --justfile "$tree/justfile" --working-directory "$tree" bump "$scratch"
  expect "  and the lock is as exported, with no temporary file left" ok '^unchanged$' unchanged "$lock"
done

echo "# snapshot-refresh records a file's bytes and nothing else"
reset_tree
refresh() { (cd "$tree" && bash "$helper" snapshot-refresh "$@"); }
snapshot_check() { (cd "$tree" && bash "$helper" snapshot-check); }
snapshot_records_only() {  # SNAPSHOT.json differs from the export in tree_sha256/tree_change only
  local records='del(.files[].tree_sha256, .files[].tree_change)'
  cmp -s <(git -C "$tree" show HEAD:deploy/lambda/SNAPSHOT.json | jq "$records") \
    <(jq "$records" "$tree/deploy/lambda/SNAPSHOT.json") && echo same
}
gz=$(jq -r '[.files[] | select(.disposition == "tracked-gzipped")][0].tree_path // "none"' \
  "$tree/deploy/lambda/SNAPSHOT.json")
expect "refuses the gzipped corpus, whose frozen sha256 is of the inflated bytes ($gz)" refuse \
  'tracked-gzipped: its frozen sha256 is not of the tracked bytes' refresh "$gz" 'why'
printf 'new\n' > "$tree/deploy/lambda/new-file.txt"
expect "refuses an untracked file" refuse 'is not a tracked file' refresh deploy/lambda/new-file.txt 'why'
git -C "$tree" add deploy/lambda/new-file.txt
expect "refuses a tracked file with no record" refuse 'has 0 records in SNAPSHOT.json' \
  refresh deploy/lambda/new-file.txt 'why'
reset_tree
printf '\nchanged\n' >> "$tree/$tampered"
expect "refuses a changed file without a reason" refuse 'differs from its frozen bytes; say why' refresh "$tampered"
expect "records a changed file with its reason" ok "recorded $tampered: tree_sha256 [0-9a-f]{64} \\(tree_change: a test edit\\)" \
  refresh "$tampered" 'a test edit'
expect "  the SNAPSHOT.json check then passes" ok 'tracked files match SNAPSHOT\.json' snapshot_check
expect "  and only tree_sha256 and tree_change changed" ok '^same$' snapshot_records_only
git -C "$tree" checkout -q -- "$tampered"
expect "records the frozen bytes again, dropping the tree_change" ok "recorded $tampered: .*its frozen bytes again" \
  refresh "$tampered"
expect "  and SNAPSHOT.json is as exported" ok '^unchanged$' unchanged deploy/lambda/SNAPSHOT.json
reset_tree

echo
printf '%s passed, %s failed (export of %s)\n' "$pass" "${#failures[@]}" "$commit"
if (( ${#failures[@]} )); then
  printf 'failed: %s\n' "${failures[@]}"
  exit 1
fi
