#!/usr/bin/env bash
# deploy/ci/lambda-workspaces.sh: the one implementation of what the repository keeps consistent
# about the standalone AWS Lambda workspaces under deploy/lambda/ (deploy/lambda/README.md). The
# hygiene lane (deploy/ci/lambda.sh), `just bump` (deploy/release/bump.sh) and
# deploy/ci/test-bump.sh all call it. Run it from the repository root:
#
#   list        The nested workspaces: every tracked deploy/lambda/**/Cargo.toml, each of which must
#               declare its own [workspace]. Prints one manifest path per line.
#   snapshot-check
#               The tracked files under deploy/lambda/ are exactly deploy/lambda/SNAPSHOT.json's tree
#               paths, each file has its recorded tree_sha256, and each one whose bytes differ from
#               its frozen bytes (sha256) records why (tree_change), except the gzipped corpus, whose
#               sha256 is of the inflated bytes (deploy/ci/lambda.sh checks those).
#   snapshot-refresh <path> [<why>]
#               Records the current bytes of the tracked file <path> in SNAPSHOT.json: its
#               tree_sha256, and <why> as its tree_change. <why> is required when the bytes differ
#               from the frozen sha256, and the tree_change is dropped when they equal it again. Use
#               it after any deliberate change to a tracked file under deploy/lambda/. Nothing else in
#               SNAPSHOT.json changes, and the frozen sha256 is never written: a path with no frozen
#               record of its own bytes (not listed, or the gzipped corpus) is refused.
#   sync-locks  The Lambda half of `just bump`. For each workspace whose Cargo.lock holds a root
#               workspace member as a path package (feat-math today) at a version other than the
#               root's, runs `cargo update -p <package>... --offline` on that workspace, refuses
#               (restoring the lock) unless the lock then differs only in those packages' version
#               lines and they read the root versions, and records the lock with snapshot-refresh.
#               Then runs verify.
#   verify      Every workspace: cargo metadata --locked --offline. Then snapshot-check.
#
# Every refusal exits 1 with a line that says what failed.
set -euo pipefail

readonly SNAPSHOT=deploy/lambda/SNAPSHOT.json
# The tree_change that sync-locks writes, and the only one it replaces.
readonly SYNC_RECORD='^cargo update( -p [A-Za-z0-9_-]+)+ --offline: .*; no other package moves$'

fail() { printf 'lambda-workspaces.sh: %s\n' "$*" >&2; exit 1; }

at_root() {
  local top
  top=$(git rev-parse --show-toplevel 2>/dev/null) || fail "not inside a git checkout"
  [[ $top == "$(pwd -P)" && -f $SNAPSHOT ]] \
    || fail "run it from the repository root (the directory holding $SNAPSHOT), not $(pwd -P)"
}

workspaces() {
  local out m
  out=$(git ls-files -- 'deploy/lambda/*Cargo.toml' | { grep -E '(^|/)Cargo\.toml$' || true; } | LC_ALL=C sort)
  [[ -n $out ]] || fail "no Cargo.toml is tracked under deploy/lambda/"
  while IFS= read -r m; do
    # A manifest without its own [workspace] would be looked up from the root workspace instead.
    grep -qx '\[workspace\]' "$m" || fail "$m declares no [workspace] of its own"
  done <<<"$out"
  printf '%s\n' "$out"
}

snapshot_check() {
  local listed tracked unexplained
  listed=$(jq -r '.files[] | select(.tree_path != null) | .tree_path' "$SNAPSHOT" | LC_ALL=C sort)
  tracked=$(git ls-files -- deploy/lambda \
    | { grep -vxE 'deploy/lambda/(SNAPSHOT\.json|README\.md|\.gitattributes)' || true; } | LC_ALL=C sort)
  if [[ $listed != "$tracked" ]]; then
    diff <(printf '%s\n' "$listed") <(printf '%s\n' "$tracked") || true
    fail "the tracked files under deploy/lambda/ differ from SNAPSHOT.json's tree paths (< SNAPSHOT.json, > git)"
  fi
  jq -r '.files[] | select(.tree_path != null) | .tree_sha256 + "  " + .tree_path' "$SNAPSHOT" \
    | sha256sum -c --strict --quiet \
    || fail "a tracked file under deploy/lambda/ differs from its SNAPSHOT.json tree_sha256 (above); record a deliberate change with: bash deploy/ci/lambda-workspaces.sh snapshot-refresh <path> '<why>'"
  unexplained=$(jq -r '.files[] | select(.tree_path != null)
    | select(if .tree_sha256 == .sha256 then has("tree_change")
             else .disposition != "tracked-gzipped" and ((.tree_change // "") | length) == 0 end)
    | .tree_path' "$SNAPSHOT")
  [[ -z $unexplained ]] \
    || fail "SNAPSHOT.json records a changed file without a tree_change, or a tree_change for an unchanged one: ${unexplained//$'\n'/, }"
  echo "ok: $(wc -l <<<"$listed") tracked files match SNAPSHOT.json"
}

snapshot_refresh() {
  local path=${1-} why=${2-} records disposition frozen now tmp
  [[ -n $path ]] || fail "usage: lambda-workspaces.sh snapshot-refresh <path> [<why>]"
  git ls-files --error-unmatch -- "$path" >/dev/null 2>&1 || fail "$path is not a tracked file"
  [[ -f $path && ! -L $path ]] || fail "$path is not a regular file"
  records=$(jq --arg p "$path" '[.files[] | select(.tree_path == $p)] | length' "$SNAPSHOT")
  [[ $records == 1 ]] \
    || fail "$path has $records records in SNAPSHOT.json, not one; a file without a frozen record needs an import record, not a refresh"
  disposition=$(jq -r --arg p "$path" '.files[] | select(.tree_path == $p) | .disposition' "$SNAPSHOT")
  frozen=$(jq -r --arg p "$path" '.files[] | select(.tree_path == $p) | .sha256 // ""' "$SNAPSHOT")
  case $disposition in
    tracked | tracked-renamed) ;;
    *) fail "$path is $disposition: its frozen sha256 is not of the tracked bytes, so a refresh would lose that record; re-import it instead" ;;
  esac
  [[ $frozen =~ ^[0-9a-f]{64}$ ]] || fail "$path has no frozen sha256 in SNAPSHOT.json to keep"
  now=$(sha256sum -- "$path")
  now=${now%% *}
  if [[ $now != "$frozen" && -z $why ]]; then
    fail "$path differs from its frozen bytes; say why: snapshot-refresh $path '<why>'"
  fi
  # A temporary file beside SNAPSHOT.json, so that the rename replaces it in one step.
  tmp=$(mktemp "$SNAPSHOT.XXXXXX")
  # shellcheck disable=SC2064  # expand $tmp now
  trap "rm -f -- '$tmp'" EXIT
  jq --arg p "$path" --arg h "$now" --arg why "$why" '
    .files |= map(if .tree_path == $p then
        .tree_sha256 = $h | if $h == .sha256 then del(.tree_change) else .tree_change = $why end
      else . end)' "$SNAPSHOT" > "$tmp"
  cmp -s <(jq 'del(.files[].tree_sha256, .files[].tree_change)' "$SNAPSHOT") \
    <(jq 'del(.files[].tree_sha256, .files[].tree_change)' "$tmp") \
    || fail "refusing to write SNAPSHOT.json: more than $path's tree_sha256 and tree_change would change"
  chmod --reference="$SNAPSHOT" -- "$tmp"
  mv -f -- "$tmp" "$SNAPSHOT"
  trap - EXIT
  if [[ $now == "$frozen" ]]; then
    echo "recorded $path: tree_sha256 $now, its frozen bytes again (no tree_change)"
  else
    echo "recorded $path: tree_sha256 $now (tree_change: $why)"
  fi
}

# lock_path_versions <lock>: "<name>\t<version>" for every package of a Cargo.lock that has no
# source line, that is, every path package.
lock_path_versions() {
  awk '
    function flush() { if (inpkg && !src && name != "") print name "\t" ver; inpkg = 0; name = ""; ver = ""; src = 0 }
    /^\[\[package\]\]$/ { flush(); inpkg = 1; next }
    /^\[/ { flush(); next }
    inpkg && /^name = "/ { name = substr($0, 9, length($0) - 9) }
    inpkg && /^version = "/ { ver = substr($0, 12, length($0) - 12) }
    inpkg && /^source = / { src = 1 }
    END { flush() }' "$1"
}

# lock_view <lock> <name>...: the lock with the version line of each named path package blanked, so
# that two views are equal exactly when the locks differ in nothing but those version lines.
lock_view() {
  local lock=$1
  shift
  awk -v names="$*" '
    BEGIN { n = split(names, list, " "); for (i = 1; i <= n; i++) want[list[i]] = 1 }
    function flush(   i) {
      for (i = 1; i <= nb; i++)
        if (inpkg && !src && (name in want) && block[i] ~ /^version = "/) print "version = <synced>"
        else print block[i]
      nb = 0; inpkg = 0; name = ""; src = 0
    }
    /^\[/ { flush(); if ($0 == "[[package]]") inpkg = 1 }
    { block[++nb] = $0 }
    inpkg && /^name = "/ { name = substr($0, 9, length($0) - 9) }
    inpkg && /^source = / { src = 1 }
    END { flush() }' "$lock"
}

# version_of <name> <"name\tversion" lines>: the version recorded for <name>, or nothing.
version_of() { awk -F'\t' -v n="$1" '$1 == n { print $2; exit }' <<<"$2"; }

# restore_lock <saved copy> <lock>: puts the lock back as it was before sync-locks touched it.
restore_lock() {
  cp -p -- "$1" "$2"
  rm -f -- "$1"
  trap - EXIT
}

sync_locks() {
  local meta root manifests m lock stale saved record names name have want got why
  local -a args entries
  meta=$(cargo metadata --no-deps --format-version 1 --locked) \
    || fail "cargo metadata --locked failed for the root workspace (above)"
  root=$(jq -r '.workspace_members as $m | .packages[] | select(.id as $id | any($m[]; . == $id))
    | "\(.name)\t\(.version)"' <<<"$meta")
  manifests=$(workspaces)
  while IFS= read -r m; do
    lock=${m%Cargo.toml}Cargo.lock
    git ls-files --error-unmatch -- "$lock" >/dev/null 2>&1 || fail "$m has no tracked Cargo.lock beside it"
    # "<name>\t<lock version>\t<root version>" for each root member that the lock pins elsewhere.
    stale=$(awk -F'\t' 'NR == FNR { v[$1] = $2; next } ($1 in v) && v[$1] != $2 { print $1 "\t" $2 "\t" v[$1] }' \
      <(printf '%s\n' "$root") <(lock_path_versions "$lock"))
    if [[ -z $stale ]]; then
      echo "ok: $lock pins no root workspace member at another version"
      continue
    fi
    record=$(jq -r --arg p "$lock" '.files[] | select(.tree_path == $p) | .tree_change // ""' "$SNAPSHOT")
    if [[ -n $record && ! $record =~ $SYNC_RECORD ]]; then
      fail "SNAPSHOT.json records another change to $lock (\"$record\"); bring it to the root versions by hand and record it (snapshot-refresh). The lock is unchanged"
    fi
    names=$(cut -f1 <<<"$stale")
    args=()
    while IFS= read -r name; do args+=(-p "$name"); done <<<"$names"
    saved=$(mktemp "$lock.XXXXXX")
    # shellcheck disable=SC2064  # expand $saved now
    trap "rm -f -- '$saved'" EXIT
    cp -p -- "$lock" "$saved"
    echo "== cargo update ${args[*]} --offline --manifest-path $m"
    if ! cargo update "${args[@]}" --offline --manifest-path "$m" </dev/null; then
      restore_lock "$saved" "$lock"
      fail "cargo update ${args[*]} --offline failed for $m (cargo's error above); the lock is unchanged. If cargo reports a missing package, the cargo cache is cold: cargo fetch --locked --manifest-path $m, then rerun; otherwise fix what cargo reports (for example a version requirement on the path dependency)"
    fi
    # shellcheck disable=SC2086  # a package name is one word
    if ! diff <(lock_view "$saved" $names) <(lock_view "$lock" $names); then
      restore_lock "$saved" "$lock"
      fail "cargo update ${args[*]} changed $lock beyond the version lines of $(paste -sd' ' <<<"$names") (diff above); the lock is unchanged. Bring it to the root versions by hand and record it (snapshot-refresh)"
    fi
    have=$(lock_path_versions "$lock")
    while IFS=$'\t' read -r name _ want; do
      got=$(version_of "$name" "$have")
      if [[ $got != "$want" ]]; then
        restore_lock "$saved" "$lock"
        fail "$lock pins $name at ${got:-nothing} after cargo update, not at the root version $want; the lock is unchanged"
      fi
    done <<<"$stale"
    rm -f -- "$saved"
    trap - EXIT
    # The record names every package that this mechanism has moved: the earlier record's and these.
    names=$( { grep -oE ' -p [A-Za-z0-9_-]+' <<<"${record%% --offline:*}" | cut -c5- || true; printf '%s\n' "$names"; } \
      | LC_ALL=C sort -u)
    args=() entries=()
    while IFS= read -r name; do
      args+=(-p "$name")
      entries+=("$name $(version_of "$name" "$have")")
    done <<<"$names"
    if (( ${#entries[@]} == 1 )); then
      why="cargo update ${args[*]} --offline: the ${entries[0]% *} path entry reads ${entries[0]##* }, the workspace version of this release; no other package moves"
    else
      why="cargo update ${args[*]} --offline: the path entries $(printf '%s, ' "${entries[@]}" | sed 's/, $//') read the workspace versions of this release; no other package moves"
    fi
    snapshot_refresh "$lock" "$why"
  done <<<"$manifests"
  verify
}

verify() {
  local manifests m
  manifests=$(workspaces)
  while IFS= read -r m; do
    cargo metadata --locked --offline --format-version 1 --manifest-path "$m" >/dev/null </dev/null \
      || fail "cargo metadata --locked --offline failed for $m (above): its Cargo.lock does not match its manifest and the root crates it takes by path, or its dependencies are not in the local cargo cache (cargo fetch --locked --manifest-path $m)"
    echo "ok: cargo metadata --locked --offline --manifest-path $m"
  done <<<"$manifests"
  snapshot_check
}

at_root
case ${1-} in
  list) (( $# == 1 )) || fail "list takes no arguments"; workspaces ;;
  snapshot-check) (( $# == 1 )) || fail "snapshot-check takes no arguments"; snapshot_check ;;
  snapshot-refresh) (( $# == 2 || $# == 3 )) || fail "usage: lambda-workspaces.sh snapshot-refresh <path> [<why>]"; snapshot_refresh "$2" "${3-}" ;;
  sync-locks) (( $# == 1 )) || fail "sync-locks takes no arguments"; sync_locks ;;
  verify) (( $# == 1 )) || fail "verify takes no arguments"; verify ;;
  *) fail "usage: lambda-workspaces.sh list | snapshot-check | snapshot-refresh <path> [<why>] | sync-locks | verify" ;;
esac
