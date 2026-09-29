#!/usr/bin/env bash
# deploy/ci/lambda.sh: build and test the standalone AWS Lambda workspaces under deploy/lambda/.
#
# Every tracked deploy/lambda/**/Cargo.toml declares its own [workspace] and has its own frozen
# Cargo.lock. None is a member of the root workspace, and none is part of the jope binary or its
# release build (deploy/lambda/README.md). The hygiene lane runs this script
# (deploy/ci/fast-lane.sh hygiene). It:
#   1. checks the tracked files under deploy/lambda/ against deploy/lambda/SNAPSHOT.json;
#   2. inflates deploy/lambda/csv/gp/corpus/train-jobs.json from its tracked .gz (the raw file is
#      over check.sh's 5 MiB size cap) and checks the sha256 of the inflated bytes before any
#      build; the inflated file is gitignored, so the tree stays clean;
#   3. per workspace: cargo metadata --locked, cargo test --locked, and cargo-deny (bans, licenses,
#      sources) against the root deny.toml when cargo-deny is installed. The hygiene lane asserts
#      the image's cargo-deny before it runs this; locally a missing cargo-deny is skipped with a
#      note.
#
# Compile-time inputs of the frozen csv sources:
#   - JOPE_SOURCE_HEAD (env! in bootstrap and bootstrap-batch): JOPE_SOURCE_SHA in CI, otherwise
#     the checked-out commit;
#   - JOPE_GP_MANIFEST_PATH (include_str!(env!(...)) in bootstrap-gp-extra): the tracked
#     extra-train-jobs.json, the manifest that binary was built and deployed with (sha256
#     3a838ea7..., deploy/lambda/csv/gp/REVIEW.md; campaign deploy-extra checked the binary embeds it);
#   - gp/corpus/train-jobs.json (include_str! in bootstrap-gp and bootstrap-gp-diagnose): step 2.
#
# CARGO_TARGET_DIR is honoured when set (CI's persistent lane target); otherwise each workspace
# builds into its own target/, which **/target/ in .gitignore covers. Nothing is written under /tmp:
# when TMPDIR is unset or under /tmp, it becomes lambda-tmp/ in the target directory.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$(pwd -P)
step() { printf '\n== %s\n' "$*"; }
fail() { printf 'lambda.sh: %s\n' "$*" >&2; exit 1; }

target_base=${CARGO_TARGET_DIR:-$root/target}
case ${TMPDIR:-/tmp} in
  /tmp | /tmp/*)
    export TMPDIR=$target_base/lambda-tmp
    mkdir -p -- "$TMPDIR"
    ;;
esac

step "tracked files match deploy/lambda/SNAPSHOT.json"
snapshot=deploy/lambda/SNAPSHOT.json
listed=$(jq -r '.files[] | select(.tree_path != null) | .tree_path' "$snapshot" | LC_ALL=C sort)
tracked=$(git ls-files -- deploy/lambda | grep -vxE 'deploy/lambda/(SNAPSHOT\.json|README\.md|\.gitattributes)' | LC_ALL=C sort)
if [[ $listed != "$tracked" ]]; then
  diff <(printf '%s\n' "$listed") <(printf '%s\n' "$tracked") || true
  fail "the tracked files under deploy/lambda/ differ from SNAPSHOT.json's tree paths (< SNAPSHOT.json, > git)"
fi
jq -r '.files[] | select(.tree_path != null) | .tree_sha256 + "  " + .tree_path' "$snapshot" \
  | sha256sum -c --strict --quiet || fail "a tracked file under deploy/lambda/ differs from its SNAPSHOT.json sha256"
echo "ok: $(wc -l <<<"$listed") tracked files match SNAPSHOT.json"

step "inflate the csv corpus manifest"
corpus=deploy/lambda/csv/gp/corpus/train-jobs.json
corpus_sha=687f22c82397af9606455473e70e825c612c469ad917378c9cfb578d3590ee97
[[ $(jq -r --arg p "$corpus.gz" '.files[] | select(.tree_path == $p) | .sha256' "$snapshot") == "$corpus_sha" ]] \
  || fail "SNAPSHOT.json does not record $corpus_sha for $corpus.gz"
if [[ -f $corpus && $(sha256sum < "$corpus" | cut -d' ' -f1) == "$corpus_sha" ]]; then
  echo "ok: $corpus is already in place (sha256 $corpus_sha)"
else
  inflated=$(mktemp "$TMPDIR/train-jobs.XXXXXX")
  gzip -dc -- "$corpus.gz" > "$inflated"
  got=$(sha256sum < "$inflated" | cut -d' ' -f1)
  if [[ $got != "$corpus_sha" ]]; then
    rm -f -- "$inflated"
    fail "$corpus.gz inflates to sha256 $got, not $corpus_sha"
  fi
  chmod 0644 -- "$inflated"
  mv -f -- "$inflated" "$corpus"
  echo "ok: inflated $corpus (sha256 $corpus_sha)"
fi
git check-ignore -q -- "$corpus" || fail "$corpus is not gitignored"

head=${JOPE_SOURCE_SHA:-}
[[ -n $head ]] || head=$(git rev-parse --verify 'HEAD^{commit}')
export JOPE_SOURCE_HEAD=$head
export JOPE_GP_MANIFEST_PATH=$root/deploy/lambda/csv/gp/corpus/extra-train-jobs.json
[[ -f $JOPE_GP_MANIFEST_PATH ]] || fail "missing $JOPE_GP_MANIFEST_PATH"

mapfile -t manifests < <(git ls-files -- 'deploy/lambda/*Cargo.toml' | grep -E '(^|/)Cargo\.toml$' | LC_ALL=C sort)
(( ${#manifests[@]} > 0 )) || fail "no Cargo.toml is tracked under deploy/lambda/"
deny=0
if cargo deny --version >/dev/null 2>&1; then deny=1; fi
passed=0
for manifest in "${manifests[@]}"; do
  # A manifest without its own [workspace] would be looked up from the root workspace instead.
  grep -qx '\[workspace\]' "$manifest" || fail "$manifest declares no [workspace] of its own"
  step "cargo metadata --locked ($manifest)"
  cargo metadata --locked --format-version 1 --manifest-path "$manifest" > /dev/null
  echo ok
  step "cargo test --locked ($manifest; JOPE_SOURCE_HEAD=$JOPE_SOURCE_HEAD)"
  log=$(mktemp "$TMPDIR/lambda-test.XXXXXX")
  cargo test --locked --manifest-path "$manifest" 2>&1 | tee "$log"
  n=$(awk '/^test result: ok\./ {s += $4} END {print s + 0}' "$log")
  rm -f -- "$log"
  (( n > 0 )) || fail "cargo test ran no tests for $manifest"
  passed=$((passed + n))
  echo "ok: $n tests passed"
  step "cargo deny check bans licenses sources ($manifest)"
  if (( deny )); then
    cargo deny --manifest-path "$manifest" --all-features check --config "$root/deny.toml" bans licenses sources
  else
    echo "cargo-deny is not installed: skipped here (the hygiene lane asserts the image's cargo-deny before it runs this)"
  fi
done
step "lambda workspaces: ${#manifests[@]} built and tested, $passed tests passed"
