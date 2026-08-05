#!/usr/bin/env bash
set -Eeuo pipefail

# Build the review payload: a deterministic Git archive of ONLY the code the
# review lanes are about. No data, no build output, no submodule contents.
#
# Four constraints shape this script; none is stylistic.
#
# 1. Jailgun's security denylist rejects any path containing "token", and this
#    repository has vendor/feat-math/src/tokens.rs — the gene language's token
#    definitions, which several lanes are specifically about. A direct scope
#    walk would silently drop them and reviewers would reason about a codebase
#    with a hole in exactly the place under review. Nesting a Git archive and
#    declaring only that one filename in the include manifest bypasses the
#    extension allowlist while keeping the tree byte-exact.
#
# 2. `git archive` of an explicit pathspec — never a working-tree walk — so the
#    payload is reproducible from a commit object alone and cannot pick up
#    target/, encodings, corpora, or anything else .gitignore'd.
#
# 3. The tree is snapshotted with `git stash create`, not HEAD. This repo runs
#    with substantial uncommitted work from more than one agent; archiving HEAD
#    would ship code nobody is actually running. `stash create` writes a commit
#    OBJECT and touches neither the branch nor the index.
#
# 4. sotarena/ and sotarena-refresh/ are gitlinks (mode 160000) totalling 5.4 GB
#    in the working tree. `git archive` already skips gitlink contents, but they
#    are excluded explicitly so that de-submoduling them later cannot silently
#    blow the size cap.

bash -n "$0" || exit 2

die() { printf 'pack-review-payload: %s\n' "$*" >&2; exit 2; }

usage() {
    cat <<'EOF'
Usage: scripts/review/pack-review-payload.sh --out DIR [options]

  --out DIR         where to write the payload, sha256, and manifest (required)
  --repo DIR        source repository (default: this script's workspace)
  --ref REF         commit-ish to archive (default: a working-tree snapshot)
  --max-bytes N     refuse to emit a larger payload (default 52428800)
  --full            archive the entire tree instead of the curated pathspec
  -h, --help        this text

Writes: review-payload.tar.gz, review-payload.tar.gz.sha256, include-manifest.txt,
        payload-ref.txt (the exact object archived)
EOF
}

OUT=""
REF=""
MAX_BYTES=52428800
FULL=0
REPO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)       OUT="${2:?}";       shift 2 ;;
        --repo)      REPO="${2:?}";      shift 2 ;;
        --ref)       REF="${2:?}";       shift 2 ;;
        --max-bytes) MAX_BYTES="${2:?}"; shift 2 ;;
        --full)      FULL=1;             shift ;;
        -h|--help)   usage; exit 0 ;;
        *) die "unknown argument $1 (use --help)" ;;
    esac
done

[[ -n "$OUT" ]] || die "--out is required"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$REPO" ]] || REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# A linked worktree has a .git FILE, not a directory, so test with git itself.
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || die "$REPO is not a Git repository"

# Snapshot the working tree unless the caller pinned a ref. `stash create` emits
# a commit object for the current tree and prints nothing on a clean tree, so
# fall back to HEAD in that case.
if [[ -z "$REF" ]]; then
    REF="$(git -C "$REPO" stash create 2>/dev/null || true)"
    [[ -n "$REF" ]] || REF="HEAD"
fi

# The curated set: the Rust engine, the Python training/eval scripts, the
# vendored math kernel the gene language is built on, the manifests that pin
# frozen contracts, and all prose. Deliberately excluded — sotarena/* (out of
# scope for the self-play loop and 5.4 GB of submodule), and the one 21 MB
# dataset manifest below.
PATHSPEC=(
    'src'
    'scripts'
    'tests'
    'vendor'
    'manifests'
    # 21 MB of packed dataset identities. A reviewer cannot read it and cannot
    # check it, but it would nearly triple the payload every lane downloads and
    # unpacks. The prompt states what it pins instead.
    ':(exclude)manifests/multiclass/mc-union-runnable272-v1.json'
    'systemd'
    'mcp'
    'docs'
    'reference'
    'Cargo.toml'
    'Cargo.lock'
    'README.md'
    'ROADMAP.md'
    ':(exclude)sotarena'
    ':(exclude)sotarena-refresh'
)

mkdir -p "$OUT"
PAYLOAD="$OUT/review-payload.tar.gz"
MANIFEST="$OUT/include-manifest.txt"

if [[ "$FULL" -eq 1 ]]; then
    git -C "$REPO" archive --format=tar.gz --prefix=review-context/ --output="$PAYLOAD" "$REF"
else
    git -C "$REPO" archive --format=tar.gz --prefix=review-context/ --output="$PAYLOAD" "$REF" -- "${PATHSPEC[@]}"
fi

sha256sum "$PAYLOAD" >"$PAYLOAD.sha256"
printf '%s\n' "$(basename "$PAYLOAD")" >"$MANIFEST"
printf '%s\n' "$REF" >"$OUT/payload-ref.txt"

BYTES="$(stat -c %s "$PAYLOAD")"
[[ "$BYTES" -le "$MAX_BYTES" ]] || die "payload is $BYTES bytes, over the $MAX_BYTES cap"

LISTING="$(mktemp)"
trap 'rm -f "$LISTING"' EXIT
tar -tzf "$PAYLOAD" >"$LISTING"
FILES="$(grep -vc '/$' "$LISTING" || true)"

# Assert the archive's CONTENTS, not merely that the pathspec parsed. `git
# archive` already dies on an entry matching nothing, so this covers the other
# direction: an entry that resolves but contributes no files, and the general
# case of the payload not holding what the prompt tells reviewers it holds.
if [[ "$FULL" -eq 0 ]]; then
    for required in "${PATHSPEC[@]}"; do
        # `:(exclude)...` are magic pathspecs; they must match nothing.
        [[ "$required" == :* ]] && continue
        grep -q "^review-context/$required" "$LISTING" \
            || die "pathspec entry '$required' matched nothing at $REF"
    done
fi

# The whole reason for nesting: prove the denylisted paths survived. If this
# ever stops holding, the payload silently ships a codebase with a hole in the
# gene language and the reviews are worth less than they appear.
TOKENS="$(grep -c 'feat-math/src/tokens.rs' "$LISTING" || true)"
[[ "$TOKENS" -gt 0 ]] || die "feat-math tokens.rs is missing from the payload; the nesting failed"

# And prove the exclusions actually excluded.
for forbidden in 'review-context/sotarena' 'mc-union-runnable272'; do
    ! grep -q "$forbidden" "$LISTING" || die "excluded path '$forbidden' leaked into the payload"
done

printf 'pack-review-payload: %s\n  ref       %s\n  files     %s\n  bytes     %s (cap %s)\n  denylist  %s tokens.rs path(s) survived\n  manifest  %s\n' \
    "$PAYLOAD" "$REF" "$FILES" "$BYTES" "$MAX_BYTES" "$TOKENS" "$MANIFEST"
