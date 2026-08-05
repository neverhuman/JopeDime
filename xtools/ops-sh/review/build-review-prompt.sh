#!/usr/bin/env bash
set -Eeuo pipefail

# Generate one epoch's review prompt from live repository facts plus the epoch's
# lane file. The lane charges are NOT written here: hardcoding them is how the
# harness stops being reusable, and how the collector's role labels drift away
# from what the reviewers were actually asked. `feedback/lanes/epoch-NN.md` is
# the single source, read by this script and by the indexer.

bash -n "$0" || exit 2

die() { printf 'build-review-prompt: %s\n' "$*" >&2; exit 2; }

EPOCH=""
OUT=""
REPO=""
REVIEWS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epoch)   EPOCH="${2:?}";   shift 2 ;;
        --out)     OUT="${2:?}";     shift 2 ;;
        --repo)    REPO="${2:?}";    shift 2 ;;
        --reviews) REVIEWS="${2:?}"; shift 2 ;;
        -h|--help) printf 'Usage: %s --epoch NN --out FILE [--repo DIR] [--reviews DIR]\n' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ -n "$EPOCH" ]] || die "--epoch is required"
[[ -n "$OUT" ]]   || die "--out is required"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$REPO" ]]    || REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
[[ -n "$REVIEWS" ]] || REVIEWS="$REPO/feedback"

LANE_FILE="$REVIEWS/lanes/epoch-$EPOCH.md"
[[ -f "$LANE_FILE" ]] || LANE_FILE="$REVIEWS/lanes/default.md"
[[ -f "$LANE_FILE" ]] || die "no lane charges at $REVIEWS/lanes/epoch-$EPOCH.md and no default.md"

# Live facts, so the prompt cannot describe a repository that no longer exists.
RUST_LOC="$(git -C "$REPO" ls-files 'src/*.rs' 'vendor/*.rs' 2>/dev/null | xargs -r wc -l 2>/dev/null | tail -1 | awk '{print $1}')"
PY_LOC="$(git -C "$REPO" ls-files 'scripts/*.py' 2>/dev/null | xargs -r wc -l 2>/dev/null | tail -1 | awk '{print $1}')"
RUST_FILES="$(git -C "$REPO" ls-files 'src/*.rs' 2>/dev/null | wc -l)"
PY_FILES="$(git -C "$REPO" ls-files 'scripts/*.py' 2>/dev/null | wc -l)"

mkdir -p "$(dirname "$OUT")"

{
cat <<EOF
# jope-dime architecture review — epoch $EPOCH

You are one of ten independent reviewers. Every reviewer receives this identical
brief and returns exactly one engineering specification. Your tab number selects
which lane charge below is yours — answer only that one.

## The archive

The attached archive contains a single file, \`review-payload.tar.gz\`.

\`\`\`
tar -xzf review-payload.tar.gz    # -> review-context/
\`\`\`

It is a Git archive of the working tree: **$RUST_FILES Rust files** (~$RUST_LOC lines
across \`src/\` and the vendored \`feat-math\` kernel) and **$PY_FILES Python files**
(~$PY_LOC lines under \`scripts/\`, the training and evaluation side), plus
\`manifests/\` (frozen contracts and receipts), \`systemd/\` (the service topology),
\`tests/\`, \`docs/\`, and the prose at the root.

Deliberately absent, so you do not waste attention looking for them:

- **No datasets and no encodings.** The corpus (~25k datasets), the embedding cache,
  and the observation database live outside the repository. Where a file pins a
  dataset by sha, the sha is present and the bytes are not.
- **\`manifests/multiclass/mc-union-runnable272-v1.json\`** — 21 MB of packed dataset
  identities, excluded on size. It enumerates the 272 multiclass datasets.
- **\`sotarena/\` and \`sotarena-refresh/\`** — separate submodules, out of scope.
- **No build output.**

## The system

A self-play flywheel over tabular ML. A deep model proposes candidate pipelines
("genes") for a dataset; a genetic programme evolves its own candidates alongside;
an evaluator scores both on held-out data; the resulting observations become the
next model's training corpus. It runs continuously on a five-node cluster under a
custom scheduler.

Four task families — regression, binary, multiclass, timeseries — sharing one
task-conditioned model. Regression and binary have millions of observations each.
Multiclass and timeseries have **zero**: their lanes were closed by two-task
hard-codes that were only just removed, which is why several charges below concern
paths that have never carried live traffic.

Key vocabulary: a **wave** evaluates one frozen **cohort** of datasets; a
**shadow** is a candidate checkpoint battling the **accepted** one; a **canary** is
the panel that decides promotion; **observations** are scored gene/dataset pairs.

## What a good answer looks like

An engineering specification, not a review summary. Name files and functions from
the payload. Where you propose a design, give the concrete shape — types, invariants,
what fails closed and how. Where you find a defect, give the failure mode: the input
or state, and the wrong behaviour that results.

Judge the design, not the incident. Several charges cite real recent failures; they
are context for reasoning about what permitted them, not the subject.

If the payload does not contain what you need to answer, say so explicitly and scope
your claim rather than guessing — a lane that refuses to rule on a question it cannot
see is more useful than one that speculates.

EOF

# Prior epochs: ask reviewers to extend rather than restate.
PRIOR="$({ find "$REVIEWS/epochs" -path '*/reports/*.md' -type f 2>/dev/null || true; } | sort)"
if [[ -n "$PRIOR" ]]; then
    printf '## What earlier epochs already concluded\n\n'
    printf 'Extend or contradict these with reasons; do not restate them.\n\n'
    while IFS= read -r report; do
        [[ -n "$report" ]] || continue
        printf -- '- **%s** — %s\n' "$(basename "$report" .md)" \
            "$(grep -m1 '^#' "$report" 2>/dev/null | sed 's/^#* *//' | cut -c1-100)"
    done <<<"$PRIOR"
    printf '\n'
fi

printf '## Lane charges\n\n'
printf 'Your tab number selects one charge. Answer only that one.\n\n'
cat "$LANE_FILE"

cat <<'EOF'

## Deliverable

Return **exactly one downloadable archive named `source.tar.gz`** containing a single
file, `lane-report.md`. Do not include the payload, any code, or any other artifact.

Start the report with a `#` title naming your lane and its conclusion in one line —
that title is what the index shows.
EOF
} >"$OUT"

printf 'build-review-prompt: %s (%s bytes, lanes from %s)\n' \
    "$OUT" "$(stat -c %s "$OUT")" "$LANE_FILE"
