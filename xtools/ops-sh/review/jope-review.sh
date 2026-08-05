#!/usr/bin/env bash
set -Eeuo pipefail

# One command: package the code, ask ten GPT-5.6 Sol Pro reviewers the epoch's
# ten lane charges, collect their specifications, and index them.
#
#     scripts/review/jope-review.sh
#
# Re-running produces a NEW epoch. Each epoch's prompt lists what earlier epochs
# concluded and asks reviewers to extend rather than repeat, so the report folder
# grows in coverage instead of accumulating restatements.

bash -n "$0" || exit 2

die() { printf 'jope-review: %s\n' "$*" >&2; exit 2; }

usage() {
    cat <<'EOF'
Usage: scripts/review/jope-review.sh [options]

  --lanes N          reviewers to run, 1..10 (default 10; two accounts x 5 tabs)
  --epoch NN         force an epoch number instead of auto-incrementing
  --collect-only NN  skip launching; just collect and index that epoch
  --dry-run          build payload + prompt and validate, submit nothing
  -h, --help         this text

Output lands in feedback/epochs/epoch-NN/reports/ and is indexed in
feedback/INDEX.md.
EOF
}

LANES=10
EPOCH=""
COLLECT_ONLY=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lanes)        LANES="${2:?}";        shift 2 ;;
        --epoch)        EPOCH="${2:?}";        shift 2 ;;
        --collect-only) COLLECT_ONLY="${2:?}"; shift 2 ;;
        --dry-run)      DRY_RUN=1;             shift ;;
        -h|--help)      usage; exit 0 ;;
        *) die "unknown argument $1 (use --help)" ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
REVIEWS="$REPO/feedback"
INDEX="$REVIEWS/INDEX.md"

# The collector names reports by tab position, so a label is how a file becomes
# readable. Labels come from the same per-epoch lane file the prompt is built
# from, which is the only way they cannot drift: a hardcoded table here would
# still say "cohort shape" for an epoch whose lane 1 is an arbitration.
role_for() {
    local lane="$1" epoch="$2" file
    file="$REVIEWS/lanes/epoch-$epoch.md"
    [[ -f "$file" ]] || file="$REVIEWS/lanes/default.md"
    if [[ -f "$file" ]]; then
        local title
        title="$(sed -n "s/^## lane 0*${lane#0} — //p;s/^## lane ${lane} — //p" "$file" | head -1)"
        [[ -z "$title" ]] || { echo "$title"; return 0; }
    fi
    echo "lane $lane"
}

index_epoch() {
    # Two statements, not one: bash declares every name in a single `local`
    # before assigning any of them, so referring to $epoch while declaring
    # `reports` reads the not-yet-assigned local and trips `set -u`.
    local epoch="$1"
    local reports="$REVIEWS/epochs/epoch-$epoch/reports"
    [[ -d "$reports" ]] || return 0
    if [[ ! -f "$INDEX" ]]; then
        mkdir -p "$REVIEWS"
        {
            printf '# jope-dime review index\n\n'
            printf 'Engineering specifications from GPT-5.6 Sol Pro review epochs. Each\n'
            printf 'epoch assigns its own ten lane charges in `feedback/lanes/epoch-NN.md`;\n'
            printf 'the role column below is read from that file.\n\n'
        } >"$INDEX"
    fi
    # Rewrite this epoch's block so re-collecting never duplicates rows.
    local tmp; tmp="$(mktemp)"
    awk -v marker="## Epoch $epoch" '
        $0 == marker { skip = 1 }
        skip && /^## Epoch / && $0 != marker { skip = 0 }
        !skip { print }
    ' "$INDEX" >"$tmp"
    mv "$tmp" "$INDEX"
    {
        printf '## Epoch %s\n\n' "$epoch"
        printf '| lane | role | title | file |\n|---|---|---|---|\n'
        local report lane title
        for report in "$reports"/*.md; do
            [[ -e "$report" ]] || continue
            lane="$(basename "$report" .md)"; lane="${lane##*-lane-}"
            title="$(grep -m1 '^#' "$report" 2>/dev/null | sed 's/^#* *//' | cut -c1-80)"
            [[ -n "$title" ]] || title="(untitled)"
            printf '| %s | %s | %s | `%s` |\n' "$lane" "$(role_for "$lane" "$epoch")" "$title" \
                "feedback/epochs/epoch-$epoch/reports/$(basename "$report")"
        done
        printf '\n'
    } >>"$INDEX"
    printf 'jope-review: indexed epoch %s into %s\n' "$epoch" "$INDEX"
}

if [[ -n "$COLLECT_ONLY" ]]; then
    "$SCRIPT_DIR/collect-review-epoch.sh" --epoch "$COLLECT_ONLY" --lanes "$LANES" --repo "$REPO"
    index_epoch "$COLLECT_ONLY"
    exit 0
fi

if [[ -z "$EPOCH" ]]; then
    # `find` on a not-yet-existing epochs directory exits non-zero, and under
    # `pipefail` that kills the whole script before it prints anything.
    LAST="$({ find "$REVIEWS/epochs" -maxdepth 1 -type d -name 'epoch-*' 2>/dev/null || true; } | sed 's/.*epoch-//' | sort -n | tail -1)"
    EPOCH="$(printf '%02d' "$(( 10#${LAST:-0} + 1 ))")"
fi

printf 'jope-review: epoch %s, %s lane(s)\n' "$EPOCH" "$LANES"

if [[ "$DRY_RUN" -eq 1 ]]; then
    "$SCRIPT_DIR/run-review-epoch.sh" --epoch "$EPOCH" --lanes "$LANES" --repo "$REPO" --dry-run
    exit 0
fi

"$SCRIPT_DIR/run-review-epoch.sh" --epoch "$EPOCH" --lanes "$LANES" --repo "$REPO"
"$SCRIPT_DIR/collect-review-epoch.sh" --epoch "$EPOCH" --lanes "$LANES" --repo "$REPO"
index_epoch "$EPOCH"

printf 'jope-review: epoch %s complete — specifications in feedback/epochs/epoch-%s/reports/\n' "$EPOCH" "$EPOCH"
