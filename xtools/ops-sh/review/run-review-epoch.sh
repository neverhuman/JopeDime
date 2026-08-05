#!/usr/bin/env bash
set -Eeuo pipefail

# Launch one review epoch: up to 10 GPT-5.6 Sol Pro tabs across two browser
# accounts, each returning one engineering specification.
#
# The parts that look like superstition are not — each cost a lost epoch:
#
#   * the payload must be a NESTED archive declared through --include-manifest,
#     because jailgun's denylist rejects any path containing "token" and this
#     repo has vendor/feat-math/src/tokens.rs;
#   * Chrome must be headed under Xvfb — headless is hard-blocked by Cloudflare
#     and never reaches the composer;
#   * JAILGUN_TAR_WAIT_MINUTES must be set IN THE ENVIRONMENT. The TOML's
#     tar_wait_minutes feeds jailgun's deploy path, not the bridge, so a config
#     saying 60 still leaves the bridge on its 30-minute default — which is
#     shorter than Pro's reasoning time and returns zero artifacts;
#   * preflight must pass BEFORE submitting, or a misselected model wastes ~35
#     minutes across ten sessions.

bash -n "$0" || exit 2

die() { printf 'run-review-epoch: %s\n' "$*" >&2; exit 2; }

EPOCH=""
LANES=10
REPO=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epoch)   EPOCH="${2:?}"; shift 2 ;;
        --lanes)   LANES="${2:?}"; shift 2 ;;
        --repo)    REPO="${2:?}";  shift 2 ;;
        --dry-run) DRY_RUN=1;      shift ;;
        -h|--help) printf 'Usage: %s --epoch NN [--lanes N] [--dry-run]\n' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ -n "$EPOCH" ]] || die "--epoch is required"
[[ "$LANES" -ge 1 && "$LANES" -le 10 ]] || die "--lanes must be 1..10 (two accounts x 5 tabs is the hard ceiling)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$REPO" ]] || REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || die "$REPO is not a Git repository"

REVIEWS="$REPO/feedback"
EPOCH_DIR="$REVIEWS/epochs/epoch-$EPOCH"
ORCH="$REVIEWS/orchestration"
CONFIG="$ORCH/jailgun-gpt56-pro.toml"
BRIDGE="${JOPE_REVIEW_BRIDGE:-$ORCH/chrome-bridge-gpt56-pro.mjs}"
PREFLIGHT="${JOPE_REVIEW_PREFLIGHT:-$ORCH/preflight-gpt56-pro.mjs}"
SUPERVISOR="${JOPE_REVIEW_SUPERVISOR:-/home/ubuntu/jbaby/setup/jailgun_browser_supervisor.sh}"
REGISTRY="${JOPE_REVIEW_REGISTRY:-/home/ubuntu/.jailgun/browser-profiles.json}"
TAR_WAIT_MINUTES="${JOPE_REVIEW_TAR_WAIT_MINUTES:-75}"
LEASES="$REGISTRY.leases.json"
JAILGUN="${JOPE_REVIEW_JAILGUN:-/home/ubuntu/.local/bin/jailgun}"

for required in "$CONFIG" "$BRIDGE" "$PREFLIGHT" "$REGISTRY" "$JAILGUN"; do
    [[ -e "$required" ]] || die "missing required component: $required"
done
# The stock jailgun bridge cannot select a model at all; only the fork can.
grep -q 'GPT-5.6 Sol' "$BRIDGE" || die "bridge $BRIDGE cannot select GPT-5.6 Sol — wrong bridge"

mkdir -p "$EPOCH_DIR/prompts" "$EPOCH_DIR/payload"
PROMPT="$EPOCH_DIR/prompts/epoch-$EPOCH.md"
"$SCRIPT_DIR/build-review-prompt.sh" --epoch "$EPOCH" --out "$PROMPT" --repo "$REPO" --reviews "$REVIEWS"

# The payload has to live inside the scoped repo, untracked, and be removed
# whichever way this script exits.
PAYLOAD="$REPO/.jailgun-review-payload-epoch-$EPOCH.tar.gz"
MANIFEST="$REPO/.jailgun-include-manifest.txt"
cleanup() { rm -f "$PAYLOAD" "$MANIFEST"; }
trap cleanup EXIT

"$SCRIPT_DIR/pack-review-payload.sh" --out "$EPOCH_DIR/payload" --repo "$REPO"
cp "$EPOCH_DIR/payload/review-payload.tar.gz" "$PAYLOAD"
printf '%s\n' "$(basename "$PAYLOAD")" >"$MANIFEST"

if [[ "$DRY_RUN" -eq 1 ]]; then
    "$JAILGUN" tar-validate "$PAYLOAD" && printf 'run-review-epoch: tar-validate ok\n'
    printf 'run-review-epoch: DRY RUN — prompt %s, payload %s bytes, %s lane(s); nothing submitted\n' \
        "$PROMPT" "$(stat -c %s "$PAYLOAD")" "$LANES"
    exit 0
fi

# Browser must be up and logged in. `ensure` is idempotent.
if [[ -x "$SUPERVISOR" ]]; then
    printf 'run-review-epoch: ensuring browser profiles ...\n'
    "$SUPERVISOR" ensure >"$EPOCH_DIR/supervisor.log" 2>&1 || die "browser supervisor failed; see $EPOCH_DIR/supervisor.log"
fi

ACTIVE_LEASES="$(jq '[.leases[]? | select(.released_at == null)] | length' "$LEASES" 2>/dev/null || echo 0)"
[[ "$ACTIVE_LEASES" == "0" ]] || die "refusing to overlap $ACTIVE_LEASES active browser lease(s)"

mapfile -t ACCOUNTS < <(jq -r '.accounts | map(select(.status == "ready")) | sort_by(.id) | .[].id' "$REGISTRY")
[[ "${#ACCOUNTS[@]}" -ge 1 ]] || die "no ready browser accounts in $REGISTRY"

printf 'run-review-epoch: preflight (every account must show GPT-5.6 Sol / Pro) ...\n'
/usr/bin/node "$PREFLIGHT" --receipt "$EPOCH_DIR/model-preflight.json" \
    >"$EPOCH_DIR/preflight.log" 2>&1 || die "preflight failed; see $EPOCH_DIR/preflight.log"

ACCOUNT_ARGS=()
for account in "${ACCOUNTS[@]}"; do ACCOUNT_ARGS+=(--account "$account"); done

MARKER="$EPOCH_DIR/launch-marker"
touch "$MARKER"
printf 'run-review-epoch: launching %s lane(s) across %s account(s); expect ~35 minutes\n' "$LANES" "${#ACCOUNTS[@]}"

set +e
(
  cd "$REPO"
  JAILGUN_BROWSER_PROFILES="$REGISTRY" \
  JAILGUN_BRIDGE_CMD="/usr/bin/node $BRIDGE" \
  JAILGUN_TAR_WAIT_MINUTES="$TAR_WAIT_MINUTES" \
  "$JAILGUN" jailhard \
    --tabs "$LANES" \
    "${ACCOUNT_ARGS[@]}" \
    --config "$CONFIG" \
    --download-only \
    --task-file "$PROMPT" \
    --include-manifest "$MANIFEST" \
    --max-bytes 52428800 \
    .
) >"$EPOCH_DIR/stdout.log" 2>"$EPOCH_DIR/stderr.log"
STATUS="$?"
set -e
printf '%s\n' "$STATUS" >"$EPOCH_DIR/exit-status"
# jailhard exits non-zero when ANY lane fails, so a 9-of-10 epoch looks like a
# total loss. The other nine archives are on disk and worth collecting, so
# record the status and continue; the collector reports per-lane outcomes.
if [[ "$STATUS" -ne 0 ]]; then
    printf 'run-review-epoch: jailhard exited %s — some lanes failed; collecting the rest\n' "$STATUS" >&2
fi

ARTIFACTS="$(sed -n 's/^artifacts_dir[[:space:]]*=[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' "$CONFIG" | head -1)"
[[ -n "$ARTIFACTS" ]] || ARTIFACTS="$REVIEWS/epochs"

mapfile -t RECEIPTS < <(find "$ARTIFACTS/jailhard" -name receipt.json -type f -newer "$MARKER" -print 2>/dev/null)
if [[ "${#RECEIPTS[@]}" -eq 1 ]]; then
    cp "${RECEIPTS[0]}" "$EPOCH_DIR/jailhard-receipt.json"
    jq -r '.run_id' "$EPOCH_DIR/jailhard-receipt.json" >"$EPOCH_DIR/run-id"
    printf 'run-review-epoch: epoch %s launched and returned; run id %s\n' "$EPOCH" "$(cat "$EPOCH_DIR/run-id")"
elif [[ "${#RECEIPTS[@]}" -gt 1 ]]; then
    die "expected one new jailhard receipt, found ${#RECEIPTS[@]}"
else
    # No receipt is normal when the run failed before writing one; recover the
    # run id from the newest download directory so collection can still work.
    NEWEST="$(find "$ARTIFACTS/downloads" -maxdepth 1 -type d -name 'run-*' -newer "$MARKER" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
    [[ -n "$NEWEST" ]] || die "no jailhard receipt and no download directory for this run"
    basename "$NEWEST" >"$EPOCH_DIR/run-id"
    printf 'run-review-epoch: no receipt; recovered run id %s from downloads\n' "$(cat "$EPOCH_DIR/run-id")" >&2
fi
