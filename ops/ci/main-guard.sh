#!/usr/bin/env bash
set -euo pipefail
: "${GH_TOKEN:?GitHub token required}"
: "${REPO:?repository required}"
: "${AFTER:?pushed head required}"
: "${PUSHER:?pusher required}"
if [ -z "${BEFORE:-}" ] || [ "$BEFORE" = "0000000000000000000000000000000000000000" ]; then
  commits=$AFTER
else
  # An API failure must fail closed rather than narrow the guard to the head commit.
  commits=$(gh api "repos/$REPO/compare/$BEFORE...$AFTER" --jq '.commits[].sha') \
    || { echo "main-guard: could not list the pushed commits; refusing to pass" >&2; exit 1; }
fi
bad=""
for sha in $commits; do
  merged=$(gh api "repos/$REPO/commits/$sha/pulls" --jq '[.[] | select(.merged_at != null)] | length') \
    || { echo "main-guard: could not look up pull requests for $sha; refusing to pass" >&2; exit 1; }
  case $merged in
    ''|*[!0-9]*) echo "main-guard: unexpected pull-request count '$merged' for $sha" >&2; exit 1 ;;
  esac
  if [ "$merged" -eq 0 ]; then bad="$bad $sha"; fi
done
if [ -n "$bad" ]; then
  body=$(printf 'Pushed by %s. Commits on main without a merged pull request:%s\n\nLand work through a pull request (ci.yml green) instead of pushing to main.' "$PUSHER" "$bad")
  gh issue create -R "$REPO" --title "main-guard: commit(s) pushed to main without a merged pull request" --body "$body"
  echo "$body"
  exit 1
fi
echo "ok: every pushed commit belongs to a merged pull request"
