#!/usr/bin/env bash
# deploy/ci/test-gh-role.sh — offline behavioral proof for ops/ci/gh-role's exact-head CI merge guard.
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p target/tmp
scratch=$(mktemp -d "$PWD/target/tmp/test-gh-role.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

bindir="$scratch/bin"
cfgdir="$scratch/gh-config"
audit="$scratch/audit.jsonl"
state="$scratch/state"
mkdir -p "$bindir" "$cfgdir" "$state"

HEAD_SHA="8219b40f10bc3ac175ffb3c74f25701ae5ae0123"
OTHER_SHA="1111111111111111111111111111111111111111"
printf '%s\n' "$HEAD_SHA" > "$state/live_head"

cat <<'EOF' > "$bindir/gh"
#!/usr/bin/env python3
import json
import os
import sys

state = os.environ["GH_STUB_STATE"]
args = sys.argv[1:]

if args[:2] == ["auth", "token"]:
    print("stub-token-not-real")
    sys.exit(0)

if args[:2] == ["api", "user"]:
    print(json.dumps({"login": "jepsontaylor", "id": 1001}))
    sys.exit(0)

if args[0] == "api":
    # Find endpoint path (first non-flag token after api)
    skip = False
    path = None
    for tok in args[1:]:
        if skip:
            skip = False
            continue
        if tok in ("-X", "--method", "-f", "--field", "-F", "--raw-field", "--jq", "-q"):
            skip = True
            continue
        if tok.startswith("-"):
            continue
        path = tok
        break
    if path == "repos/neverhuman/JopeDime/pulls/2318":
        with open(os.path.join(state, "live_head"), encoding="utf-8") as fh:
            head = fh.read().strip()
        print(json.dumps({"number": 2318, "user": {"login": "jepsontaylor"}, "head": {"sha": head}}))
        sys.exit(0)
    if path and path.startswith("repos/neverhuman/JopeDime/pulls/2318/commits"):
        print(json.dumps([{"sha": "8219b40f10bc", "author": {"login": "jepsontaylor"}, "committer": {"login": "jepsontaylor"}}]))
        sys.exit(0)
    if path and "/check-runs" in path:
        with open(os.path.join(state, "checks.json"), encoding="utf-8") as fh:
            sys.stdout.write(fh.read())
        sys.exit(0)
    if path == "repos/neverhuman/JopeDime/pulls/2318/merge":
        with open(os.path.join(state, "merged"), "w", encoding="utf-8") as fh:
            fh.write("api-merge\n")
        print(json.dumps({"merged": True}))
        sys.exit(0)

if args[:2] == ["pr", "merge"]:
    with open(os.path.join(state, "merged"), "w", encoding="utf-8") as fh:
        fh.write("pr-merge\n")
    sys.exit(0)

sys.stderr.write("unexpected stubbed gh invocation: %r\n" % (args,))
sys.exit(99)
EOF
chmod 0755 "$bindir/gh"

run_guard() {
  GH_STUB_STATE="$state" \
  GH_ROLE_GH_BIN="$bindir/gh" \
  GH_ROLE_CONFIG_DIR="$cfgdir" \
  GH_ROLE_AUDIT_PATH="$audit" \
  python3 ops/ci/gh-role "$@"
}

expect_refused() {
  local needle="$1"; shift
  local err="$scratch/err.txt"
  if run_guard "$@" >"$scratch/out.txt" 2>"$err"; then
    echo "expected refusal containing '$needle', got exit 0" >&2
    exit 1
  fi
  if ! grep -Fq "$needle" "$err"; then
    echo "expected '$needle' in stderr, got:" >&2
    cat "$err" >&2
    exit 1
  fi
  if [ -e "$state/merged" ]; then
    echo "refused merge still invoked underlying gh merge!" >&2
    exit 1
  fi
}

# 1. Green checks fixture setup first so only --match-head-commit is missing/mismatched in cases 1-2.
cat <<'EOF' > "$state/checks.json"
{
  "total_count": 3,
  "check_runs": [
    {"id": 10, "name": "hygiene", "status": "completed", "conclusion": "success"},
    {"id": 11, "name": "export-parity", "status": "completed", "conclusion": "skipped"},
    {"id": 12, "name": "aggregate", "status": "completed", "conclusion": "success"}
  ]
}
EOF

# Case 1: Missing --match-head-commit refuses closed.
expect_refused "requires an explicit 40-hex head guard" \
  writer pr merge 2318 --repo neverhuman/JopeDime --rebase

# Case 2: Mismatched --match-head-commit refuses closed.
expect_refused "head SHA mismatch" \
  writer pr merge 2318 --repo neverhuman/JopeDime --rebase --match-head-commit "$OTHER_SHA"

# Case 3: Pending `aggregate` (reproduces PR #2314/#2315 premature merge) refuses closed.
cat <<'EOF' > "$state/checks.json"
{
  "total_count": 2,
  "check_runs": [
    {"id": 10, "name": "changes", "status": "completed", "conclusion": "success"},
    {"id": 12, "name": "aggregate", "status": "in_progress", "conclusion": null}
  ]
}
EOF
expect_refused '`aggregate` is status=in_progress' \
  writer pr merge 2318 --repo neverhuman/JopeDime --rebase --match-head-commit "$HEAD_SHA"

# Case 4: Direct REST `gh api -X PUT repos/.../pulls/2318/merge` is also intercepted and refused.
expect_refused '`aggregate` is status=in_progress' \
  writer api -X PUT repos/neverhuman/JopeDime/pulls/2318/merge -f "sha=$HEAD_SHA"

# Case 5: Failed sibling lane refuses closed even if aggregate were marked success.
cat <<'EOF' > "$state/checks.json"
{
  "total_count": 2,
  "check_runs": [
    {"id": 10, "name": "hygiene", "status": "completed", "conclusion": "failure"},
    {"id": 12, "name": "aggregate", "status": "completed", "conclusion": "success"}
  ]
}
EOF
expect_refused "hygiene(completed/failure)" \
  writer pr merge 2318 --repo neverhuman/JopeDime --rebase --match-head-commit "$HEAD_SHA"

# Case 6: Re-run where attempt 1 failed (id=10) and attempt 2 succeeded (id=25) + aggregate==success -> allowed!
cat <<'EOF' > "$state/checks.json"
{
  "total_count": 4,
  "check_runs": [
    {"id": 10, "name": "hygiene", "status": "completed", "conclusion": "failure"},
    {"id": 25, "name": "hygiene", "status": "completed", "conclusion": "success"},
    {"id": 26, "name": "export-parity", "status": "completed", "conclusion": "skipped"},
    {"id": 27, "name": "aggregate", "status": "completed", "conclusion": "success"}
  ]
}
EOF
run_guard writer pr merge 2318 --repo neverhuman/JopeDime --rebase --match-head-commit "$HEAD_SHA" >/dev/null 2>"$scratch/allow.err"
test "$(cat "$state/merged")" = "pr-merge"
grep -Fq "\"head\": \"$HEAD_SHA\"" "$audit"

echo "gh-role merge guard tests: 6 passed"
