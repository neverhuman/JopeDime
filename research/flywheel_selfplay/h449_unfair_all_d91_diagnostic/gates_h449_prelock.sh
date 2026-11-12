#!/bin/bash
# Hard pre-lock gate for H449 (`gates_h449_prelock.sh`).
# Refuses (exit 3) unless H448's official FAIR dev91 score and S3 archive are published and GET-verified on origin/v1/next,
# and verifies zero APEX-CONFIRM / DL-Holdout exposure.
set -euo pipefail
REPO=/home/ubuntu/JopeDime
BRD=research/elo/reference-board/boards/dev91-audit-v1
H448_CLOSE=/home/ubuntu/jope-data/h448-closeout/archive_manifest.json
OUT=/home/ubuntu/jope_plan/evidence/H449/logs/prelock_gate.json
GH=/home/ubuntu/.local/bin/gh-role
CRED=(-c credential.helper= -c "credential.helper=!$GH writer auth git-credential")
URL=https://github.com/neverhuman/JopeDime.git

mkdir -p "$(dirname "$OUT")"
git -C "$REPO" "${CRED[@]}" fetch -q "$URL" +refs/heads/v1/next:refs/remotes/origin/v1/next
V=$(git -C "$REPO" rev-parse origin/v1/next)

has_h448_score=false
if git -C "$REPO" cat-file -e "$V:$BRD/scores/h448.APEX_FAIR_CHAMPION.jsonl" 2>/dev/null; then
  nrows=$(git -C "$REPO" show "$V:$BRD/scores/h448.APEX_FAIR_CHAMPION.jsonl" | wc -l)
  [ "$nrows" = 91 ] && has_h448_score=true
fi

has_h448_arch_git=false
if git -C "$REPO" show "$V:research/flywheel_selfplay/h448_fair_champion_d91/ARCHIVE.json" 2>/dev/null | grep -q 'h448_closeout_bundle.tar.zst' \
   && git -C "$REPO" show "$V:$BRD/ARCHIVE.json" 2>/dev/null | grep -q 'Pre-H448 snapshot'; then
  has_h448_arch_git=true
fi

has_h448_s3_get=false
if [ -s "$H448_CLOSE" ] && [ "$(jq -r '.verified // ""' "$H448_CLOSE")" = "EXACT_VERSION_GET_AND_MEMBERS_VERIFIED" ]; then
  has_h448_s3_get=true
fi

apex_661=false
[ "$(find /home/ubuntu/jope-data/h335-cl/apex/tables -maxdepth 1 -type f | wc -l)" = 661 ] && apex_661=true

x0_views_ok=false
if ssh -i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes xbabe0 \
  '[ "$(find /home/ubuntu/h335-stage-v/dlval-view/tables -maxdepth 1 -type f | wc -l)" = 1310 ] && [ "$(find /home/ubuntu/jope-data/h335-cl/apex/tables -maxdepth 1 -type f | wc -l)" = 221 ]'; then
  x0_views_ok=true
fi

jq -n \
  --arg v "$V" --arg at "$(date -u +%FT%TZ)" \
  --argjson s "$has_h448_score" --argjson ag "$has_h448_arch_git" --argjson s3 "$has_h448_s3_get" \
  --argjson ap "$apex_661" --argjson x0 "$x0_views_ok" '
  {schema: "jope.h449.prelock_gate/v1", at: $at, v1_next: $v,
   gates: {
     g1_h448_fair_score_on_v1_next: $s,
     g2_h448_archive_in_git: $ag,
     g3_h448_s3_exact_get_verified: $s3,
     g4_xbabe2_apex_no_confirm: $ap,
     g5_xbabe0_views_no_confirm_no_holdout: $x0
   }} | .pass = ([.gates[]] | all)' > "$OUT"

cat "$OUT"
[ "$(jq -r .pass "$OUT")" = true ] || { echo "REFUSING H449: prelock gate failed (H448 fair score & S3 archive must land first)" >&2; exit 3; }
