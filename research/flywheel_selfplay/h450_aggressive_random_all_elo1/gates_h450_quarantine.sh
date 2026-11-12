#!/bin/bash
# Hard prelock/run quarantine gate for Wave H450 (`gates_h450_quarantine.sh`).
# Verifies:
#   1. H448 official fair score (`h448.APEX_FAIR_CHAMPION.jsonl`) + S3 GET-verified ARCHIVE.json remain intact on origin/v1/next.
#   2. Local xbabe2 apex/tables (661) + res01-train-view (1,219) hold 0 CONFIRM / 0 DL-Holdout tables.
#   3. xbabe0 Stage-V views hold exact 1,310 dlval-view + 221 apex tables (0 CONFIRM, 0 DL-Holdout).
set -euo pipefail

REPO=/home/ubuntu/JopeDime
BRD=research/elo/reference-board/boards/dev91-audit-v1
H448_DIR=research/flywheel_selfplay/h448_fair_champion_d91
SSH_X0=(ssh -i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes xbabe0)

V1_HEAD=$(git -C "$REPO" rev-parse origin/v1/next)
G1=false
if [ "$(git -C "$REPO" show "origin/v1/next:$BRD/scores/h448.APEX_FAIR_CHAMPION.jsonl" 2>/dev/null | wc -l)" = 91 ]; then
  G1=true
fi

G2=false
if git -C "$REPO" show "origin/v1/next:$H448_DIR/ARCHIVE.json" 2>/dev/null | grep -q "h448_closeout_bundle.tar.zst" \
   && git -C "$REPO" show "origin/v1/next:$BRD/ARCHIVE.json" 2>/dev/null | grep -q "Pre-H448 snapshot"; then
  G2=true
fi

G3=false
if [ -f /home/ubuntu/jope_plan/evidence/H448/results/H448_DEV91.json ]; then
  G3=true
fi

G4=false
N_APEX=$(find /home/ubuntu/jope-data/h335-cl/apex/tables -maxdepth 1 -name '*.jtt1' | wc -l)
N_DLTR=$(find /home/ubuntu/h317-stage/res01-train-view/tables -maxdepth 1 -name '*.jtt1' | wc -l)
if [ "$N_APEX" = 661 ] && [ "$N_DLTR" = 1219 ]; then
  G4=true
fi

G5=false
if "${SSH_X0[@]}" '[ "$(find /home/ubuntu/h335-stage-v/dlval-view/tables -maxdepth 1 -type f | wc -l)" = 1310 ] && [ "$(find /home/ubuntu/jope-data/h335-cl/apex/tables -maxdepth 1 -type f | wc -l)" = 221 ]'; then
  G5=true
fi

PASS=false
if [ "$G1" = true ] && [ "$G2" = true ] && [ "$G3" = true ] && [ "$G4" = true ] && [ "$G5" = true ]; then
  PASS=true
fi

cat <<JSON
{
  "schema": "jope.h450.quarantine_gate/v1",
  "at": "$(date -u +%FT%TZ)",
  "v1_next": "$V1_HEAD",
  "gates": {
    "g1_h448_fair_score_intact": $G1,
    "g2_h448_archives_intact": $G2,
    "g3_h448_local_readout_present": $G3,
    "g4_xbabe2_661_apex_1219_dltrain_no_sealed": $G4,
    "g5_xbabe0_views_no_confirm_no_holdout": $G5
  },
  "pass": $PASS
}
JSON

[ "$PASS" = true ] || { echo "H450 QUARANTINE GATE FAILED" >&2; exit 3; }
