#!/bin/bash
# H438 gate for addendum 2 (checkpoint displacement). Run BEFORE the read. Create-once output.
#   gates3_h438.sh ADDENDUM2_COMMIT      -> $EV/results/H438.gates3.json
set -euo pipefail
AC=${1:?commit that published addendum 2}
EV=/home/ubuntu/jope_plan/evidence/H438; MON=/home/ubuntu/jope-data/apex-icl/diag/mon
STAGE=$(cat "$EV/logs/pub_stage3"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h438_icl_104m_lag_diagnosis
OUT=$EV/results/H438.gates3.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$AC" origin/v1/next)
add=$(ok cmp -s "$STAGE/code/PREREG_H438_ADDENDUM_2.md" <(git -C "$REPO" show "$AC:$WDIR/PREREG_ADDENDUM_2.md"))
code=$(ok cmp -s "$STAGE/code/disp_h438.py" <(git -C "$REPO" show "$AC:$WDIR/disp_h438.py"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/H438.displacement.json" ] && echo true || echo false)
n=0; for f in M_r3_h012 M_r3_h018 M_r3_h024 L_w2_h012 L_w2_h018 L_w2_h024; do [ -f "$MON/$f.pt" ] && [ -f "$MON/$f.pt.sha256" ] && n=$((n + 1)); done
jq -n --argjson anc "$anc" --argjson add "$add" --argjson code "$code" --argjson stage "$stage" --argjson noread "$noread" --argjson n "$n" \
  --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  {schema: "jope.h438.gates3/v1", at: $at, checks: {checkpoint_copies_with_sha: $n, stage_mode: $mode},
   gates: {g1_addendum2_and_code_published: ($anc and $add and $code), g2_six_checkpoint_copies: ($n == 6), g3_no_displacement_result: $noread,
           g4_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass}' "$OUT"
jq -e '.pass' "$OUT" > /dev/null
