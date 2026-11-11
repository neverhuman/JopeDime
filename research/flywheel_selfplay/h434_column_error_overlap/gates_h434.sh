#!/bin/bash
# H434 integrity gates (PREREG section 6). The whole wave is one read of stored values, so the gates run BEFORE it:
# the driver runs this script and starts the read only if every gate passes. Create-once output.
#   gates_h434.sh PREREG_COMMIT      -> $EV/results/H434.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H434
STAGE=$(cat "$EV/logs/pub_stage"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h434_column_error_overlap
BANK=/home/ubuntu/jope-data/h335-cl/h424/pub/bank_DEV.npz
OUT=$EV/results/H434.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H434.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/H434_DEV.overlap.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
jq -n --slurpfile bj "$BANK.json" --argjson ptime "$ptime" --argjson now "$(date +%s)" --argjson anc "$anc" --argjson prereg "$prereg" \
  --argjson stage "$stage" --argjson noread "$noread" --argjson view "$view" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" \
  --arg a20 "$(sha /home/ubuntu/jope_plan/evidence/H424/results/H424_DEV.apply.json)" \
  --arg scr "$(sha /home/ubuntu/jope_plan/evidence/H425/results/SCREEN_DEV.json)" \
  --arg need "$(sha /home/ubuntu/jope_plan/evidence/H432/results/H432_DEV.need_per_table.jsonl)" \
  --argjson cols "$(ls /home/ubuntu/jope-data/h424/pub/cols_DEV/*.npz | wc -l)" '
  {schema: "jope.h434.gates/v1", at: $at,
   checks: {prereg_commit_time: $ptime, gate_time: $now, stage_mode: $mode, column_files: $cols,
            inputs: {apply: $a20, screen: $scr, bank: $bj[0].sha256, h432_need_per_table: $need}},
   gates: {
     g1_prereg_before_read: ($anc and $prereg and $now > $ptime and $noread),
     g2_pinned_inputs: ($a20 == "ce0eb88749ebfb5c92aa6c10cd621563b1ab4a041756cc6e882b282be824faca"
         and $scr == "fa8edd8a08a9aef30af98b8240f3e3b1f40d11e9681b157d7190cb00ff58891e"
         and $bj[0].sha256 == "dccf9beb3312f42e5a28c32aba33389c13f9c582add85084b0fc7d236f613c9b"
         and $need == "69081a5bee17b4ba59a6a7a810370a8534621ad1551f97adfc6e9675ec811e01" and $cols == 115),
     g3_confirm_untouched: $view,
     g4_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass}' "$OUT"
jq -e '.pass' "$OUT" > /dev/null
