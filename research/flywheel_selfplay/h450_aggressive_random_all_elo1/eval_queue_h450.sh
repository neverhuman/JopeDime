#!/bin/bash
# H450 pub-r11 (cl_fwriter): dev91 evaluation queue for the 104M runs r4_L_A (xbabe2) and r4_L_B (xbabe3).
#
# Each pass:
#   - waits until no h450-eval-* unit is running;
#   - scores the newest epoch that both runs have written and that has not been scored yet (eval_gpu_h450.sh epoch E);
#   - appends one line to TRAJECTORY.jsonl with the Elo arms and each run's exposure ledger from epoch_E.json;
#   - posts that line to AGENT_CHAT and the coord file.
# Epochs in between are skipped: one eval takes longer than one epoch, so N = the epochs that come due while an eval
# runs.
# Plateau: when neither the champion stack nor the ICL-only arm has beaten its best for 4 scored epochs in a row, the
# queue posts PLATEAU once per streak and keeps scoring (pub-r13; pub-r10..r12 posted EXHAUSTED and stopped).
#
#   eval_queue_h450.sh            (run as the user unit h450-eval-queue)
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
ST=$(dirname "$(dirname "$(readlink -f "$0")")")
EVAL=$ST/code/eval_gpu_h450.sh
R4=/home/ubuntu/jope-data/h450-quarantine/r4
EV=/home/ubuntu/jope_plan/evidence/H450
TRAJ=$EV/diag_results/r4_TRAJECTORY.jsonl
X3=xbabe3
X3_RUNS=/home/ubuntu/jope-data/h450-r4-x3/runs
RA=r4_L_A
RB=r4_L_B
PATIENCE=4
log() { echo "$(date -u +%FT%TZ) $*"; }
post() {
  local m="[$(date -u +%FT%H:%MZ)] cl_fwriter -> agy_apex_w10, Jepson (cc all): $1"
  (cd /home/ubuntu/JopeDime && flock AGENT_CHAT.md.lock bash -c 'printf "%s\n" "$1" >> AGENT_CHAT.md' _ "$m")
  flock /home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md.lock \
    bash -c 'printf "%s\n" "$1" >> /home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md' _ "$m"
}
max_epoch() {  # newest epoch both runs have written
  local a b
  a=$(find "$R4/runs/$RA" -maxdepth 1 -name 'ckpt_epoch_[0-9][0-9].pt' | sed 's/.*_\([0-9][0-9]\)\.pt/\1/' | sort -n | tail -1)
  b=$(ssh -o BatchMode=yes "$X3" "find $X3_RUNS/$RB -maxdepth 1 -name 'ckpt_epoch_[0-9][0-9].pt'" | sed 's/.*_\([0-9][0-9]\)\.pt/\1/' | sort -n | tail -1)
  [ -n "$a" ] && [ -n "$b" ] || { echo 0; return; }
  echo $(( 10#$a < 10#$b ? 10#$a : 10#$b ))
}
upd() {  # LINE -> update best_c, best_i, stale
  local c i imp
  c=$(jq -r .champion_stack <<<"$1"); i=$(jq -r .icl_only <<<"$1")
  imp=$(jq -n --argjson c "$c" --argjson i "$i" --argjson bc "$best_c" --argjson bi "$best_i" '($c > $bc) or ($i > $bi)')
  if [ "$imp" = true ]; then
    stale=0
    best_c=$(jq -n --argjson c "$c" --argjson bc "$best_c" '[$c, $bc] | max')
    best_i=$(jq -n --argjson i "$i" --argjson bi "$best_i" '[$i, $bi] | max')
  else
    stale=$((stale + 1))
  fi
}
# pub-r11: the streak starts from the epoch-0 baseline and replays TRAJECTORY.jsonl, so a queue restart keeps it.
best_c=-1e9; best_i=-1e9; stale=0
until [ -s "$EV/diag_results/r4_g00_init/SUMMARY.json" ]; do sleep 120; done
upd "$(cat "$EV/diag_results/r4_g00_init/SUMMARY.json")"; stale=0
if [ -s "$TRAJ" ]; then while read -r l; do upd "$l"; done < "$TRAJ"; fi
log "start: best champion $best_c, best ICL-only $best_i, no-gain streak $stale"
while true; do
  while systemctl --user list-units --no-legend 'h450-eval-g*' | grep -q 'running'; do sleep 60; done
  E=$(max_epoch)
  e=$(printf '%02d' "$E")
  if [ "$E" -lt 1 ] || [ -e "$R4/cols/g${e}_${RA}_${RB}" ]; then sleep 120; continue; fi
  [ -s "$EV/diag_results/r4_g00_init/SUMMARY.json" ] || { sleep 120; continue; }   # the analysed baseline comes first
  log "scoring epoch $e"
  if ! "$EVAL" epoch "$E" "$RA" "$RB" > "$EV/logs/eval_g${e}.log" 2>&1; then
    post "H450 dev91 eval of epoch $e FAILED (see $EV/logs/eval_g${e}.log); queue paused for cl_fwriter."
    exit 1
  fi
  S=$EV/diag_results/r4_g${e}_${RA}_${RB}/SUMMARY.json
  line=$(jq -c --arg e "$e" --slurpfile a "$R4/runs/$RA/epoch_$e.json" --slurpfile b "$R4/runs/$RB/epoch_$e.json" '
    . + {epoch: ($e | tonumber),
         run_a: {run: "r4_L_A", unique_datasets: $a[0].exposure.unique_datasets_seen, dup_real: $a[0].exposure.duplicate_real_datasets,
                 dup_synth: $a[0].exposure.duplicate_synthetic_datasets, src_tables: $a[0].exposure.distinct_source_tables_seen},
         run_b: {run: "r4_L_B", unique_datasets: $b[0].exposure.unique_datasets_seen, dup_real: $b[0].exposure.duplicate_real_datasets,
                 dup_synth: $b[0].exposure.duplicate_synthetic_datasets, src_tables: $b[0].exposure.distinct_source_tables_seen},
         label: "in-corpus (dev91 in training), canonical_promotion_forbidden"}' "$S")
  echo "$line" >> "$TRAJ"
  upd "$line"
  post "H450 dev91 in-corpus Elo, epoch $e (Kumo L 2070.2): $line; no-gain streak $stale/$PATIENCE"
  # pub-r13: at the plateau the queue keeps scoring. ENG:watcher (AGENT_CHAT, 2026-11-13 05:32 MST) relayed Jepson's
  # standing directive: keep r4_L_A / r4_L_B training past queue exhaustion and propose the next recipe change instead.
  if [ "$stale" -eq "$PATIENCE" ]; then
    post "H450 PLATEAU: no dev91 gain for $PATIENCE scored epochs (best champion $best_c, best ICL-only $best_i). Trainers keep running per the relayed directive; scoring continues; cl_fwriter proposes the next recipe change for Jepson."
  fi
done
