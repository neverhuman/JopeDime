#!/bin/bash
# xbabe4 (Apple GPU) supervisor for one bounded H419 arm. Written after the 2026-11-10 20:25Z memory exhaustion:
# a trainer's memory on the MPS backend grows with the number of distinct batch shapes it has seen (buffer pool up
# to its watermark, plus about 10 GiB an hour outside the pool), so the process is restarted from its own
# checkpoint before it gets large. Training arguments and code stage are the arm's own; only --resume is added.
#   x4_supervise_arm.sh NAME SOFT_GIB HARD_GIB STAGE_DIR -- <icl_train.py arguments, without --resume>
#     physical footprint >= SOFT and a checkpoint written 15-60 s ago -> restart (loses seconds)
#     physical footprint >= HARD                                      -> restart at once (loses up to one checkpoint interval)
#     system free memory below FREE_MIN percent (default 25)       -> stop this arm and the supervisor
#     4 trainer failures within an hour                            -> supervisor stops
# Ends when the run's DONE.json exists (the arm's own --stop-h budget). macOS bash 3.2: no arrays tricks.
NAME=${1:?name}; SOFT=${2:?soft}; HARD=${3:?hard}; STAGE=${4:?stage}; shift 5
X=$HOME/jope-data/apex-icl/x4; FREE_MIN=${FREE_MIN:-25}
OUT=""; prev=""; for a in "$@"; do [ "$prev" = "--out" ] && OUT=$a; prev=$a; done
[ -f "$OUT/ckpt_last.pt" ] || { echo "no checkpoint in $OUT" >&2; exit 2; }
LOG=$X/sup_$NAME.log
say() { echo "$(date -u +%FT%TZ) $NAME $*" >> "$LOG"; }
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_HIGH:-0.14} PYTORCH_MPS_LOW_WATERMARK_RATIO=${MPS_LOW:-0.10}
fails=0; win=$(date +%s); k=0
say "supervisor start soft=$SOFT hard=$HARD free_min=$FREE_MIN mps_high=$PYTORCH_MPS_HIGH_WATERMARK_RATIO mps_low=$PYTORCH_MPS_LOW_WATERMARK_RATIO stage=$STAGE"
while [ ! -e "$OUT/DONE.json" ]; do
  cd "$STAGE/code" || exit 2
  "$HOME/apex-venv/bin/python" -B icl_train.py "$@" --resume >> "$X/sup_$NAME.out" 2>&1 &
  p=$!; echo $p > "$X/sup_$NAME.pid"; t0=$(date +%s); why=exit; halt=0
  say "trainer started pid $p"
  while kill -0 $p 2>/dev/null; do
    sleep 20
    g=$(/usr/bin/footprint -p $p 2>/dev/null | awk '/phys_footprint:/ {v=$2; u=$3; if (u=="KB") v/=1048576; else if (u=="MB") v/=1024; else if (u=="TB") v*=1024; printf "%d", v; exit}')
    [ -n "$g" ] || { kill -0 $p 2>/dev/null || break; continue; }
    k=$(( k + 1 )); [ $(( k % 30 )) = 1 ] && say "footprint ${g} GiB"
    f=$(memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%","",$NF); print $NF}')
    if [ "${f:-100}" -lt "$FREE_MIN" ]; then why="system free memory ${f}% < ${FREE_MIN}% (footprint ${g} GiB)"; halt=1; kill -TERM $p; break; fi
    if [ "$g" -ge "$HARD" ]; then why="hard limit: footprint ${g} GiB"; kill -TERM $p; break; fi
    if [ "$g" -ge "$SOFT" ]; then
      m=$(stat -f %m "$OUT/ckpt_last.pt" 2>/dev/null || echo 0); now=$(date +%s); age=$(( now - m ))
      if [ "$m" -gt "$t0" ] && [ "$age" -ge 15 ] && [ "$age" -le 60 ]; then
        why="soft limit: footprint ${g} GiB, checkpoint ${age} s old"; kill -TERM $p; break
      fi
    fi
  done
  wait $p 2>/dev/null; rc=$?
  say "trainer ended: $why (exit $rc) after $(( $(date +%s) - t0 )) s"
  [ -e "$OUT/DONE.json" ] && break
  [ "$halt" = 1 ] && { say "supervisor stops: system memory low"; exit 6; }
  if [ "$why" = exit ]; then
    now=$(date +%s); [ $(( now - win )) -gt 3600 ] && { fails=0; win=$now; }
    fails=$(( fails + 1 ))
    [ "$fails" -ge 4 ] && { say "supervisor stops: 4 trainer failures within an hour"; exit 5; }
  fi
  sleep 10
done
say "run complete (DONE.json)"
