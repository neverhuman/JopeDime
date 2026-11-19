#!/bin/bash
# H472 waiter (xbabe2, seeds 1 2 3, MAXALIVE=2). Pattern = H471 waiter.sh (paths renamed only). Usage: SEEDS="1 2" MEMGATE=30 bash waiter.sh
# Tracks the real pid in results/seedN/pid (rewritten on every (re)launch). Per seed in $SEEDS:
#  - never launched -> launch when MemAvailable >= $MEMGATE GB AND the CPU gate passes (busy% + 16 threads <= 85%) on 2 consecutive checks;
#  - dead without summary.json (ANY cause: own 15 GB MemAvailable guard abort, silent death) -> relaunch, resume-safe via
#    results/fold_sig, same gate on 2 consecutive checks, max 5 relaunches per seed; at most 2 seeds alive at once;
#  - summary.json written but its own h472 process still alive 10 min later -> SIGTERM that pid only
#    (only if its cmdline is h472_trainfold_cv.py --seeds N).
# One launch per 15 min. Exits when all its seeds have summary.json and no own seed process is alive, or after 48 h.
# Requires PROBE_OK + intact DOPE in probe.log (xbabe2) or probe_ok.txt (copied to xbabe1) and the pinned code sha.
# Never touches any other process. Never xbabe0. Never xbabe3.
cd ~/jope-data/hwave-h472 || exit 1
SEEDS=${SEEDS:-"1 2 3"}; MEMGATE=${MEMGATE:-30}
{ grep -q "PROBE_OK" probe.log 2>/dev/null || grep -q "PROBE_OK" probe_ok.txt 2>/dev/null; } || { echo "$(date -u +%FT%TZ) waiter: probe not OK" >> waiter.log; exit 1; }
sha256sum -c h472_trainfold_cv.py.sha256 >/dev/null || { echo "$(date -u +%FT%TZ) waiter: code sha mismatch" >> waiter.log; exit 1; }
. ./cpugate.sh
ok=0; end=$(( $(date +%s) + 48*3600 )); declare -A n
echo "$(date -u +%FT%TZ) waiter start on $(hostname) pid $$ seeds [$SEEDS] memgate ${MEMGATE}G" >> waiter.log
own() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null && tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -q "h472_trainfold_cv.py --seeds $2 "; }
while [ $(date +%s) -lt $end ]; do
  m=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo); c=$(cpu_busy_pct)
  if cpu_ok "$c" && [ "$m" -ge $MEMGATE ]; then ok=$((ok+1)); else ok=0; fi
  act=""; done_n=0; alive=0; ns=0
  for s in $SEEDS; do
    ns=$((ns+1)); p=$(cat results/seed$s/pid 2>/dev/null)
    if [ -f results/seed$s/summary.json ]; then
      done_n=$((done_n+1))
      if own "$p" $s; then
        alive=$((alive+1)); age=$(( $(date +%s) - $(stat -c %Y results/seed$s/summary.json) ))
        if [ $age -gt 600 ]; then kill -TERM "$p"; echo "$(date -u +%FT%TZ) waiter: seed $s pid $p alive ${age}s after summary.json - SIGTERM (own teardown hang)" >> waiter.log; fi
      fi
      continue
    fi
    if own "$p" $s; then alive=$((alive+1)); continue; fi
    if [ -z "$p" ]; then [ -z "$act" ] && [ $ok -ge 2 ] && act="launch $s"; continue; fi
    if [ "${n[$s]:-0}" -lt 5 ]; then [ -z "$act" ] && [ $ok -ge 2 ] && act="relaunch $s"
    elif [ "${n[$s]:-0}" -eq 5 ]; then echo "$(date -u +%FT%TZ) waiter: seed $s dead again after 5 relaunches - giving up" >> waiter.log; n[$s]=6; fi
  done
  [ $done_n -eq $ns ] && [ $alive -eq 0 ] && { echo "$(date -u +%FT%TZ) waiter: all seeds [$SEEDS] done (summary.json, no h472 seed process alive)" >> waiter.log; exit 0; }
  [ -n "$act" ] && [ $alive -ge ${MAXALIVE:-2} ] && act=""
  if [ -n "$act" ]; then
    s=${act#* }; [ "${act% *}" = relaunch ] && { n[$s]=$(( ${n[$s]:-0} + 1 )); tail -n 2 results/seed$s/run.log | sed "s/^/  last: /" >> waiter.log; }
    mkdir -p results/seed$s; echo "--- $(date -u +%FT%TZ) waiter $act on $(hostname) mem=${m}G cpu_busy=${c}% ---" >> results/seed$s/run.log
    setsid nohup nice -n 19 python3 h472_trainfold_cv.py --seeds $s --threads 16 --out results/seed$s >> results/seed$s/run.log 2>&1 < /dev/null &
    echo $! > results/seed$s/pid; echo "$(date -u +%FT%TZ) waiter $act on $(hostname) pid $! mem=${m}G cpu_busy=${c}% (relaunches seed $s: ${n[$s]:-0})" >> waiter.log; ok=0; sleep 900; continue
  fi
  sleep 300
done
echo "$(date -u +%FT%TZ) waiter: timed out" >> waiter.log
