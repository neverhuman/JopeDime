#!/usr/bin/env bash
# Every 180 s (runs on xbabe2, screen ta-sync-v8b): mirror the small status files of the two v8b runs into the probe
# dashboard dir (both on xbabe1 disk: v8braw and v8bft2 — the final v8b-ft run). Create/overwrite only; nothing is deleted.
RUN=ta-20261007T2030Z; X1=/home/ubuntu/ta_encoders/$RUN; X3=/dev/shm/ta_encoders/$RUN; D=/home/ubuntu/jopedime_probe_exp/dashboard/ta
mkdir -p $D
while true; do
  for f in timeseries.jsonl progress.json run.json; do
    ssh -n -o ConnectTimeout=10 xbabe1 "cat $X1/v8braw/$f 2>/dev/null" > $D/.v8braw_$f.tmp && [ -s $D/.v8braw_$f.tmp ] && mv $D/.v8braw_$f.tmp $D/v8braw_$f
    ssh -n -o ConnectTimeout=10 xbabe1 "cat $X1/v8bft2/$f 2>/dev/null" > $D/.v8bft_$f.tmp && [ -s $D/.v8bft_$f.tmp ] && mv $D/.v8bft_$f.tmp $D/v8bft_$f
  done
  echo "$(date -u +%FT%TZ) v8b synced" >> $D/sync_v8b.log
  sleep 180
done
