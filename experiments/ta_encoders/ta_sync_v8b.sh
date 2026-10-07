#!/usr/bin/env bash
# Every 180 s (runs on xbabe2, screen ta-sync-v8b): mirror the small status files of the two v8b runs (xbabe1) into the
# probe dashboard dir. Create/overwrite only; nothing is deleted anywhere.
RUN=ta-20261007T2030Z; X1=/home/ubuntu/ta_encoders/$RUN; D=/home/ubuntu/jopedime_probe_exp/dashboard/ta
mkdir -p $D
while true; do
  for k in v8braw v8bft; do
    for f in timeseries.jsonl progress.json run.json; do
      ssh -n -o ConnectTimeout=10 xbabe1 "cat $X1/$k/$f 2>/dev/null" > $D/.${k}_$f.tmp && [ -s $D/.${k}_$f.tmp ] && mv $D/.${k}_$f.tmp $D/${k}_$f
    done
  done
  echo "$(date -u +%FT%TZ) v8b synced" >> $D/sync_v8b.log
  sleep 180
done
