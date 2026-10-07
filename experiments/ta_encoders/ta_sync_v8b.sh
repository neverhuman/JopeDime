#!/usr/bin/env bash
# Every 180 s (runs on xbabe2, screen ta-sync-v8b): mirror the small status files of the two v8b runs into the probe
# dashboard dir (v8braw on xbabe1 disk; v8bft on xbabe3 /dev/shm), and mirror v8bft's new checkpoints/embeddings onto
# xbabe1 disk (xbabe3-mirror/v8bft). Create/overwrite only; nothing is deleted anywhere.
RUN=ta-20261007T2030Z; X1=/home/ubuntu/ta_encoders/$RUN; X3=/dev/shm/ta_encoders/$RUN; D=/home/ubuntu/jopedime_probe_exp/dashboard/ta
mkdir -p $D
while true; do
  for f in timeseries.jsonl progress.json run.json; do
    ssh -n -o ConnectTimeout=10 xbabe1 "cat $X1/v8braw/$f 2>/dev/null" > $D/.v8braw_$f.tmp && [ -s $D/.v8braw_$f.tmp ] && mv $D/.v8braw_$f.tmp $D/v8braw_$f
    ssh -n -o ConnectTimeout=10 xbabe3 "cat $X3/v8bft/$f 2>/dev/null" > $D/.v8bft_$f.tmp && [ -s $D/.v8bft_$f.tmp ] && mv $D/.v8bft_$f.tmp $D/v8bft_$f
  done
  have=$(ssh -n xbabe1 "mkdir -p $X1/xbabe3-mirror/v8bft && cd $X1/xbabe3-mirror && find v8bft -type f \( -name '*.pt' -o -name '*.npy' -o -name 'meta_*.json' \) | sort")
  want=$(ssh -n xbabe3 "cd $X3 && [ -d v8bft ] && find v8bft -type f \( -name '*.pt' -o -name '*.npy' -o -name 'meta_*.json' \) | sort")
  new=$(comm -13 <(echo "$have") <(echo "$want") | tr '\n' ' ')
  [ -n "$want" ] && ssh -n xbabe3 "cd $X3 && tar c v8bft/timeseries.jsonl v8bft/progress.json v8bft/run.json v8bft/probe.log v8bft.log $new 2>/dev/null" | ssh xbabe1 "cd $X1/xbabe3-mirror && tar x 2>/dev/null"
  echo "$(date -u +%FT%TZ) v8b synced new_ft=$(echo $new | wc -w)" >> $D/sync_v8b.log
  sleep 180
done
