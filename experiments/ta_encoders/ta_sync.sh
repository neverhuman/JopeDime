#!/usr/bin/env bash
# Every 180 s (runs on xbabe2, tmux ta-sync): mirror the small TA status files into the probe dashboard dir, and mirror
# xbabe3's /dev/shm TA_Dope_V2 run (checkpoints/embeddings, new files only) onto xbabe1 disk (xbabe3 root disk is full).
# Create/overwrite only; nothing is deleted anywhere.
RUN=ta-20261007T2030Z; X1=/home/ubuntu/ta_encoders/$RUN; X3=/dev/shm/ta_encoders/$RUN; D=/home/ubuntu/jopedime_probe_exp/dashboard/ta
mkdir -p $D
while true; do
  for f in timeseries.jsonl progress.json run.json; do
    ssh -n -o ConnectTimeout=10 xbabe1 "cat $X1/hv8r2/$f 2>/dev/null" > $D/.hv8_$f.tmp && [ -s $D/.hv8_$f.tmp ] && mv $D/.hv8_$f.tmp $D/hv8_$f
    ssh -n -o ConnectTimeout=10 xbabe3 "cat $X3/dv2r2/$f 2>/dev/null" > $D/.dv2_$f.tmp && [ -s $D/.dv2_$f.tmp ] && mv $D/.dv2_$f.tmp $D/dv2_$f
  done
  have=$(ssh -n xbabe1 "mkdir -p $X1/xbabe3-mirror/dv2r2 && cd $X1/xbabe3-mirror && find dv2r2 -type f \( -name '*.pt' -o -name '*.npy' -o -name 'meta_*.json' \) | sort")
  want=$(ssh -n xbabe3 "cd $X3 && find dv2r2 -type f \( -name '*.pt' -o -name '*.npy' -o -name 'meta_*.json' \) | sort")
  new=$(comm -13 <(echo "$have") <(echo "$want") | tr '\n' ' ')
  ssh -n xbabe3 "cd $X3 && tar c dv2r2/timeseries.jsonl dv2r2/progress.json dv2r2/run.json dv2r2/probe.log dv2r2.log $new 2>/dev/null" | ssh xbabe1 "cd $X1/xbabe3-mirror && tar x 2>/dev/null"
  echo "$(date -u +%FT%TZ) synced new=$(echo $new | wc -w)" >> $D/sync.log
  sleep 180
done
