#!/bin/bash
# v8b launches (Jepson approval 2026-10-07 3:10pm MT). Runs alongside hv8r2 (xbabe1) and dv2r2 (xbabe3).
cd /home/ubuntu/ta_encoders/ta-20261007T2030Z
COMMON="--data data --probe-cids meta/probe_cids.json --probe-data /home/ubuntu/jopedime_probe_exp/20261007T1710Z/data --probe-cache /home/ubuntu/ta_encoders/ta-20261007T2030Z/probe_cache --exclude meta/exclude_sealed.json --probe-threads 6 --seed 0 --batch 16 --lr 0.0003 --eval-min 20 --pcap 128 --nmax 384 --max-hours 12 --guard-siblings hv8raw,hv8ft"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
screen -dmS ta-v8braw bash -c "nice -n 5 python3 code-v8b/train_ta.py --model hv8raw --out v8braw --epochs 400 $COMMON > logs/v8braw.out 2>&1"
# moved to xbabe3 (v8b_launch_xbabe3_ft.sh) at 21:22Z; original xbabe1 launch was:
# screen -dmS ta-v8bft bash -c "nice -n 5 python3 code-v8b/train_ta.py --model hv8ft --out v8bft --epochs 120 --k-train 2 --trunk-lr 2e-5 --eval-batch 8 $COMMON > logs/v8bft.out 2>&1"
