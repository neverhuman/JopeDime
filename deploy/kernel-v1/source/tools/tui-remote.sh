#!/usr/bin/env bash
set -euo pipefail
exec rtk proxy ssh -tt -o BatchMode=yes -i /home/ubuntu/.ssh/lambda_cloud_a100_20260907_ed25519 ubuntu@158.101.20.226 'python3 /home/ubuntu/jope-dime/parallel/tools/tui-session.py --check && exec tmux attach-session -t =jope'
