#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/jope-dime/parallel
test -x bin/jope-parallel
test -x bin/jope-parallel-cpu
test -s manifests/generated/gp-qualification-jobs.json
sudo install -m 0644 systemd/jope-parallel-monitor.service /etc/systemd/system/
sudo install -m 0644 systemd/jope-parallel-gp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jope-parallel-monitor.service
sudo systemctl enable --now jope-parallel-gp.service
