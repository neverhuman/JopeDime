#!/usr/bin/env bash
# collect.sh [host…] : copy case-study outputs from the remote run hosts into this host's tree.
#
# Every host writes ~/jope-data/case_study/<case>/<stamp>/… (xbabe4 inside its colima VM under
# /home/agent.guest/resfit-arm64/home/jope-data/case_study). Stamps are unique per launch, so
# copying them side by side never overwrites another host's run; an existing local copy is only
# updated, never removed (rsync without --delete). Each run.json names the host that produced it.
# Default hosts: xbabe1 xbabe3 xbabe4.
set -euo pipefail
dest=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}
hosts=("$@"); [ ${#hosts[@]} -gt 0 ] || hosts=(xbabe1 xbabe3 xbabe4)
mkdir -p "$dest"
for h in "${hosts[@]}"; do
  case $h in
    xbabe4)
      ssh -o BatchMode=yes -o LogLevel=ERROR xbabe4 \
        'export PATH=/opt/homebrew/bin:$PATH; colima ssh -p jope-ci -- sh -c "tar czf - -C /home/agent.guest/resfit-arm64/home/jope-data case_study"' 2>/dev/null \
        | tar xzf - -C "$(dirname "$dest")" --keep-newer-files 2>/dev/null || true
      ;;
    *)
      rsync -a --exclude 'test-registry/' "$h:jope-data/case_study/" "$dest/" ;;
  esac
  echo "collect: $h done"
done
