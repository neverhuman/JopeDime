#!/usr/bin/env bash
# Historical unit name retained; reporting only. Evidence requires explicit custody cleanup.
set -euo pipefail
printf 'runner storage report %s (no deletion)\n' "$(date -u +%FT%TZ)"
df -B1 --output=source,size,used,avail,pcent / /var/lib/docker
docker system df
docker volume ls --filter name=jope-
