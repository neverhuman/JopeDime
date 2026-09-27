#!/usr/bin/env bash
# A recovered upload does not rerun or change the result of the numerical/test lane.
set -euo pipefail
[ "$#" = 3 ] || { echo 'three publication outcomes are required' >&2; exit 1; }
printf 'CI publication attempts: first=%s second=%s third=%s\n' "$1" "$2" "$3"
case "$1:$2:$3" in
  success:skipped:skipped|failure:success:skipped|failure:failure:success) exit 0 ;;
  *) echo 'no successful bounded publication sequence; evidence gate failed' >&2; exit 1 ;;
esac
