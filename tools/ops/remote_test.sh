#!/bin/bash
# remote_test.sh <host|auto> <branch|sha> <cargo args...>   e.g. remote_test.sh auto HEAD test -p jope-feat-math --lib
# Ships an exact commit bundle to one reusable source checkout on the target, runs cargo there
# (xbabe4 = Mac: inside Docker rust:1.97.1-slim-bookworm with cached volumes), streams output back.
# Reuses one locked source checkout and one shared Cargo target per host. Picks the least-loaded
# enabled host whose load stays < 85%. NEVER xbabe0. Needs >=40G free disk generally; xbabe1 requires >=100G. xbabe3 is temporarily disabled by owner call t0855.
# Existing remote-shell expansions and word splitting are intentional; Cargo args are shell-escaped before transport.
# shellcheck disable=SC2016,SC2086,SC2088
set -o pipefail
HOST=$1; REF=$2; shift 2; [ $# -eq 0 ] && set -- check --workspace --locked
REPO=$(git rev-parse --show-toplevel) || exit 2
SHA=$(git -C "$REPO" rev-parse "$REF") || exit 2
min_free_gb(){
  case "$1" in xbabe1) echo 100;; xbabe3) echo 60;; *) echo 40;; esac
}
auto_hosts(){
  printf '%s\n' xbabe1 xbabe4
}
host_is_disabled(){
  [ "$1" = xbabe3 ]
}
probe(){ # prints "free_slots" or -1
  local h=$1 out
  if [ "$h" = xbabe4 ]; then out=$(timeout 10 ssh -o BatchMode=yes $h 'echo $(sysctl -n hw.ncpu) $(sysctl -n vm.loadavg|awk "{print \$2}") $(df -g ~|awk "NR==2{print \$4}") $(/usr/local/bin/docker ps -q --filter ancestor=rust:1.97.1-slim-bookworm 2>/dev/null|wc -l)' 2>/dev/null)
  elif [ "$h" = xbabe2 ]; then out="$(nproc) $(cut -d" " -f1 /proc/loadavg) $(df -BG --output=avail ~|tail -1|tr -dc 0-9)"
  else out=$(timeout 10 ssh -o BatchMode=yes $h 'q=0; if [ -e "$HOME/remote_tests/.remote-test.lock" ]; then flock -n "$HOME/remote_tests/.remote-test.lock" -c true >/dev/null 2>&1 || q=1; fi; echo $(nproc) $(cut -d" " -f1 /proc/loadavg) $(df -BG --output=avail ~|tail -1|tr -dc 0-9) $q' 2>/dev/null); fi
  [ -z "$out" ] && { echo -1; return; }
  set -- $out; floor=$(min_free_gb "$h"); [ "${3:-0}" -lt "$floor" ] && { echo -1; return; }  # protect host-specific free-space floors
  # queued cargo jobs share one target lock and burn no CPU while waiting: charge 8 slots each (gb 2026-10-31)
  awk -v c=$1 -v l=$2 -v q=${4:-0} 'BEGIN{s=int(c*0.85-l); if(s>32)s=32; s-=8*q; print (s<4?-1:s)}'
}
if [ "$HOST" = auto ]; then best=-1
  for h in $(auto_hosts);  # t0855 owner call: CPU-only pool excludes low-space xbabe3
  do s=$(probe $h); echo "probe $h slots=$s" >&2; [ "$s" -gt "$best" ] && { best=$s; HOST=$h; }; done
  [ "$best" -lt 4 ] && { echo "no host under 85% load; retry later" >&2; exit 3; }
fi
[ "$HOST" = xbabe0 ] && { echo "xbabe0 forbidden" >&2; exit 2; }
host_is_disabled "$HOST" && { echo "xbabe3 disabled by owner call; re-enable only after disk use is below 90% and a follow-up config update; use auto (xbabe1/xbabe4)" >&2; exit 2; }
# Keep the source checkout and shared target locked as one unit. The lock lives
# outside target so the low-space guard cannot unlink a lock held by a build.
GUARD='T=$HOME/remote_tests/target; L=$HOME/remote_tests/.remote-test.lock; [ -d "$T" ] || exit 0; mkdir -p "$HOME/remote_tests"; flock -n "$L" bash -c '\''T=$1; sz=$(du -sBG "$T" 2>/dev/null|cut -f1|tr -dc 0-9); fr=$(df -BG --output=avail "$HOME"|tail -1|tr -dc 0-9); if [ "${sz:-0}" -gt 60 ] || [ "${fr:-999}" -lt 150 ]; then echo "guard: clearing $T (${sz}G, free ${fr}G)" >&2; rm -rf "$T"; fi'\'' _ "$T" || echo "guard: shared remote test source/target is in use; not clearing" >&2'
case $HOST in xbabe2) bash -c "$GUARD";; xbabe1|xbabe4) ;; xbabe3) ;; *) ssh -o BatchMode=yes "$HOST" "$GUARD";; esac
J=$(probe $HOST); [ "$J" -lt 4 ] && { echo "$HOST busy/low disk (slots=$J)" >&2; exit 3; }
[ -n "${JMAX:-}" ] && [ "$J" -gt "$JMAX" ] && J=$JMAX  # JMAX=8 honours lane job caps (gb 2026-10-31)
echo ">> $HOST sha=${SHA:0:9} jobs=$J: cargo $*" >&2
ARGS=$(printf '%q ' "$@")
mkdir -p "$HOME/remote_tests" || exit 4
# Keep local, SSH, and container Cargo builds from writing incremental state
# and full development debug info into the shared test target.
if [ "$HOST" = xbabe2 ]; then  # gb 2026-10-31: test the exact SHA (not the dirty checkout) with an absolute cargo path
  S=$HOME/remote_tests/src; T=$HOME/remote_tests/target; L=$HOME/remote_tests/.remote-test.lock
  mkdir -p "$S" "$T" || exit 4
  flock "$L" bash -c 'set -e; S=$1; T=$2; REPO=$3; SHA=$4; J=$5; CARGO=$6; shift 6; if [ ! -e "$S/.git" ]; then git -C "$S" init -q; fi; git -C "$S" fetch -q "$REPO" "$SHA"; git -C "$S" checkout -q --force "$SHA"; git -C "$S" clean -ffdx; cd "$S"; CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_TARGET_DIR="$T" nice -n 10 "$CARGO" "$@" -j "$J"' _ "$S" "$T" "$REPO" "$SHA" "$J" "$HOME/.cargo/bin/cargo" "$@"
  exit $?
fi
TREE=$(git -C "$REPO" rev-parse "$SHA^{tree}") || exit 4
TRANSFER_SHA=$(GIT_AUTHOR_NAME='JopeDime remote test' GIT_AUTHOR_EMAIL='jope-remote-test@localhost' GIT_COMMITTER_NAME='JopeDime remote test' GIT_COMMITTER_EMAIL='jope-remote-test@localhost' GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' git -C "$REPO" commit-tree "$TREE" -m "remote source snapshot $SHA") || exit 4
REF="refs/jope-remote-test/$$"
git -C "$REPO" update-ref "$REF" "$TRANSFER_SHA" || exit 4
trap 'git -C "$REPO" update-ref -d "$REF" "$TRANSFER_SHA"' EXIT
if [ "$HOST" = xbabe4 ]; then
  INNER="set -e; S=/workspace/src; B=/workspace/source.bundle; cleanup(){ rm -f \"\$B\"; }; trap cleanup EXIT; cat > \"\$B\"; if [ ! -e \"\$S/.git\" ]; then git -C \"\$S\" init -q; fi; git -C \"\$S\" fetch -q \"\$B\" $REF; git -C \"\$S\" checkout -q --force FETCH_HEAD; git -C \"\$S\" clean -ffdx; git -C \"\$S\" config jope.sourceSha $SHA; rm -f \"\$B\"; cd \"\$S\"; command -v g++ >/dev/null || (apt-get update -qq && apt-get install -y -qq build-essential pkg-config libssl-dev >/dev/null); nice -n 10 cargo $ARGS -j $J"
  printf -v INNER_Q '%q' "$INNER"
  CONTAINER_SCRIPT="set -e; mkdir -p /workspace/src /target; flock /workspace/.remote-test.lock bash -c $INNER_Q"
  printf -v CONTAINER_SCRIPT_Q '%q' "$CONTAINER_SCRIPT"
  CONTAINER_CMD="/usr/local/bin/docker run -i --rm --cpus=$J -v jope_source:/workspace -v jope_target:/target -v jope_cargo:/usr/local/cargo/registry -e CARGO_TARGET_DIR=/target -e CARGO_INCREMENTAL=0 -e CARGO_PROFILE_DEV_DEBUG=line-tables-only rust:1.97.1-slim-bookworm bash -c $CONTAINER_SCRIPT_Q"
  git -C "$REPO" bundle create - "$REF" | ssh -o BatchMode=yes "$HOST" "$CONTAINER_CMD"
else
  INNER="set -e; S=\$HOME/remote_tests/src; T=\$HOME/remote_tests/target; B=\$(mktemp \"\$HOME/remote_tests/source.bundle.XXXXXX\"); cleanup(){ rm -f \"\$B\"; }; trap cleanup EXIT; cat > \"\$B\"; if [ ! -e \"\$S/.git\" ]; then git -C \"\$S\" init -q; fi; git -C \"\$S\" fetch -q \"\$B\" $REF; git -C \"\$S\" checkout -q --force FETCH_HEAD; git -C \"\$S\" clean -ffdx; git -C \"\$S\" config jope.sourceSha $SHA; rm -f \"\$B\"; cd \"\$S\"; CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CUDA_VISIBLE_DEVICES= CARGO_TARGET_DIR=\"\$T\" nice -n 10 ~/.cargo/bin/cargo $ARGS -j $J"
  printf -v INNER_Q '%q' "$INNER"
  REMOTE_CMD="mkdir -p \$HOME/remote_tests/src \$HOME/remote_tests/target && flock \$HOME/remote_tests/.remote-test.lock bash -c $INNER_Q"
  git -C "$REPO" bundle create - "$REF" | ssh -o BatchMode=yes "$HOST" "$REMOTE_CMD"
fi
