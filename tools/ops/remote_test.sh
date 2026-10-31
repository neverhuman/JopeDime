#!/bin/bash
# remote_test.sh <host|auto> <branch|sha> <cargo args...>   e.g. remote_test.sh auto HEAD test -p jope-feat-math --lib
# Ships `git archive <sha>` of the current repo to a scratch dir on the target, runs cargo there
# (xbabe4 = Mac: inside Docker rust:1.97.1-slim-bookworm with cached volumes), streams output back.
# Picks least-loaded enabled host whose load stays < 85%. NEVER xbabe0. Needs >=40G free disk generally; xbabe1 requires >=100G. xbabe3 is temporarily disabled by owner call t0855.
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
  else out=$(timeout 10 ssh -o BatchMode=yes $h 'echo $(nproc) $(cut -d" " -f1 /proc/loadavg) $(df -BG --output=avail ~|tail -1|tr -dc 0-9) $(pgrep -fc "remote_tests/src\\.[0-9]+ &&")' 2>/dev/null); fi
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
# disk guard: clear ~/remote_tests/target if >60G or free <150G, only when no other src.* job runs
GUARD='T=$HOME/remote_tests/target; [ -d $T ] || exit 0; sz=$(du -sBG $T 2>/dev/null|cut -f1|tr -dc 0-9); fr=$(df -BG --output=avail $HOME|tail -1|tr -dc 0-9); if [ "${sz:-0}" -gt 60 ] || [ "${fr:-999}" -lt 150 ]; then if pgrep -f "remote_tests/src\.[0-9]" >/dev/null; then echo "guard: target ${sz}G free ${fr}G but jobs running; not clearing" >&2; else echo "guard: clearing $T (${sz}G, free ${fr}G)" >&2; rm -rf $T; fi; fi'
case $HOST in xbabe2) bash -c "$GUARD";; xbabe1|xbabe4) ;; xbabe3) ;; *) ssh -o BatchMode=yes "$HOST" "$GUARD";; esac
J=$(probe $HOST); [ "$J" -lt 4 ] && { echo "$HOST busy/low disk (slots=$J)" >&2; exit 3; }
[ -n "${JMAX:-}" ] && [ "$J" -gt "$JMAX" ] && J=$JMAX  # JMAX=8 honours lane job caps (gb 2026-10-31)
echo ">> $HOST sha=${SHA:0:9} jobs=$J: cargo $*" >&2
ARGS=$(printf '%q ' "$@")
if [ "$HOST" = xbabe2 ]; then  # gb 2026-10-31: test the exact SHA (not the dirty checkout) with an absolute cargo path
  S=$HOME/remote_tests/src.$$; mkdir -p $S/target $HOME/remote_tests/target && git -C "$REPO" archive --format=tar "$SHA" | tar -x -C $S || exit 4
  cd $S && CARGO_TARGET_DIR=$HOME/remote_tests/target nice -n 10 $HOME/.cargo/bin/cargo "$@" -j $J; rc=$?; cd /; rm -rf $S; exit $rc
fi
W='~/remote_tests'
if [ "$HOST" = xbabe4 ]; then
  git -C "$REPO" archive --format=tar "$SHA" | ssh -o BatchMode=yes $HOST "/usr/local/bin/docker run -i --rm --cpus=$J -v jope_target:/target -v jope_cargo:/usr/local/cargo/registry -e CARGO_TARGET_DIR=/target rust:1.97.1-slim-bookworm bash -c 'mkdir -p /w && tar -x -C /w && cd /w && command -v g++ >/dev/null || (apt-get update -qq && apt-get install -y -qq build-essential pkg-config libssl-dev >/dev/null); nice -n 10 cargo $ARGS -j $J'"
else
  git -C "$REPO" archive --format=tar "$SHA" | ssh -o BatchMode=yes $HOST "mkdir -p $W/src.$$/target $W/target && tar -x -C $W/src.$$" || exit 4
  ssh -o BatchMode=yes $HOST "cd $W/src.$$ && CUDA_VISIBLE_DEVICES='' CARGO_TARGET_DIR=\$HOME/remote_tests/target nice -n 10 ~/.cargo/bin/cargo $ARGS -j $J; rc=\$?; rm -rf $W/src.$$; exit \$rc"
fi
