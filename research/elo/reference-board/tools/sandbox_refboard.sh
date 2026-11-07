#!/bin/bash
# Reference-board sandbox (bwrap, xbabe2). No network, own PID namespace.
# Visible: / read-only, with /home/ubuntu, /tmp and the other data roots as EMPTY tmpfs. Bound back:
#   ~/.local (ro; the system-python libraries), the refboard venvs (venv, venv-gpu, venv-icl, venv-mitra), HF cache and weights (ro), this reference-board
#   source dir (ro), BOARD_ROOT/BOARD.json + BOARD_ROOT/envs (ro), BOARD_ROOT/scores + BOARD_ROOT/preds (rw).
# Nothing else under /home/ubuntu is visible: no other board, no Stage V root, no DL-Val table outside the board's own
# envelopes, no sealed / holdout / meta-test path.
# Usage: sandbox_refboard.sh BOARD_ROOT <command...>
set -euo pipefail
BR=$(readlink -f "$1"); shift
R=/home/ubuntu/jope-data/refboard
SRCDIR=$(dirname "$(dirname "$(readlink -f "$0")")")
[ -s "$BR/BOARD.json" ] && [ -d "$BR/envs" ] || { echo "not a board root: $BR" >&2; exit 2; }
mkdir -p "$BR/scores" "$BR/preds"
# environments: whichever of the CPU venv (xbabe2) / GPU venv (xbabe3) exists on this host
EXTRA=()
for d in "$R/venv" "$R/venv-gpu" "$R/venv-icl" "$R/venv-mitra" "$R/hf" "$R/weights"; do [ -d "$d" ] && EXTRA+=(--ro-bind "$d" "$d"); done
# REFBOARD_GPU=1: expose the NVIDIA device nodes (and nothing else) for CUDA models
if [ "${REFBOARD_GPU:-0}" = 1 ]; then
  for n in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
    [ -e "$n" ] && EXTRA+=(--dev-bind "$n" "$n")
  done
fi
TMPFS=()
for d in /mnt /media /srv /app /review /Users /opt/work; do [ -d "$d" ] && TMPFS+=(--tmpfs "$d"); done
# network: own (empty) namespace where unprivileged bwrap may create one (xbabe2). Where it may not (xbabe3: the
# loopback setup is refused), the network is shared and only HF_HUB_OFFLINE=1 / TABPFN_DISABLE_TELEMETRY=1 apply.
NET=(--unshare-net)
bwrap --ro-bind / / --unshare-net true 2>/dev/null || NET=()
# No bwrap at all (xbabe3: unprivileged user namespaces are blocked, "setting up uid map: Permission denied"):
# refuse unless REFBOARD_ALLOW_NOSANDBOX=1 is set explicitly; then run with the same environment and no confinement
# (score_reference.py itself opens only BOARD.json and envs/<cid>.jtt1 of this board). Documented in README.md.
if ! bwrap --ro-bind / / true 2>/dev/null; then
  [ "${REFBOARD_ALLOW_NOSANDBOX:-0}" = 1 ] || { echo "bwrap unavailable on $(hostname); set REFBOARD_ALLOW_NOSANDBOX=1 to run unconfined" >&2; exit 4; }
  cd "$BR"
  exec env HOME=/home/ubuntu HF_HOME="$R/hf" HF_HUB_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 "$@"
fi
exec bwrap --ro-bind / / \
  --tmpfs /home/ubuntu --tmpfs /tmp "${TMPFS[@]}" \
  --dev /dev --proc /proc \
  --ro-bind /home/ubuntu/.local /home/ubuntu/.local \
  "${EXTRA[@]}" \
  --ro-bind "$SRCDIR" "$SRCDIR" \
  --ro-bind "$BR/BOARD.json" "$BR/BOARD.json" --ro-bind "$BR/envs" "$BR/envs" \
  --bind "$BR/scores" "$BR/scores" --bind "$BR/preds" "$BR/preds" \
  "${NET[@]}" --unshare-pid --die-with-parent \
  --setenv HOME /home/ubuntu --setenv HF_HOME "$R/hf" --setenv HF_HUB_OFFLINE 1 \
  --setenv TABPFN_DISABLE_TELEMETRY 1 --setenv PYTHONDONTWRITEBYTECODE 1 \
  "$@"
