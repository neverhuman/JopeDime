#!/bin/bash
# Host the operator-facing JOPE self-play TUI in a detached tmux session.
#
# Why this exists: the TUI used to run as a bare foreground process inside one
# operator SSH session (pid 2907910 on pts/36, 2026-07-27..28). Every binary
# upgrade then needed a human at that exact terminal to restart it, and a crash
# or a dropped SSH connection silently killed the dashboard. That is the kind of
# hand-holding the operator directive forbids.
#
#   attach:  tmux -L jope attach -t jope-tui        (detach with ctrl-b d)
#   upgrade: install the new binary, then
#            pkill -f '^/home/ubuntu/jope-dime/target/release/jope_selfplay_tui$'
#            -- the inner loop below respawns it from the new inode within 3s,
#            without dropping the tmux session an operator may be attached to.
#
#            The ^…$ anchors are load-bearing. An unanchored
#            `pkill -f 'target/release/jope_selfplay_tui'` also matches the tmux
#            server (the binary path is inside its new-session command line),
#            the inner `bash -c while true` loop, AND the very shell running the
#            pkill -- verified 2026-07-28 04:07Z: it killed the tmux server and
#            self-terminated with exit 144. systemd rebuilt the session 13s
#            later, but any attached operator was dropped. Anchoring matches
#            only the TUI process, whose argv is exactly the binary path.
#
# Two supervision layers, on purpose:
#   * inner loop  -> survives TUI exits/crashes/binary swaps, session preserved
#   * --supervise -> holds jope-selfplay-tui.service open so systemd's
#                    Restart=always recreates the session if the tmux server
#                    itself dies. Without this the unit's cgroup teardown kills
#                    the very server it just started.
set -uo pipefail

SOCKET=jope
SESSION=jope-tui
BIN=/home/ubuntu/jope-dime/target/release/jope_selfplay_tui

export TERM="${TERM:-xterm-256color}"

ensure_session() {
    if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
        return 0
    fi
    if [ ! -x "$BIN" ]; then
        echo "jope_tui_session: TUI binary missing or not executable: $BIN" >&2
        return 1
    fi
    # The respawn loop lives inside tmux rather than in systemd because the TUI
    # needs a real pty and the operator needs to be able to attach to it.
    tmux -L "$SOCKET" new-session -d -s "$SESSION" -x 240 -y 64 \
        "while true; do '$BIN'; sleep 3; done" || return 1
    tmux -L "$SOCKET" set-option -t "$SESSION" history-limit 2000 >/dev/null 2>&1
    echo "jope_tui_session: started tmux -L $SOCKET session $SESSION"
}

ensure_session || exit 1

if [ "${1:-}" = "--supervise" ]; then
    while tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; do
        sleep 10
    done
    echo "jope_tui_session: session $SESSION vanished — exiting for systemd to restart" >&2
    exit 1
fi
