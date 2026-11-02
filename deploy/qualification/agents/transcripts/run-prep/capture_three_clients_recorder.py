"""Signal handling and bounded recorder drain helpers."""

from __future__ import annotations

import signal
import time

def _current_sigmask() -> tuple[set[signal.Signals], BaseException | None]:
    pending_error = None
    while True:
        try:
            return signal.pthread_sigmask(signal.SIG_BLOCK, ()), pending_error
        except KeyboardInterrupt as error:
            if pending_error is None:
                pending_error = error


def _block_sigint() -> tuple[set[signal.Signals], BaseException | None]:
    # Read the mask before changing it: pthread_sigmask applies the new mask and
    # only then runs pending handlers, so an interrupt raised by the blocking call
    # must not be retried into a "previous" mask that already blocks SIGINT.
    previous_mask, pending_error = _current_sigmask()
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    except KeyboardInterrupt as error:
        if pending_error is None:
            pending_error = error
    return previous_mask, pending_error


def _restore_sigint(previous_mask: set[signal.Signals]) -> BaseException | None:
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    except BaseException as error:
        return error
    return None


def wait_for_recorder(
    process, capture_dir, recorder, check_interrupt, deadline
) -> None:
    """Bound normal-exit EOF/receipt draining without signaling either session."""
    while True:
        check_interrupt()
        require(
            time.monotonic() < deadline, "recorder flush/EOF drain deadline exceeded"
        )
        groups = recorder.groups()
        client_members = any(p["group"] == process.pid for p in _live_processes())
        receipt = capture_dir / "capture.json"
        require(
            not capture_dir.is_symlink() and not receipt.is_symlink(),
            "capture receipt path is a symlink",
        )
        if receipt.is_file() and not groups and not client_members:
            return
        require(
            time.monotonic() < deadline, "recorder flush/EOF drain deadline exceeded"
        )
        time.sleep(0.02)

