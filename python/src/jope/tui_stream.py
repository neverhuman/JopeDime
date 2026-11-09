"""Dramatic real-time "gene stream" TUI screen (rich.Live, no Textual).

Left pane: a live table tailing ``gene_events.jsonl`` — one row per gene, updated in
place across proposed -> scored -> selected, color-coded by score, with a genes/sec
readout and a best-gene ticker.

Right pane: the standalone algorithm SOURCE (Python or Rust) of the highlighted gene,
syntax-highlighted via ``rich.syntax.Syntax`` and fetched on demand through
``EngineRouter.emit`` (LRU-cached). Genes whose engine can't yet emit (newmath invention
ops) show the unsupported-op list instead.

Keyboard (cbreak on /dev/tty): j/k or arrows move the cursor, p/r switch python/rust,
l toggles latest-follow, s saves the current source, q quits. ``handle_key`` is a pure
function so the whole input model is unit-testable without a tty.

Perf: an incremental byte-offset ``JsonlTailer`` reads only appended bytes; the in-memory
gene map is capped; header stats come from the O(1) ``.meta.json`` sidecar, never a feed
scan; the code pane is built only when the selection or language changes.
"""

from __future__ import annotations

import json
import os
import select
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from rich.text import Text

from .engines import EngineRouter
from .tui import _truncate as _truncate
from .tui_stream_state import Gene, JsonlTailer, StreamState, handle_key

CANON_PREVIEW = 60
CODE_PREVIEW_CHARS = 20_000
MAX_GENES = 2000
INITIAL_TAIL_BYTES = 2 * 1024 * 1024


_SPARK = "▁▂▃▄▅▆▇█"


def _tail_jsonl(path: Path | None, limit: int) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-limit:]


def _sparkline(values: list[float], width: int = 48) -> str:
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    return "".join(_SPARK[min(len(_SPARK) - 1, int((v - lo) / span * (len(_SPARK) - 1)))] for v in vals)


def _bar(value: float, total: float, width: int, style: str) -> Text:
    frac = 0.0 if total <= 0 else max(0.0, min(1.0, value / total))
    filled = int(round(frac * width))
    t = Text()
    t.append("█" * filled, style=style)
    t.append("░" * (width - filled), style="grey30")
    t.append(f" {value:3.0f}%", style=style)
    return t


def _state_glyph(state: str) -> tuple[str, str]:
    return {
        "proposed": ("○", "dim"),
        "selected": ("◍", "yellow"),
        "scored": ("●", "green"),
    }.get(state, ("·", "dim"))


def _score_cell(value: float | None) -> tuple[str, str]:
    if value is None:
        return ("—", "dim")
    if value > 0.3:
        style = "bold green"
    elif value >= 0.0:
        style = "cyan"
    else:
        style = "red"
    return (f"{value:+.3f}", style)


def _fmt(value: float | None, *, pct: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _read_key(fd: int) -> str | None:
    """Decode a single keypress (handles 3-byte arrow escapes). Non-blocking caller."""
    ch = os.read(fd, 1).decode("utf-8", "ignore")
    if ch == "\x1b":
        # arrow keys arrive as ESC [ A/B/C/D
        rest = os.read(fd, 2).decode("utf-8", "ignore")
        return {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}.get(rest, "ESC")
    return ch


def _load_split_mixins():
    from .tui_stream_data import StreamDataMixin as _StreamDataMixin
    from .tui_stream_display import StreamDisplayMixin as _StreamDisplayMixin
    from .tui_stream_panels import StreamPanelsMixin as _StreamPanelsMixin

    return _StreamDataMixin, _StreamDisplayMixin, _StreamPanelsMixin


_StreamDataMixin, _StreamDisplayMixin, _StreamPanelsMixin = _load_split_mixins()
del _load_split_mixins


class StreamView(_StreamDataMixin, _StreamPanelsMixin, _StreamDisplayMixin):
    def __init__(
        self,
        run_dir: str | Path,
        *,
        router: EngineRouter | None = None,
        train_metrics: str | Path | None = None,
        cluster: Any = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.feed_path = self.run_dir / "gene_events.jsonl"
        self.meta_path = self.run_dir / "gene_events.jsonl.meta.json"
        self.router = router or EngineRouter()
        self.tailer = JsonlTailer(self.feed_path)
        self.genes: OrderedDict[str, Gene] = OrderedDict()
        self.state = StreamState()
        self._seq = 0
        self._code_cache: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
        # Live training loss: explicit path, else auto-detect a metrics feed in the run dir.
        # Auto-detected paths are RE-RESOLVED each render so a long-running TUI follows
        # self-play across generations (gen1_train_metrics -> gen2_train_metrics -> ...).
        self._train_metrics_pinned = bool(train_metrics)
        self.train_metrics_path = Path(train_metrics) if train_metrics else self._auto_train_metrics()
        self.cluster = cluster  # ClusterMonitor | None (already started by the caller)


def run_stream_tui(
    run_dir: str | Path,
    *,
    refresh: float = 10.0,
    once: bool = False,
    train_metrics: str | Path | None = None,
    cluster_hosts: tuple[str, ...] | None = None,
) -> None:
    from rich.live import Live

    cluster = None
    if cluster_hosts:
        from .cluster_monitor import ClusterMonitor

        cluster = ClusterMonitor(hosts=cluster_hosts)
        # one-shot renders poll synchronously (no time for a background cycle); live mode
        # runs the background thread and refreshes every couple of seconds.
        cluster.poll_once() if once else cluster.start()
    view = StreamView(run_dir, train_metrics=train_metrics, cluster=cluster)
    if once:
        from rich.console import Console

        Console().print(view.render())
        return

    import termios
    import tty

    fd = None
    saved = None
    try:
        fd = os.open("/dev/tty", os.O_RDWR)
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except OSError:
        fd = None  # not a tty (piped) — render without keyboard

    period = max(0.05, 1.0 / max(1.0, refresh))
    try:
        with Live(view.render(), refresh_per_second=refresh, screen=True, auto_refresh=False) as live:
            while not view.state.quit:
                if fd is not None:
                    ready, _, _ = select.select([fd], [], [], period)
                    if ready:
                        key = _read_key(fd)
                        if key:
                            view.state = handle_key(view.state, key, len(view.state.order))
                            if view.state.save:
                                view.save_current()
                else:
                    time.sleep(period)
                live.update(view.render(), refresh=True)
    finally:
        if cluster is not None:
            cluster.stop()
        if fd is not None and saved is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            os.close(fd)
