from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, BinaryIO

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}


class JsonlTail:
    """Incremental JSONL reader that tolerates partial writes and rename rotation."""

    def __init__(
        self,
        path: Path,
        *,
        limit: int | None,
        initial_bytes: int | None = 1024 * 1024,
    ) -> None:
        self.path = path
        self.rows: deque[dict[str, Any]] = deque(maxlen=limit)
        self.initial_bytes = initial_bytes
        self.handle: BinaryIO | None = None
        self.inode: tuple[int, int] | None = None
        self.partial = b""

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.handle = None
        self.inode = None
        self.partial = b""

    def _open(self) -> bool:
        try:
            handle = self.path.open("rb")
            stat = os.fstat(handle.fileno())
        except OSError:
            return False
        if self.initial_bytes is not None and stat.st_size > self.initial_bytes:
            handle.seek(stat.st_size - self.initial_bytes)
            handle.readline()
        self.handle = handle
        self.inode = (stat.st_dev, stat.st_ino)
        self.partial = b""
        return True

    def _consume(self) -> None:
        assert self.handle is not None
        data = self.partial + self.handle.read()
        parts = data.split(b"\n")
        self.partial = parts.pop()
        for raw in parts:
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                self.rows.append(value)

    def poll(self) -> list[dict[str, Any]]:
        if self.handle is None and not self._open():
            return list(self.rows)
        assert self.handle is not None
        self._consume()
        try:
            path_stat = self.path.stat()
            current = (path_stat.st_dev, path_stat.st_ino)
        except OSError:
            return list(self.rows)
        if current != self.inode or path_stat.st_size < self.handle.tell():
            # Consume the renamed file to EOF, discard any incomplete old tail,
            # then begin at byte zero of the replacement.
            self._consume()
            self.close()
            if self._open():
                self._consume()
        return list(self.rows)


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def _sparkline(values: list[Any], width: int = 64) -> str:
    finite = [float(value) for value in values[-width:] if isinstance(value, (int, float))]
    if not finite:
        return "-"
    blocks = "▁▂▃▄▅▆▇█"
    low, high = min(finite), max(finite)
    if high - low < 1e-12:
        return blocks[3] * len(finite)
    return "".join(blocks[min(7, int(7 * (value - low) / (high - low)))] for value in finite)


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _downsample_full_history(values: list[Any], width: int) -> list[float | None]:
    """Compress the entire series into equal-width mean buckets."""

    if not values or width <= 0:
        return []
    bucket_count = min(len(values), width)
    sampled: list[float | None] = []
    for column in range(bucket_count):
        start = column * len(values) // bucket_count
        end = (column + 1) * len(values) // bucket_count
        finite = [number for value in values[start:end] if (number := _finite_float(value)) is not None]
        sampled.append(sum(finite) / len(finite) if finite else None)
    return sampled


def _scaled_sparkline(
    values: list[Any],
    *,
    low: float,
    high: float,
    width: int = 64,
    style: str = "white",
) -> Text:
    """Render a whole-series sparkline against a caller-provided shared scale."""

    blocks = "▁▂▃▄▅▆▇█"
    span = high - low
    result = Text()
    for value in _downsample_full_history(values, width):
        if value is None:
            result.append("·", style="dim")
            continue
        level = (
            3
            if span < 1e-12
            else min(
                7,
                max(0, int(7 * (value - low) / span)),
            )
        )
        result.append(blocks[level], style=style)
    return result or Text("-", style="dim")


def _history_plot(
    title: str,
    epochs: list[Any],
    series: list[tuple[str, list[Any], str]],
    *,
    fixed_range: tuple[float, float] | None = None,
    width: int = 64,
) -> Group:
    finite = [
        number for _, values, _ in series for value in values if (number := _finite_float(value)) is not None
    ]
    if fixed_range is not None:
        low, high = fixed_range
    elif finite:
        low, high = min(finite), max(finite)
    else:
        low, high = 0.0, 1.0

    numeric_epochs = [number for epoch in epochs if (number := _finite_float(epoch)) is not None]
    epoch_range = f"{int(numeric_epochs[0])}–{int(numeric_epochs[-1])}" if numeric_epochs else "-"
    header = Text.assemble(
        (title, "bold"),
        (
            f"  ·  full epochs {epoch_range}  ·  {len(epochs)} points  ·  shared scale {low:.4g}–{high:.4g}",
            "dim",
        ),
    )
    table = Table(show_header=True, box=box.SIMPLE, expand=True, padding=(0, 1))
    table.add_column("series", no_wrap=True)
    table.add_column("latest", justify="right", no_wrap=True)
    table.add_column("oldest → newest (entire history)", ratio=1)
    for label, values, style in series:
        latest = next(
            (number for value in reversed(values) if (number := _finite_float(value)) is not None),
            None,
        )
        table.add_row(
            Text(label, style=style),
            Text(_fmt(latest, 5), style=style),
            _scaled_sparkline(
                values,
                low=low,
                high=high,
                width=width,
                style=style,
            ),
        )
    return Group(header, table)


class DimeScoreView:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.run = self.root
        self.live = JsonlTail(self.run / "live-metrics.jsonl", limit=1200)
        self.metrics = JsonlTail(
            self.run / "metrics.jsonl",
            limit=None,
            initial_bytes=None,
        )
        self.errors = JsonlTail(self.run / "errors.jsonl", limit=5)

    def _set_run(self, run: Path) -> None:
        if run == self.run:
            return
        self.live.close()
        self.metrics.close()
        self.errors.close()
        self.run = run
        self.live = JsonlTail(run / "live-metrics.jsonl", limit=1200)
        self.metrics = JsonlTail(
            run / "metrics.jsonl",
            limit=None,
            initial_bytes=None,
        )
        self.errors = JsonlTail(run / "errors.jsonl", limit=5)

    def _suite(self) -> dict[str, Any]:
        current = self.root
        suites: list[dict[str, Any]] = []
        for _ in range(4):
            suite = _read_json(current / "suite-progress.v1.json")
            if not suite:
                break
            suites.append(suite)
            active = suite.get("active_run")
            if not isinstance(active, str) or not active:
                break
            selected = Path(active)
            if not selected.is_absolute():
                selected = current / selected
            if selected == current:
                break
            current = selected
        self._set_run(current)
        if not suites:
            return {}
        combined = dict(suites[0])
        for suite in suites[1:]:
            combined.update(suite)
        outer_state = suites[0].get("checkpoint_state")
        if outer_state in {"waiting_for_admission", "waiting_for_data"}:
            combined["active_phase"] = suites[0].get("active_phase")
            combined["checkpoint_state"] = outer_state
        combined["suite_path"] = str(current)
        return combined

    def render(self) -> Group:
        suite = self._suite()
        run = self.run
        identity = _read_json(run / "run.v1.json")
        progress = _read_json(run / "progress.v1.json")
        summary = _read_json(run / "summary.v1.json")
        pilots = _read_json(self.root / "initialization-pilots.v1.json")
        if not pilots:
            pilots = _read_json(run / "initialization-pilots.v1.json")
        preflight = _read_json(self.root / "warm-start-preflight.v1.json")
        if not preflight:
            preflight = _read_json(run / "warm-start-preflight.v1.json")
        ablation = _read_json(self.root / "ablation.v1.json")
        if not ablation:
            ablation = _read_json(run / "ablation.v1.json")
        live_rows = self.live.poll()
        durable_rows = self.metrics.poll()
        errors = self.errors.poll()
        latest_live = live_rows[-1] if live_rows else {}
        last = durable_rows[-1] if durable_rows else {}
        validation = last.get("validation") or progress.get("validation") or summary.get("validation") or {}
        test_metrics = summary.get("test") or progress.get("test") or {}

        state = suite.get(
            "checkpoint_state",
            progress.get("checkpoint_state", "waiting"),
        )
        quality_gate = summary.get(
            "quality_gate",
            suite.get("quality_gate", progress.get("quality_gate", "unrated")),
        )
        title = Text("DIME GENOME SCORE TRAINER", style="bold bright_cyan")
        subtitle = Text(
            f"{run.name}  ·  {state}  ·  {quality_gate}",
            style="white",
        )

        identity_table = Table.grid(expand=True)
        identity_table.add_column()
        identity_table.add_column(justify="right")
        identity_table.add_row(
            "suite phase",
            str(suite.get("active_phase", identity.get("run_kind", "-"))),
        )
        identity_table.add_row(
            "corpus",
            str(
                identity.get(
                    "corpus_archive_sha256",
                    suite.get("corpus_archive_sha256", "-"),
                )
            )[:20],
        )
        identity_table.add_row(
            "embeddings",
            str(
                identity.get(
                    "embedding_sha256",
                    suite.get("embedding_sha256", "-"),
                )
            )[:20],
        )
        identity_table.add_row(
            "grammar",
            str(
                identity.get(
                    "grammar_contract_sha256",
                    suite.get("grammar_contract_sha256", "-"),
                )
            )[:20],
        )
        identity_table.add_row("initialization", str(identity.get("initialization", "-")))
        identity_table.add_row(
            "warm-start",
            ("eligible" if preflight.get("eligible") else str(preflight.get("reason", "-"))),
        )
        identity_table.add_row("seed", str(identity.get("seed", "-")))
        row_split = identity.get("row_split") or {}
        validation_fraction = row_split.get("validation_fraction")
        identity_table.add_row(
            "example holdout",
            (
                f"{100 * validation_fraction:.0f}% · {row_split.get('method', '-')}"
                if isinstance(validation_fraction, (int, float))
                else (f"global genomes · {row_split.get('method', '-')}" if row_split.get("method") else "-")
            ),
        )
        split_dataset_counts = row_split.get("dataset_counts") or {}
        identity_table.add_row(
            "datasets train · val · test",
            " · ".join(str(split_dataset_counts.get(name, "-")) for name in ("train", "validation", "test")),
        )
        randomization = identity.get("batch_randomization") or {}
        identity_table.add_row(
            "batch shuffle",
            str(randomization.get("strategy", "-")),
        )
        packing = identity.get("binary_packing") or {}
        identity_table.add_row(
            "binary packing",
            (
                f"{packing.get('format')} · {packing.get('tokens', '-')} tokens"
                if packing.get("format")
                else "-"
            ),
        )

        losses = Table(show_header=True, box=box.SIMPLE_HEAVY, expand=True)
        losses.add_column("realtime loss")
        losses.add_column("latest", justify="right")
        losses.add_column("history")
        for label, key in (
            ("total", "total_loss"),
            ("Huber", "huber_loss"),
            ("Bradley–Terry", "bradley_terry_loss"),
            ("validity", "validity_loss"),
        ):
            losses.add_row(
                label,
                _fmt(latest_live.get(key), 5),
                _sparkline([row.get(key) for row in live_rows]),
            )

        epochs = [row.get("epoch") for row in durable_rows]

        def train_history(key: str) -> list[Any]:
            return [(row.get("train_losses") or {}).get(key) for row in durable_rows]

        def validation_history(key: str) -> list[Any]:
            return [((row.get("validation") or {}).get("losses") or {}).get(key) for row in durable_rows]

        def quality_history(key: str) -> list[Any]:
            return [(row.get("validation") or {}).get(key) for row in durable_rows]

        epoch_history = Group(
            _history_plot(
                "Roll-up objective ↓",
                epochs,
                [
                    ("train total", train_history("total"), "bright_cyan"),
                    (
                        "validation total",
                        validation_history("total"),
                        "bright_magenta",
                    ),
                ],
            ),
            _history_plot(
                "Percentile fit (Huber) ↓",
                epochs,
                [
                    ("train Huber", train_history("huber"), "cyan"),
                    (
                        "validation Huber",
                        validation_history("huber"),
                        "magenta",
                    ),
                ],
            ),
            _history_plot(
                "Validation generalization",
                epochs,
                [
                    (
                        "Spearman ↑",
                        quality_history("mean_per_dataset_spearman"),
                        "bright_green",
                    ),
                    (
                        "pairwise accuracy ↑",
                        quality_history("pairwise_accuracy"),
                        "yellow",
                    ),
                    ("MAE ↓", quality_history("mae"), "bright_red"),
                    (
                        "constant MAE ↓",
                        quality_history("constant_percentile_mae"),
                        "dim red",
                    ),
                ],
                fixed_range=(0.0, 1.0),
            ),
        )

        learning = Table(show_header=True, box=box.SIMPLE_HEAVY, expand=True)
        learning.add_column("training")
        learning.add_column("value", justify="right")
        learning.add_row(
            "epoch · step",
            (
                f"{latest_live.get('epoch', progress.get('epoch', '-'))}"
                f" · {latest_live.get('step', progress.get('batch', '-'))}/"
                f"{latest_live.get('steps', progress.get('batches', '-'))}"
            ),
        )
        learning.add_row(
            "samples/s",
            _fmt(
                latest_live.get("samples_per_second", progress.get("samples_per_second")),
                0,
            ),
        )
        learning.add_row(
            "loader wait",
            _fmt(
                latest_live.get("loader_wait_pct", progress.get("loader_wait_pct")),
                1,
            )
            + "%",
        )
        learning.add_row(
            "H2D · compute · optimizer",
            " · ".join(
                _fmt(
                    latest_live.get(name, progress.get(name)),
                    2,
                )
                + " ms"
                for name in ("h2d_ms", "compute_ms", "optimizer_ms")
            ),
        )
        learning.add_row(
            "queue depth · starvation",
            (
                f"{latest_live.get('queue_depth', progress.get('queue_depth', '-'))}"
                f" · {latest_live.get('queue_starvations', progress.get('queue_starvations', '-'))}"
            ),
        )
        learning.add_row(
            "batch shape",
            " × ".join(
                map(
                    str,
                    latest_live.get("batch_shape", progress.get("batch_shape", [])),
                )
            )
            or "-",
        )
        learning.add_row(
            "LR · gradient norm",
            f"{_fmt(latest_live.get('learning_rate', progress.get('learning_rate')), 7)}"
            f" · {_fmt(latest_live.get('gradient_norm', progress.get('gradient_norm')), 3)}",
        )
        eta = latest_live.get("eta_seconds", progress.get("eta_seconds"))
        learning.add_row("ETA", _fmt(eta, 0) + " s" if eta is not None else "-")

        quality = Table(show_header=True, box=box.SIMPLE_HEAVY, expand=True)
        quality.add_column("validation")
        quality.add_column("value", justify="right")
        for label, key in (
            ("mean/dataset Spearman", "mean_per_dataset_spearman"),
            ("pairwise accuracy", "pairwise_accuracy"),
            ("MAE", "mae"),
            ("constant MAE", "constant_percentile_mae"),
            ("calibration MAE", "calibration_mae"),
            ("top-decile precision", "top_decile_precision"),
            ("top-decile regret", "top_decile_regret"),
            ("validity AUROC", "validity_auroc"),
            ("validity PR-AUC", "validity_pr_auc"),
        ):
            quality.add_row(label, _fmt(validation.get(key), 4))
        cold_start = (ablation.get("dataset_held_out") or {}).get("validation") or {}
        genome_only = (ablation.get("genome_only") or {}).get("validation") or {}
        quality.add_row(
            "cold-start Spearman",
            _fmt(cold_start.get("mean_per_dataset_spearman"), 4),
        )
        quality.add_row("cold-start MAE", _fmt(cold_start.get("mae"), 4))
        quality.add_row(
            "genome-only Spearman",
            _fmt(genome_only.get("mean_per_dataset_spearman"), 4),
        )
        quality.add_row(
            "conditioning beats genome-only",
            str(ablation.get("dataset_conditioning_beats_genome_only", "-")),
        )
        if test_metrics:
            quality.add_section()
            quality.add_row(
                "untouched test examples",
                str(test_metrics.get("examples", "-")),
            )
            for label, key in (
                ("test Spearman", "mean_per_dataset_spearman"),
                ("test pairwise accuracy", "pairwise_accuracy"),
                ("test MAE", "mae"),
                ("test constant MAE", "constant_percentile_mae"),
                ("test top-decile precision", "top_decile_precision"),
                ("test top-decile regret", "top_decile_regret"),
            ):
                quality.add_row(label, _fmt(test_metrics.get(key), 4))

        system = Table.grid(expand=True)
        system.add_column()
        system.add_column(justify="right")
        system.add_row(
            "GPU utilization",
            _fmt(progress.get("gpu_utilization_pct"), 0) + "%"
            if progress.get("gpu_utilization_pct") is not None
            else "-",
        )
        used = progress.get("gpu_memory_used_mib")
        total = progress.get("gpu_memory_total_mib")
        system.add_row("VRAM", f"{_fmt(used, 0)} / {_fmt(total, 0)} MiB")
        headroom = progress.get("gpu_headroom_mib")
        system.add_row(
            "4 GiB headroom",
            (
                f"{_fmt(headroom, 0)} MiB · {'OK' if progress.get('gpu_headroom_ok') else 'LOW'}"
                if headroom is not None
                else "-"
            ),
        )
        system.add_row(
            "temperature · power",
            f"{_fmt(progress.get('gpu_temperature_c'), 0)} °C · {_fmt(progress.get('gpu_power_watts'), 0)} W",
        )
        system.add_row(
            "graphics · SM · memory clocks",
            " · ".join(
                _fmt(progress.get(name), 0) + " MHz"
                for name in (
                    "gpu_graphics_clock_mhz",
                    "gpu_sm_clock_mhz",
                    "gpu_memory_clock_mhz",
                )
            ),
        )
        system.add_row(
            "checkpoint",
            str(summary.get("checkpoint", progress.get("checkpoint_state", "-"))),
        )

        comparison = Table(show_header=True, box=box.SIMPLE, expand=True)
        comparison.add_column("init")
        comparison.add_column("eligible", justify="right")
        comparison.add_column("median Spearman", justify="right")
        comparison.add_column("median pairwise", justify="right")
        eligibility = pilots.get("eligibility") or {}
        for name in sorted(set(eligibility) | set(pilots.get("comparison") or {})):
            values = (pilots.get("comparison") or {}).get(name, {})
            eligible = (eligibility.get(name) or {}).get("eligible", True)
            comparison.add_row(
                name,
                str(eligible),
                _fmt(values.get("median_mean_per_dataset_spearman"), 4),
                _fmt(values.get("median_pairwise_accuracy"), 4),
            )
        if not pilots:
            comparison.add_row("waiting", "-", "-", "-")
        selected = pilots.get("selected_initialization", "-")

        error_text = "\n".join(str(row.get("error", row))[:180] for row in errors) or "none"
        return Group(
            Panel(Group(title, subtitle), box=box.DOUBLE, border_style="bright_blue"),
            Panel(identity_table, title="Immutable identity", border_style="cyan"),
            Panel(losses, title="10 Hz loss windows", border_style="magenta"),
            Panel(
                epoch_history,
                title=(f"Full-history train vs {validation.get('split_name', '10% validation')}"),
                border_style="bright_magenta",
            ),
            Panel(learning, title="Throughput and timing", border_style="magenta"),
            Panel(
                quality,
                title="Validation, cold-start, and genome-only comparison",
                border_style="green",
            ),
            Panel(
                comparison,
                title=f"Initialization pilots · selected {selected}",
                border_style="yellow",
            ),
            Panel(
                system,
                title="1 Hz hardware and checkpoint state",
                border_style="blue",
            ),
            Panel(error_text, title="Recent errors", border_style="red"),
        )


def render_dime_score(run_dir: str | Path) -> Group:
    return DimeScoreView(run_dir).render()


def run_dime_score_tui(
    run_dir: str | Path,
    *,
    refresh: float = 0.1,
    once: bool = False,
) -> None:
    console = Console()
    view = DimeScoreView(run_dir)
    if once:
        console.print(view.render())
        return
    period = max(0.05, refresh)
    with Live(view.render(), console=console, screen=True, auto_refresh=False) as live:
        while True:
            live.update(view.render(), refresh=True)
            time.sleep(period)
