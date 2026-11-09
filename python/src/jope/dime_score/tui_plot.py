from __future__ import annotations

import math
from typing import Any

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text


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
