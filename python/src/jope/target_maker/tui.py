from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def _read_tail(path: Path, limit: int = 200) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: deque[dict[str, object]] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(rows)


def _nvidia_smi() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, timeout=2)
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append({"idx": parts[0], "name": parts[1], "util": parts[2], "used": parts[3], "total": parts[4]})
    return rows


def _latest(rows: Iterable[dict[str, object]], phase: str | None = None) -> dict[str, object]:
    chosen: dict[str, object] = {}
    for row in rows:
        if phase is None or row.get("phase") == phase:
            chosen = row
    return chosen


def _format_float(value: object, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "-"


def render_dashboard(run_dir: Path) -> Group:
    rows = _read_tail(run_dir / "metrics.jsonl")
    train = _latest(rows, "train")
    val = _latest(rows, "val")
    last = rows[-1] if rows else {}

    gpu_table = Table(expand=True)
    gpu_table.add_column("GPU", justify="right")
    gpu_table.add_column("Util", justify="right")
    gpu_table.add_column("Memory", justify="right")
    for gpu in _nvidia_smi():
        gpu_table.add_row(gpu["idx"], f'{gpu["util"]}%', f'{gpu["used"]}/{gpu["total"]} MiB')
    if gpu_table.row_count == 0:
        gpu_table.add_row("-", "-", "-")

    loss_table = Table(expand=True)
    loss_table.add_column("Metric")
    loss_table.add_column("Train", justify="right")
    loss_table.add_column("Validation", justify="right")
    for key in ["loss", "instance", "contrastive", "profile", "task", "collapse"]:
        loss_table.add_row(key, _format_float(train.get(key)), _format_float(val.get(key)))

    metrics_table = Table(expand=True)
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")
    for key in [
        "retrieval@1",
        "retrieval@10",
        "group_retrieval@1",
        "profile_mae",
        "binary_rate_mae",
        "regression_quantile_mae",
        "embedding_rank",
    ]:
        metrics_table.add_row(key, _format_float(val.get(key)))

    speed_table = Table(expand=True)
    speed_table.add_column("Field")
    speed_table.add_column("Value", justify="right")
    samples = train.get("samples_per_sec")
    speed_table.add_row("samples/sec", _format_float(samples, 2))
    speed_table.add_row("datasets/sec", _format_float(train.get("datasets_per_sec"), 2))
    speed_table.add_row("epoch", str(last.get("epoch", "-")))
    speed_table.add_row("step", str(last.get("step", "-")))
    speed_table.add_row("checkpoint", str(last.get("checkpoint") or last.get("best_checkpoint") or "-"))
    speed_table.add_row("updated", time.strftime("%Y-%m-%d %H:%M:%S"))

    return Group(
        Panel(gpu_table, title="GPU"),
        Panel(speed_table, title="Run"),
        Panel(loss_table, title="Loss"),
        Panel(metrics_table, title="Validation"),
    )


def watch(run_dir: Path, refresh_seconds: float = 2.0) -> None:
    run_dir = Path(run_dir).expanduser()
    with Live(render_dashboard(run_dir), refresh_per_second=max(1.0, 1.0 / refresh_seconds), screen=True) as live:
        while True:
            live.update(render_dashboard(run_dir))
            time.sleep(refresh_seconds)
