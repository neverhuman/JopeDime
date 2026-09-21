from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .status import artifact_freshness

try:  # system-load telemetry; degrade gracefully if unavailable
    import psutil
except Exception:  # pragma: no cover - psutil is a hard dep but keep the TUI resilient
    psutil = None  # type: ignore[assignment]

GENOME_PREVIEW = 96
CODE_PREVIEW_CHARS = 10_000
_HEAD_OPS = {
    "elasticnet",
    "ensemble",
    "ensemble_ridge_gbm",
    "gbm",
    "gpr",
    "histgbm",
    "histgbm2",
    "huber_ridge",
    "knn",
    "krr",
    "ksmooth",
    "logistic",
    "random_forest",
    "ridge",
    "stack",
}
_HEAT = " ▁▂▃▄▅▆▇█"


def run_tui(
    *,
    run_dir: str | Path,
    art_dir: str | Path,
    refresh: float = 1.0,
    once: bool = False,
    no_gpu: bool = False,
) -> None:
    run = RunView(Path(run_dir), Path(art_dir), no_gpu=no_gpu)
    if once:
        run.console.print(run.render())
        return
    period = max(0.25, 1.0 / max(0.2, refresh))
    with Live(run.render(), refresh_per_second=max(0.2, refresh), screen=True, auto_refresh=False) as live:
        while True:
            live.update(run.render(), refresh=True)
            time.sleep(period)


class RunView:
    def __init__(self, run_dir: Path, art_dir: Path, *, no_gpu: bool = False) -> None:
        from rich.console import Console
        from rich.layout import Layout

        self.console = Console()
        self.Layout = Layout
        self.run_dir = run_dir
        self.art_dir = art_dir
        self.no_gpu = no_gpu
        # Persistent sampling state across renders (needed for realtime deltas).
        self._proc_cache: dict[int, Any] = {}
        self._ckpt_cache: dict[str, Any] = {}
        self._prime_cpu()

    def _prime_cpu(self) -> None:
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None, percpu=True)
            except Exception:
                pass

    def render(self):
        layout = self.Layout(name="root")
        layout.split_column(
            self.Layout(name="header", size=4),
            self.Layout(name="main", ratio=1),
            self.Layout(name="footer", size=7),
        )
        layout["main"].split_row(
            self.Layout(name="left", ratio=2),
            self.Layout(name="center", ratio=3),
            self.Layout(name="right", ratio=3),
        )
        layout["left"].split_column(
            self.Layout(name="corpus"),
            self.Layout(name="vocab"),
        )
        layout["center"].split_column(
            self.Layout(name="learning", ratio=4),
            self.Layout(name="loss_curves", ratio=3),
            self.Layout(name="proposals", ratio=6),
        )
        layout["right"].split_column(
            self.Layout(name="quality", ratio=2),
            self.Layout(name="validity", ratio=2),
            self.Layout(name="code", ratio=8),
        )
        layout["footer"].split_row(
            self.Layout(name="system", ratio=5),
            self.Layout(name="gp", ratio=4),
            self.Layout(name="log", ratio=4),
        )
        state = self._state()
        layout["header"].update(self._header(state))
        layout["corpus"].update(self._corpus_panel(state))
        layout["vocab"].update(self._vocab_panel(state))
        layout["learning"].update(self._learning_panel(state))
        layout["loss_curves"].update(self._loss_curve_panel(state))
        layout["proposals"].update(self._gene_stream_panel(state))
        layout["quality"].update(self._quality_panel(state))
        layout["validity"].update(self._validity_panel(state))
        layout["code"].update(self._code_panel(state))
        if state.get("loop_metrics"):
            layout["gp"].update(self._loop_panel(state))
        else:
            layout["gp"].update(self._gp_panel(state))
        layout["system"].update(self._system_panel(state))
        layout["log"].update(self._log_panel(state))
        return layout

    def _state(self) -> dict[str, Any]:
        manifest = _read_json(self.art_dir / "manifest.json")
        vocab = _read_json(self.art_dir / "active_vocab.json")
        proposals = _read_jsonl(self.run_dir / "proposals.jsonl", limit=200)
        candidate_audit = _read_jsonl(self.run_dir / "candidate_audit.jsonl", limit=400)
        evaluations = _read_jsonl(self.run_dir / "evaluation.jsonl", limit=100)
        eval_summary = _read_json(self.run_dir / "eval_summary.json") or _read_json(
            self.run_dir / "summary.json"
        )
        log_tail = _tail_lines(self.run_dir / "run.log", 14)
        log_json = _json_objects_from_log(self.run_dir / "run.log")
        metrics = _read_jsonl(self.run_dir / "train_metrics.jsonl", limit=400)
        loop_metrics = _read_jsonl(self.run_dir / "loop_metrics.jsonl", limit=100)
        checkpoint = _first_existing(
            [
                self.art_dir / "checkpoints" / "active_model.pt",
                self.art_dir / "checkpoints" / "model.pt",
                self.art_dir / "checkpoints" / "smoke.pt",
            ]
        )
        ckpt = self._checkpoint_info(checkpoint)
        if not metrics and ckpt.get("loss_history"):
            metrics = [
                {"epoch": i + 1, "epochs": len(ckpt["loss_history"]), "loss": value}
                for i, value in enumerate(ckpt["loss_history"])
            ]
        build_summary = next((row for row in log_json if "manifest" in row), {})
        train_summary = next((row for row in reversed(log_json) if "loss_history" in row), {})
        return {
            "manifest": manifest,
            "vocab": vocab,
            "proposals": proposals,
            "candidate_audit": candidate_audit,
            "evaluations": evaluations,
            "eval_summary": eval_summary,
            "log_tail": log_tail,
            "build_summary": build_summary,
            "train_summary": train_summary,
            "metrics": metrics,
            "loop_metrics": loop_metrics,
            "checkpoint": checkpoint,
            "checkpoint_exists": checkpoint.exists(),
            "loss_weights": (
                metrics[-1].get("loss_weights")
                if metrics and isinstance(metrics[-1].get("loss_weights"), dict)
                else None
            )
            or ckpt.get("loss_weights")
            or {},
            "freshness": artifact_freshness(art_dir=self.art_dir, run_dir=self.run_dir),
            "metrics_age": _file_age(self.run_dir / "train_metrics.jsonl"),
            "proposals_age": _file_age(self.run_dir / "proposals.jsonl"),
            "candidate_audit_age": _file_age(self.run_dir / "candidate_audit.jsonl"),
            "system": self._system_state(),
            "gpus": [] if self.no_gpu else _gpu_states(),
            "run_dir": str(self.run_dir),
            "art_dir": str(self.art_dir),
            "now": time.time(),
        }

    # ---- header -------------------------------------------------------------
    def _header(self, state: dict[str, Any]) -> Panel:
        manifest = state["manifest"]
        metrics = state["metrics"]
        last = metrics[-1] if metrics else {}
        title = Text("JOPE ACTIVE JAIN-GENE TRAINER", style="bold bright_cyan")
        sub = Text()
        if last.get("distributed"):
            sub.append(f"{last.get('world_size', '?')} ranks", style="bright_magenta")
            sub.append("  |  ")
        sub.append(f"run {Path(state['run_dir']).name}", style="white")
        sub.append("  |  ")
        sub.append(f"{manifest.get('datasets_with_sidecar', '?')} datasets", style="bright_green")
        sub.append("  |  ")
        epoch = last.get("epoch")
        epochs = last.get("epochs")
        if epoch is not None:
            sub.append(f"epoch {epoch}/{epochs}", style="bright_yellow")
            sub.append("  |  ")
        state_word = "checkpoint ready" if state["checkpoint_exists"] else "training/building"
        sub.append(state_word, style="yellow")
        sub.append("  |  ")
        sub.append(f"metrics {_fmt_age(state['metrics_age'])}", style=_freshness_style(state["metrics_age"]))
        freshness = state.get("freshness") or {}
        if freshness.get("warning"):
            sub.append("  |  ")
            sub.append("stale artifacts", style="bold red")
        return Panel(Align.center(Group(title, sub)), box=box.DOUBLE, style="bright_blue")

    # ---- left column --------------------------------------------------------
    def _corpus_panel(self, state: dict[str, Any]) -> Panel:
        manifest = state["manifest"]
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        splits = manifest.get("splits") or {}
        table.add_row("candidate rows", _fmt_int(manifest.get("rows", 0)))
        table.add_row("dataset sidecars", _fmt_int(manifest.get("datasets_with_sidecar", 0)))
        table.add_row("train datasets", _fmt_int(splits.get("train", 0)))
        table.add_row("validation datasets", _fmt_int(splits.get("validation", 0)))
        table.add_row("target encoder", str(manifest.get("target_encoder", "none")))
        if manifest.get("target_sidecar"):
            table.add_row("target sidecar", str(manifest.get("target_sidecar")))
        table.add_row("artifacts", _dir_size(self.art_dir))
        return Panel(table, title="Corpus", border_style="green", box=box.ROUNDED)

    def _vocab_panel(self, state: dict[str, Any]) -> Panel:
        vocab = state["vocab"]
        transforms = vocab.get("transforms") or {}
        heads = vocab.get("heads") or {}
        table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("space")
        table.add_column("n", justify="right")
        table.add_column("examples")
        table.add_row("transforms", str(len(transforms)), ", ".join(list(transforms)[:5]))
        table.add_row("heads", str(len(heads)), ", ".join(heads))
        table.add_row(
            "transform counts",
            str(len(vocab.get("transform_counts") or [])),
            str(vocab.get("transform_counts") or []),
        )
        param_count = sum(len(p) for p in transforms.values()) + sum(len(p) for p in heads.values())
        table.add_row("observed params", str(param_count), "from canonical reports")
        return Panel(table, title="Active Vocabulary", border_style="cyan", box=box.ROUNDED)

    # ---- center column ------------------------------------------------------
    def _learning_panel(self, state: dict[str, Any]) -> Panel:
        metrics = state["metrics"]
        losses = _metric_values(metrics, "train_loss", fallback="loss")
        val_losses = _metric_values(metrics, "val_loss")
        last = metrics[-1] if metrics else {}

        progress = Progress(
            TextColumn("[bold]epoch[/]"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            expand=True,
        )
        total = int(last.get("epochs") or len(metrics) or 1)
        progress.add_task("epochs", total=max(1, total), completed=min(len(metrics), max(1, total)))

        # Headline: within-dataset relative-performance ranking (was frozen at chance).
        pw_series = _metric_values(metrics, "val_pairwise_win_accuracy")
        headline = Text()
        if pw_series:
            pw = pw_series[-1]
            headline.append("PAIRWISE WIN  ", style="bold")
            headline.append(f"{pw * 100:0.1f}%", style=_ranking_style(pw))
            headline.append(f"  {_trend_arrow(pw_series)}  ", style=_ranking_style(pw))
            headline.append("(chance 50%)  ", style="dim")
            headline.append(_spark(pw_series, width=28, label="pw"), style=_ranking_style(pw))
        else:
            headline.append("PAIRWISE WIN  waiting for validation metrics", style="dim")

        spark = _spark(losses, label="train")

        status = Table.grid(expand=True)
        status.add_column()
        status.add_column(justify="right")
        status.add_column(justify="right")
        status.add_row("[dim]metric[/]", "[dim]value[/]", "[dim]trend[/]")
        status.add_row(
            "train loss", _fmt_float(losses[-1]) if losses else "…", _spark(losses, width=14, label="")
        )
        status.add_row(
            "val loss",
            _fmt_float(val_losses[-1]) if val_losses else "…",
            _spark(val_losses, width=14, label=""),
        )
        self._ranking_rows(status, metrics, last)
        body = Group(progress, Text(spark, style="bold magenta"), headline, status)
        return Panel(
            body, title="Learning Signal · relative performance", border_style="magenta", box=box.ROUNDED
        )

    def _ranking_rows(self, status: Table, metrics: list[dict[str, Any]], last: dict[str, Any]) -> None:
        rbt = _metric_values(metrics, "train_loss_reward_bt")
        rlw = _metric_values(metrics, "train_loss_reward_listwise")
        regret = _metric_values(metrics, "val_top1_historical_regret")
        head_acc = _metric_values(metrics, "val_head_accuracy")
        grammar = _metric_values(metrics, "val_grammar_exact_match")
        vmae = _metric_values(metrics, "val_value_mae")
        status.add_row("reward BT loss", _fmt_float(rbt[-1]) if rbt else "-", _spark(rbt, width=14, label=""))
        if rlw:
            status.add_row("reward listwise", _fmt_float(rlw[-1]), _spark(rlw, width=14, label=""))
        status.add_row(
            "top-1 regret", _fmt_float(regret[-1]) if regret else "-", _spark(regret, width=14, label="")
        )
        status.add_row(
            "head accuracy", _fmt_pct(head_acc[-1]) if head_acc else "-", _spark(head_acc, width=14, label="")
        )
        status.add_row(
            "grammar exact", _fmt_pct(grammar[-1]) if grammar else "-", _spark(grammar, width=14, label="")
        )
        status.add_row("value MAE", _fmt_float(vmae[-1]) if vmae else "-", _spark(vmae, width=14, label=""))
        eps = last.get("examples_per_sec")
        if isinstance(eps, (int, float)):
            status.add_row("throughput", f"{eps:0.0f} ex/s", f"{_fmt_plain_float(last.get('epoch_sec'))}s/ep")

    def _loss_curve_panel(self, state: dict[str, Any]) -> Panel:
        metrics = state["metrics"]
        train_losses = _metric_values(metrics, "train_loss", fallback="loss")
        val_losses = _metric_values(metrics, "val_loss")
        weights = state.get("loss_weights") or {}
        lines = Text()
        lines.append(_spark(train_losses, label="train"), style="bold magenta")
        lines.append("\n")
        lines.append(_spark(val_losses, label="valid"), style="bold cyan")
        lines.append("\n")
        if train_losses and val_losses:
            gap = val_losses[-1] - train_losses[-1]
            lines.append(f"gen. gap: {_fmt_float(gap)}", style="yellow" if gap > 0 else "green")
        elif not val_losses:
            lines.append("validation loss: waiting", style="yellow")

        table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("component")
        table.add_column("train", justify="right")
        table.add_column("valid", justify="right")
        table.add_column("wt", justify="right")
        last = metrics[-1] if metrics else {}
        train_c = (
            last.get("train_loss_components") if isinstance(last.get("train_loss_components"), dict) else {}
        )
        val_c = last.get("val_loss_components") if isinstance(last.get("val_loss_components"), dict) else {}
        names = sorted(
            set(train_c) | set(val_c),
            key=lambda n: float(train_c.get(n) or 0.0) + float(val_c.get(n) or 0.0),
            reverse=True,
        )
        for name in names[:9]:
            wt = weights.get(name)
            style = "bright_green" if name.startswith("reward") else None
            table.add_row(
                Text(name, style=style) if style else name,
                _fmt_float(train_c.get(name)),
                _fmt_float(val_c.get(name)),
                f"{wt:0.2f}" if isinstance(wt, (int, float)) else "-",
            )
        if not names:
            table.add_row("-", "-", "-", "-")
        return Panel(
            Group(lines, table),
            title="Loss Curves · task-weighted",
            border_style="bright_magenta",
            box=box.ROUNDED,
        )

    def _proposal_panel(self, state: dict[str, Any]) -> Panel:
        proposals = state["proposals"]
        table = Table(show_header=True, header_style="bold bright_white", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("rk", justify="right", width=3)
        table.add_column("ok", justify="center", width=4)
        table.add_column("reward", justify="right", width=7)
        table.add_column("head", width=9, no_wrap=True, overflow="ellipsis")
        table.add_column("unc", justify="right", width=7)
        table.add_column("nvl", justify="center", width=3)
        table.add_column("genome", ratio=1, min_width=16, no_wrap=True, overflow="ellipsis")
        recent = proposals[-8:]
        for row in recent:
            genome = row.get("canonical_genome") or row.get("input_genome") or ""
            reward = row.get("reward")
            table.add_row(
                str(row.get("reward_rank")) if row.get("reward_rank") is not None else "-",
                "[green]ok[/]"
                if row.get("valid")
                else ("[yellow]fb[/]" if row.get("fallback") else "[red]bad[/]"),
                f"{float(reward):+0.3f}" if isinstance(reward, (int, float)) else "-",
                str(row.get("head_type") or _head_from_genome(genome) or "?"),
                _fmt_float(row.get("uncertainty")),
                "y" if row.get("novel") else "-",
                _truncate(genome, GENOME_PREVIEW),
            )
        if not recent:
            table.add_row("-", "wait", "-", "-", "-", "-", "proposals appear after checkpoint save")
        title = f"Candidate Proposals · reward-ranked ({len(proposals)} total, {_fmt_age(state['proposals_age'])})"
        return Panel(table, title=title, border_style="bright_blue", box=box.ROUNDED)

    def _gene_stream_panel(self, state: dict[str, Any]) -> Panel:
        entries = _gene_stream_entries(
            state.get("proposals") or [],
            state.get("candidate_audit") or [],
            state.get("evaluations") or [],
            limit=10,
        )
        table = Table(show_header=True, header_style="bold bright_white", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("src", width=5, no_wrap=True)
        table.add_column("slot", width=12, no_wrap=True, overflow="ellipsis")
        table.add_column("engine/head", width=16, no_wrap=True, overflow="ellipsis")
        table.add_column("reward", justify="right", width=8)
        table.add_column("score", justify="right", width=8)
        table.add_column("win", justify="right", width=7)
        table.add_column("risk", justify="right", width=7)
        table.add_column("cvΔ", justify="right", width=8)
        table.add_column("code", justify="right", width=7)
        table.add_column("genome", ratio=1, min_width=18, no_wrap=True, overflow="ellipsis")
        for source, row, eval_row in entries:
            genome = _row_genome(row)
            slot = str(row.get("selection_slot") or ("selected" if row.get("candidate_selected") else source))
            engine = str(row.get("engine") or "?")
            head = str(row.get("head_type") or _head_from_genome(genome) or "?")
            cv_delta = _eval_delta(eval_row) if eval_row else _eval_delta(row)
            risk = row.get("p_bad")
            if not isinstance(risk, (int, float)):
                risk = row.get("calibrated_tail_risk")
            code_tokens = _code_token_total(row)
            style = "bright_green" if source == "gene" or row.get("candidate_selected") else "dim"
            if row.get("valid") is False or row.get("candidate_risk_reason"):
                style = "yellow"
            table.add_row(
                Text(source, style=style),
                Text(_truncate(slot, 18), style=style),
                f"{_truncate(engine, 8)}/{_truncate(head, 8)}",
                _fmt_float(row.get("reward")),
                _fmt_float(row.get("selection_score")),
                _fmt_pct(row.get("calibrated_win_rate")),
                _fmt_float(risk),
                _fmt_float(cv_delta),
                _fmt_int(code_tokens) if code_tokens is not None else "-",
                _truncate(genome, GENOME_PREVIEW),
            )
        if not entries:
            table.add_row(
                "wait",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "candidate stream appears after proposal generation",
            )
        freshest = _freshest_age(state.get("proposals_age"), state.get("candidate_audit_age"))
        title = (
            "Live Gene Stream · score/eval/code "
            f"({len(state.get('candidate_audit') or [])} audit, {len(state.get('proposals') or [])} selected, {_fmt_age(freshest)})"
        )
        return Panel(table, title=title, border_style="bright_blue", box=box.ROUNDED)

    def _code_panel(self, state: dict[str, Any]) -> Panel:
        row = _latest_algorithm_row(state)
        if row is None:
            text = Text("waiting for a genome with generated source metadata", style="dim")
            return Panel(
                text,
                title="Algorithm Preview · Python / Rust",
                border_style="bright_magenta",
                box=box.ROUNDED,
            )
        roots = [self.run_dir, self.art_dir]
        python_code, python_source = _code_for_row(row, "python", roots)
        rust_code, rust_source = _code_for_row(row, "rust", roots)
        genome = _row_genome(row)
        engine = str(row.get("engine") or "?")
        head = str(row.get("head_type") or _head_from_genome(genome) or "?")
        token_count = _code_token_total(row)

        code_table = Table(
            show_header=True, header_style="bold bright_white", box=box.SIMPLE_HEAVY, expand=True
        )
        code_table.add_column(f"Python · {python_source}", ratio=1)
        code_table.add_column(f"Rust · {rust_source}", ratio=1)
        code_table.add_row(
            Syntax(python_code, "python", theme="monokai", line_numbers=False, word_wrap=True),
            Syntax(rust_code, "rust", theme="monokai", line_numbers=False, word_wrap=True),
        )
        title = (
            f"Algorithm Preview · {row.get('selection_slot') or '-'} · {engine}/{head} · "
            f"score {_fmt_float(row.get('selection_score'))} · tok {_fmt_int(token_count) if token_count is not None else '-'}"
        )
        return Panel(code_table, title=title, border_style="bright_magenta", box=box.ROUNDED)

    # ---- right column -------------------------------------------------------
    def _quality_panel(self, state: dict[str, Any]) -> Panel:
        summary = state["eval_summary"]
        evaluations = state["evaluations"]
        table = Table.grid(expand=True)
        table.add_column()
        table.add_column(justify="right")
        completed = summary.get("eval_completed_count", summary.get("evaluated_count", len(evaluations)))
        requested = summary.get("eval_requested_count")
        progress = (
            f"{_fmt_int(completed)}/{_fmt_int(requested)}" if requested is not None else _fmt_int(completed)
        )
        if summary.get("evaluation_in_progress"):
            progress += " [bright_yellow]running[/]"
        table.add_row("eval progress", progress)
        table.add_row(
            "win rate",
            Text(
                _fmt_pct(summary.get("win_rate_vs_incumbent")),
                style=_win_style(summary.get("win_rate_vs_incumbent")),
            ),
        )
        table.add_row("median delta", _fmt_float(summary.get("median_delta_vs_incumbent")))
        table.add_row("best-of-n median", _fmt_float(summary.get("best_of_n_median_delta")))
        table.add_row("reward corr", _fmt_float(summary.get("reward_actual_correlation")))
        table.add_row("novel evals", _fmt_int(summary.get("novel_eval_count", len(evaluations))))
        heads = summary.get("head_outcomes") if isinstance(summary.get("head_outcomes"), dict) else {}
        for head, outcome in sorted(
            heads.items(), key=lambda item: float((item[1] or {}).get("win_rate") or 0.0), reverse=True
        )[:4]:
            if not isinstance(outcome, dict):
                continue
            table.add_row(
                f"{head} wins",
                f"{_fmt_pct(outcome.get('win_rate'))} ({_fmt_int(outcome.get('wins'))}/{_fmt_int(outcome.get('count'))})",
            )
        freshness = state.get("freshness") or {}
        if freshness.get("warning"):
            table.add_row("freshness", Text(str(freshness["warning"]), style="bold red"))
        outliers = (
            summary.get("worst_reward_outliers")
            if isinstance(summary.get("worst_reward_outliers"), list)
            else []
        )
        if outliers:
            worst = outliers[0]
            table.add_row(
                "worst outlier",
                f"{worst.get('dataset_hash')} {worst.get('head_type')} Δ {_fmt_float(worst.get('oof_delta_vs_incumbent'))}",
            )
        return Panel(table, title="Quality · validation CV", border_style="bright_yellow", box=box.ROUNDED)

    def _validity_panel(self, state: dict[str, Any]) -> Panel:
        proposals = state["proposals"]
        evaluations = state["evaluations"]
        summary = state["eval_summary"]
        valid = sum(1 for r in proposals if r.get("valid"))
        fallback = sum(1 for r in proposals if r.get("fallback"))
        invalid = len(proposals) - valid
        table = Table.grid(expand=True)
        table.add_column()
        table.add_column(justify="right")
        table.add_row("proposal valid", f"[green]{valid}[/]")
        table.add_row("proposal invalid", f"[red]{invalid}[/]")
        table.add_row("fallbacks", f"[yellow]{fallback}[/]")
        completed = summary.get("eval_completed_count", summary.get("evaluated_count", len(evaluations)))
        requested = summary.get("eval_requested_count")
        progress = (
            f"{_fmt_int(completed)}/{_fmt_int(requested)}" if requested is not None else _fmt_int(completed)
        )
        if summary.get("evaluation_in_progress"):
            progress += " [bright_yellow]running[/]"
        table.add_row("CV eval progress", progress)
        table.add_row("novel evals", _fmt_int(summary.get("novel_eval_count", len(evaluations))))
        table.add_row("cache hits", _fmt_pct(summary.get("cache_hit_rate")))
        win = summary.get("win_rate_vs_incumbent")
        table.add_row("win vs incumbent", Text(_fmt_pct(win), style=_win_style(win)))
        table.add_row("median delta", _fmt_float(summary.get("median_delta_vs_incumbent")))
        table.add_row("best-of-n delta", _fmt_float(summary.get("best_of_n_median_delta")))
        table.add_row("rust authority", "[green]active[/]")
        table.add_row("symbolic/krr/stack", "[red]rejected[/]")
        errors = [err for r in proposals for err in (r.get("errors") or [])]
        if errors:
            table.add_row("latest error", _truncate(errors[-1], 42))
        return Panel(table, title="Validity Gate · vs incumbent", border_style="yellow", box=box.ROUNDED)

    def _gp_panel(self, state: dict[str, Any]) -> Panel:
        proposals = state["proposals"]
        evaluations = state["evaluations"]
        gpr_eval = [
            r
            for r in evaluations
            if r.get("head_type") == "gpr" or "(gpr" in str(r.get("canonical_genome") or "")
        ]
        lines = Text()
        heads = Counter(
            str(r.get("head_type") or _head_from_genome(r.get("canonical_genome") or "") or "?")
            for r in proposals
        )
        if heads:
            lines.append("head mix  ", style="bold")
            lines.append(", ".join(f"{h}:{c}" for h, c in heads.most_common(6)), style="cyan")
            lines.append("\n\n")
        if gpr_eval:
            for r in gpr_eval[-2:]:
                cv = r.get("cv") if isinstance(r.get("cv"), dict) else {}
                lines.append(
                    f"GPR CV {_fmt_float(cv.get('oof_r2'))} Δ {_fmt_float(r.get('oof_delta_vs_incumbent'))}\n",
                    style="green",
                )
        lines.append("uncertainty ridge (top proposals)\n", style="bold")
        top_unc = sorted(proposals, key=lambda r: float(r.get("uncertainty") or 0.0), reverse=True)[:5]
        for r in top_unc:
            unc = float(r.get("uncertainty") or 0.0)
            head = str(r.get("head_type") or _head_from_genome(r.get("canonical_genome") or "") or "?")
            lines.append(f"{_bar(unc, scale=1.0)} {unc:0.3f} {head}\n")
        rank_status = _latest_rank_status(state["metrics"])
        if rank_status:
            lines.append("\ncluster ranks\n", style="bold cyan")
            for r in rank_status[:8]:
                lines.append(
                    f"r{r.get('rank')} {r.get('hostname') or '?'}: util {_fmt_plain_float(r.get('nvidia_smi_gpu_utilization_pct'))}% "
                    f"mem {_fmt_plain_float(r.get('nvidia_smi_memory_used_mb'))}/{_fmt_plain_float(r.get('nvidia_smi_memory_total_mb'))}\n",
                    style="dim",
                )
        return Panel(
            lines, title="Heads / Uncertainty / Cluster", border_style="bright_cyan", box=box.ROUNDED
        )

    def _loop_panel(self, state: dict[str, Any]) -> Panel:
        rows = state.get("loop_metrics") or []
        last = rows[-1] if rows else {}
        lines = Text()
        lines.append(f"iteration {last.get('iteration', '?')}", style="bold bright_yellow")
        if last.get("policy_refits_halted"):
            lines.append("  POLICY REFITS HALTED", style="bold red")
        lines.append("\n")
        spear = [r.get("surrogate_vs_cv_spearman") for r in rows]
        spear = [s for s in spear if isinstance(s, (int, float))]
        if spear:
            lines.append(_spark(spear, width=24, label="surrogate-vs-cv"), style="bright_cyan")
            lines.append(f"  {spear[-1]:+.2f}\n", style="bright_cyan")
        per_engine = last.get("per_engine") or {}
        for engine, stats in sorted(per_engine.items()):
            style = "bright_green" if engine == "newmath" else "bright_blue"
            lines.append(f"{engine:9}", style=style)
            lines.append(
                f" sampled {int(stats.get('sampled', 0))}"
                f" uniq {int(stats.get('unique', 0))}"
                f" cv {int(stats.get('cv_completed', 0))}/{int(stats.get('cv_requested', 0))}"
                f" t/o {int(stats.get('timeouts', 0))}"
                f" blow {int(stats.get('blowups', 0))}\n",
                style="white",
            )
        best = [r.get("best_robust_score") for r in rows]
        best = [b for b in best if isinstance(b, (int, float))]
        if best:
            lines.append(_spark(best, width=24, label="best robust"), style="bright_magenta")
            lines.append(f"  {best[-1]:+.3f}\n", style="bright_magenta")
        lines.append(f"loop rows {int(last.get('corpus_loop_rows', 0))}", style="dim")
        refit = last.get("refit") or {}
        if refit:
            lines.append(
                f"  refit loss {refit.get('refit_loss', 0):.3f}"
                f" ({'policy' if refit.get('refit_policy') else 'reward'})",
                style="dim",
            )
        return Panel(lines, title="Closed Loop", border_style="bright_yellow", box=box.ROUNDED)

    # ---- footer: system load ------------------------------------------------
    def _system_panel(self, state: dict[str, Any]) -> Panel:
        sys = state["system"]
        gpus = state["gpus"]
        text = Text()
        if not sys:
            text.append("psutil unavailable — system telemetry off", style="dim")
            return Panel(text, title="System Load", border_style="red", box=box.ROUNDED)

        cpu = sys["cpu_percent"]
        text.append("CPU  ", style="bold")
        text.append(_meter(cpu, 100.0), style=_load_style(cpu))
        text.append(f" {cpu:5.1f}%  ", style=_load_style(cpu))
        text.append(f"{sys['cpu_count']} cores", style="dim")
        la = sys.get("loadavg")
        if la:
            text.append(f"   load {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}", style="dim")
        text.append("\n")
        percpu = sys.get("percpu") or []
        if percpu:
            text.append("core ", style="dim")
            for value in percpu:
                text.append(
                    _HEAT[min(len(_HEAT) - 1, int(value / 100.0 * (len(_HEAT) - 1)))],
                    style=_load_style(value),
                )
            text.append("\n")

        mem = sys["mem"]
        text.append("RAM  ", style="bold")
        text.append(_meter(mem["percent"], 100.0), style=_load_style(mem["percent"]))
        text.append(f" {mem['percent']:5.1f}%  ", style=_load_style(mem["percent"]))
        text.append(f"{mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB", style="dim")
        swap = sys.get("swap")
        if swap and swap["total_gb"] > 0:
            text.append(f"   swap {swap['used_gb']:.1f}/{swap['total_gb']:.1f} GB", style="dim")
        text.append("\n")

        disk = sys.get("disk")
        if disk:
            text.append("DISK ", style="bold")
            text.append(_meter(disk["percent"], 100.0), style=_load_style(disk["percent"]))
            text.append(f" {disk['percent']:5.1f}%  ", style=_load_style(disk["percent"]))
            text.append(f"{disk['free_gb']:.0f} GB free ({disk['mount']})", style="dim")
            text.append("\n")

        for g in gpus:
            util = float(g.get("util") or 0.0)
            mem_used = float(g.get("mem_used") or 0.0)
            mem_total = float(g.get("mem_total") or 0.0)
            text.append(f"GPU{g.get('index', 0)} ", style="bold bright_green")
            text.append(_meter(util, 100.0), style=_load_style(util))
            text.append(f" {util:5.1f}%  ", style=_load_style(util))
            memp = (mem_used / mem_total * 100.0) if mem_total else 0.0
            text.append(f"{mem_used / 1024:.1f}/{mem_total / 1024:.1f} GB ({memp:0.0f}%)", style="dim")
            if g.get("temp") is not None:
                text.append(f"  {g['temp']:0.0f}°C", style=_temp_style(g["temp"]))
            if g.get("power") is not None:
                lim = f"/{g['power_limit']:0.0f}" if g.get("power_limit") else ""
                text.append(f"  {g['power']:0.0f}{lim}W", style="dim")
            text.append("\n")
        if not gpus and not self.no_gpu:
            text.append("GPU  none detected\n", style="dim")

        procs = sys.get("top_procs") or []
        if procs:
            text.append("top  ", style="bold")
            text.append(
                "  ".join(
                    f"{p['name'][:12]}·{p['pid']} {p['cpu']:0.0f}%c {p['mem']:0.0f}%m" for p in procs[:4]
                ),
                style="dim",
            )
        return Panel(text, title="System Load · realtime", border_style="bright_green", box=box.ROUNDED)

    def _log_panel(self, state: dict[str, Any]) -> Panel:
        text = Text()
        for line in state["log_tail"]:
            text.append(_truncate(line, 200) + "\n", style="dim")
        if not state["log_tail"]:
            text.append("run.log empty — driver writes progress here\n", style="dim")
        return Panel(text, title="Run Log Tail", border_style="white", box=box.ROUNDED)

    # ---- sampling helpers ---------------------------------------------------
    def _system_state(self) -> dict[str, Any]:
        if psutil is None:
            return {}
        out: dict[str, Any] = {}
        try:
            out["cpu_percent"] = float(psutil.cpu_percent(interval=None))
            out["percpu"] = [float(v) for v in psutil.cpu_percent(interval=None, percpu=True)]
            out["cpu_count"] = psutil.cpu_count(logical=True) or len(out["percpu"])
        except Exception:
            out["cpu_percent"] = 0.0
            out["percpu"] = []
            out["cpu_count"] = 0
        try:
            out["loadavg"] = list(psutil.getloadavg())
        except Exception:
            out["loadavg"] = None
        try:
            vm = psutil.virtual_memory()
            out["mem"] = {
                "percent": float(vm.percent),
                "used_gb": vm.used / 2**30,
                "total_gb": vm.total / 2**30,
            }
        except Exception:
            out["mem"] = {"percent": 0.0, "used_gb": 0.0, "total_gb": 0.0}
        try:
            sw = psutil.swap_memory()
            out["swap"] = {
                "percent": float(sw.percent),
                "used_gb": sw.used / 2**30,
                "total_gb": sw.total / 2**30,
            }
        except Exception:
            out["swap"] = None
        try:
            target = self.run_dir if self.run_dir.exists() else Path(".")
            du = psutil.disk_usage(str(target))
            out["disk"] = {
                "percent": float(du.percent),
                "free_gb": du.free / 2**30,
                "mount": _mount_label(target),
            }
        except Exception:
            out["disk"] = None
        out["top_procs"] = self._top_processes()
        return out

    def _top_processes(self, limit: int = 6) -> list[dict[str, Any]]:
        if psutil is None:
            return []
        procs: list[dict[str, Any]] = []
        seen: set[int] = set()
        try:
            for proc in psutil.process_iter(["pid", "name", "memory_percent"]):
                pid = proc.info.get("pid")
                if pid is None:
                    continue
                seen.add(pid)
                cached = self._proc_cache.get(pid)
                if cached is None:
                    cached = proc
                    self._proc_cache[pid] = cached
                try:
                    cpu = cached.cpu_percent(interval=None)
                except Exception:
                    cpu = 0.0
                procs.append(
                    {
                        "pid": pid,
                        "name": str(proc.info.get("name") or "?"),
                        "cpu": float(cpu),
                        "mem": float(proc.info.get("memory_percent") or 0.0),
                    }
                )
        except Exception:
            return []
        for pid in list(self._proc_cache):
            if pid not in seen:
                self._proc_cache.pop(pid, None)
        procs.sort(key=lambda p: (p["cpu"], p["mem"]), reverse=True)
        return procs[:limit]

    def _checkpoint_info(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return {}
        key = str(path)
        cached = self._ckpt_cache.get(key)
        if cached and cached.get("mtime") == mtime:
            return cached
        info: dict[str, Any] = {"mtime": mtime, "loss_history": [], "loss_weights": {}}
        try:
            import torch

            state = torch.load(path, map_location="cpu")
        except Exception:
            self._ckpt_cache[key] = info
            return info
        if isinstance(state, dict):
            history = state.get("loss_history")
            if isinstance(history, list):
                info["loss_history"] = [float(v) for v in history if isinstance(v, (int, float))]
            model_sd = state.get("model") if isinstance(state.get("model"), dict) else {}
            for k, v in model_sd.items():
                if k.startswith("loss_log_vars."):
                    name = k.split(".", 1)[1]
                    try:
                        info["loss_weights"][name] = float(math.exp(-float(v)))
                    except Exception:
                        continue
        self._ckpt_cache[key] = info
        return info


# ---- module helpers ---------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-limit:]


def _json_objects_from_log(path: Path) -> list[dict[str, Any]]:
    return [row for row in _read_jsonl(path, limit=200) if isinstance(row, dict)]


def _tail_lines(path: Path, count: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-count:]


def _file_age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _mount_label(path: Path) -> str:
    try:
        dev = os.stat(path).st_dev
        for part in psutil.disk_partitions(all=False) if psutil else []:
            try:
                if os.stat(part.mountpoint).st_dev == dev:
                    return part.mountpoint
            except OSError:
                continue
    except Exception:
        pass
    return "/"


def _gpu_states() -> list[dict[str, Any]]:
    query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": _to_int(parts[0]),
                "name": parts[1],
                "util": _to_float(parts[2]),
                "mem_used": _to_float(parts[3]),
                "mem_total": _to_float(parts[4]),
                "temp": _to_float(parts[5]) if len(parts) > 5 else None,
                "power": _to_float(parts[6]) if len(parts) > 6 else None,
                "power_limit": _to_float(parts[7]) if len(parts) > 7 else None,
            }
        )
    return gpus


def _metric_values(rows: list[dict[str, Any]], key: str, *, fallback: str | None = None) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) and fallback is not None:
            value = row.get(fallback)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _latest_rank_status(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in reversed(rows):
        value = row.get("rank_status")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _spark(values: list[float], width: int = 52, *, label: str = "loss") -> str:
    prefix = f"{label}: " if label else ""
    if not values:
        return f"{prefix}waiting" if label else ""
    values = values[-width:]
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return prefix + "▁" * len(values)
    chars = "▁▂▃▄▅▆▇█"
    out = []
    for value in values:
        idx = int((value - lo) / (hi - lo) * (len(chars) - 1))
        out.append(chars[max(0, min(idx, len(chars) - 1))])
    return prefix + "".join(out)


def _trend_arrow(values: list[float]) -> str:
    if len(values) < 2:
        return "•"
    delta = values[-1] - values[max(0, len(values) - 4)]
    if delta > 1e-4:
        return "▲"
    if delta < -1e-4:
        return "▼"
    return "→"


def _meter(value: float, scale: float, width: int = 12) -> str:
    filled = max(0, min(width, int((value / max(scale, 1e-9)) * width)))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _bar(value: float, *, scale: float, width: int = 16) -> str:
    filled = max(0, min(width, int((value / max(scale, 1e-9)) * width)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _row_genome(row: dict[str, Any]) -> str:
    return str(row.get("canonical_genome") or row.get("input_genome") or row.get("genome") or "")


def _candidate_key(row: dict[str, Any]) -> tuple[str, str] | None:
    genome = _row_genome(row)
    if not genome:
        return None
    dataset = str(row.get("dataset_hash") or row.get("dataset_id") or "")
    return (dataset, genome)


def _evaluation_maps(
    evaluations: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_genome: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        key = _candidate_key(row)
        if key is not None:
            by_key[key] = row
            by_genome[key[1]] = row
    return by_key, by_genome


def _matching_eval(
    row: dict[str, Any],
    by_key: dict[tuple[str, str], dict[str, Any]],
    by_genome: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    key = _candidate_key(row)
    if key is not None and key in by_key:
        return by_key[key]
    genome = _row_genome(row)
    return by_genome.get(genome) if genome else None


def _gene_stream_entries(
    proposals: list[dict[str, Any]],
    candidate_audit: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    *,
    limit: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    by_key, by_genome = _evaluation_maps(evaluations)
    audit_rows = [row for row in candidate_audit[-max(3, limit // 2) :] if _row_genome(row)]
    proposal_rows = [row for row in proposals[-max(4, limit - len(audit_rows)) :] if _row_genome(row)]
    combined: list[tuple[str, dict[str, Any]]] = [("audit", row) for row in audit_rows] + [
        ("gene", row) for row in proposal_rows
    ]

    seen: set[tuple[str, str, str]] = set()
    entries: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    for source, row in combined:
        key = _candidate_key(row)
        dedupe = (source, key[0], key[1]) if key is not None else (source, "", _row_genome(row))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        entries.append((source, row, _matching_eval(row, by_key, by_genome)))
    return entries[-limit:]


def _eval_delta(row: dict[str, Any] | None) -> float | None:
    if not isinstance(row, dict):
        return None
    for key in (
        "oof_delta_vs_incumbent",
        "delta_vs_incumbent",
        "cv_delta_vs_incumbent",
        "median_delta_vs_incumbent",
        "actual_delta",
    ):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
    for key in ("oof_delta_vs_incumbent", "delta_vs_incumbent", "oof_delta"):
        value = cv.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _code_token_total(row: dict[str, Any]) -> int | None:
    tokens = row.get("code_tokens")
    if isinstance(tokens, dict):
        total = tokens.get("total")
        if isinstance(total, (int, float)):
            return int(total)
        subtotal = 0
        any_value = False
        for value in tokens.values():
            if isinstance(value, (int, float)):
                subtotal += int(value)
                any_value = True
        return subtotal if any_value else None
    if isinstance(tokens, (int, float)):
        return int(tokens)
    return None


def _freshest_age(*ages: float | None) -> float | None:
    present = [age for age in ages if isinstance(age, (int, float))]
    return min(present) if present else None


def _latest_algorithm_row(state: dict[str, Any]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for key in ("proposals", "candidate_audit", "evaluations"):
        rows.extend(reversed([row for row in state.get(key) or [] if isinstance(row, dict)]))
    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for row in rows:
        if not _row_genome(row):
            continue
        if row.get("valid") is False or row.get("candidate_selected") is False:
            fallback.append(row)
            continue
        preferred.append(row)
    return (preferred or fallback or [None])[0]


def _code_for_row(row: dict[str, Any], lang: str, roots: list[Path]) -> tuple[str, str]:
    inline = _inline_code(row, lang)
    if inline:
        return (_clip_code(inline), "inline")
    path = _code_path(row, lang, roots)
    if path is not None:
        code = _read_code(path)
        if code:
            return (_clip_code(code), _truncate(str(path), 34))
    if lang == "rust":
        return (_rust_algorithm_preview(row), "preview")
    return (_python_algorithm_preview(row), "preview")


def _inline_code(row: dict[str, Any], lang: str) -> str | None:
    generated = row.get("generated_code")
    if isinstance(generated, dict):
        value = generated.get(lang) or generated.get(f"{lang}_code")
        if isinstance(value, str) and value.strip():
            return value
    for key in (f"{lang}_code", f"{lang}_source", f"generated_{lang}"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    code = row.get("code")
    if isinstance(code, dict):
        value = code.get(lang) or code.get(f"{lang}_code")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _code_path(row: dict[str, Any], lang: str, roots: list[Path]) -> Path | None:
    values: list[Any] = []
    for key in (f"{lang}_path", f"{lang}_source_path", f"{lang}_code_path"):
        values.append(row.get(key))
    for container_key in ("code_paths", "source_paths", "artifact_paths", "artifacts"):
        container = row.get(container_key)
        if isinstance(container, dict):
            values.extend(
                [container.get(lang), container.get(f"{lang}_path"), container.get(f"{lang}_source_path")]
            )
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        candidates = [path] if path.is_absolute() else [root / path for root in roots] + [path]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _read_code(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text[:CODE_PREVIEW_CHARS]


def _clip_code(code: str) -> str:
    code = code.strip("\n")
    if len(code) <= CODE_PREVIEW_CHARS:
        return code
    return code[:CODE_PREVIEW_CHARS] + "\n# ... clipped ..."


def _genome_stages(genome: str) -> list[tuple[str, dict[str, str]]]:
    stages: list[tuple[str, dict[str, str]]] = []
    for match in re.finditer(r"\(([A-Za-z_][A-Za-z0-9_\-]*)([^()]*)\)", genome):
        op = match.group(1)
        if op in {"pipe", "sym"}:
            continue
        stages.append((op, _stage_params(match.group(2))))
    return stages


def _stage_params(raw: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for idx, token in enumerate(str(raw).strip().split()):
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            params[key.strip()] = value.strip()
        else:
            params[f"arg{idx}"] = token.strip('"')
    return params


def _split_transforms_head(
    row: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, str]]], tuple[str, dict[str, str]]]:
    stages = _genome_stages(_row_genome(row))
    if not stages:
        head = str(row.get("head_type") or "ridge")
        return [], (head, {})
    last_op, last_params = stages[-1]
    if last_op in _HEAD_OPS:
        return stages[:-1], (last_op, last_params)
    head = str(row.get("head_type") or _head_from_genome(_row_genome(row)) or "ridge")
    return stages, (head, {})


def _python_algorithm_preview(row: dict[str, Any]) -> str:
    transforms, head = _split_transforms_head(row)
    lines = [
        "class Model:",
        "    def fit(self, X, y):",
        "        Z = X",
    ]
    if not transforms:
        lines.append("        # raw features")
    for op, params in transforms[:5]:
        lines.append(f"        Z = apply_stage(Z, {op!r}, {params!r}, y=y)")
    if len(transforms) > 5:
        lines.append(f"        # ... {len(transforms) - 5} more transform stages ...")
    lines.extend(
        [
            f"        self.head = fit_head({head[0]!r}, Z, y, {head[1]!r})",
            "        return self",
            "",
            "    def predict(self, X):",
            "        Z = X",
        ]
    )
    for index, (op, _params) in enumerate(transforms[:5]):
        lines.append(f"        Z = self.transforms[{index}].transform(Z)  # {op}")
    if len(transforms) > 5:
        lines.append("        # remaining fitted transforms replay here")
    lines.append("        return self.head.predict(Z)")
    return "\n".join(lines)


def _rust_algorithm_preview(row: dict[str, Any]) -> str:
    transforms, head = _split_transforms_head(row)
    lines = [
        "pub struct Model {",
        "    transforms: Vec<FittedStage>,",
        "    head: FittedHead,",
        "}",
        "",
        "impl Model {",
        "    pub fn fit(x: &Matrix, y: &[f64]) -> Self {",
        "        let mut z = x.clone();",
        "        let mut transforms = Vec::new();",
    ]
    for op, params in transforms[:4]:
        lines.append(
            "        z = fit_stage("
            f"&mut transforms, z, {json.dumps(op)}, {json.dumps(params, sort_keys=True)}, Some(y));"
        )
    if len(transforms) > 4:
        lines.append(f"        // ... {len(transforms) - 4} more transform stages ...")
    lines.extend(
        [
            "        let head = fit_head("
            f"{json.dumps(head[0])}, &z, y, {json.dumps(head[1], sort_keys=True)});",
            "        Self { transforms, head }",
            "    }",
            "",
            "    pub fn predict(&self, x: &Matrix) -> Vec<f64> {",
            "        let mut z = x.clone();",
        ]
    )
    for index, (op, _params) in enumerate(transforms[:4]):
        lines.append(f"        z = self.transforms[{index}].transform(z); // {op}")
    if len(transforms) > 4:
        lines.append("        // remaining fitted transforms replay here")
    lines.extend(["        self.head.predict(&z)", "    }", "}"])
    return "\n".join(lines)


def _head_from_genome(genome: str) -> str | None:
    if not genome:
        return None
    for head in sorted(_HEAD_OPS, key=len, reverse=True):
        if f"({head}" in genome:
            return head
    return None


def _load_style(pct: float) -> str:
    if pct >= 90:
        return "bold red"
    if pct >= 70:
        return "yellow"
    if pct >= 40:
        return "green"
    return "bright_green"


def _temp_style(temp: float) -> str:
    if temp >= 85:
        return "bold red"
    if temp >= 70:
        return "yellow"
    return "green"


def _ranking_style(pw: float) -> str:
    if pw >= 0.55:
        return "bold bright_green"
    if pw >= 0.52:
        return "green"
    if pw >= 0.5:
        return "yellow"
    return "red"


def _win_style(win: Any) -> str:
    if not isinstance(win, (int, float)):
        return "dim"
    if win > 0.15:
        return "bold bright_green"
    if win > 0.0:
        return "green"
    return "red"


def _freshness_style(age: float | None) -> str:
    if age is None:
        return "dim"
    if age < 15:
        return "bright_green"
    if age < 120:
        return "yellow"
    return "red"


def _fmt_age(age: float | None) -> str:
    if age is None:
        return "—"
    if age < 90:
        return f"{age:0.0f}s ago"
    if age < 5400:
        return f"{age / 60:0.0f}m ago"
    return f"{age / 3600:0.1f}h ago"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "?"


def _fmt_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):+.4f}"


def _fmt_plain_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):0.1f}"


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value) * 100.0:0.1f}%"


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _truncate(value: str, width: int) -> str:
    text = str(value).replace("\n", " ")
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def _dir_size(path: Path) -> str:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:0.1f} {unit}"
        total /= 1024
    return f"{total:0.1f} TB"
