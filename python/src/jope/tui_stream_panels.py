"""Training, self-play, and cluster panels for the gene stream TUI."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .tui_stream import _bar, _sparkline, _tail_jsonl


class StreamPanelsMixin:
    def _footer(self) -> Any:
        from rich.layout import Layout

        row = Layout()
        panels = [Layout(self._training_panel(), ratio=1)]
        sp = self._self_play_panel()
        if sp is not None:
            panels.append(Layout(sp, ratio=1))
        panels.append(Layout(self._cluster_panel(), ratio=1))
        row.split_row(*panels)
        return row

    def _self_play_panel(self) -> Panel | None:
        """Live self-play view: generation + phase, learning-curve sparklines, gate verdicts.

        Tails self_play_state.json (atomic, phase-granular) + self_play_metrics.jsonl
        (one row per finished generation). Hidden when no self-play run exists.
        """
        state_p = self.run_dir / "self_play_state.json"
        metrics_p = self.run_dir / "self_play_metrics.jsonl"
        if not state_p.exists() and not metrics_p.exists():
            return None
        try:
            state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}
        except (OSError, json.JSONDecodeError):
            state = {}
        gens = _tail_jsonl(metrics_p, 64) if metrics_p.exists() else []
        body = Text()
        gen = state.get("generation", "?")
        phase = str(state.get("phase", "?"))
        phases = ["TRAIN", "PROPOSE", "EVAL", "GATE", "BENCH", "DONE"]
        line = Text(f" gen {gen}  ", style="bold white")
        for p in phases:
            if p == phase:
                line.append(f"▸{p} ", style="bold bright_green")
            else:
                line.append(f"{p.lower()} ", style="dim")
        # phase age + staleness dot: silence shows as AGE, never as an ambiguous freeze
        now = time.time()
        started = state.get("phase_started")
        if isinstance(started, (int, float)):
            age = int(now - started)
            line.append(f"· {age // 60}m{age % 60:02d}s ", style="dim")
        try:
            # liveness = freshest of state / per-batch train progress / miner progress
            candidates = [state_p, self.run_dir / "train_progress.json", self.run_dir / "miner_progress.json"]
            newest = max((p.stat().st_mtime for p in candidates if p.exists()), default=0.0)
            mtime_age = now - newest
            dot_style = "bold green" if mtime_age < 20 else ("bold yellow" if mtime_age < 90 else "bold red")
            line.append("●", style=dot_style)
        except OSError:
            pass
        body.append(line)
        body.append("\n")
        done, total = state.get("phase_done"), state.get("phase_total")
        if isinstance(done, int) and isinstance(total, int) and total > 0:
            body.append(" ")
            body.append(_bar(100.0 * done / total, 100.0, 16, "bright_green"))
            body.append(f" {done}/{total}", style="dim")
            eps = state.get("evals_per_sec")
            if isinstance(eps, (int, float)):
                body.append(f"  {eps:.0f}/s", style="cyan")
            # live phase ETA from the observed rate since phase start
            if isinstance(started, (int, float)) and done > 0 and now > started:
                rate = done / (now - started)
                if rate > 0 and done < total:
                    eta = int((total - done) / rate)
                    body.append(f"  ETA {eta // 60}m{eta % 60:02d}s", style="bold white")
            body.append("\n")
        # generation-level ETA: remaining phases estimated from the last completed gen
        if gens and isinstance(started, (int, float)):
            durs = gens[-1]
            phase_costs = {
                "TRAIN": durs.get("train_sec"),
                "PROPOSE": durs.get("propose_sec"),
                "EVAL": durs.get("eval_sec"),
                "BENCH": max(
                    (durs.get("total_sec") or 0)
                    - sum(durs.get(k) or 0 for k in ("train_sec", "propose_sec", "eval_sec")),
                    0,
                ),
                "GATE": 5.0,
                "DONE": 0.0,
            }
            order = ["TRAIN", "PROPOSE", "EVAL", "GATE", "BENCH", "DONE"]
            if phase in order and all(isinstance(v, (int, float)) for v in phase_costs.values()):
                idx = order.index(phase)
                cur_cost = float(phase_costs.get(phase) or 0.0)
                cur_left = max(cur_cost - (now - started), 0.0)
                rest = sum(float(phase_costs[p] or 0.0) for p in order[idx + 1 :])
                gen_eta = int(cur_left + rest)
                body.append(f" gen ETA ~{gen_eta // 60}m{gen_eta % 60:02d}s", style="bold magenta")
                body.append(f"  (last gen {int((durs.get('total_sec') or 0) // 60)}m)\n", style="dim")
        # concurrent champion miner: live raced/total + champions found this slice
        miner_p = self.run_dir / "miner_progress.json"
        if state.get("miner_active") and miner_p.exists():
            try:
                mp_ = json.loads(miner_p.read_text(encoding="utf-8"))
                body.append(" ⛏ ", style="yellow")
                body.append(
                    _bar(100.0 * mp_.get("raced", 0) / max(mp_.get("total", 1), 1), 100.0, 12, "yellow")
                )
                body.append(f" {mp_.get('raced', 0)}/{mp_.get('total', '?')} mutants", style="dim")
                body.append(f"  +{mp_.get('champions', 0)} champions\n", style="bold yellow")
            except (OSError, json.JSONDecodeError, ZeroDivisionError):
                pass
        if gens:
            last = gens[-1]

            # deployed capability = cheap-probe selector (elo_probe); elo_top1 is the
            # reward-pick baseline, kept as a diagnostic. Fall back for pre-cutover rows.
            def _dep(g):
                for key in ("elo_deployed", "elo_probe", "elo_cv", "elo_top1"):
                    v = g.get(key)
                    if isinstance(v, (int, float)):
                        return v
                return None

            elos = [_dep(g) for g in gens if isinstance(_dep(g), (int, float))]
            beats = [
                g["beat_rate_vs_prev"] for g in gens if isinstance(g.get("beat_rate_vs_prev"), (int, float))
            ]
            if elos:
                body.append(" Elo↥ ", style="dim")
                body.append(_sparkline(elos[-32:]), style="magenta")
                body.append(f" {elos[-1]:.0f}", style="bold magenta")
                rk = last.get("elo_probe_rank") or last.get("elo_rank")
                if rk:
                    body.append(f" #{rk}", style="magenta")
                if len(elos) >= 2:
                    d = elos[-1] - elos[0]
                    body.append(f" ({'+' if d >= 0 else ''}{d:.0f})", style="green" if d > 0 else "red")
                body.append("\n")
            # selection ladder: reward-pick top1 → cheap-probe (deployed) → best-K ceiling
            t1, pr, ck = last.get("elo_top1"), last.get("elo_probe"), last.get("elo_best_k")
            if isinstance(pr, (int, float)) and isinstance(t1, (int, float)):
                body.append(" sel  ", style="dim")
                body.append(f"top1 {t1:.0f}", style="dim")
                body.append(" → ", style="dim")
                body.append(f"probe {pr:.0f}", style="bold green")
                body.append(f" (+{pr - t1:.0f})", style="green")
                if isinstance(ck, (int, float)):
                    body.append(f"  ceil {ck:.0f}", style="dim cyan")
                body.append("\n")
            if beats:
                body.append(" beat ", style="dim")
                body.append(_sparkline(beats[-32:]), style="cyan")
                verdict = last.get("shipped")
                body.append(f" {beats[-1]:.2f} ", style="cyan")
                if verdict is not None:
                    body.append(
                        "✓shipped" if verdict else "✗held", style="bold green" if verdict else "bold red"
                    )
                body.append("\n")
            nb = last.get("new_bests")
            champs = last.get("miner_new_champions")
            info = Text(" ")
            if nb is not None:
                info.append(f"new_bests {nb} ", style="green")
            if champs is not None:
                info.append(f"⛏champions {champs} ", style="yellow")
            if last.get("mean_rank_pct") is not None:
                info.append(f"rank_pct {last['mean_rank_pct']:.2f}", style="dim")
            body.append(info)
        else:
            body.append(" first generation running…", style="dim")
        return Panel(
            body, title="self-play · beat all former selves", border_style="magenta", box=box.ROUNDED
        )

    def _read_progress(self) -> dict | None:
        """Read the atomic per-batch progress file (sibling of the metrics jsonl)."""
        if not self.train_metrics_path:
            return None
        p = self.train_metrics_path.parent / "train_progress.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _training_panel(self) -> Panel:
        if not self._train_metrics_pinned:
            self.train_metrics_path = self._auto_train_metrics()
        rows = _tail_jsonl(self.train_metrics_path, 200) if self.train_metrics_path else []
        prog = self._read_progress()
        if not rows and not prog:
            return Panel(
                Text("no active training run\n(point --train-metrics at a metrics.jsonl)", style="dim"),
                title="training loss",
                border_style="cyan",
                box=box.ROUNDED,
            )
        losses = [float(r["loss"]) for r in rows if isinstance(r.get("loss"), (int, float))]
        last = rows[-1] if rows else {}
        body = Text()
        ep = (prog or last).get("epoch", "?")
        eps = (prog or last).get("epochs", "?")
        phase = (prog or last).get("phase", "")
        body.append(f" epoch {ep}/{eps}", style="bold white")
        if phase:
            body.append(f"  phase {phase}", style="magenta")
        if losses:
            body.append(f"   loss {losses[-1]:.4f}", style="bold cyan")
        body.append("\n")
        # Part D: live in-epoch % progress bar + running batch loss (updates <100ms)
        if prog and not prog.get("epoch_done"):
            body.append(" ")
            body.append(_bar(float(prog.get("pct", 0.0)), 100.0, 18, "bright_green"))
            body.append(f"  batch {prog.get('batch', '?')}/{prog.get('total_batches', '?')}", style="white")
            rl = prog.get("running_loss")
            if isinstance(rl, (int, float)):
                body.append(f"  running {rl:.3f}", style="cyan")
            body.append("\n")
        # train vs val loss + overfit flag (red when val exceeds train and the gap is growing).
        # Use the latest completed-epoch metrics for val (progress mid-epoch may not have it yet).
        train_now = (
            (prog or {}).get("running_loss")
            if (prog and not prog.get("epoch_done"))
            else last.get("train_loss")
        )
        vl = (
            (prog or {}).get("val_loss") if (prog or {}).get("val_loss") is not None else last.get("val_loss")
        )
        gap = (
            (prog or {}).get("overfit_gap")
            if (prog or {}).get("overfit_gap") is not None
            else last.get("overfit_gap")
        )
        if isinstance(vl, (int, float)):
            gaps = [float(r["overfit_gap"]) for r in rows if isinstance(r.get("overfit_gap"), (int, float))]
            growing = len(gaps) >= 2 and gaps[-1] > gaps[-2]
            overfit = isinstance(gap, (int, float)) and gap > 0 and growing
            body.append(" train ", style="dim")
            if isinstance(train_now, (int, float)):
                body.append(f"{train_now:.3f}", style="cyan")
            body.append("  val ", style="dim")
            body.append(f"{vl:.3f}", style="red" if overfit else "green")
            if isinstance(gap, (int, float)):
                body.append(f"  gap {gap:+.3f}", style="red" if overfit else "dim")
                if overfit:
                    body.append(" ⚠OVERFIT", style="bold red")
            body.append("\n")
        pw = last.get("val_pairwise_win_accuracy")
        gr = last.get("val_one_shot_greedy_valid_rate")
        stab = (last.get("train_loss_components") or {}).get("stability_bce")
        line = Text()
        if isinstance(pw, (int, float)):
            line.append(f" pairwise_win {pw:.4f} ", style="green" if pw > 0.55 else "yellow")
            # Elo derived from the win probability vs a 1500 incumbent baseline (logistic
            # scale): rises live as pairwise_win climbs. 0.5 win rate == 1500.
            p = min(max(float(pw), 1e-4), 1 - 1e-4)
            elo = 1500 + 400 * math.log10(p / (1 - p))
            line.append(f"↦ Elo {elo:4.0f} ", style="bold magenta")
        if isinstance(gr, (int, float)):
            line.append(f" greedy_valid {gr:.2f} ", style="green" if gr > 0.99 else "yellow")
        if isinstance(stab, (int, float)):
            line.append(f" stab {stab:.3f} ", style="dim")
        body.append(line)
        body.append("\n")
        # Elo trajectory sparkline across the run (relative-performance headline)
        pws = [
            float(r["val_pairwise_win_accuracy"])
            for r in rows
            if isinstance(r.get("val_pairwise_win_accuracy"), (int, float))
        ]
        if len(pws) >= 2:
            elos = [
                1500 + 400 * math.log10(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
                for p in pws
            ]
            body.append("loss ", style="dim")
            body.append(_sparkline(losses[-64:]), style="cyan")
            body.append(f" {losses[-1]:.3f}\n", style="dim")
            body.append("Elo  ", style="dim")
            body.append(_sparkline(elos[-64:]), style="magenta")
            body.append(f" {elos[-1]:.0f}", style="dim")
        elif len(losses) >= 2:
            body.append(_sparkline(losses[-64:]), style="cyan")
            body.append(f"  min {min(losses):.4f}", style="dim")
        return Panel(
            body,
            title=f"training loss · {self.train_metrics_path.name}",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def _cluster_panel(self) -> Panel:
        if self.cluster is None:
            return Panel(
                Text("cluster monitor off", style="dim"),
                title="cluster",
                border_style="blue",
                box=box.ROUNDED,
            )
        table = Table(box=box.SIMPLE, expand=True, pad_edge=False, show_edge=False)
        table.add_column("host", width=7)
        table.add_column("GPU", ratio=1)
        table.add_column("mem", width=11, justify="right")
        table.add_column("°C", width=4, justify="right")
        table.add_column("CPU", ratio=1)
        table.add_column("lanes", width=5, justify="right")
        for st in self.cluster.snapshot():
            if not st.ok:
                table.add_row(st.host[:7], Text("unreachable", style="red"), "", "", "", "")
                continue
            g = st.gpus[0] if st.gpus else {}
            util = g.get("util") or 0.0
            gpu_bar = _bar(util, 100.0, 12, "green" if util < 70 else "yellow" if util < 90 else "red")
            mem_u, mem_t = g.get("mem_used") or 0.0, g.get("mem_total") or 1.0
            mem = f"{mem_u / 1024:.1f}/{mem_t / 1024:.0f}G"
            temp = f"{int(g.get('temp') or 0)}"
            load = st.load1 or 0.0
            cores = st.cores or 1
            cpu_pct = min(100.0, 100.0 * load / cores)
            cpu_bar = _bar(
                cpu_pct, 100.0, 12, "green" if cpu_pct < 70 else "yellow" if cpu_pct < 90 else "red"
            )
            lanes = str(st.lanes if st.lanes is not None else "?")
            host_style = "bold cyan" if st.host == "local" else "bold"
            table.add_row(
                Text(st.host[:7], style=host_style),
                gpu_bar,
                mem,
                temp,
                Text.assemble(cpu_bar, f" {load:.0f}"),
                lanes,
            )
        return Panel(table, title="cluster · GPU / cores / eval lanes", border_style="blue", box=box.ROUNDED)
