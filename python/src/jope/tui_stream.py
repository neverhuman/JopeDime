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
import math
import os
import select
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .engines import EngineRouter
from .tui import _truncate

CANON_PREVIEW = 60
CODE_PREVIEW_CHARS = 20_000
MAX_GENES = 2000
INITIAL_TAIL_BYTES = 2 * 1024 * 1024


class JsonlTailer:
    """Yield only rows appended since the last poll; reset on rotation (inode/size)."""

    def __init__(self, path: str | Path, *, initial_tail_bytes: int | None = INITIAL_TAIL_BYTES) -> None:
        self.path = Path(path)
        self._offset = 0
        self._inode: int | None = None
        self._buf = ""
        self._initial_tail_bytes = initial_tail_bytes

    def poll(self) -> list[dict[str, Any]]:
        try:
            st = self.path.stat()
        except OSError:
            return []
        # rotation / truncation → reopen from the start
        reset = self._inode is not None and (st.st_ino != self._inode or st.st_size < self._offset)
        if reset:
            self._offset = 0
            self._buf = ""
        if (self._inode is None or reset) and self._initial_tail_bytes is not None:
            self._offset = max(0, st.st_size - max(0, self._initial_tail_bytes))
        self._inode = st.st_ino
        if st.st_size <= self._offset:
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        self._buf += chunk
        rows: list[dict[str, Any]] = []
        *complete, self._buf = self._buf.split("\n")
        for line in complete:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


@dataclass
class Gene:
    gene_id: str
    engine: str = ""
    canonical: str = ""
    head: str | None = None
    mu: float | None = None
    sigma: float | None = None
    p_bad: float | None = None
    oof: float | None = None
    test_r2: float | None = None
    state: str = "proposed"  # proposed -> selected -> scored
    total_ms: float | None = None
    seq: int = 0


@dataclass
class StreamState:
    cursor: int = 0
    lang: str = "python"  # or "rust"
    follow: bool = True
    quit: bool = False
    save: bool = False
    order: list[str] = field(default_factory=list)  # gene_ids in arrival order


def handle_key(state: StreamState, key: str, n_genes: int) -> StreamState:
    """Pure key dispatch (no I/O) so the input model is testable without a tty."""
    s = replace(state, save=False)
    if key in ("q", "\x03"):  # q or Ctrl-C
        s.quit = True
    elif key in ("j", "DOWN"):
        s.follow = False
        s.cursor = min(max(0, n_genes - 1), s.cursor + 1)
    elif key in ("k", "UP"):
        s.follow = False
        s.cursor = max(0, s.cursor - 1)
    elif key == "p":
        s.lang = "python"
    elif key == "r":
        s.lang = "rust"
    elif key == "l":
        s.follow = not s.follow
    elif key == "s":
        s.save = True
    return s


class StreamView:
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

    def _auto_train_metrics(self) -> Path | None:
        # self-play runs write per-generation metrics: follow the LATEST generation live
        gen_files = sorted(
            self.run_dir.glob("gen*_train_metrics.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        if gen_files:
            return gen_files[-1]
        for name in ("train_metrics.jsonl", "policy_metrics.jsonl", "stream_metrics.jsonl", "metrics.jsonl"):
            p = self.run_dir / name
            if p.exists():
                return p
        return None

    # -- data ingestion ------------------------------------------------------
    def ingest(self) -> None:
        for ev in self.tailer.poll():
            gid = ev.get("gene_id")
            if not gid:
                continue
            gene = self.genes.get(gid)
            if gene is None:
                self._seq += 1
                gene = Gene(gene_id=gid, seq=self._seq)
                self.genes[gid] = gene
                self.state.order.append(gid)
            event = ev.get("event")
            if ev.get("engine"):
                gene.engine = ev["engine"]
            if ev.get("canonical"):
                gene.canonical = ev["canonical"]
            if ev.get("head"):
                gene.head = ev["head"]
            if event == "proposed":
                gene.mu = ev.get("mu", gene.mu)
                gene.sigma = ev.get("sigma", gene.sigma)
                gene.p_bad = ev.get("p_bad", gene.p_bad)
            elif event == "selected":
                if gene.state == "proposed":
                    gene.state = "selected"
            elif event == "scored":
                gene.state = "scored"
                gene.oof = ev.get("oof_r2", ev.get("oof_auc", gene.oof))
                gene.test_r2 = ev.get("test_r2", ev.get("test_auc", gene.test_r2))
                gene.total_ms = ev.get("total_ms", gene.total_ms)
        # cap memory: drop oldest genes beyond MAX_GENES
        while len(self.genes) > MAX_GENES:
            old_id, _ = self.genes.popitem(last=False)
            if old_id in self.state.order:
                self.state.order.remove(old_id)
        if self.state.follow and self.state.order:
            self.state.cursor = len(self.state.order) - 1
        self.state.cursor = min(self.state.cursor, max(0, len(self.state.order) - 1))

    def _meta(self) -> dict[str, Any]:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # -- code fetch (on demand, cached) --------------------------------------
    def _code_for(self, gene: Gene, lang: str) -> dict[str, Any]:
        key = (gene.engine, gene.canonical, lang)
        cached = self._code_cache.get(key)
        if cached is not None:
            self._code_cache.move_to_end(key)
            return cached
        if not gene.engine or not gene.canonical:
            return {"supported": False, "errors": ["no gene selected"]}
        try:
            emitted = self.router.emit_one(gene.engine, gene.canonical)
        except Exception as exc:  # bridge unavailable etc.
            emitted = {"supported": False, "errors": [str(exc)]}
        self._code_cache[key] = emitted
        if len(self._code_cache) > 64:
            self._code_cache.popitem(last=False)
        return emitted

    # -- rendering -----------------------------------------------------------
    def render(self) -> Panel:
        self.ingest()
        from rich.layout import Layout

        layout = Layout()
        # header / body (genes | code) / footer (training loss | cluster GPUs+cores)
        layout.split_column(
            Layout(self._header(), size=4, name="header"),
            Layout(name="body"),
            Layout(self._footer(), size=9, name="footer"),
        )
        layout["body"].split_row(Layout(self._gene_table(), ratio=3), Layout(self._code_pane(), ratio=4))
        return Panel(layout, title="JOPE · GENE STREAM", border_style="bright_magenta", box=box.HEAVY)

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

    def _header(self) -> Panel:
        meta = self._meta()
        counts = meta.get("counts", {})
        gps = meta.get("genes_per_sec_ewma", 0.0)
        best = meta.get("best") or {}
        text = Text()
        text.append(f" {gps:6.1f} genes/s ", style="bold black on bright_green")
        text.append(
            f"  proposed {counts.get('proposed', 0)}  scored {counts.get('scored', 0)}"
            f"  selected {counts.get('selected', 0)}  ",
            style="white",
        )
        if best:
            text.append(
                f" ★ best {best.get('score', 0):+.3f} {best.get('engine', '')}/{best.get('head', '')} ",
                style="bold yellow",
            )
        foot = Text("  j/k move · p/r python/rust · l follow · s save · q quit", style="dim")
        return Panel(Group(text, foot), box=box.MINIMAL, border_style="magenta")

    def _gene_table(self) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
        table.add_column("#", justify="right", width=4, style="dim")
        table.add_column("eng", width=4)
        table.add_column("head", width=8)
        table.add_column("μ", justify="right", width=6)
        table.add_column("p_bad", justify="right", width=5)
        table.add_column("oof", justify="right", width=7)
        table.add_column("", width=1)  # state glyph
        table.add_column("genome", ratio=1, no_wrap=True)

        order = self.state.order
        # show a window around the cursor
        height = 30
        if len(order) > height:
            lo = min(max(0, self.state.cursor - height // 2), len(order) - height)
        else:
            lo = 0
        window = order[lo : lo + height]
        for local, gid in enumerate(window):
            gene = self.genes[gid]
            idx = lo + local
            selected = idx == self.state.cursor
            glyph, gstyle = _state_glyph(gene.state)
            oof_txt, oof_style = _score_cell(gene.oof)
            row_style = "on grey23" if selected else ""
            eng_style = "cyan" if gene.engine == "featmath" else "magenta"
            table.add_row(
                str(gene.seq),
                Text(gene.engine[:4], style=eng_style),
                _truncate(gene.head or "", 8),
                _fmt(gene.mu),
                _fmt(gene.p_bad, pct=False),
                Text(oof_txt, style=oof_style),
                Text(glyph, style=gstyle),
                _truncate(gene.canonical, CANON_PREVIEW),
                style=row_style,
            )
        title = f"genes ({len(order)})" + ("  ⏵follow" if self.state.follow else "")
        return Panel(table, title=title, border_style="blue", box=box.ROUNDED)

    def _code_pane(self) -> Panel:
        order = self.state.order
        if not order:
            return Panel(Text("waiting for genes…", style="dim"), title="algorithm", border_style="green")
        gene = self.genes[order[min(self.state.cursor, len(order) - 1)]]
        emitted = self._code_for(gene, self.state.lang)
        lang = self.state.lang
        if emitted.get("supported"):
            source = emitted.get(lang, "") or ""
            body: Any = Syntax(
                source[:CODE_PREVIEW_CHARS], lang, theme="ansi_dark", line_numbers=True, word_wrap=False
            )
            tokens = (emitted.get("code_tokens") or {}).get(lang, "?")
            title = f"{gene.engine} · {lang} · {gene.head or ''} · oof {_fmt(gene.oof)} · {tokens} tok"
            border = "green"
        else:
            unsupported = emitted.get("unsupported") or []
            lines = Text()
            if unsupported:
                lines.append("emit unavailable — ops not yet transpilable:\n", style="yellow")
                lines.append("  " + ", ".join(unsupported) + "\n\n", style="bold yellow")
            for err in emitted.get("errors", []):
                lines.append(f"  {err}\n", style="dim red")
            lines.append(f"\ngenome:\n  {gene.canonical}", style="dim")
            body = lines
            title = f"{gene.engine} · {lang} · (no source)"
            border = "yellow"
        return Panel(body, title=title, border_style=border, box=box.ROUNDED)

    def save_current(self) -> Path | None:
        order = self.state.order
        if not order:
            return None
        gene = self.genes[order[min(self.state.cursor, len(order) - 1)]]
        emitted = self._code_for(gene, self.state.lang)
        if not emitted.get("supported"):
            return None
        ext = "py" if self.state.lang == "python" else "rs"
        out_dir = self.run_dir / "gene_exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{gene.gene_id}.{ext}"
        path.write_text(emitted.get(self.state.lang, ""), encoding="utf-8")
        return path


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
