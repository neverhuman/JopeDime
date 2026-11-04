"""Main layout and gene/code panes for the gene stream TUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .tui import _truncate
from .tui_stream import CANON_PREVIEW, CODE_PREVIEW_CHARS, _fmt, _score_cell, _state_glyph


class StreamDisplayMixin:
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
