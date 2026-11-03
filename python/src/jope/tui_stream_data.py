"""Feed ingestion and source fetching for the gene stream view."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .engines import EngineRouter
from .tui_stream_state import Gene, JsonlTailer, StreamState
from .tui_stream import INITIAL_TAIL_BYTES, MAX_GENES

class StreamDataMixin:
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
        except (Exception,) as exc:  # bridge unavailable etc.
            emitted = {"supported": False, "errors": [str(exc)]}
        self._code_cache[key] = emitted
        if len(self._code_cache) > 64:
            self._code_cache.popitem(last=False)
        return emitted

    # -- rendering -----------------------------------------------------------
