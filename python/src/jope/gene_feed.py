"""Per-gene event feed: a live JSONL stream the TUI tails in real time.

One line per lifecycle event of a proposed genome ("gene"):
  proposed  — decoded, bridge-validated, surrogate-scored (mu/sigma/p_bad), pre-CV
  scored    — real CV completed (oof_r2 / oof_auc, timings)
  selected  — entered the corpus / ranked within the top-n

A stable ``gene_id = sha1(engine|dataset_hash|canonical)[:12]`` correlates the three
events so a viewer can update ONE row in place across the lifecycle. Scores are inlined
(each event ~300 B); emitted source code is NOT (10-50x larger) — the TUI fetches code
on demand via ``EngineRouter.emit`` and keys it by (engine, canonical).

Durability/perf choices:
  - single append handle, line-buffered; a ``threading.Lock`` guards write+seq so the
    stream driver's sampler and eval-consumer threads can share one feed;
  - time-based flush (every ``flush_interval_s``) so a tailer sees data within ~50 ms
    without an fsync per event;
  - NUMBERED rotation (``feed.jsonl.1``, ``.2`` …) — never truncate-in-place, which
    would corrupt a byte-offset tailer;
  - an atomically-rewritten ``<feed>.meta.json`` sidecar carries running counts /
    genes-per-sec / best-gene so the TUI reads header stats from O(1) bytes instead of
    scanning the feed.

Everything is opt-in: producers take ``gene_feed: GeneFeed | None = None`` and default to
None, so existing callers and tests are unaffected.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = "jope.gene_event.v1"
EVENTS = ("proposed", "scored", "selected")


def gene_id(engine: str, dataset_hash: str, canonical: str) -> str:
    """Stable correlation id for a gene across its lifecycle events."""
    key = f"{engine}|{dataset_hash}|{canonical}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:12]


def _clean(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop None values so events stay compact and event-type keys stay meaningful."""
    return {k: v for k, v in fields.items() if v is not None}


class GeneFeed:
    """Append-only writer for a ``gene_events.jsonl`` feed plus its meta sidecar."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        task_type: str = "regression",
        rotate_bytes: int = 512 * 1024 * 1024,
        keep_rotations: int = 3,
        flush_interval_s: float = 0.05,
        meta_interval_s: float = 0.5,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path.with_name(self.path.name + ".meta.json")
        self.run_id = run_id
        self.task_type = task_type
        self.rotate_bytes = int(rotate_bytes)
        self.keep_rotations = int(keep_rotations)
        self.flush_interval_s = float(flush_interval_s)
        self.meta_interval_s = float(meta_interval_s)

        self._lock = threading.Lock()
        self._seq = 0
        self._bytes = self.path.stat().st_size if self.path.exists() else 0
        self._counts = {event: 0 for event in EVENTS}
        self._best: dict[str, Any] | None = None
        self._genes_per_sec = 0.0
        self._last_event_ts: float | None = None
        self._last_flush = 0.0
        self._last_meta = 0.0
        self._handle = self.path.open("a", buffering=1, encoding="utf-8")
        self._closed = False
        self._write_meta(force=True)

    # -- lifecycle events ----------------------------------------------------
    def proposed(
        self,
        *,
        engine: str,
        canonical: str,
        dataset_hash: str,
        head: str | None = None,
        families: list[str] | None = None,
        decoder: str | None = None,
        mu: float | None = None,
        sigma: float | None = None,
        p_bad: float | None = None,
        percentile: float | None = None,
        calibrated_win_rate: float | None = None,
        calibrated_expected_delta: float | None = None,
        novel: bool | None = None,
        risk: str | None = None,
        iteration: int | None = None,
    ) -> str:
        gid = gene_id(engine, dataset_hash, canonical)
        self.event(
            "proposed",
            _clean(
                {
                    "gene_id": gid,
                    "dataset_hash": dataset_hash,
                    "engine": engine,
                    "canonical": canonical,
                    "head": head,
                    "families": families,
                    "decoder": decoder,
                    "mu": mu,
                    "sigma": sigma,
                    "p_bad": p_bad,
                    "percentile": percentile,
                    "calibrated_win_rate": calibrated_win_rate,
                    "calibrated_expected_delta": calibrated_expected_delta,
                    "novel": novel,
                    "risk": risk,
                    "iteration": iteration,
                }
            ),
        )
        return gid

    def scored(
        self,
        gene_id: str,
        *,
        engine: str | None = None,
        dataset_hash: str | None = None,
        canonical: str | None = None,
        head: str | None = None,
        oof_r2: float | None = None,
        oof_auc: float | None = None,
        fold_scores: list[float] | None = None,
        test_r2: float | None = None,
        test_auc: float | None = None,
        score_robust: float | None = None,
        total_ms: float | None = None,
        cache_hit: bool = False,
        timeout: bool = False,
        blowup: bool = False,
    ) -> None:
        primary = oof_auc if oof_auc is not None else oof_r2
        self._update_best(primary, engine, head, canonical, dataset_hash)
        self.event(
            "scored",
            _clean(
                {
                    "gene_id": gene_id,
                    "engine": engine,
                    "dataset_hash": dataset_hash,
                    "canonical": canonical,
                    "head": head,
                    "oof_r2": oof_r2,
                    "oof_auc": oof_auc,
                    "fold_scores": fold_scores,
                    "test_r2": test_r2,
                    "test_auc": test_auc,
                    "score_robust": score_robust,
                    "total_ms": total_ms,
                    "cache_hit": cache_hit or None,
                    "timeout": timeout or None,
                    "blowup": blowup or None,
                }
            ),
        )

    def selected(
        self,
        gene_id: str,
        *,
        engine: str | None = None,
        dataset_hash: str | None = None,
        canonical: str | None = None,
        selection_slot: str | None = None,
        reward_rank: int | None = None,
        generation: int | None = None,
        candidate_pool_id: str | None = None,
        selection_key: float | None = None,
        selection_score: float | None = None,
        selected_for_eval: bool | None = None,
    ) -> None:
        self.event(
            "selected",
            _clean(
                {
                    "gene_id": gene_id,
                    "engine": engine,
                    "dataset_hash": dataset_hash,
                    "canonical": canonical,
                    "selected": True,
                    "selection_slot": selection_slot,
                    "reward_rank": reward_rank,
                    "generation": generation,
                    "candidate_pool_id": candidate_pool_id,
                    "selection_key": selection_key,
                    "selection_score": selection_score,
                    "selected_for_eval": selected_for_eval,
                }
            ),
        )

    # -- core write ----------------------------------------------------------
    def event(self, event: str, fields: dict[str, Any]) -> None:
        if event not in EVENTS:
            raise ValueError(f"unknown gene event {event!r}")
        now = time.time()
        line = json.dumps(
            {
                "schema": SCHEMA,
                "event": event,
                "ts": now,
                "run_id": self.run_id,
                "task_type": self.task_type,
                **fields,
            }
        )
        with self._lock:
            if self._closed:
                return
            self._seq += 1
            payload = line + "\n"
            self._handle.write(payload)
            self._bytes += len(payload.encode("utf-8"))
            self._counts[event] = self._counts.get(event, 0) + 1
            # EWMA of inter-event rate for the header genes/sec readout.
            if self._last_event_ts is not None:
                dt = now - self._last_event_ts
                if dt > 0:
                    inst = 1.0 / dt
                    self._genes_per_sec = (
                        inst if self._genes_per_sec == 0.0 else 0.9 * self._genes_per_sec + 0.1 * inst
                    )
            self._last_event_ts = now
            if now - self._last_flush >= self.flush_interval_s:
                self._handle.flush()
                self._last_flush = now
            if now - self._last_meta >= self.meta_interval_s:
                self._write_meta()
                self._last_meta = now
            if self._bytes >= self.rotate_bytes:
                self._rotate()

    # -- internals -----------------------------------------------------------
    def _update_best(
        self,
        score: float | None,
        engine: str | None,
        head: str | None,
        canonical: str | None,
        dataset_hash: str | None,
    ) -> None:
        if score is None:
            return
        if self._best is None or score > self._best.get("score", float("-inf")):
            self._best = _clean(
                {
                    "score": float(score),
                    "engine": engine,
                    "head": head,
                    "canonical": canonical,
                    "dataset_hash": dataset_hash,
                }
            )

    def _write_meta(self, *, force: bool = False) -> None:
        payload = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "task_type": self.task_type,
            "feed": self.path.name,
            "seq": self._seq,
            "counts": dict(self._counts),
            "genes_per_sec_ewma": round(self._genes_per_sec, 3),
            "best": self._best,
            "bytes": self._bytes,
            "updated_at": time.time(),
        }
        tmp = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.meta_path)

    def _rotate(self) -> None:
        self._handle.flush()
        self._handle.close()
        # Shift numbered rotations up: .(k-1) -> .k, dropping the oldest.
        for k in range(self.keep_rotations, 0, -1):
            src = self.path.with_name(f"{self.path.name}.{k - 1}") if k > 1 else self.path
            dst = self.path.with_name(f"{self.path.name}.{k}")
            if src.exists():
                os.replace(src, dst)
        self._handle = self.path.open("a", buffering=1, encoding="utf-8")
        self._bytes = 0

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._handle.flush()
            self._write_meta(force=True)
            self._handle.close()

    def __enter__(self) -> "GeneFeed":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def gene_id(engine: str, dataset_hash: str, canonical: str) -> str:
        return gene_id(engine, dataset_hash, canonical)
