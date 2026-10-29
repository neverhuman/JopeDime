"""GP generation recorder (TELEMETRY_SPEC rows G1-G12 and the E2 status split).

The flywheel's evolution loop calls ``add`` with each evaluator result dict and ``finish`` at
the end of a generation. This module only marshals: it maps result dicts to the JSON that
``jope_telemetry``'s ``GpSession.summarize_json`` takes, and hands the returned summary to
``emit``. Diversity, entropy and novelty are computed in Rust.

The result dicts come from ``research/flywheel_selfplay/rust_evaluator.py``: ``status`` is
``completed`` or ``failed`` (timeouts, nonzero exits and parse failures are not told apart
there), ``fitness`` is lower-is-better with ``100.0`` for failures, and ``proj16`` is a 16-D
float projection whose signs are packed into bits here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from typing import Any, Protocol

GEN_KIND = "jope.telemetry.gp-generation/v1"

Emit = Callable[[str, dict[str, Any]], Any]


class Session(Protocol):
    def summarize_json(self, generation_json: str) -> str: ...


def proj16_bits(proj16: Any) -> int | None:
    """Bit i is set when component i of the 16-D projection is positive."""
    if proj16 is None:
        return None
    vals = [float(v) for v in proj16]
    if len(vals) != 16:
        return None
    return sum(1 << i for i, v in enumerate(vals) if v > 0.0)


def error_profile(result: dict[str, Any]) -> list[float | None]:
    """The G4 proxy available today: the candidate's error profile on its own context."""

    def fin(v: Any) -> float | None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    test = fin(result.get("test_rmse_ratio"))
    return [
        fin(result.get("cv_mean_log_ratio")),
        fin(result.get("cv_tail_log_ratio")),
        fin(result.get("cv_log_ratio_sd")),
        math.log(test) if test is not None and test > 0.0 else None,
    ]


def status_of(result: dict[str, Any]) -> str:
    """``completed`` stays as is; a ``failed`` result is split by its return code if the
    worker recorded one (``rc``), and stays ``failed`` (cause unknown) otherwise."""
    status = str(result.get("status", "failed"))
    if status == "completed":
        return "completed"
    rc = result.get("rc")
    if rc is None:
        return "failed"
    return "timeout" if int(rc) == -1 else "nonzero_rc" if int(rc) != 0 else "parse_fail"


class GenerationRecorder:
    """Collects one generation of one context and summarizes it through ``session``.

    ``bank_shas`` and ``corpus_shas`` are the program hashes in the champion bank and the
    corpus at round start; a candidate in neither is ``novel``.
    """

    def __init__(
        self,
        session: Session,
        context: str,
        round_idx: int,
        emit: Emit | None = None,
        bank_shas: Iterable[str] = (),
        corpus_shas: Iterable[str] = (),
    ) -> None:
        self.session = session
        self.context = context
        self.round = int(round_idx)
        self.emit = emit
        self.bank = frozenset(bank_shas)
        self.corpus = frozenset(corpus_shas)
        self._cands: list[dict[str, Any]] = []
        self._proposals = 0
        self._seen_rejects = 0

    def proposals(self, n: int, seen_rejects: int) -> None:
        """Proposal-time counters for G2: candidates proposed, and those dropped by ``seen_dsl``."""
        self._proposals += int(n)
        self._seen_rejects += int(seen_rejects)

    def provenance(self, program_sha256: str) -> str:
        if program_sha256 in self.bank:
            return "bank"
        if program_sha256 in self.corpus:
            return "replayed"
        return "novel"

    def add(
        self,
        result: dict[str, Any],
        parent_dsl: str | None = None,
        improved: bool | None = None,
        hwm_winner: bool = False,
        behavior: list[float | None] | None = None,
    ) -> None:
        dsl = result["dsl"]
        sha = result.get("program_sha256") or hashlib.sha256(dsl.encode("utf-8")).hexdigest()
        complexity = result.get("complexity_open_parens")
        self._cands.append(
            {
                "dsl": dsl,
                "parent_dsl": parent_dsl,
                "operator": str(result.get("operator", "")),
                "status": status_of(result),
                "cv_gate_passed": bool(result.get("cv_gate_passed", False)),
                "fitness": float(result.get("fitness", 100.0)),
                "improved": improved,
                "hwm_winner": bool(hwm_winner),
                "complexity": int(complexity) if complexity is not None else None,
                "proj16_bits": proj16_bits(result.get("proj16")),
                "behavior": behavior if behavior is not None else error_profile(result),
                "provenance": self.provenance(sha),
            }
        )

    def finish(self, generation: int) -> dict[str, Any]:
        payload = {
            "context": self.context,
            "round": self.round,
            "generation": int(generation),
            "proposals": self._proposals,
            "seen_rejects": self._seen_rejects,
            "candidates": self._cands,
        }
        summary = json.loads(self.session.summarize_json(json.dumps(payload)))
        self._cands = []
        self._proposals = self._seen_rejects = 0
        if self.emit is not None:
            self.emit(GEN_KIND, summary)
        return summary
