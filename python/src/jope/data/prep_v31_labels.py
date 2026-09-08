"""One-shot corpus label pass for v3.1 self-play training.

Stamps onto every row of an existing active corpus (in place, backup kept):
- ``metadata.task_type = "regression"`` (this corpus is regression; explicit beats implicit)
- ``metadata.rank_pct``: per-dataset rank percentile of the robust score in [0,1]
  (the listwise/utility target that is NOT compressed at zero like delta-to-incumbent)
- ``metadata.rwr_weight_global``: RWR weight computed per dataset over the FULL candidate
  set (batch-local recomputation was noisy; this wires the trainer's ``precomputed_rwr``)
- sanitized ``metadata.runtime_label = log1p(min(cost_ms, cap))`` + ``runtime_censored``

Runs in minutes; does not add/remove rows.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from .labels import assign_elo  # noqa: F401 (kept for parity with corpus builders)
from ..policy.losses import rwr_weights

RUNTIME_CAP_MS = 600_000.0  # 10 min — beyond this treat as censored timeout


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def prep_v31_labels(corpus_path: str | Path, *, write: bool = True) -> dict:
    corpus_path = Path(corpus_path)
    raw = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # group scored rows per dataset for rank percentiles
    by_ds: dict[str, list[int]] = defaultdict(list)
    scores: list[float | None] = []
    for i, r in enumerate(raw):
        md = r.get("metadata") or {}
        s = _finite(md.get("score_robust", r.get("score")))
        scores.append(s)
        if s is not None:
            by_ds[str(r.get("dataset_id"))].append(i)

    n_ranked = 0
    for ds, idxs in by_ds.items():
        vals = sorted((scores[i] for i in idxs))  # type: ignore[type-var]
        n = len(idxs)
        for i in idxs:
            s = scores[i]
            # average-rank percentile in [0,1]; single-row datasets get 0.5
            if n == 1:
                pct = 0.5
            else:
                lo = 0
                hi = 0
                for v in vals:
                    if v < s:
                        lo += 1
                    elif v == s:
                        hi += 1
                pct = (lo + 0.5 * (hi - 1)) / (n - 1) if n > 1 else 0.5
                pct = min(max(pct, 0.0), 1.0)
            raw[i].setdefault("metadata", {})["rank_pct"] = round(pct, 6)
            n_ranked += 1

    # global RWR (per dataset over the full candidate set)
    dataset_ids = [str(r.get("dataset_id")) for r in raw]
    provenance = [str((r.get("metadata") or {}).get("provenance") or "gp_report") for r in raw]
    weights = rwr_weights(dataset_ids, scores, provenance)

    for r, w in zip(raw, weights):
        md = r.setdefault("metadata", {})
        md["task_type"] = md.get("task_type") or "regression"
        md["rwr_weight_global"] = round(float(w), 6)
        cost = _finite(r.get("cost_ms"))
        if cost is not None:
            md["runtime_label"] = round(math.log1p(min(cost, RUNTIME_CAP_MS)), 6)
            md["runtime_censored"] = bool(cost >= RUNTIME_CAP_MS)

    report = {
        "rows": len(raw),
        "datasets": len(by_ds),
        "rows_ranked": n_ranked,
        "rows_censored_runtime": sum(1 for r in raw if (r.get("metadata") or {}).get("runtime_censored")),
    }
    if write:
        backup = corpus_path.with_suffix(".v3.bak.jsonl")
        if not backup.exists():
            shutil.copy2(corpus_path, backup)
        with corpus_path.open("w", encoding="utf-8") as fh:
            for r in raw:
                fh.write(json.dumps(r) + "\n")
    return report
