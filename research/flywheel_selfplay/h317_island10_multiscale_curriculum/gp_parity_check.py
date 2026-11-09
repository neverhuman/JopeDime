#!/usr/bin/env python3
"""H317 GP host-parity check (xbabe2 bwrap sandbox vs xbabe0 docker sandbox). Infrastructure only, no pilot outcome.

1. worker binary sha256 == pinned e3be9386...;
2. one champion evaluation on a deterministic NON-pilot TRAIN context (first sorted cid outside pilot_shard(205)),
   on its SEARCH and AUDIT envelopes; digest = sha256 of the canonical JSON of the worker's inner_cv records with the
   timing fields removed (predictions, targets, metrics, model/scaler sha256 kept);
3. torch / numpy import + versions; guard report.
Usage: gp_parity_check.py <out.json> [--core N] [--expect <reference.json>]   (exit 1 on any mismatch)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()

import gp_engine_h317 as E  # noqa: E402
import gp_grammar_h317 as G  # noqa: E402

WORKER_SHA256 = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
TIMING_KEYS = {"started_utc_us", "finished_utc_us", "fit_finished_utc_us", "wall_us"}


def digest_doc(doc: dict) -> str:
    recs = [
        {k: v for k, v in r.items() if k not in TIMING_KEYS}
        for r in doc.get("inner_cv") or []
    ]
    return hashlib.sha256(json.dumps(recs, sort_keys=True).encode()).hexdigest()


def screener_smoke(run_dir: Path, cid: str, rows) -> dict:
    """Load both seeds' OOF models per fold (load_fold_screener default = every seed under run_dir, scores averaged),
    score 160 bank programs for a NON-pilot target with all-zero view features (no data), and check that every
    other fold's model refuses the target (ScreenerLeak)."""
    import time

    import numpy as np

    import distributed_10gen_island_gp as D
    import screener_h317 as S

    folds = C.family_folds(rows)
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    zeros = S.SearchViewProvider(
        view_fn=lambda c, kind: np.zeros((32, 997), np.float32)
    )
    t0 = time.time()
    fs = S.load_neural_fold_screeners(run_dir, folds, zeros, family_of=fam)
    load_s = time.time() - t0
    bank, _ = D.build_bank(rows)
    dsls = [b.dsl for b in bank[:160]]
    ad = fs.for_context(cid)
    t0 = time.time()
    sc = ad.score(cid, list(range(32)), dsls)
    call_s = time.time() - t0
    refused = 0
    for f, other in fs.by_fold.items():
        if f == folds[cid]:
            continue
        try:
            other.score(cid, [0], dsls[:2])
        except Exception as e:  # noqa: BLE001
            refused += type(e).__name__ == "ScreenerLeak"
    n_models = {f: len(a.model.models) for f, a in fs.by_fold.items()}
    return {
        "load_s": load_s,
        "score_32x160_s": call_s,
        "shape": list(sc.shape),
        "finite": bool(np.isfinite(sc).all()),
        "models_per_fold": n_models,
        "other_folds_refused": refused,
        "cacheable": ad.cache_scores,
        "ok": bool(np.isfinite(sc).all())
        and refused == C.N_FOLDS - 1
        and all(v == 2 for v in n_models.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--expect", default=None)
    ap.add_argument(
        "--screener-run",
        default=None,
        help="smoke-load the OOF fold screeners (zero view features)",
    )
    args = ap.parse_args()
    rows = C.train_rows()
    pilot = set(C.pilot_shard(rows, C.family_folds(rows)))
    cid = next(r["context_sha256"] for r in rows if r["context_sha256"] not in pilot)
    fam = {r["context_sha256"]: r["family_id"] for r in rows}[cid]
    champ = next(
        r
        for r in json.loads((C.RES01 / "champions.json").read_text())
        if r["context_sha256"] == cid
    )
    cand = G.Candidate.from_dsl(champ["dsl"], "parity")
    root = C.DATA_ROOT / "gp" / "parity"
    docs = {}

    def capture(ctx, kind, c, core, slot):
        doc, wall, err = E.rust_runner(C.WORKER_BIN, 120.0)(ctx, kind, c, core, slot)
        docs[kind] = doc
        return doc, wall, err

    eng = E.EvalEngine([args.core], root=root, runner=capture, persist=False)
    ctx = eng.context(cid, fam)
    s = eng.search(ctx, [cand])[0]
    a = eng.audit(ctx, [cand])[0]
    eng.close()
    out = {
        "cid": cid,
        "dsl": cand.dsl,
        "worker_sha256": C.sha256_file(C.WORKER_BIN),
        "worker_sha_ok": C.sha256_file(C.WORKER_BIN) == WORKER_SHA256,
        "search_env_sha256": ctx.env_sha["search"],
        "audit_env_sha256": ctx.env_sha["audit"],
        "search_digest": digest_doc(docs["search"]),
        "audit_digest": digest_doc(docs["audit"]),
        "search_utility": s.utility,
        "audit_utility": a.utility,
    }
    try:
        import numpy
        import torch

        out["torch"], out["numpy"] = torch.__version__, numpy.__version__
        out["torch_ok"] = bool(torch.ones(2).sum().item() == 2.0)
    except Exception as e:  # noqa: BLE001
        out["torch"], out["torch_ok"] = f"import failed: {e}", False
    if args.screener_run:
        out["screener_smoke"] = screener_smoke(Path(args.screener_run), cid, rows)
    out["guard"] = C.guard_report()
    C.write_json(Path(args.out), out)
    ok = (
        out["worker_sha_ok"]
        and out["torch_ok"]
        and out.get("screener_smoke", {}).get("ok", True)
    )
    if args.expect:
        ref = json.loads(Path(args.expect).read_text())
        for k in (
            "cid",
            "dsl",
            "worker_sha256",
            "search_env_sha256",
            "audit_env_sha256",
            "search_digest",
            "audit_digest",
            "search_utility",
            "audit_utility",
        ):
            same = ref[k] == out[k]
            print(f"{k}: {'MATCH' if same else 'MISMATCH'}")
            ok = ok and same
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "cid",
                    "worker_sha_ok",
                    "search_digest",
                    "audit_digest",
                    "torch",
                )
            }
        )
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
