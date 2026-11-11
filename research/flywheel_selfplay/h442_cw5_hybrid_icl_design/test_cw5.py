#!/usr/bin/env python3
"""Synthetic unit verification for CW-5 Hybrid (`test_cw5.py`):
Checks key + formula extraction, count scale alignment, wide-table keep-S bagging, and strict query-row permutation invariance.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("WAVE_BASE_CODE"):
    sys.path.append(os.environ["WAVE_BASE_CODE"])

import eval_cw5_fitsel as CW5  # noqa: E402
import struct_cols as SC  # noqa: E402
from icl_model import ICL  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(436)
    n_c, n_q, f = 240, 80, 10
    X = rng.uniform(1.0, 5.0, size=(n_c + n_q, f))
    g = rng.integers(0, 20, size=n_c + n_q)
    X[:, 2] = g * 3.0
    key_eff = rng.normal(size=20)[g]
    y = 1.5 * key_eff + 2.0 * (X[:, 0] / X[:, 1]) + 0.05 * rng.normal(size=n_c + n_q)

    S_c, S_q, meta = SC.build_struct_cols(X[:n_c], y[:n_c], X[n_c:], seed=77)
    # 1. Finds both exact key x2 and raw ratio x0/x1
    c1 = bool(meta["n_keys"] >= 1 and meta["n_formulas"] >= 1 and S_c.shape[1] == S_q.shape[1] and S_c.shape[1] >= 3)
    c2 = bool(np.all(np.isfinite(S_c)) and np.all(np.isfinite(S_q)))

    # 2. Strict query-row permutation invariance (zero query cross-row leakage!)
    perm = rng.permutation(n_q)
    _, S_q_perm, _ = SC.build_struct_cols(X[:n_c], y[:n_c], X[n_c:][perm], seed=77)
    c3 = bool(np.max(np.abs(S_q[perm] - S_q_perm)) < 1e-12)

    # 3. Wide-table bagging subsets ALWAYS include all S columns
    subs = CW5._subsets_keep_s("0" * 64, f_raw=45, d_s=6)
    s_expected = set(range(45, 51))
    c4 = bool(len(subs) == 8 and all(s_expected <= set(t.tolist()) and len(t) == 30 for t in subs))

    # 4. R0..R5 + ICL-free controls run cleanly through tiny ICL("S"), and R0 matches DG.run_variant("ridgeres") on F<=30
    import icl_diag as DG

    torch.manual_seed(0)
    tiny = ICL("S", layers=2, col_blocks=1, row_layers=1)
    tiny.eval()
    x_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    preds, ctrls, rmeta = CW5.score_context_recipes(tiny, x_t, y_t, n_c, "0" * 64, 0)
    y_pad = torch.cat([y_t[:n_c], torch.zeros(n_q, dtype=y_t.dtype)])
    ref_r0 = DG.run_variant(tiny, "ridgeres", x_t, y_pad, n_c, 419_500 + 0)
    yq_d = y_t[n_c:].double()
    ym_d = y_t[:n_c].double().mean()
    u_r0 = DG.u_of(preds["R0"], yq_d, ym_d)
    u_ref = DG.u_of(ref_r0, yq_d, ym_d)
    u_rx = DG.u_of(ctrls["ridge_X"], yq_d, ym_d)
    u_rxs = DG.u_of(ctrls["ridge_XS"], yq_d, ym_d)
    c5 = bool(
        set(preds.keys()) == set(CW5.RECIPES)
        and set(ctrls.keys()) == set(CW5.CONTROLS)
        and all(preds[r].shape == (n_q,) and bool(torch.isfinite(preds[r]).all()) for r in CW5.RECIPES)
        and all(ctrls[c].shape == (n_q,) and bool(torch.isfinite(ctrls[c]).all()) for c in CW5.CONTROLS)
        and rmeta["ctx_ridge_gain_pct"] > 5.0
        and rmeta["r5_chose_struct"] is True
        and u_rxs > u_rx + 0.30
    )
    c6 = bool(abs(u_r0 - u_ref) / max(1.0, abs(u_ref)) <= 1e-12)

    checks = {
        "finds_key_and_formula": c1,
        "finite_struct_cols": c2,
        "query_permutation_invariant": c3,
        "wide_subsets_keep_all_S": c4,
        "recipes_and_controls_finite_and_gated": c5,
        "r0_matches_icl_diag_ridgeres_exact": c6,
    }
    res = {"checks": checks, "all_passed": all(checks.values())}
    print(json.dumps(res, sort_keys=True))
    if not res["all_passed"]:
        raise SystemExit(1)
    print("CW5_UNIT_TESTS_OK 6/6")


if __name__ == "__main__":
    main()
