#!/usr/bin/env python3
"""CW-5 Hybrid (`eval_cw5_fitsel.py`):
Design-stage evaluation of Hybrid `[X, S]` -> 24M ICL (`ICL_M_r3_ckpt_h020.pt`) recipes `R0..R5` on `FIT-SEL`
(`fitsel_search_val.npz`), sharded across single-thread CPU workers, plus single-shot `FROZEN.json` selection & readout
on `clean_non_copy_88` (primary, per `H431` `EXCLUSION_FITSEL.json`) and `all_106` (descriptive).

Recipes (`R0..R5`):
  R0  H424/H428 baseline (`ridgeres` on X for F<=30; H428 T1 ridge-on-all + 8x30 bagged residual for F>30)
  R1  `[X, S]` with ridge base on `X` (`yr` from X) and ICL residual on `[X, S]` (8 bagged subsets always keeping S when wide)
  R2  `[X, S]` with structural ridge base on `[X, S]` (`RES.residual_targets([X, S], ...)`) + ICL residual on `[X, S]`
  R3  Two-pass residual: Pass 1 = R2 (4-fold inner context OOF `p1_c` + `p1_q`), Pass 2 = ICL residual on `[X, S, p1]`
      with inner-CV shrinkage `eta in {0.0, 0.35, 0.70}`
  R4  `R2` averaged over `4` feature permutations (`perm4`) when `F + d_S <= 30`, else `K=8` bagged subsets keeping `S`
  R5  Context-CV gated hybrid: selects `R2` when 5-fold ridge OOF RMSE on `[X, S]` beats `X` by >= 0.5% (zero query y read!), else `R0`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("WAVE_BASE_CODE"):
    sys.path.append(os.environ["WAVE_BASE_CODE"])

import icl_diag as DG  # noqa: E402
import icl_resid as RES  # noqa: E402
import struct_cols as SC  # noqa: E402
from icl_inject import RealVal  # noqa: E402
from icl_model import ICL  # noqa: E402

WIDE, SUB, K_BAG = 30, 30, 8
SEED_CW5 = 436_500
RECIPES = ("R0", "R1", "R2", "R3", "R4", "R5")
CONTROLS = ("ridge_X", "ridge_XS")


def _subsets_keep_s(cid: str, f_raw: int, d_s: int, tag: str = "CW5") -> list[torch.Tensor]:
    """Returns K_BAG index tensors into `[X, S]` (width `f_raw + d_s`) that ALWAYS include all `d_s` structural columns."""
    s_idx = np.arange(f_raw, f_raw + d_s, dtype=np.int64)
    k_raw = max(1, SUB - d_s)
    if f_raw <= k_raw:
        full = torch.from_numpy(np.concatenate([np.arange(f_raw, dtype=np.int64), s_idx]))
        return [full]
    out = []
    for j in range(K_BAG):
        h = int.from_bytes(hashlib.sha256(f"{tag}\0{cid}\0{j}".encode()).digest()[:8], "big")
        raw_pick = np.sort(np.random.default_rng(h).choice(f_raw, k_raw, replace=False))
        out.append(torch.from_numpy(np.concatenate([raw_pick, s_idx])))
    return out


@torch.no_grad()
def _icl_residual_pass(
    model,
    x_base: torch.Tensor,
    x_feat: torch.Tensor,
    y_full: torch.Tensor,
    n_ctx: int,
    cid: str,
    f_raw: int,
    d_s: int,
    seed: int,
    perms: int = 0,
    return_base: bool = False,
):
    """Computes `RES.residual_targets` on `x_base`, then predicts residual using `x_feat` (`[X, S]`)."""
    yr, used, bq = RES.residual_targets(x_base[None], y_full[None], n_ctx, torch.Generator().manual_seed(seed))
    yc_res = yr[0, :n_ctx]
    w_tot = x_feat.shape[1]
    if w_tot <= WIDE:
        res_q = DG.predict(model, x_feat, yc_res, n_ctx, perms=perms, seed=seed + 11)
    else:
        subs = _subsets_keep_s(cid, f_raw, d_s)
        acc = torch.zeros(x_feat.shape[0] - n_ctx, dtype=torch.float64)
        for S_idx in subs:
            acc += DG.predict(model, x_feat[:, S_idx], yc_res, n_ctx) / len(subs)
        res_q = acc
    n_q = int(x_feat.shape[0] - n_ctx)
    ym_val = float(y_full[:n_ctx].double().mean())
    base_q = bq[0].cpu().double() if bool(used[0]) else torch.full((n_q,), ym_val, dtype=torch.float64)
    pred_q = res_q + bq[0].cpu() if bool(used[0]) else res_q
    return (pred_q, base_q) if return_base else pred_q


def score_context_recipes(model, x_raw: torch.Tensor, y_full: torch.Tensor, n_ctx: int, cid: str, idx: int) -> tuple[dict, dict, dict]:
    """Evaluates R0..R5 + descriptive ICL-free controls strictly from `(x_raw[:n_ctx], y_full[:n_ctx], x_raw[n_ctx:])`."""
    seed0 = 419_500 + idx
    f_raw = int(x_raw.shape[1])
    xc_np = x_raw[:n_ctx].double().numpy()
    yc_np = y_full[:n_ctx].double().numpy()
    xq_np = x_raw[n_ctx:].double().numpy()
    n_q = int(x_raw.shape[0] - n_ctx)
    y_pad = torch.cat([y_full[:n_ctx], torch.zeros(n_q, dtype=y_full.dtype)])

    # R0: H424/H428 baseline on X only (+ descriptive base ridge_X)
    p_r0, b_r0 = _icl_residual_pass(model, x_raw, x_raw, y_pad, n_ctx, cid, f_raw, 0, seed0, perms=0, return_base=True)

    # Build S strictly on context rows
    S_c, S_q, smeta = SC.build_struct_cols(xc_np, yc_np, xq_np, seed=SEED_CW5 + idx)
    d_s = int(S_c.shape[1])

    if d_s == 0:
        preds = {r: p_r0 for r in RECIPES}
        ctrls = {"ridge_X": b_r0, "ridge_XS": b_r0}
        smeta.update({"r3_eta": 0.0, "r5_chose_struct": False, "ctx_ridge_gain_pct": 0.0})
        return preds, ctrls, smeta

    S_all = torch.from_numpy(np.vstack([S_c, S_q]).astype(np.float32))
    x_xs = torch.cat([x_raw.float(), S_all], dim=1)

    # R1: ridge base on X, ICL residual on [X, S]
    p_r1 = _icl_residual_pass(model, x_raw, x_xs, y_pad, n_ctx, cid, f_raw, d_s, seed0, perms=0)

    # R2: structural ridge base on [X, S], ICL residual on [X, S] (+ descriptive base ridge_XS)
    p_r2, b_r2 = _icl_residual_pass(model, x_xs, x_xs, y_pad, n_ctx, cid, f_raw, d_s, seed0, perms=0, return_base=True)

    # R4: perm4 on narrow [X, S], bagged keep-S on wide [X, S]
    p_r4 = _icl_residual_pass(model, x_xs, x_xs, y_pad, n_ctx, cid, f_raw, d_s, seed0, perms=4 if (f_raw + d_s) <= WIDE else 0)

    # R3: 2-pass residual (4-fold inner context OOF of R2 -> second ICL residual pass on [X, S, p1] with shrinkage eta)
    g_in = np.random.default_rng(SEED_CW5 + 1000 + idx)
    in_fold = g_in.permutation(n_ctx) % 4
    p1_oof = torch.zeros(n_ctx, dtype=torch.float64)
    p2_oof = torch.zeros(n_ctx, dtype=torch.float64)
    xc_t = x_xs[:n_ctx]
    yc_t = y_full[:n_ctx]
    for fk in range(4):
        ci = torch.from_numpy(np.nonzero(in_fold != fk)[0])
        vi = torch.from_numpy(np.nonzero(in_fold == fk)[0])
        sub_x = torch.cat([xc_t[ci], xc_t[vi]], dim=0)
        sub_y = torch.cat([yc_t[ci], torch.zeros(len(vi), dtype=yc_t.dtype)])
        p1_v = _icl_residual_pass(model, sub_x, sub_x, sub_y, int(len(ci)), cid, f_raw, d_s, seed0 + 30 + fk, perms=0)
        p1_oof[vi] = p1_v
    r1_c = (yc_t.double() - p1_oof).float()
    if float(r1_c.std()) > 1e-4 * max(float(yc_t.double().std()), 1e-9):
        x_aug_c = torch.cat([xc_t, p1_oof.float()[:, None]], dim=1)
        for fk in range(4):
            ci = torch.from_numpy(np.nonzero(in_fold != fk)[0])
            vi = torch.from_numpy(np.nonzero(in_fold == fk)[0])
            sub_xa = torch.cat([x_aug_c[ci], x_aug_c[vi]], dim=0)
            p2_oof[vi] = _icl_residual_pass(
                model,
                sub_xa,
                sub_xa,
                torch.cat([r1_c[ci], torch.zeros(len(vi), dtype=r1_c.dtype)]),
                int(len(ci)),
                cid,
                f_raw,
                d_s + 1,
                seed0 + 60 + fk,
                perms=0,
            )
        yc_d = yc_t.double()
        best_eta, best_mse = 0.0, float(((yc_d - p1_oof) ** 2).mean())
        for eta in (0.35, 0.70):
            m_e = float(((yc_d - (p1_oof + eta * p2_oof)) ** 2).mean())
            if m_e < 0.995 * best_mse:
                best_mse = m_e
                best_eta = eta
        if best_eta > 0.0:
            x_aug_all = torch.cat([x_xs, torch.cat([p1_oof, p_r2]).float()[:, None]], dim=1)
            r2_q = _icl_residual_pass(
                model,
                x_aug_all,
                x_aug_all,
                torch.cat([r1_c, torch.zeros(n_q, dtype=r1_c.dtype)]),
                n_ctx,
                cid,
                f_raw,
                d_s + 1,
                seed0 + 99,
                perms=0,
            )
            p_r3 = p_r2 + best_eta * r2_q
        else:
            p_r3 = p_r2
    else:
        best_eta = 0.0
        p_r3 = p_r2

    # R5: Context-only 5-fold ridge OOF gate comparing [X, S] vs X (zero query target access!)
    oof_x, _ = RES.ridge_base(x_raw[:n_ctx][None], y_full[:n_ctx][None], n_ctx, torch.Generator().manual_seed(seed0 + 7))
    oof_xs, _ = RES.ridge_base(x_xs[:n_ctx][None], y_full[:n_ctx][None], n_ctx, torch.Generator().manual_seed(seed0 + 7))
    rm_x = float(torch.sqrt(((y_full[:n_ctx].double() - oof_x[0]) ** 2).mean()))
    rm_xs = float(torch.sqrt(((y_full[:n_ctx].double() - oof_xs[0]) ** 2).mean()))
    gain_pct = 100.0 * (rm_x - rm_xs) / max(rm_x, 1e-12)
    chose_s = bool(rm_xs <= 0.995 * rm_x)
    p_r5 = p_r2 if chose_s else p_r0

    smeta.update({
        "r3_eta": best_eta,
        "r5_chose_struct": chose_s,
        "ctx_ridge_gain_pct": round(gain_pct, 3),
    })
    ctrls = {"ridge_X": b_r0, "ridge_XS": b_r2}
    return {"R0": p_r0, "R1": p_r1, "R2": p_r2, "R3": p_r3, "R4": p_r4, "R5": p_r5}, ctrls, smeta


def cmd_run(a) -> None:
    t0, c0 = time.time(), time.process_time()
    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)
    if hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest() != a.ckpt_sha256:
        raise SystemExit("ckpt sha256 mismatch")
    if hashlib.sha256(Path(a.val).read_bytes()).hexdigest() != a.val_sha256:
        raise SystemExit("val sha256 mismatch")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ICL(a.size)
    model.load_state_dict(ck["ema"])
    model.eval()

    v = np.load(a.val, allow_pickle=False)
    cids = [str(c) for c in v["cids"].tolist()]
    rv = RealVal(a.val, "cpu")
    order = sorted(range(len(cids)), key=lambda i: (rv.items[i][0].shape[0] * (rv.items[i][0].shape[1] - 1), cids[i]))
    k, m = (int(z) for z in a.shard.split("/"))
    mine = order[k::m]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for i in mine:
        cid = cids[i]
        p = out / f"ctx_{i:04d}.json"
        if p.exists():
            skipped += 1
            continue
        t1 = time.time()
        mat, n_ctx = rv.items[i]
        x_raw, y_full = mat[:, :-1], mat[:, -1]
        yq = y_full[n_ctx:].double()
        ym = y_full[:n_ctx].double().mean()
        preds, ctrls, smeta = score_context_recipes(model, x_raw, y_full, n_ctx, cid, i)
        rq = DG.ridge_fit_predict(
            torch.nan_to_num(x_raw[:n_ctx].double()), y_full[:n_ctx].double(), [torch.nan_to_num(x_raw[n_ctx:].double())]
        )[0]
        u_rec = {r: DG.u_of(preds[r], yq, ym) for r in RECIPES}
        u_ctrl = {c: DG.u_of(ctrls[c], yq, ym) for c in CONTROLS}
        ok = all(bool(torch.isfinite(preds[r]).all()) for r in RECIPES) and all(
            bool(torch.isfinite(ctrls[c]).all()) for c in CONTROLS
        )
        rec = {
            "i": i,
            "cid": cid,
            "family": rv.family[i],
            "F": int(x_raw.shape[1]),
            "n_ctx": int(n_ctx),
            "n_q": int(len(yq)),
            "u_ridge": DG.u_of(rq, yq, ym),
            "u": u_rec,
            "u_ctrl": u_ctrl,
            "struct": smeta,
            "ok": ok,
            "wall_s": round(time.time() - t1, 3),
        }
        tmp = p.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(rec, sort_keys=True))
        tmp.rename(p)
        done += 1

    st = {
        "schema": "jope.cw5.shard/v1",
        "shard": a.shard,
        "assigned": len(mine),
        "written": done,
        "already_there": skipped,
        "ckpt_sha256": a.ckpt_sha256,
        "val_sha256": a.val_sha256,
        "threads": 1,
        "device": "cpu",
        "cuda_visible": bool(torch.cuda.is_available()),
        "wall_s": round(time.time() - t0, 2),
        "cpu_s": round(time.process_time() - c0, 2),
    }
    (out / f"status_{k}of{m}_{int(t0)}.json").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st), flush=True)


def _summ_rows(rows: list[dict], T16, H) -> dict:
    unit = {r["cid"]: r["family"] for r in rows}
    vals = {rec: {r["cid"]: r["u"][rec] for r in rows} for rec in RECIPES}
    ctrl_vals = {c: {r["cid"]: r["u_ctrl"][c] for r in rows} for c in CONTROLS}
    pairs = {f"{rec}_vs_R0": T16.pair_stats(np, H, vals[rec], vals["R0"], unit) for rec in RECIPES if rec != "R0"}
    ctrl_pairs = {
        "ridge_XS_vs_ridge_X": T16.pair_stats(np, H, ctrl_vals["ridge_XS"], ctrl_vals["ridge_X"], unit),
        "R0_vs_ridge_X": T16.pair_stats(np, H, vals["R0"], ctrl_vals["ridge_X"], unit),
        "R2_vs_ridge_XS": T16.pair_stats(np, H, vals["R2"], ctrl_vals["ridge_XS"], unit),
        "R2_vs_ridge_X": T16.pair_stats(np, H, vals["R2"], ctrl_vals["ridge_X"], unit),
    }
    return {
        "n_contexts": len(rows),
        "n_families": len(set(unit.values())),
        "contexts_with_struct": int(sum(r["struct"]["d_S"] > 0 for r in rows)),
        "contexts_with_keys": int(sum(r["struct"]["n_keys"] > 0 for r in rows)),
        "contexts_with_formulas": int(sum(r["struct"]["n_formulas"] > 0 for r in rows)),
        "family_mean_u": {rec: round(float(H.family_mean(vals[rec], unit)), 4) for rec in RECIPES},
        "controls_family_mean_u": {c: round(float(H.family_mean(ctrl_vals[c], unit)), 4) for c in CONTROLS},
        "pairs_vs_R0": pairs,
        "controls_pairs": ctrl_pairs,
    }


def cmd_readout(a) -> None:
    out_p = Path(a.out)
    frozen_p = Path(a.frozen_out)
    if out_p.exists() or frozen_p.exists():
        raise SystemExit("readout or FROZEN.json already exists (create-once)")
    gates = json.loads(Path(a.gates).read_text())
    if not gates.get("pass"):
        raise SystemExit("REFUSING readout: integrity gates did not pass")

    import train_h416 as T16

    _, H = T16._stats_mod()
    excl = json.loads(Path(a.excl).read_text())
    tier_of = {cid: info["tier"] for cid, info in excl["contexts"].items()}

    rows_all = [json.loads(p.read_text()) for p in sorted(Path(a.data_dir).glob("ctx_*.json"))]
    if len(rows_all) != 106:
        raise SystemExit(f"expected 106 contexts, got {len(rows_all)}")
    rows_clean = [r for r in rows_all if tier_of[r["cid"]] != "copy"]
    rows_struct = [r for r in rows_clean if r["struct"]["d_S"] > 0]

    clean_sum = _summ_rows(rows_clean, T16, H)
    all_sum = _summ_rows(rows_all, T16, H)
    struct_sum = _summ_rows(rows_struct, T16, H) if rows_struct else None

    # Select winning frozen recipe among R1..R5 strictly by clipped family-mean gain vs R0 on clean_88!
    # Controls (ridge_X, ridge_XS) are strictly descriptive and non-selectable.
    best_r = max(
        ("R1", "R2", "R3", "R4", "R5"),
        key=lambda rec: (clean_sum["pairs_vs_R0"][f"{rec}_vs_R0"]["clipped_mean"], clean_sum["family_mean_u"][rec]),
    )
    best_stats = clean_sum["pairs_vs_R0"][f"{best_r}_vs_R0"]
    promote = bool(best_stats["clipped_mean"] >= 0.01 and best_stats["signflip_p"] < 0.05)
    verdict = "CW5_HYBRID_READY_FOR_JOINT_DEV" if promote else "CW5_HYBRID_DESIGN_MODEST"

    frozen = {
        "schema": "jope.cw5.frozen/v1",
        "selected_recipe": best_r if promote else None,
        "best_design_recipe": best_r,
        "clean88_clipped_gain_vs_R0": best_stats["clipped_mean"],
        "clean88_unclipped_gain_vs_R0": best_stats["gain"],
        "clean88_signflip_p": best_stats["signflip_p"],
        "promote_to_joint_dev": promote,
        "ckpt_sha256": gates["checks"]["ckpt_sha256"],
    }
    res = {
        "schema": "jope.cw5.readout/v1",
        "verdict": verdict,
        "frozen": frozen,
        "clean_non_copy_88": clean_sum,
        "clean_with_struct": struct_sum,
        "all_fitsel_106": all_sum,
        "gates_pass": gates["pass"],
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    frozen_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2, sort_keys=True))
    frozen_p.write_text(json.dumps(frozen, indent=2, sort_keys=True))
    out_p.chmod(0o444)
    frozen_p.chmod(0o444)
    print(json.dumps({"verdict": verdict, "frozen": frozen, "clean88_mean_u": clean_sum["family_mean_u"]}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--val", required=True)
    p.add_argument("--val-sha256", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-sha256", required=True)
    p.add_argument("--size", default="M")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", default="0/1")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("readout")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--excl", required=True)
    p.add_argument("--gates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frozen-out", required=True)
    p.set_defaults(fn=cmd_readout)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
