#!/usr/bin/env python3
"""Synthetic self-test suite for `H448` (`selftest_h448.py`).

Touches ZERO `dev91` bytes. Exercises:
  1. H425 feature-only contamination screen (`sketch_features_only` + `screen_one_h425`: copy vs clear, `tk is None`).
  2. Full 91-context / 19-family / 32-entrant synthetic board scoring + `tools/ladder.py` `< 1e-9` cross-check.
  3. Gate refusal cases (`missing_h445_without_owner_token`, `tampered_sxb_replay`) ensuring exit `3`, `INCOMPLETE`,
     and zero `rows` written, plus `OWNER_DIRECT_D91` receipt recording.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_h448 as AN  # noqa: E402
import guard_matern_core as GM  # noqa: E402
import meta_h416 as M  # noqa: E402
import train_h416 as T16  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_h425_feature_only_screen(tmp_dir: Path) -> None:
    """Import `d91_h448` under H335_GROUP=AGG in a subprocess or directly to test H425 feature-only functions."""
    os.environ["H335_GROUP"] = "AGG"
    os.environ["H448_D91"] = str(tmp_dir / "d91_dummy")
    import d91_h448 as D48

    rng = np.random.default_rng(44801)
    X_ref = rng.normal(size=(240, 8))
    X_other = rng.uniform(-10.0, 10.0, size=(200, 8))
    X_copy = X_ref[:160].copy()  # shared rows & specific columns -> must flag "copy"
    X_clear = rng.exponential(scale=3.0, size=(180, 6))

    fit = [
        D48.sketch_features_only("fit_0001", X_ref),
        D48.sketch_features_only("fit_0002", X_other),
    ]
    q_copy = D48.sketch_features_only("q_copy", X_copy)
    q_clear = D48.sketch_features_only("q_clear", X_clear)
    assert q_copy["tk"] is None and q_clear["tk"] is None

    col_idx, row_idx, pair_idx, pair_df = (
        collections.defaultdict(list),
        collections.defaultdict(list),
        collections.defaultdict(list),
        collections.Counter(),
    )
    for i, f_sk in enumerate(fit):
        for j, s in enumerate(f_sk["spec"]):
            for h in s.tolist():
                col_idx[h].append((i, j))
        for h in f_sk["rows"].tolist():
            row_idx[h].append(i)
        for p in f_sk["pairs"]:
            pair_idx[p].append(i)
    for sk_any in fit + [q_copy, q_clear]:
        for p in sk_any["pairs"]:
            pair_df[p] += 1
    names_fit = [D48.h425_name_tokens("fresh:srcfile:alpha_housing"), D48.h425_name_tokens("fresh:srcfile:beta_energy")]

    res_copy = D48.screen_one_h425(q_copy, fit, col_idx, row_idx, pair_idx, pair_df, names_fit, None)
    res_clear = D48.screen_one_h425(q_clear, fit, col_idx, row_idx, pair_idx, pair_df, names_fit, None)
    assert res_copy["flag_feature"] is True and res_copy["tier"] == "copy"
    assert res_clear["flag_feature"] is False and res_clear["tier"] == "clear"


def build_synthetic_fixture(base: Path) -> dict:
    rng = np.random.default_rng(44802)
    _, H = T16._stats_mod()

    cids = [hashlib.sha256(f"syn_cid_{i:03d}".encode()).hexdigest() for i in range(91)]
    cids.sort()
    fams = [f"fam-{hashlib.sha256(f'syn_fam_{i % 19:02d}'.encode()).hexdigest()}" for i in range(91)]
    fam_map = dict(zip(cids, fams))
    envs = {c: hashlib.sha256(f"env_{c}".encode()).hexdigest() for c in cids}

    P = base / "pulled"
    P418 = base / "h418_pulled"
    B = base / "board"
    (B / "scores").mkdir(parents=True, exist_ok=True)
    for d in ("cols_matern_D91", "cols_k3_D91", "cols_f4_D91", "cols_icl_D91"):
        (P / d).mkdir(parents=True, exist_ok=True)
    for t in ("D1", "D2"):
        (P418 / t / "stages" / f"H416_{t}").mkdir(parents=True, exist_ok=True)
        (P418 / t / f"h416_{t}_npz").mkdir(parents=True, exist_ok=True)

    design = {"schema": "jope.h416.design/v1", "design": "BCAR.mix"}
    design_p = base / "DESIGN.json"
    design_p.write_text(json.dumps(design))

    a2_cid = cids[3]
    a2_doc = {
        "schema": "jope.h418.a2-cells/v1",
        "cells": [{"cid": a2_cid, "seed": 335102, "source": "GE"}],
    }
    a2_p = base / "a2_cells.json"
    a2_p.write_text(json.dumps(a2_doc))

    tb_sha = "a" * 64
    frozen = {
        "schema": "jope.h447.frozen_fair_champion/v1",
        "wave": "H447",
        "selected_arm": "STACK_R4_E8_BASE",
        "blocks": {
            "gp_pool": "stored_h416_64_candidates",
            "matern_m7": list(GM.PRIMARY_NAMES),
            "exact_keys_k3": list(AN.H430_K3),
            "formulas_f4": list(AN.H433_F4),
            "icl": {
                "enabled": True,
                "recipe": "R4_E8",
                "col_name": "r4_e8",
                "training_bank": "fit_icl_bank.npz",
                "training_bank_sha256": tb_sha,
            },
        },
    }
    frz_p = base / "FROZEN_FAIR_CHAMPION.json"
    frz_p.write_text(json.dumps(frozen))

    S_list, AU_list, labs_list, ys_list, ya_shas, y_shas = [], [], [], [], [], []
    x0_ps = {}
    sxb_u = {}
    scr_per = {}

    for idx, c in enumerate(cids):
        n_s, n_a, f = 40, 16, 4
        Xs = rng.normal(size=(n_s, f))
        Xa = rng.normal(size=(n_a, f))
        y = 2.0 * Xs[:, 0] - 1.0 * Xs[:, 1] + rng.normal(0.0, 0.25, size=n_s)
        ya = 2.0 * Xa[:, 0] - 1.0 * Xa[:, 1] + rng.normal(0.0, 0.25, size=n_a)
        ym = float(np.mean(y))
        ba = float(np.sqrt(np.mean((ya - ym) ** 2)))
        lab = (np.arange(n_s) % 5).astype(np.int8)

        s_mat = np.column_stack([Xs, y]).astype(np.float32)
        xa_mat = Xa.astype(np.float32)
        S_list.append(s_mat)
        AU_list.append(xa_mat)
        labs_list.append(lab)
        ys_list.append(y)
        ya_shas.append(hashlib.sha256(np.ascontiguousarray(ya, np.float64).tobytes()).hexdigest())
        y_shas.append(hashlib.sha256(np.ascontiguousarray(y, np.float64).tobytes()).hexdigest())

        x0_ps[c] = [0.25 + 0.01 * (idx % 7), 0.23 + 0.01 * (idx % 7)]
        per_sxb = []
        for s_i, (s, t) in enumerate(((335101, "D1"), (335102, "D2"))):
            Q = np.column_stack([
                1.8 * Xs[:, 0] + rng.normal(0.0, 0.3, size=n_s),
                -0.9 * Xs[:, 1] + rng.normal(0.0, 0.3, size=n_s),
            ])
            AKF = np.column_stack([1.8 * Xa[:, 0], -0.9 * Xa[:, 1]])
            PA = AKF + rng.normal(0.0, 0.02, size=AKF.shape)
            npz_p = P418 / t / f"h416_{t}_npz" / f"{c}.npz"
            np.savez_compressed(
                npz_p,
                Q=Q, y=y, lab=lab, AKF=AKF, PA=PA,
                akf_nfold=np.array([5, 5]), pa_ok=np.array([True, True]),
                ya=ya, ba=np.float64(ba), ym=np.float64(ym),
                shas=np.array(["s0", "s1"]),
            )
            w = M.meta_weights("BCAR", Q, y, c)
            keep = np.flatnonzero(w > 0)
            pred, _ = M.deploy_pred("mix", w[keep], [AKF[:, j] for j in keep], [PA[:, j] for j in keep], ym, n_a)
            u_val = float(M.utility(pred, ya, ba)[0])
            rec = {
                "type": "result",
                "cid": c,
                "tag": t,
                "mode": "confirm",
                "npz": {"sha256": _sha(npz_p)},
                "outputs": {"P3+LIB3.SXB": {"value": u_val, "invalid": False, "raw": {"status": "completed"}}},
            }
            (P418 / t / "stages" / f"H416_{t}" / f"{c}__confirm.jsonl").write_text(json.dumps(rec) + "\n")
            per_sxb.append(x0_ps[c][s_i] if (c == a2_cid and s == 335102) else u_val)
        sxb_u[c] = 0.5 * sum(per_sxb)

        # Synthetic external blocks (strong true signal so stack improves cleanly)
        sig_q = 2.0 * Xs[:, 0] - 1.0 * Xs[:, 1] + rng.normal(0.0, 0.08, size=n_s)
        sig_a = 2.0 * Xa[:, 0] - 1.0 * Xa[:, 1]
        for folder, names in (
            ("cols_matern_D91", GM.PRIMARY_NAMES),
            ("cols_k3_D91", AN.H430_K3),
            ("cols_f4_D91", AN.H433_F4),
            ("cols_icl_D91", ("r4_e8",)),
        ):
            kw = {"ok": np.array(True)}
            for nm in names:
                kw[f"q_{nm}"] = sig_q + rng.normal(0.0, 0.02, size=n_s)
                kw[f"bag_{nm}"] = sig_a + rng.normal(0.0, 0.01, size=n_a)
                kw[f"ref_{nm}"] = sig_a + rng.normal(0.0, 0.01, size=n_a)
            np.savez(P / folder / f"{c}.npz", **kw)

        flagged = idx < 5
        scr_per[c] = {
            "cohort": "D91",
            "family_id": fam_map[c],
            "flag_feature": flagged,
            "tier": "copy" if flagged else "clear",
        }

    # Build 32 stored entrants on synthetic board
    board_sha = "b" * 64
    board_doc = {
        "board": "dev91-audit-v1",
        "board_sha256": board_sha,
        "n_contexts": 91,
        "n_families": 19,
        "contexts": [{"cid": c, "family_id": fam_map[c], "envelope_sha256": envs[c]} for c in cids],
    }
    (B / "BOARD.json").write_text(json.dumps(board_doc))

    required_ids = [
        AN.ANCHOR, AN.X0B, AN.XSTACK, AN.SXB_ENTRANT, AN.KUMO_LARGE, AN.KUMO_SMALL, AN.TABFM,
    ]
    all_ids = required_ids + [f"syn_model_{i:02d}.v1" for i in range(32 - len(required_ids))]
    field_vals = {}
    for m_idx, eid in enumerate(sorted(all_ids)):
        rows_txt = []
        field_vals[eid] = {}
        for c_i, c in enumerate(cids):
            if eid == AN.X0B:
                u_v = 0.5 * sum(x0_ps[c])
                ps = x0_ps[c]
            elif eid == AN.SXB_ENTRANT:
                u_v = sxb_u[c]
                ps = [u_v, u_v]
            elif eid == AN.KUMO_LARGE:
                u_v = 0.85 + 0.01 * (c_i % 5)
                ps = None
            else:
                u_v = 0.15 + 0.015 * (m_idx % 9) + 0.005 * (c_i % 6)
                ps = None
            field_vals[eid][c] = u_v
            row = {"board": "dev91-audit-v1", "board_sha256": board_sha, "cid": c, "envelope_sha256": envs[c], "model_id": eid, "utility": u_v}
            if ps is not None:
                row["per_seed"] = ps
            rows_txt.append(json.dumps(row))
        (B / "scores" / f"{eid}.jsonl").write_text("\n".join(rows_txt) + "\n")

    ordered_field = dict(sorted(field_vals.items(), key=lambda kv: kv[0] + ".jsonl"))
    ep = H.elo_panel(ordered_field, fam_map, AN.ANCHOR, seed=AN.LADDER_SEED, n_boot=AN.N_BOOT)
    (B / "ladder.json").write_text(json.dumps({"elo": ep}))

    bman_p = base / "board_audit.sha256"
    bman_lines = [f"{_sha(p)}  {p.relative_to(B)}" for p in sorted(B.rglob("*")) if p.is_file()]
    bman_p.write_text("\n".join(bman_lines) + "\n")

    lock_doc = {
        "schema": "jope.h448.lock/v1",
        "wave": "H448",
        "entrant": AN.ENTRANT,
        "seeds": [335101, 335102],
        "design": "BCAR.mix",
        "design_sha256": _sha(design_p),
        "frozen_champion_sha256": _sha(frz_p),
        "a2_cells_sha256": _sha(a2_p),
        "prereg_commit": "1" * 40,
        "inputs_sha256": {str(bman_p): _sha(bman_p)},
    }
    lock_p = base / "LOCK.json"
    lock_p.write_text(json.dumps(lock_doc))
    lock_sha = _sha(lock_p)

    receipt_p = base / "H448_LOCK_sha256.txt"
    receipt_p.write_text(f"{lock_sha}  {lock_p}\n")

    (P / "cids91.json").write_text(json.dumps({
        "schema": "jope.h448.cids91/v1", "cids": cids, "family_of": fam_map, "lock_sha256": lock_sha,
    }))
    bank_p = P / "bank_d91_search.npz"
    np.savez(
        bank_p,
        search=np.concatenate([t.ravel() for t in S_list]),
        s_shape=np.array([t.shape for t in S_list], np.int64),
        audit=np.concatenate([t.ravel() for t in AU_list]),
        a_shape=np.array([t.shape for t in AU_list], np.int64),
        lab=np.concatenate(labs_list),
        y64=np.concatenate(ys_list),
        cids=np.array(cids),
        family=np.array([fam_map[c] for c in cids]),
        cohort=np.array(["D91"] * 91),
        ya_sha256=np.array(ya_shas),
        y_sha256=np.array(y_shas),
    )
    (P / "bank_d91_search.npz.json").write_text(json.dumps({
        "audit_targets_written": False, "sha256": _sha(bank_p), "lock_sha256": lock_sha,
    }))
    (P / "SCREEN_D91_SEARCH.json").write_text(json.dumps({
        "schema": "jope.h448.screen_d91_search/v1",
        "training_bank": "fit_icl_bank.npz",
        "training_bank_sha256": tb_sha,
        "confirm_targets_keyed": False,
        "audit_rows_read": 0,
        "lock_sha256": lock_sha,
        "summary": {"contexts": 91, "flag_feature": 5, "clear_without_feature": 86},
        "per_context": scr_per,
    }))

    h445_ok = base / "READOUT_DEV_OK.json"
    h445_ok.write_text(json.dumps({"verdict": "JOINT_DEV_CONFIRMED"}))

    return {
        "P": P, "P418": P418, "B": B, "bman": bman_p, "lock": lock_p, "lock_sha": lock_sha,
        "receipt": receipt_p, "frozen": frz_p, "design": design_p, "a2": a2_p, "h445_ok": h445_ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ladder", required=True)
    a = ap.parse_args()
    root = Path(a.dir)
    root.mkdir(parents=True, exist_ok=True)

    # 1. H425 feature-only screen unit test
    test_h425_feature_only_screen(root / "t_h425")

    # 2. Full synthetic 91-ctx / 19-fam / 32-entrant scoring + ladder.py cross-check
    fx = build_synthetic_fixture(root / "fx")
    out_ok = root / "out_ok.json"
    rows_ok = root / "rows_ok.jsonl"
    rc = AN.main([
        "--pulled", str(fx["P"]),
        "--h418-pulled", str(fx["P418"]),
        "--board", str(fx["B"]),
        "--board-manifest", str(fx["bman"]),
        "--lock", str(fx["lock"]),
        "--receipt", str(fx["receipt"]),
        "--frozen", str(fx["frozen"]),
        "--design", str(fx["design"]),
        "--a2-cells", str(fx["a2"]),
        "--h445-readout", str(fx["h445_ok"]),
        "--procs", "8",
        "--desc-boot", "500",
        "--out", str(out_ok),
        "--rows", str(rows_ok),
    ])
    assert rc == 0, f"expected rc=0, got {rc}"
    doc = json.loads(out_ok.read_text())
    assert doc["status"] == "SCORED"
    assert doc["readout"]["h425_without_feature_clear"]["contexts"] == 86
    assert doc["readout"]["canonical_board_91"]["contexts"] == 91
    assert len(rows_ok.read_text().splitlines()) == 91

    # Cross-check with published tools/ladder.py (both point Elo AND 10,000-draw CI95!)
    bn = root / "board_new"
    shutil.copytree(fx["B"], bn)
    shutil.copy2(rows_ok, bn / "scores" / f"{AN.ENTRANT}.jsonl")
    env = os.environ.copy()
    env["REFBOARD_H335_DIR"] = env.get("REFBOARD_H335_DIR", "/home/ubuntu/h335-stage/code")
    subprocess.run([sys.executable, "-B", a.ladder, str(bn), "--write"], env=env, check=True, stdout=subprocess.DEVNULL)
    lad_new = json.loads((bn / "ladder.json").read_text())
    elo_lad = float(lad_new["elo"]["elo"][AN.ENTRANT]["elo"])
    ci_lad = lad_new["elo"]["elo"][AN.ENTRANT]["ci95"]
    elo_an = float(doc["readout"]["canonical_board_91"]["entrant_elo"]["elo"])
    ci_an = doc["readout"]["canonical_board_91"]["entrant_elo"]["ci95"]
    assert abs(elo_lad - elo_an) < 1e-9, f"ladder.py mismatch: {elo_lad} vs {elo_an}"
    assert max(abs(ci_lad[0] - ci_an[0]), abs(ci_lad[1] - ci_an[1])) < 1e-9, f"CI95 mismatch: {ci_lad} vs {ci_an}"

    # 3. Missing H445 readout without OWNER_DIRECT_D91 -> must refuse (exit 3, INCOMPLETE, no rows)
    out_miss = root / "out_miss.json"
    rows_miss = root / "rows_miss.jsonl"
    rc_miss = AN.main([
        "--pulled", str(fx["P"]),
        "--h418-pulled", str(fx["P418"]),
        "--board", str(fx["B"]),
        "--board-manifest", str(fx["bman"]),
        "--lock", str(fx["lock"]),
        "--receipt", str(fx["receipt"]),
        "--frozen", str(fx["frozen"]),
        "--design", str(fx["design"]),
        "--a2-cells", str(fx["a2"]),
        "--n-boot", "50",
        "--out", str(out_miss),
        "--rows", str(rows_miss),
    ])
    assert rc_miss == 3 and not rows_miss.exists()
    d_miss = json.loads(out_miss.read_text())
    assert d_miss["status"] == "INCOMPLETE" and "h445_precondition" in d_miss["gate_failed"]

    # 4. OWNER_DIRECT_D91 token in receipt -> allowed & explicitly disclosed
    rec_dir = root / "receipt_direct.txt"
    rec_dir.write_text(f"{fx['lock_sha']}  {fx['lock']}  OWNER_DIRECT_D91\n")
    out_dir = root / "out_direct.json"
    rows_dir = root / "rows_direct.jsonl"
    rc_dir = AN.main([
        "--pulled", str(fx["P"]),
        "--h418-pulled", str(fx["P418"]),
        "--board", str(fx["B"]),
        "--board-manifest", str(fx["bman"]),
        "--lock", str(fx["lock"]),
        "--receipt", str(rec_dir),
        "--frozen", str(fx["frozen"]),
        "--design", str(fx["design"]),
        "--a2-cells", str(fx["a2"]),
        "--procs", "8",
        "--n-boot", "500",
        "--desc-boot", "500",
        "--out", str(out_dir),
        "--rows", str(rows_dir),
    ])
    assert rc_dir == 0 and json.loads(out_dir.read_text())["h445_dev_status"] == "OWNER_DIRECT_D91"

    # 5. Tampered SXB replay -> must refuse (exit 3, INCOMPLETE, no rows)
    b_bad = root / "board_bad"
    shutil.copytree(fx["B"], b_bad)
    sxb_lines = (b_bad / "scores" / "h416.SXB.jsonl").read_text().splitlines()
    r0 = json.loads(sxb_lines[0])
    r0["utility"] += 0.05
    sxb_lines[0] = json.dumps(r0)
    (b_bad / "scores" / "h416.SXB.jsonl").write_text("\n".join(sxb_lines) + "\n")
    # Recompute ladder & manifest for b_bad so board_files passes and h418_sxb_replay specifically catches the drift
    _, H = T16._stats_mod()
    bd_vals = {}
    fam_m = {c["cid"]: c["family_id"] for c in json.loads((b_bad / "BOARD.json").read_text())["contexts"]}
    for sp in sorted((b_bad / "scores").glob("*.jsonl")):
        bd_vals[sp.stem] = {json.loads(x)["cid"]: float(json.loads(x)["utility"]) for x in sp.read_text().splitlines() if x.strip()}
    ep_bad = H.elo_panel(dict(sorted(bd_vals.items(), key=lambda kv: kv[0] + ".jsonl")), fam_m, AN.ANCHOR, seed=AN.LADDER_SEED, n_boot=50)
    (b_bad / "ladder.json").write_text(json.dumps({"elo": ep_bad}))
    bman_bad = root / "board_audit.sha256"
    bman_bad.write_text("\n".join(f"{_sha(p)}  {p.relative_to(b_bad)}" for p in sorted(b_bad.rglob("*")) if p.is_file()) + "\n")
    lk_bad = json.loads(fx["lock"].read_text())
    lk_bad["inputs_sha256"] = {str(bman_bad): _sha(bman_bad)}
    lock_bad_p = root / "LOCK_bad.json"
    lock_bad_p.write_text(json.dumps(lk_bad))
    lock_bad_sha = _sha(lock_bad_p)
    rec_bad_p = root / "rec_bad.txt"
    rec_bad_p.write_text(f"{lock_bad_sha}  {lock_bad_p}\n")
    p_bad = root / "pulled_bad"
    shutil.copytree(fx["P"], p_bad)
    for meta_fn in ("cids91.json", "bank_d91_search.npz.json", "SCREEN_D91_SEARCH.json"):
        mp = p_bad / meta_fn
        md = json.loads(mp.read_text())
        md["lock_sha256"] = lock_bad_sha
        mp.chmod(0o644)
        mp.write_text(json.dumps(md))

    out_bad = root / "out_bad.json"
    rows_bad = root / "rows_bad.jsonl"
    rc_bad = AN.main([
        "--pulled", str(p_bad),
        "--h418-pulled", str(fx["P418"]),
        "--board", str(b_bad),
        "--board-manifest", str(bman_bad),
        "--lock", str(lock_bad_p),
        "--receipt", str(rec_bad_p),
        "--frozen", str(fx["frozen"]),
        "--design", str(fx["design"]),
        "--a2-cells", str(fx["a2"]),
        "--h445-readout", str(fx["h445_ok"]),
        "--n-boot", "50",
        "--out", str(out_bad),
        "--rows", str(rows_bad),
    ])
    assert rc_bad == 3 and not rows_bad.exists()
    d_bad = json.loads(out_bad.read_text())
    assert d_bad["status"] == "INCOMPLETE" and "h418_sxb_replay" in d_bad["gate_failed"]

    print(json.dumps({"all_passed": True, "cases": 5, "ladder_elo_diff": abs(elo_lad - elo_an)}), flush=True)
    print("H448_SELFTEST_OK", flush=True)


if __name__ == "__main__":
    main()
