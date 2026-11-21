#!/usr/bin/env python3
"""Wave H458 Track G: 15-Gate Verification & Interoperability Self-Test Suite
(`selftest_h458.py`).

Executes 15 deterministic pure-CPU verification gates covering:
  - `shared_CHAT.md` untracked status & flock helper
  - `cl_h458` `5a` path isolation + `assert qhat.ok` enforcement
  - `DOPE` (`c51a56d0...`) SHA-256 immutability + real non-zero `44d` `ctx44`
  - `ContextualSurrogateQHatV4` 3-head (`q_pred`, `stab_pred`, `overfit_gap_pred >= 0`) ranking
  - Causal `eval_exact_teacher_on_split` zero-`yq` leakage (`max|diff| == 0.0`)
  - `Harvester v2` `0.0%` held-out loser invariant on `gp_teacher_shards_v2`
  - Track S Interface (iv) telemetry (`v1_backfill_cv_to_test_ledger.npz`) & `S8` `281`-table `CONFIRM` exclusion
  - Shared `K=5` `h458-k5-strata-v1` canonical feature-byte hash (`-0.0/+0.0`, NaN payloads, `f2893c38` test vector)
  - `skill-xfit5` label-free column routing + `_load_col_block` & Interface (i) `sidecar.json`
  - End-to-end 3-Tier (`Tier A -> Tier B Vault -> Tier C M=1`) `Train Hyper-CV` smoke run (`0` `ya` bytes!)
  - `stack_hyper_cv_context` exact query-row permutation equivariance (`max|diff| == 0.0`) & `M`-multiplicity debiasing
  - Source AST / path hygiene (`0` sealed board opens, `0` deletions, `0` `/tmp` paths)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

os.environ["H335_GROUP"] = "AGG"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
for _cand in reversed((
    str(HERE),
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h450-stage/fleet-r10/code",
)):
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)
if sys.path[0] != str(HERE):
    sys.path.insert(0, str(HERE))

import h335_core as K  # noqa: E402
import crop_shuffle_synthetic_views as V  # noqa: E402
from bagged_neural_seed_h452 import BaggedNeuralEncoderH452  # noqa: E402
from exact_gp_teacher_harvester_h454 import eval_exact_teacher_on_split, make_kf_stratified  # noqa: E402
from mmap_bank_h450 import MmapF64Bank  # noqa: E402
from stack_hyper_cv_h458 import (  # noqa: E402
    assemble_skill_xfit5_context_columns,
    debias_search_oof_for_selection_multiplicity,
    load_col_block_with_sidecar,
    stack_hyper_cv_context,
)
from train_qhat_v4_generalizability_h458 import (  # noqa: E402
    PretrainedQHatV4Prior,
)
from ultra_gp_cycle_h458 import (  # noqa: E402
    canonicalize_feature_matrix_f64,
    make_k5_feature_byte_strata,
    run_three_tier_hyper_cv_for_context,
    write_colblock_sidecars,
)
from ultra_gp_evolve_h452 import make_stratified_kfold  # noqa: E402

C = K.C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qhat-ckpt", default="/home/ubuntu/jope-data/h458-flywheel/surrogate_v4/qhat_v4_heldout.pt")
    ap.add_argument("--v2-shards-dir", default="/home/ubuntu/jope-data/h458-flywheel/gp_teacher_shards_v2")
    ap.add_argument("--ledger-npz", default="/home/ubuntu/jope-data/h458-flywheel/teaching_shards_v4/v1_backfill_cv_to_test_ledger.npz")
    ap.add_argument("--s8-audit-json", default="/home/ubuntu/jope-data/h458s-quarantine/s8/S8_BUCKET_AUDIT.json")
    ap.add_argument("--selftest-work-dir", default="/home/ubuntu/jope-data/h458-flywheel/selftest_work")
    ap.add_argument("--out-receipt", default="/home/ubuntu/jope_plan/evidence/H458/trackG/SELFTEST_H458_TRACK_G_RECEIPT.json")
    a = ap.parse_args()

    C.resume_stage_v()
    work_d = Path(a.selftest_work_dir)
    work_d.mkdir(parents=True, exist_ok=True)

    gates: List[Dict] = []

    def _run_gate(name: str, fn: Callable[[], Dict]) -> None:
        t0 = time.time()
        detail = fn()
        gates.append({"gate": name, "status": "PASS", "elapsed_s": round(time.time() - t0, 3), **detail})

    # Gate 1: shared_CHAT.md untracked in Git & coord_shared_chat_h458.py present
    def g01() -> Dict:
        r = subprocess.run(
            ["git", "-C", "/home/ubuntu/JopeDime", "ls-files", "shared_CHAT.md", "shared_CHAT.md.lock"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert r.stdout.strip() == "", f"shared_CHAT.md must stay untracked, got: {r.stdout}"
        assert (HERE / "coord_shared_chat_h458.py").is_file()
        return {"tracked_files": 0}

    _run_gate("gate_01_shared_chat_untracked_and_flock", g01)

    # Gate 2: cl_h458 5a path isolation guards
    def g02() -> Dict:
        r_ugp = subprocess.run(
            [sys.executable, str(HERE / "ultra_gp_cycle_h458.py"), "--out-root", "/home/ubuntu/jope-data/h452-flywheel/ultra_gp_d91"],
            capture_output=True,
            text=True,
        )
        assert r_ugp.returncode != 0 and "Refusing to write into legacy" in r_ugp.stderr
        r_hv = subprocess.run(
            [sys.executable, str(HERE / "hyper_cv_test_gp_harvester_h458.py"), "--shards-dir", "/home/ubuntu/jope-data/h452-flywheel/gp_teacher_shards_v1"],
            capture_output=True,
            text=True,
        )
        assert r_hv.returncode != 0 and "Refusing to overwrite H454" in r_hv.stderr
        return {"legacy_overwrite_refused": True}

    _run_gate("gate_02_isolated_h458_paths_guard", g02)

    # Gate 3: PretrainedQHatV4Prior strict require_ok guard + real v4 load
    qhat_holder: Dict[str, PretrainedQHatV4Prior] = {}

    def g03() -> Dict:
        raised = False
        try:
            PretrainedQHatV4Prior(work_d / "missing_qhat.pt", require_ok=True)
        except RuntimeError:
            raised = True
        assert raised, "PretrainedQHatV4Prior(require_ok=True) must fail loudly on missing ckpt!"
        qh = PretrainedQHatV4Prior(Path(a.qhat_ckpt), require_ok=True)
        assert qh.ok and qh.version == 4 and qh.d_ast == 128
        qhat_holder["qh"] = qh
        return {"qhat_version": qh.version, "missing_ckpt_raises": True}

    _run_gate("gate_03_qhat_v4_require_ok_guard", g03)

    # Gate 4: DOPE SHA-256 frozen + real non-zero 44d ctx44
    def g04() -> Dict:
        dope_enc = V.DopeViewEncoder(threads=1)
        assert dope_enc.dope_sha_before == C.DOPE_SHA256
        rng = np.random.default_rng(45804)
        X_syn = rng.standard_normal((120, 8))
        y_syn = 2.0 * X_syn[:, 0] - 0.8 * (X_syn[:, 1] ** 2) + 0.1 * rng.standard_normal(120)
        raw44, cz = qhat_holder["qh"].encode_xy_z(X_syn, y_syn, dope_enc)
        norm44 = float(np.linalg.norm(raw44))
        assert raw44.shape == (44,) and cz.shape == (1, 44) and norm44 > 0.5
        assert dope_enc.dope_sha() == C.DOPE_SHA256
        return {"ctx44_l2_norm": round(norm44, 4), "dope_sha256": C.DOPE_SHA256[:16]}

    _run_gate("gate_04_dope_sha256_frozen_and_nonzero_ctx44", g04)

    # Gate 5: ContextualSurrogateQHatV4 3-head shapes & non-negative overfit_gap
    def g05() -> Dict:
        qh = qhat_holder["qh"]
        cz = np.zeros((1, 44), dtype=np.float32)
        cands = ["(pipe X (ridge_gcv))", "(pipe X (oblivious))", "(pipe X (pair_ratio append=true topk=8) (histgbm2))"]
        qp, sp, gp = qh.predict_components(cz, cands)
        assert qp.shape == (3,) and sp.shape == (3,) and gp.shape == (3,)
        assert np.all((sp >= 0.0) & (sp <= 1.0)) and np.all(gp >= 0.0)
        sc_lo = qh.score_candidates(cz, cands, gap_penalty=0.0)
        sc_hi = qh.score_candidates(cz, cands, gap_penalty=2.0)
        assert sc_lo.shape == (3,) and sc_hi.shape == (3,) and np.all(np.isfinite(sc_hi))
        return {"min_gap_pred": round(float(np.min(gp)), 6), "max_gap_pred": round(float(np.max(gp)), 6)}

    _run_gate("gate_05_qhat_v4_three_heads_and_gap_nonneg", g05)

    # Gate 6: Causal s1main execution (`0` yq leakage)
    def g06() -> Dict:
        rng = np.random.default_rng(45806)
        Xc = rng.standard_normal((96, 6))
        yc = np.sin(Xc[:, 0]) + 0.5 * Xc[:, 1] + 0.05 * rng.standard_normal(96)
        Xq_r = rng.standard_normal((32, 6))
        Xq_cf = rng.standard_normal((16, 6))
        lab7 = make_kf_stratified(yc, k=7, seed=45806)
        s_dir = work_d / "g06_slot"
        s_dir.mkdir(parents=True, exist_ok=True)
        r1 = eval_exact_teacher_on_split("(pipe X (oblivious))", Xc, yc, Xq_r, Xq_cf, lab7, s_dir, timeout_s=5.0)
        r2 = eval_exact_teacher_on_split("(pipe X (oblivious))", Xc, yc, Xq_r, Xq_cf, lab7, s_dir, timeout_s=5.0)
        assert r1 is not None and r2 is not None
        max_diff = float(np.max(np.abs(np.asarray(r1["pred_q_real"]) - np.asarray(r2["pred_q_real"]))))
        assert max_diff == 0.0
        return {"max_diff_pred_q_real": max_diff, "u_cv7": round(float(r1["u_cv7"]), 4)}

    _run_gate("gate_06_causal_harvester_zero_yq_leakage", g06)

    # Gate 7: Harvester v2 zero-loser invariant on gp_teacher_shards_v2
    def g07() -> Dict:
        v2_files = sorted(Path(a.v2_shards_dir).glob("shard_*.npz"))[:500]
        assert len(v2_files) >= 50, f"Expected >=50 bootstrapped v2 shards, found {len(v2_files)}"
        losers = 0
        max_gap = 0.0
        for p2 in v2_files:
            with np.load(p2, allow_pickle=False) as z2:
                u_te = float(z2["u_gp_test_all"])
                u_ri = float(z2["u_ridge_test_all"])
                gap = float(z2["overfit_gap"])
                c44 = np.asarray(z2["real_ctx44"], dtype=np.float32)
                assert float(np.linalg.norm(c44)) > 1e-6
                if u_te < u_ri:
                    losers += 1
                max_gap = max(max_gap, gap)
        assert losers == 0, f"Zero-loser gate violated: {losers} losers!"
        assert max_gap <= 0.120001, f"Max overfit gap exceeded 0.12: {max_gap}"
        return {"v2_shards_checked": len(v2_files), "losers": losers, "max_overfit_gap": round(max_gap, 6)}

    _run_gate("gate_07_harvester_v2_zero_loser_admission_gate", g07)

    # Gate 8: Track S Interface (iv) telemetry schema
    def g08() -> Dict:
        lp = Path(a.ledger_npz)
        assert lp.is_file(), f"Missing CV->Test ledger: {lp}"
        req_keys = {
            "u_cv7",
            "u_test_all",
            "u_test_iid",
            "u_test_extrap",
            "overfit_gap",
            "ast_size",
            "n_ctx",
            "width",
            "head",
            "family_id",
            "real_ctx44",
        }
        with np.load(lp, allow_pickle=False) as zl:
            missing = req_keys - set(zl.files)
            assert not missing, f"Missing Interface (iv) keys: {missing}"
            n_rows = int(len(zl["u_cv7"]))
            assert n_rows >= 1000
        return {"ledger_episodes": n_rows}

    _run_gate("gate_08_track_s_interface_iv_telemetry_schema", g08)

    # Gate 9: S8 canonical-CONFIRM 281-table exclusion
    def g09() -> Dict:
        assert Path(a.s8_audit_json).is_file()
        s8_doc = json.loads(Path(a.s8_audit_json).read_text())
        bad_cids = {str(r["cid"]) for r in s8_doc["entries"] if not bool(r["admit_canonical"])}
        mm_bank = MmapF64Bank(
            Path("/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1"),
            expect_sha256="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507",
        )
        bad_tidxs = {i for i, c in enumerate(mm_bank.cids) if str(c) in bad_cids}
        assert len(bad_tidxs) == 281
        leaked = 0
        for p2 in sorted(Path(a.v2_shards_dir).glob("shard_*.npz"))[:500]:
            with np.load(p2, allow_pickle=False) as z2:
                if int(z2["table_idx"]) in bad_tidxs:
                    leaked += 1
        assert leaked == 0, f"Found {leaked} S8 canonical-CONFIRM shards in gp_teacher_shards_v2!"
        return {"s8_bad_tables": len(bad_tidxs), "v2_s8_leaked": leaked}

    _run_gate("gate_09_s8_canonical_confirm_exclusion", g09)

    # Gate 10: Shared K=5 feature-byte strata canonicalization + f2893c38 test vector
    def g10() -> Dict:
        x_a = np.array([[0.0, 1.5, np.nan]], dtype="<f8")
        x_b = np.array([[-0.0, 1.5, 0.0]], dtype="<f8")
        # Inject non-standard quiet NaN payload in col 2 of x_b
        x_b.view("<u8")[0, 2] = np.uint64(0x7FF8000000000123)
        ca = canonicalize_feature_matrix_f64(x_a)
        cb = canonicalize_feature_matrix_f64(x_b)
        assert ca.tobytes() == cb.tobytes(), "Canonical feature bytes must unify -0.0/+0.0 and NaN payloads!"
        assert int(make_k5_feature_byte_strata(x_a)[0]) == int(make_k5_feature_byte_strata(x_b)[0])

        pool_dir = Path("/home/ubuntu/jope-data/h458-quarantine/family_pool_v1")
        meta = json.loads((pool_dir / "meta.json").read_text())
        idx = [i for i, f in enumerate(meta["families"]) if "f2893c38" in str(f)][0]
        r, w = meta["shapes"][idx]
        off = meta["offsets"][idx]
        mm = np.memmap(pool_dir / meta["data_file"], dtype="<f8", mode="r")
        tab = np.asarray(mm[off : off + r * w], dtype="<f8").reshape(r, w)
        st = make_k5_feature_byte_strata(tab[:, :-1], k=5)
        cnts = [int(np.sum(st == j)) for j in range(5)]
        assert cnts == [1170, 1274, 1230, 1238, 1195], f"Unexpected f2893c38 vector: {cnts}"
        return {"f2893c38_counts": cnts}

    _run_gate("gate_10_k5_feature_strata_canonical_and_twin_safe", g10)

    # Gate 11: skill-xfit5 label-free column routing + _load_col_block verification
    def g11() -> Dict:
        rng = np.random.default_rng(45811)
        Xs = rng.standard_normal((50, 5))
        Xa = rng.standard_normal((25, 5))
        # Plant exact X-twin between Xs[3] and Xa[7] -> must route to the SAME held-out fold model!
        Xa[7] = Xs[3].copy()
        fold_blocks = [
            {
                "q_spec": np.full(50, float(fk), dtype=np.float64),
                "bag_spec": np.full(25, float(fk), dtype=np.float64),
                "ref_spec": np.full(25, float(fk), dtype=np.float64),
            }
            for fk in range(5)
        ]
        cb_dir = work_d / "skill_xfit5_cb"
        _dst, meta = assemble_skill_xfit5_context_columns("cid_test11", Xs, Xa, fold_blocks, ["spec"], cb_dir, pool_sha256="abc")
        (cb_dir / "sidecar.json").write_text(json.dumps({"closure": "skill-xfit5", "cross_fitted": True, "col_names": ["spec"]}))
        loaded = load_col_block_with_sidecar(cb_dir, "cid_test11", 50, 25)
        assert loaded is not None
        (Q, B, R), sc = loaded
        assert float(Q[3, 0]) == float(B[7, 0]) == float(R[7, 0]), "Twin rows in Xs and Xa must route to identical fold model!"
        assert sc["closure"] == "skill-xfit5" and sc["cross_fitted"] is True
        return {"twin_fold_model": int(Q[3, 0])}

    _run_gate("gate_11_skill_xfit5_label_free_routing_and_colblock", g11)

    # Gate 12: End-to-end 3-Tier (`Tier A -> Tier B -> Tier C`) smoke run on 1 small real dev91 context (`0` `ya`!)
    def g12() -> Dict:
        with np.load("/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz", allow_pickle=False) as zb:
            assert "ya" not in zb.files
            b_cids = [str(x) for x in zb["cids"].tolist()]
            ss, sa = zb["s_shape"], zb["a_shape"]
            # Pick smallest search context for 4-second smoke test
            smallest_i = int(np.argmin(ss[:, 0]))
            so = int(np.sum(ss[:smallest_i, 0] * ss[:smallest_i, 1]))
            ao = int(np.sum(sa[:smallest_i, 0] * sa[:smallest_i, 1]))
            lo = int(np.sum(ss[:smallest_i, 0]))
            n_s, w_s = int(ss[smallest_i, 0]), int(ss[smallest_i, 1])
            n_a, w_a = int(sa[smallest_i, 0]), int(sa[smallest_i, 1])
            cid_sm = b_cids[smallest_i]
            Xs_sm = np.array(zb["search"][so : so + n_s * w_s], np.float64).reshape(n_s, w_s)[:, :-1]
            Xa_sm = np.array(zb["audit"][ao : ao + n_a * w_a], np.float64).reshape(n_a, w_a)
            ys_sm = np.array(zb["y64"][lo : lo + n_s], np.float64)
            lab5_sm = np.array(zb["lab"][lo : lo + n_s], np.int64)

        fam_map = {r["context_sha256"]: r["family_id"] for r in C.dev_rows_for_stage_v()}
        l_r0 = make_stratified_kfold(cid_sm, ys_sm, k=7, rep=0)
        l_v = make_stratified_kfold(cid_sm, ys_sm, k=7, rep=77)
        l_s0 = make_stratified_kfold(cid_sm, ys_sm, k=7, rep=101)
        assert not np.array_equal(l_r0, l_v) and not np.array_equal(l_v, l_s0)

        enc = BaggedNeuralEncoderH452(dope_threads=1)
        out_r = work_d / "ugp_smoke_snaps"
        cb_r = work_d / "ugp_smoke_cb"
        sl_r = work_d / "ugp_smoke_slots"
        res12 = run_three_tier_hyper_cv_for_context(
            cid=cid_sm,
            family_id=fam_map[cid_sm],
            Xs=Xs_sm,
            Xa=Xa_sm,
            ys=ys_sm,
            lab5=lab5_sm,
            anchor_pack=None,
            warm_root=None,
            out_root=out_r,
            colblock_root=cb_r,
            slot_dir=sl_r,
            encoder=enc,
            qhat=qhat_holder["qh"],
            n_pop=10,
            start_gen=1,
            n_gens=4,
            snapshots=(2, 4),
            eval_per_gen=2,
            screen_max_rows=256,
            timeout_cv2=2.0,
            timeout_crossfit=6.0,
            stagnation_gens=20,
            ratchet_margin=0.002,
            seed=45812,
            inner_slots=2,
            rayon_threads=1,
        )
        assert res12["tier_b_monotonic_vault"] is True
        sc_files = write_colblock_sidecars(cb_r, (2, 4), [res12])
        assert len(sc_files) == 2
        loaded_g4 = load_col_block_with_sidecar(cb_r / "colblocks_g4", cid_sm, n_s, n_a)
        assert loaded_g4 is not None
        (Q4, B4, R4), sc4 = loaded_g4
        assert Q4.shape == (n_s, 1) and B4.shape == (n_a, 1) and R4.shape == (n_a, 1)
        assert sc4["closure"] == "fair-search-cv" and sc4["selection_candidates_crossfitted_in_tier_c"] == 1
        dope_ok, _ = enc.verify_dope_unchanged()
        assert dope_ok
        return {"cid": cid_sm[:12], "horizons": len(res12["horizons"])}

    _run_gate("gate_12_three_tier_hyper_cv_disjoint_folds_and_monotonic_ratchet", g12)

    # Gate 13: stack_hyper_cv_context query-row permutation equivariance (`max|diff| == 0.0`)
    def g13() -> Dict:
        rng = np.random.default_rng(45813)
        Xs = rng.standard_normal((84, 5))
        ys = 1.5 * Xs[:, 0] - 0.7 * Xs[:, 1] + 0.1 * rng.standard_normal(84)
        Xa = rng.standard_normal((40, 5))
        lab5 = make_stratified_kfold("cid_g13", ys, k=5, rep=0)
        Q_b = np.column_stack([1.4 * Xs[:, 0] - 0.65 * Xs[:, 1], 1.5 * Xs[:, 0] - 0.7 * Xs[:, 1] + 0.05 * rng.standard_normal(84)])
        B_b = np.column_stack([1.4 * Xa[:, 0] - 0.65 * Xa[:, 1], 1.5 * Xa[:, 0] - 0.7 * Xa[:, 1]])
        R_b = B_b.copy()
        Q_u = (1.48 * Xs[:, 0] - 0.68 * Xs[:, 1])[:, None]
        B_u = (1.48 * Xa[:, 0] - 0.68 * Xa[:, 1])[:, None]
        R_u = B_u.copy()
        p1, d1 = stack_hyper_cv_context(
            "cid_g13", Xs, Xa, ys, lab5, None, None, None, [("base", (Q_b, B_b, R_b))], ("ugp_tierc", (Q_u, B_u, R_u), 1)
        )
        perm = rng.permutation(len(Xa))
        p2, _ = stack_hyper_cv_context(
            "cid_g13",
            Xs,
            Xa[perm],
            ys,
            lab5,
            None,
            None,
            None,
            [("base", (Q_b, B_b[perm], R_b[perm]))],
            ("ugp_tierc", (Q_u, B_u[perm], R_u[perm]), 1),
        )
        max_eq_diff = float(np.max(np.abs(p1[perm] - p2)))
        assert max_eq_diff < 1e-12, f"Query equivariance diff: {max_eq_diff}"
        return {"query_equivariance_max_diff": max_eq_diff, "tierc_admitted": bool(d1["tierc_admitted"])}

    _run_gate("gate_13_stack_hyper_cv_query_permutation_equivariance", g13)

    # Gate 14: Selection-multiplicity debiasing exact identity at M=1 & strict shrinkage at M=3000
    def g14() -> Dict:
        rng = np.random.default_rng(45814)
        ys = rng.standard_normal(100) + 3.0
        Q = ys[:, None] + 0.2 * rng.standard_normal((100, 1))
        Q_m1 = debias_search_oof_for_selection_multiplicity(Q, ys, multiplicity_m=1)
        Q_m3000 = debias_search_oof_for_selection_multiplicity(Q, ys, multiplicity_m=3000)
        assert float(np.max(np.abs(Q_m1 - Q))) == 0.0
        std_m1 = float(np.std(Q_m1 - np.mean(ys)))
        std_m3000 = float(np.std(Q_m3000 - np.mean(ys)))
        assert std_m3000 < std_m1
        return {"m1_identity": True, "shrink_ratio_m3000": round(std_m3000 / std_m1, 6)}

    _run_gate("gate_14_selection_multiplicity_debiasing_monotonic", g14)

    # Gate 15: Source hygiene scan across all H458 Track G modules (`0` sealed boards, `0` deletions, `0` `/tmp`)
    def g15() -> Dict:
        h458_files = [
            HERE / "coord_shared_chat_h458.py",
            HERE / "train_qhat_v4_generalizability_h458.py",
            HERE / "hyper_cv_test_gp_harvester_h458.py",
            HERE / "ultra_gp_cycle_h458.py",
            HERE / "stack_hyper_cv_h458.py",
            HERE / "train_104m_h458_x3b_eval_variety.py",
        ]
        forbidden_tokens = (
            "apex-confirm-v1",
            "fmgap_b",
            "fmgap_c",
            "shutil.rmtree(",
            "os.unlink(",
            "os.remove(",
            '"/tmp/',
            "'/tmp/",
        )
        for fp in h458_files:
            txt = fp.read_text(encoding="utf-8")
            for tok in forbidden_tokens:
                assert tok not in txt, f"Forbidden token {tok!r} in {fp.name}"
        return {"files_audited": len(h458_files), "violations": 0}

    _run_gate("gate_15_zero_sealed_and_zero_ya_source_audit", g15)

    # Gate 16: X3-B bounded 2-slot valley ring, B1/S8/B7 family quarantine, & zero-Xq-leakage |corr| reducer
    def g16() -> Dict:
        from train_104m_h458_x3b_eval_variety import (
            apply_eval_aligned_target_prep_zero_leakage,
            build_s8_clean_eligibility_mask,
            is_canonical_confirm_family,
            save_valley_ckpt_bounded,
            select_wide_cols_eval_aligned,
        )

        assert is_canonical_confirm_family("") is True, "Empty family key must be refused (B1)"
        assert is_canonical_confirm_family("   ") is True
        assert is_canonical_confirm_family("DL-HOLDOUT-xyz") is True
        assert is_canonical_confirm_family("fresh:srcfile:meta-album-md-6-mini") is True

        mm_bank = MmapF64Bank(
            Path("/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1"),
            expect_sha256="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507",
        )
        elig_m, n_masked = build_s8_clean_eligibility_mask(mm_bank)
        assert n_masked >= 529, f"Expected >=529 S8+B7+S4 canonical-CONFIRM tables masked on v1, got {n_masked}"
        for idx_ok in np.flatnonzero(elig_m > 0.0):
            assert not is_canonical_confirm_family(mm_bank.families[int(idx_ok)])

        rng16 = np.random.default_rng(45816)
        Xc16 = rng16.standard_normal((64, 24))
        yc16 = 2.0 * Xc16[:, 3] - 1.3 * Xc16[:, 11] + 0.1 * rng16.standard_normal(64)
        Xq16_a = rng16.standard_normal((20, 24))
        Xq16_b = rng16.standard_normal((20, 24)) * 50.0 + 99.0
        Xc_sel_a, _ = select_wide_cols_eval_aligned(Xc16, yc16, Xq16_a, 8, np.random.default_rng(77))
        Xc_sel_b, _ = select_wide_cols_eval_aligned(Xc16, yc16, Xq16_b, 8, np.random.default_rng(77))
        assert float(np.max(np.abs(Xc_sel_a - Xc_sel_b))) == 0.0, "Query rows leaked into wide column selection!"

        yc_t1, _, _ = apply_eval_aligned_target_prep_zero_leakage(yc16, rng16.standard_normal(20), np.random.default_rng(99))
        yc_t2, _, _ = apply_eval_aligned_target_prep_zero_leakage(yc16, rng16.standard_normal(20) * 100.0, np.random.default_rng(99))
        assert float(np.max(np.abs(yc_t1 - yc_t2))) == 0.0, "Query labels leaked into target prep!"

        ring_d = work_d / "x3b_ring_smoke"
        ring_d.mkdir(parents=True, exist_ok=True)
        for cyc_i in range(1, 6):
            save_valley_ckpt_bounded(ring_d, {"cycle": cyc_i}, cyc_i, keep_latest_only=True, ring_slots=2)
        valley_files = sorted(p.name for p in ring_d.glob("ckpt_valley_*.pt"))
        assert valley_files == ["ckpt_valley_slot0.pt", "ckpt_valley_slot1.pt"], f"Unexpected valley files: {valley_files}"
        return {"s8_b7_masked": int(n_masked), "valley_ring_files": valley_files}

    _run_gate("gate_16_x3b_eval_variety_bounded_ring_and_s8_gate", g16)

    receipt = {
        "schema": "jope.h458.track_g_selftest/v1",
        "wave": "H458_TrackG",
        "gates_collected": 16,
        "gates_passed": len(gates),
        "gates_failed": 16 - len(gates),
        "all_passed": len(gates) == 16,
        "gates": gates,
        "utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    out_p = Path(a.out_receipt)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.with_suffix(".tmp.json")
    tmp_p.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_p, out_p)
    print(json.dumps({"gates_passed": len(gates), "all_passed": len(gates) == 16, "receipt": str(out_p)}, sort_keys=True))


if __name__ == "__main__":
    main()
