#!/usr/bin/env python3
"""Wave C (`w5_online_ast_screener`):
  eval_w5_fitsel.py must-shas --run-dir DIR --tag TBOTH --out FILE
  eval_w5_fitsel.py apply     --run-dir DIR --tag TBOTH --bank FILE --h01a-npz FILE --excl FILE --out FILE
  eval_w5_fitsel.py readout   --apply FILE --gates FILE --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("WAVE_BASE_CODE"):
    sys.path.append(os.environ["WAVE_BASE_CODE"])
import cache_h422 as CA  # noqa: E402
import meta_h416 as M  # noqa: E402
import w5_screener_core as W5  # noqa: E402

EVAL_SEED = 435_202
BOOT_SEED, N_BOOT = 435_300, 10_000
ARMS = ("W5_ONLINE_UCB", "W5_ZERO_SHOT", "H01A_ONLINE", "H01A_RAW", "RND", "ORC")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= np.mean(ra)
    rb -= np.mean(rb)
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(np.dot(ra, rb) / den) if den > 1e-12 else 0.0


def _stack_u(cid: str, a: dict, idxs: list[int]) -> float:
    sel = sorted(set(idxs))
    Q_sub = a["Q"][:, sel]
    w = CA.bcar(Q_sub, a["y"], cid)
    pred = 0.5 * (a["BAG"][:, sel] @ w) + 0.5 * (a["REF"][:, sel] @ w)
    return float(M.utility(pred, a["ya"], a["ba"])[0])


def cmd_must_shas(a) -> None:
    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists (create-once)")
    cfg = json.loads((Path(a.run_dir) / f"config_{a.tag}.json").read_text())
    must = {}
    for cid in sorted(cfg["cids"]):
        _, st = CA.stored(a.run_dir, a.tag, cid)
        must[cid] = list(st["shas"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(must, sort_keys=True))
    print(json.dumps({"contexts": len(must), "median_tboth_shas": int(np.median([len(v) for v in must.values()]))}), flush=True)


def cmd_apply(a) -> None:
    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists (create-once)")
    t0 = time.time()

    excl = json.loads(Path(a.excl).read_text())
    tier_of = {cid: info["tier"] for cid, info in excl["contexts"].items()}

    with np.load(a.bank, allow_pickle=False) as b:
        b_cids = [str(c) for c in b["cids"].tolist()]
        s_shape = b["s_shape"]
        search = b["search"]
        y64 = b["y64"]
        so = lo = 0
        phi_by_cid = {}
        for i, cid in enumerate(b_cids):
            n, w = (int(v) for v in s_shape[i])
            S = np.array(search[so : so + n * w], np.float64).reshape(n, w)
            Xs = np.array(S[:, :-1], np.float64, copy=True)
            ys = np.array(y64[lo : lo + n], np.float64, copy=True)
            phi_by_cid[cid] = W5.compute_dataset_meta_profile(Xs, ys)
            so += n * w
            lo += n

    with np.load(a.h01a_npz, allow_pickle=False) as hz:
        h_cids = [str(c) for c in hz["cids"].tolist()]
        h_fams = [str(f) for f in hz["families"].tolist()]
        offsets = hz["offsets"].astype(np.int64)
        shas_all = [str(s) for s in hz["shas"].tolist()]
        dsls_all = [str(d) for d in hz["dsls"].tolist()]
        su_all = hz["search_u"].astype(np.float64)
        h01_all = hz["h01a_score"].astype(np.float64)

    ctx_pool = {}
    for idx, cid in enumerate(h_cids):
        s0, s1 = int(offsets[idx]), int(offsets[idx + 1])
        ctx_pool[cid] = {
            "cid": cid,
            "family": h_fams[idx],
            "tier": tier_of[cid],
            "clean": tier_of[cid] != "copy",
            "phi": phi_by_cid[cid],
            "shas": shas_all[s0:s1],
            "dsls": dsls_all[s0:s1],
            "search_u": su_all[s0:s1],
            "h01a": h01_all[s0:s1],
        }

    # Strict 5-fold cross-family CV trained ONLY on clean (non-copy) families!
    clean_fams = sorted({v["family"] for v in ctx_pool.values() if v["clean"]})
    fam_fold = {f: (i % 5) for i, f in enumerate(clean_fams)}
    for v in ctx_pool.values():
        if v["family"] not in fam_fold:
            fam_fold[v["family"]] = int(hashlib.sha256(v["family"].encode()).hexdigest()[:4], 16) % 5

    surr_by_fold = {}
    for fold in range(5):
        tr_records = [
            (v["family"], v["phi"], v["dsls"], v["search_u"])
            for v in sorted(ctx_pool.values(), key=lambda x: x["cid"])
            if v["clean"] and fam_fold[v["family"]] != fold
        ]
        # Verify strict family disjointness
        tr_fam_set = {r[0] for r in tr_records}
        va_fam_set = {v["family"] for v in ctx_pool.values() if fam_fold[v["family"]] == fold}
        assert not (tr_fam_set & va_fam_set), "Cross-family leak!"
        surr_by_fold[fold] = W5.GrammarCompleteASTSurrogate(n_members=5).fit(tr_records, seed=4350 + fold)

    per_context = {}
    for cid in sorted(ctx_pool.keys()):
        v = ctx_pool[cid]
        fold = fam_fold[v["family"]]
        surr = surr_by_fold[fold]
        dsls = v["dsls"]
        u_vec = v["search_u"]
        h01_vec = v["h01a"]
        shas = v["shas"]
        n_c = len(dsls)

        rng = np.random.default_rng(M.stream_seed(EVAL_SEED, "MEMO_SPLIT", cid))
        perm = rng.permutation(n_c)
        n_obs = min(24, max(8, n_c // 3))
        obs_idx = perm[:n_obs].tolist()
        rem_idx = perm[n_obs:].tolist()
        obs_dsls = [dsls[i] for i in obs_idx]
        rem_dsls = [dsls[i] for i in rem_idx]
        obs_u = u_vec[obs_idx]
        rem_u = u_vec[rem_idx]
        k_pick = min(8, len(rem_idx))

        w5_on_ucb, w5_on_mean = surr.adapt_online_and_score(v["phi"], obs_dsls, obs_u, rem_dsls)
        w5_zs_mean, w5_zs_std = surr.score_candidates(v["phi"], rem_dsls)
        w5_zs_ucb = w5_zs_mean + 0.20 * w5_zs_std
        h01_on = W5.h01a_head_residual_adapt(h01_vec[obs_idx], obs_dsls, obs_u, h01_vec[rem_idx], rem_dsls)
        h01_raw = h01_vec[rem_idx]
        rnd_sc = rng.standard_normal(len(rem_idx))

        picks_memo = {
            "W5_ONLINE_UCB": W5.select_diverse_topk(w5_on_ucb, rem_dsls, k_pick, lam_div=0.03),
            "W5_ZERO_SHOT": W5.select_diverse_topk(w5_zs_ucb, rem_dsls, k_pick, lam_div=0.03),
            "H01A_ONLINE": np.argsort(-h01_on)[:k_pick].tolist(),
            "H01A_RAW": np.argsort(-h01_raw)[:k_pick].tolist(),
            "RND": np.argsort(-rnd_sc)[:k_pick].tolist(),
            "ORC": np.argsort(-rem_u)[:k_pick].tolist(),
        }
        rhos_memo = {
            "W5_ONLINE_UCB": _spearman(w5_on_mean, rem_u),
            "W5_ZERO_SHOT": _spearman(w5_zs_mean, rem_u),
            "H01A_ONLINE": _spearman(h01_on, rem_u),
            "H01A_RAW": _spearman(h01_raw, rem_u),
            "RND": _spearman(rnd_sc, rem_u),
            "ORC": 1.0,
        }
        top8_search_u = {arm: float(np.max(rem_u[idxs])) for arm, idxs in picks_memo.items()}

        # Endpoint 2: Held-out AUDIT stack utility on H420_FIT-SEL TBOTH (12 seed + 16 screened out of m0)
        rec, st = CA.stored(a.run_dir, a.tag, cid)
        sha_to_pool = {s: idx_p for idx_p, s in enumerate(shas)}
        tb_shas = st["shas"]
        valid_tb = [j for j, s in enumerate(tb_shas) if s in sha_to_pool]
        rng_tb = np.random.default_rng(M.stream_seed(EVAL_SEED, "TBOTH_SPLIT", cid))
        perm_tb = rng_tb.permutation(valid_tb).tolist()
        n_seed_tb = min(12, max(6, len(perm_tb) // 4))
        seed_tb = perm_tb[:n_seed_tb]
        cand_tb = perm_tb[n_seed_tb:]
        k_scr_tb = min(16, len(cand_tb))

        if k_scr_tb > 0:
            s_pidx = [sha_to_pool[tb_shas[j]] for j in seed_tb]
            c_pidx = [sha_to_pool[tb_shas[j]] for j in cand_tb]
            s_dsls = [dsls[i] for i in s_pidx]
            c_dsls = [dsls[i] for i in c_pidx]
            s_u = u_vec[s_pidx]

            tb_w5_on, _ = surr.adapt_online_and_score(v["phi"], s_dsls, s_u, c_dsls)
            tb_w5_zs_m, tb_w5_zs_s = surr.score_candidates(v["phi"], c_dsls)
            tb_h01_on = W5.h01a_head_residual_adapt(h01_vec[s_pidx], s_dsls, s_u, h01_vec[c_pidx], c_dsls)
            tb_h01_raw = h01_vec[c_pidx]
            tb_rnd = rng_tb.standard_normal(len(cand_tb))
            tb_orc = u_vec[c_pidx]

            sel_tb = {
                "W5_ONLINE_UCB": seed_tb + [cand_tb[i] for i in W5.select_diverse_topk(tb_w5_on, c_dsls, k_scr_tb, lam_div=0.03)],
                "W5_ZERO_SHOT": seed_tb + [cand_tb[i] for i in W5.select_diverse_topk(tb_w5_zs_m + 0.20 * tb_w5_zs_s, c_dsls, k_scr_tb, lam_div=0.03)],
                "H01A_ONLINE": seed_tb + [cand_tb[i] for i in np.argsort(-tb_h01_on)[:k_scr_tb].tolist()],
                "H01A_RAW": seed_tb + [cand_tb[i] for i in np.argsort(-tb_h01_raw)[:k_scr_tb].tolist()],
                "RND": seed_tb + [cand_tb[i] for i in np.argsort(-tb_rnd)[:k_scr_tb].tolist()],
                "ORC": seed_tb + [cand_tb[i] for i in np.argsort(-tb_orc)[:k_scr_tb].tolist()],
            }
            audit_u = {arm: _stack_u(cid, st, idxs) for arm, idxs in sel_tb.items()}
        else:
            base_u = float(rec["outputs"]["P3+LIB3.SXB"]["value"])
            audit_u = {arm: base_u for arm in ARMS}

        per_context[cid] = {
            "cid": cid,
            "family": v["family"],
            "tier": v["tier"],
            "clean": v["clean"],
            "fold": fold,
            "n_pool": n_c,
            "n_tboth": len(valid_tb),
            "spearman_rho": rhos_memo,
            "top8_search_u": top8_search_u,
            "audit_stack_u": audit_u,
            "full_sxb_audit_u": float(rec["outputs"]["P3+LIB3.SXB"]["value"]),
        }

    out = {
        "schema": "jope.w5_screener.apply/v1",
        "n_total": len(per_context),
        "n_clean": sum(1 for r in per_context.values() if r["clean"]),
        "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
        "h01a_npz_sha256": hashlib.sha256(Path(a.h01a_npz).read_bytes()).hexdigest(),
        "excl_sha256": hashlib.sha256(Path(a.excl).read_bytes()).hexdigest(),
        "wall_s": round(time.time() - t0, 2),
        "per_context": per_context,
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(out, sort_keys=True))
    tmp.rename(out_p)
    print(json.dumps({k: out[k] for k in ("n_total", "n_clean", "wall_s")}), flush=True)


def _summarize_subset(rows: list[dict], T16, H) -> dict:
    cids = [r["cid"] for r in rows]
    unit = {r["cid"]: r["family"] for r in rows}
    mean_rho = {arm: round(float(np.mean([r["spearman_rho"][arm] for r in rows])), 4) for arm in ARMS}

    search_vals = {arm: {r["cid"]: r["top8_search_u"][arm] for r in rows} for arm in ARMS}
    audit_vals = {arm: {r["cid"]: r["audit_stack_u"][arm] for r in rows} for arm in ARMS}

    pairs = (
        ("W5_ONLINE_UCB", "H01A_ONLINE"),
        ("W5_ONLINE_UCB", "RND"),
        ("W5_ZERO_SHOT", "H01A_RAW"),
        ("W5_ZERO_SHOT", "RND"),
        ("H01A_ONLINE", "RND"),
        ("ORC", "W5_ONLINE_UCB"),
        ("ORC", "H01A_ONLINE"),
    )
    s_pairs = {f"{x}_vs_{y}": T16.pair_stats(np, H, search_vals[x], search_vals[y], unit) for x, y in pairs}
    a_pairs = {f"{x}_vs_{y}": T16.pair_stats(np, H, audit_vals[x], audit_vals[y], unit) for x, y in pairs}
    an, point, boot, _ = T16.elo_boot(np, H, audit_vals, unit, "H01A_ONLINE", BOOT_SEED, N_BOOT)
    a_elo = {f"{x}_vs_{y}": T16.delta_elo(np, an, point, boot, x, y) for x, y in pairs}

    return {
        "n_contexts": len(cids),
        "n_families": len(set(unit.values())),
        "mean_spearman_rho": mean_rho,
        "family_mean_top8_search_u": {arm: round(float(H.family_mean(search_vals[arm], unit)), 4) for arm in ARMS},
        "family_mean_audit_stack_u": {arm: round(float(H.family_mean(audit_vals[arm], unit)), 4) for arm in ARMS},
        "search_top8_pairs": s_pairs,
        "audit_stack_pairs": a_pairs,
        "audit_stack_delta_elo": a_elo,
    }


def cmd_readout(a) -> None:
    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists (create-once)")
    gates = json.loads(Path(a.gates).read_text())
    if not gates.get("pass"):
        raise SystemExit("REFUSING readout: integrity gates did not pass")

    import train_h416 as T16

    _, H = T16._stats_mod()
    ap_data = json.loads(Path(a.apply).read_text())
    rows_all = [ap_data["per_context"][c] for c in sorted(ap_data["per_context"])]
    rows_clean = [r for r in rows_all if r["clean"]]

    clean_sum = _summarize_subset(rows_clean, T16, H)
    all_sum = _summarize_subset(rows_all, T16, H)

    # Holm-Bonferroni FWER = 0.025 across the 2 primary clean_88 tests:
    #   P1: W5_ONLINE_UCB vs H01A_ONLINE on top8_search_u
    #   P2: W5_ONLINE_UCB vs H01A_ONLINE on audit_stack_u
    p1 = float(clean_sum["search_top8_pairs"]["W5_ONLINE_UCB_vs_H01A_ONLINE"]["signflip_p"])
    p2 = float(clean_sum["audit_stack_pairs"]["W5_ONLINE_UCB_vs_H01A_ONLINE"]["signflip_p"])
    pass1 = bool(clean_sum["search_top8_pairs"]["W5_ONLINE_UCB_vs_H01A_ONLINE"]["pass"])
    pass2 = bool(clean_sum["audit_stack_pairs"]["W5_ONLINE_UCB_vs_H01A_ONLINE"]["pass"])
    ordered = sorted([("P1_search_top8_vs_H01A", p1, pass1), ("P2_audit_stack_vs_H01A", p2, pass2)], key=lambda t: t[1])
    first_ok = ordered[0][1] < 0.0125 and ordered[0][2]
    second_ok = first_ok and ordered[1][1] < 0.025 and ordered[1][2]
    holm = {ordered[0][0]: bool(first_ok), ordered[1][0]: bool(second_ok)}

    verdict = "W5_SCREENER_CONFIRMED" if any(holm.values()) else "W5_SCREENER_NULL"
    res = {
        "schema": "jope.w5_screener.readout/v1",
        "verdict": verdict,
        "holm_primary_clean88": holm,
        "clean_non_copy_88": clean_sum,
        "all_fitsel_106": all_sum,
        "gates_pass": gates["pass"],
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(res, indent=2, sort_keys=True))
    tmp.rename(out_p)
    out_p.chmod(0o444)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "holm_primary_clean88": holm,
                "clean88_rho": clean_sum["mean_spearman_rho"],
                "clean88_top8_search_u": clean_sum["family_mean_top8_search_u"],
                "clean88_audit_stack_u": clean_sum["family_mean_audit_stack_u"],
                "P1_search_vs_H01A": clean_sum["search_top8_pairs"]["W5_ONLINE_UCB_vs_H01A_ONLINE"],
                "P2_audit_vs_H01A": clean_sum["audit_stack_pairs"]["W5_ONLINE_UCB_vs_H01A_ONLINE"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("must-shas")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", default="TBOTH")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_must_shas)

    p = sub.add_parser("apply")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", default="TBOTH")
    p.add_argument("--bank", required=True)
    p.add_argument("--h01a-npz", required=True)
    p.add_argument("--excl", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("readout")
    p.add_argument("--apply", required=True)
    p.add_argument("--gates", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_readout)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
