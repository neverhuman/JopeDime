"""One H285 CPU attempt. Selection is frozen before outcome arrays are read."""

from pathlib import Path
import hashlib
import json
import time
import sys
import importlib.util
import math
import gc
import numpy as np
from dependencies import ROOT, topk, core
import h124
import fresh_inputs as fi
from analysis_h285 import summarize, decide, random_valid_picks, NULL_SEEDS

DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
STOP = 1793759400
S1 = Path(
    "/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod/contexts"
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def check():
    core.check_cooperative_pause()
    if time.time() >= STOP:
        raise RuntimeError("compute deadline")
    h124.check_rss()


def load_net(checkpoint):
    import torch

    spec = importlib.util.spec_from_file_location(
        "h285_cpu_fused", "/home/ubuntu/jope-data/h50/input/cpu_exact_fused.py"
    )
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert ck["dope_sha256"] == DOPE
    net.load_state_dict(ck["state_dict"], strict=True)
    for p in net.parameters():
        p.requires_grad_(False)
    guard = h124.source_module(
        "python/src/jope/north_star/dope_freeze_guard.py", "h285_runtime_guard"
    )
    guard.lock_dope_weights_and_eval_mode(net)
    net.eval()
    assert guard.snapshot_dope_weights(net).composite_sha256 == DOPE
    return net, guard


def features(net, ws, table, prog):
    import torch
    import fresh_features as ff

    nt, _, width = fi.struct.unpack_from("<III", table, 4)
    values = (
        np.frombuffer(table, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
    )
    tok, mask, d, lm, ctx = ff.extract_single_table_features(
        net, ws, values[:, :-1], values[:, -1], torch.device("cpu")
    )
    ap, sp = ff.compute_candidate_priors(net, prog, lm, torch.device("cpu"))
    import rust_evaluator as evaluator

    pj = (
        prog.astype(np.float64) @ evaluator.get_w1_sign_matrix() / np.sqrt(412)
    ).astype(np.float32)
    return dict(
        tokens_10x32=tok,
        mask_10=mask,
        dope629=d,
        lm48_z=lm,
        ctx_feat_584=ctx,
        base_prior_cost=ap,
        sos_prior=sp,
        proj16=pj,
    )


def predict(net, pool, f):
    import torch

    prog = pool["program"]
    n = len(prog)
    with torch.no_grad():
        out = net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((1, 584)),
            prog_412=torch.from_numpy(prog)[None],
            proj_16=torch.from_numpy(f["proj16"])[None],
            base_prior_cost=torch.zeros((1, n)),
            scale=torch.tensor([0.4208]),
            sos_prior=torch.zeros((1, n, 2)),
            mm_tokens_10x32=torch.from_numpy(f["tokens_10x32"])[None],
            mm_mask_10=torch.from_numpy(f["mask_10"])[None].bool(),
            revealed_indices=torch.arange(4)[None],
            revealed_raw_f=torch.as_tensor(pool["fitness"][:4], dtype=torch.float32)[
                None
            ],
            revealed_gate=torch.as_tensor(pool["gate"][:4], dtype=torch.float32)[None],
        )
    return out["total_cost"][0].numpy().astype(np.float64)


def regression_builder(net, ws, out):
    """One reused VAL context proves table/pool reconstruction and numeric1 H58 context features."""
    row = json.loads((ROOT / "inputs/selected.json").read_text())["dev"][0]
    cid = row["context_sha256"]
    adm = {
        r["context_sha256"]: r
        for r in json.loads(
            Path(
                "/home/ubuntu/jope-evidence/dope-lift-20261021/deep-learning-dl03/admitted-input-contexts.json"
            ).read_text()
        )
    }
    a = adm[cid]
    table = fi.train_only_table(Path(a["context_json_path"]).parent / "teacher.jtt1.gz")
    assert (
        table
        == (
            Path("/home/ubuntu/jope-data/h50/input/tables") / (cid + ".jtt1")
        ).read_bytes()
    )
    names = json.loads(
        Path(
            "/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study/program-feature-names.json"
        ).read_text()
    )
    pool = fi.extract_pool(
        Path(a["final_path"]).parent / "summary.tar.gz", fi.features, names
    )
    with np.load("/home/ubuntu/jope-data/h50/input/pools/" + cid + ".npz") as ref:
        for k in ["program", "fitness", "gate", "program_sha256"]:
            assert np.array_equal(pool[k], ref[k]), k
    f = features(net, ws, table, pool["program"])
    checks = {}
    with np.load("/home/ubuntu/jope-data/h58/cache/context_features.npz") as z:
        i = list(z["cids"]).index(cid)
        for k in ["tokens_10x32", "mask_10", "dope629", "lm48_z", "ctx_feat_584"]:
            checks[k] = dict(
                bit_exact=bool(np.array_equal(f[k], z[k][i])),
                max_abs=float(
                    np.max(np.abs(f[k].astype(float) - z[k][i].astype(float)))
                ),
            )
    with np.load("/home/ubuntu/jope-data/h58/cache/pool_priors.npz") as z:
        i = list(z["cids"]).index(cid)
        lo, hi = map(int, z["offsets"][i : i + 2])
        for k in ["base_prior_cost", "sos_prior", "proj16"]:
            checks[k] = dict(
                bit_exact=bool(np.array_equal(f[k], z[k][lo:hi])),
                max_abs=float(np.max(np.abs(f[k] - z[k][lo:hi]))),
            )
    write(
        out / "builder-regression.json",
        dict(cid=cid, table_bit_exact=True, pool_bit_exact=True, checks=checks),
    )
    # H58 cache was numeric4; this run is numeric1. The approved CPU cost regression remains exact;
    # fresh features must be numerically stable (absolute <=1e-5 or relative <=1e-5).
    with np.load("/home/ubuntu/jope-data/h58/cache/context_features.npz") as z:
        i = list(z["cids"]).index(cid)
        for k in ["tokens_10x32", "mask_10", "dope629", "lm48_z", "ctx_feat_584"]:
            assert np.allclose(f[k], z[k][i], atol=1e-5, rtol=1e-5), k
    with np.load("/home/ubuntu/jope-data/h58/cache/pool_priors.npz") as z:
        i = list(z["cids"]).index(cid)
        lo, hi = map(int, z["offsets"][i : i + 2])
        for k in ["base_prior_cost", "sos_prior", "proj16"]:
            assert np.allclose(f[k], z[k][lo:hi], atol=1e-5, rtol=1e-5), k
    return names


def main():
    for path, digest in json.loads(
        (ROOT / "source/runtime-pins.json").read_text()
    ).items():
        assert sha(path) == digest, path
    for name, digest in json.loads((ROOT / "source/manifest.json").read_text())[
        "source_sha256"
    ].items():
        assert sha(ROOT / "source" / name) == digest, name
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    h124.REPO = Path("/home/ubuntu/JopeDime")
    h124.RSS_LIMIT_KIB = 4 * 1024 * 1024
    source = json.loads((ROOT / "prereg/published.json").read_text())["source_commit"]
    out = ROOT / "attempts/attempt1"
    out.mkdir()
    write(
        out / "launch.json",
        dict(
            source_commit=source,
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            numeric_threads=1,
            aggregate_thread_limit=4,
            rss_limit_bytes=4 * 1024**3,
            affinity=[48, 50, 51, 52],
            nice=19,
            compute_stop=STOP,
            attempt=1,
            neural_updates=0,
        ),
    )
    sys.addaudithook(__import__("data").audit_hook)
    from cpu_score import regression_and_swa_val

    val = regression_and_swa_val(out / "regression")
    check()
    gc.collect()
    h124.release_memory()
    net, guard = load_net(ROOT / "inputs/checkpoints/round-3175.pt")
    swa, sg = load_net(ROOT / "inputs/checkpoints/swa.pt")
    import fresh_features as ff

    ws = ff.base.FusedLandmarkWorkspace(num_threads=1)
    names = regression_builder(net, ws, out)
    cohort = json.loads((ROOT / "inputs/fresh-selected.json").read_text())
    pins = {
        r["cid"]: r
        for r in json.loads((ROOT / "inputs/fresh-metadata-receipts.json").read_text())[
            "rows"
        ]
    }
    for d in ["pools", "tables", "features", "costs"]:
        (out / d).mkdir()
    rows = []
    packets = []
    excluded = []
    start = time.monotonic()
    planned = len(cohort)
    for index, e in enumerate(cohort):
        if index >= planned:
            break
        check()
        cx = e["context"]
        cid = cx["context_sha256"]
        p = pins[cid]
        teacher = Path(cx["context_dir"]) / "teacher.jtt1.gz"
        tar = S1 / cid / "summary.tar.gz"
        assert (
            sha(teacher) == p["teacher_compressed_sha256"]
            and sha(tar) == p["summary_sha256"]
        )
        table = fi.train_only_table(teacher)
        (out / "tables" / f"{cid}.jtt1").write_bytes(table)
        pool = fi.extract_pool(tar, fi.features, names)
        np.savez_compressed(out / "pools" / f"{cid}.npz", **pool)
        if len(pool["program"]) < 8:
            excluded.append(
                dict(
                    cid=cid,
                    reason="fewer_than_8_candidates",
                    family_key=e["family_key"],
                )
            )
        else:
            f = features(net, ws, table, pool["program"])
            np.savez_compressed(out / "features" / f"{cid}.npz", **f)
            rc = predict(net, pool, f)
            sc = predict(swa, pool, f)
            assert np.isfinite(rc).all() and np.isfinite(sc).all()
            np.savez_compressed(out / "costs" / f"{cid}.npz", r3175=rc, swa=sc)
            order = np.argsort(rc.astype(np.float32), kind="mergesort")
            ix, X = topk.build_topk(
                rc.astype(np.float32),
                f["base_prior_cost"],
                pool["program"][order[:8]],
                core.compute_sem24_numpy(pool["program"]),
                f["proj16"],
                f["lm48_z"],
                f["dope629"],
                8,
            )
            assert X.shape == (8, 568) and np.isfinite(X).all()
            packets.append((cid, ix, X, rc[ix], sc[ix]))
            rows.append(
                dict(
                    cid=cid,
                    family_key=e["family_key"],
                    dataset_id=e["dataset_id"],
                    n_candidates=len(rc),
                )
            )
        # Only wall time and completed index enter the one permitted outcome-blind prefix decision.
        if index == 9:
            elapsed = time.monotonic() - start
            per = elapsed / 10
            available = max(0, STOP - time.time() - 900)
            planned = min(60, max(10, 10 + math.floor(available / max(1, 1.5 * per))))
            write(
                out / "prefix-lock.json",
                dict(
                    smoke_contexts=10,
                    smoke_seconds=elapsed,
                    seconds_per_context=per,
                    remaining_seconds_less_900s_reserve=available,
                    safety_factor=1.5,
                    planned_prefix=planned,
                    family_keys=[r["family_key"] for r in cohort[:planned]],
                    utility_or_model_outcomes_used=False,
                ),
            )
        print(
            json.dumps(dict(phase="fresh_cpu", completed=index + 1, planned=planned)),
            flush=True,
        )
        h124.release_memory()
    assert (
        guard.snapshot_dope_weights(net).composite_sha256 == DOPE
        and sg.snapshot_dope_weights(swa).composite_sha256 == DOPE
        and not torch.cuda.is_initialized()
    )
    write(
        out / "dope.json",
        dict(before=DOPE, after=DOPE, neural_updates=0, CUDA_initialized=False),
    )
    del net, swa, ws
    gc.collect()
    h124.release_memory()
    write(
        out / "selected.json",
        dict(rows=rows, excluded=excluded, planned_prefix=planned),
    )
    if not packets:
        raise RuntimeError("no scorable fresh contexts")
    cids = [t[0] for t in packets]
    X = np.stack([t[2] for t in packets])
    indices = np.stack([t[1] for t in packets])
    np.savez_compressed(
        out / "fresh-top8-no-picks.npz",
        cids=np.asarray(cids),
        X=X,
        indices=indices,
        r3175_costs=np.stack([t[3] for t in packets]),
        swa_costs_at_r3175_top8=np.stack([t[4] for t in packets]),
    )
    from custody import put_file

    receipt = put_file(out / "fresh-top8-no-picks.npz", "fresh/fresh-top8-no-picks.npz")
    receipt.update(
        wave="H285",
        source_commit=source,
        cohort_sha256=sha(ROOT / "inputs/fresh_cohort_v1.json"),
        half="ODD",
        families=len(rows),
        features=568,
        contains_picks=False,
        selection="r3175 ascending mergesort top8",
        rows=rows,
    )
    write(out / "fresh-pools-receipt.json", receipt)
    # Atomic single-wave handoff; consumers get only costs/features and identity metadata.
    shared = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_pools_H285.json")
    assert not shared.exists()
    write(shared, receipt)
    check()
    import lightgbm as lgb
    import veto

    manifest = json.loads((ROOT / "export/export_manifest.json").read_text())
    picks = []
    for cell in manifest["cells"]:
        rank = lgb.Booster(model_file=str(ROOT / "export" / cell["ranker"]["file"]))
        gate = lgb.Booster(model_file=str(ROOT / "export" / cell["veto"]["file"]))
        score = rank.predict(
            X.reshape(-1, 568),
            num_iteration=cell["selected_n_estimators"],
            num_threads=1,
        ).reshape(-1, 8)
        prob = gate.predict(X.reshape(-1, 568), num_threads=1).reshape(-1, 8)
        masked = veto.mask(score, prob, cell["selected_tau"])
        picks.append(
            [core.select_from_top4_scores(v, cell["strict_margin"]) for v in masked]
        )
    # Selector predictions are sealed before decoding full-pool utility or invalid arrays for analysis.
    picks = np.asarray(picks)
    np.savez_compressed(
        out / "frozen-predictions.npz", cids=np.asarray(cids), slots=picks
    )
    u = np.empty((3, len(rows), 4))
    bad = np.empty_like(u, dtype=bool)
    program_picks = np.empty_like(u, dtype=int)
    gates = []
    utilities = []
    swa_picks = []
    for i, cid in enumerate(cids):
        with np.load(out / "pools" / f"{cid}.npz") as z:
            fitness = z["fitness"]
            g = z["gate"]
            utility = np.where(g, -fitness, -2.0)
        with np.load(out / "costs" / f"{cid}.npz") as z:
            sp = int(np.argmin(z["swa"]))
        utilities.append(utility)
        gates.append(g[indices[i]])
        swa_picks.append(sp)
        for s in range(3):
            pp = [int(indices[i, 0]), int(indices[i, picks[s, i]]), 0, sp]
            for arm in (0, 1, 3):
                program_picks[s, i, arm] = pp[arm]
                u[s, i, arm] = utility[pp[arm]]
                bad[s, i, arm] = not bool(g[pp[arm]])
    for s, seed in enumerate(NULL_SEEDS):
        random = random_valid_picks(cids, gates, seed)
        for i, j in enumerate(random):
            pp = int(indices[i, j])
            program_picks[s, i, 2] = pp
            u[s, i, 2] = utilities[i][pp]
            bad[s, i, 2] = not bool(gates[i][j])
    families = [r["family_key"] for r in rows]
    stats = summarize(u, bad, program_picks, families)
    decision = decide(stats)
    np.savez_compressed(
        out / "outcomes.npz",
        cids=np.asarray(cids),
        families=np.asarray(families),
        utility=u,
        invalid=bad,
        program_picks=program_picks,
    )
    metrics = dict(
        wave="H285",
        status="COMPLETE",
        source_commit=source,
        verdict=decision["verdict"],
        decision=decision,
        fresh=stats,
        reused_DLVal=val,
        neural_updates=0,
        compute_finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write(out / "metrics-full.json", metrics)
    print(json.dumps(dict(status="COMPLETE", verdict=decision["verdict"])), flush=True)


if __name__ == "__main__":
    main()
