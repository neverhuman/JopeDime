"""Single-attempt H168 executor: H156 W10 and W20 r882 continuation trajectories under OLD strict vs NEW prospective remedy inner-B selectors."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np

import runtime as rt
import science as sc

HERE = Path(__file__).resolve().parent


def save(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    data = (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def verify(pins):
    if rt.sha(HERE / "PREREG.md") != pins["prereg_sha256"]:
        raise RuntimeError("PREREG.md modified after manifest pin")
    if rt.sha(HERE / "fold_manifest.json") != pins["fold_manifest_sha256"]:
        raise RuntimeError("fold_manifest.json modified after manifest pin")
    for rel, expected in pins.get("executor_sha256", {}).items():
        if rt.sha(HERE / rel) != expected:
            raise RuntimeError(f"executor modified after pin: {rel}")
    for kind in ("inputs", "assets"):
        for p, h in pins[kind].items():
            if rt.sha(p) != h:
                raise RuntimeError(f"pinned {kind} hash mismatch: {p}")
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"package version mismatch: {name}")


def selected():
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train = packet["train"]
    val = packet["dev"]
    if len(train) != 1219 or len(val) != 91:
        raise RuntimeError(f"unexpected split counts: {len(train)} / {len(val)}")
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if len({r["context_sha256"] for r in rows}) != len(rows):
            raise RuntimeError("duplicate context")
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("protected row")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("train/val family overlap")
    return train, val


def baseline_gate(values, ref):
    raw_expected = ref["utility"]["per_context"]
    expected = np.asarray(
        [x["utility"] if isinstance(x, dict) else float(x) for x in raw_expected],
        dtype=np.float64,
    )
    actual = np.asarray(values, dtype=np.float64)
    if (
        expected.shape != (91,)
        or actual.shape != (91,)
        or not np.isfinite(actual).all()
    ):
        raise RuntimeError("91 finite contexts required")
    err = float(np.max(np.abs(actual - expected)))
    mean_err = abs(float(actual.mean()) - float(ref["best_utility"]))
    if err != 0.0 or mean_err != 0.0:
        raise RuntimeError(f"r882 baseline exact parity failed: {err}, {mean_err}")
    return dict(
        matches=91,
        max_context_error=err,
        mean_error=mean_err,
        mean=float(actual.mean()),
        status="PASS",
    )


def verify_native_score_bit_identity(uncached_raw, cached_raw, rows):
    """Verifies per-candidate total_cost bit-identity (exact float32 bytes and torch.equal) for native W10 r882 across all 91 DL-Val contexts."""
    import torch

    if len(rows) != 91:
        raise RuntimeError("91 DL-Val contexts required for native score bit-identity")
    total_candidates = 0
    for r in rows:
        cid = r["context_sha256"]
        if cid not in uncached_raw or cid not in cached_raw:
            raise RuntimeError(f"missing context score vector: {cid}")
        u = np.ascontiguousarray(uncached_raw[cid], dtype=np.float32)
        c = np.ascontiguousarray(cached_raw[cid], dtype=np.float32)
        if (
            u.shape != c.shape
            or u.tobytes() != c.tobytes()
            or not torch.equal(torch.from_numpy(u), torch.from_numpy(c))
        ):
            max_abs = (
                float(np.max(np.abs(u - c))) if u.shape == c.shape else float("inf")
            )
            raise RuntimeError(
                f"W10 r882 native vs cached total_cost bit-identity failed on {cid}: max_abs={max_abs}"
            )
        total_candidates += int(u.size)
    return dict(
        contexts_checked=len(rows),
        candidates_checked=total_candidates,
        bit_identical=True,
        max_abs_cost_diff=0.0,
        status="PASS",
    )


def verify_external_launch_binding(ext_path, head, manifest_sha, prereg_sha, fold_sha):
    """Verifies external launch.json binding written after PR merge and exact-head review."""
    p = Path(ext_path)
    if not p.exists():
        raise RuntimeError(f"missing external launch binding: {p}")
    ext = json.loads(p.read_text())
    if ext.get("merged_commit") != head:
        raise RuntimeError(
            f"external launch merged_commit mismatch: {ext.get('merged_commit')} != {head}"
        )
    if ext.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("external launch manifest_sha256 mismatch")
    if ext.get("prereg_sha256") != prereg_sha:
        raise RuntimeError("external launch prereg_sha256 mismatch")
    if ext.get("fold_manifest_sha256") != fold_sha:
        raise RuntimeError("external launch fold_manifest_sha256 mismatch")
    if not ext.get("review_id"):
        raise RuntimeError("external launch review_id required")
    return ext


def subpanel(full, rows):
    idx = {c: i for i, c in enumerate(full["context_ids"])}
    order = [idx[r["context_sha256"]] for r in rows]
    return dict(
        values=[full["values"][i] for i in order],
        choices=[full["choices"][i] for i in order],
        invalid=[full["invalid"][i] for i in order],
        context_ids=[full["context_ids"][i] for i in order],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    train, val = selected()
    if len({r["family_id"] for r in train}) != 205:
        raise RuntimeError("205 DL-Train families required")
    groups = sc.folds(train)
    if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
        raise RuntimeError("fold_manifest mismatch")
    allowed = set(pins["inputs"])
    allowed.add(str((out.parent / "launch.json").resolve()))
    allowed.add(str((out / "launch.json").resolve()))
    allowed.update(
        str(rt.INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    updates = 0
    started = time.monotonic()
    frozen_hash = None
    head = None

    def write_audit():
        import torch

        save(
            out / "audit.json",
            dict(
                **audit.receipt(),
                optimizer_updates=updates,
                wall_seconds=time.monotonic() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                threads_at_end=len(list(Path("/proc/self/task").iterdir())),
                cuda_initialized=torch.cuda.is_initialized(),
                dope_sha256=rt.DOPE,
                frozen_sha256=frozen_hash,
                source_commit=head,
                manifest_sha256=rt.sha(HERE / "manifest.json"),
            ),
        )

    def _sigterm_handler(signum, frame):
        write_audit()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        frozen_hash = frozen
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or any(m["p"] != 0 for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
            or census["adapter_training"]
            or census["dope_training"]
        ):
            raise RuntimeError("unexpected dropout or training mode in adapter")
        rt.check_frozen(net, guard, names, frozen)
        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        width_counts = {}
        for arm in sc.WIDTH_ARMS:
            params_w, _, _ = rt.configure_adapter_width(
                net, initial, names, sc.WIDTH_MULTS[arm], sc.SEEDS[0]
            )
            width_counts[arm] = sum(t.numel() for t in params_w)
            rt.check_frozen(net, guard, names, frozen)
        if width_counts != {"W10": 29211, "W20": 46595}:
            raise RuntimeError(f"unexpected width parameter counts: {width_counts}")
        rt.configure_adapter_width(net, initial, names, 1.0, sc.SEEDS[0])
        rt.check_frozen(net, guard, names, frozen)

        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    wave="H168",
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=len(names),
                    trainable_parameters=width_counts["W10"],
                    width_parameter_counts=width_counts,
                    census=census,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    seeds=list(sc.SEEDS),
                    fold_seed=sc.FOLD_SEED,
                    boot_seed=sc.BOOT_SEED,
                    primary_arms=list(sc.PRIMARY_ARMS),
                    diagnostic_arms=list(sc.DIAGNOSTIC_ARMS),
                    packages=pins["packages"],
                ),
            )
            return

        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
        manifest_sha = rt.sha(HERE / "manifest.json")
        ext_launch = verify_external_launch_binding(
            out.parent / "launch.json",
            head,
            manifest_sha,
            pins["prereg_sha256"],
            pins["fold_manifest_sha256"],
        )
        launch = dict(
            wave="H168",
            source_commit=head,
            manifest_sha256=manifest_sha,
            prereg_sha256=pins["prereg_sha256"],
            fold_manifest_sha256=pins["fold_manifest_sha256"],
            external_launch_binding=ext_launch,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        save(out / "launch.json", launch)

        ref = json.loads(Path(pins["reference"]).read_text())
        contract = ref["evaluation_contract"]
        expected = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in val
        ]
        if (
            contract["contexts"] != expected
            or hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            != pins["evaluation_identity"]
        ):
            raise RuntimeError("evaluation contract mismatch")

        import torch

        pools = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for row in train + val:
            cid = row["context_sha256"]
            path = rt.INPUT / "pools" / f"{cid}.npz"
            if rt.sha(path) != row["pool_sha256"]:
                raise RuntimeError(f"pool hash mismatch: {cid}")
            with np.load(path, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            if set(ids) != set(pools) or len(ids) != len(pools):
                raise RuntimeError("context feature cache mismatch")
            tok, mask = z["tokens_10x32"], z["mask_10"]
            tokens = {c: tok[i].astype(np.float32) for i, c in enumerate(ids)}
            masks = {c: mask[i].astype(bool) for i, c in enumerate(ids)}

        # Precompute frozen anchor outputs once across all 1,310 contexts (exact 0.0 diff vs uncached)
        anchor_cache = {}
        all_groups = {}
        for r in train + val:
            c = r["context_sha256"]
            all_groups.setdefault(len(pools[c]["fitness"]), []).append(c)
        with torch.no_grad():
            for size, cids in sorted(all_groups.items()):
                for start in range(0, len(cids), 8):
                    batch = cids[start : start + 8]
                    b = len(batch)
                    k = size
                    progs = torch.as_tensor(
                        np.stack([pools[c]["program"] for c in batch])
                    )
                    proj = torch.as_tensor(
                        np.stack([pools[c]["proj16"] for c in batch])
                    )
                    rev_idx = torch.arange(min(4, k)).reshape(1, -1).expand(b, -1)
                    rev_raw_f = torch.as_tensor(
                        np.stack([pools[c]["fitness"][: min(4, k)] for c in batch]),
                        dtype=torch.float32,
                    )
                    rev_gate = torch.as_tensor(
                        np.stack([pools[c]["gate"][: min(4, k)] for c in batch]),
                        dtype=torch.float32,
                    )
                    scale_b = torch.full((b,), 0.4208, dtype=torch.float32)
                    sos_b = torch.zeros((b, k, 2), dtype=torch.float32)
                    a_out = net.adapter_10tok.anchor.predict_costs(
                        ctx_feat=torch.zeros((b, 584), dtype=torch.float32),
                        prog_412=progs,
                        proj_16=proj,
                        base_prior_cost=torch.zeros((b, k), dtype=torch.float32),
                        scale=scale_b,
                        sos_prior=sos_b,
                        revealed_indices=rev_idx,
                        revealed_raw_f=rev_raw_f,
                        revealed_gate=rev_gate,
                        enable_prior_correction=True,
                        enable_struct_kernel=True,
                        enable_cross_attn=True,
                        enable_ucb=False,
                    )
                    a_tot = a_out["total_cost"].detach().float()
                    a_eff = a_out["eff_prior"].detach().float()
                    for i_b, c in enumerate(batch):
                        anchor_cache[c] = (
                            a_tot[i_b].clone(),
                            a_eff[i_b].clone(),
                            rev_raw_f[i_b].clone(),
                            rev_gate[i_b].clone(),
                        )

        def evaluate_uncached(rows, return_raw_scores=False):
            groups_r = {}
            result = {}
            raw_scores = {}
            for r in rows:
                c = r["context_sha256"]
                groups_r.setdefault(len(pools[c]["fitness"]), []).append(c)
            with torch.no_grad():
                for size, cids in sorted(groups_r.items()):
                    for start in range(0, len(cids), 8):
                        batch = cids[start : start + 8]
                        progs = np.stack([pools[c]["program"] for c in batch])
                        proj = np.stack([pools[c]["proj16"] for c in batch])
                        fit = np.stack([pools[c]["fitness"] for c in batch])
                        gate_arr = np.stack([pools[c]["gate"] for c in batch])
                        tok_arr = np.stack([tokens[c] for c in batch])
                        msk_arr = np.stack([masks[c] for c in batch])
                        scores = (
                            net.predict_with_10tok_adapter(
                                ctx_feat=torch.zeros(
                                    (len(batch), 584), dtype=torch.float32
                                ),
                                prog_412=torch.as_tensor(progs),
                                proj_16=torch.as_tensor(proj),
                                base_prior_cost=torch.zeros(
                                    (len(batch), size), dtype=torch.float32
                                ),
                                scale=torch.full(
                                    (len(batch),), 0.4208, dtype=torch.float32
                                ),
                                sos_prior=torch.zeros(
                                    (len(batch), size, 2), dtype=torch.float32
                                ),
                                mm_tokens_10x32=torch.as_tensor(tok_arr),
                                mm_mask_10=torch.as_tensor(msk_arr),
                                revealed_indices=torch.arange(min(4, size))
                                .reshape(1, -1)
                                .expand(len(batch), -1),
                                revealed_raw_f=torch.as_tensor(
                                    fit[:, : min(4, size)], dtype=torch.float32
                                ),
                                revealed_gate=torch.as_tensor(
                                    gate_arr[:, : min(4, size)], dtype=torch.float32
                                ),
                                enable_prior_correction=True,
                                enable_struct_kernel=True,
                                enable_cross_attn=True,
                                enable_ucb=False,
                            )["total_cost"]
                            .cpu()
                            .numpy()
                        )
                        for cid, cost in zip(batch, scores, strict=True):
                            if return_raw_scores:
                                raw_scores[cid] = np.ascontiguousarray(
                                    cost, dtype=np.float32
                                )
                            idx_min = int(np.argmin(cost))
                            p_c = pools[cid]
                            valid = bool(p_c["gate"][idx_min])
                            result[cid] = dict(
                                choice=idx_min,
                                utility=float(
                                    -p_c["fitness"][idx_min] if valid else -2.0
                                ),
                                invalid=not valid,
                            )
            rt.check_frozen(net, guard, names, frozen)
            out_dict = dict(
                values=[result[r["context_sha256"]]["utility"] for r in rows],
                choices=[result[r["context_sha256"]]["choice"] for r in rows],
                invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
                context_ids=[r["context_sha256"] for r in rows],
            )
            if return_raw_scores:
                out_dict["raw_scores"] = raw_scores
            return out_dict

        def evaluate(rows, return_raw_scores=False):
            groups_r = {}
            result = {}
            raw_scores = {}
            for r in rows:
                c = r["context_sha256"]
                groups_r.setdefault(len(pools[c]["fitness"]), []).append(c)
            with torch.no_grad():
                for size, cids in sorted(groups_r.items()):
                    for start in range(0, len(cids), 8):
                        batch = cids[start : start + 8]
                        b = len(batch)
                        k = size
                        prog_b = torch.as_tensor(
                            np.stack([pools[c]["program"] for c in batch])
                        )
                        a_tot_b = torch.stack([anchor_cache[c][0] for c in batch])
                        a_eff_b = torch.stack([anchor_cache[c][1] for c in batch])
                        rev_raw_b = torch.stack([anchor_cache[c][2] for c in batch])
                        rev_gate_b = torch.stack([anchor_cache[c][3] for c in batch])
                        tok_b = torch.as_tensor(np.stack([tokens[c] for c in batch]))
                        mask_b = torch.as_tensor(np.stack([masks[c] for c in batch]))
                        rev_idx = torch.arange(min(4, k)).reshape(1, -1).expand(b, -1)
                        scale_b = torch.full((b,), 0.4208, dtype=torch.float32)
                        sos_b = torch.zeros((b, k, 2), dtype=torch.float32)
                        scores = (
                            rt.forward_adapter_with_anchor(
                                net,
                                prog_b,
                                sos_b,
                                tok_b,
                                mask_b,
                                scale_b,
                                rev_idx,
                                rev_raw_b,
                                rev_gate_b,
                                a_tot_b,
                                a_eff_b,
                            )
                            .cpu()
                            .numpy()
                        )
                        if not np.isfinite(scores).all():
                            raise RuntimeError("nonfinite predictions")
                        for cid, cost in zip(batch, scores, strict=True):
                            if return_raw_scores:
                                raw_scores[cid] = np.ascontiguousarray(
                                    cost, dtype=np.float32
                                )
                            idx_min = int(np.argmin(cost))
                            p_c = pools[cid]
                            valid = bool(p_c["gate"][idx_min])
                            result[cid] = dict(
                                choice=idx_min,
                                utility=float(
                                    -p_c["fitness"][idx_min] if valid else -2.0
                                ),
                                invalid=not valid,
                            )
            rt.check_frozen(net, guard, names, frozen)
            out_dict = dict(
                values=[result[r["context_sha256"]]["utility"] for r in rows],
                choices=[result[r["context_sha256"]]["choice"] for r in rows],
                invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
                context_ids=[r["context_sha256"] for r in rows],
            )
            if return_raw_scores:
                out_dict["raw_scores"] = raw_scores
            return out_dict

        # Verify exact 91-ctx r882 baseline on BOTH uncached and cached paths (including per-candidate total_cost bit-identity for native W10 r882)
        baseline_uncached = evaluate_uncached(val, return_raw_scores=True)
        baseline_with_raw = evaluate(val, return_raw_scores=True)
        w10_score_bit_identity = verify_native_score_bit_identity(
            baseline_uncached["raw_scores"], baseline_with_raw["raw_scores"], val
        )
        baseline = {k: v for k, v in baseline_with_raw.items() if k != "raw_scores"}
        if (
            baseline_uncached["values"] != baseline["values"]
            or baseline_uncached["choices"] != baseline["choices"]
        ):
            raise RuntimeError("cached vs uncached baseline mismatch")
        sanity = baseline_gate(baseline["values"], ref)

        # Verify W20 (46,595-param research-only) step-0 utility tolerance (<= 1e-6; not native score bit-exactness)
        rt.configure_adapter_width(net, initial, names, 2.0, sc.SEEDS[0])
        w20_step0_val = evaluate(val, return_raw_scores=True)
        w20_step0_max_diff = float(
            np.max(
                np.abs(
                    np.asarray(w20_step0_val["values"]) - np.asarray(baseline["values"])
                )
            )
        )
        w20_step0_score_max_abs_diff = float(
            max(
                np.max(
                    np.abs(
                        w20_step0_val["raw_scores"][r["context_sha256"]]
                        - baseline_with_raw["raw_scores"][r["context_sha256"]]
                    )
                )
                for r in val
            )
        )
        if w20_step0_max_diff > 1e-6:
            raise RuntimeError(
                f"W20 step-0 DL-Val utility tolerance failed: max_diff={w20_step0_max_diff}"
            )
        rt.configure_adapter_width(net, initial, names, 1.0, sc.SEEDS[0])

        save(
            out / "baseline.json",
            dict(
                **baseline,
                sanity=sanity,
                w10_native_score_bit_identity=w10_score_bit_identity,
                w20_step0_val_utility_max_diff=w20_step0_max_diff,
                w20_step0_score_max_abs_diff=w20_step0_score_max_abs_diff,
                w20_step0_scope="research_only_46595_param_utility_tolerance_le_1e-6_not_native_score_bit_exact",
            ),
        )
        print(
            "BASELINE PASS",
            sanity,
            "w10_bit_identity",
            w10_score_bit_identity,
            "w20_step0_util_diff",
            w20_step0_max_diff,
            flush=True,
        )

        r882_train = evaluate(train)
        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}

        primary = np.full((len(sc.SEEDS), len(train), len(sc.COLUMNS)), np.nan)
        secondary = np.full((len(sc.SEEDS), 5, len(val), len(sc.COLUMNS)), np.nan)
        diag_primary = np.full(
            (len(sc.SEEDS), len(train), len(sc.DIAG_COLUMNS)), np.nan
        )
        diag_secondary = np.full(
            (len(sc.SEEDS), 5, len(val), len(sc.DIAG_COLUMNS)), np.nan
        )

        for si in range(len(sc.SEEDS)):
            primary[si, :, 0] = r882_train["values"]
            diag_primary[si, :, 0] = r882_train["values"]
            for outer in range(5):
                secondary[si, outer, :, 0] = baseline["values"]
                diag_secondary[si, outer, :, 0] = baseline["values"]

        nonzero_selections = {a: 0 for a in sc.PRIMARY_ARMS}
        diag_nonzero_selections = {a: 0 for a in sc.DIAGNOSTIC_ARMS}
        drifts_by_width = {w: [] for w in sc.WIDTH_ARMS}
        cells_meta = []

        for outer in range(5):
            fit, select, assess = sc.partition(train, outer)
            fit_ids = {r["context_sha256"] for r in fit}
            excluded = {r["context_sha256"] for r in select + assess + val}
            assert not (fit_ids & excluded)
            orders = {c: np.argsort(pools[c]["fitness"]) for c in fit_ids}
            r882_sel_scores = subpanel(r882_train, select)
            r882_assess_scores = subpanel(r882_train, assess)
            ai_idx = [train_index[r["context_sha256"]] for r in assess]

            for si, seed in enumerate(sc.SEEDS):
                cell_tick = time.monotonic()
                plan = sc.batch_plan(fit_ids, orders, seed)
                if len(plan) != 40:
                    raise RuntimeError("40-step batch plan required")

                # Precompute frozen anchor terms for the 40 minibatches of (outer, seed)
                batch_cache = []
                with torch.no_grad():
                    for step, (stream, cids, ixs) in enumerate(plan, 1):
                        if not set(cids) <= fit_ids or (set(cids) & excluded):
                            raise RuntimeError("minibatch family leakage")
                        b = len(cids)
                        k = ixs[0].shape[0]
                        progs = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["program"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            )
                        )
                        proj = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["proj16"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            )
                        )
                        tok_b = torch.as_tensor(np.stack([tokens[c] for c in cids]))
                        mask_b = torch.as_tensor(np.stack([masks[c] for c in cids]))
                        rev_idx = torch.arange(min(4, k)).reshape(1, -1).expand(b, -1)
                        rev_raw_f = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["fitness"][ix[: min(4, k)]]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                        )
                        rev_gate = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["gate"][ix[: min(4, k)]]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                        )
                        fitness = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["fitness"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                        )
                        gate = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["gate"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.bool,
                        )
                        scale_b = torch.full((b,), 0.4208, dtype=torch.float32)
                        sos_b = torch.zeros((b, k, 2), dtype=torch.float32)
                        a_out = net.adapter_10tok.anchor.predict_costs(
                            ctx_feat=torch.zeros((b, 584), dtype=torch.float32),
                            prog_412=progs,
                            proj_16=proj,
                            base_prior_cost=torch.zeros((b, k), dtype=torch.float32),
                            scale=scale_b,
                            sos_prior=sos_b,
                            revealed_indices=rev_idx,
                            revealed_raw_f=rev_raw_f,
                            revealed_gate=rev_gate,
                            enable_prior_correction=True,
                            enable_struct_kernel=True,
                            enable_cross_attn=True,
                            enable_ucb=False,
                        )
                        batch_cache.append(
                            (
                                step,
                                stream,
                                cids,
                                ixs,
                                progs,
                                sos_b,
                                tok_b,
                                mask_b,
                                scale_b,
                                rev_idx,
                                rev_raw_f,
                                rev_gate,
                                a_out["total_cost"].detach().float(),
                                a_out["eff_prior"].detach().float(),
                                fitness,
                                gate,
                            )
                        )

                arm_snapshots = {}
                arm_sel_scores = {}
                arm_losses = {}
                arm_drifts = {}
                arm_none_grad = {}

                for w_arm in sc.WIDTH_ARMS:
                    wm = sc.WIDTH_MULTS[w_arm]
                    params, _, init_arm_vec = rt.configure_adapter_width(
                        net, initial, names, wm, seed
                    )
                    rt.check_frozen(net, guard, names, frozen)
                    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                    snapshots, sel_scores, losses, step_drifts = {}, [], [], {}
                    none_grad_set = set()

                    def cur_drift():
                        vec = torch.cat(
                            [
                                dict(net.named_parameters())[n]
                                .detach()
                                .cpu()
                                .reshape(-1)
                                for n in names
                            ]
                        )
                        return float(torch.linalg.vector_norm(vec - init_arm_vec))

                    def take_snapshot(step_num, arm_label=w_arm):
                        st = {
                            n: net.state_dict()[n].detach().cpu().clone() for n in names
                        }
                        snapshots[step_num] = st
                        step_drifts[str(step_num)] = (
                            0.0 if step_num == 0 else cur_drift()
                        )
                        if arm_label == "W10" and step_num == 0:
                            sel_scores.append(r882_sel_scores)
                        else:
                            sel_scores.append(evaluate(select))

                    take_snapshot(0)
                    for (
                        step,
                        stream,
                        cids,
                        ixs,
                        progs,
                        sos_b,
                        tok_b,
                        mask_b,
                        scale_b,
                        rev_idx,
                        rev_raw_f,
                        rev_gate,
                        a_tot_b,
                        a_eff_b,
                        fitness,
                        gate,
                    ) in batch_cache:
                        torch.manual_seed(stream)
                        optimizer.zero_grad(set_to_none=True)
                        pred = rt.forward_adapter_with_anchor(
                            net,
                            progs,
                            sos_b,
                            tok_b,
                            mask_b,
                            scale_b,
                            rev_idx,
                            rev_raw_f,
                            rev_gate,
                            a_tot_b,
                            a_eff_b,
                        )
                        objective = rt.loss(pred, fitness, gate)
                        if not torch.isfinite(objective):
                            raise RuntimeError("nonfinite objective")
                        objective.backward()
                        _, none_ix = rt.safe_grad_norm_and_clip(params, max_norm=1.0)
                        none_grad_set.update(names[i] for i in none_ix)
                        optimizer.step()
                        updates += 1
                        rt.check_frozen(net, guard, names, frozen)
                        losses.append(float(objective.detach()))
                        if step in sc.STEPS:
                            take_snapshot(step)

                    if step_drifts["40"] <= 0.0:
                        raise RuntimeError(f"no parameter drift for {w_arm}")
                    drifts_by_width[w_arm].append(step_drifts["40"])
                    arm_snapshots[w_arm] = snapshots
                    arm_sel_scores[w_arm] = sel_scores
                    arm_losses[w_arm] = losses
                    arm_drifts[w_arm] = step_drifts
                    arm_none_grad[w_arm] = sorted(none_grad_set)

                # Lock OLD, NEW, RELAXED, and NONZERO inner-B selections BEFORE evaluating outer fold A or DL-Val
                r882_sel_vals = np.asarray(r882_sel_scores["values"], dtype=np.float64)
                decisions = {}
                for wi, w_arm in enumerate(sc.WIDTH_ARMS):
                    panel_b = np.array(
                        [r["values"] for r in arm_sel_scores[w_arm]], dtype=np.float64
                    ).T
                    decisions[w_arm] = sc.choose_dual(
                        select,
                        panel_b,
                        sc.BOOT_SEED + 100 * outer + 10 * si + wi,
                        r882_col=r882_sel_vals,
                    )

                cellname = f"A{outer}-{seed}"
                lockpath = out / f"{cellname}-selection.json"
                save(lockpath, decisions)

                # Collect unique non-zero (width_arm, step) checkpoints needed across OLD, NEW, RELAXED, NONZERO
                needed_pairs = {("W10", 0)}
                for w_arm in sc.WIDTH_ARMS:
                    d_w = decisions[w_arm]
                    for k_step in (
                        "old_selected_step",
                        "new_selected_step",
                        "relaxed_selected_step",
                        "nonzero_selected_step",
                    ):
                        st_pick = int(d_w[k_step])
                        needed_pairs.add(
                            ("W10", 0) if st_pick == 0 else (w_arm, st_pick)
                        )

                assessment = {("W10", 0): r882_assess_scores}
                val_assessment = {("W10", 0): baseline}
                for arm_k, step_k in sorted(needed_pairs):
                    if (arm_k, step_k) in assessment:
                        continue
                    rt.configure_adapter_width(
                        net, initial, names, sc.WIDTH_MULTS[arm_k], seed
                    )
                    with torch.no_grad():
                        for n, p_t in net.named_parameters():
                            if n in arm_snapshots[arm_k][step_k]:
                                p_t.copy_(
                                    arm_snapshots[arm_k][step_k][n].to(p_t.device)
                                )
                    assessment[(arm_k, step_k)] = evaluate(assess)
                    val_assessment[(arm_k, step_k)] = evaluate(val)

                rt.configure_adapter_width(net, initial, names, 1.0, seed)

                # Populate primary/secondary panels: ("r882", "W10_OLD", "W10_NEW", "W20_OLD", "W20_NEW")
                primary_map = [
                    ("W10_OLD", "W10", decisions["W10"]["old_selected_step"]),
                    ("W10_NEW", "W10", decisions["W10"]["new_selected_step"]),
                    ("W20_OLD", "W20", decisions["W20"]["old_selected_step"]),
                    ("W20_NEW", "W20", decisions["W20"]["new_selected_step"]),
                ]
                for col, (p_name, w_arm, st_pick) in enumerate(primary_map, start=1):
                    if st_pick > 0:
                        nonzero_selections[p_name] += 1
                    key = ("W10", 0) if st_pick == 0 else (w_arm, st_pick)
                    primary[si, ai_idx, col] = assessment[key]["values"]
                    secondary[si, outer, :, col] = val_assessment[key]["values"]

                # Populate diagnostic panels: ("r882", "W10_RELAXED", "W10_NONZERO", "W20_RELAXED", "W20_NONZERO")
                diag_map = [
                    ("W10_RELAXED", "W10", decisions["W10"]["relaxed_selected_step"]),
                    ("W10_NONZERO", "W10", decisions["W10"]["nonzero_selected_step"]),
                    ("W20_RELAXED", "W20", decisions["W20"]["relaxed_selected_step"]),
                    ("W20_NONZERO", "W20", decisions["W20"]["nonzero_selected_step"]),
                ]
                for col, (d_name, w_arm, st_pick) in enumerate(diag_map, start=1):
                    if st_pick > 0:
                        diag_nonzero_selections[d_name] += 1
                    key = ("W10", 0) if st_pick == 0 else (w_arm, st_pick)
                    diag_primary[si, ai_idx, col] = assessment[key]["values"]
                    diag_secondary[si, outer, :, col] = val_assessment[key]["values"]

                cell_record = dict(
                    outer_fold=outer,
                    selection_fold=(outer + 1) % 5,
                    seed=seed,
                    updates_per_width_arm=40,
                    total_cell_updates=80,
                    fit_contexts=sorted(fit_ids),
                    selector_contexts=[r["context_sha256"] for r in select],
                    assessment_contexts=[r["context_sha256"] for r in assess],
                    losses=arm_losses,
                    adapter_l2_drift=arm_drifts,
                    none_grad_tensors=arm_none_grad,
                    snapshot_sha256={
                        w_arm: {
                            str(s): rt.tensor_hash(sorted(v.items()))
                            for s, v in arm_snapshots[w_arm].items()
                        }
                        for w_arm in sc.WIDTH_ARMS
                    },
                    selection_lock_sha256=rt.sha(lockpath),
                    decisions=decisions,
                    assessment={f"{k[0]}_step{k[1]}": v for k, v in assessment.items()},
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    wall_seconds=time.monotonic() - cell_tick,
                )
                save(out / f"{cellname}.json", cell_record)
                cells_meta.append(
                    dict(
                        cell=cellname,
                        outer_fold=outer,
                        seed=seed,
                        w10_old=decisions["W10"]["old_selected_step"],
                        w10_new=decisions["W10"]["new_selected_step"],
                        w10_relaxed=decisions["W10"]["relaxed_selected_step"],
                        w10_nonzero=decisions["W10"]["nonzero_selected_step"],
                        w20_old=decisions["W20"]["old_selected_step"],
                        w20_new=decisions["W20"]["new_selected_step"],
                        w20_relaxed=decisions["W20"]["relaxed_selected_step"],
                        w20_nonzero=decisions["W20"]["nonzero_selected_step"],
                        w10_step40_drift=arm_drifts["W10"]["40"],
                        w20_step40_drift=arm_drifts["W20"]["40"],
                    )
                )
                print(
                    "CELL COMPLETE",
                    cellname,
                    "updates",
                    updates,
                    "W10(old/new/rel/nz)",
                    (
                        decisions["W10"]["old_selected_step"],
                        decisions["W10"]["new_selected_step"],
                        decisions["W10"]["relaxed_selected_step"],
                        decisions["W10"]["nonzero_selected_step"],
                    ),
                    "W20(old/new/rel/nz)",
                    (
                        decisions["W20"]["old_selected_step"],
                        decisions["W20"]["new_selected_step"],
                        decisions["W20"]["relaxed_selected_step"],
                        decisions["W20"]["nonzero_selected_step"],
                    ),
                    "seconds",
                    round(cell_record["wall_seconds"], 2),
                    flush=True,
                )

        if updates != 1200 or len(cells_meta) != 15:
            raise RuntimeError(
                f"incomplete wave: updates={updates} cells={len(cells_meta)}"
            )

        primary_summary = sc.summarize(train, primary, col_names=sc.COLUMNS)
        secondary_summary = sc.summarize(
            val, secondary.mean(axis=1), col_names=sc.COLUMNS
        )
        diag_primary_summary = sc.summarize(
            train, diag_primary, col_names=sc.DIAG_COLUMNS
        )
        diag_secondary_summary = sc.summarize(
            val, diag_secondary.mean(axis=1), col_names=sc.DIAG_COLUMNS
        )
        decision = sc.decide(primary_summary, nonzero_selections)

        drift_summary = {
            w: dict(
                median_step40_l2_drift=float(np.median(drifts_by_width[w])),
                min_step40_l2_drift=float(np.min(drifts_by_width[w])),
                max_step40_l2_drift=float(np.max(drifts_by_width[w])),
            )
            for w in sc.WIDTH_ARMS
        }

        metrics = dict(
            wave="H168",
            status="COMPLETE",
            verdict=decision["verdict"],
            decision=decision,
            primary=primary_summary,
            secondary_dl_val_descriptive_only=secondary_summary,
            prelocked_diagnostics_oof_train=diag_primary_summary,
            prelocked_diagnostics_dl_val=diag_secondary_summary,
            nonzero_selections=nonzero_selections,
            diagnostic_nonzero_selections=diag_nonzero_selections,
            cells_per_width_arm=15,
            total_trajectories=30,
            optimizer_updates=updates,
            width_parameter_counts=width_counts,
            drift_summary=drift_summary,
            cell_selections=cells_meta,
            baseline=sanity,
            w20_step0_val_max_diff=w20_step0_max_diff,
            source_commit=head,
            manifest_sha256=rt.sha(HERE / "manifest.json"),
            fold_manifest_sha256=rt.sha(HERE / "fold_manifest.json"),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        np.savez_compressed(
            out / "panels.npz",
            primary=primary,
            secondary=secondary,
            diagnostic_primary=diag_primary,
            diagnostic_secondary=diag_secondary,
        )
        verify(pins)
        save(out / "metrics.json", metrics)
    finally:
        write_audit()


if __name__ == "__main__":
    main()
