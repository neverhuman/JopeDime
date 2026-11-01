"""Single-attempt H173 executor: hard-context (regret-weighted / focal-style) loss reweighting on r882 continuation."""

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
    """Verifies per-candidate total_cost bit-identity (exact float32 bytes and torch.equal) for native r882 across all 91 DL-Val contexts."""
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
                f"r882 native vs cached total_cost bit-identity failed on {cid}: max_abs={max_abs}"
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


def predict_uncached_adapter_costs(net, progs, proj, fit, gate_arr, tok_arr, msk_arr):
    """Calls net.adapter_10tok.predict_costs directly for uncached native r882 baseline verification."""
    import torch

    b, size = progs.shape[:2]
    return (
        net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((b, 584), dtype=torch.float32),
            prog_412=torch.as_tensor(progs),
            proj_16=torch.as_tensor(proj),
            base_prior_cost=torch.zeros((b, size), dtype=torch.float32),
            scale=torch.full((b,), 0.4208, dtype=torch.float32),
            sos_prior=torch.zeros((b, size, 2), dtype=torch.float32),
            mm_tokens_10x32=torch.as_tensor(tok_arr),
            mm_mask_10=torch.as_tensor(msk_arr),
            revealed_indices=torch.arange(min(4, size)).reshape(1, -1).expand(b, -1),
            revealed_raw_f=torch.as_tensor(fit[:, : min(4, size)], dtype=torch.float32),
            revealed_gate=torch.as_tensor(
                gate_arr[:, : min(4, size)], dtype=torch.float32
            ),
        )["total_cost"]
        .cpu()
        .numpy()
    )


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
        params_check, _, _ = rt.reset_adapter_to_r882(net, initial, names)
        if sum(t.numel() for t in params_check) != 29211:
            raise RuntimeError("unexpected trainable parameter count")
        rt.check_frozen(net, guard, names, frozen)

        if not args.smoke:
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
                wave="H173",
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

        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            expected_ids = {r["context_sha256"] for r in train + val}
            if set(ids) != expected_ids or len(ids) != len(expected_ids):
                raise RuntimeError("context feature cache mismatch")
            tok, mask = z["tokens_10x32"], z["mask_10"]
            tokens = {c: tok[i].astype(np.float32) for i, c in enumerate(ids)}
            masks = {c: mask[i].astype(bool) for i, c in enumerate(ids)}

        pools = {}
        anchor_cache = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)

        def load_pools_and_anchor_cache(rows):
            groups_sub = {}
            for row in rows:
                cid = row["context_sha256"]
                path = rt.INPUT / "pools" / f"{cid}.npz"
                if rt.sha(path) != row["pool_sha256"]:
                    raise RuntimeError(f"pool hash mismatch: {cid}")
                with np.load(path, allow_pickle=False) as z:
                    program = z["program"].astype(np.float32)
                    pools[cid] = dict(
                        program=program,
                        proj16=(program.astype(np.float64) @ projection).astype(
                            np.float32
                        ),
                        fitness=z["fitness"].astype(np.float64),
                        gate=z["gate"].astype(bool),
                    )
                groups_sub.setdefault(len(pools[cid]["fitness"]), []).append(cid)
            with torch.no_grad():
                for size, cids in sorted(groups_sub.items()):
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
                        scores = predict_uncached_adapter_costs(
                            net, progs, proj, fit, gate_arr, tok_arr, msk_arr
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

        # 1. Load ONLY the 91 DL-Val contexts and run the exact 91-ctx baseline & total_cost bit-identity gate BEFORE opening any DL-Train pools
        load_pools_and_anchor_cache(val)
        baseline_uncached = evaluate_uncached(val, return_raw_scores=True)
        baseline_with_raw = evaluate(val, return_raw_scores=True)
        native_score_bit_identity = verify_native_score_bit_identity(
            baseline_uncached["raw_scores"], baseline_with_raw["raw_scores"], val
        )
        baseline = {k: v for k, v in baseline_with_raw.items() if k != "raw_scores"}
        if (
            baseline_uncached["values"] != baseline["values"]
            or baseline_uncached["choices"] != baseline["choices"]
        ):
            raise RuntimeError("cached vs uncached baseline mismatch")
        sanity = baseline_gate(baseline["values"], ref)

        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    wave="H173",
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=len(val),
                    dl_train_pools_opened=0,
                    baseline_sanity=sanity,
                    native_score_bit_identity=native_score_bit_identity,
                    adapter_tensors=len(names),
                    trainable_parameters=29211,
                    census=census,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    seeds=list(sc.SEEDS),
                    fold_seed=sc.FOLD_SEED,
                    boot_seed=sc.BOOT_SEED,
                    arms=list(sc.ARMS),
                    treatment_arms=list(sc.TREATMENT_ARMS),
                    packages=pins["packages"],
                ),
            )
            return

        save(
            out / "baseline.json",
            dict(
                **baseline,
                sanity=sanity,
                native_score_bit_identity=native_score_bit_identity,
            ),
        )
        print(
            "BASELINE PASS",
            sanity,
            "native_bit_identity",
            native_score_bit_identity,
            flush=True,
        )

        # 2. Load the 1,219 DL-Train pools and precompute frozen anchor outputs after DL-Val baseline gate passes
        load_pools_and_anchor_cache(train)
        r882_train = evaluate(train)
        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}
        r882_choice_map = {
            cid: int(ch)
            for cid, ch in zip(
                r882_train["context_ids"], r882_train["choices"], strict=True
            )
        }

        primary = np.full((len(sc.SEEDS), len(train), len(sc.COLUMNS)), np.nan)
        secondary = np.full((len(sc.SEEDS), 5, len(val), len(sc.COLUMNS)), np.nan)
        diag_uncon_primary = np.full(
            (len(sc.SEEDS), len(train), len(sc.DIAG_UNCONSTRAINED_COLUMNS)), np.nan
        )
        diag_uncon_secondary = np.full(
            (len(sc.SEEDS), 5, len(val), len(sc.DIAG_UNCONSTRAINED_COLUMNS)), np.nan
        )
        diag_nz_primary = np.full(
            (len(sc.SEEDS), len(train), len(sc.DIAG_NONZERO_COLUMNS)), np.nan
        )
        diag_nz_secondary = np.full(
            (len(sc.SEEDS), 5, len(val), len(sc.DIAG_NONZERO_COLUMNS)), np.nan
        )

        for si in range(len(sc.SEEDS)):
            primary[si, :, 0] = r882_train["values"]
            diag_uncon_primary[si, :, 0] = r882_train["values"]
            diag_nz_primary[si, :, 0] = r882_train["values"]
            for outer in range(5):
                secondary[si, outer, :, 0] = baseline["values"]
                diag_uncon_secondary[si, outer, :, 0] = baseline["values"]
                diag_nz_secondary[si, outer, :, 0] = baseline["values"]

        nonzero_selections = {a: 0 for a in sc.ARMS}
        uncon_nonzero_selections = {a: 0 for a in sc.DIAG_UNCONSTRAINED_ARMS}
        nz_nonzero_selections = {a: 0 for a in sc.DIAG_NONZERO_ARMS}
        drifts_by_arm = {a: [] for a in sc.ARMS}
        ess_records_by_arm = {a: [] for a in sc.ARMS}
        selected_steps_by_arm_and_seed = {
            a: [[0] * 5 for _ in range(len(sc.SEEDS))] for a in sc.ARMS
        }
        cells_meta = []
        inner_b_npz_dict = {}

        (out / "locks").mkdir(exist_ok=True)
        (out / "cells").mkdir(exist_ok=True)

        for outer in range(5):
            fit, select, assess = sc.partition(train, outer)
            fit_ids = {r["context_sha256"] for r in fit}
            excluded = {r["context_sha256"] for r in select + assess + val}
            assert not (fit_ids & excluded)
            orders = {c: np.argsort(pools[c]["fitness"]) for c in fit_ids}
            r882_sel_scores = subpanel(r882_train, select)
            r882_assess_scores = subpanel(r882_train, assess)
            ai_idx = [train_index[r["context_sha256"]] for r in assess]

            inner_b_npz_dict[f"A{outer}_select_context_ids"] = np.asarray(
                [r["context_sha256"] for r in select]
            )
            inner_b_npz_dict[f"A{outer}_select_family_ids"] = np.asarray(
                [r["family_id"] for r in select]
            )

            # Build FIT-only regret source dictionary (strictly restricted to fit_ids)
            fit_regret_sources = {
                c: dict(
                    fitness=pools[c]["fitness"],
                    gate=pools[c]["gate"],
                    r882_choice=r882_choice_map[c],
                )
                for c in fit_ids
            }

            for si, seed in enumerate(sc.SEEDS):
                cell_tick = time.monotonic()
                weight_bundle = sc.compute_fit_regret_weights(
                    fit_ids=fit_ids,
                    pool_regret_sources=fit_regret_sources,
                    seed=seed,
                    outer=outer,
                    excluded_ids=excluded,
                )
                for arm in sc.ARMS:
                    ess_records_by_arm[arm].append(weight_bundle["ess_by_arm"][arm])

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
                arm_sel_panels = {}
                arm_choices = {}
                arm_losses = {}
                arm_drifts = {}
                arm_none_grad = {}

                for arm in sc.ARMS:
                    arm_w_map = weight_bundle["weights_by_arm"][arm]
                    params, _, init_arm_vec = rt.reset_adapter_to_r882(
                        net, initial, names
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

                    def take_snapshot(step_num):
                        st = {
                            n: net.state_dict()[n].detach().cpu().clone() for n in names
                        }
                        snapshots[step_num] = st
                        step_drifts[str(step_num)] = (
                            0.0 if step_num == 0 else cur_drift()
                        )
                        if step_num == 0:
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
                        w_batch = torch.tensor(
                            [arm_w_map[c] for c in cids], dtype=torch.float32
                        )
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
                        lval = rt.weighted_loss(pred, fitness, gate, w_batch)
                        if not torch.isfinite(lval):
                            raise RuntimeError("nonfinite training loss")
                        lval.backward()
                        raw_grad_norm, none_ix = rt.safe_grad_norm_and_clip(
                            params, max_norm=1.0
                        )
                        none_grad_set.update(none_ix)
                        optimizer.step()
                        updates += 1
                        rt.check_frozen(net, guard, names, frozen)
                        losses.append(
                            dict(
                                step=step,
                                stream=stream,
                                loss=float(lval.detach().cpu()),
                                grad_norm=raw_grad_norm,
                                batch_weight_mean=float(w_batch.mean()),
                                batch_weight_max=float(w_batch.max()),
                            )
                        )
                        if step in sc.STEPS[1:]:
                            take_snapshot(step)

                    panel_b = np.column_stack([s["values"] for s in sel_scores])
                    inner_b_npz_dict[f"A{outer}-{seed}-{arm}"] = panel_b
                    choice_info = sc.choose_strict_and_diagnostics(
                        select,
                        panel_b,
                        seed=sc.selector_seed_for(outer, seed, sc.BOOT_SEED),
                    )
                    arm_snapshots[arm] = snapshots
                    arm_sel_panels[arm] = panel_b
                    arm_choices[arm] = choice_info
                    arm_losses[arm] = losses
                    arm_drifts[arm] = step_drifts
                    arm_none_grad[arm] = sorted(none_grad_set)
                    drifts_by_arm[arm].append(step_drifts["40"])

                cell_id = f"A{outer}-{seed}"
                lock_payload = dict(
                    cell=cell_id,
                    outer_fold=outer,
                    seed=seed,
                    fit_contexts=len(fit_ids),
                    select_contexts=len(select),
                    assess_contexts=len(assess),
                    raw_median_fit_regret=weight_bundle["raw_median_fit_regret"],
                    eps=weight_bundle["eps"],
                    zero_regret_count=weight_bundle["zero_regret_count"],
                    positive_regret_count=weight_bundle["positive_regret_count"],
                    mean_fit_regret=weight_bundle["mean_fit_regret"],
                    max_fit_regret=weight_bundle["max_fit_regret"],
                    ess_by_arm=weight_bundle["ess_by_arm"],
                    shuf_seed=weight_bundle["shuf_seed"],
                    arms={
                        arm: dict(
                            **arm_choices[arm],
                            step_drifts=arm_drifts[arm],
                            none_grad_param_indices=arm_none_grad[arm],
                        )
                        for arm in sc.ARMS
                    },
                    locked_before_outer_assessment=True,
                )
                save(out / "locks" / f"{cell_id}.json", lock_payload)

                # Evaluate locked checkpoints on outer assessment fold A_k and DL-Val
                def eval_arm_step(arm_name, step_num):
                    if step_num == 0:
                        return r882_assess_scores, baseline
                    rt.reset_adapter_to_r882(net, initial, names)
                    sd = net.state_dict()
                    for n, t in arm_snapshots[arm_name][step_num].items():
                        sd[n] = t.clone()
                    net.load_state_dict(sd, strict=True)
                    rt.check_frozen(net, guard, names, frozen)
                    return evaluate(assess), evaluate(val)

                eval_cache = {}
                for arm in sc.ARMS:
                    ch = arm_choices[arm]
                    for st_step in {
                        ch["selected_step"],
                        ch["unconstrained_selected_step"],
                        ch["nonzero_selected_step"],
                    }:
                        if (arm, st_step) not in eval_cache:
                            eval_cache[(arm, st_step)] = eval_arm_step(arm, st_step)

                for arm_i, arm in enumerate(sc.ARMS, start=1):
                    ch = arm_choices[arm]
                    s_strict = ch["selected_step"]
                    s_uncon = ch["unconstrained_selected_step"]
                    s_nz = ch["nonzero_selected_step"]
                    selected_steps_by_arm_and_seed[arm][si][outer] = int(s_strict)

                    if s_strict > 0:
                        nonzero_selections[arm] += 1
                    if s_uncon > 0:
                        uncon_nonzero_selections[f"{arm}_UNCONSTRAINED"] += 1
                    if s_nz > 0:
                        nz_nonzero_selections[f"{arm}_NONZERO"] += 1

                    a_strict, v_strict = eval_cache[(arm, s_strict)]
                    primary[si, ai_idx, arm_i] = a_strict["values"]
                    secondary[si, outer, :, arm_i] = v_strict["values"]

                    a_uncon, v_uncon = eval_cache[(arm, s_uncon)]
                    diag_uncon_primary[si, ai_idx, arm_i] = a_uncon["values"]
                    diag_uncon_secondary[si, outer, :, arm_i] = v_uncon["values"]

                    a_nz, v_nz = eval_cache[(arm, s_nz)]
                    diag_nz_primary[si, ai_idx, arm_i] = a_nz["values"]
                    diag_nz_secondary[si, outer, :, arm_i] = v_nz["values"]

                cell_summary = dict(
                    cell=cell_id,
                    outer_fold=outer,
                    seed=seed,
                    eps=weight_bundle["eps"],
                    ess_ratio_rw05=weight_bundle["ess_by_arm"]["RW05"]["ess_ratio"],
                    ess_ratio_rw10=weight_bundle["ess_by_arm"]["RW10"]["ess_ratio"],
                    ess_ratio_rw10_shuf=weight_bundle["ess_by_arm"]["RW10_SHUF"][
                        "ess_ratio"
                    ],
                    ctrl_selected=arm_choices["CTRL"]["selected_step"],
                    rw05_selected=arm_choices["RW05"]["selected_step"],
                    rw10_selected=arm_choices["RW10"]["selected_step"],
                    rw10_shuf_selected=arm_choices["RW10_SHUF"]["selected_step"],
                    ctrl_unconstrained=arm_choices["CTRL"][
                        "unconstrained_selected_step"
                    ],
                    rw05_unconstrained=arm_choices["RW05"][
                        "unconstrained_selected_step"
                    ],
                    rw10_unconstrained=arm_choices["RW10"][
                        "unconstrained_selected_step"
                    ],
                    rw10_shuf_unconstrained=arm_choices["RW10_SHUF"][
                        "unconstrained_selected_step"
                    ],
                    ctrl_nonzero=arm_choices["CTRL"]["nonzero_selected_step"],
                    rw05_nonzero=arm_choices["RW05"]["nonzero_selected_step"],
                    rw10_nonzero=arm_choices["RW10"]["nonzero_selected_step"],
                    rw10_shuf_nonzero=arm_choices["RW10_SHUF"]["nonzero_selected_step"],
                    ctrl_step40_drift=arm_drifts["CTRL"]["40"],
                    rw05_step40_drift=arm_drifts["RW05"]["40"],
                    rw10_step40_drift=arm_drifts["RW10"]["40"],
                    rw10_shuf_step40_drift=arm_drifts["RW10_SHUF"]["40"],
                )
                cells_meta.append(cell_summary)
                save(
                    out / "cells" / f"{cell_id}.json",
                    dict(
                        **cell_summary,
                        losses=arm_losses,
                    ),
                )
                print(
                    f"CELL COMPLETE {cell_id} updates {updates} "
                    f"strict(ctrl/rw05/rw10/shuf)=({cell_summary['ctrl_selected']}, {cell_summary['rw05_selected']}, {cell_summary['rw10_selected']}, {cell_summary['rw10_shuf_selected']}) "
                    f"uncon=({cell_summary['ctrl_unconstrained']}, {cell_summary['rw05_unconstrained']}, {cell_summary['rw10_unconstrained']}, {cell_summary['rw10_shuf_unconstrained']}) "
                    f"seconds {time.monotonic() - cell_tick:.2f}",
                    flush=True,
                )

        rt.reset_adapter_to_r882(net, initial, names)
        rt.check_frozen(net, guard, names, frozen)

        if (
            np.isnan(primary).any()
            or np.isnan(secondary).any()
            or np.isnan(diag_uncon_primary).any()
            or np.isnan(diag_uncon_secondary).any()
            or np.isnan(diag_nz_primary).any()
            or np.isnan(diag_nz_secondary).any()
        ):
            raise RuntimeError("incomplete evaluation matrix")

        np.savez_compressed(out / "inner_b_panels.npz", **inner_b_npz_dict)
        np.savez_compressed(
            out / "primary.npz",
            primary=primary,
            secondary=secondary,
            diag_unconstrained_primary=diag_uncon_primary,
            diag_unconstrained_secondary=diag_uncon_secondary,
            diag_nonzero_primary=diag_nz_primary,
            diag_nonzero_secondary=diag_nz_secondary,
            train_context_ids=np.asarray([r["context_sha256"] for r in train]),
            train_family_ids=np.asarray([r["family_id"] for r in train]),
            val_context_ids=np.asarray([r["context_sha256"] for r in val]),
            val_family_ids=np.asarray([r["family_id"] for r in val]),
        )

        primary_summary = sc.summarize(
            train, primary, cols=sc.COLUMNS, seed=sc.BOOT_SEED, draws=10000
        )
        secondary_summary = sc.summarize(
            val, secondary, cols=sc.COLUMNS, seed=sc.BOOT_SEED + 500, draws=10000
        )
        diag_uncon_train_summary = sc.summarize(
            train,
            diag_uncon_primary,
            cols=sc.DIAG_UNCONSTRAINED_COLUMNS,
            seed=sc.BOOT_SEED + 1500,
            draws=10000,
        )
        diag_uncon_val_summary = sc.summarize(
            val,
            diag_uncon_secondary,
            cols=sc.DIAG_UNCONSTRAINED_COLUMNS,
            seed=sc.BOOT_SEED + 2000,
            draws=10000,
        )
        diag_nz_train_summary = sc.summarize(
            train,
            diag_nz_primary,
            cols=sc.DIAG_NONZERO_COLUMNS,
            seed=sc.BOOT_SEED + 2500,
            draws=10000,
        )
        diag_nz_val_summary = sc.summarize(
            val,
            diag_nz_secondary,
            cols=sc.DIAG_NONZERO_COLUMNS,
            seed=sc.BOOT_SEED + 3000,
            draws=10000,
        )
        decision = sc.decide(primary_summary, nonzero_selections)
        pseudorep = sc.pseudoreplication_report(primary, selected_steps_by_arm_and_seed)

        drift_summary = {
            arm: dict(
                median_step40_l2_drift=float(np.median(drifts_by_arm[arm])),
                min_step40_l2_drift=float(np.min(drifts_by_arm[arm])),
                max_step40_l2_drift=float(np.max(drifts_by_arm[arm])),
            )
            for arm in sc.ARMS
        }
        ess_summary = {
            arm: dict(
                median_ess=float(
                    np.median([x["ess"] for x in ess_records_by_arm[arm]])
                ),
                min_ess=float(np.min([x["ess"] for x in ess_records_by_arm[arm]])),
                max_ess=float(np.max([x["ess"] for x in ess_records_by_arm[arm]])),
                median_ess_ratio=float(
                    np.median([x["ess_ratio"] for x in ess_records_by_arm[arm]])
                ),
                min_ess_ratio=float(
                    np.min([x["ess_ratio"] for x in ess_records_by_arm[arm]])
                ),
                max_ess_ratio=float(
                    np.max([x["ess_ratio"] for x in ess_records_by_arm[arm]])
                ),
                median_max_weight=float(
                    np.median([x["max"] for x in ess_records_by_arm[arm]])
                ),
                median_min_weight=float(
                    np.median([x["min"] for x in ess_records_by_arm[arm]])
                ),
            )
            for arm in sc.ARMS
        }

        metrics = dict(
            wave="H173",
            status="COMPLETE",
            verdict=decision["verdict"],
            decision=decision,
            primary=primary_summary,
            secondary_dl_val_descriptive_only=secondary_summary,
            prelocked_unconstrained_oof_train=diag_uncon_train_summary,
            prelocked_unconstrained_dl_val=diag_uncon_val_summary,
            prelocked_nonzero_oof_train=diag_nz_train_summary,
            prelocked_nonzero_dl_val=diag_nz_val_summary,
            pseudoreplication_report=pseudorep,
            nonzero_selections=nonzero_selections,
            unconstrained_nonzero_selections=uncon_nonzero_selections,
            nonzero_diag_selections=nz_nonzero_selections,
            ess_summary=ess_summary,
            drift_summary=drift_summary,
            cells_per_arm=15,
            total_trajectories=60,
            optimizer_updates=updates,
            trainable_parameters=29211,
            inner_b_raw_context_panels_retained=True,
            inner_b_panels_artifact="inner_b_panels.npz",
            cell_selections=cells_meta,
            baseline=sanity,
            native_score_bit_identity=native_score_bit_identity,
            source_commit=head,
            manifest_sha256=rt.sha(HERE / "manifest.json"),
            fold_manifest_sha256=pins["fold_manifest_sha256"],
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        save(out / "metrics.json", metrics)
    finally:
        write_audit()


if __name__ == "__main__":
    main()
