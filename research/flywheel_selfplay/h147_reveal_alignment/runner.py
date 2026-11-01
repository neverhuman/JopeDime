"""H147 execution runner: 5 whole-family folds x 3 seeds x 3 matched-support reveal arms (45 cells / 1,800 updates) on the 1,219-context / 205-family DL-Train cohort."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from runtime import (
    DOPE,
    HERE,
    INPUT,
    MIXED_H58_CACHE,
    S3_INVENTORY,
    TRAIN_CACHE,
    TRAIN_SELECTED,
    VAL_CACHE,
    Audit,
    assert_frozen_anchor,
    atomic_claim,
    bind_source,
    build,
    check_frozen,
    cpu_seconds_used,
    dropout_census,
    eval_batch_cache_key,
    guards,
    load_elo_tracker,
    loss,
    revealed_loss_and_grad_diagnostics,
    save,
    sha,
    tensor_hash,
)
from science import (
    ARMS,
    CANDIDATE_ARMS,
    LEARNING_RATE,
    SEEDS,
    SLATE_SIZE,
    WEIGHT_DECAY,
    batch_plan,
    bradley_terry_elo_summary,
    contrast,
    decide,
    family_matrix,
    folds,
    stable_fitness_order,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ONE_USE_LEDGER = Path("/home/ubuntu/jope_plan/ns/receipts/H147_exec_claim.txt")


def verify(pins):
    for name, version in pins.get("packages", {}).items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package version mismatch: " + name)
    for p, h in pins["inputs"].items():
        if sha(p) != h:
            raise RuntimeError("input identity mismatch: " + p)
    for p, h in pins.get("assets", {}).items():
        if sha(p) != h:
            raise RuntimeError("asset identity mismatch: " + p)
    for p, h in pins.get("executor_sha256", {}).items():
        if sha(HERE / p) != h:
            raise RuntimeError("executor identity mismatch: " + p)


def predict_dict(net, pools, tokens, masks, cids, indexes):
    progs = np.stack(
        [pools[c]["program"][ix] for c, ix in zip(cids, indexes, strict=True)]
    )
    proj = np.stack(
        [pools[c]["proj16"][ix] for c, ix in zip(cids, indexes, strict=True)]
    )
    b, k = progs.shape[:2]
    return net.adapter_10tok.predict_costs(
        ctx_feat=torch.zeros((b, 584)),
        prog_412=torch.as_tensor(progs),
        proj_16=torch.as_tensor(proj),
        base_prior_cost=torch.zeros((b, k)),
        scale=torch.full((b,), 0.4208),
        sos_prior=torch.zeros((b, k, 2)),
        mm_tokens_10x32=torch.stack([tokens[c] for c in cids]),
        mm_mask_10=torch.stack([masks[c] for c in cids]),
        revealed_indices=torch.arange(min(4, k)).reshape(1, -1).expand(b, -1),
        revealed_raw_f=torch.as_tensor(
            np.stack(
                [
                    pools[c]["fitness"][ix[: min(4, k)]]
                    for c, ix in zip(cids, indexes, strict=True)
                ]
            ),
            dtype=torch.float32,
        ),
        revealed_gate=torch.as_tensor(
            np.stack(
                [
                    pools[c]["gate"][ix[: min(4, k)]]
                    for c, ix in zip(cids, indexes, strict=True)
                ]
            ),
            dtype=torch.float32,
        ),
    )


def predict(net, pools, tokens, masks, cids, indexes):
    return predict_dict(net, pools, tokens, masks, cids, indexes)["total_cost"].float()


def candidate_utility(fitness_arr, gate_arr, idx):
    return float(-fitness_arr[idx] if bool(gate_arr[idx]) else -2.0)


def score_context_from_costs(cost_1d, pool):
    """Outcome-blind assessment choice and descriptive U4/unrevealed/anchor diagnostics on full pool in storage order."""
    cost = np.asarray(cost_1d, dtype=np.float64)
    if cost.ndim != 1 or len(cost) < SLATE_SIZE or not np.isfinite(cost).all():
        raise RuntimeError("invalid or nonfinite cost vector")
    fit = pool["fitness"]
    gate = pool["gate"]
    # Primary native top-1 choice (ties broken by lowest storage index via np.argmin)
    choice = int(np.argmin(cost))
    valid = bool(gate[choice])
    utility = candidate_utility(fit, gate, choice)

    # Unrevealed choices: indices 4..N-1 in storage order, ranked purely by predicted cost before inspecting gate/fitness
    unrev_order = 4 + np.argsort(cost[4:], kind="stable")
    unrev_top1_idx = int(unrev_order[0])
    unrev_top1_util = candidate_utility(fit, gate, unrev_top1_idx)

    top4_unrev = [int(x) for x in unrev_order[:4]]
    u4_unrev_util = max(candidate_utility(fit, gate, i) for i in top4_unrev)
    union8 = [0, 1, 2, 3, *top4_unrev]
    u4_union_util = max(candidate_utility(fit, gate, i) for i in union8)

    return dict(
        choice=choice,
        utility=utility,
        invalid=not valid,
        on_anchor=bool(choice < 4),
        unrev_top1_choice=unrev_top1_idx,
        unrev_top1_utility=unrev_top1_util,
        u4_top4_unrev_utility=u4_unrev_util,
        u4_union_anchors_and_top4_unrev_utility=u4_union_util,
    )


def evaluate(
    net,
    pools,
    tokens,
    masks,
    rows,
    split_name=None,
    use_cache=True,
    return_scores=False,
):
    if split_name is None:
        splits = {r.get("dl_split", "UNKNOWN") for r in rows}
        split_name = splits.pop() if len(splits) == 1 else "MIXED"
    groups = {}
    result = {}
    raw_scores = {}
    for r in rows:
        groups.setdefault(len(pools[r["context_sha256"]]["fitness"]), []).append(
            r["context_sha256"]
        )
    with torch.no_grad():
        for size, cids in sorted(groups.items()):
            for start in range(0, len(cids), 8):
                batch = cids[start : start + 8]
                ixs = [np.arange(size, dtype=np.int64) for _ in batch]
                net._eval_batch_key = (
                    eval_batch_cache_key(split_name, batch, ixs, pools)
                    if use_cache
                    else None
                )
                try:
                    scores = predict(
                        net,
                        pools,
                        tokens,
                        masks,
                        batch,
                        ixs,
                    ).numpy()
                finally:
                    net._eval_batch_key = None
                if not np.isfinite(scores).all():
                    raise RuntimeError("nonfinite predictions")
                for cid, cost in zip(batch, scores, strict=True):
                    p = pools[cid]
                    result[cid] = score_context_from_costs(cost, p)
                    if return_scores:
                        raw_scores[cid] = cost.copy()
    out = dict(
        values=[result[r["context_sha256"]]["utility"] for r in rows],
        choices=[result[r["context_sha256"]]["choice"] for r in rows],
        invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
        on_anchor=[result[r["context_sha256"]]["on_anchor"] for r in rows],
        unrev_top1_values=[
            result[r["context_sha256"]]["unrev_top1_utility"] for r in rows
        ],
        u4_top4_unrev_values=[
            result[r["context_sha256"]]["u4_top4_unrev_utility"] for r in rows
        ],
        u4_union_values=[
            result[r["context_sha256"]]["u4_union_anchors_and_top4_unrev_utility"]
            for r in rows
        ],
        context_ids=[r["context_sha256"] for r in rows],
    )
    if return_scores:
        out["scores"] = [raw_scores[r["context_sha256"]] for r in rows]
    return out


def selected_and_catalog_check():
    packet = json.loads((INPUT / "selected.json").read_text())
    train_only_packet = json.loads(TRAIN_SELECTED.read_text())
    train = train_only_packet["train"]
    val = packet["dev"]
    if train != packet["train"]:
        raise RuntimeError(
            "train_only_selected_1219.json mismatch with selected.json['train']"
        )
    if len(train) != 1219 or len(val) != 91:
        raise RuntimeError("incomplete cohort")
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
        raise RuntimeError("family overlap between DL-Train and DL-Val")

    # Identity-only check against authoritative S3_INVENTORY (PREREG.md lines 46-48)
    s3_inv = json.loads(S3_INVENTORY.read_text())
    sealed_ids = {
        str(item["id"])
        for item in s3_inv.get("items", [])
        if bool(item.get("has_meta_test", False))
        or str(item.get("origin", "")) != "real"
    }
    train_dataset_ids = {
        str(r.get("dataset_id", r.get("context_sha256"))) for r in train
    }
    overlap = sorted(train_dataset_ids & sealed_ids)
    if overlap:
        raise RuntimeError(f"sealed/non-real S3 catalog overlap: {overlap[:5]}")
    catalog_receipt = dict(
        s3_inventory_path=str(S3_INVENTORY),
        train_contexts_checked=len(train),
        train_families_checked=len({r["family_id"] for r in train}),
        known_catalog_sealed_id_overlap=0,
        global_sealed_alias_roster_complete=False,
        disclosure="Checked by identity only against S3_INVENTORY; missing global sealed alias roster remains disclosed and is not reported as a complete universal exclusion certificate.",
    )
    return train, val, catalog_receipt


def baseline_gate(values, reference):
    expected = np.asarray(reference["utility"]["per_context"], dtype=np.float64)
    actual = np.asarray(values, dtype=np.float64)
    if (
        expected.shape != (91,)
        or actual.shape != (91,)
        or not np.isfinite(actual).all()
    ):
        raise RuntimeError("91 finite contexts required")
    err = float(np.max(np.abs(actual - expected)))
    mean_err = abs(float(actual.mean()) - reference["best_utility"])
    if err != 0.0 or mean_err != 0.0:
        raise RuntimeError(f"baseline exact parity failed: {err}, {mean_err}")
    return dict(
        matches=91,
        max_context_error=err,
        mean_error=mean_err,
        mean=float(actual.mean()),
        status="PASS",
    )


def two_load_train_baseline_gate(eval_load1, eval_load2):
    u1 = np.asarray(eval_load1["values"], dtype=np.float64)
    u2 = np.asarray(eval_load2["values"], dtype=np.float64)
    if (
        u1.shape != (1219,)
        or u2.shape != (1219,)
        or not np.isfinite(u1).all()
        or not np.isfinite(u2).all()
    ):
        raise RuntimeError("1219 finite DL-Train contexts required")
    max_u_diff = float(np.max(np.abs(u1 - u2)))
    choices_match = eval_load1["choices"] == eval_load2["choices"]
    scores_match = all(
        np.array_equal(a, b)
        for a, b in zip(eval_load1["scores"], eval_load2["scores"], strict=True)
    )
    if max_u_diff != 0.0 or not choices_match or not scores_match:
        raise RuntimeError(
            f"two-load DL-Train r882 parity failed: max_u_diff={max_u_diff}, choices_match={choices_match}, scores_match={scores_match}"
        )
    vec_sha = hashlib.sha256(np.ascontiguousarray(u1).tobytes()).hexdigest()
    choice_sha = hashlib.sha256(
        np.ascontiguousarray(
            np.asarray(eval_load1["choices"], dtype=np.int64)
        ).tobytes()
    ).hexdigest()
    return dict(
        matches=1219,
        max_context_error=max_u_diff,
        choices_identical=True,
        raw_scores_bitwise_identical=True,
        context_mean_utility=float(u1.mean()),
        utility_vector_sha256=vec_sha,
        choice_vector_sha256=choice_sha,
        status="PASS",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()  # Never accepts cached cells or an old attempt.
    started_monotonic = time.monotonic()
    launch_guard = guards(out.parent, started_monotonic=started_monotonic)
    pins = json.loads((HERE / "manifest.json").read_text())
    allowed = set(pins["inputs"].keys())
    train, val, catalog_receipt = selected_and_catalog_check()
    allowed.update(
        str(INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    # Explicitly forbid opening the mixed 1,310-context H58 cache (PREREG.md line 52)
    audit = Audit(allowed, out, forbidden_reads=(MIXED_H58_CACHE,))
    sys.addaudithook(audit.check)
    updates = 0
    try:
        verify(pins)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialization forbidden")
        checkpoint = Path(pins["checkpoint"])
        net, guard, names, frozen, fused = build(checkpoint)
        assert_frozen_anchor(net, names, net._anchor_sha256)
        elo_mod = load_elo_tracker()
        census = dropout_census(net)
        if census["adapter_dropout_modules"] or any(
            m["p"] != 0 for m in census["adapter_attention_dropout"]
        ):
            raise RuntimeError("prospective dropout exclusion invalid")
        if (
            census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
            or census["adapter_training"]
            or census["dope_training"]
        ):
            raise RuntimeError("effective mode changed")
        if args.smoke:
            check_frozen(net, guard, names, frozen)
            end_guard = guards(out.parent, started_monotonic=started_monotonic)
            assets = {
                p: sha(p)
                for p in sorted(audit.opened)
                if p.startswith("/home/ubuntu/JopeDimeMicro/")
                and Path(p).is_file()
                and not p.endswith(".pyc")
            }
            for m in list(sys.modules.values()):
                p = getattr(m, "__file__", None)
                if (
                    p
                    and str(p).startswith("/home/ubuntu/JopeDimeMicro/")
                    and str(p).endswith(".py")
                ):
                    assets[p] = sha(p)
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    candidate_arms_scored=0,
                    census=census,
                    catalog_receipt=catalog_receipt,
                    assets=assets,
                    packages={
                        n: importlib.metadata.version(n)
                        for n in ("numpy", "torch", "psutil")
                    },
                    python=sys.version,
                    frozen_sha256=frozen,
                    anchor_sha256=net._anchor_sha256,
                    adapter_names=list(names),
                    dope_sha256=DOPE,
                    launch_guard=launch_guard,
                    end_guard=end_guard,
                ),
            )
            return

        if not ONE_USE_LEDGER.exists():
            raise RuntimeError(f"missing one-use ledger: {ONE_USE_LEDGER}")
        atomic_claim(
            out / "exec_claim.json",
            dict(study="H147", ledger=str(ONE_USE_LEDGER)),
        )
        launch = json.loads((out.parent / "launch.json").read_text())
        if not launch["merged_source_verified"] or launch["manifest_sha256"] != sha(
            HERE / "manifest.json"
        ):
            raise RuntimeError("merged launch required")
        binding = bind_source(launch, prereg_sha=pins["prereg_sha256"])

        fold_doc = json.loads((HERE / "fold_manifest.json").read_text())
        calc_groups = folds(train)
        doc_groups = [f["assessment_families"] for f in fold_doc["folds"]]
        if calc_groups != doc_groups:
            raise RuntimeError("fold_manifest.json family groups mismatch")

        fold_specs = []
        seen_assess_cids = set()
        for f_entry in fold_doc["folds"]:
            fk = int(f_entry["fold"])
            assess_fams = set(f_entry["assessment_families"])
            assess_rows = [r for r in train if r["family_id"] in assess_fams]
            fit_rows = [r for r in train if r["family_id"] not in assess_fams]
            if [r["context_sha256"] for r in assess_rows] != f_entry[
                "assessment_context_ids"
            ]:
                raise RuntimeError(f"fold {fk} assessment_context_ids mismatch")
            if (
                len(assess_fams) != 41
                or len({r["family_id"] for r in fit_rows}) != 164
                or ({r["family_id"] for r in fit_rows} & assess_fams)
            ):
                raise RuntimeError(f"fold {fk} family partition invalid")
            seen_assess_cids.update(r["context_sha256"] for r in assess_rows)
            fold_specs.append(
                dict(
                    fold=fk,
                    fit_rows=fit_rows,
                    assess_rows=assess_rows,
                    fit_ids={r["context_sha256"] for r in fit_rows},
                    assess_ids={r["context_sha256"] for r in assess_rows},
                )
            )
        if len(seen_assess_cids) != 1219:
            raise RuntimeError(
                "5 assessment folds do not partition 1219 DL-Train contexts"
            )

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

        pools = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for r in train + val:
            cid = r["context_sha256"]
            p = INPUT / "pools" / f"{cid}.npz"
            if sha(p) != r["pool_sha256"]:
                raise RuntimeError("pool SHA mismatch")
            with np.load(p, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                fitness = z["fitness"].astype(np.float64)
                gate = z["gate"].astype(bool)
                if len(fitness) < SLATE_SIZE:
                    raise RuntimeError(
                        f"pool {cid} has fewer than {SLATE_SIZE} entries"
                    )
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=fitness,
                    gate=gate,
                )

        # Load isolated TRAIN-only feature packet (1,219 contexts) and isolated VAL feature packet (91 contexts)
        train_cids_set = {r["context_sha256"] for r in train}
        val_cids_set = {r["context_sha256"] for r in val}
        tokens, masks = {}, {}
        with np.load(TRAIN_CACHE, allow_pickle=False) as z:
            tr_ids = list(map(str, z["cids"]))
            if set(tr_ids) != train_cids_set or len(tr_ids) != 1219:
                raise RuntimeError("TRAIN_CACHE closure mismatch")
            tok, msk = z["tokens_10x32"], z["mask_10"]
            for i, c in enumerate(tr_ids):
                tokens[c] = torch.as_tensor(tok[i], dtype=torch.float32)
                masks[c] = torch.as_tensor(msk[i], dtype=torch.bool)
        with np.load(VAL_CACHE, allow_pickle=False) as z:
            va_ids = list(map(str, z["cids"]))
            if set(va_ids) != val_cids_set or len(va_ids) != 91:
                raise RuntimeError("VAL_CACHE closure mismatch")
            tok, msk = z["tokens_10x32"], z["mask_10"]
            for i, c in enumerate(va_ids):
                tokens[c] = torch.as_tensor(tok[i], dtype=torch.float32)
                masks[c] = torch.as_tensor(msk[i], dtype=torch.bool)

        # Step 1a: Exact 91/91 r882 DL-Val baseline gate (hwave_t0000_h147exec_gpu_fused_agy.md)
        baseline_val = evaluate(net, pools, tokens, masks, val, split_name="DL-Val")
        val_sanity = baseline_gate(baseline_val["values"], ref)
        save(out / "baseline.json", dict(**baseline_val, sanity=val_sanity))
        print("BASELINE DL-VAL PASS", val_sanity, flush=True)

        # Step 1b: Two independent r882 loads on 1,219-ctx DL-Train (PREREG.md lines 167-169)
        # Evaluate per assessment fold so each fold's frozen anchor/sem24 cache is populated for cell assessments
        net2, guard2, names2, frozen2, _ = build(checkpoint)
        if frozen2 != frozen or names2 != names:
            raise RuntimeError("second r882 load frozen hash mismatch")
        train_idx_map = {r["context_sha256"]: i for i, r in enumerate(train)}
        load1_by_cid, load2_by_cid = {}, {}
        for fspec in fold_specs:
            fk = fspec["fold"]
            arows = fspec["assess_rows"]
            ev1 = evaluate(
                net,
                pools,
                tokens,
                masks,
                arows,
                split_name=f"ASSESS-fold{fk}",
                use_cache=True,
                return_scores=True,
            )
            ev2 = evaluate(
                net2,
                pools,
                tokens,
                masks,
                arows,
                split_name=f"ASSESS-fold{fk}",
                use_cache=False,
                return_scores=True,
            )
            for idx_r, r in enumerate(arows):
                cid = r["context_sha256"]
                load1_by_cid[cid] = {k: ev1[k][idx_r] for k in ev1}
                load2_by_cid[cid] = {k: ev2[k][idx_r] for k in ev2}
        del net2, guard2

        def assemble_train_eval(by_cid_map, include_scores=False):
            keys = (
                "values",
                "choices",
                "invalid",
                "on_anchor",
                "unrev_top1_values",
                "u4_top4_unrev_values",
                "u4_union_values",
                "context_ids",
            )
            out_d = {
                k: [by_cid_map[r["context_sha256"]][k] for r in train] for k in keys
            }
            if include_scores:
                out_d["scores"] = [
                    by_cid_map[r["context_sha256"]]["scores"] for r in train
                ]
            return out_d

        baseline_train_1 = assemble_train_eval(load1_by_cid, include_scores=True)
        baseline_train_2 = assemble_train_eval(load2_by_cid, include_scores=True)
        train_sanity = two_load_train_baseline_gate(baseline_train_1, baseline_train_2)
        baseline_train = assemble_train_eval(load1_by_cid, include_scores=False)
        save(
            out / "baseline_train.json",
            dict(**baseline_train, sanity=train_sanity),
        )
        check_frozen(net, guard, names, frozen)
        print("BASELINE DL-TRAIN TWO-LOAD PASS", train_sanity, flush=True)

        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        params = [p for p in net.parameters() if p.requires_grad]
        initial_vec = torch.cat([initial[n].reshape(-1) for n in names])
        orders = {
            r["context_sha256"]: stable_fitness_order(
                pools[r["context_sha256"]]["fitness"]
            )
            for r in train
        }

        identity = dict(
            source_commit=launch["source_commit"],
            reviewed_head=launch["reviewed_head"],
            binding=binding,
            manifest_sha256=sha(HERE / "manifest.json"),
            prereg_sha256=pins["prereg_sha256"],
            prereg_amendment_a1_sha256=pins["prereg_amendment_a1_sha256"],
            fold_manifest_sha256=sha(HERE / "fold_manifest.json"),
            one_use_ledger=str(ONE_USE_LEDGER),
            one_use_ledger_sha256=sha(ONE_USE_LEDGER),
        )

        # Panels for out-of-fold DL-Train (3 seeds x 1219 contexts) and descriptive DL-Val (3 seeds x 5 folds x 91 contexts)
        oof_values = {arm: np.full((3, 1219), np.nan, dtype=np.float64) for arm in ARMS}
        oof_choices = {arm: np.full((3, 1219), -1, dtype=np.int64) for arm in ARMS}
        oof_invalid = {arm: np.zeros((3, 1219), dtype=bool) for arm in ARMS}
        oof_on_anchor = {arm: np.zeros((3, 1219), dtype=bool) for arm in ARMS}
        oof_unrev_top1 = {
            arm: np.full((3, 1219), np.nan, dtype=np.float64) for arm in ARMS
        }
        oof_u4_unrev = {
            arm: np.full((3, 1219), np.nan, dtype=np.float64) for arm in ARMS
        }
        oof_u4_union = {
            arm: np.full((3, 1219), np.nan, dtype=np.float64) for arm in ARMS
        }
        val_fold_values = {
            arm: np.full((3, 5, 91), np.nan, dtype=np.float64) for arm in ARMS
        }
        val_fold_invalid = {arm: np.zeros((3, 5, 91), dtype=bool) for arm in ARMS}

        cells = {}
        slate_stats_total = dict(
            total_slates=0,
            a_equals_b_set_slates=0,
            a_equals_b_ordered_slates=0,
            support_changed_vs_h144_slates=0,
        )
        grad_mass_by_arm = {arm: [] for arm in ARMS}
        loss_mass_by_arm = {arm: [] for arm in ARMS}

        for fspec in fold_specs:
            fk = fspec["fold"]
            fit_ids = fspec["fit_ids"]
            assess_ids = fspec["assess_ids"]
            assess_rows = fspec["assess_rows"]
            assess_ix = [train_idx_map[r["context_sha256"]] for r in assess_rows]

            for s_idx, seed in enumerate(SEEDS):
                guards(out.parent, started_monotonic=started_monotonic)
                plan = batch_plan(fit_ids, orders, seed, fk)
                if len(plan) != 40:
                    raise RuntimeError("40-step plan required")

                for step_item in plan:
                    slate_stats_total["total_slates"] += 16
                    slate_stats_total["a_equals_b_set_slates"] += int(
                        step_item["a_equals_b_set_count"]
                    )
                    slate_stats_total["a_equals_b_ordered_slates"] += int(
                        step_item["a_equals_b_ordered_count"]
                    )
                    slate_stats_total["support_changed_vs_h144_slates"] += int(
                        step_item["support_changed_vs_h144_count"]
                    )

                for arm in ARMS:
                    guards(out.parent, started_monotonic=started_monotonic)
                    cell_id = f"{arm}-fold{fk}-{seed}"
                    atomic_claim(
                        out / f"{cell_id}.started",
                        dict(
                            cell_id=cell_id,
                            arm=arm,
                            fold=fk,
                            seed=seed,
                            started_utc=datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    t_cell = time.monotonic()
                    net.load_state_dict(initial, strict=True)
                    check_frozen(net, guard, names, frozen)
                    optimizer = torch.optim.AdamW(
                        params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                    )
                    mask_rev = arm == "STORAGE_MASK"
                    losses = []
                    step_grad_shares = []
                    step_loss_shares = []

                    for step_idx, step_item in enumerate(plan, start=1):
                        stream = step_item["stream"]
                        cids = step_item["cids"]
                        if (
                            not set(cids) <= fit_ids
                            or (set(cids) & assess_ids)
                            or (set(cids) & val_cids_set)
                        ):
                            raise RuntimeError("minibatch family/split leakage")
                        ix_ctrl = step_item["ix_support_ctrl"]
                        ix_stor = step_item["ix_storage"]
                        for ic, is_ in zip(ix_ctrl, ix_stor, strict=True):
                            if len(ic) != SLATE_SIZE or set(ic.tolist()) != set(
                                is_.tolist()
                            ):
                                raise RuntimeError(
                                    "common support S mismatch across arms"
                                )

                        ixs = ix_ctrl if arm == "SUPPORT_CTRL" else ix_stor
                        split_tag = (
                            f"FIT-CTRL-f{fk}-s{seed}-t{step_idx}"
                            if arm == "SUPPORT_CTRL"
                            else f"FIT-STOR-f{fk}-s{seed}-t{step_idx}"
                        )
                        torch.manual_seed(stream)
                        optimizer.zero_grad(set_to_none=True)
                        net._eval_batch_key = eval_batch_cache_key(
                            split_tag, cids, ixs, pools
                        )
                        try:
                            pred = predict(net, pools, tokens, masks, cids, ixs)
                        finally:
                            net._eval_batch_key = None

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
                        if step_idx in (1, 20, 40):
                            diag = revealed_loss_and_grad_diagnostics(
                                pred, fitness, gate, mask_revealed=mask_rev
                            )
                            if mask_rev and (
                                diag["revealed_loss_share"] != 0.0
                                or diag["revealed_pred_grad_norm2_share"] != 0.0
                            ):
                                raise RuntimeError(
                                    "STORAGE_MASK nonzero revealed loss/grad share"
                                )
                            step_loss_shares.append(diag["revealed_loss_share"])
                            step_grad_shares.append(
                                diag["revealed_pred_grad_norm2_share"]
                            )

                        objective = loss(pred, fitness, gate, mask_revealed=mask_rev)
                        if not torch.isfinite(objective):
                            raise RuntimeError("nonfinite loss")
                        objective.backward()
                        torch.nn.utils.clip_grad_norm_(
                            params, 1.0, error_if_nonfinite=True
                        )
                        optimizer.step()
                        updates += 1
                        check_frozen(net, guard, names, frozen)
                        losses.append(float(objective.detach()))

                    drift = float(
                        torch.linalg.vector_norm(
                            torch.cat(
                                [
                                    dict(net.named_parameters())[n].detach().reshape(-1)
                                    for n in names
                                ]
                            )
                            - initial_vec
                        )
                    )
                    if not drift > 0:
                        raise RuntimeError(f"no parameter update for {cell_id}")

                    assess_eval = evaluate(
                        net,
                        pools,
                        tokens,
                        masks,
                        assess_rows,
                        split_name=f"ASSESS-fold{fk}",
                        use_cache=True,
                    )
                    check_frozen(net, guard, names, frozen)

                    oof_values[arm][s_idx, assess_ix] = assess_eval["values"]
                    oof_choices[arm][s_idx, assess_ix] = assess_eval["choices"]
                    oof_invalid[arm][s_idx, assess_ix] = assess_eval["invalid"]
                    oof_on_anchor[arm][s_idx, assess_ix] = assess_eval["on_anchor"]
                    oof_unrev_top1[arm][s_idx, assess_ix] = assess_eval[
                        "unrev_top1_values"
                    ]
                    oof_u4_unrev[arm][s_idx, assess_ix] = assess_eval[
                        "u4_top4_unrev_values"
                    ]
                    oof_u4_union[arm][s_idx, assess_ix] = assess_eval["u4_union_values"]

                    # Lock out-of-fold assessment receipt to disk before descriptive DL-Val evaluation
                    cell_summary = dict(
                        cell_id=cell_id,
                        arm=arm,
                        fold=fk,
                        seed=seed,
                        effective_stream_seed=int(seed) + 1_000_000 * int(fk),
                        updates=40,
                        adapter_l2_drift=drift,
                        adapter_sha256=tensor_hash(
                            {n: net.state_dict()[n] for n in names}
                        ),
                        frozen_sha256=frozen,
                        dope_sha256=DOPE,
                        assess_context_count=len(assess_rows),
                        assess_mean_utility=float(np.mean(assess_eval["values"])),
                        assess_invalid_count=int(sum(assess_eval["invalid"])),
                        assess_anchor_choice_fraction=float(
                            np.mean(assess_eval["on_anchor"])
                        ),
                        revealed_loss_share_mean=float(np.mean(step_loss_shares)),
                        revealed_pred_grad_norm2_share_mean=float(
                            np.mean(step_grad_shares)
                        ),
                        loss_first=losses[0],
                        loss_last=losses[-1],
                        wall_seconds=time.monotonic() - t_cell,
                    )
                    save(out / f"{cell_id}.json", cell_summary)

                    val_eval = evaluate(
                        net,
                        pools,
                        tokens,
                        masks,
                        val,
                        split_name="DL-Val",
                        use_cache=True,
                    )
                    check_frozen(net, guard, names, frozen)
                    val_fold_values[arm][s_idx, fk, :] = val_eval["values"]
                    val_fold_invalid[arm][s_idx, fk, :] = val_eval["invalid"]

                    grad_mass_by_arm[arm].extend(step_grad_shares)
                    loss_mass_by_arm[arm].extend(step_loss_shares)
                    cells[arm, fk, seed] = cell_summary
                    print(
                        "CELL COMPLETE",
                        cell_id,
                        f"({len(cells)}/45)",
                        "drift",
                        round(drift, 6),
                        "assess_mean",
                        round(cell_summary["assess_mean_utility"], 7),
                        "val_mean",
                        round(float(np.mean(val_eval["values"])), 7),
                        "sec",
                        round(cell_summary["wall_seconds"], 2),
                        flush=True,
                    )

                # Evict FIT-* step cache entries after all 3 arms of (seed, fold) complete to bound RAM
                for cache_dict in (net._anchor_cache, net._sem24_cache):
                    for k in [k for k in cache_dict if k[0].startswith("FIT-")]:
                        del cache_dict[k]

        if updates != 1800 or len(cells) != 45:
            raise RuntimeError(
                f"incomplete wave: updates={updates}, cells={len(cells)}"
            )
        for arm in ARMS:
            if (
                np.isnan(oof_values[arm]).any()
                or (oof_choices[arm] < 0).any()
                or np.isnan(val_fold_values[arm]).any()
            ):
                raise RuntimeError(f"missing out-of-fold or val entries for {arm}")

        base_tr_vals = np.asarray(baseline_train["values"], dtype=np.float64)
        vs_r882 = {arm: contrast(train, oof_values[arm], base_tr_vals) for arm in ARMS}
        vs_support_ctrl = {
            arm: contrast(train, oof_values[arm], oof_values["SUPPORT_CTRL"])
            for arm in CANDIDATE_ARMS
        }
        mask_minus_reveal = contrast(
            train, oof_values["STORAGE_MASK"], oof_values["STORAGE_REVEAL"]
        )
        decision = decide(vs_r882, vs_support_ctrl)
        bt_elo = bradley_terry_elo_summary(train, base_tr_vals, oof_values, elo_mod)

        # Descriptive 91-ctx DL-Val readout (mean across the 5 fold models per seed, plus per-fold breakdown)
        val_seed_panels = {arm: val_fold_values[arm].mean(axis=1) for arm in ARMS}
        descriptive_dl_val = {}
        for arm in ARMS:
            v_mat = val_seed_panels[arm]
            descriptive_dl_val[arm] = dict(
                context_mean_utility_by_seed=v_mat.mean(axis=1).tolist(),
                family_mean_utility_by_seed=family_matrix(val, v_mat)[1]
                .mean(axis=1)
                .tolist(),
                vs_r882=contrast(val, v_mat, baseline_val["values"]),
                vs_SUPPORT_CTRL=contrast(val, v_mat, val_seed_panels["SUPPORT_CTRL"]),
                invalid_count_mean_across_folds_by_seed=val_fold_invalid[arm]
                .sum(axis=2)
                .mean(axis=1)
                .tolist(),
            )
        descriptive_dl_val["bradley_terry_elo"] = bradley_terry_elo_summary(
            val, baseline_val["values"], val_seed_panels, elo_mod
        )

        # Descriptive diagnostics on 1,219-ctx DL-Train (U4, unrevealed top-1, anchor choice fraction, A=B slates, loss/grad mass)
        descriptive_train = {
            "r882": dict(
                context_mean_utility=float(base_tr_vals.mean()),
                family_mean_utility=float(
                    family_matrix(train, np.tile(base_tr_vals, (3, 1)))[1].mean()
                ),
                invalid_count=int(sum(baseline_train["invalid"])),
                anchor_choice_fraction=float(np.mean(baseline_train["on_anchor"])),
                unrev_top1_family_mean=float(
                    family_matrix(
                        train,
                        np.tile(baseline_train["unrev_top1_values"], (3, 1)),
                    )[1].mean()
                ),
                u4_top4_unrev_family_mean=float(
                    family_matrix(
                        train,
                        np.tile(baseline_train["u4_top4_unrev_values"], (3, 1)),
                    )[1].mean()
                ),
                u4_union_anchors_and_top4_unrev_family_mean=float(
                    family_matrix(
                        train, np.tile(baseline_train["u4_union_values"], (3, 1))
                    )[1].mean()
                ),
            )
        }
        primary_arms_summary = {}
        for arm in ARMS:
            drifts_arm = [
                cells[arm, fk, s]["adapter_l2_drift"] for s in SEEDS for fk in range(5)
            ]
            drifts_by_seed = [
                float(
                    np.mean([cells[arm, fk, s]["adapter_l2_drift"] for fk in range(5)])
                )
                for s in SEEDS
            ]
            primary_arms_summary[arm] = dict(
                context_mean_utility_by_seed=oof_values[arm].mean(axis=1).tolist(),
                family_mean_utility_by_seed=family_matrix(train, oof_values[arm])[1]
                .mean(axis=1)
                .tolist(),
                vs_r882=vs_r882[arm],
                vs_SUPPORT_CTRL=(
                    vs_support_ctrl[arm]
                    if arm in CANDIDATE_ARMS
                    else contrast(train, oof_values[arm], oof_values["SUPPORT_CTRL"])
                ),
                bradley_terry_elo=bt_elo[arm],
                invalid_counts_by_seed=oof_invalid[arm].sum(axis=1).tolist(),
                anchor_choice_fraction_by_seed=oof_on_anchor[arm].mean(axis=1).tolist(),
                drift_mean_by_seed=drifts_by_seed,
                drift_median_all_15_cells=float(np.median(drifts_arm)),
                drift_min_max=[float(min(drifts_arm)), float(max(drifts_arm))],
            )
            descriptive_train[arm] = dict(
                unrev_top1_vs_r882=contrast(
                    train,
                    oof_unrev_top1[arm],
                    baseline_train["unrev_top1_values"],
                ),
                u4_top4_unrev_vs_r882=contrast(
                    train,
                    oof_u4_unrev[arm],
                    baseline_train["u4_top4_unrev_values"],
                ),
                u4_union_anchors_and_top4_unrev_vs_r882=contrast(
                    train, oof_u4_union[arm], baseline_train["u4_union_values"]
                ),
                anchor_choice_fraction_by_seed=oof_on_anchor[arm].mean(axis=1).tolist(),
                revealed_loss_share_mean=float(np.mean(loss_mass_by_arm[arm])),
                revealed_pred_grad_norm2_share_mean=float(
                    np.mean(grad_mass_by_arm[arm])
                ),
            )

        tot_slates = max(1, slate_stats_total["total_slates"])
        slate_diagnostics = dict(
            **slate_stats_total,
            a_equals_b_set_percentage=100.0
            * slate_stats_total["a_equals_b_set_slates"]
            / tot_slates,
            a_equals_b_ordered_percentage=100.0
            * slate_stats_total["a_equals_b_ordered_slates"]
            / tot_slates,
            support_changed_vs_h144_percentage=100.0
            * slate_stats_total["support_changed_vs_h144_slates"]
            / tot_slates,
        )

        seed_pseudoreplication = {
            arm: dict(
                identical_oof_utilities_across_seeds=len(
                    {tuple(oof_values[arm][s].tolist()) for s in range(3)}
                )
                == 1,
                identical_oof_choices_across_seeds=len(
                    {tuple(oof_choices[arm][s].tolist()) for s in range(3)}
                )
                == 1,
                unique_cell_adapter_hashes=len(
                    {
                        cells[arm, fk, s]["adapter_sha256"]
                        for fk in range(5)
                        for s in SEEDS
                    }
                ),
            )
            for arm in ARMS
        }
        cross_arm_pseudoreplication = {
            f"{arm}_vs_SUPPORT_CTRL": dict(
                choice_diffs_out_of_1219_by_seed=[
                    int(np.sum(oof_choices[arm][s] != oof_choices["SUPPORT_CTRL"][s]))
                    for s in range(3)
                ],
                identical_oof_utilities_by_seed=[
                    bool(
                        np.array_equal(
                            oof_values[arm][s], oof_values["SUPPORT_CTRL"][s]
                        )
                    )
                    for s in range(3)
                ],
            )
            for arm in CANDIDATE_ARMS
        }
        cross_arm_pseudoreplication["STORAGE_MASK_vs_STORAGE_REVEAL"] = dict(
            choice_diffs_out_of_1219_by_seed=[
                int(
                    np.sum(
                        oof_choices["STORAGE_MASK"][s]
                        != oof_choices["STORAGE_REVEAL"][s]
                    )
                )
                for s in range(3)
            ],
            identical_oof_utilities_by_seed=[
                bool(
                    np.array_equal(
                        oof_values["STORAGE_MASK"][s],
                        oof_values["STORAGE_REVEAL"][s],
                    )
                )
                for s in range(3)
            ],
        )

        end_guard = guards(out.parent, started_monotonic=started_monotonic)
        metrics = dict(
            wave="H147",
            status="COMPLETE",
            verdict=decision["verdict"],
            best_advancing_arm=decision["best_advancing_arm"],
            primary_oof_dl_train_1219ctx_205fam=primary_arms_summary,
            decision=decision,
            bradley_terry_elo_oof_dl_train=bt_elo,
            masking_contrast_storage_mask_minus_storage_reveal=mask_minus_reveal,
            descriptive_diagnostics_dl_train=descriptive_train,
            descriptive_slate_support_diagnostics=slate_diagnostics,
            descriptive_dl_val_91ctx_19fam=descriptive_dl_val,
            seed_pseudoreplication=seed_pseudoreplication,
            cross_arm_pseudoreplication=cross_arm_pseudoreplication,
            baseline_dl_val_91ctx=val_sanity,
            baseline_dl_val_family_utility=float(
                family_matrix(val, np.tile(baseline_val["values"], (3, 1)))[1].mean()
            ),
            baseline_dl_train_two_load_1219ctx=train_sanity,
            baseline_dl_train_family_utility=descriptive_train["r882"][
                "family_mean_utility"
            ],
            catalog_receipt=catalog_receipt,
            historical_r882_train_exposure=True,
            continuation_held_out_only=True,
            serving_promotion=False,
            planned_cells=45,
            completed_cells=len(cells),
            planned_optimizer_updates=1800,
            actual_optimizer_updates=updates,
            seeds=list(SEEDS),
            arms=list(ARMS),
            identity=identity,
            launch_guard=launch_guard,
            end_guard=end_guard,
            cpu_seconds=round(cpu_seconds_used(), 3),
            wall_seconds=time.monotonic() - started_monotonic,
        )
        if len(json.dumps(metrics).encode()) >= 100000:
            raise RuntimeError("metrics.json exceeds 100 KB limit")
        np.savez_compressed(
            out / "panels.npz",
            r882_train=base_tr_vals,
            **{f"oof_values_{a}": oof_values[a] for a in ARMS},
            **{f"oof_choices_{a}": oof_choices[a] for a in ARMS},
            **{f"val_fold_values_{a}": val_fold_values[a] for a in ARMS},
        )
        verify(pins)
        save(out / "metrics.json", metrics)
    finally:
        save(
            out / "audit.json",
            dict(
                **audit.receipt(),
                optimizer_updates=updates,
                cpu_seconds=round(cpu_seconds_used(), 3),
                wall_seconds=time.monotonic() - started_monotonic,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                threads_at_end=len(list(Path("/proc/self/task").iterdir())),
                cuda_initialized=torch.cuda.is_initialized(),
            ),
        )


if __name__ == "__main__":
    main()
