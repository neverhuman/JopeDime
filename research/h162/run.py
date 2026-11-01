"""Single H162 continuation wave: cosine warm-restart LR schedule with snapshot selection vs LR-matched constant control."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as stat  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def verify(pins):
    for kind in ("inputs", "assets"):
        for p, h in pins[kind].items():
            if sha(p) != h:
                raise RuntimeError("identity mismatch: " + p)
    for p, h in pins["source_sha256"].items():
        if sha(HERE / p) != h:
            raise RuntimeError("source identity mismatch: " + p)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package version: " + name)


def parent_executor():
    parent = HERE.parent / "flywheel_selfplay/h146_lookahead_exec"
    sys.path.insert(0, str(parent))
    spec = importlib.util.spec_from_file_location("h162_parent", parent / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module, sys.modules["runtime"]


def forward_adapter_with_anchor(
    net,
    prog_t,
    sos_t,
    tok_t,
    mask_t,
    scale_t,
    rev_idx,
    rev_raw_f,
    rev_gate,
    anchor_total,
    anchor_eff_prior,
):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    ctx_pooled, cand_q, sem24, mm_prior_adj = (
        ad.encode_multimodal_context_and_candidates(
            mm_tokens_10x32=tok_t,
            mm_mask_10=mask_t,
            prog_412=prog_t,
            sos_prior=sos_t,
            active_modality_override_mask=None,
        )
    )
    static = {
        "eff_prior": anchor_eff_prior + mm_prior_adj,
        "mm_prior_adj": mm_prior_adj,
        "sem24": sem24,
        "cand_q": cand_q,
        "cand_q_norm": F.normalize(cand_q, p=2, dim=-1, eps=1e-8),
        "ctx_pooled": ctx_pooled,
    }
    delta = ad.compute_adapter_delta_from_static_terms(
        static_terms=static,
        scale=scale_t,
        revealed_indices=rev_idx,
        revealed_raw_f=rev_raw_f,
        revealed_gate=rev_gate,
    )
    return (anchor_total + delta).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()  # Exclusive, no result/cache reuse or automatic retry.
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    parent, rt = parent_executor()
    import torch

    train, val = parent.selected()
    if len({r["family_id"] for r in train}) != 205:
        raise RuntimeError("TRAIN family inventory")
    groups = stat.folds(train)
    if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"])
    if args.launch is not None:
        allowed.add(str(args.launch.resolve()))
    allowed.update(
        str(rt.INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    updates = 0
    started = time.monotonic()
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or any(m["p"] != 0 for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
            or census["adapter_training"]
            or census["dope_training"]
        ):
            raise RuntimeError("effective inference/training mode changed")
        rt.check_frozen(net, guard, names, frozen)
        params_check = [dict(net.named_parameters())[n] for n in names]
        trainable_params = sum(p.numel() for p in params_check)
        if trainable_params != 29211:
            raise RuntimeError(
                f"expected 29211 trainable adapter params, got {trainable_params}"
            )
        for arm in stat.ARMS:
            sched = stat.lr_schedule(arm, 40)
            if len(sched) != 40 or not np.isfinite(sched).all():
                raise RuntimeError(f"invalid schedule for {arm}")

        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        init_vec = torch.cat(
            [initial[n].detach().cpu().reshape(-1) for n in names]
        ).float()

        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    wave="H162",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=len(names),
                    trainable_parameters=trainable_params,
                    arms=list(stat.ARMS),
                    arm_steps={k: list(v) for k, v in stat.ARM_STEPS.items()},
                    mean_lrs={
                        k: float(np.mean(stat.lr_schedule(k, 40))) for k in stat.ARMS
                    },
                    census=census,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    packages=pins["packages"],
                ),
            )
            return

        launch = json.loads(args.launch.read_text())
        if (
            launch["manifest_sha256"] != sha(HERE / "manifest.json")
            or not launch["merged_source_verified"]
        ):
            raise RuntimeError("merged preregistration required")

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
            raise RuntimeError("evaluation identity")

        pools = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for row in train + val:
            cid = row["context_sha256"]
            path = rt.INPUT / "pools" / f"{cid}.npz"
            if rt.sha(path) != row["pool_sha256"]:
                raise RuntimeError("pool identity")
            with np.load(path, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )

        device = torch.device("cpu")
        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            if set(ids) != set(pools) or len(ids) != len(pools):
                raise RuntimeError("feature cache closure")
            tok, mask = z["tokens_10x32"], z["mask_10"]
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32, device=device)
                for i, c in enumerate(ids)
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool, device=device)
                for i, c in enumerate(ids)
            }

        eval_cache = {}

        def precompute_cohort(cohort_rows):
            with torch.no_grad():
                cohort_groups = {}
                for r in cohort_rows:
                    c = r["context_sha256"]
                    cohort_groups.setdefault(len(pools[c]["fitness"]), []).append(c)
                for size, cids in sorted(cohort_groups.items()):
                    for start in range(0, len(cids), 8):
                        batch = cids[start : start + 8]
                        b = len(batch)
                        k = size
                        progs = torch.as_tensor(
                            np.stack([pools[c]["program"] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        proj = torch.as_tensor(
                            np.stack([pools[c]["proj16"] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        rev_idx = (
                            torch.arange(min(4, k), device=device)
                            .reshape(1, -1)
                            .expand(b, -1)
                        )
                        rev_raw_f = torch.as_tensor(
                            np.stack([pools[c]["fitness"][: min(4, k)] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        rev_gate = torch.as_tensor(
                            np.stack([pools[c]["gate"][: min(4, k)] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        scale_b = torch.full(
                            (b,), 0.4208, dtype=torch.float32, device=device
                        )
                        sos_b = torch.zeros(
                            (b, k, 2), dtype=torch.float32, device=device
                        )
                        a_out = net.adapter_10tok.anchor.predict_costs(
                            ctx_feat=torch.zeros(
                                (b, 584), dtype=torch.float32, device=device
                            ),
                            prog_412=progs,
                            proj_16=proj,
                            base_prior_cost=torch.zeros(
                                (b, k), dtype=torch.float32, device=device
                            ),
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
                            eval_cache[c] = (
                                a_tot[i_b].clone(),
                                a_eff[i_b].clone(),
                                rev_raw_f[i_b].clone(),
                                rev_gate[i_b].clone(),
                            )

        def evaluate(rows):
            groups_r = {}
            result = {}
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
                            np.stack([pools[c]["program"] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        a_tot_b = torch.stack([eval_cache[c][0] for c in batch])
                        a_eff_b = torch.stack([eval_cache[c][1] for c in batch])
                        rev_raw_b = torch.stack([eval_cache[c][2] for c in batch])
                        rev_gate_b = torch.stack([eval_cache[c][3] for c in batch])
                        tok_b = torch.stack([tokens[c] for c in batch])
                        mask_b = torch.stack([masks[c] for c in batch])
                        rev_idx = (
                            torch.arange(min(4, k), device=device)
                            .reshape(1, -1)
                            .expand(b, -1)
                        )
                        scale_b = torch.full(
                            (b,), 0.4208, dtype=torch.float32, device=device
                        )
                        sos_b = torch.zeros(
                            (b, k, 2), dtype=torch.float32, device=device
                        )
                        scores = (
                            forward_adapter_with_anchor(
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
                            idx_min = int(np.argmin(cost))
                            p = pools[cid]
                            valid = bool(p["gate"][idx_min])
                            result[cid] = dict(
                                choice=idx_min,
                                utility=float(
                                    -p["fitness"][idx_min] if valid else -2.0
                                ),
                                invalid=not valid,
                            )
            rt.check_frozen(net, guard, names, frozen)
            return dict(
                values=[result[r["context_sha256"]]["utility"] for r in rows],
                choices=[result[r["context_sha256"]]["choice"] for r in rows],
                invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
                context_ids=[r["context_sha256"] for r in rows],
            )

        precompute_cohort(val)
        baseline = evaluate(val)
        sanity = parent.baseline_gate(baseline["values"], ref)
        save(out / "baseline.json", dict(**baseline, sanity=sanity))
        print("BASELINE PASS", sanity, flush=True)
        for outer in range(5):
            precompute_cohort(stat.partition(train, outer)[2])

        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}
        primary = np.full((len(stat.SEEDS), len(train), len(stat.ALL_ARMS)), np.nan)
        secondary = np.full((len(stat.SEEDS), 5, len(val), len(stat.ALL_ARMS)), np.nan)

        selections = []
        step40_drifts = {arm: [] for arm in stat.ARMS}
        selected_drifts = {arm: [] for arm in stat.ARMS}

        for outer in range(5):
            fit, select, assess = stat.partition(train, outer)
            fit_ids = {r["context_sha256"] for r in fit}
            excluded = {r["context_sha256"] for r in select + assess + val}
            assert not fit_ids & excluded
            orders = {c: np.argsort(pools[c]["fitness"]) for c in fit_ids}
            net.load_state_dict(initial, strict=True)
            r882_sel_scores = evaluate(select)
            r882_assess_scores = evaluate(assess)
            ai_idx = [train_index[r["context_sha256"]] for r in assess]

            for si, seed in enumerate(stat.SEEDS):
                primary[si, ai_idx, 0] = r882_assess_scores["values"]
                secondary[si, outer, :, 0] = baseline["values"]
                cell_tick = time.monotonic()
                plan = stat.batch_plan(fit_ids, orders, seed)
                if len(plan) != 40:
                    raise RuntimeError("40-update plan required")

                # Precompute frozen anchor terms for the 40 minibatches of this (outer, seed)
                batch_cache = []
                with torch.no_grad():
                    for step, (stream, cids, ixs) in enumerate(plan, 1):
                        if not set(cids) <= fit_ids or set(cids) & excluded:
                            raise RuntimeError("minibatch family leakage")
                        b = len(cids)
                        k = ixs[0].shape[0]
                        progs = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["program"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        proj = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["proj16"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        tok_b = torch.stack([tokens[c] for c in cids])
                        mask_b = torch.stack([masks[c] for c in cids])
                        rev_idx = (
                            torch.arange(min(4, k), device=device)
                            .reshape(1, -1)
                            .expand(b, -1)
                        )
                        rev_raw_f = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["fitness"][ix[: min(4, k)]]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        rev_gate = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["gate"][ix[: min(4, k)]]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        fitness = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["fitness"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        gate = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["gate"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.bool,
                            device=device,
                        )
                        scale_b = torch.full(
                            (b,), 0.4208, dtype=torch.float32, device=device
                        )
                        sos_b = torch.zeros(
                            (b, k, 2), dtype=torch.float32, device=device
                        )
                        a_out = net.adapter_10tok.anchor.predict_costs(
                            ctx_feat=torch.zeros(
                                (b, 584), dtype=torch.float32, device=device
                            ),
                            prog_412=progs,
                            proj_16=proj,
                            base_prior_cost=torch.zeros(
                                (b, k), dtype=torch.float32, device=device
                            ),
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
                arm_step_drifts = {}

                for arm in stat.ARMS:
                    net.load_state_dict(initial, strict=True)
                    rt.check_frozen(net, guard, names, frozen)
                    params = [dict(net.named_parameters())[n] for n in names]
                    sched = stat.lr_schedule(arm, 40)
                    optimizer = torch.optim.AdamW(
                        params, lr=sched[0], weight_decay=stat.WEIGHT_DECAY
                    )
                    arm_steps = stat.ARM_STEPS[arm]
                    snapshots = {}
                    sel_scores = []
                    losses = []
                    step_drifts = {}

                    def take_snapshot(step_num, arm_label=arm):
                        st = {
                            n: net.state_dict()[n].detach().cpu().clone() for n in names
                        }
                        snapshots[step_num] = st
                        np.savez_compressed(
                            out / f"A{outer}-{seed}-{arm_label}-step{step_num}.npz",
                            **{n: t.numpy() for n, t in st.items()},
                        )
                        cur_vec = torch.cat([st[n].reshape(-1) for n in names]).float()
                        step_drifts[str(step_num)] = float(
                            torch.linalg.vector_norm(cur_vec - init_vec)
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
                        torch.manual_seed(stream)
                        lr_now = sched[step - 1]
                        for pg in optimizer.param_groups:
                            pg["lr"] = lr_now
                        optimizer.zero_grad(set_to_none=True)
                        pred = forward_adapter_with_anchor(
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
                        torch.nn.utils.clip_grad_norm_(
                            params, 1.0, error_if_nonfinite=True
                        )
                        optimizer.step()
                        updates += 1
                        rt.check_frozen(net, guard, names, frozen)
                        losses.append(float(objective.detach()))
                        if step in arm_steps:
                            take_snapshot(step)

                    drift = step_drifts["40"]
                    if drift <= 0:
                        raise RuntimeError("no optimizer parameter drift")
                    arm_snapshots[arm] = snapshots
                    arm_sel_scores[arm] = sel_scores
                    arm_losses[arm] = losses
                    arm_drifts[arm] = drift
                    arm_step_drifts[arm] = step_drifts
                    step40_drifts[arm].append(drift)

                # Selection on inner selector fold B locked BEFORE outer fold A assessment
                decisions = {}
                for ai, arm in enumerate(stat.ARMS):
                    arm_panel = np.array(
                        [r["values"] for r in arm_sel_scores[arm]],
                        dtype=np.float64,
                    ).T
                    decisions[arm] = stat.choose(
                        select,
                        arm_panel,
                        stat.ARM_STEPS[arm],
                        stat.BOOT_SEED + 100 * outer + 10 * si + ai,
                    )
                    sel_step = decisions[arm]["selected_step"]
                    selected_drifts[arm].append(arm_step_drifts[arm][str(sel_step)])

                cellname = f"A{outer}-{seed}"
                lockpath = out / f"{cellname}-selection.json"
                save(lockpath, decisions)

                assessment = {("R882", 0): r882_assess_scores}
                val_assessment = {("R882", 0): baseline}
                for col, arm in enumerate(stat.ARMS, start=1):
                    st_pick = decisions[arm]["selected_step"]
                    if st_pick == 0:
                        a_res = r882_assess_scores
                        v_res = baseline
                    else:
                        with torch.no_grad():
                            for n, p in net.named_parameters():
                                if n in arm_snapshots[arm][st_pick]:
                                    p.copy_(arm_snapshots[arm][st_pick][n].to(p.device))
                        a_res = evaluate(assess)
                        v_res = evaluate(val)
                        assessment[(arm, st_pick)] = a_res
                        val_assessment[(arm, st_pick)] = v_res
                    primary[si, ai_idx, col] = a_res["values"]
                    secondary[si, outer, :, col] = v_res["values"]

                net.load_state_dict(initial, strict=True)
                cell = dict(
                    outer_fold=outer,
                    selection_fold=(outer + 1) % 5,
                    seed=seed,
                    updates_per_arm=40,
                    total_cell_updates=160,
                    fit_contexts=sorted(fit_ids),
                    selector_contexts=[r["context_sha256"] for r in select],
                    assessment_contexts=[r["context_sha256"] for r in assess],
                    losses=arm_losses,
                    adapter_l2_drift=arm_drifts,
                    adapter_l2_drift_by_step=arm_step_drifts,
                    snapshot_sha256={
                        arm: {
                            str(s): rt.tensor_hash(v)
                            for s, v in arm_snapshots[arm].items()
                        }
                        for arm in stat.ARMS
                    },
                    selection_lock_sha256=sha(lockpath),
                    decisions=decisions,
                    assessment={f"{k[0]}_step{k[1]}": v for k, v in assessment.items()},
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    wall_seconds=time.monotonic() - cell_tick,
                )
                save(out / f"{cellname}.json", cell)
                selections.append(decisions)
                print(
                    "CELL COMPLETE",
                    cellname,
                    "updates",
                    updates,
                    "picks",
                    {arm: decisions[arm]["selected_step"] for arm in stat.ARMS},
                    "unconstrained_picks",
                    {
                        arm: decisions[arm]["unconstrained_selected_step"]
                        for arm in stat.ARMS
                    },
                    "seconds",
                    round(cell["wall_seconds"], 2),
                    flush=True,
                )

        drift_summary = {
            arm: dict(
                step40_l2_median=float(np.median(step40_drifts[arm])),
                step40_l2_mean=float(np.mean(step40_drifts[arm])),
                step40_l2_min=float(np.min(step40_drifts[arm])),
                step40_l2_max=float(np.max(step40_drifts[arm])),
                selected_step_l2_median=float(np.median(selected_drifts[arm])),
                selected_step_l2_mean=float(np.mean(selected_drifts[arm])),
            )
            for arm in stat.ARMS
        }

        train_summary = stat.summarize(train, primary, drift_by_arm=drift_summary)
        val_summary = stat.summarize(
            val, secondary.mean(axis=1), drift_by_arm=drift_summary
        )
        metrics = dict(
            wave="H162",
            status="COMPLETE",
            verdict=stat.verdict(train_summary, val_summary, selections),
            rule_neutral=bool(
                all(
                    all(d[arm]["selected_step"] == 0 for arm in stat.ARMS)
                    for d in selections
                )
            ),
            h145_train_1219_205fam=train_summary,
            dl_val_91_19fam=val_summary,
            primary=train_summary,
            secondary_dl_val=val_summary,
            drift_summary=drift_summary,
            lr_schedules={
                arm: dict(
                    steps=list(stat.ARM_STEPS[arm]),
                    mean_lr=float(np.mean(stat.lr_schedule(arm, 40))),
                    min_lr=float(np.min(stat.lr_schedule(arm, 40))),
                    max_lr=float(np.max(stat.lr_schedule(arm, 40))),
                )
                for arm in stat.ARMS
            },
            trainable_parameters=trainable_params,
            selections=selections,
            optimizer_updates=updates,
            cells=len(selections),
            baseline=sanity,
            gpu_lock_held=False,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        if updates != 2400 or len(selections) != 15:
            raise RuntimeError("incomplete wave")
        np.savez_compressed(out / "panels.npz", primary=primary, secondary=secondary)
        verify(pins)
        save(out / "metrics.json", metrics)
    finally:
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
            ),
        )


if __name__ == "__main__":
    main()
