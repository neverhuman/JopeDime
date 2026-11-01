"""Single H161 wave: robust per-context normalized/Huber loss + family-balanced sampling on the H148 5-fold cross-fit harness."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from runtime import (
    HERE,
    INPUT,
    CACHE,
    DOPE,
    Audit,
    build,
    check_frozen,
    dropout_census,
    loss,
    per_context_loss,
    robust_loss,
    save,
    sha,
)
from science import (
    ACCUM_STEPS,
    ARMS,
    BASE_LR,
    HUBER_DELTA,
    SEEDS,
    STEPS,
    TREATMENT_ARMS,
    WEIGHT_DECAY,
    batch_plan,
    choose,
    decide,
    drift_lr,
    drift_match,
    fambal_batch_plan,
    family_matrix,
    fold_median_scale,
    folds,
    partition,
    selection_seed,
    summarize,
)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


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


def predict_uncached(net, pools, tokens, masks, cids, indexes):
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
    )["total_cost"].float()


def predict_cached(net, pools, tokens, masks, cids, indexes, anchor_cache):
    """Bitwise-exact equivalent of predict_uncached that caches frozen FlywheelGameNet anchor outputs."""
    key = (
        tuple(cids),
        tuple(
            hashlib.sha256(np.asarray(ix, dtype="<i8").tobytes()).digest()[:8]
            for ix in indexes
        ),
    )
    progs = np.stack(
        [pools[c]["program"][ix] for c, ix in zip(cids, indexes, strict=True)]
    )
    proj = np.stack(
        [pools[c]["proj16"][ix] for c, ix in zip(cids, indexes, strict=True)]
    )
    b, k = progs.shape[:2]
    prog_t = torch.as_tensor(progs)
    proj_t = torch.as_tensor(proj)
    scale = torch.full((b,), 0.4208)
    sos_prior = torch.zeros((b, k, 2))
    mm_tokens = torch.stack([tokens[c] for c in cids])
    mm_mask = torch.stack([masks[c] for c in cids])
    rev_idx = torch.arange(min(4, k)).reshape(1, -1).expand(b, -1)
    rev_raw_f = torch.as_tensor(
        np.stack(
            [
                pools[c]["fitness"][ix[: min(4, k)]]
                for c, ix in zip(cids, indexes, strict=True)
            ]
        ),
        dtype=torch.float32,
    )
    rev_gate = torch.as_tensor(
        np.stack(
            [
                pools[c]["gate"][ix[: min(4, k)]]
                for c, ix in zip(cids, indexes, strict=True)
            ]
        ),
        dtype=torch.float32,
    )
    ad = net.adapter_10tok
    if key not in anchor_cache:
        with torch.no_grad():
            anchor_out = ad.anchor.predict_costs(
                ctx_feat=torch.zeros((b, 584)),
                prog_412=prog_t,
                proj_16=proj_t,
                base_prior_cost=torch.zeros((b, k)),
                scale=scale,
                sos_prior=sos_prior,
                revealed_indices=rev_idx,
                revealed_raw_f=rev_raw_f,
                revealed_gate=rev_gate,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
            anchor_cache[key] = (
                anchor_out["total_cost"].detach().float(),
                anchor_out["eff_prior"].detach().float(),
            )
    anchor_total, anchor_eff_prior = anchor_cache[key]
    ctx_pooled, cand_q, sem24, mm_prior_adj = (
        ad.encode_multimodal_context_and_candidates(
            mm_tokens_10x32=mm_tokens,
            mm_mask_10=mm_mask,
            prog_412=prog_t,
            sos_prior=sos_prior,
        )
    )
    static_terms = dict(
        eff_prior=anchor_eff_prior + mm_prior_adj,
        mm_prior_adj=mm_prior_adj,
        sem24=sem24,
        cand_q=cand_q,
        cand_q_norm=F.normalize(cand_q, p=2, dim=-1, eps=1e-8),
        ctx_pooled=ctx_pooled,
    )
    delta = ad.compute_adapter_delta_from_static_terms(
        static_terms=static_terms,
        scale=scale,
        revealed_indices=rev_idx,
        revealed_raw_f=rev_raw_f,
        revealed_gate=rev_gate,
    )
    return (anchor_total + delta).float()


def evaluate(net, pools, tokens, masks, rows, anchor_cache=None, record_losses=False):
    groups = {}
    result = {}
    for r in rows:
        groups.setdefault(len(pools[r["context_sha256"]]["fitness"]), []).append(
            r["context_sha256"]
        )
    with torch.no_grad():
        for size, cids in sorted(groups.items()):
            for start in range(0, len(cids), 8):
                batch = cids[start : start + 8]
                ixs = [np.arange(size) for _ in batch]
                if anchor_cache is None:
                    scores_t = predict_uncached(net, pools, tokens, masks, batch, ixs)
                else:
                    scores_t = predict_cached(
                        net, pools, tokens, masks, batch, ixs, anchor_cache
                    )
                scores = scores_t.numpy()
                if not np.isfinite(scores).all():
                    raise RuntimeError("nonfinite predictions")
                if record_losses:
                    fit_t = torch.as_tensor(
                        np.stack([pools[c]["fitness"] for c in batch]),
                        dtype=torch.float32,
                    )
                    gate_t = torch.as_tensor(
                        np.stack([pools[c]["gate"] for c in batch]),
                        dtype=torch.bool,
                    )
                    ctx_losses = per_context_loss(scores_t, fit_t, gate_t).numpy()
                else:
                    ctx_losses = [None] * len(batch)
                for cid, cost, closs in zip(batch, scores, ctx_losses, strict=True):
                    i = int(np.argmin(cost))
                    p = pools[cid]
                    valid = bool(p["gate"][i])
                    result[cid] = dict(
                        choice=i,
                        utility=float(-p["fitness"][i] if valid else -2.0),
                        invalid=not valid,
                        loss=float(closs) if closs is not None else None,
                    )
    out = dict(
        values=[result[r["context_sha256"]]["utility"] for r in rows],
        choices=[result[r["context_sha256"]]["choice"] for r in rows],
        invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
        context_ids=[r["context_sha256"] for r in rows],
    )
    if record_losses:
        out["losses"] = {
            r["context_sha256"]: result[r["context_sha256"]]["loss"] for r in rows
        }
    return out


def selected():
    packet = json.loads((INPUT / "selected.json").read_text())
    train = packet["train"]
    val = packet["dev"]
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
        raise RuntimeError("family overlap")
    return train, val


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    allowed = set(pins["inputs"])
    allowed.add(str((out.parent / "launch.json").resolve()))
    if sha(INPUT / "selected.json") != pins["inputs"][str(INPUT / "selected.json")]:
        raise RuntimeError("selected mismatch")
    train, val = selected()
    allowed.update(
        str(INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = Audit(allowed, out)
    sys.addaudithook(audit.check)
    updates = 0
    started = time.monotonic()
    try:
        verify(pins)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialization forbidden")
        checkpoint = Path(pins["checkpoint"])
        net, guard, names, frozen, fused = build(checkpoint)
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
        check_frozen(net, guard, names, frozen)
        if args.smoke:
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
                    adapter_tensors=len(names),
                    trainable_parameters=sum(
                        p.numel() for p in net.parameters() if p.requires_grad
                    ),
                    census=census,
                    assets=assets,
                    packages={
                        n: importlib.metadata.version(n)
                        for n in ("numpy", "torch", "psutil")
                    },
                    python=sys.version,
                    frozen_sha256=frozen,
                    adapter_names=list(names),
                    dope_sha256=DOPE,
                ),
            )
            return

        launch = json.loads((out.parent / "launch.json").read_text())
        if not launch["merged_source_verified"] or launch["manifest_sha256"] != sha(
            HERE / "manifest.json"
        ):
            raise RuntimeError("merged launch required")
        groups = folds(train)
        if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
            raise RuntimeError("fold identity")
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
            raise RuntimeError("evaluation contract")

        pools = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for r in train + val:
            cid = r["context_sha256"]
            p = INPUT / "pools" / f"{cid}.npz"
            if sha(p) != r["pool_sha256"]:
                raise RuntimeError("pool mismatch")
            with np.load(p, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
        with np.load(CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            tok = z["tokens_10x32"]
            mask = z["mask_10"]
            if set(ids) != set(pools) or len(ids) != len(pools):
                raise RuntimeError("cache closure")
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32)
                for i, c in enumerate(ids)
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool) for i, c in enumerate(ids)
            }

        anchor_cache = {}
        # 1. Exact 91-context r882 baseline gate FIRST (uncached + cached bitwise parity proof).
        baseline = evaluate(net, pools, tokens, masks, val, anchor_cache=None)
        baseline_cached = evaluate(
            net, pools, tokens, masks, val, anchor_cache=anchor_cache
        )
        if (
            baseline["values"] != baseline_cached["values"]
            or baseline["choices"] != baseline_cached["choices"]
        ):
            raise RuntimeError("cached anchor parity mismatch on DL-Val")
        sanity = baseline_gate(baseline["values"], ref)
        check_frozen(net, guard, names, frozen)
        save(out / "baseline.json", dict(**baseline, sanity=sanity))
        print("BASELINE PASS", sanity, flush=True)

        # 2. Precompute step-0 r882 utilities and detached per-context losses per fold.
        fold_rows = [
            [r for r in train if r["family_id"] in set(groups[g])] for g in range(5)
        ]
        base_fold = {}
        base_losses = {}
        for g in range(5):
            ev = evaluate(
                net,
                pools,
                tokens,
                masks,
                fold_rows[g],
                anchor_cache=anchor_cache,
                record_losses=True,
            )
            check_frozen(net, guard, names, frozen)
            base_losses.update(ev.pop("losses"))
            base_fold[g] = ev

        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        params = dict(net.named_parameters())
        plist = [params[n] for n in names]
        sizes = [params[n].numel() for n in names]
        theta0 = torch.cat([initial[n].double().reshape(-1) for n in names])

        def load_adapter(vec):
            with torch.no_grad():
                for n, chunk in zip(names, torch.split(vec, sizes), strict=True):
                    params[n].copy_(chunk.reshape(params[n].shape).to(params[n].dtype))

        def adapter_vec():
            return torch.cat([params[n].detach().double().reshape(-1) for n in names])

        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}
        executed_arms = list(ARMS)
        cols = ["r882"] + executed_arms
        primary = np.full((3, len(train), len(cols)), np.nan)
        primary_inv = np.zeros((3, len(train), len(cols)), dtype=bool)
        secondary = np.full((3, 5, len(val), len(cols)), np.nan)
        secondary_inv = np.zeros((3, 5, len(val), len(cols)), dtype=bool)

        for outer in range(5):
            ai = [train_index[r["context_sha256"]] for r in fold_rows[outer]]
            for si in range(3):
                primary[si, ai, 0] = base_fold[outer]["values"]
                primary_inv[si, ai, 0] = base_fold[outer]["invalid"]
                secondary[si, outer, :, 0] = baseline["values"]
                secondary_inv[si, outer, :, 0] = baseline["invalid"]

        cells = []
        drifts = {a: [] for a in ARMS}
        selected_drifts = {a: [] for a in ARMS}
        nonzero = {a: 0 for a in ARMS}
        fold_scales = {}
        selector_step_gains = {a: {s: [] for s in STEPS} for a in ARMS}
        selector_best_gains = {a: [] for a in ARMS}

        def run_arm(arm, lr, use_clip, use_fambal):
            nonlocal updates
            col = cols.index(arm)
            for outer in range(5):
                fit, select, assess = partition(train, outer)
                sel_fold = (outer + 1) % 5
                fit_ids = {r["context_sha256"] for r in fit}
                excluded = {r["context_sha256"] for r in select + assess + val}
                assert not fit_ids & excluded
                scale_outer = fold_median_scale(fit_ids, base_losses)
                fold_scales[str(outer)] = scale_outer
                orders = {c: np.argsort(pools[c]["fitness"]) for c in fit_ids}
                for si, seed in enumerate(SEEDS):
                    tick = time.monotonic()
                    net.load_state_dict(initial, strict=True)
                    check_frozen(net, guard, names, frozen)
                    optimizer = torch.optim.AdamW(
                        plist, lr=lr, weight_decay=WEIGHT_DECAY
                    )
                    snapshots, selector_scores, step_drifts, losses_log, batches = (
                        {},
                        [],
                        {},
                        [],
                        [],
                    )
                    cellname = f"{arm}-A{outer}-{seed}"

                    def snapshot(step):
                        vec = adapter_vec()
                        snapshots[step] = vec.clone()
                        step_drifts[str(step)] = float(
                            torch.linalg.vector_norm(vec - theta0)
                        )
                        selector_scores.append(
                            base_fold[sel_fold]
                            if step == 0
                            else evaluate(
                                net,
                                pools,
                                tokens,
                                masks,
                                select,
                                anchor_cache=anchor_cache,
                            )
                        )
                        check_frozen(net, guard, names, frozen)

                    snapshot(0)
                    if use_fambal:
                        plan = fambal_batch_plan(fit, orders, seed)
                        if len(plan) != 40:
                            raise RuntimeError("40-update fambal plan required")
                        for step, (stream, cids, ixs, step_fams, mbs) in enumerate(
                            plan, 1
                        ):
                            if (
                                not set(cids) <= fit_ids
                                or set(cids) & excluded
                                or len(set(step_fams)) != 16
                            ):
                                raise RuntimeError("fambal minibatch family leakage")
                            torch.manual_seed(stream)
                            optimizer.zero_grad(set_to_none=True)
                            step_loss = 0.0
                            for mb_cids, mb_ixs in mbs:
                                pred = predict_cached(
                                    net,
                                    pools,
                                    tokens,
                                    masks,
                                    mb_cids,
                                    mb_ixs,
                                    anchor_cache,
                                )
                                fit_t = torch.as_tensor(
                                    np.stack(
                                        [
                                            pools[c]["fitness"][ix]
                                            for c, ix in zip(
                                                mb_cids, mb_ixs, strict=True
                                            )
                                        ]
                                    ),
                                    dtype=torch.float32,
                                )
                                gate_t = torch.as_tensor(
                                    np.stack(
                                        [
                                            pools[c]["gate"][ix]
                                            for c, ix in zip(
                                                mb_cids, mb_ixs, strict=True
                                            )
                                        ]
                                    ),
                                    dtype=torch.bool,
                                )
                                obj = (
                                    robust_loss(
                                        pred,
                                        fit_t,
                                        gate_t,
                                        scale_outer,
                                        delta=HUBER_DELTA,
                                    )
                                    if use_clip
                                    else loss(pred, fit_t, gate_t)
                                )
                                if not torch.isfinite(obj):
                                    raise RuntimeError("nonfinite objective")
                                (obj / float(ACCUM_STEPS)).backward()
                                step_loss += float(obj.detach()) / float(ACCUM_STEPS)
                            torch.nn.utils.clip_grad_norm_(
                                plist, 1.0, error_if_nonfinite=True
                            )
                            optimizer.step()
                            updates += 1
                            check_frozen(net, guard, names, frozen)
                            losses_log.append(step_loss)
                            batches.append(
                                dict(
                                    step=step,
                                    stream=stream,
                                    context_ids=cids,
                                    families=step_fams,
                                    indices_sha256=hashlib.sha256(
                                        np.asarray(ixs, dtype="<i8").tobytes()
                                    ).hexdigest(),
                                )
                            )
                            if step in STEPS:
                                snapshot(step)
                    else:
                        plan = batch_plan(fit_ids, orders, seed)
                        if len(plan) != 40:
                            raise RuntimeError("40-update plan required")
                        for step, (stream, cids, ixs) in enumerate(plan, 1):
                            if not set(cids) <= fit_ids or set(cids) & excluded:
                                raise RuntimeError("minibatch family leakage")
                            torch.manual_seed(stream)
                            optimizer.zero_grad(set_to_none=True)
                            pred = predict_cached(
                                net, pools, tokens, masks, cids, ixs, anchor_cache
                            )
                            fit_t = torch.as_tensor(
                                np.stack(
                                    [
                                        pools[c]["fitness"][ix]
                                        for c, ix in zip(cids, ixs, strict=True)
                                    ]
                                ),
                                dtype=torch.float32,
                            )
                            gate_t = torch.as_tensor(
                                np.stack(
                                    [
                                        pools[c]["gate"][ix]
                                        for c, ix in zip(cids, ixs, strict=True)
                                    ]
                                ),
                                dtype=torch.bool,
                            )
                            obj = (
                                robust_loss(
                                    pred,
                                    fit_t,
                                    gate_t,
                                    scale_outer,
                                    delta=HUBER_DELTA,
                                )
                                if use_clip
                                else loss(pred, fit_t, gate_t)
                            )
                            if not torch.isfinite(obj):
                                raise RuntimeError("nonfinite objective")
                            obj.backward()
                            torch.nn.utils.clip_grad_norm_(
                                plist, 1.0, error_if_nonfinite=True
                            )
                            optimizer.step()
                            updates += 1
                            check_frozen(net, guard, names, frozen)
                            losses_log.append(float(obj.detach()))
                            batches.append(
                                dict(
                                    step=step,
                                    stream=stream,
                                    context_ids=cids,
                                    indices_sha256=hashlib.sha256(
                                        np.asarray(ixs, dtype="<i8").tobytes()
                                    ).hexdigest(),
                                )
                            )
                            if step in STEPS:
                                snapshot(step)

                    drift = step_drifts["40"]
                    if drift <= 0:
                        raise RuntimeError("no optimizer parameter drift")
                    drifts[arm].append(drift)
                    panel = np.array([r["values"] for r in selector_scores]).T
                    decision = choose(
                        select,
                        panel,
                        selection_seed(outer, si, arm, arms=executed_arms),
                    )
                    if arm in selector_step_gains:
                        fmeans = decision["family_means"]
                        for idx_s, st in enumerate(STEPS):
                            selector_step_gains[arm][st].append(
                                float(fmeans[idx_s] - fmeans[0])
                            )
                        selector_best_gains[arm].append(float(decision["gain"]))
                    lockpath = out / f"{cellname}-selection.json"
                    save(lockpath, decision)
                    step = decision["selected_step"]
                    selected_drifts[arm].append(step_drifts[str(step)])
                    if step:
                        nonzero[arm] += 1
                        load_adapter(snapshots[step])
                        assessed = evaluate(
                            net,
                            pools,
                            tokens,
                            masks,
                            assess,
                            anchor_cache=anchor_cache,
                        )
                        validated = evaluate(
                            net,
                            pools,
                            tokens,
                            masks,
                            val,
                            anchor_cache=anchor_cache,
                        )
                        check_frozen(net, guard, names, frozen)
                    else:
                        assessed = base_fold[outer]
                        validated = baseline
                    ai = [train_index[r["context_sha256"]] for r in assess]
                    primary[si, ai, col] = assessed["values"]
                    primary_inv[si, ai, col] = assessed["invalid"]
                    secondary[si, outer, :, col] = validated["values"]
                    secondary_inv[si, outer, :, col] = validated["invalid"]
                    cell = dict(
                        arm=arm,
                        outer_fold=outer,
                        selection_fold=sel_fold,
                        seed=seed,
                        lr=lr,
                        weight_decay=WEIGHT_DECAY,
                        use_clip=use_clip,
                        use_fambal=use_fambal,
                        fold_median_scale=scale_outer,
                        updates=40,
                        fit_contexts=sorted(fit_ids),
                        losses=losses_log,
                        batches=batches,
                        adapter_l2_drift=drift,
                        step_l2_drifts=step_drifts,
                        selected_l2_drift=step_drifts[str(step)],
                        snapshot_sha256={
                            str(s): hashlib.sha256(v.numpy().tobytes()).hexdigest()
                            for s, v in snapshots.items()
                        },
                        selection_lock_sha256=sha(lockpath),
                        decision=decision,
                        selector_scores=[r["values"] for r in selector_scores],
                        assessment=assessed["values"],
                        validation=validated["values"],
                        frozen_sha256=frozen,
                        dope_sha256=DOPE,
                        wall_seconds=time.monotonic() - tick,
                    )
                    save(out / f"{cellname}.json", cell)
                    cells.append(
                        dict(
                            arm=arm,
                            outer=outer,
                            seed=seed,
                            best_step=decision["best_step"],
                            selected_step=step,
                            selector_gain=decision["gain"],
                            drift=drift,
                        )
                    )
                    print(
                        "CELL",
                        cellname,
                        "updates",
                        updates,
                        "pick",
                        step,
                        "best_step",
                        decision["best_step"],
                        "sel_gain",
                        round(decision["gain"], 6),
                        "drift",
                        round(drift, 6),
                        "seconds",
                        round(cell["wall_seconds"], 2),
                        flush=True,
                    )

        arm_configs = {
            "CTRL": (BASE_LR, False, False),
            "CLIP": (BASE_LR, True, False),
            "FAMBAL": (BASE_LR, False, True),
            "CLIP_FAMBAL": (BASE_LR, True, True),
        }
        for arm in ARMS:
            lr, u_clip, u_fambal = arm_configs[arm]
            run_arm(arm, lr, u_clip, u_fambal)

        psummary = summarize(train, primary, primary_inv, arms=executed_arms)
        prelim_decision = decide(psummary, nonzero)
        winning_treatments = [
            a
            for a in TREATMENT_ARMS
            if prelim_decision["treatment_arms"][a]["efficacy_pass"]
        ]
        drift_match_receipt = None
        winner_arm = None
        ctrl_drift_lr = None
        if winning_treatments:
            winner_arm = max(
                winning_treatments,
                key=lambda a: psummary["contrasts"][f"{a}_minus_r882"]["median_gain"],
            )
            ctrl_drift_lr = drift_lr(drifts[winner_arm], drifts["CTRL"])
            executed_arms.append("CTRL_DRIFT")
            cols.append("CTRL_DRIFT")
            drifts["CTRL_DRIFT"] = []
            selected_drifts["CTRL_DRIFT"] = []
            nonzero["CTRL_DRIFT"] = 0
            primary = np.concatenate(
                [primary, np.full((3, len(train), 1), np.nan)], axis=2
            )
            primary_inv = np.concatenate(
                [primary_inv, np.zeros((3, len(train), 1), dtype=bool)], axis=2
            )
            secondary = np.concatenate(
                [secondary, np.full((3, 5, len(val), 1), np.nan)], axis=2
            )
            secondary_inv = np.concatenate(
                [secondary_inv, np.zeros((3, 5, len(val), 1), dtype=bool)], axis=2
            )
            run_arm("CTRL_DRIFT", ctrl_drift_lr, False, False)
            drift_match_receipt = drift_match(drifts[winner_arm], drifts["CTRL_DRIFT"])
            psummary = summarize(train, primary, primary_inv, arms=executed_arms)

        val_mean_panel = secondary.mean(axis=1)
        val_inv_panel = secondary_inv.mean(axis=1) >= 0.5
        ssummary = summarize(val, val_mean_panel, val_inv_panel, arms=executed_arms)
        decision = decide(
            psummary,
            nonzero,
            drift_match_receipt=drift_match_receipt,
            winner_arm=winner_arm,
        )
        ctrl_med_drift = float(np.median(drifts["CTRL"]))
        drift_summary = {
            a: dict(
                median_step40_l2_drift=float(np.median(v)),
                min_step40_l2_drift=float(np.min(v)),
                max_step40_l2_drift=float(np.max(v)),
                ratio_vs_ctrl=float(np.median(v) / ctrl_med_drift),
                median_selected_l2_drift=float(np.median(selected_drifts[a])),
            )
            for a, v in drifts.items()
        }
        selector_diagnostics = {
            a: dict(
                median_gain_by_step={
                    str(st): float(np.median(selector_step_gains[a][st]))
                    for st in STEPS
                },
                mean_gain_by_step={
                    str(st): float(np.mean(selector_step_gains[a][st])) for st in STEPS
                },
                median_best_step_gain=float(np.median(selector_best_gains[a])),
                max_best_step_gain=float(np.max(selector_best_gains[a])),
                cells_with_positive_best_gain=int(
                    sum(g > 0.0 for g in selector_best_gains[a])
                ),
            )
            for a in ARMS
        }
        expected_cells = 75 if "CTRL_DRIFT" in executed_arms else 60
        expected_updates = expected_cells * 40
        metrics = dict(
            wave="H161",
            status="COMPLETE",
            verdict=decision["verdict"],
            decision=decision,
            primary=psummary,
            secondary_dl_val_descriptive_only=ssummary,
            nonzero_selections=nonzero,
            cells_per_arm=15,
            total_cells=len(cells),
            fold_median_scales=fold_scales,
            drift_summary=drift_summary,
            ctrl_drift_executed="CTRL_DRIFT" in executed_arms,
            ctrl_drift_lr=ctrl_drift_lr,
            drift_match=drift_match_receipt,
            selector_fold_diagnostics=selector_diagnostics,
            optimizer_updates=updates,
            baseline=sanity,
            baseline_dl_val_family_utility=float(
                family_matrix(
                    np.asarray(baseline["values"])[:, None],
                    [r["family_id"] for r in val],
                )[1].mean()
            ),
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            fold_manifest_sha256=sha(HERE / "fold_manifest.json"),
            dope_sha256=DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        if updates != expected_updates or len(cells) != expected_cells:
            raise RuntimeError("incomplete wave inventory")
        if len(json.dumps(metrics).encode()) >= 100_000:
            raise RuntimeError("metrics.json exceeds 100 KB limit")
        np.savez_compressed(
            out / "panels.npz",
            primary=primary,
            primary_inv=primary_inv,
            secondary=secondary,
            secondary_inv=secondary_inv,
        )
        verify(pins)
        save(out / "metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], "updates", updates, flush=True)
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
