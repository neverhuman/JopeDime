"""CPU-only H174 fit-only SWAG posterior predictive cross-fit."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import analysis as stat
from treatment import (
    SNAPSHOT_STEPS,
    SELECT_STEPS,
    RANK,
    SAMPLES,
    SCALES,
    average_cost_vectors,
    adapter_forward,
    gradient_norm,
    adapter_drift,
    fit_swag,
    sample_vectors,
    unflatten_state,
    validate_fit_only,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources"):
        for path, digest in pins[kind].items():
            p = REPO / path if kind == "parent_sources" else Path(path)
            if sha(p) != digest:
                raise RuntimeError("pin mismatch: " + str(p))
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("source pin mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def runtime():
    path = REPO / "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    spec = importlib.util.spec_from_file_location("h174_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohorts(rt):
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train, val = packet["train"], packet["dev"]
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
    ):
        raise RuntimeError("cohort size")
    if len({r["context_sha256"] for r in train + val}) != 1310:
        raise RuntimeError("duplicate context")
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("protected context")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("family overlap")
    return train, val


class Engine:
    def __init__(self, net, rt, names, frozen, guard, fused, rows):
        self.net, self.rt, self.names, self.frozen, self.guard = (
            net,
            rt,
            names,
            frozen,
            guard,
        )
        self.pools, self.cached = {}, {}
        self.projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            if len(ids) != 1310 or set(ids) != {r["context_sha256"] for r in rows}:
                raise RuntimeError("feature cache closure")
            tok, mask = z["tokens_10x32"], z["mask_10"]
            self.tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32)
                for i, c in enumerate(ids)
            }
            self.masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool) for i, c in enumerate(ids)
            }

    def load(self, rows):
        for row in rows:
            cid = row["context_sha256"]
            if cid in self.pools:
                continue
            path = self.rt.INPUT / "pools" / (cid + ".npz")
            if sha(path) != row["pool_sha256"]:
                raise RuntimeError("pool hash")
            with np.load(path, allow_pickle=False) as z:
                p = z["program"].astype(np.float32)
                self.pools[cid] = dict(
                    program=p,
                    proj16=(p.astype(np.float64) @ self.projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )

    def batch(self, cids, indexes):
        b, k = len(cids), len(indexes[0])

        def stack(key):
            return np.stack(
                [self.pools[c][key][ix] for c, ix in zip(cids, indexes, strict=True)]
            )

        prog = torch.as_tensor(stack("program"), dtype=torch.float32)
        proj = torch.as_tensor(stack("proj16"), dtype=torch.float32)
        fitness = torch.as_tensor(stack("fitness"), dtype=torch.float32)
        gate = torch.as_tensor(stack("gate"), dtype=torch.bool)
        kwargs = dict(
            ctx_feat=torch.zeros((b, 584)),
            prog_412=prog,
            proj_16=proj,
            base_prior_cost=torch.zeros((b, k)),
            scale=torch.full((b,), 0.4208),
            sos_prior=torch.zeros((b, k, 2)),
            revealed_indices=torch.arange(min(4, k)).reshape(1, -1).expand(b, -1),
            revealed_raw_f=fitness[:, : min(4, k)],
            revealed_gate=gate[:, : min(4, k)].float(),
        )
        return (
            kwargs,
            torch.stack([self.tokens[c] for c in cids]),
            torch.stack([self.masks[c] for c in cids]),
            fitness,
            gate,
        )

    def anchored(self, kwargs):
        with torch.no_grad():
            a = self.net.adapter_10tok.anchor.predict_costs(
                **kwargs,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
        return a["total_cost"].detach().float(), a["eff_prior"].detach().float()

    def forward(self, kwargs, tok, mask, anchor):
        return adapter_forward(
            self.net,
            kwargs["prog_412"],
            kwargs["sos_prior"],
            tok,
            mask,
            kwargs["scale"],
            kwargs["revealed_indices"],
            kwargs["revealed_raw_f"],
            kwargs["revealed_gate"],
            *anchor,
        )

    def check(self):
        self.rt.check_frozen(self.net, self.guard, self.names, self.frozen)
        if (
            self.net.dope_model.training
            or self.net.adapter_10tok.anchor.training
            or torch.cuda.is_initialized()
        ):
            raise RuntimeError("frozen mode/device violation")

    def evaluate(self, rows, native=False, prove_identity=False):
        groups, result, raw = {}, {}, {}
        for r in rows:
            cid = r["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.no_grad():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    batch = cids[start : start + 8]
                    kwargs, tok, mask, _, _ = self.batch(
                        batch, [np.arange(k) for c in batch]
                    )
                    if native:
                        scores = self.net.adapter_10tok.predict_costs(
                            **kwargs, mm_tokens_10x32=tok, mm_mask_10=mask
                        )["total_cost"].float()
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        if prove_identity and not torch.equal(
                            scores, self.forward(kwargs, tok, mask, anchor)
                        ):
                            raise RuntimeError(
                                "step0 cached forward differs from native r882 bits"
                            )
                    else:
                        missing = [c for c in batch if c not in self.cached]
                        if missing:
                            anchor = self.anchored(kwargs)
                            for i, cid in enumerate(batch):
                                self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        anchor = tuple(
                            torch.stack([self.cached[c][j] for c in batch])
                            for j in range(2)
                        )
                        scores = self.forward(kwargs, tok, mask, anchor)
                    costs = scores.numpy()
                    if not np.isfinite(costs).all():
                        raise RuntimeError("nonfinite prediction")
                    for cid, cost in zip(batch, costs, strict=True):
                        choice = int(np.argmin(cost))
                        pool = self.pools[cid]
                        result[cid] = dict(
                            choice=choice,
                            utility=float(
                                -pool["fitness"][choice]
                                if pool["gate"][choice]
                                else -2.0
                            ),
                            invalid=not bool(pool["gate"][choice]),
                        )
                        raw[cid] = cost.copy()
        self.check()
        return dict(
            context_ids=[r["context_sha256"] for r in rows],
            values=[result[r["context_sha256"]]["utility"] for r in rows],
            choices=[result[r["context_sha256"]]["choice"] for r in rows],
            invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
        ), raw


def utility_panel(engine, rows, costs):
    result = []
    for row in rows:
        cid = row["context_sha256"]
        values = np.asarray(costs[cid], dtype=np.float32)
        choice = int(np.argmin(values))
        pool = engine.pools[cid]
        result.append(
            dict(
                context=cid,
                choice=choice,
                utility=float(
                    -pool["fitness"][choice] if pool["gate"][choice] else -2.0
                ),
                invalid=not bool(pool["gate"][choice]),
            )
        )
    return dict(
        context_ids=[x["context"] for x in result],
        values=[x["utility"] for x in result],
        choices=[x["choice"] for x in result],
        invalid=[x["invalid"] for x in result],
    )


def assign_state(net, names, state):
    with torch.no_grad():
        params = dict(net.named_parameters())
        for name in names:
            params[name].copy_(state[name])


def assign_vector(net, names, template, vector):
    assign_state(net, names, unflatten_state(vector, template, names))


def average_costs(engine, rows, vectors, net, names, template, restore):
    raw_samples, utility_samples = [], []
    for vector in vectors:
        assign_vector(net, names, template, vector)
        _, raw = engine.evaluate(rows)
        raw_samples.append(raw)
        utility_samples.append(utility_panel(engine, rows, raw))
    restore(template)
    averaged = {
        row["context_sha256"]: average_cost_vectors(
            [sample[row["context_sha256"]] for sample in raw_samples]
        ).numpy()
        for row in rows
    }
    return utility_panel(engine, rows, averaged), averaged, utility_samples, raw_samples


def locked_b_panel(rows, raw, panel):
    ids = [row["context_sha256"] for row in rows]
    if set(raw) != set(ids) or panel["context_ids"] != ids:
        raise ValueError("B panel context identity mismatch")
    costs = [np.asarray(raw[cid], dtype=np.float64).tolist() for cid in ids]
    if any(
        len(values) != 24 or not np.isfinite(values).all()
        for values in map(np.asarray, costs)
    ):
        raise ValueError("B cost vector shape or finiteness")
    return {
        "context_ids": ids,
        "cost_vectors": costs,
        "choices": list(panel["choices"]),
        "utilities": list(panel["values"]),
    }


def replication_report(primary, val_panel, cells):
    groups = {}
    for si, seed in enumerate(stat.SEEDS):
        for ai, arm in enumerate(stat.ALL_ARMS):
            groups[f"seed{seed}:{arm}:OOF"] = primary[si, :, ai]
            groups[f"seed{seed}:{arm}:DLVal"] = val_panel[si, :, ai]
    selector_patterns, choice_patterns = {}, {}
    for arm in stat.ARMS:
        selector_patterns[arm] = stat.duplicate_groups(
            {
                f"seed{seed}": [
                    next(
                        c["decisions"][arm]["selected"]
                        for c in cells
                        if c["seed"] == seed and c["outer_fold"] == outer
                    )
                    for outer in range(5)
                ]
                for seed in stat.SEEDS
            }
        )
        choice_patterns[arm] = stat.duplicate_groups(
            {
                f"seed{seed}": [
                    int(
                        next(
                            c["decisions"][arm]["selected_step"]
                            if arm == "CTRL"
                            else c["decisions"][arm]["selected_candidate"]
                            for c in cells
                            if c["seed"] == seed and c["outer_fold"] == outer
                        )
                    )
                    for outer in range(5)
                ]
                for seed in stat.SEEDS
            }
        )
    return dict(
        identical_utility_panels=stat.duplicate_groups(groups),
        identical_selector_choices_by_seed=choice_patterns,
        identical_selection_patterns_by_seed=selector_patterns,
        note="Seeds are not treated as independent when their exact selected-policy or utility panels match.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    validate_launch(
        launch,
        source_head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    rt = runtime()
    train, val = cohorts(rt)
    provenance = json.loads((HERE / "provenance.json").read_text())
    ids = sorted({r["dataset_id"] for r in train + val})
    if (
        provenance["status"] != "PASS"
        or not provenance["zero_synthetic"]
        or ids != provenance["dataset_ids"]
        or provenance["real_catalog_joined"] != 276
    ):
        raise RuntimeError("real S3 origin closure")
    fold_doc = json.loads((HERE / "fold_manifest.json").read_text())
    if (
        fold_doc["fold_seed"] != stat.FOLD_SEED
        or stat.folds(train) != fold_doc["families"]
    ):
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")
    updates = 0
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        initial = {n: net.state_dict()[n].detach().cpu().float().clone() for n in names}
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            sum(p.numel() for p in params) != 29211
            or len(names) != 47
            or census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
        ):
            raise RuntimeError("adapter/mode inventory")
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        if {id(p) for g in optimizer.param_groups for p in g["params"]} != {
            id(p) for p in params
        }:
            raise RuntimeError("optimizer exclusion")
        rt.check_frozen(net, guard, names, frozen)
        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=47,
                    trainable_parameters=29211,
                    launch_exact_path_after_audit=True,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    initial_adapter_sha256=initial_hash,
                    census=census,
                ),
            )
            return
        ref = json.loads(Path(pins["reference"]).read_text())
        expected_contract = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in val
        ]
        if (
            ref["evaluation_contract"]["contexts"] != expected_contract
            or hashlib.sha256(
                json.dumps(ref["evaluation_contract"], sort_keys=True).encode()
            ).hexdigest()
            != pins["evaluation_identity"]
        ):
            raise RuntimeError("evaluation contract")
        engine = Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, baseline_costs = engine.evaluate(
            val, native=True, prove_identity=True
        )
        if (
            not np.array_equal(
                np.asarray(baseline["values"]),
                np.asarray(ref["utility"]["per_context"]),
            )
            or float(np.mean(baseline["values"])) != 1.1377941898164525
        ):
            raise RuntimeError("exact 91 r882 baseline")
        np.savez_compressed(out / "baseline_costs.npz", **baseline_costs)
        save(
            out / "baseline.json",
            dict(
                **baseline,
                status="PASS",
                matches=91,
                mean=1.1377941898164525,
                eval_forward_bit_exact=True,
            ),
        )
        print("BASELINE PASS 91/91; cached/native eval forward BIT-EXACT", flush=True)

        primary = np.full((3, 1219, 5), np.nan)
        secondary_by_fold = np.full((3, 5, 91, 5), np.nan)
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        cells, selection_records = [], []

        def restore(state):
            assign_state(net, names, state)
            engine.check()

        for outer in range(5):
            fit, select, assess = stat.partition(train, outer)
            fit_ids = {r["context_sha256"] for r in fit}
            select_ids = {r["context_sha256"] for r in select}
            assess_ids = {r["context_sha256"] for r in assess}
            val_ids = {r["context_sha256"] for r in val}
            if fit_ids & (select_ids | assess_ids | val_ids) or select_ids & assess_ids:
                raise RuntimeError("family disjointness")
            engine.load(fit)
            orders = {cid: np.argsort(engine.pools[cid]["fitness"]) for cid in fit_ids}
            for si, seed in enumerate(stat.SEEDS):
                cellname = f"A{outer}-{seed}"
                restore(initial)
                torch.manual_seed(stat.stream_seed(seed, outer, "adamw-init"))
                optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                plan, plan_digest, trajectory_ids = [], hashlib.sha256(), set()
                for stream, cids, indexes in stat.batch_plan(fit_ids, orders, seed):
                    if not set(cids) <= fit_ids:
                        raise RuntimeError("non-fit minibatch")
                    trajectory_ids.update(cids)
                    kwargs, tok, mask, fitness, gate = engine.batch(cids, indexes)
                    anchor = engine.anchored(kwargs)
                    plan.append(
                        (
                            stream,
                            cids,
                            indexes,
                            kwargs,
                            tok,
                            mask,
                            anchor,
                            fitness,
                            gate,
                        )
                    )
                    plan_digest.update(
                        json.dumps(
                            [stream, cids, [ix.tolist() for ix in indexes]]
                        ).encode()
                    )
                traces, snapshots, none_names = [], {0: initial}, set()
                for step, (
                    stream,
                    cids,
                    indexes,
                    kwargs,
                    tok,
                    mask,
                    anchor,
                    fitness,
                    gate,
                ) in enumerate(plan, 1):
                    torch.manual_seed(stream)
                    optimizer.zero_grad(set_to_none=True)
                    pred = adapter_forward(
                        net,
                        kwargs["prog_412"],
                        kwargs["sos_prior"],
                        tok,
                        mask,
                        kwargs["scale"],
                        kwargs["revealed_indices"],
                        kwargs["revealed_raw_f"],
                        kwargs["revealed_gate"],
                        *anchor,
                    )
                    objective = rt.loss(pred, fitness, gate)
                    if not torch.isfinite(objective):
                        raise RuntimeError("nonfinite objective")
                    raw_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            params, 1.0, error_if_nonfinite=True
                        )
                    )
                    clipped_norm = gradient_norm(params)
                    missing = tuple(
                        n
                        for n, p in net.named_parameters()
                        if n in initial and p.grad is None
                    )
                    none_names.add(missing)
                    optimizer.step()
                    updates += 1
                    rt.check_frozen(net, guard, names, frozen)
                    traces.append(
                        dict(
                            step=step,
                            stream=stream,
                            fit_context_ids=cids,
                            context_indices=[ix.tolist() for ix in indexes],
                            loss=float(objective.detach()),
                            raw_grad_l2=raw_norm,
                            clipped_grad_l2=clipped_norm,
                            learning_rate=optimizer.param_groups[0]["lr"],
                            beta1=optimizer.param_groups[0]["betas"][0],
                            beta2=optimizer.param_groups[0]["betas"][1],
                            none_gradient_names=list(missing),
                        )
                    )
                    if step in SNAPSHOT_STEPS:
                        snapshot = {
                            n: net.state_dict()[n].detach().cpu().float().clone()
                            for n in names
                        }
                        snapshots[step] = snapshot
                        np.savez_compressed(
                            out / f"{cellname}-fit-step{step}.npz",
                            **{n: t.numpy() for n, t in snapshot.items()},
                        )
                validate_fit_only(
                    fit_ids, trajectory_ids, select_ids, assess_ids, val_ids
                )
                if tuple(sorted(snapshots)[1:]) != SNAPSHOT_STEPS:
                    raise RuntimeError("snapshot schedule")
                if rt.tensor_hash(snapshots[40]) == initial_hash:
                    raise RuntimeError("unchanged trajectory")
                swag = fit_swag({s: snapshots[s] for s in SNAPSHOT_STEPS}, names)
                swag_seed = stat.stream_seed(seed, "H174-SWAG-SAMPLES-v1")
                vectors = sample_vectors(swag, swag_seed)
                candidate_states = {
                    "SWA": unflatten_state(swag["mean"], initial, names),
                    "BMA05": [vectors[0.5][i] for i in range(SAMPLES)],
                    "BMA10": [vectors[1.0][i] for i in range(SAMPLES)],
                }
                del plan
                restore(initial)
                engine.load(select)
                base_b, base_b_raw = engine.evaluate(select)
                raw_b_costs = {
                    "r882": locked_b_panel(select, base_b_raw, base_b),
                    "CTRL": {"0": locked_b_panel(select, base_b_raw, base_b)},
                    "SWA": {},
                    "BMA05": {},
                    "BMA10": {},
                }
                selector_panels = {
                    "CTRL": [base_b],
                    "SWA": [],
                    "BMA05": [],
                    "BMA10": [],
                }
                ctrl_snapshots = {}
                for step in SELECT_STEPS[1:]:
                    restore(snapshots[step])
                    panel, raw = engine.evaluate(select)
                    ctrl_snapshots[step] = panel
                    selector_panels["CTRL"].append(panel)
                    raw_b_costs["CTRL"][str(step)] = locked_b_panel(select, raw, panel)
                decisions = {
                    "CTRL": stat.choose(
                        select,
                        np.asarray([p["values"] for p in selector_panels["CTRL"]]).T,
                        stat.stream_seed(stat.BOOT_SEED, outer, seed, "CTRL"),
                    )
                }
                decisions["CTRL"].update(
                    choice=decisions["CTRL"]["selected_step"],
                    selected=decisions["CTRL"]["selected_step"] != 0,
                )
                bma_spread = {}
                swag_hashes = {}
                for arm in ("SWA", "BMA05", "BMA10"):
                    if arm == "SWA":
                        restore(candidate_states[arm])
                        panel, raw = engine.evaluate(select)
                        raw_b_costs[arm] = {
                            "candidate_mean": locked_b_panel(select, raw, panel)
                        }
                        swag_hashes[arm] = rt.tensor_hash(candidate_states[arm])
                        sample_panels = [panel]
                    else:
                        scale = 0.5 if arm == "BMA05" else 1.0
                        panel, raw, sample_panels, raw_samples_for_lock = average_costs(
                            engine,
                            select,
                            candidate_states[arm],
                            net,
                            names,
                            initial,
                            restore,
                        )
                        raw_b_costs[arm] = {
                            "posterior_samples": [
                                locked_b_panel(select, sample, sample_panel)
                                for sample, sample_panel in zip(
                                    raw_samples_for_lock, sample_panels, strict=True
                                )
                            ],
                            "candidate_mean": locked_b_panel(select, raw, panel),
                        }
                        swag_hashes[arm] = hashlib.sha256(
                            b"".join(v.numpy().tobytes() for v in candidate_states[arm])
                        ).hexdigest()
                        sample_means = [
                            float(np.mean(p["values"])) for p in sample_panels
                        ]
                        bma_spread[arm] = dict(
                            sample_utility_means=sample_means,
                            mean=float(np.mean(sample_means)),
                            sd=float(np.std(sample_means)),
                            min=float(np.min(sample_means)),
                            max=float(np.max(sample_means)),
                        )
                        bma_spread[arm]["scale"] = scale
                    selector_panels[arm] = [base_b, panel]
                    decisions[arm] = stat.choose_candidate(
                        select,
                        np.column_stack([base_b["values"], panel["values"]]),
                        stat.stream_seed(stat.BOOT_SEED, outer, seed, arm),
                        draws=1000,
                    )
                    decisions[arm]["choice"] = (
                        arm if decisions[arm]["selected_candidate"] else "r882"
                    )
                    decisions[arm]["selected"] = decisions[arm]["selected_candidate"]
                restore(initial)
                lock = out / f"{cellname}-selection.json"
                step40_drift = adapter_drift(initial, snapshots[40])
                save(
                    lock,
                    dict(
                        wave="H174",
                        outer_fold=outer,
                        seed=seed,
                        decisions=decisions,
                        selector_panels=selector_panels,
                        raw_b_cost_vectors=raw_b_costs,
                        bma_posterior_sample_utility_spread=bma_spread,
                        fit_context_ids=sorted(fit_ids),
                        trajectory_context_ids=sorted(trajectory_ids),
                        select_context_ids=[r["context_sha256"] for r in select],
                        assess_context_ids=[r["context_sha256"] for r in assess],
                        snapshot_steps=list(SNAPSHOT_STEPS),
                        batch_plan_sha256=plan_digest.hexdigest(),
                        swag_sample_seed=swag_seed,
                        swag_rank=RANK,
                        swag_samples=SAMPLES,
                        swag_scales=list(SCALES),
                        swag_diagnostics=swag["diagnostics"],
                        step40_drift=step40_drift,
                    ),
                )
                lock_sha = sha(lock)
                selection_records.append(decisions)

                # A and candidate Val are first opened only after the durable B decisions.
                engine.load(assess)
                base_a, _ = engine.evaluate(assess)
                primary[si, [positions[r["context_sha256"]] for r in assess], 0] = (
                    base_a["values"]
                )
                secondary_by_fold[si, outer, :, 0] = baseline["values"]
                assessment, flips = {}, {}
                for ai, arm in enumerate(stat.ARMS, 1):
                    if arm == "CTRL":
                        choice = decisions[arm]["selected_step"]
                        restore(snapshots[choice])
                        candidate_a, _ = (
                            engine.evaluate(assess) if choice else (base_a, {})
                        )
                        candidate_v, _ = (
                            engine.evaluate(val) if choice else (baseline, {})
                        )
                    elif arm == "SWA":
                        restore(candidate_states[arm])
                        candidate_a, _ = engine.evaluate(assess)
                        candidate_v, _ = engine.evaluate(val)
                    else:
                        vectors_for_arm = candidate_states[arm]
                        candidate_a, _, _ = average_costs(
                            engine,
                            assess,
                            vectors_for_arm,
                            net,
                            names,
                            initial,
                            restore,
                        )
                        candidate_v, _, _ = average_costs(
                            engine, val, vectors_for_arm, net, names, initial, restore
                        )
                    selected = decisions[arm]["selected"]
                    use_a, use_v = (
                        (candidate_a, candidate_v) if selected else (base_a, baseline)
                    )
                    primary[
                        si, [positions[r["context_sha256"]] for r in assess], ai
                    ] = use_a["values"]
                    secondary_by_fold[si, outer, :, ai] = use_v["values"]
                    flips[arm] = dict(
                        assessment=int(
                            np.sum(
                                np.asarray(candidate_a["choices"])
                                != np.asarray(base_a["choices"])
                            )
                        ),
                        val=int(
                            np.sum(
                                np.asarray(candidate_v["choices"])
                                != np.asarray(baseline["choices"])
                            )
                        ),
                    )
                    assessment[arm] = dict(
                        candidate_A=candidate_a,
                        candidate_Val=candidate_v,
                        selected_A=use_a,
                        selected_Val=use_v,
                    )
                cell = dict(
                    wave="H174",
                    outer_fold=outer,
                    seed=seed,
                    decisions=decisions,
                    selection_sha256=lock_sha,
                    selector_panels=selector_panels,
                    candidate_assessment=assessment,
                    argmin_flip_count=flips,
                    bma_posterior_sample_utility_spread=bma_spread,
                    swag_diagnostics=swag["diagnostics"],
                    swag_rank=RANK,
                    swag_samples=SAMPLES,
                    swag_snapshot_steps=list(SNAPSHOT_STEPS),
                    swag_sample_seed=swag_seed,
                    step40_drift=step40_drift,
                    adapter_sha256={
                        str(s): rt.tensor_hash(w) for s, w in snapshots.items()
                    },
                    candidate_state_sha256=swag_hashes,
                    traces=traces,
                    none_gradient_name_sets=[list(x) for x in sorted(none_names)],
                    batch_plan_sha256=plan_digest.hexdigest(),
                )
                save(out / f"{cellname}.json", cell)
                cells.append(cell)
                save(
                    out / "progress.json",
                    dict(
                        complete_cells=len(cells),
                        optimizer_updates=updates,
                        last_cell=cellname,
                    ),
                )
                print(
                    "CELL COMPLETE",
                    cellname,
                    {a: decisions[a]["choice"] for a in stat.ARMS},
                    flush=True,
                )
                restore(initial)
                del snapshots, candidate_states, vectors, swag

        if updates != 600 or len(cells) != 15:
            raise RuntimeError("incomplete H174 wave")
        val_panel = secondary_by_fold.mean(axis=1)
        np.savez_compressed(
            out / "panels.npz", primary=primary, secondary_by_fold=secondary_by_fold
        )
        metrics = dict(
            wave="H174",
            status="COMPLETE",
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            optimizer_updates=updates,
            complete_cells=15,
            baseline_mean=1.1377941898164525,
            eval_forward_bit_exact=True,
            primary=stat.summarize(train, primary),
            secondary=stat.summarize(val, val_panel),
            selections=selection_records,
            unconstrained_inner_B={
                a: dict(
                    per_cell=[c["decisions"][a].get("gain", 0.0) for c in cells],
                    median=float(
                        np.median([c["decisions"][a].get("gain", 0.0) for c in cells])
                    ),
                )
                for a in stat.ARMS
            },
            step40_drift={
                f"A{c['outer_fold']}-seed{c['seed']}": c["step40_drift"] for c in cells
            },
            argmin_flip_count={
                a: dict(
                    assessment=sum(
                        c["argmin_flip_count"][a]["assessment"] for c in cells
                    ),
                    val=sum(c["argmin_flip_count"][a]["val"] for c in cells),
                )
                for a in stat.ARMS
            },
            swag_diagnostics=[
                dict(
                    outer_fold=c["outer_fold"], seed=c["seed"], **c["swag_diagnostics"]
                )
                for c in cells
            ],
            bma_posterior_sample_utility_spread={
                a: [
                    dict(
                        outer_fold=c["outer_fold"],
                        seed=c["seed"],
                        **{
                            k: v
                            for k, v in c["bma_posterior_sample_utility_spread"][
                                a
                            ].items()
                            if k != "sample_utility_means"
                        },
                    )
                    for c in cells
                ]
                for a in ("BMA05", "BMA10")
            },
            pseudoreplication=replication_report(primary, val_panel, cells),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            trainable_parameters=29211,
            initial_adapter_sha256=initial_hash,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
            audit=audit.receipt(),
        )
        save(out / "metrics.json", metrics)
        print("COMPLETE", flush=True)
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(
                status="INCOMPLETE",
                optimizer_updates=updates,
                error=repr(exc),
                audit=audit.receipt(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
