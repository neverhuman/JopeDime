"""One monitored H182 CPU attempt: nested 5/10/20 fresh-adapter ensembles."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import analysis as stat
from treatment import adapter_drift, adapter_forward, gradient_norm, validate_launch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    with path.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources"):
        for path, digest in pins[kind].items():
            p = REPO / path if kind == "parent_sources" else Path(path)
            if sha(p) != digest:
                raise RuntimeError(f"{kind} pin mismatch: {p}")
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("source pin mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def runtime():
    path = REPO / "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    spec = importlib.util.spec_from_file_location("h176_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohorts(rt, provenance):
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train, val = packet["train"], packet["dev"]
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
        or len({r["family_id"] for r in val}) != 19
        or len({r["context_sha256"] for r in train + val}) != 1310
    ):
        raise RuntimeError("cohort size or duplicate context")
    for rows, split in ((train, "DL-Train"), (val, "DL-Val")):
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("protected context or wrong split")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("TRAIN/DL-Val family overlap")
    ids = sorted({r["dataset_id"] for r in train + val})
    if ids != provenance["dataset_ids"] or provenance["real_catalog_joined"] != 276:
        raise RuntimeError("catalog provenance closure")
    if provenance["status"] != "PASS" or not provenance["zero_synthetic"]:
        raise RuntimeError("real-S3 source provenance")
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
            cids = list(map(str, z["cids"]))
            if len(cids) != 1310 or set(cids) != {r["context_sha256"] for r in rows}:
                raise RuntimeError("feature cache closure")
            tokens, masks = z["tokens_10x32"], z["mask_10"]
            self.tokens = {
                c: torch.as_tensor(tokens[i], dtype=torch.float32)
                for i, c in enumerate(cids)
            }
            self.masks = {
                c: torch.as_tensor(masks[i], dtype=torch.bool)
                for i, c in enumerate(cids)
            }

    def load(self, rows):
        for row in rows:
            cid = row["context_sha256"]
            if cid in self.pools:
                continue
            path = self.rt.INPUT / "pools" / (cid + ".npz")
            if sha(path) != row["pool_sha256"]:
                raise RuntimeError("TRAIN/DL-Val pool hash")
            with np.load(path, allow_pickle=False) as z:
                prog = z["program"].astype(np.float32)
                self.pools[cid] = dict(
                    program=prog,
                    proj16=(prog.astype(np.float64) @ self.projection).astype(
                        np.float32
                    ),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )

    def batch(self, cids, indexes):
        def stack(key):
            return np.stack(
                [self.pools[c][key][ix] for c, ix in zip(cids, indexes, strict=True)]
            )

        b, k = len(cids), len(indexes[0])
        fitness = torch.as_tensor(stack("fitness"), dtype=torch.float32)
        gate = torch.as_tensor(stack("gate"), dtype=torch.bool)
        kwargs = dict(
            ctx_feat=torch.zeros((b, 584)),
            prog_412=torch.as_tensor(stack("program"), dtype=torch.float32),
            proj_16=torch.as_tensor(stack("proj16"), dtype=torch.float32),
            base_prior_cost=torch.zeros((b, k)),
            scale=torch.full((b,), 0.4208),
            sos_prior=torch.zeros((b, k, 2)),
            revealed_indices=torch.arange(4).reshape(1, -1).expand(b, -1),
            revealed_raw_f=fitness[:, :4],
            revealed_gate=gate[:, :4].float(),
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
            out = self.net.adapter_10tok.anchor.predict_costs(
                **kwargs,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
        return out["total_cost"].detach().float(), out["eff_prior"].detach().float()

    def forward(self, kwargs, tokens, masks, anchor):
        return adapter_forward(
            self.net,
            kwargs["prog_412"],
            kwargs["sos_prior"],
            tokens,
            masks,
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
        groups, results, raw = {}, {}, {}
        for row in rows:
            cid = row["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.no_grad():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    batch = cids[start : start + 8]
                    kwargs, tokens, masks, _, _ = self.batch(
                        batch, [np.arange(k) for _ in batch]
                    )
                    if native:
                        scores = self.net.adapter_10tok.predict_costs(
                            **kwargs, mm_tokens_10x32=tokens, mm_mask_10=masks
                        )["total_cost"].float()
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        if prove_identity and not torch.equal(
                            scores, self.forward(kwargs, tokens, masks, anchor)
                        ):
                            raise RuntimeError("native/cached r882 costs differ")
                    else:
                        if any(cid not in self.cached for cid in batch):
                            anchor = self.anchored(kwargs)
                            for i, cid in enumerate(batch):
                                self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        anchor = tuple(
                            torch.stack([self.cached[c][j] for c in batch])
                            for j in range(2)
                        )
                        scores = self.forward(kwargs, tokens, masks, anchor)
                    costs = scores.numpy()
                    if not np.isfinite(costs).all():
                        raise RuntimeError("nonfinite prediction")
                    for cid, cost in zip(batch, costs, strict=True):
                        pool = self.pools[cid]
                        choice = int(np.argmin(cost))
                        results[cid] = float(
                            -pool["fitness"][choice] if pool["gate"][choice] else -2.0
                        )
                        raw[cid] = cost.copy()
        self.check()
        return {r["context_sha256"]: results[r["context_sha256"]] for r in rows}, raw

    def oracle(self, rows):
        out = {}
        for row in rows:
            pool = self.pools[row["context_sha256"]]
            utility = stat.context_utility(pool["fitness"], pool["gate"])
            out[row["context_sha256"]] = float(np.max(utility))
        return out


def adapter_state(net, names):
    return {name: net.state_dict()[name].detach().cpu().clone() for name in names}


def utility_from_scores(engine, rows, score_rows):
    result = {}
    for row in rows:
        cid = row["context_sha256"]
        pool = engine.pools[cid]
        choice = int(np.argmin(score_rows[cid]))
        result[cid] = float(-pool["fitness"][choice] if pool["gate"][choice] else -2.0)
    return result


def restore(net, names, state):
    with torch.no_grad():
        params = dict(net.named_parameters())
        for name in names:
            params[name].copy_(state[name])


def random_adapter_state(net, names, seed):
    torch.manual_seed(seed)
    old = net.adapter_10tok
    fresh = (
        type(old)(
            frozen_anchor=old.anchor,
            prog_mean=old.prog_mean.detach().cpu().numpy(),
            prog_std=old.prog_std.detach().cpu().numpy(),
            d_token=old.d_token,
            n_modalities=old.n_modalities,
            modality_dropout_p=old.modality_dropout_p,
        )
        .cpu()
        .eval()
    )
    params = dict(fresh.named_parameters())
    state = {}
    for name in names:
        suffix = name.removeprefix("adapter_10tok.")
        if suffix not in params or not params[suffix].requires_grad:
            raise RuntimeError("fresh adapter trainable inventory mismatch")
        state[name] = params[suffix].detach().cpu().clone()
    if len(state) != 47 or sum(t.numel() for t in state.values()) != 29211:
        raise RuntimeError("fresh adapter must have 47 tensors/29,211 parameters")
    del fresh
    return state


def run_cell(
    net,
    rt,
    engine,
    names,
    frozen,
    guard,
    state,
    fit_rows,
    seed,
    updates,
    output,
    tag,
    plan=None,
    optimizer=None,
    start_step=1,
    excluded_context_ids=(),
):
    if optimizer is None:
        if state is None:
            raise ValueError("initial state required for a new cell")
        restore(net, names, state)
        params = [p for p in net.parameters() if p.requires_grad]
        if len(params) != 47 or sum(p.numel() for p in params) != 29211:
            raise RuntimeError("optimizer parameter inventory")
        optimizer = torch.optim.AdamW(
            params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
        )
    else:
        params = [p for p in net.parameters() if p.requires_grad]
        if state is not None:
            raise ValueError("continuation must retain in-memory optimizer and weights")
    if plan is None:
        cids = [r["context_sha256"] for r in fit_rows]
        orders = {cid: np.argsort(engine.pools[cid]["fitness"]) for cid in cids}
        plan = stat.batch_plan(cids, orders, seed, updates=max(40, updates))
    if len(plan) < updates:
        raise RuntimeError("batch plan shorter than common update budget")
    stat.validate_fit_plan(
        plan[:updates],
        [row["context_sha256"] for row in fit_rows],
        excluded_context_ids,
        (),
    )
    traces, times = [], []
    for step in range(start_step, updates + 1):
        stream, cids, indexes = plan[step - 1]
        if not set(cids) <= {r["context_sha256"] for r in fit_rows}:
            raise RuntimeError("FIT-only batch leak")
        start = time.monotonic()
        kwargs, tokens, masks, fitness, gate = engine.batch(cids, indexes)
        anchor = engine.anchored(kwargs)
        torch.manual_seed(stream)
        optimizer.zero_grad(set_to_none=True)
        prediction = engine.forward(kwargs, tokens, masks, anchor)
        objective = rt.loss(prediction, fitness, gate)
        if not torch.isfinite(objective):
            raise RuntimeError("nonfinite objective")
        objective.backward()
        raw_norm = float(
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        )
        clipped = gradient_norm(params)
        none = [n for n, p in net.named_parameters() if n in names and p.grad is None]
        optimizer.step()
        elapsed = time.monotonic() - start
        times.append(elapsed)
        rt.check_frozen(net, guard, names, frozen)
        engine.check()
        traces.append(
            dict(
                step=step,
                stream=stream,
                fit_context_ids=cids,
                loss=float(objective.detach()),
                raw_grad_l2=raw_norm,
                clipped_grad_l2=clipped,
                none_gradient_names=none,
                learning_rate=1e-4,
                elapsed_seconds=elapsed,
            )
        )
    return dict(
        state=adapter_state(net, names),
        traces=traces,
        elapsed_update_seconds=times,
        optimizer=optimizer,
        batch_plan=plan,
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
        raise RuntimeError("CPU xbabe2 nice19 only")
    output = args.output.resolve()
    output.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    expected_trajectories = 5 * len(stat.SEEDS) * stat.MEMBER_COUNT
    expected_updates = expected_trajectories * 40
    if (
        pins.get("max_trajectories") != expected_trajectories
        or pins.get("max_total_updates") != expected_updates
        or pins.get("max_updates_per_member") != 40
        or "fractions" in pins
    ):
        raise RuntimeError(
            "H182 manifest execution bounds do not match the twenty-member design"
        )
    verify(pins)
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    rt = runtime()
    provenance = json.loads((HERE / "provenance.json").read_text())
    if pins["dope_composite_sha256"] != rt.DOPE:
        raise RuntimeError("DOPE composite pin mismatch")
    train, val = cohorts(rt, provenance)
    fold_record = json.loads((HERE / "fold_manifest.json").read_text())
    folds = fold_record["families"]
    if (
        fold_record.get("seed") != stat.FOLD_SEED
        or folds
        != stat.family_folds([r["family_id"] for r in train], seed=stat.FOLD_SEED)
        or len(folds) != 5
    ):
        raise RuntimeError("family fold manifest identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")
    started, updates = time.monotonic(), 0
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        if len(names) != 47 or sum(p.numel() for p in params) != 29211:
            raise RuntimeError("trainable adapter inventory")
        initial = adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
        ):
            raise RuntimeError("dropout/eval mode mismatch")
        rt.check_frozen(net, guard, names, frozen)
        if args.smoke:
            save(
                output / "smoke.json",
                dict(
                    status="PASS",
                    scored_contexts=0,
                    optimizer_updates=0,
                    adapter_tensors=47,
                    trainable_parameters=29211,
                    initial_adapter_sha256=initial_hash,
                    dope_sha256=rt.DOPE,
                    frozen_sha256=frozen,
                ),
            )
            return
        reference = json.loads(Path(pins["reference"]).read_text())
        engine = Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, _ = engine.evaluate(val, native=True, prove_identity=True)
        baseline_values = np.asarray([baseline[r["context_sha256"]] for r in val])
        if not np.array_equal(
            baseline_values, np.asarray(reference["utility"]["per_context"])
        ):
            raise RuntimeError("exact native 91 DL-Val r882 baseline")
        if float(np.mean(baseline_values)) != 1.1377941898164525:
            raise RuntimeError("exact native 91 DL-Val mean")
        save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                families=19,
                mean=1.1377941898164525,
                bit_exact=True,
                initial_adapter_sha256=initial_hash,
            ),
        )
        print("BASELINE PASS: 91/91 native r882 exact", flush=True)
        if rt.tensor_hash(adapter_state(net, names)) != initial_hash:
            raise RuntimeError("r882 state drift before treatment")
        contexts_by_family = {r["context_sha256"]: r["family_id"] for r in train}
        policy_results = {seed: {arm: {} for arm in stat.ARMS} for seed in stat.SEEDS}
        member_scores = {
            seed: {member: {} for member in range(stat.MEMBER_COUNT)}
            for seed in stat.SEEDS
        }
        fold_count = None
        common_updates = None
        pilot_record = None
        cell_records = []

        def score_r882_a(rows):
            engine.load(rows)
            active = adapter_state(net, names)
            restore(net, names, initial)
            values, _ = engine.evaluate(rows, native=True)
            restore(net, names, active)
            for outer_seed in stat.SEEDS:
                policy_results[outer_seed]["R882"].update(values)

        for outer in range(5):
            fit_fams, b_fams, a_fams = stat.partition(
                {r["family_id"] for r in train}, outer, folds
            )
            fit_rows_full = [r for r in train if r["family_id"] in fit_fams]
            b_rows = [r for r in train if r["family_id"] in b_fams]
            a_rows = [r for r in train if r["family_id"] in a_fams]
            if not fit_fams.isdisjoint(b_fams | a_fams) or b_fams & a_fams:
                raise RuntimeError("family partition leakage")
            # The timing pilot opens FIT pools only; A pools are opened after its
            # fold/update plan has been locked by the timing-only rule.
            engine.load(fit_rows_full)
            if outer >= (fold_count or 5):
                break
            # The timing plan must be locked before any TRAIN outcome is scored.
            if fold_count is not None:
                score_r882_a(a_rows)
            for seed in stat.SEEDS:
                fit_context_ids = [r["context_sha256"] for r in fit_rows_full]
                a_context_ids = [r["context_sha256"] for r in a_rows]
                b_context_ids = [r["context_sha256"] for r in b_rows]
                orders = {
                    cid: np.argsort(engine.pools[cid]["fitness"])
                    for cid in fit_context_ids
                }
                member_raw, fresh_final_hashes = [], []
                # Every member has a distinct constructor seed and independent batch stream.
                member_trajectory = [
                    (f"MEMBER{i}", s)
                    for i, s in enumerate(stat.fresh_member_seeds(seed), 1)
                ]
                fresh_initial_states = {
                    member_seed: random_adapter_state(net, names, member_seed)
                    for _, member_seed in member_trajectory
                }
                stat.validate_member_init_hashes(
                    rt.tensor_hash(fresh_initial_states[member_seed])
                    for _, member_seed in member_trajectory
                )
                for label, init_seed in member_trajectory:
                    start_state = fresh_initial_states[init_seed]
                    batch_seed = stat.stream_seed(
                        seed, outer, label, "h182-h156-batches-v1"
                    )
                    batch_plan = stat.batch_plan(
                        fit_context_ids, orders, batch_seed, updates=40
                    )
                    stat.validate_fit_plan(
                        batch_plan, fit_context_ids, a_context_ids, b_context_ids
                    )
                    tag = f"A{outer}-seed{seed}-{label}"
                    if fold_count is None:
                        if (outer, seed, label) != (0, stat.SEEDS[0], "MEMBER1"):
                            raise RuntimeError("timing pilot order changed")
                        pilot = run_cell(
                            net,
                            rt,
                            engine,
                            names,
                            frozen,
                            guard,
                            start_state,
                            fit_rows_full,
                            batch_seed,
                            4,
                            output,
                            tag,
                            plan=batch_plan,
                            excluded_context_ids=a_context_ids + b_context_ids,
                        )
                        updates += 4
                        seconds = float(np.median(pilot["elapsed_update_seconds"][1:]))
                        execution = stat.execution_plan(seconds)
                        fold_count, common_updates = (
                            execution["outer_folds"],
                            execution["updates_per_member"],
                        )
                        pilot_record = dict(
                            elapsed_update_seconds=pilot["elapsed_update_seconds"],
                            seconds_per_update=seconds,
                            **execution,
                            rule="timing only; four updates count toward MEMBER1; five folds retained unless 12,000-update projection exceeds four hours; then three folds; no A outcomes or training metrics influence plan",
                        )
                        save(output / "pilot_budget.json", pilot_record)
                        if not execution["within_cap"]:
                            raise RuntimeError(
                                "INCOMPLETE: three-fold 40-update/member projection still exceeds four hours"
                            )
                        score_r882_a(a_rows)
                        if outer >= fold_count:
                            break
                        if common_updates > 4:
                            tail = run_cell(
                                net,
                                rt,
                                engine,
                                names,
                                frozen,
                                guard,
                                None,
                                fit_rows_full,
                                batch_seed,
                                common_updates,
                                output,
                                tag,
                                plan=batch_plan,
                                optimizer=pilot["optimizer"],
                                start_step=5,
                                excluded_context_ids=a_context_ids + b_context_ids,
                            )
                            updates += common_updates - 4
                            cell = dict(
                                state=tail["state"],
                                traces=pilot["traces"] + tail["traces"],
                                elapsed_update_seconds=pilot["elapsed_update_seconds"]
                                + tail["elapsed_update_seconds"],
                                batch_plan=batch_plan,
                            )
                        else:
                            cell = pilot
                    else:
                        cell = run_cell(
                            net,
                            rt,
                            engine,
                            names,
                            frozen,
                            guard,
                            start_state,
                            fit_rows_full,
                            batch_seed,
                            common_updates,
                            output,
                            tag,
                            plan=batch_plan,
                            excluded_context_ids=a_context_ids + b_context_ids,
                        )
                        updates += common_updates
                    final_hash = rt.tensor_hash(cell["state"])
                    drift = adapter_drift(start_state, cell["state"])
                    if final_hash == rt.tensor_hash(start_state) or drift == 0:
                        raise RuntimeError("unchanged adapter trajectory")
                    fresh_final_hashes.append(final_hash)
                    restore(net, names, cell["state"])
                    _, raw = engine.evaluate(a_rows)
                    member_index = int(label.removeprefix("MEMBER")) - 1
                    member_raw.append(raw)
                    member_context_ids = [row["context_sha256"] for row in a_rows]
                    member_family_ids = [row["family_id"] for row in a_rows]
                    member_score_rows = [
                        np.asarray(raw[cid], dtype=np.float32)
                        for cid in member_context_ids
                    ]
                    member_candidate_counts = np.asarray(
                        [len(values) for values in member_score_rows], dtype=np.int16
                    )
                    member_max_candidates = int(member_candidate_counts.max())
                    member_score_matrix = np.full(
                        (len(member_context_ids), member_max_candidates),
                        np.nan,
                        dtype=np.float32,
                    )
                    member_choice_vector = np.empty(
                        len(member_context_ids), dtype=np.int16
                    )
                    for row_index, (cid, values) in enumerate(
                        zip(member_context_ids, member_score_rows, strict=True)
                    ):
                        member_score_matrix[row_index, : len(values)] = values
                        member_choice_vector[row_index] = int(np.argmin(values))
                        row = a_rows[row_index]
                        if row["context_sha256"] != cid:
                            raise RuntimeError("member score context order changed")
                        member_scores[seed][member_index][cid] = values
                    np.savez_compressed(
                        output / f"member-a-score-{tag}.npz",
                        context_ids=np.asarray(member_context_ids),
                        family_ids=np.asarray(member_family_ids),
                        candidate_counts=member_candidate_counts,
                        seed_id=np.asarray(seed, dtype=np.int64),
                        member_index=np.asarray(member_index, dtype=np.int16),
                        member_seed=np.asarray(init_seed, dtype=np.int64),
                        scores=member_score_matrix,
                        choices=member_choice_vector,
                        score_dtype=np.asarray("float32"),
                    )
                    cell_records.append(
                        dict(
                            outer_fold=outer,
                            seed=seed,
                            trajectory=label,
                            init_seed=init_seed,
                            fit_families=sorted(fit_fams),
                            fit_context_count=len(fit_rows_full),
                            updates=common_updates,
                            initial_adapter_sha256=rt.tensor_hash(start_state),
                            final_adapter_sha256=final_hash,
                            drift_l2=drift,
                            a_context_count=len(a_rows),
                            a_family_ids=sorted(a_fams),
                            batch_plan_sha256=hashlib.sha256(
                                json.dumps(
                                    [
                                        [s, c, [ix.tolist() for ix in indexes]]
                                        for s, c, indexes in batch_plan[:common_updates]
                                    ],
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest(),
                            trace=cell["traces"],
                        )
                    )
                    save(output / ("cell-" + tag + ".json"), cell_records[-1])
                    restore(net, names, initial)
                    engine.check()
                if outer >= fold_count:
                    break
                if len(member_raw) != stat.MEMBER_COUNT:
                    raise RuntimeError("twenty fresh member panels required")
                for k, arm in zip(stat.MEMBER_COUNTS, stat.FRESH_ARMS, strict=True):
                    averaged = {
                        row["context_sha256"]: np.mean(
                            [member_raw[m][row["context_sha256"]] for m in range(k)],
                            axis=0,
                        )
                        for row in a_rows
                    }
                    policy_results[seed][arm].update(
                        utility_from_scores(engine, a_rows, averaged)
                    )
                if len(set(fresh_final_hashes)) != stat.MEMBER_COUNT:
                    raise RuntimeError(
                        "fresh member final states unexpectedly identical"
                    )
                if outer >= fold_count:
                    break
            if fold_count is not None and outer + 1 >= fold_count:
                break
        if (
            fold_count is None
            or common_updates is None
            or len(cell_records) != fold_count * len(stat.SEEDS) * stat.MEMBER_COUNT
        ):
            raise RuntimeError("incomplete trajectory grid")
        assess_families = sorted(
            {
                contexts_by_family[cid]
                for seed in stat.SEEDS
                for arm in stat.ARMS
                for cid in policy_results[seed][arm]
            }
        )
        family_panels = np.zeros(
            (3, len(assess_families), len(stat.ARMS)), dtype=np.float64
        )
        assess_context_ids = sorted(
            {
                cid
                for seed in stat.SEEDS
                for arm in stat.ARMS
                for cid in policy_results[seed][arm]
            }
        )
        context_panels = np.zeros(
            (3, len(assess_context_ids), len(stat.ARMS)), dtype=np.float64
        )
        context_family_ids = [contexts_by_family[cid] for cid in assess_context_ids]
        for si, seed in enumerate(stat.SEEDS):
            for pi, cid in enumerate(assess_context_ids):
                context_panels[si, pi, 0] = policy_results[seed]["R882"][cid]
                for ai, arm in enumerate(stat.ARMS[1:], 1):
                    if cid not in policy_results[seed][arm]:
                        raise RuntimeError("A-context policy panel incomplete")
                    context_panels[si, pi, ai] = policy_results[seed][arm][cid]
            for ci, family in enumerate(assess_families):
                ids = [cid for cid, fid in contexts_by_family.items() if fid == family]
                base_values = [policy_results[seed]["R882"][cid] for cid in ids]
                family_panels[si, ci, 0] = float(np.mean(base_values))
                for ai, arm in enumerate(stat.ARMS[1:], 1):
                    vals = [
                        policy_results[seed][arm][cid]
                        for cid in ids
                        if cid in policy_results[seed][arm]
                    ]
                    if not vals:
                        raise RuntimeError("A-family policy panel incomplete")
                    family_panels[si, ci, ai] = float(np.mean(vals))
        summary = stat.verdict(
            stat.ensemble_summary(
                assess_families,
                family_panels,
                context_panel=context_panels,
                context_families=context_family_ids,
            )
        )
        summary["minus2_context_count_by_seed_arm"] = {
            str(seed): {
                arm: int(np.count_nonzero(context_panels[si, :, ai] == -2.0))
                for ai, arm in enumerate(stat.ARMS)
            }
            for si, seed in enumerate(stat.SEEDS)
        }
        score_context_ids = sorted(member_scores[stat.SEEDS[0]][0])
        expected_contexts = set(score_context_ids)
        if set(score_context_ids) != set(assess_context_ids):
            raise RuntimeError(
                "member score A-context closure differs from outcome panel"
            )
        for seed in stat.SEEDS:
            for member_index in range(stat.MEMBER_COUNT):
                if set(member_scores[seed][member_index]) != expected_contexts:
                    raise RuntimeError("member A-score panel incomplete")
        candidate_counts = np.asarray(
            [len(engine.pools[cid]["fitness"]) for cid in score_context_ids],
            dtype=np.int16,
        )
        max_candidates = int(candidate_counts.max())
        score_panel = np.full(
            (
                len(stat.SEEDS),
                stat.MEMBER_COUNT,
                len(score_context_ids),
                max_candidates,
            ),
            np.nan,
            dtype=np.float32,
        )
        choice_panel = np.full(
            (len(stat.SEEDS), stat.MEMBER_COUNT, len(score_context_ids)),
            -1,
            dtype=np.int16,
        )
        family_panel_ids = [contexts_by_family[cid] for cid in score_context_ids]
        context_positions = {cid: i for i, cid in enumerate(score_context_ids)}
        for si, seed in enumerate(stat.SEEDS):
            for member_index in range(stat.MEMBER_COUNT):
                for cid, values in member_scores[seed][member_index].items():
                    ci = context_positions[cid]
                    if (
                        len(values) != candidate_counts[ci]
                        or not np.isfinite(values).all()
                    ):
                        raise RuntimeError(
                            "member candidate score shape/value mismatch"
                        )
                    score_panel[si, member_index, ci, : len(values)] = values
                    choice_panel[si, member_index, ci] = int(np.argmin(values))
        np.savez_compressed(
            output / "member_a_scores.npz",
            context_ids=np.asarray(score_context_ids),
            family_ids=np.asarray(family_panel_ids),
            candidate_counts=candidate_counts,
            seed_ids=np.asarray(stat.SEEDS, dtype=np.int64),
            member_seeds=np.asarray(
                [stat.fresh_member_seeds(seed) for seed in stat.SEEDS], dtype=np.int64
            ),
            member_indices=np.arange(stat.MEMBER_COUNT, dtype=np.int16),
            scores=score_panel,
            choices=choice_panel,
            score_dtype=np.asarray("float32"),
        )
        disagreement_by_k = {}
        for k in stat.MEMBER_COUNTS:
            rates = []
            denominator = k * (k - 1) // 2
            for si in range(len(stat.SEEDS)):
                for ci in range(len(score_context_ids)):
                    choices = choice_panel[si, :k, ci]
                    rates.append(
                        sum(
                            choices[i] != choices[j]
                            for i in range(k)
                            for j in range(i + 1, k)
                        )
                        / denominator
                    )
            disagreement_by_k[str(k)] = float(np.mean(rates))
        gain_vs_k = []
        for k, arm in zip(stat.MEMBER_COUNTS, stat.FRESH_ARMS, strict=True):
            gain = summary["gain_vs_r882"][arm]
            gain_vs_k.append(
                {
                    "k": k,
                    "arm": arm,
                    "median_gain": gain["median_over_seeds"],
                    "ci95": gain["family_bootstrap_ci95"],
                    "per_seed": gain["per_seed"],
                    "raw_p": gain["one_sided_centered_bootstrap_p"],
                    "holm_m3_adjusted_p": gain["holm_m3_adjusted_p"],
                    "median_elo": summary["elo_vs_r882"][arm]["median_over_seeds"],
                    "elo_ci95": summary["elo_vs_r882"][arm]["family_bootstrap_ci95"],
                    "member_choice_disagreement": disagreement_by_k[str(k)],
                }
            )
        summary.update(
            outer_folds=fold_count,
            updates_per_member=common_updates,
            trajectories=len(cell_records),
            optimizer_updates=updates,
            assessed_family_count=len(assess_families),
            gain_vs_k=gain_vs_k,
            member_pairwise_choice_disagreement_by_k=disagreement_by_k,
            member_pairwise_choice_disagreement_rate=disagreement_by_k["20"],
            member_a_score_persistence={
                "file": "member_a_scores.npz",
                "shape": list(score_panel.shape),
                "dtype": "float32",
                "choices_persisted": True,
            },
            diagnostic_only=True,
            serving_promotion=False,
            dope_sha256=rt.DOPE,
        )
        save(output / "metrics.json", summary)
        np.savez_compressed(
            output / "learning_curve_family_panels.npz",
            family_ids=np.array(assess_families),
            panel=family_panels,
            seed_ids=np.array(stat.SEEDS),
            arm_names=np.array(stat.ARMS),
        )
        np.savez_compressed(
            output / "learning_curve_context_panels.npz",
            context_ids=np.array(assess_context_ids),
            family_ids=np.array(context_family_ids),
            panel=context_panels,
            seed_ids=np.array(stat.SEEDS),
            arm_names=np.array(stat.ARMS),
        )
        save(
            output / "audit.json",
            dict(
                status="PASS",
                opened=audit.receipt()["opened"],
                refused=audit.receipt()["refused"],
                protected_reads=audit.receipt()["protected_reads"],
                optimizer_updates=updates,
                forward_scored_contexts=1310
                + sum(c["a_context_count"] for c in cell_records),
            ),
        )
        print("COMPLETE " + summary["verdict"], flush=True)
    except BaseException as exc:
        save(
            output / "failure.json",
            dict(
                status="INCOMPLETE",
                optimizer_updates=updates,
                error=repr(exc),
                audit=audit.receipt(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
