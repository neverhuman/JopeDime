"""One monitored H188 CPU attempt: kernel-INPUT ablation on the r882 adapter, 0 updates; pooled-B select -> lock -> pooled A."""

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
from treatment import (
    adapter_components,
    adapter_forward,
    anchor_features,
    assert_anchor_decomposition,
    assert_component_sum,
    gradient_norm,
    huber_clip_residuals,
    kernel_term,
    rank_residuals,
    static_terms,
    validate_launch,
    variant_scores,
    zero_adapter_scores,
)

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
    spec = importlib.util.spec_from_file_location("h188_runtime", path)
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

    def evaluate_anchor(self, rows):
        """ZERO_ADAPTER scores (anchor_total) with the live-decomposition proof on every batch."""
        groups, results, raw, proved = {}, {}, {}, 0
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
                    native = self.net.adapter_10tok.predict_costs(
                        **kwargs, mm_tokens_10x32=tokens, mm_mask_10=masks
                    )
                    anchor_total, _ = self.anchored(kwargs)
                    assert_anchor_decomposition(native, anchor_total)
                    proved += len(batch)
                    costs = zero_adapter_scores(anchor_total).numpy()
                    if not np.isfinite(costs).all():
                        raise RuntimeError("nonfinite anchor prediction")
                    for cid, cost in zip(batch, costs, strict=True):
                        pool = self.pools[cid]
                        choice = int(np.argmin(cost))
                        results[cid] = float(
                            -pool["fitness"][choice] if pool["gate"][choice] else -2.0
                        )
                        raw[cid] = cost.copy()
        self.check()
        return (
            {r["context_sha256"]: results[r["context_sha256"]] for r in rows},
            raw,
            proved,
        )

    def evaluate_kernel_inputs(self, rows):
        """R882 and the three kernel-input arms from one pass on the r882 adapter. Per batch, bit-exact gates: the
        PRIOR+KERNEL+ATTN sum equals the live delta; the replayed kernel with identity inputs equals the live KERNEL
        term; R882 equals the H176/H180 adapter_forward score."""
        arms = ("R882", "KIN_ANCHOR", "KIN_RANKRES", "KIN_CLIPRES")
        results = {a: {} for a in arms}
        raw = {a: {} for a in arms}
        groups, proved = {}, 0
        ad = self.net.adapter_10tok
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
                    if any(cid not in self.cached for cid in batch):
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                    anchor_total, anchor_prior = (
                        torch.stack([self.cached[c][j] for c in batch])
                        for j in range(2)
                    )
                    static = static_terms(
                        self.net,
                        kwargs["prog_412"],
                        kwargs["sos_prior"],
                        tokens,
                        masks,
                        anchor_prior,
                    )
                    rev = (
                        kwargs["scale"],
                        kwargs["revealed_indices"],
                        kwargs["revealed_raw_f"],
                        kwargs["revealed_gate"],
                    )
                    live_delta = ad.compute_adapter_delta_from_static_terms(
                        static_terms=static,
                        scale=rev[0],
                        revealed_indices=rev[1],
                        revealed_raw_f=rev[2],
                        revealed_gate=rev[3],
                    )
                    parts = adapter_components(ad, static, *rev)
                    assert_component_sum(parts, live_delta)
                    identity = kernel_term(
                        ad,
                        static["sem24"],
                        static["cand_q_norm"],
                        static["eff_prior"],
                        *rev,
                    )
                    if not torch.equal(identity, parts["KERNEL"]):
                        raise RuntimeError("replayed kernel != live KERNEL term")
                    full = (anchor_total + live_delta).float()
                    if not torch.equal(
                        full,
                        self.forward(
                            kwargs, tokens, masks, (anchor_total, anchor_prior)
                        ),
                    ):
                        raise RuntimeError("R882 != H176/H180 adapter_forward")
                    a_sem, a_q = anchor_features(
                        ad.anchor,
                        kwargs["ctx_feat"],
                        kwargs["prog_412"],
                        kwargs["proj_16"],
                        kwargs["sos_prior"],
                    )
                    variants = {
                        "KIN_ANCHOR": kernel_term(
                            ad, a_sem, a_q, static["eff_prior"], *rev
                        ),
                        "KIN_RANKRES": kernel_term(
                            ad,
                            static["sem24"],
                            static["cand_q_norm"],
                            static["eff_prior"],
                            *rev,
                            residual_transform=rank_residuals,
                        ),
                        "KIN_CLIPRES": kernel_term(
                            ad,
                            static["sem24"],
                            static["cand_q_norm"],
                            static["eff_prior"],
                            *rev,
                            residual_transform=huber_clip_residuals,
                        ),
                    }
                    proved += len(batch)
                    scores = {
                        "R882": full,
                        **{
                            a: variant_scores(anchor_total, parts, v)
                            for a, v in variants.items()
                        },
                    }
                    for arm, sc in scores.items():
                        costs = sc.numpy()
                        if not np.isfinite(costs).all():
                            raise RuntimeError("nonfinite prediction")
                        for cid, cost in zip(batch, costs, strict=True):
                            pool = self.pools[cid]
                            choice = int(np.argmin(cost))
                            results[arm][cid] = float(
                                -pool["fitness"][choice]
                                if pool["gate"][choice]
                                else -2.0
                            )
                            raw[arm][cid] = cost.copy()
        self.check()
        return results, raw, proved

    def oracle(self, rows):
        out = {}
        for row in rows:
            pool = self.pools[row["context_sha256"]]
            utility = stat.context_utility(pool["fitness"], pool["gate"])
            out[row["context_sha256"]] = float(np.max(utility))
        return out


def adapter_state(net, names):
    return {name: net.state_dict()[name].detach().cpu().clone() for name in names}


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
    folds = json.loads((HERE / "fold_manifest.json").read_text())["families_by_seed"]
    if sorted(folds) != sorted(str(s) for s in stat.SEEDS):
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
        engine.load(train)
        train_r882, train_raw = engine.evaluate(train, native=True)
        train_oracle = engine.oracle(train)
        val_r882 = [baseline[r["context_sha256"]] for r in val]
        val_oracle = list(engine.oracle(val).values())
        gap = stat.zero_update_gap(
            train,
            [train_r882[r["context_sha256"]] for r in train],
            [train_oracle[r["context_sha256"]] for r in train],
            val,
            val_r882,
            val_oracle,
        )
        save(output / "zero_update_gap.json", gap)
        np.savez_compressed(
            output / "zero_update_family_panels.npz",
            train_family_ids=np.array(list(gap["family_values"]["train"])),
            train_family_utility=np.array(list(gap["family_values"]["train"].values())),
            val_family_ids=np.array(list(gap["family_values"]["dl_val"])),
            val_family_utility=np.array(list(gap["family_values"]["dl_val"].values())),
        )
        np.savez_compressed(
            output / "zero_update_context_panels.npz",
            train_context_ids=np.array([r["context_sha256"] for r in train]),
            train_family_ids=np.array([r["family_id"] for r in train]),
            train_r882_utility=np.array(
                [train_r882[r["context_sha256"]] for r in train]
            ),
            train_oracle_utility=np.array(
                [train_oracle[r["context_sha256"]] for r in train]
            ),
            val_context_ids=np.array([r["context_sha256"] for r in val]),
            val_family_ids=np.array([r["family_id"] for r in val]),
            val_r882_utility=np.array(val_r882),
            val_oracle_utility=np.array(val_oracle),
        )
        if rt.tensor_hash(adapter_state(net, names)) != initial_hash:
            raise RuntimeError("r882 state drift before treatment")
        _, _, anchor_proved = engine.evaluate_anchor(
            val
        )  # anchor-only decomposition proof on all 91 DL-Val
        arms = stat.ARMS
        index = {r["context_sha256"]: i for i, r in enumerate(train)}
        panel = {
            stage: np.full((3, len(train), len(arms)), np.nan) for stage in ("B", "A")
        }
        choices = {
            stage: {s: {a: {} for a in arms} for s in stat.SEEDS}
            for stage in ("B", "A")
        }
        proved, cells = 0, []

        def score(stage, rows, si, seed, outer):
            nonlocal proved
            results, raws, n = engine.evaluate_kernel_inputs(rows)
            proved += n
            for cid, value in results["R882"].items():
                if value != train_r882[cid]:
                    raise RuntimeError("live-path R882 != native r882 TRAIN utility")
            for j, arm in enumerate(arms):
                for cid, value in results[arm].items():
                    panel[stage][si, index[cid], j] = value
                    choices[stage][seed][arm][cid] = int(np.argmin(raws[arm][cid]))
            cells.append(dict(stage=stage, seed=seed, outer=outer, contexts=len(rows)))
            print(
                "CELL", stage, f"A{outer}-seed{seed}", "contexts", len(rows), flush=True
            )

        fold_sets = {}
        for si, seed in enumerate(stat.SEEDS):
            groups = stat.seed_folds([r["family_id"] for r in train], seed)
            if groups != folds[str(seed)]:
                raise RuntimeError("fold manifest identity")
            fold_sets[seed] = groups
            for outer in range(5):
                _, b_fams, _ = stat.partition(
                    {r["family_id"] for r in train}, outer, groups
                )
                score(
                    "B", [r for r in train if r["family_id"] in b_fams], si, seed, outer
                )
        if not np.isfinite(panel["B"]).all():
            raise RuntimeError("pooled B panel incomplete")
        decisions = stat.pooled_b_select_m3(
            train,
            panel["B"][:, :, 0],
            {a: panel["B"][:, :, arms.index(a)] for a in stat.CANDIDATES},
        )
        lock = output / "pooled_b_selection_lock.json"
        save(lock, dict(decisions=decisions, written_before_any_A_scoring=True))
        lock_sha = sha(lock)
        print(
            "LOCK",
            {a: d["admitted"] for a, d in decisions.items()},
            lock_sha[:12],
            flush=True,
        )
        for si, seed in enumerate(stat.SEEDS):
            for outer in range(5):
                _, _, a_fams = stat.partition(
                    {r["family_id"] for r in train}, outer, fold_sets[seed]
                )
                score(
                    "A", [r for r in train if r["family_id"] in a_fams], si, seed, outer
                )
        if not np.isfinite(panel["A"]).all():
            raise RuntimeError("pooled A panel incomplete")
        minus2 = {
            a: [int(np.count_nonzero(panel["A"][s, :, j] == -2.0)) for s in range(3)]
            for j, a in enumerate(arms)
        }
        agreement = {
            a: [
                stat.choice_agreement(choices["A"][seed][a], choices["A"][seed]["R882"])
                for seed in stat.SEEDS
            ]
            for a in arms
        }
        summary_a = stat.summarize_a(train, panel["A"], minus2, agreement)
        decision = stat.verdict(decisions, summary_a)
        val_results, _, val_proved = engine.evaluate_kernel_inputs(val)
        proved += val_proved
        dl_val = {}
        for a in arms:
            v = np.array([val_results[a][r["context_sha256"]] for r in val])
            _, fm = stat.family_values(val, v)
            dl_val[a] = dict(
                context_mean=float(v.mean()),
                equal_family_mean=float(fm.mean()),
                minus2=int(np.count_nonzero(v == -2.0)),
            )
        base_v = np.array([val_results["R882"][r["context_sha256"]] for r in val])
        if not np.array_equal(base_v, np.asarray(reference["utility"]["per_context"])):
            raise RuntimeError("live-path R882 DL-Val != pinned reference")
        for a in stat.CANDIDATES:
            dl_val[a]["equal_family_gain_vs_r882"] = (
                dl_val[a]["equal_family_mean"] - dl_val["R882"]["equal_family_mean"]
            )
        if rt.tensor_hash(adapter_state(net, names)) != initial_hash or updates != 0:
            raise RuntimeError("r882 adapter changed or updates occurred")
        summary = dict(
            verdict=decision["verdict"],
            decision=decision,
            pooled_b=decisions,
            pooled_b_lock_sha256=lock_sha,
            pooled_a=summary_a,
            dl_val_descriptive=dl_val,
            pooled_b_equals_pooled_a=bool(np.array_equal(panel["B"], panel["A"])),
            identical_seed_panels_b=bool(
                all(np.array_equal(panel["B"][0], panel["B"][s]) for s in range(3))
            ),
            optimizer_updates=updates,
            cells=len(cells),
            component_and_kernel_identity_proved_contexts=proved,
            zero_adapter_decomposition_proved_contexts=anchor_proved,
            h178_receipt_sha256=pins["h178_receipt_sha256"],
            zero_update_gap={k: v for k, v in gap.items() if k != "family_values"},
            diagnostic_only=True,
            serving_promotion=False,
            dope_sha256=rt.DOPE,
        )
        raw_bytes = json.dumps(summary, allow_nan=False).encode()
        if len(raw_bytes) >= 100_000:
            raise RuntimeError("metrics would exceed 100 KB")
        save(output / "metrics.json", summary)
        np.savez_compressed(
            output / "panels.npz",
            pooled_b=panel["B"],
            pooled_a=panel["A"],
            context_ids=np.array([r["context_sha256"] for r in train]),
            family_ids=np.array([r["family_id"] for r in train]),
            arm_names=np.array(arms),
            seed_ids=np.array(stat.SEEDS),
        )
        save(
            output / "choices.json",
            {st: {str(s): choices[st][s] for s in stat.SEEDS} for st in ("B", "A")},
        )
        save(
            output / "audit.json",
            dict(
                status="PASS",
                opened=audit.receipt()["opened"],
                refused=audit.receipt()["refused"],
                protected_reads=audit.receipt()["protected_reads"],
                optimizer_updates=updates,
            ),
        )
        print(
            "COMPLETE",
            summary["verdict"],
            "pooled_b_equals_pooled_a",
            summary["pooled_b_equals_pooled_a"],
            flush=True,
        )
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
