"""One CPU-only H243 wave: zero-update module-wise soups of r882 and r782 (hw_t0435). The 47 adapter tensors are
partitioned into the 11 module groups frozen in groups.json; a candidate replaces group g of r882 by
(1 - beta) r882_g + beta r782_g, beta in {0.5, 1.0}. Arms: MODSWAP_SELECT (best single (g, beta) per (seed, outer) cell
on DL-Train FIT folds, strict), the exploratory GREEDY_SOUP (Wortsman 2022 greedy forward add of groups on FIT, strict)
and the report-only PLACEBO (the selected group moved by a seeded random direction of the same L2). Merged H242/H241
zero-update harness, 0 optimizer updates; all selections are locked before any DL-Val91 read beyond the Native91
regression; every policy is then scored once on DL-Val91."""

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
import soup as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAUSE = None


def cooperate():
    """H239 A1: block between evaluation batches/pool loads while the supervisor requests a pause."""
    if PAUSE is not None:
        while PAUSE.exists():
            time.sleep(0.25)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return sha(path)


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


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runtime():
    module = load_module(
        "h243_runtime", "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    )
    # Merged H232 A2: release the early `research` namespace binding so the pinned H50 fused model can import
    # JopeDimeMicro's `research.levels`.
    for name in [
        n for n in sys.modules if n == "research" or n.startswith("research.")
    ]:
        del sys.modules[name]
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
    """Merged H237 engine (cached frozen anchor; native/cached identity proven on r882) with H239 A1 cooperation."""

    def __init__(self, net, rt, names, frozen, guard, fused, rows, forward):
        self.net, self.rt, self.names, self.frozen, self.guard = (
            net,
            rt,
            names,
            frozen,
            guard,
        )
        self.adapter_forward = forward
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
            cooperate()
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

    def batch(self, cids, k):
        b = len(cids)

        def stack(key):
            return np.stack([self.pools[c][key][np.arange(k)] for c in cids])

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
        return self.adapter_forward(
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
        groups, result = {}, {}
        for r in rows:
            cid = r["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.no_grad():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    cooperate()
                    batch = cids[start : start + 8]
                    kwargs, tok, mask = self.batch(batch, k)
                    missing = [c for c in batch if c not in self.cached]
                    if native or missing:
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                    if native:
                        scores = self.net.adapter_10tok.predict_costs(
                            **kwargs, mm_tokens_10x32=tok, mm_mask_10=mask
                        )["total_cost"].float()
                        if prove_identity and not torch.equal(
                            scores, self.forward(kwargs, tok, mask, anchor)
                        ):
                            raise RuntimeError(
                                "cached forward differs from native r882 bits"
                            )
                    else:
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
        self.check()
        return dict(
            context_ids=[r["context_sha256"] for r in rows],
            values=[result[r["context_sha256"]]["utility"] for r in rows],
            choices=[result[r["context_sha256"]]["choice"] for r in rows],
            invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
        )


def recover(prior, launch):
    """H239 A1-style resume: reuse only hash-verified completed evaluation files of the same science."""
    if prior is None:
        return {}
    prior = prior.resolve()
    if (
        str(prior) != launch.get("resume_from")
        or not 1 <= launch.get("resume_number", 0) <= 2
    ):
        raise RuntimeError("resume source binding")
    progress = json.loads((prior / "progress.json").read_text())
    if progress.get("manifest_sha256") != launch["manifest_sha256"]:
        raise RuntimeError("resume changed science")
    reused = {}
    for name, digest in progress["files"].items():
        if sha(prior / name) != digest:
            raise RuntimeError("resume file hash: " + name)
        reused[name] = json.loads((prior / name).read_text())
    return reused


def build_policies(initial, previous, groups, rt):
    """r882, the 22 single-group soups and their 22 matched placebos, keyed by policy name (r882 first)."""
    policies, l2 = {"r882": initial}, {}
    for g in groups:
        for beta in sp.BETAS:
            name = sp.label(g, beta)
            policies[name] = sp.soup(initial, previous, {g: beta}, groups)
            l2[name] = sp.group_l2(initial, previous, groups[g], beta)
    for g in groups:
        for beta in sp.BETAS:
            name = sp.label(g, beta)
            policies["PL:" + name] = sp.placebo(initial, previous, groups, g, beta)
            moved = sp.adapter_l2(initial, policies["PL:" + name])
            if abs(moved - l2[name]) > 1e-5 * max(1.0, l2[name]):
                raise RuntimeError("placebo L2 mismatch: " + name)
    return policies, l2


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()
    PAUSE = args.output.with_suffix(".pause.json")
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
    out.mkdir()  # Exclusive; never reuse completed/partial study outputs.
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    sp.validate_launch(
        launch,
        source_head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    reused = recover(args.resume_from, launch)
    groups = json.loads((HERE / "groups.json").read_text())["groups"]
    treatment = load_module(
        "h243_h237_treatment", "research/flywheel_selfplay/h237_adabelief/treatment.py"
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
    if {str(s): stat.folds(train, s) for s in stat.SEEDS} != json.loads(
        (HERE / "fold_manifest.json").read_text()
    )["families_by_seed"]:
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")
    files, started = {}, time.monotonic()

    def record(name, obj):
        files[name] = save(out / name, obj)
        tmp = out / "progress.json.tmp"
        tmp.write_text(
            json.dumps(
                dict(manifest_sha256=launch["manifest_sha256"], files=files), indent=2
            )
            + "\n"
        )
        os.replace(tmp, out / "progress.json")

    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        dope_before = guard.snapshot_dope_weights(net).composite_sha256
        params = [p for p in net.parameters() if p.requires_grad]
        initial = {n: net.state_dict()[n].detach().clone() for n in names}
        initial_hash = rt.tensor_hash(initial)
        if (
            sum(p.numel() for p in params) != 29211
            or len(names) != 47
            or dope_before != rt.DOPE
        ):
            raise RuntimeError("adapter/DOPE inventory")
        sp.check_groups(groups, names)
        anchor_state = torch.load(
            pins["checkpoint"], map_location="cpu", weights_only=True
        )
        ck = torch.load(pins["previous"], map_location="cpu", weights_only=True)
        if ck.get("round") != 782 or ck.get("dope_sha256") != rt.DOPE:
            raise RuntimeError("wrong r782 checkpoint")
        if any(
            k not in names
            and ck["state_dict"][k].numpy().tobytes()
            != anchor_state["state_dict"][k].numpy().tobytes()
            for k in anchor_state["state_dict"]
        ) or set(ck["state_dict"]) != set(anchor_state["state_dict"]):
            raise RuntimeError("r782 non-adapter tensors differ from r882")
        previous = {n: ck["state_dict"][n].float().clone() for n in names}
        del anchor_state, ck
        policies, group_l2 = build_policies(initial, previous, groups, rt)
        identity = {
            p: dict(
                adapter_sha256=rt.tensor_hash(w), l2_to_r882=sp.adapter_l2(initial, w)
            )
            for p, w in policies.items()
        }
        if len({v["adapter_sha256"] for v in identity.values()}) != len(identity):
            raise RuntimeError("duplicate candidate adapters")
        rt.check_frozen(net, guard, names, frozen)
        if args.smoke:
            record(
                "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=47,
                    trainable_parameters=29211,
                    frozen_sha256=frozen,
                    dope_sha256=dope_before,
                    initial_adapter_sha256=initial_hash,
                    r782_adapter_sha256=rt.tensor_hash(previous),
                    groups={g: len(v) for g, v in groups.items()},
                    group_l2=group_l2,
                    policies=identity,
                ),
            )
            return

        def install(weights):
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if n in weights:
                        p.copy_(weights[n])
            engine.check()

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
        engine = Engine(
            net, rt, names, frozen, guard, fused, train + val, treatment.adapter_forward
        )
        engine.load(
            val
        )  # No TRAIN outcomes opened until the exact Native91 baseline passes.
        baseline = engine.evaluate(val, native=True, prove_identity=True)
        if (
            not np.array_equal(
                np.asarray(baseline["values"]),
                np.asarray(ref["utility"]["per_context"]),
            )
            or float(np.mean(baseline["values"])) != stat.R882_MEAN
        ):
            raise RuntimeError("exact 91 r882 baseline")
        record(
            "baseline.json",
            dict(
                **baseline,
                status="PASS",
                matches=91,
                mean=stat.R882_MEAN,
                eval_forward_bit_exact=True,
            ),
        )
        print("BASELINE PASS 91/91; cached/native eval forward BIT-EXACT", flush=True)
        engine.load(train)

        def scored(policy, split, rows_):
            name = f"eval-{policy}-{split}.json"
            prior = reused.get(name)
            if (
                prior is not None
                and prior["adapter_sha256"] == identity[policy]["adapter_sha256"]
                and prior["context_ids"] == [r["context_sha256"] for r in rows_]
            ):
                record(name, dict(prior, recovered=True))
                return prior
            install(policies[policy])
            result = dict(
                engine.evaluate(rows_),
                policy=policy,
                split=split,
                adapter_sha256=identity[policy]["adapter_sha256"],
            )
            if (
                rt.tensor_hash({n: net.state_dict()[n] for n in names})
                != identity[policy]["adapter_sha256"]
            ):
                raise RuntimeError("installed adapter identity")
            record(name, result)
            print("SCORED", policy, split, flush=True)
            return result

        # Phase 1: DL-Train only. Every decision is made and locked before any further DL-Val read.
        train_scores = {p: scored(p, "train", train) for p in list(policies)}
        tv = {p: np.asarray(s["values"]) for p, s in train_scores.items()}

        def train_values(recipe):
            name = stat.recipe_label(recipe)
            if name not in policies:
                policies[name] = sp.soup(initial, previous, recipe, groups)
                identity[name] = dict(
                    adapter_sha256=rt.tensor_hash(policies[name]),
                    l2_to_r882=sp.adapter_l2(initial, policies[name]),
                    recipe=recipe,
                )
            if name not in tv:
                train_scores[name] = scored(name, "train", train)
                tv[name] = np.asarray(train_scores[name]["values"])
            return tv[name]

        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        single_labels = ["r882"] + [sp.label(g, b) for g in groups for b in sp.BETAS]
        cells = []
        for seed in stat.SEEDS:
            for outer in range(5):
                fit, assess = stat.partition(train, outer, seed)
                if {r["family_id"] for r in fit} & {
                    r["family_id"] for r in assess + val
                }:
                    raise RuntimeError("family leakage")
                fi = [positions[r["context_sha256"]] for r in fit]
                modswap = stat.choose(
                    fit,
                    np.stack([tv[c][fi] for c in single_labels], axis=1),
                    stat.stream_seed(stat.BOOT_SEED, outer, seed, "MODSWAP_SELECT"),
                    single_labels,
                )
                soup_decision = stat.greedy(
                    fit,
                    tv["r882"][fi],
                    {(g, b): tv[sp.label(g, b)][fi] for g in groups for b in sp.BETAS},
                    sp.BETAS,
                    lambda recipe: train_values(recipe)[fi],
                    stat.stream_seed(stat.BOOT_SEED, outer, seed, "GREEDY_SOUP"),
                )
                decisions = dict(
                    CONTROL=dict(selected="r882", best="r882", accepted=False),
                    MODSWAP_SELECT=modswap,
                    GREEDY_SOUP=dict(
                        soup_decision,
                        selected=stat.recipe_label(soup_decision["selected"]),
                        best=stat.recipe_label(soup_decision["recipe"]),
                        selected_recipe=soup_decision["selected"],
                    ),
                    PLACEBO=dict(
                        selected="r882"
                        if modswap["selected"] == "r882"
                        else "PL:" + modswap["selected"],
                        best="r882"
                        if modswap["best"] == "r882"
                        else "PL:" + modswap["best"],
                        matched_to="MODSWAP_SELECT",
                    ),
                )
                cells.append(
                    dict(
                        seed=seed,
                        outer_fold=outer,
                        fit_ids=sorted(r["context_sha256"] for r in fit),
                        assess_ids=[r["context_sha256"] for r in assess],
                        decisions=decisions,
                    )
                )
                print(
                    "CELL",
                    seed,
                    outer,
                    modswap["selected"],
                    decisions["GREEDY_SOUP"]["selected"],
                    flush=True,
                )
        record("selection.json", dict(cells=cells, policies=identity))
        selection_sha = files["selection.json"]
        print("SELECTIONS LOCKED", selection_sha, flush=True)

        # Phase 2: every policy scored once on DL-Val91 (r882 cached forward must equal the native baseline).
        val_scores = {p: scored(p, "val", val) for p in policies}
        if val_scores["r882"]["values"] != baseline["values"]:
            raise RuntimeError("CONTROL differs from native r882 on DL-Val91")
        vv = {p: np.asarray(s["values"]) for p, s in val_scores.items()}

        npol = len(stat.ALL_ARMS)
        primary = np.full((3, 1219, npol), np.nan)
        unconstrained = np.full_like(primary, np.nan)
        secondary = np.full((3, 5, 91, npol), np.nan)
        unconstrained_val = np.full_like(secondary, np.nan)
        for cell in cells:
            si, outer = stat.SEEDS.index(cell["seed"]), cell["outer_fold"]
            ai_rows = [positions[c] for c in cell["assess_ids"]]
            primary[si, ai_rows, 0] = unconstrained[si, ai_rows, 0] = tv["r882"][
                ai_rows
            ]
            secondary[si, outer, :, 0] = unconstrained_val[si, outer, :, 0] = vv["r882"]
            for ai, arm in enumerate(stat.ARMS, 1):
                dec = cell["decisions"][arm]
                primary[si, ai_rows, ai] = tv[dec["selected"]][ai_rows]
                unconstrained[si, ai_rows, ai] = tv[dec["best"]][ai_rows]
                secondary[si, outer, :, ai] = vv[dec["selected"]]
                unconstrained_val[si, outer, :, ai] = vv[dec["best"]]
        dope_after = guard.snapshot_dope_weights(net).composite_sha256
        if dope_after != rt.DOPE:
            raise RuntimeError("DOPE drift")
        np.savez_compressed(
            out / "panels.npz",
            primary=primary,
            secondary=secondary,
            unconstrained=unconstrained,
            unconstrained_val=unconstrained_val,
        )
        fixed = [p for p in policies if p != "r882"]
        cliff = stat.CLIFF_CONTEXT
        metrics = dict(
            wave="H243",
            status="COMPLETE",
            arms=stat.ALL_ARMS,
            decision_arms=stat.DECISION_ARMS,
            seeds=stat.SEEDS,
            betas=sp.BETAS,
            groups={g: v for g, v in groups.items()},
            optimizer_updates=0,
            complete_cells=len(cells),
            baseline_mean=stat.R882_MEAN,
            eval_forward_bit_exact=True,
            primary_dlval=stat.summarize(val, secondary.mean(axis=1)),
            secondary_oof=stat.summarize(train, primary),
            descriptive_unconstrained_val=stat.summarize(
                val, unconstrained_val.mean(axis=1)
            ),
            descriptive_unconstrained_oof=stat.summarize(train, unconstrained),
            fixed_dlval=stat.fixed_readout(
                val, np.stack([vv[c] for c in ["r882", *fixed]], axis=1), fixed
            ),
            fixed_train_descriptive=stat.fixed_readout(
                train, np.stack([tv[c] for c in ["r882", *fixed]], axis=1), fixed
            ),
            cliff_context=dict(
                index=cliff,
                context_sha256=val[cliff]["context_sha256"],
                family_id=val[cliff]["family_id"],
                r882=float(vv["r882"][cliff]),
                per_policy={p: float(vv[p][cliff]) for p in policies},
                per_arm_by_seed=secondary[:, :, cliff, :].mean(axis=1).tolist(),
                note="Reported separately and never excluded from any statistic (hw_t0435).",
            ),
            group_l2=group_l2,
            invalid_picks={
                p: dict(
                    val=int(sum(val_scores[p]["invalid"])),
                    train=int(sum(train_scores[p]["invalid"])),
                )
                for p in policies
            },
            selected_counts={
                arm: {
                    p: sum(c["decisions"][arm]["selected"] == p for c in cells)
                    for p in sorted({c["decisions"][arm]["selected"] for c in cells})
                }
                for arm in stat.ARMS[1:]
            },
            best_counts={
                arm: {
                    p: sum(c["decisions"][arm]["best"] == p for c in cells)
                    for p in sorted({c["decisions"][arm]["best"] for c in cells})
                }
                for arm in stat.ARMS[1:]
            },
            selections=[c["decisions"] for c in cells],
            selection_sha256=selection_sha,
            policies=identity,
            dope_before=dope_before,
            dope_after=dope_after,
            frozen_sha256=frozen,
            trainable_parameters=29211,
            initial_adapter_sha256=initial_hash,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            resume_number=launch.get("resume_number", 0),
            recovered_files=sorted(n for n in reused if n.startswith("eval-")),
            serving_promotion=False,
            audit=audit.receipt(),
            elapsed_seconds=time.monotonic() - started,
        )
        metrics["verdict"] = (
            "ADVANCE"
            if any(
                metrics["primary_dlval"]["contrasts"][a + "_minus_r882"]["advance"]
                for a in stat.DECISION_ARMS
            )
            else "NO_ADVANCE"
        )
        record("metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as exc:
        (out / "failure.json").write_text(
            json.dumps(
                dict(status="INCOMPLETE", error=repr(exc), audit=audit.receipt()),
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
