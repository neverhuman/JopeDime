"""One CPU-only H249 wave: zero-init low-rank residual side branch on a frozen r882 adapter (SIDE4 rank 4 primary,
SIDE16 rank 16 secondary) vs exact H170 CTRL full-adapter continuation (CONTROL). Reuses merged H247/H232/H170
machinery with one fresh fold seed 249140, seeds 249101-249103, 15 cells/arm, 40 updates/cell (1,800 total updates),
and H239 A1 cooperative pauses on xbabe2 cores 52-55 (hw_t1035)."""

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
    TRAINABLE_PARAMS,
    adapter_drift,
    adapter_forward,
    arm_loss,
    gradient_norm,
    settings,
    setup_arm,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GIT_REPO = Path("/home/ubuntu/JopeDime")
STUDY_REL = "research/flywheel_selfplay/h249_side_branch"
PAUSE = None
OUT = None


def cooperate():
    """H239 A1: block between optimizer steps, evaluation batches and pool loads while a pause is requested."""
    if PAUSE is not None:
        while PAUSE.exists():
            time.sleep(0.25)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def save_cell(name, obj):
    """Completed-cell checkpoint (H239 A1): fsynced JSON whose sha256 is recorded in progress.json."""
    save(OUT / name, obj)
    return sha(OUT / name)


def recover(prior, launch):
    """H239 A1 resume: reuse only hash-verified completed cells of the same manifest (at most two resumes)."""
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
    cells = {}
    for name, digest in progress["files"].items():
        if sha(prior / name) != digest:
            raise RuntimeError("resume cell hash: " + name)
        cells[name.removesuffix(".json")] = json.loads((prior / name).read_text())
    return cells


def verify(pins, launch=None, smoke=False):
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
    if not smoke and launch is not None:
        git_root = GIT_REPO if (GIT_REPO / ".git").exists() else REPO
        commit = launch["source_commit"]
        subprocess.check_call(
            ["git", "-C", str(git_root), "cat-file", "-e", f"{commit}^{{commit}}"]
        )
        for path in pins["source_sha256"]:
            blob = subprocess.check_output(
                ["git", "-C", str(git_root), "show", f"{commit}:{STUDY_REL}/{path}"]
            )
            if blob != (HERE / path).read_bytes():
                raise RuntimeError(f"git blob mismatch at {commit}: {path}")


def runtime():
    path = REPO / "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    spec = importlib.util.spec_from_file_location("h170_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in [
        n for n in list(sys.modules) if n == "research" or n.startswith("research.")
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

    def forward(self, kwargs, tok, mask, anchor, branch=None):
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
            branch=branch,
        )

    def check(self):
        self.rt.check_frozen(self.net, self.guard, self.names, self.frozen)
        if (
            self.net.dope_model.training
            or self.net.adapter_10tok.anchor.training
            or torch.cuda.is_initialized()
        ):
            raise RuntimeError("frozen mode/device violation")

    def evaluate(self, rows, native=False, prove_identity=False, branch=None):
        groups, result, raw = {}, {}, {}
        for r in rows:
            cid = r["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.no_grad():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    cooperate()
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
                        scores = self.forward(kwargs, tok, mask, anchor, branch=branch)
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


def compact_summary(summary):
    """Strip verbose per-seed harm family string lists from the Git-published summary while retaining counts."""
    out = dict(summary)
    out["contrasts"] = {}
    for k, v in summary["contrasts"].items():
        cv = dict(v)
        cv["harm_families_by_seed"] = None
        out["contrasts"][k] = cv
    return out


def audit_allowed_paths(pins, launch_path, study_dir=HERE, pool_paths=()):
    study = Path(study_dir).resolve()
    study_files = set(pins.get("source_sha256", ())) | {
        "manifest.json",
        "schedules.json",
        "PREREG.md",
        "fold_manifest.json",
        "provenance.json",
    }
    allowed = (
        set(pins["inputs"])
        | set(pins["assets"])
        | {str(Path(launch_path).resolve())}
        | {str((study / name).resolve()) for name in study_files}
        | set(pool_paths)
    )
    return allowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()
    global PAUSE
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
    global OUT
    out = OUT = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    manifest_sha256 = sha(HERE / "manifest.json")
    schedules_sha256 = sha(HERE / "schedules.json")
    launch = json.loads(args.launch.read_text())
    verify(pins, launch=launch, smoke=args.smoke)
    validate_launch(
        launch,
        launch.get("source_commit"),
        manifest_sha256,
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    recovered = recover(args.resume_from, launch)
    expected_schedule = json.loads((HERE / "schedules.json").read_text())
    actual_schedule = {
        a: [
            dict(
                step=t,
                lr=settings(a, t)[0],
                beta1=settings(a, t)[1],
                beta2=settings(a, t)[2],
                weight_decay=settings(a, t)[3],
                rank=settings(a, t)[4],
                alpha=settings(a, t)[5],
                trainable_params=settings(a, t)[6],
            )
            for t in range(1, 41)
        ]
        for a in stat.ARMS
    }
    if expected_schedule["arms"] != actual_schedule:
        raise RuntimeError("schedule table identity")
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
    pool_paths = {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    allowed = audit_allowed_paths(
        pins, args.launch, study_dir=HERE, pool_paths=pool_paths
    )
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")
    if sha(HERE / "schedules.json") != schedules_sha256:
        raise RuntimeError("schedules mutation")
    updates, started = 0, time.monotonic()
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params_ctrl = [p for p in net.parameters() if p.requires_grad]
        initial = {n: net.state_dict()[n].detach().clone() for n in names}
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            sum(p.numel() for p in params_ctrl) != 29211
            or len(names) != 47
            or sum(initial[n].dim() == 2 for n in names) != 16
            or census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
        ):
            raise RuntimeError("adapter/mode inventory")
        rt.check_frozen(net, guard, names, frozen)
        coverage = {}
        for arm in stat.ARMS:
            branch, arm_params, named_trainable, probe = setup_arm(
                arm, net, names, initial, seed=stat.SEEDS[0]
            )
            covered = sorted(id(p) for g in probe.param_groups for p in g["params"])
            if covered != sorted(id(p) for p in arm_params):
                raise RuntimeError("optimizer coverage: " + arm)
            n_numel = sum(p.numel() for p in arm_params)
            if n_numel != TRAINABLE_PARAMS[arm]:
                raise RuntimeError(
                    f"trainable param count mismatch for {arm}: {n_numel}"
                )
            if arm != "CONTROL":
                if any(p.requires_grad for p in net.parameters()):
                    raise RuntimeError(f"r882 adapter not frozen in {arm}")
            coverage[arm] = dict(
                groups=len(probe.param_groups),
                trainable_tensors=len(arm_params),
                trainable_parameters=n_numel,
                r882_requires_grad=any(p.requires_grad for p in net.parameters()),
                rank=None if branch is None else branch.rank,
                alpha=None if branch is None else branch.alpha,
            )
            del branch, arm_params, named_trainable, probe
        for n, p in net.named_parameters():
            p.requires_grad_(n in set(names))
        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=47,
                    adapter_2d_matrices=16,
                    trainable_parameters=TRAINABLE_PARAMS,
                    launch_exact_path_after_audit=True,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    initial_adapter_sha256=initial_hash,
                    census=census,
                    optimizer_coverage=coverage,
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
        baseline, costs = engine.evaluate(val, native=True, prove_identity=True)
        if (
            not np.array_equal(
                np.asarray(baseline["values"]),
                np.asarray(ref["utility"]["per_context"]),
            )
            or float(np.mean(baseline["values"])) != 1.1377941898164525
        ):
            raise RuntimeError("exact 91 r882 baseline")
        # Assert update-0 scores bit-identical to r882 on Native91 for SIDE4 and SIDE16 before training.
        for arm in ("SIDE4", "SIDE16"):
            b0, _, _, _ = setup_arm(arm, net, names, initial, seed=stat.SEEDS[0])
            b0_eval, b0_costs = engine.evaluate(val, native=False, branch=b0)
            if (
                b0_eval["values"] != baseline["values"]
                or b0_eval["choices"] != baseline["choices"]
            ):
                raise RuntimeError(
                    f"{arm} update-0 utility/choice differs from r882 on Native91"
                )
            for cid in costs:
                if not np.array_equal(b0_costs[cid], costs[cid]):
                    raise RuntimeError(
                        f"{arm} update-0 cost bits differ from r882 on Native91 ({cid})"
                    )
            if rt.tensor_hash(b0.merged_state()) != initial_hash:
                raise RuntimeError(
                    f"{arm} update-0 merged state hash differs from r882"
                )
            del b0, b0_eval, b0_costs
        np.savez_compressed(out / "baseline_costs.npz", **costs)
        save(
            out / "baseline.json",
            dict(
                **baseline,
                status="PASS",
                matches=91,
                mean=1.1377941898164525,
                eval_forward_bit_exact=True,
                side4_step0_bit_exact=True,
                side16_step0_bit_exact=True,
            ),
        )
        print(
            "BASELINE PASS 91/91; cached/native and SIDE4/SIDE16 update-0 eval forward BIT-EXACT",
            flush=True,
        )
        del costs
        engine.load(train)
        npol = len(stat.ALL_ARMS)
        primary = np.full((3, 1219, npol), np.nan)
        secondary = np.full((3, 5, 91, npol), np.nan)
        unconstrained = np.full_like(primary, np.nan)
        unconstrained_val = np.full_like(secondary, np.nan)
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        selections, cells = [], []
        step40 = np.full_like(primary, np.nan)
        step40_val = np.full_like(secondary, np.nan)
        files, recovered_updates = {}, 0

        def fill(si, outer, cell):
            indices = [positions[c] for c in cell["assess_ids"]]
            base = {int(k): v for k, v in cell["assessment"][stat.ARMS[0]].items()}[0]
            primary[si, indices, 0] = unconstrained[si, indices, 0] = base["outer"][
                "values"
            ]
            secondary[si, outer, :, 0] = unconstrained_val[si, outer, :, 0] = base[
                "val"
            ]["values"]
            step40[si, indices, 0] = base["outer"]["values"]
            step40_val[si, outer, :, 0] = base["val"]["values"]
            for ai, arm in enumerate(stat.ARMS, 1):
                steps = {int(k): v for k, v in cell["assessment"][arm].items()}
                dec = cell["decisions"][arm]
                chosen, best, last = (
                    steps[dec["selected_step"]],
                    steps[dec["best_step"]],
                    steps[40],
                )
                primary[si, indices, ai] = chosen["outer"]["values"]
                secondary[si, outer, :, ai] = chosen["val"]["values"]
                unconstrained[si, indices, ai] = best["outer"]["values"]
                unconstrained_val[si, outer, :, ai] = best["val"]["values"]
                step40[si, indices, ai] = last["outer"]["values"]
                step40_val[si, outer, :, ai] = last["val"]["values"]

        def restore(snapshot):
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if n in snapshot:
                        p.copy_(snapshot[n])
            engine.check()

        for outer in range(5):
            for si, seed in enumerate(stat.SEEDS):
                if f"A{outer}-{seed}" in recovered:
                    cell = recovered[f"A{outer}-{seed}"]
                    fill(si, outer, cell)
                    files[f"A{outer}-{seed}.json"] = save_cell(
                        f"A{outer}-{seed}.json", cell
                    )
                    cells.append(cell)
                    selections.append(cell["decisions"])
                    recovered_updates += 40 * len(stat.ARMS)
                    print("CELL RECOVERED", f"A{outer}-{seed}", flush=True)
                    continue
                fit, select, assess = stat.partition(train, outer, seed)
                fitids = {r["context_sha256"] for r in fit}
                excluded = {r["context_sha256"] for r in select + assess + val}
                if fitids & excluded:
                    raise RuntimeError("family leakage")
                restore(initial)
                selbase, _ = engine.evaluate(select)
                abase, _ = engine.evaluate(assess)
                indices = [positions[r["context_sha256"]] for r in assess]
                orders = {c: np.argsort(engine.pools[c]["fitness"]) for c in fitids}
                cellname = f"A{outer}-{seed}"
                batch_cache, plan_digest = [], hashlib.sha256()
                for stream, cids, indexes in stat.batch_plan(fitids, orders, seed):
                    if not set(cids) <= fitids or set(cids) & excluded:
                        raise RuntimeError("fit-only minibatches")
                    kwargs, tok, mask, fitness, gate = engine.batch(cids, indexes)
                    batch_cache.append(
                        (
                            stream,
                            kwargs,
                            tok,
                            mask,
                            engine.anchored(kwargs),
                            fitness,
                            gate,
                        )
                    )
                    plan_digest.update(
                        json.dumps(
                            [stream, cids, [ix.tolist() for ix in indexes]]
                        ).encode()
                    )
                snapshots, selector_panels, traces = {}, {}, {}
                drifts, branch_norms, no_grad_names = {}, {}, {}
                for arm in stat.ARMS:
                    restore(initial)
                    arm_seed = stat.stream_seed(seed, outer, arm)
                    torch.manual_seed(arm_seed)
                    branch, arm_params, named_trainable, optimizer = setup_arm(
                        arm, net, names, initial, seed=arm_seed
                    )
                    if sorted(
                        id(p)
                        for group in optimizer.param_groups
                        for p in group["params"]
                    ) != sorted(id(p) for p in arm_params):
                        raise RuntimeError("optimizer coverage: " + arm)
                    snapshots[arm], selector_panels[arm], traces[arm] = (
                        {0: initial},
                        [selbase],
                        [],
                    )
                    for step, (
                        stream,
                        kwargs,
                        tok,
                        mask,
                        anchor,
                        fitness,
                        gate,
                    ) in enumerate(batch_cache, 1):
                        cooperate()
                        torch.manual_seed(stream)
                        optimizer.zero_grad(set_to_none=True)
                        pred = engine.forward(kwargs, tok, mask, anchor, branch=branch)
                        objective = arm_loss(arm, rt.loss, pred, fitness, gate)
                        if not torch.isfinite(objective):
                            raise RuntimeError("nonfinite objective")
                        objective.backward()
                        optimizer.filter_gradients()
                        norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                arm_params, 1.0, error_if_nonfinite=True
                            )
                        )
                        clipped = gradient_norm(arm_params)
                        missing = [n for n, p in named_trainable if p.grad is None]
                        no_grad_names.setdefault(arm, set()).add(tuple(missing))
                        inactive_elements = sum(
                            p.numel() for _, p in named_trainable if p.grad is None
                        )
                        optimizer.step()
                        updates += 1
                        if branch is not None:
                            for n in names:
                                if not torch.equal(net.state_dict()[n], initial[n]):
                                    raise RuntimeError(
                                        f"frozen r882 weight mutated in {arm}: {n}"
                                    )
                        engine.check()
                        traces[arm].append(
                            dict(
                                step=step,
                                loss=float(objective.detach()),
                                raw_grad_l2=norm,
                                clipped_grad_l2=clipped,
                                learning_rate=optimizer.param_groups[0]["lr"],
                                beta1=optimizer.param_groups[0]["betas"][0],
                                beta2=optimizer.param_groups[0]["betas"][1],
                                weight_decay=optimizer.param_groups[0]["weight_decay"],
                                optimizer=type(optimizer).__name__,
                                none_gradient_tensors=len(missing),
                                none_gradient_elements=inactive_elements,
                            )
                        )
                        if step in stat.STEPS:
                            if branch is None:
                                snapshot = {
                                    n: net.state_dict()[n].detach().clone()
                                    for n in names
                                }
                                extra_arrays = {}
                            else:
                                snapshot = branch.merged_state()
                                extra_arrays = {
                                    k: t.numpy()
                                    for k, t in branch.branch_tensors().items()
                                }
                            snapshots[arm][step] = snapshot
                            np.savez_compressed(
                                out / f"{cellname}-{arm}-step{step}.npz",
                                **{n: t.numpy() for n, t in snapshot.items()},
                                **extra_arrays,
                            )
                            restore(snapshot)
                            panel, _ = engine.evaluate(select)
                            selector_panels[arm].append(panel)
                            if branch is not None:
                                restore(initial)
                    drifts[arm] = adapter_drift(initial, snapshots[arm][40])
                    branch_norms[arm] = 0.0 if branch is None else branch.branch_norm()
                    if rt.tensor_hash(snapshots[arm][40]) == initial_hash:
                        raise RuntimeError("unchanged adapter")
                decisions = {
                    arm: stat.choose(
                        select,
                        np.array([p["values"] for p in selector_panels[arm]]).T,
                        stat.stream_seed(stat.BOOT_SEED, outer, seed, arm),
                    )
                    for arm in stat.ARMS
                }
                lock = out / (cellname + "-selection.json")
                save(
                    lock,
                    dict(
                        decisions=decisions,
                        fit_ids=sorted(fitids),
                        select_ids=[r["context_sha256"] for r in select],
                        assess_ids=[r["context_sha256"] for r in assess],
                        batch_plan_sha256=plan_digest.hexdigest(),
                        optimizer_schedule_sha256=sha(HERE / "schedules.json"),
                        step40_drift=drifts,
                        step40_branch_norm=branch_norms,
                    ),
                )
                lock_sha = sha(lock)
                primary[si, indices, 0], unconstrained[si, indices, 0] = (
                    abase["values"],
                    abase["values"],
                )
                secondary[si, outer, :, 0], unconstrained_val[si, outer, :, 0] = (
                    baseline["values"],
                    baseline["values"],
                )
                assessment = {}
                for ai, arm in enumerate(stat.ARMS, 1):
                    assessment[arm] = {0: dict(outer=abase, val=baseline)}
                    for step in sorted(
                        {
                            decisions[arm]["selected_step"],
                            decisions[arm]["best_step"],
                            40,
                        }
                        - {0}
                    ):
                        restore(snapshots[arm][step])
                        a, _ = engine.evaluate(assess)
                        v, _ = engine.evaluate(val)
                        assessment[arm][step] = dict(outer=a, val=v)
                    chosen = assessment[arm][decisions[arm]["selected_step"]]
                    best = assessment[arm][decisions[arm]["best_step"]]
                    primary[si, indices, ai] = chosen["outer"]["values"]
                    secondary[si, outer, :, ai] = chosen["val"]["values"]
                    unconstrained[si, indices, ai] = best["outer"]["values"]
                    unconstrained_val[si, outer, :, ai] = best["val"]["values"]
                cell = dict(
                    outer_fold=outer,
                    seed=seed,
                    assess_ids=[r["context_sha256"] for r in assess],
                    decisions=decisions,
                    selection_sha256=lock_sha,
                    selector_panels=selector_panels,
                    assessment=assessment,
                    traces=traces,
                    step40_drift=drifts,
                    step40_branch_norm=branch_norms,
                    none_gradient_names={
                        a: [list(group) for group in sorted(groups)]
                        for a, groups in no_grad_names.items()
                    },
                    adapter_sha256={
                        a: {str(s): rt.tensor_hash(w) for s, w in snaps.items()}
                        for a, snaps in snapshots.items()
                    },
                )
                files[cellname + ".json"] = save_cell(cellname + ".json", cell)
                cells.append(cell)
                selections.append(decisions)
                fill(si, outer, cell)
                save(
                    out / "progress.json",
                    dict(
                        complete_cells=len(selections),
                        optimizer_updates=updates + recovered_updates,
                        last_cell=cellname,
                        manifest_sha256=launch["manifest_sha256"],
                        files=files,
                    ),
                )
                print(
                    "CELL COMPLETE",
                    cellname,
                    "selected",
                    {a: decisions[a]["selected_step"] for a in stat.ARMS},
                    "drift",
                    {a: round(drifts[a], 4) for a in stat.ARMS},
                    "branch_norm",
                    {a: round(branch_norms[a], 4) for a in stat.ARMS},
                    flush=True,
                )
                restore(initial)
                del batch_cache, snapshots
        if (
            updates + recovered_updates != 15 * len(stat.ARMS) * 40
            or len(selections) != 15
        ):
            raise RuntimeError("incomplete wave")
        val_panel, val_unconstrained = (
            secondary.mean(axis=1),
            unconstrained_val.mean(axis=1),
        )
        np.savez_compressed(
            out / "panels.npz",
            primary=primary,
            secondary=secondary,
            unconstrained=unconstrained,
            unconstrained_val=unconstrained_val,
            step40=step40,
            step40_val=step40_val,
        )
        if guard.snapshot_dope_weights(net).composite_sha256 != rt.DOPE:
            raise RuntimeError("DOPE drift")
        p_dlval = stat.summarize(val, val_panel)
        s_oof = stat.summarize(train, primary)
        u_oof = stat.summarize(train, unconstrained)
        u_val = stat.summarize(val, val_unconstrained)
        s40_val = stat.summarize(val, step40_val.mean(axis=1))
        s40_oof = stat.summarize(train, step40)
        verdict = (
            "ADVANCE"
            if any(
                p_dlval["contrasts"][a + "_minus_r882"]["advance"]
                for a in stat.ARMS[1:]
            )
            else "NO_ADVANCE"
        )
        full_metrics = dict(
            wave="H249",
            status="COMPLETE",
            verdict=verdict,
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            fold_seed=stat.FOLD_SEED,
            optimizer_updates=updates + recovered_updates,
            complete_cells=15,
            trainable_parameters=TRAINABLE_PARAMS,
            r882_adapter_parameters=29211,
            adapter_2d_matrices=16,
            baseline_mean=1.1377941898164525,
            eval_forward_bit_exact=True,
            side4_step0_bit_exact=True,
            side16_step0_bit_exact=True,
            primary_dlval=p_dlval,
            secondary_oof=s_oof,
            descriptive_unconstrained=u_oof,
            descriptive_unconstrained_val=u_val,
            diagnostic_step40_val=s40_val,
            diagnostic_step40_oof=s40_oof,
            cliff_context=dict(
                index=stat.CLIFF_CONTEXT,
                context_sha256=val[stat.CLIFF_CONTEXT]["context_sha256"],
                family_id=val[stat.CLIFF_CONTEXT]["family_id"],
                r882=float(baseline["values"][stat.CLIFF_CONTEXT]),
                strict_per_arm_by_seed=val_panel[:, stat.CLIFF_CONTEXT, :].tolist(),
                step40_per_arm_by_seed=step40_val.mean(axis=1)[
                    :, stat.CLIFF_CONTEXT, :
                ].tolist(),
                note="Reported separately and never excluded from any statistic (hw_t1035).",
            ),
            dope_before=rt.DOPE,
            dope_after=guard.snapshot_dope_weights(net).composite_sha256,
            resume_number=launch.get("resume_number", 0),
            recovered_cells=sorted(recovered),
            recovered_optimizer_updates=recovered_updates,
            selections=selections,
            schedule_table=json.loads((HERE / "schedules.json").read_text()),
            step40_drift={
                a: {
                    "per_cell": [c["step40_drift"][a] for c in cells],
                    "median": float(np.median([c["step40_drift"][a] for c in cells])),
                    "min": min(c["step40_drift"][a] for c in cells),
                    "max": max(c["step40_drift"][a] for c in cells),
                }
                for a in stat.ARMS
            },
            step40_branch_norm={
                a: {
                    "per_cell": [c["step40_branch_norm"][a] for c in cells],
                    "median": float(
                        np.median([c["step40_branch_norm"][a] for c in cells])
                    ),
                    "min": min(c["step40_branch_norm"][a] for c in cells),
                    "max": max(c["step40_branch_norm"][a] for c in cells),
                }
                for a in stat.ARMS
            },
            pseudoreplication=stat.pseudoreplication(
                primary, val_panel, unconstrained, val_unconstrained, cells
            ),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            initial_adapter_sha256=initial_hash,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
            audit=audit.receipt(),
            elapsed_seconds=time.monotonic() - started,
        )
        save(out / "full_metrics.json", full_metrics)
        compact = dict(full_metrics)
        for k in (
            "primary_dlval",
            "secondary_oof",
            "descriptive_unconstrained",
            "descriptive_unconstrained_val",
            "diagnostic_step40_val",
            "diagnostic_step40_oof",
        ):
            compact[k] = compact_summary(full_metrics[k])
        compact.pop("schedule_table", None)
        compact["schedule_table_sha256"] = sha(HERE / "schedules.json")
        compact["full_metrics_sha256"] = sha(out / "full_metrics.json")
        save(out / "metrics.json", compact)
        print("COMPLETE", verdict, flush=True)
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
