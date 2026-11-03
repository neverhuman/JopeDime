"""H244 zero-update prediction smoothing; pinned H242 engine, H239 A1 monitor and immutable Git inputs.
All FIT decisions are fsync-locked before post-Native91 DL-Val scoring. No optimizer is constructed.
"""

import argparse
import hashlib
import importlib.metadata
import types
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import analysis as stat
import smoothing as sp

HERE = Path(__file__).resolve().parent
REPO = Path("/home/ubuntu/JopeDime")
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


def git_source(path, pins):
    return subprocess.check_output(
        ["git", "show", pins["parent_commit"] + ":" + path], cwd=REPO
    )


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources"):
        for path, digest in pins[kind].items():
            raw = (
                git_source(path, pins)
                if kind == "parent_sources"
                else Path(path).read_bytes()
            )
            if hashlib.sha256(raw).hexdigest() != digest:
                raise RuntimeError("pin mismatch: " + path)
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("source pin mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def runtime(pins):
    # Load pinned parent bytes, never the stale/shared working-tree source.
    for name in ("research", "research.flywheel_selfplay"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    for name, path in [
        (
            "research.flywheel_selfplay.runtime_audit",
            "research/flywheel_selfplay/runtime_audit.py",
        ),
        ("h244_runtime", "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"),
    ]:
        m = types.ModuleType(name)
        m.__file__ = str(REPO / path)
        sys.modules[name] = m
        exec(compile(git_source(path, pins), m.__file__, "exec"), m.__dict__)
    m.HERE = REPO  # guard git-show uses the canonical object database only
    # H232 A2: release this namespace before Micro's research.levels import.
    for name in [
        n for n in sys.modules if n == "research" or n.startswith("research.")
    ]:
        del sys.modules[name]
    return m


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

    def evaluate(self, rows, native=False, prove_identity=False, members=None):
        groups, result, single, saved_costs = {}, {}, {}, {}
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
                    if members is not None:
                        all_costs = [costs.copy()]
                        for weights in members:
                            cooperate()
                            for n, parameter in self.net.named_parameters():
                                if n in weights:
                                    parameter.copy_(weights[n])
                            member_cost = (
                                self.forward(kwargs, tok, mask, anchor).numpy().copy()
                            )
                            all_costs.append(member_cost)
                        costs = sp.mean_cost(all_costs)
                        for cid, mean, one in zip(
                            batch, costs, all_costs[1], strict=True
                        ):
                            saved_costs[cid + "_mean"] = mean.copy()
                            saved_costs[cid + "_single"] = one.copy()
                        for cid, cost in zip(batch, all_costs[1], strict=True):
                            choice = int(np.argmin(cost))
                            pool = self.pools[cid]
                            single[cid] = dict(
                                choice=choice,
                                utility=float(
                                    -pool["fitness"][choice]
                                    if pool["gate"][choice]
                                    else -2.0
                                ),
                                invalid=not bool(pool["gate"][choice]),
                            )
                        for n, parameter in self.net.named_parameters():
                            if n in self.initial:
                                parameter.copy_(self.initial[n])
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
        output = dict(
            context_ids=[r["context_sha256"] for r in rows],
            values=[result[r["context_sha256"]]["utility"] for r in rows],
            choices=[result[r["context_sha256"]]["choice"] for r in rows],
            invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
        )

        if members is not None:
            np.savez_compressed(self.cost_path, **saved_costs)
            output["single"] = dict(
                context_ids=output["context_ids"],
                values=[single[x]["utility"] for x in output["context_ids"]],
                choices=[single[x]["choice"] for x in output["context_ids"]],
                invalid=[single[x]["invalid"] for x in output["context_ids"]],
            )
        return output


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    args = ap.parse_args()
    PAUSE = args.output.with_suffix(".pause.json")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or sorted(os.sched_getaffinity(0)) != [48, 49, 50, 51]
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("CPU xbabe2 cores48-51 nice19 required")
    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    source_head = os.environ["H244_SOURCE"]
    sp.validate_launch(
        launch,
        source_head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
    )
    for f in HERE.iterdir():
        if f.is_file() and f.read_bytes() != subprocess.check_output(
            [
                "git",
                "show",
                source_head + ":research/flywheel_selfplay/h244_wsmooth/" + f.name,
            ],
            cwd=REPO,
        ):
            raise RuntimeError("merged source differs: " + f.name)
    if launch.get("resume_number", 0) != 0:
        raise RuntimeError("single run only")
    rt = runtime(pins)
    treatment = types.ModuleType("h244_forward")
    treatment.__file__ = str(
        REPO / "research/flywheel_selfplay/h237_adabelief/treatment.py"
    )
    exec(
        compile(
            git_source("research/flywheel_selfplay/h237_adabelief/treatment.py", pins),
            treatment.__file__,
            "exec",
        ),
        treatment.__dict__,
    )
    train, val = cohorts(rt)
    provenance = json.loads((HERE / "provenance.json").read_text())
    if (
        provenance["status"] != "PASS"
        or not provenance["zero_synthetic"]
        or sorted({r["dataset_id"] for r in train + val}) != provenance["dataset_ids"]
        or provenance["real_catalog_joined"] != 276
    ):
        raise RuntimeError("S3 real provenance")
    if {str(seed): stat.folds(train, seed) for seed in stat.SEEDS} != json.loads(
        (HERE / "fold_manifest.json").read_text()
    )["families_by_seed"]:
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    files = {}
    started = time.monotonic()

    def record(name, obj):
        files[name] = save(out / name, obj)
        tmp = out / "progress.json.tmp"
        with tmp.open("w") as f:
            json.dump(
                dict(
                    manifest_sha256=sha(HERE / "manifest.json"),
                    files=files,
                    optimizer_updates=0,
                ),
                f,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out / "progress.json")

    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        initial = {n: net.state_dict()[n].detach().clone() for n in names}
        initial_hash = rt.tensor_hash(initial)
        dope_before = guard.snapshot_dope_weights(net).composite_sha256
        inventory = json.loads((HERE / "perturbations.json").read_text())
        if (
            len(names) != 47
            or sum(w.numel() for w in initial.values()) != 29211
            or dope_before != rt.DOPE
            or sp.inventory(initial, names) != inventory["tensors"]
        ):
            raise RuntimeError("frozen tensor/RMS inventory")
        if inventory["member_seeds"] != {str(k): v for k, v in sp.MEMBER_SEEDS.items()}:
            raise RuntimeError("perturbation seed drift")
        # No autograd or optimizer; all members derived from the same immutable r882 state.
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        engine = Engine(
            net, rt, names, frozen, guard, fused, train + val, treatment.adapter_forward
        )
        engine.initial = initial
        ref = json.loads(Path(pins["reference"]).read_text())
        expected = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in val
        ]
        if (
            ref["evaluation_contract"]["contexts"] != expected
            or hashlib.sha256(
                json.dumps(ref["evaluation_contract"], sort_keys=True).encode()
            ).hexdigest()
            != pins["evaluation_identity"]
        ):
            raise RuntimeError("evaluation contract")
        engine.load(val)
        baseline = engine.evaluate(val, native=True, prove_identity=True)
        if (
            not np.array_equal(baseline["values"], ref["utility"]["per_context"])
            or float(np.mean(baseline["values"])) != stat.R882_MEAN
        ):
            raise RuntimeError("Native91 mismatch")
        record(
            "baseline.json",
            dict(**baseline, status="PASS", matches=91, mean=stat.R882_MEAN),
        )
        print("NATIVE91 PASS 91/91 bit-exact", flush=True)
        engine.load(train)
        tv = {"r882": engine.evaluate(train)}
        record("eval-r882-train.json", tv["r882"])
        identities = {}

        def score(seed, label, rows, split):
            sigma = stat.SIGMA_OF[label]
            members = [
                sp.perturb(initial, names, sigma, seed_)
                for seed_ in sp.MEMBER_SEEDS[seed]
            ]
            key = f"{seed}-{label}"
            hashes = [rt.tensor_hash(w) for w in members]
            if key in identities and hashes != identities[key]:
                raise RuntimeError("member weights changed across panels")
            identities[key] = hashes
            engine.cost_path = out / f"costs-{key}-{split}.npz"
            result = engine.evaluate(rows, members=members)
            if rt.tensor_hash({n: net.state_dict()[n] for n in names}) != initial_hash:
                raise RuntimeError("r882 not restored")
            record(
                f"eval-{key}-{split}.json",
                dict(**result, member_hashes=hashes, cost_sha256=sha(engine.cost_path)),
            )
            print("SCORED", key, split, flush=True)
            return result

        # One fixed Gaussian sample per seed/member, shared across sigma values and folds.
        for seed in stat.SEEDS:
            for label in stat.LABELS[1:]:
                result = score(seed, label, train, "train")
                tv[f"{seed}-{label}"] = result
                tv[f"{seed}-{stat.PLACEBO_OF[label]}"] = result["single"]
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}

        def key(seed, label):
            return "r882" if label == "r882" else f"{seed}-{label}"

        cells = []
        for seed in stat.SEEDS:
            for outer in range(5):
                fit, assess = stat.partition(train, outer, seed)
                if {r["family_id"] for r in fit} & {
                    r["family_id"] for r in assess + val
                }:
                    raise RuntimeError("family leakage")
                fi = [positions[r["context_sha256"]] for r in fit]
                panel = np.stack(
                    [
                        np.asarray(tv[key(seed, label)]["values"])[fi]
                        for label in stat.LABELS
                    ],
                    axis=1,
                )
                choice = stat.choose(
                    fit,
                    panel,
                    stat.stream_seed(stat.BOOT_SEED, seed, outer, "WSMOOTH_SELECT"),
                    stat.LABELS,
                )
                decisions = dict(
                    CONTROL=dict(selected="r882", best="r882", accepted=False),
                    WSMOOTH_SELECT=choice,
                    WSINGLE=dict(
                        selected=stat.PLACEBO_OF[choice["selected"]],
                        best=stat.PLACEBO_OF[choice["best"]],
                        matched_to="WSMOOTH_SELECT",
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
        record("selection.json", dict(cells=cells, member_hashes=identities))
        selection_sha = files["selection.json"]
        lock = out / "selection.lock.json"
        save(
            lock,
            dict(
                selection_sha256=selection_sha,
                locked_at=time.time(),
                source_commit=source_head,
            ),
        )
        directory_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory_fd)
        os.close(directory_fd)
        print("SELECTIONS LOCKED", selection_sha, flush=True)
        vv = {"r882": baseline}  # CONTROL reuses exact Native91; no duplicate scoring.
        for seed in stat.SEEDS:
            for label in stat.LABELS[1:]:
                if (
                    sha(out / "selection.json") != selection_sha
                    or json.loads(lock.read_text())["selection_sha256"] != selection_sha
                ):
                    raise RuntimeError("selection mutation")
                result = score(seed, label, val, "val")
                vv[f"{seed}-{label}"] = result
                vv[f"{seed}-{stat.PLACEBO_OF[label]}"] = result["single"]
        primary = np.full((3, 5, 91, len(stat.ALL_ARMS)), np.nan)
        oof = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        c27 = []
        for cell in cells:
            seed = cell["seed"]
            si = stat.SEEDS.index(seed)
            outer = cell["outer_fold"]
            ai = [positions[x] for x in cell["assess_ids"]]
            primary[si, outer, :, 0] = baseline["values"]
            oof[si, ai, 0] = np.asarray(tv["r882"]["values"])[ai]
            diagnostic = dict(seed=seed, outer_fold=outer, policies={})
            for col, arm in enumerate(stat.ARMS, 1):
                label = key(seed, cell["decisions"][arm]["selected"])
                primary[si, outer, :, col] = vv[label]["values"]
                oof[si, ai, col] = np.asarray(tv[label]["values"])[ai]
                diagnostic["policies"][arm] = {
                    k: vv[label][k][27] for k in ["values", "choices", "invalid"]
                }
            c27.append(diagnostic)
        engine.check()
        dope_after = guard.snapshot_dope_weights(net).composite_sha256
        if (
            dope_after != dope_before
            or rt.tensor_hash({n: net.state_dict()[n] for n in names}) != initial_hash
        ):
            raise RuntimeError("frozen/restore violation")
        np.savez_compressed(
            out / "panels.npz", primary_dlval=primary, secondary_oof=oof
        )
        metrics = dict(
            wave="H244",
            status="COMPLETE",
            optimizer_updates=0,
            complete_cells=len(cells),
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            primary_dlval=stat.summarize(val, primary.mean(axis=1)),
            secondary_oof=stat.summarize(train, oof),
            selections=cells,
            selection_sha256=selection_sha,
            selected_counts={
                arm: {
                    label: sum(c["decisions"][arm]["selected"] == label for c in cells)
                    for label in sorted(
                        {c["decisions"][arm]["selected"] for c in cells}
                    )
                }
                for arm in stat.ARMS
            },
            context27=dict(
                index_zero_based=27,
                context_sha256=val[27]["context_sha256"],
                family_id=val[27]["family_id"],
                baseline={k: baseline[k][27] for k in ["values", "choices", "invalid"]},
                selected=c27,
                fixed={
                    k: {f: v[f][27] for f in ["values", "choices", "invalid"]}
                    for k, v in vv.items()
                },
            ),
            member_hashes=identities,
            dope_before=dope_before,
            dope_after=dope_after,
            frozen_sha256=frozen,
            initial_adapter_sha256=initial_hash,
            source_commit=source_head,
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
            audit=audit.receipt(),
            elapsed_seconds=time.monotonic() - started,
        )
        metrics["verdict"] = (
            "ADVANCE"
            if metrics["primary_dlval"]["contrasts"]["WSMOOTH_SELECT_minus_r882"][
                "advance"
            ]
            else "NO_ADVANCE"
        )
        record("metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(status="INCOMPLETE", error=repr(exc), audit=audit.receipt()),
        )
        raise


if __name__ == "__main__":
    main()
