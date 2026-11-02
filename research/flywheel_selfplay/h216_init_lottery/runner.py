"""Guarded H216 CPU inference-only runner: random-init lottery with B-locked strict-margin selection."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import boto3
import numpy as np
import psutil
import torch

import analysis as stat
from parents import runner_parent

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
P = runner_parent()
_FORK_STAGE1_FN = None


def _fork_stage1_entry(outer):
    return _FORK_STAGE1_FN(outer)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with Path(path).open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def resource_snapshot():
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    disk = psutil.disk_usage("/home/ubuntu")
    proc = psutil.Process()
    tree = [proc] + proc.children(recursive=True)
    load1, load5, load15 = os.getloadavg()
    disk_pct = 100.0 * disk.used / (disk.used + disk.free)
    return dict(
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        load1=float(load1),
        load5=float(load5),
        load15=float(load15),
        available_ram_gib=float(vm.available / (1024**3)),
        swap_total_gib=float(sw.total / (1024**3)),
        swap_used_gib=float(sw.used / (1024**3)),
        swap_free_gib=float(sw.free / (1024**3)),
        disk_used_pct=float(disk_pct),
        disk_free_gib=float(disk.free / (1024**3)),
        process_tree_rss_gib=float(
            sum(p.memory_info().rss for p in tree if p.is_running()) / (1024**3)
        ),
        process_tree_threads=int(sum(p.num_threads() for p in tree if p.is_running())),
    )


def enforce_resource_guards(snap=None):
    s = resource_snapshot() if snap is None else snap
    if (
        s["load1"] >= 85.0
        or s["available_ram_gib"] < 16.0
        or s["disk_used_pct"] > 96.0
        or s["process_tree_rss_gib"] >= 12.0
    ):
        raise RuntimeError("H216 resource guard refusal: " + json.dumps(s))
    return s


class S3Store:
    def __init__(self, output, prefix):
        self.output = Path(output)
        self.prefix = prefix.rstrip("/") + "/"
        self.bucket = "veox-jopedime-learning-use2"
        self.client = boto3.Session(profile_name="veox-jepson-prod").client(
            "s3", region_name="us-east-2"
        )
        self.receipts = {}

    def put_bytes(self, name, data):
        key = self.prefix + name
        sha_hex = digest(data)
        resp = self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            IfNoneMatch="*",
            Metadata={"sha256": sha_hex},
        )
        version = resp["VersionId"]
        got = self.client.get_object(Bucket=self.bucket, Key=key, VersionId=version)
        body = got["Body"].read()
        got["Body"].close()
        if got["VersionId"] != version or digest(body) != sha_hex:
            raise RuntimeError("S3 exact-version verification failed for " + key)
        rec = dict(
            s3_uri=f"s3://{self.bucket}/{key}",
            version_id=version,
            sha256=sha_hex,
            bytes=len(data),
        )
        self.receipts[name] = rec
        return rec

    def put_json(self, name, obj):
        raw = (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode("utf-8")
        return self.put_bytes(name, raw)

    def put_npz(self, name, **arrays):
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return self.put_bytes(name, buf.getvalue())


def verify_pins(pins):
    for kind in ("inputs", "assets", "parent_sources"):
        for path, expected in pins[kind].items():
            target = REPO / path if kind == "parent_sources" else Path(path)
            if sha(target) != expected:
                raise RuntimeError(f"pin mismatch for {kind}: {target}")
    for path, expected in pins["source_sha256"].items():
        if sha(HERE / path) != expected:
            raise RuntimeError("source mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def build_prebatches(engine, rows):
    groups = {}
    for row in rows:
        cid = row["context_sha256"]
        groups.setdefault(len(engine.pools[cid]["fitness"]), []).append(cid)
    prebatches = []
    with torch.no_grad():
        for k, cids in sorted(groups.items()):
            for start in range(0, len(cids), 8):
                batch = cids[start : start + 8]
                kwargs, tokens, masks, _, _ = engine.batch(
                    batch, [np.arange(k) for _ in batch]
                )
                if any(cid not in engine.cached for cid in batch):
                    anchor = engine.anchored(kwargs)
                    for i, cid in enumerate(batch):
                        engine.cached[cid] = tuple(t[i].clone() for t in anchor)
                anchor = tuple(
                    torch.stack([engine.cached[c][j] for c in batch]) for j in range(2)
                )
                prebatches.append((batch, kwargs, tokens, masks, anchor))
    return prebatches


def evaluate_prebatched(engine, rows, prebatches):
    results, raw = {}, {}
    with torch.no_grad():
        for batch, kwargs, tokens, masks, anchor in prebatches:
            scores = engine.forward(kwargs, tokens, masks, anchor)
            costs = scores.numpy()
            if not np.isfinite(costs).all():
                raise RuntimeError("nonfinite prediction")
            for cid, cost in zip(batch, costs, strict=True):
                pool = engine.pools[cid]
                choice = int(np.argmin(cost))
                results[cid] = float(
                    -pool["fitness"][choice] if pool["gate"][choice] else -2.0
                )
                raw[cid] = cost.copy()
    engine.check()
    return {r["context_sha256"]: results[r["context_sha256"]] for r in rows}, raw


def persist_scores(output, tag, rows, raw, engine):
    ids = [r["context_sha256"] for r in rows]
    arrays = [np.asarray(raw[c], dtype=np.float32) for c in ids]
    counts = np.array([len(x) for x in arrays], dtype=np.int32)
    if any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays):
        raise RuntimeError("invalid candidate score vector")
    choices = np.array([int(np.argmin(x)) for x in arrays], dtype=np.int32)
    utilities = np.array(
        [
            (-engine.pools[c]["fitness"][j] if engine.pools[c]["gate"][j] else -2.0)
            for c, j in zip(ids, choices, strict=True)
        ],
        dtype=np.float64,
    )
    payload = dict(
        context_ids=np.array(ids),
        family_ids=np.array([r["family_id"] for r in rows]),
        candidate_counts=counts,
        offsets=np.r_[0, np.cumsum(counts, dtype=np.int64)],
        scores=np.concatenate(arrays),
        choices=choices,
        utilities=utilities,
    )
    np.savez_compressed(output / (tag + ".npz"), **payload)
    return (
        dict(zip(ids, utilities.tolist(), strict=True)),
        dict(zip(ids, choices.tolist(), strict=True)),
        payload,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-one-fold", action="store_true")
    ap.add_argument("--s3-prefix", type=str, default=None)
    args = ap.parse_args()

    try:
        if os.getpriority(os.PRIO_PROCESS, 0) < 19:
            os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    except OSError:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("CPU xbabe2 nice19 with CUDA uninitialized required")
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        val = int(os.environ.get(env_var, "4"))
        if val < 1 or val > 16:
            raise RuntimeError(f"{env_var} must be in 1..16")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pins = json.loads((HERE / "manifest.json").read_text())
    verify_pins(pins)
    if pins["max_total_updates"] != 0 or pins["arms"] != list(stat.ARMS):
        raise RuntimeError("H216 requires zero updates and fixed four-arm roster")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    snap_start = enforce_resource_guards()
    rt = P.runtime()
    if pins["dope_composite_sha256"] != rt.DOPE:
        raise RuntimeError("DOPE composite SHA-256 mismatch")

    train, val = P.cohorts(rt, json.loads((HERE / "provenance.json").read_text()))
    folds = stat.family_folds([r["family_id"] for r in train])
    if folds != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
        raise RuntimeError("fold_manifest.json family partition mismatch")

    allowed = set(pins["inputs"]) | set(pins["assets"])
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    started = time.monotonic()

    def interrupted(signum, _frame):
        raise RuntimeError("received signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    original_step = torch.optim.AdamW.step

    def forbidden_step(*_a, **_kw):
        raise RuntimeError("H216 is inference-only; optimizer.step is forbidden")

    torch.optim.AdamW.step = forbidden_step
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        initial = P.adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        params = [p for p in net.parameters() if p.requires_grad]
        if (
            len(names) != 47
            or len(params) != 47
            or sum(p.numel() for p in params) != 29211
        ):
            raise RuntimeError("adapter inventory must be 47 tensors / 29,211 params")
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
        ):
            raise RuntimeError("effective dropout/frozen mode mismatch")
        rt.check_frozen(net, guard, names, frozen)

        if args.smoke:
            save(
                output / "smoke.json",
                dict(
                    status="PASS",
                    wave="H216",
                    scored_contexts=0,
                    optimizer_updates=0,
                    trainable_parameters=29211,
                    adapter_tensors=47,
                    dope_sha256=rt.DOPE,
                    frozen_sha256=frozen,
                    initial_adapter_sha256=initial_hash,
                    resources=snap_start,
                ),
            )
            return

        engine = P.Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, baseline_costs = engine.evaluate(
            val, native=True, prove_identity=True
        )
        values = np.array(
            [baseline[r["context_sha256"]] for r in val], dtype=np.float64
        )
        expected = np.asarray(
            json.loads(Path(pins["reference"]).read_text())["utility"]["per_context"],
            dtype=np.float64,
        )
        if (
            values.tobytes() != expected.tobytes()
            or float(values.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("exact 91-context native r882 baseline mismatch")
        with np.load(pins["baseline_cost_reference"], allow_pickle=False) as ref:
            if any(
                baseline_costs[c].dtype != np.float32
                or baseline_costs[c].tobytes() != ref[c].tobytes()
                for c in baseline_costs
            ):
                raise RuntimeError("native candidate-cost reference bytes mismatch")
        save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                mean=float(values.mean()),
                bit_exact=True,
                candidate_cost_bytes_exact=True,
                TRAIN_outcome_pools_opened=0,
                dope_sha256=rt.DOPE,
                frozen_sha256=frozen,
                initial_adapter_sha256=initial_hash,
            ),
        )
        print(
            "BASELINE PASS: 91/91 utility and native cost bytes; 0 updates", flush=True
        )

        fresh = {}
        init_hashes = {}
        for init_seed in stat.all_candidate_seeds():
            fresh[init_seed] = P.random_adapter_state(net, names, init_seed)
            init_hashes[init_seed] = rt.tensor_hash(fresh[init_seed])
            engine.check()
        stat.validate_candidate_init_hashes(init_hashes, initial_hash)
        save(
            output / "initial_hashes.json",
            {str(k): v for k, v in init_hashes.items()},
        )

        outer_folds = (0,) if args.smoke_one_fold else (0, 1, 2, 3, 4)
        all_train_fams = [r["family_id"] for r in train]
        s3 = S3Store(output, args.s3_prefix) if args.s3_prefix else None

        # Stage 1: Evaluate B_outer ONLY and lock LOTTERY16 and PLACEBO choices before any A_outer outcome is read.
        b_locks = {}
        b_resource_snapshots = [enforce_resource_guards()]
        fold_raw_r882 = {}
        fold_raw_cand = {}

        def _eval_stage1_fold(outer):
            torch.set_num_threads(1)
            fit_f, b_f, a_f = stat.partition(all_train_fams, outer, folds)
            b_fold_idx = (outer + 1) % 5
            b_rows = [r for r in train if r["family_id"] in b_f]
            a_rows = [r for r in train if r["family_id"] in a_f]
            if {r["context_sha256"] for r in b_rows} & {
                r["context_sha256"] for r in a_rows
            }:
                raise RuntimeError("B/A context overlap")
            for r in a_rows:
                engine.pools.pop(r["context_sha256"], None)
            engine.load(b_rows)
            if any(r["context_sha256"] in engine.pools for r in a_rows):
                raise RuntimeError("A_outer outcome pool present during B_outer lock")
            P.restore(net, names, initial)
            b_prebatches = build_prebatches(engine, b_rows)
            r882_b_u, r882_b_raw = evaluate_prebatched(engine, b_rows, b_prebatches)
            persist_scores(output, f"B{outer}-R882", b_rows, r882_b_raw, engine)
            b_fams, r882_b_fam = stat.family_equal_vector(b_rows, r882_b_u)
            if set(b_fams) != b_f or not set(b_fams).isdisjoint(a_f):
                raise RuntimeError("B family closure mismatch")

            fold_lock = {}
            fold_cand_raw = {}
            for seed in stat.SEEDS:
                cand_seeds = stat.candidate_seeds(seed)
                cand_b_fam = []
                for k_idx, c_seed in enumerate(cand_seeds, 1):
                    P.restore(net, names, fresh[c_seed])
                    c_u, c_raw = evaluate_prebatched(engine, b_rows, b_prebatches)
                    fold_cand_raw[(seed, k_idx)] = c_raw
                    persist_scores(
                        output,
                        f"B{outer}-S{seed}-CAND{k_idx}",
                        b_rows,
                        c_raw,
                        engine,
                    )
                    fams_k, fam_vec_k = stat.family_equal_vector(b_rows, c_u)
                    if fams_k != b_fams:
                        raise RuntimeError("B candidate family ordering mismatch")
                    cand_b_fam.append(fam_vec_k)
                cand_b_matrix = np.stack(cand_b_fam, axis=0)
                lottery_sel = stat.select_on_b(r882_b_fam, cand_b_matrix, seed, outer)
                placebo_sel = stat.select_placebo_on_b(
                    r882_b_fam, cand_b_matrix, seed, outer
                )
                fold_lock[str(seed)] = dict(
                    outer_fold=outer,
                    seed=seed,
                    B_families=b_fams,
                    A_families_unscored=sorted(a_f),
                    FIT_families_unused=sorted(fit_f),
                    r882_b_equal_family_utility=float(r882_b_fam.mean()),
                    LOTTERY16=lottery_sel,
                    PLACEBO=placebo_sel,
                )
            P.restore(net, names, initial)
            engine.check()
            print(f"STAGE 1 B-LOCK COMPLETE: outer={outer}", flush=True)
            return outer, b_fold_idx, fold_lock, r882_b_raw, fold_cand_raw

        if len(outer_folds) == 1:
            results_stage1 = [_eval_stage1_fold(outer_folds[0])]
        else:
            global _FORK_STAGE1_FN
            _FORK_STAGE1_FN = _eval_stage1_fold
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=len(outer_folds)) as pool:
                results_stage1 = pool.map(_fork_stage1_entry, outer_folds)

        for outer, b_fold_idx, fold_lock, r882_b_raw, fold_cand_raw in results_stage1:
            b_locks[outer] = fold_lock
            fold_raw_r882[b_fold_idx] = r882_b_raw
            fold_raw_cand[b_fold_idx] = fold_cand_raw
        b_resource_snapshots.append(enforce_resource_guards())

        lock_payload = dict(
            schema="jope.h216-b-selection-lock/v1",
            wave="H216",
            status="LOCKED_BEFORE_A",
            outer_folds=list(outer_folds),
            seeds=list(stat.SEEDS),
            margin=stat.B_MARGIN,
            quantile=stat.B_QUANTILE,
            b_bootstrap_draws=stat.DRAWS,
            a_contexts_scored_before_lock=0,
            optimizer_updates=0,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            folds={str(k): v for k, v in b_locks.items()},
        )
        save(output / "b_selection_lock.json", lock_payload)
        lock_sha = sha(output / "b_selection_lock.json")
        if s3 is not None:
            s3.put_json("b_selection_lock.json", lock_payload)

        if args.smoke_one_fold:
            rt.check_frozen(net, guard, names, frozen)
            verify_pins(pins)
            snap_end = enforce_resource_guards()
            save(
                output / "smoke_fold0.json",
                dict(
                    status="PASS",
                    wave="H216",
                    mode="SMOKE_ONE_FOLD_B_ONLY",
                    source_commit=head,
                    outer_fold=0,
                    scored_a_contexts=0,
                    b_selection_lock_sha256=lock_sha,
                    optimizer_updates=0,
                    dope_sha256_before=rt.DOPE,
                    dope_sha256_after=rt.DOPE,
                    frozen_sha256=frozen,
                    initial_adapter_sha256=initial_hash,
                    fold0_locks={
                        s: dict(
                            lottery16_selected=b_locks[0][s]["LOTTERY16"][
                                "selected_label"
                            ],
                            placebo_selected=b_locks[0][s]["PLACEBO"]["selected_label"],
                        )
                        for s in map(str, stat.SEEDS)
                    },
                    resources_start=snap_start,
                    resources_end=snap_end,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            print("SMOKE ONE FOLD PASS (0 A contexts scored)", flush=True)
            return

        # Stage 2: Score A_outer using locked B selections (plus diagnostic ORACLE_A).
        policies = {s: {a: {} for a in stat.ARMS} for s in stat.SEEDS}
        oracle_records = {str(outer): {} for outer in outer_folds}
        a_resource_snapshots = []
        for outer in outer_folds:
            a_resource_snapshots.append(enforce_resource_guards())
            _, b_f, a_f = stat.partition(all_train_fams, outer, folds)
            a_rows = [r for r in train if r["family_id"] in a_f]
            if set(a_f) & set(b_f):
                raise RuntimeError("A/B family overlap")
            engine.load(a_rows)
            if outer in fold_raw_r882:
                r882_a_raw = fold_raw_r882[outer]
                r882_a_u, _, _ = persist_scores(
                    output, f"A{outer}-R882", a_rows, r882_a_raw, engine
                )
            else:
                P.restore(net, names, initial)
                r882_a_u, r882_a_raw = engine.evaluate(
                    a_rows, native=True, prove_identity=True
                )
                persist_scores(output, f"A{outer}-R882", a_rows, r882_a_raw, engine)
            a_fams, r882_a_fam = stat.family_equal_vector(a_rows, r882_a_u)

            for seed in stat.SEEDS:
                policies[seed]["R882"].update(r882_a_u)
                cand_seeds = stat.candidate_seeds(seed)
                cand_a_u = {}
                cand_a_raw = {}
                cand_a_fam = []
                for k_idx, c_seed in enumerate(cand_seeds, 1):
                    if outer in fold_raw_cand:
                        c_raw = fold_raw_cand[outer][(seed, k_idx)]
                        c_u, _, _ = persist_scores(
                            output,
                            f"A{outer}-S{seed}-CAND{k_idx}",
                            a_rows,
                            c_raw,
                            engine,
                        )
                    else:
                        P.restore(net, names, fresh[c_seed])
                        c_u, c_raw = engine.evaluate(a_rows, native=False)
                        persist_scores(
                            output,
                            f"A{outer}-S{seed}-CAND{k_idx}",
                            a_rows,
                            c_raw,
                            engine,
                        )
                    cand_a_u[k_idx] = c_u
                    cand_a_raw[k_idx] = c_raw
                    fams_k, fam_vec_k = stat.family_equal_vector(a_rows, c_u)
                    if fams_k != a_fams:
                        raise RuntimeError("A candidate family ordering mismatch")
                    cand_a_fam.append(fam_vec_k)
                cand_a_matrix = np.stack(cand_a_fam, axis=0)
                oracle_sel = stat.select_oracle_on_a(
                    r882_a_fam, cand_a_matrix, seed, outer
                )
                oracle_records[str(outer)][str(seed)] = oracle_sel

                cell_lock = b_locks[outer][str(seed)]
                for arm_name, sel_rec in (
                    ("LOTTERY16", cell_lock["LOTTERY16"]),
                    ("PLACEBO", cell_lock["PLACEBO"]),
                    ("ORACLE_A", oracle_sel),
                ):
                    idx = int(sel_rec["selected_candidate_index"])
                    chosen_u = r882_a_u if idx == 0 else cand_a_u[idx]
                    chosen_raw = r882_a_raw if idx == 0 else cand_a_raw[idx]
                    persist_scores(
                        output,
                        f"A{outer}-S{seed}-{arm_name}",
                        a_rows,
                        chosen_raw,
                        engine,
                    )
                    policies[seed][arm_name].update(chosen_u)

            P.restore(net, names, initial)
            engine.check()
            print(f"STAGE 2 A-EVAL COMPLETE: outer={outer}", flush=True)

        # Verify DOPE and all pinned inputs/assets unchanged after completion.
        rt.check_frozen(net, guard, names, frozen)
        verify_pins(pins)
        snap_end = enforce_resource_guards()

        if args.smoke_one_fold:
            smoke_ids = sorted(policies[stat.SEEDS[0]]["R882"])
            save(
                output / "smoke_fold0.json",
                dict(
                    status="PASS",
                    wave="H216",
                    mode="SMOKE_ONE_FOLD",
                    source_commit=head,
                    outer_fold=0,
                    scored_a_contexts=len(smoke_ids),
                    b_selection_lock_sha256=lock_sha,
                    optimizer_updates=0,
                    dope_sha256_before=rt.DOPE,
                    dope_sha256_after=rt.DOPE,
                    frozen_sha256=frozen,
                    initial_adapter_sha256=initial_hash,
                    fold0_locks={
                        s: dict(
                            lottery16_selected=b_locks[0][s]["LOTTERY16"][
                                "selected_label"
                            ],
                            placebo_selected=b_locks[0][s]["PLACEBO"]["selected_label"],
                            oracle_a_selected=oracle_records["0"][s]["selected_label"],
                        )
                        for s in map(str, stat.SEEDS)
                    },
                    resources_start=snap_start,
                    resources_end=snap_end,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            print("SMOKE ONE FOLD PASS", flush=True)
            return

        ids = sorted(policies[stat.SEEDS[0]]["R882"])
        if len(ids) != 1219:
            raise RuntimeError(f"expected 1219 OOF contexts, got {len(ids)}")
        family_of = {r["context_sha256"]: r["family_id"] for r in train}
        fam_ids = [family_of[c] for c in ids]
        panel = np.array(
            [
                [[policies[s][arm][c] for arm in stat.ARMS] for c in ids]
                for s in stat.SEEDS
            ],
            dtype=np.float64,
        )

        b_summary = {
            "b_selection_lock_sha256": lock_sha,
            "lottery16_non_r882_adoptions": sum(
                1
                for outer in outer_folds
                for s in map(str, stat.SEEDS)
                if not b_locks[outer][s]["LOTTERY16"]["fallback_to_r882"]
            ),
            "placebo_non_r882_adoptions": sum(
                1
                for outer in outer_folds
                for s in map(str, stat.SEEDS)
                if not b_locks[outer][s]["PLACEBO"]["fallback_to_r882"]
            ),
            "per_fold_selections": {
                str(outer): {
                    s: dict(
                        lottery16_selected=b_locks[outer][s]["LOTTERY16"][
                            "selected_label"
                        ],
                        lottery16_b_boot_median_gain=b_locks[outer][s]["LOTTERY16"][
                            "candidates"
                        ][
                            b_locks[outer][s]["LOTTERY16"]["selected_candidate_index"]
                            - 1
                        ]["b_boot_median_gain"]
                        if b_locks[outer][s]["LOTTERY16"]["selected_candidate_index"]
                        > 0
                        else 0.0,
                        lottery16_max_b_q10=max(
                            c["b_boot_q10_gain"]
                            for c in b_locks[outer][s]["LOTTERY16"]["candidates"]
                        ),
                        lottery16_max_b_boot_median=max(
                            c["b_boot_median_gain"]
                            for c in b_locks[outer][s]["LOTTERY16"]["candidates"]
                        ),
                        placebo_selected=b_locks[outer][s]["PLACEBO"]["selected_label"],
                        placebo_max_b_q10=max(
                            c["b_boot_q10_gain"]
                            for c in b_locks[outer][s]["PLACEBO"]["candidates"]
                        ),
                        oracle_a_selected=oracle_records[str(outer)][s][
                            "selected_label"
                        ],
                        oracle_a_family_gain=oracle_records[str(outer)][s][
                            "oracle_a_family_gain"
                        ],
                    )
                    for s in map(str, stat.SEEDS)
                }
                for outer in outer_folds
            },
        }

        summary = stat.summarize(panel, fam_ids, b_selections=b_summary)
        summary.update(
            wave="H216",
            status="COMPLETE",
            source_commit=head,
            manifest_sha256=sha(HERE / "manifest.json"),
            outer_folds=5,
            candidate_inits_per_seed=stat.CANDIDATE_COUNT,
            total_distinct_candidate_inits=len(init_hashes),
            actual_optimizer_updates=0,
            dope_sha256_before=rt.DOPE,
            dope_sha256_after=rt.DOPE,
            frozen_sha256=frozen,
            initial_adapter_sha256=initial_hash,
            baseline_dl_val_mean=1.1377941898164525,
            b_locked_before_a=True,
            resources_start=snap_start,
            resources_end=snap_end,
            elapsed_seconds=time.monotonic() - started,
        )

        oof_payload = dict(
            context_ids=np.array(ids),
            family_ids=np.array(fam_ids),
            seed_ids=np.array(stat.SEEDS),
            arm_names=np.array(stat.ARMS),
            utility=panel,
        )
        np.savez_compressed(output / "OOF-panels.npz", **oof_payload)
        audit_payload = dict(
            status="PASS",
            **audit.receipt(),
            optimizer_updates=0,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            elapsed_seconds=time.monotonic() - started,
        )
        save(output / "audit.json", audit_payload)

        if s3 is not None:
            s3.put_npz("OOF-panels.npz", **oof_payload)
            s3.put_json("audit.json", audit_payload)
            s3.put_json(
                "initial_hashes.json",
                {str(k): v for k, v in init_hashes.items()},
            )
            summary["s3_custody"] = s3.receipts
            metrics_rec = s3.put_json("metrics.json", summary)
            summary["s3_custody"]["metrics.json"] = metrics_rec

        save(output / "metrics.json", summary)
        print("COMPLETE " + summary["verdict"], flush=True)
    except BaseException as exc:
        save(
            output / "failure.json",
            dict(
                status="INCOMPLETE",
                error=repr(exc),
                optimizer_updates=0,
                audit=audit.receipt(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise
    finally:
        torch.optim.AdamW.step = original_step


if __name__ == "__main__":
    main()
