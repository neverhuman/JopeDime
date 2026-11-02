#!/usr/bin/env python3
"""H214: reuse pinned H212 execution, with a guarded supervisor and two16-wide pools."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from research.flywheel_selfplay.h214_pool_diversity import analysis as stat
from research.flywheel_selfplay.h214_pool_diversity import protocol

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path.home() / "jope-data"
H212 = REPO / "research/flywheel_selfplay/h212_pool_width"
H212_COMMIT = "c39a96589f340e3d8faa833840e8ddf224aed2a5"
H212_SHA = "cf42b33391aee91d6650b72ad46da10ae85ce9ab9b5e1d2758ba44556d91cce6"
R882 = DATA / "h214/input/round-882.pt"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temp = Path(str(path) + ".new")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def upstream_blob(name):
    return subprocess.check_output(
        [
            "git",
            "show",
            f"{H212_COMMIT}:research/flywheel_selfplay/h212_pool_width/{name}",
        ],
        cwd=REPO,
    )


def load_input_manifest():
    manifest = json.loads((HERE / "manifest.json").read_text())
    original = upstream_blob("manifest.json")
    if (
        hashlib.sha256(original).hexdigest()
        != manifest["upstream"]["h212_manifest_sha256"]
    ):
        raise ValueError("H212 input manifest drift")
    return json.loads(original)


def load_h212():
    if (
        hashlib.sha256(upstream_blob("runner.py")).hexdigest() != H212_SHA
        or sha(H212 / "runner.py") != H212_SHA
    ):
        raise ValueError("merged H212 runner pin differs")
    analysis = stat.load_upstream_analysis()

    previous = sys.modules.get("analysis")
    sys.modules["analysis"] = analysis
    try:
        spec = importlib.util.spec_from_file_location("h214_h212", H212 / "runner.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("analysis", None)
        else:
            sys.modules["analysis"] = previous


def family_bank(champions, target_family):
    if any(row["dl_split"] != "DL-Train" for row in champions):
        raise ValueError("retrieval bank must contain TRAIN records only")
    return [row for row in champions if row["family_id"] != target_family]


def retrieval_parent(bank, landmark):
    if not bank:
        raise ValueError("empty family-disjoint TRAIN parent bank")
    target = np.asarray(landmark[:12], dtype=np.float64)
    return min(
        bank,
        key=lambda row: (
            float(
                np.sum(
                    (target - np.asarray(row["landmark_summary"], dtype=np.float64))
                    ** 2
                )
            ),
            row["program_sha256"],
        ),
    )


def score_panels(pools, union, results, costs):
    identities = [row["program_sha256"] for row in union]
    if (
        len(set(identities)) != len(identities)
        or len(results) != len(union)
        or [row.get("program_sha256") for row in results] != identities
        or any(row.get("status") != "completed" for row in results)
    ):
        raise ValueError("complete ordered native evaluator results required")
    costs = np.asarray(costs, dtype=np.float64)
    fitness = np.asarray([row["fitness"] for row in results], dtype=np.float64)
    gates = np.asarray([row["cv_gate_passed"] for row in results], dtype=bool)
    if (
        costs.shape != (len(union),)
        or not np.isfinite(costs).all()
        or not np.isfinite(fitness).all()
    ):
        raise ValueError("complete finite cost/fitness panel required")
    index = {key: i for i, key in enumerate(identities)}
    out = {}
    for arm in stat.ARMS:
        rows = pools[arm]
        if len(rows) != 16 or len({r["program_sha256"] for r in rows}) != 16:
            raise ValueError("exact16 unique scored programs per arm required")
        keep = [index[row["program_sha256"]] for row in rows]
        utility, selected = stat.first_argmin_utility(
            costs[keep], fitness[keep], gates[keep]
        )
        out[arm] = dict(
            utility=utility,
            selected_index=selected,
            oracle_utility=float(np.where(gates[keep], -fitness[keep], -2.0).max()),
            ordered_program_sha256=[row["program_sha256"] for row in rows],
        )
    return out


def setup(root):
    h212 = load_h212()
    original = load_input_manifest()
    # Relocate only the already pinned checkpoint COPY; other identities are untouched.
    original["inputs"]["r882_owned_copy_path"] = str(R882)
    relocated = root / "h212-inputs-relocated.json"
    write(relocated, original)
    h212.MANIFEST, h212.R882 = relocated, R882
    h212.verify_inputs()
    h139 = h212.load_source(h212.H139_PATH, "h214_h139", h212.H139_SHA)
    h124 = h139.load_h124()
    h124.R882 = R882
    h124.FILES[R882] = h212.R882_SHA
    train, val, reference = h139.roster(h124)
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
        or len({r["family_id"] for r in val}) != 19
        or {r["family_id"] for r in train} & {r["family_id"] for r in val}
    ):
        raise ValueError("approved family-disjoint H50/H58 roster required")
    source = json.loads((h212.INPUT / "source_manifest.json").read_text())
    if source["excluded_payloads_opened"] != 0 or source["outer_rows_delivered"] != 0:
        raise ValueError("input read closure differs")
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            for r in rows
        ):
            raise ValueError("unauthorized context")
    selected = [*val, train[0]]
    context, _ = h124.train_context(selected, [])
    projection = h124.train_projection(selected, [])
    pools = h139.pools_for(selected)
    costs, net = h124.baseline(root, selected, context, pools, projection)
    import torch

    torch.set_num_threads(1)  # native evaluator uses4 workers with single-thread math
    if (
        net.training
        or any(p.requires_grad for p in net.parameters())
        or torch.cuda.is_initialized()
    ):
        raise ValueError("frozen CPU inference required")
    native = np.asarray(
        [
            h139.native(
                costs[r["context_sha256"]],
                pools[r["context_sha256"]]["fitness"],
                pools[r["context_sha256"]]["gate"],
            )
            for r in val
        ]
    )
    parity = h139.regression_check(native, reference)
    if float(native.mean()) != h212.REFERENCE_MEAN:
        raise ValueError("native r882 baseline mean differs")
    write(root / "baseline-parity.json", dict(parity, exact_mean=float(native.mean())))
    generator = h212.load_source(
        h212.GENERATOR_PATH, "h214_generator", h212.GENERATOR_SHA
    )
    if sha(generator.WORKER_BIN) != generator.WORKER_SHA256:
        raise ValueError("pinned Rust worker differs")
    envelope = h212.load_source(
        h212.H58_ENVELOPE_PATH, "h214_envelope", h212.H58_ENVELOPE_SHA
    )
    champions = json.loads((h212.CACHE / "champions_featurized.json").read_text())
    if len(champions) != 1219 or {r["context_sha256"] for r in champions} != {
        r["context_sha256"] for r in train
    }:
        raise ValueError("complete TRAIN champion bank required")
    family_bank(champions, "")
    generator.EVAL_SCRATCH_ROOT = root / "rust_eval"
    generator.EVAL_SCRATCH_ROOT.mkdir()
    return dict(
        h212=h212,
        h124=h124,
        generator=generator,
        envelope=envelope,
        train=train,
        val=val,
        context=context,
        projection=projection,
        pools=pools,
        costs=costs,
        net=net,
        champions=champions,
        baseline=native,
    )


def measure(runtime, row, seed, root):
    h212, generator = runtime["h212"], runtime["generator"]
    cid = row["context_sha256"]
    bank = family_bank(runtime["champions"], row["family_id"])
    parent = retrieval_parent(bank, runtime["context"][cid]["lm48_z"])
    arguments = dict(
        parent_programs=[parent],
        landmark_v2_48=runtime["context"][cid]["lm48_z"],
        champion_bank=bank,
        target_family_id=row["family_id"],
        seed=stat.stream_seed(seed, cid),
    )
    proposals = {
        arm: protocol.propose_arm(generator.propose_flywheel_candidates, arguments, arm)
        for arm in stat.ARMS
    }
    all_rows = stat.unique_programs(
        parent, [*proposals["NATIVE16"], *proposals["FPS16"]]
    )
    projections = {
        p["program_sha256"]: generator.featurize_dsl_412(p["dsl"])[1] for p in all_rows
    }
    pools = dict(
        NATIVE16=stat.native_pool(parent, proposals["NATIVE16"]),
        FPS16=stat.fps_pool(parent, proposals["FPS16"], projections),
    )
    union_by_sha = {}
    for arm in stat.ARMS:
        for program in pools[arm]:
            union_by_sha.setdefault(program["program_sha256"], program)
    union = list(union_by_sha.values())
    raw = (h212.INPUT / "tables" / (cid + ".jtt1")).read_bytes()
    view = runtime["envelope"].training_cv_envelope(raw)
    h212.validate_train_cv_envelope(
        raw,
        view,
        row["train_only_jtt1_sha256"],
        runtime["envelope"].training_cv_envelope,
    )
    table = root / (cid + ".jtt1.gz")
    table.write_bytes(gzip.compress(view, mtime=0))
    results = generator.evaluate_gp_batch_rust(
        str(table), union, run_outer_test=False, max_workers=4, timeout_s=25.0
    )
    costs = h212.predict_dynamic(
        runtime["net"],
        runtime["h124"],
        cid,
        runtime["context"][cid],
        runtime["pools"][cid],
        runtime["projection"],
        union,
        generator,
    )
    panels = score_panels(pools, union, results, costs[4:])
    record = dict(
        seed=seed,
        context_sha256=cid,
        family_id=row["family_id"],
        parent_sha256=parent["program_sha256"],
        evaluated_union=len(union),
        train_only_jtt1_sha256=row["train_only_jtt1_sha256"],
        cv_envelope_sha256=hashlib.sha256(view).hexdigest(),
        programs=union,
        native_results=results,
        costs=costs[4:].tolist(),
        arms=panels,
        diversity={arm: stat.diversity(pools[arm], projections) for arm in stat.ARMS},
    )
    with (root / "panels.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    # Only fresh derived evaluator scratch is reclaimed, after raw panel evidence is durable.
    tag = hashlib.sha256(str(table).encode()).hexdigest()[:16]
    shutil.rmtree(generator.EVAL_SCRATCH_ROOT / ("ctx-" + tag), ignore_errors=True)
    table.unlink()
    return record


def elo(runtime, panels):
    helper = runtime["h124"].source_module(
        "python/src/jope/north_star/elo_tracker.py", "h214_elo"
    )
    ratings = {anchor: {} for anchor in ["R882", "NATIVE16"]}
    for seed in stat.SEEDS:
        records = [
            helper.DatasetEvalRecord(
                row["context_sha256"],
                row["family_id"],
                "regression",
                dict(
                    R882=float(runtime["baseline"][i]),
                    **{arm: float(panels[seed][arm][i]) for arm in stat.ARMS},
                ),
            )
            for i, row in enumerate(runtime["val"])
        ]
        for anchor in ratings:
            ratings[anchor][str(seed)] = (
                helper.compute_family_grouped_bradley_terry_elo(
                    records,
                    anchor,
                    anchor_elo=1500.0,
                    tie_margin=1e-4,
                    prior=0.01,
                    n_iters=400,
                )["ratings"]
            )
    return {
        anchor: dict(
            by_seed=values,
            median_seed={
                name: float(np.median([values[str(seed)][name] for seed in stat.SEEDS]))
                for name in ["R882", *stat.ARMS]
            },
        )
        for anchor, values in ratings.items()
    }


def frozen_after(runtime):
    import torch

    runtime["h212"].verify_inputs()
    guard = runtime["h124"].source_module(
        "python/src/jope/north_star/dope_freeze_guard.py", "h214_freeze"
    )
    pin = guard.snapshot_dope_weights(runtime["net"]).composite_sha256
    if (
        pin != runtime["h212"].DOPE_SHA
        or sha(R882) != runtime["h212"].R882_SHA
        or torch.cuda.is_initialized()
    ):
        raise ValueError("DOPE/checkpoint/CPU freeze failed at completion")
    return dict(
        dope_sha256_after=pin,
        r882_sha256_after=sha(R882),
        optimizer_updates=0,
        CUDA_initialized=False,
    )


def worker(root, phase, source, smoke, smoke_sha=None):
    metrics = dict(
        schema="jope.h214.pool-diversity-metrics/v1",
        status="INCOMPLETE",
        source_commit=source,
        phase=phase,
        TEMP16="NOT_RUN: no native temperature hook; coordinator ruling",
        optimizer_updates=0,
        utility_readouts=0,
    )
    try:
        if phase == "run":
            if sha(smoke) != smoke_sha:
                raise ValueError("TRAIN smoke receipt hash differs")
            checked = json.loads(Path(smoke).read_text())
            if checked["status"] != "SMOKE_PASS" or checked["source_commit"] != source:
                raise ValueError("exact-head TRAIN smoke required")
            protocol.require_smoke(checked["h212_smoke"], source)
        runtime = setup(root)
        if phase == "smoke":
            h212 = runtime["h212"]
            row = runtime["train"][0]
            cid = row["context_sha256"]
            folds = json.loads(h212.FOLD_MANIFEST.read_text())["folds"]
            fold = next(f for f in folds if row["family_id"] in f["families"])
            excluded = set(fold["families"])
            bank = [r for r in runtime["champions"] if r["family_id"] not in excluded]
            champion = next(
                r for r in runtime["champions"] if r["context_sha256"] == cid
            )
            receipt = h212.end_to_end_smoke(
                root / "h212-smoke",
                row,
                fold["fold"],
                runtime["context"],
                runtime["projection"],
                runtime["pools"][cid],
                champion,
                bank,
                runtime["net"],
                runtime["h124"],
                runtime["generator"],
            )
            receipt["source_commit"] = source
            protocol.require_smoke(receipt, source)
            runtime["generator"].EVAL_SCRATCH_ROOT = root / "rust_eval"
            result = measure(runtime, row, stat.SEEDS[0], root)
            metrics.update(
                status="SMOKE_PASS",
                h212_smoke=receipt,
                scored_widths={
                    arm: len(result["arms"][arm]["ordered_program_sha256"])
                    for arm in stat.ARMS
                },
                freeze=frozen_after(runtime),
            )
        else:
            prior = json.loads(Path(smoke).read_text())
            if prior["status"] != "SMOKE_PASS" or prior["source_commit"] != source:
                raise ValueError("real exact-head TRAIN smoke required")
            protocol.require_smoke(prior["h212_smoke"], source)
            panels = {seed: {arm: [] for arm in stat.ARMS} for seed in stat.SEEDS}
            diagnostics = {arm: [] for arm in stat.ARMS}
            oracle = {arm: [] for arm in stat.ARMS}
            for seed in stat.SEEDS:
                for i, row in enumerate(runtime["val"]):
                    result = measure(runtime, row, seed, root)
                    for arm in stat.ARMS:
                        panels[seed][arm].append(result["arms"][arm]["utility"])
                        oracle[arm].append(result["arms"][arm]["oracle_utility"])
                        diagnostics[arm].append(
                            result["diversity"][arm]["mean_pairwise_distance"]
                        )
                    metrics["utility_readouts"] += 1
                    if i % 10 == 0:
                        print(
                            json.dumps(dict(seed=seed, contexts=i + 1, total=91)),
                            flush=True,
                        )
            gains = np.stack(
                [
                    np.asarray(panels[s]["FPS16"]) - np.asarray(panels[s]["NATIVE16"])
                    for s in stat.SEEDS
                ]
            )
            stats = stat.family_gain_stats(
                gains, [r["family_id"] for r in runtime["val"]]
            )
            decision = stat.decide({"FPS16": stats})
            np.savez_compressed(
                root / "utility-panels.npz",
                **{
                    str(s) + "_" + a: panels[s][a]
                    for s in stat.SEEDS
                    for a in stat.ARMS
                },
                R882=runtime["baseline"],
            )
            metrics.update(
                status="COMPLETE",
                **decision,
                FPS16_vs_NATIVE16=stats,
                seeds=list(stat.SEEDS),
                contexts=91,
                families=19,
                elo=elo(runtime, panels),
                diversity={
                    a: dict(
                        unique_programs=16,
                        mean_pairwise_distance=float(np.mean(diagnostics[a])),
                    )
                    for a in stat.ARMS
                },
                oracle_best_of_scored16={
                    a: float(np.mean(oracle[a])) for a in stat.ARMS
                },
                freeze=frozen_after(runtime),
            )
    except BaseException as error:
        metrics.update(error=str(error), traceback=traceback.format_exc())
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if observed != source:
        metrics.update(status="INCOMPLETE", error="source changed during execution")
    write(root / "metrics.json", metrics)
    return 0 if metrics["status"] in ("COMPLETE", "SMOKE_PASS") else 1


def processes(root_pid):
    table = {}
    for p in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = p.read_text()
            parts = text[text.rfind(")") + 2 :].split()
            table[int(p.parent.name)] = (
                int(parts[1]),
                int(parts[19]),
                int(parts[21]) * os.sysconf("SC_PAGE_SIZE"),
            )
        except (OSError, ValueError, IndexError):
            continue
    owned = {root_pid}
    while True:
        updated = owned | {
            pid for pid, (parent, _, _) in table.items() if parent in owned
        }
        if updated == owned:
            break
        owned = updated
    return {pid: table[pid] for pid in owned if pid in table}


def terminate_owned(child):
    owned = processes(child.pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, (_, start, _) in reversed(list(owned.items())):
            try:
                text = Path(f"/proc/{pid}/stat").read_text()
                parts = text[text.rfind(")") + 2 :].split()
                if int(parts[19]) == start:
                    os.kill(pid, sig)
            except (OSError, ValueError, IndexError):
                continue
        if sig == signal.SIGTERM:
            time.sleep(2)
    child.wait(timeout=10)


def supervise(root, phase, launch, smoke):
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if (
        source != launch["source_commit"]
        or sha(HERE / "PREREG.md") != launch["prereg_sha256"]
        or sha(HERE / "manifest.json") != launch["manifest_sha256"]
    ):
        raise ValueError("launch source/registration pin differs")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source, "origin/main"],
        cwd=REPO,
        check=True,
    )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, text=True
    ).strip():
        raise ValueError("clean merged checkout required")
    parent = DATA / "h214"
    expected = "preflight-" + source[:12] if phase == "smoke" else "attempt-001"
    if root != parent / expected or root.exists():
        raise ValueError("one fresh registered output root required")
    deadline = dt.datetime.fromisoformat(launch["deadline_utc"])
    started = dt.datetime.fromisoformat(launch["campaign_started_utc"])
    if (deadline - started).total_seconds() > 7200:
        raise ValueError("campaign wall cap exceeds2h")
    if phase == "run":
        if sha(smoke) != launch["smoke_sha256"]:
            raise ValueError("TRAIN smoke receipt hash differs")
        receipt = json.loads(Path(smoke).read_text())
        if receipt["status"] != "SMOKE_PASS" or receipt["source_commit"] != source:
            raise ValueError("exact-head TRAIN smoke absent")
        protocol.require_smoke(receipt["h212_smoke"], source)
    h212 = load_h212()
    usage = shutil.disk_usage("/")
    protocol.resource_guard(
        dict(
            host=os.uname().nodename.split(".")[0],
            nice=os.getpriority(os.PRIO_PROCESS, 0),
            disk_used_fraction=usage.used / (usage.used + usage.free),
            disk_free_bytes=usage.free,
            mem_available_bytes=h212.mem_available(),
            rss_bytes=0,
            load1=os.getloadavg()[0],
            load5=os.getloadavg()[1],
            cores=os.cpu_count(),
            cpu_pct=h212.host_cpu_pct(),
        ),
        time.monotonic(),
        None,
    )
    if dt.datetime.now(dt.timezone.utc) >= deadline:
        raise TimeoutError("campaign deadline before admission")
    root.mkdir(parents=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(REPO),
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    child_args = [
        sys.executable,
        "-B",
        str(HERE / "runner.py"),
        "--worker",
        "--phase",
        phase,
        "--root",
        str(root),
        "--launch",
        str(launch["_path"]),
    ]
    if smoke:
        child_args += ["--smoke", str(smoke)]
    h212 = load_h212()
    high_since = None
    previous = h212.cpu_ticks()
    last = time.monotonic()
    with (
        (root / "worker.stdout").open("x") as out,
        (root / "worker.stderr").open("x") as err,
    ):
        child = subprocess.Popen(
            child_args,
            cwd=REPO,
            env=env,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        write(
            root / "launch-receipt.json",
            dict(
                host="xbabe1",
                pid=child.pid,
                supervisor_pid=os.getpid(),
                source_commit=source,
                phase=phase,
                root=str(root),
                deadline_utc=launch["deadline_utc"],
                attempts_consumed=int(phase == "run"),
            ),
        )
        try:
            while child.poll() is None:
                time.sleep(0.5)
                now = time.monotonic()
                ticks = h212.cpu_ticks()
                total = ticks[0] - previous[0]
                idle = ticks[1] - previous[1]
                cpu = 100 * (1 - idle / total) if total else 0
                previous = ticks
                usage = shutil.disk_usage("/")
                observation = dict(
                    host=os.uname().nodename.split(".")[0],
                    nice=os.getpriority(os.PRIO_PROCESS, 0),
                    disk_used_fraction=usage.used / (usage.used + usage.free),
                    disk_free_bytes=usage.free,
                    mem_available_bytes=h212.mem_available(),
                    rss_bytes=sum(v[2] for v in processes(child.pid).values()),
                    load1=os.getloadavg()[0],
                    load5=os.getloadavg()[1],
                    cores=os.cpu_count(),
                    cpu_pct=cpu,
                )
                high_since = protocol.resource_guard(observation, now, high_since)
                if dt.datetime.now(dt.timezone.utc) >= deadline:
                    raise TimeoutError("shared2h campaign deadline")
                if now - last >= 30:
                    with (root / "resources.jsonl").open("a") as f:
                        f.write(json.dumps(observation) + "\n")
                    last = now
        except BaseException as error:
            terminate_owned(child)
            write(
                root / "metrics.json",
                dict(
                    schema="jope.h214.pool-diversity-metrics/v1",
                    status="INCOMPLETE",
                    source_commit=source,
                    phase=phase,
                    error=str(error),
                    TEMP16="NOT_RUN",
                    optimizer_updates=0,
                ),
            )
            return 1
    return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "run"], required=True)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    launch = json.loads(args.launch.read_text())
    launch["_path"] = str(args.launch)
    if os.uname().nodename.split(".")[0] != "xbabe1":
        raise ValueError("xbabe1 only")
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError("nice19 required")
    if args.worker:
        return worker(
            args.root,
            args.phase,
            launch["source_commit"],
            args.smoke,
            launch.get("smoke_sha256"),
        )
    return supervise(args.root, args.phase, launch, args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
