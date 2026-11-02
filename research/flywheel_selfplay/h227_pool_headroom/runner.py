"""H227 frozen candidate-generation headroom; no optimizer exists in this worker."""

from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np

import backend as b

stat = b.module(b.PREFIX + "analysis.py", "h227_analysis")
protocol = b.module(b.PREFIX + "protocol.py", "h227_protocol")


def array_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def append(path, value):
    import os

    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False, default=array_json) + "\n")
        f.flush()
        os.fsync(f.fileno())


def parent_bank(champions, row, folds=None):
    excluded = {row["family_id"]}
    if row["dl_split"] == "DL-Train" and folds is not None:
        excluded = next(
            set(f["families"]) for f in folds if row["family_id"] in f["families"]
        )
    bank = [p for p in champions if p["family_id"] not in excluded]
    if not bank or any(p["dl_split"] != "DL-Train" for p in bank):
        raise ValueError("nonempty family-disjoint TRAIN retrieval bank required")
    return bank


def measure(runtime, row, seed, root, folds=None):
    if (
        row.get("dl_split") not in ["DL-Train", "DL-Val"]
        or row.get("task") != "regression"
        or row.get("has_meta_test") is True
        or (runtime.get("phase") == "smoke" and row["dl_split"] != "DL-Train")
    ):
        raise ValueError(
            "approved regression split only; treatment smoke is TRAIN-only"
        )
    cid = row["context_sha256"]
    generator = runtime["generator"]
    h212 = runtime["h212"]
    bank = parent_bank(runtime["champions"], row, folds)
    landmark = runtime["context"][cid]["lm48_z"]
    parent = min(
        bank,
        key=lambda p: (
            float(np.sum((np.asarray(p["landmark_summary"]) - landmark[:12]) ** 2)),
            p["program_sha256"],
        ),
    )
    args = dict(
        parent_programs=[parent],
        landmark_v2_48=landmark,
        champion_bank=bank,
        target_family_id=row["family_id"],
        seed=stat.stream_seed(seed, cid),
    )
    pools = stat.build_pools(
        parent, generator.propose_flywheel_candidates, args, generator.featurize_dsl_412
    )
    locked = dict(
        seed=seed,
        context_sha256=cid,
        family_id=row["family_id"],
        arms={a: [p["program_sha256"] for p in pools[a]] for a in stat.ARMS},
    )
    append(
        root / "pool-locks.jsonl", locked
    )  # Durable outcome-free selection before native evaluator/score calls.
    union = list({p["program_sha256"]: p for a in stat.ARMS for p in pools[a]}.values())
    scratch = b.ensure_scratch(root / "scratch" / f"{seed}-{cid}")
    generator.EVAL_SCRATCH_ROOT = b.ensure_scratch(scratch / "rust_eval")
    raw = (h212.INPUT / "tables" / f"{cid}.jtt1").read_bytes()
    envelope = runtime["envelope"].training_cv_envelope(raw)
    b.validate_envelope(
        raw,
        envelope,
        row["train_only_jtt1_sha256"],
        runtime["envelope"].training_cv_envelope,
    )
    table = scratch / (cid + ".jtt1.gz")
    table.write_bytes(gzip.compress(envelope, mtime=0))
    started = time.monotonic()
    results = generator.evaluate_gp_batch_rust(
        str(table), union, run_outer_test=False, max_workers=4, timeout_s=25.0
    )
    ids = [p["program_sha256"] for p in union]
    if (
        len(results) != len(union)
        or [r.get("program_sha256") for r in results] != ids
        or any(r.get("status") not in ["completed", "failed"] for r in results)
    ):
        raise ValueError("complete ordered native evaluation required")
    if not any(r["status"] == "completed" for r in results):
        append(
            root / "native-failure-panel.jsonl",
            dict(seed=seed, context_sha256=cid, results=results),
        )
        raise ValueError("no completed native candidate; operational panel unqualified")
    if any(r["status"] == "failed" and r["cv_gate_passed"] for r in results):
        raise ValueError("failed native result cannot pass gate")
    result_by_id = dict(zip(ids, results, strict=True))
    panels = {}
    costs = {}
    for arm in stat.ARMS:
        # Score each registered pool separately, with four revealed anchors used only as conditioning.
        prediction = h212.predict_dynamic(
            runtime["net"],
            runtime["h124"],
            cid,
            runtime["context"][cid],
            runtime["pools"][cid],
            runtime["projection"],
            pools[arm],
            generator,
        )
        if len(prediction) != 4 + len(pools[arm]):
            raise ValueError("dynamic prediction width differs")
        costs[arm] = prediction[4:].tolist()
        panels[arm] = stat.score(
            prediction[4:], [result_by_id[p["program_sha256"]] for p in pools[arm]]
        )
    record = dict(
        **locked,
        parent_sha256=parent["program_sha256"],
        programs=union,
        native_results=results,
        predicted_cost=costs,
        scored=panels,
        native_evaluated_union=len(union),
        logical_evaluation_counts={a: len(pools[a]) for a in stat.ARMS},
        cv_envelope_sha256=hashlib.sha256(envelope).hexdigest(),
        raw_train_only_jtt1_sha256=row["train_only_jtt1_sha256"],
        evaluation_seconds=time.monotonic() - started,
        oracle_role="diagnostic only; not used for selection",
    )
    append(root / "panels.jsonl", record)
    if scratch.parent != root / "scratch" or scratch.is_symlink():
        raise ValueError("unexpected scratch path")
    shutil.rmtree(
        scratch
    )  # Own generated scratch only, after all returned results and analysis row are durable.
    return record


def elo(runtime, panels):
    helper = runtime["h124"].source_module(
        "python/src/jope/north_star/elo_tracker.py", "h227_elo"
    )
    ratings = {}
    for seed in stat.SEEDS:
        rows = [
            helper.DatasetEvalRecord(
                r["context_sha256"],
                r["family_id"],
                "regression",
                dict(
                    R882=float(runtime["baseline"][i]),
                    **{a: float(panels[seed][a][i]) for a in stat.ARMS},
                ),
            )
            for i, r in enumerate(runtime["val"])
        ]
        ratings[str(seed)] = helper.compute_family_grouped_bradley_terry_elo(
            rows, "R882", anchor_elo=1500.0, tie_margin=0.0, prior=0.01, n_iters=400
        )["ratings"]
    return dict(
        by_seed=ratings,
        median_seed={
            a: float(np.median([ratings[str(s)][a] for s in stat.SEEDS]))
            for a in ["R882", *stat.ARMS]
        },
        diagnostic=True,
    )


def run(root, phase, smoke=None):
    root = Path(root)
    doc = b.registration()
    if phase == "smoke":
        import datetime as dt

        protocol.DEADLINE = min(
            protocol.DEADLINE,
            dt.datetime.fromisoformat("2026-11-02T17:45:00+00:00").timestamp(),
        )
    metrics = dict(
        schema="jope.h227.pool-headroom-metrics/v1",
        status="INCOMPLETE",
        verdict="INCOMPLETE",
        source_commit=b.SOURCE,
        phase=phase,
        optimizer_updates=0,
        completed_context_seed_units=0,
        statistics=None,
        elo=None,
        dope_after=None,
    )
    b.write(root / "metrics.json", metrics)
    try:
        if phase == "run":
            passed = json.loads(Path(smoke).read_text())
            if (
                passed["status"] != "SMOKE_PASS"
                or passed["source_commit"] != b.SOURCE
                or passed["full_path"] is not True
            ):
                raise ValueError("exact-source full-path TRAIN smoke required")
        runtime = b.setup(root, phase)
        metrics["freeze_before"] = b.freeze(runtime)
        folds = json.loads(b.blob(b.PREFIX + "fold_manifest.json"))
        if phase == "smoke":
            row = runtime["train"][0]
            protocol.wait(root)
            result = measure(
                runtime, row, stat.SEEDS[0], root, folds["train"][str(stat.SEEDS[0])]
            )
            if set(result["scored"]) != set(stat.ARMS) or result[
                "logical_evaluation_counts"
            ] != dict(NATIVE16=16, NATIVE64=64, FPS16of64=16):
                raise ValueError("smoke omitted a treatment or analysis path")
            metrics.update(
                status="SMOKE_PASS",
                verdict=None,
                full_path=True,
                smoke_split="DL-Train",
                completed_context_seed_units=1,
                scored_arms=list(stat.ARMS),
                scratch_deleted=not any((root / "scratch").iterdir()),
                freeze_after=b.freeze(runtime),
            )
        else:
            panels = {s: {a: [] for a in stat.ARMS} for s in stat.SEEDS}
            oracle = {s: {a: [] for a in stat.ARMS} for s in stat.SEEDS}
            # Every complete unit is retained; no reuse/retry of a partial primary attempt.
            for seed in stat.SEEDS:
                for row in runtime["val"]:
                    protocol.wait(root)
                    result = measure(runtime, row, seed, root)
                    for arm in stat.ARMS:
                        panels[seed][arm].append(result["scored"][arm]["utility"])
                        oracle[seed][arm].append(
                            result["scored"][arm]["oracle_utility"]
                        )
                    metrics["completed_context_seed_units"] += 1
                    b.write(root / "metrics.json", metrics)
                    print(
                        "H227 completed unit",
                        metrics["completed_context_seed_units"],
                        "/273",
                        flush=True,
                    )
            stats = {
                a: stat.gain_stats(
                    np.asarray(
                        [
                            np.asarray(panels[s][a]) - np.asarray(panels[s]["NATIVE16"])
                            for s in stat.SEEDS
                        ]
                    ),
                    [r["family_id"] for r in runtime["val"]],
                )
                for a in stat.CONTRASTS
            }
            arrays = {f"{s}_{a}": panels[s][a] for s in stat.SEEDS for a in stat.ARMS}
            arrays.update(
                {f"oracle_{s}_{a}": oracle[s][a] for s in stat.SEEDS for a in stat.ARMS}
            )
            arrays["R882"] = runtime["baseline"]
            np.savez_compressed(root / "utility-panels.npz", **arrays)
            metrics.update(
                status="COMPLETE",
                **stat.decide(stats),
                statistics=stats,
                elo=elo(runtime, panels),
                oracle_diagnostic={
                    a: dict(
                        context_mean_by_seed=[
                            float(np.mean(oracle[s][a])) for s in stat.SEEDS
                        ],
                        selected_gap_by_seed=[
                            float(
                                np.mean(
                                    np.asarray(oracle[s][a]) - np.asarray(panels[s][a])
                                )
                            )
                            for s in stat.SEEDS
                        ],
                    )
                    for a in stat.ARMS
                },
                freeze_after=b.freeze(runtime),
                utility_panel_sha256=b.digest(root / "utility-panels.npz"),
                seeds=list(stat.SEEDS),
                contexts=91,
                families=19,
                compute_deadline_utc=doc["resources"]["compute_deadline_utc"],
            )
    except BaseException as exc:
        metrics.update(
            status="INCOMPLETE",
            verdict="INCOMPLETE",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    if metrics.get("freeze_after") is not None:
        metrics["dope_after"] = metrics["freeze_after"]["dope_sha256"]
    b.write(root / "metrics.json", metrics)
    return 0 if metrics["status"] in ("SMOKE_PASS", "COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(
        run(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    )
