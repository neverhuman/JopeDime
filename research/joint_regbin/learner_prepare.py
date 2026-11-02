"""Materialize the admitted family-key roles into the JRB learner payload format.

All raw roles first rederive through current admission. Regression training and
validation retain their hash-pinned, TRAIN-only H50 cuts. Binary contexts use
the admitted real-S3 TRAIN caches. Panel top-ups reuse eligible declared TRAIN
sources and are evaluated by held-family models, as prescribed by the driver.

Freeze at most 300 rows and the first 64 columns before native evaluation. The
sampling is deterministic, label-stratified only for explicitly binary tasks.
Insufficient rows/classes are input exclusions, recorded before any scores.
Sixteen current-proposer programs and one fixed native ridge control share the
same two inner folds. Dataset RMSE is pooled from actual inner-fold predictions.
Binary probabilities are the native least-squares predictions clipped to
[1e-6,1-1e-6], a fixed link with no calibration fit on scored labels. AUC and
log-loss are computed on the concatenated out-of-fold predictions. These cached
metrics are inputs to task-separated Elo, not a learning-study verdict.

Native query padding contains ten TRAIN feature rows and zero targets; no test
or sealed payload is read. Only the pinned CPU S1 executable runs. No adapter
optimizer, GPU, qualification lock, bank write, or score-store append occurs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import torch

from . import admission, candidate_builder as cb, learner, protocol, runtime, worker

PRODUCER_SHA = protocol.file_sha(__file__)
H50 = Path("/home/ubuntu/jope-data/h50/input")
BINARY = Path("/home/ubuntu/jope-data/jrb-01/binary-s3-train-t2210")
NATIVE = Path(
    "/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod/binaries/bootstrap.s1main"
)
RIDGE_DSL = "(pipe X (standardize append=false) (ridge lambda_frac=0.001))"
RECIPE = dict(
    rows=300,
    columns=64,
    proposals=16,
    seed=20261029,
    binary_link="clip-native-posterior-mean-1e-6",
    ridge=RIDGE_DSL,
    evaluation=protocol.EVALUATION_PROTOCOL,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verified_bytes(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"source hash mismatch: {path}")
    return raw


def decode_train_table(raw):
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    magic, n, nq, p, folds = struct.unpack_from("<4sIIIB", raw)
    if (
        magic != b"JTT1"
        or nq != 0
        or folds != 2
        or len(raw) != 17 + n + 8 * n * (p + 1)
    ):
        raise ValueError("original regression cut is not a TRAIN-only two-fold JTT1")
    table = np.frombuffer(raw, dtype="<f8", offset=17 + n).reshape(n, p + 1)
    return table[:, :-1].copy(), table[:, -1].copy()


def select_input(x, y, task, context_id):
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("nonfinite or unaligned raw TRAIN table")
    if len(y) < 50 or x.shape[1] < 1:
        raise ValueError("input exclusion: native S1 requires at least50 TRAIN rows")
    if task == "binary":
        classes, counts = np.unique(y, return_counts=True)
        if set(classes.tolist()) != {0.0, 1.0} or min(counts) < 2:
            raise ValueError(
                "input exclusion: each explicit binary class needs two rows"
            )
    rng = np.random.default_rng(
        int(protocol.digest([RECIPE["seed"], context_id])[:16], 16)
    )
    if len(y) <= RECIPE["rows"]:
        rows = np.arange(len(y))
    elif task == "binary":
        # Retain at least two of each class, then fill by a fixed random order.
        required = np.concatenate(
            [rng.permutation(np.flatnonzero(y == c))[:2] for c in (0, 1)]
        )
        rest = np.setdiff1d(np.arange(len(y)), required)
        rows = np.sort(
            np.concatenate([required, rng.permutation(rest)[: RECIPE["rows"] - 4]])
        )
    else:
        rows = np.sort(rng.choice(len(y), RECIPE["rows"], replace=False))
    columns = np.arange(min(RECIPE["columns"], x.shape[1]))
    return (
        np.ascontiguousarray(x[rows][:, columns]),
        np.ascontiguousarray(y[rows]),
        rows,
        columns,
    )


def padded_native_input(encoded):
    raw = gzip.decompress(encoded)
    magic, n, nq, p, folds = struct.unpack_from("<4sIIIB", raw)
    if magic != b"JTT1" or nq != 0 or folds != 2:
        raise ValueError("TRAIN-only encoder contract differs")
    table = np.frombuffer(raw, dtype="<f8", offset=17 + n).reshape(n, p + 1)
    query = table[:10].copy()
    query[:, -1] = 0
    result = (
        b"JTT1" + struct.pack("<IIIB", n, 10, p, folds) + raw[17:] + query.tobytes()
    )
    return gzip.compress(result, mtime=0)


def dataset_metrics(document, task, expected_targets=None):
    folds = document.get("inner_cv")
    if (
        document.get("status") != "completed"
        or not isinstance(folds, list)
        or len(folds) != 2
    ):
        raise ValueError("native inner-CV result is unavailable")
    predictions, targets = [], []
    if {f["index"] for f in folds} != {0, 1}:
        raise ValueError("native fold inventory differs")
    for fold in sorted(folds, key=lambda x: x["index"]):
        pred = np.asarray(fold["predictions"], dtype=np.float64)
        target = np.asarray(fold["targets"], dtype=np.float64)
        if (
            pred.shape != target.shape
            or pred.ndim != 1
            or not len(pred)
            or not np.isfinite(pred).all()
        ):
            raise ValueError("native prediction/target inventory mismatch")
        if (
            not np.isfinite(target).all()
            or fold.get("serialized_prediction_replay_exact") is not True
        ):
            raise ValueError("native target/replay proof differs")
        if expected_targets is not None and (
            not np.array_equal(target, expected_targets[fold["index"]])
            or fold["score_rows"] != len(target)
            or fold["train_rows"] != len(expected_targets[1 - fold["index"]])
        ):
            raise ValueError("native fold targets differ from the frozen TRAIN cut")
        predictions.append(pred)
        targets.append(target)
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    if task == "regression":
        return dict(rmse=float(np.sqrt(np.mean((pred - target) ** 2))))
    if task != "binary" or set(np.unique(target).tolist()) != {0.0, 1.0}:
        raise ValueError("explicit binary task/labels required for binary metrics")
    probability = np.clip(pred, 1e-6, 1 - 1e-6)
    module = worker.binary_metrics_module()
    t = torch.as_tensor(target, dtype=torch.float32).reshape(1, -1)
    p = torch.as_tensor(probability, dtype=torch.float32).reshape(1, -1)
    auc = float(
        module.batched_roc_auc_gpu(p, t, torch.ones_like(t, dtype=torch.bool))[0]
    )
    loss = float(torch.nn.functional.binary_cross_entropy(p, t))
    if not math.isfinite(auc) or not math.isfinite(loss):
        raise ValueError("binary dataset metrics unavailable")
    return dict(auc=auc, log_loss=loss)


def _read_raw(row, settings):
    selected = settings["selected"]
    source_id = settings["panel_sources"].get(row["id"], row["id"])
    if source_id in selected:
        original = selected[source_id]
        if row["role"] == "train" and original["dl_split"] != "DL-Train":
            raise ValueError("nontraining original cut in training closure")
        if original["dataset_id"] != row["dataset"]:
            raise ValueError("original regression context dataset mismatch")
        if source_id == row["id"]:
            verified_bytes(H50 / "pools" / (source_id + ".npz"), row["payload_sha256"])
        raw = verified_bytes(
            H50 / "tables" / (source_id + ".jtt1"), original["train_only_jtt1_sha256"]
        )
        x, y = decode_train_table(raw)
        return (
            x,
            y,
            dict(
                kind="frozen_flywheel_train_only_cut",
                source_context_id=source_id,
                source_sha256=original["train_only_jtt1_sha256"],
            ),
        )
    if row["task"] == "binary":
        raw = verified_bytes(
            BINARY / "binary" / (row["dataset"] + ".npz"), row["payload_sha256"]
        )
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            if (
                str(z["dataset"].item()) != row["dataset"]
                or str(z["task"].item()) != "binary"
            ):
                raise ValueError("binary raw-cache identity differs")
            if str(z["catalog_sha256"].item()) != cb.CATALOG_V246_SHA256:
                raise ValueError("binary raw-cache catalog differs")
            return (
                z["x"].copy(),
                z["y"].copy(),
                dict(
                    kind="catalog_real_s3_train_cache",
                    source_sha256=row["payload_sha256"],
                ),
            )
    catalog = settings["catalog"][row["dataset"]]
    if (
        row["role"] != "panel"
        or catalog["task"] != "regression"
        or catalog["train_sha256"] != row["payload_sha256"]
    ):
        raise ValueError("unexpected raw regression source")
    import boto3

    client = boto3.Session(profile_name="veox-jepson-prod").client(
        "s3", region_name="us-east-2"
    )
    obj = client.get_object(
        Bucket="veox-jopedime-datasets-use2",
        Key=f"datasets/{row['dataset']}/manifest.json",
    )
    manifest_raw = obj["Body"].read()
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("id") != row["dataset"]
        or manifest.get("task") != "regression"
        or manifest["files"]["train"]["sha256"] != catalog["train_sha256"]
        or manifest["files"]["train"]["rows"] != catalog["train_rows"]
        or manifest["files"]["train"]["columns"] != catalog["cols"]
    ):
        raise ValueError("S3 TRAIN manifest drift")
    blob = (
        Path("/home/ubuntu/scratch/jopedime-corpus-benchmark-20260928/blobs")
        / catalog["train_sha256"]
    )
    if blob.is_file():
        raw = verified_bytes(blob, catalog["train_sha256"])
    else:
        raw = client.get_object(
            Bucket="veox-jopedime-datasets-use2",
            Key="blobs/sha256/" + catalog["train_sha256"],
        )["Body"].read()
        if hashlib.sha256(raw).hexdigest() != catalog["train_sha256"]:
            raise ValueError("S3 TRAIN body drift")
    index = manifest["target"]["index"]
    if index < 0:
        manifest = dict(
            manifest,
            target=dict(
                manifest["target"], index=manifest["files"]["train"]["columns"] + index
            ),
        )
    # Only this approved TRAIN CSV is decoded; never another native split.
    table = np.asarray(
        [[float(v) for v in r] for r in csv.reader(io.StringIO(raw.decode()))],
        dtype=np.float64,
    )
    index = manifest["target"]["index"]
    if table.shape != (catalog["train_rows"], catalog["cols"]):
        raise ValueError("S3 TRAIN table shape differs")
    return (
        np.delete(table, index, axis=1),
        table[:, index],
        dict(
            kind="catalog_real_s3_train_csv",
            source_sha256=hashlib.sha256(raw).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            manifest_version_id=obj.get("VersionId"),
        ),
    )


def _initialize(settings):
    global _SETTINGS, _PINNED, _CALLBACKS, _EVALUATOR
    _SETTINGS = settings
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    learner.verify_sources()
    cb._require(
        protocol.file_sha(NATIVE) == cb.NATIVE_S1_SHA256, "native S1 source drift"
    )
    _PINNED = runtime.initialize_r882_worker()
    _CALLBACKS = cb.flywheel_callbacks(
        Path(settings["cache"]) / f"landmarks-{os.getpid()}"
    )
    _EVALUATOR = sys.modules["jrb01_flywheel_evaluator"]


def _native(proposal, input_path, directory, index):
    remaining = _SETTINGS["deadline_unix"] - time.time()
    if remaining <= 0:
        raise TimeoutError("preparation deadline exhausted")
    folder = directory / f"native-{index:02d}"
    folder.mkdir()
    candidate = folder / "candidate.json"
    output = folder / "inner-cv.json"
    write_json(
        candidate,
        dict(
            dsl=proposal["dsl"],
            program_sha256=proposal["program_sha256"],
            birth_seq=0,
            parents=[],
            operator=proposal.get("operator", "fixed_ridge_control"),
        ),
    )
    env = dict(
        PATH="/usr/bin:/bin",
        HOME=str(folder),
        TMPDIR=str(folder),
        RAYON_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    with (folder / "stderr.txt").open("xb") as errors:
        result = subprocess.run(
            [str(NATIVE), "--fit-s1", str(input_path), str(candidate), str(output)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=errors,
            timeout=min(60, remaining),
        )
    if result.returncode or not output.is_file():
        raise ValueError(f"native candidate unavailable: exit{result.returncode}")
    raw = output.read_bytes()
    doc = json.loads(raw)
    if doc.get("program", {}).get("program_sha256") != proposal["program_sha256"]:
        raise ValueError("native program identity mismatch")
    return doc, hashlib.sha256(raw).hexdigest()


def _prepare(row):
    settings = _SETTINGS
    directory = Path(settings["cache"]) / row["id"]
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        cached = json.loads(receipt_path.read_text())
        if cached["producer_sha256"] != PRODUCER_SHA or cached["raw_context"] != row:
            raise ValueError("prepared context source drift; no implicit replacement")
        verified_bytes(
            protocol.safe_path(settings["packet"], cached["context"]["payload"]),
            cached["context"]["payload_sha256"],
        )
        return cached
    learner.resource_check(torch.device("cpu"))
    if time.time() > settings["deadline_unix"]:
        raise TimeoutError("preparation deadline exhausted")
    directory.mkdir(exist_ok=False)
    x, y, source = _read_raw(row, settings)
    x, y, rows, columns = select_input(x, y, row["task"], row["id"])
    encoded = cb.encode_two_fold_jtt1(
        x, y, task=row["task"], dataset=row["dataset"], seed=RECIPE["seed"]
    )
    decoded = gzip.decompress(encoded)
    fold_ids = np.frombuffer(decoded, dtype=np.uint8, count=len(y), offset=17)
    expected_targets = [y[fold_ids == index] for index in range(2)]
    padded = padded_native_input(encoded)
    native_path = directory / "train-only-native.jtt1"
    native_path.write_bytes(gzip.decompress(padded))
    landmark, propose, _unused, featurize = _CALLBACKS
    proposals = propose(
        [], landmark(encoded), [], row["family"], n_proposals=16, seed=RECIPE["seed"]
    )
    if len(proposals) != 16:
        raise ValueError("fixed sixteen-proposal count differs")
    program, projection, fitness, gates, native_metrics, native_hashes = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, proposal in enumerate(proposals):
        learner.resource_check(torch.device("cpu"))
        document, receipt_sha = _native(proposal, native_path, directory, index)
        score = _EVALUATOR.compute_s1_metrics_from_inner_cv(
            proposal["dsl"], document["inner_cv"]
        )
        if score["status"] != "completed":
            raise ValueError(
                "native S1 fitness unavailable; no zero/sentinel substitution"
            )
        f, p = featurize(proposal["dsl"])
        program.append(f)
        projection.append(p)
        fitness.append(score["fitness"])
        gates.append(score["cv_gate_passed"])
        native_metrics.append(dataset_metrics(document, row["task"], expected_targets))
        native_hashes.append(receipt_sha)
    ridge = dict(
        dsl=RIDGE_DSL, program_sha256=hashlib.sha256(RIDGE_DSL.encode()).hexdigest()
    )
    control, control_sha = _native(ridge, native_path, directory, 16)
    baseline = dataset_metrics(control, row["task"], expected_targets)
    metric = "rmse" if row["task"] == "regression" else "auc"
    evaluation = {
        metric: [m[metric] for m in native_metrics],
        "ridge_" + metric: baseline[metric],
    }
    if row["task"] == "binary":
        evaluation["log_loss"] = [m["log_loss"] for m in native_metrics]
    xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
    yt = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
    yt = (yt - yt.mean(1, keepdim=True)) / yt.var(
        1, unbiased=False, keepdim=True
    ).sqrt().clamp_min(1e-8)
    with torch.inference_mode():
        vec, col = _PINNED.net._extract_stage1a_landmarks_batched_gpu(
            xt, yt, torch.ones(1, len(y)), torch.ones(1, x.shape[1])
        )
        dope, lm = _PINNED.net._run_dope_mlc_v3_and_lm48(vec, col, use_landmarks=True)
        tokens, mask, _ = _PINNED.net._project_stage1b_tokens_batched_gpu(lm, dope)
    _PINNED.verify_frozen_state()
    identity = dict(
        context_id=row["id"],
        **{
            key: row[key]
            for key in (
                "dataset",
                "family",
                "task",
                "role",
                "source_split",
                "origin",
                "has_meta_test",
            )
        },
    )
    pool = dict(
        schema=cb.POOL_SCHEMA,
        status="CANDIDATE_TENSORS_NOT_ADMITTED",
        eligibility_verified=False,
        context=identity,
        task_target="s1_fitness"
        if row["task"] == "regression"
        else "s1_cv_gate_passed",
        proposal_source_sha256=cb.FLYWHEEL_EVALUATOR_SHA256,
        landmark_source_sha256=cb.FLYWHEEL_LANDMARK_SHA256,
        native_s1_sha256=cb.NATIVE_S1_SHA256,
    )
    target = np.asarray(
        fitness if row["task"] == "regression" else gates, dtype=np.float32
    )
    result_row = dict(row, payload=row["id"] + ".npz")
    path = protocol.safe_path(settings["packet"], result_row["payload"])
    with path.open("xb") as f:
        np.savez(
            f,
            pool_json=json.dumps(
                dict(
                    learner_payload_schema=learner.PAYLOAD_SCHEMA,
                    pool=pool,
                    evaluation=evaluation,
                )
            ),
            program=np.asarray(program, dtype=np.float32),
            proj16=np.asarray(projection, dtype=np.float64),
            target=target,
            tokens_10x32=tokens[0].numpy(),
            mask_10=mask[0].numpy(),
        )
    result_row["payload_sha256"] = protocol.file_sha(path)
    # Exercise the actual maintained packer and all score cases before admission.
    pools, token_map, mask_map, metrics = learner.load_payloads(
        [result_row], settings["packet"]
    )
    pack = cb.make_worker_packer(
        pools,
        token_map,
        mask_map,
        checkpoint_sha256=runtime.R882_CHECKPOINT_SHA256,
        dope_sha256=protocol.DOPE,
    )
    _inputs, actual_target = pack(result_row)
    for index in range(16):
        pred = torch.zeros_like(actual_target)
        pred[0, index] = -1 if row["task"] == "regression" else 1
        learner.score(
            result_row,
            pred,
            metrics[row["id"]],
            dict(round=5, checkpoint_sha256="0" * 64),
        )
    result = dict(
        raw_context=row,
        context=result_row,
        producer_sha256=PRODUCER_SHA,
        recipe=RECIPE,
        source=source,
        row_indices=rows.tolist(),
        column_indices=columns.tolist(),
        original_train_jtt1_sha256=hashlib.sha256(encoded).hexdigest(),
        native_input_sha256=hashlib.sha256(padded).hexdigest(),
        native_receipt_hashes=native_hashes,
        ridge_receipt_sha256=control_sha,
        dope_sha256=protocol.DOPE,
        adapter_optimizer_updates=0,
        evaluation_protocol=protocol.EVALUATION_PROTOCOL,
    )
    write_json(receipt_path, result)
    return result


def prepare(
    raw_manifest, family_manifest, packet, cache, *, budget, workers=8, limit=None
):
    raw = admission.rederive_role_manifest(raw_manifest)
    admission.verify_family_key_pool_manifest(
        family_manifest, admission.DEFAULT_ROLE_CATALOG
    )
    if any(
        raw["family_by_dataset"].get(r["dataset"])
        != family_manifest["family_by_dataset"].get(r["dataset"])
        for r in raw["contexts"]
    ):
        raise ValueError("role family map differs from current family-key proof")
    for p in (packet, cache):
        driver_path = Path(p).resolve()
        if driver_path.is_relative_to(Path("/tmp")) or any(
            (q / ".git").exists() for q in (driver_path, *driver_path.parents)
        ):
            raise ValueError("data preparation belongs outside checkout and /tmp")
        driver_path.mkdir(parents=True, exist_ok=True)
    budget = Path(budget)
    if not budget.exists():
        start = time.time()
        write_json(
            budget,
            dict(
                start_unix=start, deadline_unix=start + 7200, cpu_workers=8, gpu=False
            ),
        )
    reservation = json.loads(budget.read_text())
    if time.time() >= reservation["deadline_unix"]:
        raise TimeoutError("original cumulative preparation deadline exhausted")
    pins = json.loads(runtime.H144_MANIFEST.read_text())["inputs"]
    verified_bytes(
        H50 / "source_manifest.json", pins[str(H50 / "source_manifest.json")]
    )
    selected = json.loads(
        verified_bytes(H50 / "selected.json", pins[str(H50 / "selected.json")])
    )
    original = {r["context_sha256"]: r for r in selected["train"] + selected["dev"]}
    catalog = json.loads(admission.DEFAULT_ROLE_CATALOG.read_text())
    rows = list(raw["contexts"])
    panel_sources = {}
    # Input exclusions use only published raw shape/class metadata, before fits.
    excluded = []
    for row in list(rows):
        if row["task"] != "binary":
            continue
        b = verified_bytes(
            BINARY / "binary" / (row["dataset"] + ".npz"), row["payload_sha256"]
        )
        with np.load(io.BytesIO(b), allow_pickle=False) as z:
            values, counts = np.unique(z["y"], return_counts=True)
            reason = (
                "fewer_than50_real_rows"
                if len(z["y"]) < 50
                else "binary_class_has_fewer_than2_rows"
                if len(values) != 2 or min(counts) < 2
                else None
            )
        if reason:
            excluded.append(dict(context=row, reason=reason))
            rows.remove(row)
    # Top up missing panels from eligible existing TRAIN sources. This is frozen
    # before any metric is computed; held-family routing remains the driver's.
    for task in protocol.TASKS:
        panel_ids = {
            r["dataset"] for r in rows if r["task"] == task and r["role"] == "panel"
        }
        sources = {}
        for row in sorted(rows, key=lambda x: x["id"]):
            if (
                row["task"] == task
                and row["role"] == "train"
                and row["dataset"] not in panel_ids
            ):
                sources.setdefault(row["dataset"], row)
        for dataset in sorted(sources)[: 240 - len(panel_ids)]:
            source = sources[dataset]
            row = dict(
                source,
                id=protocol.digest(
                    ["family-key-panel-topup", task, dataset, source["id"]]
                ),
                role="panel",
            )
            if task == "regression":
                panel_sources[row["id"]] = source["id"]
            rows.append(row)
            panel_ids.add(dataset)
        if len(panel_ids) != 240:
            raise ValueError("insufficient eligible panel top-ups")
    write_json(
        Path(cache) / "frozen-source-cohort.json",
        dict(
            raw_manifest_sha256=protocol.digest(raw),
            recipe=RECIPE,
            contexts=rows,
            exclusions=excluded,
            panel_sources=panel_sources,
            producer_sha256=PRODUCER_SHA,
        ),
    )
    chosen = rows
    if limit is not None:
        chosen = [
            next(r for r in rows if r["task"] == t and r["role"] == "train")
            for t in protocol.TASKS
        ][:limit]
    settings = dict(
        selected=original,
        panel_sources=panel_sources,
        catalog={r["id"]: r for r in catalog["datasets"]},
        packet=str(packet),
        cache=str(cache),
        deadline_unix=reservation["deadline_unix"],
    )
    results, failures = [], []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(settings,),
    ) as pool:
        futures = {pool.submit(_prepare, row): row for row in chosen}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        dict(
                            prepared=len(results),
                            total=len(chosen),
                            id=result["context"]["id"],
                        )
                    ),
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    dict(
                        context=futures[future],
                        error=type(error).__name__ + ":" + str(error),
                    )
                )
                print(json.dumps(dict(unavailable=failures[-1])), flush=True)
    summary = dict(
        status="COMPLETE"
        if not failures and limit is None
        else "PILOT_COMPLETE"
        if not failures
        else "INCOMPLETE",
        producer_sha256=PRODUCER_SHA,
        recipe=RECIPE,
        prepared=len(results),
        expected=len(chosen),
        exclusions=excluded,
        failures=failures,
        contexts=[r["context"] for r in results],
        receipts=results,
        source_manifest=raw,
        adapter_optimizer_updates=0,
        gpu=False,
    )
    write_json(Path(cache) / "preparation-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--family-manifest", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("one to eight CPU workers required")
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    summary = prepare(
        json.loads(args.raw_manifest.read_text()),
        json.loads(args.family_manifest.read_text()),
        args.packet,
        args.cache,
        budget=args.budget,
        workers=args.workers,
        limit=2 if args.pilot else None,
    )
    return 1 if summary["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
