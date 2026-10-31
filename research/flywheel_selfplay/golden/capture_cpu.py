#!/usr/bin/env python3
"""Capture the bounded RES-07 training-only CPU oracle; never CUDA qualification."""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import tempfile

import numpy as np
import torch

if __package__:
    from . import generate
else:
    import generate


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(generate.canonical(value) + b"\n")


def checked(path, expected):
    data = generate.regular(Path(path))
    generate.require(digest(data) == expected, "input identity differs: " + str(path))
    return data


def validate_plan(plan):
    """Reject mixed roles before following any training-table path."""
    contexts = plan["contexts"]
    generate.require(
        len(contexts) == 50, "exactly 50 pinned training contexts required"
    )
    generate.require(
        len({c["context_sha256"] for c in contexts}) == 50, "duplicate context"
    )
    selected = generate.read_json(checked(plan["selected"], plan["selected_sha256"]))
    train = {r["context_sha256"]: r for r in selected["train"]}
    for row in contexts:
        original = train.get(row["context_sha256"])
        generate.require(
            original is not None
            and row["dl_split"] == "DL-Train"
            and row["source_split"] in ("train", "inner"),
            "nontraining context refused",
        )
        for field in (
            "family_id",
            "dataset_id",
            "source_split",
            "train_only_jtt1_sha256",
        ):
            generate.require(
                row[field] == original[field], "training identity differs: " + field
            )
        generate.require(
            0 < row["rows"] <= 1024 and row["columns"] > 0,
            "complete bounded training table required",
        )
        generate.require(
            len(row["program_sha256"]) == 20 and len(set(row["program_sha256"])) == 20,
            "twenty distinct prior programs required",
        )
    budget = plan["budget"]
    generate.require(
        0 < budget["after_lock_wall_seconds"] <= 1799
        and budget["cpu_affinity_slots"] <= 4
        and budget["math_threads"] == 1
        and budget["max_native_replays"] <= 1000
        and budget["max_native_fold_fits"] <= 2000
        and budget["max_output_bytes"] <= 2**31,
        "original resource ceiling exceeded",
    )
    return contexts


def source_definition(reference, suffix, constants=False):
    paths = [p for p in reference.pins if p.endswith("/" + suffix)]
    generate.require(len(paths) == 1, "ambiguous source: " + suffix)
    original = paths[0]
    raw = checked(
        reference.custody / "files" / original.lstrip("/"),
        reference.pins[original]["sha256"],
    )
    # Import only pinned definitions/constants; never execute a historical runner.
    nodes = [
        n
        for n in ast.parse(raw).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        or (constants and isinstance(n, (ast.Assign, ast.AnnAssign)))
    ]
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), original, "exec"), reference.env)
    reference.source_hashes[original] = digest(raw)
    return digest(raw)


def native_training_transport(raw):
    """Old S1 validates an ignored query tail; supply only copied training rows.

    The pinned parser decodes exactly nt rows. The ten extra slots contain
    unchanged copies of original training rows0..9, never outer/heldout data.
    Training bytes, labels and fold assignments are untouched.
    """
    generate.require(raw[:4] == b"JTT1" and len(raw) >= 17, "invalid training header")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    stride = (width + 1) * 8
    generate.require(
        nq == 0
        and raw[16] == 2
        and nt >= 50
        and 0 < width <= 512
        and len(raw) == 17 + nt + nt * stride,
        "complete two-fold training-only source required",
    )
    encoded = bytearray(raw)
    struct.pack_into("<I", encoded, 8, 10)
    encoded.extend(raw[17 + nt : 17 + nt + 10 * stride])
    generate.require(
        encoded[17:] == raw[17:] + raw[17 + nt : 17 + nt + 10 * stride],
        "training prefix changed",
    )
    return bytes(encoded)


def capture(plan, output):
    contexts = validate_plan(plan)
    generate.require(
        not output.exists() and not output.is_symlink(), "new external output required"
    )
    generate.require(
        not torch.cuda.is_initialized()
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "CPU route requires hidden CUDA and no initialized CUDA",
    )
    generate.require(
        os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 or lower priority required"
    )
    cpus = sorted(os.sched_getaffinity(0))
    generate.require(
        os.getloadavg()[0] + 4 <= 0.85 * len(cpus), "host CPU admission failed"
    )
    os.sched_setaffinity(0, set(cpus[:4]))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for path, expected in plan["restored_source_sha256"].items():
        checked(path, expected)
    generate.CUSTODY = Path(plan["custody"])
    manifest, custody, reference, validation = generate.validate_bundle(
        Path(plan["bundle"]), plan["bundle_digest"]
    )
    generate.require(
        plan["selected_sha256"] == custody["training_corpus"]["selected_sha256"],
        "frozen training roster differs",
    )
    generate.require(
        all(
            c["family_id"] in custody["training_corpus"]["train_family_ids"]
            for c in contexts
        ),
        "family outside training closure",
    )
    generate.require(
        all(v.device.type == "cpu" for v in reference.state.values()),
        "CPU weights required",
    )
    generate.require(
        validation["frozen_dope_composite"].startswith("c51a56d0"),
        "frozen DOPE differs",
    )
    source_definition(
        reference, "JopeDimeMicro/python/research/levels/descriptors.py", constants=True
    )
    source_definition(
        reference,
        "JopeDimeMicro/python/research/levels/building_block_landmarks.py",
        constants=True,
    )
    native = next(
        r for r in custody["assets"] if r["path"].endswith("/bin/libjope_hpc.so")
    )
    library = reference.custody / "files" / native["path"].lstrip("/")
    checked(library, native["sha256"])
    reference.env.update(ctypes=ctypes, Path=Path, LIB_PATH=library)
    workspace_sha = source_definition(
        reference, "JopeDimeMicro/python/hpc/landmark_workspace.py"
    )
    source_definition(
        reference, "frozen-source/research/flywheel_selfplay/device_pool_cache.py"
    )
    reference.define(
        "JopeDimeMicro/python/rust_evaluator.py", ["compute_s1_metrics_from_inner_cv"]
    )
    workspace = reference.env["FusedLandmarkWorkspace"](lib_path=library, num_threads=4)
    cast = reference.env["landmark_float32"]
    metrics = reference.env["compute_s1_metrics_from_inner_cv"]
    worker = custody["evaluator"]["binary"]
    executable = reference.custody / "files" / worker["path"].lstrip("/")
    checked(executable, worker["sha256"])
    bank = [
        generate.read_json(r)
        for r in generate.decompress(
            (Path(plan["bundle"]) / "bank.jsonl.zst").read_bytes()
        ).splitlines()
    ]
    programs = generate.model_prior_programs(bank)
    order = [r["program_sha256"] for r in programs[:20]]
    dsls = {r["program_sha256"]: r["dsl"] for r in programs}
    generate.require(
        all(c["program_sha256"] == order for c in contexts),
        "frozen prior order differs",
    )
    caller = next(
        v
        for p, v in reference.source_hashes.items()
        if p.endswith("/run_fused_closed_loop.py")
    )
    output.mkdir(parents=True)
    captures = output / "captures"
    captures.mkdir()
    selected = checked(plan["selected"], plan["selected_sha256"])
    (captures / "selected.json").write_bytes(selected)
    packet = dict(
        schema="jope.golden-source-captures/v1",
        bundle_digest=plan["bundle_digest"],
        selected=dict(path="selected.json", sha256=digest(selected)),
        contexts=[],
    )
    calls = folds = 0
    started = time.monotonic()

    def save(name, data):
        path = captures / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return dict(path=name, sha256=digest(data))

    for context in contexts:
        generate.require(
            os.getloadavg()[0] <= 0.85 * len(cpus), "host CPU limit exceeded"
        )
        cid = context["context_sha256"]
        raw = checked(context["table_path"], context["train_only_jtt1_sha256"])
        generate.require(raw[:4] == b"JTT1", "training table header differs")
        n, query, width = struct.unpack_from("<III", raw, 4)
        generate.require(
            query == 0 and n == context["rows"] and width == context["columns"],
            "query rows, truncated table or shape mismatch refused",
        )
        values = np.frombuffer(raw, dtype="<f8", offset=17 + n).reshape(n, width + 1)
        x, y = values[:, :-1], values[:, -1]
        generate.require(bool(np.isfinite(y).all()), "nonfinite training targets")
        yn = np.asarray((y - y.mean()) / max(float(y.std()), 1e-8), dtype=np.float32)
        xn = cast(x)
        v, full, _ = workspace.fit_transform_and_extract_bundle_high(
            xn, np.ascontiguousarray(yn)
        )
        v2, full2, _ = workspace.fit_transform_and_extract_bundle_high(
            xn, np.ascontiguousarray(yn)
        )
        generate.require(
            v.tobytes() == v2.tobytes() and full.tobytes() == full2.tobytes(),
            "native regeneration differs",
        )
        col = np.zeros((64, 17), dtype=np.float32)
        col[: min(64, width)] = full[:64, :17]
        tensors = {
            "input.X": torch.from_numpy(xn.copy()),
            "input.y": torch.from_numpy(yn.copy()),
            "input.row_ids": torch.arange(n, dtype=torch.int64),
            "v256": torch.from_numpy(v.copy()),
            "col_full": torch.from_numpy(full.copy()),
            "col": torch.from_numpy(col),
        }
        # Missing raw features may be NaN; computed outputs must remain finite.
        header, chunks, offset = {}, [], 0
        for name, tensor in sorted(tensors.items()):
            generate.require(
                name == "input.X" or bool(torch.isfinite(tensor).all()),
                "nonfinite native result",
            )
            data = (
                tensor.contiguous()
                .numpy()
                .astype("<i8" if tensor.dtype == torch.int64 else "<f4")
                .tobytes()
            )
            header[name] = dict(
                dtype="I64" if tensor.dtype == torch.int64 else "F32",
                shape=list(tensor.shape),
                data_offsets=[offset, offset + len(data)],
            )
            chunks.append(data)
            offset += len(data)
        encoded = generate.canonical(header)
        encoded += b" " * (-len(encoded) % 8)
        landmark = save(
            "landmarks/" + cid + ".safetensors",
            struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks),
        )
        landmark.update(
            reference_source_sha256=workspace_sha,
            native_library_sha256=native["sha256"],
            caller_source_sha256=caller,
            outer_query_rows=0,
        )
        measured = {}
        scratch = output / "native-evaluator" / cid
        scratch.mkdir(parents=True)
        native_input = scratch / "training-transport.jtt1"
        transport = native_training_transport(raw)
        native_input.write_bytes(transport)
        write_json(
            scratch / "training-transport.json",
            dict(
                original_training_sha256=digest(raw),
                transport_sha256=digest(transport),
                unchanged_training_prefix=True,
                ignored_transport_source_row_ids=list(range(10)),
                outer_query_rows=0,
                parser_source_sha256="e369e13bfb498dee924cd0afca23a82fb43c6997864e37fdc28063f092cc1fb1",
            ),
        )
        for key in order[:4]:
            calls += 1
            ledger_path = Path(plan["budget_ledger"])
            ledger = generate.read_json(ledger_path.read_bytes())
            ledger["native_calls_reserved"] += 1
            ledger["native_folds_reserved"] += 2
            generate.require(
                ledger["native_calls_reserved"] <= plan["budget"]["max_native_replays"]
                and ledger["native_folds_reserved"]
                <= plan["budget"]["max_native_fold_fits"],
                "original cumulative native budget exceeded",
            )
            write_json(ledger_path, ledger)
            candidate, result = (
                scratch / (key + ".candidate.json"),
                scratch / (key + ".result.json"),
            )
            write_json(
                candidate,
                dict(
                    dsl=dsls[key],
                    program_sha256=key,
                    birth_seq=0,
                    parents=[],
                    operator="edit_zero",
                ),
            )
            command = [
                str(executable),
                "--fit-s1",
                str(native_input),
                str(candidate),
                str(result),
            ]
            env = dict(
                PATH="/usr/bin:/bin",
                RAYON_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                TMPDIR=str(scratch),
            )
            before = time.monotonic()
            run = subprocess.run(
                command,
                capture_output=True,
                env=env,
                timeout=plan["budget"]["native_call_timeout_seconds"],
            )
            write_json(
                scratch / (key + ".receipt.json"),
                dict(
                    command=command,
                    exit=run.returncode,
                    seconds=time.monotonic() - before,
                    input_sha256=context["train_only_jtt1_sha256"],
                    transport_sha256=digest(transport),
                    evaluator_sha256=worker["sha256"],
                    stdout=run.stdout.decode(errors="replace")[-8192:],
                    stderr=run.stderr.decode(errors="replace")[-8192:],
                ),
            )
            generate.require(
                run.returncode == 0 and result.is_file(), "native evaluator failed"
            )
            doc = generate.read_json(result.read_bytes())
            inner = doc.get("inner_cv") or []
            generate.require(len(inner) == 2, "frozen two-fold recipe required")
            folds += len(inner)
            generate.require(
                folds <= plan["budget"]["max_native_fold_fits"],
                "native fold budget exceeded",
            )
            outcome = metrics(dsls[key], inner)
            generate.require(
                outcome["status"] == "completed",
                "unknown or failed outcome is not a fixture",
            )
            measured[key] = dict(
                raw_f=outcome["fitness"],
                gate=int(outcome["cv_gate_passed"]),
                status=outcome["status"],
                inner_cv=inner,
                result_sha256=digest(result.read_bytes()),
            )
        surrogate = save(
            "surrogate/" + cid + ".json",
            generate.canonical(
                dict(
                    evaluator_sha256=worker["sha256"],
                    context_sha256=cid,
                    recipe=manifest["surrogate_recipe"],
                    measurements=measured,
                )
            )
            + b"\n",
        )
        packet["contexts"].append(
            dict(
                context_sha256=cid,
                family_id=context["family_id"],
                dl_split="DL-Train",
                landmarks=landmark,
                surrogate=surrogate,
                slates={
                    str(size): dict(
                        program_sha256=order[:size],
                        revealed_raw_f=[measured[k]["raw_f"] for k in order[:4]],
                        revealed_gate=[measured[k]["gate"] for k in order[:4]],
                    )
                    for size in (12, 20)
                },
            )
        )
        print(
            json.dumps(
                dict(
                    contexts=len(packet["contexts"]),
                    native_calls=calls,
                    fold_fits=folds,
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
    write_json(captures / "captures.json", packet)
    pin = digest((captures / "captures.json").read_bytes())
    indexes = []
    for name in ("generation-1", "generation-2"):
        indexes.append(
            generate._generate_locked(
                Path(plan["bundle"]),
                plan["bundle_digest"],
                captures / "captures.json",
                pin,
                output / name,
            )
        )
    generate.require(indexes[0] == indexes[1], "regenerated fixture inventories differ")
    for name, spec in indexes[0]["files"].items():
        generate.require(
            checked(output / "generation-1" / name, spec["sha256"])
            == checked(output / "generation-2" / name, spec["sha256"]),
            "regeneration bytes differ",
        )
    generate.require(not torch.cuda.is_initialized(), "CPU route initialized CUDA")
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    generate.require(
        total <= plan["budget"]["max_output_bytes"], "output budget exceeded"
    )
    report = dict(
        schema="jope.cpu-golden-capture/v1",
        device="cpu",
        contexts=50,
        slates=100,
        native_calls=calls,
        fold_fits=folds,
        capture_manifest_sha256=pin,
        fixture_index_sha256=digest((output / "generation-1/index.json").read_bytes()),
        fixture_bytes=indexes[0]["total_bytes"],
        total_output_bytes=total,
        byte_identical_regeneration=True,
        cuda_initialized=False,
        qualification=dict(
            RNK06_M2="pending Rust replay",
            RNK08_M2="pending Rust replay",
            CUDA=False,
            Q1=False,
            M3=False,
            M5=False,
        ),
    )
    write_json(output / "cpu-capture-receipt.json", report)
    return report


def regenerate(plan, captures, capture_pin, index_pin, output):
    """Repackage saved TRAIN captures on CPU; never rerun the native evaluator."""
    generate.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CPU regeneration requires hidden CUDA and no initialized CUDA",
    )
    generate.require(
        os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 or lower priority required"
    )
    generate.require(
        len(index_pin) == 64 and all(c in "0123456789abcdef" for c in index_pin),
        "explicit published fixture index SHA256 required",
    )
    packet = generate.read_json(checked(captures, capture_pin))
    expected = {row["context_sha256"]: row["family_id"] for row in plan["contexts"]}
    generate.require(
        len(packet["contexts"]) == len(expected)
        and {row["context_sha256"]: row["family_id"] for row in packet["contexts"]}
        == expected
        and packet["bundle_digest"] == plan["bundle_digest"],
        "regeneration capture differs from the reviewed plan population or bundle",
    )
    torch.set_num_threads(1)
    generate.CUSTODY = Path(plan["custody"])
    index = generate._generate_locked(
        Path(plan["bundle"]), plan["bundle_digest"], captures, capture_pin, output
    )
    checked(output / "index.json", index_pin)
    generate.require(
        index["total_bytes"] <= generate.MAX_BYTES,
        "regenerated packet exceeds byte bound",
    )
    generate.require(
        not torch.cuda.is_initialized(), "CPU regeneration initialized CUDA"
    )
    return dict(
        schema="jope.cpu-golden-regeneration/v1",
        contexts=index["contexts"],
        file_count=index["file_count"],
        fixture_bytes=index["total_bytes"],
        capture_manifest_sha256=capture_pin,
        fixture_index_sha256=index_pin,
        native_calls=0,
        fold_fits=0,
        cuda_initialized=False,
        qualification=index["qualification"],
    )


def historical_packet(plan, plan_pin, pools, packet, index_pin, store, output):
    """Export captured TRAIN feature rows and frozen projections, not guessed DSLs."""
    generate.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CPU historical export requires hidden CUDA",
    )
    generate.require(
        os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 or lower priority required"
    )
    generate.require(
        not output.exists() and not output.is_symlink(), "output must be new"
    )
    source_index = checked(packet / "index.json", index_pin)
    original = generate.read_json(source_index)
    contexts = {row["context_sha256"] for row in plan["contexts"]}
    generate.require(
        original["bundle_digest"] == plan["bundle_digest"]
        and original["contexts"] == len(contexts)
        and {
            Path(name).stem
            for name in original["files"]
            if name.startswith("contexts/")
        }
        == contexts,
        "packet differs from reviewed plan",
    )
    files = {"packet/index.json": source_index}
    for name, spec in original["files"].items():
        data = generate.captured(packet, dict(path=name, sha256=spec["sha256"]))
        generate.require(len(data) == spec["bytes"], "source packet length differs")
        files["packet/" + name] = data
    generate.require(
        sum(len(data) for data in files.values()) == original["total_bytes"],
        "source packet size differs",
    )
    torch.set_num_threads(1)
    generate.CUSTODY = Path(plan["custody"])
    _, _, reference, _ = generate.validate_bundle(
        Path(plan["bundle"]), plan["bundle_digest"]
    )
    matrix = np.asarray(reference.env["get_w1_sign_matrix"](), dtype=np.float64)
    generate.require(
        matrix.shape == (412, 16) and np.isfinite(matrix).all(),
        "frozen projection matrix differs",
    )
    total_rows, programs = 0, set()
    for row in sorted(plan["contexts"], key=lambda r: r["context_sha256"]):
        cid = row["context_sha256"]
        raw = checked(pools / (cid + ".npz"), row["pool_sha256"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as pool:
            features = pool["program"]
            identities = pool["program_sha256"]
            generate.require(
                features.dtype == np.float32
                and features.shape == (row["n_candidates"], 412)
                and np.isfinite(features).all()
                and identities.shape == (len(features),),
                "historical row count/feature contract differs",
            )
            hashes = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in identities.tolist()
            ]
        generate.require(
            all(
                len(key) == 64 and all(c in "0123456789abcdef" for c in key)
                for key in hashes
            ),
            "invalid historical program hash",
        )
        projection = features.astype(np.float64) @ matrix / np.sqrt(412)
        files[f"historical/{cid}.safetensors"] = generate.tensor_bytes(
            dict(feat412=features, proj16=projection)
        )
        files[f"historical/{cid}.json"] = (
            generate.canonical(
                dict(
                    context_sha256=cid,
                    family_id=row["family_id"],
                    dl_split="DL-Train",
                    pool_sha256=row["pool_sha256"],
                    program_sha256=hashes,
                    feature_source="captured historical NPZ program F32; DSL strings absent",
                )
            )
            + b"\n"
        )
        total_rows += len(features)
        programs.update(hashes)
    generate.require(
        sum(len(data) for data in files.values()) <= 511 * 1024 * 1024,
        "historical set exceeds byte bound",
    )
    specs = {
        name: generate.custody_blob(store, data) for name, data in sorted(files.items())
    }
    index, encoded = generate.custody_index(
        specs,
        dict(
            plan_sha256=plan_pin,
            source_packet_index_sha256=index_pin,
            bundle_digest=plan["bundle_digest"],
            contexts=len(contexts),
            historical_program_rows=total_rows,
            historical_unique_program_hashes=len(programs),
            historical_scope="reviewed 50-context TRAIN pool snapshot; all rows/order retained",
            historical_corpus_complete=False,
            features_regenerated_from_dsl=False,
            projection_formula="program.astype(float64) @ get_w1_sign_matrix() / sqrt(412)",
            reference_source_sha256=reference.source_hashes,
            qualification=dict(M2=False, M3=False, M5=False, proposer_retrieval=False),
        ),
    )
    generate.require(
        index["total_bytes"] <= 512 * 1024 * 1024, "historical set exceeds byte bound"
    )
    catalog = generate.custody_blob(store, encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="golden-catalog-", dir=output.parent
    ) as temp:
        staged = Path(temp) / "catalog"
        staged.mkdir()
        (staged / "custody-index.json").write_bytes(encoded)
        generate.exporter()._publish_directory(staged, output)
    generate.require(
        not torch.cuda.is_initialized(), "CPU historical export initialized CUDA"
    )
    return dict(
        catalog_sha256=catalog["sha256"],
        file_count=index["file_count"],
        total_bytes=index["total_bytes"],
        unique_custody_bytes=index["unique_custody_bytes"],
        hash_custody_required=index["hash_custody_required"],
        contexts=len(contexts),
        historical_program_rows=total_rows,
        historical_unique_program_hashes=len(programs),
        native_calls=0,
        fold_fits=0,
    )


def train_dsl_records(plan, paths, pins):
    generate.require(
        len(paths) == len(pins) and paths and len(set(pins)) == len(pins),
        "distinct DSL sources and explicit pins required",
    )
    selected = generate.read_json(
        checked(Path(plan["selected"]), plan["selected_sha256"])
    )
    train = {row["context_sha256"]: row for row in selected["train"]}
    programs, aliases, sources = {}, [], []
    for path, pin in sorted(zip(paths, pins), key=lambda item: item[1]):
        doc = generate.read_json(checked(path, pin))
        if isinstance(doc, list):
            rows, kind = doc, "train-records"
        else:
            generate.require(
                set(doc) == {"all_90_qualified_novel_winner_records"},
                "unsupported DSL source inventory",
            )
            rows, kind = (
                doc["all_90_qualified_novel_winner_records"],
                "novel-winner-records",
            )
        generate.require(rows, "empty DSL source inventory")
        for position, row in enumerate(rows):
            cid = row["context_sha256"]
            generate.require(
                cid in train and train[cid]["dl_split"] == "DL-Train",
                "DSL source is outside TRAIN",
            )
            family = train[cid]["family_id"]
            generate.require(
                row.get(
                    "dl_split", "DL-Train" if kind == "novel-winner-records" else None
                )
                == "DL-Train"
                and row.get("family_id", family) == family,
                "DSL TRAIN family/role differs",
            )
            key, dsl = row["program_sha256"], row["dsl"]
            generate.require(
                isinstance(dsl, str) and generate.sha(dsl.encode()) == key,
                "DSL program hash mismatch",
            )
            generate.require(
                key not in programs or programs[key] == dsl,
                "ambiguous DSL program identity",
            )
            programs[key] = dsl
            aliases.append(
                dict(
                    source_sha256=pin,
                    source_row=position,
                    source_kind=kind,
                    context_sha256=cid,
                    family_id=family,
                    dl_split="DL-Train",
                    program_sha256=key,
                )
            )
        sources.append(dict(sha256=pin, kind=kind, records=len(rows)))
    return programs, aliases, sources


def dsl_coverage(plan, plan_pin, store, historical_sha, paths, pins, output):
    """Recover exact delivered DSLs, never infer missing text from feature vectors."""
    generate.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CPU DSL coverage requires hidden CUDA",
    )
    generate.require(
        os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 or lower priority required"
    )
    generate.require(
        not output.exists() and not output.is_symlink(), "output must be new"
    )
    parent, parent_bytes = generate.read_custody_index(store, historical_sha)
    generate.require(
        parent["plan_sha256"] == plan_pin
        and parent["bundle_digest"] == plan["bundle_digest"]
        and not any(name.startswith("dsl/") for name in parent["files"]),
        "historical catalog plan/bundle/scope differs",
    )

    def object_bytes(name):
        spec = parent["files"][name]
        data = generate.regular(store / spec["sha256"])
        generate.require(
            len(data) == spec["bytes"] and generate.sha(data) == spec["sha256"],
            "historical custody object differs",
        )
        return data

    for name in parent["files"]:
        object_bytes(name)
    programs, aliases, sources = train_dsl_records(plan, paths, pins)
    torch.set_num_threads(1)
    generate.CUSTODY = Path(plan["custody"])
    _, _, reference, _ = generate.validate_bundle(
        Path(plan["bundle"]), plan["bundle_digest"]
    )
    keys = sorted(programs)
    canonical = {key: reference.featurize(programs[key]) for key in keys}
    extras, shards = {}, []
    for start in range(0, len(keys), 8192):
        chunk = keys[start : start + 8192]
        name = f"dsl/programs-{len(shards):05d}.safetensors"
        extras[name] = generate.tensor_bytes(
            dict(
                feat412=np.stack([canonical[k][0] for k in chunk]),
                proj16=np.stack([canonical[k][1] for k in chunk]),
            )
        )
        shards.append(dict(path=name, start=start, end=start + len(chunk)))
    extras["dsl/programs.jsonl"] = b"".join(
        generate.canonical(dict(program_sha256=key, dsl=programs[key])) + b"\n"
        for key in keys
    )
    extras["dsl/source-rows.jsonl"] = b"".join(
        generate.canonical(row) + b"\n" for row in aliases
    )
    historical, covered, contexts = set(), set(), []
    covered_rows, total_rows, raw_projection_difference = 0, 0, 0.0
    matrix = reference.env["get_w1_sign_matrix"]()
    for row in sorted(plan["contexts"], key=lambda value: value["context_sha256"]):
        cid = row["context_sha256"]
        meta = generate.read_json(object_bytes(f"historical/{cid}.json"))
        identities = meta["program_sha256"]
        generate.require(
            meta["pool_sha256"] == row["pool_sha256"]
            and meta["dl_split"] == "DL-Train"
            and len(identities) == row["n_candidates"],
            "historical context/row identity differs",
        )
        tensors = generate.load_tensors(object_bytes(f"historical/{cid}.safetensors"))
        f, p = tensors["feat412"].numpy(), tensors["proj16"].numpy()
        generate.require(
            f.shape == (len(identities), 412)
            and f.dtype == np.float32
            and p.shape == (len(identities), 16)
            and p.dtype == np.float64,
            "historical numeric contract differs",
        )
        count = 0
        for i, key in enumerate(identities):
            historical.add(key)
            if key in canonical:
                actual, projection = canonical[key]
                generate.require(
                    np.array_equal(actual, f[i]),
                    "canonical feature differs from captured history: "
                    + cid
                    + "/"
                    + key,
                )
                captured_domain = actual.astype(np.float64) @ matrix / np.sqrt(412)
                generate.require(
                    np.max(np.abs(captured_domain - p[i])) <= 1e-12,
                    "captured projection domain differs",
                )
                raw_projection_difference = max(
                    raw_projection_difference, float(np.max(np.abs(projection - p[i])))
                )
                covered.add(key)
                count += 1
        covered_rows += count
        total_rows += len(identities)
        contexts.append(
            dict(context_sha256=cid, rows=len(identities), covered_rows=count)
        )
    generate.require(
        total_rows == parent["historical_program_rows"]
        and len(historical) == parent["historical_unique_program_hashes"],
        "historical coverage inventory differs",
    )
    missing = sorted(historical - covered)
    prior = {
        generate.read_json(line)["program_sha256"]
        for line in object_bytes("packet/programs.jsonl").splitlines()
    }
    coverage = dict(
        sources=sources,
        delivered_source_records=len(aliases),
        delivered_unique_programs=len(keys),
        extra_definitions_vs_prior=len(set(keys) - prior),
        historical_unique_programs=len(historical),
        historical_rows=total_rows,
        covered_program_sha256=sorted(covered),
        missing_program_sha256=missing,
        covered_rows=covered_rows,
        contexts=contexts,
        canonical_tensor_shards=shards,
        canonical_projection_domain="raw frozen featurizer FP64",
        captured_projection_domain="captured feature F32 -> FP64 frozen caller",
        raw_vs_captured_projection_max_abs_difference=raw_projection_difference,
        historical_snapshot_full_dsl_coverage=not missing,
        global_historical_corpus_complete=False,
    )
    extras["dsl/coverage.json"] = generate.canonical(coverage) + b"\n"
    generate.require(
        parent["total_bytes"] + sum(len(data) for data in extras.values())
        < 511 * 1024 * 1024,
        "DSL coverage set exceeds byte bound",
    )
    files = dict(parent["files"])
    files["sources/historical-custody-index.json"] = generate.custody_blob(
        store, parent_bytes
    )
    files.update(
        {
            name: generate.custody_blob(store, data)
            for name, data in sorted(extras.items())
        }
    )
    metadata = {
        key: parent[key]
        for key in [
            "plan_sha256",
            "source_packet_index_sha256",
            "bundle_digest",
            "contexts",
            "historical_program_rows",
            "historical_unique_program_hashes",
            "historical_scope",
            "historical_corpus_complete",
            "features_regenerated_from_dsl",
            "projection_formula",
            "reference_source_sha256",
            "qualification",
        ]
    }
    metadata.update(
        source_historical_catalog_sha256=historical_sha,
        dsl_coverage=dict(
            sources=sources,
            delivered_source_records=len(aliases),
            delivered_unique_programs=len(keys),
            covered_historical_programs=len(covered),
            missing_historical_programs=len(missing),
        ),
    )
    index, encoded = generate.custody_index(files, metadata)
    generate.require(
        index["total_bytes"] <= 512 * 1024 * 1024, "DSL coverage set exceeds byte bound"
    )
    catalog = generate.custody_blob(store, encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="golden-dsl-", dir=output.parent) as temp:
        staged = Path(temp) / "catalog"
        staged.mkdir()
        (staged / "custody-index.json").write_bytes(encoded)
        generate.exporter()._publish_directory(staged, output)
    generate.require(
        not torch.cuda.is_initialized(), "CPU DSL coverage initialized CUDA"
    )
    return dict(
        catalog_sha256=catalog["sha256"],
        file_count=index["file_count"],
        total_bytes=index["total_bytes"],
        delivered_source_records=len(aliases),
        delivered_unique_programs=len(keys),
        extra_definitions_vs_prior=coverage["extra_definitions_vs_prior"],
        covered_historical_programs=len(covered),
        missing_historical_programs=len(missing),
        covered_historical_rows=covered_rows,
        full_historical_corpus_complete=False,
        native_calls=0,
        fold_fits=0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regenerate-captures", type=Path)
    parser.add_argument("--expected-captures-sha256")
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--historical-pools", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--custody-store", type=Path)
    parser.add_argument("--dsl-records", type=Path, action="append")
    parser.add_argument("--expected-dsl-sha256", action="append")
    parser.add_argument("--historical-catalog-sha256")
    args = parser.parse_args(argv)
    dsl_mode = (
        args.dsl_records,
        args.expected_dsl_sha256,
        args.historical_catalog_sha256,
    )
    if any(dsl_mode) and (
        not all(dsl_mode)
        or not args.custody_store
        or args.historical_pools
        or args.packet
        or args.regenerate_captures
        or args.expected_captures_sha256
        or args.expected_index_sha256
    ):
        parser.error(
            "DSL records/pins, historical catalog and custody store are required together, separately from other modes"
        )
    history = (args.historical_pools, args.packet, args.custody_store)
    if (
        not any(dsl_mode)
        and any(history)
        and (
            not all(history)
            or not args.expected_index_sha256
            or args.regenerate_captures
            or args.expected_captures_sha256
        )
    ):
        parser.error(
            "historical pools, packet, custody store and expected index are required together; regeneration flags are separate"
        )
    replay = (
        args.regenerate_captures,
        args.expected_captures_sha256,
        args.expected_index_sha256,
    )
    if not any(dsl_mode) and not any(history) and any(replay) and not all(replay):
        parser.error(
            "regenerate-captures and both expected SHA256 pins are required together"
        )
    plan = generate.read_json(checked(args.plan, args.expected_plan_sha256))
    validate_plan(plan)
    if args.dsl_records is not None:
        report = dsl_coverage(
            plan,
            args.expected_plan_sha256,
            args.custody_store,
            args.historical_catalog_sha256,
            args.dsl_records,
            args.expected_dsl_sha256,
            args.output,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.historical_pools is not None:
        report = historical_packet(
            plan,
            args.expected_plan_sha256,
            args.historical_pools,
            args.packet,
            args.expected_index_sha256,
            args.custody_store,
            args.output,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.regenerate_captures is not None:
        report = regenerate(
            plan,
            args.regenerate_captures,
            args.expected_captures_sha256,
            args.expected_index_sha256,
            args.output,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(
            TimeoutError("remaining CPU fixture budget exhausted")
        ),
    )
    remaining = min(
        plan["budget"]["after_lock_wall_seconds"],
        int(plan["cpu_deadline_epoch_seconds"] - time.time()),
    )
    generate.require(remaining > 0, "original absolute CPU deadline exhausted")
    signal.alarm(remaining)
    print(json.dumps(capture(plan, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
