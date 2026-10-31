#!/usr/bin/env python3
"""Capture the bounded RES-07 training-only CPU oracle; never CUDA qualification."""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = generate.read_json(checked(args.plan, args.expected_plan_sha256))
    validate_plan(plan)
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
