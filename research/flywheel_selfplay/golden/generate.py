#!/usr/bin/env python3
"""RES-07 CPU pre-stage validation and lock-gated reference fixture packaging.

No historical runner, checkpoint loader, GPU or network is used by validation.
Generation requires pinned training-only reference captures; see README.md.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import jsonschema
import numpy as np
import torch

if __package__:
    from .reference import Reference
else:
    from reference import Reference
from safetensors.torch import load as load_tensors

ROOT = Path(__file__).resolve().parents[3]
CUSTODY = ROOT / "research/flywheel_selfplay/custody"
MIN_CONTEXTS = 50
MAX_BYTES = 64 * 1024 * 1024
GIT_FIXTURE_LIMIT = 50_000_000
CONSUMERS = {
    "RNK-02": ["programs.safetensors:proj16"],
    "RNK-03/RNK-04": ["programs.safetensors:feat412", "programs.jsonl"],
    "RNK-05": ["landmarks/<context>.safetensors"],
    "RNK-06": ["contexts/<context>.safetensors:lm48,lm48_z"],
    "RNK-07": ["contexts/<context>.safetensors:tokens,mask"],
    "RNK-08": ["slates/<context>-12.safetensors", "slates/<context>-20.safetensors"],
    "RNK-09": [],  # Parked: no admitted raw proposer snapshot in RES-01/RES-02.
    "DSC-07": ["landmark-pools.json"],
    "RES-09": ["index.json", "all *.safetensors (exact per-file tensor schema)"],
    "M3 evaluator": ["surrogate/<context>.json"],
}


class Refusal(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    def invalid(value):
        raise Refusal("non-finite JSON constant: " + value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def regular(path, maximum=MAX_BYTES):
    require(
        path.is_file() and not path.is_symlink(), "regular input required: " + str(path)
    )
    require(path.stat().st_size <= maximum, "input exceeds byte bound")
    data = path.read_bytes()
    require(len(data) <= maximum, "input grew past byte bound")
    return data


def exporter():
    spec = importlib.util.spec_from_file_location(
        "res07_export_contract",
        ROOT / "research/flywheel_selfplay/export_fused_bundle.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decompress(data):
    # File-backed input prevents a pipe deadlock while bounding decompressed bytes.
    with tempfile.TemporaryFile() as source:
        source.write(data)
        source.seek(0)
        process = subprocess.Popen(
            ["zstd", "--decompress", "--stdout", "--quiet"],
            stdin=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            output = process.stdout.read(MAX_BYTES + 1)
            require(len(output) <= MAX_BYTES, "decompressed bank exceeds byte bound")
            require(process.wait(timeout=15) == 0, "bank decompression failed")
            return output
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()


def model_prior_programs(bank):
    """Keep the admitted prior population distinct from raw proposer coordinates."""
    require(len(bank) == 881, "RES-06 model_prior inventory differs")
    programs = {}
    for row in bank:
        require(
            row.get("role") == "model_prior"
            and row.get("coordinate_space") == "standardized_lm48_l2_unit",
            "only the tagged model_prior / standardized_lm48_l2_unit bank is admitted",
        )
        key = sha(row["dsl"].encode())
        programs.setdefault(key, dict(program_sha256=key, dsl=row["dsl"]))
    return list(programs.values())


def validate_bundle(directory, expected_digest):
    require(
        directory.is_dir() and not directory.is_symlink(),
        "bundle must be a real directory",
    )
    contract = exporter()
    manifest = read_json(regular(directory / "manifest.json"))
    jsonschema.validate(manifest, read_json(regular(contract.SCHEMA_PATH)))
    require(
        manifest["bundle_digest"] == expected_digest,
        "bundle differs from explicit expected digest",
    )
    require(
        {p.name for p in directory.iterdir()} == {"manifest.json", *contract.PAYLOADS},
        "bundle must contain exactly the RES-06 payload inventory",
    )
    files = {name: regular(directory / name) for name in sorted(contract.PAYLOADS)}
    require(
        contract.compute_bundle_digest(manifest, files) == expected_digest,
        "BundleDigest mismatch",
    )
    for name, field in [
        ("dope.safetensors", "dope_weights_sha256"),
        ("jope.safetensors", "jope_weights_sha256"),
    ]:
        require(sha(files[name]) == manifest[field], "weight-container digest mismatch")
    state = load_tensors(files["dope.safetensors"])
    jope = load_tensors(files["jope.safetensors"])
    require(not set(state) & set(jope), "duplicate tensor across bundle partitions")
    state.update(jope)
    contract.validate_state(state)
    features = read_json(files["featurizer.json"])
    names = features["feature_names"]
    require(
        len(names) == len(set(names)) == 412,
        "412 distinct ordered feature names required",
    )
    require(
        features["proj16_domain"] == "jope-context-cross-v1",
        "projection domain mismatch",
    )
    custody = read_json(regular(CUSTODY / "MANIFEST.json"))
    require(
        manifest["reference_source_sha256"]["custody/MANIFEST.json"]
        == sha(regular(CUSTODY / "MANIFEST.json")),
        "custody manifest pin mismatch",
    )
    reference = Reference(CUSTODY, custody, state, names)
    bank = [read_json(row) for row in decompress(files["bank.jsonl.zst"]).splitlines()]
    programs = model_prior_programs(bank)
    max_proj_error = 0.0
    prior_projection_differences = 0
    for row in bank:
        f, p = reference.featurize(row["dsl"])
        require(
            np.array_equal(f, np.asarray(row["feat412"], dtype=np.float32)),
            "pinned featurizer does not reproduce prior-bank feat412",
        )
        error = float(np.max(np.abs(p - np.asarray(row["proj16"], dtype=np.float64))))
        # These are frozen prior-bank inputs, not a raw-FP64 proj16 oracle.
        # Preserve/report differences rather than relabelling either coordinate contract.
        prior_projection_differences += int(
            not np.array_equal(
                p.astype(np.float32), np.asarray(row["proj16"], dtype=np.float32)
            )
        )
        max_proj_error = max(max_proj_error, error)
    report = dict(
        stage="cpu-pre-stage",
        bundle_digest=expected_digest,
        tensors=len(state),
        frozen_dope_composite=contract.FROZEN_DOPE_COMPOSITE,
        payloads={
            name: dict(sha256=sha(data), bytes=len(data))
            for name, data in files.items()
        },
        prior_bank_records=len(bank),
        validated_prior_rows=len(bank),
        prior_unique_programs=len({row["dsl"] for row in bank}),
        subset=dict(
            role="model_prior",
            coordinate_space="standardized_lm48_l2_unit",
            records=len(bank),
            programs=len(programs),
            historical_corpus_complete=False,
            proposer_retrieval_qualified=False,
        ),
        prior_proj16_vs_raw_max_absolute_difference=max_proj_error,
        prior_proj16_vs_raw_differing_programs=prior_projection_differences,
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device="cpu"),
        reference_source_sha256=reference.source_hashes,
        qualification=dict(M2=False, M3=False, M5=False),
        fixture_files_written=0,
        bank_scope="fixed prior bank only; not the 1219-context raw-SVD proposer bank",
    )
    return manifest, custody, reference, report


def tensor_bytes(values):
    """Canonical SafeTensors, preserving FP32/FP64/I64 rather than lossy casts."""
    header, chunks, offset = {}, [], 0
    dtypes = {"<f4": "F32", "<f8": "F64", "<i8": "I64", "|u1": "U8"}
    for name in sorted(values):
        value = values[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        array = np.asarray(value)
        dtype = array.dtype.newbyteorder("<")
        require(dtype.str in dtypes, "unsupported fixture tensor dtype")
        require(np.isfinite(array).all(), "non-finite fixture tensor: " + name)
        shape = array.shape
        array = np.ascontiguousarray(array, dtype=dtype).reshape(shape)
        data = array.tobytes()
        header[name] = dict(
            dtype=dtypes[dtype.str],
            shape=list(array.shape),
            data_offsets=[offset, offset + len(data)],
        )
        chunks.append(data)
        offset += len(data)
    encoded = canonical(header)
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)


def captured(root, item):
    name = Path(item["path"])
    require(
        not name.is_absolute()
        and name.parts
        and all(p not in (".", "..") for p in name.parts),
        "capture path must be canonical and relative",
    )
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "capture escapes its root")
    data = regular(path)
    require(sha(data) == item["sha256"], "capture hash mismatch: " + str(name))
    return data


def custody_blob(store, data):
    """Publish an immutable-by-convention SHA256 object without replacement."""
    require(len(data) <= MAX_BYTES, "custody object exceeds byte bound")
    store.mkdir(parents=True, exist_ok=True)
    require(
        store.is_dir() and not store.is_symlink(), "real custody directory required"
    )
    key = sha(data)
    path = store / key
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=store) as staged:
            staged.write(data)
            staged.flush()
            os.fsync(staged.fileno())
            os.fchmod(staged.fileno(), 0o444)
            try:
                os.link(staged.name, path)
            except FileExistsError:
                pass  # A competing publisher must have produced the same bytes.
    require(regular(path) == data, "existing custody object differs")
    return dict(sha256=key, bytes=len(data))


def custody_index(files, metadata):
    unique = {}
    for spec in files.values():
        require(
            spec["sha256"] not in unique or unique[spec["sha256"]] == spec["bytes"],
            "inconsistent custody length",
        )
        unique[spec["sha256"]] = spec["bytes"]
    index = dict(
        metadata,
        schema="jope.golden-custody/v1",
        files=files,
        file_count=len(files) + 1,
        payload_bytes=sum(spec["bytes"] for spec in files.values()),
        unique_payload_bytes=sum(unique.values()),
        total_bytes=0,
        unique_custody_bytes=0,
        storage="sha256-custody",
        hash_custody_required=False,
    )
    while True:
        encoded = canonical(index) + b"\n"
        sizes = dict(
            total_bytes=index["payload_bytes"] + len(encoded),
            unique_custody_bytes=index["unique_payload_bytes"] + len(encoded),
            hash_custody_required=index["payload_bytes"] + len(encoded)
            > GIT_FIXTURE_LIMIT,
        )
        if all(index[key] == value for key, value in sizes.items()):
            return index, encoded
        index.update(sizes)


def read_custody_index(store, expected_sha256):
    """Authenticate catalog metadata without copying any payloads."""
    require(
        store.is_dir() and not store.is_symlink(), "real custody directory required"
    )
    require(
        len(expected_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_sha256),
        "invalid custody index SHA256",
    )
    encoded = regular(store / expected_sha256)
    require(sha(encoded) == expected_sha256, "custody index hash mismatch")
    index = read_json(encoded)
    require(index["schema"] == "jope.golden-custody/v1", "unsupported custody index")
    rebuilt, expected = custody_index(
        index["files"],
        {
            key: value
            for key, value in index.items()
            if key
            not in {
                "schema",
                "files",
                "file_count",
                "payload_bytes",
                "unique_payload_bytes",
                "total_bytes",
                "unique_custody_bytes",
                "storage",
                "hash_custody_required",
            }
        },
    )
    require(index == rebuilt and encoded == expected, "custody byte accounting differs")
    require(index["total_bytes"] <= 512 * 1024 * 1024, "custody set exceeds byte bound")
    for name, spec in index["files"].items():
        path = Path(name)
        require(
            path.parts
            and not path.is_absolute()
            and name == path.as_posix()
            and all(p not in (".", "..") for p in path.parts)
            and name != "custody-index.json",
            "invalid custody logical path",
        )
        key = spec["sha256"]
        require(
            len(key) == 64 and all(c in "0123456789abcdef" for c in key),
            "invalid custody object SHA256",
        )
    return index, encoded


def materialize_custody(store, expected_sha256, output):
    """Fetch only authenticated objects into one new atomic fixture directory."""
    require(not output.exists() and not output.is_symlink(), "output must be new")
    index, encoded = read_custody_index(store, expected_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="golden-fetch-", dir=output.parent) as temp:
        staging = Path(temp) / "packet"
        staging.mkdir()
        for name, spec in index["files"].items():
            data = regular(store / spec["sha256"])
            require(
                len(data) == spec["bytes"] and sha(data) == spec["sha256"],
                "custody object hash/length mismatch",
            )
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (staging / "custody-index.json").write_bytes(encoded)
        exporter()._publish_directory(staging, output)
    return index


@contextmanager
def qualification_lease(path):
    require(
        path.exists() and not path.is_symlink(),
        "existing real qualification lock required",
    )
    with path.open("r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Refusal(
                "qualification lock busy; never preempt another owner"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _generate_locked(
    bundle, expected_digest, captures_path, expected_captures_sha256, output
):
    """Generate only from pinned train-only oracle captures, never table discovery."""
    require(not output.exists() and not output.is_symlink(), "output must be new")
    manifest, custody, reference, validation = validate_bundle(bundle, expected_digest)
    root = captures_path.parent
    capture_bytes = regular(captures_path)
    require(
        sha(capture_bytes) == expected_captures_sha256,
        "capture differs from explicit reviewed manifest pin",
    )
    capture = read_json(capture_bytes)
    require(
        capture["schema"] == "jope.golden-source-captures/v1",
        "unsupported capture schema",
    )
    require(
        capture["bundle_digest"] == expected_digest, "capture bundle identity differs"
    )
    require(
        not {"proposer_bank", "historical_inventory", "corpus"}.intersection(capture),
        "this model_prior subset does not admit proposer or replacement corpus inputs",
    )
    bank_bytes = regular(bundle / "bank.jsonl.zst")
    require(
        sha(bank_bytes) == validation["payloads"]["bank.jsonl.zst"]["sha256"],
        "model_prior payload changed after validation",
    )
    bank = [read_json(row) for row in decompress(bank_bytes).splitlines()]
    programs = model_prior_programs(bank)
    hashes = [row["program_sha256"] for row in programs]
    vectors = [reference.featurize(row["dsl"]) for row in programs]
    features = np.stack([p[0] for p in vectors])
    projections = np.stack([p[1] for p in vectors])
    positions = {key: i for i, key in enumerate(hashes)}
    contexts = capture["contexts"]
    require(
        len(contexts) >= MIN_CONTEXTS
        and len({c["context_sha256"] for c in contexts}) == len(contexts),
        "at least 50 distinct training contexts required",
    )
    train_families = set(custody["training_corpus"]["train_family_ids"])
    selected_bytes = captured(root, capture["selected"])
    require(
        sha(selected_bytes) == custody["training_corpus"]["selected_sha256"],
        "training selection pin differs",
    )
    # Read identifiers only from the pinned row-free metadata, never follow its table paths.
    selected = read_json(selected_bytes)
    train = {row["context_sha256"]: row["family_id"] for row in selected["train"]}
    require(
        len(train) == custody["training_corpus"]["train_contexts"],
        "training roster count differs",
    )
    source_pins = {row["path"]: row["sha256"] for row in custody["import_files"]}
    workspace = next(
        value
        for key, value in source_pins.items()
        if key.endswith("/hpc/landmark_workspace.py")
    )
    evaluator = custody["evaluator"]["binary"]["sha256"]
    caller = next(
        value
        for key, value in reference.source_hashes.items()
        if key.endswith(
            "/frozen-source/research/flywheel_selfplay/run_fused_closed_loop.py"
        )
    )
    native = next(
        row["sha256"]
        for row in custody["assets"]
        if row["path"].endswith("/bin/libjope_hpc.so")
    )
    files = {
        "programs.safetensors": tensor_bytes(
            dict(feat412=features, proj16=projections)
        ),
        "programs.jsonl": b"".join(canonical(p) + b"\n" for p in programs),
        "model-prior.jsonl": b"".join(canonical(row) + b"\n" for row in bank),
    }
    pools = []
    for context in sorted(contexts, key=lambda c: c["context_sha256"]):
        cid = context["context_sha256"]
        require(
            len(cid) == 64 and all(c in "0123456789abcdef" for c in cid),
            "invalid context hash",
        )
        require(
            context["dl_split"] == "DL-Train"
            and cid in train
            and context["family_id"] == train[cid]
            and context["family_id"] in train_families,
            "context is outside the pinned training families",
        )
        landmark = context["landmarks"]
        require(
            landmark["reference_source_sha256"] == workspace
            and landmark["outer_query_rows"] == 0
            and landmark["caller_source_sha256"] == caller
            and landmark["native_library_sha256"] == native,
            "full trained landmark recipe and zero outer-query rows required",
        )
        data = captured(root, landmark)
        tensors = load_tensors(data)
        require(
            set(tensors)
            >= {"input.X", "input.y", "input.row_ids", "v256", "col", "col_full"},
            "landmark capture must retain actual training inputs and full/padded outputs",
        )
        x, y, rows, full_col = (
            tensors[key] for key in ("input.X", "input.y", "input.row_ids", "col_full")
        )
        require(
            x.ndim == 2
            and 0 < x.shape[0] <= manifest["landmark_sample_rows"]
            and y.shape == rows.shape == (x.shape[0],)
            and rows.dtype == torch.int64
            and len(torch.unique(rows)) == len(rows)
            and bool((rows >= 0).all()),
            "landmark inputs require the bounded actual training rows",
        )
        require(
            full_col.shape == (x.shape[1], 24) and full_col.dtype == torch.float32,
            "original native full-recipe column outputs required",
        )
        padded = torch.zeros((64, 17), dtype=torch.float32)
        n = min(64, len(full_col))
        padded[:n] = full_col[:n, :17]
        require(
            torch.equal(padded, tensors["col"].reshape(64, 17)),
            "caller column truncation/padding differs",
        )
        inter = reference.intermediates(tensors["v256"].numpy(), tensors["col"].numpy())
        files[f"landmarks/{cid}.safetensors"] = data
        files[f"contexts/{cid}.safetensors"] = tensor_bytes(inter)
        stages, heads = reference.preferred_pools(inter["lm48"].numpy().reshape(-1))
        pools.append(dict(context_sha256=cid, stages=stages, heads=heads))
        measured = read_json(captured(root, context["surrogate"]))
        require(
            measured["evaluator_sha256"] == evaluator
            and measured["context_sha256"] == cid
            and measured["recipe"] == manifest["surrogate_recipe"],
            "surrogate comparison identity differs",
        )
        files[f"surrogate/{cid}.json"] = canonical(measured) + b"\n"
        require(
            set(context["slates"]) == {"12", "20"},
            "both 4+8 and 4+16 serving slates required",
        )
        for width in (12, 20):
            slate = context["slates"][str(width)]
            require(
                len(slate["program_sha256"]) == width, "serving slate width differs"
            )
            require(
                len(slate["revealed_raw_f"]) == len(slate["revealed_gate"]) == 4,
                "exactly four measured reveals required",
            )
            require(
                all(math.isfinite(v) for v in slate["revealed_raw_f"])
                and all(v in (0, 1) for v in slate["revealed_gate"]),
                "invalid reveal values",
            )
            for key, value, gate in zip(
                slate["program_sha256"][:4],
                slate["revealed_raw_f"],
                slate["revealed_gate"],
            ):
                observation = measured["measurements"][key]
                require(
                    observation["raw_f"] == value and observation["gate"] == gate,
                    "revealed evidence differs from source measurement",
                )
            indices = [positions[key] for key in slate["program_sha256"]]
            result = reference.costs(
                inter,
                features[indices],
                projections[indices],
                slate["revealed_raw_f"],
                slate["revealed_gate"],
            )
            files[f"slates/{cid}-{width}.safetensors"] = tensor_bytes(result)
    files["landmark-pools.json"] = canonical(pools) + b"\n"
    total = sum(len(data) for data in files.values())
    index = dict(
        schema="jope.golden-fixtures/v1",
        bundle_digest=expected_digest,
        capture_manifest_sha256=sha(regular(captures_path)),
        contexts=len(contexts),
        model_prior_programs=len(programs),
        subset=validation["subset"],
        qualification=dict(M2=False, M3=False, M5=False, proposer_retrieval=False),
        consumers=CONSUMERS,
        reference_source_sha256=reference.source_hashes,
        payload_bytes=total,
        total_bytes=0,
        file_count=len(files) + 1,
        storage="git-eligible",
        files={
            name: dict(
                sha256=sha(data),
                bytes=len(data),
                **(
                    {
                        "tensors": {
                            key: {"dtype": value["dtype"], "shape": value["shape"]}
                            for key, value in read_json(
                                data[8 : 8 + struct.unpack("<Q", data[:8])[0]]
                            ).items()
                            if key != "__metadata__"
                        }
                    }
                    if name.endswith(".safetensors")
                    else {}
                ),
            )
            for name, data in sorted(files.items())
        },
    )
    while True:
        encoded = canonical(index) + b"\n"
        final_size = total + len(encoded)
        storage = "hash-custody" if final_size > GIT_FIXTURE_LIMIT else "git-eligible"
        if index["total_bytes"] == final_size and index["storage"] == storage:
            break
        index.update(total_bytes=final_size, storage=storage)
    files["index.json"] = encoded
    require(
        not (final_size > GIT_FIXTURE_LIMIT and output.resolve().is_relative_to(ROOT)),
        "fixtures above 50MB must remain in external hash custody",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="golden-", dir=output.parent) as temp:
        staging = Path(temp) / "payload"
        staging.mkdir()
        for name, data in sorted(files.items()):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        # Existing export primitive uses Linux RENAME_NOREPLACE; no racing output overwrite.
        exporter()._publish_directory(staging, output)
    return index


def generate(bundle, expected_digest, captures_path, expected_captures_sha256, output):
    merged = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            "7b30157ca2289db20b684e68f3c6fe0fbb84673d",
            "HEAD",
        ],
        check=False,
    )
    require(merged.returncode == 0, "merged RES-06 source required")
    with qualification_lease(Path.home() / "jope_plan/qualification.lock"):
        return _generate_locked(
            bundle, expected_digest, captures_path, expected_captures_sha256, output
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-digest", required=True)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-captures-sha256")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if (
            args.captures is None
            and args.output is None
            and args.expected_captures_sha256 is None
        ):
            *_, report = validate_bundle(args.bundle, args.expected_bundle_digest)
            report["consumers"] = CONSUMERS
        else:
            require(
                args.captures is not None
                and args.output is not None
                and args.expected_captures_sha256,
                "captures, explicit reviewed capture pin and output are required together",
            )
            report = generate(
                args.bundle,
                args.expected_bundle_digest,
                args.captures,
                args.expected_captures_sha256,
                args.output,
            )
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, jsonschema.ValidationError) as error:
        print("RES-07 refused: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
