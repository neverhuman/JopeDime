#!/usr/bin/env python3
"""Train one R10 adapter-only candidate from the pinned R882 state on DL-Train."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Mapping

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
import psutil
import torch
from safetensors.torch import load_file, save_file

from objective import training_loss, choose_training_pair

SOURCE_COMMIT = "3b4f349fd4691db1672ad9243d9cc056b7e055f0"
CHECKPOINT_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
SELECTED_SHA256 = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
SOURCE_MANIFEST_SHA256 = (
    "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
)
TRAIN_PACKET_SHA256 = "a2fc16d2a8112dd6966bcfc34a923a29cf727a3a91d497a74f56b300e6cc0bbd"
SIGNED_JOPE_SHA256 = "ff3f9ea5d76fa6c13cd74d40a9effbf7e9c891d98746dc542d851726b9e3138c"
SIGNED_DOPE_FILE_SHA256 = (
    "0f5e98fa29cf9c45bfaac5a3cf001b13a0de37d7b6e154c582d7ec689b21a84d"
)
SIGNED_MANIFEST_SHA256 = (
    "9a83b2225f57ee451b4813790dad3cb4c41249d4f5a14f7c80e2c0e41540891f"
)
SIGNED_CATALOG_SHA256 = (
    "3af56a817bfae39b70f0b2f60668fe88fae178650ee596e1ba7a180acba1e0c6"
)
SIGNED_PACKAGE_SHA256 = (
    "7523af9700e21325a69f9f71b8b60eb4897d46f3948546e4024010b1ce129a60"
)
SIGNING_MESSAGE_SHA256 = (
    "45e21d576cec91b5171bb8e279ef24fef761456fa916e4de1ed8ed2b8bd6eba5"
)
SIGNING_SIGNATURE_SHA256 = (
    "d35ac4dcdeba1aae42d9ad4469f4681c83ef57df969939ad0101215524a1a780"
)
SIGNING_RECEIPT_SHA256 = (
    "be1e4a4584d89d5b09b26287c8a58a43391943d03f12616280bdbb87b217f5f0"
)
CPU_FUSED_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
TRAINABLE_PARAMS = 29_211
TRAIN_CONTEXTS = 1_219
TRAIN_FAMILIES = 205
SOURCE_SLATE_SIZE = 192
OPTIMIZER_UPDATES = 2
MICROBATCH = 8
SEED = 2_026_1101
MICRO_ROOT = Path("/home/ubuntu/JopeDimeMicro")
MICRO_PYTHON = MICRO_ROOT / "python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_state_for_export(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Copy tied module aliases into independent contiguous storage for Safetensors."""
    return {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in state.items()
    }


def source_blob(path: str, expected_sha: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(result).hexdigest() != expected_sha:
        raise RuntimeError(f"pinned training source hash mismatch: {path}")
    return result.decode("utf-8")


def load_training_indices():
    namespace: dict[str, object] = {"__name__": "r10_pinned_automatic_promotion"}
    source = source_blob(
        "research/flywheel_selfplay/automatic_promotion.py",
        "37e50568fd097944cf7125de340a19da3dbf27f7be80f47e28fb93b6cc24d16d",
    )
    exec(compile(source, "automatic_promotion.py@3b4f349f", "exec"), namespace)
    return namespace["training_indices"]


def verify_sources(input_dir: Path) -> dict[str, str]:
    signed_dir = input_dir / "signed-r882"
    expected = {
        input_dir / "source_manifest.json": SOURCE_MANIFEST_SHA256,
        input_dir / "cpu_exact_fused.py": CPU_FUSED_SHA256,
        signed_dir / "jope.safetensors": SIGNED_JOPE_SHA256,
        signed_dir / "dope.safetensors": SIGNED_DOPE_FILE_SHA256,
        signed_dir / "r882.jopedime": SIGNED_PACKAGE_SHA256,
        signed_dir / "manifest.json": SIGNED_MANIFEST_SHA256,
        signed_dir / "release-catalog.json": SIGNED_CATALOG_SHA256,
        signed_dir / "signing-message.bin": SIGNING_MESSAGE_SHA256,
        signed_dir / "manifest.sig": SIGNING_SIGNATURE_SHA256,
        input_dir / "signed-r882-receipt.json": SIGNING_RECEIPT_SHA256,
        MICRO_PYTHON
        / "phase12_multimodal_model.py": "7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a",
        MICRO_PYTHON
        / "model.py": "0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a",
        MICRO_PYTHON
        / "rust_evaluator.py": "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900",
        MICRO_ROOT
        / "assets/models/dope_mlc_v3_multitask_best.pt": "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952",
        MICRO_ROOT
        / "assets/models/flywheel_winner_pool_v3a_q25_train_only_430.pt": "6f1646afce6be9e88eab289cda0c8b5edec3da13a373518709183dbbfff856b9",
        MICRO_ROOT
        / "assets/models/h40b_round2_dope_self_contained_10tok.pt": "3a1b62754b372a2572b5664929bfa8706da654446a35122d3a8b7702fc2b25db",
        MICRO_ROOT
        / "assets/projections/frozen_stage1_stage2_inference_params_v3.pt": "136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d",
        MICRO_PYTHON
        / "hpc/landmark_workspace.py": "834b3a13db9a257f2b8f3cb892cbdd5571092a41b0e5257826b7d9d3cf61f91a",
        MICRO_PYTHON
        / "research/levels/building_block_landmarks.py": "fc2cc82ec4c654a512f592211e21a940a2c7a5baf3e002fe5a1e7649fdfb9186",
        MICRO_PYTHON
        / "research/levels/descriptors.py": "18cfefdddca262616dadd4841119abfd2b073f9e2ee28fb58c31cad09dd1b928",
        MICRO_ROOT
        / "bin/libjope_hpc.so": "f37f3f4d8ed9954a8ceb23baca08f8b08a79234ad3afd7f90638d68242484901",
    }
    observed = {}
    for path, expected_hash in expected.items():
        got = sha256(path)
        if got != expected_hash:
            raise RuntimeError(f"source/input pin mismatch: {path}")
        observed[str(path)] = got
    source_blob(
        "research/flywheel_selfplay/device_pool_cache.py",
        "cdcb1d672a643716ffba3f2c20aa75d5ae0b19766db616ccd6f6ad65a9adbd76",
    )
    receipt = json.loads((input_dir / "signed-r882-receipt.json").read_text())
    if (
        receipt.get("kind") != "release"
        or receipt.get("label") != "r882"
        or receipt.get("verify") != "PASS"
        or receipt.get("message_sha256") != SIGNING_MESSAGE_SHA256
    ):
        raise RuntimeError(
            "watcher-signed R882 receipt does not match pinned signed inputs"
        )
    manifest = json.loads((signed_dir / "manifest.json").read_text())
    if (
        manifest.get("architecture") != "dope-fused-ranker/h40b"
        or manifest.get("jope_weights_sha256") != SIGNED_JOPE_SHA256
        or manifest.get("dope_weights_sha256") != SIGNED_DOPE_FILE_SHA256
        or manifest.get("bundle_digest")
        != "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc"
        or CHECKPOINT_SHA256 not in manifest.get("reference_source_sha256", {}).values()
    ):
        raise RuntimeError("signed R882 manifest identity mismatch")
    verify_signature = subprocess.run(
        [
            "cosign",
            "verify-blob",
            "--insecure-ignore-tlog",
            "--key",
            str(Path(__file__).resolve().parents[3] / "docs/release/signing.pub"),
            "--signature",
            str(signed_dir / "manifest.sig"),
            str(signed_dir / "signing-message.bin"),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if verify_signature.returncode != 0:
        raise RuntimeError(
            "signed R882 signature verification failed: "
            + verify_signature.stdout.strip()
        )
    return observed


def read_jtt1(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 17 or raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError(f"expected train-only JTT1/v2 table: {path.name}")
    n_train, n_query, width = struct.unpack_from("<III", raw, 4)
    if n_query or len(raw) != 17 + n_train + n_train * (width + 1) * 8:
        raise ValueError(f"invalid train-only JTT1 table: {path.name}")
    values = np.frombuffer(raw, dtype="<f8", offset=17 + n_train).reshape(
        n_train, width + 1
    )
    return values[:, :-1].copy(), values[:, -1].copy()


def landmark_float32(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    maxima = np.max(np.where(np.isfinite(values), np.abs(values), 0.0), axis=0)
    columns = maxima > np.finfo(np.float32).max
    if columns.any():
        values = values.copy()
        values[:, columns] /= maxima[columns]
    return np.ascontiguousarray(values, dtype=np.float32)


def authenticate_train_roster(
    input_dir: Path,
    *,
    packet_sha256: str = TRAIN_PACKET_SHA256,
    source_manifest_sha256: str = SOURCE_MANIFEST_SHA256,
) -> tuple[dict, list[dict]]:
    """Authenticate all row identities before a caller can open any payload."""
    source_path = input_dir / "source_manifest.json"
    packet_path = input_dir / "selected_train.json"
    if sha256(source_path) != source_manifest_sha256:
        raise RuntimeError("train source-manifest file hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("source_manifest_sha256")
        != "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
    ):
        raise RuntimeError("train source-manifest identity mismatch")
    if sha256(packet_path) != packet_sha256:
        raise RuntimeError("train roster packet hash mismatch")
    manifest = json.loads(packet_path.read_text())
    rows = manifest.get("rows")
    if (
        manifest.get("schema") != "jope.rnk10-m5-train-packet/v1"
        or manifest.get("source_selected_sha256") != SELECTED_SHA256
        or manifest.get("source_manifest_sha256") != source_manifest_sha256
        or manifest.get("train_contexts") != TRAIN_CONTEXTS
        or manifest.get("train_families") != TRAIN_FAMILIES
        or manifest.get("task") != "regression"
        or manifest.get("split") != "DL-Train"
        or not isinstance(rows, list)
        or len(rows) != TRAIN_CONTEXTS
    ):
        raise RuntimeError("train-only roster identity mismatch")
    context_ids, families = set(), set()
    for row in rows:
        cid = row.get("context_sha256")
        family = row.get("family_id")
        if (
            not isinstance(cid, str)
            or len(cid) != 64
            or any(char not in "0123456789abcdef" for char in cid)
            or row.get("dl_split") != "DL-Train"
            or row.get("task") != "regression"
            or row.get("source_split") not in ("train", "inner")
            or not isinstance(family, str)
            or not family
            or not isinstance(row.get("pool_sha256"), str)
            or len(row["pool_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["pool_sha256"])
            or not isinstance(row.get("train_only_jtt1_sha256"), str)
            or len(row["train_only_jtt1_sha256"]) != 64
            or any(
                char not in "0123456789abcdef" for char in row["train_only_jtt1_sha256"]
            )
        ):
            raise RuntimeError("invalid or non-training row in authenticated roster")
        context_ids.add(cid)
        families.add(family)
    if len(context_ids) != TRAIN_CONTEXTS:
        raise RuntimeError("duplicate train context")
    if len(families) != TRAIN_FAMILIES:
        raise RuntimeError("train family inventory mismatch")
    return manifest, rows


def load_train_packet(
    input_dir: Path,
) -> tuple[list[dict], dict[str, dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    # This gate must remain ahead of the first pool/table open below.
    _, rows = authenticate_train_roster(input_dir)

    import sys

    sys.path.insert(0, str(MICRO_PYTHON))
    import rust_evaluator as evaluator

    projection = evaluator.get_w1_sign_matrix() / np.sqrt(412.0)
    pools: dict[str, dict] = {}
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, row in enumerate(rows, 1):
        cid = row["context_sha256"]
        pool_path = input_dir / "pools" / f"{cid}.npz"
        table_path = input_dir / "tables" / f"{cid}.jtt1"
        if sha256(pool_path) != row.get("pool_sha256"):
            raise RuntimeError(f"train pool hash mismatch: {cid}")
        if sha256(table_path) != row.get("train_only_jtt1_sha256"):
            raise RuntimeError(f"train table hash mismatch: {cid}")
        with np.load(pool_path, allow_pickle=False) as archive:
            programs = archive["program"].astype(np.float32)
            ids = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in archive["program_sha256"].tolist()
            ]
            fitness = archive["fitness"].astype(np.float32)
            gates = archive["gate"].astype(bool)
        if programs.ndim != 2 or programs.shape[1] != 412 or len(ids) != len(programs):
            raise RuntimeError(f"invalid program pool shape: {cid}")
        pools[cid] = {
            "pids": ids,
            "program": programs,
            "proj16": (programs.astype(np.float64) @ projection).astype(np.float32),
            "fitness": fitness,
            "gate": gates,
        }
        tables[cid] = read_jtt1(table_path)
        if index % 100 == 0:
            wait_for_cpu_headroom()
    return rows, pools, tables


def wait_for_cpu_headroom() -> None:
    while psutil.cpu_percent(interval=1.0) > 85.0:
        print("host CPU above 85%; pausing at train batch boundary", flush=True)
        time.sleep(5)


def prepare_tokens(
    net, tables: dict[str, tuple[np.ndarray, np.ndarray]], context_ids: list[str]
):
    sys.path.insert(0, str(MICRO_PYTHON))
    from hpc.landmark_workspace import FusedLandmarkWorkspace

    workspace = FusedLandmarkWorkspace(num_threads=4)
    tokens: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    device = torch.device("cpu")
    for index, cid in enumerate(context_ids, 1):
        x, y = tables[cid]
        y64 = np.asarray(y, dtype=np.float64)
        y_normalized = ((y64 - y64.mean()) / max(float(y64.std()), 1e-8)).astype(
            np.float32
        )
        vec256, column_lm, _ = workspace.fit_transform_and_extract_bundle_high(
            landmark_float32(x), np.ascontiguousarray(y_normalized)
        )
        columns = np.zeros((64, 17), dtype=np.float32)
        n = min(len(column_lm), 64)
        columns[:n, : min(column_lm.shape[1], 17)] = column_lm[:n, :17]
        with torch.no_grad():
            dope629, lm48 = net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(vec256, dtype=torch.float32, device=device).reshape(
                    1, 256
                ),
                torch.as_tensor(columns, dtype=torch.float32, device=device).reshape(
                    1, 64, 17
                ),
                use_landmarks=True,
            )
            token, mask, _ = net._project_stage1b_tokens_batched_gpu(lm48, dope629)
        tokens[cid] = token[0].detach()
        masks[cid] = mask[0].detach()
        if index % 32 == 0:
            wait_for_cpu_headroom()
    return tokens, masks


def predict_batch(net, pools, tokens, masks, context_ids, selections):
    batch = len(context_ids)
    slate = len(selections[0])
    programs = np.stack(
        [
            pools[c]["program"][sel]
            for c, sel in zip(context_ids, selections, strict=True)
        ]
    )
    proj16 = np.stack(
        [
            pools[c]["proj16"][sel]
            for c, sel in zip(context_ids, selections, strict=True)
        ]
    )
    fitness = np.stack(
        [
            pools[c]["fitness"][sel]
            for c, sel in zip(context_ids, selections, strict=True)
        ]
    )
    gates = np.stack(
        [pools[c]["gate"][sel] for c, sel in zip(context_ids, selections, strict=True)]
    )
    device = torch.device("cpu")
    args = {
        "ctx_feat": torch.zeros((batch, 584), dtype=torch.float32, device=device),
        "prog_412": torch.as_tensor(programs, dtype=torch.float32, device=device),
        "proj_16": torch.as_tensor(proj16, dtype=torch.float32, device=device),
        "base_prior_cost": torch.zeros(
            (batch, slate), dtype=torch.float32, device=device
        ),
        "scale": torch.full((batch,), 0.4208, dtype=torch.float32, device=device),
        "sos_prior": torch.zeros((batch, slate, 2), dtype=torch.float32, device=device),
        "mm_mask_10": torch.stack([masks[c] for c in context_ids]).float(),
        "revealed_indices": torch.arange(4, device=device)
        .reshape(1, 4)
        .expand(batch, -1),
        "revealed_raw_f": torch.as_tensor(
            fitness[:, :4], dtype=torch.float32, device=device
        ),
        "revealed_gate": torch.as_tensor(
            gates[:, :4], dtype=torch.float32, device=device
        ),
    }
    baseline_tokens = torch.stack([tokens[c] for c in context_ids]).float()
    pre_baseline_rng = torch.get_rng_state()
    baseline = net.adapter_10tok.predict_costs(**args, mm_tokens_10x32=baseline_tokens)[
        "total_cost"
    ].float()
    post_baseline_rng = torch.get_rng_state()
    ablated_tokens = baseline_tokens.clone()
    ablated_tokens[:, [0, 1, 7, 9], :] = 0.0
    torch.set_rng_state(pre_baseline_rng)
    ablated = net.adapter_10tok.predict_costs(**args, mm_tokens_10x32=ablated_tokens)[
        "total_cost"
    ].float()
    torch.set_rng_state(post_baseline_rng)
    return (
        baseline,
        ablated,
        torch.as_tensor(fitness, dtype=torch.float32, device=device),
        torch.as_tensor(gates, dtype=torch.bool, device=device),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify signed model/source pins without opening train payloads",
    )
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "xbabe2":
        raise SystemExit("R10 candidate training must run on xbabe2")
    if os.nice(0) < 19:
        raise SystemExit("R10 candidate training must run at nice 19")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    input_dir = args.input.resolve()
    # Authenticate roster and source manifest before opening model or data payloads.
    authenticate_train_roster(input_dir)
    source_pins = verify_sources(input_dir)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() > 8:
        raise SystemExit("CPU thread cap exceeded")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    sys.path.insert(0, str(MICRO_PYTHON))
    spec = importlib.util.spec_from_file_location(
        "r10_cpu_fused", input_dir / "cpu_exact_fused.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned CPU fused model")
    fused = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fused
    spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    initial_state = load_file(
        str(input_dir / "signed-r882/jope.safetensors"), device="cpu"
    )
    dope_state = load_file(
        str(input_dir / "signed-r882/dope.safetensors"), device="cpu"
    )
    if set(initial_state) & set(dope_state):
        raise RuntimeError("signed R882 JOPE and DOPE tensor names overlap")
    initial_state.update(dope_state)
    net.load_state_dict(initial_state, strict=True)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor."):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    count = sum(parameter.numel() for parameter in trainable)
    if count != TRAINABLE_PARAMS:
        raise RuntimeError(f"expected {TRAINABLE_PARAMS} trainable values, got {count}")

    # The source freeze guard is hash-pinned and keeps DOPE in eval mode.
    guard_sha = "07de5bc34f181e8ef14c3e5cdc20716716b2cc7dd418792f82f3db1d0d576f9c"
    repo_root = Path(__file__).resolve().parents[3]
    guard_path = repo_root / "python/src/jope/north_star/dope_freeze_guard.py"
    if sha256(guard_path) != guard_sha:
        raise RuntimeError("DOPE freeze-guard source mismatch")
    guard_spec = importlib.util.spec_from_file_location("r10_dope_guard", guard_path)
    assert guard_spec is not None and guard_spec.loader is not None
    guard = importlib.util.module_from_spec(guard_spec)
    sys.modules[guard_spec.name] = guard
    guard_spec.loader.exec_module(guard)
    guard.lock_dope_weights_and_eval_mode(net)
    dope_before = guard.snapshot_dope_weights(net)
    if dope_before.composite_sha256 != DOPE_SHA256:
        raise RuntimeError("initial DOPE checksum differs from the frozen R882 pin")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "PASS",
                    "signed_bundle_digest": "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc",
                    "model_loaded_strict": True,
                    "dope_sha256": dope_before.composite_sha256,
                    "train_payloads_opened": 0,
                    "device": "cpu",
                    "threads": torch.get_num_threads(),
                    "nice": os.nice(0),
                },
                sort_keys=True,
            )
        )
        return 0
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)

    rows, pools, tables = load_train_packet(input_dir)
    context_ids = [row["context_sha256"] for row in rows]
    tokens, masks = prepare_tokens(net, tables, context_ids)
    indices_fn = load_training_indices()
    losses = []
    net.train()
    guard.lock_dope_weights_and_eval_mode(net)
    for step in range(OPTIMIZER_UPDATES):
        wait_for_cpu_headroom()
        np.random.seed(SEED + step)
        order = np.random.permutation(context_ids).tolist()
        selections = {
            cid: indices_fn(pools[cid], SOURCE_SLATE_SIZE).astype(np.int64)
            for cid in order
        }
        optimizer.zero_grad(set_to_none=True)
        step_total = step_regression = step_causal = 0.0
        chunks = [
            order[start : start + MICROBATCH]
            for start in range(0, len(order), MICROBATCH)
        ]
        for chunk in chunks:
            wait_for_cpu_headroom()
            selected = [selections[cid] for cid in chunk]
            baseline, ablated, fitness, gate = predict_batch(
                net, pools, tokens, masks, chunk, selected
            )
            targets = torch.where(gate, fitness, torch.full_like(fitness, 2.0))
            pairs = [
                choose_training_pair(fitness[i], gate[i]) for i in range(len(chunk))
            ]
            best = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
            worst = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
            total, regression, causal = training_loss(
                baseline, ablated, targets, best, worst
            )
            weight = len(chunk) / TRAIN_CONTEXTS
            (total * weight).backward()
            step_total += float(total.detach()) * weight
            step_regression += float(regression.detach()) * weight
            step_causal += float(causal.detach()) * weight
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        )
        optimizer.step()
        guard.verify_dope_frozen_unchanged(net, dope_before)
        losses.append(
            {
                "step": step + 1,
                "combined_loss": step_total,
                "regression_loss": step_regression,
                "causal_hinge": step_causal,
                "gradient_norm_before_clip": grad_norm,
                "contexts": TRAIN_CONTEXTS,
            }
        )
        print(json.dumps(losses[-1], sort_keys=True), flush=True)

    guard.verify_dope_frozen_unchanged(net, dope_before)
    candidate = args.output / "r10-landmark-candidate.safetensors"
    candidate_state = independent_state_for_export(net.state_dict())
    save_file(candidate_state, str(candidate))
    result = {
        "schema": "jope.rnk10-m5-causal-candidate/v1",
        "status": "TRAINED_PENDING_M5_AND_ELO",
        "source_commit": SOURCE_COMMIT,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "starting_signed_bundle_digest": "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc",
        "starting_signed_package_sha256": SIGNED_PACKAGE_SHA256,
        "starting_jope_weights_sha256": SIGNED_JOPE_SHA256,
        "starting_dope_weights_file_sha256": SIGNED_DOPE_FILE_SHA256,
        "signing_message_sha256": SIGNING_MESSAGE_SHA256,
        "signing_receipt_sha256": SIGNING_RECEIPT_SHA256,
        "candidate_weights_sha256": sha256(candidate),
        "dope_sha256": dope_before.composite_sha256,
        "train_contexts": TRAIN_CONTEXTS,
        "train_families": TRAIN_FAMILIES,
        "sealed_payloads_opened": 0,
        "validation_payloads_opened": 0,
        "trainable_parameters": count,
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "updates": OPTIMIZER_UPDATES,
        },
        "seed": SEED,
        "source_slate_size": SOURCE_SLATE_SIZE,
        "microbatch": MICROBATCH,
        "auxiliary_weight": 0.1,
        "auxiliary_margin": 0.04208,
        "losses": losses,
        "device": "cpu",
        "host": socket.gethostname(),
        "nice": os.nice(0),
        "torch_threads": torch.get_num_threads(),
        "source_pins": source_pins,
        "source_automatic_promotion_sha256": "37e50568fd097944cf7125de340a19da3dbf27f7be80f47e28fb93b6cc24d16d",
        "source_device_pool_cache_sha256": "cdcb1d672a643716ffba3f2c20aa75d5ae0b19766db616ccd6f6ad65a9adbd76",
        "dope_freeze_guard_sha256": guard_sha,
        "admission_and_val_reg_qualification": "pending independent rnk02 exact M5 replay and dev-fold Elo",
    }
    (args.output / "candidate-receipt.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate),
                "sha256": result["candidate_weights_sha256"],
                "updates": OPTIMIZER_UPDATES,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
