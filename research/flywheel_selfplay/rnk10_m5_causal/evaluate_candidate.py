#!/usr/bin/env python3
"""Candidate-loading M5 sensitivity and DL-Val Elo research runners."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import subprocess
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from safetensors.numpy import load_file as load_numpy
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MICRO = Path("/home/ubuntu/JopeDimeMicro")
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
CPU_FUSED_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
GOLDEN_INDEX_SHA256 = "27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393"
TRAIN_PACKET_SHA256 = "a2fc16d2a8112dd6966bcfc34a923a29cf727a3a91d497a74f56b300e6cc0bbd"
SOURCE_MANIFEST_FILE_SHA256 = (
    "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
)
SELECTED_ROSTER_SHA256 = (
    "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
)
DEV_ROSTER_SHA256 = "eedd7e98eb02e625f75ea71245a32d8765b75610f0b7f56689088a10b0a08193"
SIGNED_R882_HASHES = {
    "jope.safetensors": "ff3f9ea5d76fa6c13cd74d40a9effbf7e9c891d98746dc542d851726b9e3138c",
    "dope.safetensors": "0f5e98fa29cf9c45bfaac5a3cf001b13a0de37d7b6e154c582d7ec689b21a84d",
    "manifest.json": "9a83b2225f57ee451b4813790dad3cb4c41249d4f5a14f7c80e2c0e41540891f",
    "release-catalog.json": "3af56a817bfae39b70f0b2f60668fe88fae178650ee596e1ba7a180acba1e0c6",
    "r882.jopedime": "7523af9700e21325a69f9f71b8b60eb4897d46f3948546e4024010b1ce129a60",
    "manifest.sig": "d35ac4dcdeba1aae42d9ad4469f4681c83ef57df969939ad0101215524a1a780",
    "signing-message.bin": "45e21d576cec91b5171bb8e279ef24fef761456fa916e4de1ed8ed2b8bd6eba5",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serving_selection(costs: np.ndarray) -> np.ndarray:
    """The unchanged CPU M5 4+8 serving rule: best four of eight proposals."""
    values = np.asarray(costs, dtype=np.float64).reshape(-1)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("M5 serving selection requires twelve finite slate costs")
    return np.argsort(values[4:], kind="stable")[:4]


def authenticate_candidate_inputs(args) -> None:
    if digest(args.train_input / "source_manifest.json") != SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError(
            "candidate train source-manifest bytes differ from their pin"
        )
    if digest(args.train_input / "selected_train.json") != TRAIN_PACKET_SHA256:
        raise RuntimeError(
            "candidate selected-train packet bytes differ from their pin"
        )
    train_packet = json.loads((args.train_input / "selected_train.json").read_text())
    if (
        train_packet.get("source_selected_sha256") != SELECTED_ROSTER_SHA256
        or train_packet.get("source_manifest_sha256") != SOURCE_MANIFEST_FILE_SHA256
        or train_packet.get("train_contexts") != 1219
        or train_packet.get("train_families") != 205
        or train_packet.get("split") != "DL-Train"
    ):
        raise RuntimeError("candidate train packet metadata identity mismatch")
    for name, expected in SIGNED_R882_HASHES.items():
        if digest(args.signed_r882 / name) != expected:
            raise RuntimeError(f"signed R882 incumbent asset hash mismatch: {name}")
    manifest = json.loads((args.signed_r882 / "manifest.json").read_text())
    if (
        manifest.get("architecture") != "dope-fused-ranker/h40b"
        or manifest.get("jope_weights_sha256") != SIGNED_R882_HASHES["jope.safetensors"]
        or manifest.get("dope_weights_sha256") != SIGNED_R882_HASHES["dope.safetensors"]
        or manifest.get("bundle_digest")
        != "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc"
    ):
        raise RuntimeError("signed R882 manifest identity mismatch")
    verified = subprocess.run(
        [
            "cosign",
            "verify-blob",
            "--insecure-ignore-tlog",
            "--key",
            str(ROOT / "docs/release/signing.pub"),
            "--signature",
            str(args.signed_r882 / "manifest.sig"),
            str(args.signed_r882 / "signing-message.bin"),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if verified.returncode:
        raise RuntimeError("signed R882 signature verification failed")


def wait_for_cpu_headroom() -> None:
    import psutil

    while psutil.cpu_percent(interval=1.0) > 85:
        print("host CPU above 85%; waiting at context boundary", flush=True)
        time.sleep(5)


def load_net(input_dir: Path, weights: Path):
    fused_path = input_dir / "cpu_exact_fused.py"
    if digest(fused_path) != CPU_FUSED_SHA256:
        raise RuntimeError("pinned CPU fused model source mismatch")
    spec = importlib.util.spec_from_file_location("rnk10_candidate_fused", fused_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned fused model")
    fused = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fused
    spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    net.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    sys.path.insert(0, str(ROOT / "python/src"))
    from jope.north_star import dope_freeze_guard

    dope_freeze_guard.lock_dope_weights_and_eval_mode(net)
    if dope_freeze_guard.snapshot_dope_weights(net).composite_sha256 != DOPE_SHA256:
        raise RuntimeError("candidate frozen DOPE identity mismatch")
    net.eval()
    return net


def score_tensor_batch(net, rows, pools, tokens, masks):
    scores = {}
    for index, row in enumerate(rows, 1):
        if index % 8 == 0:
            wait_for_cpu_headroom()
        cid = row["context_sha256"]
        pool = pools[cid]
        programs = pool["program"]
        args = {
            "ctx_feat": torch.zeros((1, 584), dtype=torch.float32),
            "prog_412": torch.as_tensor(programs[None], dtype=torch.float32),
            "proj_16": torch.as_tensor(pool["proj16"][None], dtype=torch.float32),
            "base_prior_cost": torch.zeros((1, len(programs)), dtype=torch.float32),
            "scale": torch.full((1,), 0.4208, dtype=torch.float32),
            "sos_prior": torch.zeros((1, len(programs), 2), dtype=torch.float32),
            "mm_tokens_10x32": tokens[cid].unsqueeze(0).float(),
            "mm_mask_10": masks[cid].unsqueeze(0).float(),
            "revealed_indices": torch.arange(4).reshape(1, -1),
            "revealed_raw_f": torch.as_tensor(
                pool["fitness"][:4][None], dtype=torch.float32
            ),
            "revealed_gate": torch.as_tensor(
                pool["gate"][:4][None], dtype=torch.float32
            ),
        }
        with torch.no_grad():
            costs = (
                net.adapter_10tok.predict_costs(**args)["total_cost"][0].float().numpy()
            )
        eligible = np.flatnonzero(pool["gate"][4:]) + 4
        chosen = eligible[np.argsort(costs[eligible], kind="stable")[:4]]
        indexes = np.concatenate((np.arange(4), chosen))
        utility = float(
            np.max(np.where(pool["gate"][indexes], -pool["fitness"][indexes], -2.0))
        )
        scores[cid] = utility
    return scores


def dev_elo(args):
    # The existing acceptance loader verifies the pinned manifest, selected roster,
    # every payload hash, and the 91-row / 19-family DL-Val inventory.
    sys.path.insert(0, str(ROOT / "python/src"))
    score_spec = importlib.util.spec_from_file_location(
        "rnk10_candidate_h49_score", ROOT / "research/flywheel_selfplay/h49/score.py"
    )
    if score_spec is None or score_spec.loader is None:
        raise RuntimeError("cannot load the pinned H49 DL-Val loader")
    score = importlib.util.module_from_spec(score_spec)
    sys.modules[score_spec.name] = score
    score_spec.loader.exec_module(score)
    h48 = score.h48
    from jope.north_star.elo_tracker import (
        DatasetEvalRecord,
        compute_family_grouped_bradley_terry_elo,
    )

    pinned_path = HERE / "dev_roster.json"
    if digest(pinned_path) != DEV_ROSTER_SHA256:
        raise RuntimeError("committed DL-Val metadata roster hash mismatch")
    pinned = json.loads(pinned_path.read_text())
    if digest(args.input / "source_manifest.json") != SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError("DL-Val source-manifest bytes differ from their pin")
    if digest(args.input / "selected.json") != SELECTED_ROSTER_SHA256:
        raise RuntimeError("DL-Val selected-roster bytes differ from their pin")
    selected = json.loads((args.input / "selected.json").read_text())
    rows = selected.get("dev")
    if len(rows) != 91 or len({row.get("family_id") for row in rows}) != 19:
        raise RuntimeError("pinned DL-Val metadata roster inventory mismatch")
    if [row.get("context_sha256") for row in rows] != [
        row["context_sha256"] for row in pinned["contexts"]
    ]:
        raise RuntimeError(
            "selected DL-Val identities differ from committed dev roster"
        )
    fields = (
        "context_sha256",
        "dataset_id",
        "dl_split",
        "family_id",
        "task",
        "pool_sha256",
        "train_only_jtt1_sha256",
    )
    for source, lock in zip(rows, pinned["contexts"], strict=True):
        if any(source.get(field) != lock.get(field) for field in fields):
            raise RuntimeError(
                f"selected DL-Val metadata differs from pin: {lock['context_sha256']}"
            )
    rows, pools, tables = score.load_validation_packet(args.input)
    if len(rows) != 91 or len({r["family_id"] for r in rows}) != 19:
        raise RuntimeError(
            "dev roster differs from pinned 91-context/19-family contract"
        )
    if [row["context_sha256"] for row in rows] != [
        row["context_sha256"] for row in pinned["contexts"]
    ]:
        raise RuntimeError("DL-Val loader changed the authenticated row sequence")
    fused_path = args.input / "cpu_exact_fused.py"
    if digest(fused_path) != CPU_FUSED_SHA256:
        raise RuntimeError("staged CPU fused model source differs from its pin")
    spec = importlib.util.spec_from_file_location("rnk10_dev_fused", fused_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load staged fused model")
    fused = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fused
    spec.loader.exec_module(fused)
    candidate = fused.FusedJopeNetwork(
        device=torch.device("cpu"), adapter_variant="h40b"
    )
    incumbent = fused.FusedJopeNetwork(
        device=torch.device("cpu"), adapter_variant="h40b"
    )
    candidate.load_state_dict(load_file(str(args.candidate), device="cpu"), strict=True)
    initial = load_file(str(args.signed_r882 / "jope.safetensors"), device="cpu")
    initial.update(load_file(str(args.signed_r882 / "dope.safetensors"), device="cpu"))
    incumbent.load_state_dict(initial, strict=True)
    guard = h48.base.guard
    for net in (candidate, incumbent):
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        for name, parameter in net.adapter_10tok.named_parameters():
            if not name.startswith("anchor."):
                parameter.requires_grad_(True)
        guard.lock_dope_weights_and_eval_mode(net)
        h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA256)
    candidate.eval()
    incumbent.eval()
    tokens, masks = h48.h46.make_tokens(candidate, tables, torch.device("cpu"))
    cand_scores = score_tensor_batch(candidate, rows, pools, tokens, masks)
    ref_scores = score_tensor_batch(incumbent, rows, pools, tokens, masks)
    records = [
        DatasetEvalRecord(
            dataset_id=row["dataset_id"],
            family_id=row["family_id"],
            task_type="regression",
            scores_by_contender={
                "candidate": cand_scores[row["context_sha256"]],
                "r882": ref_scores[row["context_sha256"]],
            },
        )
        for row in rows
    ]
    ratings = compute_family_grouped_bradley_terry_elo(
        records, anchor_contender="r882", anchor_elo=1500.0
    )
    result = {
        "schema": "jope.rnk10-candidate-dev-elo/v1",
        "scope": "reused DL-Val development cohort; not independent generalization evidence",
        "criterion": "candidate dev-fold Elo exceeds signed R882 incumbent",
        "contexts": 91,
        "families": 19,
        "sealed_payloads_opened": 0,
        "candidate_sha256": digest(args.candidate),
        "candidate_receipt_sha256": args.receipt_sha256,
        "candidate_execution_evidence_sha256": args.evidence_sha256,
        "runner_sha256": args.runner_sha256,
        "source_roster_sha256": "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1",
        "ratings": ratings,
        "candidate_elo_lift": ratings["ratings"]["candidate"]
        > ratings["ratings"]["r882"],
        "qualification": "NOT_CLAIMED",
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


def m5(args):
    index_path = args.fixtures / "index.json"
    if digest(index_path) != GOLDEN_INDEX_SHA256:
        raise RuntimeError("M5 golden fixture index hash mismatch")
    index = json.loads(index_path.read_text())
    if index.get("schema") != "jope.golden-fixtures/v1" or index.get("contexts") != 50:
        raise RuntimeError("M5 fixture index identity mismatch")
    train_ids = {
        row["context_sha256"]
        for row in json.loads((args.train_input / "selected_train.json").read_text())[
            "rows"
        ]
    }
    dev_roster_path = HERE / "dev_roster.json"
    if digest(dev_roster_path) != DEV_ROSTER_SHA256:
        raise RuntimeError("committed DL-Val metadata roster hash mismatch")
    dev_ids = {
        row["context_sha256"]
        for row in json.loads(dev_roster_path.read_text())["contexts"]
    }
    indexed_context_ids = {
        Path(name).stem.rsplit("-", 1)[0]
        for name in index["files"]
        if name.startswith("slates/") and name.endswith("-12.safetensors")
    }
    train_overlap = len(indexed_context_ids & train_ids)
    dev_overlap = len(indexed_context_ids & dev_ids)
    if len(indexed_context_ids) != 50 or train_overlap != 50 or dev_overlap != 0:
        raise RuntimeError(
            "M5 fixtures do not match the pinned in-sample TRAIN roster closure"
        )
    net = load_net(args.input, args.candidate)
    slates = 0
    serving_slates = 0
    landmark_changed_contexts: set[str] = set()
    fitness_changed_contexts: set[str] = set()
    disconnected_contexts: set[str] = set()
    intervention_contexts = {-6.0: set(), 0.0: set(), 6.0: set()}
    finite_cost_values = 0
    intervention_cost_values = 0
    fitness_control_cost_values = 0
    invalid_cost_values = 0
    contexts_seen: set[str] = set()
    for name, spec in sorted(index["files"].items()):
        if not name.startswith("slates/") or not (
            name.endswith("-12.safetensors") or name.endswith("-20.safetensors")
        ):
            continue
        path = args.fixtures / name
        if digest(path) != spec["sha256"]:
            raise RuntimeError(f"M5 slate fixture hash mismatch: {name}")
        data = load_numpy(str(path))
        fields = {key.removeprefix("input."): value for key, value in data.items()}
        width = len(fields["prog_412"]) // 412
        context_id = Path(name).stem.rsplit("-", 1)[0]
        contexts_seen.add(context_id)

        # Fixture fields are the exact serving tensor inputs; apply the same deterministic
        # four-lowest-cost admission rule while varying only landmark token slots.
        def predict(token):
            payload = {
                "ctx_feat": torch.as_tensor(fields["ctx_feat"]).reshape(1, 584),
                "prog_412": torch.as_tensor(fields["prog_412"]).reshape(1, width, 412),
                "proj_16": torch.as_tensor(fields["proj_16"]).reshape(1, width, 16),
                "base_prior_cost": torch.as_tensor(fields["base_prior_cost"]).reshape(
                    1, width
                ),
                "scale": torch.as_tensor(fields["scale"]).reshape(1),
                "sos_prior": torch.as_tensor(fields["sos_prior"]).reshape(1, width, 2),
                "mm_tokens_10x32": torch.as_tensor(token).reshape(1, 10, 32),
                "mm_mask_10": torch.as_tensor(fields["mm_mask_10"]).reshape(1, 10),
                "revealed_indices": torch.as_tensor(fields["revealed_indices"]).reshape(
                    1, 4
                ),
                "revealed_raw_f": torch.as_tensor(fields["revealed_raw_f"]).reshape(
                    1, 4
                ),
                "revealed_gate": torch.as_tensor(fields["revealed_gate"]).reshape(1, 4),
            }
            with torch.no_grad():
                out = (
                    net.adapter_10tok.predict_costs(**payload)["total_cost"]
                    .numpy()
                    .reshape(-1)
                )
            return out, serving_selection(out) if width == 12 else None

        base_costs, base_choice = predict(fields["mm_tokens_10x32"])
        finite_cost_values += int(np.isfinite(base_costs).sum())
        invalid_cost_values += int((~np.isfinite(base_costs)).sum())
        if name.endswith("-12.safetensors"):
            serving_slates += 1
            context_id = Path(name).stem.removesuffix("-12")
            if not np.array_equal(base_choice, np.arange(4)):
                disconnected_contexts.add(context_id)
            for value in (-6.0, 0.0, 6.0):
                altered = np.array(fields["mm_tokens_10x32"], copy=True)
                altered[..., [0, 1, 7, 9], :] = value
                costs, selected = predict(altered)
                intervention_cost_values += int(np.isfinite(costs).sum())
                invalid_cost_values += int((~np.isfinite(costs)).sum())
                if not np.array_equal(base_choice, selected):
                    intervention_contexts[value].add(context_id)
                    landmark_changed_contexts.add(context_id)

            fitness_args = {
                "ctx_feat": torch.as_tensor(fields["ctx_feat"]).reshape(1, 584),
                "prog_412": torch.as_tensor(fields["prog_412"]).reshape(1, width, 412),
                "proj_16": torch.as_tensor(fields["proj_16"]).reshape(1, width, 16),
                "base_prior_cost": torch.as_tensor(fields["base_prior_cost"]).reshape(
                    1, width
                ),
                "scale": torch.as_tensor(fields["scale"]).reshape(1),
                "sos_prior": torch.as_tensor(fields["sos_prior"]).reshape(1, width, 2),
                "mm_tokens_10x32": torch.as_tensor(fields["mm_tokens_10x32"]).reshape(
                    1, 10, 32
                ),
                "mm_mask_10": torch.as_tensor(fields["mm_mask_10"]).reshape(1, 10),
                "revealed_indices": torch.as_tensor(fields["revealed_indices"]).reshape(
                    1, 4
                ),
                "revealed_raw_f": torch.as_tensor(fields["revealed_raw_f"])[
                    [3, 2, 1, 0]
                ].reshape(1, 4),
                "revealed_gate": torch.as_tensor(fields["revealed_gate"]).reshape(1, 4),
            }
            with torch.no_grad():
                fitness_costs = (
                    net.adapter_10tok.predict_costs(**fitness_args)["total_cost"]
                    .numpy()
                    .reshape(-1)
                )
            fitness_control_cost_values += int(np.isfinite(fitness_costs).sum())
            invalid_cost_values += int((~np.isfinite(fitness_costs)).sum())
            if not np.array_equal(base_choice, serving_selection(fitness_costs)):
                fitness_changed_contexts.add(context_id)
        slates += 1
        if slates % 16 == 0:
            wait_for_cpu_headroom()
    if slates != 100 or serving_slates != 50 or len(contexts_seen) != 50:
        raise RuntimeError(
            f"expected 100 slates across 50 contexts and 50 serving slates, found {slates}/{len(contexts_seen)}/{serving_slates}"
        )
    if finite_cost_values != 1600:
        raise RuntimeError(
            f"expected 1,600 baseline cost values, found {finite_cost_values}"
        )
    result = {
        "schema": "jope.rnk10-candidate-m5/v1",
        "candidate_sha256": digest(args.candidate),
        "candidate_receipt_sha256": args.receipt_sha256,
        "candidate_execution_evidence_sha256": args.evidence_sha256,
        "runner_sha256": args.runner_sha256,
        "fixture_index_sha256": GOLDEN_INDEX_SHA256,
        "selection_policy": "unchanged deterministic M5 4+8 serving selection: stable lowest four costs from proposal indices 4..11",
        "contexts": 50,
        "slates": slates,
        "serving_slates": serving_slates,
        "finite_cost_values": finite_cost_values,
        "intervention_cost_values_checked": intervention_cost_values,
        "fitness_control_cost_values_checked": fitness_control_cost_values,
        "invalid_cost_values": invalid_cost_values,
        "landmark_only_changed_admission_contexts": len(landmark_changed_contexts),
        "fitness_changed_admission_contexts": len(fitness_changed_contexts),
        "disconnected_heuristic_mismatch_contexts": len(disconnected_contexts),
        "landmark_intervention_changed_contexts": {
            str(value): len(contexts)
            for value, contexts in intervention_contexts.items()
        },
        "landmark_slots": [0, 1, 7, 9],
        "interventions": [-6.0, 0.0, 6.0],
        "masks_preserved": True,
        "quality_gate_granted": False,
        "Q1_ready": False,
        "speed_verdict": "NOT_MEASURED",
        "sensitivity_scope": "in_sample",
        "overlap": {
            "fixture_contexts_in_train": train_overlap,
            "fixture_contexts_in_dev": dev_overlap,
        },
        "m5_predicate_passed_in_sample": (
            invalid_cost_values == 0
            and len(landmark_changed_contexts) > 0
            and len(fitness_changed_contexts) > 0
            and len(disconnected_contexts) > 0
        ),
        "qualification": "NOT_CLAIMED; report in-sample sensitivity only; dev-fold Elo is the sole lift criterion",
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("m5", "dev-elo"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-input", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--signed-r882", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixtures", type=Path, default=Path("/home/ubuntu/jope-data/golden")
    )
    args = parser.parse_args()
    if args.mode == "dev-elo" and args.train_input is None:
        parser.error(
            "dev-elo requires --train-input with the authenticated candidate train packet"
        )
    if args.train_input is None:
        args.train_input = args.input
    if os.nice(0) < 19:
        raise SystemExit("research candidate evaluation requires nice 19")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    wait_for_cpu_headroom()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite evaluation output: {args.output}")
    receipt_path = args.receipt or args.candidate.parent / "candidate-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("candidate_weights_sha256") != digest(args.candidate)
        or receipt.get("dope_sha256") != DOPE_SHA256
        or receipt.get("train_contexts") != 1219
        or receipt.get("sealed_payloads_opened") != 0
    ):
        raise SystemExit(
            "candidate receipt, weight hash, or frozen DOPE identity mismatch"
        )
    evidence_path = Path(
        receipt.get("training_implementation", {}).get("execution_evidence_file", "")
    )
    if not evidence_path.is_file() or digest(evidence_path) != receipt.get(
        "training_implementation", {}
    ).get("execution_evidence_sha256"):
        raise SystemExit(
            "candidate execution provenance evidence is missing or changed"
        )
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence.get("candidate_weights_sha256") != digest(args.candidate)
        or evidence.get("selected_train_packet_sha256") != TRAIN_PACKET_SHA256
        or receipt.get("training_input_identity", {}).get(
            "selected_train_packet_sha256"
        )
        != TRAIN_PACKET_SHA256
        or receipt.get("training_input_identity", {}).get(
            "source_selected_roster_sha256"
        )
        != SELECTED_ROSTER_SHA256
    ):
        raise SystemExit(
            "candidate execution evidence does not bind the pinned candidate and training roster"
        )
    args.receipt_sha256 = digest(receipt_path)
    args.evidence_sha256 = digest(evidence_path)
    args.runner_sha256 = digest(Path(__file__))
    if args.signed_r882 is None:
        parser.error(
            "both modes require --signed-r882 with the pinned signed incumbent assets"
        )
    authenticate_candidate_inputs(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "m5":
        m5(args)
    else:
        dev_elo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
