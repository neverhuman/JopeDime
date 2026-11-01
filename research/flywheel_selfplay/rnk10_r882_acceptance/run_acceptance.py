#!/usr/bin/env python3
"""One explicit, CPU-only research acceptance read for the pinned r882 model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RNK10_SOURCE_ROOT", Path(__file__).resolve().parents[3])).resolve()
H49_DIR = ROOT / "research/flywheel_selfplay/h49"
INPUT_DIR = Path("/home/ubuntu/jope-data/flywheel-full-20261030/input")
CHECKPOINT_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
RUN_DIR = Path(os.environ.get("RNK10_RUN_DIR", "/home/ubuntu/jope-data/RNK10-t2115-run-20261101")).resolve()
WEIGHTS = RUN_DIR / "round-882-state.safetensors"
IMPORT_RECEIPT = RUN_DIR / "round-882-import.json"
SOURCE_MANIFEST_SHA256 = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
SELECTED_SHA256 = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
CPU_FUSED_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
MODEL_SOURCE_SHA256 = "0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a"
ADAPTER_SOURCE_SHA256 = "7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a"
MICRO_ROOT = Path("/home/ubuntu/JopeDimeMicro")
MODEL_SUPPORT_FILES = {
    "assets/models/dope_mlc_v3_multitask_best.pt": "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952",
    "assets/models/flywheel_winner_pool_v3a_q25_train_only_430.pt": "6f1646afce6be9e88eab289cda0c8b5edec3da13a373518709183dbbfff856b9",
    "assets/models/h40b_round2_dope_self_contained_10tok.pt": "3a1b62754b372a2572b5664929bfa8706da654446a35122d3a8b7702fc2b25db",
    "assets/projections/frozen_stage1_stage2_inference_params_v3.pt": "136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d",
    "python/research/levels/multilevel_model.py": "39ce38d17a87a2f9cf1a02dbad6efb6606100d46c66ab00fcd56c0416e9643b6",
    "python/research/levels/budget_planner.py": "ae76dd8c587c00d28e57cb152de6719f77cc516c660819d7990823c6a9ff1e4f",
    "python/research/levels/building_block_landmarks.py": "fc2cc82ec4c654a512f592211e21a940a2c7a5baf3e002fe5a1e7649fdfb9186",
    "python/research/levels/multiclass_head.py": "d4c9e750330487b4c9c2cd6c702ed58201eb9322735b911e45ec7cff883f4cda",
    "python/model.py": MODEL_SOURCE_SHA256,
    "python/phase12_multimodal_model.py": ADAPTER_SOURCE_SHA256,
    "python/rust_evaluator.py": "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900",
}
SCALE = 0.4208
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 214_001
DEADLINE_UTC = dt.datetime.fromisoformat("2026-11-01T07:00:00+00:00")
MANIFEST = Path(__file__).with_name("evaluation_manifest.json")
LEDGER = RUN_DIR / "outcome_ledger.json"
OUTPUT = RUN_DIR / "acceptance-result.json"
SMOKE_OUTPUT = RUN_DIR / "model-smoke.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_family_mean(deltas: dict[str, float]) -> list[float]:
    families = sorted(deltas)
    if len(families) != 19:
        raise ValueError("the registered DL-Val family roster must contain 19 families")
    values = np.asarray([deltas[family] for family in families], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = np.mean(values[rng.integers(0, len(values), len(values))])
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def top4_utility(scores: np.ndarray, pool: dict) -> tuple[float, list[int]]:
    gate = pool["gate"]
    fitness = pool["fitness"]
    eligible = np.flatnonzero(gate[4:]) + 4
    if len(eligible) < 4:
        raise ValueError("DL-Val context has fewer than four gate-passing unrevealed proposals")
    chosen = eligible[np.argsort(scores[eligible], kind="stable")[:4]]
    indexes = np.concatenate((np.arange(4), chosen))
    utilities = np.where(gate[indexes], -fitness[indexes], -2.0)
    return float(np.max(utilities)), chosen.astype(int).tolist()


def paired_kl(guided_cost: np.ndarray, anchor_cost: np.ndarray, gate: np.ndarray) -> float:
    eligible = np.flatnonzero(gate[4:]) + 4
    if not len(eligible):
        raise ValueError("DL-Val context has no gate-passing unrevealed proposals")
    q_logits = -guided_cost[eligible].astype(np.float64) / SCALE
    p_logits = -anchor_cost[eligible].astype(np.float64) / SCALE
    q_logits -= np.max(q_logits)
    p_logits -= np.max(p_logits)
    q = np.exp(q_logits)
    p = np.exp(p_logits)
    q /= np.sum(q)
    p /= np.sum(p)
    return float(np.sum(q * (np.log(q) - np.log(p))))


def predict_pair(net, tokens, masks, pool: dict) -> tuple[np.ndarray, np.ndarray]:
    cid = pool["rec"]["context_sha256"]
    programs = pool["program"]
    proj16 = pool["proj16"]
    batch, slate = 1, len(programs)
    device = torch.device("cpu")
    tensors = {
        "ctx_feat": torch.zeros((batch, 584), dtype=torch.float32, device=device),
        "prog_412": torch.as_tensor(programs[None, :, :], dtype=torch.float32, device=device),
        "proj_16": torch.as_tensor(proj16[None, :, :], dtype=torch.float32, device=device),
        "base_prior_cost": torch.zeros((batch, slate), dtype=torch.float32, device=device),
        "scale": torch.full((batch,), SCALE, dtype=torch.float32, device=device),
        "sos_prior": torch.zeros((batch, slate, 2), dtype=torch.float32, device=device),
        "mm_tokens_10x32": tokens[cid].unsqueeze(0),
        "mm_mask_10": masks[cid].unsqueeze(0),
        "revealed_indices": torch.arange(4, device=device).reshape(1, -1),
        "revealed_raw_f": torch.as_tensor(pool["fitness"][:4][None, :], dtype=torch.float32, device=device),
        "revealed_gate": torch.as_tensor(pool["gate"][:4][None, :], dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        guided = net.adapter_10tok.predict_costs(**tensors)["total_cost"][0].float().cpu().numpy()
        anchor_tensors = {
            key: value for key, value in tensors.items()
            if key not in ("mm_tokens_10x32", "mm_mask_10")
        }
        anchor = net.adapter_10tok.anchor.predict_costs(
            **anchor_tensors,
            enable_prior_correction=True,
            enable_struct_kernel=True,
            enable_cross_attn=True,
            enable_ucb=False,
        )["total_cost"][0].float().cpu().numpy()
    return guided, anchor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-eval", action="store_true", help="required; blocks production use")
    parser.add_argument("--smoke-only", action="store_true", help="load the pinned candidate without opening evaluation outcomes")
    args = parser.parse_args()
    if not args.research_eval and not args.smoke_only:
        parser.error("refusing outcome access without explicit --research-eval")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise SystemExit("effective nice value must be at least 10")
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("refusing when CUDA is visible")
    if OUTPUT.exists():
        raise SystemExit("refusing to repeat/overwrite the registered one-use evaluation")
    if args.smoke_only and SMOKE_OUTPUT.exists():
        raise SystemExit("refusing to repeat/overwrite the model smoke")
    manifest = json.loads(MANIFEST.read_text())
    ledger = json.loads(LEDGER.read_text())
    manifest_sha = sha256(MANIFEST)
    if manifest.get("source_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise SystemExit("manifest candidate pin mismatch")
    if manifest.get("source_manifest_file_sha256") != SOURCE_MANIFEST_SHA256 or manifest.get("selected_roster_sha256") != SELECTED_SHA256:
        raise SystemExit("manifest input pin mismatch")
    if manifest.get("runner_sha256") != sha256(Path(__file__)):
        raise SystemExit("runner source SHA differs from the preregistration")
    if manifest.get("status") != "PRE_REGISTERED_ONE_SHOT":
        raise SystemExit("registered evaluation is closed; refusing another run")
    if manifest.get("attempt_index") != 2 or manifest.get("evaluation_id") != "RNK10-T2210-R882-CPU-02":
        raise SystemExit("runner is restricted to the owner-authorized t2210 attempt 2")
    if ledger.get("status") != "UNUSED" or ledger.get("manifest_sha256") != manifest_sha:
        raise SystemExit("ledger is not the unused ledger bound to this manifest")
    attempts = ledger.get("attempts", [])
    prior_attempts = manifest.get("prior_attempts", [])
    if (
        len(attempts) != 2
        or len(prior_attempts) != 1
        or attempts[0].get("status") != "FAILED_CONSUMED"
        or attempts[0].get("terminal_outcomes_scored") != 0
        or attempts[1].get("attempt_index") != 2
        or attempts[1].get("status") != "UNUSED"
        or attempts[0].get("manifest_sha256") != prior_attempts[0].get("manifest_sha256")
        or attempts[0].get("runner_sha256") != prior_attempts[0].get("runner_sha256")
        or prior_attempts[0].get("source_checkpoint_sha256") != CHECKPOINT_SHA256
        or prior_attempts[0].get("source_pointer_sha256") != manifest.get("source_pointer_sha256")
        or prior_attempts[0].get("selected_roster_sha256") != SELECTED_SHA256
    ):
        raise SystemExit("ledger does not contain the authorized attempt-1 failure and unused attempt 2")
    if dt.datetime.now(dt.timezone.utc) >= DEADLINE_UTC:
        raise SystemExit("owner-authorized one-shot deadline has expired")
    receipt = json.loads(IMPORT_RECEIPT.read_text())
    if (
        receipt.get("research_only") is not True
        or receipt.get("production_bundle") is not False
        or receipt.get("source_checkpoint_sha256") != CHECKPOINT_SHA256
        or sha256(WEIGHTS) != receipt.get("imported_safetensors_sha256")
    ):
        raise SystemExit("research importer receipt or derived weights do not match the preregistration")
    if sha256(INPUT_DIR / "source_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise SystemExit("DL-Val source manifest file SHA mismatch")
    if sha256(INPUT_DIR / "selected.json") != SELECTED_SHA256:
        raise SystemExit("DL-Val selected-roster SHA mismatch")
    if sha256(INPUT_DIR / "cpu_exact_fused.py") != CPU_FUSED_SHA256:
        raise SystemExit("pinned CPU fused source SHA mismatch")
    if sha256(MICRO_ROOT / "python/model.py") != MODEL_SOURCE_SHA256:
        raise SystemExit("pinned Flywheel model source SHA mismatch")
    if sha256(MICRO_ROOT / "python/phase12_multimodal_model.py") != ADAPTER_SOURCE_SHA256:
        raise SystemExit("pinned adapter source SHA mismatch")
    if manifest.get("model_support_files") != MODEL_SUPPORT_FILES:
        raise SystemExit("manifest model support inventory differs from runner pins")
    for relative_path, expected in manifest.get("evaluator_sources", {}).items():
        if sha256(ROOT / relative_path) != expected:
            raise SystemExit(f"pinned evaluator source SHA mismatch: {relative_path}")
    for relative_path, expected in manifest.get("runtime_source_files", {}).items():
        if sha256(ROOT / relative_path) != expected:
            raise SystemExit(f"pinned runtime source SHA mismatch: {relative_path}")
    for relative_path, expected in MODEL_SUPPORT_FILES.items():
        if sha256(MICRO_ROOT / relative_path) != expected:
            raise SystemExit(f"pinned model support SHA mismatch: {relative_path}")
    for optional_path in ("expanded_gpu_bank_1766.pt", "distilled_stage1a_bridge_real_s3.pt"):
        if (INPUT_DIR / optional_path).exists():
            raise SystemExit(f"unregistered model sidecar is present: {optional_path}")
    selected_rows = json.loads((INPUT_DIR / "selected.json").read_text()).get("dev", [])
    manifest_rows = manifest.get("contexts", [])
    by_cid = {row.get("context_sha256"): row for row in manifest_rows}
    if len(by_cid) != 91 or len(selected_rows) != 91:
        raise SystemExit("manifest and selected DL-Val rosters must each have 91 unique contexts")
    if {row.get("context_sha256") for row in selected_rows} != set(by_cid):
        raise SystemExit("selected context identities differ from manifest")
    for row in selected_rows:
        pinned = by_cid[row["context_sha256"]]
        if any(row.get(key) != pinned.get(key) for key in (
            "dataset_id", "family_id", "dl_split", "task", "pool_sha256", "train_only_jtt1_sha256"
        )):
            raise SystemExit("selected metadata differs from manifest roster")

    sys.path.insert(0, str(H49_DIR))
    score_spec = importlib.util.spec_from_file_location("rnk10_h49_score", H49_DIR / "score.py")
    if score_spec is None or score_spec.loader is None:
        raise SystemExit("cannot load the pinned DL-Val packet verifier")
    score = importlib.util.module_from_spec(score_spec)
    sys.modules[score_spec.name] = score
    score_spec.loader.exec_module(score)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    h48 = score.h48
    device = torch.device("cpu")
    fused_spec = importlib.util.spec_from_file_location(
        "rnk10_fused", INPUT_DIR / "cpu_exact_fused.py"
    )
    if fused_spec is None or fused_spec.loader is None:
        raise SystemExit("cannot load the pinned CPU fused-model source")
    fused = importlib.util.module_from_spec(fused_spec)
    fused_spec.loader.exec_module(fused)
    guard = h48.base.guard
    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for name, parameter in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor."):
            parameter.requires_grad_(True)
    guard.lock_dope_weights_and_eval_mode(net)
    state = load_file(str(WEIGHTS), device="cpu")
    net.load_state_dict(state, strict=True)
    if h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA256) != DOPE_SHA256:
        raise SystemExit("DOPE guard failed")
    net.eval()

    if args.smoke_only:
        smoke = {
            "schema": "jope.rnk10.r882-model-smoke/v1",
            "evaluation_id": manifest["evaluation_id"],
            "manifest_sha256": manifest_sha,
            "runner_sha256": sha256(Path(__file__)),
            "source_checkpoint_sha256": CHECKPOINT_SHA256,
            "imported_weights_sha256": receipt["imported_safetensors_sha256"],
            "dope_sha256": DOPE_SHA256,
            "model_loaded_strict": True,
            "device": "cpu",
            "threads": 1,
            "optimizer_updates": 0,
            "evaluation_payloads_opened": 0,
            "four_gates": "NOT_RUN",
        }
        SMOKE_OUTPUT.write_text(json.dumps(smoke, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"smoke": str(SMOKE_OUTPUT), "evaluation_payloads_opened": 0}, sort_keys=True))
        return 0

    import psutil

    while psutil.cpu_percent(interval=1) > 85:
        if dt.datetime.now(dt.timezone.utc) >= DEADLINE_UTC:
            raise SystemExit("deadline expired while waiting for CPU <=85%")
        print("host CPU above 85%; waiting", flush=True)
        time.sleep(5)
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    ledger.update({"status": "RUNNING", "run_count": 2, "active_attempt": 2, "started_utc": started_utc, "manifest_sha256": manifest_sha})
    ledger["attempts"][1].update({"status": "RUNNING", "started_utc": started_utc})
    temp_ledger = LEDGER.with_suffix(".json.new")
    temp_ledger.write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n")
    temp_ledger.replace(LEDGER)
    dev_rows, pools, tables = score.load_validation_packet(INPUT_DIR)
    if len(dev_rows) != 91 or len({row["family_id"] for row in dev_rows}) != 19:
        raise SystemExit("DL-Val inventory changed")
    if any(row.get("dl_split") != "DL-Val" for row in dev_rows):
        raise SystemExit("non-DL-Val row entered the registered evaluation")
    tokens, masks = h48.h46.make_tokens(net, tables, device)

    guided_context, anchor_context, context_kl = {}, {}, {}
    terminal_outcomes: set[tuple[str, int]] = set()
    for index, row in enumerate(dev_rows, 1):
        cid = row["context_sha256"]
        guided_cost, anchor_cost = predict_pair(net, tokens, masks, pools[cid])
        guided_context[cid], guided_selected = top4_utility(guided_cost, pools[cid])
        anchor_context[cid], anchor_selected = top4_utility(anchor_cost, pools[cid])
        terminal_outcomes.update((cid, idx) for idx in set(guided_selected + anchor_selected))
        context_kl[cid] = paired_kl(guided_cost, anchor_cost, pools[cid]["gate"])
        while psutil.cpu_percent(interval=0.2) > 85:
            if dt.datetime.now(dt.timezone.utc) >= DEADLINE_UTC:
                raise SystemExit("deadline expired while waiting for CPU <=85%")
            time.sleep(5)
        if dt.datetime.now(dt.timezone.utc) >= DEADLINE_UTC:
            raise SystemExit("owner-authorized deadline expired during scoring")
        if index % 16 == 0:
            print(f"scored {index}/91 registered DL-Val contexts", flush=True)

    family_delta: dict[str, list[float]] = {}
    family_guided: dict[str, list[float]] = {}
    family_anchor: dict[str, list[float]] = {}
    for row in dev_rows:
        cid, family = row["context_sha256"], row["family_id"]
        family_delta.setdefault(family, []).append(guided_context[cid] - anchor_context[cid])
        family_guided.setdefault(family, []).append(guided_context[cid])
        family_anchor.setdefault(family, []).append(anchor_context[cid])
    family_means = {family: float(np.mean(vals)) for family, vals in family_delta.items()}
    interval = bootstrap_family_mean(family_means)

    elo_path = ROOT / "python/src/jope/north_star/elo_tracker.py"
    elo_spec = importlib.util.spec_from_file_location("rnk10_elo_tracker", elo_path)
    if elo_spec is None or elo_spec.loader is None:
        raise SystemExit("cannot load the pinned family-grouped Elo implementation")
    elo = importlib.util.module_from_spec(elo_spec)
    sys.modules[elo_spec.name] = elo
    elo_spec.loader.exec_module(elo)
    records = [
        elo.DatasetEvalRecord(
            dataset_id=row["context_sha256"],
            family_id=row["family_id"],
            task_type="regression",
            scores_by_contender={
                "anchor": anchor_context[row["context_sha256"]],
                "r882-guided": guided_context[row["context_sha256"]],
            },
        )
        for row in dev_rows
    ]
    elo_result = elo.compute_family_grouped_bradley_terry_elo(
        records, anchor_contender="anchor", anchor_elo=1500.0
    )
    max_kl = max(context_kl.values())
    median_family_delta = float(np.median(list(family_means.values())))
    elo_guided = float(elo_result["ratings"]["r882-guided"])
    gates = {
        "max_conditional_base_kl_lte_0_35": max_kl <= 0.35,
        "median_family_mean_delta_gt_0": median_family_delta > 0.0,
        "family_bootstrap_97_5pct_lower_bound_gt_0": interval[0] > 0.0,
        "family_grouped_elo_gt_1500": elo_guided > 1500.0,
    }
    if len(terminal_outcomes) > ledger["max_unique_terminal_outcomes"]:
        raise SystemExit("terminal outcome ledger budget exceeded")
    result = {
        "schema": "jope.rnk10.r882-guided-acceptance/v1",
        "evaluation_id": manifest["evaluation_id"],
        "attempt_index": manifest["attempt_index"],
        "research_only": True,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "imported_weights_sha256": receipt["imported_safetensors_sha256"],
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "selected_roster_sha256": SELECTED_SHA256,
        "dope_sha256": DOPE_SHA256,
        "cohort": {"split": "DL-Val", "contexts": len(dev_rows), "families": len(family_means)},
        "evaluation_identity": manifest["evaluation_identity"],
        "manifest_sha256": manifest_sha,
        "runner_sha256": sha256(Path(__file__)),
        "evaluator_source_commit": manifest["evaluator_source_commit"],
        "terminal_unique_candidate_outcomes": len(terminal_outcomes),
        "comparison": "r882-guided top-4 vs its frozen anchor's top-4; first four slots revealed; gate-failing proposals excluded from both supports",
        "kl": {"distribution": "softmax(-cost / 0.4208) on gate-passing unrevealed proposals", "max_conditional_base_kl": max_kl, "per_context": context_kl},
        "utility": {
            "median_family_mean_delta": median_family_delta,
            "family_bootstrap_97_5pct_ci": interval,
            "family_mean_delta": family_means,
            "guided_family_utility": {family: float(np.mean(vals)) for family, vals in family_guided.items()},
            "anchor_family_utility": {family: float(np.mean(vals)) for family, vals in family_anchor.items()},
            "family_grouped_elo": elo_result,
        },
        "gates": gates,
        "four_gate_diagnostic_all_pass": all(gates.values()),
        "qualified_for_rnk10_acceptance": False,
        "scope_note": "Known reused DL-Val development cohort; r882 has prior validation exposure. This one-shot result is not independent generalization evidence and does not close M5, S4, S2, Q1, or production qualification.",
        "sealed_holdout_meta_test_payloads_opened": 0,
        "optimizer_updates": 0,
        "device": "cpu",
        "threads": 1,
    }
    OUTPUT.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    finished_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    ledger["attempts"][1].update({
        "status": "CONSUMED",
        "finished_utc": finished_utc,
        "result_sha256": sha256(OUTPUT),
        "terminal_outcomes_scored": len(terminal_outcomes),
        "four_gates": gates,
    })
    ledger.update({
        "status": "CONSUMED",
        "run_count": 2,
        "active_attempt": None,
        "finished_utc": finished_utc,
        "result_sha256": sha256(OUTPUT),
        "unique_terminal_outcomes_read": len(terminal_outcomes),
        "four_gates": gates,
    })
    temp_ledger.write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n")
    temp_ledger.replace(LEDGER)
    print(json.dumps({"output": str(OUTPUT), "gates": gates, "qualified": all(gates.values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
