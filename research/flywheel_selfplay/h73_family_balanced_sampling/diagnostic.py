#!/usr/bin/env python3
"""Measure family concentration in the frozen r882 DL-Train objective.

This is a descriptive, CPU-only diagnostic. It consumes only the DL-Train
records listed before the `dev` key in the admitted H50 selected.json and the
training-only H67 family-fold roster. It never opens validation or holdout
payloads and performs no optimizer updates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
H48_PATH = ROOT / "research/flywheel_selfplay/h48/run.py"
EXPECTED_CHECKPOINT_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
EXPECTED_DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
EXPECTED_SOURCE_MANIFEST_SHA256 = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
EXPECTED_SOURCE_FILE_SHA256 = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
EXPECTED_CONTEXTS = 1219
EXPECTED_FAMILIES = 205
FOLD_SEED = 4900
THREADS = 4
MAX_RSS_BYTES = 8 * 1024**3
MIN_AVAILABLE_BYTES = 16 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonspace(stream: TextIO) -> str:
    while True:
        value = stream.read(1)
        if not value:
            return ""
        if not value.isspace():
            return value


def _json_string(stream: TextIO, first: str) -> str:
    if first != '"':
        raise ValueError("expected JSON object key")
    chars = [first]
    escaped = False
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("truncated JSON key")
        chars.append(char)
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return json.loads("".join(chars))


def _json_object(stream: TextIO, first: str) -> dict:
    if first != "{":
        raise ValueError("expected a training row object")
    chars = [first]
    depth = 1
    in_string = False
    escaped = False
    while depth:
        char = stream.read(1)
        if not char:
            raise ValueError("truncated training row")
        chars.append(char)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    value = json.loads("".join(chars))
    if not isinstance(value, dict):
        raise ValueError("training row is not a JSON object")
    return value


def read_train_rows_only(path: Path) -> tuple[list[dict], str]:
    """Read only the first top-level `train` array; stop before the next key."""
    digest = hashlib.sha256()

    class HashingReader:
        def __init__(self, inner: TextIO):
            self.inner = inner

        def read(self, count: int = 1) -> str:
            value = self.inner.read(count)
            digest.update(value.encode("utf-8"))
            return value

    with path.open("r", encoding="utf-8") as raw:
        stream = HashingReader(raw)
        if _nonspace(stream) != "{":
            raise ValueError("selected input must start with a JSON object")
        key = _json_string(stream, _nonspace(stream))
        if key != "train":
            raise ValueError("train must be the first selected.json field")
        if _nonspace(stream) != ":" or _nonspace(stream) != "[":
            raise ValueError("selected train field must be an array")
        rows: list[dict] = []
        first = _nonspace(stream)
        if first == "]":
            return rows, digest.hexdigest()
        while True:
            rows.append(_json_object(stream, first))
            separator = _nonspace(stream)
            if separator == "]":
                break
            if separator != ",":
                raise ValueError("expected a training row separator")
            first = _nonspace(stream)
        return rows, digest.hexdigest()


def _read_linux_mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _cpu_totals() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def host_cpu_percent(interval: float = 0.1) -> float:
    before_total, before_idle = _cpu_totals()
    time.sleep(interval)
    after_total, after_idle = _cpu_totals()
    total_delta = after_total - before_total
    idle_delta = after_idle - before_idle
    if total_delta <= 0:
        return 0.0
    return 100.0 * (total_delta - idle_delta) / total_delta


class ResourceGuard:
    def __init__(self) -> None:
        self.max_cpu = 0.0
        self.max_rss = 0
        self.min_available = _read_linux_mem_available()

    def check(self) -> None:
        rss = _rss_bytes()
        available = _read_linux_mem_available()
        self.max_rss = max(self.max_rss, rss)
        self.min_available = min(self.min_available, available)
        if rss >= MAX_RSS_BYTES:
            raise RuntimeError(f"RSS guard reached: {rss} bytes >= 8 GiB")
        if available < MIN_AVAILABLE_BYTES:
            raise RuntimeError(f"MemAvailable below 16 GiB: {available} bytes")
        cpu = host_cpu_percent()
        self.max_cpu = max(self.max_cpu, cpu)
        if cpu > 85.0:
            raise RuntimeError(f"host CPU utilization exceeded 85%: {cpu:.2f}%")


def import_h48():
    spec = importlib.util.spec_from_file_location("h73_h48", H48_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import H48 helpers from {H48_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_train_rows(input_dir: Path, roster_path: Path) -> tuple[list[dict], dict, str]:
    # The committed H67 roster binds this admitted packet. Avoid opening its
    # source_manifest.json because that summary also carries non-training split
    # counts. The selected.json reader below stops at the end of `train`.
    rows, selected_train_prefix_sha256 = read_train_rows_only(input_dir / "selected.json")
    if len(rows) != EXPECTED_CONTEXTS:
        raise RuntimeError(f"expected {EXPECTED_CONTEXTS} DL-Train rows, got {len(rows)}")
    if any(
        row.get("dl_split") != "DL-Train"
        or row.get("source_split") not in {"train", "inner"}
        or row.get("task") != "regression"
        for row in rows
    ):
        raise RuntimeError("non-training or non-regression row in training array")
    ids = [row.get("context_sha256") for row in rows]
    if len(set(ids)) != EXPECTED_CONTEXTS:
        raise RuntimeError("duplicate or missing DL-Train context identity")

    roster = json.loads(roster_path.read_text())
    if (
        roster.get("schema") != "jopedime.w4-active-reveal-training-roster.v1"
        or roster.get("preparation_only") is not True
        or roster.get("source_manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or roster.get("source_manifest_file_sha256") != EXPECTED_SOURCE_FILE_SHA256
        or roster.get("fold_seed") != FOLD_SEED
        or roster.get("fold_count") != 5
        or roster.get("declared_train_contexts") != EXPECTED_CONTEXTS
        or roster.get("declared_train_families") != EXPECTED_FAMILIES
        or roster.get("outcomes_in_manifest") is not False
    ):
        raise RuntimeError("H67 training-only roster identity or scope mismatch")
    assignments = roster.get("context_fold_assignments")
    if not isinstance(assignments, list) or len(assignments) != EXPECTED_CONTEXTS:
        raise RuntimeError("H67 roster must contain exactly 1,219 training assignments")
    assignment_by_id = {item.get("context_sha256"): item for item in assignments}
    if set(assignment_by_id) != set(ids):
        raise RuntimeError("H67 training roster and admitted DL-Train contexts differ")
    if len({item.get("family_group") for item in assignments}) != EXPECTED_FAMILIES:
        raise RuntimeError("H67 resolved family inventory mismatch")
    family_folds = roster.get("family_folds")
    if not isinstance(family_folds, list) or len(family_folds) != 5:
        raise RuntimeError("H67 family fold map is incomplete")
    fold_by_family = {}
    for fold in family_folds:
        fold_id = fold.get("fold")
        for family in fold.get("family_groups", []):
            if family in fold_by_family:
                raise RuntimeError("H67 family group appears in multiple folds")
            fold_by_family[family] = fold_id
    for item in assignments:
        if item.get("fold") != fold_by_family.get(item.get("family_group")):
            raise RuntimeError("H67 context assignment splits or mislabels a family")
    for row in rows:
        item = assignment_by_id[row["context_sha256"]]
        row["family_group"] = item["family_group"]
        row["fold"] = item["fold"]
    return rows, roster, selected_train_prefix_sha256


def _load_train_payloads(h48, input_dir: Path, rows: list[dict], guard: ResourceGuard):
    base = h48.base
    pools: dict[str, dict] = {}
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    w1 = base.evaluator.get_w1_sign_matrix()
    for index, row in enumerate(rows, 1):
        cid = row["context_sha256"]
        pool_path = input_dir / "pools" / f"{cid}.npz"
        table_path = input_dir / "tables" / f"{cid}.jtt1"
        if sha256(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"training pool hash mismatch for {cid}")
        if sha256(table_path) != row["train_only_jtt1_sha256"]:
            raise RuntimeError(f"training-only table hash mismatch for {cid}")
        with np.load(pool_path, allow_pickle=False) as source:
            programs = source["program"].astype(np.float32)
            pids = [value.decode() if isinstance(value, bytes) else str(value) for value in source["program_sha256"].tolist()]
            pools[cid] = {
                "rec": row,
                "pids": pids,
                "program": programs,
                "proj16": programs.astype(np.float64) @ w1 / np.sqrt(412),
                "struct_sig": base.compute_semantic_struct_sig_24d(programs),
                "fitness": source["fitness"].astype(np.float64),
                "logratio": source["logratio"].astype(np.float64),
                "gate": source["gate"].astype(bool),
                "prior_cost": np.zeros(len(programs), dtype=np.float64),
                "scale": 0.4208,
            }
        raw = table_path.read_bytes()
        if len(raw) < 17 or raw[:4] != b"JTT1" or raw[16] != 2:
            raise RuntimeError(f"invalid train-only JTT1 table for {cid}")
        nt, nq, width = base.struct.unpack_from("<III", raw, 4)
        if nq:
            raise RuntimeError(f"unexpected validation rows in train-only table {cid}")
        values = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
        tables[cid] = (values[:, :-1], values[:, -1])
        if index % 8 == 0:
            guard.check()
    return pools, tables


def _build_frozen_model(h48, checkpoint_path: Path, tables: dict, guard: ResourceGuard):
    if sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("offline r882 checkpoint copy hash mismatch")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("round") != 882 or state.get("dope_sha256") != EXPECTED_DOPE_SHA256:
        raise RuntimeError("r882 round/DOPE checkpoint metadata mismatch")
    fused_spec = importlib.util.spec_from_file_location(
        "h73_h50_fused", Path.home() / "jope-data/h50/input/cpu_exact_fused.py"
    )
    if fused_spec is None or fused_spec.loader is None:
        raise RuntimeError("cannot load pinned H50 fused model source")
    fused = importlib.util.module_from_spec(fused_spec)
    sys.modules[fused_spec.name] = fused
    fused_spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for name, parameter in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor."):
            parameter.requires_grad_(True)
    h48.base.guard.lock_dope_weights_and_eval_mode(net)
    net.load_state_dict(state["state_dict"], strict=True)
    if h48.h46.checkpoint_guard(h48.base.guard, net, expected=EXPECTED_DOPE_SHA256) != EXPECTED_DOPE_SHA256:
        raise RuntimeError("frozen DOPE hash mismatch")
    if sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad) != 29_211:
        raise RuntimeError("unexpected trainable adapter parameter count")
    net.eval()
    tokens: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    workspace = h48.base.FusedLandmarkWorkspace(num_threads=THREADS)
    for index, (cid, (x, y)) in enumerate(tables.items(), 1):
        raw_y = y.astype(np.float64)
        normalized_y = ((raw_y - raw_y.mean()) / max(float(raw_y.std()), 1e-8)).astype(np.float32)
        vector, columns, _ = workspace.fit_transform_and_extract_bundle_high(
            h48.base.landmark_float32(x), np.ascontiguousarray(normalized_y)
        )
        column_buffer = np.zeros((64, 17), np.float32)
        count = min(len(columns), 64)
        column_buffer[:count, : min(columns.shape[1], 17)] = columns[:count, :17]
        with torch.no_grad():
            dope, lm = net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(vector, device="cpu").reshape(1, 256),
                torch.as_tensor(column_buffer, device="cpu").reshape(1, 64, 17),
                use_landmarks=True,
            )
            token, mask, _ = net._project_stage1b_tokens_batched_gpu(lm, dope)
        tokens[cid], masks[cid] = token[0].detach(), mask[0].detach()
        if index % 4 == 0:
            guard.check()
    return state, net, tokens, masks


def _context_loss(prediction: torch.Tensor, pool: dict) -> dict:
    target = torch.as_tensor(pool["fitness"].copy(), dtype=torch.float32)
    gate = torch.as_tensor(pool["gate"].copy(), dtype=torch.bool)
    target = torch.where(gate, target, torch.full_like(target, 2.0))
    huber_terms = F.smooth_l1_loss(prediction, target, reduction="none")
    log_pred = F.log_softmax(-prediction / 0.2, dim=-1)
    target_prob = F.softmax(-target / 0.2, dim=-1)
    kl_terms = F.kl_div(log_pred, target_prob, reduction="none")
    candidate_terms = huber_terms + 0.1 * kl_terms
    objective_loss = float(huber_terms.mean() + 0.1 * kl_terms.sum())
    selected = int(torch.argmin(prediction).item())
    selected_gate = bool(pool["gate"][selected])
    selected_utility = -float(pool["fitness"][selected]) if selected_gate else -2.0
    failed = ~pool["gate"]
    return {
        "objective_loss": objective_loss,
        "failure_candidate_loss_mass": float(candidate_terms[torch.as_tensor(failed)].sum() / len(pool["gate"])),
        "all_candidate_loss_mass": float(candidate_terms.sum() / len(pool["gate"])),
        "selected_utility": selected_utility,
        "selected_failure": not selected_gate,
    }


def _gini(values: list[float]) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if not len(array) or float(array.sum()) == 0.0:
        return 0.0
    ranks = np.arange(1, len(array) + 1, dtype=np.float64)
    return float((2 * np.sum(ranks * array) / (len(array) * array.sum())) - (len(array) + 1) / len(array))


def _average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if float(np.std(left_rank)) == 0.0 or float(np.std(right_rank)) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def aggregate_metrics(rows: list[dict], results: dict[str, dict]) -> dict:
    counts = Counter(row["family_group"] for row in rows)
    context_histogram = Counter(counts.values())
    count_values = list(counts.values())
    total_contexts = len(rows)
    top10_context_share = sum(sorted(count_values, reverse=True)[:10]) / total_contexts
    raw_family_mass: dict[str, float] = defaultdict(float)
    balanced_family_mass: dict[str, float] = defaultdict(float)
    total_raw = sum(results[row["context_sha256"]]["objective_loss"] for row in rows)
    weights = {}
    family_count = len(counts)
    for row in rows:
        family = row["family_group"]
        weights[row["context_sha256"]] = total_contexts / (family_count * counts[family])
        raw_family_mass[family] += results[row["context_sha256"]]["objective_loss"]
        balanced_family_mass[family] += (
            weights[row["context_sha256"]] * results[row["context_sha256"]]["objective_loss"]
        )
    total_balanced = sum(balanced_family_mass.values())
    raw_shares = [value / total_raw for value in raw_family_mass.values()]
    balanced_shares = [value / total_balanced for value in balanced_family_mass.values()]
    raw_top = sorted(raw_family_mass.values(), reverse=True)
    balanced_top = sorted(balanced_family_mass.values(), reverse=True)
    raw_top_families = sorted(raw_family_mass, key=raw_family_mass.get, reverse=True)[:10]
    balanced_top_families = sorted(balanced_family_mass, key=balanced_family_mass.get, reverse=True)[:10]
    raw_top_context_share = sum(counts[family] for family in raw_top_families) / total_contexts
    balanced_top_context_share = sum(counts[family] for family in balanced_top_families) / total_contexts
    family_keys = list(raw_family_mass)
    family_profiles = [
        [
            counts[family],
            raw_family_mass[family] / total_raw,
            balanced_family_mass[family] / total_balanced,
            raw_family_mass[family] / counts[family],
            sum(
                results[row["context_sha256"]]["selected_failure"]
                for row in rows
                if row["family_group"] == family
            ),
        ]
        for family in sorted(family_keys, key=raw_family_mass.get, reverse=True)
    ]
    failure_rows = [row for row in rows if results[row["context_sha256"]]["selected_failure"]]
    failure_loss = sum(results[row["context_sha256"]]["objective_loss"] for row in failure_rows)
    failure_candidate_mass = sum(
        results[row["context_sha256"]]["failure_candidate_loss_mass"] for row in rows
    )
    candidate_mass = sum(results[row["context_sha256"]]["all_candidate_loss_mass"] for row in rows)
    fold_summaries = []
    for fold in range(5):
        fold_rows = [row for row in rows if row["fold"] == fold]
        fold_loss = [results[row["context_sha256"]]["objective_loss"] for row in fold_rows]
        fold_families = {row["family_group"] for row in fold_rows}
        fold_failures = sum(results[row["context_sha256"]]["selected_failure"] for row in fold_rows)
        fold_summaries.append({
            "fold": fold,
            "contexts": len(fold_rows),
            "families": len(fold_families),
            "mean_context_loss": float(np.mean(fold_loss)) if fold_loss else None,
            "selected_minus_two_contexts": int(fold_failures),
        })
    return {
        "schema": "jopedime.h73.family-balance-diagnostic.v1",
        "diagnostic_only": True,
        "training_data_only": True,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "dope_sha256": EXPECTED_DOPE_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "fold_seed": FOLD_SEED,
        "fold_count": 5,
        "contexts": total_contexts,
        "resolved_families": family_count,
        "context_count_histogram": {str(k): int(v) for k, v in sorted(context_histogram.items())},
        "contexts_per_family": {
            "min": min(count_values),
            "median": float(np.median(count_values)),
            "max": max(count_values),
            "gini": _gini(count_values),
            "top10_family_context_share": top10_context_share,
        },
        "r882_training_objective": {
            "definition": "H48 CONTROL row loss: mean smooth-L1 on fitness/gate-cost targets plus 0.1 times KL(softmax(-target/0.2) || softmax(-prediction/0.2)); each context is a row.",
            "mean_context_loss": total_raw / total_contexts,
            "total_unweighted_loss_mass": total_raw,
            "raw_family_loss_share": {
                "gini": _gini(raw_shares),
                "top1_family_share": max(raw_shares),
                "top10_family_share": sum(sorted(raw_shares, reverse=True)[:10]),
            "top10_loss_families_context_share": raw_top_context_share,
            "family_size_vs_total_loss_spearman_rho": _spearman(
                [counts[family] for family in family_keys],
                [raw_family_mass[family] for family in family_keys],
            ),
                "median_family_share": float(np.median(raw_shares)),
            },
            "full_cohort_family_balanced_loss_share": {
                "gini": _gini(balanced_shares),
                "top1_family_share": max(balanced_shares),
                "top10_family_share": sum(sorted(balanced_shares, reverse=True)[:10]),
            "top10_loss_families_context_share": balanced_top_context_share,
                "median_family_share": float(np.median(balanced_shares)),
            },
        "per_family_profile_columns": [
            "context_count", "unweighted_loss_share", "family_balanced_loss_share",
            "mean_context_loss", "selected_minus_two_contexts",
        ],
        "per_family_loss_profiles": family_profiles,
            "context_loss_distribution": _distribution([results[row["context_sha256"]]["objective_loss"] for row in rows]),
            "selected_minus_two_failure_contexts": {
                "count": len(failure_rows),
                "fraction": len(failure_rows) / total_contexts,
                "selected_utility": -2.0,
                "summed_context_objective_loss": failure_loss,
                "share_of_unweighted_context_loss_mass": failure_loss / total_raw if total_raw else None,
            },
            "failed_candidate_target_loss_mass": {
                "target_cost": 2.0,
                "signed_mass": failure_candidate_mass,
                "all_candidate_mass": candidate_mass,
                "signed_share": failure_candidate_mass / candidate_mass if candidate_mass else None,
                "note": "Sum of failed-entry smooth-L1 plus its additive KL terms, divided by each pool size; signed because individual KL summands can be negative.",
            },
        },
        "fold_summaries": fold_summaries,
        "interpretation_limit": "The r882 adapter was trained on the full DL-Train cohort. Family folds partition this diagnostic only; they do not create out-of-sample predictions or a causal estimate of gradient dominance.",
        "excluded_data": {"dl_val_payloads_read": 0, "fold_b_payloads_read": 0, "outer_test_payloads_read": 0, "holdout_payloads_read": 0, "sealed_payloads_read": 0},
    }


def run(args: argparse.Namespace) -> dict:
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError("launch with nice -n 19")
    if any(int(os.environ.get(name, "0")) > THREADS for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise RuntimeError("OMP/MKL/OPENBLAS thread limits must be <=4")
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    guard = ResourceGuard()
    guard.check()
    h48 = import_h48()
    rows, roster, selected_train_prefix_sha256 = load_train_rows(args.input, args.roster)
    pools, tables = _load_train_payloads(h48, args.input, rows, guard)
    state, net, tokens, masks = _build_frozen_model(h48, args.checkpoint, tables, guard)
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device("cpu"))
    results: dict[str, dict] = {}
    by_size: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        cid = row["context_sha256"]
        by_size[len(pools[cid]["pids"])].append(cid)
    for size, contexts in sorted(by_size.items()):
        for start in range(0, len(contexts), 4):
            batch = contexts[start : start + 4]
            indexes = [np.arange(size) for _ in batch]
            with torch.no_grad():
                predictions = predict(batch, indexes).detach().cpu()
            for cid, prediction in zip(batch, predictions, strict=True):
                if not torch.isfinite(prediction).all():
                    raise RuntimeError(f"non-finite frozen prediction for training context {cid}")
                results[cid] = _context_loss(prediction, pools[cid])
            guard.check()
    if len(results) != EXPECTED_CONTEXTS:
        raise RuntimeError("incomplete DL-Train diagnostic coverage")
    metrics = aggregate_metrics(rows, results)
    metrics["input_provenance"] = {
        "training_roster_sha256": sha256(args.roster),
        "selected_train_prefix_sha256": selected_train_prefix_sha256,
        "h67_selected_file_sha256_reference": roster["selected_train_file_sha256"],
        "pool_and_table_closure_sha256": hashlib.sha256(
            "\n".join(
                f"{row['context_sha256']}:{row['pool_sha256']}:{row['train_only_jtt1_sha256']}"
                for row in sorted(rows, key=lambda item: item["context_sha256"])
            ).encode()
        ).hexdigest(),
    }
    metrics["resource_observation"] = {
        "max_host_cpu_percent_sampled": guard.max_cpu,
        "max_rss_bytes": guard.max_rss,
        "min_mem_available_bytes": guard.min_available,
        "torch_threads": THREADS,
        "device": "cpu",
    }
    metrics["scientific_verdict"] = "diagnostic-only; H73 allocation/preregistration required before any arm is trained"
    if len(json.dumps(metrics, allow_nan=False).encode()) >= 50 * 1024:
        raise RuntimeError("metrics JSON must remain below 50 KiB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h51/ckpt/round-882.pt")
    parser.add_argument(
        "--roster",
        type=Path,
        default=ROOT / "research/flywheel_selfplay/h67_w4_active_reveal_2plus2/family_fold_manifest.json",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("metrics.json"))
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "contexts": result["contexts"],
        "families": result["resolved_families"],
        "context_gini": result["contexts_per_family"]["gini"],
        "top10_context_share": result["contexts_per_family"]["top10_family_context_share"],
        "raw_top10_loss_share": result["r882_training_objective"]["raw_family_loss_share"]["top10_family_share"],
        "selected_minus_two_contexts": result["r882_training_objective"]["selected_minus_two_failure_contexts"]["count"],
        "metrics_bytes": args.output.stat().st_size,
        "resource_observation": result["resource_observation"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
