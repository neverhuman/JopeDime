from __future__ import annotations

from test_exact_train import *

def load_optional_task_readiness(path: Path = OPTIONAL_TASK_READINESS) -> dict | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get("schema") != "JopeOptionalTaskReadiness.v2":
        raise RuntimeError("optional-task readiness schema drift")
    generated = int(value.get("generated_at_unix_ms") or 0)
    budget = int(value.get("freshness_budget_seconds") or 0)
    now = int(time.time() * 1000)
    if generated <= 0 or budget <= 0 or now < generated or now - generated > budget * 1000:
        raise RuntimeError("optional-task readiness is stale or future-invalid")
    inputs = value.get("input_digests")
    if (
        not isinstance(inputs, list)
        or not inputs
        or inputs != sorted(set(inputs))
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(digest)) for digest in inputs)
    ):
        raise RuntimeError("optional-task readiness input digests are invalid")
    rows = value.get("tasks")
    if (
        not isinstance(rows, list)
        or [row.get("task") for row in rows] != ["multiclass", "timeseries"]
    ):
        raise RuntimeError("optional-task readiness rows are incomplete or unordered")
    return value


def optional_task_training_allowed(task: str, row: dict, datasets: int) -> bool:
    floor = OPTIONAL_TASK_DATASET_FLOORS[task]
    common = (
        row.get("state") == "shadow_ready"
        and row.get("training_allowed") is True
        and int(row.get("distinct_dataset_identities") or 0) >= floor
        and datasets >= floor
        and int(row.get("valid_v2_cache_bindings") or 0) >= datasets
        and int(row.get("valid_context_transform_bindings") or 0) >= datasets
        and row.get("evaluator_retained") is True
        and int(row.get("task_valid_head_count") or 0) > 0
        and row.get("fresh_unbiased_shadow_evidence") is True
        and not row.get("blockers")
    )
    if task == "multiclass":
        return (
            common
            and int(row.get("minimum_required") or 0) == 500
            and row.get("primary_metric") == "macro_f1"
            and row.get("native_error_metric") == "one_minus_macro_f1"
        )
    return (
        common
        and datasets == 42
        and int(row.get("distinct_dataset_identities") or 0) == 42
        and int(row.get("minimum_required") or 0) == 42
        and int(row.get("minimum_rolling_origins_per_dataset") or 0) >= 3
        and row.get("causal_reduction_verified") is True
        and row.get("no_future_leakage") is True
        and row.get("primary_metric") == "negative_mase"
        and row.get("native_error_metric") == "mase"
    )


def eight_slot_task_weights(readiness: dict | None) -> dict[str, tuple[int, int]]:
    rows = {
        row["task"]: row for row in (readiness or {}).get("tasks", [])
        if isinstance(row, dict) and row.get("task") in OPTIONAL_TASK_DATASET_FLOORS
    }
    multiclass = rows.get("multiclass", {}).get("training_allowed") is True
    timeseries = rows.get("timeseries", {}).get("training_allowed") is True
    missing = int(not multiclass) + int(not timeseries)
    return {
        "regression": (6 + missing, 16),
        "binary": (6 + missing, 16),
        "multiclass": (2 if multiclass else 0, 16),
        "timeseries": (2 if timeseries else 0, 16),
    }


def require_optional_task_admission(
    task: str, readiness_path: Path | None
) -> dict | None:
    if task in MANDATORY_TASKS:
        return None
    path = readiness_path or OPTIONAL_TASK_READINESS
    readiness = load_optional_task_readiness(path)
    if readiness is None:
        raise ValueError(f"{task} training requires JopeOptionalTaskReadiness.v2")
    row = next(row for row in readiness["tasks"] if row["task"] == task)
    if row.get("training_allowed") is not True:
        raise ValueError(f"{task} is not admitted for optional-task training")
    return readiness


def trainable_task_metrics(con, readiness: dict | None = None) -> dict:
    """Tasks whose observations may enter a checkpoint, task -> metric.

    Mandatory tasks always qualify. Optional tasks require both corpus counts
    and a fresh V2 readiness authority; missing readiness is typed absence.
    """

    trainable = {
        task: metric
        for task, metric in ENABLED_TASK_METRICS.items()
        if task in MANDATORY_TASKS
    }
    if readiness is None:
        readiness = load_optional_task_readiness()
    readiness_rows = {
        row["task"]: row for row in (readiness or {}).get("tasks", [])
        if isinstance(row, dict) and row.get("task") in OPTIONAL_TASK_DATASET_FLOORS
    }
    for task, metric in sorted(ENABLED_TASK_METRICS.items()):
        if task in MANDATORY_TASKS:
            continue
        (datasets,) = con.execute(
            "SELECT count(DISTINCT dataset_hash) FROM observations "
            "WHERE task=? AND metric=?",
            (task, metric),
        ).fetchone()
        row = readiness_rows.get(task)
        if row is not None and optional_task_training_allowed(task, row, datasets):
            trainable[task] = metric
        else:
            print(
                f"[trainer] {task} withheld from training: {datasets} distinct "
                f"datasets; JopeOptionalTaskReadiness.v2 is absent or not "
                f"shadow-ready at task floor {OPTIONAL_TASK_DATASET_FLOORS[task]}; "
                f"evaluation continues",
                flush=True,
            )
    return trainable


RECIPE_E_MODEL_OUTPUTS = (
    "immediate_utility",
    "strict_win_probability",
    "descendant_value_h1",
    "descendant_value_h2",
    "descendant_value_h4",
    "descendant_value_h8",
    "mutation_operator_distribution",
    "parent_conditioned_structured_edit_distribution",
    "validity_probability",
    "instability_probability",
    "reward_quantile_lower",
    "reward_quantile_median",
    "reward_quantile_upper",
)


def recipe_e_training_trigger(
    new_eligible_labels: int,
    new_completed_descendant_receipts: int,
) -> bool:
    """Recipe E trains after either fixed information-mass threshold."""

    if min(new_eligible_labels, new_completed_descendant_receipts) < 0:
        raise ValueError("recipe E trigger counts cannot be negative")
    return (
        new_eligible_labels >= 50_000
        or new_completed_descendant_receipts >= 5_000
    )


def recipe_e_batch_lane_counts(batch_size: int, batch_index: int) -> dict[str, int]:
    """Return an exact deterministic 50/25/15/10 cumulative allocation."""

    if batch_size <= 0 or batch_index < 0 or batch_size % 8:
        raise ValueError("recipe E requires a positive batch size divisible by eight")
    # The pair lane must contain whole DL/GP pairs in every individual batch.
    pair_rows = batch_size // 4
    if pair_rows % 2:
        raise ValueError("recipe E pair lane must contain an even row count")
    previous_total = batch_index * batch_size
    current_total = (batch_index + 1) * batch_size
    broad_rows = batch_size // 2
    descendant_rows = (
        (current_total * 15) // 100 - (previous_total * 15) // 100
    )
    hard_rows = batch_size - broad_rows - pair_rows - descendant_rows
    return {
        "uniform_broad_replay": broad_rows,
        "exact_current_parent_pairs": pair_rows,
        "completed_descendant_trajectories": descendant_rows,
        "validity_instability_hard_negatives": hard_rows,
    }


def recipe_e_evidence_lane(metadata: dict, accepted_parent_sha256: str) -> str:
    """Classify one row into one non-overlapping recipe-E evidence lane."""

    pair_id = metadata.get("causal_pair_id")
    pair_lane = metadata.get("causal_lane")
    if (
        pair_id
        and pair_lane in {"native_dl_direct", "matched_gp_comparator"}
        and metadata.get("causal_accepted_parent_checkpoint_sha256")
        == accepted_parent_sha256
    ):
        return "exact_current_parent_pairs"
    if metadata.get("descendant_credit_receipt_sha256"):
        if metadata.get("descendant_censored", False):
            return "censored_descendant"
        if metadata.get("descendant_horizon") not in {1, 2, 4, 8}:
            raise ValueError("recipe E descendant row has an invalid horizon")
        return "completed_descendant_trajectories"
    if any(
        bool(metadata.get(name))
        for name in (
            "validity_hard_negative",
            "instability_hard_negative",
            "adversarial_hard_case",
            "candidate_invalid",
        )
    ):
        return "validity_instability_hard_negatives"
    return "uniform_broad_replay"


class GpuUtilizationMonitor:
    """Collect low-overhead utilization samples over the real training window."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.values: list[float] = []
        self.started = 0.0
        self.finished = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.started = time.monotonic()

        def sample() -> None:
            while not self._stop.is_set():
                observed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if observed.returncode == 0:
                    for line in observed.stdout.splitlines():
                        try:
                            self.values.append(float(line.strip()))
                        except ValueError:
                            pass
                self._stop.wait(15)

        self._thread = threading.Thread(
            target=sample, name="gpu-utilization-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20)
        self.finished = time.monotonic()

    def summary(self) -> dict:
        elapsed = max(0.0, self.finished - self.started)
        median = statistics.median(self.values) if self.values else None
        return {
            "samples": len(self.values),
            "sample_interval_seconds": 15,
            "elapsed_seconds": elapsed,
            "median_gpu_utilization_percent": median,
            "fifteen_minute_window_complete": elapsed >= 900 and len(self.values) >= 60,
            "meets_sixty_percent_gate": (
                elapsed >= 900
                and len(self.values) >= 60
                and median is not None
                and median >= 60.0
            ),
        }


def four_worker_policy_batch_iterator(train_v3, original):
    """Ordered four-worker CPU collation with pinned-memory CUDA prefetch."""

    def iterator(batch_rows, vocab, *, device, artifact_cache):
        if not str(device).startswith("cuda"):
            yield from original(
                batch_rows,
                vocab,
                device=device,
                artifact_cache=artifact_cache,
            )
            return
        torch = train_v3.torch
        row_iter = iter(batch_rows)
        futures = deque()

        def collate(rows):
            return train_v3._pin_policy_batch(
                train_v3.rows_to_policy_v3_batch(
                    rows,
                    vocab,
                    device="cpu",
                    artifact_cache=artifact_cache,
                )
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="packed-loader"
        ) as workers:
            for _ in range(8):
                try:
                    futures.append(workers.submit(collate, next(row_iter)))
                except StopIteration:
                    break
            copy_stream = torch.cuda.Stream(device=device)
            current_stream = torch.cuda.current_stream(device=device)
            while futures:
                host_batch = futures.popleft().result()
                try:
                    futures.append(workers.submit(collate, next(row_iter)))
                except StopIteration:
                    pass
                with torch.cuda.stream(copy_stream):
                    device_batch = train_v3._move_policy_batch(
                        host_batch, device, non_blocking=True
                    )
                    ready = torch.cuda.Event()
                    ready.record(copy_stream)
                current_stream.wait_event(ready)
                train_v3._record_policy_batch_stream(device_batch, current_stream)
                yield device_batch
                del host_batch

    return iterator
