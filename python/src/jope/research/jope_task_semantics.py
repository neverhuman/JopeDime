"""Generated dependency-free view of the embedded Rust task semantics.

Do not hand-edit policy in this module. The JSON bytes and digest mirror
`manifests/epoch6/JopeTaskSemantics.v1.json`; import fails immediately if the
generated view is internally inconsistent.
"""

from __future__ import annotations

import hashlib
import json


RUST_TASK_SEMANTICS_SHA256 = (
    "d158798880758dafcce9241226a1585015c4a95c29c2c2a830fdd888fa1b5422"
)
_RUST_TASK_SEMANTICS_JSON = r"""{"schema":"JopeTaskSemantics.v1","canonical_order":["regression","binary","multiclass","timeseries"],"legacy_replay_order":["regression","binary","timeseries","multiclass"],"tasks":[{"task":"regression","corpus_directory":"regression","primary_metric":"r2","native_error_metric":"rmse","baseline_meaning":"training_mean","reward_transform":"clamp_unit","target_validation":"finite_continuous","required_head_capabilities":["regression"],"gp_support":"qualified","evaluation_readiness":"active","training_readiness":"active","minimum_distinct_v2_identities":0,"minimum_rolling_origins":0},{"task":"binary","corpus_directory":"binary","primary_metric":"auc","native_error_metric":"log_loss","baseline_meaning":"training_prevalence","reward_transform":"clamp_unit","target_validation":"binary_zero_one","required_head_capabilities":["binary_probability"],"gp_support":"qualified","evaluation_readiness":"active","training_readiness":"active","minimum_distinct_v2_identities":0,"minimum_rolling_origins":0},{"task":"multiclass","corpus_directory":"multi_label","primary_metric":"macro_f1","native_error_metric":"one_minus_macro_f1","baseline_meaning":"training_majority_class","reward_transform":"clamp_unit","target_validation":"contiguous_class_index","required_head_capabilities":["multiclass_probability"],"gp_support":"shadow_only","evaluation_readiness":"shadow_ready","training_readiness":"requires_readiness_receipt","minimum_distinct_v2_identities":500,"minimum_rolling_origins":0},{"task":"timeseries","corpus_directory":"timeseries_reduced_v2","primary_metric":"negative_mase","native_error_metric":"mase","baseline_meaning":"seasonal_naive","reward_transform":"negative_mase_parity","target_validation":"ordered_finite_continuous","required_head_capabilities":["ordered_forecast"],"gp_support":"shadow_only","evaluation_readiness":"shadow_ready","training_readiness":"requires_readiness_receipt","minimum_distinct_v2_identities":42,"minimum_rolling_origins":3}]}
"""


def _verify_generated_view() -> None:
    observed = hashlib.sha256(_RUST_TASK_SEMANTICS_JSON.encode("utf-8")).hexdigest()
    if observed != RUST_TASK_SEMANTICS_SHA256:
        raise RuntimeError(
            "generated Python task semantics do not match the embedded Rust digest"
        )


_verify_generated_view()
TASK_SEMANTICS = json.loads(_RUST_TASK_SEMANTICS_JSON)
_BY_TASK = {row["task"]: row for row in TASK_SEMANTICS["tasks"]}


def semantics(task: str) -> dict:
    """Return one task's generated semantics; reject compatibility aliases."""

    try:
        return _BY_TASK[task]
    except KeyError as exc:
        raise ValueError(f"unsupported task {task!r}") from exc


def primary_metric(task: str) -> str:
    return str(semantics(task)["primary_metric"])


def corpus_directory(task: str) -> str:
    return str(semantics(task)["corpus_directory"])
