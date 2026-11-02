from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from dltrain import common, live
from dltrain import model as model_mod
from dltrain import objective as objective_mod

RUN_PLAN_SCHEMA = "jope.dl-run-plan/v1"
BASE_REF_KEYS = ("epoch", "holdout_binding", "tokenizer", "corpus", "encoding")
REF_KEYS = BASE_REF_KEYS + ("grammar", "encoding_arm", "normalizers")
IDENTITY_REF_KEYS = ("epoch", "holdout_binding", "tokenizer", "grammar", "encoding_arm", "normalizers")
RESUME_IDENTITY_KEYS = ("run_id", "run_plan_sha256", "objective_sha256", "model_config_sha256",
                        "sampler_config_sha256", "sampler_fit_sha256", "determinism_sha256", "train_data_sha256",
                        "source_tree_sha256", "runtime_sha256")
PLAN_KEYS = ("schema", "run_id", "experiment_id", "arm", "objective", "objective_sha256", "run_seed_hex", "refs",
             "model", "sampler", "train", "ceilings", "schedule", "determinism")
STATUSES = ("COMPLETED", "YIELDED_TO_INTAKE", "YIELDED_UNPUBLISHED", "STOPPED_BUDGET", "FAILED")
_AUTOCAST = {"bfloat16": torch.bfloat16, "float16": torch.float16}
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


@dataclass(frozen=True)
class TrainConfig:
    """Optimization and cadence settings (§6.3.2 Optimization, §6.6 Cadence, §6.7 writer intervals)."""

    lr: float = 3e-4
    lr_min: float = 3e-5
    warmup: int = 1000
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    commit_every_updates: int = 500
    commit_every_seconds: float | None = 1800.0
    eval_every_epochs: int | None = 1
    autocast_dtype: str | None = None
    num_threads: int = 4
    ctx_drop_prob: float = 0.0
    ctx_noise_std: float = 0.0
    live_interval_s: float = 3.0
    receipt_interval_s: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        checks = (
            (self.lr > 0, "lr > 0"), (0 <= self.lr_min <= self.lr, "0 <= lr_min <= lr"),
            (_is_int(self.warmup) and self.warmup >= 0, "warmup is an integer >= 0"),
            (len(self.betas) == 2 and all(0 <= b < 1 for b in self.betas), "two betas in [0, 1)"),
            (self.eps > 0, "eps > 0"), (self.weight_decay >= 0, "weight_decay >= 0"),
            (self.clip_norm > 0, "clip_norm > 0"),
            (_is_int(self.commit_every_updates) and self.commit_every_updates >= 1, "commit_every_updates >= 1"),
            (self.commit_every_seconds is None or self.commit_every_seconds > 0, "commit_every_seconds > 0 or None"),
            (self.eval_every_epochs is None or (_is_int(self.eval_every_epochs) and self.eval_every_epochs >= 1),
             "eval_every_epochs >= 1 or None"),
            (self.autocast_dtype is None or self.autocast_dtype in _AUTOCAST, f"autocast_dtype in {sorted(_AUTOCAST)}"),
            (_is_int(self.num_threads) and self.num_threads >= 1, "num_threads >= 1"),
            (0.0 <= self.ctx_drop_prob <= 1.0, "ctx_drop_prob in [0, 1]"),
            (self.ctx_noise_std >= 0, "ctx_noise_std >= 0"),
            (0 < self.live_interval_s <= live.MAX_INTERVAL_S, f"0 < live_interval_s <= {live.MAX_INTERVAL_S}"),
            (self.receipt_interval_s >= 0, "receipt_interval_s >= 0"),
        )
        for ok, rule in checks:
            if not ok:
                raise ValueError(f"TrainConfig: {rule}")

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"TrainConfig: unknown keys {unknown}")
        return cls(**d)


def _check_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != RUN_PLAN_SCHEMA:
        raise ValueError(f"not a {RUN_PLAN_SCHEMA} document")
    missing = [k for k in PLAN_KEYS if k not in plan]
    if missing:
        raise ValueError(f"run plan lacks {missing}")
    if plan["schedule"].get("id") != objective_mod.SCHEDULE_ID:
        raise ValueError(f"run plan schedule must be {objective_mod.SCHEDULE_ID}")


def make_run_plan(*, run_id: str, run_seed: bytes, arm: str, objective: str, model, sampler, train: TrainConfig,
                  max_updates: int, deadline_utc_us: int | None = None, storage_cap_bytes: int | None = None,
                  refs: dict | None = None, experiment_id: str = "gp-policy-dl-v1-e0") -> dict:
    """The ``jope.dl-run-plan/v1`` document of §5 P9 Step 0 (``model``/``sampler`` are PolicyConfig/SamplerConfig).

    ``determinism`` is the dict returned by ``model.configure_determinism(train.num_threads)`` (it is applied here
    and re-checked by ``run``). The schedule is recorded as a formula id plus its parameters, never a lambda.
    """
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError(f"run_id {run_id!r} must match {_RUN_ID.pattern}")
    if not isinstance(run_seed, (bytes, bytearray)) or len(run_seed) != 32:
        raise ValueError("run_seed must be 32 bytes")
    if not isinstance(arm, str) or not arm:
        raise ValueError("arm must be a non-empty string")
    if not _is_int(max_updates) or max_updates < 0:
        raise ValueError("max_updates must be an integer >= 0")
    for name, value in (("deadline_utc_us", deadline_utc_us), ("storage_cap_bytes", storage_cap_bytes)):
        if value is not None and (not _is_int(value) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer or None")
    if not isinstance(train, TrainConfig):
        raise TypeError("train must be a TrainConfig")
    ref_block = dict.fromkeys(BASE_REF_KEYS)
    for key, value in (refs or {}).items():
        if key not in REF_KEYS:
            raise ValueError(f"unknown ref {key!r}; allowed {REF_KEYS}")
        ref_block[key] = value
    objective_sha256 = objective_mod.objective_sha256(objective)
    determinism = model_mod.configure_determinism(train.num_threads)
    return {
        "schema": RUN_PLAN_SCHEMA,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "arm": arm,
        "objective": objective,
        "objective_sha256": objective_sha256,
        "run_seed_hex": bytes(run_seed).hex(),
        "refs": ref_block,
        "model": model.to_json(),
        "sampler": sampler.to_json(),
        "train": train.to_json(),
        "ceilings": {"max_updates": max_updates, "deadline_utc_us": deadline_utc_us,
                     "storage_cap_bytes": storage_cap_bytes},
        "schedule": {"id": objective_mod.SCHEDULE_ID, "warmup": train.warmup, "max_updates": max_updates,
                     "lr": train.lr, "lr_min": train.lr_min},
        "determinism": dict(determinism),
    }


def plan_sha256(plan: dict) -> str:
    return common.sha256_bytes(common.canonical(plan))


def write_run_plan(path, plan: dict) -> dict:
    """Create-only canonical write of the run plan (§5 P9 Step 0)."""
    _check_plan(plan)
    live.assert_outside_claude(path, "run plan path")
    return common.immutable_write(path, common.canonical(plan))


def load_run_plan(path) -> dict:
    plan = json.loads(Path(path).read_bytes())
    _check_plan(plan)
    return plan
