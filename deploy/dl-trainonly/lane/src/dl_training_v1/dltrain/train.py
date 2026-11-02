"""CPU trainer for the dltrain research lane (PLAN §5 P7 Build/tests 4-6, §5 P9 Step 0, §6.3.2, §6.6, §6.7, §6.8).

- ``make_run_plan``/``write_run_plan``: the ``jope.dl-run-plan/v1`` step-0 record (refs, objective sha256, seed,
  ceilings, hashed warmup+cosine schedule formula, determinism controls), written create-only.
- ``run``: deterministic seeding, the step loop (STOP file, absolute deadline, NaN stop), the consumption
  journal (deterministic fields only), ``events.jsonl`` (the only place with wall-clock times), cadence local
  commits and publishes (dependencies -> manifest -> pointer CAS, ``state/expected-pointer.json``), update
  receipts and live snapshots, and dev-only validation history.
- ``resume_from_commit`` (local, no store) and ``resume_from_store`` (the §6.8 at-resume rules through
  ``ckpt.resume_point``) continue a run; a resume is refused unless every closure identity equals this launch's:
  run plan, objective, model, sampler config and fit, determinism, the training-data content, the trainer's source
  tree and the runtime.
- ``gradient_scale_audit``: the step-0 per-term gradient-norm audit (§6.3.2 Optimization).

Global RNG states are consumed only inside a training step, in the fixed order python (context drop), numpy
(context noise), torch (dropout). The states after each update form the step boundary: commits capture exactly that
state, and anything that draws from a global RNG between steps (publishing and its retries, receipts, live writes,
evaluation, hooks) is undone before the next step and recorded as an ``rng_restored`` event.

Event seqs are unique across the launches of a run: a resume continues above the closure's ``last_event_seq``, the
event high-water mark of the predecessor's ``events.jsonl`` when it is reachable, and the mark the predecessor
recorded in the resume pointer's writer block (§6.6 Events).
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import safetensors
import torch

from dltrain import ckpt, common, evaluate, live
from dltrain import model as model_mod
from dltrain import objective as objective_mod
from dltrain import sampler as sampler_mod

RUN_PLAN_SCHEMA = "jope.dl-run-plan/v1"
GRADIENT_AUDIT_SCHEMA = "jope.dl-gradient-scale-audit/v1"
BASE_REF_KEYS = ("epoch", "holdout_binding", "tokenizer", "corpus", "encoding")
REF_KEYS = BASE_REF_KEYS + ("grammar", "encoding_arm", "normalizers")
IDENTITY_REF_KEYS = ("epoch", "holdout_binding", "tokenizer", "grammar", "encoding_arm", "normalizers")
RESUME_IDENTITY_KEYS = ("run_id", "run_plan_sha256", "objective_sha256", "model_config_sha256",
                        "sampler_config_sha256", "sampler_fit_sha256", "determinism_sha256", "train_data_sha256",
                        "source_tree_sha256", "runtime_sha256")
PLAN_KEYS = ("schema", "run_id", "experiment_id", "arm", "objective", "objective_sha256", "run_seed_hex", "refs",
             "model", "sampler", "train", "ceilings", "schedule", "determinism")
STATUSES = ("COMPLETED", "YIELDED_TO_INTAKE", "YIELDED_UNPUBLISHED", "STOPPED_BUDGET", "FAILED")
_FINAL_PHASE = {"COMPLETED": "completed", "YIELDED_TO_INTAKE": "yielded", "YIELDED_UNPUBLISHED": "yielded_unpublished",
                "STOPPED_BUDGET": "stopped_budget", "FAILED": "failed"}
_AUTOCAST = {"bfloat16": torch.bfloat16, "float16": torch.float16}
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RESUME_ALLOWED = {"restore", "state"}
_PACKAGE = "dltrain"
_TRAINER_MODULE = "train"


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


def build_training_objects(plan: dict):
    """PolicyLM, AdamW (two named groups, fused/foreach off) and LambdaLR over the hashed schedule (§6.3.2)."""
    _check_plan(plan)
    cfg = model_mod.PolicyConfig.from_json(plan["model"])
    tc = TrainConfig.from_json(plan["train"])
    schedule = plan["schedule"]
    net = model_mod.PolicyLM(cfg)
    optimizer = torch.optim.AdamW(model_mod.param_groups(net, tc.weight_decay), lr=tc.lr, betas=tc.betas,
                                  eps=tc.eps, weight_decay=tc.weight_decay, fused=False, foreach=False)
    factor = functools.partial(objective_mod.lr_factor, warmup=schedule["warmup"],
                               max_updates=schedule["max_updates"], lr=schedule["lr"], lr_min=schedule["lr_min"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
    return net, optimizer, scheduler


def _seed_int(tag: bytes, run_seed: bytes, nbytes: int) -> int:
    return int.from_bytes(hashlib.sha256(tag + b"\0" + bytes(run_seed)).digest()[:nbytes], "big")


def seed_fresh(run_seed: bytes) -> None:
    """Seed torch, Python and numpy from the run seed with domain-separated sha256 (§5 P9 Step 0)."""
    torch.manual_seed(_seed_int(b"torch", run_seed, 8))
    random.seed(_seed_int(b"python", run_seed, 8))
    np.random.seed(_seed_int(b"numpy", run_seed, 4))


def apply_augmentation(batch, cfg: TrainConfig) -> None:
    """Training-step-only context augmentation in fixed order: python RNG drop, then numpy RNG noise."""
    if cfg.ctx_drop_prob <= 0 and cfg.ctx_noise_std <= 0:
        return
    z = batch.ctx_z.clone()
    n_ctx, width = z.shape
    if cfg.ctx_drop_prob > 0:
        for c in range(n_ctx):
            if random.random() < cfg.ctx_drop_prob:
                z[c] = 0
    if cfg.ctx_noise_std > 0:
        z += torch.from_numpy(np.random.standard_normal((n_ctx, width)).astype(np.float32)) * cfg.ctx_noise_std
    batch.ctx_z = z


def epoch_steps(data) -> int:
    """Reporting epoch length ceil(|C_train_fit| / B_ctx) in updates (§6.1 Epoch)."""
    return max(1, math.ceil(data.fit.n_contexts() / data.config.b_ctx))


# -------------------------------------------------------------------------------- code and runtime identities

def _package_imports(tree: ast.AST, package_dir: Path) -> set[str]:
    """The dltrain modules a parsed module imports: ``from dltrain import x``, ``from dltrain.x import y``,
    ``import dltrain.x``, ``from . import x`` and ``from .x import y`` (anywhere in the module)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (node.level == 0 and module == _PACKAGE) or (node.level == 1 and not module):
                found.update(alias.name for alias in node.names)
            elif node.level == 0 and module.startswith(_PACKAGE + "."):
                found.add(module.split(".")[1])
            elif node.level == 1:
                found.add(module.split(".")[0])
        elif isinstance(node, ast.Import):
            found.update(alias.name.split(".")[1] for alias in node.names if alias.name.startswith(_PACKAGE + "."))
    return {name for name in found if (package_dir / f"{name}.py").is_file()}


@functools.cache
def _source_tree() -> tuple[tuple[str, str], ...]:
    package_dir = Path(__file__).resolve().parent
    digests: dict[str, str] = {}
    pending_modules = [_TRAINER_MODULE]
    while pending_modules:
        name = pending_modules.pop()
        if f"{name}.py" in digests:
            continue
        raw = (package_dir / f"{name}.py").read_bytes()
        digests[f"{name}.py"] = common.sha256_bytes(raw)
        pending_modules.extend(sorted(_package_imports(ast.parse(raw, filename=f"{name}.py"), package_dir)))
    digests["__init__.py"] = common.sha256_file(package_dir / "__init__.py")
    return tuple((f"{_PACKAGE}/{name}", digest) for name, digest in sorted(digests.items()))


def source_tree_files() -> list[tuple[str, str]]:
    """(relative path, sha256) of the trainer's static source closure: ``dltrain/__init__.py`` plus every dltrain
    module reachable from train.py through import statements, sorted by path. Other modules of the package (other
    packets' code) are not in it, so editing them never changes the identity of a training run."""
    return list(_source_tree())


def source_tree_sha256() -> str:
    """The closure identity ``source_tree_sha256`` (PLAN §6.6): sha256 over ``path NUL sha256 LF`` lines."""
    return common.sha256_bytes("".join(f"{path}\0{digest}\n" for path, digest in _source_tree()).encode("utf-8"))


@functools.cache
def _runtime() -> tuple[tuple[str, str], ...]:
    executable = os.path.realpath(sys.executable)
    doc = {"python": sys.version, "executable_sha256": common.sha256_file(executable),
           "torch": torch.__version__, "torch_config_sha256": common.sha256_bytes(torch.__config__.show().encode()),
           "cpu_capability": torch.backends.cpu.get_cpu_capability(), "numpy": np.__version__,
           "safetensors": safetensors.__version__}
    return tuple(sorted(doc.items()))


def runtime_identity() -> dict:
    """What decides this process's numerics beyond the source (PLAN §6.6 'runtime'): the interpreter version and
    binary sha256, torch's version, build configuration and CPU capability, and numpy's and safetensors' versions."""
    return dict(_runtime())


def runtime_sha256() -> str:
    return common.sha256_bytes(common.canonical(runtime_identity()))


@dataclass
class TrainHooks:
    after_step: Callable[[int], None] | None = None


@dataclass(frozen=True)
class RunResult:
    status: str
    reason: str | None
    step: int
    initial_parameter_sha256: str
    parameter_sha256: str
    optimizer_parameter_change_verified: bool
    last_commit: str | None
    published: dict | None
    journal_sha256: str
    events_path: str
    wall_s: float

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _rng_snapshot():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def _rng_unchanged(before) -> bool:
    t0, p0, n0 = before
    t1, p1, n1 = _rng_snapshot()
    return (torch.equal(t0, t1) and p0 == p1 and n0[0] == n1[0] and np.array_equal(n0[1], n1[1])
            and tuple(n0[2:]) == tuple(n1[2:]))


def _rng_restore(state) -> None:
    t, p, n = state
    torch.set_rng_state(t.clone())
    random.setstate(p)
    np.random.set_state(n)


def _objective_fn(name: str):
    if name == objective_mod.SUCCESSOR_V1:
        return objective_mod.successor_v1
    if name == objective_mod.BASELINE_V1:
        return objective_mod.baseline_v1
    raise ValueError(f"unknown objective {name!r}")


def _identities(plan: dict, data) -> dict:
    refs = plan["refs"]
    ids = {
        "run_id": plan["run_id"],
        "run_plan_sha256": plan_sha256(plan),
        "objective_sha256": plan["objective_sha256"],
        "model_config_sha256": model_mod.PolicyConfig.from_json(plan["model"]).sha256(),
        "sampler_config_sha256": data.config.sha256(),
        "sampler_fit_sha256": data.fit.sha256(),
        "determinism_sha256": common.sha256_bytes(common.canonical(plan["determinism"])),
        "train_data_sha256": data.content_sha256,
        "source_tree_sha256": source_tree_sha256(),
        "runtime_sha256": runtime_sha256(),
    }
    ids.update({k: refs.get(k) for k in IDENTITY_REF_KEYS})
    return ids


def _check_data(plan: dict, data) -> None:
    cfg = model_mod.PolicyConfig.from_json(plan["model"])
    if common.canonical(plan["sampler"]) != common.canonical(data.config.to_json()):
        raise ValueError("run plan sampler config differs from the data's sampler config")
    if common.canonical(data.fit.config.to_json()) != common.canonical(data.config.to_json()):
        raise ValueError("sampler fit was made with a different sampler config")
    if cfg.vocab_size != data.vocab_size or cfg.ctx_dim != data.ctx_dim:
        raise ValueError("model vocab_size/ctx_dim differ from the training data")
    stamped = data.refs or {}
    for key in sampler_mod.DATA_REF_KEYS:
        if plan["refs"].get(key) != stamped.get(key):
            raise ValueError(f"run plan ref {key!r} differs from the identity stamped on the training data")
    if plan["objective_sha256"] != objective_mod.objective_sha256(plan["objective"]):
        raise ValueError("run plan objective_sha256 does not match its objective")
    if plan["objective"] == objective_mod.BASELINE_V1 and (data.config.group_mode != "baseline-v1"
                                                           or not cfg.context_free_head):
        raise ValueError("baseline-v1 needs the baseline-v1 sampler groups and a context-free head")


class _Events:
    """Append-only ``events.jsonl``: the only trainer output that carries wall-clock times (§6.6 Events)."""

    def __init__(self, path: Path, last_seq: int):
        self.path, self.seq = path, last_seq

    def emit(self, event: str, step: int, **fields) -> int:
        self.seq += 1
        record = {"seq": self.seq, "event": event, "step": step, "utc_us": common.utc_us(), **fields}
        with open(self.path, "ab") as f:
            f.write(common.canonical(record) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        return self.seq


def _local_event_seq(commits_root) -> int:
    """The event high-water mark of the launch whose trainer layout ``<workdir>/commits/`` holds ``commits_root``:
    the highest seq among the complete lines of ``<workdir>/events.jsonl`` (a torn tail is not an event); 0 when
    ``commits_root`` is not such a directory or the launch wrote no events."""
    if commits_root is None:
        return 0
    root = Path(commits_root)
    path = root.parent / "events.jsonl"
    if root.name != "commits" or not path.is_file():
        return 0
    data = path.read_bytes()
    high = 0
    for line in data[:data.rfind(b"\n") + 1].splitlines():
        try:
            seq = json.loads(line)["seq"]
        except (ValueError, KeyError, TypeError) as err:
            raise ckpt.CheckpointRefused(f"{path}: unreadable event line, so its seq mark is unknown: {err}") from err
        if not _is_int(seq) or seq < 1:
            raise ckpt.CheckpointRefused(f"{path}: invalid event seq {seq!r}")
        high = max(high, seq)
    return high


def _writer_block(native: dict) -> dict:
    """The pointer writer block (§6.6 pointer body) of this process; ``_publish`` adds its ``event_seq`` mark."""
    return {k: native[k] for k in ("host", "pid", "starttick")} | {"unit": None}


@dataclass(frozen=True)
class _Resume:
    """How a launch continues (§6.8 At resume): the local commit to load, whether that commit is the resume pointer's
    target (so it is never published again), the highest event seq an earlier launch may have used, and the rule."""

    commit_dir: Path
    published: bool
    event_seq_floor: int
    rule: str


class _Run:
    """One launch of the trainer over a run plan; see ``run`` for the contract."""

    def __init__(self, plan, data, workdir, *, resume: _Resume | None = None, store=None, store_prefix: str = "",
                 stop_file=None, live_writer=None, update_dir=None, training_admitted: bool = False,
                 admission_ref: dict | None = None, hooks: TrainHooks | None = None):
        self.t0 = time.monotonic()
        _check_plan(plan)
        self.tc = TrainConfig.from_json(plan["train"])
        determinism = model_mod.configure_determinism(self.tc.num_threads)
        if common.canonical(dict(determinism)) != common.canonical(plan["determinism"]):
            raise ValueError("runtime determinism controls differ from the run plan")
        self.workdir = Path(workdir)
        live.assert_outside_claude(self.workdir, "workdir")
        if update_dir is not None:
            live.assert_outside_claude(update_dir, "update_dir")
        _check_data(plan, data)
        self.plan, self.data = plan, data
        self.cfg = model_mod.PolicyConfig.from_json(plan["model"])
        self.run_id = plan["run_id"]
        self.run_seed = bytes.fromhex(plan["run_seed_hex"])
        self.max_updates = plan["ceilings"]["max_updates"]
        self.deadline = plan["ceilings"]["deadline_utc_us"]
        self.identities = _identities(plan, data)
        self.store, self.store_prefix = store, store_prefix
        self.stop_file = Path(stop_file) if stop_file is not None else None
        self.live_writer = live_writer
        self.update_dir = Path(update_dir) if update_dir is not None else None
        self.training_admitted, self.admission_ref = bool(training_admitted), admission_ref
        self.hooks = hooks or TrainHooks()
        self.objective_fn = _objective_fn(plan["objective"])
        self.epoch_len = epoch_steps(data)
        self.journal_path = self.workdir / "journal.jsonl"
        self.native = live.native_process_identity()
        self.writer = _writer_block(self.native)
        # run state
        self.step = 0
        self.verified = False
        self.verified_utc_us = None
        self.current_sha = None
        self.last_local = None            # ckpt.LocalCommit of the newest commit made or restored
        self.last_local_published = False
        self.last_commit_t = self.t0
        self.expected = None
        self.published = None
        self.resumed = False
        self.storage_new_bytes = 0
        self.checkpoint_info = {"ref": None, "utc_us": None, "readback": None, "checkpoint": None}
        self._boundary = None             # global RNG states at the current step boundary
        self._finished = False
        # reporting state (this process only)
        self.receipt = self.receipt_ref = None
        self.last_report_t = None
        self.window = {"t": None, "steps": 0, "sequences": 0, "tokens": 0, "contexts": 0}
        self.rates = dict.fromkeys(live.RATE_KEYS)
        self.last_loss = self.last_grad = self.last_lr = self.batch_size = None
        self.history: list[dict] = []
        self.epoch_losses: list[float] = []
        self.last_eval = None
        self.events = None
        self.workdir.mkdir(parents=True, exist_ok=True)
        if resume is None:
            self._start_fresh()
        else:
            self._start_resume(resume)
        self.parameter_count = model_mod.count_parameters(self.net)

    # ------------------------------------------------------------------ setup
    def _start_fresh(self) -> None:
        for name in ("journal.jsonl", "events.jsonl", "state/expected-pointer.json"):
            if (self.workdir / name).exists():
                raise FileExistsError(f"fresh run refused: {self.workdir / name} exists (resume instead)")
        if (self.workdir / "commits").exists() and any((self.workdir / "commits").iterdir()):
            raise FileExistsError(f"fresh run refused: {self.workdir / 'commits'} holds commits")
        if self.store is not None:
            self._refuse_existing_pointer()
        seed_fresh(self.run_seed)
        self.net, self.opt, self.sched = build_training_objects(self.plan)
        self.initial_sha = model_mod.parameter_sha256(self.net)
        self.sampler = sampler_mod.MacroSampler(self.data.fit, self.run_seed, self.data.config)
        self._boundary = _rng_snapshot()
        common.atomic_write(self.journal_path, b"")
        self.events = _Events(self.workdir / "events.jsonl", 0)
        self.events.emit("run_started", 0, run_id=self.run_id, run_plan_sha256=self.identities["run_plan_sha256"],
                         initial_parameter_sha256=self.initial_sha, native_process_identity=self.native)

    def _refuse_existing_pointer(self) -> None:
        """A fresh run never starts under an existing resume pointer of its run_id (that is a resume, §6.8)."""
        try:
            self.store.get(ckpt.pointer_key(self.store_prefix, self.run_id))
        except ckpt.NoSuchKey:
            return
        raise ckpt.CheckpointRefused(f"fresh run refused: run {self.run_id!r} already has a resume pointer "
                                     "(resume_from_store instead)")

    def _start_resume(self, resume: _Resume) -> None:
        extra = sorted(p.name for p in self.workdir.iterdir() if p.name not in _RESUME_ALLOWED)
        state_dir = self.workdir / "state"
        if state_dir.exists():
            extra += [f"state/{p.name}" for p in state_dir.iterdir() if p.name != "expected-pointer.json"]
        if extra:
            raise ckpt.CheckpointRefused(f"resume workdir must be empty apart from restore/ and state/: {extra}")
        self.expected = ckpt.load_expected(self.workdir)
        if self.store is not None and self.expected is None:
            raise ckpt.CheckpointRefused("a store-backed resume needs the verified pointer state in "
                                         "state/expected-pointer.json (resume_from_store, §6.8)")
        seed_fresh(self.run_seed)                     # reproduces the run's initial parameters for the proof
        self.net, self.opt, self.sched = build_training_objects(self.plan)
        self.initial_sha = model_mod.parameter_sha256(self.net)
        closure = ckpt.read_commit(resume.commit_dir)
        if closure.run_id != self.run_id:
            raise ckpt.CheckpointRefused(f"commit run_id {closure.run_id!r} != plan run_id {self.run_id!r}")
        for key in RESUME_IDENTITY_KEYS:
            if closure.identities.get(key) != self.identities[key]:
                raise ckpt.CheckpointRefused(f"resume refused: identity {key} differs from this launch's")
        ckpt.apply_closure(closure, model=self.net, optimizer=self.opt, scheduler=self.sched)  # RNG restored last
        self._boundary = _rng_snapshot()
        self.sampler = sampler_mod.MacroSampler(self.data.fit, self.run_seed, self.data.config)
        self.sampler.load_state_dict(closure.sampler_state)
        self.step = closure.step
        commit_raw = (resume.commit_dir / "commit.json").read_bytes()
        self.last_local = ckpt.LocalCommit(dir=resume.commit_dir, step=closure.step,
                                           commit_sha256=common.sha256_bytes(commit_raw), doc=json.loads(commit_raw))
        self.last_local_published = resume.published and self.store is not None
        self.resumed = True
        self.current_sha = model_mod.parameter_sha256(self.net)
        if self.step >= 1:                            # a committed step >= 1 passed the first-update check
            self.verified, self.verified_utc_us = True, common.utc_us()
        common.atomic_write(self.journal_path, closure.journal_bytes)
        self.events = _Events(self.workdir / "events.jsonl", max(closure.last_event_seq, resume.event_seq_floor))
        self.events.emit("resumed", self.step, rule=resume.rule, commit_dir=str(resume.commit_dir),
                         commit_sha256=self.last_local.commit_sha256,
                         manifest_sha256=self.expected.manifest_sha256 if self.expected else None,
                         sampler_cursor=self.sampler.cursor, event_seq_floor=resume.event_seq_floor,
                         native_process_identity=self.native)

    # ------------------------------------------------------------------ RNG step boundary
    def _restore_boundary(self) -> bool:
        """Undo any global RNG draw made since the step boundary; True when something had drawn (§6.6)."""
        if self._boundary is None or _rng_unchanged(self._boundary):
            return False
        _rng_restore(self._boundary)
        return True

    def _end_between_steps(self, during: str) -> None:
        if self._restore_boundary() and not self._finished:
            self.events.emit("rng_restored", self.step, during=during)

    # ------------------------------------------------------------------ live view
    def _live_state(self, phase: str) -> live.LiveState:
        ev = self.last_eval
        metrics = None
        if ev is not None:
            metrics = {"token_validity": ev["token_validity"],
                       "ranking_pair_accuracy": ev["pairwise_accuracy_family_macro"]}
        info = self.checkpoint_info
        return live.LiveState(
            phase=phase, training_admitted=self.training_admitted, admission_ref=self.admission_ref,
            run_id=self.run_id, native_process_identity=self.native, launch_receipt_ref=None,
            update_receipt=self.receipt, update_ref=self.receipt_ref,
            optimizer_parameter_change_verified=self.verified, initial_parameter_sha256=self.initial_sha,
            current_parameter_sha256=self.current_sha, optimizer_updates_total=self.step if self.verified else None,
            verified_utc_us=self.verified_utc_us, train_loss=self.last_loss,
            validation_loss=ev["ce_per_token_family_macro"] if ev else None,
            history=tuple(self.history), epoch=self.step // self.epoch_len, rates=dict(self.rates),
            batch_size=self.batch_size, gradient_norm=self.last_grad, learning_rate=self.last_lr,
            encoding_dimensions={"arm": self.plan["arm"], "input_width": self.data.ctx_dim,
                                 "decoder_width": self.cfg.d_model},
            run_info={"arm": self.plan["arm"], "objective_sha256": self.plan["objective_sha256"],
                      "model_config_sha256": self.identities["model_config_sha256"],
                      "parameters": self.parameter_count},
            checkpoint_ref=info["ref"], checkpoint_utc_us=info["utc_us"], checkpoint_readback=info["readback"],
            checkpoint=info["checkpoint"],
            storage_bytes=self.storage_new_bytes if (self.store is not None and not self.resumed) else None,
            storage_cap_bytes=self.plan["ceilings"]["storage_cap_bytes"],
            validation_split=ev["validation_split"] if ev else None, metrics=metrics,
        )

    def _set_phase(self, phase: str) -> None:
        if self.live_writer is not None:
            self.live_writer.update(self._live_state(phase))

    # ------------------------------------------------------------------ checkpoints
    def _parent(self):
        """The previous local commit plus the last published manifest (the pointer target the CAS must observe)."""
        if self.last_local is None:
            return None
        return {"commit_sha256": self.last_local.commit_sha256,
                "manifest_sha256": self.expected.manifest_sha256 if self.expected else None,
                "manifest_version_id": self.expected.manifest_version_id if self.expected else None}

    def _commit(self):
        """Local commit of the step-boundary state: any global RNG draw since the boundary is undone first, so the
        closure always holds the training streams' state after the last update."""
        if self._restore_boundary():
            self.events.emit("rng_restored", self.step, during="before_commit")
        local = ckpt.save_local_commit(
            self.workdir, run_id=self.run_id, step=self.step, model=self.net, optimizer=self.opt,
            scheduler=self.sched, sampler_state=self.sampler.state_dict(), journal_bytes=self.journal_path.read_bytes(),
            identities=self.identities, parent=self._parent(), last_event_seq=self.events.seq,
            expected_pointer=self.expected.to_json() if self.expected else None)
        self.last_local, self.last_local_published = local, False
        self.last_commit_t = time.monotonic()
        self.checkpoint_info = {"ref": None, "utc_us": local.doc.get("wall_time_utc_us"), "readback": None,
                                "checkpoint": {"step": local.step, "upload_status": "LOCAL_ONLY"}}
        self.events.emit("commit", self.step, commit_sha256=local.commit_sha256, dir=str(local.dir),
                         parameter_sha256=local.doc.get("parameter_sha256"))
        return local

    def _publish(self, local, *, tail: int) -> None:
        """Publish a local commit (§6.6 order). The pointer's writer block records ``event_seq``: the seq of this
        publish's outcome event plus ``tail`` more that the caller emits before stopping, so a resume from this
        pointer continues above every event this launch emits up to then. A global RNG draw during the publish
        (e.g. client retry jitter) is undone before the outcome event, which records it."""
        self.checkpoint_info["checkpoint"] = {"step": local.step, "upload_status": "UPLOADING"}
        self._set_phase("checkpointing")
        writer = dict(self.writer, event_seq=self.events.seq + 1 + tail)
        try:
            res = ckpt.publish_commit(self.store, local, prefix=self.store_prefix, expected=self.expected,
                                      writer=writer)
        except Exception as exc:
            restored = self._restore_boundary()
            self.checkpoint_info["checkpoint"] = {"step": local.step, "upload_status": "FAILED"}
            self.checkpoint_info["readback"] = "fail"
            self.events.emit("publish_failed", self.step, error_type=type(exc).__name__, error=str(exc)[:500],
                             rng_restored=restored)
            raise
        restored = self._restore_boundary()
        ckpt.save_expected(self.workdir, res["pointer"])
        self.expected = res["pointer"]
        self.last_local_published = True
        self.storage_new_bytes += int(res["new_bytes"])
        self.published = {"manifest_ref": res["manifest_ref"], "deps": res["deps"],
                          "pointer": res["pointer"].to_json(), "new_bytes": res["new_bytes"]}
        s3ref = {k: v for k, v in res["manifest_ref"].items() if k != "new_bytes"}
        self.checkpoint_info = {"ref": s3ref, "utc_us": common.utc_us(), "readback": "pass",
                                "checkpoint": {"step": local.step, "upload_status": "PUBLISHED_VERIFIED",
                                               "pointer_version_id": res["pointer"].version_id}}
        self.events.emit("published", self.step, manifest_sha256=res["pointer"].manifest_sha256,
                         manifest_version_id=res["pointer"].manifest_version_id, pointer_seq=res["pointer"].seq,
                         pointer_version_id=res["pointer"].version_id, new_bytes=res["new_bytes"],
                         event_seq_reserved=writer["event_seq"], rng_restored=restored)

    def _cadence_checkpoint(self):
        """Local commit plus publish at the §6.6 cadence; returns a terminal RunResult only on CONFLICT/POISONED_KEY."""
        self._set_phase("checkpointing")
        local = self._commit()
        if self.store is not None:
            try:
                self._publish(local, tail=0)
            except ckpt.Conflict:
                return self._finish("FAILED", "CONFLICT")
            except ckpt.PoisonedKey:
                return self._finish("FAILED", "POISONED_KEY")
            except Exception:  # noqa: BLE001 - logged in events; the next cadence publish retries the chain
                pass
        self._set_phase("stepping")
        return None

    def _terminal(self, status: str, reason: str | None) -> RunResult:
        """Commit (unless this step already has one) and publish, then stop (§6.8 yield sequence). After a publish
        attempt exactly one event follows its outcome: ``finished`` (reserved in the pointer's ``event_seq``)."""
        self._set_phase("checkpointing")
        local = self.last_local if self.last_local is not None and self.last_local.step == self.step else self._commit()
        if self.store is not None and not self.last_local_published:
            try:
                self._publish(local, tail=1)
            except ckpt.Conflict:
                return self._finish("FAILED", "CONFLICT")
            except ckpt.PoisonedKey:
                return self._finish("FAILED", "POISONED_KEY")
            except Exception:  # noqa: BLE001 - recorded as publish_failed; the next launch publishes it first (§6.8)
                status = "YIELDED_UNPUBLISHED"
        return self._finish(status, reason)

    # ------------------------------------------------------------------ steps
    def _report_rates(self, now: float) -> None:
        w = self.window
        dt = now - w["t"] if w["t"] is not None else None
        if dt is not None and dt > 0 and w["steps"] > 0:
            self.rates = {"dataset_splits_per_second": w["contexts"] / dt, "examples_per_second": w["sequences"] / dt,
                          "batches_per_second": w["steps"] / dt, "tokens_per_second": w["tokens"] / dt}
        self.window = {"t": now, "steps": 0, "sequences": 0, "tokens": 0, "contexts": 0}

    def _step(self):
        """One optimizer update, then the between-step work on the RNG step boundary; returns a terminal RunResult
        on failure, else None (§6.6, §6.3.2)."""
        if self.window["t"] is None:
            self.window["t"] = time.monotonic()
        draw = self.sampler.draw_next()
        batch = sampler_mod.collate(draw, self.data)
        report = self.live_writer is not None and (
            self.last_report_t is None or time.monotonic() - self.last_report_t >= self.tc.receipt_interval_s)
        before = model_mod.parameter_sha256(self.net) if report else None
        apply_augmentation(batch, self.tc)
        self.net.train()
        autocast = (torch.autocast(device_type="cpu", dtype=_AUTOCAST[self.tc.autocast_dtype])
                    if self.tc.autocast_dtype else contextlib.nullcontext())
        with autocast:
            out = self.net(batch.tokens, batch.lengths, batch.ctx_z, batch.seq_ctx)
        try:
            result = self.objective_fn(out, batch)
        except objective_mod.ObjectiveNonFinite as exc:
            self.events.emit("objective_nonfinite", self.step, error=str(exc)[:500])
            return self._finish("FAILED", "nonfinite_objective")
        if not bool(torch.isfinite(result.loss.detach())):
            self.events.emit("objective_nonfinite", self.step, error="loss is not finite")
            return self._finish("FAILED", "nonfinite_objective")
        self.opt.zero_grad(set_to_none=True)
        result.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.tc.clip_norm, foreach=False)
        if not bool(torch.isfinite(grad_norm)):
            self.events.emit("gradient_nonfinite", self.step)
            return self._finish("FAILED", "nonfinite_objective")
        lr_used = float(self.opt.param_groups[0]["lr"])
        self.opt.step()
        self.sched.step()
        self.step += 1
        loss_value, grad_value = float(result.loss.detach()), float(grad_norm)
        record = draw.journal_record() | {"lr": lr_used, "loss": loss_value, "grad_norm": grad_value}
        with open(self.journal_path, "ab") as f:
            f.write(common.canonical(record) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        after = None
        if not self.verified:
            after = model_mod.parameter_sha256(self.net)
            if after == self.initial_sha:
                self.events.emit("parameter_unchanged", self.step, parameter_sha256=after)
                return self._finish("FAILED", "parameter_unchanged")
            self.verified, self.verified_utc_us, self.current_sha = True, common.utc_us(), after
            self.events.emit("optimizer_parameter_change_verified", self.step, initial=self.initial_sha, after=after)
        self.last_loss, self.last_grad, self.last_lr = loss_value, grad_value, lr_used
        self.batch_size = int(batch.tokens.shape[0])
        self.epoch_losses.append(loss_value)
        w = self.window
        w["steps"] += 1
        w["sequences"] += self.batch_size
        w["tokens"] += int(batch.n_tokens)
        w["contexts"] += len(batch.ctx_ids)
        self._boundary = _rng_snapshot()              # the training streams' state after this update
        try:
            return self._between_steps(report, before, after)
        finally:
            self._end_between_steps("between_steps")

    def _between_steps(self, report: bool, before: str | None, after: str | None):
        if report:
            failed = self._report(before, after)
            if failed is not None:
                return failed
        due = self.step % self.tc.commit_every_updates == 0 or (
            self.tc.commit_every_seconds is not None
            and time.monotonic() - self.last_commit_t >= self.tc.commit_every_seconds)
        if due and self.step < self.max_updates:
            terminal = self._cadence_checkpoint()
            if terminal is not None:
                return terminal
        if self.tc.eval_every_epochs and self.step % (self.epoch_len * self.tc.eval_every_epochs) == 0:
            self._evaluate()
        if self.hooks.after_step is not None:
            self.hooks.after_step(self.step)
        return None

    def _report(self, before: str, after: str | None):
        after = after if after is not None else model_mod.parameter_sha256(self.net)
        self.current_sha = after
        if self.update_dir is not None:
            if before == after:
                self.events.emit("update_receipt_skipped", self.step, reason="before == after")
            else:
                try:
                    self.receipt, self.receipt_ref = live.write_update_receipt(
                        self.update_dir, run_id=self.run_id, optimizer_step=self.step, process_identity=self.native,
                        before_sha256=before, after_sha256=after)
                except common.ImmutableConflict as exc:   # a replayed step after resume: never overwritten
                    self.events.emit("update_receipt_conflict", self.step, error=str(exc)[:500])
                except live.LiveRefused as exc:
                    self.events.emit("live_writer_error", self.step, error=str(exc)[:500])
                    return self._finish("FAILED", "live_writer_error")
        now = time.monotonic()
        self._report_rates(now)
        self.last_report_t = now
        self._set_phase("stepping")
        return None

    def _evaluate(self) -> None:
        self._set_phase("validating")
        metrics = evaluate.dev_metrics(self.net, self.data, split="dev")
        epoch = self.step // self.epoch_len
        train_loss = math.fsum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else None
        self.epoch_losses = []
        self.events.emit("dev_validation", self.step, epoch=epoch, metrics=metrics)
        if metrics["ce_per_token_family_macro"] is not None and train_loss is not None:
            self.last_eval = metrics
            self.history.append({"epoch": epoch, "optimizer_step": self.step, "train_loss": train_loss,
                                 "validation_loss": metrics["ce_per_token_family_macro"], "utc_us": common.utc_us()})
        self._set_phase("stepping")

    # ------------------------------------------------------------------ main loop
    def _finish(self, status: str, reason: str | None) -> RunResult:
        """Final live document, then ``finished`` as this launch's last event (a live writer error is a field)."""
        final_sha = model_mod.parameter_sha256(self.net)
        if self.verified:
            self.current_sha = final_sha
        writer_error = None
        if self.live_writer is not None:
            self.live_writer.update(self._live_state(_FINAL_PHASE[status]))
            self.live_writer.stop()
            if self.live_writer.error is not None:
                writer_error = repr(self.live_writer.error)[:500]
        self._finished = True
        self.events.emit("finished", self.step, status=status, reason=reason, parameter_sha256=final_sha,
                         last_commit=str(self.last_local.dir) if self.last_local else None,
                         live_writer_error=writer_error)
        return RunResult(
            status=status, reason=reason, step=self.step, initial_parameter_sha256=self.initial_sha,
            parameter_sha256=final_sha, optimizer_parameter_change_verified=self.verified,
            last_commit=str(self.last_local.dir) if self.last_local else None, published=self.published,
            journal_sha256=common.sha256_file(self.journal_path), events_path=str(self.events.path),
            wall_s=time.monotonic() - self.t0)

    def loop(self) -> RunResult:
        try:
            if self.live_writer is not None:
                self.live_writer.update(self._live_state("preflight"))
                self.live_writer.start()
                self._end_between_steps("preflight")
            while self.step < self.max_updates:
                if self.stop_file is not None and self.stop_file.exists():
                    return self._terminal("YIELDED_TO_INTAKE", "stop_file")
                if self.deadline is not None and common.utc_us() >= self.deadline:
                    return self._terminal("STOPPED_BUDGET", "deadline")
                if self.live_writer is not None and self.live_writer.error is not None:
                    self.events.emit("live_writer_error", self.step, error=repr(self.live_writer.error)[:500])
                    return self._finish("FAILED", "live_writer_error")
                terminal = self._step()
                if terminal is not None:
                    return terminal
            return self._terminal("COMPLETED", None)
        except BaseException as exc:
            self.events.emit("crashed", self.step, error_type=type(exc).__name__, error=str(exc)[:500])
            if self.live_writer is not None and self.live_writer.running:
                self.live_writer.update(self._live_state("failed"))
                self.live_writer.stop()
            raise


def run(plan: dict, data, workdir, *, store=None, store_prefix: str = "", stop_file=None,
        live_writer: live.LiveWriter | None = None, update_dir=None, training_admitted: bool = False,
        admission_ref: dict | None = None, hooks: TrainHooks | None = None) -> RunResult:
    """Train ``plan`` on ``data`` from step 0 in ``workdir`` until a terminal status (§5 P7, §6.6, §6.8).

    Refuses an existing ``journal.jsonl``, events, expected pointer or commit, and (with a store) an existing resume
    pointer of the run; seeds from the run seed and builds the objects. Each iteration: STOP file ->
    YIELDED_TO_INTAKE (YIELDED_UNPUBLISHED if the publish fails); deadline -> STOPPED_BUDGET; live writer error ->
    FAILED; then one update (non-finite objective or gradient -> FAILED with no commit of the bad state; an unchanged
    first update -> FAILED 'parameter_unchanged'), the journal line, and the between-step work on the RNG step
    boundary: update receipt and live snapshot when reporting, cadence commit/publish, dev validation and the hook.
    CONFLICT and POISONED_KEY are terminal FAILED. Reaching ``max_updates`` commits, publishes and COMPLETES.
    """
    return _Run(plan, data, workdir, store=store, store_prefix=store_prefix, stop_file=stop_file,
                live_writer=live_writer, update_dir=update_dir, training_admitted=training_admitted,
                admission_ref=admission_ref, hooks=hooks).loop()


def resume_from_commit(plan: dict, data, commit_dir, workdir, **kw) -> RunResult:
    """Continue a run from a local commit into an otherwise empty workdir, without a store (§6.8). Store-backed
    resumes go through ``resume_from_store``, which applies the at-resume rules against the pointer. Events continue
    above the closure's ``last_event_seq`` and above the events of the launch that wrote the commit, when the commit
    sits in that launch's ``<workdir>/commits/``."""
    if kw.get("store") is not None:
        raise ValueError("a store-backed resume goes through resume_from_store (PLAN §6.8 At resume)")
    commit_dir = Path(commit_dir)
    resume = _Resume(commit_dir=commit_dir, published=False, event_seq_floor=_local_event_seq(commit_dir.parent),
                     rule="local_commit")
    return _Run(plan, data, workdir, resume=resume, **kw).loop()


def resume_from_store(plan: dict, data, store, workdir, *, store_prefix: str = "", local_commits=None,
                      **kw) -> RunResult:
    """Continue a run under the §6.8 at-resume rules (``ckpt.resume_point``), CPU-only before any training.

    ``local_commits`` is the previous launch's ``commits/`` directory when it still exists. Depending on the pointer,
    the newest local commit is continued from, published first and continued from, or the pointer's closure is
    restored by exact VersionId into ``workdir/restore``; anything else is refused before anything is written. The
    verified pointer state goes to ``workdir/state/expected-pointer.json``; ``workdir`` must be absent or empty.
    """
    workdir = Path(workdir)
    live.assert_outside_claude(workdir, "workdir")
    if workdir.exists() and any(workdir.iterdir()):
        raise ckpt.CheckpointRefused(f"resume_from_store needs an absent or empty workdir: {workdir}")
    _check_plan(plan)
    workdir.mkdir(parents=True, exist_ok=True)
    point = ckpt.resume_point(store, prefix=store_prefix, run_id=plan["run_id"], local_commits=local_commits,
                              restore_dir=workdir / "restore", writer=_writer_block(live.native_process_identity()),
                              event_seq_floor=_local_event_seq(local_commits))
    ckpt.save_expected(workdir, point.pointer)
    resume = _Resume(commit_dir=point.commit.dir, published=True, event_seq_floor=point.event_seq, rule=point.rule)
    return _Run(plan, data, workdir, resume=resume, store=store, store_prefix=store_prefix, **kw).loop()


def _grad_norm(params) -> float:
    return math.sqrt(math.fsum(float(p.grad.detach().double().pow(2).sum()) for p in params if p.grad is not None))


def gradient_scale_audit(model, batch, objective_name: str) -> dict:
    """Per-term gradient norms of the weighted objective terms on one batch (§6.3.2: stop if any term > 10x CE).

    Runs in eval mode with no augmentation. Each term's gradient comes from a separate forward/backward with every
    other coefficient set to zero inside the objective's coefficient table (restored afterwards); the patched loss
    must equal that term's weighted value, otherwise the audit refuses instead of reporting a wrong norm. Gradients
    are cleared afterwards and the global torch/Python/numpy RNG states are asserted unchanged.
    """
    coefficients = {objective_mod.SUCCESSOR_V1: objective_mod.COEF_SUCCESSOR,
                    objective_mod.BASELINE_V1: objective_mod.COEF_BASELINE}.get(objective_name)
    if coefficients is None:
        raise ValueError(f"unknown objective {objective_name!r}")
    fn = _objective_fn(objective_name)
    ce_name = "ce" if objective_name == objective_mod.SUCCESSOR_V1 else "seq_nll"
    original = dict(coefficients)
    params = [p for p in model.parameters() if p.requires_grad]
    rng_before = _rng_snapshot()
    was_training = model.training

    def forward():
        return model(batch.tokens, batch.lengths, batch.ctx_z, batch.seq_ctx)

    model.eval()
    norms, support = {}, {}
    try:
        for p in params:
            p.grad = None
        with torch.no_grad():
            reference = fn(forward(), batch)
        support = dict(reference.support)
        for name in sorted(original):
            coefficients.clear()
            coefficients.update({k: (v if k == name else 0.0) for k, v in original.items()})
            result = fn(forward(), batch)
            expected = reference.weighted.get(name)
            if expected is None or not math.isclose(float(result.loss.detach()), expected, rel_tol=1e-4, abs_tol=1e-6):
                raise RuntimeError(f"objective coefficient table is not the one the loss reads (term {name}); "
                                   "per-term audit refused")
            result.loss.backward()
            norms[name] = _grad_norm(params)
            for p in params:
                p.grad = None
    finally:
        coefficients.clear()
        coefficients.update(original)
        model.train(was_training)
        for p in params:
            p.grad = None
    if not _rng_unchanged(rng_before):
        raise AssertionError("gradient-scale audit consumed global RNG state")
    ce_norm = norms[ce_name]
    exceeds = sorted(k for k, v in norms.items() if k != ce_name and v > 10.0 * ce_norm)
    return {"schema": GRADIENT_AUDIT_SCHEMA, "objective": objective_name, "ce_term": ce_name, "ce_norm": ce_norm,
            "term_norms": norms, "exceeds_10x": exceeds, "support": support}
