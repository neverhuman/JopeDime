"""Frozen game identities, committed before disclosing an execution result."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_sha(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a full sha256")


def local_path(bench, path, area=None):
    bench, path = Path(bench).resolve(), Path(path).resolve()
    protected = "flywheel_neural_continuous_20261009_v1"
    if protected in bench.parts or protected in path.parts:
        raise ValueError("refusing the protected service root")
    root = bench / area if area else bench
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"path must be below benchmark {area or 'root'}")
    return path


@dataclass(frozen=True)
class ContextSpec:
    context_id: str
    lineage_id: str
    role: str
    identity_sha256: str
    hwm_program_sha256: str | None

    def __post_init__(self):
        if not self.context_id or not self.lineage_id or self.role != "development":
            raise ValueError("arena contexts require a development-role lineage")
        require_sha(self.identity_sha256, "context identity")
        if self.hwm_program_sha256 is not None:
            require_sha(self.hwm_program_sha256, "incumbent program")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    checkpoint_sha256: str | None
    trie_sha256: str
    normalizer_sha256: str
    tokenizer_sha256: str
    temperature: float = 1.0

    def __post_init__(self):
        if (
            not self.name
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("invalid arm name or temperature")
        for name in ("trie_sha256", "normalizer_sha256", "tokenizer_sha256"):
            require_sha(getattr(self, name), name)
        if self.checkpoint_sha256 is not None:
            require_sha(self.checkpoint_sha256, "checkpoint")
        if (
            self.name != "hwm"
            and not self.name.startswith("uniform_trie")
            and self.checkpoint_sha256 is None
        ):
            raise ValueError("policy arms require a sealed checkpoint identity")


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    bench_root: str
    out_dir: str
    state_path: str
    contexts: tuple[ContextSpec, ...]
    arms: tuple[ArmSpec, ...]
    seeds: tuple[int, ...]
    role_map_sha256: str
    cache_snapshot_sha256: str
    evaluator_sha256: str
    canonicalizer_sha256: str
    protocol: str = "jope.flywheel-s1-inner-rmse/v1"
    budget: int = 8
    tau: float = 0.01
    order_seed: int = 20261009
    bootstrap: int = 2000
    bootstrap_seed: int = 7
    slate_k: int = 8
    kpi_version: str = "rlfw-kpi/v1"
    comparator: str = "original-program-reexecuted-current-protocol"
    outcome_classes: tuple[str, ...] = (
        "qualified_complete",
        "program_failed",
        "timeout",
        "parity_custody",
        "infra_unknown",
        "censored",
    )
    publish_prefix: str = "program-results/benchmark-rlfw/v1/"
    benchmark_valid: bool = False
    runtime_binding_sha256: str | None = None
    policy_code_manifest_sha256: str | None = None
    policy_qualification_sha256: str | None = None
    schedule: str = "one-per-lineage-then-all"
    deadline_unix: float | None = None
    runtime_files: tuple[tuple[str, str], ...] = ()
    device: str = "cpu"
    device_parity_sha256: str | None = None

    def __post_init__(self):
        # Nested tuples and frozen members prevent in-process spec edits.
        if not all(
            isinstance(x, tuple)
            for x in (self.contexts, self.arms, self.seeds, self.outcome_classes)
        ):
            raise ValueError("game collections must be immutable tuples")
        if not self.contexts or not self.arms or not self.seeds or not self.game_id:
            raise ValueError("empty game")
        if len({c.context_id for c in self.contexts}) != len(self.contexts):
            raise ValueError("duplicate context")
        if len({a.name for a in self.arms}) != len(self.arms) or len(
            set(self.seeds)
        ) != len(self.seeds):
            raise ValueError("duplicate arm or seed")
        if any(type(s) is not int for s in self.seeds):
            raise ValueError("seeds must be integers")
        if (
            self.budget != 8
            or self.tau != 0.01
            or self.bootstrap != 2000
            or self.bootstrap_seed != 7
        ):
            raise ValueError(
                "tonight's preregistered B/tau/bootstrap settings are frozen"
            )
        if type(self.slate_k) is not int or self.slate_k not in (8, 32):
            raise ValueError("slate size must be baseline eight or rerank 32")
        if (
            any(a.name.startswith("rerank32:") for a in self.arms)
            and self.slate_k != 32
        ):
            raise ValueError("rerank32 requires preregistered K=32")
        if self.benchmark_valid is not False or self.kpi_version != "rlfw-kpi/v1":
            raise ValueError("research results only")
        if (
            self.protocol != "jope.flywheel-s1-inner-rmse/v1"
            or self.comparator != "original-program-reexecuted-current-protocol"
        ):
            raise ValueError("incompatible protocol or comparator")
        for name in (
            "role_map_sha256",
            "cache_snapshot_sha256",
            "evaluator_sha256",
            "canonicalizer_sha256",
        ):
            require_sha(getattr(self, name), name)
        for name in (
            "runtime_binding_sha256",
            "policy_code_manifest_sha256",
            "policy_qualification_sha256",
            "device_parity_sha256",
        ):
            if getattr(self, name) is not None:
                require_sha(getattr(self, name), name)
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.schedule not in ("shuffle", "one-per-lineage-then-all"):
            raise ValueError("unknown context-major schedule")
        if self.deadline_unix is not None and (
            not math.isfinite(self.deadline_unix) or self.deadline_unix <= 0
        ):
            raise ValueError("finite campaign deadline required")
        if "hwm" not in {a.name for a in self.arms}:
            raise ValueError(
                "include the original comparator arm, even when reusing its receipt"
            )
        if not isinstance(self.runtime_files, tuple) or len(
            {item[0] for item in self.runtime_files}
        ) != len(self.runtime_files):
            raise ValueError("immutable distinct runtime byte pins required")
        for relative, pin in self.runtime_files:
            if Path(relative).is_absolute():
                raise ValueError("runtime pins are benchmark-relative")
            local_path(self.bench_root, Path(self.bench_root) / relative)
            require_sha(pin, "runtime file")
        local_path(self.bench_root, self.out_dir, "runs")
        local_path(self.bench_root, self.state_path, "state")
        if self.publish_prefix != "program-results/benchmark-rlfw/v1/":
            raise ValueError(
                "publication requires the separate benchmark-rlfw/v1 prefix"
            )
        supports = {
            (a.trie_sha256, a.normalizer_sha256, a.tokenizer_sha256) for a in self.arms
        }
        if len(supports) != 1:
            raise ValueError("all arms require the same support and preprocessing")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return digest(self.to_dict())

    @property
    def pairing_sha256(self):
        # Checkpoint-specific qualifications remain bound by the full game SHA.
        # They certify each arm, but are not shared comparison conditions.
        d = self.to_dict()
        certificates = {self.policy_qualification_sha256, self.device_parity_sha256}
        d["runtime_files"] = [
            item for item in d["runtime_files"] if item[1] not in certificates
        ]
        for key in (
            "game_id",
            "bench_root",
            "out_dir",
            "state_path",
            "arms",
            "slate_k",
            "policy_qualification_sha256",
            "device_parity_sha256",
        ):
            d.pop(key)
        d["support"] = [
            self.arms[0].trie_sha256,
            self.arms[0].normalizer_sha256,
            self.arms[0].tokenizer_sha256,
        ]
        return digest(d)

    @classmethod
    def from_dict(cls, data):
        d = dict(data)
        d["contexts"] = tuple(ContextSpec(**c) for c in d["contexts"])
        d["arms"] = tuple(ArmSpec(**a) for a in d["arms"])
        d["seeds"] = tuple(d["seeds"])
        if "outcome_classes" in d:
            d["outcome_classes"] = tuple(d["outcome_classes"])
        if "runtime_files" in d:
            d["runtime_files"] = tuple(tuple(item) for item in d["runtime_files"])
        return cls(**d)

    def freeze(self):
        out = local_path(self.bench_root, self.out_dir, "runs")
        out.mkdir(parents=True, exist_ok=True)
        path = out / "game_spec.json"
        payload = canonical({"spec_sha256": self.sha256, "spec": self.to_dict()}) + "\n"
        try:
            with path.open("x") as f:
                f.write(payload)
        except FileExistsError:
            if path.read_text() != payload:
                raise ValueError("GameSpec hash changed; freeze and re-spec")
        return path

    def verify_frozen(self):
        path = local_path(self.bench_root, self.out_dir, "runs") / "game_spec.json"
        payload = json.loads(path.read_text())
        if payload != {
            "spec_sha256": self.sha256,
            "spec": json.loads(canonical(self.to_dict())),
        }:
            raise ValueError("GameSpec hash changed mid-run")
        for relative, pin in self.runtime_files:
            file = local_path(self.bench_root, Path(self.bench_root) / relative)
            if hashlib.sha256(file.read_bytes()).hexdigest() != pin:
                raise ValueError("runtime dependency bytes changed mid-run")

    def require_pair(self, other):
        if self.pairing_sha256 != other.pairing_sha256:
            raise ValueError("paired GameSpec hash mismatch")
