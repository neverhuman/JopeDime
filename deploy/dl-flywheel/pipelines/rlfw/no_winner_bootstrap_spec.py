"""Immutable preregistered no-winner bootstrap specification."""

from dataclasses import asdict, dataclass
from pathlib import Path

from rlfw.arena.game_spec import (
    R5_KPI_VERSION,
    ArmSpec,
    GameSpec,
    digest,
    local_path,
    require_sha,
)
from rlfw.research_hwm import HWMKey
from rlfw.no_winner_bootstrap import CampaignWindow, TrainingContext


@dataclass(frozen=True)
class BootstrapSpec:
    game_id: str
    bench_root: str
    out_dir: str
    state_path: str
    contexts: tuple[TrainingContext, ...]
    arms: tuple[ArmSpec, ...]
    role_map_sha256: str
    evaluator_sha256: str
    canonicalizer_sha256: str
    calibration_sha256: str
    runtime_binding_sha256: str
    policy_code_manifest_sha256: str
    policy_qualification_sha256: str
    runtime_files: tuple[tuple[str, str], ...]
    campaign_window: CampaignWindow
    previously_labeled_parents: tuple[str, ...] = ()
    cache_snapshot_sha256: str = digest({})
    protocol: str = "jope.flywheel-s1-inner-rmse/v1"
    seeds: tuple[int, ...] = (20261009,)
    budget: int = 8
    slate_k: int = 32
    tau: float = 0.01
    order_seed: int = 20261009
    schedule: str = "shuffle"
    device: str = "cpu"
    device_parity_sha256: str | None = None
    operator: str = "coverage_5_2_1"
    kpi_version: str = R5_KPI_VERSION

    def __post_init__(self):
        if (
            not isinstance(self.contexts, tuple)
            or not 1 <= len(self.contexts) <= 100
            or any(not isinstance(c, TrainingContext) for c in self.contexts)
        ):
            raise ValueError("wave requires 1..100 original train contexts")
        if len({c.context_id for c in self.contexts}) != len(self.contexts) or len(
            {c.lineage_id for c in self.contexts}
        ) != len(self.contexts):
            raise ValueError("one context per original parent per wave")
        if (
            not isinstance(self.arms, tuple)
            or len(self.arms) != 1
            or not self.arms[0].name.startswith("rerank_mix:")
            or self.arms[0].temperature != 1.0
        ):
            raise ValueError("one fixed-temperature mixed coverage arm required")
        if (
            self.seeds,
            self.budget,
            self.slate_k,
            self.tau,
            self.order_seed,
            self.operator,
        ) != ((20261009,), 8, 32, 0.01, 20261009, "coverage_5_2_1"):
            raise ValueError("preregistered wave settings cannot change")
        if self.device not in ("cpu", "cuda") or not isinstance(
            self.campaign_window, CampaignWindow
        ):
            raise ValueError("explicit device and injected campaign window required")
        if self.kpi_version != R5_KPI_VERSION:
            raise ValueError("bootstrap waves use the R5 program-parity outcomes")
        if self.cache_snapshot_sha256 != digest({}):
            raise ValueError("bootstrap starts with the frozen empty research cache")
        for field in (
            "role_map_sha256",
            "evaluator_sha256",
            "canonicalizer_sha256",
            "calibration_sha256",
            "runtime_binding_sha256",
            "policy_code_manifest_sha256",
            "policy_qualification_sha256",
        ):
            require_sha(getattr(self, field), field)
        if (
            not isinstance(self.runtime_files, tuple)
            or len(dict(self.runtime_files)) != len(self.runtime_files)
            or not self.runtime_files
        ):
            raise ValueError("distinct immutable runtime byte pins required")
        pins = dict(self.runtime_files)
        if (
            self.calibration_sha256 not in pins.values()
            or self.runtime_binding_sha256 not in pins.values()
        ):
            raise ValueError(
                "binding and coverage calibration must be sealed runtime dependencies"
            )
        for name, pin in self.runtime_files:
            if Path(name).is_absolute():
                raise ValueError("runtime files are benchmark-relative")
            local_path(self.bench_root, Path(self.bench_root) / name)
            require_sha(pin, name)
        if (
            self.schedule != "shuffle"
            or self.protocol != "jope.flywheel-s1-inner-rmse/v1"
            or not isinstance(self.previously_labeled_parents, tuple)
        ):
            raise ValueError(
                "fixed native protocol, seeded schedule and immutable prior-parent inventory required"
            )
        local_path(self.bench_root, self.out_dir, "runs")
        local_path(self.bench_root, self.state_path, "state")
        for c in self.contexts:
            key = c.hwm_key
            if (
                key.evaluator_sha256,
                key.canonicalizer_sha256,
                key.protocol,
                key.vocabulary_sha256,
                key.trie_sha256,
            ) != (
                self.evaluator_sha256,
                self.canonicalizer_sha256,
                self.protocol,
                self.arms[0].tokenizer_sha256,
                self.arms[0].trie_sha256,
            ):
                raise ValueError("wave/research comparison identity mismatch")

    def to_dict(self):
        return {"schema": "jope.rlfw-bootstrap-spec/v2", **asdict(self)}

    @property
    def sha256(self):
        return digest(self.to_dict())

    @property
    def deadline_unix(self):
        # The shared scope and oracle checks read the injected window's deadline.
        return self.campaign_window.deadline_unix

    # Reuse byte-custody operations; no development GameSpec is constructed.
    freeze = GameSpec.freeze
    verify_frozen = GameSpec.verify_frozen

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        if d.pop("schema") != "jope.rlfw-bootstrap-spec/v2":
            raise ValueError("incompatible bootstrap schema")
        if d.get("campaign_window") is None:
            raise ValueError("bootstrap spec must bind its injected campaign window")
        d["campaign_window"] = CampaignWindow(**d["campaign_window"])
        contexts = []
        for item in d["contexts"]:
            c = dict(item)
            key = dict(c["hwm_key"])
            key["inner_folds"] = tuple(key["inner_folds"])
            c["hwm_key"] = HWMKey(**key)
            contexts.append(TrainingContext(**c))
        d["contexts"] = tuple(contexts)
        d["arms"] = tuple(ArmSpec(**a) for a in d["arms"])
        for field in ("seeds", "previously_labeled_parents"):
            d[field] = tuple(d[field])
        d["runtime_files"] = tuple(tuple(p) for p in d["runtime_files"])
        return cls(**d)
