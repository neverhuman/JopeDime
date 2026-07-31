"""WS-2: the scripts/ lane's binding to the ONE scq spec policy.

The policy itself lives in Rust (`src/spec_policy.rs`). This module reads the
generated projection `scripts/jope-spec-policy.json` so a Python renderer emits
byte-identical `[resource]` text to its Rust counterpart instead of keeping its
own copy of the numbers.

Regenerate the JSON after changing the Rust policy:

    cargo run --release --bin jope_spec_drift -- --emit-json \\
        > scripts/jope-spec-policy.json

`cargo test --lib spec_drift` fails if the checked-in JSON goes stale, and
`jope_spec_drift` fails if any renderer disagrees with it.

Why this exists: on 2026-07-28 the same job family was rendered by five
independent code paths with five different `ram_mb` values -- 49152, 16384,
12288, 8192, 512 -- against a measured peak of ~1 GB for a propose job. Every
cluster stall that day was that one bug class.
"""

from __future__ import annotations

import json
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parent / "jope-spec-policy.json"
SCHEMA = "JopeSpecPolicy.v1"

FAMILIES = ("trainer", "propose", "eval", "encode")


class SpecPolicyError(RuntimeError):
    """The policy file is missing, stale, or does not describe this family."""


def _load() -> dict:
    try:
        raw = POLICY_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecPolicyError(
            f"{POLICY_PATH} is unreadable ({exc}). Regenerate with: "
            "cargo run --release --bin jope_spec_drift -- --emit-json "
            "> scripts/jope-spec-policy.json"
        ) from exc
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA:
        raise SpecPolicyError(
            f"{POLICY_PATH} declares schema {doc.get('schema')!r}, expected {SCHEMA!r}"
        )
    return doc


_DOC = _load()

#: scq subtracts this from a node's available RAM before placing anything
#: (``MEM_SAFETY_MB``, scq-controller/src/state.rs:185). A 49152 MB reservation
#: therefore demands 56 GiB free, which is why it could only ever land on one node.
SCQ_SAFETY_FLOOR_MB: int = _DOC["scq_safety_floor_mb"]

#: Cohort schema. v1 never filtered sealed families, so ~99.9% of 128-draws were
#: unbuildable.
COHORT_SCHEMA: str = _DOC["cohort_schema"]

#: Sealed families are never eligible for cohort draws. Non-negotiable.
EXCLUDE_SEALED_FAMILIES: bool = _DOC["exclude_sealed_families"]


def policy(family: str) -> dict:
    """The full policy record for one job family."""
    try:
        return _DOC["families"][family]
    except KeyError:
        raise SpecPolicyError(
            f"unknown job family {family!r}; known: {sorted(_DOC['families'])}"
        ) from None


def ram_mb(family: str) -> int:
    return int(policy(family)["ram_mb"])


def placement_demand_mb(family: str) -> int:
    """Free RAM a node must have for scq to place this family."""
    return ram_mb(family) + SCQ_SAFETY_FLOOR_MB


def render_resource_block(family: str) -> str:
    """The `[resource]` block, byte-identical to the Rust renderer's."""
    p = policy(family)
    return (
        "[resource]\n"
        f"gpus_per_task = {p['gpus_per_task']}\n"
        f"cpus_reserved = {p['cpus_reserved']}\n"
        f"omp_threads = {p['omp_threads']}\n"
        f"ram_mb = {p['ram_mb']}\n"
        f"gang_size = {p['gang_size']}\n"
        f'placement = "{p["placement"]}"\n'
    )


def render_placement_block(family: str) -> str:
    """The `[[dependencies.smoke]]` blocks -- the ONLY placement mechanism scq
    enforces.

    `[dependencies] hostname = [...]` is a serde NO-OP: `hostname` is not a field
    of scq's DependencySpec (scq-core/src/deps.rs:17-31) and there is no
    `deny_unknown_fields`, so it is dropped, `DependencySpec::is_empty()` becomes
    true, and `prepare_dependencies` CLEARS `dependency_allowed_nodes`
    (scq-controller/src/state.rs:3038-3040) -- placement ends up unconstrained.

    `[resource] node` is rejected outright at submit as controller-only
    (scq-core/src/spec.rs:286-294).

    CAUTION: declaring ANY dependency changes submit semantics. scq probes only
    nodes that ALREADY have the requested GPU/cores/RAM free
    (scq-controller/src/state.rs:3092-3140) and REJECTS the submit when none of
    them pass, instead of queueing. Only add probes to a lane whose submitter
    retries.
    """
    out = []
    for probe in policy(family)["smoke"]:
        args = ", ".join(json.dumps(a) for a in probe["args"])
        out.append(
            "[[dependencies.smoke]]\n"
            f'name = "{probe["name"]}"\n'
            f'command = "{probe["command"]}"\n'
            f"args = [{args}]\n"
            f"timeout_secs = {probe['timeout_secs']}\n"
        )
    return "".join(out)


def render_env_block(family: str, pythonpath: str) -> str:
    """`[env]`. OMP_NUM_THREADS must match `omp_threads` or the node is
    oversubscribed."""
    p = policy(family)
    return f'[env]\nPYTHONPATH = "{pythonpath}"\nOMP_NUM_THREADS = "{p["omp_threads"]}"\n'


def render_limits_block(family: str) -> str:
    p = policy(family)
    return f"[limits]\nmax_runtime_hours = {p['max_runtime_hours']}\nmax_restarts = 1\n"


if __name__ == "__main__":  # pragma: no cover - operator convenience
    for fam in FAMILIES:
        p = policy(fam)
        peak = p["measured_peak_mb"]
        margin = f"{p['ram_mb'] / peak:.2f}x" if peak else "UNMEASURED"
        print(
            f"{fam:9s} ram_mb={p['ram_mb']:6d}  peak={str(peak):>6s}  margin={margin:>10s}"
            f"  demands={placement_demand_mb(fam):6d} MB free  probes={len(p['smoke'])}"
        )
