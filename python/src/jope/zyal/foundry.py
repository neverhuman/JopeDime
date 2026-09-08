"""Evidence-first control plane for the fourth ZYAL feature-foundry campaign.

This module contains the deterministic parts of the r4 protocol.  It does not
call a model and it never opens DEV or TEST.  Keeping those concerns here makes
``compile`` useful when credentials are absent while leaving the expensive
screening/model stages behind explicit receipts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TAXONOMY = (
    "sequence/flow",
    "AST/program semantics",
    "parameters",
    "expressions",
    "capacity/resources",
    "dataset compatibility",
    "head compatibility",
    "counterfactuals",
    "cost/deployment",
    "invalid/uncomputable",
)

CURRICULUM: dict[int, str] = {
    1: "Executable reconciliation: translate the strongest r3 clusters into precise, non-generic families and repair uncovered promoted concepts.",
    2: "Representation-flow dynamics: bottlenecks, destructive pressure, recovery, width/memory/cost trajectories, and information-loss timing.",
    3: "Counterfactual influence: stage deletion, duplication, reordering, head substitution, redundancy, brittleness, and marginal structural contribution.",
    4: "Parameter geometry: grammar-normalized values, defaults, bounds, coupled constraints, repeated-stage drift, seeds, and capacity/regularization balance.",
    5: "Expression semantics: abstract interpretation, guards, singularities, saturation, curvature, effective degree, symmetry, and variable-interaction topology.",
    6: "Head/pre-head compatibility: effective capacity, sample complexity, representation consumption, sparse/dense behavior, and failure margins.",
    7: "Dataset/genome matching: train-only diagnostics and fold-local embedding interactions, emphasizing regime reversals without dataset identity.",
    8: "Multiscale program structure: graph/path kernels, compression complexity, self-supervised genome embeddings, and long-range motifs beyond n-grams.",
    9: "Adversarial replication: negate, perturb, and recombine SELECT survivors; target source, engine, head, and time instability and synthetic-noise aliases.",
    10: "Final complement and compression: low-latency approximations, orthogonal bundles, missing mechanism coverage, and deployable sufficient statistics.",
}

ROLE_COUNTS = {
    "domain_inventor": 50,
    "cross_domain_synthesizer": 10,
    "blind_judge": 6,
    "memory_curator": 4,
}

BASELINE_COMPONENTS = {
    "story": 124,
    "rich_hashed_genome": 1095,
    "hyperion_pca": 32,
    "tabpfn_v3_pca": 16,
    "embedding_genome_interactions": 64,
}

MODEL_FAMILIES = (
    "ridge",
    "elastic_net",
    "extra_trees",
    "lightgbm_percentile",
    "lightgbm_lambdarank",
    "xgboost_pairwise",
    "catboost_ranker",
    "factorization_machine",
    "dataset_genome_two_tower_mlp",
    "oof_blend_distilled_two_tower_mlp",
)

SELECT_LABELS = frozenset(
    {
        "duplicate",
        "constant",
        "noise_like",
        "select_promising",
        "unstable",
        "head_specific",
        "complementary",
    }
)

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("invalid/uncomputable", ("invalid", "uncomputable", "impossible", "outcome", "label correlation")),
    ("counterfactuals", ("counterfactual", "leave one", "deletion", "delete", "duplicate stage", "swap", "ablation")),
    ("cost/deployment", ("deployment", "latency", "runtime", "cache", "parallel", "memory cost", "cost")),
    ("dataset compatibility", ("dataset", "missingness", "sparsity", "intrinsic dimension", "conditioning", "effective rank")),
    ("head compatibility", ("head", "terminal", "pre-head", "sample complexity")),
    ("expressions", ("expression", "symbolic", "singularity", "saturation", "guard", "lipschitz", "curvature")),
    ("parameters", ("parameter", "hyperparameter", "seed", "default", "bound", "regularization")),
    ("capacity/resources", ("capacity", "width", "bottleneck", "compression", "expansion", "resource", "information retention")),
    ("AST/program semantics", ("ast", "program", "graph", "path kernel", "motif", "compression complexity")),
    ("sequence/flow", ("sequence", "flow", "trajectory", "stage", "operator", "transition", "position")),
)

_FORBIDDEN_THEORY_TERMS = (
    "raw transformed data",
    "historical outcome",
    "historical score",
    "observed outcome",
    "label correlation",
    "correlation with label",
    "target correlation",
    "test score",
    "proxy score",
    "partial cv",
    "candidate rank",
    "dataset identity",
    "undefined embedding",
)

_ALLOWED_INPUTS = frozenset(
    {
        "operator_sequence",
        "operator_multiset",
        "stage_positions",
        "ast_paths",
        "parameter_keys",
        "parameter_values",
        "expression_ast",
        "head_family",
        "resolved_width_trace",
        "runtime_cost_estimates",
        "dataset_train_shape",
        "dataset_train_statistics",
        "dataset_embedding",
        "engine",
    }
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_append_only_json(
    directory: Path, stem: str, identity: Mapping[str, Any], value: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Publish a content-addressed receipt without replacing an earlier byte."""

    identity_sha256 = object_hash(identity)
    path = directory / f"{stem}-{identity_sha256[:20]}.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("identity_sha256") != identity_sha256:
            raise ValueError(f"append-only receipt collision at {path}")
        stored_sha256 = current.get("receipt_sha256")
        unhashed = dict(current)
        unhashed.pop("receipt_sha256", None)
        if stored_sha256 != object_hash(unhashed):
            raise ValueError(f"append-only receipt was modified at {path}")
        return path, current
    directory.mkdir(parents=True, exist_ok=True)
    receipt = dict(value)
    receipt["identity_sha256"] = identity_sha256
    receipt["receipt_sha256"] = object_hash(receipt)
    payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path, receipt


def read_append_only_json(path: Path) -> dict[str, Any]:
    """Read a content-addressed receipt and reject byte-level tampering."""

    current = json.loads(path.read_text(encoding="utf-8"))
    stored_sha256 = current.get("receipt_sha256")
    unhashed = dict(current)
    unhashed.pop("receipt_sha256", None)
    if stored_sha256 != object_hash(unhashed):
        raise ValueError(f"append-only receipt was modified at {path}")
    return current


def normalize_category(category: Any, concept: Mapping[str, Any] | None = None) -> str:
    """Map a free-form label into the controller-owned taxonomy."""

    pieces = [str(category or "")]
    if concept:
        pieces.extend(str(concept.get(key) or "") for key in ("feature_name", "mechanism", "formula"))
    text = " ".join(pieces).casefold().replace("_", "-")
    for normalized, needles in _CATEGORY_RULES:
        if any(needle in text for needle in needles):
            return normalized
    return "invalid/uncomputable"


def category_record(concept: Mapping[str, Any]) -> dict[str, str]:
    original = str(concept.get("category") or "")
    return {"original_category": original, "normalized_category": normalize_category(original, concept)}


def generation_zero_errors(concept: Mapping[str, Any]) -> list[str]:
    """Return concrete legality/computability failures for a theory."""

    text = " ".join(str(value) for value in concept.values()).casefold().replace("_", " ")
    errors = [f"requires forbidden source: {term}" for term in _FORBIDDEN_THEORY_TERMS if term in text]
    inputs = concept.get("inputs")
    if not isinstance(inputs, list):
        errors.append("inputs must be a list of generation-zero authorities")
    else:
        unknown = sorted({str(item) for item in inputs}.difference(_ALLOWED_INPUTS))
        if unknown:
            errors.append(f"unsupported generation-zero inputs: {unknown}")
    if "embedding" in text:
        qualified = any(
            term in text
            for term in ("dataset_embedding", "dataset embedding", "self-supervised", "masked-stage", "invariance")
        )
        if not qualified:
            errors.append("embedding authority/objective is undefined")
    cyclic_structure = re.search(
        r"\bcyclic(?:\s+\w+){0,2}\s+(?:expression|ast|tree|path)s?\b"
        r"|\b(?:expression|ast|tree|path)(?:\s+\w+){0,2}\s+cyclic\b",
        text,
    )
    if cyclic_structure:
        errors.append("expression trees must be acyclic")
    return sorted(set(errors))


_PRIMITIVES_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "sequence/flow": (
        "operator_entropy", "transition_entropy", "long_range_repeat", "family_switches",
        "cost_centroid", "cost_switches", "n_stages", "n_transforms",
    ),
    "AST/program semantics": (
        "path_complexity", "genome_compression_ratio", "long_range_repeat", "ast_depth",
        "ast_constant_count", "expr_len", "n_exprs", "col_refs",
    ),
    "parameters": (
        "param_default_distance", "param_bound_distance", "param_constraint_ratio",
        "repeated_param_drift", "seed_collision", "regularization_capacity_balance",
        "missing_default_topology", "numeric_param_count",
    ),
    "expressions": (
        "expr_constant_folded_size", "expr_effective_variables", "expr_degree_proxy",
        "expr_frequency_proxy", "expr_interval_growth", "expr_singularity_risk",
        "expr_saturation_risk", "expr_guard_coverage", "expr_lipschitz_proxy",
        "expr_monotonicity_proxy", "expr_symmetry_proxy", "expr_branch_balance",
        "expr_variable_interactions",
    ),
    "capacity/resources": (
        "cumulative_log_width", "information_retention_proxy", "width_cost_area",
        "bottleneck_depth", "bottleneck_duration", "bottleneck_location",
        "recovery_after_compression", "expansion_overshoot", "maximum_width_drawdown",
        "destructive_pressure_waves", "width_change_points",
    ),
    "dataset compatibility": (
        "dataset_effective_rank", "dataset_intrinsic_dimension", "dataset_sparsity",
        "train_missingness", "condition_diagnostic", "rank_diagnostic", "train_rows",
        "train_width", "head_supply_demand_gap", "regularization_capacity_balance",
    ),
    "head compatibility": (
        "head_representation_supply", "head_representation_demand", "head_supply_demand_gap",
        "effective_capacity", "regularization_capacity_balance", "runtime_cost_estimate",
        "train_rows", "train_width",
    ),
    "counterfactuals": (
        "leave_one_stage_out_delta", "adjacent_swap_delta", "duplicate_stage_delta",
        "head_swap_delta", "parameter_default_delta", "operator_entropy",
        "cost_centroid", "family_switches",
    ),
    "cost/deployment": (
        "semantic_cacheability", "runtime_cost_estimate", "peak_memory_proxy",
        "parallelism_proxy", "width_cost_area", "total_cost", "max_cost", "cost_centroid",
    ),
}

_KEYWORD_PRIMITIVES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("recovery",), "recovery_after_compression"),
    (("bottleneck",), "bottleneck_depth"),
    (("drawdown",), "maximum_width_drawdown"),
    (("leave one", "deletion", "delete"), "leave_one_stage_out_delta"),
    (("adjacent swap", "reorder"), "adjacent_swap_delta"),
    (("duplicate stage", "duplication"), "duplicate_stage_delta"),
    (("default",), "param_default_distance"),
    (("bound",), "param_bound_distance"),
    (("seed",), "seed_collision"),
    (("singular",), "expr_singularity_risk"),
    (("saturat",), "expr_saturation_risk"),
    (("guard",), "expr_guard_coverage"),
    (("symmetr",), "expr_symmetry_proxy"),
    (("head demand", "consumption"), "head_representation_demand"),
    (("effective rank",), "dataset_effective_rank"),
    (("intrinsic dimension",), "dataset_intrinsic_dimension"),
    (("spars",), "dataset_sparsity"),
    (("cache",), "semantic_cacheability"),
    (("parallel",), "parallelism_proxy"),
)


def theory_descriptor(concept: Mapping[str, Any]) -> dict[str, Any]:
    categories = category_record(concept)
    normalized = categories["normalized_category"]
    text = " ".join(
        str(concept.get(key) or "") for key in ("feature_name", "mechanism", "formula", "category")
    ).casefold()
    formula_text = str(concept.get("formula") or "").casefold()
    candidates = list(_PRIMITIVES_BY_CATEGORY.get(normalized, ()))
    observed: list[str] = []
    for source in (formula_text, text):
        for needles, primitive in _KEYWORD_PRIMITIVES:
            if any(needle in source for needle in needles) and primitive not in observed:
                observed.append(primitive)
    for primitive in candidates:
        if primitive not in observed:
            observed.append(primitive)
    return {
        **categories,
        "observables": observed,
        "transformations": ["raw", "signed", "absolute", "log", "normalized", "thresholded", "positional"],
        "interactions": int(concept.get("interaction_order") or 1),
        "invariances": sorted(map(str, concept.get("invariances") or [])),
        "falsifier": str(concept.get("falsifier") or ""),
        "engine_scope": "both" if "engine" in (concept.get("inputs") or []) else "declared_by_corpus",
        "dataset_scope": "train_only" if any(str(item).startswith("dataset_") for item in concept.get("inputs") or []) else "genome_only",
        "extraction_cost_class": str(concept.get("cost_class") or "unknown"),
    }


def expression_hash(expression: Any) -> str:
    return object_hash(expression)


def expression_set_hash(outputs: Sequence[Mapping[str, Any]]) -> str:
    return object_hash(sorted(expression_hash(output["expr"]) for output in outputs))


def shared_baseline_family() -> dict[str, Any]:
    outputs = [
        {"name": "stage_count", "expr": ["n_stages"]},
        {"name": "transform_count", "expr": ["n_transforms"]},
        {"name": "stage_count_log", "expr": ["log1p", ["n_stages"]]},
        {"name": "total_cost", "expr": ["total_cost"]},
        {"name": "total_cost_log", "expr": ["log1p", ["total_cost"]]},
        {"name": "maximum_cost", "expr": ["max_cost"]},
        {"name": "cost_centroid", "expr": ["cost_centroid"]},
        {"name": "family_switches", "expr": ["family_switches"]},
    ]
    spec = {
        "schema": "FeatureSpec.v1",
        "family_id": "zyal.shared.anchors",
        "rationale": "Frozen common stage/count/cost anchors, emitted once for the campaign.",
        "outputs": outputs,
    }
    return {
        "schema": "FeatureFamilySpec.v1",
        "family_id": spec["family_id"],
        "cluster_id": "shared.baseline",
        "disposition": "shared_baseline",
        "reason": None,
        "columns": [item["name"] for item in outputs],
        "spec": spec,
        "spec_sha256": object_hash(spec),
        "theory": {"normalized_category": "shared_baseline", "observables": []},
        "output_expression_hashes": [expression_hash(item["expr"]) for item in outputs],
        "output_expression_set_sha256": expression_set_hash(outputs),
        "translation_attempts": 1,
        "alias_of": None,
    }


def _name(value: str, suffix: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if len(base) > 24:
        base = f"{base[:20]}_{hashlib.sha256(base.encode()).hexdigest()[:4]}"
    return f"{base}_{suffix}"[:40]


def _local_variants(observables: Sequence[str], dataset_conditioned: bool) -> list[dict[str, Any]]:
    central = observables[0]
    complement = observables[1] if len(observables) > 1 else "n_stages"
    raw = [central]
    other = [complement]
    outputs = [
        {"name": _name(central, "raw"), "expr": raw},
        {"name": _name(central, "signed"), "expr": ["sub", raw, ["div", ["n_stages"], 2]]},
        {"name": _name(central, "absolute"), "expr": ["abs", raw]},
        {"name": _name(central, "log_abs"), "expr": ["log1p", ["abs", raw]]},
        {"name": _name(central, "per_stage"), "expr": ["div", raw, ["max", 1, ["n_stages"]]]},
        {"name": _name(central, "threshold"), "expr": ["max", 0, ["sub", raw, 1]]},
        {"name": _name(central, "position"), "expr": ["mul", raw, ["cost_centroid"]]},
        {"name": _name(complement, "raw"), "expr": other},
        {"name": _name(central, "complement"), "expr": ["mul", raw, other]},
        {"name": _name(central, "cost"), "expr": ["div", ["mul", raw, ["total_cost"]], ["max", 1, ["n_stages"]]]},
    ]
    if len(observables) > 2:
        third = [observables[2]]
        outputs.extend(
            [
                {"name": _name(observables[2], "raw"), "expr": third},
                {"name": _name(central, "three_way"), "expr": ["mul", ["mul", raw, other], third]},
            ]
        )
    if dataset_conditioned and central not in {"train_rows", "train_width"}:
        outputs.extend(
            [
                {"name": _name(central, "rows"), "expr": ["div", ["mul", raw, ["log1p", ["train_rows"]]], ["max", 1, ["train_width"]]]},
                {"name": _name(central, "missing"), "expr": ["mul", raw, ["train_missingness"]]},
            ]
        )
    return outputs[:25]


def compile_theory_family(
    cluster_id: str,
    concept: Mapping[str, Any],
    used_expression_sets: dict[str, str],
    *,
    alias_of: str | None = None,
) -> dict[str, Any]:
    """Translate one theory, rejecting rather than fabricating an unfaithful family."""

    theory = theory_descriptor(concept)
    errors = generation_zero_errors(concept)
    if theory["normalized_category"] == "invalid/uncomputable":
        errors.append("category cannot be translated to an executable mechanism")
    observables = theory["observables"]
    if not observables:
        errors.append("no supported executable observable was identified")
    family_id = "zyal." + cluster_id.removeprefix("cluster.")
    if errors:
        return {
            "schema": "FeatureFamilySpec.v1", "family_id": None, "cluster_id": cluster_id,
            "disposition": "rejected", "reason": "; ".join(sorted(set(errors))), "columns": [],
            "spec": None, "spec_sha256": "0" * 64, "theory": theory,
            "output_expression_hashes": [], "output_expression_set_sha256": "0" * 64,
            "translation_attempts": 1, "alias_of": alias_of,
        }
    outputs = _local_variants(observables, theory["dataset_scope"] == "train_only")
    set_sha256 = expression_set_hash(outputs)
    duplicate = used_expression_sets.get(set_sha256)
    if duplicate and alias_of is None:
        return {
            "schema": "FeatureFamilySpec.v1", "family_id": None, "cluster_id": cluster_id,
            "disposition": "rejected", "reason": f"identical output-expression set to non-alias family {duplicate}",
            "columns": [], "spec": None, "spec_sha256": "0" * 64, "theory": theory,
            "output_expression_hashes": [expression_hash(item["expr"]) for item in outputs],
            "output_expression_set_sha256": set_sha256, "translation_attempts": 2,
            "alias_of": None,
        }
    if alias_of:
        return {
            "schema": "FeatureFamilySpec.v1", "family_id": None, "cluster_id": cluster_id,
            "disposition": "alias", "reason": "inherits representative family", "columns": [],
            "spec": None, "spec_sha256": "0" * 64, "theory": theory,
            "output_expression_hashes": [expression_hash(item["expr"]) for item in outputs],
            "output_expression_set_sha256": set_sha256, "translation_attempts": 1,
            "alias_of": alias_of,
        }
    spec = {
        "schema": "FeatureSpec.v1",
        "family_id": family_id,
        "rationale": str(concept.get("mechanism") or "")[:1000],
        "outputs": outputs,
    }
    used_expression_sets[set_sha256] = family_id
    return {
        "schema": "FeatureFamilySpec.v1", "family_id": family_id, "cluster_id": cluster_id,
        "disposition": "declarative_featurespec", "reason": None,
        "columns": [item["name"] for item in outputs], "spec": spec,
        "spec_sha256": object_hash(spec), "theory": theory,
        "output_expression_hashes": [expression_hash(item["expr"]) for item in outputs],
        "output_expression_set_sha256": set_sha256, "translation_attempts": 1,
        "alias_of": None,
    }


def validate_expression_set_uniqueness(families: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, str] = {}
    errors: list[str] = []
    for family in families:
        if family.get("disposition") not in {"declarative_featurespec", "llm_featurespec", "sandboxed_plugin"}:
            continue
        digest = str(family.get("output_expression_set_sha256") or "")
        family_id = str(family.get("family_id") or "")
        if digest in seen and not family.get("alias_of"):
            errors.append(f"{family_id} and {seen[digest]} share an output-expression set")
        else:
            seen[digest] = family_id
    return errors


def build_concept_graph(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build bounded semantic edges without a quadratic all-pairs comparison."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    cluster_representatives: dict[str, str] = {}
    observable_owners: dict[str, str] = {}
    for record in records:
        concept_id = str(record["concept_id"])
        cluster_id = str(record["cluster_id"])
        concept = record.get("payload") or {}
        descriptor = theory_descriptor(concept)
        nodes.append(
            {
                "concept_id": concept_id,
                "cluster_id": cluster_id,
                "valid": bool(record.get("valid")),
                "normalized_category": descriptor["normalized_category"],
                "observables": descriptor["observables"],
            }
        )
        representative = cluster_representatives.setdefault(cluster_id, concept_id)
        if representative != concept_id:
            edges.append({"source": concept_id, "target": representative, "kind": "semantic_alias"})
        for observable in descriptor["observables"][:3]:
            owner = observable_owners.setdefault(observable, concept_id)
            if owner != concept_id:
                edges.append({"source": concept_id, "target": owner, "kind": "shared_observable"})
        text = f"{concept.get('mechanism', '')} {concept.get('formula', '')}".casefold()
        if "counterfactual" in text or "ablation" in text:
            edges.append({"source": concept_id, "target": representative, "kind": "counterfactual"})
        if "refine" in text or "approximation" in text:
            edges.append({"source": concept_id, "target": representative, "kind": "refinement"})
    return {
        "schema": "ZyalConceptGraph.v1",
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def executable_coverage(
    records: Sequence[Mapping[str, Any]], families: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_cluster = {str(item["cluster_id"]): item for item in families}
    mechanism_counts: Counter[str] = Counter()
    unsupported: Counter[str] = Counter()
    expression_counts: Counter[str] = Counter()
    generic_only: list[str] = []
    for record in records:
        if not record.get("valid"):
            continue
        cluster_id = str(record["cluster_id"])
        family = by_cluster.get(cluster_id)
        descriptor = theory_descriptor(record.get("payload") or {})
        if not family or family.get("disposition") != "declarative_featurespec":
            unsupported.update(descriptor["observables"] or [descriptor["normalized_category"]])
            continue
        mechanism_counts.update(family["theory"]["observables"])
        expression_counts.update(family.get("output_expression_hashes") or [])
        if set(family["theory"]["observables"]).issubset({"n_stages", "n_transforms", "total_cost"}):
            generic_only.append(cluster_id)
    all_mechanisms = {item for values in _PRIMITIVES_BY_CATEGORY.values() for item in values}
    return {
        "schema": "ZyalExecutableCoverage.v1",
        "executable_observables": dict(sorted(mechanism_counts.items())),
        "missing_mechanisms": sorted(all_mechanisms.difference(mechanism_counts)),
        "unsupported_primitives": dict(sorted(unsupported.items())),
        "duplicated_output_expressions": sorted(
            digest for digest, count in expression_counts.items() if count > 1
        ),
        "generic_count_only_clusters": sorted(set(generic_only)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    return value


def sqlite_table_hash(database: Path, table: str) -> tuple[int, str]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        columns = [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]
        if not columns:
            return 0, object_hash([])
        digest = hashlib.sha256()
        count = 0
        for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
            digest.update(canonical_json([_jsonable(value) for value in row]))
            count += 1
        return count, digest.hexdigest()
    finally:
        connection.close()


def artifact_forensics(
    run_id: str,
    run_root: Path,
    legacy_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Hash preserved evidence and publish an append-only forensic receipt."""

    roots = [(run_id, run_root)]
    if legacy_root and legacy_root.is_dir() and legacy_root.resolve() != run_root.resolve():
        roots.insert(0, (legacy_root.name, legacy_root))
    artifacts: list[dict[str, Any]] = []
    table_receipts: list[dict[str, Any]] = []
    evidence_summary: dict[str, dict[str, Any]] = {}
    table_names = (
        "concepts", "probe_results", "reviews", "curations", "memory",
        "concept_ledger", "compilation_dispositions", "roles", "attempts",
    )

    def generated_by_r4_controller(relative: Path) -> bool:
        """Keep later controller output from recursively changing source evidence."""

        posix = relative.as_posix()
        return (
            posix.startswith("compiled/r4/")
            or posix.startswith("matrices/r4/")
            or posix.startswith("receipts/r4-")
            or posix.startswith("memory/r4-epoch-zero-")
        )

    for source_run_id, root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if generated_by_r4_controller(path.relative_to(root)):
                continue
            artifacts.append(
                {
                    "source_run_id": source_run_id,
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_hash(path),
                }
            )
        database = root / "campaign.sqlite3"
        if database.is_file():
            connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
            existing = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            summary: dict[str, Any] = {}
            if "concepts" in existing:
                concept_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(concepts)")
                }
                summary["concepts"] = int(connection.execute("SELECT COUNT(*) FROM concepts").fetchone()[0])
                if "sandbox_valid" in concept_columns:
                    summary["valid_concepts"] = int(
                        connection.execute("SELECT COALESCE(SUM(sandbox_valid),0) FROM concepts").fetchone()[0]
                    )
                if "payload_json" in concept_columns:
                    summary["original_category_labels"] = int(
                        connection.execute(
                            "SELECT COUNT(DISTINCT json_extract(payload_json,'$.category')) FROM concepts"
                        ).fetchone()[0]
                    )
            for table, key in (
                ("probe_results", "probes"), ("reviews", "judgments"), ("curations", "curations")
            ):
                if table in existing:
                    summary[key] = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            if "reviews" in existing:
                summary["judged_concepts"] = int(
                    connection.execute("SELECT COUNT(DISTINCT concept_id) FROM reviews").fetchone()[0]
                )
                summary["maximum_reviews_per_concept"] = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(n),0) FROM (SELECT COUNT(*) n FROM reviews GROUP BY concept_id)"
                    ).fetchone()[0]
                )
            if "curations" in existing:
                summary["curated_concepts"] = int(
                    connection.execute("SELECT COUNT(DISTINCT concept_id) FROM curations").fetchone()[0]
                )
            if "concept_ledger" in existing:
                total, clusters, null_probes = connection.execute(
                    "SELECT COUNT(*),COUNT(DISTINCT cluster_id),"
                    "SUM(probe_receipt_sha256 IS NULL) FROM concept_ledger"
                ).fetchone()
                pending_concepts, pending_unique = connection.execute(
                    "SELECT COUNT(*),COUNT(DISTINCT cluster_id) FROM concept_ledger WHERE cluster_id IN ("
                    "SELECT cluster_id FROM concept_ledger GROUP BY cluster_id "
                    "HAVING SUM(probe_receipt_sha256 IS NOT NULL)=0)"
                ).fetchone()
                summary.update(
                    {
                        "combined_concepts": int(total),
                        "semantic_clusters": int(clusters),
                        "raw_null_probe_links": int(null_probes or 0),
                        "pending_after_alias_inheritance": int(pending_concepts),
                        "pending_unique_representatives": int(pending_unique),
                    }
                )
            evidence_summary[source_run_id] = summary
            connection.close()
            for table in table_names:
                if table in existing:
                    count, sha256 = sqlite_table_hash(database, table)
                    table_receipts.append(
                        {"source_run_id": source_run_id, "table": table, "rows": count, "sha256": sha256}
                    )
    identity = {
        "run_id": run_id,
        "artifacts": artifacts,
        "tables": table_receipts,
        "evidence_summary": evidence_summary,
    }
    counts = {f"{item['source_run_id']}:{item['table']}": item["rows"] for item in table_receipts}
    receipt = {
        "schema": "ZyalArtifactForensicsReceipt.v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "preservation": "source artifacts are read-only; this receipt is append-only",
        "artifacts": artifacts,
        "table_receipts": table_receipts,
        "counts": counts,
        "evidence_summary": evidence_summary,
        "concept_hashes": [item for item in table_receipts if item["table"] in {"concepts", "concept_ledger"}],
        "probe_hashes": [item for item in table_receipts if item["table"] == "probe_results"],
        "judgment_hashes": [item for item in table_receipts if item["table"] == "reviews"],
        "curation_hashes": [item for item in table_receipts if item["table"] in {"curations", "memory"}],
        "duplicate_hashes": [item for item in artifacts if "duplicate" in item["path"]],
        "compilation_hashes": [item for item in table_receipts if item["table"] == "compilation_dispositions"],
        "coverage_hashes": [item for item in artifacts if "coverage" in item["path"]],
    }
    return write_append_only_json(run_root / "receipts/r4-forensics", "artifact-forensics", identity, receipt)


def probe_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Resolve SELECT probes and aliases while retaining legacy evidence as diagnostic."""

    select_by_cluster: dict[str, str] = {}
    valid_by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("valid"):
            continue
        cluster_id = str(record["cluster_id"])
        valid_by_cluster[cluster_id].append(record)
        probe = record.get("select_probe_receipt_sha256")
        if not probe and record.get("source_kind") != "legacy_seed":
            probe = record.get("probe_receipt_sha256")
        if probe:
            select_by_cluster.setdefault(cluster_id, str(probe))

    pending_representatives: dict[str, str] = {}
    for cluster_id, members in sorted(valid_by_cluster.items()):
        if cluster_id in select_by_cluster:
            continue
        representative = next(
            (row for row in members if row.get("probe_receipt_sha256")), members[0]
        )
        pending_representatives[cluster_id] = str(representative["concept_id"])

    ledger = []
    for record in records:
        concept_id = str(record["concept_id"])
        cluster_id = str(record["cluster_id"])
        legacy = record.get("probe_receipt_sha256")
        direct = record.get("select_probe_receipt_sha256")
        if not direct and record.get("source_kind") != "legacy_seed":
            direct = legacy
        inherited = select_by_cluster.get(cluster_id)
        if not record.get("valid"):
            state, receipt = "not_applicable_invalid", None
        elif direct:
            state, receipt = "select_probe", str(direct)
        elif inherited:
            state, receipt = "alias_inherited", inherited
        elif legacy:
            state, receipt = "legacy_exploratory_nonselection", None
        else:
            state, receipt = "pending_select_probe", None
        ledger.append(
            {
                "concept_id": concept_id,
                "cluster_id": cluster_id,
                "state": state,
                "probe_receipt_sha256": receipt,
                "legacy_probe_receipt_sha256": str(legacy) if legacy else None,
            }
        )
    pending_clusters = set(pending_representatives)
    return {
        "schema": "ZyalProbeCoverage.v1",
        "records": ledger,
        "total": len(ledger),
        "covered": sum(
            item["state"] in {"select_probe", "alias_inherited", "not_applicable_invalid"}
            for item in ledger
        ),
        "pending_concepts": sum(
            item["cluster_id"] in pending_clusters and item["state"] != "not_applicable_invalid"
            for item in ledger
        ),
        "pending_unique": len(pending_representatives),
        "pending_representatives": pending_representatives,
    }


def epoch_curriculum_receipt(run_id: str, roles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    role_ids = [str(role["id"]) for role in roles]
    counts_by_epoch: dict[int, Counter[str]] = defaultdict(Counter)
    for role in roles:
        counts_by_epoch[int(role["epoch"])][str(role["class"])] += 1
    errors = []
    if len(roles) != 700 or len(set(role_ids)) != 700:
        errors.append("campaign must contain 700 distinct original role identities")
    for epoch in range(1, 11):
        if dict(counts_by_epoch[epoch]) != ROLE_COUNTS:
            errors.append(f"epoch {epoch} role allocation is {dict(counts_by_epoch[epoch])}")
    return {
        "schema": "EpochCurriculum.v1",
        "run_id": run_id,
        "epochs": [{"epoch": epoch, "curriculum": CURRICULUM[epoch], "roles": ROLE_COUNTS} for epoch in range(1, 11)],
        "original_role_identities": role_ids,
        "complete": not errors,
        "errors": errors,
        "sha256": object_hash({"curriculum": CURRICULUM, "role_ids": role_ids}),
    }


def baseline_schema() -> dict[str, Any]:
    total = sum(BASELINE_COMPONENTS.values())
    if total != 1331:
        raise AssertionError(f"baseline schema drifted to {total} columns")
    return {
        "schema": "ZyalBaselineSchema.v1",
        "components": BASELINE_COMPONENTS,
        "feature_count": total,
        "outer_folds": 5,
        "pca_fit_inside_outer_training_fold": True,
        "production_stratum": "honest_featmath",
        "newmath_head": "independent",
        "legacy_strata_authority": "confounding_diagnostics_only",
    }


def validate_authority_file(path: Path, *, expected_fields: int, expected_schema_sha256: str | None = None) -> dict[str, Any]:
    """Validate content/schema evidence; filenames carry no authority."""

    raw_sha256 = file_hash(path)
    field_count: int | None = None
    schema_sha256: str | None = None
    errors: list[str] = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict):
        names = value.get("field_names") or value.get("fields") or value.get("columns")
        if isinstance(names, list):
            field_count = len(names)
            schema_sha256 = object_hash(list(map(str, names)))
        elif isinstance(value.get("field_count"), int) and isinstance(value.get("schema_sha256"), str):
            field_count = int(value["field_count"])
            schema_sha256 = str(value["schema_sha256"])
    if field_count is None:
        errors.append("content does not carry a verifiable field schema")
    elif field_count != expected_fields:
        errors.append(f"expected {expected_fields} fields, found {field_count}")
    if expected_schema_sha256 and schema_sha256 != expected_schema_sha256:
        errors.append("schema sha256 does not match the pinned authority")
    return {
        "path": str(path), "sha256": raw_sha256, "field_count": field_count,
        "schema_sha256": schema_sha256, "verified": not errors, "errors": errors,
    }


def prospective_holdout_receipt(
    *,
    run_id: str,
    frozen_manifest_sha256: str,
    holdout_snapshot_sha256: str,
    evaluated_at: str,
    feature_manifest_sha256: str,
    model_manifest_sha256: str,
    source_family_reconciled: bool,
    fastcuda_recertified: bool,
    paired_bootstrap_ci: Sequence[float] | None,
    synthetic_noise_rejected: bool,
    capture_regression_material: bool,
    regret_regression_material: bool,
) -> dict[str, Any]:
    ci = list(paired_bootstrap_ci) if paired_bootstrap_ci is not None else None
    positive_ci = bool(ci and len(ci) == 2 and all(math.isfinite(float(x)) for x in ci) and float(ci[0]) > 0)
    gates = {
        "post_freeze": bool(frozen_manifest_sha256),
        "positive_paired_bootstrap_ci": positive_ci,
        "synthetic_noise_rejected": bool(synthetic_noise_rejected),
        "capture_no_material_regression": not capture_regression_material,
        "regret_no_material_regression": not regret_regression_material,
        "source_family_reconciled": bool(source_family_reconciled),
        "fastcuda_host_recertified": bool(fastcuda_recertified),
    }
    return {
        "schema": "ProspectiveHoldoutReceipt.v1",
        "run_id": run_id,
        "evaluated_at": evaluated_at,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "holdout_snapshot_sha256": holdout_snapshot_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "paired_bootstrap_ci": ci,
        "gates": gates,
        "immutable_after_evaluation": True,
        "promotion_authority": all(gates.values()),
        "shadow_only": not all(gates.values()),
    }


def challenger_protocol() -> dict[str, Any]:
    return {
        "schema": "ZyalChallengerProtocol.v1",
        "model_families": MODEL_FAMILIES,
        "lexicographic_metrics": [
            "within_group_spearman", "capture_at_3", "mean_regret", "percentile_pearson", "latency"
        ],
        "subset_sizes": [200, 300, 400, 500],
        "subset_selection": "nested_independently_inside_each_outer_training_fold",
        "winners": ["unrestricted_research_accuracy", "best_200_500_subset", "latency_qualified_deployment"],
    }
