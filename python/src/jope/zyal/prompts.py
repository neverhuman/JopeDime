"""Role roster, genome sampling, and the strict-JSON prompt envelope.

Every prompt this module emits obeys the three campaign invariants:

1. **The controller owns every ID.** Roles are minted here as
   ``zyal.e{epoch:02d}.{role_class}-{i:02d}``; family artifacts as
   ``zyal.e{epoch:02d}.{role_class}.{slug}`` via :func:`mint_id`. Agents propose a *slug* and
   nothing else -- `validate_payload` never trusts a proposed identifier, and the envelope tells
   the agent so explicitly.
2. **Feature-proposing roles never see numbers.** :func:`build_prompt` runs
   `assert_categorical_only` over every feedback dict in the context *before* serialising, and
   raises `LeakageError` rather than emitting a prompt containing a score. This is a hard gate on
   the serialisation path, not a convention the caller is asked to honour.
3. **Only SELECT is ever exposed.** :func:`sample_genomes` filters to ``split == 'select'`` before
   it samples. DEV and TEST genome text never enters a prompt.

The anti-laziness contract is the other half of the design. An LLM asked for "a feature family"
will happily return ``feature-family-01`` with a one-line rationale forty times in a row; the
minimum-length and placeholder rules in :func:`family_card_contract` and :func:`validate_payload`
are what turn a role into a proposal that can actually be falsified.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
import re
from typing import Any, NamedTuple

from . import (
    CAMPAIGN_SEED,
    GENOME_PREVIEW_N,
    MAX_ATTEMPTS_PER_ROLE,
    PRIOR_SLUG_WINDOW,
    PROMPT_BYTE_CAP,
    SCHEMA_VERSION,
    LeakageError,
    assert_categorical_only,
    assert_no_forbidden_inputs,
)
from .security import MAX_PLUGIN_BYTES

# --- Roster ------------------------------------------------------------------------------------
# NOTE ON THE TOTAL: the campaign brief describes this roster as "70 per epoch, exactly" but the
# per-class figures it specifies sum to 76. The per-class numbers are the concrete specification
# and every downstream budget is per-role-class, so they are preserved verbatim and the import-time
# assertion checks the roster against its own computed total. Resolve the 70-vs-76 discrepancy
# upstream before the first epoch is costed; do NOT silently drop six roles to make a summary line
# come out even.
ROLE_COUNTS: dict[str, int] = {
    "genome_architect": 20,
    "spec_builder": 20,
    "interaction_synthesizer": 8,
    "expression_specialist": 6,
    "param_topology": 6,
    "plugin_writer": 4,
    "data_prospector": 8,
    "adversarial_tester": 2,
    "blind_judge": 2,
}
ROLES_PER_EPOCH = 76
assert sum(ROLE_COUNTS.values()) == ROLES_PER_EPOCH, (
    f"roster is internally inconsistent: {sum(ROLE_COUNTS.values())} != {ROLES_PER_EPOCH}"
)

ENGINES = ("featmath", "newmath")

ROLE_OBJECTIVES: dict[str, str] = {
    "genome_architect": (
        "Invent feature FAMILIES: structural properties of a genome S-expression that plausibly "
        "predict its within-dataset rank, stated as a mechanism that can be falsified."
    ),
    "spec_builder": (
        "Turn one structural observable into a precise, deterministic extraction rule the "
        "controller can compile without asking you a follow-up question."
    ),
    "interaction_synthesizer": (
        "Propose interactions between two structural observables where the pair carries signal "
        "that neither observable carries alone."
    ),
    "expression_specialist": (
        "Mine the arithmetic expression strings inside stage params for structure that "
        "distinguishes strong pipelines from weak ones."
    ),
    "param_topology": (
        "Treat the stage parameter settings as a topology and propose statistics over that "
        "space -- adjacency, dispersion, coherence between neighbouring stages."
    ),
    "plugin_writer": (
        "Write one self-contained plugin defining compute_family(structure, config) that computes "
        "a feature block from parsed genome structure alone."
    ),
    "data_prospector": (
        "Propose analysis probes to run against the SELECT shard only, to discover which "
        "structural observables separate ranks before any family is committed."
    ),
    "adversarial_tester": (
        "Attack the surviving families: name the structure-preserving intervention that should "
        "break each one, so that a family surviving by luck is exposed."
    ),
    "blind_judge": (
        "Rank proposed families on mechanism quality and novelty alone. You are shown no scores "
        "and no outcomes; judge the argument, not the result."
    ),
}

# Which roles run agent-authored code against real labels. Analysis mode is SELECT-only and is
# enforced by the container mount table in security.sandbox_command, not by this flag.
ANALYSIS_ROLES = frozenset({"data_prospector"})

# --- Output schemas ------------------------------------------------------------------------------
# `n_items` is an EXACT count. `item_keys` is an EXACT key set: missing and unexpected keys are both
# errors, because a payload that merely happens to contain what we need is a payload we cannot
# compile deterministically.


class _Schema(NamedTuple):
    output_key: str
    n_items: int
    item_keys: tuple[str, ...]


ROLE_SCHEMAS: dict[str, _Schema] = {
    "genome_architect": _Schema(
        "family_cards", 3,
        ("engine_scope", "falsifier", "family_slug", "mechanism_hypothesis",
         "novelty_argument", "structural_observables"),
    ),
    "spec_builder": _Schema(
        "feature_specs", 4,
        ("aggregation", "extraction_rule", "falsifier", "feature_slug",
         "mechanism_hypothesis", "structural_observables"),
    ),
    "interaction_synthesizer": _Schema(
        "interactions", 4,
        ("combiner", "falsifier", "interaction_slug", "left_observable",
         "mechanism_hypothesis", "right_observable"),
    ),
    "expression_specialist": _Schema(
        "expression_features", 4,
        ("extraction_rule", "falsifier", "feature_slug", "mechanism_hypothesis",
         "target_construct"),
    ),
    "param_topology": _Schema(
        "param_features", 4,
        ("falsifier", "feature_slug", "mechanism_hypothesis", "op_scope", "param_scope",
         "topology_statistic"),
    ),
    "plugin_writer": _Schema(
        "plugins", 1,
        ("falsifier", "family_slug", "mechanism_hypothesis", "novelty_argument",
         "output_columns", "source"),
    ),
    "data_prospector": _Schema(
        "probes", 3,
        ("falsifier", "method", "probe_slug", "question", "structural_observable"),
    ),
    "adversarial_tester": _Schema(
        "attacks", 5,
        ("attack_slug", "expected_breakage", "failure_mode", "structure_preserving_intervention",
         "target_family_slug"),
    ),
    "blind_judge": _Schema(
        "judgements", 8,
        ("card_slug", "reason", "verdict"),
    ),
}
assert set(ROLE_SCHEMAS) == set(ROLE_COUNTS), "every role class needs an output schema"

# Minimum prose lengths. These are the anti-laziness teeth: without them a role returns
# "captures pipeline depth" forty times and nothing is falsifiable.
CONTRACT_MIN_CHARS: dict[str, int] = {
    "mechanism_hypothesis": 80,
    "novelty_argument": 80,
    "falsifier": 40,
    "structure_preserving_intervention": 40,
    "extraction_rule": 40,
    "reason": 40,
}

# `*_slug` fields the agent INVENTS (format-checked) versus fields that REFER to an existing,
# controller-minted identifier (checked only for non-emptiness).
REFERENCE_SLUG_KEYS = frozenset({"card_slug", "target_family_slug"})

LIST_VALUED_KEYS = frozenset({"structural_observables", "output_columns"})

JUDGE_VERDICTS = ("advance", "drop", "hold")

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# Generic names an unmotivated model reaches for first.
_PLACEHOLDER_RE = re.compile(
    r"^(?:feature|family|feature-family|spec|item|candidate|interaction|probe|attack|plugin|"
    r"expression|param|my|new|test)(?:-(?:family|spec|feature|card))?-?\d*$"
)
_PLACEHOLDER_WORDS = ("todo", "tbd", "placeholder", "example", "lorem", "xxx", "foobar")
_SLUG_MIN, _SLUG_MAX = 6, 48

RULES: tuple[str, ...] = (
    "Return one strict JSON object and no markdown or prose.",
    "The first non-whitespace character must be { and the last must be }.",
    "Do not wrap JSON in code fences.",
    "Never use labels, target values, scores, credentials, or network access.",
    "Controller-generated IDs replace all proposed IDs after validation.",
    f"Plugin source is limited to {MAX_PLUGIN_BYTES // 1024} KB and must define "
    "compute_family(structure, config).",
)

_SHRINK_GENOME_FLOOR = 4          # a preview below this teaches the role nothing


# --- IDs ----------------------------------------------------------------------------------------

def mint_id(epoch: int, role_class: str, slug: str) -> str:
    """The controller's artifact ID. Agents never construct one; their slug is an input only."""
    return f"zyal.e{epoch:02d}.{role_class}.{slug}"


def role_id(epoch: int, role_class: str, index: int) -> str:
    return f"zyal.e{epoch:02d}.{role_class}-{index:02d}"


# --- Roster -------------------------------------------------------------------------------------

def role_specs(epoch: int) -> list[dict]:
    """The full deterministic roster for one epoch. No randomness anywhere in here.

    Each spec carries its own output schema so that `validate_payload` reads the contract from the
    role the controller actually dispatched, rather than re-deriving it and risking a drift between
    what was asked for and what is enforced.
    """
    specs: list[dict] = []
    for role_class, count in ROLE_COUNTS.items():
        schema = ROLE_SCHEMAS[role_class]
        for i in range(count):
            specs.append({
                "id": role_id(epoch, role_class, i),
                "epoch": epoch,
                "role_class": role_class,
                "index": i,
                "schema_version": SCHEMA_VERSION,
                "objective": ROLE_OBJECTIVES[role_class],
                "engines": list(ENGINES),
                "mode": "analysis" if role_class in ANALYSIS_ROLES else "feature",
                "output_key": schema.output_key,
                "n_items": schema.n_items,
                "item_keys": list(schema.item_keys),
                "max_attempts": MAX_ATTEMPTS_PER_ROLE,
            })
    assert len(specs) == ROLES_PER_EPOCH
    return specs


# --- Genome sampling ------------------------------------------------------------------------------

class GenomeSample(NamedTuple):
    """Genomes plus the provenance needed to reproduce the draw.

    Returned as a tuple so the seed can be recorded in the prompt artifact -- a preview that cannot
    be reproduced makes the whole prompt unauditable.
    """
    genomes: list[str]
    seed: int
    seed_key: str
    n_available: int


def _seed_for(epoch: int, role: str) -> tuple[int, str]:
    key = f"{CAMPAIGN_SEED}:{epoch}:{role}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:16], 16), key


def sample_genomes(rows, role_id: str, epoch: int, n: int = GENOME_PREVIEW_N) -> GenomeSample:
    """Draw `n` distinct SELECT-split genome strings, reproducibly and distinctly per role.

    Sampling is over the *deduplicated, sorted* set of genome texts: a popular genome appearing on
    two hundred datasets would otherwise dominate every preview, and sorting makes the draw
    independent of row order in the database.
    """
    split = getattr(rows, "split", None)
    genome = getattr(rows, "genome", None)
    if split is None or genome is None:
        raise TypeError("rows must expose .split and .genome arrays")

    pool = sorted({str(g) for g, s in zip(genome, split) if str(s) == "select"})
    seed, key = _seed_for(epoch, role_id)
    rng = random.Random(seed)
    take = min(int(n), len(pool))
    drawn = rng.sample(pool, take) if take else []
    return GenomeSample(genomes=drawn, seed=seed, seed_key=key, n_available=len(pool))


# --- Domain context -------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def grammar_snapshot() -> dict:
    """Compact grammar summary, read from `jope.grammars` (the source of truth shared with Rust).

    Computed, never hardcoded: if the grammar gains an op, the prompts say so without an edit here.
    """
    from jope.grammars import FAMILY_LIST_V3, grammar_for

    engines = {}
    for name in ENGINES:
        g = grammar_for(name)
        engines[name] = {
            "n_transforms": len(g.transforms),
            "n_heads": len(g.heads),
            "max_transforms": int(g.max_transforms),
        }
    expr = grammar_for("featmath").expr
    return {
        "engines": engines,
        "families": list(FAMILY_LIST_V3),
        "expr": {
            "funcs1": list(expr.funcs1),
            "funcs2": list(expr.funcs2),
            "binops": list(expr.binops),
            "max_col": int(expr.max_col),
            "max_depth": int(expr.max_depth),
            "max_nodes": int(expr.max_nodes),
            "vars_common": list(expr.vars_common),
        },
        "notes": (
            "A genome is (pipe X <transform>... <head>). Stages carry params; some params hold "
            "arithmetic expression strings. Symbolic (sym ...) genomes are not pipelines and "
            "parse to ok=false -- any proposal must say what it does with them."
        ),
    }


def family_card_contract() -> dict:
    """The anti-laziness contract. Every clause here exists because a role failed it somewhere."""
    return {
        "family_slug": (
            "invent a unique lowercase kebab-case name for the concrete mechanism; generic names "
            "such as feature-family-01 are forbidden"
        ),
        "mechanism_hypothesis": (
            "at least 80 characters; name a concrete GENOME-STRUCTURAL observable and the "
            "mechanism linking it to within-dataset rank"
        ),
        "novelty_argument": (
            "at least 80 characters; identify what makes this distinct from the 1,331 columns "
            "already in the baseline registry"
        ),
        "falsifier": (
            "at least 40 characters; state a structure-preserving intervention that would "
            "disprove the mechanism"
        ),
        "distinctness": (
            "every card you return must propose a DIFFERENT mechanism; near-duplicates of your "
            "own other cards are rejected as a set"
        ),
        "no_placeholders": (
            "no placeholder, template, or to-be-decided content in any field; a field you cannot "
            "fill concretely means the proposal is not ready"
        ),
        "structural_only": (
            "observables must be computable from genome structure alone -- never from scores, "
            "labels, dataset identity, runtime, or rank"
        ),
    }


# --- Envelope --------------------------------------------------------------------------------------

def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _nbytes(s: str) -> int:
    return len(s.encode("utf-8"))


def _assert_feedback_categorical(node: Any, path: str = "feedback") -> None:
    """Recursive leakage gate. A nested feedback map is still feedback."""
    if isinstance(node, dict):
        if node and all(isinstance(v, (dict, list)) for v in node.values()):
            for k, v in node.items():
                _assert_feedback_categorical(v, f"{path}.{k}")
        else:
            assert_categorical_only(node)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _assert_feedback_categorical(v, f"{path}[{i}]")
    elif node is not None:
        raise LeakageError(f"{path} must be a dict of categorical strings, got {type(node).__name__}")


_NUMERIC_CONTEXT_ALLOWLIST = frozenset({
    # Structural counts an agent legitimately needs and which reveal nothing about the target.
    "epoch", "n_epochs", "concept_count", "preview_count", "max_concepts", "min_concepts",
    "byte_cap", "seed", "attempt",
})


def _scan_context(node, path: str) -> None:
    """Recursively reject numbers and forbidden identifiers ANYWHERE in the context.

    The previous version checked a three-key allowlist for feedback and only one nesting level of
    keys for forbidden names. An adversarial pass drove the champion's real metrics straight into a
    prompt via `{"scores": {"champion_spearman": 0.1284}}`, and hid `dataset_id` at depth three and
    inside lists. Both got through.

    A numeric leak is not a lint failure here -- it is the failure mode the whole campaign design
    exists to prevent, so the gate has to be exhaustive rather than a list of the places we thought
    of. Structural counts an agent genuinely needs are named explicitly in
    _NUMERIC_CONTEXT_ALLOWLIST; everything else numeric is refused.
    """
    if isinstance(node, dict):
        assert_no_forbidden_inputs(node.keys())
        for key, value in node.items():
            _scan_context(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _scan_context(value, f"{path}[{index}]")
    elif isinstance(node, bool) or isinstance(node, (int, float)):
        leaf = path.rsplit(".", 1)[-1].split("[")[0]
        if leaf not in _NUMERIC_CONTEXT_ALLOWLIST:
            raise LeakageError(
                f"context{'.' + path if path else ''} = {node!r} is numeric; agents may not "
                f"receive numbers. Bucket it via scoring.categorical_feedback, or add the key to "
                f"_NUMERIC_CONTEXT_ALLOWLIST if it is a structural count."
            )
    elif isinstance(node, str):
        assert_no_forbidden_inputs([node])


def _guard_context(context: dict) -> None:
    """Gate everything leaving for a prompt: no numbers, no forbidden identifiers, at any depth."""
    for key in ("feedback", "prior_feedback", "epoch_feedback"):
        if key in context and context[key] is not None:
            _assert_feedback_categorical(context[key], key)
    _scan_context(context, "")


def _contract_for(role: dict) -> dict:
    """Only ship the clauses that bind this role's actual fields -- bytes are scarce."""
    keys = set(role.get("item_keys") or ROLE_SCHEMAS[role["role_class"]].item_keys)
    full = family_card_contract()
    contract = {k: v for k, v in full.items() if k in keys}
    contract["distinctness"] = full["distinctness"]
    contract["no_placeholders"] = full["no_placeholders"]
    contract["structural_only"] = full["structural_only"]
    if "mechanism_hypothesis" in keys and "mechanism_hypothesis" not in contract:
        contract["mechanism_hypothesis"] = full["mechanism_hypothesis"]
    for slug_key in (k for k in keys if k.endswith("_slug") and k not in REFERENCE_SLUG_KEYS):
        contract[slug_key] = full["family_slug"]
    return contract


def _schema_block(role: dict) -> dict:
    schema = ROLE_SCHEMAS[role["role_class"]]
    item_keys = list(role.get("item_keys") or schema.item_keys)
    block = {
        "output_key": role.get("output_key", schema.output_key),
        "n_items_exact": int(role.get("n_items", schema.n_items)),
        "item_keys_exact": sorted(item_keys),
        "min_chars": {k: v for k, v in CONTRACT_MIN_CHARS.items() if k in item_keys},
    }
    if any(k in LIST_VALUED_KEYS for k in item_keys):
        block["list_valued"] = sorted(k for k in item_keys if k in LIST_VALUED_KEYS)
    if role["role_class"] == "blind_judge":
        block["verdict_enum"] = list(JUDGE_VERDICTS)
    if role["role_class"] == "plugin_writer":
        block["source_max_bytes"] = MAX_PLUGIN_BYTES
        block["source_entrypoint"] = "compute_family(structure, config)"
    return block


def _output_contract(role: dict) -> dict:
    schema = ROLE_SCHEMAS[role["role_class"]]
    return {
        "top_level_keys_exact": [role.get("output_key", schema.output_key)],
        "format": "strict_json_object",
        "max_attempts": int(role.get("max_attempts", MAX_ATTEMPTS_PER_ROLE)),
        "on_invalid": (
            "the controller returns the schema errors verbatim and you get one corrected attempt; "
            "a second invalid payload discards the role"
        ),
        "ids": (
            "propose slugs only; the controller mints every final ID, filename and parameter and "
            "your proposed IDs are discarded"
        ),
    }


def _shrink(payload: dict, *, genome_key: str = "genomes") -> str:
    """Serialise under the byte cap.

    Primary lever is the genome preview, halved and retried as specified. Once the preview is at
    its useful floor the prior-slug window is halved instead (keeping the MOST RECENT slugs, which
    are the ones that actually constrain novelty), and only then is the preview taken to zero.
    """
    context = payload.get("context")
    while True:
        text = _dumps(payload)
        if _nbytes(text) <= PROMPT_BYTE_CAP:
            return text
        if not isinstance(context, dict):
            raise ValueError(
                f"prompt envelope is {_nbytes(text)} bytes with nothing left to shrink "
                f"(cap {PROMPT_BYTE_CAP})"
            )
        genomes = context.get(genome_key) or []
        slugs = context.get("prior_slugs") or []
        # Every branch must strictly REDUCE length or the loop cannot terminate. The previous
        # version used slugs[len(slugs)//2:], which for a single-element list is slugs[0:] -- a
        # no-op fixed point that hung build_prompt forever instead of raising. A hang on the
        # prompt-emission path stalls a whole wave with no error to diagnose.
        if len(genomes) > _SHRINK_GENOME_FLOOR:
            context[genome_key] = genomes[: len(genomes) // 2]
        elif len(slugs) > 1:
            context["prior_slugs"] = slugs[: len(slugs) // 2]
        elif slugs:
            context["prior_slugs"] = []
        elif genomes:
            context[genome_key] = genomes[: len(genomes) // 2]
        else:
            raise ValueError(
                f"prompt envelope is {_nbytes(text)} bytes with nothing left to shrink "
                f"(cap {PROMPT_BYTE_CAP})"
            )


def build_prompt(role: dict, context: dict) -> str:
    """The minified strict-JSON envelope handed to one role.

    Raises `LeakageError` before serialising if any feedback in `context` is not categorical.
    """
    context = dict(context or {})
    _guard_context(context)

    if "prior_slugs" in context and context["prior_slugs"]:
        context["prior_slugs"] = list(context["prior_slugs"])[-PRIOR_SLUG_WINDOW:]
    context.setdefault("grammar", grammar_snapshot())
    context.setdefault("epoch", role.get("epoch"))

    envelope = {
        "role": {
            "id": role["id"],
            "role_class": role["role_class"],
            "epoch": role.get("epoch"),
            "objective": role.get("objective") or ROLE_OBJECTIVES[role["role_class"]],
            "mode": role.get("mode", "feature"),
            "schema_version": role.get("schema_version", SCHEMA_VERSION),
        },
        "contract": _contract_for(role),
        "rules": list(RULES),
        "schema": _schema_block(role),
        "output_contract": _output_contract(role),
        "context": context,
    }
    return _shrink(envelope)


def build_schema_retry_prompt(role: dict, invalid_payload: Any, errors: list[str]) -> str:
    """The one corrective attempt. Shows what was wrong, not how to be clever."""
    shown = [str(e) for e in (errors or [])][:30]
    artifact = _dumps(invalid_payload) if not isinstance(invalid_payload, str) else invalid_payload

    while True:
        envelope = {
            "role": {
                "id": role["id"],
                "role_class": role["role_class"],
                "epoch": role.get("epoch"),
                "attempt": "corrective",
                "schema_version": role.get("schema_version", SCHEMA_VERSION),
            },
            "contract": _contract_for(role),
            "rules": list(RULES),
            "schema": _schema_block(role),
            "output_contract": _output_contract(role),
            "context": {
                "instruction": (
                    "Your previous payload failed schema validation. Return one corrected strict "
                    "JSON object satisfying every listed error. Do not explain the errors."
                ),
                "errors": shown,
                "n_errors_total": len(errors or []),
                "invalid_payload_excerpt": artifact,
                "truncated": len(artifact) < len(_dumps(invalid_payload)
                                                 if not isinstance(invalid_payload, str)
                                                 else invalid_payload),
            },
        }
        text = _dumps(envelope)
        if _nbytes(text) <= PROMPT_BYTE_CAP:
            return text
        if artifact:
            artifact = artifact[: len(artifact) // 2]
        elif len(shown) > 1:
            shown = shown[: len(shown) // 2]
        elif shown:
            shown = []
        else:
            raise ValueError(
                f"retry envelope is {_nbytes(text)} bytes with nothing left to shrink "
                f"(cap {PROMPT_BYTE_CAP})"
            )


# --- Validation -------------------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def _check_slug(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}: slug must be a non-empty string")
        return
    slug = value.strip()
    if not _SLUG_RE.match(slug):
        errors.append(f"{where}: {slug!r} is not lowercase kebab-case (a-z, 0-9, single hyphens)")
        return
    if not (_SLUG_MIN <= len(slug) <= _SLUG_MAX):
        errors.append(f"{where}: {slug!r} must be {_SLUG_MIN}-{_SLUG_MAX} characters")
    if _PLACEHOLDER_RE.match(slug):
        errors.append(
            f"{where}: {slug!r} is a generic placeholder name; name the concrete mechanism"
        )
    if any(w in slug for w in _PLACEHOLDER_WORDS):
        errors.append(f"{where}: {slug!r} contains placeholder content")


def validate_payload(role: dict, payload: Any) -> list[str]:
    """Exact key sets, exact counts, and the anti-laziness contract.

    Returns human-readable errors; an empty list means the payload is compilable as-is.
    """
    errors: list[str] = []
    role_class = role.get("role_class")
    if role_class not in ROLE_SCHEMAS:
        return [f"unknown role_class {role_class!r}"]
    schema = ROLE_SCHEMAS[role_class]
    output_key = role.get("output_key", schema.output_key)
    n_items = int(role.get("n_items", schema.n_items))
    item_keys = set(role.get("item_keys") or schema.item_keys)

    if not isinstance(payload, dict):
        return [f"payload must be a JSON object, got {type(payload).__name__}"]

    got = set(payload)
    want = {output_key}
    for extra in sorted(got - want):
        errors.append(f"unexpected top-level key {extra!r}; the only allowed key is {output_key!r}")
    for missing in sorted(want - got):
        errors.append(f"missing top-level key {missing!r}")
    if errors:
        return errors

    items = payload[output_key]
    if not isinstance(items, list):
        errors.append(f"{output_key!r} must be a list, got {type(items).__name__}")
        return errors
    if len(items) != n_items:
        errors.append(f"{output_key!r} must contain exactly {n_items} items, got {len(items)}")

    seen_slugs: dict[str, str] = {}
    seen_mechanisms: dict[str, str] = {}

    for i, item in enumerate(items):
        where = f"{output_key}[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{where}: must be a JSON object, got {type(item).__name__}")
            continue
        keys = set(item)
        for extra in sorted(keys - item_keys):
            errors.append(f"{where}: unexpected key {extra!r}")
        for missing in sorted(item_keys - keys):
            errors.append(f"{where}: missing key {missing!r}")

        for key in sorted(keys & item_keys):
            value = item[key]
            if key in LIST_VALUED_KEYS:
                if not isinstance(value, list) or not value:
                    errors.append(f"{where}.{key}: must be a non-empty list")
                elif not all(isinstance(v, str) and v.strip() for v in value):
                    errors.append(f"{where}.{key}: every entry must be a non-empty string")
                continue
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}.{key}: must be a non-empty string")
                continue
            if key.endswith("_slug"):
                if key in REFERENCE_SLUG_KEYS:
                    continue
                _check_slug(value, f"{where}.{key}", errors)
                norm = value.strip().casefold()
                if norm in seen_slugs:
                    errors.append(
                        f"{where}.{key}: duplicate slug {value.strip()!r}, already used at "
                        f"{seen_slugs[norm]}"
                    )
                else:
                    seen_slugs[norm] = where
                continue
            minimum = CONTRACT_MIN_CHARS.get(key)
            if minimum and len(value.strip()) < minimum:
                errors.append(
                    f"{where}.{key}: {len(value.strip())} characters, contract requires at least "
                    f"{minimum}"
                )
            if any(w in value.casefold() for w in _PLACEHOLDER_WORDS):
                errors.append(f"{where}.{key}: contains placeholder content")

        mechanism = item.get("mechanism_hypothesis")
        if isinstance(mechanism, str) and mechanism.strip():
            norm = _norm(mechanism)
            if norm in seen_mechanisms:
                errors.append(
                    f"{where}.mechanism_hypothesis: duplicates {seen_mechanisms[norm]}; every "
                    "item from one role must propose a DIFFERENT mechanism"
                )
            else:
                seen_mechanisms[norm] = where

        if role_class == "blind_judge":
            verdict = item.get("verdict")
            if isinstance(verdict, str) and verdict.strip().casefold() not in JUDGE_VERDICTS:
                errors.append(
                    f"{where}.verdict: {verdict!r} is not one of {list(JUDGE_VERDICTS)}"
                )

        if role_class == "plugin_writer":
            source = item.get("source")
            if isinstance(source, str) and source.strip():
                if len(source.encode("utf-8")) > MAX_PLUGIN_BYTES:
                    errors.append(
                        f"{where}.source: exceeds the {MAX_PLUGIN_BYTES} byte plugin limit"
                    )
                if "def compute_family(" not in source:
                    errors.append(f"{where}.source: must define compute_family(structure, config)")

    return errors
