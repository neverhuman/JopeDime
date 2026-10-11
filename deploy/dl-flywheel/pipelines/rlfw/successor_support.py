"""Versioned SCRATCH training-role program availability, separate from labels.

The builder never admits validation, heldout or unknown-role measured programs.
Availability is global within the frozen support; it grants no context reward
or optimizer eligibility. Qualified use awaits Jepson's HWMKey extension ruling.
"""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass

V1_TRIE_SHA256 = "116154767f2df56fb065c9b6bc3ce07df38c3672843c0eb4b944d84e8e8473bd"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ContextOutcomeEligibility:
    context_id: str
    program_sha256: str
    outcome: str
    q: float | None
    baseline_verified: bool
    role: str

    @property
    def optimizer_eligible(self):
        return (
            self.role == "train"
            and self.baseline_verified
            and self.outcome == "completed"
            and self.q is not None
            and math.isfinite(self.q)
        )


class SuccessorBuilder:
    """Consume canonical measured rows after exact native shard/role checking.

    validate_program must enforce token roundtrip, vocabulary, grammar and model
    length. These source checks are independent of optional outcome labels.
    """

    def __init__(
        self,
        catalog,
        *,
        v1_trie_sha256,
        vocabulary_sha256,
        max_seq,
        role_map,
        validate_program,
    ):
        actual = digest(
            {
                "schema": "jope.flywheel-full-catalog-trie/v1",
                "vocabulary_sha256": vocabulary_sha256,
                "max_seq": max_seq,
                "programs": catalog,
            }
        )
        if actual != v1_trie_sha256 or actual != V1_TRIE_SHA256 or len(catalog) != 6640:
            raise ValueError("exact v1 6640-program support required")
        self.role_map = dict(role_map)
        self.validate_program = validate_program
        self.programs = {}
        self.sources = {}
        self.counts = Counter()
        self.catalog_shas = set()
        self.frozen = False
        for program in catalog:
            ph = self._insert(program)
            if ph in self.catalog_shas:
                raise ValueError("duplicate v1 catalog identity")
            self.catalog_shas.add(ph)

    def _insert(self, program):
        text, ph = program["dsl"], program["program_sha256"]
        if hashlib.sha256(text.encode()).hexdigest() != ph:
            raise ValueError("canonical program SHA mismatch")
        self.validate_program(program)
        if ph in self.programs and self.programs[ph]["dsl"] != text:
            raise ValueError("program identity collision")
        self.programs[ph] = {"dsl": text, "program_sha256": ph}
        return ph

    def add_measured(self, context_id, program, *, shard_sha256, outcome):
        if self.frozen:
            raise ValueError("support generation is frozen; build a new version")
        if self.role_map.get(context_id) != "train":
            raise ValueError("measured support extension requires training ROLE")
        if (
            len(shard_sha256) != 64
            or any(c not in "0123456789abcdef" for c in shard_sha256)
            or not isinstance(outcome, str)
            or not outcome
        ):
            raise ValueError("bound measured shard and explicit outcome required")
        ph = self._insert(program)
        self.sources.setdefault(
            ph, {"context_id": context_id, "shard_sha256": shard_sha256}
        )
        self.counts["measured_rows"] += 1
        self.counts["outcome:" + outcome] += 1
        return ph

    def freeze(self, *, vocabulary_sha256, max_seq, source_pins, operator_names):
        # Reused availability is a cache, not evidence of current TRAIN custody.
        # Only add_measured records sources during the current native scan.
        missing = (self.programs.keys() - self.catalog_shas) - self.sources.keys()
        if missing:
            raise ValueError(
                f"{len(missing)} noncatalog programs lack current TRAIN provenance"
            )
        programs = [dict(self.programs[ph]) for ph in sorted(self.programs)]
        added = sorted(set(self.programs) - self.catalog_shas)
        operators = Counter()
        added_operators = Counter()
        for program in programs:
            names = set(operator_names(program["dsl"]))
            operators.update(names)
            if program["program_sha256"] in self.catalog_shas:
                continue
            added_operators.update(names)
        manifest = {
            "schema": "jope.rlfw-successor-support/v1",
            "label": "SCRATCH NON-QUALIFYING",
            "qualified_use": "BLOCKED_JEPSON",
            "qualified_use_requirement": "HWMKey/trie-extension ruling",
            "parent_trie_sha256": V1_TRIE_SHA256,
            "catalog_size": 6640,
            "size": len(programs),
            "added_size": len(added),
            "added_program_sha256": added,
            "availability_sha256": digest(programs),
            "vocabulary_sha256": vocabulary_sha256,
            "max_seq": max_seq,
            "source_pins": json.loads(json.dumps(source_pins, allow_nan=False)),
            "role_map_sha256": digest(self.role_map),
            "counts": dict(sorted(self.counts.items())),
            "operator_program_coverage": dict(sorted(operators.items())),
            "added_operator_program_coverage": dict(sorted(added_operators.items())),
            "reference": {
                "parameter_sha256": "ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb",
                "mask": "this successor support",
                "cross_support_forward_kl": "REFUSED",
            },
            "availability_grants_context_outcome_eligibility": False,
        }
        manifest["support_sha256"] = digest(manifest)
        self.frozen = True
        return programs, manifest
