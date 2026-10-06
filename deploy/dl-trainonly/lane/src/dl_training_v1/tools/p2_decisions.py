#!/usr/bin/env python3
"""DL-D5 owner-answer expansion for the P2 family review (PLAN §5 P2/P3, §7.2): tools/p2_decisions.py, version 2.

Version 2 is pinned to P2 generation 2 (RUN/families/g2: review.csv aad99b1a..., registry.pre-review.json e21690dd...,
alias-sources.json f818efaf..., built by families 7dcd65ff, family_keys 9b688e9b, family_pin 05187c72 and
family_screen 76365b12; shape builtin:aad99b1a). It applies the Track D review findings D-MAP-1..7 and the tool-side parts
of P3I-1..6.

The tool turns the owner's DL-D5 confirmation into the review decision file that `dltrain.families build` reads
(jope.dl-family-review-decisions/v1). It decides nothing itself:
- every review.csv row gets its own decision from one fixed, versioned mapping (dl-d5-map/v2, MAPPING_TABLE);
- union_subset sets are copied verbatim from the row's recommendation_basis;
- no class decisions are used.
The bytes are common.canonical(doc): sorted keys, compact, no clock field. The same answers and inputs therefore always
give the same bytes, which is what `check` re-derives.

Owner text (D-MAP-1, D-MAP-2, D-MAP-3). The owner's message, or the one cited line of OWNER_DIRECTIVES.md, must be exactly
one line in this grammar (dl-d5-confirm/v1):
    DL-D5 confirmed for review.csv <sha12>: Q1=x, Q2=x, Q3=x, Q4=x, Q5=x; split-registry excluded seals nothing.
Letter case and spacing around the separators are free, the final period is optional, and a directive line may begin
with one '- ' or '* '. Anything else fails: a second line, a question, a negation, a quoted suggestion, another decision
label. <sha12> must be the first 12 hex digits of the bound review.csv sha256, so the confirmation binds exactly the rows
the owner was shown. Q1..Q4 must equal --answers. Q5 must match --rights in both directions (a = rights_note unknown,
b = undecided; c = prohibited is a per-row owner call and is refused). A message must not predate --question-posted-utc.

expand modes:
  real       The owner's confirmation. It writes, create-only:
             - the owner message RUN/receipts/P2/dl-d5-owner-message-<sha16>.txt;
             - <bound families dir>/review-decisions.json (mode 0444);
             - the receipt RUN/receipts/P2/decisions-expand-<utc_us>.json.
             Before any write it runs families.check_provenance(real=True) (D-MAP-5). An owner_directive is accepted only
             from the owner's own OWNER_DIRECTIVES.md (OWNER_DIRECTIVES_PATH).
             It then stores the owner message and simulates the final build with real=True and receipts_dir
             RUN/receipts/P2, exactly as `families build --out RUN/families/g2` will run it. The receipt therefore
             predicts the sha256 of the final registry.json.
  --draft    The relayed DL-D5 paste (sha256 b785a123..., answered against g1, UNVERIFIED). It takes Q1..Q4 = a only and
             --rights undecided, because the relay names no rights_note answer, and it leaves the split-registry ruling
             null and unconfirmed. A DRAFT adds no choice the relay did not make (D-MAP-4).
  --what-if  Any answers, labelled as an agent's hypothesis, for example the recommended answers with Q5=a.
DRAFT and what-if files are structurally unpromotable (D-MAP-6). Each of these is refused by families.validate_decisions
for every generation, real or staging, and by holdout.review_gate:
  - the schema: jope.dl-p2-decisions-draft-unconfirmed/v1 or jope.dl-p2-decisions-what-if/v1;
  - the provenance kind: relay_draft_unconfirmed or agent_what_if_not_a_decision;
  - the reviewer, which is not 'owner'.
They are written only to the staging dir, with a DRAFT-UNCONFIRMED- or WHATIF-NOT-A-DECISION- prefix.
Their outcome is measured on an in-memory rehearsal variant that is never written: the decision schema, reviewer
'owner', the split ruling assumed true, and a DRAFT-marked provenance that families refuses for every real generation.
families' own SIMULATION_SCHEMA path cannot be used: it never grants 'reviewed', so it cannot measure N or k.

Subcommands (run from SRC inside the lane slice with the offline prefix, AWS_CONFIG_FILE=/dev/null):
  shape   --review-csv P --review-csv-sha256 H [--raw] [--out STAGE_RECEIPTS/dl-d5-shape-<sha16>.json]
          [--expect-shape builtin:aad99b1a | F --expect-shape-sha256 H]
      The shape (jope.dl-d5-review-shape/v2) is derived from review.csv and the frozen families code:
      - review_csv_sha256 and the row count;
      - the counts per (kind, recommendation) and the link-row summary;
      - the union_subset set counts and sizes, and the breaker ids;
      - rows_digest, a sha256 over every (review_group_id, kind, recommendation, family_ids, sets, options) tuple;
      - families.DECISION_OPTIONS of the kinds present;
      - the four family code shas.
      --out writes it create-only.
  expand  --review-csv P --review-csv-sha256 H --pre-review P --pre-review-sha256 H --alias-sources P
          --alias-sources-sha256 H --answers Q1=x,Q2=x,Q3=x,Q4=x --rights unknown|undecided
          --expect-shape builtin:aad99b1a | --expect-shape SHAPE.json --expect-shape-sha256 H
          [--simulate --expect-outcome families=797,N_pool=652,k=33]
      REAL (all required):
          - --out <bound families dir>/review-decisions.json; the bound dir must also hold the alias-sources.json;
          - --receipt RUN/receipts/P2/decisions-expand-<utc_us>.json;
          - --split-registry-confirmed --simulate --expect-outcome ...;
          - one provenance:
            --provenance-kind owner_verbatim_message|owner_session_selections --session S --message-utc U
                --message-file F --message-sha256 H --question-posted-utc P
            --provenance-kind owner_directive --directive-file OWNER_DIRECTIVES_PATH --directive-version V
                --directive-lines N --directive-sha256 H
      DRAFT:   --draft --relay-sha256 b785a123... --answers Q1=a,Q2=a,Q3=a,Q4=a --rights undecided
      WHAT-IF: --what-if
          Outputs: STAGE/<prefix>review-decisions-<sha16>.json and STAGE_RECEIPTS/<prefix>p2-decisions-expand-<utc_us>.json.
      --simulate runs families.run_build(alias-sources, /proc/self/fd/<memfd>) in memory. It needs verdict PASS, phase
      final, the rebuilt registry.pre-review.json and review.csv byte-equal to the bound files, and the counts of
      --expect-outcome.
  check   --decisions D --decisions-sha256 H + the three bound inputs [--real]
          [--registry R --registry-sha256 H [--dryrun-receipt X --dryrun-receipt-sha256 H]]
      Re-expands the file from its recorded answers and provenance and needs byte equality. It then validates against
      the in-memory pre-review model, which must rebuild the bound bytes.
      - A real file is validated with families.validate_decisions(real=--real, receipts_dir=RUN/receipts/P2).
      - A DRAFT or what-if file is validated through its rehearsal variant, after families and holdout are shown to
        refuse the file itself.
      --real refuses unconfirmed files, the relay sha and a decision file outside the bound families dir.
      --registry runs holdout.load_registry and review_gate, plus the tool-side P3I-1 gate: the registry's
      decision_deltas.provenance must be draft false, simulation false, and a pinned owner record of this decision
      file's message or directive.
      --dryrun-receipt (P3I-4) needs a PASS rehearsal receipt whose holdout source tree and family code shas equal
      the current ones.
  dryrun  --decisions STAGE/<prefix>review-decisions-<sha16>.json --decisions-sha256 H + the three bound inputs
          --owner-decisions O --owner-decisions-sha256 H --epoch E --epoch-sha256 H --master M --master-sha256 H
          --registry-db DB --expect-outcome ... [--attempt N] [--quick-rehash]
      The P2->P3 readiness rehearsal of a DRAFT or what-if file. It runs:
      - families build, then verify (the API, with receipts_dir = RUN/receipts/P2 as the real S3 verify scans it; full
        rehash by default), then publish;
      - holdout commit -> freeze -> verify --assignment-poison -> closure -> publish, through holdout's non-CLI DRAFT
        dry-run mode (_main(..., draft_dry_run=True): fake clients only, families.check_provenance(real=False), never
        RUN/holdout, every receipt stamped draft_dry_run).
      Everything goes through the real entry points with one shared fake bucket. tests_p3.fixtures_p3 applies the real
      s3io.Guard; the real client factories are patched to raise, and the write audit hook is armed.
      The rehearsal bytes reach families and holdout only through a memfd, never as a file.
      All outputs go to one private tree, STAGE/<prefix>rehearsal-<sha16>-a<N>/. The tree is inventoried (path, sha256,
      bytes) into the create-only receipt STAGE_RECEIPTS/<prefix>p3-dryrun-<sha16>-a<N>.json and then removed (D-MAP-6),
      so the rehearsal leaves no P2 packet, registry or holdout behind.

Exit codes: 0 PASS, 2 STOP (refused; one JSON line on stderr names the refusal), 3 ERROR. A refusal writes nothing.
Two exceptions:
- a real expand that fails its simulation (R11) keeps the owner message it stored (an owner record, not a decision);
- a dryrun step that fails after the pipeline started also exits 2, with its receipt written.
Terminal output: counts, review_group_ids, family ids, paths and sha256 values only; never a trigger_value, members or
evidence cell.

Imports: dltrain.common, dltrain.families and dltrain.family_pin (CpuGuard) at module level. dltrain.holdout is imported
only inside check, dryrun and the check helpers (holdout_refuses, dryrun_code_gate). tests_p3.fixtures_p3 (which itself
brings boto3 and dltrain.s3io) is imported only inside dryrun. The tool never imports s3io, boto3, jope, gp_worker, local_gateway, reuse_cache or intake_registry itself.

R7 code binding: registry.pre-review.json records the sha256 of families, family_keys, family_pin and family_screen
("code"). expand, check and dryrun refuse when current_code_shas() differs, because the final build would then rebuild
different pre-review bytes and STOP on the decision binds. An edited module therefore needs a rebuilt pre-review set, a
new shape and a new owner confirmation.

Refusal codes:
  R1   --answers
  R2   Q3=b (package level)
  R3   split-registry confirmation
  R4   rights_note
  R5   row kinds and options
  R6   union_subset sets
  R7   pins and bindings
  R8   shape
  R9   provenance and owner text
  R10  modes and paths
  R11  simulation
  R12  the unpromotable self-check
  VALIDATE, EXPANSION, GATE, DRYRUN, USAGE
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from dltrain import common
from dltrain import families
from dltrain import family_pin

TOOL = "tools/p2_decisions.py"
MAPPING = "dl-d5-map/v2"
EXPAND_SCHEMA = "jope.dl-p2-decisions-expand/v2"
DRYRUN_SCHEMA = "jope.dl-p2-decisions-dryrun/v2"
SHAPE_SCHEMA = "jope.dl-d5-review-shape/v2"
CONFIRM_GRAMMAR = "dl-d5-confirm/v1"
DRAFT_SCHEMA = "jope.dl-p2-decisions-draft-unconfirmed/v1"     # never a decision file: families and holdout refuse it
WHATIF_SCHEMA = "jope.dl-p2-decisions-what-if/v1"              # never a decision file: families and holdout refuse it
UNCONFIRMED_SCHEMA = {"draft": DRAFT_SCHEMA, "what_if": WHATIF_SCHEMA}
UNCONFIRMED_REVIEWER = "none: not an owner review (DRAFT-UNCONFIRMED or WHATIF-NOT-A-DECISION)"
DRAFT_KIND = "relay_draft_unconfirmed"                        # not a families.PROVENANCE_KINDS kind
WHATIF_KIND = "agent_what_if_not_a_decision"                  # not a families.PROVENANCE_KINDS kind
EXIT = {"PASS": 0, "STOP": 2, "ERROR": 3}

RUN = families.RUN
FAMILIES_ROOT = families.FAMILIES_DIR
RECEIPTS_ROOT = families.RECEIPTS_DIR
STAGE = RUN / "tmp" / "dl-d5-g2"
STAGE_RECEIPTS = RECEIPTS_ROOT / "staging"
PREFIX = {"draft": "DRAFT-UNCONFIRMED-", "what_if": "WHATIF-NOT-A-DECISION-"}
DRAFT_PREFIX = PREFIX["draft"]
OUT_NAME = families.DECISIONS_NAME
MESSAGE_NAME = families.OWNER_MESSAGE_NAME
# PLAN §7.2: an owner_directive cites the owner's versioned directive file; the tool accepts no other (D-MAP-5)
OWNER_DIRECTIVES_PATH = Path("/home/ubuntu/JopeDime/.agent/plans/OWNER_DIRECTIVES.md")

RELAY_SHA256 = "b785a123a4f17bd9bc52502fe70f448db981417ccc84b73affb122e73899d6f1"
RELAY_ANSWERS = {"Q1": "a", "Q2": "a", "Q3": "a", "Q4": "a"}
DRAFT_RIGHTS = "undecided"           # the relay names no rights_note answer: those rows keep their default (D-MAP-4)
DRAFT_PROVENANCE = {
    "kind": DRAFT_KIND,
    "session": "DRAFT-UNCONFIRMED-RELAY-b785a123",
    "relay_sha256": RELAY_SHA256,
    "relay_utc": "2026-10-05T10:07Z relay arrival (not an owner timestamp)",
    "relay_answered": "g1 review.csv c7377cf8 (superseded); letters for Q1-Q4 only",
    "draft": True,
    "confirmed_by_owner": False,
    "unconfirmed": ["Q1-Q4", "Q5 rights_note (not in the relay: the rights_note rows stay undecided)",
                    "split_registry_excluded_seals_nothing"],
    "note": "DRAFT: relayed paste attributed to an owner loop; PLAN §7.2: relayed text never counts",
}
WHATIF_PROVENANCE = {
    "kind": WHATIF_KIND,
    "session": "WHATIF-NOT-A-DECISION",
    "draft": True,
    "simulation": True,
    "confirmed_by_owner": False,
    "unconfirmed": ["Q1-Q5", "split_registry_excluded_seals_nothing"],
    "note": "SIMULATION: an agent's what-if expansion, never an owner decision (PLAN §7.2)",
}
UNCONFIRMED_PROVENANCE = {"draft": DRAFT_PROVENANCE, "what_if": WHATIF_PROVENANCE}
NOTICE = {
    "draft": "DRAFT-UNCONFIRMED: expanded from the relayed DL-D5 paste (sha256 b785a123..., answered against g1), awaiting "
             "Jepson's own typed confirmation under PLAN §7.2. Not a decision file: families and holdout refuse its "
             "schema, kind and reviewer.",
    "what_if": "WHATIF-NOT-A-DECISION: an agent's hypothetical expansion for the owner question set. Not a decision "
               "file: families and holdout refuse its schema, kind and reviewer.",
}
RELATED_CLASS_POLICY = "lineage-B related-class families stay unresolved: train, never held out"

# ---- the mapping (dl-d5-map/v2): row kind -> decision
LINK_TRIGGER_KINDS = ("rdatasets_pkg", "publisher", "socrata_domain", "github_owner", "legacy_source_family",
                      "tidytuesday_year", "name_candidate", "classic_candidate", "nds_value_weak")
LINK_KINDS = LINK_TRIGGER_KINDS + ("multi_member_family",)
SEED_KIND, SEALED_KIND, RIGHTS_KIND, BREAKER_KIND = ("seed_source_mapping", "sealed_adjacent", "rights_note",
                                                     "circuit_breaker")
ALWAYS_UNRESOLVED_KINDS = ("coverage_values_only", "coverage_exact_only", "lineage_b_related_class",
                           "lineage_b_audit_family")
COVERED_KINDS = LINK_KINDS + (SEED_KIND, SEALED_KIND, RIGHTS_KIND, BREAKER_KIND) + ALWAYS_UNRESOLVED_KINDS
ANSWER_LETTERS = {"Q1": ("a", "b"), "Q2": ("a", "b", "c"), "Q3": ("a", "b"), "Q4": ("a", "b")}
RIGHTS_CHOICES = ("unknown", "undecided")
Q5_RIGHTS = {"a": "unknown", "b": "undecided"}          # 'c' (prohibited) is refused: a per-row owner call
RIGHTS_Q5 = {v: k for k, v in Q5_RIGHTS.items()}
MAPPING_TABLE = {
    "seed_source_mapping": "confirm_mapping under Q1=a and Q1=b (reject_mapping is never generated)",
    "link rows (" + ", ".join(LINK_KINDS) + ")": "Q1=a: the row's recommendation (union_subset with its sets copied "
                                                 "verbatim; a recommended unresolved stays unresolved); Q1=b: unresolved",
    "sealed_adjacent": "Q2=a: keep_sealed (ruling keep); Q2=b: the row's recommendation (override); Q2=c: unseal "
                       "(override)",
    "rdatasets granularity": "Q3=a: item (ruling); Q3=b: refused (R2)",
    "circuit_breaker": "Q4=a: unresolved; Q4=b: union",
    ", ".join(ALWAYS_UNRESOLVED_KINDS): "unresolved under Q4=a and Q4=b",
    "rights_note": "Q5=a (--rights unknown): unknown; Q5=b (--rights undecided): no entry; Q5=c: refused (R4)",
    "split-registry": "real: rulings.split_registry_excluded_seals_nothing true (stated in the owner line); DRAFT and "
                      "what-if: null and listed in rulings_unconfirmed",
    "any other kind": "STOP (R5)",
}

UNION_SUBSET_BASIS = re.compile(r"agent-generated \(deterministic rule, not a decision\): pairwise nds/name/legacy "
                                r"evidence; sets=(\[.*\])")
FID_RE = re.compile(r"fam-[0-9a-f]{64}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
IDENT_RE = re.compile(r"[a-z0-9_]{1,40}")
ANSWER_ITEM_RE = re.compile(r"(Q[0-9]+)=([a-z])")
UTC_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?Z")
LINES_RE = re.compile(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?")
REAL_RECEIPT_RE = re.compile(r"decisions-expand-[0-9]{13,20}\.json")
SHAPE_NAME_RE = re.compile(r"dl-d5-shape-[0-9a-f]{16}\.json")
_SP = r"[ \t]"
CONFIRM_TEMPLATE = ("DL-D5 confirmed for review.csv <sha12>: Q1=x, Q2=x, Q3=x, Q4=x, Q5=x; "
                    "split-registry excluded seals nothing.")
CONFIRM_RE = re.compile(
    rf"DL-D5{_SP}+confirmed{_SP}+for{_SP}+review\.csv{_SP}+(?P<sha12>[0-9a-f]{{12}}){_SP}*:"
    + "".join(rf"{_SP}*Q{n}{_SP}*={_SP}*(?P<Q{n}>[a-z]){_SP}*{sep}" for n, sep in
              ((1, ","), (2, ","), (3, ","), (4, ","), (5, ";")))
    + rf"{_SP}*split-registry{_SP}+excluded{_SP}+seals{_SP}+nothing{_SP}*\.?",
    re.IGNORECASE | re.ASCII)          # ASCII: no Unicode case folding (U+017F would otherwise match 's')
BULLET_RE = re.compile(r"[-*][ \t]+")
MESSAGE_KINDS = ("owner_verbatim_message", "owner_session_selections")
PROVENANCE_CHOICES = MESSAGE_KINDS + ("owner_directive",)
FORBIDDEN_SESSION_PREFIXES = ("DRAFT", "SIMULATION", "WHATIF")
OUTCOME_KEYS = ("families", "N_pool", "k", "pool_contexts", "auto", "reviewed", "unresolved", "sealed", "sealed_hard",
                "sealed_provisional", "seed_exposed")
OUTCOME_REQUIRED = ("families", "N_pool", "k")
FRAC = 0.05                                             # holdout.FRAC: k = ceil(0.05 * N)
REAL_ONLY_ARGS = ("out", "receipt", "split_registry_confirmed", "provenance_kind", "session", "message_utc",
                  "message_file", "message_sha256", "directive_file", "directive_version", "directive_lines",
                  "directive_sha256", "question_posted_utc")
MESSAGE_ARGS = ("session", "message_utc", "message_file", "message_sha256", "question_posted_utc")
DIRECTIVE_ARGS = ("directive_file", "directive_version", "directive_lines", "directive_sha256")

# ---- the review shape the owner is shown (P2 generation 2, review.csv aad99b1a...): compute_shape of that review.csv
# under the frozen families code, pinned here literally (shape sha256 6790b102...; also stored create-only as
# RUN/receipts/P2/staging/dl-d5-shape-6790b1020be2a218.json)
_TRIGGER4 = ["independent", "union", "union_subset", "unresolved"]
BUILTIN_SHAPES = {
    "builtin:aad99b1a": {
        "schema": SHAPE_SCHEMA,
        "review_csv_sha256": "aad99b1a9af6056a4cd812a962871990ae35fce24fb8b699cbd9df116d188fcb",
        "rows": 413,
        "by_kind": {
            "circuit_breaker": {"unresolved": 1},
            "classic_candidate": {"union_subset": 1, "unresolved": 10},
            "coverage_exact_only": {"unresolved": 62},
            "coverage_values_only": {"unresolved": 22},
            "github_owner": {"independent": 4, "union_subset": 1},
            "legacy_source_family": {"union_subset": 28},
            "lineage_b_audit_family": {"union_subset": 1},
            "lineage_b_related_class": {"unresolved": 2},
            "multi_member_family": {"confirm_union": 88},
            "name_candidate": {"union_subset": 14},
            "nds_value_weak": {"union_subset": 7},
            "publisher": {"independent": 31, "union_subset": 6},
            "rdatasets_pkg": {"independent": 37, "union_subset": 14},
            "rights_note": {"unknown": 14},
            "sealed_adjacent": {"keep_sealed": 33, "unseal": 4},
            "seed_source_mapping": {"confirm_mapping": 10},
            "socrata_domain": {"independent": 20},
            "tidytuesday_year": {"independent": 2, "union_subset": 1},
        },
        "link_rows": {"rows": 264, "confirm_union": 88, "independent": 94, "union_subset": 72, "unresolved": 10},
        "union_subset_sets": {"classic_candidate": 1, "github_owner": 1, "legacy_source_family": 28,
                              "lineage_b_audit_family": 1, "name_candidate": 14, "nds_value_weak": 7, "publisher": 7,
                              "rdatasets_pkg": 21, "tidytuesday_year": 1},
        "union_subset_set_sizes": {"12": 1, "13": 1, "2": 52, "3": 20, "4": 2, "5": 2, "6": 1, "8": 1, "9": 1},
        "circuit_breaker_ids": ["rg-e6fb3d57ef2782074eeb468d"],
        "rows_digest": "981d26ba21161dcb790bba124c5bc6c8c499d1345e37577d277a8be3f2132811",
        "decision_options": {
            "circuit_breaker": ["union", "union_subset", "unresolved"],
            "classic_candidate": _TRIGGER4, "coverage_exact_only": ["independent", "unresolved"],
            "coverage_values_only": ["independent", "unresolved"], "github_owner": _TRIGGER4,
            "legacy_source_family": _TRIGGER4, "lineage_b_audit_family": _TRIGGER4,
            "lineage_b_related_class": _TRIGGER4, "multi_member_family": ["confirm_union", "split", "unresolved"],
            "name_candidate": _TRIGGER4, "nds_value_weak": _TRIGGER4, "publisher": _TRIGGER4,
            "rdatasets_pkg": _TRIGGER4, "rights_note": ["unknown", "prohibited"],
            "sealed_adjacent": ["keep_sealed", "unseal"],
            "seed_source_mapping": ["confirm_mapping", "add_members", "reject_mapping"],
            "socrata_domain": _TRIGGER4, "tidytuesday_year": _TRIGGER4,
        },
        "code": {"families": "7dcd65ff5646cdf93bcf450fa9e8f09bbe3532c0a77d42d827be7a4698d575e2",
                 "family_keys": "9b688e9b8a36a32485524243dc1adea7d30ee7404df4027d2b4d17bd6a900f7e",
                 "family_pin": "05187c72156c75f3787d67b1ee94e584da192b56fbd9522d7fcb93ad7dbd7e25",
                 "family_screen": "76365b12de5fa704569d302fd0d86de4de7a37f985714063cdb3a3ba7dc401cf"},
    },
}
BUILTIN_SHAPE_SHA256 = {"builtin:aad99b1a": "6790b1020be2a21896ffa275a5ffa811062600654ee180b999adc91d0808bc17"}


class Refused(Exception):
    """A STOP. code is the refusal (R1..R12) or a check/dryrun stage name."""

    def __init__(self, code: str, reason: str):
        super().__init__(f"{code}: {reason}")
        self.code, self.reason = code, reason


@dataclass(frozen=True)
class Roots:
    """Where the tool may write. Tests inject temporary roots under TMPDIR."""
    families: Path = FAMILIES_ROOT
    receipts: Path = RECEIPTS_ROOT
    stage: Path = STAGE
    stage_receipts: Path = STAGE_RECEIPTS


@dataclass(frozen=True)
class Bound:
    path: Path
    raw: bytes
    sha256: str

    def ref(self) -> dict:
        return {"path": str(self.path), "sha256": self.sha256, "bytes": len(self.raw)}


# ------------------------------------------------------------------------------------------------- small helpers
def tool_sha256() -> str:
    return common.sha256_file(Path(__file__).resolve())


def current_code_shas() -> dict:
    """The four module shas registry.pre-review.json records ("code"), as families._code_shas computes them now. R7
    compares the two; any difference means the final build would rebuild different pre-review bytes and STOP."""
    return families._code_shas()


def code_shas() -> dict:
    """The four module shas R7 compared, plus this tool."""
    return dict(current_code_shas(), tool=tool_sha256())


def _ident(v) -> str:
    return v if isinstance(v, str) and IDENT_RE.fullmatch(v) else "<non-identifier>"


def _inside(path, root) -> bool:
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def _present(args, name) -> bool:
    v = getattr(args, name, None)
    return v not in (None, False)


def read_pinned(path, pin, what: str, code: str = "R7") -> Bound:
    if not isinstance(pin, str) or not HEX64_RE.fullmatch(pin):
        raise Refused(code, f"{what}: the sha256 pin must be 64 lowercase hex")
    if not path:
        raise Refused(code, f"{what}: no path given")
    p = Path(path)
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        raise Refused(code, f"{what}: {p} is missing") from None
    got = common.sha256_bytes(raw)
    if got != pin:
        raise Refused(code, f"{what}: sha256 {got} differs from its pin {pin}")
    return Bound(p, raw, got)


def strict_json(raw: bytes, what: str, code: str):
    """Strict JSON for files a person or this tool authored: no duplicate keys, no NaN or Infinity."""
    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)

    def bad_constant(name):
        raise ValueError(f"non-finite number {name}")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (ValueError, UnicodeDecodeError):
        raise Refused(code, f"{what}: not strict JSON") from None


def k_of(n) -> int | None:
    return math.ceil(FRAC * n) if isinstance(n, int) and not isinstance(n, bool) else None


def utc_key(text) -> tuple | None:
    """A sortable key of an ISO-8601 UTC timestamp (YYYY-MM-DDTHH:MM[:SS[.f]]Z), or None."""
    m = UTC_RE.fullmatch(text) if isinstance(text, str) else None
    if m is None:
        return None
    y, mo, d, h, mi, s, frac = m.groups()
    return int(y), int(mo), int(d), int(h), int(mi), int(s or 0), int((frac or "0").ljust(9, "0"))


def _rmtree(path: Path) -> None:
    """Remove a tree this tool created, including the 0444 files and read-only directories its steps leave."""
    def retry(fn, p, _exc):
        parent = os.path.dirname(p)
        for target in (parent, p):
            with contextlib.suppress(OSError):
                os.chmod(target, 0o700)
        fn(p)
    if path.exists():
        shutil.rmtree(path, onexc=retry)


# ------------------------------------------------------------------------------------------------- review.csv
def parse_rows(raw: bytes) -> list[dict]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused("R7", "review.csv is not UTF-8") from None
    reader = csv.DictReader(io.StringIO(text))
    try:
        rows = list(reader)
    except csv.Error:
        raise Refused("R7", "review.csv is not parseable CSV") from None
    if tuple(reader.fieldnames or ()) != families.REVIEW_COLUMNS:
        raise Refused("R7", "review.csv columns differ from families.REVIEW_COLUMNS")
    for n, r in enumerate(rows):
        if None in r or any(v is None for v in r.values()):
            raise Refused("R7", f"review.csv data row {n + 1} has a missing or extra field")
    ids = [r["review_group_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise Refused("R7", f"review_group_ids are not unique ({len(ids) - len(set(ids))} duplicate(s))")
    return rows


def family_ids(row: dict) -> list[str]:
    return [f for f in row["family_ids"].split(";") if f]


def row_options(row: dict) -> tuple[tuple, tuple]:
    return (tuple(o for o in row["decision_options"].split(";") if o),
            tuple(families.DECISION_OPTIONS.get(row["kind"], ())))


def parse_sets(row: dict) -> list:
    """R6: the union_subset sets of a row, copied exactly as review.csv lists them in recommendation_basis."""
    gid, basis = row["review_group_id"], row["recommendation_basis"]
    if basis.count("sets=") != 1:
        raise Refused("R6", f"{gid}: recommendation_basis must hold exactly one 'sets='")
    m = UNION_SUBSET_BASIS.fullmatch(basis)
    if m is None:
        raise Refused("R6", f"{gid}: recommendation_basis does not match the union_subset rule text")
    text = m.group(1)
    sets = strict_json(text.encode("utf-8"), f"{gid}: sets", "R6")
    if not isinstance(sets, list) or not sets:
        raise Refused("R6", f"{gid}: sets must be a non-empty list")
    fids, used = set(family_ids(row)), set()
    for s in sets:
        if not isinstance(s, list) or not all(isinstance(x, str) and FID_RE.fullmatch(x) for x in s):
            raise Refused("R6", f"{gid}: a set is not a list of fam-<64hex> ids")
        if len(s) < 2 or len(set(s)) != len(s):
            raise Refused("R6", f"{gid}: a set does not hold >= 2 distinct families")
        if s != sorted(s):
            raise Refused("R6", f"{gid}: a set is not sorted")
        if not set(s) <= fids:
            raise Refused("R6", f"{gid}: a set names a family outside the row's family_ids")
        if set(s) & used:
            raise Refused("R6", f"{gid}: the sets are not pairwise disjoint")
        used |= set(s)
    if common.canonical(sets).decode("utf-8") != text:
        raise Refused("R6", f"{gid}: the sets text is not canonical JSON")
    return sets


# ------------------------------------------------------------------------------------------------- shape (R8)
def link_summary(rows: list) -> dict:
    c = Counter(r["recommendation"] for r in rows if r["kind"] in LINK_KINDS)
    return {"rows": sum(c.values()), **dict(sorted(c.items()))}


def rows_digest(rows: list) -> str:
    """sha256 over the decision-relevant tuple of every row, in review.csv order (D-MAP-3: equal counts with different
    content are a different shape)."""
    tuples = [[r["review_group_id"], r["kind"], r["recommendation"], sorted(family_ids(r)),
               parse_sets(r) if r["recommendation"] == "union_subset" else None, list(row_options(r)[0])]
              for r in rows]
    return common.sha256_bytes(common.canonical(tuples))


def compute_shape(rows: list, csv_sha256: str) -> dict:
    """The review shape, derived from review.csv and the frozen families code."""
    by_kind, us_sets, sizes = {}, Counter(), Counter()
    for r in rows:
        by_kind.setdefault(r["kind"], Counter())[r["recommendation"]] += 1
        if r["recommendation"] == "union_subset":
            sets = parse_sets(r)
            us_sets[r["kind"]] += len(sets)
            for s in sets:
                sizes[str(len(s))] += 1
    kinds = sorted(by_kind)
    return {"schema": SHAPE_SCHEMA, "review_csv_sha256": csv_sha256, "rows": len(rows),
            "by_kind": {k: dict(sorted(v.items())) for k, v in sorted(by_kind.items())},
            "link_rows": link_summary(rows),
            "union_subset_sets": dict(sorted(us_sets.items())),
            "union_subset_set_sizes": dict(sorted(sizes.items())),
            "circuit_breaker_ids": sorted(r["review_group_id"] for r in rows if r["kind"] == BREAKER_KIND),
            "rows_digest": rows_digest(rows),
            "decision_options": {k: list(families.DECISION_OPTIONS.get(k, ())) for k in kinds},
            "code": current_code_shas()}


def shape_sha256(shape: dict) -> str:
    return common.sha256_bytes(common.canonical(shape))


SHAPE_KEYS = ("schema", "review_csv_sha256", "rows", "link_rows", "union_subset_sets", "union_subset_set_sizes",
              "circuit_breaker_ids", "rows_digest", "decision_options", "code")


def shape_diff(want: dict, got: dict) -> list[str]:
    """Names (never values) of the shape entries that differ."""
    out = [key for key in SHAPE_KEYS if want.get(key) != got.get(key)]
    wk, gk = want.get("by_kind") or {}, got.get("by_kind") or {}
    for kind in sorted(set(wk) | set(gk)):
        if wk.get(kind) != gk.get(kind):
            out.append(f"by_kind.{_ident(kind)}")
    if set(want) != set(got):
        out.append("keys")
    return out


def expected_shape(spec, sha) -> tuple[dict, dict]:
    if not spec:
        raise Refused("R8", "--expect-shape is required (builtin:aad99b1a, or a shape file with its sha256)")
    if spec.startswith("builtin:"):
        if sha is not None:
            raise Refused("R8", "--expect-shape-sha256 goes only with a shape file")
        if spec not in BUILTIN_SHAPES:
            raise Refused("R8", f"unknown built-in shape {spec!r} (g1 builtin:c7377cf8 is superseded)")
        if shape_sha256(BUILTIN_SHAPES[spec]) != BUILTIN_SHAPE_SHA256[spec]:
            raise Refused("R8", f"the built-in shape {spec} no longer hashes to its pin {BUILTIN_SHAPE_SHA256[spec][:16]}")
        return BUILTIN_SHAPES[spec], {"builtin": spec, "sha256": BUILTIN_SHAPE_SHA256[spec]}
    b = read_pinned(spec, sha, "--expect-shape file", code="R8")
    doc = strict_json(b.raw, "--expect-shape file", "R8")
    if not isinstance(doc, dict) or doc.get("schema") != SHAPE_SCHEMA:
        raise Refused("R8", f"--expect-shape file is not a {SHAPE_SCHEMA} document")
    if not isinstance(doc.get("review_csv_sha256"), str) or not HEX64_RE.fullmatch(doc["review_csv_sha256"]):
        raise Refused("R8", "--expect-shape file binds no review.csv sha256")
    return doc, {"file": b.ref()}


def require_shape(rows: list, want: dict, csv_sha256: str) -> tuple[dict, str]:
    got = compute_shape(rows, csv_sha256)
    if got != want:
        raise Refused("R8", "review.csv shape differs from --expect-shape (a new review set needs a new shape and a new "
                            f"owner confirmation): {shape_diff(want, got)}")
    return got, shape_sha256(got)


def require_shape_binds(want: dict, csv_b: Bound) -> None:
    if want.get("review_csv_sha256") != csv_b.sha256:
        raise Refused("R8", f"the expected shape binds review.csv {str(want.get('review_csv_sha256'))[:12]}, not the bound "
                            f"review.csv {csv_b.sha256[:12]} (a new review set needs a new shape and a new owner "
                            "confirmation)")


# ------------------------------------------------------------------------------------------------- bindings (R7)
def check_bindings(csv_b: Bound, pre_b: Bound, alias_b: Bound) -> dict:
    try:
        pre = json.loads(pre_b.raw)
    except (ValueError, UnicodeDecodeError):
        raise Refused("R7", "registry.pre-review.json is not JSON") from None
    if not isinstance(pre, dict) or pre.get("schema") != families.REGISTRY_SCHEMA:
        raise Refused("R7", f"registry.pre-review.json is not a {families.REGISTRY_SCHEMA} document")
    if pre.get("phase") != "pre_review":
        raise Refused("R7", f"registry.pre-review.json has phase {pre.get('phase')!r}, not 'pre_review'")
    if pre.get("review_csv_sha256") != csv_b.sha256:
        raise Refused("R7", "registry.pre-review.json binds a different review.csv")
    if pre.get("alias_sources_sha256") != alias_b.sha256:
        raise Refused("R7", "registry.pre-review.json binds a different alias-sources.json")
    want, have = pre.get("code") if isinstance(pre.get("code"), dict) else {}, current_code_shas()
    differ = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    if differ:
        raise Refused("R7", f"current code differs from registry.pre-review.json 'code' for {differ}: the final build "
                            "would rebuild different pre-review bytes and STOP (freeze the code, or rebuild and "
                            "re-review)")
    if not isinstance(pre.get("families"), dict):
        raise Refused("R7", "registry.pre-review.json has no families table")
    return pre


def read_inputs(args) -> tuple[Bound, Bound, Bound]:
    return (read_pinned(args.review_csv, args.review_csv_sha256, "review.csv"),
            read_pinned(args.pre_review, args.pre_review_sha256, "registry.pre-review.json"),
            read_pinned(args.alias_sources, args.alias_sources_sha256, "alias-sources.json"))


def bound_dir(csv_b: Bound, pre_b: Bound) -> Path:
    d = csv_b.path.resolve().parent
    if pre_b.path.resolve().parent != d:
        raise Refused("R10", "review.csv and registry.pre-review.json must sit in one bound families dir")
    return d


# ------------------------------------------------------------------------------------------------- answers
def parse_answers(text) -> dict:
    """R1/R2: exactly Q1..Q4 once each, letters within each question's options; Q3=b is refused."""
    if not isinstance(text, str) or not text:
        raise Refused("R1", "--answers is required: Q1=x,Q2=x,Q3=x,Q4=x")
    out = {}
    for item in text.split(","):
        m = ANSWER_ITEM_RE.fullmatch(item)
        if m is None:
            raise Refused("R1", "--answers items must read Qn=x with a lowercase letter, comma-separated, no spaces")
        q, letter = m.groups()
        if q not in ANSWER_LETTERS:
            raise Refused("R1", f"--answers has an unknown key {q} (Q5 is given as --rights)")
        if q in out:
            raise Refused("R1", f"--answers names {q} twice")
        if letter not in ANSWER_LETTERS[q]:
            raise Refused("R1", f"{q}={letter} is outside {q} in {list(ANSWER_LETTERS[q])}")
        out[q] = letter
    missing = [q for q in ANSWER_LETTERS if q not in out]
    if missing:
        raise Refused("R1", f"--answers lacks {missing}")
    if out["Q3"] == "b":
        raise Refused("R2", "Q3=b (package level) is not expanded: under rulings.rdatasets_granularity 'package' "
                            "families unions every row whose trigger kinds hold rdatasets_pkg and STOPs on any other "
                            "decision there, which contradicts Q1's per-row link decisions; a package-level answer needs "
                            "its own owner review")
    return {q: out[q] for q in ANSWER_LETTERS}


def parse_rights(value) -> str:
    if value is None:
        raise Refused("R4", "--rights is required (unknown or undecided; there is no default)")
    if value not in RIGHTS_CHOICES:
        raise Refused("R4", "--rights must be unknown or undecided ('prohibited' is a per-row owner call, not a DL-D5 "
                            "answer)")
    return value


def parse_outcome(text) -> dict:
    if not text:
        raise Refused("R11", "--simulate needs --expect-outcome (at least families=,N_pool=,k=)")
    out = {}
    for item in text.split(","):
        key, sep, val = item.partition("=")
        if not sep or key not in OUTCOME_KEYS or not re.fullmatch(r"[0-9]+", val):
            raise Refused("R11", f"--expect-outcome items must read key=int with key in {list(OUTCOME_KEYS)}")
        if key in out:
            raise Refused("R11", f"--expect-outcome names {key} twice")
        out[key] = int(val)
    missing = [k for k in OUTCOME_REQUIRED if k not in out]
    if missing:
        raise Refused("R11", f"--expect-outcome lacks {missing}")
    return out


def outcome_of(counts: dict) -> dict:
    iq = counts.get("independence_qualified") or {}
    sealed = counts.get("sealed") or {}
    n = counts.get("N_pool")
    return {"families": counts.get("families"), "N_pool": n, "k": k_of(n), "pool_contexts": counts.get("pool_contexts"),
            "auto": iq.get("auto", 0), "reviewed": iq.get("reviewed", 0), "unresolved": iq.get("unresolved", 0),
            "sealed": sealed.get("families"), "sealed_hard": sealed.get("hard_families"),
            "sealed_provisional": sealed.get("provisional_families"), "seed_exposed": counts.get("seed_exposed_families")}


# ------------------------------------------------------------------------------------------------- owner text (R9)
def parse_confirmation(text, csv_sha256: str, *, directive: bool = False) -> dict:
    """D-MAP-1/D-MAP-3: the whole text is the one affirmative confirmation line of dl-d5-confirm/v1, and it names the
    bound review.csv. Letters are taken from that line only."""
    if not isinstance(text, str):
        raise Refused("R9", "the owner text is not text")
    body = text.replace("\r\n", "\n").strip()
    if directive:
        body = BULLET_RE.sub("", body, count=1) if BULLET_RE.match(body) else body
    if not body:
        raise Refused("R9", "the owner text is empty")
    if "\n" in body or "\r" in body:
        raise Refused("R9", "the owner text must be exactly one line, the affirmative DL-D5 confirmation "
                            f"{CONFIRM_TEMPLATE!r}, with no other text")
    m = CONFIRM_RE.fullmatch(body)
    if m is None:
        raise Refused("R9", f"the owner text is not the affirmative DL-D5 confirmation line {CONFIRM_TEMPLATE!r} "
                            "(no other words, questions, negations or quoted suggestions)")
    sha12 = m.group("sha12").lower()
    if sha12 != csv_sha256[:12]:
        raise Refused("R9", f"the owner text confirms review.csv {sha12}, not the bound review.csv {csv_sha256[:12]}")
    return {"review_csv_sha12": sha12, **{f"Q{n}": m.group(f"Q{n}").lower() for n in range(1, 6)}}


def check_letters(parsed: dict, answers: dict, rights: str) -> dict:
    """D-MAP-2: Q1..Q4 equal --answers and Q5 equals --rights in both directions."""
    q14 = {q: parsed[q] for q in ANSWER_LETTERS}
    if q14 != answers:
        raise Refused("R9", f"the owner text answers {q14}, not the --answers letters {answers}")
    q5 = parsed["Q5"]
    if q5 == "c":
        raise Refused("R4", "the owner text answers Q5=c (prohibited): a per-row owner call this tool does not expand")
    if q5 not in Q5_RIGHTS:
        raise Refused("R9", f"the owner text answers Q5={q5}: Q5 must be a (rights_note unknown) or b (undecided)")
    if Q5_RIGHTS[q5] != rights:
        raise Refused("R9", f"the owner text answers Q5={q5} (rights_note {Q5_RIGHTS[q5]}), not --rights {rights}")
    return {"grammar": CONFIRM_GRAMMAR, "parsed": dict(sorted(parsed.items())), "rights_note": rights,
            "split_registry_stated": True}


def _check_session(session) -> None:
    if not isinstance(session, str) or not session.strip():
        raise Refused("R9", "provenance session is required (screen/pid/starttick/cwd of the owner's session)")
    if session.strip().upper().startswith(FORBIDDEN_SESSION_PREFIXES):
        raise Refused("R9", "provenance session starts with DRAFT, SIMULATION or WHATIF")


def message_provenance(kind, session, message_utc, message: Bound, answers, rights, csv_sha256: str,
                       posted_utc) -> tuple[dict, dict, dict]:
    if kind not in MESSAGE_KINDS:
        raise Refused("R9", "provenance kind must be owner_verbatim_message or owner_session_selections here")
    _check_session(session)
    if utc_key(message_utc) is None:
        raise Refused("R9", "provenance message_utc must be ISO-8601 UTC ending in Z")
    if utc_key(posted_utc) is None:
        raise Refused("R9", "--question-posted-utc must be ISO-8601 UTC ending in Z (when the g2 DL-D5 question set was "
                            "posted)")
    if utc_key(message_utc) < utc_key(posted_utc):
        raise Refused("R9", "the owner message predates the question set it answers (--question-posted-utc): it binds "
                            "recommendations the owner had not been shown")
    if message.sha256 == RELAY_SHA256:
        raise Refused("R9", "the message is the relayed DL-D5 paste b785a123...: PLAN §7.2 needs the owner's own typed "
                            "message")
    try:
        text = message.raw.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused("R9", "the owner message is not UTF-8") from None
    parsed = parse_confirmation(text, csv_sha256)
    checks = check_letters(parsed, answers, rights)
    confirmation = {"grammar": CONFIRM_GRAMMAR, "review_csv_sha12": parsed["review_csv_sha12"],
                    "letters": {q: parsed[q] for q in ("Q1", "Q2", "Q3", "Q4", "Q5")}, "question_posted_utc": posted_utc}
    return ({"kind": kind, "session": session, "message_utc": message_utc, "message_sha256": message.sha256,
             "message_text": text}, checks, confirmation)


def directive_text(raw: bytes, lines: str) -> str:
    m = LINES_RE.fullmatch(lines or "")
    if m is None:
        raise Refused("R9", "--directive-lines must name one line N")
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    if a != b:
        raise Refused("R9", "--directive-lines must name exactly one line: the confirmation line")
    try:
        all_lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise Refused("R9", "the directive file is not UTF-8") from None
    if not a <= len(all_lines):
        raise Refused("R9", f"--directive-lines {a} is outside the file's {len(all_lines)} lines")
    return all_lines[a - 1]


def directive_provenance(file_b: Bound, version, lines, answers, rights, csv_sha256: str) -> tuple[dict, dict, dict]:
    """D-MAP-5: only the owner's own OWNER_DIRECTIVES.md, re-hashed, one cited line in the confirmation grammar."""
    if file_b.path.resolve() != Path(OWNER_DIRECTIVES_PATH).resolve():
        raise Refused("R9", f"owner_directive provenance accepts only the owner's directive file {OWNER_DIRECTIVES_PATH} "
                            "(PLAN §7.2: text from an agent, orchestrator or workflow never counts)")
    if not isinstance(version, str) or not version.strip():
        raise Refused("R9", "--directive-version is required")
    if file_b.sha256 == RELAY_SHA256:
        raise Refused("R9", "the directive file is the relayed DL-D5 paste")
    parsed = parse_confirmation(directive_text(file_b.raw, lines), csv_sha256, directive=True)
    checks = check_letters(parsed, answers, rights)
    confirmation = {"grammar": CONFIRM_GRAMMAR, "review_csv_sha12": parsed["review_csv_sha12"],
                    "letters": {q: parsed[q] for q in ("Q1", "Q2", "Q3", "Q4", "Q5")}, "question_posted_utc": None}
    return ({"kind": "owner_directive", "file": str(Path(OWNER_DIRECTIVES_PATH).resolve()), "version": version,
             "lines": lines, "file_sha256": file_b.sha256}, checks, confirmation)


def real_provenance(args, answers, rights, csv_b: Bound) -> tuple[dict, dict, Bound | None, dict]:
    kind = args.provenance_kind
    if kind not in PROVENANCE_CHOICES:
        raise Refused("R9", f"real mode needs --provenance-kind in {list(PROVENANCE_CHOICES)}")
    if kind in MESSAGE_KINDS:
        if any(getattr(args, a) is not None for a in DIRECTIVE_ARGS):
            raise Refused("R9", "directive arguments go only with --provenance-kind owner_directive")
        for a in MESSAGE_ARGS:
            if getattr(args, a) is None:
                raise Refused("R9", f"--{a.replace('_', '-')} is required with {kind}")
        message = read_pinned(args.message_file, args.message_sha256, "--message-file", code="R9")
        prov, checks, conf = message_provenance(kind, args.session, args.message_utc, message, answers, rights,
                                                csv_b.sha256, args.question_posted_utc)
        return prov, checks, message, conf
    if any(getattr(args, a) is not None for a in MESSAGE_ARGS):
        raise Refused("R9", "message arguments (and --question-posted-utc) go only with a message provenance kind")
    for a in DIRECTIVE_ARGS:
        if getattr(args, a) is None:
            raise Refused("R9", f"--{a.replace('_', '-')} is required with owner_directive")
    file_b = read_pinned(args.directive_file, args.directive_sha256, "--directive-file", code="R9")
    prov, checks, conf = directive_provenance(file_b, args.directive_version, args.directive_lines, answers, rights,
                                              csv_b.sha256)
    return prov, checks, None, conf


def recorded_provenance(prov: dict, answers: dict, rights: str, csv_sha256: str, confirmation) -> tuple[dict, dict, dict]:
    """check: rebuild a real provenance block from its recorded fields, re-running the R9 rules."""
    if not isinstance(prov, dict):
        raise Refused("R9", "the decision file has no provenance block")
    kind = prov.get("kind")
    if kind in MESSAGE_KINDS:
        text = prov.get("message_text")
        if not isinstance(text, str):
            raise Refused("R9", "the recorded provenance lacks message_text")
        raw = text.encode("utf-8")
        if common.sha256_bytes(raw) != prov.get("message_sha256"):
            raise Refused("R9", "the recorded message_text does not hash to message_sha256")
        posted = confirmation.get("question_posted_utc") if isinstance(confirmation, dict) else None
        p, c, conf = message_provenance(kind, prov.get("session"), prov.get("message_utc"),
                                        Bound(Path("<recorded message_text>"), raw, common.sha256_bytes(raw)), answers,
                                        rights, csv_sha256, posted)
        return p, c, conf
    if kind == "owner_directive":
        file_b = read_pinned(prov.get("file"), prov.get("file_sha256"), "the recorded directive file", code="R9")
        return directive_provenance(file_b, prov.get("version"), prov.get("lines"), answers, rights, csv_sha256)
    raise Refused("R9", "the recorded provenance kind is not one this tool writes")


def real_generation_gate(prov: dict, message: Bound | None, receipts_dir) -> dict:
    """D-MAP-5: families.check_provenance(real=True) on the provenance about to be written, before any write. A message
    provenance resolves through an explicit record of the pinned message file (the stored copy is written next); an
    owner_directive is re-hashed in place."""
    probe = dict(prov) if message is None else dict(prov, record={"path": str(message.path.resolve()),
                                                                  "sha256": message.sha256})
    try:
        res = families.check_provenance(probe, real=True, receipts_dir=Path(receipts_dir).resolve())
    except families.FamilyStop as err:
        raise Refused("R9", f"families refuses this provenance for a real generation (CL-2): {str(err)[:300]}") from None
    rec = res.get("record") or {}
    return {"draft": res.get("draft"), "record_form": rec.get("form"), "record_sha256": rec.get("sha256")}


# ------------------------------------------------------------------------------------------------- the expansion
def decide(kind: str, rec: str, answers: dict, rights: str) -> str | None:
    """dl-d5-map/v2. None means: write no entry for this row."""
    if kind == SEED_KIND:
        return "confirm_mapping"
    if kind in LINK_KINDS:
        return rec if answers["Q1"] == "a" else "unresolved"
    if kind == SEALED_KIND:
        return {"a": "keep_sealed", "b": rec, "c": "unseal"}[answers["Q2"]]
    if kind == RIGHTS_KIND:
        return "unknown" if rights == "unknown" else None
    if kind == BREAKER_KIND:
        return "unresolved" if answers["Q4"] == "a" else "union"
    if kind in ALWAYS_UNRESOLVED_KINDS:
        return "unresolved"
    raise Refused("R5", f"row kind {_ident(kind)} is not covered by {MAPPING}")


def expand_rows(rows: list, answers: dict, rights: str) -> list[dict]:
    out = []
    for r in rows:
        gid, kind, rec = r["review_group_id"], r["kind"], r["recommendation"]
        if kind not in COVERED_KINDS:
            raise Refused("R5", f"{gid}: row kind {_ident(kind)} is not covered by {MAPPING} (withdrawn_family_key, "
                                "hf_owner, pmlb_stem, socrata_4x4_cross_host, hunt_semantic_signature and new kinds STOP)")
        csv_opts, fam_opts = row_options(r)
        if rec not in csv_opts or rec not in fam_opts:
            raise Refused("R5", f"{gid}: recommendation {_ident(rec)} is not among the row's decision options")
        d = decide(kind, rec, answers, rights)
        if d is None:
            continue
        if d not in csv_opts or d not in fam_opts:
            raise Refused("R5", f"{gid}: decision {_ident(d)} is outside the row's decision options")
        entry = {"review_group_id": gid, "decision": d}
        if d == "union_subset":
            entry["sets"] = parse_sets(r)
        out.append(entry)
    return out


def rulings_for(answers: dict, *, split) -> dict:
    return {"rdatasets_granularity": "item", "provisional_seals": "keep" if answers["Q2"] == "a" else "override",
            "related_class_policy": RELATED_CLASS_POLICY, "split_registry_excluded_seals_nothing": split}


def answers_block(mode: str, answers: dict, rights: str, confirmation) -> dict:
    if mode == "real":
        return dict(answers, Q5=confirmation["letters"]["Q5"], rights_note=rights, split_registry_confirmed=True)
    return dict(answers, Q5=None if mode == "draft" else RIGHTS_Q5[rights], rights_note=rights,
                split_registry_confirmed=False)


def decision_doc(*, mode: str, provenance: dict, binds: dict, answers: dict, rights: str, decisions: list,
                 shape_sha: str, confirmation=None) -> dict:
    expansion = {"tool": TOOL, "tool_sha256": tool_sha256(), "mapping": MAPPING, "shape_sha256": shape_sha}
    block = answers_block(mode, answers, rights, confirmation)
    if mode == "real":
        return {"schema": families.DECISIONS_SCHEMA, "reviewer": "owner", "provenance": provenance, "binds": binds,
                "rulings": rulings_for(answers, split=True), "decisions": decisions, "class_decisions": [],
                "answers": block, "confirmation": confirmation, "expansion": expansion}
    return {"schema": UNCONFIRMED_SCHEMA[mode], "reviewer": UNCONFIRMED_REVIEWER, "notice": NOTICE[mode],
            "provenance": provenance, "binds": binds, "rulings": rulings_for(answers, split=None),
            "rulings_unconfirmed": ["split_registry_excluded_seals_nothing"], "decisions": decisions,
            "class_decisions": [], "answers": block, "expansion": expansion}


def rehearsal_provenance(mode: str, unconfirmed_sha256: str) -> dict:
    """The provenance of the in-memory rehearsal variant. families needs a known kind for any final build; the DRAFT
    markers make every real generation refuse it (CL-2). It is never written to a file."""
    return {"kind": "owner_verbatim_message", "session": f"{PREFIX[mode]}REHEARSAL in memory (no owner message)",
            "message_utc": "DRAFT: no owner timestamp", "message_sha256": unconfirmed_sha256, "draft": True,
            "confirmed_by_owner": False, "unconfirmed": [f"the {mode} file {unconfirmed_sha256}"],
            "note": "DRAFT rehearsal of an unconfirmed expansion: in memory only, never a decision file"}


def rehearsal_doc(doc: dict, mode: str, unconfirmed_sha256: str) -> dict:
    """The in-memory variant that families can build in a staging (non-real) final build. reviewer 'owner' and the
    split ruling are what families.validate_decisions requires of every final build; both are assumptions of the
    rehearsal, recorded in rehearsal_of, and the variant never reaches a file."""
    return {"schema": families.DECISIONS_SCHEMA, "reviewer": "owner",
            "provenance": rehearsal_provenance(mode, unconfirmed_sha256), "binds": doc["binds"],
            "rulings": dict(doc["rulings"], split_registry_excluded_seals_nothing=True),
            "decisions": doc["decisions"], "class_decisions": doc["class_decisions"],
            "rehearsal_of": {"schema": doc["schema"], "sha256": unconfirmed_sha256,
                             "assumed": ["reviewer owner", "split_registry_excluded_seals_nothing"]}}


def tally(rows: list, decisions: list) -> dict:
    decided = {d["review_group_id"]: d["decision"] for d in decisions}
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], Counter())[decided.get(r["review_group_id"], "<undecided>")] += 1
    undecided = Counter(r["kind"] for r in rows if r["review_group_id"] not in decided)
    return {"rows": len(rows), "decisions": len(decided), "decision_counts": dict(sorted(Counter(decided.values()).items())),
            "undecided_rows": sum(undecided.values()), "undecided_by_kind": dict(sorted(undecided.items())),
            "by_kind": {k: dict(sorted(v.items())) for k, v in sorted(by_kind.items())},
            "union_subset_sets": sum(len(d.get("sets") or ()) for d in decisions)}


def _vrows(rows: list) -> list:
    return [{"review_group_id": r["review_group_id"], "kind": r["kind"], "family_ids": family_ids(r),
             "recommendation": r["recommendation"]} for r in rows]


def validate(body: bytes, rows: list, binds: dict, pre_doc: dict) -> dict:
    """families.validate_decisions on the exact bytes, against the review.csv rows (add_members is never generated, so
    the pre-review families table stands in for the Model)."""
    try:
        res = families.validate_decisions(body, _vrows(rows), review_csv_sha256=binds["review_csv_sha256"],
                                          pre_sha256=binds["registry_pre_review_sha256"],
                                          alias_sha256=binds["alias_sources_sha256"],
                                          pre=SimpleNamespace(families=pre_doc["families"]))
    except families.FamilyStop as err:
        raise Refused("VALIDATE", f"families.validate_decisions refused the expansion: {str(err)[:300]}") from None
    breakers = [r["review_group_id"] for r in rows if r["kind"] == BREAKER_KIND
                and r["review_group_id"] not in res["effective"]]
    if breakers:
        raise Refused("VALIDATE", f"{len(breakers)} circuit_breaker row(s) undecided: run_build would STOP")
    return {"accepted": True, "effective_decisions": len(res["effective"])}


def unpromotable_families(body: bytes, rows: list, binds: dict, pre_doc: dict) -> dict:
    """D-MAP-6 self-check: families.validate_decisions refuses an unconfirmed file for a staging and a real generation."""
    out = {}
    for real in (False, True):
        try:
            families.validate_decisions(body, _vrows(rows), review_csv_sha256=binds["review_csv_sha256"],
                                        pre_sha256=binds["registry_pre_review_sha256"],
                                        alias_sha256=binds["alias_sources_sha256"],
                                        pre=SimpleNamespace(families=pre_doc["families"]), real=real,
                                        receipts_dir=RECEIPTS_ROOT if real else None)
        except families.FamilyStop as err:
            out["families_real" if real else "families_staging"] = str(err)[:200]
            continue
        raise Refused("R12", f"families.validate_decisions accepted an unconfirmed file (real={real})")
    return out


def holdout_refuses(d_raw: bytes, csv_b: Bound, alias_b: Bound) -> dict:
    """D-MAP-6 self-check: holdout.review_gate refuses an unconfirmed file even next to a registry that binds it, in its
    real mode and in its non-CLI DRAFT dry-run mode (the schema alone is refused there)."""
    from dltrain import holdout          # only in check and dryrun
    d_sha = common.sha256_bytes(d_raw)
    stub = SimpleNamespace(path="<unconfirmed self-check>", sha256=d_sha, bytes=0, families={},
                           doc={"review_decisions_sha256": d_sha, "review_csv_sha256": csv_b.sha256,
                                "alias_sources_sha256": alias_b.sha256})
    out = {}
    for name, kw in (("holdout", {}), ("holdout_draft_dry_run", {"draft_dry_run": True})):
        try:
            holdout.review_gate(stub, d_raw, csv_b.raw, **kw)
        except holdout.HoldoutStop as err:
            out[name] = str(err)[:300]
            continue
        except Exception as err:  # noqa: BLE001 (not a refusal by holdout's rules)
            raise Refused("R12", f"holdout.review_gate self-check ({name}) raised {type(err).__name__}, not a "
                                 "HoldoutStop") from None
        raise Refused("R12", f"holdout.review_gate ({name}) accepted an unconfirmed file")
    return out


@dataclass
class Expansion:
    mode: str
    doc: dict
    body: bytes
    rows: list
    shape: dict
    shape_sha256: str
    binds: dict
    tally: dict
    validation: dict
    build_body: bytes            # what a final build reads: the file itself (real) or its rehearsal variant


def build_expansion(*, mode: str, csv_b: Bound, pre_b: Bound, alias_b: Bound, answers: dict, rights: str,
                    provenance: dict, want_shape: dict, confirmation=None) -> Expansion:
    """Everything up to the bytes, in memory: R7 binding, R8 shape (with R6 parsing), R5/R6 mapping, validation and, for
    an unconfirmed file, the R12 self-check that families refuses it."""
    pre_doc = check_bindings(csv_b, pre_b, alias_b)
    rows = parse_rows(csv_b.raw)
    shape, s_sha = require_shape(rows, want_shape, csv_b.sha256)
    decisions = expand_rows(rows, answers, rights)
    binds = {"review_csv_sha256": csv_b.sha256, "registry_pre_review_sha256": pre_b.sha256,
             "alias_sources_sha256": alias_b.sha256}
    doc = decision_doc(mode=mode, provenance=provenance, binds=binds, answers=answers, rights=rights,
                       decisions=decisions, shape_sha=s_sha, confirmation=confirmation)
    body = common.canonical(doc)
    if mode == "real":
        build_body = body
        validation = validate(body, rows, binds, pre_doc)
    else:
        build_body = common.canonical(rehearsal_doc(doc, mode, common.sha256_bytes(body)))
        validation = dict(validate(build_body, rows, binds, pre_doc), rehearsal_sha256=common.sha256_bytes(build_body),
                          unpromotable=unpromotable_families(body, rows, binds, pre_doc))
    return Expansion(mode=mode, doc=doc, body=body, rows=rows, shape=shape, shape_sha256=s_sha, binds=binds,
                     tally=tally(rows, decisions), validation=validation, build_body=build_body)


def mode_of(doc) -> str:
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if schema == families.DECISIONS_SCHEMA:
        return "real"
    for mode, s in UNCONFIRMED_SCHEMA.items():
        if schema == s:
            return mode
    raise Refused("EXPANSION", "the file is neither a decision file nor a DRAFT or what-if file this tool writes")


def rederive(d: Bound, csv_b: Bound, pre_b: Bound, alias_b: Bound) -> tuple[Expansion, dict, dict | None]:
    """check/dryrun: re-expand a file from its recorded answers and provenance; byte equality or EXPANSION."""
    doc = strict_json(d.raw, "decisions", "EXPANSION")
    if not isinstance(doc, dict) or not isinstance(doc.get("answers"), dict) or not isinstance(doc.get("expansion"), dict):
        raise Refused("EXPANSION", "the file lacks the answers/expansion blocks this tool writes")
    mode = mode_of(doc)
    rec = doc["answers"]
    if not all(isinstance(rec.get(q), str) for q in ANSWER_LETTERS):
        raise Refused("EXPANSION", "the file's answers lack Q1..Q4")
    answers = parse_answers(",".join(f"{q}={rec[q]}" for q in ANSWER_LETTERS))
    rights = parse_rights(rec.get("rights_note"))
    split, prov = rec.get("split_registry_confirmed"), doc.get("provenance")
    prov_checks, confirmation = None, None
    if mode == "draft":
        if prov != DRAFT_PROVENANCE or answers != RELAY_ANSWERS or rights != DRAFT_RIGHTS or split is not False:
            raise Refused("R9", "a DRAFT file must carry the exact DRAFT provenance, the relayed letters, rights_note "
                                "undecided and split_registry_confirmed false")
        provenance = dict(DRAFT_PROVENANCE)
    elif mode == "what_if":
        if prov != WHATIF_PROVENANCE or split is not False:
            raise Refused("R9", "a what-if file must carry the exact what-if provenance and split_registry_confirmed false")
        provenance = dict(WHATIF_PROVENANCE)
    else:
        if split is not True:
            raise Refused("R3", "a real decision file needs answers.split_registry_confirmed true")
        provenance, prov_checks, confirmation = recorded_provenance(prov, answers, rights, csv_b.sha256,
                                                                    doc.get("confirmation"))
    check_bindings(csv_b, pre_b, alias_b)          # R7 first: under code drift the shape (which binds the code) differs
    rows = parse_rows(csv_b.raw)
    shape = compute_shape(rows, csv_b.sha256)
    if doc["expansion"].get("shape_sha256") != shape_sha256(shape):
        raise Refused("R8", "the file's shape_sha256 is not the shape of the bound review.csv")
    exp = build_expansion(mode=mode, csv_b=csv_b, pre_b=pre_b, alias_b=alias_b, answers=answers, rights=rights,
                          provenance=provenance, want_shape=shape, confirmation=confirmation)
    if exp.body != d.raw:
        differ = sorted(k for k in set(doc) | set(exp.doc) if doc.get(k) != exp.doc.get(k))
        hint = ""
        if "expansion" in differ and doc["expansion"].get("tool_sha256") != exp.doc["expansion"]["tool_sha256"]:
            hint = " (expansion.tool_sha256 differs: the file came from another version of this tool)"
        raise Refused("EXPANSION", f"the file is not the mechanical expansion of its recorded answers and provenance; "
                                   f"differing top-level keys: {differ}{hint}")
    return exp, doc, prov_checks


# ------------------------------------------------------------------------------------------------- simulation (R11)
def simulate(alias_b: Bound, body: bytes, pre_b: Bound, csv_b: Bound, expect: dict, *, build_fn, guard, real: bool,
             receipts_dir) -> dict:
    """families.run_build on the exact bytes from a memfd: nothing is written. real: the CL-2 real-generation path with
    the stored owner record under receipts_dir, as `families build --out RUN/families/<gen>` runs it."""
    started = time.monotonic()
    kw = {"real": True, "receipts_dir": str(receipts_dir)} if real else {}
    fd = os.memfd_create("dl-d5-review-decisions", os.MFD_CLOEXEC)
    try:
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view):]
        res = build_fn(str(alias_b.path), f"/proc/self/fd/{fd}", guard=guard,
                       decisions_sha256=common.sha256_bytes(body), **kw)
    except (families.FamilyStop, family_pin.PinStop) as err:
        raise Refused("R11", f"the simulated final build STOPped: {str(err)[:400]}") from None
    finally:
        os.close(fd)
    summary, outputs = res.summary, res.outputs
    got = outcome_of(summary.get("counts") or {})
    final = outputs.get(families.OUT_FINAL)
    deltas = summary.get("decision_deltas") or {}
    result = {"verdict": summary.get("verdict"), "phase": summary.get("phase"), "real": real, "outcome": got,
              "expect": expect, "pre_review_equal": outputs.get(families.OUT_PRE) == pre_b.raw,
              "review_csv_equal": outputs.get(families.OUT_CSV) == csv_b.raw,
              "registry_final": None if final is None else {"sha256": common.sha256_bytes(final), "bytes": len(final)},
              "decisions_sha256": common.sha256_bytes(body), "decision_deltas": summary.get("decision_deltas"),
              "counts": {k: (summary.get("counts") or {}).get(k) for k in
                         ("families", "N_pool", "pool_contexts", "independence_qualified", "sealed",
                          "seed_exposed_families", "rights_status", "review_rows")},
              "wall_s": round(time.monotonic() - started, 3),
              "cpu_guard": guard.summary() if hasattr(guard, "summary") else None}
    problems = []
    if result["verdict"] != "PASS" or result["phase"] != "final" or final is None:
        problems.append(f"verdict {result['verdict']!r} phase {result['phase']!r} (needs PASS, final)")
    if not result["pre_review_equal"]:
        problems.append("the rebuilt registry.pre-review.json differs from the bound bytes")
    if not result["review_csv_equal"]:
        problems.append("the rebuilt review.csv differs from the bound bytes")
    prov = deltas.get("provenance") if isinstance(deltas, dict) else None
    if real and isinstance(prov, dict) and (prov.get("draft") is not False or not isinstance(prov.get("record"), dict)):
        problems.append("the simulated real build did not resolve a pinned owner record")
    diff = {k: {"expected": v, "got": got.get(k)} for k, v in sorted(expect.items()) if got.get(k) != v}
    if diff:
        problems.append(f"counts differ from --expect-outcome: {diff}")
    if problems:
        raise Refused("R11", "; ".join(problems))
    return result


# ------------------------------------------------------------------------------------------------- shape
def cmd_shape(args, *, roots: Roots, **_) -> dict:
    csv_b = read_pinned(args.review_csv, args.review_csv_sha256, "review.csv")
    rows = parse_rows(csv_b.raw)
    shape = compute_shape(rows, csv_b.sha256)
    body = common.canonical(shape)
    sha = common.sha256_bytes(body)
    expect = None
    if args.expect_shape:
        want, src = expected_shape(args.expect_shape, args.expect_shape_sha256)
        if want != shape:
            raise Refused("R8", f"review.csv shape differs from --expect-shape: {shape_diff(want, shape)}")
        expect = {"source": src, "equal": True}
    written = None
    if args.out:
        out = Path(args.out).resolve()
        allowed = {Path(roots.stage_receipts).resolve(), Path(roots.stage).resolve()}
        if out.parent not in allowed or out.name != f"dl-d5-shape-{sha[:16]}.json" or not SHAPE_NAME_RE.fullmatch(out.name):
            raise Refused("R10", "--out must be <staging receipts dir or staging dir>/dl-d5-shape-<shape sha16>.json "
                                 f"(here dl-d5-shape-{sha[:16]}.json)")
        w = common.immutable_write(out, body)
        written = {"path": w["path"], "sha256": w["sha256"], "bytes": w["bytes"], "existed": w["existed"]}
    if args.raw:
        return {"verdict": "PASS", "_raw": body}
    return {"verdict": "PASS", "review_csv": csv_b.ref(), "shape_sha256": sha, "shape": shape,
            "builtin_equal": {name: doc == shape for name, doc in sorted(BUILTIN_SHAPES.items())},
            "link_rows": shape["link_rows"], "expect": expect, "written": written}


# ------------------------------------------------------------------------------------------------- expand
def cmd_expand(args, *, roots: Roots, build_fn, guard_factory, argv, **_) -> dict:
    started = common.utc_us()
    if args.draft and args.what_if:
        raise Refused("R10", "--draft and --what-if exclude each other")
    mode = "draft" if args.draft else "what_if" if args.what_if else "real"
    # ---- static argument checks (R10, R1, R2, R4, R9 draft, R3, R11)
    if mode != "real":
        extra = [a for a in REAL_ONLY_ARGS if _present(args, a)]
        if extra:
            raise Refused("R10", f"real-mode arguments in {mode} mode: {['--' + a.replace('_', '-') for a in extra]}")
    if mode != "draft" and args.relay_sha256 is not None:
        raise Refused("R10", "--relay-sha256 is a DRAFT argument (pass --draft)")
    answers = parse_answers(args.answers)
    rights = parse_rights(args.rights)
    if mode == "draft":
        if args.relay_sha256 != RELAY_SHA256:
            raise Refused("R9", "--relay-sha256 must name the relayed DL-D5 paste b785a123...")
        if answers != RELAY_ANSWERS:
            raise Refused("R9", "a DRAFT expands only the relayed letters Q1=a,Q2=a,Q3=a,Q4=a")
        if rights != DRAFT_RIGHTS:
            raise Refused("R9", "a DRAFT adds no choice the relay did not make (D-MAP-4): the relay names no rights_note "
                                "answer, so --rights must be undecided (use --what-if for a Q5 variant)")
    if mode == "real" and not args.split_registry_confirmed:
        raise Refused("R3", "real mode needs --split-registry-confirmed: validate_decisions requires rulings."
                            "split_registry_excluded_seals_nothing to be literally true, and the owner line must state it")
    if mode == "real" and not args.simulate:
        raise Refused("R11", "real mode needs --simulate (and --expect-outcome)")
    if args.expect_outcome and not args.simulate:
        raise Refused("R11", "--expect-outcome goes only with --simulate")
    expect = parse_outcome(args.expect_outcome) if args.simulate else None
    want_shape, shape_src = expected_shape(args.expect_shape, args.expect_shape_sha256)
    if mode == "real":
        for a in ("out", "receipt"):
            if not getattr(args, a):
                raise Refused("R10", f"real mode needs --{a}")
    # ---- inputs (R7) and the review.csv the expected shape binds (R8)
    csv_b, pre_b, alias_b = read_inputs(args)
    require_shape_binds(want_shape, csv_b)
    # ---- provenance (R9)
    message, prov_checks, confirmation = None, None, None
    if mode == "real":
        provenance, prov_checks, message, confirmation = real_provenance(args, answers, rights, csv_b)
    else:
        provenance = dict(UNCONFIRMED_PROVENANCE[mode])
    # ---- the expansion in memory (R7 binding, R8 shape, R6, R5, validation, R12)
    exp = build_expansion(mode=mode, csv_b=csv_b, pre_b=pre_b, alias_b=alias_b, answers=answers, rights=rights,
                          provenance=provenance, want_shape=want_shape, confirmation=confirmation)
    sha = common.sha256_bytes(exp.body)
    # ---- output paths (R10)
    msg_path = None
    if mode != "real":
        stage = Path(roots.stage).resolve()
        if _inside(stage, roots.families):
            raise Refused("R10", "the staging dir is inside the families root")
        out = stage / f"{PREFIX[mode]}review-decisions-{sha[:16]}.json"
        receipt = Path(roots.stage_receipts).resolve() / f"{PREFIX[mode]}p2-decisions-expand-{started}.json"
    else:
        bdir = bound_dir(csv_b, pre_b)
        if not _inside(bdir, roots.families):
            raise Refused("R10", "the bound families dir is outside the families root")
        if alias_b.path.resolve() != bdir / "alias-sources.json":
            raise Refused("R10", f"real mode needs {bdir / 'alias-sources.json'}: a generation under RUN/families builds "
                                 "only from its own alias-sources.json")
        out = Path(args.out).resolve()
        if out != bdir / OUT_NAME:
            raise Refused("R10", f"--out must be {bdir / OUT_NAME} (the bound families dir)")
        receipt = Path(args.receipt).resolve()
        if receipt.parent != Path(roots.receipts).resolve() or not REAL_RECEIPT_RE.fullmatch(receipt.name):
            raise Refused("R10", f"--receipt must be {Path(roots.receipts).resolve()}/decisions-expand-<utc_us>.json")
        if message is not None:
            msg_path = Path(roots.receipts).resolve() / MESSAGE_NAME.format(sha16=message.sha256[:16])
            if msg_path.exists() and msg_path.read_bytes() != message.raw:
                raise Refused("R10", f"{msg_path} exists with different bytes")
    if out.exists() and out.read_bytes() != exp.body:
        raise Refused("R10", f"{out} exists with different bytes ({common.sha256_file(out)} != {sha})")
    if receipt.exists():
        raise Refused("R10", f"{receipt} exists (one receipt per expansion)")
    # ---- the real-generation provenance rule (families CL-2), before any write (D-MAP-5)
    cl2 = real_generation_gate(provenance, message, roots.receipts) if mode == "real" else None
    # ---- real: the owner message is stored first, so the simulated build resolves it exactly as the final build will
    msg_ref = None
    if msg_path is not None:
        w = common.immutable_write(msg_path, message.raw)
        msg_ref = {"path": w["path"], "sha256": w["sha256"], "bytes": w["bytes"], "existed": w["existed"],
                   "source": str(message.path)}
    # ---- simulation (R11), in memory
    sim = None
    if args.simulate:
        try:
            sim = simulate(alias_b, exp.build_body, pre_b, csv_b, expect, build_fn=build_fn, guard=guard_factory(),
                           real=(mode == "real"), receipts_dir=Path(roots.receipts).resolve())
        except Refused as err:
            if msg_ref is not None and not msg_ref["existed"]:
                raise Refused(err.code, f"{err.reason}; the owner message stays stored at {msg_ref['path']} (an owner "
                                        "record, not a decision)") from None
            raise
    # ---- writes: the decisions, then the receipt
    w = common.immutable_write(out, exp.body)
    out_ref = {"path": w["path"], "sha256": w["sha256"], "bytes": w["bytes"], "existed": w["existed"]}
    doc = {"schema": EXPAND_SCHEMA, "experiment_id": families.EXPERIMENT_ID, "verdict": "PASS", "mode": mode,
           "notice": NOTICE.get(mode), "answers": answers, "rights": rights,
           "split_registry_confirmed": mode == "real",
           "inputs": {"review_csv": csv_b.ref(), "registry_pre_review": pre_b.ref(), "alias_sources": alias_b.ref()},
           "code": code_shas(), "mapping": MAPPING, "mapping_table": MAPPING_TABLE,
           "shape": {"expect": shape_src, "sha256": exp.shape_sha256, "rows_digest": exp.shape["rows_digest"]},
           "tally": exp.tally, "validation": exp.validation,
           "provenance": {k: v for k, v in provenance.items() if k != "message_text"}, "provenance_checks": prov_checks,
           "confirmation": confirmation, "real_generation_gate": cl2, "owner_message": msg_ref, "output": out_ref,
           "simulation": sim, "argv": common.redact_argv([TOOL] + list(argv)), "started_utc_us": started,
           "utc_us": common.utc_us()}
    rec = common.write_receipt(receipt, doc)
    return {"verdict": "PASS", "mode": mode, "out": out_ref, "owner_message": msg_ref,
            "receipt": {k: rec[k] for k in ("path", "sha256", "bytes")}, "binds": exp.binds,
            "shape_sha256": exp.shape_sha256, "decision_counts": exp.tally["decision_counts"],
            "undecided_rows": exp.tally["undecided_rows"], "rows": exp.tally["rows"],
            "unpromotable": exp.validation.get("unpromotable"),
            "simulation": None if sim is None else {"outcome": sim["outcome"], "registry_final": sim["registry_final"],
                                                    "real": sim["real"], "wall_s": sim["wall_s"]}}


# ------------------------------------------------------------------------------------------------- check
def registry_provenance_gate(reg_doc: dict, prov: dict, receipts_dir) -> dict:
    """P3I-1 (tool side, before holdout's own rule): the final registry must come from a real-generation build of this
    owner decision file. Its decision_deltas.provenance must read draft false and simulation false, with the pinned
    owner record of this file: the stored owner message directly in the receipts dir, or the owner's
    OWNER_DIRECTIVES.md."""
    dd = reg_doc.get("decision_deltas")
    rp = dd.get("provenance") if isinstance(dd, dict) else None
    problems = []
    if not isinstance(rp, dict):
        problems.append("the registry carries no decision_deltas.provenance")
        rp = {}
    if rp.get("draft") is not False:
        problems.append("the registry's decision_deltas.provenance.draft is not false (built from a DRAFT)")
    if rp.get("simulation") is not False:
        problems.append("the registry's decision_deltas.provenance.simulation is not false")
    rec = rp.get("record")
    if prov.get("kind") in MESSAGE_KINDS:
        want = {"form": "stored_owner_message", "sha256": prov.get("message_sha256"),
                "path": Path(receipts_dir).resolve() / MESSAGE_NAME.format(sha16=str(prov.get("message_sha256"))[:16])}
    else:
        want = {"form": "owner_directive", "sha256": prov.get("file_sha256"), "path": Path(OWNER_DIRECTIVES_PATH).resolve()}
    if not (isinstance(rec, dict) and rec.get("sha256") == want["sha256"] and rec.get("form") == want["form"]
            and isinstance(rec.get("path"), str) and Path(rec["path"]).resolve() == want["path"]):
        problems.append("the registry's decision_deltas.provenance.record is not the pinned owner record of this "
                        "decision file (a staging build, or another provenance)")
    if problems:
        raise Refused("GATE", "; ".join(problems))
    return {"draft": False, "simulation": False, "record": {"form": rec["form"], "sha256": rec["sha256"],
                                                            "path": rec["path"]}}


def dryrun_code_gate(path, pin) -> dict:
    """P3I-4: a PASS rehearsal receipt whose P3 code (holdout source tree) and family code equal the current code."""
    from dltrain import holdout          # only in check and dryrun
    b = read_pinned(path, pin, "--dryrun-receipt", code="GATE")
    doc = strict_json(b.raw, "--dryrun-receipt", "GATE")
    if not isinstance(doc, dict) or doc.get("schema") != DRYRUN_SCHEMA or doc.get("verdict") != "PASS":
        raise Refused("GATE", f"--dryrun-receipt is not a PASS {DRYRUN_SCHEMA} receipt")
    have = dict(current_code_shas(), holdout_source_tree=holdout.source_tree_sha256())
    code = doc.get("code") if isinstance(doc.get("code"), dict) else {}
    differ = sorted(k for k in have if code.get(k) != have[k])
    if differ:
        raise Refused("GATE", f"the P3 rehearsal ran other code than the current code for {differ}: rehearse again")
    return {"path": str(b.path), "sha256": b.sha256, "code_equal": sorted(have)}


def cmd_check(args, *, roots: Roots, build_fn, guard_factory, **_) -> dict:
    d = read_pinned(args.decisions, args.decisions_sha256, "decisions")
    csv_b, pre_b, alias_b = read_inputs(args)
    if (args.registry is None) != (args.registry_sha256 is None):
        raise Refused("R7", "--registry and --registry-sha256 go together")
    if (args.dryrun_receipt is None) != (args.dryrun_receipt_sha256 is None):
        raise Refused("R7", "--dryrun-receipt and --dryrun-receipt-sha256 go together")
    if args.dryrun_receipt is not None and args.registry is None:
        raise Refused("R10", "--dryrun-receipt goes only with --registry")
    exp, doc, prov_checks = rederive(d, csv_b, pre_b, alias_b)
    mode, prov = exp.mode, doc.get("provenance")
    real_checked = None
    if args.real:
        if mode != "real":
            raise Refused("R9", f"check --real refuses a {'DRAFT' if mode == 'draft' else 'what-if'} file: it is never a "
                                "decision file")
        if prov.get("message_sha256") == RELAY_SHA256 or prov.get("file_sha256") == RELAY_SHA256:
            raise Refused("R9", "check --real refuses the relayed DL-D5 paste as provenance")
        bdir = bound_dir(csv_b, pre_b)
        if not _inside(bdir, roots.families):
            raise Refused("R10", "the bound families dir is outside the families root")
        if d.path.resolve() != bdir / OUT_NAME:
            raise Refused("R10", f"check --real needs the decision file at {bdir / OUT_NAME}")
        stored = None
        if prov.get("kind") in MESSAGE_KINDS:
            p = Path(roots.receipts).resolve() / MESSAGE_NAME.format(sha16=prov["message_sha256"][:16])
            if not p.is_file() or common.sha256_file(p) != prov["message_sha256"]:
                raise Refused("R9", f"the stored owner message {p} is missing or differs")
            stored = {"path": str(p), "sha256": prov["message_sha256"]}
        real_checked = {"bound_dir": str(bdir), "owner_message": stored}
    if args.registry is not None and mode != "real":
        raise Refused("R9", "an unconfirmed file never reaches the holdout gate (--registry)")
    # ---- the in-memory pre-review model: it must rebuild the bound bytes, and it must accept the decisions
    guard = guard_factory()
    try:
        res = build_fn(str(alias_b.path), None, guard=guard)
    except (families.FamilyStop, family_pin.PinStop) as err:
        raise Refused("R7", f"the in-memory pre-review build STOPped: {str(err)[:400]}") from None
    if res.outputs.get(families.OUT_PRE) != pre_b.raw or res.outputs.get(families.OUT_CSV) != csv_b.raw:
        raise Refused("R7", "the current code rebuilds different registry.pre-review.json or review.csv bytes")
    vbody = d.raw if mode == "real" else exp.build_body
    try:
        v = families.validate_decisions(vbody, res.pre.rows, review_csv_sha256=csv_b.sha256, pre_sha256=pre_b.sha256,
                                        alias_sha256=alias_b.sha256, pre=res.pre, real=bool(args.real),
                                        receipts_dir=Path(roots.receipts).resolve() if args.real else None)
    except families.FamilyStop as err:
        raise Refused("VALIDATE", f"families.validate_decisions (in-memory pre-review model): {str(err)[:300]}") from None
    undecided_breakers = [r["review_group_id"] for r in res.pre.rows
                          if r["kind"] == BREAKER_KIND and r["review_group_id"] not in v["effective"]]
    if undecided_breakers:
        raise Refused("VALIDATE", f"{len(undecided_breakers)} circuit_breaker row(s) undecided: run_build would STOP")
    unpromotable = None
    if mode != "real":
        unpromotable = dict(exp.validation["unpromotable"], **holdout_refuses(d.raw, csv_b, alias_b))
    gate = None
    if args.registry is not None:
        from dltrain import holdout          # only here (and in dryrun)
        try:
            reg = holdout.load_registry(args.registry, args.registry_sha256)
            gate = holdout.review_gate(reg, d.raw, csv_b.raw)
        except holdout.HoldoutStop as err:
            raise Refused("GATE", f"holdout review gate: {str(err)[:400]}") from None
        extra = []
        if reg.doc.get("registry_pre_review_sha256") != pre_b.sha256:
            extra.append("the registry's registry_pre_review_sha256 is not the bound pre-review registry")
        if reg.doc.get("alias_sources_sha256") != alias_b.sha256:
            extra.append("the registry's alias_sources_sha256 is not the bound alias-sources.json")
        if extra:
            raise Refused("GATE", "; ".join(extra))
        gate = dict(gate, registry={"path": str(args.registry), "sha256": reg.sha256, "bytes": reg.bytes},
                    registry_provenance=registry_provenance_gate(reg.doc, prov, roots.receipts))
        if args.dryrun_receipt is not None:
            gate["dryrun_code"] = dryrun_code_gate(args.dryrun_receipt, args.dryrun_receipt_sha256)
    return {"verdict": "PASS", "mode": mode, "real": real_checked, "decisions": d.ref(), "binds": exp.binds,
            "re_expansion_byte_equal": True, "provenance_checks": prov_checks, "pre_review_rebuilt_equal": True,
            "validate": {"of": "file" if mode == "real" else "rehearsal variant", "real": bool(args.real),
                         "effective_decisions": len(v["effective"]),
                         "undecided_rows": sum(1 for r in res.pre.rows if r["review_group_id"] not in v["effective"])},
            "unpromotable": unpromotable, "decision_counts": exp.tally["decision_counts"], "review_gate": gate,
            "cpu_guard": guard.summary() if hasattr(guard, "summary") else None}


# ------------------------------------------------------------------------------------------------- dryrun
def dryrun_tree(stage: Path, mode: str, sha: str, attempt: int) -> Path:
    return stage / f"{PREFIX[mode]}rehearsal-{sha[:16]}-a{attempt}"


def dryrun_dirs(tree: Path) -> dict:
    return {"families": tree / "families", "receipts": tree / "receipts" / "P2", "holdout": tree / "holdout"}


def families_argv(*, dirs: dict, alias: Path, decisions: str, decisions_sha256: str, utc: int) -> list:
    """families build (CLI) and publish (CLI); verify runs through the API (see cmd_dryrun)."""
    fam, rec = dirs["families"], dirs["receipts"]
    return [
        ("families.build", ["build", "--config", str(alias), "--out", str(fam), "--decisions", decisions,
                            "--decisions-sha256", decisions_sha256, "--receipt", str(rec / f"build-{utc}.json")]),
        ("families.publish", ["publish", "--out", str(fam), "--receipt", str(rec / "receipt.json"),
                              "--verify-receipt", str(rec / f"verify-{utc}.json")]),
    ]


def holdout_argv(*, dirs: dict, decisions: str, registry_sha256: str, p2_receipt_sha256: str, owner: Bound,
                 epoch: Bound, master: Bound, registry_db: Path, alias: Bound) -> list:
    """commit -> freeze -> verify --assignment-poison -> closure -> publish (closure before publish: interface gap 5).
    The verify receipt path for publish is filled in after verify ran."""
    fam, rec, hold = dirs["families"], dirs["receipts"], dirs["holdout"]
    reg = str(fam / families.OUT_FINAL)
    return [
        ("holdout.commit", ["commit", "--registry", reg, "--registry-sha256", registry_sha256,
                            "--decisions", decisions, "--review-csv", str(fam / families.OUT_CSV),
                            "--p2-receipt", str(rec / "receipt.json"), "--p2-receipt-sha256", p2_receipt_sha256,
                            "--owner-decisions", str(owner.path), "--owner-decisions-sha256", owner.sha256,
                            "--out", str(hold)]),
        ("holdout.freeze", ["freeze", "--registry", reg, "--registry-sha256", registry_sha256,
                            "--seed-file", str(hold / "seed.hex"), "--out", str(hold)]),
        ("holdout.verify", ["verify", "--holdout", str(hold / "holdout.json"), "--registry", reg,
                            "--registry-sha256", registry_sha256, "--seed-file", str(hold / "seed.hex"),
                            "--assignment-poison"]),
        ("holdout.closure", ["closure", "--holdout", str(hold / "holdout.json"), "--registry", reg,
                             "--registry-sha256", registry_sha256, "--epoch", str(epoch.path),
                             "--epoch-sha256", epoch.sha256, "--master", str(master.path),
                             "--master-sha256", master.sha256, "--registry-db", str(registry_db),
                             "--alias-sources", str(alias.path), "--alias-sources-sha256", alias.sha256,
                             "--out", str(hold)]),
        ("holdout.publish", ["publish", "--holdout", str(hold / "holdout.json"), "--verify-receipt", "<verify receipt>",
                             "--closure", str(hold / "closure.json")]),
    ]


def _last_json(*texts) -> dict:
    for text in texts:
        for line in reversed(text.strip().splitlines()):
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if isinstance(v, dict):
                return v
    return {}


def _call(fn, argv, **kw) -> tuple[int, dict]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(list(argv), **kw)
    return code, _last_json(out.getvalue(), err.getvalue())


def inventory(tree: Path) -> list:
    """Every file of the rehearsal tree: relative path, sha256 and bytes (recorded before the tree is removed)."""
    out = []
    if tree.is_dir():
        for p in sorted(tree.rglob("*")):
            if p.is_file() and not p.is_symlink():
                out.append({"path": str(p.relative_to(tree)), "sha256": common.sha256_file(p), "bytes": p.stat().st_size})
    return out


def cmd_dryrun(args, *, roots: Roots, guard_factory, argv, **_) -> dict:
    started = common.utc_us()
    # ---- refusals: nothing written
    d = read_pinned(args.decisions, args.decisions_sha256, "unconfirmed decisions")
    dpath, stage = d.path.resolve(), Path(roots.stage).resolve()
    doc = strict_json(d.raw, "unconfirmed decisions", "R9")
    mode = next((m for m, s in UNCONFIRMED_SCHEMA.items() if isinstance(doc, dict) and doc.get("schema") == s), None)
    if mode is None:
        raise Refused("R9", "dryrun rehearses only a DRAFT or what-if file (a real decision file goes through the real "
                            "S2-S6 sequence)")
    if dpath.parent != stage or not dpath.name.startswith(PREFIX[mode]):
        raise Refused("R10", f"dryrun takes only a {PREFIX[mode]}* file directly under {stage}")
    if doc.get("provenance") != UNCONFIRMED_PROVENANCE[mode]:
        raise Refused("R9", f"dryrun needs the exact {mode} provenance")
    tmp = os.environ.get("TMPDIR") or ""
    if not tmp or not stage.is_relative_to(Path(tmp).resolve()) or _inside(stage, roots.families):
        raise Refused("R10", "the staging dir must sit under TMPDIR (the write audit's allowed root)")
    csv_b, pre_b, alias_b = read_inputs(args)
    exp, _, _ = rederive(d, csv_b, pre_b, alias_b)
    owner = read_pinned(args.owner_decisions, args.owner_decisions_sha256, "owner decisions")
    epoch = read_pinned(args.epoch, args.epoch_sha256, "epoch")
    master = read_pinned(args.master, args.master_sha256, "master manifest")
    db = Path(args.registry_db or "")
    if not db.is_file():
        raise Refused("R7", "--registry-db is missing")
    expect = parse_outcome(args.expect_outcome)
    if not isinstance(args.attempt, int) or args.attempt < 1:
        raise Refused("R10", "--attempt must be >= 1")
    tree = dryrun_tree(stage, mode, d.sha256, args.attempt)
    receipt = Path(roots.stage_receipts).resolve() / f"{PREFIX[mode]}p3-dryrun-{d.sha256[:16]}-a{args.attempt}.json"
    if tree.exists() or receipt.exists():
        raise Refused("R10", f"rehearsal attempt {args.attempt} already ran (pass --attempt {args.attempt + 1})")
    dirs = dryrun_dirs(tree)
    reh_sha = common.sha256_bytes(exp.build_body)
    # ---- lazy imports: holdout is only needed here; the fake bucket fixtures bring boto3 and s3io (the tool never
    # imports either itself and reaches s3io only through the fixtures, to disable its client factory)
    from unittest import mock
    from dltrain import holdout
    from tests_p3 import fixtures_p3 as fx
    s3io = fx.s3io
    bucket = fx.FakeBucket()
    clients = fx.fake_clients(bucket)
    guard = guard_factory()
    steps, verdict, stop = [], "PASS", None
    ctx = {"rehearsal": {"sha256": reh_sha, "bytes": len(exp.build_body), "via": "memfd (never a file)"}}

    def record(name, code, summary, extra=None):
        steps.append({"step": name, "exit": code, "summary": summary, **(extra or {})})

    def fail(msg):
        raise Refused("DRYRUN", msg)

    refuse_real = AssertionError("a real S3 client was requested in the dry run")
    violations, files = [], []
    fd = os.memfd_create("dl-d5-rehearsal", os.MFD_CLOEXEC)
    try:
        view = memoryview(exp.build_body)
        while view:
            view = view[os.write(fd, view):]
        dec_path = f"/proc/self/fd/{fd}"
        with mock.patch.object(holdout, "_default_clients", side_effect=refuse_real), \
                mock.patch.object(families, "_default_publish_clients", side_effect=refuse_real), \
                mock.patch.object(s3io, "make_client", side_effect=refuse_real):
            fx.arm()
            try:
                utc = common.utc_us()
                build_step, publish_step = families_argv(dirs=dirs, alias=alias_b.path, decisions=dec_path,
                                                         decisions_sha256=reh_sha, utc=utc)
                # families build (CLI)
                guard.wait("dryrun:families.build", force=True)
                code, summary = _call(families._main, build_step[1], roots=(dirs["families"], dirs["receipts"]),
                                      clients=clients, guard=guard)
                record("families.build", code, {k: summary.get(k) for k in ("verdict", "stop", "phase", "counts",
                                                                            "decision_deltas") if k in summary},
                       {"argv": build_step[1], "fake_puts": 0})
                if code != 0 or summary.get("verdict") != "PASS":
                    fail(f"families.build did not PASS (exit {code})")
                c, outs = summary.get("counts") or {}, summary.get("outputs") or {}
                ctx["build"] = {"families": c.get("families"), "N_pool": c.get("N_pool"), "k": k_of(c.get("N_pool")),
                                "pre_review_equal": (outs.get(families.OUT_PRE) or {}).get("sha256") == pre_b.sha256,
                                "review_csv_equal": (outs.get(families.OUT_CSV) or {}).get("sha256") == csv_b.sha256,
                                "registry_final": outs.get(families.OUT_FINAL)}
                want = {k: expect[k] for k in ("families", "N_pool", "k")}
                got = {k: ctx["build"][k] for k in want}
                if got != want or not ctx["build"]["pre_review_equal"] or not ctx["build"]["review_csv_equal"]:
                    fail(f"families.build counts {got} != {want}, or the rebuilt pre-review bytes differ")
                # families verify (API): the real S3 verify scans RUN/receipts/P2, so this one does too (P3I-3)
                guard.wait("dryrun:families.verify", force=True)
                vpath = Path(publish_step[1][publish_step[1].index("--verify-receipt") + 1])
                try:
                    rep = families.verify(str(alias_b.path), dirs["families"], dec_path, guard=guard,
                                          expect=families.EXPECT, full_rehash=not args.quick_rehash,
                                          receipt_path=vpath, decisions_sha256=reh_sha,
                                          receipts_dir=Path(roots.receipts).resolve())
                    vcode = 0 if rep.get("verdict") == "PASS" else 2
                except (families.FamilyStop, family_pin.PinStop) as err:
                    rep, vcode = {"verdict": "STOP", "stop": str(err)[:300]}, 2
                canary = rep.get("canary") or {}
                record("families.verify", vcode, {"verdict": rep.get("verdict"), "phase": rep.get("phase"),
                                                  "failed": rep.get("failed"), "stop": rep.get("stop"),
                                                  "canary_scope": canary.get("scope"),
                                                  "canary_hit_files": canary.get("hit_files"),
                                                  "undecodable": len(canary.get("undecodable") or [])},
                       {"api": "families.verify", "full_rehash": not args.quick_rehash,
                        "receipts_dir": str(Path(roots.receipts).resolve()), "fake_puts": 0})
                if vcode != 0:
                    fail(f"families.verify did not PASS (failed {rep.get('failed')}): the real S3 verify would FAIL too")
                # families publish (CLI): publish reads <out>/alias-sources.json, a byte-identical copy
                w = common.immutable_write(dirs["families"] / "alias-sources.json", alias_b.raw)
                ctx["alias_copy"] = {"path": w["path"], "sha256": w["sha256"], "existed": w["existed"]}
                guard.wait("dryrun:families.publish", force=True)
                puts_before = len(bucket.puts())
                code, summary = _call(families._main, publish_step[1], roots=(dirs["families"], dirs["receipts"]),
                                      clients=clients, guard=guard)
                record("families.publish", code, {k: summary.get(k) for k in ("verdict", "stop", "phase") if k in summary},
                       {"argv": publish_step[1], "fake_puts": len(bucket.puts()) - puts_before})
                if code != 0 or summary.get("verdict") != "PASS":
                    fail(f"families.publish did not PASS (exit {code})")
                if steps[-1]["fake_puts"] != 7:
                    fail(f"families.publish made {steps[-1]['fake_puts']} fake PUTs, not 7")
                ctx["registry_sha256"] = common.sha256_file(dirs["families"] / families.OUT_FINAL)
                ctx["p2_receipt_sha256"] = common.sha256_file(dirs["receipts"] / "receipt.json")
                hsteps = holdout_argv(dirs=dirs, decisions=dec_path, registry_sha256=ctx["registry_sha256"],
                                      p2_receipt_sha256=ctx["p2_receipt_sha256"], owner=owner, epoch=epoch,
                                      master=master, registry_db=db, alias=alias_b)
                for name, hargv in hsteps:
                    if name == "holdout.publish":
                        hargv = [ctx["verify_receipt"] if a == "<verify receipt>" else a for a in hargv]
                    guard.wait(f"dryrun:{name}", force=True)
                    puts_before = len(bucket.puts())
                    # holdout's non-CLI DRAFT dry-run mode: fake clients only, families.check_provenance(real=False),
                    # never RUN/holdout, every receipt stamped draft_dry_run (holdout P3I-1/D-MAP-6)
                    code, summary = _call(holdout._main, hargv, clients=clients, draft_dry_run=True)
                    rpath = (summary.get("receipt") or {}).get("path")
                    full = json.loads(Path(rpath).read_text()) if rpath and Path(rpath).is_file() else {}
                    if full and full.get("draft_dry_run") is not True:
                        fail(f"{name}: the holdout receipt is not stamped draft_dry_run")
                    pick = {}
                    if name == "holdout.commit":
                        pick = {k: full.get(k) for k in ("pool_N", "k_expected", "s3_outcome", "seed_sha256")}
                    elif name == "holdout.freeze":
                        pick = {k: full.get(k) for k in ("N", "k", "dev", "diag", "train_fit", "holdout_sha256")}
                        pick["expect_complete"] = (full.get("expect") or {}).get("complete")
                    elif name == "holdout.verify":
                        ap = full.get("assignment_poison") or {}
                        pick = {"N": full.get("N"), "k": full.get("k"), "bytes_equal": full.get("bytes_equal"),
                                "failed_checks": full.get("failed_checks"), "assignment_poison_ran": ap.get("ran"),
                                "assignment_poison_pass": ap.get("pass"), "poison_variants": len(ap.get("variants") or [])}
                        ctx["verify_receipt"] = rpath
                    elif name == "holdout.closure":
                        cl = full.get("closure") or {}
                        pick = {"counts": full.get("counts"), "closure_sha256": cl.get("sha256"),
                                "blocking_links": len(full.get("blocking_links") or [])}
                    elif name == "holdout.publish":
                        pick = {"binding_outcome": (full.get("binding") or {}).get("outcome"),
                                "objects": [x for x in ("manifest", "campaign_manifest", "closure_object") if full.get(x)]}
                    record(name, code, {"status": full.get("status") or summary.get("status"),
                                        "stop": summary.get("stop") or summary.get("error"), **pick},
                           {"argv": hargv, "receipt": {"path": rpath,
                                                       "sha256": common.sha256_file(rpath) if rpath else None},
                            "fake_puts": len(bucket.puts()) - puts_before})
                    if code != 0 or full.get("status") != "PASS":
                        fail(f"{name} did not PASS (exit {code}, status {full.get('status') or summary.get('status')!r})")
                    if name == "holdout.commit" and (pick["pool_N"], pick["k_expected"]) != (expect["N_pool"], expect["k"]):
                        fail(f"holdout.commit pool_N/k_expected {pick['pool_N']}/{pick['k_expected']} != "
                             f"{expect['N_pool']}/{expect['k']}")
                    if name == "holdout.freeze" and (pick["N"], pick["k"]) != (expect["N_pool"], expect["k"]):
                        fail("holdout.freeze N/k differ from --expect-outcome")
                    if name == "holdout.verify" and not (pick["assignment_poison_ran"] and pick["assignment_poison_pass"]
                                                         and pick["bytes_equal"]):
                        fail("holdout.verify: assignment poison did not run and pass, or holdout bytes differ")
                    if name == "holdout.closure" and pick["blocking_links"]:
                        fail("holdout.closure reported blocking cross-family links")
                    if name == "holdout.publish" and (steps[-1]["fake_puts"] != 4 or len(pick["objects"]) != 3):
                        fail(f"holdout.publish made {steps[-1]['fake_puts']} fake PUTs (want 3 objects plus the binding)")
            except Refused as err:
                verdict, stop = "STOP", err.reason
            except Exception as err:  # noqa: BLE001 (recorded in the receipt; exit 2)
                verdict, stop = "STOP", f"{type(err).__name__} in the dry-run pipeline"
                ctx["error"] = _safe_error(err)
            finally:
                violations = fx.disarm()
    finally:
        os.close(fd)
        # D-MAP-6: the rehearsal leaves no P2 packet, registry or holdout behind (inventoried first)
        files = inventory(tree)
        _rmtree(tree)
    if violations:
        verdict, stop = "STOP", stop or "the write audit saw writes outside TMPDIR"
    calls = Counter(op for op, _ in bucket.calls)
    doc = {"schema": DRYRUN_SCHEMA, "experiment_id": families.EXPERIMENT_ID, "verdict": verdict, "stop": stop,
           "mode": mode, "notice": NOTICE[mode] + " Rehearsal with fake S3 only: the dry-run seed comes from a fake "
                                                 "VersionId and must never be reused; every output was removed after "
                                                 "this inventory.",
           "decisions": d.ref(), "rehearsal": ctx["rehearsal"],
           "inputs": {"review_csv": csv_b.ref(), "registry_pre_review": pre_b.ref(), "alias_sources": alias_b.ref(),
                      "owner_decisions": owner.ref(), "epoch": epoch.ref(), "master": master.ref(),
                      "registry_db": str(db)},
           "tree": {"path": str(tree), "removed": not tree.exists(), "files": files},
           "attempt": args.attempt, "expect": expect, "steps": steps, "context": ctx,
           "fake_s3": {"puts": bucket.puts(), "calls": dict(sorted(calls.items()))},
           "write_audit": {"allowed_root": fx.ALLOWED_ROOT, "violations": [list(v) for v in violations]},
           "code": dict(code_shas(), holdout=common.sha256_file(holdout.__file__),
                        holdout_source_tree=holdout.source_tree_sha256(), s3io=common.sha256_file(s3io.__file__),
                        common=common.sha256_file(common.__file__), fixtures_p3=common.sha256_file(fx.__file__)),
           "cpu_guard": guard.summary() if hasattr(guard, "summary") else None,
           "argv": common.redact_argv([TOOL] + list(argv)), "started_utc_us": started, "utc_us": common.utc_us()}
    rec = common.write_receipt(receipt, doc)
    return {"verdict": verdict, "stop": stop, "receipt": {k: rec[k] for k in ("path", "sha256", "bytes")},
            "steps": [{"step": s["step"], "exit": s["exit"], **s["summary"]} for s in steps],
            "fake_puts": len(bucket.puts()), "write_audit_violations": len(violations),
            "tree_removed": not tree.exists(), "tree_files": len(files)}


# ------------------------------------------------------------------------------------------------- CLI
class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refused("USAGE", message)


def _bound_args(p) -> None:
    for name in ("review-csv", "pre-review", "alias-sources"):
        p.add_argument(f"--{name}", required=True)
        p.add_argument(f"--{name}-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    ap = _Parser(prog=TOOL, description="DL-D5 owner-answer expansion (PLAN §5 P2/P3, §7.2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shape")
    s.add_argument("--review-csv", required=True)
    s.add_argument("--review-csv-sha256", required=True)
    s.add_argument("--raw", action="store_true")
    s.add_argument("--out")
    s.add_argument("--expect-shape")
    s.add_argument("--expect-shape-sha256")
    e = sub.add_parser("expand")
    _bound_args(e)
    e.add_argument("--answers")
    e.add_argument("--rights")
    e.add_argument("--expect-shape")
    e.add_argument("--expect-shape-sha256")
    e.add_argument("--simulate", action="store_true")
    e.add_argument("--expect-outcome")
    e.add_argument("--draft", action="store_true")
    e.add_argument("--what-if", action="store_true")
    e.add_argument("--relay-sha256")
    e.add_argument("--out")
    e.add_argument("--receipt")
    e.add_argument("--split-registry-confirmed", action="store_true")
    e.add_argument("--provenance-kind")
    for name in ("session", "message-utc", "message-file", "message-sha256", "question-posted-utc", "directive-file",
                 "directive-version", "directive-lines", "directive-sha256"):
        e.add_argument(f"--{name}")
    c = sub.add_parser("check")
    c.add_argument("--decisions", required=True)
    c.add_argument("--decisions-sha256", required=True)
    _bound_args(c)
    c.add_argument("--real", action="store_true")
    c.add_argument("--registry")
    c.add_argument("--registry-sha256")
    c.add_argument("--dryrun-receipt")
    c.add_argument("--dryrun-receipt-sha256")
    r = sub.add_parser("dryrun")
    r.add_argument("--decisions", required=True)
    r.add_argument("--decisions-sha256", required=True)
    _bound_args(r)
    for name in ("owner-decisions", "epoch", "master"):
        r.add_argument(f"--{name}", required=True)
        r.add_argument(f"--{name}-sha256", required=True)
    r.add_argument("--registry-db", required=True)
    r.add_argument("--expect-outcome", required=True)
    r.add_argument("--attempt", type=int, default=1)
    r.add_argument("--quick-rehash", action="store_true")
    return ap


def _safe_error(err: BaseException) -> dict:
    """Type and location only (messages of foreign exceptions could carry values); OSError adds errno and path."""
    out = {"error_type": type(err).__name__}
    tb = traceback.extract_tb(err.__traceback__)
    if tb:
        out["where"] = f"{Path(tb[-1].filename).name}:{tb[-1].lineno} {tb[-1].name}"
    if isinstance(err, OSError):
        out.update(errno=err.errno, filename=str(err.filename) if err.filename else None)
    return out


def _emit(obj, stream) -> None:
    print(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str), file=stream, flush=True)


def _main(argv=None, *, roots: Roots | None = None, build_fn=None, guard_factory=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    roots = roots or Roots()
    build_fn = build_fn or families.run_build
    guard_factory = guard_factory or family_pin.CpuGuard
    handlers = {"shape": cmd_shape, "expand": cmd_expand, "check": cmd_check, "dryrun": cmd_dryrun}
    try:
        args = _parser().parse_args(argv)
        summary = handlers[args.cmd](args, roots=roots, build_fn=build_fn, guard_factory=guard_factory, argv=argv)
    except Refused as err:
        _emit({"verdict": "STOP", "refusal": err.code, "reason": err.reason[:700]}, sys.stderr)
        return EXIT["STOP"]
    except (families.FamilyStop, family_pin.PinStop, common.ImmutableConflict) as err:
        _emit({"verdict": "STOP", "refusal": type(err).__name__, "reason": str(err)[:700]}, sys.stderr)
        return EXIT["STOP"]
    except Exception as err:  # noqa: BLE001 (exit 3; type and location only)
        _emit({"verdict": "ERROR", **_safe_error(err)}, sys.stderr)
        return EXIT["ERROR"]
    if "_raw" in summary:
        sys.stdout.write(summary["_raw"].decode("utf-8"))
        sys.stdout.flush()
        return EXIT["PASS"]
    _emit(summary, sys.stdout)
    return EXIT[summary["verdict"]]


if __name__ == "__main__":
    raise SystemExit(_main())
