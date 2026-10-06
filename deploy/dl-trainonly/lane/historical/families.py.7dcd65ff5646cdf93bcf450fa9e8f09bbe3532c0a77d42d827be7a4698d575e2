"""PLAN §5 P2 — family registry v2: build, verify and publish, plus the P2 CLI.

Spec: RUN/receipts/P2/survey/wf_a43390a0-83c.output.json result.spec (algorithm steps 2-5); PLAN §5 P2 and the §6.2
freeze() contract (registry[f]['sealed'], ['rights_status'], ['contexts'], ['frozen_fold0'].get(c)).

CLI (every command runs inside the lane slice, PLAN §2). A generation is a directory under RUN/families holding its
alias-sources.json, inputs/ and outputs together (generation 1: RUN/families; later ones e.g. RUN/families/g2):
  python -m dltrain.families pin --out <gen> [--preflight] [--ledger L] [--receipt R]
      dltrain.family_pin, the ONLY step that touches SQLite or S3 (policy = POLICY below).
  python -m dltrain.families pin --from-local <pinned gen> --from-receipt <its pin receipt> --out <new gen>
      [--withhold W --withhold-sha256 Z] [--receipt R]
      network-free re-pin (family_pin.repin_local): re-verifies every pinned file of the source generation against its
      pin receipt and S3 ledger, then writes redacted inputs and a new alias-sources.json (review fixes CL-5, SC-4, SC-3).
  python -m dltrain.families build --config <gen>/alias-sources.json --out <gen>
      [--decisions <gen>/review-decisions.json --decisions-sha256 X] [--receipt RUN/receipts/P2/<name>.json]
      [--previous-registry P --previous-registry-sha256 Y]      (post-freeze rebuilds only)
  python -m dltrain.families verify --config ... --out ... [--decisions ...] [--receipt ...] [--quick-rehash]
  python -m dltrain.families publish --out <gen> --receipt RUN/receipts/P2/receipt.json --verify-receipt V
  For a generation under RUN/families (review fix CL-2): --config must be <out>/alias-sources.json; --decisions must be
  <out>/review-decisions.json, read-only (no write bit), with --decisions-sha256; its provenance must not mark DRAFT or
  SIMULATION and must resolve to a pinned owner record. Staging directories elsewhere (TMPDIR) are not generations.
Exit codes: 0 PASS, 2 STOP/FAIL (a breaker trip writes the pre-review outputs and still exits 2), 3 error.

build is offline and deterministic. It re-hashes the pinned inputs (a mismatch is a STOP 're-pin'), reads only what pin
pinned and never touches SQLite or S3. It writes create-only (common.immutable_write):
  without --decisions: registry.pre-review.json (phase pre_review), review.csv and screen/{links.jsonl, stop-columns.json,
      sensitivity.json, coverage.json};
  with --decisions: the same pre-review outputs (rebuilt in memory; they must equal the bytes on disk and the shas the
      decision file binds) plus registry.json (phase final), which P3 pins.

Build steps (spec numbering):
  2.1 config: the pinned policy must equal POLICY (else 're-pin'); every pinned file is re-hashed.
  2.2 identities: 'item:<id>' for the frozen, incoming and sibling registry items whose dataset is in the epoch, plus
      'beyondarena:<name>' for the seed sources. Every epoch row maps to exactly one identity.
  2.3 keys: family_keys.derive() per identity (exact, context-level, source-aware keys; triggers).
  2.4 union-find over shared (kind, value) keys, iterated in sorted order; root = min identity; every union is an edge.
  2.5 screen: family_screen.near_dup_screen(); content / nds:value_strong / nds:name_jaccard links union; curated aliases
      union only when value-confirmed (a content or strong link, or a strong BeyondArena pair check), else STOP; the
      circuit breaker (BREAKER_RULE in POLICY) holds the breaker-kind edges of an over-grown component back as a
      circuit_breaker review row (the build verdict is STOP).
  2.6 family ids: 'fam-' + sha256(FID_DOMAIN + anchor).hexdigest(), anchor = min member identity at first mint. A final
      build keeps the pre-review ids (merges keep the smaller anchor's id and retire the other; splits mint new ids for
      the parts without the old anchor). A post-freeze build with --previous-registry STOPs on any merge or split.
  2.7 sealed: lineage-B v1..v5 members of SEALED_PARTITIONS (exact dataset_hash, the Rdatasets item alias, the
      metadata_sha256, an exact or source-aware key of a sealed dataset outside the epoch), the union with them, value
      or name near-duplicates of out-of-epoch sealed references, and the provisional seal of families adjacent to a
      hard-sealed family (legacy source family, spanning lineage-B audit family, a weak or stop-relaxed value link, all
      names of a small sealed reference, a shared name token, a held breaker edge); never through package, publisher,
      domain, owner or year triggers. Cross-checks STOP on failure.
  2.8 seeds: SEED192 asserts; seed_selection_exposed = the family holds a 'beyondarena:<seed source>' identity.
  2.9 exposure per encoder: Exposed (a member shares a key with the fit set), Unknown (fit set not enumerated, or a fresh
      or external member) or Unseen.
  2.10 rights; 2.11 screen coverage; 2.12 review groups with deterministic agent recommendations; 2.13 qualification;
  2.14 contexts, frozen_fold0, excluded_contexts, registry_only_contexts; 2.15 decisions; 2.16 assembly.

Disclosure: sealed members (every member of a sealed family, provisional included) and withheld members carry
display = null and value_sha256 in place of key values; trigger values of review rows that touch such a family are
written as 'sha256:<hex>'. Withheld = the alias-sources repin.withhold list, or a family that a recommended union
(a union_subset set or a circuit-breaker row) would join with a sealed family. Any other display string, key value or
trigger value that collides with a canary (normalized, word boundaries for strings shorter than 12 characters, minimum
length CANARY_MIN_LEN = 4) is withheld or hashed too. verify() scans every output, its own receipt, the summary,
alias-sources.json, every pinned local input under the config directory (inputs/**) and, from the CLI, every file under
RUN/receipts/P2 for the in-memory canaries (sealed and withheld display names, URLs, source keys, lineage-B family,
audit and legacy names); any hit FAILs. Nothing here prints anything but counts.
Threat model (review fix SC-7): value_sha256 is an unsalted sha256 of the plaintext value. It is a pseudonym that
prevents accidental display (review files, logs, receipts, published objects); it is not a secret. Low-entropy values
(URLs, Rdatasets items, legacy names) are invertible with a dictionary built from public catalogs, so a registry must
not be treated as safe to hand to an adversary.

Interpretations and deviations (each reported to the owner):
- Curated aliases are value-confirmed by a content or nds:value_strong link, or by a 'strong' BeyondArena pair check
  (family_screen step 10): a BeyondArena table is the full source and an epoch object a <= 2k-row sample, so plain
  column Jaccard cannot confirm a true copy (houses). The check never unions on its own.
- Provisional seals (DL-D12, conservative, owner-reversible): adjacency also counts a sealed Rdatasets item name
  contained as a word sequence in a non-sealed member's display name, URL basename or dataset_id (the algorithm's
  "name token or the same Rdatasets item name"; display names are never keys), and a legacy source family shared with
  a sealed dataset outside the epoch. Adjacency is computed against the hard-sealed families only (no chains).
- 'reviewed' needs at least one review row (a family no row touches was never in front of the owner); auto families
  stay auto with decisions.
- Extra row kinds: 'hf_owner' (trigger) and per-family hunt content mismatches (kind hunt_semantic_signature,
  trigger_kinds hunt_content_mismatch).
- Members list exact and source-aware union keys; context-level keys (input_raw_sha256, reuse_identity) appear as
  context_keys {count, set_sha256} to keep the registry within the P2 S3 budget; any union they cause is an edge.
- build skips the re-hash of the annotation-only v6 parquets (6.7 GB); verify re-hashes every pinned input.
- The CPU guard is consulted between stages; the screen itself (about a minute) runs as one call.
- POLICY is frozen once pinned (a change is 'policy drift: re-pin'), and the registry records the code sha256 of
  families, family_keys, family_pin and family_screen: freeze the code between the pre-review and the final build.
- publish writes the packet receipt and a <receipt>.publish.json with every S3 ref; the locked AGENT_CHAT line is left
  to the RUN-phase agent (publish returns a suggested line).

Review fixes (2026-10-05, findings CL-1..CL-6, SC-1..SC-9; POLICY gained the new seal bases, adjacency kinds and rules,
so generation 1 is re-pinned locally into a new generation):
- CL-1 recommendations: a row of an OWN_EVIDENCE_KINDS kind (legacy source family, name token, lineage-B audit family,
  PMLB stem, Socrata 4x4 across hosts) relates its families through its own trigger, which is pairwise evidence (spec
  rule: pairwise evidence -> union_subset, sets in recommendation_basis as before). A classic or semantic-signature row
  whose only evidence is its own curated match is 'unresolved', never 'independent'. A sealed_adjacent row is
  keep_sealed when its adjacency holds nds, name, held-edge or lineage-B audit evidence, or when another row recommends
  uniting its family with a hard-sealed family; otherwise unseal.
- CL-2 decisions: check_provenance() always requires a known kind and its fields, 64-hex non-degenerate hashes, an
  ISO-8601 UTC message timestamp and a message_text that hashes to message_sha256. A real generation (an output
  directory under RUN/families) refuses DRAFT/SIMULATION provenance and needs a pinned owner record (an explicit
  record {path, sha256} naming the message sha256 and timestamp, or the owner message the DL-D5 expansion stores as
  RUN/receipts/P2/dl-d5-owner-message-<sha16>.txt); an owner_directive is re-hashed and must be OWNER_DIRECTIVES.md or
  a file named in a pinned owner message. A SIMULATION_SCHEMA file never validates as a decision file;
  run_build(simulation=True) evaluates one in memory (phase 'simulation', never written).
- CL-3 rulings are applied mechanically or STOP: rdatasets_granularity 'package' turns every row whose trigger kinds
  hold rdatasets_pkg into a union (STOP on any other decision for such a row); provisional_seals 'keep' STOPs on any
  unseal ('override': each sealed_adjacent decision applies); related_class_policy 'unresolved' (also the DL-D5
  expansion's wording) STOPs on any related-class decision other than unresolved ('per_row': row decisions apply).
  decision_deltas.rulings reports ruled versus applied.
- CL-4/SC-5 breaker: the always-applied edges (every kind outside BREAKER_KINDS) are contracted into super-nodes; a
  component of super-nodes joined by breaker-kind edges trips when its merged datasets exceed BREAKER_MAX_DATASETS, and
  only the breaker-kind edges between its super-nodes are held (never a content or value_strong edge touching a
  super-node with a base-sealed member or a sealed-reference near-duplicate). A held edge between a hard-sealed family
  and another family is provisional-seal adjacency. The breaker row lists every held cross-family edge (families, kind,
  J, containment). A final build STOPs when a held content or value_strong edge spans a pool-eligible family and a
  family that trains.
- CL-5/SC-4: see family_pin.repin_local (redacted inputs); build STOPs when a sealed or withheld member's inputs were not
  redacted by the re-pin, or when the re-pin's seal floor is not reproduced.
- CL-6 split: a split drops SPLIT_DEFAULT_KINDS edges (github_repo, nds:name_jaccard) by default and nds:value_strong
  or curated_alias edges only when its edge_kinds list names them; multi_member_family rows show per-kind counts and
  the value evidence of their nds edges.
- SC-1: screen coverage counts informative non-stop columns; a sealed reference with fewer than NAME_MIN_SHARED names
  gets 'nds:name_subset' links; sealed-reference pairs strong without the stop-column exclusion are stop-relaxed weak
  links; both are provisional-seal adjacency. Legacy adjacency matches singular and plural.
- SC-2: the sealed Rdatasets join set (SK) also holds the catalog v1 and dataset-record items of every sealed dataset
  (a construction gap STOPs), and every exact or source-aware catalog key of a sealed dataset outside the epoch seals an
  identity that carries it (basis lineage_b_source_key); verify recomputes the base seal through a separate path.
- SC-3: the pre-review withholds every family a recommended union would join with a sealed family; a final build STOPs
  when a decision seals a member the pre-review showed in clear, and its STOP receipt lists the identities: re-pin them
  with --withhold into a new generation, rebuild the pre-review there and re-bind the decisions.
- SC-6: sealed_adjacent rows carry non-identifying adjacency evidence (kind, value sha256, hard families, screen
  statistics). An owner appendix with provisional members' names in clear is not produced (they are sealed members).
- SC-7: canaries of 4 characters are kept (word boundaries); bare Rdatasets item names are not scan canaries (generic
  words; sealed item names in other datasets' names already seal provisionally); value_sha256 is a pseudonym (above).
- SC-8: verify scans RUN/receipts/P2; the local re-pin pins a redacted copy of a receipts-resident input (the survey).
- SC-9: references + content_unavailable must cover S - E exactly (and stay within EXPECT when given); a sealed seed
  family's seed row trigger is hashed.

Never imports jope, gp_worker, local_gateway, reuse_cache, intake_registry or family_protocol (their rules are copied
in family_keys / family_pin), and never imports sqlite3: family_pin owns the only SQLite read.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dltrain import common
from dltrain import family_keys as fk
from dltrain import family_pin as fp
from dltrain import family_screen as fs

EXPERIMENT_ID = fp.EXPERIMENT_ID
REGISTRY_SCHEMA = "jope.dl-family-registry/v1"
DECISIONS_SCHEMA = "jope.dl-family-review-decisions/v1"
SIMULATION_SCHEMA = "jope.dl-family-review-decisions-simulation/v1"     # never a decision file (review fix CL-2)
WITHHOLD_SCHEMA = "jope.dl-family-withhold/v1"
DECISIONS_NAME = "review-decisions.json"
BUILD_RECEIPT_SCHEMA = "jope.dl-family-build-receipt/v1"
VERIFY_RECEIPT_SCHEMA = "jope.dl-family-verify-receipt/v1"
PUBLISH_SCHEMA = "jope.dl-family-publish/v1"
PACKET_RECEIPT_SCHEMA = "jope.dl-packet-receipt/v1"
MANIFEST_SCHEMA = "jope.dl-family-manifest/v1"
SCREEN_BUNDLE_SCHEMA = "jope.dl-family-screen-bundle/v1"
RUN = fp.RUN
FAMILIES_DIR = RUN / "families"
RECEIPTS_DIR = RUN / "receipts" / "P2"
OUT_PRE = "registry.pre-review.json"
OUT_FINAL = "registry.json"
OUT_SIMULATION = "registry.simulation.json"      # in memory only: build() never writes a simulation
OUT_CSV = "review.csv"
SCREEN_DIR = "screen"
EXIT = {"PASS": 0, "STOP": 2, "FAIL": 2, "ERROR": 3}

FID_DOMAIN = b"jope.dl-family-id/v1\0"
NDS_DOMAIN = b"jope.dl-family-nds/v1"
RG_DOMAIN = b"jope.dl-family-review/v1\0"

# Live-run expectations (not policy: offline tests pass their own). verify() also checks them when given.
EXPECT = {"seed_sources": 10, "seed_programs": 192, "broad_programs": 96, "epoch_sha256": fp.EPOCH_SHA256,
          "seed_set_sha256": "3ce7a0b47e76cf015751b997eaa63b631a0bd0dba65e81c11eb4e24ea99e8b61",
          "sealed_content_unavailable": ("9e1de934553cac43",)}     # S - E ids without a dataset-record (SC-9)

BREAKER_KINDS = ("curated_alias", "github_repo", "nds:name_jaccard", "nds:value_strong")
SPLITTABLE_KINDS = ("curated_alias", "github_repo", "nds:name_jaccard", "nds:value_strong")
SCREEN_UNION_KINDS = ("content", "nds:name_jaccard", "nds:value_strong")
# value links never held back by the breaker when they touch a sealed super-node (review fix CL-4)
NEVER_HELD_KINDS = ("content", "nds:value_strong")
PROVISIONAL_SEAL_BY = ("legacy_source_family", "lineage_b_audit_family", "nds:value_weak", "nds:name_subset", "name_token",
                       "held_edge")
NEVER_SEAL_BY = ("rdatasets_pkg", "publisher", "socrata_domain", "github_owner", "tidytuesday_year")
SEALED_BASES = ("lineage_b_member", "lineage_b_rdatasets_alias", "lineage_b_metadata_sha256", "lineage_b_source_key",
                "sealed_union", "value_near_dup_of_sealed_reference", "provisional_pending_review", "owner_decision")
HARD_BASES = SEALED_BASES[:6]
# kinds of the out-of-epoch sealed catalog keys that never seal (identities share them by construction or not at all)
SEALED_KEY_SKIP = ("dataset_hash", "canonical_sha256")
CANARY_MIN_LEN = 4                # review fix SC-7: 4-character canaries are kept (word boundaries)

TRIGGER_ROW_KINDS = ("lineage_b_related_class", "withdrawn_family_key", "lineage_b_audit_family", "legacy_source_family",
                     "rdatasets_pkg", "socrata_4x4_cross_host", "socrata_domain", "tidytuesday_year", "github_owner",
                     "hf_owner", "pmlb_stem", "hunt_semantic_signature", "publisher")
CANDIDATE_ROW_KINDS = ("nds_value_weak", "name_candidate", "classic_candidate")
SINGLE_ROW_KINDS = ("multi_member_family", "seed_source_mapping", "sealed_adjacent", "rights_note", "coverage_exact_only",
                    "coverage_values_only", "circuit_breaker")
ROW_KINDS = SINGLE_ROW_KINDS + TRIGGER_ROW_KINDS + CANDIDATE_ROW_KINDS
MANDATORY_KINDS = ("seed_source_mapping", "sealed_adjacent", "rights_note")
# Review fix CL-1 (code only, not POLICY): the row's own trigger relates its families (pairwise evidence) ...
OWN_EVIDENCE_KINDS = ("legacy_source_family", "name_candidate", "lineage_b_audit_family", "pmlb_stem",
                      "socrata_4x4_cross_host")
# ... or is a curated or semantic match that alone recommends 'unresolved' (never 'independent')
OWN_CANDIDATE_KINDS = ("classic_candidate", "hunt_semantic_signature")
# sealed_adjacent adjacency kinds that recommend keep_sealed (nds, name, held breaker edges, spanning audit family)
KEEP_SEALED_ADJACENCY = ("nds:value_weak", "name_token", "nds:name_subset", "lineage_b_audit_family", "held_edge")
UNION_RECOMMENDATIONS = ("union", "union_subset")
# Review fix CL-6: a split drops these edge kinds by default; the others only when the decision lists them
SPLIT_DEFAULT_KINDS = ("github_repo", "nds:name_jaccard")
_TRIGGER_OPTIONS = ("independent", "union", "union_subset", "unresolved")
DECISION_OPTIONS = {k: _TRIGGER_OPTIONS for k in TRIGGER_ROW_KINDS + CANDIDATE_ROW_KINDS}
DECISION_OPTIONS.update({
    "multi_member_family": ("confirm_union", "split", "unresolved"),
    "sealed_adjacent": ("keep_sealed", "unseal"),
    "seed_source_mapping": ("confirm_mapping", "add_members", "reject_mapping"),
    "rights_note": ("unknown", "prohibited"),
    "coverage_exact_only": ("independent", "unresolved"),
    "coverage_values_only": ("independent", "unresolved"),
    "circuit_breaker": ("union", "union_subset", "unresolved"),
})
DEFAULT_IF_UNDECIDED = {k: "unresolved" for k in ROW_KINDS}
DEFAULT_IF_UNDECIDED.update({"sealed_adjacent": "provisional_sealed", "rights_note": "unknown",
                             "seed_source_mapping": "P3 STOP"})
REVIEW_COLUMNS = ("review_group_id", "kind", "trigger_kinds", "trigger_value", "family_ids", "members", "n_families",
                  "n_datasets", "n_contexts", "n_fold0", "cohorts", "evidence", "flags", "impact_if_union",
                  "recommendation", "recommendation_basis", "default_if_undecided", "decision_options")
COVERAGE_RANK = {"full": 3, "names_only": 2, "values_only": 1, "exact_only": 0}
NAME_TOKEN_STOP = frozenset({
    "api", "blob", "content", "contents", "csv", "data", "data_csv", "datafile", "dataset", "dataset_csv", "datasets",
    "default", "dl", "doc", "docs", "download", "downloads", "export", "export_csv", "figshare", "file", "file_download",
    "files", "full", "get", "github", "html", "index", "input", "json", "kaggle", "main", "master", "openml", "output",
    "page", "pmlb", "query", "raw", "rdatasets", "readme", "record", "records", "resource", "resources", "result",
    "results", "rows", "rows_csv", "sample", "socrata", "table", "tables", "test", "testing", "train", "training", "tree",
    "uci", "values", "view", "views", "zenodo"})
NAME_TOKEN_MIN_LEN = 4
_EXT = re.compile(r"(?i)\.(csv|tsv|txt|json|jsonl|xlsx?|zip|gz|bz2|xz|html?|parquet|data|dat|arff|rda|rdata|rds|sav|dta|xml|ods)$")
_TOKEN_DROP = (re.compile(r"[0-9a-f]{8,}"), re.compile(r"[a-z0-9]{4}_[a-z0-9]{4}"), re.compile(r".*[0-9]{5,}.*"))
_SPDX = re.compile(r"[A-Za-z0-9.+\-]+")
_WHEN = re.compile(r"recommendation==([a-z_]+)")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX16 = re.compile(r"[0-9a-f]{16}")
_HEX_ID = re.compile(r"(?:sha256:)?[0-9a-f]{16}(?:[0-9a-f]{16}){0,3}")

ENCODERS = {
    "lineageA_normalizers@3df46159+6c07305f": {
        "enumerated": True, "fit": "lineage_a_split_manifest datasets[split == 'train']",
        "weights_sha256": ["3df4615971e7974b09f91d6ebcf22672e02b4900d4f71a18616dae8dc37b8ff9",
                           "6c07305fc1de841ea387fe5b01d0074b9ba386d9d15089fe2a3c10acdf623145"]},
    "lineageB_transforms@1b779c6f": {
        "enumerated": True, "fit": "lineage_b_assembly meta_training_families -> lineage_b_qualification_v4 datasets",
        "weights_sha256": ["1b779c6ff7a1a812e3001342d663353d3c7ff54eb93392a600f01813f4c66e22",
                           "eb4dce65c7fe3a72f13700ce7667183745b531744fe59229718f0c0458d834b6"]},
    "target_fused_v1@be96adc6": {
        "enumerated": False, "fit": "target_metadata fit_records (classical blocks); neural-block cache unrecoverable",
        "weights_sha256": ["be96adc6549a478c6a1759e40ce8c403fa529694b54a74a7d93a6dd56d35a40f"]},
    "hyperion_v5_arm_b@88b9e834": {
        "enumerated": False, "fit": None,
        "weights_sha256": ["88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be"]},
    "starlight@f663ae49": {
        "enumerated": False, "fit": None,
        "weights_sha256": ["f663ae49cc0f1504f07800a69dd6c31ca22fc473f26bbc0684eac25129a04128"]},
    "tabpfn35@ece4d67e": {
        "enumerated": False, "fit": None,
        "weights_sha256": ["ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"]},
}
TARGET_FIT = {"rule": "blake2b(f'{task}:{root}/{task}/{prefix}{id}'.encode(), digest_size=16).hexdigest() "
                      "(target_maker/data.py stable_hash of the resolved dataset dir; copied)",
              "root": "/home/ubuntu/quant", "tasks": ("regression", "binary"), "prefixes": ("", "synth_", ".incoming-")}

POLICY = {
    "schema": "jope.dl-family-policy/v1",
    "FID_DOMAIN": FID_DOMAIN, "NDS_DOMAIN": NDS_DOMAIN, "RG_DOMAIN": RG_DOMAIN,
    "ROUND": 5, "INFORMATIVE_MIN_DISTINCT": 12, "MINHASH_K": 64, "LSH": (16, 4), "COL_J": 0.8, "MIN_COLS": 2, "STOP_T": 5,
    "SENS_T": (10, 20), "CONTAIN": 0.5, "MIN_TUPLES": 50, "MAX_JOIN_COLS": 8, "NAME_J": 0.8, "NAME_MIN_SHARED": 3,
    "MAX_CANDIDATE_PAIRS": 5_000_000, "BREAKER_MAX_DATASETS": 10, "BREAKER_KINDS": BREAKER_KINDS,
    "SPLITTABLE_KINDS": SPLITTABLE_KINDS, "FEATURES_ONLY": True,
    "AGGREGATOR_REPOS": dict(fk.AGGREGATOR_REPOS), "PAGES_FILE_LEVEL": fk.PAGES_FILE_LEVEL, "HOSTING": fk.HOSTING,
    "FRESH_COHORTS": frozenset({"incoming"}), "SEALED_PARTITIONS": frozenset(fp.SEALED_PARTITIONS),
    "FORBIDDEN": fk.FORBIDDEN,
    "GENERIC_NAMES": frozenset({"x", "y", "id", "index", "value", "date", "time", "year", "month", "day", "target",
                                "label", "unnamed_0"}),
    "GENERIC_NAME_RE": r"^(x|v|var|col|column|feature|f)_?[0-9]+$",
    "RESTRICTIVE_NOTE": fk.RESTRICTIVE_NOTE,
    "KNOWN": "isinstance(v, str) and v.strip().lower() not in ('', 'unknown', 'none', 'null', 'n/a')",
    "KEYS_POLICY": fk.KEYS_POLICY, "KEYS_POLICY_SHA256": fk.KEYS_POLICY_SHA256,
    "SCREEN_UNION_KINDS": SCREEN_UNION_KINDS,
    "CURATED_CONFIRMATION": ("content", "nds:value_strong", "beyondarena_check:strong"),
    "PROVISIONAL_SEAL_BY": PROVISIONAL_SEAL_BY, "NEVER_SEAL_BY": NEVER_SEAL_BY, "SEALED_BASES": SEALED_BASES,
    "BREAKER_RULE": "contract every edge whose kind is not in BREAKER_KINDS into super-nodes; a component of super-nodes "
                    "joined by BREAKER_KINDS edges trips when its merged datasets exceed BREAKER_MAX_DATASETS; only the "
                    "breaker-kind edges between its super-nodes are held, never a NEVER_HELD_KINDS edge touching a "
                    "super-node with a base-sealed member or a sealed-reference near-duplicate",
    "NEVER_HELD_KINDS": NEVER_HELD_KINDS,
    "SEALED_SOURCE_KEYS": "every exact or source-aware catalog v1 key of a sealed dataset outside the epoch (kinds other "
                          "than dataset_hash and canonical_sha256) seals an identity carrying it (lineage_b_source_key)",
    "LEGACY_MATCH": "legacy source families match when equal or when one is the other plus a trailing 's'",
    "CANARY_MIN_LEN": CANARY_MIN_LEN,
    "NAME_TOKEN": {"min_len": NAME_TOKEN_MIN_LEN, "stop": NAME_TOKEN_STOP, "extensions": _EXT.pattern,
                   "drop": [p.pattern for p in _TOKEN_DROP],
                   "rule": "lower, non-alnum -> '_', strip '_', strip a leading '<digits>_'; from Rdatasets item names, "
                           "PMLB names, URL basenames, dataset_id tails and BeyondArena names; never display names"},
    "ENCODERS": ENCODERS, "TARGET_FIT": TARGET_FIT,
    "REVIEW": {"row_kinds": ROW_KINDS, "mandatory": MANDATORY_KINDS, "decision_options": DECISION_OPTIONS,
               "default_if_undecided": DEFAULT_IF_UNDECIDED, "columns": REVIEW_COLUMNS},
    "AUTO_RULE": "exactly one dataset with included contexts (canonical-equal siblings allowed), complete provenance "
                 "(canonical + normalized_url + known platform or a source-aware key + pinned meta read), screen_coverage "
                 "in {full, names_only}, no edge other than between canonical-equal members, no review row, not sealed, "
                 "not seed-exposed",
}


class FamilyStop(RuntimeError):
    """A P2 build/verify/publish STOP. Messages carry only ids, hashes, kinds and counts; detail (optional) holds ids
    for the STOP receipt (for example the identities a re-pin must withhold)."""

    def __init__(self, msg: str, detail: dict | None = None):
        super().__init__(msg)
        self.detail = detail


# ----------------------------------------------------------------------------------------------------- helpers
def _need(cond, msg: str, detail: dict | None = None) -> None:
    if not cond:
        raise FamilyStop(msg, detail)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _vsha(value) -> str:
    text = value if isinstance(value, str) else common.canonical(value).decode("utf-8")
    return _sha(text.encode("utf-8"))


def _hidden(value) -> str:
    return "sha256:" + _vsha(value)


def _known(v) -> bool:
    return fk.known(v)


def _is_hex64(v) -> bool:
    return isinstance(v, str) and bool(_HEX64.fullmatch(v))


def fid_of_anchor(anchor: str) -> str:
    return "fam-" + _sha(FID_DOMAIN + anchor.encode("utf-8"))


def _json(raw: bytes, what: str):
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise FamilyStop(f"{what}: not JSON") from None


def _jsonl(raw: bytes, what: str) -> list:
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError):
            raise FamilyStop(f"{what}: corrupt gzip") from None
    out = []
    for n, line in enumerate(raw.splitlines()):
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                raise FamilyStop(f"{what}: line {n + 1} is not JSON") from None
    return out


def _strict_json(raw: bytes, what: str):
    """Strict JSON for human-authored files: no NaN/Infinity, no duplicate keys."""
    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise FamilyStop(f"{what}: duplicate key")
        return dict(items)

    def bad_constant(name):
        raise FamilyStop(f"{what}: non-finite number {name}")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (ValueError, UnicodeDecodeError):
        raise FamilyStop(f"{what}: not strict JSON") from None


def _code_shas() -> dict:
    mods = {"families": __file__, "family_keys": fk.__file__, "family_pin": fp.__file__, "family_screen": fs.__file__}
    return {k: common.sha256_file(v) for k, v in sorted(mods.items())}


# ----------------------------------------------------------------------------------------------------- inputs
@dataclass
class Inputs:
    path: Path
    base: Path
    sha256: str
    cfg: dict
    idx: dict
    rehash: dict
    epoch_doc: dict
    rows: list
    prepared: dict
    snapshot: dict
    readiness: dict
    catalog: dict
    incoming_rows: dict
    metas: dict
    legacy: dict
    expansion: dict
    lineage_b: list
    freeze: dict
    exclusion_policy: dict
    native_sealed: set
    canonical_hsm: set
    corpus_hsm: set
    split_excluded: set
    seed_rows: list
    broad_rows: list
    pointers: dict
    seed_sources: list
    v6_names: list
    classic: dict
    split_manifest: dict
    assembly: dict
    target_meta: dict
    sealed_refs: dict
    sealed_records: dict
    skip_roles: tuple = ()
    policy_from_code: bool = False          # a local re-pin reading inputs pinned under an earlier POLICY
    pinned_policy_sha256: str | None = None


REQUIRED_INPUTS = ("epoch", "readiness", "catalog_v1", "legacy_inventory_tsv", "lineage_b_partition_freeze",
                   "lineage_b_family_exclusion_policy", "native_index", "canonical_exclusions", "corpus_audit_v2",
                   "split_registry_native", "seed_pool_recovery192", "seed_pool_broad96", "lineage_a_split_manifest",
                   "lineage_b_assembly", "target_metadata", "classic_aliases", "registry_snapshot",
                   "epoch_prepared_set") + fp.LINEAGE_B_VERSIONS
DEFAULT_SKIP_ROLES = ("v6_parquet",)


def _entry(idx: dict, name: str) -> dict:
    e = idx.get(name)
    _need(isinstance(e, dict), f"alias-sources lacks the pinned input {name} (re-pin)")
    return e


def _input_bytes(base: Path, idx: dict, name: str) -> bytes:
    """The bytes of one pinned input, re-checked against its pin (path -> sha256/bytes; local_path -> local_*)."""
    e = _entry(idx, name)
    if e.get("path"):
        raw = Path(e["path"]).read_bytes() if Path(e["path"]).is_file() else None
        _need(raw is not None, f"{name}: pinned file is missing (re-pin)")
        _need(_sha(raw) == e.get("sha256") and (e.get("bytes") is None or len(raw) == e["bytes"]),
              f"{name}: pinned file changed (re-pin)")
        return raw
    if e.get("local_path"):
        p = base / e["local_path"]
        _need(p.is_file(), f"{name}: local copy is missing (re-pin)")
        raw = p.read_bytes()
        _need(_sha(raw) == e.get("local_sha256"), f"{name}: local copy changed (re-pin)")
        return raw
    raise FamilyStop(f"{name}: no path or local_path to read")


def _project_context(ctx: dict) -> dict:
    out = {k: ctx.get(k) for k in fk.REUSE_FIELDS}
    ref = ctx.get("input_ref")
    if isinstance(ref, dict) and _is_hex64(ref.get("raw_sha256")):
        out["input_ref"] = {"raw_sha256": ref["raw_sha256"]}
    return out


def load_inputs(config_path, *, guard=None, skip_roles=DEFAULT_SKIP_ROLES, rehash: bool = True, raw: bytes | None = None,
                policy_from_code: bool = False) -> Inputs:
    """raw: the alias-sources bytes when they are not on disk yet (the local re-pin checks a new generation in memory
    before it writes alias-sources.json; config_path still names the generation directory's alias-sources.json).
    policy_from_code: read inputs pinned under an earlier POLICY with the current POLICY (local re-pin only: the pinned
    bytes do not depend on the policy; any output is then a re-pin computation, never a build of that config)."""
    guard = guard or fp.NoGuard()
    path = Path(config_path)
    if raw is None:
        _need(path.is_file(), f"config {path} is missing")
        raw = path.read_bytes()
    cfg = _json(raw, "alias-sources.json")
    _need(isinstance(cfg, dict) and cfg.get("schema") == fp.ALIAS_SOURCES_SCHEMA, "alias-sources.json: wrong schema")
    _need(cfg.get("experiment_id") == EXPERIMENT_ID, "alias-sources.json: wrong experiment_id")
    pinned_policy_sha256 = _sha(common.canonical(cfg.get("policy")))
    if policy_from_code:
        cfg = dict(cfg, policy=fp.jsonable(POLICY))
    _need(cfg.get("policy") == fp.jsonable(POLICY), "policy drift: alias-sources.json['policy'] != families.POLICY (re-pin)")
    base = path.parent
    idx = fp.input_index(cfg)
    for name in REQUIRED_INPUTS:
        _entry(idx, name)
    rh = {"checked": 0, "mismatches": [], "ok": True, "skipped_roles": list(skip_roles)}
    if rehash:
        guard.wait("rehash", force=True)
        rh = fp.rehash_inputs(cfg, base, skip_roles=tuple(skip_roles), guard=guard) | {"skipped_roles": list(skip_roles)}
        bad = rh["mismatches"]
        _need(not bad, f"re-pin: {len(bad)} pinned input(s) changed or missing: "
                       + ", ".join(sorted({b['name'] for b in bad})[:5]))
    rb = lambda name: _input_bytes(base, idx, name)  # noqa: E731
    # epoch and prepared contexts
    guard.wait("epoch", force=True)
    epoch_doc = _json(rb("epoch"), "epoch")
    rows = epoch_doc.get("contexts")
    _need(isinstance(rows, list) and rows, "epoch has no contexts")
    rows = sorted(rows, key=lambda r: r.get("context_sha256") or "")
    prepared, digest = {}, []
    for n, r in enumerate(rows):
        if n % 500 == 0:
            guard.wait("prepared")
        ref, ctx = r.get("prepared_ref") or {}, r.get("context_sha256")
        _need(_is_hex64(ctx) and ctx not in prepared, f"epoch row: context_sha256 missing or duplicated ({ctx})")
        p = Path(str(ref.get("path") or ""))
        _need(p.is_file(), f"prepared.json of {ctx} is missing (re-pin)")
        body = p.read_bytes()
        _need(_sha(body) == ref.get("sha256") and len(body) == ref.get("bytes"), f"prepared.json of {ctx} changed (re-pin)")
        doc = _json(body, f"prepared.json of {ctx}")
        c = doc.get("context") if isinstance(doc, dict) else None
        _need(isinstance(c, dict) and c.get("context_sha256") == ctx and c.get("dataset_hash") == r.get("dataset_hash"),
              f"prepared.json of {ctx} disagrees with the epoch row")
        prepared[ctx] = _project_context(c)
        digest.append([ctx, ref["sha256"], ref["bytes"]])
    _need(fp._set_sha256(digest) == _entry(idx, "epoch_prepared_set").get("sha256"),
          "epoch prepared set differs from its pin (re-pin)")
    # registry snapshot
    snap_ref = cfg.get("registry_snapshot") or {}
    snap_raw = rb("registry_snapshot")
    _need(_sha(snap_raw) == snap_ref.get("sha256"), "registry snapshot sha256 differs from alias-sources (re-pin)")
    snapshot = _json(snap_raw, "registry snapshot")
    # frozen provenance and catalog
    readiness = {}
    for rec in _json(rb("readiness"), "readiness").get("records") or []:
        if isinstance(rec, dict) and rec.get("cohort") == "frozen":
            _need(rec.get("dataset_hash") not in readiness, "readiness: duplicate frozen record")
            readiness[rec.get("dataset_hash")] = rec
    catalog = {}
    for r in _jsonl(rb("catalog_v1"), "catalog v1"):
        h = r.get("dataset_hash")
        _need(h not in catalog, "catalog v1: duplicate dataset_hash")
        catalog[h] = r
    # incoming catalog rows (plaintext from the pinned item files)
    incoming_rows = {}
    for name, e in sorted(idx.items()):
        if e.get("role") == "incoming_catalog_row":
            row = _json(rb(name), name)
            _need(isinstance(row, dict), f"{name}: not an object")
            incoming_rows[name.split(":", 1)[1]] = row
    # metas (META_KEEP extracts written by pin)
    metas = {}
    for name, e in sorted(idx.items()):
        if e.get("role") == "meta":
            sha = name.split(":", 1)[1]
            doc = _json(rb(name), name)
            _need(isinstance(doc, dict), f"{name}: not an object")
            metas[sha] = doc
    # legacy inventory and expansion families
    legacy = {}
    tsv = rb("legacy_inventory_tsv").decode("utf-8")
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t")
    _need(reader.fieldnames and "dataset_hash" in reader.fieldnames and "source_family" in reader.fieldnames,
          "legacy inventory: missing dataset_hash/source_family columns")
    for r in reader:
        h, fam = r.get("dataset_hash"), r.get("source_family")
        if isinstance(h, str) and _known(fam):
            legacy[h] = fam.strip()
    expansion = {}
    if "expansion_inventory_v7" in idx:
        for d in (_json(rb("expansion_inventory_v7"), "expansion inventory").get("datasets") or []):
            if isinstance(d, dict) and isinstance(d.get("dataset_hash"), str) and _known(d.get("family")):
                expansion[d["dataset_hash"]] = d["family"].strip()
    # lineage-B and sealed cross-check sources
    lineage_b = [(name, _json(rb(name), name)) for name in fp.LINEAGE_B_VERSIONS]
    freeze = _json(rb("lineage_b_partition_freeze"), "partition freeze")
    exclusion_policy = _json(rb("lineage_b_family_exclusion_policy"), "family exclusion policy")
    native = _json(rb("native_index"), "native index")
    sealed_parts = set(fp.SEALED_PARTITIONS)
    native_sealed = {e.get("dataset") for e in native.get("encodings") or []
                     if isinstance(e, dict) and e.get("partition") in sealed_parts}
    canonical_hsm = {r.get("dataset_hash") for r in _jsonl(rb("canonical_exclusions"), "canonical exclusions")
                     if "historical_sealed_meta_test" in (r.get("reasons") or [])}
    corpus_hsm = {r.get("dataset_hash") for r in _jsonl(rb("corpus_audit_v2"), "corpus audit")
                  if "historical_sealed_meta_test" in ((r.get("classification") or {}).get("exclusion_reasons") or [])}
    split_excluded = {r.get("dataset_hash") for r in _jsonl(rb("split_registry_native"), "split registry")
                      if (r.get("classification") or {}).get("status") == "excluded"}
    # seeds and BeyondArena
    seed_rows = _jsonl(rb("seed_pool_recovery192"), "seed pool")
    broad_rows = _jsonl(rb("seed_pool_broad96"), "broad seed pool")
    pointers = {}
    for name, e in sorted(idx.items()):
        if e.get("role") == "ba_pointer":
            pointers[name.split(":", 1)[1]] = _json(rb(name), name)
    seeds_block = cfg.get("seeds") or {}
    seed_sources = sorted(seeds_block.get("seed_sources") or [])
    _need(seed_sources, "alias-sources lists no seed sources")
    v6_names = sorted((cfg.get("beyondarena") or {}).get("v6_names") or [])
    classic = _json(rb("classic_aliases"), "classic aliases")
    _need(classic.get("schema") == fp.CLASSIC_ALIASES_SCHEMA, "classic aliases: wrong schema")
    split_manifest = _json(rb("lineage_a_split_manifest"), "lineage-A split manifest")
    assembly = _json(rb("lineage_b_assembly"), "lineage-B assembly")
    target_meta = _json(rb("target_metadata"), "target metadata")
    # sealed references (out-of-epoch v5 sealed ids with content) and their records
    sealed_refs, sealed_records = defaultdict(dict), {}
    for name, e in sorted(idx.items()):
        role = e.get("role") or ""
        if role.startswith("sealed_ref_"):
            part, h = role[len("sealed_ref_"):], e.get("dataset_hash")
            _need(isinstance(h, str) and part in ("manifest", "train", "test", "meta"), f"{name}: malformed sealed ref entry")
            if part in ("manifest", "meta"):
                sealed_refs[h][part] = _json(rb(name), name)
            else:
                sealed_refs[h][part] = e
        elif role == "sealed_dataset_record":
            sealed_records[e.get("dataset_hash")] = _json(rb(name), name)
    for h, ref in sealed_refs.items():
        _need("train" in ref and "test" in ref, f"sealed reference {h}: train/test entries missing (re-pin)")
    return Inputs(path=path, base=base, sha256=_sha(raw), cfg=cfg, idx=idx, rehash=rh, epoch_doc=epoch_doc, rows=rows,
                  prepared=prepared, snapshot=snapshot, readiness=readiness, catalog=catalog, incoming_rows=incoming_rows,
                  metas=metas, legacy=legacy, expansion=expansion, lineage_b=lineage_b, freeze=freeze,
                  exclusion_policy=exclusion_policy, native_sealed=native_sealed, canonical_hsm=canonical_hsm,
                  corpus_hsm=corpus_hsm, split_excluded=split_excluded, seed_rows=seed_rows, broad_rows=broad_rows,
                  pointers=pointers, seed_sources=seed_sources, v6_names=v6_names, classic=classic,
                  split_manifest=split_manifest, assembly=assembly, target_meta=target_meta,
                  sealed_refs=dict(sealed_refs), sealed_records=sealed_records, skip_roles=tuple(skip_roles),
                  policy_from_code=policy_from_code, pinned_policy_sha256=pinned_policy_sha256)


# ----------------------------------------------------------------------------------------------------- lineage-B
def lineage_b_sets(inp: Inputs) -> dict:
    """S, SK, MS, the v1 meta_test set, partition changes, spanning audit families and (memory only) sealed names."""
    sealed_parts = set(fp.SEALED_PARTITIONS)
    S, SK, MS, part_of, changes, prev = set(), set(), set(), {}, 0, None
    v1_meta_test, audit_parts, ds_audit = set(), defaultdict(set), defaultdict(set)
    sealed_names, partition_counts = set(), {}
    for n, (name, doc) in enumerate(inp.lineage_b):
        parent = doc.get("parent")
        psha = parent.get("sha256") if isinstance(parent, dict) else parent
        _need(psha == prev, f"{name}: parent chain broken")
        prev = _entry(inp.idx, name).get("sha256")
        fams = doc.get("families")
        fams = sorted(fams.items()) if isinstance(fams, dict) else [(None, f) for f in fams or []]
        for key, f in fams:
            if not isinstance(f, dict):
                continue
            part = f.get("partition")
            sealed = part in sealed_parts
            if sealed:
                for nm in (key, f.get("family")):
                    if isinstance(nm, str):
                        sealed_names.add(nm)
                for sk in f.get("source_keys") or []:
                    if isinstance(sk, str):
                        sealed_names.add(sk)
            for ds in f.get("datasets") or []:
                h = ds.get("dataset") if isinstance(ds, dict) else None
                _need(isinstance(h, str) and bool(_HEX16.fullmatch(h)), f"{name}: a dataset id is not 16 hex")
                if part_of.get(h, part) != part:
                    changes += 1
                part_of[h] = part
                af = ds.get("audit_family")
                if _known(af):
                    audit_parts[af.strip()].add(part)
                    ds_audit[h].add(af.strip())
                if n == 0 and part == "meta_test":
                    v1_meta_test.add(h)
                if sealed:
                    S.add(h)
                    for k in fp.lineage_b_rdatasets_keys(ds):
                        SK.add("rdatasets:" + k)
                    for text in [ds.get("source_key")] + [e.get("source_url") for e in ds.get("evidence") or []
                                                          if isinstance(e, dict)]:
                        key = fk.rdatasets_item_key(text) if isinstance(text, str) else None
                        if key:
                            SK.add(key)
                        if isinstance(text, str):
                            sealed_names.add(text)
                    if _is_hex64(ds.get("metadata_sha256")):
                        MS.add(ds["metadata_sha256"])
                    if _known(af):
                        sealed_names.add(af.strip())
        partition_counts = dict(Counter(part_of.values()))
    spanning = {a for a, parts in audit_parts.items() if parts & sealed_parts and parts - sealed_parts}
    # review fix SC-2: the sealed join set also holds every Rdatasets item that catalog v1 or the dataset-record names
    # for a sealed dataset (a lineage-B source_key may be a local path), and every exact or source-aware catalog key of a
    # sealed dataset outside the epoch is a sealing key (an identity carrying it shares a source with a sealed dataset)
    E = {r.get("dataset_hash") for r in inp.rows}
    S_out_keys, sk_sources = defaultdict(set), {"lineage_b": len(SK), "catalog_v1": 0, "dataset_record": 0}
    catalog_items = set()
    for h in sorted(S):
        row = inp.catalog.get(h)
        if isinstance(row, dict):
            try:
                keys = fk.derive_exact_and_source_aware(row)
            except (fk.ForbiddenKey, fk.MalformedKey):
                raise FamilyStop(f"sealed dataset {h}: catalog v1 keys are malformed") from None
            for k, v in keys:
                if k == "rdatasets_item":
                    catalog_items.add(v)
                    if v not in SK:
                        SK.add(v)
                        sk_sources["catalog_v1"] += 1
                if h not in E and k not in SEALED_KEY_SKIP and k not in fk.CONTEXT_KINDS:
                    S_out_keys[(k, v)].add(h)
        rec = inp.sealed_records.get(h) or {}
        key = fk.rdatasets_item_key(rec.get("source_key")) if isinstance(rec.get("source_key"), str) else None
        if key:
            catalog_items.add(key)
            if key not in SK:
                SK.add(key)
                sk_sources["dataset_record"] += 1
            if h not in E:
                S_out_keys[("rdatasets_item", key)].add(h)
        if isinstance(rec.get("source_key"), str):
            sealed_names.add(rec["source_key"])
    # review fix SC-2: STOP if a sealed catalog or record key is missing from the join set (a construction bug)
    _need(catalog_items <= SK, f"sealed join set lacks {len(catalog_items - SK)} catalog/record Rdatasets item(s) (bug)")
    for wf in inp.freeze.get("withdrawn_families") or []:
        if isinstance(wf, str):
            sealed_names.add(wf)
    related = inp.freeze.get("related_meta_test_never_clean_unseen_family_heldout") or {}
    for cls, names in sorted(related.items()) if isinstance(related, dict) else []:
        for nm in names or []:
            if isinstance(nm, str):
                sealed_names.add(nm)
    return {"S": S, "SK": SK, "MS": MS, "v1_meta_test": v1_meta_test, "partition_changes": changes, "part_of": part_of,
            "spanning_audit": spanning, "ds_audit": {h: sorted(a & spanning) for h, a in ds_audit.items() if a & spanning},
            "sealed_names": sealed_names, "partition_counts": partition_counts,
            "related_classes": sorted(related) if isinstance(related, dict) else [],
            "withdrawn": sorted(w for w in inp.freeze.get("withdrawn_families") or [] if isinstance(w, str)),
            "S_out_keys": {kv: sorted(hs) for kv, hs in S_out_keys.items()}, "sk_sources": sk_sources}


def sealed_crosschecks(inp: Inputs, lb: dict, expect: dict | None = None) -> dict:
    E = {r["dataset_hash"] for r in inp.rows}
    S = lb["S"]
    # review fix SC-9: every sealed dataset outside the epoch is either a pinned content reference or listed as
    # content_unavailable (a missing or partial dataset-records directory must never drop the screen silently)
    unavailable = (inp.cfg.get("sealed") or {}).get("content_unavailable") or []
    refs = set(inp.sealed_refs)
    _need(len(set(unavailable)) == len(unavailable) and not (refs & set(unavailable)),
          "sealed references and content_unavailable overlap or repeat (re-pin)")
    _need(refs | set(unavailable) == S - E and len(refs) + len(unavailable) == len(S - E),
          f"sealed references ({len(refs)}) + content_unavailable ({len(unavailable)}) do not cover the "
          f"{len(S - E)} sealed datasets outside the epoch (STOP: re-pin)")
    if expect and expect.get("sealed_content_unavailable") is not None:
        extra = sorted(set(unavailable) - set(expect["sealed_content_unavailable"]))
        _need(not extra, f"{len(extra)} sealed dataset(s) outside the epoch lack content beyond the expected list (STOP)")
    _need(sorted(inp.exclusion_policy.get("sealed_partitions") or []) == sorted(fp.SEALED_PARTITIONS),
          "family exclusion policy sealed_partitions differ from DL-D12")
    _need(lb["partition_changes"] == 0, f"lineage-B: {lb['partition_changes']} partition change(s) across v1..v5")
    _need(inp.native_sealed <= S, f"native index: {len(inp.native_sealed - S)} sealed id(s) outside S")
    _need(inp.canonical_hsm == lb["v1_meta_test"], "canonical-exclusions historical_sealed_meta_test != v1 meta_test")
    _need(inp.canonical_hsm <= S, "canonical-exclusions historical_sealed_meta_test is not a subset of S")
    _need(inp.corpus_hsm <= S, "corpus-audit historical_sealed_meta_test is not a subset of S")
    _need(not (inp.split_excluded & E), f"split-registry 'excluded' meets the epoch in {len(inp.split_excluded & E)} "
                                        "dataset(s): STOP for an owner ruling")
    pinned = (inp.cfg.get("sealed") or {}).get("S")
    _need(pinned is None or sorted(pinned) == sorted(S), "alias-sources sealed.S differs from the lineage-B registries")
    return {"S": len(S), "S_in_epoch": len(S & E), "S_out_of_epoch": len(S - E), "SK": len(lb["SK"]), "MS": len(lb["MS"]),
            "native_index_sealed": len(inp.native_sealed), "canonical_exclusions_hsm": len(inp.canonical_hsm),
            "corpus_audit_hsm": len(inp.corpus_hsm), "split_registry_excluded": len(inp.split_excluded),
            "split_registry_excluded_in_epoch": 0, "partition_changes": 0, "v1_meta_test": len(lb["v1_meta_test"]),
            "spanning_audit_families": len(lb["spanning_audit"]),
            "sealed_refs_with_content": len(inp.sealed_refs),
            "sealed_content_unavailable": len(unavailable),
            "sealed_join_rdatasets_items": dict(lb["sk_sources"], total=len(lb["SK"])),
            "sealed_out_of_epoch_source_keys": dict(sorted(Counter(k for k, _ in lb["S_out_keys"]).items()))}


# ----------------------------------------------------------------------------------------------------- seeds
def seed_check(inp: Inputs, expect: dict | None) -> dict:
    rows, broad = inp.seed_rows, inp.broad_rows
    _need(rows and all(isinstance(r, dict) for r in rows), "seed pool is empty or malformed")
    shas, srcs, per = [], [], defaultdict(lambda: {"broad": 0, "recovery": 0})
    for n, r in enumerate(rows):
        p = r.get("program") or {}
        sha, dsl = p.get("program_sha256"), p.get("dsl")
        _need(_is_hex64(sha) and isinstance(dsl, str), f"seed row {n}: program_sha256/dsl missing")
        _need(_sha(dsl.encode("utf-8")) == sha, f"seed row {n}: sha256(dsl) != program_sha256")
        try:
            src = r["source_evidence"][0]["source_view"].split("__")[0]
        except (KeyError, IndexError, TypeError, AttributeError):
            raise FamilyStop(f"seed row {n}: source_evidence[0].source_view missing") from None
        shas.append(sha)
        srcs.append(src)
        per[src]["broad" if n < len(broad) else "recovery"] += 1
    _need(len(set(shas)) == len(shas), "seed pool: program_sha256 values are not distinct")
    _need(rows[:len(broad)] == broad, "seed pool: the first rows are not the broad seed pool")
    _need(not (set(shas[:len(broad)]) & set(shas[len(broad):])), "seed pool: broad and recovery halves overlap")
    set_sha = _sha(common.canonical(sorted(shas)))
    _need(sorted(set(srcs)) == inp.seed_sources, "seed pool sources differ from the pinned seed sources")
    if expect:
        _need(len(rows) == expect["seed_programs"], f"seed pool holds {len(rows)} programs, not {expect['seed_programs']}")
        _need(len(broad) == expect["broad_programs"], f"broad seed pool holds {len(broad)} programs")
        _need(set_sha == expect["seed_set_sha256"], "SEED192 canonical set sha256 differs")
        _need(len(inp.seed_sources) == expect["seed_sources"], f"{len(inp.seed_sources)} seed sources")
    return {"programs": len(rows), "broad": len(broad), "set_sha256": set_sha, "source_of": dict(zip(shas, srcs)),
            "per_source": {k: dict(v) for k, v in sorted(per.items())}}


# ----------------------------------------------------------------------------------------------------- identities
@dataclass
class Ident:
    identity: str
    kind: str
    role: str
    dataset_hash: str | None = None
    canonical: str | None = None
    sample: str | None = None
    item_id: str | None = None
    cohort: str = "external"
    epoch_status: str = "external"
    registry_state: str | None = None
    rows: list = field(default_factory=list)
    row: dict | None = None
    display_name: str | None = None
    platform: str | None = None
    url: str | None = None
    license: dict | None = None
    meta_sha256: str | None = None
    meta: dict | None = None
    U: list = field(default_factory=list)
    T: list = field(default_factory=list)
    tokens: frozenset = frozenset()
    input_names: list = field(default_factory=list)
    reg_contexts: list = field(default_factory=list)


def _str(v):
    return v.strip() if isinstance(v, str) and _known(v) else None


def make_identities(inp: Inputs) -> dict:
    items = inp.snapshot.get("items")
    _need(isinstance(items, list) and items, "registry snapshot has no items")
    rows_frozen, rows_item, cohort_of = defaultdict(list), defaultdict(list), {}
    for r in inp.rows:
        h, c = r.get("dataset_hash"), r.get("cohort")
        _need(c in ("frozen", "incoming"), f"epoch row {r.get('context_sha256')}: cohort {c!r}")
        _need(cohort_of.setdefault(h, c) == c, f"dataset {h} appears in both cohorts")
        if c == "frozen":
            rows_frozen[h].append(r)
        else:
            _need(_is_hex64(r.get("item_id")), f"incoming row {r.get('context_sha256')} lacks item_id")
            rows_item[r["item_id"]].append(r)
    reg_ctx = defaultdict(list)
    for pair in inp.snapshot.get("contexts") or []:
        if isinstance(pair, list) and len(pair) == 2:
            reg_ctx[pair[1]].append(pair[0])
    idents, frozen_seen = {}, set()
    for it in sorted(items, key=lambda x: str(x.get("id"))):
        iid, h = it.get("id"), it.get("dataset")
        _need(_is_hex64(iid) and isinstance(h, str) and _is_hex64(it.get("canonical")) and _is_hex64(it.get("sample")),
              f"registry snapshot item {iid}: malformed identity fields")
        _need(iid == fp.identity_key(h, it["canonical"], it["sample"]) and h == it["canonical"][:16],
              f"registry item {iid}: identity rule broken")
        role = it.get("epoch_role")
        I = Ident(identity="item:" + iid, kind="registry_item", role=role, dataset_hash=h, canonical=it["canonical"],
                  sample=it["sample"], registry_state=it.get("state"), cohort=cohort_of.get(h))
        _need(I.cohort in ("frozen", "incoming"), f"registry item {iid}: dataset {h} is not an epoch dataset")
        I.reg_contexts = sorted(reg_ctx.get(iid, []))
        if role == "frozen":
            _need(I.cohort == "frozen" and h not in frozen_seen, f"frozen dataset {h}: zero or several frozen items")
            frozen_seen.add(h)
            I.rows = rows_frozen.get(h, [])
            I.row = inp.catalog.get(h)
            rec = inp.readiness.get(h)
            _need(isinstance(I.row, dict) and isinstance(rec, dict), f"frozen dataset {h}: no catalog v1 row or readiness record")
            _need(I.row.get("canonical_sha256") == I.canonical, f"frozen dataset {h}: catalog canonical differs")
            I.display_name = _str(rec.get("display_name")) or _str(I.row.get("display_name"))
            I.input_names = [f"frozen_manifest:{h}", f"frozen_data:{h}", f"dataset_record:{h}"]
        elif role == "incoming":
            _need(I.cohort == "incoming", f"incoming item {iid}: dataset {h} is not an incoming epoch dataset")
            I.item_id = iid
            I.rows = rows_item.get(iid, [])
            _need(I.rows, f"incoming item {iid} has no epoch rows")
            I.row = inp.incoming_rows.get(iid)
            _need(isinstance(I.row, dict), f"incoming item {iid}: pinned catalog-row.json missing")
            _need(I.row.get("dataset_hash") == h and I.row.get("canonical_sha256") == I.canonical,
                  f"incoming item {iid}: catalog row identity differs")
            I.display_name = _str(I.row.get("display_name"))
            I.input_names = [f"incoming_catalog_row:{iid}", f"incoming_materialization:{iid}", f"incoming_data:{iid}",
                             f"incoming_manifest:{iid}"]
        elif role == "sibling":
            I.item_id = iid
            row = (it.get("payload") or {}).get("catalog_row")
            I.row = row if isinstance(row, dict) else None
            I.display_name = _str((I.row or {}).get("display_name"))
        else:
            raise FamilyStop(f"registry item {iid}: unknown epoch_role {role!r}")
        prov = (I.row or {}).get("provenance") if isinstance((I.row or {}).get("provenance"), dict) else {}
        I.platform, I.url = _str(prov.get("platform")), _str(prov.get("url"))
        I.license = prov.get("license") if isinstance(prov.get("license"), dict) else None
        meta = (((I.row or {}).get("files") or {}).get("meta") or {})
        I.meta_sha256 = meta.get("sha256") if _is_hex64(meta.get("sha256")) else None
        I.meta = inp.metas.get(I.meta_sha256) if I.meta_sha256 else None
        if I.meta_sha256:
            I.input_names.append("meta:" + I.meta_sha256)
        if role == "sibling":
            I.epoch_status = "registry_sibling"
        else:
            I.epoch_status = "included" if any(r.get("status") == "included" for r in I.rows) else "withdrawn_only"
        idents[I.identity] = I
    _need(frozen_seen == set(rows_frozen), f"{len(set(rows_frozen) - frozen_seen)} frozen dataset(s) lack a frozen item")
    for iid in rows_item:
        _need(("item:" + iid) in idents and idents["item:" + iid].role == "incoming", f"epoch item {iid} is not mapped")
    for name in inp.seed_sources:
        ptr = inp.pointers.get(name)
        _need(isinstance(ptr, dict), f"seed source {name}: no pinned BeyondArena pointer")
        I = Ident(identity="beyondarena:" + name, kind="beyondarena", role="beyondarena", row=ptr, display_name=name,
                  platform="beyondarena", url=_str(ptr.get("original_source")),
                  license={"status": "external", "note": ptr.get("license")},
                  input_names=[f"ba_pointer:{name}", f"ba_parquet:{name}"])
        idents[I.identity] = I
    mapped = Counter(r["context_sha256"] for I in idents.values() for r in I.rows)
    _need(sum(mapped.values()) == len(inp.rows) and len(mapped) == len(inp.rows), "an epoch row is unmapped or mapped twice")
    return idents


def _norm_token(text: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"^[0-9]+_", "", t)


def _valid_token(t: str) -> bool:
    return (len(t) >= NAME_TOKEN_MIN_LEN and not t.isdigit() and t not in NAME_TOKEN_STOP
            and not any(p.fullmatch(t) for p in _TOKEN_DROP))


def _basename_token(text) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip().split("?")[0].split("#")[0].rstrip("/")
    s = s.rsplit("/", 1)[-1]
    for _ in range(3):
        s2 = _EXT.sub("", s)
        if s2 == s:
            break
        s = s2
    t = _norm_token(s)
    return t if _valid_token(t) else None


def name_tokens(I: Ident) -> frozenset:
    """Name tokens (never display names): Rdatasets item names, PMLB names, URL basenames, dataset_id tails, BA names."""
    toks = set()
    for kind, v in I.U:
        if kind == "rdatasets_item":
            toks.add(_norm_token(v.split("/", 1)[-1]))
        elif kind == "pmlb_name":
            toks.add(_norm_token(v[len("pmlb:"):]))
    prov = (I.row or {}).get("provenance") if isinstance((I.row or {}).get("provenance"), dict) else {}
    for text in (I.url, _str(prov.get("dataset_id"))):
        t = _basename_token(text.replace(":", "/")) if isinstance(text, str) else None
        if t:
            toks.add(t)
    if I.kind == "beyondarena":
        toks.add(_norm_token(I.identity.split(":", 1)[1]))
    return frozenset(t for t in toks if _valid_token(t))


def _words(text) -> list:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w] if isinstance(text, str) else []


def _contained_words(I: Ident, needles: list) -> list:
    """Sealed item names (word tuples) contained as a contiguous word sequence in the member's display name, URL basename
    or dataset_id. Used only for the provisional seal (a review trigger); never a union key."""
    if I.kind != "registry_item" or not needles:
        return []
    prov = (I.row or {}).get("provenance") if isinstance((I.row or {}).get("provenance"), dict) else {}
    hays = [_words(I.display_name), _words((I.url or "").split("?")[0].rstrip("/").rsplit("/", 1)[-1]),
            _words(_str(prov.get("dataset_id")))]
    out = []
    for n in needles:
        k = len(n)
        if any(h[i:i + k] == list(n) for h in hays for i in range(len(h) - k + 1)):
            out.append(n)
    return out


def _related_classes(legacy, classes) -> list:
    if not _known(legacy):
        return []
    lf = fk.norm(legacy)
    out = []
    for c in classes:
        cn = fk.norm(c)
        if lf in (cn, cn + "s") or lf + "s" == cn:
            out.append(c)
    return sorted(out)


def derive_identity_keys(inp: Inputs, idents: dict, lb: dict) -> None:
    for i in sorted(idents):
        I = idents[i]
        if I.kind == "beyondarena":
            attrs = fk.attrs_from_beyondarena_pointer(I.row)
        else:
            attrs = fk.attrs_from_catalog_row(I.row) if isinstance(I.row, dict) else {}
            attrs["dataset_hash"] = I.dataset_hash
            attrs["canonical_sha256"] = I.canonical
            if I.meta is not None:
                attrs["meta"] = I.meta
            ctxs = [inp.prepared[r["context_sha256"]] for r in sorted(I.rows, key=lambda r: r["context_sha256"])]
            if ctxs:
                attrs["contexts"] = ctxs
            legacy = inp.legacy.get(I.dataset_hash)
            if _known(legacy):
                attrs["legacy_source_family"] = legacy
            audit = lb["ds_audit"].get(I.dataset_hash)
            if audit:
                attrs["lineage_b_audit_family"] = audit
            classes = _related_classes(legacy, lb["related_classes"])
            if classes:
                attrs["lineage_b_related_class"] = classes
        try:
            I.U, I.T = fk.derive(attrs)
        except fk.ForbiddenKey:
            raise FamilyStop(f"{i}: a FORBIDDEN field reached derive() (bug)") from None
        except fk.MalformedKey:
            raise FamilyStop(f"{i}: contradictory key values (canonical, meta or context keys)") from None
        I.tokens = name_tokens(I)


# ----------------------------------------------------------------------------------------------------- edges
@dataclass(frozen=True, order=True)
class Edge:
    kind: str
    a: str
    b: str
    value: str
    evidence: str = ""          # canonical JSON text (immutable), '' when none

    def ev(self):
        return json.loads(self.evidence) if self.evidence else None


def _edge(a: str, b: str, kind: str, value: str, evidence=None) -> Edge:
    a, b = sorted((a, b))
    return Edge(kind, a, b, value, common.canonical(evidence).decode("utf-8") if evidence is not None else "")


def key_edges(idents: dict) -> list:
    inv = defaultdict(set)
    for i in sorted(idents):
        for kv in idents[i].U:
            inv[kv].add(i)
    edges = []
    for (kind, value), ids in sorted(inv.items()):
        if len(ids) < 2:
            continue
        try:
            fk.check_edge(kind, value)
        except fk.ForbiddenKey:
            raise FamilyStop(f"a union through a forbidden or null key of kind {kind!r} (bug)") from None
        ids = sorted(ids)
        for b in ids[1:]:
            edges.append(_edge(ids[0], b, kind, value))
    return edges


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if rb < ra:
            ra, rb = rb, ra
        self.p[rb] = ra
        return True

    def components(self) -> list:
        groups = defaultdict(list)
        for x in self.p:
            groups[self.find(x)].append(x)
        return sorted(sorted(g) for g in groups.values())


def components(items, edges) -> list:
    uf = _UF(sorted(items))
    for e in sorted(edges):
        uf.union(e.a, e.b)
    return uf.components()


# ----------------------------------------------------------------------------------------------------- screen
def _sealed_ref_table(inp: Inputs, h: str) -> fs.Table:
    ref = inp.sealed_refs[h]
    mats, src = [], {}
    for part in ("train", "test"):
        e = ref[part]
        raw = (inp.base / e["local_path"]).read_bytes()
        _need(_sha(raw) == e.get("local_sha256") == e.get("sha256"), f"sealed reference {h} {part}: bytes changed (re-pin)")
        if e.get("has_header"):
            data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
            raw = data.split(b"\n", 1)[1] if b"\n" in data else b""
        try:
            mats.append(fs.parse_csv_matrix(raw, f"sealedref:{h} {part}"))
        except fs.ScreenInputError:
            raise FamilyStop(f"sealed reference {h} {part}: not a numeric CSV") from None
        src[part] = {"local_path": e["local_path"], "sha256": e["sha256"], "bytes": e.get("bytes")}
    _need(mats[0].shape[1] == mats[1].shape[1] and mats[0].shape[1] >= 2, f"sealed reference {h}: train/test columns differ")
    import numpy as np
    return fs.make_table("sealedref:" + h, "sealed_ref", np.vstack(mats)[:, :-1], source=src)


def run_screen(inp: Inputs, idents: dict) -> fs.ScreenResult:
    tables, names, metas = [], {}, {}
    try:
        for i in sorted(idents):
            I = idents[i]
            if I.kind == "beyondarena":
                n = i.split(":", 1)[1]
                e = _entry(inp.idx, f"ba_parquet:{n}")
                tables.append(fs.load_beyondarena(i, e["path"], e.get("target"), expected_sha256=e["sha256"],
                                                  expected_bytes=e.get("bytes")))
            elif I.role in ("frozen", "incoming"):
                name = f"frozen_data:{I.dataset_hash}" if I.role == "frozen" else f"incoming_data:{I.item_id}"
                e = _entry(inp.idx, name)
                tables.append(fs.load_data_object(i, e["path"], expected_sha256=e["sha256"], expected_bytes=e.get("bytes")))
                if I.meta is not None:
                    names[i] = fs.names_from_meta(I.meta)
                    metas[i] = I.meta
        for h in sorted(inp.sealed_refs):
            tables.append(_sealed_ref_table(inp, h))
            m = inp.sealed_refs[h].get("meta")
            if isinstance(m, dict):
                names["sealedref:" + h] = fs.names_from_meta(m)
        ba_pairs = []
        for entry in curated_union_entries(inp):
            ba = "beyondarena:" + entry["name"]
            for m in entry["members"]:
                j = _member_identity(idents, m)
                if ba in idents and j is not None:
                    ba_pairs.append((ba, j))
        return fs.near_dup_screen(tables, names, policy=inp.cfg["policy"], metas=metas, ba_pairs=ba_pairs)
    except fs.ScreenLimitError:
        raise FamilyStop("screen: LSH candidates exceed MAX_CANDIDATE_PAIRS (STOP for review)") from None
    except fs.ScreenInputError as err:
        raise FamilyStop(f"screen input: {str(err)[:200]}") from None


def curated_union_entries(inp: Inputs) -> list:
    out = []
    for e in inp.classic.get("entries") or []:
        if isinstance(e, dict) and e.get("action") == "union":
            _need(e.get("name") in inp.seed_sources, f"classic alias {e.get('name')}: a union entry must name a seed source")
            out.append({"name": e["name"], "members": sorted(e.get("members") or [])})
    return sorted(out, key=lambda e: e["name"])


def _member_identity(idents: dict, dataset_hash: str):
    cands = sorted(i for i, I in idents.items() if I.dataset_hash == dataset_hash and I.role in ("frozen", "incoming"))
    inc = [i for i in cands if idents[i].epoch_status == "included"]
    return (inc or cands or [None])[0]


def split_links(scr: fs.ScreenResult, idents: dict, redact: set) -> dict:
    """Screen links between identities (union or weak) and links to sealed references (by identity): sref (content,
    strong or name link: hard seal), sref_weak (weak or stop-relaxed value link) and sref_subset (all names of a sealed
    reference with fewer than NAME_MIN_SHARED names, review fix SC-1): provisional-seal adjacency."""
    union, weak, sref, sref_weak, sref_subset = [], [], defaultdict(list), defaultdict(list), defaultdict(list)
    red = {(r["a"], r["b"], r["kind"]): r for r in scr.redacted_links(redact)}
    for r in scr.links():
        a, b, kind = r["a"], r["b"], r["kind"]
        ev = red[(a, b, kind)]["evidence"]
        ia, ib = a in idents, b in idents
        if ia and ib:
            if kind in SCREEN_UNION_KINDS:
                value = r.get("value") or ("sha256:" + _sha(common.canonical([a, b, kind, ev])))
                union.append(_edge(a, b, kind, value, {"screen": ev}))
            elif kind == "nds:value_weak":
                weak.append({"a": a, "b": b, "evidence": ev})
        elif ia != ib:
            i, other = (a, b) if ia else (b, a)
            if not other.startswith("sealedref:"):
                continue
            if kind in SCREEN_UNION_KINDS:
                sref[i].append({"kind": kind, "ref": other, "evidence": ev})
            elif kind == "nds:value_weak":
                sref_weak[i].append({"ref": other, "evidence": ev})
            elif kind == "nds:name_subset":
                sref_subset[i].append({"ref": other, "evidence": ev})
    return {"union": sorted(union), "weak": sorted(weak, key=lambda w: (w["a"], w["b"])), "sref": dict(sref),
            "sref_weak": dict(sref_weak), "sref_subset": dict(sref_subset)}


def curated_edges(inp: Inputs, idents: dict, scr: fs.ScreenResult, union_links: list) -> tuple[list, dict]:
    strong = {(e.a, e.b): e for e in union_links if e.kind in ("content", "nds:value_strong")}
    edges, evidence = [], {}
    for entry in curated_union_entries(inp):
        n = entry["name"]
        ba = "beyondarena:" + n
        _need(ba in idents, f"classic alias {n}: its seed-source identity is missing")
        conf = []
        for m in entry["members"]:
            j = _member_identity(idents, m)
            _need(j is not None, f"classic alias {n}: member {m} is not an epoch dataset")
            a, b = sorted((ba, j))
            via = None
            if (a, b) in strong:
                e = strong[(a, b)]
                via = {"member": m, "identity": j, "via": e.kind, "screen": (e.ev() or {}).get("screen")}
            else:
                chk = scr.beyondarena_check(a, b)
                if chk and chk.get("class") == "strong":
                    via = {"member": m, "identity": j, "via": "beyondarena_check",
                           "scaled": {k: (chk.get("scaled") or {}).get(k) for k in ("class", "containment", "n_mapped")},
                           "raw": {k: (chk.get("raw") or {}).get(k) for k in ("status", "class", "containment", "n_mapped")}}
            _need(via is not None, f"classic alias {n}: member {m} has no strong or content screen link (STOP)")
            conf.append(via)
            edges.append(_edge(ba, j, "curated_alias", n, {"curated": n, "confirmation": via}))
        evidence[n] = {"members": entry["members"], "confirmations": conf}
    return edges, evidence


def breaker(edges: list, idents: dict, max_datasets: int, never_hold: frozenset = frozenset()) -> tuple[set, list]:
    """Circuit breaker (review fix CL-4). The edges that always apply (every kind outside BREAKER_KINDS: exact,
    same-source and content keys) are contracted into super-nodes. A component of super-nodes joined by breaker-kind
    edges trips when the datasets of its merged super-nodes exceed max_datasets; then the breaker-kind edges between
    different super-nodes of that component are held back. Breaker-kind edges inside one super-node are redundant and
    never count or hold. A NEVER_HELD_KINDS edge touching a super-node that holds a never_hold identity (base-sealed, or
    a near-duplicate of a sealed reference) is never held: it seals what it reaches."""
    sup_of = {}
    for n, comp in enumerate(components(idents, [e for e in edges if e.kind not in BREAKER_KINDS])):
        for i in comp:
            sup_of[i] = n
    cross = sorted(e for e in edges if e.kind in BREAKER_KINDS and sup_of[e.a] != sup_of[e.b])
    members = defaultdict(list)
    for i, n in sup_of.items():
        members[n].append(i)
    sealed_sup = {sup_of[i] for i in never_hold if i in sup_of}
    uf = _UF(sorted({sup_of[x] for e in cross for x in (e.a, e.b)}))
    for e in cross:
        uf.union(sup_of[e.a], sup_of[e.b])
    held, trips = set(), []
    for comp in uf.components():
        ids = sorted(i for n in comp for i in members[n])
        units = {idents[i].dataset_hash or i for i in ids}
        if len(units) <= max_datasets:
            continue
        cs = set(comp)
        es = [e for e in cross if sup_of[e.a] in cs]
        kept = [e for e in es if e.kind in NEVER_HELD_KINDS and (sup_of[e.a] in sealed_sup or sup_of[e.b] in sealed_sup)]
        hold = sorted(set(es) - set(kept))
        held |= set(hold)
        sizes = sorted((len({idents[i].dataset_hash or i for i in members[n]}) for n in comp), reverse=True)
        trips.append({"identities": ids, "datasets": len(units), "edges": hold, "kept_sealed_edges": sorted(kept),
                      "supernode_datasets": sizes})
    return held, trips


# ----------------------------------------------------------------------------------------------------- exposure
def target_fit_id(task: str, name: str, root: str = TARGET_FIT["root"]) -> str:
    return hashlib.blake2b(f"{task}:{root}/{task}/{name}".encode("utf-8"), digest_size=16).hexdigest()


def fit_sets(inp: Inputs, lb: dict) -> dict:
    E = {r["dataset_hash"] for r in inp.rows}
    sm = inp.split_manifest.get("datasets")
    _need(isinstance(sm, dict) and sm, "lineage-A split manifest has no datasets")
    a_ids = sorted(h for h, d in sm.items() if isinstance(d, dict) and d.get("split") == "train")
    a_val = sorted(h for h, d in sm.items() if isinstance(d, dict) and d.get("split") == "validation")
    mtf = inp.assembly.get("meta_training_families")
    _need(isinstance(mtf, (dict, list)) and mtf, "lineage-B assembly lacks meta_training_families")
    fam_names = sorted(mtf) if isinstance(mtf, dict) else sorted(mtf)
    v4 = dict(inp.lineage_b)["lineage_b_qualification_v4"].get("families") or {}
    b_ids = set()
    for f in fam_names:
        _need(isinstance(v4, dict) and f in v4, "lineage-B assembly names a family absent from qualification v4")
        for d in v4[f].get("datasets") or []:
            b_ids.add(d.get("dataset"))
    records = inp.target_meta.get("fit_records") or []
    cand = sorted(set(inp.catalog) | E)
    resolved = {}
    for task in TARGET_FIT["tasks"]:
        for prefix in TARGET_FIT["prefixes"]:
            for c in cand:
                rid = target_fit_id(task, prefix + c)
                if rid in records and rid not in resolved:
                    resolved[rid] = {"task": task, "prefix": prefix, "dataset_hash": c}
    t_ids = sorted({v["dataset_hash"] for v in resolved.values() if v["prefix"] != "synth_"})
    out = {}
    for enc, spec in ENCODERS.items():
        if enc.startswith("lineageA"):
            ids = a_ids
        elif enc.startswith("lineageB"):
            ids = sorted(b_ids)
        elif enc.startswith("target"):
            ids = t_ids
        else:
            ids = []
        keys = defaultdict(set)
        for fid in ids:
            row = inp.catalog.get(fid)
            try:
                ks = fk.derive_exact_and_source_aware(row) if isinstance(row, dict) else fk.derive({"dataset_hash": fid})[0]
            except (fk.ForbiddenKey, fk.MalformedKey):
                raise FamilyStop(f"fit set {enc}: keys of fit id {fid} are malformed") from None
            for kv in ks:
                keys[kv].add(fid)
        out[enc] = {"enumerated": spec["enumerated"], "ids": ids, "keys": {k: sorted(v) for k, v in keys.items()}}
    out["_annotations"] = {"lineage_a_validation": a_val,
                           "target_resolved": {k: resolved[k] for k in sorted(resolved)},
                           "target_unresolved": sorted(set(records) - set(resolved))}
    return out


# ----------------------------------------------------------------------------------------------------- base
@dataclass
class Base:
    inp: Inputs
    idents: dict
    lb: dict
    crosscheck: dict
    seeds: dict
    scr: fs.ScreenResult
    links: dict
    key_edges: list
    curated: list
    curated_evidence: dict
    held: set
    trips: list
    applied: list
    base_sealed: dict
    fits: dict
    ext: dict
    redact: set
    withhold: frozenset = frozenset()       # identities shown as sealed in every output (re-pin --withhold, SC-3)
    seal_floor: frozenset | None = None     # identities the local re-pin found sealed or withheld (alias-sources)
    redacted_members: frozenset | None = None   # identities whose pinned inputs the local re-pin redacted


def member_bases(I: Ident, lb: dict) -> list:
    if I.kind != "registry_item":
        return []
    out = []
    if I.dataset_hash in lb["S"]:
        out.append("lineage_b_member")
    if any(k == "rdatasets_item" and v in lb["SK"] for k, v in I.U):
        out.append("lineage_b_rdatasets_alias")
    if any(k == "lineage_b_metadata_sha256" and v in lb["MS"] for k, v in I.U):
        out.append("lineage_b_metadata_sha256")
    # review fix SC-2: an exact or source-aware key shared with a sealed dataset outside the epoch
    if any(kv in lb.get("S_out_keys", {}) for kv in I.U):
        out.append("lineage_b_source_key")
    return out


def external_alias_keys(inp: Inputs, lb: dict, fits: dict) -> dict:
    """(kind, value) -> [{source, external_id, key_kind}] for the annotation-only external alias sources."""
    out = defaultdict(list)
    for h in fits["_annotations"]["lineage_a_validation"]:
        row = inp.catalog.get(h)
        ks = fk.derive_exact_and_source_aware(row) if isinstance(row, dict) else fk.derive({"dataset_hash": h})[0]
        for kv in ks:
            out[kv].append({"source": "lineage_a_validation", "external_id": h, "key_kind": kv[0]})
    for h, part in sorted(lb["part_of"].items()):
        if part == "meta_train":
            out[("dataset_hash", h)].append({"source": "lineage_b_meta_train", "external_id": h, "key_kind": "dataset_hash"})
    v6 = set(inp.v6_names)
    for name, ptr in sorted(inp.pointers.items()):
        try:
            ks = fk.derive(fk.attrs_from_beyondarena_pointer(ptr))[0]
        except (fk.ForbiddenKey, fk.MalformedKey):
            continue
        for kv in ks:
            if kv[0] == "beyondarena":
                continue
            out[kv].append({"source": "beyondarena", "external_id": "beyondarena:" + name, "key_kind": kv[0]})
            if name in v6:
                out[kv].append({"source": "v6_teacher_search", "external_id": "beyondarena:" + name, "key_kind": kv[0]})
    return dict(out)


def prepare(inp: Inputs, *, guard=None, expect: dict | None = None, extra_withhold=frozenset()) -> Base:
    """extra_withhold: identities to withhold beyond alias-sources repin.withhold (the local re-pin's plaintext pass)."""
    guard = guard or fp.NoGuard()
    if expect and expect.get("epoch_sha256"):
        _need(_entry(inp.idx, "epoch").get("sha256") == expect["epoch_sha256"], "the pinned epoch is not epoch-000 (STOP)")
    lb = lineage_b_sets(inp)
    xc = sealed_crosschecks(inp, lb, expect)
    seeds = seed_check(inp, expect)
    idents = make_identities(inp)
    derive_identity_keys(inp, idents, lb)
    kedges = key_edges(idents)
    base_sealed = {i: member_bases(I, lb) for i, I in idents.items() if member_bases(I, lb)}
    repin = inp.cfg.get("repin") if isinstance(inp.cfg.get("repin"), dict) else {}
    withhold = frozenset((repin.get("withhold") or {}).get("identities") or []) | frozenset(extra_withhold)
    _need(withhold <= set(idents), "repin.withhold names an unknown identity (re-pin)")
    floor = repin.get("seal_floor")
    seal_floor = frozenset(floor["identities"]) if isinstance(floor, dict) else None
    _need(seal_floor is None or (_sha(common.canonical(sorted(seal_floor))) == floor.get("sha256")
                                 and seal_floor <= set(idents)), "alias-sources repin.seal_floor is malformed (re-pin)")
    red = repin.get("redaction") if isinstance(repin.get("redaction"), dict) else None
    redacted_members = frozenset(red.get("identities") or []) if red else None
    guard.wait("screen", force=True)
    scr = run_screen(inp, idents)
    guard.wait("screen-done", force=True)
    # redact name evidence for every identity whose family could be sealed: computed conservatively over the
    # pre-screen union-find (base-sealed or withheld components) plus every sealed reference (the screen redacts those)
    pre = components(idents, kedges)
    sealed_ids = set()
    for comp in pre:
        if any(i in base_sealed or i in withhold for i in comp):
            sealed_ids |= set(comp)
    links = split_links(scr, idents, sealed_ids)
    cur, cur_ev = curated_edges(inp, idents, scr, links["union"])
    all_edges = sorted(set(kedges) | set(links["union"]) | set(cur))
    never_hold = frozenset(base_sealed) | frozenset(links["sref"])
    held, trips = breaker(all_edges, idents, POLICY["BREAKER_MAX_DATASETS"], never_hold=never_hold)
    applied = sorted(e for e in all_edges if e not in held)
    fits = fit_sets(inp, lb)
    ext = external_alias_keys(inp, lb, fits)
    return Base(inp=inp, idents=idents, lb=lb, crosscheck=xc, seeds=seeds, scr=scr, links=links, key_edges=kedges,
                curated=cur, curated_evidence=cur_ev, held=held, trips=trips, applied=applied, base_sealed=base_sealed,
                fits=fits, ext=ext, redact=sealed_ids, withhold=withhold, seal_floor=seal_floor,
                redacted_members=redacted_members)


# ----------------------------------------------------------------------------------------------------- model
@dataclass
class Model:
    phase: str
    families: dict
    fid_of: dict
    edges: list
    retired: dict
    rows: list = field(default_factory=list)
    effective: dict = field(default_factory=dict)
    decisions_sha256: str | None = None
    deltas: dict = field(default_factory=dict)
    parts: dict = field(default_factory=dict)
    fam_rows: dict = field(default_factory=dict)
    by_fid: dict = field(default_factory=dict)


def _canon_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class Canaries:
    """In-memory canary strings (sealed display names, URLs, source keys, lineage-B names). Never written or printed.
    withhold(text) is True when a non-sealed display string or trigger value contains a canary after normalization
    (lower case, every non-alphanumeric run -> '_', word boundaries), so a colliding value is withheld or hashed."""

    def __init__(self, strings):
        # hex identifiers (dataset hashes, sha256 values) are ids that every output lists by design, not names; strings
        # of CANARY_MIN_LEN (4) characters or more are kept and matched on word boundaries (review fix SC-7); shorter
        # strings are too generic to scan for
        self.strings = sorted({s.strip() for s in strings if isinstance(s, str) and len(s.strip()) >= CANARY_MIN_LEN
                               and not _HEX_ID.fullmatch(s.strip())})
        short = sorted({re.escape(n) for n in (_canon_text(s) for s in self.strings if len(s) < 12)
                        if len(n) >= CANARY_MIN_LEN}, key=lambda s: (-len(s), s))
        long = sorted({re.escape(n) for n in (_canon_text(s) for s in self.strings if len(s) >= 12) if n},
                      key=lambda s: (-len(s), s))
        alts = ([r"(?<![a-z0-9])(?:" + "|".join(short) + r")(?![a-z0-9])"] if short else []) + (
            ["(?:" + "|".join(long) + ")"] if long else [])
        self.rx = re.compile("|".join(alts)) if alts else None

    def withhold(self, text) -> bool:
        return isinstance(text, str) and self.rx is not None and self.rx.search(_canon_text(text)) is not None

    def __iter__(self):
        return iter(self.strings)

    def __len__(self):
        return len(self.strings)


def assign_ids(comps: list, *, pre: Model | None = None, previous: dict | None = None) -> tuple[list, dict]:
    """[(fid, anchor, members)] and retired_family_ids. pre: the pre-review model of this build (decision-driven
    merges and splits allowed); previous: a published registry (post-freeze: any merge or split is a STOP)."""
    out, retired = [], {}
    if previous is not None:
        prev_of = previous.get("identities") or {}
        prev_fams = previous.get("families") or {}
        known_ids = set(prev_fams) | set(previous.get("retired_family_ids") or {})
        retired = dict(previous.get("retired_family_ids") or {})
        seen = set()
        for comp in comps:
            prev = sorted({prev_of[i] for i in comp if i in prev_of})
            if not prev:
                anchor = comp[0]
                fid = fid_of_anchor(anchor)
                _need(fid not in known_ids, "a new family would reuse an existing family_id (STOP)")
            elif len(prev) == 1:
                fid = prev[0]
                old = {m["identity"] for m in prev_fams[fid]["members"]}
                _need(old <= set(comp), f"post-freeze split of existing family {fid}: STOP")
                anchor = prev_fams[fid]["anchor_identity"]
            else:
                raise FamilyStop(f"post-freeze merge of {len(prev)} existing families: STOP (CONTAMINATED if their "
                                 "assignments differ)")
            _need(fid not in seen, f"family {fid} would map to two components (post-freeze split): STOP")
            seen.add(fid)
            out.append((fid, anchor, comp))
        missing = sorted(set(prev_fams) - seen)
        _need(not missing, f"{len(missing)} existing families have no successor (STOP)")
        return out, retired
    for comp in comps:
        anchor = comp[0]
        out.append((fid_of_anchor(anchor), anchor, comp))
    if pre is not None:
        new_by_anchor = {anchor: fid for fid, anchor, _ in out}
        comp_of = {i: fid for fid, _, comp in out for i in comp}
        for fid, F in sorted(pre.families.items()):
            if F["anchor_identity"] not in new_by_anchor:
                retired[fid] = comp_of[F["anchor_identity"]]
            else:
                _need(new_by_anchor[F["anchor_identity"]] == fid, "family_id rule broken between pre-review and final")
    return out, retired


def _rights_member(I: Ident) -> dict:
    lic = I.license if isinstance(I.license, dict) else {}
    status = lic.get("status") if _known(lic.get("status")) else "unknown"
    spdx = lic.get("spdx") if _known(lic.get("spdx")) else None
    note = lic.get("note") if _known(lic.get("note")) else None
    return {"status": status.strip().lower(), "spdx": spdx.strip() if spdx else None,
            "note_sha256": _vsha(note) if note else None,
            "restrictive_note": bool(fk.restrictive_note(note) or fk.restrictive_note(spdx))}


def family_rights(members: list, idents: dict, prohibited: bool) -> tuple[str, dict]:
    rm = {i: _rights_member(idents[i]) for i in members}
    if prohibited:
        return "prohibited", rm
    regs = [rm[i] for i in members if idents[i].kind == "registry_item"]
    spdx = {r["spdx"] for r in regs}
    if regs and all(r["status"] == "recorded" for r in regs) and len(spdx) == 1:
        s = spdx.pop()
        if isinstance(s, str) and _SPDX.fullmatch(s):
            return "recorded:" + s, rm
    return "unknown", rm


def sealed_out_legacy(inp: Inputs, lb: dict) -> set:
    """Normalized legacy source families of the sealed datasets outside the epoch (S - E)."""
    E = {r["dataset_hash"] for r in inp.rows}
    return {fk.norm(inp.legacy[h]) for h in lb["S"] - E if _known(inp.legacy.get(h))}


def _legacy_variants(v: str) -> set:
    """A legacy source family and its singular/plural twin (the rule _related_classes applies; review fix SC-1)."""
    out = {v, v + "s"}
    if len(v) > 1 and v.endswith("s"):
        out.add(v[:-1])
    return out


def compute_model(base: Base, edges: list, *, phase: str, pre: Model | None = None, effective: dict | None = None,
                  rows: list | None = None, previous: dict | None = None) -> Model:
    idents, lb, inp = base.idents, base.lb, base.inp
    comps = components(idents, edges)
    ids, retired = assign_ids(comps, pre=pre, previous=previous)
    families, fid_of = {}, {}
    for fid, anchor, members in ids:
        _need(fid not in families, "family_id collision (STOP)")
        families[fid] = {"family_id": fid, "anchor_identity": anchor, "_members": members}
        for m in members:
            fid_of[m] = fid
    by_fid = defaultdict(list)
    for e in edges:
        by_fid[fid_of[e.a]].append(e)
    effective = effective or {}
    rows = rows or []
    row_by_id = {r["review_group_id"]: r for r in rows}
    # pre-review family ids per final family (decision lookups)
    parts = {fid: sorted({pre.fid_of[i] for i in F["_members"]}) if pre else [fid] for fid, F in families.items()}
    rows_touching = defaultdict(set)
    for r in rows:
        for f in r["family_ids"]:
            rows_touching[f].add(r["review_group_id"])
    prohibited_pre = {f for g, d in effective.items() if d == "prohibited" for f in row_by_id[g]["family_ids"]}
    fam_rows = {fid: sorted(set().union(*[rows_touching.get(p, set()) for p in parts[fid]]))
                for fid in families}
    # ---- sealed (hard bases first, then provisional adjacency over the hard-sealed families)
    hard = {}
    for fid, F in families.items():
        mem = F["_members"]
        bases = sorted({b for m in mem for b in base.base_sealed.get(m, [])}, key=SEALED_BASES.index)
        near = sorted({x["ref"] for m in mem for x in base.links["sref"].get(m, [])})
        if bases:
            reg = [m for m in mem if idents[m].kind == "registry_item"]
            all_base = all(m in base.base_sealed for m in reg)
            extra = (["sealed_union"] if not all_base else []) + (["value_near_dup_of_sealed_reference"] if near else [])
            hard[fid] = {"basis": bases[0] if all_base else "sealed_union", "bases": bases + extra, "near_refs": near}
        elif near:
            hard[fid] = {"basis": "value_near_dup_of_sealed_reference", "bases": ["value_near_dup_of_sealed_reference"],
                         "near_refs": near}
    sealed_tokens = set()
    for fid in hard:
        for m in families[fid]["_members"]:
            sealed_tokens |= set(idents[m].tokens)
    for name in lb["sealed_names"]:
        t = _basename_token(name.replace(":", "/")) if isinstance(name, str) else None
        if t:
            sealed_tokens.add(t)
    for k in lb["SK"]:
        sealed_tokens.add(_norm_token(k.split("/", 1)[-1]))
    for rec in inp.sealed_records.values():
        for text in (rec.get("source_key"),):
            t = _basename_token(text.replace(":", "/")) if isinstance(text, str) else None
            if t:
                sealed_tokens.add(t)
    sealed_tokens = {t for t in sealed_tokens if _valid_token(t)}
    # sealed Rdatasets item names as word sequences: contained in a non-sealed member's display name, URL basename or
    # dataset_id ("the same Rdatasets item name in another package or platform"; review trigger, never a union)
    sealed_items = {_norm_token(k.split("/", 1)[-1]) for k in lb["SK"]}
    for fid in hard:
        for m in families[fid]["_members"]:
            sealed_items |= {_norm_token(v.split("/", 1)[-1]) for k, v in idents[m].U if k == "rdatasets_item"}
    sealed_words = sorted({tuple(t.split("_")) for t in sealed_items if _valid_token(t)})
    hard_legacy = defaultdict(set)
    for fid in hard:
        for m in families[fid]["_members"]:
            for k, v in idents[m].T:
                if k == "legacy_source_family":
                    hard_legacy[v].add(fid)
    out_legacy = sealed_out_legacy(inp, lb)        # legacy families of sealed datasets outside the epoch
    # held breaker edges still held in this model (a 'union' decision re-admits them as edges)
    edge_set = set(edges)
    held_adj = defaultdict(list)
    for e in sorted(base.held):
        if e in edge_set:
            continue
        fa, fb = fid_of[e.a], fid_of[e.b]
        if fa != fb:
            held_adj[fa].append((e, fb))
            held_adj[fb].append((e, fa))
    adjacency = {}
    for fid, F in families.items():
        if fid in hard:
            continue
        adj = []
        for m in F["_members"]:
            I = idents[m]
            for k, v in I.T:
                if k == "legacy_source_family":
                    variants = _legacy_variants(v)          # review fix SC-1: singular and plural match
                    hits = set().union(*[hard_legacy.get(x, set()) for x in variants])
                    if hits or variants & out_legacy:
                        adj.append(("legacy_source_family", v, sorted(hits)))
                if k == "lineage_b_audit_family":
                    adj.append(("lineage_b_audit_family", v, []))
            for t in sorted(I.tokens & sealed_tokens):
                adj.append(("name_token", t, sorted(f for f in hard if t in set().union(
                    *[idents[x].tokens for x in families[f]["_members"]]))))
            for w in _contained_words(I, sealed_words):
                adj.append(("name_token", "_".join(w), []))
            for w in base.links["sref_weak"].get(m, []):
                adj.append(("nds:value_weak", w["ref"], []))
            for w in base.links.get("sref_subset", {}).get(m, []):
                adj.append(("nds:name_subset", w["ref"], []))
        for w in base.links["weak"]:
            for x, y in ((w["a"], w["b"]), (w["b"], w["a"])):
                if fid_of.get(x) == fid and fid_of.get(y) in hard:
                    adj.append(("nds:value_weak", f"{x}|{y}", [fid_of[y]]))
        for e, other in held_adj.get(fid, []):           # review fixes CL-4 / SC-5
            if other in hard:
                adj.append(("held_edge", f"{e.kind}|{e.a}|{e.b}", [other]))
        if adj:
            adjacency[fid] = sorted({(k, v, tuple(s)) for k, v, s in adj})
    sealed_rows_of = defaultdict(list)
    for r in rows:
        if r["kind"] == "sealed_adjacent":
            for f in r["_subject"]:
                sealed_rows_of[f].append(r["review_group_id"])
    for fid, F in families.items():
        if fid in hard:
            F.update(sealed=True, sealed_basis=hard[fid]["basis"], sealed_bases=hard[fid]["bases"],
                     _near_refs=hard[fid]["near_refs"], _adjacency=[])
        elif fid in adjacency:
            sa = sorted({g for p in parts[fid] for g in sealed_rows_of.get(p, [])})
            decided = [effective.get(g) for g in sa]
            if sa and all(d == "unseal" for d in decided):
                F.update(sealed=False, sealed_basis=None, sealed_bases=["unsealed_by_owner_decision"])
            elif any(d == "keep_sealed" for d in decided):
                F.update(sealed=True, sealed_basis="owner_decision", sealed_bases=["owner_decision", "provisional_pending_review"])
            else:
                F.update(sealed=True, sealed_basis="provisional_pending_review", sealed_bases=["provisional_pending_review"])
            F["_adjacency"] = adjacency[fid]
            F["_near_refs"] = []
        else:
            F.update(sealed=False, sealed_basis=None, sealed_bases=[], _adjacency=[], _near_refs=[])
        # display withholding (review fix SC-3): alias-sources repin.withhold; run_build adds the auto-withheld families
        wh = sorted(m for m in F["_members"] if m in base.withhold)
        F["withheld"] = bool(wh)
        F["withheld_basis"] = ["repin_withhold"] if wh else []
    # ---- seeds
    seed_of = base.seeds["source_of"]
    for fid, F in families.items():
        srcs = sorted(m for m in F["_members"] if idents[m].kind == "beyondarena")
        names = {s.split(":", 1)[1] for s in srcs}
        F["seed_selection_exposed"] = bool(srcs)
        F["seed_sources"] = srcs
        F["seeds_from_sources"] = sum(1 for p, s in seed_of.items() if s in names)
        ev = {n: base.curated_evidence[n] for n in sorted(names) if n in base.curated_evidence}
        F["seed_evidence"] = ({"curated": ev, "seeds_from_sources": F["seeds_from_sources"]} if srcs else None)
    # ---- exposure
    fresh = set(POLICY["FRESH_COHORTS"])
    for fid, F in families.items():
        mem = F["_members"]
        exp = {}
        for enc in ENCODERS:
            fit = base.fits[enc]
            ev = sorted({(m, kv[0], f) for m in mem for kv in idents[m].U for f in fit["keys"].get(kv, ())})
            if ev:
                state, basis = "Exposed", "shared_key"
            elif not fit["enumerated"]:
                state, basis = "Unknown", "fit_set_not_enumerated"
            elif any(idents[m].cohort in fresh or idents[m].kind == "beyondarena" for m in mem):
                state, basis = "Unknown", "fresh_or_external_member"
            else:
                state, basis = "Unseen", "no_shared_key_with_enumerated_fit_set"
            exp[enc] = {"state": state, "basis": basis}
            if ev:
                exp[enc]["evidence"] = [{"identity": m, "key_kind": k, "fit_id": f} for m, k, f in ev[:3]]
                exp[enc]["evidence_total"] = len(ev)
        F["encoder_weight_exposure"] = exp
    # ---- rights
    for fid, F in families.items():
        prohibited = any(p in prohibited_pre for p in parts[fid])
        F["rights_status"], F["rights_members"] = family_rights(F["_members"], idents, prohibited)
    # ---- coverage
    tables = base.scr.tables
    for fid, F in families.items():
        cov = [tables[m]["screen_coverage"] for m in F["_members"] if m in tables]
        F["screen_coverage"] = min(cov, key=lambda c: COVERAGE_RANK[c]) if cov else "exact_only"
    # ---- contexts
    epoch_ctx = {r["context_sha256"] for r in inp.rows}
    for fid, F in families.items():
        inc, exc, fold0, folds, reg_only = [], {}, {}, Counter(), set()
        for m in F["_members"]:
            I = idents[m]
            for r in I.rows:
                c = r["context_sha256"]
                if r.get("status") == "included":
                    f0 = r["cohort"] == "frozen" and r.get("fold") == 0
                    _need(f0 == str(r.get("eligibility_source") or "").startswith("frozen_fold0_by_construction"),
                          f"context {c}: frozen_fold0 disagrees with eligibility_source")
                    inc.append(c)
                    fold0[c] = f0
                    folds[str(r.get("fold"))] += 1
                else:
                    exc[c] = r.get("exclusion_reason") or r.get("status")
            reg_only |= set(I.reg_contexts) - epoch_ctx
        F["contexts"] = sorted(inc)
        F["frozen_fold0"] = {c: fold0[c] for c in F["contexts"]}
        F["excluded_contexts"] = dict(sorted(exc.items()))
        F["registry_only_contexts"] = sorted(reg_only)
        F["folds"] = dict(sorted(folds.items()))
    return Model(phase=phase, families=families, fid_of=fid_of, edges=sorted(edges), retired=retired, parts=parts,
                 fam_rows=fam_rows, by_fid=dict(by_fid))


# ----------------------------------------------------------------------------------------------------- review rows
def _units(F: dict, idents: dict) -> set:
    return {idents[m].dataset_hash for m in F["_members"] if idents[m].kind == "registry_item"}


def _fam_stats(fids, model: Model, idents: dict) -> dict:
    fams = [model.families[f] for f in fids]
    ds = set().union(*[_units(F, idents) for F in fams]) if fams else set()
    ctx = sum(len(F["contexts"]) for F in fams)
    f0 = sum(sum(1 for v in F["frozen_fold0"].values() if v) for F in fams)
    coh = sorted({idents[m].cohort for F in fams for m in F["_members"]})
    return {"n_families": len(fams), "n_datasets": len(ds), "n_contexts": ctx, "n_fold0": f0, "cohorts": coh}


WITHHELD_NAME = "[name withheld: collides with a sealed name]"


def _display_withheld(F: dict) -> bool:
    """A sealed (hard or provisional) or withheld family never shows member names, URLs or key values (review fix
    SC-3: withheld = alias-sources repin.withhold, or a family a recommended union would join with a sealed family)."""
    return bool(F["sealed"] or F.get("withheld"))


def _member_text(fids, model: Model, idents: dict, cans: Canaries) -> str:
    names, sealed_n, withheld_n = [], 0, 0
    for f in fids:
        F = model.families[f]
        for m in F["_members"]:
            if F["sealed"]:
                sealed_n += 1
            elif F.get("withheld"):
                withheld_n += 1
            else:
                names.append(_safe_display(idents[m].display_name or idents[m].dataset_hash or m, cans))
    out = sorted(set(names))
    if sealed_n:
        out.append(f"[sealed member withheld x{sealed_n}]")
    if withheld_n:
        out.append(f"[member withheld x{withheld_n}]")
    return "; ".join(out)


def _safe_display(text, cans: Canaries):
    if text is None:
        return None
    return WITHHELD_NAME if cans.withhold(text) else text


def _legacy_key(v: str) -> str:
    """Singular/plural-insensitive grouping key of a legacy source family (trailing 's' dropped)."""
    return v[:-1] if len(v) > 1 and v.endswith("s") else v


def _pairwise(fids: list, model: Model, base: Base, own: frozenset = frozenset(),
              own_evidence: bool = False) -> tuple[list, list]:
    """Pairwise evidence between different families of a row: weak value links, shared name tokens and shared legacy
    source families (singular and plural match). Returns (evidence items, evidence components).

    own: the row's own trigger (kind, value) pairs. A row of an OWN_EVIDENCE_KINDS kind (legacy source family, name
    token, lineage-B audit family, PMLB stem, Socrata 4x4 across hosts) relates its families through its own trigger, so
    that trigger is pairwise evidence (spec rule 'pairwise evidence -> union_subset'; review fix CL-1). For every other
    row kind (package, publisher, domain, owner, year) the shared trigger alone is not evidence."""
    idents, fs_ = base.idents, set(fids)
    items = []
    for w in base.links["weak"]:
        fa, fb = model.fid_of.get(w["a"]), model.fid_of.get(w["b"])
        if fa in fs_ and fb in fs_ and fa != fb:
            ev = w["evidence"]
            items.append(("nds:value_weak", fa, fb, {"J": ev.get("J"), "containment": ev.get("containment"),
                                                     "n_mapped": ev.get("n_mapped"), "T": ev.get("T")}))
    tok, leg, trig = defaultdict(set), defaultdict(set), defaultdict(set)
    for f in fids:
        for m in model.families[f]["_members"]:
            for t in idents[m].tokens:
                tok[t].add(f)
            for k, v in idents[m].T:
                if k == "legacy_source_family":
                    leg[_legacy_key(v)].add(f)
                elif own_evidence and (k, v) in own:
                    trig[(k, v)].add(f)
    for t, s in sorted(tok.items()):
        if len(s) > 1 and (own_evidence or ("name_candidate", t) not in own):
            s = sorted(s)
            items.append(("name_token", s[0], s[-1], {"token": t, "families": s}))
    own_legacy = {_legacy_key(v) for k, v in own if k == "legacy_source_family"}
    for v, s in sorted(leg.items()):
        if len(s) > 1 and (own_evidence or v not in own_legacy):
            s = sorted(s)
            items.append(("legacy_source_family", s[0], s[-1], {"legacy_family": v, "families": s}))
    for (k, _v), s in sorted(trig.items()):
        if len(s) > 1:
            s = sorted(s)
            items.append(("own_trigger", s[0], s[-1], {"trigger_kind": k, "families": s}))
    pairs = []
    for kind, a, b, ev in items:
        fam = ev.get("families") if isinstance(ev, dict) and "families" in ev else [a, b]
        pairs += [(fam[0], x) for x in fam[1:]]
    nodes = sorted({x for p in pairs for x in p})
    uf = _UF(nodes)
    for a, b in pairs:
        uf.union(a, b)
    comps = [c for c in uf.components() if len(c) > 1]
    return items, comps


def _render_items(items: list, sealed_row: bool, cans: Canaries) -> list:
    out = []
    for kind, a, b, ev in items:
        ev = dict(ev)
        if sealed_row:
            for k in ("token", "legacy_family"):
                if k in ev:
                    ev[k] = _hidden(ev[k])
        for k in ("token", "legacy_family"):
            if k in ev and not str(ev[k]).startswith("sha256:") and cans.withhold(ev[k]):
                ev[k] = _hidden(ev[k])
        out.append({"kind": kind, "a": a, "b": b, **ev})
    return out


def _impact(fids, model: Model, idents: dict) -> dict:
    fams = [model.families[f] for f in fids]
    sealed = any(F["sealed"] for F in fams)
    seed = any(F["seed_selection_exposed"] for F in fams)
    st = _fam_stats(fids, model, idents)
    d_ds = sum(len(_units(F, idents)) for F in fams if sealed and not F["sealed"])
    d_ctx = sum(len(F["contexts"]) for F in fams if sealed and not F["sealed"])
    return {"sealed_datasets_delta": d_ds, "sealed_contexts_delta": d_ctx,
            "seed_propagation": bool(seed and any(not F["seed_selection_exposed"] for F in fams)),
            "merged": {"families": st["n_families"], "datasets": st["n_datasets"], "contexts": st["n_contexts"]}}


def _group_id(kind: str, value: str, fids) -> str:
    return "rg-" + _sha(RG_DOMAIN + common.canonical([kind, value, sorted(fids)]))[:24]


def _edge_screen(e: Edge) -> dict:
    """J / containment / mapped and shared counts of an edge's screen evidence (never names or key values)."""
    ev = (e.ev() or {}).get("screen") or {}
    return {k: ev.get(k) for k in ("J", "containment", "n_mapped", "n_shared") if ev.get(k) is not None}


def _adjacency_evidence(f: str, F: dict, base: Base) -> list:
    """Non-identifying evidence of a provisional seal (review fix SC-6): per adjacency, its kind, the value as sha256,
    the hard-sealed families it touches and, for value or name links, the screen statistics. No names or URLs."""
    out, members = [], set(F["_members"])
    stats = {}
    for m in sorted(members):
        for w in base.links["sref_weak"].get(m, []):
            ev = w["evidence"] or {}
            stats[("nds:value_weak", w["ref"])] = {k: ev.get(k) for k in ("J", "containment", "n_mapped", "stop_relaxed")
                                                   if ev.get(k) is not None}
        for w in base.links.get("sref_subset", {}).get(m, []):
            ev = w["evidence"] or {}
            stats[("nds:name_subset", w["ref"])] = {k: ev.get(k) for k in ("n_ref_names", "n_names", "J", "rule")
                                                    if ev.get(k) is not None}
    for w in base.links["weak"]:
        if w["a"] in members or w["b"] in members:
            ev = w["evidence"] or {}
            stats[("nds:value_weak", f"{w['a']}|{w['b']}")] = {k: ev.get(k) for k in ("J", "containment", "n_mapped")
                                                              if ev.get(k) is not None}
            stats[("nds:value_weak", f"{w['b']}|{w['a']}")] = stats[("nds:value_weak", f"{w['a']}|{w['b']}")]
    held = {f"{e.kind}|{e.a}|{e.b}": e for e in base.held}
    for k, v, touch in F["_adjacency"]:
        item = {"kind": k, "value_sha256": _vsha(v), "hard_families": list(touch)}
        if (k, v) in stats:
            item["screen"] = stats[(k, v)]
        if k == "held_edge" and v in held:
            item["screen"] = dict(_edge_screen(held[v]), edge_kind=held[v].kind)
        out.append(item)
    return out


def review_rows(base: Base, model: Model, cans: Canaries) -> list:
    idents, fams = base.idents, model.families
    rows = []

    def add(kind, fids, *, value, trigger_kinds=None, evidence=None, subject=None, flags=(), held=None, kvs=None):
        fids = sorted(set(fids))
        hidden_row = any(_display_withheld(fams[f]) for f in fids)
        hide = hidden_row or cans.withhold(value)           # review fix SC-9: a sealed seed family's row too
        shown = (value if value.startswith("sha256:") else _hidden(value)) if hide else value
        rows.append({"kind": kind, "trigger_kinds": sorted(trigger_kinds or [kind]), "_value": value, "trigger_value": shown,
                     "family_ids": fids, "_subject": sorted(subject or fids), "_evidence": evidence or {},
                     "_flags": set(flags), "_held": held or [], "_rec": None, "_sealed": hidden_row,
                     "_kvs": frozenset(kvs or [(kind, value)])})

    hard_fids = {f for f, F in fams.items() if F["sealed"] and F["sealed_basis"] in HARD_BASES}
    # trigger groups (shared by >= 2 families; related class and withdrawn keys by >= 1)
    inv = defaultdict(set)
    for f, F in fams.items():
        for m in F["_members"]:
            for k, v in idents[m].T:
                inv[(k, v)].add(f)
    wf = {}
    for w in base.lb["withdrawn"]:
        wf[re.sub(r"[^a-z0-9]+", "-", w.lower()).strip("-")] = w
    for f, F in fams.items():
        for m in F["_members"]:
            I = idents[m]
            slugs = set()
            for k, v in I.U:
                if k == "rdatasets_item":
                    pkg, _, item = v[len("rdatasets:"):].partition("/")
                    slugs.add(re.sub(r"[^a-z0-9]+", "-", f"rdatasets-{pkg}-{item}".lower()).strip("-"))
            exp = base.inp.expansion.get(I.dataset_hash)
            if exp:
                slugs.add(re.sub(r"[^a-z0-9]+", "-", exp.lower()).strip("-"))
            for s in slugs & set(wf):
                inv[("withdrawn_family_key", wf[s])].add(f)
    groups = defaultdict(list)
    for (k, v), fset in sorted(inv.items()):
        if k in ("rights_note", "hunt_content_mismatch"):
            continue
        need = 1 if k in ("lineage_b_related_class", "withdrawn_family_key") else 2
        if k not in TRIGGER_ROW_KINDS or len(fset) < need:
            continue
        groups[tuple(sorted(fset))].append((k, v))
    for fids, kvs in sorted(groups.items()):
        kvs = sorted(kvs, key=lambda kv: (TRIGGER_ROW_KINDS.index(kv[0]), kv[1]))
        kind = kvs[0][0]
        value = kvs[0][1] if len(kvs) == 1 else ";".join(f"{k}={v}" for k, v in kvs)
        always_hide = any(k in ("lineage_b_audit_family", "withdrawn_family_key", "lineage_b_related_class")
                          for k, _ in kvs)
        add(kind, fids, value=_hidden(value) if always_hide else value, trigger_kinds=[k for k, _ in kvs],
            flags=["related_class"] if kind == "lineage_b_related_class" else (), kvs=kvs)
    # hunt content mismatch: per family (folded into the hunt_semantic_signature kind)
    for f, F in sorted(fams.items()):
        mism = sorted({v for m in F["_members"] for k, v in idents[m].T if k == "hunt_content_mismatch"})
        if mism:
            add("hunt_semantic_signature", [f], value=_hidden(mism), trigger_kinds=["hunt_content_mismatch"],
                evidence={"hunt_content_mismatch": len(mism)})
    # multi-dataset families (not hard-sealed, not seed families); review fix CL-6: per-kind counts and value evidence
    for f, F in sorted(fams.items()):
        if len(_units(F, idents)) > 1 and f not in hard_fids and not F["seed_selection_exposed"]:
            fe = model.by_fid.get(f, [])
            kinds = Counter(e.kind for e in fe)
            nds = [dict(_edge_screen(e), kind=e.kind, a=e.a, b=e.b) for e in sorted(fe)
                   if e.kind.startswith("nds:") or e.kind == "content"][:40]
            add("multi_member_family", [f], value=f"datasets={len(_units(F, idents))}",
                evidence={"edge_kinds": dict(sorted(kinds.items())), "nds_edges": nds,
                          "split_default_drops": sorted(k for k in kinds if k in SPLIT_DEFAULT_KINDS),
                          "split_needs_explicit": sorted(k for k in kinds if k in SPLITTABLE_KINDS
                                                         and k not in SPLIT_DEFAULT_KINDS)})
    # weak value links between families
    wpairs = []
    for w in base.links["weak"]:
        fa, fb = model.fid_of[w["a"]], model.fid_of[w["b"]]
        if fa != fb:
            wpairs.append((fa, fb))
    nodes = sorted({x for p in wpairs for x in p})
    for comp in components(nodes, [Edge("w", a, b, "w") for a, b in sorted(wpairs)]):
        if len(comp) > 1:
            add("nds_value_weak", comp, value=f"weak_links={sum(1 for a, b in wpairs if a in comp)}")
    # name candidates: a token shared by >= 2 families (seed-source names included through the BA identities)
    tokinv = defaultdict(set)
    for f, F in fams.items():
        for m in F["_members"]:
            for t in idents[m].tokens:
                tokinv[t].add(f)
    for t, fset in sorted(tokinv.items()):
        if len(fset) > 1:
            add("name_candidate", fset, value=t)
    # classic candidates (agent-drafted review entries and keyword hits; never value-confirmed)
    for e in sorted(base.inp.classic.get("entries") or [], key=lambda e: str(e.get("name"))):
        if not isinstance(e, dict) or e.get("action") != "review":
            continue
        fset = set()
        for m in e.get("members") or []:
            j = _member_identity(idents, m)
            if j is not None:
                fset.add(model.fid_of[j])
        kw = e.get("keyword")
        if isinstance(kw, str) and kw.strip():
            needle = re.sub(r"[^a-z0-9]+", "", kw.lower())
            for f, F in fams.items():
                for m in F["_members"]:
                    I = idents[m]
                    if I.kind != "registry_item":
                        continue
                    prov = (I.row or {}).get("provenance") if isinstance((I.row or {}).get("provenance"), dict) else {}
                    hay = re.sub(r"[^a-z0-9]+", "", " ".join(x for x in (I.url, _str(prov.get("dataset_id"))) if x).lower())
                    if needle and needle in hay:
                        fset.add(f)
        if fset:
            for s in e.get("related_seed_sources") or []:
                if ("beyondarena:" + s) in model.fid_of:
                    fset.add(model.fid_of["beyondarena:" + s])
            add("classic_candidate", fset, value=str(e.get("name")))
    # seed-source mapping rows (one per seed source)
    for name in base.inp.seed_sources:
        ba = "beyondarena:" + name
        f = model.fid_of[ba]
        add("seed_source_mapping", [f], value=ba, flags=["seed"],
            evidence={"curated": base.curated_evidence.get(name),
                      "seeds_from_source": sum(1 for s in base.seeds["source_of"].values() if s == name),
                      "family_members": len(fams[f]["_members"])})
    # sealed-adjacent rows: one per provisionally sealed family (plus the hard-sealed families it touches)
    for f, F in sorted(fams.items()):
        if F["sealed_basis"] not in ("provisional_pending_review", "owner_decision") or not F["_adjacency"]:
            continue
        adj = F["_adjacency"]
        touch = sorted({x for _, _, s in adj for x in s if x in fams})
        kinds = sorted({k for k, _, _ in adj})
        add("sealed_adjacent", [f] + touch, value=_hidden(sorted([k, v] for k, v, _ in adj)), trigger_kinds=kinds,
            subject=[f], flags=["sealed_adjacent"],
            evidence={"adjacency_kinds": dict(Counter(k for k, _, _ in adj)),
                      "nds_or_name_evidence": any(k in KEEP_SEALED_ADJACENCY for k in kinds),
                      "adjacency": _adjacency_evidence(f, F, base)})
    # rights rows (restrictive licence notes), per family
    for f, F in sorted(fams.items()):
        if f in hard_fids:
            continue
        notes = sorted({v for m in F["_members"] for k, v in idents[m].T if k == "rights_note"
                        and idents[m].kind == "registry_item"})
        if notes:
            add("rights_note", [f], value=_hidden(notes) if _display_withheld(F) else " | ".join(n[:160] for n in notes),
                flags=["rights"], evidence={"notes": len(notes)})
    # coverage rows: families with included contexts whose screen coverage is values_only or exact_only
    for f, F in sorted(fams.items()):
        if f in hard_fids or not F["contexts"] or F["screen_coverage"] not in ("values_only", "exact_only"):
            continue
        add("coverage_" + F["screen_coverage"], [f], value=F["screen_coverage"], flags=["coverage"],
            evidence={"tables": {m: {k: base.scr.tables[m].get(k) for k in ("n_informative", "n_informative_nonstop",
                                                                              "n_names", "names_known")}
                                 for m in F["_members"] if m in base.scr.tables}})
    # circuit breaker rows (review fix CL-4: every held cross-family edge with its statistics)
    for trip in base.trips:
        fset = {model.fid_of[i] for i in trip["identities"]}
        held = [dict(_edge_screen(e), kind=e.kind, fa=model.fid_of[e.a], fb=model.fid_of[e.b]) for e in trip["edges"]]
        add("circuit_breaker", fset, value=f"datasets={trip['datasets']}", flags=["breaker"], held=trip["edges"],
            evidence={"datasets": trip["datasets"], "held_edges": dict(Counter(e.kind for e in trip["edges"])),
                      "supernode_datasets": trip.get("supernode_datasets"), "held_cross_family_edges": held,
                      "kept_sealed_edges": len(trip.get("kept_sealed_edges") or [])})
    # finalize: ids, stats, evidence, recommendation
    out, seen = [], set()
    for r in rows:
        r["review_group_id"] = _group_id(r["kind"], r["trigger_value"], r["family_ids"])
        if r["review_group_id"] in seen:
            continue
        seen.add(r["review_group_id"])
        out.append(r)
    for r in out:
        st = _fam_stats(r["family_ids"], model, idents)
        r.update(st)
        r["members"] = _member_text(r["family_ids"], model, idents, cans)
        flags = set(r["_flags"])
        if any(fams[f]["sealed"] for f in r["family_ids"]):
            flags.add("sealed_adjacent")
        if any(fams[f].get("withheld") and not fams[f]["sealed"] for f in r["family_ids"]):
            flags.add("withheld")
        if any(fams[f]["seed_selection_exposed"] for f in r["family_ids"]):
            flags.add("seed")
        if any(fams[f]["rights_status"] == "prohibited" or any(v["restrictive_note"] for v in fams[f]["rights_members"].values())
               for f in r["family_ids"]):
            flags.add("rights")
        if any(idents[m].T and any(k == "lineage_b_related_class" for k, _ in idents[m].T)
               for f in r["family_ids"] for m in fams[f]["_members"]):
            flags.add("related_class")
        if any(fams[f]["screen_coverage"] in ("values_only", "exact_only") for f in r["family_ids"]):
            flags.add("coverage")
        r["flags"] = sorted(flags)
        own_ev = r["kind"] in OWN_EVIDENCE_KINDS
        items, comps = (_pairwise(r["family_ids"], model, base, own=r["_kvs"], own_evidence=own_ev)
                        if len(r["family_ids"]) > 1 else ([], []))
        ev = dict(r["_evidence"])
        if items:
            ev["pairwise"] = _render_items(items, r["_sealed"], cans)
        if r["kind"] == "multi_member_family":
            f = r["family_ids"][0]
            ev["union_edges"] = [{k: v for k, v in _edge_doc(e, r["_sealed"], cans).items() if k != "evidence"}
                                 for e in sorted(model.by_fid.get(f, []))[:40]]
        r["evidence"] = ev
        r["impact_if_union"] = _impact(r["family_ids"], model, idents)
        if r["kind"] != "sealed_adjacent":
            r["recommendation"], r["recommendation_basis"], r["_sets"] = _recommend(r, comps)
        r["default_if_undecided"] = DEFAULT_IF_UNDECIDED[r["kind"]]
        r["decision_options"] = list(DECISION_OPTIONS[r["kind"]])
    # sealed_adjacent recommendations (review fix CL-1): keep_sealed when the adjacency holds nds, name, held-edge or
    # spanning lineage-B audit evidence, or when another row recommends uniting the family with a hard-sealed family
    united_with_hard = set()
    for r in out:
        if r["kind"] == "sealed_adjacent":
            continue
        sets = r["_sets"] if r["recommendation"] == "union_subset" else (
            [r["family_ids"]] if r["recommendation"] == "union" else [])
        for s in sets:
            if set(s) & hard_fids:
                united_with_hard |= set(s) - hard_fids
    tag = "agent-generated (deterministic rule, not a decision): "
    for r in out:
        if r["kind"] != "sealed_adjacent":
            continue
        f = r["_subject"][0]
        kinds = {k for k, _, _ in fams[f]["_adjacency"]}
        strong = sorted(kinds & set(KEEP_SEALED_ADJACENCY))
        if strong or f in united_with_hard:
            why = []
            if strong:
                why.append("adjacency " + "/".join(strong))
            if f in united_with_hard:
                why.append("another row recommends uniting it with a hard-sealed family")
            r["recommendation"], r["recommendation_basis"] = "keep_sealed", tag + "keep sealed: " + "; ".join(why)
        else:
            r["recommendation"], r["recommendation_basis"] = "unseal", tag + (
                "trigger adjacency only (legacy source family; no nds, name, held-edge or audit-family evidence and no "
                "recommended union with a hard-sealed family): unseal")
        r["_sets"] = []
    rank = {k: n for n, k in enumerate(MANDATORY_KINDS)}
    out.sort(key=lambda r: (rank.get(r["kind"], len(rank)), -r["n_contexts"], r["kind"], r["review_group_id"]))
    return out


def _recommend(r: dict, comps: list) -> tuple[str, str, list]:
    kind = r["kind"]
    tag = "agent-generated (deterministic rule, not a decision): "
    if kind == "seed_source_mapping":
        return "confirm_mapping", tag + "seed source mapping", []
    if kind == "rights_note":
        return "unknown", tag + "restrictive licence note; no machine-readable prohibition", []
    if kind in ("lineage_b_related_class", "coverage_exact_only", "coverage_values_only", "circuit_breaker",
                "withdrawn_family_key"):
        return "unresolved", tag + f"{kind} rows default to unresolved", []
    if kind == "multi_member_family":
        return "confirm_union", tag + "multi-dataset family", []
    if comps:
        sets = [sorted(c) for c in comps]
        return "union_subset", tag + "pairwise nds/name/legacy evidence; sets=" + common.canonical(sets).decode(), sets
    if kind in OWN_CANDIDATE_KINDS:
        return "unresolved", tag + (f"{kind}: its only evidence is its own curated or semantic match (no value, name or "
                                    "legacy link between its families); owner judgment, never independent"), []
    return "independent", tag + "trigger-only; no pairwise nds, name or legacy evidence", []


# ----------------------------------------------------------------------------------------------------- qualification
def _complete_provenance(F: dict, idents: dict) -> bool:
    regs = [idents[m] for m in F["_members"] if idents[m].kind == "registry_item" and idents[m].epoch_status == "included"]
    if not regs:
        return False
    for I in regs:
        kinds = {k for k, _ in I.U}
        if not (_known(I.canonical) and "normalized_url" in kinds and (_known(I.platform) or kinds & fk.SOURCE_AWARE_KINDS)
                and I.meta is not None):
            return False
    return True


def qualify(base: Base, model: Model, effective: dict | None = None) -> None:
    idents = base.idents
    effective = effective or {}
    rows_by_id = {r["review_group_id"]: r for r in model.rows}
    for fid, F in model.families.items():
        regs = [m for m in F["_members"] if idents[m].kind == "registry_item"]
        inc_units = {idents[m].dataset_hash for m in regs if idents[m].epoch_status == "included"}
        canon = {idents[m].canonical for m in regs}
        edges_ok = all(idents[e.a].canonical == idents[e.b].canonical and idents[e.a].canonical is not None
                       for e in model.by_fid.get(fid, []))
        complete = _complete_provenance(F, idents)
        row_ids = model.fam_rows.get(fid, [])
        basis = {"complete_provenance": complete, "screen_coverage": F["screen_coverage"], "decision_refs": [],
                 "reason": None}
        if F["sealed"]:
            q, basis["reason"] = "unresolved", "sealed"
        elif not F["contexts"]:
            q, basis["reason"] = "unresolved", "contextless"
        elif (len(inc_units) == 1 and len(canon) == 1 and len(regs) == len(F["_members"]) and complete
              and F["screen_coverage"] in ("full", "names_only") and edges_ok and not row_ids
              and not F["seed_selection_exposed"]):
            q, basis["reason"] = "auto", "singleton with complete provenance, full screen coverage and no link or row"
        elif row_ids and model.phase == "final":
            # spec: reviewed iff every review row touching the family carries an owner decision other than unresolved
            # (seed rows: confirm_mapping or add_members). A family no row touches was never in front of the owner, so
            # it cannot become reviewed.
            decided = [(g, effective.get(g)) for g in row_ids]
            basis["decision_refs"] = [{"review_group_id": g, "decision": d} for g, d in decided]
            ok = all(d is not None and d != "unresolved" and d != "reject_mapping"
                     and (rows_by_id[g]["kind"] != "seed_source_mapping" or d in ("confirm_mapping", "add_members"))
                     for g, d in decided)
            if ok:
                q, basis["reason"] = "reviewed", "every review row touching the family is decided"
            else:
                q, basis["reason"] = "unresolved", "an undecided, unresolved or rejected review row"
        else:
            q = "unresolved"
            why = []
            if len(inc_units) != 1 or len(regs) != len(F["_members"]):
                why.append("not a single dataset")
            if not complete:
                why.append("incomplete provenance")
            if F["screen_coverage"] not in ("full", "names_only"):
                why.append(f"screen coverage {F['screen_coverage']}")
            if not edges_ok:
                why.append("linked")
            if row_ids:
                why.append("review rows pending")
            if F["seed_selection_exposed"]:
                why.append("seed-exposed")
            basis["reason"] = "; ".join(why) or "unresolved"
        F["independence_qualified"] = q
        F["qualification_basis"] = basis
        F["review_group_ids"] = row_ids


# ----------------------------------------------------------------------------------------------------- canaries
def canaries(base: Base, model: Model) -> Canaries:
    """In-memory canary strings of sealed and withheld members and lineage-B sealed families (never written or
    printed)."""
    idents, out = base.idents, set(base.lb["sealed_names"])
    for F in model.families.values():
        if not _display_withheld(F):
            continue
        for m in F["_members"]:
            I = idents[m]
            prov = (I.row or {}).get("provenance") if isinstance((I.row or {}).get("provenance"), dict) else {}
            for v in (I.display_name, I.url, _str(prov.get("dataset_id"))):
                if v:
                    out.add(v)
            for k, v in I.U:
                if k in fk.SOURCE_AWARE_KINDS or k in ("normalized_url", "platform_dataset_id", "hunt_platform_id"):
                    out.add(v)
                    if k == "rdatasets_item":
                        out.add(v[len("rdatasets:"):])
            for k, v in I.T:
                if k in ("legacy_source_family", "lineage_b_audit_family", "lineage_b_related_class"):
                    out.add(v)
            meta = I.meta or {}
            for k in ("source_url", "source_dataset_id"):
                if isinstance(meta.get(k), str):
                    out.add(meta[k])
    out |= sealed_out_legacy(base.inp, base.lb)
    for h in base.lb["S"]:
        row = base.inp.catalog.get(h) or {}
        prov = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        for v in (row.get("display_name"), prov.get("url"), prov.get("dataset_id")):
            if isinstance(v, str):
                out.add(v)
        rec = base.inp.sealed_records.get(h) or {}
        for k in ("name", "source_key"):
            if isinstance(rec.get(k), str):
                out.add(rec[k])
    return Canaries(out)


def canary_scan(texts: dict, cans) -> dict:
    """Per output, the number of canary strings found: case-sensitive (strings < 12 chars need word boundaries), plus a
    normalized pass (lower case, non-alphanumeric runs -> '_') for strings of 12 chars or more."""
    cans = list(cans)
    hits = {}
    long = [c for c in cans if len(c) >= 12]
    short = [c for c in cans if len(c) < 12]
    pats = [re.compile(r"(?<![A-Za-z0-9])" + re.escape(c) + r"(?![A-Za-z0-9])") for c in short]
    esc = [json.dumps(c, ensure_ascii=False)[1:-1] for c in long]
    norm = [_canon_text(c) for c in long]
    for name, text in sorted(texts.items()):
        ntext = _canon_text(text)
        n = sum(1 for c, e, k in zip(long, esc, norm) if c in text or e in text or (k and k in ntext))
        n += sum(1 for p in pats if p.search(text))
        hits[name] = n
    return hits


# ----------------------------------------------------------------------------------------------------- rendering
def _ext_aliases(F: dict, base: Base, cans: Canaries) -> list:
    out = set()
    for m in F["_members"]:
        for kv in base.idents[m].U:
            for a in base.ext.get(kv, ()):
                if a["external_id"] == m:
                    continue
                out.add((a["source"], a["external_id"], a["key_kind"]))
        if base.idents[m].kind == "beyondarena":
            n = m.split(":", 1)[1]
            out.add(("ten_case_v3", f"{n}__r0__f0", "seed_view"))
            out.add(("ten_case_hybrid", f"{n}__r0__f0", "seed_view"))
    res = []
    for src, ext, kk in sorted(out):
        if _display_withheld(F) or cans.withhold(ext):
            res.append({"source": src, "external_id_sha256": _vsha(ext), "key_kind": kk, "role": "annotation"})
        else:
            res.append({"source": src, "external_id": ext, "key_kind": kk, "role": "annotation"})
    return res


def _member_doc(I: Ident, sealed: bool, cans: Canaries) -> dict:
    keys, ctx = [], defaultdict(list)
    for k, v in I.U:
        if k in fk.CONTEXT_KINDS:
            ctx[k].append(v)
            continue
        keys.append({"kind": k, "value_sha256": _vsha(v)} if sealed or cans.withhold(v) else {"kind": k, "value": v})
    ctx_keys = {k: {"count": len(v), "set_sha256": _sha(common.canonical(sorted(v)))} for k, v in sorted(ctx.items())}
    doc = {"identity": I.identity, "kind": I.kind, "dataset_hash": I.dataset_hash, "canonical_sha256": I.canonical,
           "sample_identity_sha256": I.sample, "cohort": I.cohort, "epoch_status": I.epoch_status,
           "registry_state": I.registry_state, "meta_sha256": I.meta_sha256, "meta_read": I.meta is not None,
           "display": None if sealed else {"name": _safe_display(I.display_name, cans),
                                           "platform": _safe_display(I.platform, cans), "url": _safe_display(I.url, cans)},
           "union_keys": keys, "context_keys": ctx_keys}
    if I.item_id and I.role in ("incoming", "sibling"):
        doc["item_id"] = I.item_id
    return {k: v for k, v in doc.items() if v is not None or k == "display"}


def _redact_ev(obj):
    """Name evidence ('shared' column names) as sha256 values, recursively (sealed families)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "shared" and isinstance(v, list):
                out["shared_sha256"] = sorted(_vsha(x) for x in v if isinstance(x, str))
            elif k in ("curated", "member") and isinstance(v, str):
                out[k] = v
            else:
                out[k] = _redact_ev(v)
        return out
    if isinstance(obj, list):
        return [_redact_ev(v) for v in obj]
    return obj


def _edge_doc(e: Edge, sealed: bool, cans: Canaries | None = None) -> dict:
    d = {"a": e.a, "b": e.b, "kind": e.kind}
    if sealed or (cans is not None and cans.withhold(e.value)):
        d["value_sha256"] = _vsha(e.value)
    else:
        d["value"] = e.value
    ev = e.ev()
    if ev is not None:
        d["evidence"] = _redact_ev(ev) if sealed else ev
    return d


def family_doc(fid: str, F: dict, base: Base, model: Model, cans: Canaries) -> dict:
    idents = base.idents
    sealed = _display_withheld(F)           # sealed or withheld: no display, keys as sha256 (SC-3)
    mem = [_member_doc(idents[m], sealed, cans) for m in F["_members"]]
    regs = [idents[m] for m in F["_members"]]
    display = None if sealed else {
        "names": sorted({_safe_display(I.display_name, cans) for I in regs if I.display_name}),
        "platforms": sorted({_safe_display(I.platform, cans) for I in regs if I.platform}),
        "urls": sorted({_safe_display(I.url, cans) for I in regs if I.url})}
    rm = {m: v for m, v in sorted(F["rights_members"].items())}
    return {
        "family_id": fid, "anchor_identity": F["anchor_identity"], "members": mem,
        "union_edges": [_edge_doc(e, sealed, cans) for e in sorted(model.by_fid.get(fid, []))],
        "independence_qualified": F["independence_qualified"], "qualification_basis": F["qualification_basis"],
        "rights_status": F["rights_status"], "rights_members": rm,
        "sealed": F["sealed"], "sealed_basis": F["sealed_basis"], "sealed_bases": F["sealed_bases"],
        "display_withheld": bool(F.get("withheld")) and not F["sealed"], "withheld_basis": F.get("withheld_basis") or [],
        "seed_selection_exposed": F["seed_selection_exposed"], "seed_sources": F["seed_sources"],
        "seed_evidence": F["seed_evidence"], "seeds_from_sources": F["seeds_from_sources"],
        "contexts": F["contexts"], "frozen_fold0": F["frozen_fold0"], "excluded_contexts": F["excluded_contexts"],
        "registry_only_contexts": F["registry_only_contexts"],
        "n_identities": len(F["_members"]), "n_datasets": len(_units(F, idents)), "n_contexts": len(F["contexts"]),
        "cohorts": sorted({idents[m].cohort for m in F["_members"]}), "folds": F["folds"],
        "screen_coverage": F["screen_coverage"], "review_group_ids": F["review_group_ids"],
        "encoder_weight_exposure": F["encoder_weight_exposure"], "external_aliases": _ext_aliases(F, base, cans),
        "display": display,
    }


def model_counts(base: Base, model: Model) -> dict:
    idents, fams = base.idents, model.families
    sizes = Counter(len(_units(F, idents)) for F in fams.values() if F["contexts"])
    q = Counter(F["independence_qualified"] for F in fams.values())
    sb = Counter(F["sealed_basis"] for F in fams.values() if F["sealed"])
    hard = [F for F in fams.values() if F["sealed"] and F["sealed_basis"] in HARD_BASES]
    prov = [F for F in fams.values() if F["sealed"] and F["sealed_basis"] not in HARD_BASES]
    pool = [F for F in fams.values() if F["independence_qualified"] in ("auto", "reviewed") and not F["sealed"]
            and not F["seed_selection_exposed"] and F["rights_status"] != "prohibited"]
    exp = {}
    for enc in ENCODERS:
        c = defaultdict(lambda: {"families": 0, "datasets": 0, "contexts": 0})
        for F in fams.values():
            s = c[F["encoder_weight_exposure"][enc]["state"]]
            s["families"] += 1
            s["datasets"] += len(_units(F, idents))
            s["contexts"] += len(F["contexts"])
        exp[enc] = dict(sorted(c.items()))
    rights = Counter(F["rights_status"] for F in fams.values())
    cohort_cross = sum(1 for F in fams.values() if len({idents[m].cohort for m in F["_members"]} - {"external"}) > 1)
    return {
        "identities": len(idents), "identities_by_role": dict(sorted(Counter(I.role for I in idents.values()).items())),
        "families": len(fams), "families_with_contexts": sum(1 for F in fams.values() if F["contexts"]),
        "contextless_families": sum(1 for F in fams.values() if not F["contexts"]),
        "datasets_per_family_with_contexts": {str(k): v for k, v in sorted(sizes.items())},
        "cross_cohort_families": cohort_cross,
        "independence_qualified": dict(sorted(q.items())),
        "sealed": {"families": len(hard) + len(prov), "hard_families": len(hard), "provisional_families": len(prov),
                   "by_basis": dict(sorted((str(k), v) for k, v in sb.items())),
                   "hard_datasets": sum(len(_units(F, idents)) for F in hard),
                   "hard_contexts": sum(len(F["contexts"]) for F in hard),
                   "hard_fold0": sum(sum(1 for v in F["frozen_fold0"].values() if v) for F in hard),
                   "provisional_datasets": sum(len(_units(F, idents)) for F in prov),
                   "provisional_contexts": sum(len(F["contexts"]) for F in prov)},
        "seed_exposed_families": sum(1 for F in fams.values() if F["seed_selection_exposed"]),
        "seed_exposed_contexts": sum(len(F["contexts"]) for F in fams.values() if F["seed_selection_exposed"]),
        "seeds_from_exposed_sources": sum(F["seeds_from_sources"] for F in fams.values() if F["seed_selection_exposed"]),
        "rights_status": dict(sorted(rights.items())),
        "screen_coverage": dict(sorted(Counter(F["screen_coverage"] for F in fams.values()).items())),
        "contexts": sum(len(F["contexts"]) for F in fams.values()),
        "excluded_contexts": sum(len(F["excluded_contexts"]) for F in fams.values()),
        "frozen_fold0_true": sum(sum(1 for v in F["frozen_fold0"].values() if v) for F in fams.values()),
        "registry_only_contexts": sum(len(F["registry_only_contexts"]) for F in fams.values()),
        "union_edges_by_kind": dict(sorted(Counter(e.kind for e in model.edges).items())),
        "encoder_weight_exposure": exp,
        "review_rows": len(model.rows), "review_rows_by_kind": dict(sorted(Counter(r["kind"] for r in model.rows).items())),
        "N_pool": len(pool), "pool_contexts": sum(len(F["contexts"]) for F in pool),
        "retired_family_ids": len(model.retired),
        "display_withheld": {"families": sum(1 for F in fams.values() if F.get("withheld") and not F["sealed"]),
                             "members": sum(len(F["_members"]) for F in fams.values()
                                            if F.get("withheld") and not F["sealed"]),
                             "by_basis": dict(sorted(Counter(b for F in fams.values() if F.get("withheld")
                                                             and not F["sealed"] for b in F.get("withheld_basis") or [])
                                                     .items()))},
    }


def review_csv_bytes(rows: list) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({
            "review_group_id": r["review_group_id"], "kind": r["kind"], "trigger_kinds": ";".join(r["trigger_kinds"]),
            "trigger_value": r["trigger_value"], "family_ids": ";".join(r["family_ids"]), "members": r["members"],
            "n_families": r["n_families"], "n_datasets": r["n_datasets"], "n_contexts": r["n_contexts"],
            "n_fold0": r["n_fold0"], "cohorts": ";".join(r["cohorts"]),
            "evidence": common.canonical(r["evidence"]).decode("utf-8"), "flags": ";".join(r["flags"]),
            "impact_if_union": common.canonical(r["impact_if_union"]).decode("utf-8"),
            "recommendation": r["recommendation"], "recommendation_basis": r["recommendation_basis"],
            "default_if_undecided": r["default_if_undecided"], "decision_options": ";".join(r["decision_options"])})
    return buf.getvalue().encode("utf-8")


def registry_doc(base: Base, model: Model, *, review_csv_sha256: str, screen_shas: dict, cans: Canaries,
                 decisions_sha256: str | None = None, pre_sha256: str | None = None, extra: dict | None = None) -> dict:
    inp = base.inp
    fams = {fid: family_doc(fid, F, base, model, cans) for fid, F in sorted(model.families.items())}
    identities = {m: fid for fid, F in sorted(model.families.items()) for m in F["_members"]}
    encoders = {}
    for enc, spec in ENCODERS.items():
        fit = base.fits[enc]
        encoders[enc] = {"weights_sha256": spec["weights_sha256"], "fit_ref": spec["fit"], "fit_size": len(fit["ids"]),
                         "enumerated": spec["enumerated"]}
    ann = base.fits["_annotations"]
    encoders["target_fused_v1@be96adc6"]["fit_records"] = {"total": len(inp.target_meta.get("fit_records") or []),
                                                           "resolved": ann["target_resolved"],
                                                           "unresolved": ann["target_unresolved"]}
    doc = {
        "schema": REGISTRY_SCHEMA, "experiment_id": EXPERIMENT_ID, "phase": model.phase,
        "epoch_ref": {k: (inp.cfg.get("epoch") or {}).get(k) for k in ("path", "sha256", "bytes", "s3", "binding")},
        "alias_sources_sha256": inp.sha256,
        "registry_snapshot": {"poll_seq": (inp.cfg.get("registry_snapshot") or {}).get("poll_seq"),
                              "sha256": (inp.cfg.get("registry_snapshot") or {}).get("sha256")},
        "review_csv_sha256": review_csv_sha256, "review_decisions_sha256": decisions_sha256,
        "registry_pre_review_sha256": pre_sha256,
        "policy": inp.cfg["policy"], "encoders": encoders, "identities": identities, "families": fams,
        "retired_family_ids": dict(sorted(model.retired.items())), "counts": model_counts(base, model),
        "sealed_summary": {"crosscheck": base.crosscheck, "lineage_b_partitions": dict(sorted(base.lb["partition_counts"].items()))},
        "seeds": {"programs": base.seeds["programs"], "broad": base.seeds["broad"], "set_sha256": base.seeds["set_sha256"],
                  "per_source": base.seeds["per_source"]},
        "external_alias_sources": {"tabpfn35_unseen_exact1": "annotation only: frozen folds 1-4 inference (counts-only "
                                                             "audit, no per-dataset list), never exposure"},
        "screen": {"files": screen_shas, "note": "test-informed; outside every learner input closure"},
        "member_rules": {"union_keys": "exact and source-aware keys; context-level keys (input_raw_sha256, reuse_identity) "
                                       "appear as context_keys {count, set_sha256 = sha256(canonical(sorted values))}, and "
                                       "any union they cause is a union edge with its value",
                         "provenance_refs": fp.INPUT_NAMES},
        "code": _code_shas(),
    }
    if extra:
        doc.update(extra)
    return doc


# ----------------------------------------------------------------------------------------------------- decisions
PROVENANCE_KINDS = {"owner_verbatim_message": ("session", "message_utc", "message_sha256"),
                    "owner_session_selections": ("session", "message_utc", "message_sha256"),
                    "owner_directive": ("file", "version", "lines", "file_sha256")}
MESSAGE_KINDS = ("owner_verbatim_message", "owner_session_selections")
DRAFT_MARKERS = ("DRAFT", "SIMULATION")
# tools/p2_decisions.py (DL-D5 expansion) stores the owner's typed message create-only here (RUN/receipts/P2)
OWNER_MESSAGE_NAME = "dl-d5-owner-message-{sha16}.txt"
OWNER_DIRECTIVE_NAMES = ("OWNER_DIRECTIVES.md",)
RULINGS = {"rdatasets_granularity": ("item", "package"), "provisional_seals": ("keep", "override"),
           "related_class_policy": ("unresolved", "per_row")}
# the DL-D5 expansion's wording of related_class_policy 'unresolved' (tools/p2_decisions.py RELATED_CLASS_POLICY)
RELATED_CLASS_POLICY_ALIASES = {"lineage-B related-class families stay unresolved: train, never held out": "unresolved"}
DECISION_FIELDS = {"review_group_id", "decision", "sets", "family_ids", "edge_kinds", "note"}
_ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?Z")
_LINES = re.compile(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _degenerate_hex(v) -> bool:
    """All-zero, single-symbol or empty-input sha256 values are never a record of anything (review fix CL-2)."""
    return isinstance(v, str) and (len(set(v)) <= 1 or v == EMPTY_SHA256)


_DRAFT_TOKEN = re.compile(r"(?i)(?<![a-z])(?:draft|simulation)(?![a-z])")


def provenance_marks_draft(prov) -> bool:
    """True when a decision file's provenance marks a DRAFT or a SIMULATION (review fix CL-2): draft/simulation true,
    confirmed_by_owner false, a non-empty 'unconfirmed' list, the word DRAFT or SIMULATION anywhere in the session,
    kind or timestamp (any case), a note starting with DRAFT or SIMULATION, or the same in a nested named_by_owner."""
    if not isinstance(prov, dict):
        return False
    if (prov.get("draft") is True or prov.get("simulation") is True or prov.get("confirmed_by_owner") is False
            or bool(prov.get("unconfirmed"))):
        return True
    for k in ("session", "kind", "message_utc"):
        v = prov.get(k)
        if isinstance(v, str) and _DRAFT_TOKEN.search(v):
            return True
    note = prov.get("note")
    if isinstance(note, str) and note.strip().upper().startswith(DRAFT_MARKERS):
        return True
    return provenance_marks_draft(prov.get("named_by_owner")) if isinstance(prov.get("named_by_owner"), dict) else False


def _record_file(ref, what: str) -> tuple[Path, bytes]:
    _need(isinstance(ref, dict) and isinstance(ref.get("path"), str) and _is_hex64(ref.get("sha256"))
          and not _degenerate_hex(ref.get("sha256")), f"{what} must be {{path, sha256}} with a real sha256")
    p = Path(ref["path"])
    _need(p.is_file(), f"{what}: the pinned record is missing")
    raw = p.read_bytes()
    _need(_sha(raw) == ref["sha256"], f"{what}: the pinned record's sha256 differs")
    return p, raw


def _message_record(prov: dict, receipts_dir) -> dict:
    """A message provenance resolves to a pinned owner record: an explicit record {path, sha256} (the message bytes
    themselves, or an owner session/decisions receipt that names the same message sha256 and timestamp), or the owner's
    message the DL-D5 expansion stored under the receipts directory (dl-d5-owner-message-<sha16>.txt)."""
    sha, utc = prov["message_sha256"], prov["message_utc"].strip()
    if prov.get("record") is not None:
        p, raw = _record_file(prov["record"], "decisions provenance record")
        if _sha(raw) != sha:
            text = raw.decode("utf-8", errors="replace")
            _need(sha in text and utc in text, "decisions provenance record names neither this message sha256 and timestamp")
        return {"path": str(p), "sha256": _sha(raw), "form": "pinned_record"}
    _need(receipts_dir is not None, "decisions provenance needs a pinned owner record (receipts directory unknown)")
    p = Path(receipts_dir) / OWNER_MESSAGE_NAME.format(sha16=sha[:16])
    _need(p.is_file(), "decisions provenance does not resolve to a pinned owner record: no stored owner message "
                       f"{p.name} under the receipts directory and no record {{path, sha256}} (CL-2)")
    raw = p.read_bytes()
    _need(_sha(raw) == sha, f"the stored owner message {p.name} does not hash to message_sha256")
    if prov.get("message_text") is not None:
        _need(prov["message_text"].encode("utf-8") == raw, "message_text differs from the stored owner message")
    return {"path": str(p), "sha256": sha, "form": "stored_owner_message"}


def check_provenance(prov, *, real: bool = False, receipts_dir=None) -> dict:
    """PLAN §7.2 (review fix CL-2). Every decision file: a known kind with its fields, 64-hex non-degenerate hashes, an
    ISO-8601 UTC message timestamp and an embedded message_text that hashes to message_sha256. A real generation (an
    output directory under RUN/families) additionally refuses DRAFT/SIMULATION provenance and needs the provenance to
    resolve to a pinned owner record; an owner_directive is re-hashed and must be OWNER_DIRECTIVES.md or a file the
    owner named in a pinned message (named_by_owner)."""
    _need(isinstance(prov, dict) and prov.get("kind") in PROVENANCE_KINDS,
          "decisions need a provenance block (PLAN §7.2 human-authored citation) with a known kind")
    kind = prov["kind"]
    for k in PROVENANCE_KINDS[kind]:
        _need(isinstance(prov.get(k), str) and prov[k].strip(), f"decisions provenance lacks {k}")
    for k in ("message_sha256", "file_sha256"):
        if k in PROVENANCE_KINDS[kind]:
            _need(_is_hex64(prov[k]), f"decisions provenance {k} must be 64 hex")
            _need(not _degenerate_hex(prov[k]), f"decisions provenance {k} is a degenerate hash (refused)")
    if prov.get("message_text") is not None:
        _need(isinstance(prov["message_text"], str) and kind in MESSAGE_KINDS
              and _sha(prov["message_text"].encode("utf-8")) == prov.get("message_sha256"),
              "decisions provenance message_text does not hash to message_sha256")
    if provenance_marks_draft(prov):
        _need(not real, "decisions provenance marks DRAFT or SIMULATION: refused for a real generation (PLAN §7.2, "
                        "review fix CL-2)")
        return {"draft": True, "record": None}
    if kind in MESSAGE_KINDS:
        _need(bool(_ISO_UTC.fullmatch(prov["message_utc"].strip())), "decisions provenance message_utc must be ISO-8601 "
                                                                     "UTC (YYYY-MM-DDTHH:MM[:SS[.f]]Z)")
    else:
        _need(bool(_LINES.fullmatch(prov["lines"].strip())), "decisions provenance lines must read N or N-M")
    if not real:
        return {"draft": False, "record": None}
    if kind in MESSAGE_KINDS:
        return {"draft": False, "record": _message_record(prov, receipts_dir)}
    p, raw = _record_file({"path": prov["file"], "sha256": prov["file_sha256"]}, "owner_directive file")
    if p.name not in OWNER_DIRECTIVE_NAMES:
        named = prov.get("named_by_owner")
        _need(isinstance(named, dict) and named.get("kind") in MESSAGE_KINDS and _is_hex64(named.get("message_sha256"))
              and isinstance(named.get("message_utc"), str),
              "an owner_directive file other than OWNER_DIRECTIVES.md needs named_by_owner (the owner's message naming it)")
        rec = _message_record(named, receipts_dir)
        _need(p.name in Path(rec["path"]).read_text(encoding="utf-8", errors="replace"),
              "the owner's message does not name the directive file")
    m = _LINES.fullmatch(prov["lines"].strip())
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    _need(a <= b <= len(raw.decode("utf-8", errors="replace").splitlines()), "owner_directive lines are outside the file")
    return {"draft": False, "record": {"path": str(p), "sha256": prov["file_sha256"], "form": "owner_directive"}}


def validate_decisions(raw: bytes, rows: list, *, review_csv_sha256: str, pre_sha256: str, alias_sha256: str,
                       pre: Model, real: bool = False, receipts_dir=None, simulation: bool = False) -> dict:
    """Validate a review decision file against the pre-review rows and apply the owner rulings mechanically.

    real: the output directory is a real generation under RUN/families (PLAN §7.2 provenance and DRAFT/SIMULATION
    refusal, review fix CL-2). simulation: evaluate a jope.dl-family-review-decisions-simulation/v1 file in memory (a
    simulation file never validates as a decision file, and a decision file never runs as a simulation).
    Rulings (review fix CL-3): provisional_seals 'keep' STOPs on any unseal ('override': each sealed_adjacent decision
    applies); related_class_policy 'unresolved' STOPs on any lineage_b_related_class decision other than unresolved
    ('per_row': row decisions apply); rdatasets_granularity 'package' turns every row whose trigger kinds hold
    rdatasets_pkg into a union and STOPs on any other decision for such a row ('item': row decisions apply)."""
    doc = _strict_json(raw, "review decisions")
    _need(isinstance(doc, dict), "review decisions must be a JSON object")
    schema = doc.get("schema")
    if simulation:
        _need(schema == SIMULATION_SCHEMA, f"a simulation needs schema {SIMULATION_SCHEMA}")
        _need(not real, "a simulation never runs for a real generation")
        prov_info = {"draft": True, "simulation": True, "record": None}
    else:
        _need(schema != SIMULATION_SCHEMA, f"a simulation file ({SIMULATION_SCHEMA}) is never a decision file (CL-2)")
        _need(schema == DECISIONS_SCHEMA, f"decisions schema is not {DECISIONS_SCHEMA}")
        prov_info = check_provenance(doc.get("provenance"), real=real, receipts_dir=receipts_dir)
    _need(doc.get("reviewer") == "owner", "decisions reviewer must be 'owner' (DL-D5)")
    binds = doc.get("binds") or {}
    _need(binds.get("review_csv_sha256") == review_csv_sha256, "decisions bind a different review.csv")
    _need(binds.get("registry_pre_review_sha256") == pre_sha256, "decisions bind a different pre-review registry")
    _need(binds.get("alias_sources_sha256") == alias_sha256, "decisions bind a different alias-sources.json")
    rul = doc.get("rulings")
    _need(isinstance(rul, dict), "decisions lack the rulings block (D-P2-2, D-P2-3, D-P2-4, split-registry)")
    ruled = {"rdatasets_granularity": rul.get("rdatasets_granularity"), "provisional_seals": rul.get("provisional_seals"),
             "related_class_policy": RELATED_CLASS_POLICY_ALIASES.get(rul.get("related_class_policy"),
                                                                       rul.get("related_class_policy"))}
    for k, allowed in RULINGS.items():
        _need(ruled[k] in allowed, f"rulings.{k} must be one of {list(allowed)}")
    _need(rul.get("split_registry_excluded_seals_nothing") is True,
          "rulings.split_registry_excluded_seals_nothing must be true")
    by_id = {r["review_group_id"]: r for r in rows}
    group = {}
    for d in doc.get("decisions") or []:
        _need(isinstance(d, dict) and set(d) <= DECISION_FIELDS, "a group decision has unknown fields")
        g, dec = d.get("review_group_id"), d.get("decision")
        _need(g in by_id, f"decision for unknown review group {g!r} (STOP)")
        _need(g not in group, f"two decisions for {g}")
        r = by_id[g]
        _need(dec in DECISION_OPTIONS[r["kind"]], f"decision {dec!r} is not allowed for a {r['kind']} row (STOP)")
        entry = {"decision": dec}
        if dec == "union_subset":
            sets = d.get("sets")
            _need(isinstance(sets, list) and sets, f"{g}: union_subset needs sets")
            used = set()
            for s in sets:
                _need(isinstance(s, list) and len(set(s)) >= 2 and set(s) <= set(r["family_ids"]) and not (set(s) & used),
                      f"{g}: each union_subset set must hold >= 2 distinct families of the row, disjoint")
                used |= set(s)
            entry["sets"] = [sorted(set(s)) for s in sets]
        elif dec == "add_members":
            fids = d.get("family_ids")
            _need(isinstance(fids, list) and fids and all(f in pre.families for f in fids), f"{g}: add_members needs known family_ids")
            entry["family_ids"] = sorted(set(fids))
        else:
            _need("sets" not in d and "family_ids" not in d, f"{g}: sets/family_ids only go with union_subset/add_members")
        if "edge_kinds" in d:
            ek = d["edge_kinds"]
            _need(dec == "split" and isinstance(ek, list) and ek and set(ek) <= set(SPLITTABLE_KINDS),
                  f"{g}: edge_kinds go only with split and name splittable kinds {list(SPLITTABLE_KINDS)}")
            entry["edge_kinds"] = sorted(set(ek))
        group[g] = entry
    classes = []
    for c in doc.get("class_decisions") or []:
        _need(isinstance(c, dict) and set(c) <= {"kind", "when", "decision", "note"}, "a class decision has unknown fields")
        _need(c.get("kind") in ROW_KINDS, f"class decision for unknown kind {c.get('kind')!r}")
        m = _WHEN.fullmatch(re.sub(r"\s+", "", str(c.get("when") or "")))
        _need(m is not None, "class decision 'when' must read recommendation==<value>")
        _need(c.get("decision") in DECISION_OPTIONS[c["kind"]], f"class decision {c.get('decision')!r} not allowed for {c['kind']}")
        _need(c["decision"] not in ("union_subset", "add_members"), "union_subset/add_members need a group decision")
        classes.append({"kind": c["kind"], "recommendation": m.group(1), "decision": c["decision"]})
    effective, extra = {}, {}
    for r in rows:
        g = r["review_group_id"]
        if g in group:
            effective[g] = group[g]["decision"]
            extra[g] = group[g]
            continue
        hits = {c["decision"] for c in classes if c["kind"] == r["kind"] and c["recommendation"] == r["recommendation"]}
        _need(len(hits) <= 1, f"{g}: ambiguous class decisions (STOP)")
        if hits:
            effective[g] = hits.pop()
            extra[g] = {"decision": effective[g]}
    # ---- rulings, applied mechanically (review fix CL-3)
    applied = {"rdatasets_granularity": ruled["rdatasets_granularity"], "provisional_seals": ruled["provisional_seals"],
               "related_class_policy": ruled["related_class_policy"], "package_unions": 0, "unseal_decisions": 0,
               "related_class_rows_unresolved": 0}
    if ruled["rdatasets_granularity"] == "package":
        for r in rows:
            if "rdatasets_pkg" not in (r.get("trigger_kinds") or [r["kind"]]):
                continue
            g = r["review_group_id"]
            _need(effective.get(g) in (None, "union"), f"{g}: rulings.rdatasets_granularity 'package' contradicts the "
                                                       f"decision {effective.get(g)!r} on an rdatasets_pkg row (STOP)")
            _need("union" in DECISION_OPTIONS[r["kind"]], f"{g}: a {r['kind']} row cannot take the package union (STOP)")
            effective[g] = "union"
            extra[g] = {"decision": "union", "by_ruling": "rdatasets_granularity=package"}
            applied["package_unions"] += 1
    unseals = sorted(g for g, d in effective.items() if d == "unseal")
    applied["unseal_decisions"] = len(unseals)
    if ruled["provisional_seals"] == "keep":
        _need(not unseals, f"rulings.provisional_seals 'keep' contradicts {len(unseals)} unseal decision(s) (STOP)")
    rc_rows = [r["review_group_id"] for r in rows if r["kind"] == "lineage_b_related_class"]
    if ruled["related_class_policy"] == "unresolved":
        bad = sorted(g for g in rc_rows if effective.get(g) not in (None, "unresolved"))
        _need(not bad, f"rulings.related_class_policy 'unresolved' contradicts {len(bad)} lineage_b_related_class "
                       "decision(s) (STOP)")
        applied["related_class_rows_unresolved"] = len(rc_rows)
    return {"doc": doc, "sha256": _sha(raw), "effective": effective, "extra": extra, "provenance": prov_info,
            "rulings": {"ruled": {k: rul.get(k) for k in ("rdatasets_granularity", "provisional_seals",
                                                          "related_class_policy",
                                                          "split_registry_excluded_seals_nothing")},
                        "applied": applied}}


def decision_edges(dec: dict, base: Base, pre: Model, rows: list) -> list:
    """The final edge set: pre-review edges minus split edges, plus owner unions and re-admitted breaker edges.
    Review fix CL-6: a split drops SPLIT_DEFAULT_KINDS edges (github_repo, nds:name_jaccard) inside the family unless
    the decision lists edge_kinds; nds:value_strong and curated_alias edges are dropped only when listed."""
    by_id = {r["review_group_id"]: r for r in rows}
    edges = set(pre.edges)
    for g, d in sorted(dec["effective"].items()):
        r = by_id[g]
        if d == "split":
            f = r["family_ids"][0]
            members = set(pre.families[f]["_members"])
            kinds = set(dec["extra"][g].get("edge_kinds") or SPLIT_DEFAULT_KINDS)
            drop = {e for e in edges if e.kind in kinds and e.a in members and e.b in members}
            _need(drop, f"{g}: split finds no {sorted(kinds)} edge inside the family (exact or same-source keys never "
                        "split; nds:value_strong and curated_alias split only when edge_kinds lists them)")
            edges -= drop
            _need(len(components(sorted(members), [e for e in edges if e.a in members])) > 1,
                  f"{g}: split leaves the family connected (exact or same-source keys, or edge kinds not listed in "
                  "edge_kinds) (STOP)")
    for g, d in sorted(dec["effective"].items()):
        r = by_id[g]
        anchors = lambda fids: [pre.families[f]["anchor_identity"] for f in sorted(fids)]  # noqa: E731
        if d == "union" and r["kind"] == "circuit_breaker":
            edges |= set(r["_held"])
        elif d == "union":
            a = anchors(r["family_ids"])
            edges |= {_edge(a[0], x, "owner_decision", g) for x in a[1:]}
        elif d == "union_subset":
            for s in dec["extra"][g]["sets"]:
                a = anchors(s)
                edges |= {_edge(a[0], x, "owner_decision", g) for x in a[1:]}
        elif d == "add_members":
            a0 = pre.families[r["family_ids"][0]]["anchor_identity"]
            edges |= {_edge(a0, x, "owner_decision", g) for x in anchors(dec["extra"][g]["family_ids"])}
        elif d == "unseal":
            _need(any(pre.families[f]["sealed_basis"] == "provisional_pending_review" for f in r["_subject"]),
                  f"{g}: unseal is allowed only for provisional_pending_review seals")
    for e in edges:
        fk.check_edge(e.kind, e.value)
    return sorted(edges)


# ----------------------------------------------------------------------------------------------------- build
@dataclass
class BuildResult:
    outputs: dict
    summary: dict
    base: Base
    pre: Model
    final: Model | None
    cans: list


def _screen_files(base: Base, pre: Model) -> dict:
    redact = set(base.redact)
    for F in pre.families.values():
        if _display_withheld(F):
            redact |= set(F["_members"])
    return {f"{SCREEN_DIR}/{k}": v for k, v in sorted(base.scr.files(redact=sorted(redact)).items())}


def auto_withhold(model: Model) -> dict:
    """Families a recommended union would join with a sealed family (a union_subset set, or a circuit-breaker row, that
    holds a sealed family): {family_id: basis}. The pre-review withholds their display (review fix SC-3), so accepting
    the agent recommendation never seals a member the pre-review showed in clear."""
    fams = model.families
    sealed = {f for f, F in fams.items() if F["sealed"]}
    out = {}
    for r in model.rows:
        if r["kind"] == "circuit_breaker":
            sets, basis = [r["family_ids"]], "breaker_row_with_sealed_family"
        elif r.get("recommendation") == "union_subset":
            sets, basis = r.get("_sets") or [], "recommended_union_with_sealed_family"
        else:
            continue
        for s in sets:
            if set(s) & sealed:
                for f in s:
                    if f not in sealed:
                        out.setdefault(f, basis)
    return out


def _pool_eligible(F: dict) -> bool:
    return (F["independence_qualified"] in ("auto", "reviewed") and not F["sealed"] and not F["seed_selection_exposed"]
            and F["rights_status"] != "prohibited")


def held_value_leaks(base: Base, model: Model) -> list:
    """Review fix CL-4: held content or value_strong edges (not re-admitted by a decision) that span a pool-eligible
    family and another family that trains (pool or train_nonpool): one copy could be held out while the other trains."""
    edge_set, out = set(model.edges), []
    for e in sorted(base.held):
        if e in edge_set or e.kind not in NEVER_HELD_KINDS:
            continue
        fa, fb = model.fid_of[e.a], model.fid_of[e.b]
        if fa == fb:
            continue
        A, B = model.families[fa], model.families[fb]
        trains = lambda F: not F["sealed"] and F["rights_status"] != "prohibited"  # noqa: E731
        if (_pool_eligible(A) and trains(B)) or (_pool_eligible(B) and trains(A)):
            out.append({"kind": e.kind, "families": sorted([fa, fb])})
    return out


def _hidden_members(model: Model) -> set:
    return {m for F in model.families.values() if _display_withheld(F) for m in F["_members"]}


def run_build(config_path, decisions_path=None, *, guard=None, expect: dict | None = None, skip_roles=DEFAULT_SKIP_ROLES,
              previous: dict | None = None, decisions_sha256: str | None = None, raw_config: bytes | None = None,
              policy_from_code: bool = False, extra_withhold=frozenset(), real: bool = False, receipts_dir=None,
              simulation: bool = False, rehash: bool = True) -> BuildResult:
    """The offline build (pure: nothing is written). real: the output directory is a real generation under RUN/families
    (PLAN §7.2 decision provenance, CL-2); receipts_dir: where the stored owner message is looked up; simulation:
    evaluate a SIMULATION_SCHEMA file in memory (phase 'simulation', output registry.simulation.json, never written).
    raw_config / policy_from_code / extra_withhold / rehash serve the local re-pin's in-memory passes."""
    guard = guard or fp.NoGuard()
    inp = load_inputs(config_path, guard=guard, skip_roles=skip_roles, raw=raw_config, policy_from_code=policy_from_code,
                      rehash=rehash)
    base = prepare(inp, guard=guard, expect=expect, extra_withhold=extra_withhold)
    guard.wait("families", force=True)
    pre = compute_model(base, base.applied, phase="pre_review", previous=previous)
    for _ in range(4):                                  # review fix SC-3: withhold what a recommended union would seal
        cans = canaries(base, pre)
        pre.rows = review_rows(base, pre, cans)
        add = {f: b for f, b in auto_withhold(pre).items() if not pre.families[f].get("withheld")}
        if not add:
            break
        for f, b in sorted(add.items()):
            F = pre.families[f]
            F["withheld"], F["withheld_basis"] = True, sorted(set(F.get("withheld_basis") or []) | {b})
    else:
        raise FamilyStop("auto-withholding did not converge (bug)")
    rows_touch = defaultdict(set)
    for r in pre.rows:
        for f in r["family_ids"]:
            rows_touch[f].add(r["review_group_id"])
    pre.fam_rows = {fid: sorted(rows_touch.get(fid, ())) for fid in pre.families}
    qualify(base, pre)
    hidden = _hidden_members(pre)
    if base.seal_floor is not None:                     # review fix CL-5: the re-pin's seal set is reproduced
        missing = base.seal_floor - hidden
        _need(not missing, f"the pre-review shows {len(missing)} identity(ies) in clear that the local re-pin sealed or "
                           "withheld (seal floor not reproduced: STOP, re-pin)")
    if base.redacted_members is not None:
        unred = hidden - base.redacted_members
        _need(not unred, f"{len(unred)} sealed or withheld identity(ies) have pinned inputs the local re-pin did not "
                         "redact (STOP: re-pin into a new generation)", detail={"withhold_identities": sorted(unred)})
    csv_b = review_csv_bytes(pre.rows)
    screen = _screen_files(base, pre)
    screen_shas = {k: _sha(v) for k, v in screen.items()}
    pre_doc = registry_doc(base, pre, review_csv_sha256=_sha(csv_b), screen_shas=screen_shas, cans=cans)
    pre_b = common.canonical(pre_doc)
    outputs = {OUT_PRE: pre_b, OUT_CSV: csv_b, **screen}
    final = None
    verdict, stop = "PASS", None
    if base.trips:
        verdict, stop = "STOP", f"circuit breaker: {len(base.trips)} component(s) held back for review"
    if decisions_path is not None:
        raw = Path(decisions_path).read_bytes()
        _need(decisions_sha256 is None or _sha(raw) == decisions_sha256, "review decisions sha256 differs from its pin")
        dec = validate_decisions(raw, pre.rows, review_csv_sha256=_sha(csv_b), pre_sha256=_sha(pre_b),
                                 alias_sha256=inp.sha256, pre=pre, real=real, receipts_dir=receipts_dir,
                                 simulation=simulation)
        undecided_breakers = [r["review_group_id"] for r in pre.rows
                              if r["kind"] == "circuit_breaker" and r["review_group_id"] not in dec["effective"]]
        _need(not undecided_breakers, f"{len(undecided_breakers)} circuit_breaker row(s) undecided: STOP")
        edges = decision_edges(dec, base, pre, pre.rows)
        final = compute_model(base, edges, phase="simulation" if simulation else "final", pre=pre,
                              effective=dec["effective"], rows=pre.rows)
        for fid, F in final.families.items():           # the pre-review withholding carries over to the final model
            wb = sorted({b for p in final.parts[fid] for b in pre.families[p].get("withheld_basis") or []}
                        | set(F.get("withheld_basis") or []))
            if wb:
                F["withheld"], F["withheld_basis"] = True, wb
        final.rows, final.effective, final.decisions_sha256 = pre.rows, dec["effective"], dec["sha256"]
        qualify(base, final, dec["effective"])
        leaks = held_value_leaks(base, final)
        _need(not leaks, f"{len(leaks)} held content/value_strong edge(s) span a pool-eligible family and a family that "
                         "trains (review fix CL-4): decide the circuit_breaker row 'union', or a union_subset that keeps "
                         "each held value edge inside one set, or leave those families unresolved (STOP)",
              detail={"held_value_edges": leaks})
        newly = sorted(m for F in final.families.values() if F["sealed"] for m in F["_members"] if m not in hidden)
        _need(not newly, f"review fix SC-3: the decisions seal {len(newly)} member(s) that the pre-review showed in clear "
                         "(STOP): re-pin into a new generation with --withhold listing them (the STOP receipt lists the "
                         "identities), rebuild the pre-review there and re-bind the decisions",
              detail={"withhold_identities": newly})
        cans = Canaries(set(cans) | set(canaries(base, final)))
        final.deltas = {
            "families": len(final.families) - len(pre.families),
            "sealed_families": sum(F["sealed"] for F in final.families.values()) - sum(F["sealed"] for F in pre.families.values()),
            "seed_exposed_families": (sum(F["seed_selection_exposed"] for F in final.families.values())
                                      - sum(F["seed_selection_exposed"] for F in pre.families.values())),
            "decisions": dict(sorted(Counter(dec["effective"].values()).items())),
            "undecided_rows": sum(1 for r in pre.rows if r["review_group_id"] not in dec["effective"]),
            "rulings": dec["rulings"],
            "provenance": {"draft": dec["provenance"].get("draft"), "simulation": bool(dec["provenance"].get("simulation")),
                           "record": dec["provenance"].get("record")}}
        fin_doc = registry_doc(base, final, review_csv_sha256=_sha(csv_b), screen_shas=screen_shas, cans=cans,
                               decisions_sha256=dec["sha256"], pre_sha256=_sha(pre_b),
                               extra={"decision_deltas": final.deltas,
                                      "rulings": dec["doc"].get("rulings")})
        outputs[OUT_SIMULATION if simulation else OUT_FINAL] = common.canonical(fin_doc)
        verdict, stop = "PASS", None
    counts = model_counts(base, final or pre)
    summary = {"verdict": verdict, "stop": stop, "phase": (final or pre).phase, "alias_sources_sha256": inp.sha256,
               "outputs": {k: {"sha256": _sha(v), "bytes": len(v)} for k, v in sorted(outputs.items())},
               "counts": counts, "pre_review_counts": model_counts(base, pre) if final else None,
               "screen": base.scr.counts, "sensitivity": base.scr.sensitivity, "sealed_crosscheck": base.crosscheck,
               "breaker_trips": [{"datasets": t["datasets"], "edges": len(t["edges"]),
                                  "supernode_datasets": t.get("supernode_datasets"),
                                  "kept_sealed_edges": len(t.get("kept_sealed_edges") or [])} for t in base.trips],
               "rehash": {"checked": inp.rehash.get("checked"), "skipped_roles": inp.rehash.get("skipped_roles")},
               "metas_missing": sum(1 for I in base.idents.values() if I.kind == "registry_item" and I.meta is None),
               "repin": {"seal_floor": len(base.seal_floor) if base.seal_floor is not None else None,
                         "redacted_members": len(base.redacted_members) if base.redacted_members is not None else None,
                         "withhold": len(base.withhold)},
               "decision_deltas": final.deltas if final else None}
    return BuildResult(outputs=outputs, summary=summary, base=base, pre=pre, final=final, cans=cans)


def write_outputs(out_dir, outputs: dict) -> dict:
    out, res = Path(out_dir), {}
    for name in sorted(outputs):
        try:
            r = common.immutable_write(out / name, outputs[name])
        except common.ImmutableConflict:
            raise FamilyStop(f"{name}: create-only output exists with different bytes (inputs or code changed)") from None
        res[name] = {"sha256": r["sha256"], "bytes": r["bytes"], "existed": r["existed"]}
    return res


def is_real_generation(out_dir) -> bool:
    """An output directory under RUN/families (FAMILIES_DIR) is a real generation: decision files there must be owner
    records (PLAN §7.2; review fix CL-2). Staging and test directories elsewhere are not."""
    return _inside(out_dir, FAMILIES_DIR)


def build(config_path, out_dir, decisions_path=None, *, guard=None, expect: dict | None = None,
          skip_roles=DEFAULT_SKIP_ROLES, previous: dict | None = None, receipt_path=None,
          decisions_sha256: str | None = None, real: bool | None = None, receipts_dir=None) -> dict:
    started = common.utc_us()
    real = bool(real) or is_real_generation(out_dir)      # a generation under RUN/families is always real (CL-2)
    try:
        res = run_build(config_path, decisions_path, guard=guard, expect=expect, skip_roles=skip_roles, previous=previous,
                        decisions_sha256=decisions_sha256, real=real,
                        receipts_dir=receipts_dir if receipts_dir is not None else (RECEIPTS_DIR if real else None))
        _need(OUT_SIMULATION not in res.outputs, "a simulation is never written")
        written = write_outputs(out_dir, res.outputs)
    except FamilyStop as err:
        if receipt_path:
            common.write_receipt(receipt_path, {"schema": BUILD_RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
                                                "verdict": "STOP", "stop": str(err)[:500], "detail": err.detail,
                                                "started_utc_us": started, "utc_us": common.utc_us()})
        raise
    summary = dict(res.summary, written=written)
    if receipt_path:
        common.write_receipt(receipt_path, {"schema": BUILD_RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
                                            "started_utc_us": started, "utc_us": common.utc_us(), **summary,
                                            "cpu_guard": (guard or fp.NoGuard()).summary(), "code": _code_shas()})
    return summary


# ----------------------------------------------------------------------------------------------------- verify
INDEPENDENT_KEY_NOTE = ("dataset_hash in S; Rdatasets 'pkg/item' from lineage-B source keys and evidence URLs plus the "
                        "catalog v1 rows and dataset-records of every sealed dataset (fp.rdatasets_keys); sealed "
                        "metadata_sha256; and, for sealed datasets outside the epoch, every URL key (fk.url_keys), the "
                        "platform dataset id, the dedupe fingerprint and the meta blob sha256 of their catalog v1 rows")


def _row_keys_independent(row) -> set:
    """Keys of one catalog/incoming row through a path separate from derive_exact_and_source_aware (review fix SC-2)."""
    if not isinstance(row, dict):
        return set()
    prov = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    out = {("rdatasets_item", "rdatasets:" + k) for k in fp.rdatasets_keys(prov)}
    if _known(prov.get("url")):
        out |= set(fk.url_keys(prov["url"])[0])
    pn = fk.norm(prov["platform"]) if _known(prov.get("platform")) else None
    if pn and _known(prov.get("dataset_id")):
        out.add(("platform_dataset_id", f"{pn}:{prov['dataset_id'].strip()}"))
    fpr = prov.get("dedupe_fingerprint")
    if isinstance(fpr, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", fpr):
        out.add(("dedupe_fingerprint", f"{pn or 'none'}:{fpr}"))
    ms = ((row.get("files") or {}).get("meta") or {}).get("sha256")
    if _is_hex64(ms):
        out.add(("lineage_b_metadata_sha256", ms))
    return out


def independent_base_sealed(inp: Inputs) -> set:
    """Base sealed epoch datasets recomputed without derive() and without lineage_b_sets (INDEPENDENT_KEY_NOTE)."""
    sealed_parts, S, SK, MS = set(fp.SEALED_PARTITIONS), set(), set(), set()
    for _, doc in inp.lineage_b:
        fams = doc.get("families")
        for f in (fams.values() if isinstance(fams, dict) else fams or []):
            if isinstance(f, dict) and f.get("partition") in sealed_parts:
                for ds in f.get("datasets") or []:
                    S.add(ds.get("dataset"))
                    SK |= fp.lineage_b_rdatasets_keys(ds)
                    if _is_hex64(ds.get("metadata_sha256")):
                        MS.add(ds["metadata_sha256"])
    E = {r["dataset_hash"] for r in inp.rows}
    out_keys = set()
    for h in S:
        row = inp.catalog.get(h)
        SK |= fp.rdatasets_keys((row or {}).get("provenance"))
        rec = inp.sealed_records.get(h) or {}
        sk = rec.get("source_key")
        if isinstance(sk, str) and sk.lower().startswith("rdatasets:") and "/" in sk:
            SK.add(sk[len("rdatasets:"):].strip().lower())
        elif isinstance(sk, str):
            SK |= fp.rdatasets_keys({"url": sk})
        if h not in E:
            out_keys |= _row_keys_independent(row)
            out_keys |= {("rdatasets_item", "rdatasets:" + k) for k in fp.rdatasets_keys({"url": sk})} if isinstance(sk, str) else set()
            if isinstance(sk, str) and sk.lower().startswith("rdatasets:") and "/" in sk:
                out_keys.add(("rdatasets_item", "rdatasets:" + sk[len("rdatasets:"):].strip().lower()))
    out = {h for h in E if h in S}
    frozen = {r["dataset_hash"] for r in inp.rows if r["cohort"] == "frozen"}
    rows = [(h, inp.catalog.get(h)) for h in sorted(frozen)] + [(r.get("dataset_hash"), r) for r in inp.incoming_rows.values()]
    for h, row in rows:
        if not isinstance(row, dict):
            continue
        keys = _row_keys_independent(row)
        if ({k[len("rdatasets:"):] for kind, k in keys if kind == "rdatasets_item"} & SK
                or ((row.get("files") or {}).get("meta") or {}).get("sha256") in MS or keys & out_keys):
            out.add(h)
    return out


def _scan_tree(root: Path, label: str) -> tuple[dict, list]:
    """Every file under root as text (gzip decompressed); undecodable files are listed, not silently passed."""
    texts, skipped = {}, []
    root = Path(root)
    if not root.is_dir():
        return texts, skipped
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = f"{label}/{p.relative_to(root)}"
        raw = p.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError):
                skipped.append(rel)
                continue
        try:
            texts[rel] = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(rel)
    return texts, skipped


def verify(config_path, out_dir, decisions_path=None, *, guard=None, expect: dict | None = None, full_rehash: bool = True,
           previous: dict | None = None, receipt_path=None, extra_texts: dict | None = None,
           decisions_sha256: str | None = None, receipts_dir=None, real: bool | None = None,
           scan_inputs: bool = True) -> dict:
    """Rebuild in memory and check every output. The canary scan covers the outputs, the summary, alias-sources.json,
    every pinned local input under the config directory (inputs/**) and, when given, every file under receipts_dir
    (review fixes CL-5, SC-4, SC-8); any hit FAILs and an undecodable file there FAILs too."""
    started = common.utc_us()
    real = bool(real) or is_real_generation(out_dir)      # a generation under RUN/families is always real (CL-2)
    res = run_build(config_path, decisions_path, guard=guard, expect=expect,
                    skip_roles=() if full_rehash else DEFAULT_SKIP_ROLES, previous=previous,
                    decisions_sha256=decisions_sha256, real=real,
                    receipts_dir=receipts_dir if receipts_dir is not None else (RECEIPTS_DIR if real else None))
    _need(OUT_SIMULATION not in res.outputs, "a simulation is never verified")
    out = Path(out_dir)
    checks, base = {}, res.base
    for name, data in sorted(res.outputs.items()):
        p = out / name
        checks[f"bytes_equal:{name}"] = p.is_file() and p.read_bytes() == data
    model = res.final or res.pre
    fams, idents = model.families, base.idents
    owner = Counter(m for F in fams.values() for m in F["_members"])
    checks["each_identity_in_one_family"] = set(owner) == set(idents) and all(v == 1 for v in owner.values())
    inc = [r for r in base.inp.rows if r.get("status") == "included"]
    exc = [r for r in base.inp.rows if r.get("status") != "included"]
    ctx_all = [c for F in fams.values() for c in F["contexts"]]
    exc_all = [c for F in fams.values() for c in F["excluded_contexts"]]
    checks["contexts_cover_included_rows"] = sorted(ctx_all) == sorted(r["context_sha256"] for r in inc)
    checks["excluded_cover_excluded_rows"] = sorted(exc_all) == sorted(r["context_sha256"] for r in exc)
    checks["no_context_overlap"] = len(set(ctx_all) | set(exc_all)) == len(ctx_all) + len(exc_all)
    f0 = sum(1 for F in fams.values() for v in F["frozen_fold0"].values() if v)
    checks["frozen_fold0_count"] = f0 == sum(1 for r in inc if r["cohort"] == "frozen" and r.get("fold") == 0)
    try:
        for e in model.edges:
            fk.check_edge(e.kind, e.value)
        checks["no_forbidden_or_null_edge"] = all(e.kind not in fk.FORBIDDEN_FIELDS for e in model.edges)
    except fk.ForbiddenKey:
        checks["no_forbidden_or_null_edge"] = False
    built = {idents[m].dataset_hash for F in fams.values() for m in F["_members"]
             if m in base.base_sealed and idents[m].role in ("frozen", "incoming")}
    indep = independent_base_sealed(base.inp)
    # a base seal the independent row-level path cannot reproduce must come from a key the member's meta adds (the
    # hunt gate's normalized URL or platform id): anything else is a disagreement
    unexplained = set()
    for h in built - indep:
        ms = [m for m, I in idents.items() if I.dataset_hash == h and m in base.base_sealed]
        rowkeys = set().union(*[_row_keys_independent(idents[m].row) for m in ms]) if ms else set()
        via_meta = any(kv in base.lb["S_out_keys"] and kv not in rowkeys for m in ms for kv in idents[m].U)
        if not via_meta:
            unexplained.add(h)
    checks["base_sealed_recomputed_equal"] = indep <= built and not unexplained
    checks["base_sealed_families_sealed"] = all(F["sealed"] for F in fams.values()
                                                if any(m in base.base_sealed for m in F["_members"]))
    checks["seed_sources_mapped"] = all(("beyondarena:" + s) in model.fid_of for s in base.inp.seed_sources)
    checks["seed_families_exposed"] = all(fams[model.fid_of["beyondarena:" + s]]["seed_selection_exposed"]
                                          for s in base.inp.seed_sources)
    checks["family_id_rule"] = all(fid == fid_of_anchor(F["anchor_identity"]) and F["anchor_identity"] in F["_members"]
                                   for fid, F in fams.items())
    if res.final is not None:
        checks["retired_ids_resolve"] = all(v in fams for v in res.final.retired.values()) and not (set(res.final.retired) & set(fams))
    checks["pool_families_have_contexts"] = all(F["contexts"] for F in fams.values()
                                                if F["independence_qualified"] in ("auto", "reviewed"))
    if expect:
        checks["seed_sources_expected"] = len(base.inp.seed_sources) == expect["seed_sources"]
    texts = {k: v.decode("utf-8", errors="replace") for k, v in res.outputs.items()}
    texts["summary"] = json.dumps(res.summary, sort_keys=True, ensure_ascii=False, default=str)
    texts["alias-sources.json"] = base.inp.path.read_text(encoding="utf-8")       # published too (pin output)
    scope, skipped = {"outputs": len(res.outputs)}, []
    if scan_inputs:
        t, s = _scan_tree(base.inp.base / "inputs", "inputs")
        texts.update(t)
        skipped += s
        scope["inputs"] = len(t)
    if receipts_dir is not None:
        t, s = _scan_tree(Path(receipts_dir), "receipts")
        texts.update(t)
        skipped += s
        scope["receipts"] = len(t)
    for k, v in (extra_texts or {}).items():
        texts[k] = v
    hits = canary_scan(texts, res.cans)
    checks["metas_complete"] = res.summary["metas_missing"] == 0
    checks["breaker_clear"] = res.summary["verdict"] == "PASS"
    checks["scan_files_decodable"] = not skipped
    draft = {"checks": checks, "outputs": res.summary["outputs"], "counts": res.summary["counts"]}
    hits["receipt"] = canary_scan({"receipt": common.receipt_bytes(draft).decode("utf-8")}, res.cans)["receipt"]
    checks["canary_scan_clean"] = not any(hits.values())
    failed = sorted(k for k, v in checks.items() if not v)
    report = {"schema": VERIFY_RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID, "verdict": "PASS" if not failed else "FAIL",
              "phase": model.phase, "failed": failed, "checks": checks,
              "canary": {"strings": len(res.cans), "files_scanned": len(texts), "scope": scope,
                         "hits": {k: n for k, n in sorted(hits.items()) if n}, "hit_files": sum(1 for n in hits.values() if n),
                         "undecodable": skipped},
              "independent_base_sealed": {"rule": INDEPENDENT_KEY_NOTE, "datasets": len(indep), "built": len(built),
                                          "built_only": len(built - indep), "unexplained": len(unexplained)},
              "outputs": res.summary["outputs"], "counts": res.summary["counts"], "alias_sources_sha256": base.inp.sha256,
              "rehash": res.summary["rehash"], "repin": res.summary["repin"], "started_utc_us": started,
              "utc_us": common.utc_us()}
    if receipt_path:
        common.write_receipt(receipt_path, dict(report, cpu_guard=(guard or fp.NoGuard()).summary(), code=_code_shas()))
    return report


# ----------------------------------------------------------------------------------------------------- publish
def _put(clients: dict, key: str, body: bytes, ledger: list, content_type: str = "application/json") -> dict:
    from dltrain import s3io
    entry = {"op": "PutObject+readback", "bucket": s3io.BUCKET, "key": key, "sha256": _sha(body), "bytes": len(body),
             "utc_us": common.utc_us()}
    try:
        res = s3io.put_create_only(clients["writer"], key, body, clients["reader"], content_type=content_type)
    except Exception as err:  # noqa: BLE001 (every outcome is journaled, then re-raised)
        ledger.append(entry | {"status": "error", "error": s3io.safe_err(err), "done_utc_us": common.utc_us()})
        raise
    ledger.append(entry | {"status": res["outcome"], "version_id": res["version_id"], "done_utc_us": common.utc_us()})
    return {"bucket": s3io.BUCKET, "key": key, "version_id": res["version_id"], "sha256": res["sha256"],
            "bytes": res["bytes"], "outcome": res["outcome"]}


def _default_publish_clients() -> dict:
    from dltrain import s3io
    return {"writer": s3io.make_client(writes=True), "reader": s3io.make_client()}


def screen_bundle(out: Path) -> bytes:
    files = {}
    for name in fs.SCREEN_FILES:
        raw = (out / SCREEN_DIR / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "bytes": len(raw), "text": raw.decode("utf-8")}
    return common.canonical({"schema": SCREEN_BUNDLE_SCHEMA, "experiment_id": EXPERIMENT_ID, "files": files})


def publish(out_dir, *, receipt_path, verify_receipt_path, clients: dict | None = None) -> dict:
    from dltrain import s3io
    out = Path(out_dir)
    rp = Path(receipt_path)
    _need(not rp.exists(), f"{rp} exists: one publish receipt per attempt")
    vr_raw = Path(verify_receipt_path).read_bytes()
    vr = _json(vr_raw, "verify receipt")
    _need(vr.get("schema") == VERIFY_RECEIPT_SCHEMA and vr.get("verdict") == "PASS", "publish needs a verify PASS receipt")
    alias = (out / "alias-sources.json").read_bytes()
    _need(vr.get("alias_sources_sha256") == _sha(alias), "the verify receipt names a different alias-sources.json")
    names = [OUT_PRE, OUT_CSV] + [f"{SCREEN_DIR}/{n}" for n in fs.SCREEN_FILES]
    if (out / OUT_FINAL).exists():
        names.append(OUT_FINAL)
    want_phase = "final" if OUT_FINAL in names else "pre_review"
    _need(vr.get("phase") == want_phase, f"the verify receipt covers phase {vr.get('phase')!r}, not {want_phase!r}")
    blobs = {n: (out / n).read_bytes() for n in names}
    for n in names:
        _need((vr.get("outputs") or {}).get(n, {}).get("sha256") == _sha(blobs[n]), f"{n} differs from the verified output")
    clients = clients or _default_publish_clients()
    ledger, refs = [], {}
    key = lambda kind, sha, ext: f"{s3io.PREFIX_KEY}objects/{kind}/sha256/{sha}{ext}"  # noqa: E731
    refs["alias_sources"] = _put(clients, key("family-alias-sources", _sha(alias), ".json"), alias, ledger)
    refs["registry_pre_review"] = _put(clients, key("family-registry", _sha(blobs[OUT_PRE]), ".json"), blobs[OUT_PRE], ledger)
    refs["review_csv"] = _put(clients, key("family-review", _sha(blobs[OUT_CSV]), ".csv"), blobs[OUT_CSV], ledger,
                              content_type="text/csv")
    sb = screen_bundle(out)
    refs["screen_bundle"] = _put(clients, key("family-screen", _sha(sb), ".json"), sb, ledger)
    if OUT_FINAL in blobs:
        refs["registry_final"] = _put(clients, key("family-registry", _sha(blobs[OUT_FINAL]), ".json"), blobs[OUT_FINAL], ledger)
    reg = _json(blobs.get(OUT_FINAL) or blobs[OUT_PRE], "registry")
    manifest = {"schema": MANIFEST_SCHEMA, "experiment_id": EXPERIMENT_ID, "packet": "P2", "phase": want_phase,
                "refs": refs, "verify_receipt_sha256": _sha(vr_raw), "counts": reg.get("counts")}
    mb = common.canonical(manifest)
    refs["manifest"] = _put(clients, f"{s3io.PREFIX_KEY}campaigns/{EXPERIMENT_ID}/manifests/{_sha(mb)}.json", mb, ledger)
    c = reg.get("counts") or {}
    receipt = {"schema": PACKET_RECEIPT_SCHEMA, "packet": "P2", "experiment_id": EXPERIMENT_ID, "verdict": "PASS",
               "phase": want_phase, "utc_us": common.utc_us(),
               "outputs": {n: {"sha256": _sha(b), "bytes": len(b)} for n, b in sorted(blobs.items())}
               | {"alias-sources.json": {"sha256": _sha(alias), "bytes": len(alias)}},
               "verify_receipt": {"path": str(verify_receipt_path), "sha256": _sha(vr_raw)},
               "publish": refs, "s3_ledger": ledger,
               "counts": {"families": c.get("families"), "independence_qualified": c.get("independence_qualified"),
                          "sealed": c.get("sealed"), "seed_exposed_families": c.get("seed_exposed_families"),
                          "N_pool": c.get("N_pool"), "review_rows": c.get("review_rows")}}
    rb = common.receipt_bytes(receipt)
    rref = _put(clients, key("receipt", _sha(rb), ".json"), rb, ledger)
    common.immutable_write(rp, rb)
    pub = {"schema": PUBLISH_SCHEMA, "experiment_id": EXPERIMENT_ID, "phase": want_phase, "refs": refs | {"receipt": rref},
           "s3_ledger": ledger, "utc_us": common.utc_us(),
           "agent_chat_line_suggested": f"P2 {want_phase} published: receipt s3://{rref['bucket']}/{rref['key']}"
                                        f"@{rref['version_id']} (sha256 {rref['sha256']})"}
    common.write_receipt(rp.with_name(rp.stem + ".publish.json"), pub)
    return pub


# ----------------------------------------------------------------------------------------------------- CLI
def _inside(path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def _summary_line(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)


def _main(argv=None, *, roots: tuple | None = None, clients: dict | None = None, guard=None,
          expect: dict | None = EXPECT) -> int:
    fam_root, rec_root = roots or (FAMILIES_DIR, RECEIPTS_DIR)
    ap = argparse.ArgumentParser(prog="dltrain.families", description="P2 family registry v2 (PLAN §5 P2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pin", help="dltrain.family_pin: the only SQLite/S3 step; --from-local re-pins without network")
    p.add_argument("--out", required=True, help="a generation directory under RUN/families (e.g. RUN/families/g2)")
    p.add_argument("--receipt")
    p.add_argument("--ledger")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--from-local", help="a pinned generation directory: re-pin it into --out with no S3 or SQLite access")
    p.add_argument("--from-receipt", help="that generation's pin receipt (names its alias-sources sha256 and ledger)")
    p.add_argument("--withhold", help=f"{fp.WITHHOLD_SCHEMA} identities to withhold (review fix SC-3)")
    p.add_argument("--withhold-sha256")
    for name in ("build", "verify"):
        b = sub.add_parser(name)
        b.add_argument("--config", required=True)
        b.add_argument("--out", required=True)
        b.add_argument("--decisions")
        b.add_argument("--decisions-sha256")
        b.add_argument("--receipt")
        b.add_argument("--previous-registry")
        b.add_argument("--previous-registry-sha256")
        if name == "verify":
            b.add_argument("--quick-rehash", action="store_true", help="skip annotation-only v6 parquets")
    u = sub.add_parser("publish")
    u.add_argument("--out", required=True)
    u.add_argument("--receipt", required=True)
    u.add_argument("--verify-receipt", required=True)
    a = ap.parse_args(argv)
    for path, root in ((getattr(a, "out", None), fam_root), (getattr(a, "receipt", None), rec_root),
                       (getattr(a, "ledger", None), rec_root), (getattr(a, "from_local", None), fam_root)):
        if path is not None and not _inside(path, root):
            print(_summary_line({"verdict": "STOP", "stop": f"{path} is outside {root}"}), file=sys.stderr)
            return EXIT["STOP"]
    try:
        if a.cmd == "pin":
            if a.from_local:
                _need(a.from_receipt and not a.ledger and not a.preflight,
                      "pin --from-local needs --from-receipt and takes no --ledger or --preflight (it makes no S3 request)")
                _need((a.withhold is None) == (a.withhold_sha256 is None), "--withhold and --withhold-sha256 go together")
                wh = fp.load_withhold(a.withhold, a.withhold_sha256) if a.withhold else None
                s = fp.repin_local(a.from_local, a.out, from_receipt=a.from_receipt, policy=POLICY,
                                   guard=guard or fp.CpuGuard(), receipt_path=a.receipt, withhold=wh, expect=expect,
                                   receipts_root=rec_root)
                print(_summary_line({k: s.get(k) for k in ("verdict", "mode", "alias_sources", "verified", "redaction",
                                                           "withhold", "check", "s3_requests")}))
                return EXIT["PASS"]
            _need(not (a.from_receipt or a.withhold or a.withhold_sha256), "--from-receipt/--withhold need --from-local")
            s = fp.pin(a.out, policy=POLICY, ledger_path=None if a.preflight else a.ledger,
                       receipt_path=None if a.preflight else a.receipt, preflight=a.preflight, guard=guard)
            print(_summary_line({k: s.get(k) for k in ("verdict", "alias_sources", "counts", "s3_totals", "s3_plan")}))
            return EXIT["PASS"]
        if a.cmd in ("build", "verify"):
            real = is_real_generation(a.out)
            if real:
                # a real generation keeps its inputs and outputs together, and its decision file is the owner's record
                _need(Path(a.config).resolve() == Path(a.out).resolve() / "alias-sources.json",
                      "a generation under RUN/families builds only from its own alias-sources.json (<out>/alias-sources.json)")
                if a.decisions is not None:
                    d = Path(a.decisions)
                    _need(d.resolve() == Path(a.out).resolve() / DECISIONS_NAME,
                          f"--decisions must be <out>/{DECISIONS_NAME} for a generation under RUN/families (CL-2)")
                    _need(isinstance(a.decisions_sha256, str) and bool(_HEX64.fullmatch(a.decisions_sha256)),
                          "--decisions-sha256 (64 hex) is mandatory for a generation under RUN/families (CL-2)")
                    _need(d.is_file() and not (d.stat().st_mode & 0o222),
                          f"{DECISIONS_NAME} must be read-only (no write bit) (CL-2)")
            elif a.decisions is not None:
                _need(a.decisions_sha256 is None or bool(_HEX64.fullmatch(a.decisions_sha256)),
                      "--decisions-sha256 must be 64 hex")
        previous = None
        if a.cmd in ("build", "verify") and a.previous_registry:
            raw = Path(a.previous_registry).read_bytes()
            _need(a.previous_registry_sha256 and _sha(raw) == a.previous_registry_sha256,
                  "--previous-registry needs a matching --previous-registry-sha256")
            previous = _json(raw, "previous registry")
        guard = guard or fp.CpuGuard()
        if a.cmd == "build":
            s = build(a.config, a.out, a.decisions, guard=guard, expect=expect, previous=previous, receipt_path=a.receipt,
                      decisions_sha256=a.decisions_sha256, real=real, receipts_dir=rec_root)
            print(_summary_line({k: s.get(k) for k in ("verdict", "stop", "phase", "outputs", "counts", "screen",
                                                       "breaker_trips", "metas_missing", "decision_deltas", "repin")}))
            return EXIT[s["verdict"]]
        if a.cmd == "verify":
            r = verify(a.config, a.out, a.decisions, guard=guard, expect=expect, full_rehash=not a.quick_rehash,
                       previous=previous, receipt_path=a.receipt, decisions_sha256=a.decisions_sha256,
                       receipts_dir=rec_root, real=real)
            print(_summary_line({k: r.get(k) for k in ("verdict", "phase", "failed", "canary", "counts")}))
            return EXIT[r["verdict"]]
        r = publish(a.out, receipt_path=a.receipt, verify_receipt_path=a.verify_receipt, clients=clients)
        print(_summary_line({"verdict": "PASS", "phase": r["phase"], "refs": r["refs"]}))
        return EXIT["PASS"]
    except (FamilyStop, fp.PinStop) as err:
        print(_summary_line({"verdict": "STOP", "stop": str(err)[:500]}), file=sys.stderr)
        return EXIT["STOP"]


if __name__ == "__main__":
    raise SystemExit(_main())
