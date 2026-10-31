# H117 — W5 data-provenance re-check (identity-only)

Status: **W5 remains HELD.** None of the three W5 proofs passes.

- **Proof 1.** Against two sealed-identity sources not used before, the W5 candidate set has **zero overlap on every
  identity key**: family, dataset, view, context and view→dataset prefix. Neither source is an authoritative global
  roster, so proof 1 moves only to UNPROVED_BOUNDED.
- **Proofs 2 and 3.** These remain UNPROVED. The approved manifest has no per-child lineage and no per-row source
  fields.

## What was required

[H98](../h98_w5_provenance/README.md) and [H102](../h102_w5_provenance_gaps/README.md) held W5 for three missing
proofs:

1. Every W5 family is absent from the global sealed meta-test roster.
2. Every child is linked to an existing parent program in its pool.
3. Every original row is traced to a source S3 object and SHA.

H117 re-tests them with sources those audits did not use. It reads identities and key names only.

## Inputs (all hash-pinned)

| Input | Identity |
|---|---|
| W5 candidate set: the v3 corpus manifest (H98's object) | `corpus_v3_manifest.json`, version `gd.JxegwvXXxSr7SNt7qJNbEij97pO7L`, sha256 `701affc6…`. **254 contexts, 123 families, 127 datasets, 254 views** (v3a ∪ v3b). |
| RLFW sealed-roster proposal | `deploy/dl-flywheel/pipelines/rlfw/judge/SEALED_ROSTER_PROPOSAL.json` on main (sha256 `799d33eb…`). **97 sealed-final families**, with 153 dataset hashes, 763 view ids and 293 context ids. **`adopted: false`.** |
| S3 dataset catalog exclusions | `S3_DATASET_CATALOG.md` (sha256 `bb3a1d20…`), with 14 sealed-matching catalog ids |

## Results ([results.json](results.json))

| Proof | Check | Result | Status |
|---|---|---|---|
| 1. Global sealed absence | Candidate family / dataset / view / context / view-prefix vs the proposal | **0 / 0 / 0 / 0 / 0** | UNPROVED_BOUNDED |
| | Candidate dataset / view-prefix vs the catalog sealed ids | **0 / 0** | |
| 2. Parent → child link | Per-child lineage fields in the manifest (`parent_program_sha256`, `parent_dsl`, `operator`, `child_program_sha256`) | **none present** | UNPROVED |
| 3. Original-row source | Per-row source fields (`source_row_locator`, `source_s3_key/version`, `source_sha256`, `original_row_ledger`) | **none present** | UNPROVED |

**Lineage-like keys that do exist are pool-level only:** `parent_pool_path`, `parent_pool_sha256`,
`parent_n_candidates`, `parent_incumbent_best_f`, and the parent corpus manifest path and SHA. H102 already noted that
a whole-pool SHA is not an AST parent link. This census confirms it for both revisions.

## Why this is not a PASS

- **Proof 1** needs an *authoritative*, identity-only global sealed roster and an independent authority receipt. The
  RLFW proposal is unadopted. The catalog list records exclusions, not a complete roster. H98's 86 sealed-marker
  components also showed zero overlap.
  - Three independent sources now agree on zero overlap.
  - That raises confidence, but it cannot certify global absence.
- **Proofs 2 and 3** need artifacts that do not exist in the approved source prefix, which still has the same 83
  objects as H98.
  - The phase-7 and v2 integration report bodies might carry lineage, but the v2 producer serializes outer-test
    outcomes. This audit keeps H98's boundary and does not open them.

## What would close W5 (owner decisions)

1. **Proof 1.** Adopt an identity-only sealed-family roster, for example by ratifying the RLFW proposal or issuing
   the meta-test roster's identity list, with an authority receipt. The checker here would then produce a real PASS
   or FAIL in seconds.
2. **Proofs 2 and 3.** Either a custodian extracts a **redacted, key-only** lineage and row-source projection from the
   phase-7 and v2 reports, with outcome fields removed before anyone else reads it, or the W5 children are
   regenerated with a lineage-recording proposer.

## Data scope

The audit decoded no pool arrays and no outcome values, read no sealed-bucket objects, and opened no report bodies.
It used no DL-Val, outer-test or holdout data, no GPU and no DOPE. My S3 helper refused to upload a copy of the
proposal file because its key contains "sealed". The committed path and blob sha on main pin it instead.

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h117_w5_provenance/test_check_w5_identity.py   # 10 synthetic tests
python3 -I research/flywheel_selfplay/h117_w5_provenance/check_w5_identity.py \
  --v3-manifest <corpus_v3_manifest.json at version gd.Jxeg...> \
  --sealed-proposal deploy/dl-flywheel/pipelines/rlfw/judge/SEALED_ROSTER_PROPOSAL.json \
  --catalog-md ~/jope_plan/master/source/S3_DATASET_CATALOG.md --out results.json
```

## Evidence

S3 `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W5/opus-h117-20261031a/` holds the results,
both sources and a catalog copy. Each was written create-only with a version-pinned readback hash; the versions are in
[results.json](results.json).
