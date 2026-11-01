# H150 — W5 provenance unblock: identity-only sealed-family manifest and the three W5 proofs

**Status: provenance record. W5 remains HELD.**

- **Proof 1 (global sealed absence):** zero overlap on every identity key and on the P3 registry/holdout crosswalk,
  against an identity-only manifest of the owner-defined sealed roster. It awaits only an authority receipt.
- **Proofs 2 and 3:** still **UNPROVED**. No per-child lineage and no original-row source evidence exists.

Per the dispatch, a failing proof means this record is published and the work stops. **No W5 training
preregistration is filed.**

> **Numbering.** Owner nudge `w5_t2045` named this wave H148. luna_v1_res08 had already claimed H148 and rd_astra
> H149, so it is **H150**, as ENG:watcher confirmed. Claim: `~/jope_plan/ns/receipts/H150_claim.txt`.

## 1. Identity-only sealed-family manifest

[sealed_family_manifest.json](sealed_family_manifest.json) is 82.7 KB, SHA-256
**`0314599fe06515d3f3e976c0d435cbe4e9fbe26738014bd89470f525702f983b`**. S3 copy:
`H150/opus-h150-20261101a/protected_family_identity_manifest.json`, VersionId `uzkqtZ0R…`.

**Content.** Family IDs (`p3.family_id`), dataset hashes, view IDs, context IDs and catalog IDs only. No sealed
rows, outcomes, feature bodies or pool arrays were opened.

**Authority definition.** Owner t2300 ("the sealed meta-test group = all P3 `sealed_excluded` families (293
production + 470 catalog)") is enumerated identity-only by the RLFW sealed-roster file
(`SEALED_ROSTER_PROPOSAL.json`, SHA-256 `799d33eb…`).
- It has 97 families, all `sealed_final`, and states that "sealed family ids are p3.family_id".
- It has **293 unique context IDs**, all from `sealed_production` families, and 763 unique view IDs. That leaves
  **470 catalog views**: exactly the owner's 293 + 470.
- The builder refuses any source that does not reproduce these counts.

**Conservative union.** The manifest also includes:
- P3 registry `sealed` families (28; 2 of them not in the roster);
- P3 `holdout.json` `heldout` families (33);
- the catalog's 14 sealed-name-match IDs.

In total: **132 families, 153 dataset hashes, 763 views, 293 contexts and 14 catalog IDs**.

**Alias registry pin.** The P3 family registry's `alias_sources_sha256` is `f818efaf…`, which equals
`g2/alias-sources.json`. The 123 W5 candidate families are all canonical registry family IDs, so the registry is the
alias resolver.

**Receipts.** [input_receipts.json](input_receipts.json) gives the SHA-256 of every input, plus S3 VersionIds where
they exist:
- the holdout assignment's own S3 commitment;
- the v3 candidate manifest;
- the manifest itself.

The registry, alias-registry and catalog copies are local custody with no established S3 version, and this is
stated.

## 2. Authority receipt

A signing request was filed:
- file `~/jope_plan/signing/requests/rd_opus-w5-manifest-20261101T025504Z.json`;
- label `w5-manifest`, with the manifest SHA above.

`signer_watch` **rejected it: "label not authorized"**. Its whitelist covers release `r782`/`rel07` and `q1`. As
instructed, this is noted in AGENT_CHAT and the request stays on file for ENG:watcher. **There is no authority
receipt yet**, so proof 1 is `PASS_PENDING_AUTHORITY_RECEIPT`, not PASS.

## 3. The three W5 proofs

Rerun with [w5_proofs.py](w5_proofs.py) (`check`) against the pinned W5 candidate set: the H98 v3 corpus manifest,
SHA-256 `701affc6…`, VersionId `gd.Jxeg…`, with 254 contexts in 123 families, 127 datasets and 254 views. Full output
is in [results.json](results.json).

| Proof | Check | Result | Status |
|---|---|---|---|
| **1. Global sealed absence** | Candidate family, dataset, view, context and view-prefix against the manifest (including catalog IDs) | **0 / 0 / 0 / 0 / 0** | **PASS_PENDING_AUTHORITY_RECEIPT** |
| | Crosswalk: candidate families absent from the P3 registry / registry-sealed / holdout `sealed_excluded` or `heldout` | **0 / 0 / 0** (all 123 are P3 `train_fit`) | |
| **2. Parent → child link** | Per-child lineage fields (`parent_program_sha256`, `parent_dsl`, `operator`, `child_program_sha256`) in the candidate manifest | none | **UNPROVED (FAIL)** |
| | Lineage artifact in the W5 source prefix, by a fresh key-only listing | 83 objects, unchanged since 2026-10-24; no lineage, parent, child or operator artifact | |
| **3. Original-row source** | Per-row source fields (`source_row_locator`, `source_s3_key/version`, `source_sha256`, `original_row_ledger`) | none | **UNPROVED (FAIL)** |
| | Row-ledger or strat300 artifact in the source prefix | none | |

**Exact failing checks.**
- Proof 2: `per_child_lineage_fields == []` and `source_prefix_lineage_artifacts == []`.
- Proof 3: `per_row_source_fields == []` and `source_prefix_row_artifacts == []`.

This agrees with H98, H102, H117 and the t0855 refresh. The manifest and receipt resolve the sealed-roster gap but
cannot create lineage or row provenance, and no proof is weakened.

## What would unblock W5

1. **Proof 1.** ENG:watcher signs the manifest (SHA `0314599f…`) under an authorized label. The checker then reports
   PASS with no other change.
2. **Proofs 2 and 3.** These need artifacts that do not exist in the admitted source:
   - a field-allowlisted per-child parent/operator/child export for the 504 W5 children;
   - an original-row/strat300 ledger, with source key, VersionId and SHA-256.

   Otherwise, W5 children must be regenerated with a lineage-recording proposer, under a separate authorization.
   Historical parenthood or rows must not be synthesized.

Only after all three proofs pass would the H148-style W5 prereg be filed: parent-child value plus bounded lookahead,
CPU-first, a 4,800-fit ceiling, and r882 91/91 baseline parity.

## Data scope

- **Reads.** Identity and registry metadata only: the roster file, P3 registry, alias registry, holdout assignment
  labels, catalog inventory IDs and the v3 manifest key structure.
- **Never opened.** Sealed or holdout rows, outcomes, feature bodies, pool arrays and per-context build files.
- **Partial over-read.** During the source search, a broad JSON scan briefly read per-context `context.json` build
  files in the P3 scratch tree. These contain feature category metadata, not sealed rows. The scan was stopped as
  out of scope, and nothing from it is used.
- **Compute.** No GPU, no model, no DOPE load and no fits.
- **Tests.** [test_w5_proofs.py](test_w5_proofs.py) has 13 synthetic controls.
