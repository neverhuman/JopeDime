# H89 — W3 completion-provenance admission disposition

**Decision: H86's signature mask alone does not satisfy W3's completed-status
provenance requirement.** It remains useful diagnostic metadata. This is a
registrant decision and bounded source check, not a W3 training result or a
claim that any particular real row failed to complete.

The decision applies to the frozen [H65 publication contract](../h65_w3_continuous_top_resolved/PUBLICATION_MANIFEST.json)
and [H86 export](../h86_w3_outcome_status_census/README.md), inspected at
`b34075ef64ad5d28d4766ae40f55c542125b9f9c`. Historical H65/H86 files are unchanged.
H87's pool/mask integrity work and H88's producer-source trace are separate
owner lanes; neither is claimed as verified evidence by this report.

## What is established

The exact-version H86 manifest and metrics were downloaded from real S3,
SHA-256 checked, and compared byte-for-byte with their committed copies.
Their recorded classifier hash matches the inspected `census.py`.
The manifest explicitly says status is inferred from value signatures and
not recorded completion status. Its five export fields contain context IDs,
offsets, inferred status, pool hashes and program-order hashes; none records
an execution event or terminal disposition.

H86 reports 1,403,344 signature candidates among 1,404,287 rows: 1,344,468 pass
and 58,876 fail, with 943 sentinel-shaped rows. These are source-reported
counts, not a repeated pool census. Calling the first two codes
`measured_pass` and `measured_fail` does not independently establish completion.
The actual number of completed rows verified by this audit is **unknown**.
Unknown completion must not be recoded as execution failure or zero utility.

Eight CPU controls passed: the exact classifier function, extracted by AST
without importing its data loaders, produced the expected six finite-pass,
finite-fail, sentinel, nonfinite and anomalous classifications. Two additional
paired-history controls held fitness/logratio/gate fixed while changing a
synthetic completion label from completed to unknown. Both classifications
stayed identical because the function has no completion input.
This is an information-boundary demonstration, **not evidence that those
unknown synthetic histories occurred in the real producer**. An independently
bound producer proof could supply information absent from this mask.

## Consumer and producer handoff

Until that proof exists, use H86 codes only as an additional diagnostic filter;
do not use `status in {0,1}` as the sole W3 admission predicate. Keep its
sentinel, nonfinite and anomaly exclusions. Do not change the registered
missing-component masks or infer logratio from a failure penalty.

A completion receipt must bind an authoritative producer event or a verified
producer-to-export derivation to the existing H65/H50 identities. It must cover:

- exact DL-Train context, original training-row/table/cut and family/alias closure;
- evaluator and vocabulary identities, program identity, pool hash and row order;
- terminal completion versus timeout/error/unknown, and which components were
  actually measured, including completed gate-fail outcomes;
- source artifact key/version/hash or approved mirrored provenance, and exact
  one-to-one joins with no duplicate, missing or foreign rows silently admitted.

A source trace alone is not proof that a particular run used those bytes.
An exact run-bound derivation can support a receipt without rerunning fits;
H88 is investigating whether that route exists. A typed completion field is
also insufficient if invented from these same numeric signatures. Missing
provenance stays masked. If the resulting eligible cohort changes, freeze the
new admission manifest and check the registered study's comparability before
training. No raw mixed retained-score bodies or protected prefixes are
authorized by this handoff. This document adds no production admission helper.

## H59 dependency disposition

The published [H59 result](../../../docs/research/studies/dope-neural-ranking-20261021/h59/README.md)
now supplies the result missing from H65's dated preparation record: all four
arms were unadmitted; CLIP hurt least but remained below r882. That result is
incorporated as completed dependency evidence here. It does **not** authorize
silently changing W3's matched a/b controls, c's floor/clip, d's targets,
40-update endpoint, contrasts, seeds or gate. H59's recommendation for later
candidate-generation waves is not a W3 protocol amendment.

W3 still requires a verified W1 winner configuration, a completion-admitted
outcome export, its registered training implementation and the separate GPU
qualification gate. This report does not select a W1 winner or grant a GPU slot.

## Verification and custody

Read-only inputs were two real S3 metadata objects plus three pinned repository
files. No outcome arrays, status-mask body, DL-Val/outer/sealed/holdout payloads,
features or checkpoints were opened. No native fit, optimizer update or GPU
work ran. Elo, val_reg_inner and U4 are null; no study effect is estimated.

The single nice-10 xbabe2 audit used 1.936953 measured CPU-seconds including
completed child calls and 2.895141 wall seconds, under a 60 CPU-second audit
ceiling. Publication/transport/review overhead is unmeasured, not zero.
There is no family/seed computation to distribute across hosts.

Create-only S3 custody manifest:

`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W3/astra-h89-admission-20261031T072056Z-594fb824/manifest.json`

Version: `hYAwxJDiMzIoiHi_aNSxLDhDfh4b86q5`
SHA-256: `4f7467b2ad1072164ef30a78fa8c6d3831b1a8fb5f9e4d2dd12189d826d97982`

All ten objects, including the manifest, passed exact-version GET and downloaded
byte SHA-256 checks. The manifest retains inputs, inspected source hashes,
`audit.py`, eight control results, resources and this decision's metrics.
The checker is external source pinned by SHA-256 in that manifest, not a
committed repository test. Restore the listed objects and run `audit.py` with
Python/NumPy, a repository containing the recorded source commit, and read-only
access to the two listed S3 versions; replace its `origin/main` resolution with
the recorded commit for historical reproduction. No bulk outputs were created;
small custody files remain locally. [metrics.json](metrics.json) is byte-identical
to the archived decision metrics.
