# H314 — dev91 lineage independence audit

**H310_LINEAGE_INDEPENDENT under the preregistered metadata rule; shadow only.**
All19 dev families are CLEAN (rule-unmatched), NEIGHBORED0, UNKNOWN0 against205
DL-Train families. Pooled CLEAN gain **+0.021310997**, wild-basic95% CI
**[+0.013483416, +0.031377680]**; positive lower
endpoints on **5/6** seeds. This passes CLEAN>=10,
pooled>+.001 and positive lower endpoints>=4/6. The confounded clause is false.
**CLEAN does not establish semantic or historical independence.** No serving,
breakthrough, new-population, neural-learning or speed qualification follows.

## Hypothesis and setup

H312's gain is concentrated in roughly10/19 reused development families. H314 asks
whether those gains track recorded source lineage or near-duplicate DL-Train
schemas. No new fits, dev91 scoring, raw feature rows or labels were used; only
archived H312 utilities and metadata. No GPU command, live writer or foreign job
was accessed. DOPE c51a56d0 remains untouched.

[PREREG.md](PREREG.md) defines namespaced source/explicit lineage equality (a),
nonempty dedupe equality (b), or normalized-name Jaccard>=.80 AND absolute log row
ratio<=.10 (c). A family matches if any member pair matches; missing comparisons
remain UNKNOWN. Normalization is NFKC/casefold and separator-to-underscore.
Primary schema is files.meta.output_headers paired with catalog.shape.rows;
raw schema hashes/counts are retained separately, not mixed. Similarity for the
association diagnostic is1 for observed identity/fingerprint matches, otherwise
maximum TRAIN column Jaccard; row-ratio constraints apply to classification.

Initial prereg **13a16cfc5af2de6ba9ddc9b17bb3789f72a09e13** preceded archive recovery and all metadata
analysis. Metadata-blob transport clarification **65bd4d6a27112b7ab0a99b94dafe1970f6ae26e1**,
two-sided CI/schema clarification **5df687ea4c2c970f6c3d7cf9cccda87786f8a48e**, and identity
correction **9faa5b42ed9616ff7d234e620fcdb33405281f7b** all preceded the first metadata/gain join. Corrected
classification/code published **ec9d12ad9ad060e5f8b440843ed6be801abb9a79** before any gain join.
Executed serialization repair **a6473ad94ea504da85a5a0b2d72c734732593b98** converts Fisher NumPy
counts to JSON integers; it changes no formula, class or decision. The first
analysis attempt stopped at JSON serialization; the second completed. Failed
attempts and their source snapshots remain archived. The amended prereg history
is explicit; this is not represented as an unchanged initial specification.

H312 archive exact VersionId IWjt3KdZCfYSgh6.XeNPrKrBXQW_MNqQ and SHA256
`d8bbd501469baaeb777ce93e5d1bf49d657a7a4aa11e8f71a008f3d0df29ed15` were verified.
Only four allowed archived utility/receipt members were extracted; models were
not decoded. [Regression](regression_receipt.json) exactly reproduces all6 seed
gains and r3175 baseline context mean **1.1389572403748425**. Catalog v246 VersionId
`eJuErIwj0Pgu.lAgflPcWvvRKyuwptLC`, SHA256
`c5f5eb6db36d24daa9abc5c7808a49773a2b94c1765620d236e91a471fe2cb54` is H313's pin.
Every registry and files.meta GET was pinned to VersionId and hashed; files.meta
hashes must equal the catalog. Train/test/raw hashes and registry CSVs were denied.

## Metadata outcome and limitations

[Family metadata](family_metadata.json) covers276 unique dataset records:
257 TRAIN records/205 families and19 dev records/19 families, representing1219
TRAIN contexts and91 dev contexts. Read598 JSON metadata objects:184 full-source
registry entries,138 selected split-registry entries,276 files.meta blobs.
Full-source registry entries matched **0/276** selected datasets; split registry
matched138. Source IDs/roots and dedupe fields therefore rely on pinned catalog
and files.meta provenance. This is a coverage limitation, not complete historical
raw-source reconstruction. All rule operands were present; unpopulated explicit
parent/raw-root fields are not invented. Hashes and exact raw metadata objects
remain in the archive; [metadata review](metadata_review.json) records conflicts.

Three TRAIN files.meta header counts exceed catalog columns (34vs33,45vs43,8vs6).
Their maximum dev Jaccards are .0541,.0755,.1111; discrepancies are retained and
no records were relabelled from outcomes. Ingested names include a generic target
column; metadata-level resemblance is weaker than semantic source verification.
The independent recalculation verified all4883 member-pair predicates and598
object hashes without opening gains. Classifier outcome-path denial is tested.

NEIGHBORED0 ID-list SHA `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
CLEAN19 ID-list SHA `bc447c2efb1793c1d7472efaf5d7547d4d103831820660483157e5dc442a88ba`.
UNKNOWN0 has the empty-list hash above. No clause fired. The complete IDs and
classification SHA `2e97b4cf06dfdfb6d93e369f93ae0994870f1371847758639892e17678299b9c` are in
[classification receipt](classification_receipt.json); all pair receipts are archived.

### Pre-gain filename correction and sensitivity

The initial extractor incorrectly treated bare source_file names as lineage roots.
Marine-water dataset12d32dd30f8c602e and airfare dataset32acfcb6504af2eb share
`export.csv?accessType=DOWNLOAD` despite different source IDs,URLs,fingerprints
and Jaccard .02857. The amendment excludes bare filenames for ALL records while
retaining them as metadata. Initial classification/pairs were preserved before
correction and before any gain join (initial SHA
`5a003c1b86e8145f1d2ad0146b6fbb40306ca887aa725afae68a9953068e45bd`).

The explicitly labelled known-false-match sensitivity gives1 apparent neighbor,
18 CLEAN, pooled CLEAN **+0.022494941**, wild CI **[0.014587309518719333, 0.03297927503801955]**,
positive bounds6/6 and the same **H310_LINEAGE_INDEPENDENT** decision.
That one apparent neighbor has pooled gain+0.000000000;
its single-family wild interval is degenerate. This is a diagnostic of the bug,
not evidence of genuine lineage. Full per-seed/group sensitivity is in metrics.

## Archived-gain results

Family-equal gain is mean of within-family arm-minus-reference utility differences;
pooled primary is median of the six seed means. Family medians are also supplied.
Exact one-sided sign-flip tests enumerate524288 shared family-sign assignments.
Wild intervals use10000 shared centered Rademacher draws, RNG314150, basic95% CI.
The decision uses its two-sided lower endpoint; one-sided95% bounds are descriptive.
W/T/L tolerance1e-4, ties never wins. Drop-top removes largest gains per seed.

| Root | Family-equal mean gain | Median family gain | W/T/L | Exact p | Wild95% CI | Drop1 | Drop5 | Elo |
|---|---:|---:|---|---:|---|---:|---:|---:|
| 30001 | +0.011448662 | +0.000000000 | 6/9/4 | 0.048828125 | [-0.000337689, +0.023322137] | +0.007014497 | -0.001096525 | 1570.29 |
| 30002 | +0.015265249 | +0.011115560 | 11/5/3 | 0.00314331055 | [+0.006144600, +0.024437180] | +0.013523778 | +0.006178801 | 1725.29 |
| 30003 | +0.027356745 | +0.004955392 | 11/5/3 | 0.00201416016 | [+0.009822969, +0.045126191] | +0.020834853 | +0.006154198 | 1725.29 |
| 30004 | +0.013542218 | +0.004955392 | 11/4/4 | 0.00930786133 | [+0.003520755, +0.023656809] | +0.010450598 | +0.002128382 | 1675.46 |
| 30005 | +0.029545876 | +0.019481281 | 14/3/2 | 0.00456237793 | [+0.009956840, +0.049735892] | +0.022709727 | +0.008353330 | 1837.30 |
| 30006 | +0.032776001 | +0.010323159 | 12/5/2 | 0.000610351562 | [+0.015005447, +0.050638610] | +0.026997727 | +0.012268982 | 1810.54 |
| Pooled median | +0.021310997 | +0.007639275 | 11/5/3 | 4.57763672e-05 | [+0.013483416, +0.031377680] | +0.017179316 | +0.006166499 | 1725.29 |

**reference = r3175 slot0 (dev91)**. Pooled counts/Elo are componentwise medians
across dependent seed policies, not an extra policy. All invalid counts0, baseline0.
NEIGHBORED and UNKNOWN are empty: their gains, intervals, W/T/L, p and drop statistics
are unmeasured/null, never zero-imputed. Seed30001's two-sided CI crosses zero;
only5/6 pass the individual bound. CLEAN=ALL here.

Spearman family-median gain vs maximum TRAIN similarity **rho=-0.019391955**,
two-sided asymptotic p=0.937194898, n19.
Fisher top10-vs-neighbor table **[[0,10],[0,9]]**, one-sided p1, odds ratio undefined:
there are no neighbors, so enrichment is not estimable. Neither is causal proof.
[metrics.json](metrics.json) (33,722bytes) contains all family gains,
per-seed/pooled summaries, hashes, association diagnostics and sensitivity results.

## Verification, custody and next step

20 focused tests pass, including threshold/missingness, forbidden metadata access,
exact enumeration, empty/degenerate groups, tie handling, decision boundaries and
Fisher JSON serialization. [Numerical review](numerical_review.json) independently
reconstructs utilities, enumerates subset sums, checks wild quantiles, W/T/L/drop,
rank correlation, Fisher hypergeometric tail and decision. This is same-agent
review with an independent calculation, not an external approval. Source compile,
ruff, diff/scope and published-source/index/link checks complete the local proof.
Full repository/hosted CI **NOT_RUN** under the owner freeze; no PR/main/fastlane action.

CPU xbabe2, nice19, four idle-sampled cores34,48,49,98, numeric threads1, systemd
MemoryMax4GiB; sampled RSS peak191,442,944bytes and hostCPU
32.3% through analysis, no resource guard violation.
Hard limits: MemAvailable>=20GiB, hostCPU<=85%, own scratch<=1,000,000,000bytes,
absolute compute12:15Z/close12:30Z Nov4. Phase commands, source hashes, environment
and resource receipts are archived. Dependencies: [pins](dependency_pins.json).

Evidence is create-only S3 under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H314/astra-20261104T102434Z/`.
[Manifest](manifest.json) and [exact-version S3 receipt](s3_manifest.json) bind source,
metrics and the verified archive. Every archive member is rehashed after exact-version
GET; own scratch is deleted only after complete recovery mapping/no-consumer checks.
Final cleanup, publication/index and lifecycle receipts live in
`/home/ubuntu/jope-rescue-20260921/h314-astra` and the same S3 prefix.

Cross-critique of H312/H313: H314 extends family-ID disjointness to a recorded
lineage/schema rule, but missing historical registry coverage, metadata conflicts,
adaptive reuse and only19 development families persist. H312's negative drop-top10
sensitivity is not undone by a metadata independence label. H313 found no fresh
catalog population; reusing these gains does not satisfy fresh confirmation.
Next step remains independently sourced new ingest and preregistered confirmation
when authorized. No new wave, automatic ingest, serving change or promotion here.
